"""Host-side contract tests for the opt-in pre-compress checkpoint API (v2).

The contract has three parts:
- providers opt in by advertising ``pre_compress_checkpoint_api_version = 2``
  (v1 is the implicit historical best-effort contract with raw messages);
- ``MemoryManager`` exposes capability probing and a ``require_checkpoint``
  mode whose failure must propagate instead of being swallowed;
- the compression host normalizes messages to direct user/assistant evidence
  before handing them to v2+ providers;
- providers advertising v3 additionally receive ``tool_evidence=`` (one entry
  per tool-result row) and transcript-stable ordinals; v1/v2 providers see
  byte-identical calls and payloads.
"""

import copy

import pytest

from agent.conversation_compression import (
    CompressionCheckpointUnavailable,
    _checkpoint_blocked,
    _direct_messages_for_pre_compress_memory,
    _tool_evidence_for_pre_compress_memory,
)
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
from agent.memory_manager import MemoryManager
from agent.memory_provider import (
    PRE_COMPRESS_CHECKPOINT_API_VERSION,
    PRE_COMPRESS_TOOL_EVIDENCE_API_VERSION,
    MemoryProvider,
)


class _BaseStubProvider(MemoryProvider):
    def __init__(self, name="stub"):
        self._name = name
        self.pre_compress_calls = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        return None

    def get_tool_schemas(self):
        return []

    def on_pre_compress(self, messages):
        self.pre_compress_calls.append(messages)
        return f"{self._name} context"


class _CheckpointProvider(_BaseStubProvider):
    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION


class _ToolEvidenceProvider(_CheckpointProvider):
    pre_compress_checkpoint_api_version = PRE_COMPRESS_TOOL_EVIDENCE_API_VERSION

    def on_pre_compress(self, messages, *, tool_evidence=None, **kwargs):
        self.pre_compress_calls.append((messages, tool_evidence))
        return f"{self._name} context"


class _FailingCheckpointProvider(_CheckpointProvider):
    def on_pre_compress(self, messages):
        raise RuntimeError("durable store unreachable")


class _FailingLegacyProvider(_BaseStubProvider):
    def on_pre_compress(self, messages):
        raise RuntimeError("legacy best-effort failure")


def test_provider_base_class_defaults_to_implicit_historical_api_version_one():
    assert MemoryProvider.pre_compress_checkpoint_api_version == 1
    assert PRE_COMPRESS_CHECKPOINT_API_VERSION == 2


def test_v1_providers_receive_raw_messages_and_v2_receive_evidence():
    """The historical (v1) contract is untouched: raw message list.

    Only providers that opted into checkpoint API v2 receive the
    host-normalized evidence handoff.
    """
    manager = MemoryManager()
    legacy = _BaseStubProvider("legacy")
    manager.add_provider(legacy)
    raw = [
        {"role": "user", "content": "evidence"},
        {"role": "tool", "content": "tool output", "tool_call_id": "t1"},
    ]
    evidence = [{"role": "user", "content": "evidence"}]

    manager.on_pre_compress(raw, evidence_messages=evidence)
    assert legacy.pre_compress_calls == [raw]

    durable_manager = MemoryManager()
    durable = _CheckpointProvider("durable")
    durable_manager.add_provider(durable)
    durable_manager.on_pre_compress(raw, evidence_messages=evidence)
    assert durable.pre_compress_calls == [evidence]


def test_direct_messages_filter_keeps_only_direct_source_evidence():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "durable user decision"},
        {"role": "assistant", "content": "direct assistant answer"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "tool output", "tool_call_id": "t1"},
        {
            "role": "assistant",
            "content": "previous compaction summary",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        "not-a-dict",
    ]

    direct = _direct_messages_for_pre_compress_memory(messages)

    assert [m["content"] for m in direct] == [
        "durable user decision",
        "direct assistant answer",
    ]


def test_direct_messages_filter_keeps_prose_of_tool_call_messages():
    """Assistant prose next to tool_calls is evidence; the payload is not."""
    messages = [
        {"role": "user", "content": "please scan the network"},
        {
            "role": "assistant",
            "content": "Scanning now — the last sweep found 26 hosts.",
            "tool_calls": [{"id": "t1", "function": {"name": "terminal"}}],
        },
        {"role": "assistant", "content": "   ", "tool_calls": [{"id": "t2"}]},
    ]

    direct = _direct_messages_for_pre_compress_memory(messages)

    assert [m["content"] for m in direct] == [
        "please scan the network",
        "Scanning now — the last sweep found 26 hosts.",
    ]
    assert all("tool_calls" not in m for m in direct)
    # The original message list is not mutated.
    assert messages[1]["tool_calls"]


def test_manager_advertises_checkpoint_capability_only_with_capable_provider():
    # The host allows one external provider per manager, so capability is
    # probed on two separate managers.
    legacy_manager = MemoryManager()
    legacy_manager.add_provider(_BaseStubProvider("legacy"))
    assert legacy_manager.supports_pre_compress_checkpoint(
        PRE_COMPRESS_CHECKPOINT_API_VERSION
    ) is False

    durable_manager = MemoryManager()
    durable_manager.add_provider(_CheckpointProvider("durable"))
    assert durable_manager.supports_pre_compress_checkpoint(
        PRE_COMPRESS_CHECKPOINT_API_VERSION
    ) is True


def test_manager_require_checkpoint_raises_without_capable_provider():
    manager = MemoryManager()
    manager.add_provider(_BaseStubProvider("legacy"))

    with pytest.raises(RuntimeError, match="pre-compress checkpoint"):
        manager.on_pre_compress(
            [{"role": "user", "content": "evidence"}],
            require_checkpoint=True,
            checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
        )


def test_manager_require_checkpoint_propagates_checkpoint_provider_failure():
    manager = MemoryManager()
    manager.add_provider(_FailingCheckpointProvider("durable"))

    with pytest.raises(RuntimeError, match="durable store unreachable"):
        manager.on_pre_compress(
            [{"role": "user", "content": "evidence"}],
            require_checkpoint=True,
            checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
        )


def test_manager_require_checkpoint_succeeds_and_returns_provider_context():
    manager = MemoryManager()
    durable = _CheckpointProvider("durable")
    manager.add_provider(durable)

    combined = manager.on_pre_compress(
        [{"role": "user", "content": "evidence"}],
        require_checkpoint=True,
        checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
    )

    assert "durable context" in combined
    assert durable.pre_compress_calls


def test_manager_best_effort_mode_keeps_historical_swallow_semantics():
    manager = MemoryManager()
    manager.add_provider(_FailingLegacyProvider("legacy"))

    combined = manager.on_pre_compress([{"role": "user", "content": "evidence"}])

    assert combined == ""


# --- Checkpoint API v3: tool evidence ---------------------------------------


def _v3_transcript():
    """Live transcript shape: tool rows as written by tool_dispatch_helpers,
    ``tool_calls`` entries as built by chat_completion_helpers."""
    call = {"id": "call_1", "type": "function",
            "function": {"name": "terminal", "arguments": '{"command": "nmap"}'}}
    return [
        {"role": "user", "content": "scan the network"},
        {"role": "assistant", "content": "Scanning now.", "tool_calls": [call]},
        {"role": "tool", "name": "terminal", "tool_name": "terminal", "content": "26 hosts up",
         "tool_call_id": "call_1", "timestamp": "2026-08-26T00:00:00Z"},
        {"role": "assistant", "content": "prior summary", COMPRESSED_SUMMARY_METADATA_KEY: True},
        {"role": "user", "content": "thanks"},
    ]


def test_tool_evidence_has_one_entry_per_tool_row_joined_to_its_call():
    messages = _v3_transcript()

    tool_evidence = _tool_evidence_for_pre_compress_memory(messages)

    assert tool_evidence == [{
        "ordinal": 3, "tool_call_id": "call_1", "tool_name": "terminal",
        "content": "26 hosts up", "timestamp": "2026-08-26T00:00:00Z",
        "call_name": "terminal", "call_arguments": '{"command": "nmap"}',
    }]


def test_tool_evidence_joins_composite_bridge_ids_on_the_call_id_half():
    """Responses-bridge transcripts store ``call_id|response_item_id`` on the
    assistant side while the tool row carries the normalized call-id half."""
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_7|fc_abc", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_7", "tool_name": "terminal",
         "content": "ok"},
    ]

    [entry] = _tool_evidence_for_pre_compress_memory(messages)

    assert entry["tool_call_id"] == "call_7"
    assert (entry["call_name"], entry["call_arguments"]) == ("terminal", "{}")


def test_v3_provider_receives_tool_evidence_keyword():
    manager = MemoryManager()
    v3 = _ToolEvidenceProvider("durable")
    manager.add_provider(v3)
    messages = _v3_transcript()
    evidence_v3 = _direct_messages_for_pre_compress_memory(messages, with_ordinals=True)
    tool_evidence = _tool_evidence_for_pre_compress_memory(messages)

    manager.on_pre_compress(
        messages,
        evidence_messages=_direct_messages_for_pre_compress_memory(messages),
        evidence_messages_v3=evidence_v3,
        tool_evidence=tool_evidence,
    )

    assert v3.pre_compress_calls == [(evidence_v3, tool_evidence)]
    assert [m["_ordinal"] for m in v3.pre_compress_calls[0][0]] == [1, 2, 5]


def test_v2_provider_payload_is_byte_identical_to_todays_evidence():
    manager = MemoryManager()
    v2 = _CheckpointProvider("durable")
    manager.add_provider(v2)
    messages = _v3_transcript()
    evidence = _direct_messages_for_pre_compress_memory(messages)

    manager.on_pre_compress(messages, evidence_messages=evidence)

    assert v2.pre_compress_calls == [evidence]
    assert all("_ordinal" not in m for m in v2.pre_compress_calls[0])
    assert all("tool_calls" not in m for m in v2.pre_compress_calls[0])


def test_mixed_manager_keeps_v2_call_unchanged_and_enriches_v3():
    # One external provider per manager; "builtin" is always accepted.
    manager = MemoryManager()
    v2 = _CheckpointProvider("builtin")
    v3 = _ToolEvidenceProvider("durable")
    manager.add_provider(v2)
    manager.add_provider(v3)
    messages = _v3_transcript()
    evidence = _direct_messages_for_pre_compress_memory(messages)
    evidence_v3 = _direct_messages_for_pre_compress_memory(messages, with_ordinals=True)
    tool_evidence = _tool_evidence_for_pre_compress_memory(messages)

    combined = manager.on_pre_compress(
        messages,
        evidence_messages=evidence,
        evidence_messages_v3=evidence_v3,
        tool_evidence=tool_evidence,
        require_checkpoint=True,
    )

    # The v2 stub's positional-only signature rejects any keyword.
    assert v2.pre_compress_calls == [evidence]
    assert all("_ordinal" not in m for m in v2.pre_compress_calls[0])
    assert v3.pre_compress_calls == [(evidence_v3, tool_evidence)]
    assert "builtin context" in combined and "durable context" in combined


def test_v3_ordinals_come_from_one_unfiltered_count():
    messages = _v3_transcript()

    direct = _direct_messages_for_pre_compress_memory(messages, with_ordinals=True)
    tool_evidence = _tool_evidence_for_pre_compress_memory(messages)

    # user=1, assistant(prose+tool_calls)=2, tool=3, summary=4 (skipped), user=5.
    assert [(m["_ordinal"], m["role"]) for m in direct] == [
        (1, "user"),
        (2, "assistant"),
        (5, "user"),
    ]
    assert [e["ordinal"] for e in tool_evidence] == [3]
    assert all("tool_calls" not in m for m in direct)


def test_gate_still_arms_with_a_v2_only_provider():
    manager = MemoryManager()
    manager.add_provider(_CheckpointProvider("durable"))

    assert manager.supports_pre_compress_checkpoint(PRE_COMPRESS_CHECKPOINT_API_VERSION)
    assert not manager.supports_pre_compress_checkpoint(
        PRE_COMPRESS_TOOL_EVIDENCE_API_VERSION
    )
    combined = manager.on_pre_compress(
        [{"role": "user", "content": "evidence"}],
        require_checkpoint=True,
        checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
    )
    assert "durable context" in combined


def test_tool_evidence_skips_malformed_rows_without_raising():
    messages = [
        {"role": "assistant", "content": "x", "tool_calls": "not-a-list"},
        {"role": "assistant", "content": "y", "tool_calls": ["not-a-dict", {"id": "c2"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c3", "function": "bad"}]},
        {"role": "tool", "content": "orphan"},
        "not-a-dict",
        {"role": "tool", "content": "prior", "tool_call_id": "c9", COMPRESSED_SUMMARY_METADATA_KEY: True},
        {"role": "tool", "tool_name": "terminal", "content": "no id"},
        {"role": "tool", "content": "unmatched", "tool_call_id": "c3"},
    ]

    tool_evidence = _tool_evidence_for_pre_compress_memory(messages)

    assert [(e["ordinal"], e["tool_call_id"], e["tool_name"]) for e in tool_evidence] == [
        (7, None, "terminal"),
        (8, "c3", None),
    ]
    assert tool_evidence[1]["call_name"] is None
    assert tool_evidence[1]["call_arguments"] is None


def _compress_context_harness(monkeypatch, provider):
    """MagicMock agent as in test_compress_signal_leak.py, with a real
    ``MemoryManager`` holding ``provider``; records the manager call kwargs."""
    from unittest.mock import MagicMock

    from agent.conversation_compression import compress_context

    agent = MagicMock()
    agent._cached_system_prompt = ""
    agent.tools = None
    agent._build_system_prompt = MagicMock(return_value="sys prompt")
    agent._emit_warning = MagicMock()
    agent._session_db.try_acquire_compression_lock.return_value = True
    agent.context_compressor.compress.return_value = [{"role": "user", "content": "[summary]"}]
    agent.context_compressor.compression_count = 0
    agent.context_compressor.last_compression_rough_tokens = 0
    agent._memory_manager = MemoryManager()
    agent._memory_manager.add_provider(provider)
    monkeypatch.setattr(
        "agent.conversation_compression._compression_lock_holder",
        lambda a: "pid=test:holder",
    )

    seen = {}
    original = MemoryManager.on_pre_compress

    def _recording(self, messages, **kwargs):
        seen.update(kwargs)
        return original(self, messages, **kwargs)

    monkeypatch.setattr(MemoryManager, "on_pre_compress", _recording)
    compress_context(agent, _v3_transcript(), "", approx_tokens=100, force=True)
    return seen


def test_compress_context_hands_v3_payloads_only_to_a_v3_manager(monkeypatch):
    v3 = _ToolEvidenceProvider("durable")
    seen = _compress_context_harness(monkeypatch, v3)

    assert [m["_ordinal"] for m in seen["evidence_messages_v3"]] == [1, 2, 5]
    assert [e["ordinal"] for e in seen["tool_evidence"]] == [3]
    assert v3.pre_compress_calls == [
        (seen["evidence_messages_v3"], seen["tool_evidence"])
    ]

    v2 = _CheckpointProvider("durable")
    seen = _compress_context_harness(monkeypatch, v2)

    assert seen["evidence_messages_v3"] is None
    assert seen["tool_evidence"] is None
    assert v2.pre_compress_calls == [seen["evidence_messages"]]


def test_v3_payloads_never_mutate_the_transcript():
    messages = _v3_transcript()
    user_row, tool_row = messages[0], messages[2]
    snapshot = copy.deepcopy(messages)

    direct = _direct_messages_for_pre_compress_memory(messages, with_ordinals=True)
    _tool_evidence_for_pre_compress_memory(messages)

    assert messages == snapshot
    assert messages[0] is user_row and messages[2] is tool_row
    assert direct[0] is not user_row and "_ordinal" not in user_row


def test_checkpoint_blocked_error_is_prefixed_and_typed():
    error = _checkpoint_blocked("no active provider")
    assert isinstance(error, CompressionCheckpointUnavailable)
    assert str(error).startswith("BLOCKED_MISSING_PREREQUISITE:")
    assert "no active provider" in str(error)


def test_compressed_summary_marker_survives_restart_via_resume_history(tmp_path):
    """The persistent marker reaches the resumed model history — and only it.

    ``get_messages_as_conversation`` keeps its existing marker-free contract;
    the resume path carries ``_compressed_summary`` so checkpoint providers
    keep excluding derivative summaries after a process restart.
    """
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "durable user evidence")
    db.append_message(
        "s1", "assistant", "derivative summary", _compressed_summary=True
    )

    reopened = SessionDB(tmp_path / "state.db")
    model_history, _display = reopened.get_resume_conversations("s1")
    by_content = {m.get("content"): m for m in model_history}
    assert by_content["derivative summary"].get("_compressed_summary") is True
    assert "_compressed_summary" not in by_content["durable user evidence"]

    plain = reopened.get_messages_as_conversation("s1")
    assert all("_compressed_summary" not in m for m in plain)


def test_compressed_summary_column_is_added_to_legacy_databases(tmp_path):
    """Pre-upgrade databases gain the marker column via declarative reconcile.

    ``_init_schema()`` diffs live columns against SCHEMA_SQL on every
    writable open and ADDs whatever is missing, so a database created
    before this feature must accept marker writes after a plain reopen.
    """
    import sqlite3

    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    SessionDB(db_path)

    # Simulate a pre-upgrade database: the marker column does not exist.
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE messages DROP COLUMN _compressed_summary")
    conn.commit()
    legacy_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(messages)")
    }
    conn.close()
    assert "_compressed_summary" not in legacy_cols

    upgraded = SessionDB(db_path)
    upgraded.create_session("legacy", source="cli")
    upgraded.append_message(
        "legacy", "assistant", "derivative summary", _compressed_summary=True
    )

    model_history, _display = upgraded.get_resume_conversations("legacy")
    assert model_history[-1].get("_compressed_summary") is True


def test_native_responses_compaction_is_suppressed_when_checkpoint_required():
    """checkpoint_required must keep ``context_management`` off the wire.

    Server-side native compaction is a lossy boundary the provider owns; no
    pre-compress checkpoint can run before it, so the gate suppresses the
    payload while ordinary checkpoint-aware Hermes compression stays
    available.
    """
    from types import SimpleNamespace

    from agent.native_compaction import native_compaction_context_management

    def agent(checkpoint_required):
        return SimpleNamespace(
            model="gpt-5.6",
            base_url="https://api.openai.com/v1",
            codex_responses_native_compaction=True,
            compression_enabled=True,
            compression_checkpoint_required=checkpoint_required,
            codex_responses_compact_threshold=0.8,
            context_compressor=None,
        )

    assert native_compaction_context_management(
        agent(False), is_codex_backend=True
    )
    assert (
        native_compaction_context_management(agent(True), is_codex_backend=True)
        is None
    )


def test_codex_app_server_turn_fails_closed_before_codex_can_compact():
    """checkpoint_required + app-server must never reach ``run_turn()``.

    The codex agent compacts its own thread; once ``run_turn()`` executes, a
    codex-owned compaction may already have happened with no checkpoint. The
    turn entrypoint must raise first — the session is never even created.
    """
    from types import SimpleNamespace

    from agent.codex_runtime import run_codex_app_server_turn

    class _ExplodingSession:
        def run_turn(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("run_turn() must not be reached")

    agent = SimpleNamespace(
        api_mode="codex_app_server",
        compression_checkpoint_required=True,
        _codex_session=_ExplodingSession(),
    )

    with pytest.raises(CompressionCheckpointUnavailable, match="codex_app_server"):
        run_codex_app_server_turn(
            agent,
            user_message="hello",
            original_user_message="hello",
            messages=[],
            effective_task_id="t1",
        )


def test_agent_init_refuses_checkpoint_required_on_codex_app_server():
    """The incompatible configuration must fail closed at init time.

    In the default "native" auto-compaction mode Hermes never initiates the
    compaction, so the compress_context() guard alone cannot cover native
    turns — init_agent has to refuse before a turn exists.
    """
    from agent.agent_init import (
        _refuse_checkpoint_required_on_codex_app_server,
    )

    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_PREREQUISITE"):
        _refuse_checkpoint_required_on_codex_app_server(True, "codex_app_server")

    # Every other combination stays permitted.
    _refuse_checkpoint_required_on_codex_app_server(True, "chat_completions")
    _refuse_checkpoint_required_on_codex_app_server(True, "codex_responses")
    _refuse_checkpoint_required_on_codex_app_server(False, "codex_app_server")
    _refuse_checkpoint_required_on_codex_app_server(False, None)


def test_turn_finalizer_never_micro_compacts_while_checkpoint_gate_armed(
    monkeypatch,
):
    """Micro-compaction is a lossy rewrite authority with no checkpoint hook.

    Even if a live agent's compressor has ``_micro_compact_enabled`` flipped
    on (agent init forces it off under the gate, but it is plain mutable
    state), the post-turn finalizer must refuse to call ``_micro_compact()``
    while ``compression_checkpoint_required`` is armed — otherwise assistant
    evidence is absorbed into a rolling summary that the checkpoint filter
    later excludes, and the evidence never reaches the durable provider.
    """
    from tests.agent.test_turn_finalizer_final_response_persistence import (
        FakeAgent,
    )
    from agent.turn_finalizer import finalize_turn

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    class _RecordingCompressor:
        _micro_compact_enabled = True

        def __init__(self):
            self.calls = 0

        def _micro_compact(self, messages):
            self.calls += 1
            return list(messages)

    def _run(checkpoint_required: bool):
        agent = FakeAgent()
        compressor = _RecordingCompressor()
        agent.context_compressor = compressor
        agent.compression_checkpoint_required = checkpoint_required
        finalize_turn(
            agent,
            final_response="Done.",
            api_call_count=1,
            interrupted=False,
            failed=False,
            messages=[
                {"role": "user", "content": "do it"},
                {"role": "assistant", "content": "Done."},
            ],
            conversation_history=[],
            effective_task_id="task",
            turn_id="turn",
            user_message="do it",
            original_user_message="do it",
            _should_review_memory=False,
            _turn_exit_reason="completed",
        )
        return compressor.calls

    # Gate armed: micro-compaction never runs.
    assert _run(checkpoint_required=True) == 0

    # Gate off: micro-compaction remains reachable — sabotage control proving
    # this harness genuinely exercises the call site (the finalizer swallows
    # compressor exceptions, so a call counter is the observable signal).
    assert _run(checkpoint_required=False) == 1


def test_agent_init_suppresses_micro_compaction_under_checkpoint_gate():
    """checkpoint_required forces micro-compaction off at init.

    Both keys can be enabled together in config; the gate must win so every
    lossy rewrite passes through the checkpoint-aware batch compressor.
    """
    import inspect

    from agent import agent_init

    source = inspect.getsource(agent_init)
    # The suppression must happen before the compressor attribute assignment.
    suppress_idx = source.find(
        "if compression_checkpoint_required and compression_micro_compact:"
    )
    assign_idx = source.find("_cc._micro_compact_enabled = compression_micro_compact")
    assert suppress_idx != -1, (
        "init_agent must suppress micro-compaction when checkpoint_required"
    )
    assert assign_idx != -1
    assert suppress_idx < assign_idx


# --- Context-engine compaction authority (engine_compacts_outside_compress) --


def _engine_stubs():
    """Build engine stubs against the CURRENTLY imported ContextEngine: sibling
    suites purge ``agent.*`` from sys.modules, and stubs bound to a stale ABC
    would fail isinstance for the wrong reason."""
    from agent.context_engine import ContextEngine

    class _CompressOnlyEngine(ContextEngine):
        """The real third-party shape: should_compress() + compress() only."""

        @property
        def name(self) -> str:
            return "compress-only"

        def update_from_response(self, usage):
            return None

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(
            self,
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            return messages

    class _TurnCompleteEngine(_CompressOnlyEngine):
        """Also takes the post-turn hook, which no checkpoint precedes."""

        def on_turn_complete(self, messages, usage=None, **kwargs):
            return None

    return _CompressOnlyEngine, _TurnCompleteEngine


def test_engine_declaration_defaults_to_undeclared():
    """The declaration is tri-state and inherits ``None``, not a bool: False
    would make every pre-existing engine an implicit waiver, True would refuse
    engines the host can prove safe."""
    from agent.context_engine import ContextEngine

    compress_only, _turn_complete = _engine_stubs()
    assert ContextEngine.compacts_outside_compress is None
    assert compress_only().compacts_outside_compress is None


def test_undeclared_engine_overriding_on_turn_complete_is_refused():
    """An undeclared engine overriding ``on_turn_complete`` fails closed; the
    verdict is the same ``__func__``-vs-ABC-default check the loop performs."""
    from agent.context_engine import engine_compacts_outside_compress

    _compress_only, turn_complete = _engine_stubs()
    unsafe, reason = engine_compacts_outside_compress(turn_complete())
    assert unsafe is True
    assert "on_turn_complete" in reason


def test_compress_only_engine_resolves_safe():
    """compress()-only engines resolve safe, also with the hooks that do not
    count: ``prune_tool_results_only`` (gated at its call site),
    ``select_context`` (request-only), ``on_session_end`` (return discarded)."""
    from agent.context_engine import engine_compacts_outside_compress

    compress_only, _turn_complete = _engine_stubs()
    unsafe, _reason = engine_compacts_outside_compress(compress_only())
    assert unsafe is False

    class _GatedHookEngine(compress_only):
        def prune_tool_results_only(self, messages, current_tokens=None):
            return messages, 0

        def select_context(self, request_messages, **kwargs):
            return request_messages

        def on_session_end(self, session_id, messages):
            return None

    unsafe, _reason = engine_compacts_outside_compress(_GatedHookEngine())
    assert unsafe is False

    # No engine installed is not an engine that compacts.
    assert engine_compacts_outside_compress(None)[0] is False


def test_non_context_engine_object_is_refused():
    """A non-ContextEngine object in the engine slot (the directory loader has
    no isinstance gate) is refused rather than hook-inferred."""
    from types import SimpleNamespace

    from agent.context_engine import (
        ContextEngine,
        engine_compacts_outside_compress,
    )

    unsafe, reason = engine_compacts_outside_compress(SimpleNamespace())
    assert unsafe is True
    assert "ContextEngine" in reason

    # A duck borrowing the ABC's default hooks matches every __func__ identity,
    # so hook inference alone would clear it — the isinstance rule must not.
    class _BorrowedDefaultsDuck:
        name = "duck"
        on_turn_complete = ContextEngine.on_turn_complete

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, **kwargs):
            return messages

    assert engine_compacts_outside_compress(_BorrowedDefaultsDuck())[0] is True


def test_explicit_declaration_overrides_inference():
    """A declaration wins over inference in both directions: a pure observer
    keeps ``on_turn_complete``; an engine with its own scheduler (invisible to
    inference) can declare True."""
    from agent.context_engine import engine_compacts_outside_compress

    compress_only, turn_complete = _engine_stubs()

    class _DeclaredObserver(turn_complete):
        compacts_outside_compress = False

    class _DeclaredScheduler(compress_only):
        compacts_outside_compress = True

    assert engine_compacts_outside_compress(_DeclaredObserver())[0] is False
    assert engine_compacts_outside_compress(_DeclaredScheduler())[0] is True

    # Truthy-but-not-True is not a declaration: plugin engines and MagicMocks
    # answer getattr with truthy auto-attributes.
    class _AutoAttributeEngine(turn_complete):
        compacts_outside_compress = 1

    assert engine_compacts_outside_compress(_AutoAttributeEngine())[0] is True


# --- Proactive tool-result prune (agent/conversation_loop.py) -------------


class _CountingPruneCompressor:
    """Built-in-shaped compressor double: counts instead of asserting, because
    the call site swallows exceptions. Publishes ``would_proactively_prune``
    with a configured trigger by default (a suppression counts only when a
    prune would have run)."""

    def __init__(self, proactive_prune_tokens: int = 48_000):
        self.calls = 0
        self.proactive_prune_tokens = proactive_prune_tokens

    def would_proactively_prune(self, current_tokens=None):
        if self.proactive_prune_tokens <= 0:
            return False
        if current_tokens is not None and current_tokens < self.proactive_prune_tokens:
            return False
        return True

    def prune_tool_results_only(self, messages, current_tokens=None):
        self.calls += 1
        return list(messages) + [{"role": "system", "content": "pruned"}], 3


def _prune_agent(checkpoint_required: bool):
    from types import SimpleNamespace

    return SimpleNamespace(compression_checkpoint_required=checkpoint_required)


def test_proactive_prune_is_suppressed_when_checkpoint_required():
    """The gate reaches the prune and the transcript survives intact:
    suppression, not refusal."""
    from agent.conversation_loop import _proactive_tool_result_prune

    compressor = _CountingPruneCompressor()
    messages = [{"role": "user", "content": "scan"}]
    agent = _prune_agent(True)

    result = _proactive_tool_result_prune(agent, compressor, messages, 400_000)

    assert compressor.calls == 0
    assert result is messages
    assert agent._checkpoint_gate_suppression_count == 1


def test_proactive_prune_still_runs_when_gate_is_off():
    """Sabotage control: gate off, the harness really reaches the prune."""
    from agent.conversation_loop import _proactive_tool_result_prune

    compressor = _CountingPruneCompressor()
    messages = [{"role": "user", "content": "scan"}]

    result = _proactive_tool_result_prune(
        _prune_agent(False), compressor, messages, 400_000
    )

    assert compressor.calls == 1
    assert result is not messages
    assert result[-1]["content"] == "pruned"


def test_engine_override_of_prune_is_suppressed_too():
    """One call site covers the built-in and every engine overriding the hook."""
    from agent.conversation_loop import _proactive_tool_result_prune

    compress_only, _turn_complete = _engine_stubs()

    class _PruningEngine(compress_only):
        calls = 0

        def prune_tool_results_only(self, messages, current_tokens=None):
            type(self).calls += 1
            return list(messages), 3

    engine = _PruningEngine()
    messages = [{"role": "user", "content": "scan"}]
    agent = _prune_agent(True)

    assert _proactive_tool_result_prune(agent, engine, messages, 400_000) is messages
    assert _PruningEngine.calls == 0
    # An overriding engine owns its trigger policy; the host would have
    # dispatched the hook, so this suppression IS reported.
    assert agent._checkpoint_gate_suppression_count == 1

    # Same engine, gate off: the override is reached (sabotage control).
    _proactive_tool_result_prune(_prune_agent(False), engine, messages, 400_000)
    assert _PruningEngine.calls == 1


def test_prune_suppression_logs_once_and_names_the_availability_consequence(
    caplog, monkeypatch,
):
    """One warning per process, naming the trade: with compaction fail-closed,
    a checkpoint-provider outage halts the session."""
    import logging

    from agent import conversation_loop
    from agent.conversation_loop import _proactive_tool_result_prune

    monkeypatch.setattr(
        conversation_loop, "_checkpoint_gate_warned", set(), raising=True
    )
    compressor = _CountingPruneCompressor()
    agent = _prune_agent(True)

    with caplog.at_level(logging.WARNING, logger=conversation_loop.__name__):
        for _ in range(2):
            _proactive_tool_result_prune(agent, compressor, [], 400_000)

    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "checkpoint_required" in r.getMessage()
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "tool-result prune" in message
    assert "halt" in message
    # Suppression itself is re-evaluated every call — only the log is deduped.
    assert agent._checkpoint_gate_suppression_count == 2


def _prune_warnings(caplog):
    import logging

    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "tool-result prune" in r.getMessage()
    ]


def test_disabled_prune_is_not_reported_as_a_suppressed_authority(
    caplog, monkeypatch,
):
    """The shipping default (``proactive_prune_tokens: 0``) suppresses nothing,
    so it must report nothing."""
    import logging

    from agent import conversation_loop
    from agent.conversation_loop import _proactive_tool_result_prune

    monkeypatch.setattr(
        conversation_loop, "_checkpoint_gate_warned", set(), raising=True
    )
    compressor = _CountingPruneCompressor(proactive_prune_tokens=0)
    messages = [{"role": "user", "content": "scan"}]
    agent = _prune_agent(True)

    with caplog.at_level(logging.WARNING, logger=conversation_loop.__name__):
        result = _proactive_tool_result_prune(agent, compressor, messages, 400_000)

    # Still suppressed: the prune is not reached and the transcript is intact.
    assert compressor.calls == 0
    assert result is messages
    # ...but nothing was withheld, so nothing is reported.
    assert getattr(agent, "_checkpoint_gate_suppression_count", 0) == 0
    assert _prune_warnings(caplog) == []


def test_prune_below_its_trigger_is_not_reported_as_a_suppressed_authority(
    caplog, monkeypatch,
):
    """Configured but below the trigger is equally a non-event: the withheld
    call would have returned the input untouched."""
    import logging

    from agent import conversation_loop
    from agent.conversation_loop import _proactive_tool_result_prune

    monkeypatch.setattr(
        conversation_loop, "_checkpoint_gate_warned", set(), raising=True
    )
    compressor = _CountingPruneCompressor(proactive_prune_tokens=400_000)
    agent = _prune_agent(True)

    with caplog.at_level(logging.WARNING, logger=conversation_loop.__name__):
        _proactive_tool_result_prune(agent, compressor, [], 399_999)

    assert getattr(agent, "_checkpoint_gate_suppression_count", 0) == 0
    assert _prune_warnings(caplog) == []

    # One token more and the same compressor IS a suppressed authority: the
    # threshold, not the harness, decided above.
    with caplog.at_level(logging.WARNING, logger=conversation_loop.__name__):
        _proactive_tool_result_prune(agent, compressor, [], 400_000)

    assert agent._checkpoint_gate_suppression_count == 1
    assert len(_prune_warnings(caplog)) == 1


def test_would_proactively_prune_is_the_compressors_own_precondition():
    """The predicate is answered by the real ``ContextCompressor`` (not a
    double), so the host's report cannot drift from the compressor's trigger."""
    from unittest.mock import patch

    from agent.context_compressor import ContextCompressor
    from agent.conversation_loop import _prune_would_have_run

    with patch(
        "agent.context_compressor.get_model_context_length", return_value=1_000_000
    ):
        default = ContextCompressor(model="test", quiet_mode=True)
        configured = ContextCompressor(
            model="test", quiet_mode=True, proactive_prune_tokens=48_000
        )

    # Shipping default: opt-in, so off.
    assert default.proactive_prune_tokens == 0
    assert default.would_proactively_prune(400_000) is False
    assert _prune_would_have_run(default, 400_000) is False

    assert configured.would_proactively_prune(47_999) is False
    assert configured.would_proactively_prune(48_000) is True
    assert _prune_would_have_run(configured, 47_999) is False
    assert _prune_would_have_run(configured, 48_000) is True

    # An unknown token count cannot rule the prune out; the prune proceeds on None.
    assert configured.would_proactively_prune(None) is True


def test_prune_hookless_and_broken_predicate_shapes_are_handled():
    """No hook = no authority; a raising predicate reads as "would have run"
    (the call was withheld) rather than silencing the report."""
    from types import SimpleNamespace

    from agent.conversation_loop import _prune_would_have_run

    assert _prune_would_have_run(SimpleNamespace(), 400_000) is False

    class _BrokenPredicate:
        def would_proactively_prune(self, current_tokens=None):
            raise RuntimeError("engine blew up")

        def prune_tool_results_only(self, messages, current_tokens=None):
            return messages, 0

    assert _prune_would_have_run(_BrokenPredicate(), 400_000) is True


def test_engine_that_never_overrode_the_prune_hook_is_not_a_suppressed_authority():
    """The inherited ABC no-op is not an authority the gate withheld; the
    ``__func__`` identity check tells occupancy from inheritance."""
    from agent.conversation_loop import _prune_would_have_run

    compress_only, _turn_complete = _engine_stubs()

    class _InheritsTheDefault(compress_only):
        pass

    class _OccupiesTheHook(compress_only):
        def prune_tool_results_only(self, messages, current_tokens=None):
            return list(messages), 1

    assert _prune_would_have_run(_InheritsTheDefault(), 400_000) is False
    assert _prune_would_have_run(_OccupiesTheHook(), 400_000) is True


# --- Init refuse at the engine slot (agent/agent_init.py) -----------------


def _init_agent_with_engine(engine, *, checkpoint_required: bool):
    """Drive a real init_agent() with ``engine`` in the context-engine slot
    (harness as in tests/run_agent/test_plugin_context_engine_init.py)."""
    from unittest.mock import patch

    cfg = {
        "context": {"engine": "stub"},
        "compression": {"checkpoint_required": checkpoint_required},
        "agent": {},
    }
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_agent_init_refuses_checkpoint_required_on_uninterceptable_context_engine():
    """The incompatible configuration fails closed at init; a waiver, a gate
    that is off and an empty slot stay permitted."""
    from agent.agent_init import (
        _refuse_checkpoint_required_on_plugin_context_engine,
    )

    _compress_only, turn_complete = _engine_stubs()

    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_PREREQUISITE"):
        _refuse_checkpoint_required_on_plugin_context_engine(True, turn_complete())

    # The resolver's reason survives into the operator-facing text.
    with pytest.raises(RuntimeError, match="on_turn_complete"):
        _refuse_checkpoint_required_on_plugin_context_engine(True, turn_complete())

    class _DeclaredObserver(turn_complete):
        compacts_outside_compress = False

    _refuse_checkpoint_required_on_plugin_context_engine(True, _DeclaredObserver())
    _refuse_checkpoint_required_on_plugin_context_engine(True, _compress_only())
    _refuse_checkpoint_required_on_plugin_context_engine(True, None)
    _refuse_checkpoint_required_on_plugin_context_engine(False, turn_complete())


def test_init_refuse_reads_the_local_config_value_not_the_agent_attribute():
    """The refuse must read the LOCAL config value: the agent attribute is
    assigned after the engine slot, so only a real init_agent() can observe
    the ordering. Negative probe (run by hand, reverted): reading
    ``getattr(agent, "compression_checkpoint_required", False)`` fails only
    this test."""
    _compress_only, turn_complete = _engine_stubs()

    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_PREREQUISITE"):
        _init_agent_with_engine(turn_complete(), checkpoint_required=True)


def test_lcm_style_compress_only_engine_still_initializes_with_gate_armed():
    """The compress()-only third-party shape keeps initializing under the gate."""
    compress_only, _turn_complete = _engine_stubs()

    engine = compress_only()
    agent = _init_agent_with_engine(engine, checkpoint_required=True)

    assert agent.context_compressor is engine
    assert agent.compression_checkpoint_required is True
    # The slot is fully configured, not merely unrefused (#9071 ordering).
    assert engine.context_length == 204_800


def test_refuse_message_survives_a_hostile_name_property():
    """The diagnosis uses the class name; ``engine.name`` is an abstract
    property whose raising getter would replace it. Negative probe (run by
    hand, reverted): ``getattr(engine, "name", ...)`` fails this test with the
    getter's RuntimeError."""
    from agent.agent_init import (
        _refuse_checkpoint_required_on_plugin_context_engine,
    )

    _compress_only, turn_complete = _engine_stubs()

    class _HostileNameEngine(turn_complete):
        @property
        def name(self) -> str:
            raise RuntimeError("name getter exploded")

    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_PREREQUISITE") as excinfo:
        _refuse_checkpoint_required_on_plugin_context_engine(
            True, _HostileNameEngine()
        )

    assert "_HostileNameEngine" in str(excinfo.value)
    assert "name getter exploded" not in str(excinfo.value)


# --- Per-turn gate at the on_turn_complete dispatch ------------------------


def _turn_agent(engine, *, checkpoint_required: bool):
    from types import SimpleNamespace

    return SimpleNamespace(
        context_compressor=engine,
        compression_checkpoint_required=checkpoint_required,
        session_id="s1",
    )


def _counting_hook(calls):
    """A hook that counts instead of asserting: the dispatch swallows
    exceptions, so an assertion inside would pass for the wrong reason."""

    def _hook(messages, usage=None, **meta):
        calls.append(messages)

    return _hook


def _notify(agent, messages=None):
    import logging

    from agent.conversation_loop import _notify_context_engine_turn_complete

    _notify_context_engine_turn_complete(
        agent,
        messages if messages is not None else [{"role": "user", "content": "hi"}],
        usage=None,
        logger=logging.getLogger("test"),
    )


def test_on_turn_complete_attached_after_init_is_suppressed_when_checkpoint_required():
    """The bypass window: an engine that passed the init refuse grows
    ``on_turn_complete`` as an instance attribute (no ``__func__``, so the
    base short-circuit does not catch it); only the per-turn gate can."""
    compress_only, _turn_complete = _engine_stubs()

    engine = compress_only()
    agent = _init_agent_with_engine(engine, checkpoint_required=True)
    assert agent.compression_checkpoint_required is True

    calls = []
    engine.on_turn_complete = _counting_hook(calls)

    _notify(agent)

    assert calls == []
    assert agent._checkpoint_gate_suppression_count == 1


def test_on_turn_complete_still_dispatches_when_gate_is_off():
    """Sabotage control: gate off, a late-attached hook is still dispatched."""
    compress_only, _turn_complete = _engine_stubs()

    engine = compress_only()
    agent = _init_agent_with_engine(engine, checkpoint_required=False)

    calls = []
    engine.on_turn_complete = _counting_hook(calls)

    _notify(agent)

    assert len(calls) == 1


def test_declared_observer_engine_still_receives_on_turn_complete():
    """A pure observer declaring ``compacts_outside_compress = False`` keeps
    receiving the hook under the gate."""
    _compress_only, turn_complete = _engine_stubs()

    calls = []

    class _DeclaredObserver(turn_complete):
        compacts_outside_compress = False

        def on_turn_complete(self, messages, usage=None, **kwargs):
            calls.append(messages)

    agent = _turn_agent(_DeclaredObserver(), checkpoint_required=True)
    _notify(agent)

    assert len(calls) == 1
    assert getattr(agent, "_checkpoint_gate_suppression_count", 0) == 0


def test_on_turn_complete_suppression_logs_once(caplog, monkeypatch):
    """One warning per process; the shared counter keeps rising every turn."""
    import logging

    from agent import conversation_loop

    _compress_only, turn_complete = _engine_stubs()

    monkeypatch.setattr(
        conversation_loop, "_checkpoint_gate_warned", set(), raising=True
    )

    calls = []

    class _CompactingEngine(turn_complete):
        def on_turn_complete(self, messages, usage=None, **kwargs):
            calls.append(messages)

    agent = _turn_agent(_CompactingEngine(), checkpoint_required=True)

    with caplog.at_level(logging.WARNING, logger=conversation_loop.__name__):
        _notify(agent)
        _notify(agent)

    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "checkpoint_required" in r.getMessage()
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "on_turn_complete" in message
    assert "halt" in message

    assert calls == []
    assert agent._checkpoint_gate_suppression_count == 2


def test_turn_gate_fails_closed_when_the_resolver_cannot_be_imported():
    """A broken lazy import (the resolver import dodges a cycle) must fail
    closed. Negative probe (run by hand, reverted): ``_unsafe = False`` in the
    except branch fails this test with the hook dispatched."""
    import sys

    from unittest.mock import patch

    _compress_only, turn_complete = _engine_stubs()

    calls = []

    class _CompactingEngine(turn_complete):
        def on_turn_complete(self, messages, usage=None, **kwargs):
            calls.append(messages)

    agent = _turn_agent(_CompactingEngine(), checkpoint_required=True)

    # A None entry in sys.modules makes the lazy import raise ImportError.
    with patch.dict(sys.modules, {"agent.context_engine": None}):
        _notify(agent)

    assert calls == []
    assert agent._checkpoint_gate_suppression_count == 1


# --- Fork sites inherit the memory provider under an armed gate ------------


def _fork_config(checkpoint_required):
    """Minimal on-disk config shape the five spawn sites read through."""
    return {
        "compression": {"checkpoint_required": checkpoint_required},
        "agent": {},
        "memory": {},
        "delegation": {},
    }


class _ConstructorStop(BaseException):
    """Abort at the child constructor. BaseException, because four of the
    five sites catch ``except Exception``."""


def _fork_site(site):
    """Resolve one fork site lazily: ``(module holding the AIAgent binding,
    spawn thunk, extra patches)``. batch_runner binds AIAgent at import, so
    its patch lands on its own global; the others resolve it via run_agent."""
    from types import SimpleNamespace
    from unittest.mock import patch

    if site == "delegate":
        import run_agent
        import tools.delegate_tool as delegate_tool

        parent = SimpleNamespace(
            model="m", provider="openrouter", api_key="k",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions", session_id="parent",
            enabled_toolsets=None, disabled_toolsets=None,
            request_overrides={}, prefill_messages=None, _delegate_depth=0,
        )
        return run_agent, (lambda: delegate_tool._build_child_agent(
            task_index=0, goal="g", context=None, toolsets=None,
            model=None, max_iterations=3, task_count=1,
            parent_agent=parent,
        )), ()

    if site == "background_review":
        import run_agent
        import agent.background_review as background_review

        parent = SimpleNamespace(
            model="m", provider="openrouter", api_key="k",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions", session_id="parent",
            enabled_toolsets=None, disabled_toolsets=None,
            request_overrides={}, prefill_messages=None, platform="cli",
            quiet_mode=True, _memory_store=None, _memory_enabled=False,
            _user_profile_enabled=False, _skip_mcp_refresh=False,
            _memory_nudge_interval=0, _skill_nudge_interval=0,
            _safe_print=lambda *a, **k: None,
            background_review_callback=None,
            _current_main_runtime=lambda: {},
            _emit_auxiliary_failure=lambda *a, **k: None,
        )
        return run_agent, (lambda: background_review._run_review_in_thread(
            parent, [{"role": "user", "content": "x"}], "prompt"
        )), ()

    if site == "curator":
        import run_agent
        import agent.curator as curator

        return run_agent, (lambda: curator._run_llm_review("prompt")), ()

    if site == "batch_runner":
        import batch_runner

        return batch_runner, (lambda: batch_runner._process_single_prompt(
            0, {"prompt": "p"}, 0,
            {"model": "m", "max_iterations": 3, "distribution": {}},
        )), (
            patch.object(
                batch_runner, "sample_toolsets_from_distribution",
                return_value=["terminal"],
            ),
        )

    if site == "feishu_comment":
        import run_agent
        import plugins.platforms.feishu.feishu_comment as feishu_comment

        return run_agent, (lambda: feishu_comment._run_comment_agent(
            "p", client=object(), session_key=""
        )), (
            patch.object(
                feishu_comment, "_resolve_model_and_runtime",
                return_value=("m", {}),
            ),
            patch.object(feishu_comment, "_load_session_history", return_value=[]),
        )

    raise AssertionError(f"unknown fork site: {site}")  # pragma: no cover


def _capture_fork_kwargs(site, *, checkpoint_required):
    """Drive one real fork site and return the kwargs it passed to AIAgent."""
    from contextlib import ExitStack
    from unittest.mock import patch

    binding, spawn, extra_patches = _fork_site(site)
    captured = {}

    def _recorder(**kwargs):
        captured.update(kwargs)
        raise _ConstructorStop()

    cfg = _fork_config(checkpoint_required)
    with ExitStack() as stack:
        stack.enter_context(patch("hermes_cli.config.load_config", return_value=cfg))
        stack.enter_context(
            patch("hermes_cli.config.load_config_readonly", return_value=cfg)
        )
        stack.enter_context(patch.object(binding, "AIAgent", _recorder))
        for extra in extra_patches:
            stack.enter_context(extra)
        with pytest.raises(_ConstructorStop):
            spawn()

    assert captured, f"{site} never reached the child constructor"
    return captured


_FORK_SITES = (
    "delegate",
    "background_review",
    "curator",
    "batch_runner",
    "feishu_comment",
)


@pytest.mark.parametrize("site", _FORK_SITES)
def test_fork_inherits_memory_provider_when_checkpoint_required(site):
    """An armed gate hands every fork the provider it must checkpoint with;
    a provider-less fork would raise ``CompressionCheckpointUnavailable`` at
    its first compaction."""
    captured = _capture_fork_kwargs(site, checkpoint_required=True)

    assert captured["skip_memory"] is False
    # The value agent/memory_provider.py already names; a fresh string would
    # fall out of providers' non-primary write gates.
    assert captured["memory_agent_context"] == "subagent"


@pytest.mark.parametrize("site", _FORK_SITES)
def test_fork_stays_provider_less_when_gate_is_off(site):
    """Gate off (the default) stays provider-less: the control against
    "always inherit"."""
    captured = _capture_fork_kwargs(site, checkpoint_required=False)

    assert captured["skip_memory"] is True


def test_checkpoint_required_from_config_reads_the_armed_key():
    """The shared resolver the five sites call before an agent exists;
    ``is True`` rather than truthiness, like the gate's other read sites."""
    from unittest.mock import patch

    from agent.agent_init import checkpoint_required_from_config

    def _with(cfg):
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            return checkpoint_required_from_config()

    assert _with({"compression": {"checkpoint_required": True}}) is True
    assert _with({"compression": {"checkpoint_required": "yes"}}) is True
    assert _with({"compression": {"checkpoint_required": False}}) is False
    assert _with({"compression": {}}) is False
    assert _with({}) is False
    # A config section of the wrong shape must not raise out of a spawn path.
    assert _with({"compression": "nonsense"}) is False
    assert _with(None) is False

    # An unreadable config reads as "not armed", matching init_agent.
    with patch(
        "hermes_cli.config.load_config_readonly",
        side_effect=RuntimeError("config unreadable"),
    ):
        assert checkpoint_required_from_config() is False


def test_memory_agent_context_reaches_the_provider_initialize():
    """The label survives the whole constructor path to the provider's
    ``initialize()``; driven through a real ``init_agent()``."""
    from unittest.mock import patch

    seen = {}

    class _RecordingProvider(MemoryProvider):
        pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

        @property
        def name(self):
            return "recording"

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            seen.clear()
            seen.update(kwargs)

        def get_tool_schemas(self):
            return []

    cfg = {
        "compression": {},
        "agent": {},
        "memory": {
            "provider": "recording",
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
    }

    def _build(**extra):
        with (
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("hermes_cli.config.load_config_readonly", return_value=cfg),
            patch(
                "plugins.memory.load_memory_provider",
                return_value=_RecordingProvider(),
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=204_800,
            ),
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            from run_agent import AIAgent

            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                **extra,
            )
        return seen.get("agent_context")

    assert _build(memory_agent_context="subagent") == "subagent"
    # Unset, blank or non-str falls back to the historical "primary".
    assert _build() == "primary"
    assert _build(memory_agent_context="   ") == "primary"
    assert _build(memory_agent_context=None) == "primary"
