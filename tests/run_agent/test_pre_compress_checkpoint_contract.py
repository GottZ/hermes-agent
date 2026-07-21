"""Contracts for fail-closed checkpoints before lossy compression."""

import inspect
import os
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression import (
    COMPACTION_STATUS,
    CompressionCheckpointUnavailable,
)
from agent.memory_manager import MemoryManager
from agent.memory_provider import (
    MemoryProvider,
    PRE_COMPRESS_CHECKPOINT_API_VERSION,
    PRE_COMPRESS_CHECKPOINT_SUMMARY_TOKEN_KEY,
)
from hermes_cli.config import validate_raw_compression_checkpoint_config
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_state import SessionDB


class RecordingMemoryManager:
    def __init__(self, *, error=None, compatible=True, events=None):
        self.error = error
        self.compatible = compatible
        self.events = events
        self.calls = []
        self.probes = []
        self.switches = []

    def supports_pre_compress_checkpoint(self, api_version=1):
        self.probes.append(api_version)
        return self.compatible

    def on_pre_compress(
        self,
        messages,
        *,
        require_checkpoint=False,
        checkpoint_api_version=1,
    ):
        self.calls.append((messages, require_checkpoint, checkpoint_api_version))
        if self.events is not None:
            self.events.append("checkpoint")
        if self.error is not None:
            raise self.error
        return "checkpoint receipt"

    def on_session_switch(self, new_session_id, **kwargs):
        self.switches.append((new_session_id, kwargs))

    def bind_pre_compress_checkpoint_session(self, new_session_id, **kwargs):
        self.switches.append((new_session_id, kwargs))


class CheckpointOnlyProvider(MemoryProvider):
    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

    def __init__(self, *, initialize_error=None, switch_error=None):
        self.initialize_calls = []
        self.switch_calls = []
        self.initialize_error = initialize_error
        self.switch_error = switch_error

    @property
    def name(self):
        return "checkpoint-only-test"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.initialize_calls.append((session_id, kwargs))
        if self.initialize_error is not None:
            raise self.initialize_error

    def on_session_switch(self, new_session_id, **kwargs):
        self.switch_calls.append((new_session_id, kwargs))
        if self.switch_error is not None:
            raise self.switch_error

    def get_tool_schemas(self):
        return []

    def on_pre_compress(self, messages):
        return "checkpoint-only receipt"


def _build_agent():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent: Any = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compress.__signature__ = inspect.signature(
        lambda messages, *, current_tokens, focus_topic=None, force=False, memory_context="", checkpoint_summary_token="": (
            None
        )
    )

    def default_compress(_messages, *, checkpoint_summary_token="", **_kwargs):
        return [
            {
                "role": "assistant",
                "content": "[CONTEXT COMPACTION] summary",
                "_compressed_summary": True,
                PRE_COMPRESS_CHECKPOINT_SUMMARY_TOKEN_KEY: checkpoint_summary_token,
            },
            {"role": "user", "content": "tail"},
        ]

    compressor.compress.side_effect = default_compress
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor._last_compression_made_progress = True
    compressor._last_summary_fallback_used = False
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    return agent, compressor


def _full_transcript():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "direct user evidence"},
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool evidence"},
        {"role": "assistant", "content": "direct assistant evidence"},
        {
            "role": "assistant",
            "content": "old summary",
            "_compressed_summary": True,
        },
    ]


def test_checkpoint_required_defaults_false():
    agent, _ = _build_agent()

    assert agent.compression_checkpoint_required is False


def test_checkpoint_required_reads_config():
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
        patch(
            "hermes_cli.config.load_config",
            return_value={"compression": {"checkpoint_required": True}},
        ),
    ):
        from run_agent import AIAgent

        agent: Any = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.compression_checkpoint_required is True


def test_required_skip_memory_agent_initializes_dedicated_checkpoint_manager():
    provider = CheckpointOnlyProvider()
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
        patch(
            "hermes_cli.config.load_config",
            return_value={
                "compression": {"checkpoint_required": True},
                "memory": {"provider": "checkpoint-only-test"},
            },
        ),
        patch("plugins.memory.load_memory_provider", return_value=provider),
    ):
        from run_agent import AIAgent

        agent: Any = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            platform="subagent",
            session_id="child-session",
            parent_session_id="parent-session",
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent._memory_manager is None
    assert agent._compression_checkpoint_manager.providers == [provider]
    assert provider.initialize_calls == [
        (
            "child-session",
            {
                "platform": "subagent",
                "hermes_home": str(get_hermes_home()),
                "agent_context": "checkpoint_only",
                "parent_session_id": "parent-session",
            },
        )
    ]


def test_checkpoint_required_blocks_codex_app_server_runtime():
    agent, compressor = _build_agent()
    agent.api_mode = "codex_app_server"
    agent.compression_checkpoint_required = True

    with pytest.raises(
        CompressionCheckpointUnavailable,
        match="authoritative thread",
    ):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    compressor.compress.assert_not_called()


def test_required_checkpoint_receives_full_fidelity_source_before_compressor():
    events = []
    agent, compressor = _build_agent()
    manager = RecordingMemoryManager(events=events)
    agent._memory_manager = manager
    agent.compression_checkpoint_required = True
    agent._emit_status = MagicMock(side_effect=lambda _message: events.append("status"))
    compressor.compress.side_effect = lambda *_args, **_kwargs: (
        events.append("compressor")
        or [
            {
                "role": "assistant",
                "content": "[CONTEXT COMPACTION] summary",
                "_compressed_summary": True,
                PRE_COMPRESS_CHECKPOINT_SUMMARY_TOKEN_KEY: _kwargs[
                    "checkpoint_summary_token"
                ],
            },
            {"role": "user", "content": "tail"},
        ]
    )

    compressed, _system_prompt = agent._compress_context(
        _full_transcript(), "system", approx_tokens=100_000
    )

    assert manager.probes == [1]
    assert manager.calls == [(_full_transcript(), True, 1)]
    assert events == ["checkpoint", "status", "compressor"]
    agent._emit_status.assert_called_once_with(COMPACTION_STATUS)
    assert (
        "[PRE-COMPRESSION CHECKPOINT RECEIPT]\ncheckpoint receipt"
        in compressed[0]["content"]
    )
    assert PRE_COMPRESS_CHECKPOINT_SUMMARY_TOKEN_KEY not in compressed[0]


def test_required_compaction_uses_session_scoped_checkpoint_only_manager():
    agent, compressor = _build_agent()
    manager = RecordingMemoryManager()
    agent.session_id = "active-child-session"
    agent._parent_session_id = "parent-session"
    agent._memory_manager = None
    agent._compression_checkpoint_manager = manager
    agent.compression_checkpoint_required = True

    agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    assert manager.switches == [
        (
            "active-child-session",
            {
                "parent_session_id": "parent-session",
                "reset": False,
                "reason": "checkpoint",
                "checkpoint_api_version": PRE_COMPRESS_CHECKPOINT_API_VERSION,
            },
        )
    ]
    assert manager.probes == [PRE_COMPRESS_CHECKPOINT_API_VERSION]
    assert len(manager.calls) == 1
    compressor.compress.assert_called_once()


def test_checkpoint_only_manager_is_closed_without_memory_extraction():
    agent, _compressor = _build_agent()
    manager = MagicMock()
    agent._memory_manager = None
    agent._compression_checkpoint_manager = manager

    agent.shutdown_memory_provider(_full_transcript())

    manager.shutdown_all.assert_called_once_with()
    manager.on_session_end.assert_not_called()


def test_checkpoint_only_session_binding_failure_blocks_compression():
    agent, compressor = _build_agent()
    provider = CheckpointOnlyProvider(switch_error=RuntimeError("bind failed"))
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all("original-session")
    agent._memory_manager = None
    agent._compression_checkpoint_manager = manager
    agent.compression_checkpoint_required = True
    agent.session_id = "active-child-session"

    with pytest.raises(
        CompressionCheckpointUnavailable,
        match="checkpoint provider session binding failed",
    ):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    assert provider.switch_calls == [
        (
            "active-child-session",
            {
                "parent_session_id": getattr(agent, "_parent_session_id", "") or "",
                "reset": False,
                "reason": "checkpoint",
            },
        )
    ]
    compressor.compress.assert_not_called()


def test_required_initialization_failure_removes_provider_capability():
    provider = CheckpointOnlyProvider(initialize_error=RuntimeError("init failed"))
    manager = MemoryManager()
    manager.add_provider(provider)

    with pytest.raises(RuntimeError, match="checkpoint-only-test"):
        manager.initialize_all("child-session", require_success=True)

    assert not manager.supports_pre_compress_checkpoint(
        PRE_COMPRESS_CHECKPOINT_API_VERSION
    )
    assert provider not in manager.providers


def test_required_checkpoint_failure_blocks_compressor_and_status():
    agent, compressor = _build_agent()
    agent._memory_manager = RecordingMemoryManager(
        error=RuntimeError("durability unavailable")
    )
    agent.compression_checkpoint_required = True
    agent._emit_status = MagicMock()

    with pytest.raises(
        CompressionCheckpointUnavailable,
        match="provider checkpoint API v1 failed",
    ):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    compressor.compress.assert_not_called()
    agent._emit_status.assert_not_called()


def test_required_checkpoint_failure_releases_real_session_lock(tmp_path):
    agent, compressor = _build_agent()
    session_db = SessionDB(tmp_path / "state.db")
    session_db.create_session("checkpoint-lock", source="test")
    agent.session_db = session_db
    agent.session_id = "checkpoint-lock"
    agent._memory_manager = RecordingMemoryManager(
        error=RuntimeError("provider failed")
    )
    agent.compression_checkpoint_required = True

    with pytest.raises(CompressionCheckpointUnavailable):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    compressor.compress.assert_not_called()
    assert session_db.get_compression_lock_holder("checkpoint-lock") is None
    assert session_db.try_acquire_compression_lock("checkpoint-lock", "second-attempt")
    session_db.release_compression_lock("checkpoint-lock", "second-attempt")


def test_required_checkpoint_without_compatible_provider_blocks_compressor():
    agent, compressor = _build_agent()
    agent._memory_manager = RecordingMemoryManager(compatible=False)
    agent.compression_checkpoint_required = True

    with pytest.raises(
        CompressionCheckpointUnavailable,
        match="does not implement checkpoint API v1",
    ):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    compressor.compress.assert_not_called()


def test_required_checkpoint_without_memory_manager_blocks_compressor():
    agent, compressor = _build_agent()
    agent._memory_manager = None
    agent.compression_checkpoint_required = True

    with pytest.raises(
        CompressionCheckpointUnavailable,
        match="no active provider implements checkpoint API v1",
    ):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    compressor.compress.assert_not_called()


def test_required_checkpoint_rejects_oversized_receipt_before_status():
    agent, compressor = _build_agent()
    manager = RecordingMemoryManager()
    manager.on_pre_compress = MagicMock(return_value="x" * 6_001)
    agent._memory_manager = manager
    agent.compression_checkpoint_required = True
    agent._emit_status = MagicMock()

    with pytest.raises(CompressionCheckpointUnavailable, match="lossless host limit"):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    compressor.compress.assert_not_called()
    agent._emit_status.assert_not_called()


def test_required_checkpoint_rejects_implicit_kwargs_engine_capability():
    agent, compressor = _build_agent()
    agent._memory_manager = RecordingMemoryManager()
    agent.compression_checkpoint_required = True
    agent._emit_status = MagicMock()
    compressor.compress.__signature__ = inspect.signature(
        lambda messages, *, current_tokens, **kwargs: None
    )

    with pytest.raises(
        CompressionCheckpointUnavailable, match="explicit memory_context"
    ):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    compressor.compress.assert_not_called()
    agent._emit_status.assert_not_called()


def test_required_checkpoint_rejects_engine_without_summary_provenance_support():
    agent, compressor = _build_agent()
    agent._memory_manager = RecordingMemoryManager()
    agent.compression_checkpoint_required = True
    agent._emit_status = MagicMock()
    compressor.compress.__signature__ = inspect.signature(
        lambda messages, *, current_tokens, memory_context="": None
    )

    with pytest.raises(
        CompressionCheckpointUnavailable, match="summary provenance support"
    ):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    compressor.compress.assert_not_called()
    agent._emit_status.assert_not_called()


def test_required_checkpoint_rejects_unmarked_compression_result():
    agent, compressor = _build_agent()
    agent._memory_manager = RecordingMemoryManager()
    agent.compression_checkpoint_required = True
    compressor.compress.side_effect = lambda *_args, **_kwargs: [
        {"role": "assistant", "content": "summary without identity"},
        {"role": "user", "content": "tail"},
    ]

    with pytest.raises(CompressionCheckpointUnavailable, match="no marked summary"):
        agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    compressor.compress.assert_called_once()


def test_required_checkpoint_rejects_stale_input_summary_as_receipt_anchor():
    agent, compressor = _build_agent()
    agent._memory_manager = RecordingMemoryManager()
    agent.compression_checkpoint_required = True
    source = _full_transcript()
    stale_summary = next(
        message for message in source if message.get("_compressed_summary")
    )
    compressor.compress.side_effect = lambda *_args, **_kwargs: [
        dict(stale_summary),
        {"role": "assistant", "content": "fresh but unmarked summary"},
        {"role": "user", "content": "tail"},
    ]

    with pytest.raises(CompressionCheckpointUnavailable, match="no marked summary"):
        agent._compress_context(source, "system", approx_tokens=100_000)

    assert "CHECKPOINT RECEIPT" not in stale_summary["content"]
    compressor.compress.assert_called_once()


def test_required_checkpoint_rejects_modified_stale_summary_as_receipt_anchor():
    agent, compressor = _build_agent()
    agent._memory_manager = RecordingMemoryManager()
    agent.compression_checkpoint_required = True
    source = _full_transcript()
    stale_summary = next(
        message for message in source if message.get("_compressed_summary")
    )
    modified_stale_summary = dict(stale_summary)
    modified_stale_summary["content"] += "\nmodified by engine"
    compressor.compress.side_effect = lambda *_args, **_kwargs: [
        modified_stale_summary,
        {"role": "assistant", "content": "fresh but unmarked summary"},
        {"role": "user", "content": "tail"},
    ]

    with pytest.raises(CompressionCheckpointUnavailable, match="no marked summary"):
        agent._compress_context(source, "system", approx_tokens=100_000)

    assert "CHECKPOINT RECEIPT" not in modified_stale_summary["content"]
    compressor.compress.assert_called_once()


def test_required_checkpoint_allows_semantic_noop_without_marked_summary():
    agent, compressor = _build_agent()
    manager = RecordingMemoryManager()
    agent._memory_manager = manager
    agent.compression_checkpoint_required = True
    agent._emit_status = MagicMock()
    original = _full_transcript()
    compressor._last_compression_made_progress = False
    compressor.compress.side_effect = lambda compression_input, **_kwargs: [
        dict(message) for message in compression_input
    ]

    compressed, _system_prompt = agent._compress_context(
        original, "system", approx_tokens=100_000
    )

    assert compressed == original
    assert manager.probes == [PRE_COMPRESS_CHECKPOINT_API_VERSION]
    assert len(manager.calls) == 1
    compressor.compress.assert_called_once()
    agent._emit_status.assert_called_once_with(COMPACTION_STATUS)


def test_required_checkpoint_isolates_live_transcript_from_mutating_engine():
    agent, compressor = _build_agent()
    agent._memory_manager = RecordingMemoryManager()
    agent.compression_checkpoint_required = True
    messages = _full_transcript()
    original = _full_transcript()
    seen_inputs = []

    def mutate_then_fail(compression_input, **_kwargs):
        seen_inputs.append(compression_input)
        compression_input[1]["content"] = "MUTATED-BEFORE-FAIL-CLOSED"
        compression_input[2]["tool_calls"][0]["id"] = "mutated-call-id"
        compression_input.append({"role": "user", "content": "injected"})
        return [{"role": "assistant", "content": "unmarked result"}]

    compressor.compress.side_effect = mutate_then_fail

    with pytest.raises(CompressionCheckpointUnavailable, match="no marked summary"):
        agent._compress_context(messages, "system", approx_tokens=100_000)

    assert len(seen_inputs) == 1
    assert seen_inputs[0] is not messages
    assert seen_inputs[0] != original
    assert messages == original
    assert messages[2]["tool_calls"][0]["id"] == "call-1"


def test_optional_checkpoint_keeps_legacy_best_effort_behavior():
    agent, compressor = _build_agent()
    manager = RecordingMemoryManager(error=RuntimeError("legacy failure"))
    agent._memory_manager = manager
    agent.compression_checkpoint_required = False

    agent._compress_context(_full_transcript(), "system", approx_tokens=100_000)

    assert manager.probes == []
    assert manager.calls == [(_full_transcript(), False, 1)]
    compressor.compress.assert_called_once()


def test_required_manager_isolates_each_provider_and_live_transcript():
    messages = _full_transcript()
    seen = []

    class Provider:
        pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

        def __init__(self, name, mutate=False):
            self.name = name
            self.mutate = mutate

        def on_pre_compress(self, payload):
            seen.append((self.name, payload))
            if self.mutate:
                payload.append({"role": "user", "content": "provider mutation"})
            return f"receipt:{self.name}"

    manager = MemoryManager()
    manager._providers = cast(
        list[MemoryProvider], [Provider("mutator", True), Provider("observer")]
    )

    manager.on_pre_compress(messages, require_checkpoint=True)

    assert messages == _full_transcript()
    assert seen[1] == ("observer", _full_transcript())


def test_required_manager_requires_all_compatible_providers_to_succeed():
    calls = []

    class Provider:
        pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def on_pre_compress(self, payload):
            calls.append((self.name, payload))
            if self.fail:
                raise RuntimeError("provider-private-error")
            return f"receipt:{self.name}"

    manager = MemoryManager()
    manager._providers = cast(
        list[MemoryProvider], [Provider("first", True), Provider("second")]
    )

    with pytest.raises(RuntimeError, match="first") as exc_info:
        manager.on_pre_compress(_full_transcript(), require_checkpoint=True)

    assert "provider-private-error" not in str(exc_info.value)
    assert [name for name, _payload in calls] == ["first", "second"]


def test_checkpoint_api_version_rejects_bool_and_string_aliases():
    class Provider:
        name = "misconfigured"

        def __init__(self, version):
            self.pre_compress_checkpoint_api_version = version

    manager = MemoryManager()
    for version in (True, "1"):
        manager._providers = cast(list[MemoryProvider], [Provider(version)])
        assert manager.supports_pre_compress_checkpoint() is False
        with pytest.raises(RuntimeError, match="No active memory provider"):
            manager.on_pre_compress(_full_transcript(), require_checkpoint=True)


@pytest.mark.parametrize(
    ("yaml_text", "error_match"),
    [
        ("compression:\n  checkpoint_required: treu\n", "must be a boolean"),
        ("compression:\n  checkpoint_required: null\n", "must be a boolean"),
        ("compression:\n  checkpoint_requiredd: true\n", "unknown .*checkpoint key"),
        ("compression: []\n", "must be a mapping"),
    ],
)
def test_real_raw_config_rejects_checkpoint_policy_downgrades(
    tmp_path, yaml_text, error_match
):
    token = set_hermes_home_override(tmp_path)
    try:
        (tmp_path / "config.yaml").write_text(yaml_text, encoding="utf-8")
        with pytest.raises(ValueError, match=error_match):
            validate_raw_compression_checkpoint_config()
    finally:
        reset_hermes_home_override(token)


def test_real_raw_config_accepts_boolean_checkpoint_policy(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        (tmp_path / "config.yaml").write_text(
            "compression:\n  checkpoint_required: true\n", encoding="utf-8"
        )
        validate_raw_compression_checkpoint_config()
    finally:
        reset_hermes_home_override(token)
