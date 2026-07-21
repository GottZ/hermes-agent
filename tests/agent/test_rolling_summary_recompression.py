"""Regression contracts for rolling-summary recompression."""

from agent.context_compressor import (
    ContextCompressor,
    LEGACY_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from hermes_state import SessionDB


def _messages_for_compression():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first decision " + "x" * 200},
        {"role": "assistant", "content": "first response " + "y" * 200},
        {"role": "user", "content": "second decision " + "z" * 200},
        {"role": "assistant", "content": "second response " + "q" * 200},
        {"role": "user", "content": "recent request"},
        {"role": "assistant", "content": "recent response"},
    ]


def test_protected_user_text_with_legacy_summary_prefix_is_not_removed(monkeypatch):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    monkeypatch.setattr(
        compressor,
        "_generate_summary",
        lambda _turns, focus_topic=None, memory_context="": "## Goal\nUpdated handoff",
    )
    messages = _messages_for_compression() + [
        {"role": "user", "content": "more context " + "a" * 400},
        {"role": "assistant", "content": "more response " + "b" * 400},
    ]
    literal_user_text = f"{LEGACY_SUMMARY_PREFIX} this is literal user content"
    messages[1]["content"] = literal_user_text

    result = compressor.compress(messages)

    assert any(message.get("content") == literal_user_text for message in result)
    assert compressor._previous_summary != "this is literal user content"


def test_unmarked_canonical_prefix_later_in_history_is_treated_as_user_text(
    monkeypatch,
):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    seen_turns = []

    def summarize(turns, focus_topic=None, memory_context=""):
        seen_turns.extend(turns)
        return "## Goal\nUpdated handoff"

    monkeypatch.setattr(compressor, "_generate_summary", summarize)
    literal_user_text = f"{SUMMARY_PREFIX}\nliteral later user request"
    messages = _messages_for_compression() + [
        {"role": "user", "content": "more context " + "a" * 400},
        {"role": "assistant", "content": "more response " + "b" * 400},
    ]
    messages[3]["content"] = literal_user_text

    compressor.compress(messages)

    assert any(message.get("content") == literal_user_text for message in seen_turns)
    assert compressor._previous_summary != "literal later user request"


def test_textual_end_marker_later_in_user_history_is_not_summary_provenance(
    monkeypatch,
):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    seen_turns = []

    def summarize(turns, focus_topic=None, memory_context=""):
        seen_turns.extend(turns)
        return "## Goal\nUpdated handoff"

    monkeypatch.setattr(compressor, "_generate_summary", summarize)
    literal_user_text = (
        f"{SUMMARY_PREFIX}\n## Goal\nQuoted summary-shaped request\n\n"
        f"{_SUMMARY_END_MARKER}"
    )
    messages = _messages_for_compression() + [
        {"role": "user", "content": "more context " + "a" * 400},
        {"role": "assistant", "content": "more response " + "b" * 400},
    ]
    messages[3]["content"] = literal_user_text

    compressor.compress(messages)

    assert any(message.get("content") == literal_user_text for message in seen_turns)
    assert "Quoted summary-shaped request" not in (compressor._previous_summary or "")


def test_structured_premetadata_handoff_in_first_slot_is_reused(monkeypatch):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    observed_prior = []

    def summarize(_turns, focus_topic=None, memory_context=""):
        observed_prior.append(compressor._previous_summary)
        return "## Goal\nRenormalized handoff"

    monkeypatch.setattr(compressor, "_generate_summary", summarize)
    assert LEGACY_SUMMARY_PREFIX == "[CONTEXT SUMMARY]:"
    assert _SUMMARY_END_MARKER == (
        "--- END OF CONTEXT SUMMARY — "
        "respond to the message below, not the summary above ---"
    )
    old_handoff = (
        "[CONTEXT SUMMARY]:\n## Historical Task Snapshot\n"
        "User asked: 'historical task A'\n\n## Goal\nHistorical goal"
    )
    messages = _messages_for_compression() + [
        {"role": "user", "content": "more context " + "a" * 400},
        {"role": "assistant", "content": "more response " + "b" * 400},
    ]
    messages[1]["content"] = old_handoff

    result = compressor.compress(messages)

    assert observed_prior and "historical task A" in observed_prior[0]
    assert all(message.get("content") != old_handoff for message in result)
    assert sum(bool(message.get("_compressed_summary")) for message in result) == 1


def test_protected_merged_summary_restores_prior_tail_content(monkeypatch):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    monkeypatch.setattr(
        compressor,
        "_generate_summary",
        lambda _turns, focus_topic=None, memory_context="": "## Goal\nResumed handoff",
    )
    protected_tail = "protected tail fact must survive"
    merged_content = (
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{protected_tail}\n\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n\n{SUMMARY_PREFIX}\n"
        f"## Goal\nOld handoff\n\n{_SUMMARY_END_MARKER}"
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "protected opening"},
        {"role": "assistant", "content": "protected response"},
        {"role": "user", "content": merged_content, "_compressed_summary": True},
    ] + [
        {
            "role": "assistant" if index % 2 == 0 else "user",
            "content": f"new cycle {index} " + "x" * 400,
        }
        for index in range(12)
    ]

    result = compressor.compress(messages)

    rendered = "\n".join(str(message.get("content", "")) for message in result)
    assert protected_tail in rendered
    assert "Old handoff" not in rendered
    visible_roles = [
        message.get("role") for message in result if message.get("role") != "system"
    ]
    assert all(left != right for left, right in zip(visible_roles, visible_roles[1:]))


def test_real_db_resume_rehydrates_marker_before_second_compression(
    monkeypatch, tmp_path
):
    first = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
    )
    first.tail_token_budget = 30
    monkeypatch.setattr(
        first,
        "_generate_summary",
        lambda _turns, focus_topic=None, memory_context="": (
            "## Goal\nFirst durable summary"
        ),
    )
    source = _messages_for_compression()
    compacted_once = first.compress(source)

    db = SessionDB(tmp_path / "state.db")
    db.create_session("rolling-resume", source="test")
    db.replace_messages("rolling-resume", source)
    db.archive_and_compact("rolling-resume", compacted_once)
    resumed = db.get_messages_as_conversation("rolling-resume")

    markers = [message for message in resumed if message.get("_compressed_summary")]
    assert len(markers) == 1
    assert "First durable summary" in str(markers[0]["content"])

    second = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
    )
    second.tail_token_budget = 30
    observed_prior = []

    def second_summary(_turns, focus_topic=None, memory_context=""):
        observed_prior.append(second._previous_summary)
        return "## Goal\nSecond durable summary"

    monkeypatch.setattr(second, "_generate_summary", second_summary)
    resumed.extend(
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"post-resume turn {index} " + "x" * 300,
        }
        for index in range(10)
    )
    compacted_twice = second.compress(resumed)

    assert observed_prior and "First durable summary" in observed_prior[0]
    assert (
        sum(bool(message.get("_compressed_summary")) for message in compacted_twice)
        == 1
    )
    rendered = "\n".join(str(message.get("content", "")) for message in compacted_twice)
    assert "Second durable summary" in rendered
    assert "First durable summary" not in rendered

    db.archive_and_compact("rolling-resume", compacted_twice)
    reloaded = db.get_messages_as_conversation("rolling-resume")
    assert sum(bool(message.get("_compressed_summary")) for message in reloaded) == 1
