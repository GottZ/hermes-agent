"""Behavior contracts for durable compressed-summary identity."""

import contextlib
import inspect
import os
import threading
from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
)
from hermes_state import SessionDB


def _summary() -> dict:
    return {
        "role": "assistant",
        "content": "opaque compressed handoff without a legacy prefix",
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }


def test_compressed_summary_marker_preserves_append_message_positional_abi():
    parameters = inspect.signature(SessionDB.append_message).parameters

    assert parameters["timestamp"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["_compressed_summary"].kind is inspect.Parameter.KEYWORD_ONLY


def test_gateway_transcript_adapter_forwards_compressed_summary_marker():
    from gateway.session import SessionStore

    store = object.__new__(SessionStore)
    store._db = MagicMock()

    store._append_transcript_message("session-a", _summary())

    assert store._db.append_message.call_args.kwargs["_compressed_summary"] is True


def test_gateway_transcript_adapter_round_trips_marker_with_real_db(tmp_path):
    from gateway.session import SessionStore

    db = SessionDB(tmp_path / "gateway-state.db")
    db.create_session("gateway-session", source="gateway")
    store = object.__new__(SessionStore)
    store._db = db

    store._append_transcript_message("gateway-session", _summary())

    restored = db.get_messages_as_conversation("gateway-session")
    assert restored[0][COMPRESSED_SUMMARY_METADATA_KEY] is True


def test_tui_branch_seed_forwards_compressed_summary_marker(monkeypatch):
    from tui_gateway import server

    db = MagicMock()

    @contextlib.contextmanager
    def session_db(_session):
        yield db

    monkeypatch.setattr(server, "_session_db", session_db)
    session = {
        "parent_session_id": "parent",
        "session_key": "branch",
        "history": [_summary()],
        "history_lock": threading.RLock(),
    }

    server._persist_branch_seed(session)

    assert db.append_message.call_args.kwargs["_compressed_summary"] is True


def test_tui_branch_seed_round_trips_marker_with_real_db(monkeypatch, tmp_path):
    from tui_gateway import server

    db = SessionDB(tmp_path / "tui-state.db")
    db.create_session("parent", source="tui")
    db.create_session("branch-key", source="tui", parent_session_id="parent")

    @contextlib.contextmanager
    def session_db(_session):
        yield db

    monkeypatch.setattr(server, "_session_db", session_db)
    session = {
        "parent_session_id": "parent",
        "session_key": "branch-key",
        "session_id": "branch",
        "history": [_summary()],
        "history_lock": threading.RLock(),
    }

    server._persist_branch_seed(session)

    restored = db.get_messages_as_conversation("branch-key")
    assert restored[0][COMPRESSED_SUMMARY_METADATA_KEY] is True


def test_append_message_round_trips_compressed_summary_marker(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="session-a", source="test")
        summary = _summary()

        message_id = db.append_message(
            "session-a",
            role=summary["role"],
            content=summary["content"],
            _compressed_summary=True,
        )

        assert db.get_messages("session-a")[0][COMPRESSED_SUMMARY_METADATA_KEY] is True
        resumed = db.get_messages_as_conversation("session-a")[0]
        assert resumed[COMPRESSED_SUMMARY_METADATA_KEY] is True
        assert ContextCompressor._has_compressed_summary_metadata(resumed)
        assert (
            db.get_messages_around("session-a", message_id)["window"][0][
                COMPRESSED_SUMMARY_METADATA_KEY
            ]
            is True
        )
    finally:
        db.close()


def test_replace_messages_round_trips_compressed_summary_marker(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="session-a", source="test")

        db.replace_messages("session-a", [_summary()])

        assert db.get_messages("session-a")[0][COMPRESSED_SUMMARY_METADATA_KEY] is True
    finally:
        db.close()


def test_archive_and_compact_round_trips_compressed_summary_marker(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="session-a", source="test")
        db.append_message("session-a", role="user", content="old turn")

        db.archive_and_compact("session-a", [_summary()])

        active = db.get_messages("session-a")
        assert len(active) == 1
        assert active[0][COMPRESSED_SUMMARY_METADATA_KEY] is True
    finally:
        db.close()


def test_agent_flush_round_trips_compressed_summary_marker(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="session-a", source="test")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=db,
                session_id="session-a",
                skip_context_files=True,
                skip_memory=True,
            )

        agent._flush_messages_to_session_db([_summary()], [])

        assert db.get_messages("session-a")[0][COMPRESSED_SUMMARY_METADATA_KEY] is True
    finally:
        db.close()
