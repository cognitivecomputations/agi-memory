"""Isolated tests for the thin inbound-disposition Python bridge."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.inbound_disposition import (
    is_disposition_enabled,
    record_passive_observation,
    resolve_disposition,
)

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_args):
        return False


def _pool(conn):
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=lambda: _Acquire(conn))
    return pool


def _conn(*, fetchval=None, side_effect=None):
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=fetchval, side_effect=side_effect)
    return conn


def _result(**overrides):
    result = {
        "disposition": "observe",
        "reason": "default_observe",
        "ambiguous": False,
        "is_operator": False,
        "reply_allowed": True,
        "trigger_stripped_text": None,
        "session_id": None,
        "audit_id": None,
    }
    result.update(overrides)
    return result


async def test_enabled_flag_is_dark_on_read_failure(caplog):
    pool = _pool(_conn(side_effect=RuntimeError("db unavailable")))

    with caplog.at_level(logging.WARNING):
        assert await is_disposition_enabled(pool) is False

    assert "retaining legacy channel gates" in caplog.text


async def test_deterministic_result_does_not_call_classifier(monkeypatch):
    result = _result(disposition="engage", reason="trigger_match")
    conn = _conn(fetchval=json.dumps(result))
    pool = _pool(conn)
    classifier = AsyncMock()
    monkeypatch.setattr("core.llm_json.chat_json", classifier)

    actual = await resolve_disposition(
        pool,
        channel_type="imessage",
        sender_id="sender-1234",
        session_id="room",
        text="hexis hello",
        metadata={"channel_id": "room"},
    )

    assert actual == result
    classifier.assert_not_awaited()
    assert conn.fetchval.await_count == 1
    args = conn.fetchval.await_args.args
    assert "resolve_inbound_disposition" in args[0]
    assert args[1:5] == (
        "imessage",
        "sender-1234",
        "room",
        "hexis hello",
    )


async def test_ambiguous_operator_classifier_engages_and_finalizes(monkeypatch):
    result = _result(
        ambiguous=True,
        is_operator=True,
        reason="ambiguous_operator",
        audit_id=42,
    )
    conn = _conn(side_effect=[json.dumps(result), True, 7, True])
    pool = _pool(conn)
    monkeypatch.setattr(
        "core.llm_config.load_llm_config",
        AsyncMock(return_value={"provider": "test", "model": "test"}),
    )
    classifier = AsyncMock(
        return_value=(
            {
                "addressed_to_hexis": True,
                "is_correction": False,
                "confidence": 0.95,
            },
            "{}",
        )
    )
    monkeypatch.setattr("core.llm_json.chat_json", classifier)

    actual = await resolve_disposition(
        pool,
        channel_type="imessage",
        sender_id="operator-1234",
        session_id="room",
        text="could you check this?",
    )

    assert actual["disposition"] == "engage"
    assert actual["classifier_used"] is True
    assert actual["classifier_label"] == "classifier_addressed"
    classifier.assert_awaited_once()
    finalize = conn.fetchval.await_args_list[-1].args
    assert "finalize_inbound_disposition" in finalize[0]
    assert finalize[1:] == (42, "engage", "classifier_addressed")


async def test_operator_classifier_correction_maps_to_wake(monkeypatch):
    result = _result(
        ambiguous=True,
        is_operator=True,
        reason="ambiguous_operator",
        audit_id=7,
    )
    conn = _conn(side_effect=[result, True, 10, True])
    pool = _pool(conn)
    monkeypatch.setattr(
        "core.llm_config.load_llm_config",
        AsyncMock(return_value={"provider": "test", "model": "test"}),
    )
    monkeypatch.setattr(
        "core.llm_json.chat_json",
        AsyncMock(
            return_value=(
                {
                    "addressed_to_hexis": False,
                    "is_correction": True,
                    "confidence": 0.9,
                },
                "{}",
            )
        ),
    )

    actual = await resolve_disposition(
        pool,
        channel_type="imessage",
        sender_id="operator-1234",
        session_id="room",
        text="that was incorrect",
    )

    assert actual["disposition"] == "wake"
    assert actual["classifier_label"] == "classifier_correction"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"addressed_to_hexis": "yes", "is_correction": False},
        {"addressed_to_hexis": False},
    ],
)
async def test_invalid_classifier_payload_retains_deterministic_result(
    monkeypatch, payload
):
    result = _result(
        ambiguous=True,
        is_operator=True,
        reason="ambiguous_operator",
        audit_id=9,
    )
    conn = _conn(side_effect=[result, True, 10])
    pool = _pool(conn)
    monkeypatch.setattr(
        "core.llm_config.load_llm_config",
        AsyncMock(return_value={"provider": "test", "model": "test"}),
    )
    monkeypatch.setattr(
        "core.llm_json.chat_json", AsyncMock(return_value=(payload, "{}"))
    )

    actual = await resolve_disposition(
        pool,
        channel_type="imessage",
        sender_id="operator-1234",
        session_id="room",
        text="ambiguous",
    )

    assert actual == result
    assert conn.fetchval.await_count == 3


async def test_sql_failure_retains_passive_observation(caplog):
    pool = _pool(_conn(side_effect=RuntimeError("resolver missing")))

    with caplog.at_level(logging.WARNING):
        result = await resolve_disposition(
            pool,
            channel_type="signal",
            sender_id="sender-9999",
            session_id="room",
            text="hello",
        )

    assert result["disposition"] == "observe"
    assert result["reply_allowed"] is False
    assert "retaining passive observation" in caplog.text


async def test_passive_observation_uses_canonical_db_function():
    stored = {
        "session_id": "00000000-0000-0000-0000-000000000001",
        "message_id": "00000000-0000-0000-0000-000000000002",
        "duplicate": False,
    }
    conn = _conn(fetchval=stored)
    pool = _pool(conn)
    message = SimpleNamespace(
        channel_type="signal",
        channel_id="room",
        sender_id="sender-1234",
        sender_name="Sender",
        content="ambient",
        message_id="platform-1",
        metadata={"is_group": True},
    )

    actual = await record_passive_observation(
        pool,
        message=message,
        disposition={
            "audit_id": 12,
            "disposition": "observe",
            "reason": "default_observe",
            "is_operator": False,
        },
    )

    assert actual == stored
    args = conn.fetchval.await_args.args
    assert "record_inbound_disposition_observation" in args[0]
    assert args[1:8] == (
        12,
        "signal",
        "room",
        "sender-1234",
        "Sender",
        "ambient",
        "platform-1",
    )
