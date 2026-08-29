from __future__ import annotations

import uuid

import httpx
import pytest

import apps.hexis_api as web_module
from apps.hexis_api import app

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


MESSAGE_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")


class _Connection:
    def __init__(self) -> None:
        self.marked_read = False

    async def fetchrow(self, query: str, message_id: uuid.UUID):
        assert message_id == MESSAGE_ID
        return {
            "id": MESSAGE_ID,
            "outbox_msg_id": "outbox-42",
            "kind": "user",
            "intent": "check_in",
            "message": "Would you like me to prepare the report?",
            "payload": {},
        }

    async def fetchval(self, query: str, message_id: uuid.UUID):
        assert "mark_web_inbox_read" in query
        assert message_id == MESSAGE_ID
        self.marked_read = True
        return 1


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _Connection:
        return self.connection

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _Pool:
    def __init__(self) -> None:
        self.connection = _Connection()

    def acquire(self) -> _Acquire:
        return _Acquire(self.connection)


class _Bridge:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.payloads: list[dict] = []

    async def ensure_ready(self) -> None:
        return None

    async def publish_inbox_payload(self, payload: dict) -> bool:
        if self.fail:
            raise RuntimeError("queue unavailable")
        self.payloads.append(payload)
        return True


async def _post_reply(pool: _Pool, bridge: _Bridge, monkeypatch: pytest.MonkeyPatch):
    original_pool = web_module._pool
    web_module._pool = pool  # type: ignore[assignment]
    monkeypatch.setattr(web_module, "RabbitMQBridge", lambda _pool: bridge)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/inbox/reply",
                json={"message_id": str(MESSAGE_ID), "reply": "Yes, please do."},
            )
    finally:
        web_module._pool = original_pool


async def test_reply_is_queued_with_context_and_then_marks_source_read(monkeypatch) -> None:
    pool = _Pool()
    bridge = _Bridge()

    response = await _post_reply(pool, bridge, monkeypatch)

    assert response.status_code == 200
    assert response.json()["queued"] is True
    assert response.json()["marked_read"] == 1
    assert pool.connection.marked_read is True
    assert len(bridge.payloads) == 1
    payload = bridge.payloads[0]
    assert payload["kind"] == "web_outbox_reply"
    assert payload["reply"] == "Yes, please do."
    assert payload["reply_to"] == {
        "web_inbox_id": str(MESSAGE_ID),
        "outbox_message_id": "outbox-42",
        "kind": "user",
        "intent": "check_in",
    }
    assert "Would you like me to prepare the report?" in payload["content"]
    assert "Yes, please do." in payload["content"]


async def test_failed_queue_does_not_mark_source_read(monkeypatch) -> None:
    pool = _Pool()
    bridge = _Bridge(fail=True)

    response = await _post_reply(pool, bridge, monkeypatch)

    assert response.status_code == 503
    assert "Reply was not queued" in response.json()["detail"]
    assert pool.connection.marked_read is False
