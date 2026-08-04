from __future__ import annotations

import json

import pytest

from core.rabbitmq_bridge import RabbitMQBridge

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


class _RoutedResponse:
    status_code = 200
    text = "{}"

    @staticmethod
    def json() -> dict:
        return {"routed": True}


class _UnroutedResponse:
    status_code = 200
    text = '{"routed": false}'

    @staticmethod
    def json() -> dict:
        return {"routed": False}


async def test_publish_outbox_preserves_delivery_metadata() -> None:
    bridge = RabbitMQBridge(pool=None)
    captured: list[dict] = []

    async def fake_request(method: str, path: str, payload: dict | None = None):
        captured.append({"method": method, "path": path, "payload": payload})
        return _RoutedResponse()

    bridge._request = fake_request  # type: ignore[method-assign]

    published = await bridge.publish_outbox_payloads([
        {
            "message_id": "msg-1",
            "kind": "user",
            "payload": {"message": "hello"},
            "delivery": {"mode": "web_inbox"},
            "task_name": "scheduled hello",
        }
    ])

    assert published == 1
    body = json.loads(captured[0]["payload"]["payload"])
    assert body == {
        "id": "msg-1",
        "kind": "user",
        "payload": {"message": "hello"},
        "delivery": {"mode": "web_inbox"},
        "task_name": "scheduled hello",
    }


async def test_publish_inbox_payload_is_durable_and_correlated() -> None:
    bridge = RabbitMQBridge(pool=None)
    captured: list[dict] = []

    async def fake_request(method: str, path: str, payload: dict | None = None):
        captured.append({"method": method, "path": path, "payload": payload})
        return _RoutedResponse()

    bridge._request = fake_request  # type: ignore[method-assign]
    body = {
        "id": "reply-1",
        "kind": "web_outbox_reply",
        "content": "User reply: yes",
        "reply_to": {"web_inbox_id": "message-1"},
    }

    assert await bridge.publish_inbox_payload(body) is True
    request = captured[0]
    assert request["method"] == "POST"
    assert request["path"].endswith("/amq.default/publish")
    assert request["payload"]["routing_key"] == "hexis.inbox"
    assert request["payload"]["properties"] == {
        "content_type": "application/json",
        "delivery_mode": 2,
        "message_id": "reply-1",
    }
    assert json.loads(request["payload"]["payload"]) == body


async def test_publish_inbox_payload_fails_when_message_is_not_routed() -> None:
    bridge = RabbitMQBridge(pool=None)

    async def fake_request(method: str, path: str, payload: dict | None = None):
        return _UnroutedResponse()

    bridge._request = fake_request  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="not routed"):
        await bridge.publish_inbox_payload({"id": "reply-1", "content": "hello"})
