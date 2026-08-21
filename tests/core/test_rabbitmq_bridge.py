from __future__ import annotations

import json

import pytest

from core.rabbitmq_bridge import RabbitMQBridge

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


_RABBITMQ_ENV_NAMES = (
    "RABBITMQ_MANAGEMENT_URL",
    "RABBITMQ_MANAGEMENT_PORT",
    "RABBITMQ_USER",
    "RABBITMQ_PASSWORD",
    "RABBITMQ_DEFAULT_USER",
    "RABBITMQ_DEFAULT_PASS",
)


def _clear_rabbitmq_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _RABBITMQ_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


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


async def test_host_bridge_uses_published_port_and_compose_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rabbitmq_env(monkeypatch)
    monkeypatch.setenv("RABBITMQ_MANAGEMENT_PORT", "46673")
    monkeypatch.setenv("RABBITMQ_DEFAULT_USER", "compose-user")
    monkeypatch.setenv("RABBITMQ_DEFAULT_PASS", "compose-password")
    monkeypatch.setenv("RABBITMQ_USER", "unrelated-user")
    monkeypatch.setenv("RABBITMQ_PASSWORD", "unrelated-password")
    captured: dict[str, object] = {}

    def fake_request(method, url, *, auth, json, timeout):
        captured.update(
            method=method,
            url=url,
            auth=auth,
            payload=json,
            timeout=timeout,
        )
        return _RoutedResponse()

    monkeypatch.setattr("core.rabbitmq_bridge.requests.request", fake_request)
    bridge = RabbitMQBridge(pool=None)

    await bridge._request("GET", "/api/overview")

    assert captured == {
        "method": "GET",
        "url": "http://127.0.0.1:46673/api/overview",
        "auth": ("compose-user", "compose-password"),
        "payload": None,
        "timeout": 5,
    }


async def test_explicit_remote_bridge_uses_direct_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rabbitmq_env(monkeypatch)
    monkeypatch.setenv("RABBITMQ_MANAGEMENT_URL", "https://rabbit.example/manage/")
    monkeypatch.setenv("RABBITMQ_USER", "remote-user")
    monkeypatch.setenv("RABBITMQ_PASSWORD", "remote-password")
    monkeypatch.setenv("RABBITMQ_DEFAULT_USER", "local-user")
    monkeypatch.setenv("RABBITMQ_DEFAULT_PASS", "local-password")

    bridge = RabbitMQBridge(pool=None)

    assert bridge.management_url == "https://rabbit.example/manage"
    assert bridge.user == "remote-user"
    assert bridge.password == "remote-password"


async def test_publish_outbox_preserves_delivery_metadata() -> None:
    bridge = RabbitMQBridge(pool=None)
    captured: list[dict] = []

    async def fake_request(method: str, path: str, payload: dict | None = None):
        captured.append({"method": method, "path": path, "payload": payload})
        return _RoutedResponse()

    bridge._request = fake_request  # type: ignore[method-assign]

    published = await bridge.publish_outbox_payloads(
        [
            {
                "message_id": "msg-1",
                "kind": "user",
                "payload": {"message": "hello"},
                "delivery": {"mode": "web_inbox"},
                "task_name": "scheduled hello",
            }
        ]
    )

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
