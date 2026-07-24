import json

import pytest

from core.tools import ToolResult
from services import connector_setup
from services.connector_setup import (
    ConnectorSetupIntent,
    detect_connector_setup_intent,
    run_connector_setup_intent,
)


class _NoDbPool:
    def acquire(self):  # pragma: no cover - should not be touched by these cases
        raise AssertionError("database should not be queried")


class _FakeConn:
    async def fetchval(self, _sql):
        return json.dumps(
            {
                "recent_attempts": [
                    {
                        "connector_id": "gmail",
                        "status": "pending_user",
                    }
                ]
            }
        )


class _Acquire:
    async def __aenter__(self):
        return _FakeConn()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _PendingPool:
    def acquire(self):
        return _Acquire()


class _FakeRegistry:
    def __init__(self):
        self.calls = []
        self.pool = _NoDbPool()

    async def execute(self, tool_name, arguments, context):
        self.calls.append((tool_name, arguments, context.session_id))
        return ToolResult.success_result(
            {
                "status": "needs_client_secret",
                "ui": {
                    "kind": "connector_setup",
                    "connector_id": "gmail",
                    "display_name": "Gmail",
                    "status": "needs_client_secret",
                },
            },
            display_output="Gmail setup needs one-time Google setup. The panel walks through the steps.",
        )


class _BuiltInGmailRegistry(_FakeRegistry):
    async def execute(self, tool_name, arguments, context):
        self.calls.append((tool_name, arguments, context.session_id))
        return ToolResult.success_result(
            {
                "status": "needs_client_secret",
                "ui": {
                    "kind": "connector_setup",
                    "connector_id": "gmail",
                    "display_name": "Gmail",
                    "status": "needs_client_secret",
                    "hexis_oauth_client_available": True,
                    "credential_step_label": "Google sign-in ready",
                    "credential_step": {
                        "modes": [
                            {
                                "id": "hosted_oauth",
                                "label": "Built-in Google sign-in",
                                "available": True,
                            }
                        ]
                    },
                },
            },
            display_output="Gmail sign-in is ready.",
        )


def _patch_classifier(monkeypatch, doc):
    async def fake_classifier(_pool, _text, *, pending, gmail_connected):
        assert pending is not None
        assert gmail_connected is False
        return doc

    monkeypatch.setattr(connector_setup, "_classify_connector_setup_intent", fake_classifier)


@pytest.mark.asyncio
async def test_direct_email_connection_request_is_agent_routed():
    intent = await detect_connector_setup_intent(
        _NoDbPool(),
        "Can you connect to my email?",
        session_id="setup-natural",
    )

    assert intent is None


@pytest.mark.asyncio
async def test_scope_then_memory_answers_route_from_pending_setup(monkeypatch):
    session_id = "setup-staged"
    first = ConnectorSetupIntent("gmail", "choose_scope")

    registry = _FakeRegistry()
    opened = await run_connector_setup_intent(
        _NoDbPool(),
        registry,  # type: ignore[arg-type]
        first,
        session_id=session_id,
        source_channel="cli",
    )
    assert opened.action == "choose_scope"
    assert opened.ui is not None
    assert opened.ui["status"] == "needs_capability_choice"

    _patch_classifier(
        monkeypatch,
        {"route": "connector_setup", "capability_tier": "read_only"},
    )
    second = await detect_connector_setup_intent(_NoDbPool(), "just read them", session_id=session_id)
    assert second is not None
    assert second.action == "choose_memory"
    assert second.arguments["base_capabilities"] == ["read", "search"]

    prompted = await run_connector_setup_intent(
        _NoDbPool(),
        registry,  # type: ignore[arg-type]
        second,
        session_id=session_id,
        source_channel="cli",
    )
    assert prompted.ui is not None
    assert prompted.ui["status"] == "needs_memory_choice"
    assert prompted.ui["memory_config_key"] == "integrations.gmail.memory_policy"

    _patch_classifier(
        monkeypatch,
        {"route": "connector_setup", "memory_policy": "forget"},
    )
    third = await detect_connector_setup_intent(_NoDbPool(), "forget what they say", session_id=session_id)
    assert third is not None
    assert third.action == "choose_autonomy"
    assert third.arguments["base_capabilities"] == ["read", "search"]
    assert third.arguments["memory_policy"] == "forget"

    autonomy_prompt = await run_connector_setup_intent(
        _NoDbPool(),
        registry,  # type: ignore[arg-type]
        third,
        session_id=session_id,
        source_channel="cli",
    )
    assert autonomy_prompt.ui is not None
    assert autonomy_prompt.ui["status"] == "needs_autonomy_choice"
    assert autonomy_prompt.ui["heartbeat_digest_config_key"] == "integrations.gmail.heartbeat_digest_enabled"

    _patch_classifier(
        monkeypatch,
        {"route": "connector_setup", "heartbeat_digest_enabled": False},
    )
    fourth = await detect_connector_setup_intent(_NoDbPool(), "only when I ask", session_id=session_id)
    assert fourth is not None
    assert fourth.action == "start"
    assert fourth.arguments["capabilities"] == ["read", "search"]
    assert fourth.arguments["memory_policy"] == "forget"
    assert fourth.arguments["heartbeat_digest_enabled"] is False


@pytest.mark.asyncio
async def test_connected_gmail_clears_stale_pending_setup():
    session_id = "setup-connected"
    connector_setup._PENDING_SETUP_BY_SESSION[session_id] = {
        "connector_id": "gmail",
        "stage": "capability_choice",
    }

    class _ConnectedConn:
        async def fetchval(self, _sql):
            return json.dumps(
                {
                    "connections": [
                        {
                            "connector_id": "gmail",
                            "status": "connected",
                            "account_key": "eric@example.com",
                        }
                    ],
                    "recent_attempts": [
                        {
                            "connector_id": "gmail",
                            "status": "pending_user",
                            "authorization_url": "https://accounts.google.com/stale",
                        }
                    ],
                }
            )

    class _ConnectedAcquire:
        async def __aenter__(self):
            return _ConnectedConn()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _ConnectedPool:
        def acquire(self):
            return _ConnectedAcquire()

    intent = await detect_connector_setup_intent(
        _ConnectedPool(),
        "read a batch of my emails and notify me if anything is urgent",
        session_id=session_id,
    )

    assert intent is None
    assert session_id not in connector_setup._PENDING_SETUP_BY_SESSION


@pytest.mark.asyncio
async def test_detects_google_setup_file_path():
    intent = await detect_connector_setup_intent(
        _NoDbPool(),
        "The Google setup file is /Users/eric/Downloads/client_secret.json",
        session_id="setup-path-first",
    )

    assert intent is not None
    assert intent.action == "choose_scope"
    assert intent.arguments["client_secret_path"] == "/Users/eric/Downloads/client_secret.json"


@pytest.mark.asyncio
async def test_pending_setup_accepts_bare_absolute_json_path():
    session_id = "setup-bare-path"
    first = ConnectorSetupIntent("gmail", "choose_scope")

    registry = _FakeRegistry()
    await run_connector_setup_intent(
        _NoDbPool(),
        registry,  # type: ignore[arg-type]
        first,
        session_id=session_id,
        source_channel="cli",
    )

    second = await detect_connector_setup_intent(
        _NoDbPool(),
        "/Users/eric/Downloads/google-setup.json",
        session_id=session_id,
    )

    assert second is not None
    assert second.action == "choose_scope"
    assert second.arguments["client_secret_path"] == "/Users/eric/Downloads/google-setup.json"


@pytest.mark.asyncio
async def test_detects_pending_gmail_oauth_redirect_as_completion():
    intent = await detect_connector_setup_intent(
        _PendingPool(),
        "http://localhost:1/?state=abc&code=4/0abc",
    )

    assert intent is not None
    assert intent.action == "complete"
    assert intent.arguments["authorization_response"].startswith("http://localhost")


@pytest.mark.asyncio
async def test_run_connector_setup_returns_assistant_text_and_ui():
    intent = ConnectorSetupIntent("gmail", "choose_scope")
    registry = _FakeRegistry()

    result = await run_connector_setup_intent(
        _NoDbPool(),
        registry,  # type: ignore[arg-type]
        intent,
        session_id="session-1",
        source_channel="cli",
    )

    assert result.assistant_message.startswith("Do you want me")
    assert result.ui is not None
    assert result.ui["kind"] == "connector_setup"
    assert result.ui["status"] == "needs_capability_choice"
    assert registry.calls == []


@pytest.mark.asyncio
async def test_run_connector_setup_start_passes_policy_separate_from_capabilities():
    intent = ConnectorSetupIntent(
        "gmail",
        "start",
        {
            "capabilities": ["read", "search", "send", "reply"],
            "memory_policy": "forget",
            "heartbeat_digest_enabled": False,
        },
    )
    registry = _FakeRegistry()

    result = await run_connector_setup_intent(
        _NoDbPool(),
        registry,  # type: ignore[arg-type]
        intent,
        session_id="session-2",
        source_channel="cli",
    )

    assert result.ui is not None
    assert registry.calls == [
        (
            "connect_gmail",
            {
                "capabilities": ["read", "search", "send", "reply"],
                "memory_policy": "forget",
                "heartbeat_digest_enabled": False,
                "source_channel": "cli",
                "source_session_id": "session-2",
            },
            "session-2",
        )
    ]


@pytest.mark.asyncio
async def test_run_connector_setup_start_uses_built_in_sign_in_copy():
    intent = ConnectorSetupIntent(
        "gmail",
        "start",
        {
            "capabilities": ["read", "search"],
            "memory_policy": "remember",
            "heartbeat_digest_enabled": True,
        },
    )
    registry = _BuiltInGmailRegistry()

    result = await run_connector_setup_intent(
        _NoDbPool(),
        registry,  # type: ignore[arg-type]
        intent,
        session_id="session-built-in",
        source_channel="cli",
    )

    assert result.assistant_message == (
        "Gmail sign-in is ready. I opened the setup panel so you can approve access with Google."
    )
    assert result.ui is not None
    assert result.ui["hexis_oauth_client_available"] is True
    assert "upload" not in result.assistant_message.lower()
    assert "json" not in result.assistant_message.lower()
