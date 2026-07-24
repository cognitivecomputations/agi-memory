from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import apps.hexis_api as hexis_api
from apps.hexis_api import IntegrationActionRequest, _integration_action_arguments, app
from core.tools import ToolResult


def test_integration_action_arguments_add_web_source_context() -> None:
    args = _integration_action_arguments(
        "start_setup",
        {"connector_id": "Telegram"},
        "web-connections",
    )

    assert args["connector_id"] == "telegram"
    assert args["source_channel"] == "web"
    assert args["source_session_id"] == "web-connections"


def test_integration_action_arguments_accept_channel_configure() -> None:
    args = _integration_action_arguments(
        "configure_channel",
        {
            "connector_id": "Slack",
            "settings": {
                "bot_token": "SLACK_BOT_TOKEN",
                "app_token": "SLACK_APP_TOKEN",
            },
        },
        "web-connections",
    )

    assert args == {
        "connector_id": "slack",
        "settings": {
            "bot_token": "SLACK_BOT_TOKEN",
            "app_token": "SLACK_APP_TOKEN",
        },
    }


def test_integration_action_request_accepts_legacy_flat_payload() -> None:
    request = IntegrationActionRequest(
        action="start_setup",
        connector_id="signal",
        source_session_id="web-connections",
    )

    assert request.model_extra == {"connector_id": "signal"}


def test_integration_action_arguments_reject_gmail_manual_setup() -> None:
    with pytest.raises(HTTPException) as excinfo:
        _integration_action_arguments(
            "start_setup",
            {"connector_id": "gmail"},
            "web-connections",
        )

    assert excinfo.value.status_code == 422
    assert "connect_gmail" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_save_gmail_client_secret_action_returns_ui_without_secret(monkeypatch) -> None:
    class FakeRegistry:
        async def execute(self, tool_name, _arguments, _context):
            assert tool_name == "gmail_setup_status"
            return ToolResult.success_result(
                {
                    "connector_id": "gmail",
                    "status": "client_secret_saved",
                    "client_secret_saved": True,
                    "ui": {
                        "kind": "connector_setup",
                        "connector_id": "gmail",
                        "status": "client_secret_saved",
                        "client_secret_saved": True,
                    },
                }
            )

    def fake_save(payload, *, source):
        assert source == "web_upload"
        assert payload["installed"]["client_secret"] == "actual-secret"
        return {
            "status": "client_secret_saved",
            "connector_id": "gmail",
            "client_secret_saved": True,
            "client_id_hint": "...client",
            "source": source,
            "next_step": "Start Google sign-in from the setup panel.",
        }

    monkeypatch.setattr(hexis_api, "_pool", object())
    monkeypatch.setattr(hexis_api, "create_default_registry", lambda _pool: FakeRegistry())
    monkeypatch.setattr(
        "core.auth.google_gmail.save_gmail_client_secret_payload",
        fake_save,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/integrations/action",
            json={
                "action": "save_gmail_client_secret",
                "arguments": {
                    "client_secret_json": {
                        "installed": {
                            "client_id": "client",
                            "client_secret": "actual-secret",
                        }
                    }
                },
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["output"]["ui"]["kind"] == "connector_setup"
    assert body["output"]["ui"]["status"] == "client_secret_saved"
    assert "actual-secret" not in response.text


@pytest.mark.asyncio
async def test_gmail_oauth_callback_completes_attempt_without_manual_paste(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_complete(_pool, *, authorization_response: str):
        calls.append(authorization_response)
        return SimpleNamespace(
            account_key="eric@example.com",
            display_name="eric@example.com",
            credential_ref="integration.gmail.default",
            granted_scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            capabilities=["read", "search"],
        )

    monkeypatch.setattr(hexis_api, "_pool", object())
    monkeypatch.setattr("core.auth.google_gmail.complete_gmail_oauth", fake_complete)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:43817",
    ) as client:
        response = await client.get("/?code=oauth-code&state=state-value")

    assert response.status_code == 200
    assert "Gmail connected" in response.text
    assert "eric@example.com" in response.text
    assert calls and "code=oauth-code" in calls[0]
