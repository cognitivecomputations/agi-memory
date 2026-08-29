from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import apps.hexis_api as hexis_api
from core.tools import ToolResult


def test_life_integration_action_arguments_are_narrow_and_add_web_context():
    args = hexis_api._integration_action_arguments(
        "connect_life",
        {"connector_id": "Home-Assistant", "base_url": "http://ha.test:8123", "token_env": "HA_TOKEN"},
        "web-wave-b",
    )
    assert args == {
        "connector_id": "home_assistant",
        "base_url": "http://ha.test:8123",
        "token_env": "HA_TOKEN",
        "source_channel": "web",
        "source_session_id": "web-wave-b",
    }

    with pytest.raises(HTTPException, match="connect_spotify"):
        hexis_api._integration_action_arguments(
            "connect_life",
            {"connector_id": "spotify"},
            "web-wave-b",
        )

    with pytest.raises(HTTPException, match="supports"):
        hexis_api._integration_action_arguments(
            "revoke_life",
            {"connector_id": "gmail"},
            "web-wave-b",
        )


@pytest.mark.asyncio
async def test_spotify_callback_completes_and_names_spotify_panel(monkeypatch):
    async def fake_complete(pool, *, authorization_response, attempt_id=None):
        assert pool is fake_pool
        assert "code=spotify-code" in authorization_response
        assert attempt_id is None
        return SimpleNamespace(
            account_key="spotify:user-1",
            display_name="Spotify User",
        )

    fake_pool = object()
    monkeypatch.setattr(hexis_api, "_pool", fake_pool)
    monkeypatch.setattr("core.auth.spotify.complete_spotify_oauth", fake_complete)

    transport = httpx.ASGITransport(app=hexis_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:43817") as client:
        response = await client.get(
            "/api/integrations/spotify/callback?code=spotify-code&state=spotify-state"
        )
    assert response.status_code == 200
    assert "Spotify connected" in response.text
    assert "Spotify setup panel" in response.text
    assert "Gmail setup panel" not in response.text


@pytest.mark.asyncio
async def test_web_action_routes_to_life_setup_tool(monkeypatch):
    captured = {}

    class FakeRegistry:
        async def execute(self, tool_name, arguments, context):
            captured.update(
                tool_name=tool_name,
                arguments=arguments,
                session_id=context.session_id,
            )
            return ToolResult.success_result(
                {
                    "status": "connected",
                    "ui": {
                        "kind": "connector_setup",
                        "connector_id": "weather",
                        "status": "connected",
                    },
                }
            )

    monkeypatch.setattr(hexis_api, "_pool", object())
    monkeypatch.setattr(hexis_api, "create_default_registry", lambda _pool: FakeRegistry())
    transport = httpx.ASGITransport(app=hexis_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/integrations/action",
            json={
                "action": "connect_life",
                "arguments": {"connector_id": "weather", "location": "Boston, MA"},
                "source_session_id": "web-wave-b",
            },
        )
    assert response.status_code == 200
    assert captured == {
        "tool_name": "connect_life_integration",
        "arguments": {
            "connector_id": "weather",
            "location": "Boston, MA",
            "source_channel": "web",
            "source_session_id": "web-wave-b",
        },
        "session_id": "web-wave-b",
    }
