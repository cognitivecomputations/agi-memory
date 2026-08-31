from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.tools.base import ToolContext, ToolErrorType, ToolExecutionContext
from core.tools.life_integrations import (
    RevokeLifeIntegrationHandler,
    SpotifyControlPlaybackHandler,
    WeatherForecastHandler,
)
from core.tools.registry import create_default_registry


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


WAVE_B_TOOL_NAMES = {
    "connect_life_integration",
    "connect_spotify",
    "complete_spotify_connection",
    "revoke_life_integration",
    "notion_search",
    "notion_get_page",
    "notion_query_data_source",
    "notion_create_page",
    "spotify_search",
    "spotify_playback_state",
    "spotify_control_playback",
    "home_assistant_states",
    "home_assistant_call_service",
    "weather_forecast",
    "trello_list_boards",
    "trello_list_cards",
    "trello_create_card",
    "trello_update_card",
}


def _context(pool=None) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="wave-b-tools",
        session_id="wave-b-tools",
        registry=SimpleNamespace(pool=pool),
    )


async def test_wave_b_tools_are_registered_with_write_and_secret_boundaries():
    registry = create_default_registry(pool=None)
    assert WAVE_B_TOOL_NAMES <= set(registry.list_names())

    writes = {
        "connect_life_integration",
        "connect_spotify",
        "complete_spotify_connection",
        "revoke_life_integration",
        "notion_create_page",
        "spotify_control_playback",
        "home_assistant_call_service",
        "trello_create_card",
        "trello_update_card",
    }
    for name in writes:
        spec = registry.get_spec(name)
        assert spec is not None
        assert spec.is_read_only is False
        assert spec.requires_approval is True
        assert spec.supports_parallel is False

    for name in WAVE_B_TOOL_NAMES - writes:
        spec = registry.get_spec(name)
        assert spec is not None
        assert spec.is_read_only is True

    life_setup = registry.get_spec("connect_life_integration").parameters["properties"]
    spotify_setup = registry.get_spec("connect_spotify").parameters["properties"]
    assert "token" not in life_setup
    assert "api_key" not in life_setup
    assert {"token_env", "api_key_env"} <= set(life_setup)
    assert "client_secret" not in spotify_setup
    assert {"client_id", "client_id_env"} <= set(spotify_setup)


async def test_spotify_control_maps_queue_and_rejects_bad_repeat(monkeypatch):
    calls: list[tuple] = []

    async def fake_api(pool, method, path, **kwargs):
        calls.append((pool, method, path, kwargs))
        return {}

    monkeypatch.setattr("core.auth.spotify.spotify_api_request", fake_api)
    handler = SpotifyControlPlaybackHandler()
    pool = object()
    queued = await handler.execute(
        {"action": "queue", "uri": "spotify:track:123", "device_id": "device-1"},
        _context(pool),
    )
    assert queued.success is True
    assert calls == [
        (
            pool,
            "POST",
            "/me/player/queue",
            {
                "capability": "playback_control",
                "account_key": None,
                "params": {"device_id": "device-1", "uri": "spotify:track:123"},
                "body": None,
            },
        )
    ]

    rejected = await handler.execute(
        {"action": "repeat", "state": "forever"},
        _context(pool),
    )
    assert rejected.success is False
    assert rejected.error_type == ToolErrorType.INVALID_PARAMS
    assert "off, context, or track" in rejected.error

    rejected_action = await handler.execute(
        {"action": "arbitrary-endpoint"},
        _context(pool),
    )
    assert rejected_action.success is False
    assert rejected_action.error_type == ToolErrorType.INVALID_PARAMS
    assert len(calls) == 1


async def test_revoke_rejects_connectors_outside_wave_b():
    result = await RevokeLifeIntegrationHandler().execute(
        {"connector_id": "gmail"},
        _context(None),
    )
    assert result.success is False
    assert result.error_type == ToolErrorType.INVALID_PARAMS


async def test_weather_tool_surfaces_location_and_bounded_days(monkeypatch):
    captured = {}

    async def fake_forecast(pool, **kwargs):
        captured.update({"pool": pool, **kwargs})
        return {
            "location": {"name": "Boston"},
            "forecast": {"current": {"temperature_2m": 20}},
        }

    monkeypatch.setattr("services.life_integrations.weather_forecast", fake_forecast)
    pool = object()
    result = await WeatherForecastHandler().execute(
        {"location": "Boston", "days": 3},
        _context(pool),
    )
    assert result.success is True
    assert result.display_output == "Weather forecast retrieved for Boston."
    assert captured == {
        "pool": pool,
        "location": "Boston",
        "latitude": None,
        "longitude": None,
        "days": 3,
        "account_key": None,
    }
