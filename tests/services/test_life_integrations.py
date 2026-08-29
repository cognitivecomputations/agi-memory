from __future__ import annotations

import json

import pytest

import services.life_integrations as life


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def test_setup_never_falls_back_to_an_ambient_secret(db_pool, monkeypatch):
    monkeypatch.setenv("NOTION_TOKEN", "ambient-secret-that-was-not-selected")
    called = False

    async def forbidden_request(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider must not be called without the selected env reference")

    monkeypatch.setattr(life, "request_json", forbidden_request)
    with pytest.raises(life.LifeIntegrationError, match="EXPLICIT_NOTION_TOKEN is not set"):
        await life.setup_life_integration(
            db_pool,
            "notion",
            {"token_env": "EXPLICIT_NOTION_TOKEN"},
            source_channel="test",
            source_session_id="wave-b-no-ambient",
        )
    assert called is False

    async with db_pool.acquire() as conn:
        persisted = await conn.fetchval(
            """
            SELECT count(*) FROM connection_attempts
            WHERE source_session_id = 'wave-b-no-ambient'
              AND (
                flow_state::text LIKE '%ambient-secret-that-was-not-selected%'
                OR COALESCE(error, '') LIKE '%ambient-secret-that-was-not-selected%'
              )
            """
        )
    assert persisted == 0


async def test_all_manual_wave_b_connectors_verify_and_use_only_selected_refs(db_pool, monkeypatch):
    monkeypatch.setenv("TEST_NOTION_TOKEN", "notion-secret")
    monkeypatch.setenv("TEST_HOME_TOKEN", "home-secret")
    monkeypatch.setenv("TEST_TRELLO_KEY", "trello-key-secret")
    monkeypatch.setenv("TEST_TRELLO_TOKEN", "trello-token-secret")
    calls: list[dict] = []

    async def fake_request(provider, method, url, **kwargs):
        calls.append({"provider": provider, "method": method, "url": url, **kwargs})
        if provider == "notion" and url.endswith("/v1/users/me"):
            assert kwargs["headers"]["Authorization"] == "Bearer notion-secret"
            assert kwargs["headers"]["Notion-Version"] == "2026-03-11"
            return {"id": "notion-bot", "name": "Hexis Notion"}
        if provider == "notion" and url.endswith("/v1/search"):
            return {"object": "list", "results": [{"id": "page-1", "object": "page"}], "has_more": False}
        if provider == "home_assistant" and method == "GET" and url.endswith("/api"):
            assert kwargs["headers"]["Authorization"] == "Bearer home-secret"
            return {"message": "API running."}
        if provider == "home_assistant" and "/services/light/turn_on" in url:
            return [{"entity_id": "light.desk", "state": "on", "attributes": {}}]
        if provider == "trello" and url.endswith("/members/me"):
            assert kwargs["params"]["key"] == "trello-key-secret"
            assert kwargs["params"]["token"] == "trello-token-secret"
            return {"id": "member-1", "username": "hexistest", "fullName": "Hexis Test"}
        if provider == "trello" and url.endswith("/members/me/boards"):
            return [{"id": "board-1", "name": "Home", "lists": [{"id": "list-1", "name": "Today"}]}]
        if provider == "trello" and url.endswith("/cards") and method == "POST":
            return {"id": "card-1", "name": kwargs["params"]["name"], "url": "https://trello.com/c/card-1"}
        if provider == "open_meteo_geocoding":
            return {
                "results": [
                    {
                        "id": 4930956,
                        "name": "Boston",
                        "admin1": "Massachusetts",
                        "country": "United States",
                        "country_code": "US",
                        "latitude": 42.35843,
                        "longitude": -71.05977,
                        "timezone": "America/New_York",
                    }
                ]
            }
        if provider == "open_meteo_forecast":
            return {
                "timezone": "America/New_York",
                "current": {"temperature_2m": 21.0, "weather_code": 1},
                "daily": {"time": ["2026-08-28"], "temperature_2m_max": [24.0]},
            }
        raise AssertionError(f"unexpected provider request: {provider} {method} {url}")

    monkeypatch.setattr(life, "request_json", fake_request)

    notion = await life.setup_life_integration(
        db_pool,
        "notion",
        {"token_env": "TEST_NOTION_TOKEN", "capabilities": ["search", "read", "query", "create"]},
        source_channel="test",
        source_session_id="wave-b-manual",
    )
    home = await life.setup_life_integration(
        db_pool,
        "home_assistant",
        {"base_url": "http://ha.test:8123", "token_env": "TEST_HOME_TOKEN", "capabilities": ["states", "service_control"]},
        source_channel="test",
        source_session_id="wave-b-manual",
    )
    weather = await life.setup_life_integration(
        db_pool,
        "weather",
        {"location": "Boston, MA"},
        source_channel="test",
        source_session_id="wave-b-manual",
    )
    trello = await life.setup_life_integration(
        db_pool,
        "trello",
        {"api_key_env": "TEST_TRELLO_KEY", "token_env": "TEST_TRELLO_TOKEN", "capabilities": ["boards", "cards", "create_card", "update_card"]},
        source_channel="test",
        source_session_id="wave-b-manual",
    )

    assert notion["status"] == home["status"] == weather["status"] == trello["status"] == "connected"
    assert weather["ui"]["matched_locations"][0]["name"] == "Boston"

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT connector_id, credential_ref, metadata::text AS metadata_text
            FROM integration_connections
            WHERE source_session_id = 'wave-b-manual'
            ORDER BY connector_id
            """
        )
    assert {row["connector_id"] for row in rows} == {"notion", "home_assistant", "weather", "trello"}
    persisted = json.dumps([dict(row) for row in rows])
    for secret in ("notion-secret", "home-secret", "trello-key-secret", "trello-token-secret"):
        assert secret not in persisted
    assert "TEST_NOTION_TOKEN" in persisted
    assert "TEST_HOME_TOKEN" in persisted
    assert "TEST_TRELLO_KEY" in persisted
    assert "TEST_TRELLO_TOKEN" in persisted

    search = await life.notion_search(db_pool, query="roadmap", account_key=notion["account_key"])
    changed = await life.home_assistant_call_service(
        db_pool,
        domain="light",
        service="turn_on",
        entity_id="light.desk",
        account_key=home["account_key"],
    )
    forecast = await life.weather_forecast(db_pool, account_key=weather["account_key"], days=3)
    boards = await life.trello_list_boards(db_pool, account_key=trello["account_key"])
    card = await life.trello_create_card(
        db_pool,
        list_id="list-1",
        name="Buy milk",
        account_key=trello["account_key"],
    )

    assert search["results"][0]["id"] == "page-1"
    assert changed[0]["state"] == "on"
    assert forecast["location"]["name"].startswith("Boston")
    assert boards[0]["id"] == "board-1"
    assert card["name"] == "Buy milk"

    trello_create_call = next(call for call in calls if call["provider"] == "trello" and call["method"] == "POST")
    assert trello_create_call["url"].endswith("/cards")
    assert "trello-token-secret" not in trello_create_call["url"]
