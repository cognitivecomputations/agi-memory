from __future__ import annotations

import json

import pytest


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_wave_b_manifests_are_available_and_derive_least_scopes(db_pool):
    async with db_pool.acquire() as conn:
        status = _json(await conn.fetchval("SELECT integration_status(NULL)"))
        connectors = {item["id"]: item for item in status["connectors"]}

        assert {"notion", "spotify", "home_assistant", "weather", "trello"} <= set(connectors)
        assert all(connectors[name]["status"] == "available" for name in connectors if name in {"notion", "spotify", "home_assistant", "weather", "trello"})
        assert connectors["notion"]["setup_manifest"]["api_version"] == "2026-03-11"
        assert connectors["spotify"]["setup_manifest"]["callback_path"] == "/api/integrations/spotify/callback"
        assert connectors["weather"]["setup_manifest"]["secret_storage"] == "none"

        notion = _json(await conn.fetchval("SELECT prepare_connection_attempt('notion', NULL)"))
        spotify = _json(await conn.fetchval("SELECT prepare_connection_attempt('spotify', NULL)"))
        home = _json(await conn.fetchval("SELECT prepare_connection_attempt('home_assistant', NULL)"))
        weather = _json(await conn.fetchval("SELECT prepare_connection_attempt('weather', NULL)"))
        trello = _json(await conn.fetchval("SELECT prepare_connection_attempt('trello', NULL)"))

        assert notion["capabilities"] == ["search", "read", "query"]
        assert notion["requested_scopes"] == ["read_content"]
        assert spotify["capabilities"] == ["search", "playback_state"]
        assert spotify["requested_scopes"] == ["user-read-private", "user-read-playback-state"]
        assert home["capabilities"] == ["states"]
        assert home["requested_scopes"] == []
        assert weather["capabilities"] == ["forecast"]
        assert weather["requested_scopes"] == []
        assert trello["capabilities"] == ["boards", "cards"]
        assert trello["requested_scopes"] == ["read"]

        spotify_write = _json(
            await conn.fetchval(
                "SELECT prepare_connection_attempt('spotify', $1::jsonb)",
                json.dumps(["control"]),
            )
        )
        assert spotify_write["capabilities"] == ["playback_control"]
        assert spotify_write["requested_scopes"] == ["user-modify-playback-state"]

        with pytest.raises(Exception, match="unsupported notion capability"):
            await conn.fetchval(
                "SELECT prepare_connection_attempt('notion', $1::jsonb)",
                json.dumps(["delete_workspace"]),
            )


async def test_wave_b_external_writes_are_in_db_action_policy_map(db_pool):
    expected = {
        "notion_create_page": ("notion", "create_page", "provider_state_change"),
        "spotify_control_playback": ("spotify", "control_playback", "provider_state_change"),
        "home_assistant_call_service": ("home_assistant", "call_service", "provider_state_change"),
        "trello_create_card": ("trello", "create_card", "provider_state_change"),
        "trello_update_card": ("trello", "update_card", "provider_state_change"),
    }
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT tool_name, connector_id, action_kind, sensitivity, enabled
            FROM connector_action_tool_map
            WHERE tool_name = ANY($1::text[])
            """,
            list(expected),
        )
    actual = {
        row["tool_name"]: (
            row["connector_id"],
            row["action_kind"],
            row["sensitivity"],
        )
        for row in rows
        if row["enabled"]
    }
    assert actual == expected
