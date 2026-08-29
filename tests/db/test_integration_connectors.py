from __future__ import annotations

import json

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_prepare_connection_attempt_derives_gmail_capabilities_and_scopes(db_pool):
    async with db_pool.acquire() as conn:
        default_plan = _j(await conn.fetchval(
            "SELECT prepare_connection_attempt('gmail', NULL)"
        ))
        assert default_plan["capabilities"] == ["read", "search"]
        assert default_plan["requested_scopes"] == [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/gmail.readonly",
        ]

        alias_plan = _j(await conn.fetchval(
            "SELECT prepare_connection_attempt('gmail', $1::jsonb)",
            json.dumps(["spam", "respond"]),
        ))
        assert alias_plan["capabilities"] == ["spam_triage", "reply"]
        assert alias_plan["requested_scopes"] == [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/gmail.send",
        ]

        delete_plan = _j(await conn.fetchval(
            "SELECT prepare_connection_attempt('gmail', $1::jsonb)",
            json.dumps(["delete"]),
        ))
        assert delete_plan["capabilities"] == ["delete"]
        assert delete_plan["requested_scopes"] == [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/gmail.modify",
        ]

        with pytest.raises(Exception, match="unsupported gmail capability"):
            await conn.fetchval(
                "SELECT prepare_connection_attempt('gmail', $1::jsonb)",
                json.dumps(["fax"]),
            )


async def test_channel_connector_manifests_are_first_class_and_honest(db_pool):
    async with db_pool.acquire() as conn:
        status = _j(await conn.fetchval("SELECT integration_status(NULL)"))
        by_id = {item["id"]: item for item in status["connectors"]}

        assert {"slack", "telegram", "signal", "twitter_x"} <= set(by_id)
        assert by_id["slack"]["status"] == "available"
        assert by_id["telegram"]["auth_type"] == "api_key"
        assert by_id["signal"]["auth_type"] == "pairing"
        assert by_id["twitter_x"]["status"] == "available"
        assert by_id["twitter_x"]["auth_type"] == "oauth2"

        slack_default = _j(await conn.fetchval(
            "SELECT prepare_connection_attempt('slack', NULL)"
        ))
        assert slack_default["capabilities"] == ["live_chat", "send", "ingest_live"]
        assert slack_default["requested_scopes"] == [
            "app_mentions:read",
            "channels:history",
            "chat:write",
            "im:write",
        ]

        telegram_alias = _j(await conn.fetchval(
            "SELECT prepare_connection_attempt('telegram', $1::jsonb)",
            json.dumps(["message", "ingest"]),
        ))
        assert telegram_alias["capabilities"] == ["send", "ingest_live"]
        assert telegram_alias["requested_scopes"] == []

        with pytest.raises(Exception, match="planned"):
            await conn.fetchval(
                "SELECT prepare_connection_attempt('signal', $1::jsonb)",
                json.dumps(["backfill"]),
            )

        twitter_default = _j(await conn.fetchval("SELECT prepare_connection_attempt('twitter_x', NULL)"))
        assert twitter_default["capabilities"] == ["read", "search", "ingest"]
        assert twitter_default["requested_scopes"] == [
            "tweet.read",
            "users.read",
            "offline.access",
        ]

        twitter_send = _j(await conn.fetchval(
            "SELECT prepare_connection_attempt('twitter_x', $1::jsonb)",
            json.dumps(["post", "dm"]),
        ))
        assert twitter_send["capabilities"] == ["send", "dm_send"]
        assert twitter_send["requested_scopes"] == [
            "tweet.read",
            "users.read",
            "offline.access",
            "tweet.write",
            "dm.write",
        ]

        twitter_archive = _j(await conn.fetchval(
            "SELECT prepare_connection_attempt('twitter_x', $1::jsonb)",
            json.dumps(["archive"]),
        ))
        assert twitter_archive["capabilities"] == ["archive_import"]
        assert twitter_archive["requested_scopes"] == []


async def test_channel_adapter_runtime_status_is_db_owned(db_pool):
    marker = get_test_identifier("channel-runtime")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            recorded = _j(await conn.fetchval(
                """
                SELECT record_channel_adapter_status(
                    'telegram',
                    'running',
                    TRUE,
                    TRUE,
                    NULL,
                    $1::jsonb
                )
                """,
                json.dumps({"test_marker": marker}),
            ))
            stopped = _j(await conn.fetchval(
                "SELECT record_channel_adapter_status('telegram', 'stopped', TRUE, FALSE)"
            ))
            listed = _j(await conn.fetchval(
                "SELECT list_channel_adapter_status('telegram')"
            ))
        finally:
            await tr.rollback()

    assert recorded["channel_type"] == "telegram"
    assert recorded["status"] == "running"
    assert recorded["running"] is True
    assert recorded["metadata"]["test_marker"] == marker
    assert stopped["status"] == "stopped"
    assert stopped["running"] is False
    assert listed[0]["channel_type"] == "telegram"
    assert listed[0]["status"] == "stopped"


async def test_gmail_connector_attempt_connection_and_revoke(db_pool):
    marker = get_test_identifier("integration")

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            status = _j(await conn.fetchval("SELECT integration_status('gmail')"))
            assert status["connectors"][0]["id"] == "gmail"
            assert status["connectors"][0]["auth_type"] == "oauth2"
            assert "read" in status["connectors"][0]["capability_manifest"]

            attempt = _j(await conn.fetchval(
                """
                SELECT start_connection_attempt(
                    'gmail',
                    '["read", "search"]'::jsonb,
                    $1::text[],
                    '{"pending_auth_ref": "test"}'::jsonb,
                    'https://accounts.google.com/o/oauth2/v2/auth?test=1',
                    'Open the URL and paste the redirect back.',
                    'cli',
                    $2,
                    CURRENT_TIMESTAMP + INTERVAL '10 minutes'
                )
                """,
                ["https://www.googleapis.com/auth/gmail.readonly"],
                marker,
            ))
            assert attempt["connector_id"] == "gmail"
            assert attempt["status"] == "pending_user"
            assert attempt["source_session_id"] == marker
            assert attempt["requested_capabilities"] == ["read", "search"]
            assert attempt["requested_scopes"] == [
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/gmail.readonly",
            ]

            completed = _j(await conn.fetchval(
                """
                SELECT complete_connection_attempt(
                    $1::uuid,
                    $2,
                    'eric@example.com',
                    'integration.gmail.default',
                    $3::text[],
                    '["read", "search"]'::jsonb,
                    '{"auth_store": "filesystem"}'::jsonb
                )
                """,
                attempt["attempt_id"],
                f"eric-{marker}@example.com",
                ["https://www.googleapis.com/auth/gmail.readonly"],
            ))
            assert completed["status"] == "connected"
            assert completed["credential_ref"] == "integration.gmail.default"

            revoked = _j(await conn.fetchval(
                "SELECT revoke_integration_connection('gmail', $1, 'test revoke')",
                f"eric-{marker}@example.com",
            ))
            assert revoked["revoked"] == 1
        finally:
            await tr.rollback()
