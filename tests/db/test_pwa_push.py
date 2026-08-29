from __future__ import annotations

import json

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def test_pwa_defaults_are_private_and_explicit(db_pool):
    async with db_pool.acquire() as conn:
        assert await conn.fetchval("SELECT get_config_bool('pwa.push.enabled')") is True
        assert (
            await conn.fetchval(
                "SELECT get_config_bool('pwa.push.show_message_previews')"
            )
            is False
        )
        assert (
            await conn.fetchval("SELECT get_config_bool('pwa.presence.enabled')")
            is True
        )


async def test_web_push_subscription_can_be_reenabled_and_revoked(db_pool):
    marker = get_test_identifier("pwa-push")
    endpoint = f"https://push.example.test/{marker}"
    async with db_pool.acquire() as conn:
        try:
            created = await conn.fetchval(
                "SELECT upsert_web_push_subscription($1, 'p256dh', 'auth', NULL, 'test', '{}'::jsonb)",
                endpoint,
            )
            if isinstance(created, str):
                created = json.loads(created)
            assert created["active"] is True
            assert (
                await conn.fetchval("SELECT revoke_web_push_subscription($1)", endpoint)
                is True
            )
            assert (
                await conn.fetchval(
                    "SELECT revoked_at IS NOT NULL FROM web_push_subscriptions WHERE endpoint = $1",
                    endpoint,
                )
                is True
            )

            restored = await conn.fetchval(
                "SELECT upsert_web_push_subscription($1, 'new-p256dh', 'new-auth', NULL, 'test', '{}'::jsonb)",
                endpoint,
            )
            if isinstance(restored, str):
                restored = json.loads(restored)
            assert restored["id"] == created["id"]
            row = await conn.fetchrow(
                "SELECT p256dh, auth, revoked_at FROM web_push_subscriptions WHERE endpoint = $1",
                endpoint,
            )
            assert tuple(row) == ("new-p256dh", "new-auth", None)
        finally:
            await conn.execute(
                "DELETE FROM web_push_subscriptions WHERE endpoint = $1", endpoint
            )


async def test_web_push_subscription_rejects_plain_http(db_pool):
    async with db_pool.acquire() as conn:
        with pytest.raises(Exception, match="HTTPS URL"):
            await conn.fetchval(
                "SELECT upsert_web_push_subscription('http://push.invalid/x', 'p', 'a')"
            )


async def test_pwa_presence_keeps_only_latest_ephemeral_device_state(db_pool):
    device_id = get_test_identifier("pwa-presence")
    async with db_pool.acquire() as conn:
        try:
            first = await conn.fetchval(
                "SELECT record_pwa_presence($1, 'online', 'standalone', 'visible')",
                device_id,
            )
            second = await conn.fetchval(
                "SELECT record_pwa_presence($1, 'idle', 'standalone', 'hidden')",
                device_id,
            )
            if isinstance(first, str):
                first = json.loads(first)
            if isinstance(second, str):
                second = json.loads(second)
            rows = await conn.fetch(
                """
                SELECT presence_kind, metadata
                FROM channel_presence_events
                WHERE channel_type = 'web' AND channel_id = $1
                """,
                device_id,
            )

            assert first["id"] != second["id"]
            assert len(rows) == 1
            assert rows[0]["presence_kind"] == "idle"
            metadata = rows[0]["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            assert metadata == {
                "display_mode": "standalone",
                "visibility": "hidden",
            }
        finally:
            await conn.execute(
                "DELETE FROM channel_presence_events WHERE channel_type = 'web' AND channel_id = $1",
                device_id,
            )
