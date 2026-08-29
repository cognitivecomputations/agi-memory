from __future__ import annotations

import json
from uuid import uuid4

import pytest


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_database_accepts_only_an_advertised_wave_c_action(db_pool):
    node_id = "f" * 64
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(
                """
                INSERT INTO hexis_nodes (
                    node_id, public_key, name, capabilities, status,
                    last_seen_at, connection_id
                ) VALUES ($1, 'test-key', 'Wave C Mac', $2::jsonb, 'online',
                          CURRENT_TIMESTAMP, $3::uuid)
                """,
                node_id,
                json.dumps(["apple.reminders.list", "onepassword.items"]),
                uuid4(),
            )
            created = _json(
                await conn.fetchval(
                    "SELECT create_node_invocation($1, $2, '{}'::jsonb, 'test', 30, '{}'::jsonb)",
                    node_id,
                    "apple.reminders.list",
                )
            )
            denied = _json(
                await conn.fetchval(
                    "SELECT create_node_invocation($1, $2, '{}'::jsonb, 'test', 30, '{}'::jsonb)",
                    node_id,
                    "apple.notes.search",
                )
            )

            assert created["queued"] is True
            assert denied["queued"] is False
            assert denied["status"] == "unsupported"
            assert (
                await conn.fetchval(
                    "SELECT action FROM node_invocations WHERE id=$1::uuid",
                    created["invocation_id"],
                )
                == "apple.reminders.list"
            )
        finally:
            await transaction.rollback()
