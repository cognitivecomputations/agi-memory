from __future__ import annotations

import json
import uuid
from pathlib import Path

import asyncpg
import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_deliberation_lifecycle_is_durable_idempotent_and_advisory(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            evidence_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, importance)
                VALUES (
                    'semantic',
                    'Deliberation evidence',
                    array_fill(0.17, ARRAY[embedding_dimension()])::vector,
                    0.7
                )
                RETURNING id
                """
            )
            started = _json(
                await conn.fetchval(
                    """
                    SELECT begin_deliberation(
                        $1, 'high', 'chat', 'session-1', NULL, 'call-1',
                        '["skeptic", "builder"]'::jsonb,
                        2,
                        '{"additional_context":"A reversible pilot is available"}'::jsonb
                    )
                    """,
                    "Should we launch the pilot?",
                )
            )
            deliberation_id = uuid.UUID(started["id"])

            first_move = await conn.fetchval(
                """
                SELECT record_deliberation_move(
                    $1, 'perspective:skeptic', 'perspective', $2,
                    1, 0, 'skeptic', NULL, ARRAY[$3]::uuid[],
                    '{"available":true}'::jsonb
                )
                """,
                deliberation_id,
                "Pilot first; define a stop condition before expanding.",
                evidence_id,
            )
            duplicate_move = await conn.fetchval(
                """
                SELECT record_deliberation_move(
                    $1, 'perspective:skeptic', 'perspective', $2,
                    1, 0, 'skeptic', NULL, ARRAY[$3]::uuid[], '{}'::jsonb
                )
                """,
                deliberation_id,
                "This duplicate must not rewrite the first move.",
                evidence_id,
            )
            assert first_move == duplicate_move

            completed = _json(
                await conn.fetchval(
                    """
                    SELECT complete_deliberation(
                        $1, $2, $3,
                        '["A pilot is reversible"]'::jsonb,
                        '["Demand is uncertain"]'::jsonb,
                        '["Support load may rise"]'::jsonb,
                        '["Current support capacity"]'::jsonb,
                        '["Wait for one more customer interview"]'::jsonb,
                        '["Pilot support exceeds the agreed cap"]'::jsonb,
                        ARRAY[$4]::uuid[], TRUE,
                        '{"degraded":false}'::jsonb
                    )
                    """,
                    deliberation_id,
                    "Run the bounded pilot; do not commit to a full launch.",
                    "The reversible pilot tests demand while preserving the stop condition.",
                    evidence_id,
                )
            )
            assert completed["applied"] is True
            assert completed["memory_id"]

            inspected = _json(
                await conn.fetchval("SELECT inspect_deliberation($1)", deliberation_id)
            )
            assert inspected["found"] is True
            assert inspected["session"]["status"] == "completed"
            assert inspected["verdict"]["recommendation"].startswith("Run the bounded")
            assert inspected["verdict"]["missing_evidence"] == [
                "Current support capacity"
            ]
            assert inspected["verdict"]["summary_memory_id"] == completed["memory_id"]
            assert [move["role"] for move in inspected["moves"]] == [
                "perspective",
                "synthesis",
            ]
            assert inspected["moves"][0]["content"].startswith("Pilot first")
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM deliberation_moves WHERE session_id = $1",
                    deliberation_id,
                )
                == 2
            )

            repeated = _json(
                await conn.fetchval(
                    "SELECT complete_deliberation($1, 'different', 'different')",
                    deliberation_id,
                )
            )
            assert repeated["applied"] is False
            assert repeated["reason"] == "already_completed"

            listed = _json(
                await conn.fetchval("SELECT list_deliberations(5, 'completed')")
            )
            assert any(item["id"] == str(deliberation_id) for item in listed["items"])
        finally:
            await tr.rollback()


async def test_failed_deliberation_preserves_cause_without_creating_memory(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            started = _json(
                await conn.fetchval(
                    """
                    SELECT begin_deliberation(
                        'A failed council run', 'material', 'heartbeat', NULL,
                        NULL, 'call-failed', '["skeptic"]'::jsonb
                    )
                    """
                )
            )
            deliberation_id = uuid.UUID(started["id"])
            failed = _json(
                await conn.fetchval(
                    "SELECT fail_deliberation($1, $2)",
                    deliberation_id,
                    "model route unavailable; configure llm.chat and retry",
                )
            )
            assert failed["applied"] is True
            inspected = _json(
                await conn.fetchval("SELECT inspect_deliberation($1)", deliberation_id)
            )
            assert inspected["session"]["status"] == "failed"
            assert "configure llm.chat" in inspected["session"]["error"]
            assert inspected["verdict"] is None
        finally:
            await tr.rollback()


async def test_database_enforces_live_context_limit(db_pool):
    async with db_pool.acquire() as conn:
        max_chars = await conn.fetchval(
            "SELECT get_config_int('deliberation.max_context_chars')"
        )
        oversized = "x" * (int(max_chars) + 1)
        with pytest.raises(asyncpg.PostgresError, match="context exceeds"):
            await conn.fetchval(
                """
                SELECT begin_deliberation(
                    'Bounded context', 'routine', 'chat', NULL, NULL, NULL,
                    '["skeptic"]'::jsonb, 0,
                    jsonb_build_object('additional_context', $1::text)
                )
                """,
                oversized,
            )


async def test_database_rejects_duplicate_personas_and_excess_signals(db_pool):
    async with db_pool.acquire() as conn:
        with pytest.raises(asyncpg.PostgresError, match="must be unique"):
            await conn.fetchval(
                """
                SELECT begin_deliberation(
                    'Duplicate council', 'routine', 'chat', NULL, NULL, NULL,
                    '["skeptic", "skeptic"]'::jsonb
                )
                """
            )

        signal_limit = await conn.fetchval(
            "SELECT get_config_int('deliberation.signal_limit')"
        )
        with pytest.raises(asyncpg.PostgresError, match="evidence signals"):
            await conn.fetchval(
                """
                SELECT begin_deliberation(
                    'Oversized evidence', 'routine', 'chat', NULL, NULL, NULL,
                    '["skeptic"]'::jsonb, $1
                )
                """,
                int(signal_limit) + 1,
            )


async def test_0226_replays_as_a_self_contained_forward_migration(db_pool):
    migration = (
        Path(__file__).resolve().parents[2]
        / "db"
        / "migrations"
        / "0226_deliberation.sql"
    ).read_text(encoding="utf-8")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "DROP TABLE deliberation_verdicts, deliberation_moves, "
                "deliberation_sessions CASCADE"
            )
            await conn.execute(migration)

            assert await conn.fetchval(
                "SELECT to_regclass('public.deliberation_sessions') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT to_regprocedure('public.inspect_deliberation(uuid)') IS NOT NULL"
            )
            started = _json(
                await conn.fetchval(
                    "SELECT begin_deliberation('Migration replay', 'routine', "
                    "'chat', NULL, NULL, 'replay', '[\"skeptic\"]'::jsonb)"
                )
            )
            assert started["status"] == "running"
        finally:
            await tr.rollback()
