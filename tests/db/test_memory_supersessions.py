from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _memory(conn, content: str, valid_from: datetime):
    return await conn.fetchval(
        """
        INSERT INTO memories (
            type, content, embedding, embedding_status, status,
            created_at, valid_from, source_attribution
        ) VALUES (
            'semantic', $1,
            array_fill(0.1::float, ARRAY[embedding_dimension()])::vector,
            'embedded', 'active', $2, $2, '{"kind":"test"}'::jsonb
        )
        RETURNING id
        """,
        content,
        valid_from,
    )


async def test_supersession_closes_validity_and_preserves_point_in_time_history(
    db_pool,
):
    jan = datetime(2025, 1, 1, tzinfo=timezone.utc)
    feb = datetime(2025, 2, 1, tzinfo=timezone.utc)
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            old_id = await _memory(conn, "The retainer is monthly", jan)
            replacement_id = await _memory(conn, "The retainer is quarterly", feb)
            event_id = await conn.fetchval(
                """
                SELECT record_supersession(
                    $1, $2, 'new contract corrected cadence', 'operator',
                    'active', $3, NULL, TRUE, '{"source":"contract"}'::jsonb
                )
                """,
                old_id,
                replacement_id,
                feb,
            )

            old = await conn.fetchrow(
                "SELECT status, valid_from, valid_until, superseded_by FROM memories WHERE id=$1",
                old_id,
            )
            event = await conn.fetchrow(
                "SELECT * FROM memory_supersessions_active WHERE id=$1", event_id
            )
            before = await conn.fetch(
                """
                SELECT id FROM memories
                WHERE id = ANY($1::uuid[])
                  AND valid_from <= $2
                  AND (valid_until IS NULL OR valid_until > $2)
                ORDER BY id
                """,
                [old_id, replacement_id],
                datetime(2025, 1, 15, tzinfo=timezone.utc),
            )
            after = await conn.fetch(
                """
                SELECT id FROM memories
                WHERE id = ANY($1::uuid[])
                  AND valid_from <= $2
                  AND (valid_until IS NULL OR valid_until > $2)
                ORDER BY id
                """,
                [old_id, replacement_id],
                datetime(2025, 2, 15, tzinfo=timezone.utc),
            )

            assert old["status"] == "active"
            assert old["valid_from"] == jan
            assert old["valid_until"] == feb
            assert old["superseded_by"] == replacement_id
            assert event["superseded_memory_id"] == old_id
            assert event["replacement_memory_id"] == replacement_id
            assert event["reason"] == "new contract corrected cadence"
            assert event["actor"] == "operator"
            assert event["replacement_planned"] is True
            assert [row["id"] for row in before] == [old_id]
            assert [row["id"] for row in after] == [replacement_id]

            # An exact retry is idempotent; changing an active decision requires
            # the explicit revert path rather than silently rewriting history.
            retry_id = await conn.fetchval(
                """
                SELECT record_supersession(
                    $1, $2, 'new contract corrected cadence', 'operator',
                    'active', $3, NULL, TRUE, '{"source":"contract"}'::jsonb
                )
                """,
                old_id,
                replacement_id,
                feb,
            )
            assert retry_id == event_id
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM memory_supersessions WHERE superseded_memory_id=$1",
                    old_id,
                )
                == 1
            )
            with pytest.raises(asyncpg.RaiseError, match="revert it explicitly"):
                await conn.fetchval(
                    "SELECT record_supersession($1, NULL, 'different decision', 'operator')",
                    old_id,
                )
        finally:
            await tr.rollback()


async def test_revert_is_explicit_and_restores_old_validity(db_pool):
    jan = datetime(2025, 1, 1, tzinfo=timezone.utc)
    feb = datetime(2025, 2, 1, tzinfo=timezone.utc)
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            old_id = await _memory(conn, "Original belief", jan)
            replacement_id = await _memory(conn, "Replacement belief", feb)
            event_id = await conn.fetchval(
                "SELECT record_supersession($1, $2, 'correction', 'operator', 'active', $3)",
                old_id,
                replacement_id,
                feb,
            )
            assert await conn.fetchval(
                "SELECT revert_supersession($1, 'the old belief was right', 'operator')",
                event_id,
            )
            event = await conn.fetchrow(
                "SELECT status, resolved_at, metadata FROM memory_supersessions WHERE id=$1",
                event_id,
            )
            old = await conn.fetchrow(
                "SELECT valid_until, superseded_by FROM memories WHERE id=$1", old_id
            )
            assert event["status"] == "reverted"
            assert event["resolved_at"] is not None
            assert _json(event["metadata"])["reverted_by"] == "operator"
            assert old["valid_until"] is None
            assert old["superseded_by"] is None
            assert not await conn.fetchval(
                "SELECT revert_supersession($1, 'duplicate', 'operator')", event_id
            )
        finally:
            await tr.rollback()


async def test_legacy_pointer_write_cannot_bypass_lineage(db_pool):
    # Keep the memory valid before the transaction timestamp used by the
    # compatibility trigger; sub-millisecond client/server ordering is noise,
    # not a future-dated belief.
    now = datetime.now(timezone.utc) - timedelta(seconds=1)
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            old_id = await _memory(conn, "A detail to consolidate", now)
            gist_id = await _memory(conn, "Consolidated detail", now)
            await conn.execute(
                """
                UPDATE memories
                SET status='archived',
                    superseded_by=$2,
                    metadata=jsonb_build_object(
                        'consolidation', jsonb_build_object('archived_at', now())
                    )
                WHERE id=$1
                """,
                old_id,
                gist_id,
            )
            event = await conn.fetchrow(
                """
                SELECT reason, actor, replacement_memory_id
                FROM memory_supersessions_active
                WHERE superseded_memory_id=$1
                """,
                old_id,
            )
            assert event["reason"] == "retention consolidation"
            assert event["actor"] == "retention"
            assert event["replacement_memory_id"] == gist_id
            assert await conn.fetchval(
                "SELECT valid_until IS NOT NULL FROM memories WHERE id=$1", old_id
            )
        finally:
            await tr.rollback()


async def test_temporal_and_self_supersession_invariants_fail_loud(db_pool):
    now = datetime.now(timezone.utc)
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            memory_id = await _memory(conn, "Invariant target", now)
            with pytest.raises(asyncpg.RaiseError, match="cannot precede valid_from"):
                async with conn.transaction():
                    await conn.execute(
                        "UPDATE memories SET valid_until=$2 WHERE id=$1",
                        memory_id,
                        datetime(2024, 1, 1, tzinfo=timezone.utc),
                    )
            with pytest.raises(asyncpg.RaiseError, match="cannot supersede itself"):
                async with conn.transaction():
                    await conn.fetchval(
                        "SELECT record_supersession($1, $1, 'bad link', 'test')",
                        memory_id,
                    )
        finally:
            await tr.rollback()
