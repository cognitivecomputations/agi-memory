from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _memory(conn, content: str, *, memory_type: str = "semantic"):
    return await conn.fetchval(
        """
        INSERT INTO memories (
            type, content, embedding, embedding_status, status,
            importance, trust_level, metadata, source_attribution
        ) VALUES (
            $1::memory_type, $2,
            array_fill(0.1::float, ARRAY[embedding_dimension()])::vector,
            'embedded', 'active', 0.5, 0.6,
            '{"confidence":0.5}'::jsonb,
            '{"kind":"test","worker_id":"belief-test"}'::jsonb
        )
        RETURNING id
        """,
        memory_type,
        content,
    )


async def test_meaningful_belief_writes_are_durable_and_filtered(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            semantic_id = await _memory(conn, "A durable belief")
            episodic_id = await _memory(
                conn, "A passing experience", memory_type="episodic"
            )
            event = await conn.fetchrow(
                "SELECT * FROM belief_update_log WHERE memory_id=$1", semantic_id
            )
            assert event is not None
            assert event["change_kind"] == "new_evidence"
            assert event["actor"] == "belief-test"
            assert event["notified"] is True
            assert not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM belief_update_log WHERE memory_id=$1)",
                episodic_id,
            )
        finally:
            await tr.rollback()


async def test_thresholds_and_contradictions_classify_the_change(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            memory_id = await _memory(conn, "The agreement renews monthly")
            await conn.execute(
                "DELETE FROM belief_update_log WHERE memory_id=$1", memory_id
            )
            await conn.execute(
                "UPDATE memories SET metadata=jsonb_set(metadata, '{confidence}', '0.55') WHERE id=$1",
                memory_id,
            )
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM belief_update_log WHERE memory_id=$1",
                    memory_id,
                )
                == 0
            )

            await conn.execute(
                "UPDATE memories SET metadata=jsonb_set(metadata, '{confidence}', '0.8') WHERE id=$1",
                memory_id,
            )
            assert (
                await conn.fetchval(
                    "SELECT change_kind FROM belief_update_log WHERE memory_id=$1 ORDER BY log_id DESC LIMIT 1",
                    memory_id,
                )
                == "confidence_change"
            )

            await conn.execute(
                """
                UPDATE memories
                SET metadata=jsonb_set(
                    metadata,
                    '{contradicting_sources}',
                    '[{"kind":"document","ref":"contract:2","trust":0.9}]'::jsonb
                ), trust_level=0.3
                WHERE id=$1
                """,
                memory_id,
            )
            latest = await conn.fetchrow(
                """
                SELECT change_kind, new_value
                FROM belief_update_log
                WHERE memory_id=$1
                ORDER BY log_id DESC LIMIT 1
                """,
                memory_id,
            )
            assert latest["change_kind"] == "contradiction"
            assert _json(latest["new_value"])["contradicting_sources"] == 1
        finally:
            await tr.rollback()


async def test_notify_rate_limit_never_drops_durable_events(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "SELECT set_config('belief.propagation_notify_per_minute', '0'::jsonb)"
            )
            ids = [
                await _memory(conn, f"Rate-limited belief {index}")
                for index in range(3)
            ]
            rows = await conn.fetch(
                """
                SELECT notified, context
                FROM belief_update_log
                WHERE memory_id = ANY($1::uuid[])
                ORDER BY log_id
                """,
                ids,
            )
            assert len(rows) == 3
            assert all(row["notified"] is False for row in rows)
            assert all(
                _json(row["context"])["notification_suppressed"] == "rate_limit"
                for row in rows
            )
        finally:
            await tr.rollback()


async def test_supersession_and_reversion_propagate_as_explicit_events(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            old_id = await _memory(conn, "The old belief")
            replacement_id = await _memory(conn, "The corrected belief")
            await conn.execute(
                "DELETE FROM belief_update_log WHERE memory_id = ANY($1::uuid[])",
                [old_id, replacement_id],
            )
            event_id = await conn.fetchval(
                "SELECT record_supersession($1, $2, 'operator correction', 'operator')",
                old_id,
                replacement_id,
            )
            assert (
                await conn.fetchval(
                    "SELECT change_kind FROM belief_update_log WHERE memory_id=$1 ORDER BY log_id DESC LIMIT 1",
                    old_id,
                )
                == "supersession"
            )
            assert await conn.fetchval(
                "SELECT revert_supersession($1, 'restored after review', 'operator')",
                event_id,
            )
            kinds = await conn.fetch(
                "SELECT change_kind FROM belief_update_log WHERE memory_id=$1 ORDER BY log_id",
                old_id,
            )
            assert [row["change_kind"] for row in kinds] == [
                "supersession",
                "reversion",
            ]
        finally:
            await tr.rollback()


async def test_recent_surface_delivery_receipt_and_retention(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            memory_id = await _memory(conn, "A surfaced belief update")
            event = await conn.fetchrow(
                "SELECT log_id, fired_at FROM belief_update_log WHERE memory_id=$1",
                memory_id,
            )
            updates = _json(
                await conn.fetchval(
                    "SELECT recent_belief_updates_json(5, $1)",
                    event["fired_at"] - timedelta(seconds=1),
                )
            )
            assert updates[0]["memory_id"] == str(memory_id)
            assert updates[0]["content"] == "A surfaced belief update"
            rendered = await conn.fetchval(
                "SELECT render_belief_updates($1::jsonb)", json.dumps(updates)
            )
            assert "Belief changes since your last heartbeat" in rendered
            assert "A surfaced belief update" in rendered

            assert await conn.fetchval(
                "SELECT record_belief_update_delivery($1, 'heartbeat', $2::jsonb)",
                event["log_id"],
                json.dumps({"path": "listen"}),
            )
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM belief_update_deliveries WHERE log_id=$1 AND subscriber='heartbeat'",
                    event["log_id"],
                )
                == 1
            )

            await conn.execute(
                "UPDATE belief_update_log SET fired_at=$2 WHERE log_id=$1",
                event["log_id"],
                datetime.now(timezone.utc) - timedelta(hours=3),
            )
            assert await conn.fetchval("SELECT prune_belief_update_log(1)") == 1
            assert not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM belief_update_deliveries WHERE log_id=$1)",
                event["log_id"],
            )
        finally:
            await tr.rollback()


async def test_start_heartbeat_definition_surfaces_prior_events(db_pool):
    async with db_pool.acquire() as conn:
        definition = await conn.fetchval(
            "SELECT pg_get_functiondef('start_heartbeat()'::regprocedure)"
        )
        assert (
            "recent_belief_updates_json(20, state_record.last_heartbeat_at)"
            in definition
        )
