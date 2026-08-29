from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "db"
    / "migrations"
    / "0233_contradiction_events.sql"
)


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _memory(conn, content: str, created_at: datetime):
    return await conn.fetchval(
        """
        INSERT INTO memories (
            type, content, embedding, embedding_status, status,
            created_at, valid_from, source_attribution, trust_level
        ) VALUES (
            'semantic', $1,
            array_fill(0.1::float, ARRAY[embedding_dimension()])::vector,
            'embedded', 'active', $2, $2,
            '{"kind":"user_testimony","ref":"test"}'::jsonb, 0.9
        )
        RETURNING id
        """,
        content,
        created_at,
    )


async def _case(conn, *, confidence: float = 0.91):
    old_at = datetime.now(timezone.utc) - timedelta(days=30)
    new_at = datetime.now(timezone.utc) - timedelta(days=1)
    old_id = await _memory(conn, "The Manning retainer is paid monthly", old_at)
    new_id = await _memory(conn, "The Manning retainer is paid quarterly", new_at)
    result = _json(
        await conn.fetchval(
            """
            SELECT file_contradiction_case(
                $1, $2, $2, 'The payment cadence conflicts.', $3,
                'test', '{"fixture":true}'::jsonb
            )
            """,
            old_id,
            new_id,
            confidence,
        )
    )
    return old_id, new_id, result


async def test_detection_queue_claims_candidates_and_retries_durably(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            old_id, new_id, _ = await _case(conn)
            await conn.execute("DELETE FROM contradiction_cases")

            claimed = _json(
                await conn.fetchval(
                    "SELECT claim_contradiction_detection_batch(10, TRUE)"
                )
            )
            items = claimed["items"]
            item_by_memory = {
                item["memory"]["memory_id"]: item for item in items
            }
            queue_ids = [item["queue_id"] for item in items]

            retried = _json(
                await conn.fetchval(
                    """
                    SELECT finish_contradiction_detection_batch(
                        $1::uuid[], '{}'::jsonb, 'temporary model failure'
                    )
                    """,
                    queue_ids,
                )
            )
            queue_rows = await conn.fetch(
                """
                SELECT status, attempts, error, next_attempt_at > CURRENT_TIMESTAMP AS delayed
                FROM contradiction_detection_queue
                WHERE id = ANY($1::uuid[])
                """,
                queue_ids,
            )
        finally:
            await transaction.rollback()

    assert claimed["skipped"] is False
    assert {str(old_id), str(new_id)}.issubset(item_by_memory)
    assert any(
        candidate["memory_id"] == str(old_id)
        for candidate in item_by_memory[str(new_id)]["candidates"]
    )
    assert retried == {"completed": 0, "retried": len(queue_ids), "failed": 0}
    assert all(row["status"] == "pending" for row in queue_rows)
    assert all(row["attempts"] == 1 for row in queue_rows)
    assert all(row["error"] == "temporary model failure" for row in queue_rows)
    assert all(row["delayed"] for row in queue_rows)


async def test_threshold_and_subconscious_observations_share_one_ledger(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            old_id, new_id, below = await _case(conn, confidence=0.4)
            before = await conn.fetchval("SELECT count(*) FROM contradiction_cases")
            await conn.fetchval(
                """
                SELECT create_strategic_memory(
                    'The contract evidence conflicts with prior testimony.',
                    'Unresolved tension between beliefs',
                    0.93,
                    jsonb_build_object(
                        'kind', 'contradiction',
                        'memory_a', $1::uuid,
                        'memory_b', $2::uuid,
                        'tension', 'The payment cadence conflicts.',
                        'confidence', 0.93
                    )
                )
                """,
                old_id,
                new_id,
            )
            cases = _json(await conn.fetchval("SELECT list_contradiction_cases('pending', 10)"))
        finally:
            await transaction.rollback()

    assert below == {"created": False, "reason": "below_threshold", "confidence": 0.4}
    assert before == 0
    assert len(cases) == 1
    assert cases[0]["detected_by"] == "subconscious"
    assert cases[0]["memory_a"]["id"] == str(old_id)
    assert cases[0]["memory_b"]["id"] == str(new_id)


@pytest.mark.parametrize(
    ("outcome", "expected_winner", "expected_loser"),
    [("new_right", "new", "old"), ("old_right", "old", "new")],
)
async def test_explicit_resolution_closes_loser_but_keeps_history(
    db_pool, outcome, expected_winner, expected_loser
):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            old_id, new_id, filed = await _case(conn)
            result = _json(
                await conn.fetchval(
                    "SELECT decide_contradiction($1::uuid, $2, 'Test decision', 'web', 'operator')",
                    filed["case_id"],
                    outcome,
                )
            )
            repeated = _json(
                await conn.fetchval(
                    "SELECT decide_contradiction($1::uuid, $2, NULL, 'web', 'operator')",
                    filed["case_id"],
                    outcome,
                )
            )
            memories = {
                row["id"]: row
                for row in await conn.fetch(
                    """
                    SELECT id, status, valid_until, superseded_by
                    FROM memories WHERE id = ANY($1::uuid[])
                    """,
                    [old_id, new_id],
                )
            }
            ledger = await conn.fetchrow(
                "SELECT * FROM contradiction_cases WHERE id=$1::uuid",
                filed["case_id"],
            )
            supersession = await conn.fetchrow(
                "SELECT * FROM memory_supersessions WHERE id=$1::uuid",
                result["supersession_id"],
            )
        finally:
            await transaction.rollback()

    ids = {"old": old_id, "new": new_id}
    winner = ids[expected_winner]
    loser = ids[expected_loser]
    assert result["ok"] is True
    assert result["winner_memory_id"] == str(winner)
    assert result["loser_memory_id"] == str(loser)
    assert repeated["already_decided"] is True
    assert memories[loser]["status"] == "active"
    assert memories[loser]["valid_until"] is not None
    assert memories[loser]["superseded_by"] == winner
    assert memories[winner]["valid_until"] is None
    assert ledger["status"] == "resolved"
    assert ledger["outcome"] == outcome
    assert supersession["superseded_memory_id"] == loser
    assert supersession["replacement_memory_id"] == winner


async def test_accepting_tension_preserves_both_memories(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            old_id, new_id, filed = await _case(conn)
            result = _json(
                await conn.fetchval(
                    "SELECT decide_contradiction($1::uuid, 'tension', NULL, 'web', 'operator')",
                    filed["case_id"],
                )
            )
            rows = await conn.fetch(
                "SELECT id, valid_until, superseded_by FROM memories WHERE id=ANY($1::uuid[])",
                [old_id, new_id],
            )
        finally:
            await transaction.rollback()

    assert result["status"] == "tension"
    assert "supersession_id" not in result
    assert all(row["valid_until"] is None for row in rows)
    assert all(row["superseded_by"] is None for row in rows)


async def test_daily_digest_batches_without_deciding_and_heartbeat_cannot_choose(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            old_id, new_id, filed = await _case(conn)
            heartbeat = _json(
                await conn.fetchval(
                    """
                    SELECT execute_heartbeat_action(
                        gen_random_uuid(), 'resolve_contradiction',
                        jsonb_build_object('case_id', $1::uuid, 'resolution', 'Prefer newer')
                    )
                    """,
                    filed["case_id"],
                )
            )
            first = _json(
                await conn.fetchval("SELECT publish_contradiction_digest_if_due(TRUE)")
            )
            second = _json(
                await conn.fetchval("SELECT publish_contradiction_digest_if_due(FALSE)")
            )
            case = await conn.fetchrow(
                "SELECT status, outcome, proposed_at, outbox_message_id FROM contradiction_cases WHERE id=$1::uuid",
                filed["case_id"],
            )
            envelope = _json(
                await conn.fetchval(
                    "SELECT envelope FROM outbox_messages WHERE id=$1", first["outbox_message_id"]
                )
            )
            validity = await conn.fetch(
                "SELECT valid_until FROM memories WHERE id=ANY($1::uuid[])",
                [old_id, new_id],
            )
        finally:
            await transaction.rollback()

    assert heartbeat["success"] is True
    assert heartbeat["result"]["decision_required"] is True
    assert heartbeat["result"]["changed_memories"] is False
    assert first["skipped"] is False
    assert first["count"] == 1
    assert second == {"skipped": True, "reason": "not_due"}
    assert case["status"] == "pending"
    assert case["outcome"] is None
    assert case["proposed_at"] is not None
    assert case["outbox_message_id"] is not None
    assert envelope["payload"]["intent"] == "contradiction_review"
    assert filed["code"] in envelope["payload"]["message"]
    assert all(row["valid_until"] is None for row in validity)


async def test_coded_private_reply_and_migration_replay_are_idempotent(db_pool):
    migration = _MIGRATION.read_text(encoding="utf-8")
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            _, _, filed = await _case(conn)
            ignored = _json(
                await conn.fetchval(
                    "SELECT try_resolve_contradiction_from_inbound('signal', 'operator', '1')"
                )
            )
            decided = _json(
                await conn.fetchval(
                    "SELECT try_resolve_contradiction_from_inbound('signal', 'operator', $1)",
                    f"3 {filed['code']}",
                )
            )
            await conn.execute(migration)
            await conn.execute(migration)
            preserved = await conn.fetchrow(
                "SELECT status, outcome FROM contradiction_cases WHERE id=$1::uuid",
                filed["case_id"],
            )
        finally:
            await transaction.rollback()

    assert ignored == {"recognized": False, "matched": False}
    assert decided["recognized"] is True
    assert decided["matched"] is True
    assert decided["status"] == "tension"
    assert preserved["status"] == "tension"
    assert preserved["outcome"] == "tension"
