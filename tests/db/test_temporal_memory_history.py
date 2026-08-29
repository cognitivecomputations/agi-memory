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
    / "0234_temporal_memory_history.sql"
)


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _memory(
    conn,
    content: str,
    at: datetime,
    *,
    confidence: float = 0.6,
    sensitivity: str | None = None,
):
    source = {
        "kind": "user_testimony",
        "ref": f"temporal-test:{content}",
        "label": "Temporal test source",
        "trust": 0.8,
    }
    if sensitivity:
        source["sensitivity"] = sensitivity
    return await conn.fetchval(
        """
        INSERT INTO memories (
            type, content, embedding, embedding_status, status,
            created_at, valid_from, source_attribution, trust_level, metadata
        ) VALUES (
            'semantic', $1,
            array_fill(0.1::float, ARRAY[embedding_dimension()])::vector,
            'embedded', 'active', $2, $2, $3::jsonb, 0.8,
            jsonb_build_object(
                'confidence', $4::float,
                'source_references', jsonb_build_array($3::jsonb)
            )
        )
        RETURNING id
        """,
        content,
        at,
        json.dumps(source),
        confidence,
    )


async def test_point_in_time_recall_returns_the_claim_valid_then(db_pool):
    now = datetime.now(timezone.utc)
    old_at = now - timedelta(days=60)
    replacement_at = now - timedelta(days=30)
    decision_at = now - timedelta(days=20)
    before = now - timedelta(days=40)
    after = now - timedelta(days=10)

    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            old_id = await _memory(conn, "The Manning retainer is monthly", old_at)
            new_id = await _memory(conn, "The Manning retainer is quarterly", replacement_at)
            await conn.fetchval(
                """
                SELECT record_supersession(
                    $1, $2, 'Signed contract changed the payment cadence',
                    'operator', 'active', $3
                )
                """,
                old_id,
                new_id,
                decision_at,
            )

            before_snapshot = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('Manning retainer', $1, 10, NULL, 0, FALSE)",
                    before,
                )
            )
            after_snapshot = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('Manning retainer', $1, 10, NULL, 0, FALSE)",
                    after,
                )
            )
        finally:
            await transaction.rollback()

    before_ids = {item["memory_id"] for item in before_snapshot["memories"]}
    after_ids = {item["memory_id"] for item in after_snapshot["memories"]}
    assert str(old_id) in before_ids
    assert str(new_id) not in before_ids
    assert str(old_id) not in after_ids
    assert str(new_id) in after_ids
    old = next(item for item in before_snapshot["memories"] if item["memory_id"] == str(old_id))
    assert old["citation_id"] == f"mem-{old_id}"
    assert old["citation"]["trust_level"] == 0.8
    assert old["valid_from"].startswith(str(old_at.date()))


async def test_history_diff_explains_addition_expiry_and_supersession(db_pool):
    now = datetime.now(timezone.utc)
    from_time = now - timedelta(days=60)
    new_at = now - timedelta(days=30)
    decision_at = now - timedelta(days=20)

    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            old_id = await _memory(conn, "The Manning retainer is monthly", from_time - timedelta(days=1))
            new_id = await _memory(conn, "The Manning retainer is quarterly", new_at)
            await conn.fetchval(
                """
                SELECT record_supersession(
                    $1, $2, 'Signed contract changed the payment cadence',
                    'operator', 'active', $3
                )
                """,
                old_id,
                new_id,
                decision_at,
            )
            result = _json(
                await conn.fetchval(
                    "SELECT diff_memory_history('Manning retainer', $1, $2, 10, NULL, 0, FALSE)",
                    from_time,
                    now,
                )
            )
        finally:
            await transaction.rollback()

    assert {item["memory_id"] for item in result["expired"]} == {str(old_id)}
    assert {item["memory_id"] for item in result["added"]} == {str(new_id)}
    assert result["summary"]["supersessions"] == 1
    event = result["supersessions"][0]
    assert event["reason"] == "Signed contract changed the payment cadence"
    assert event["actor"] == "operator"
    assert event["superseded_memory_id"] == str(old_id)
    assert event["replacement_memory_id"] == str(new_id)


async def test_reverted_supersession_reconstructs_the_closed_interval(db_pool):
    now = datetime.now(timezone.utc)
    old_at = now - timedelta(days=60)
    replacement_at = now - timedelta(days=30)
    superseded_at = now - timedelta(days=20)
    during = now - timedelta(days=10)

    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            old_id = await _memory(conn, "The venue is Hartford", old_at)
            new_id = await _memory(conn, "The venue is Boston", replacement_at)
            event_id = await conn.fetchval(
                "SELECT record_supersession($1, $2, 'Tentative correction', 'operator', 'active', $3)",
                old_id,
                new_id,
                superseded_at,
            )
            await conn.fetchval(
                "SELECT revert_supersession($1, 'Correction was mistaken', 'operator')",
                event_id,
            )
            during_snapshot = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('venue', $1, 10, NULL, 0, FALSE)",
                    during,
                )
            )
            after_snapshot = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('venue', CURRENT_TIMESTAMP, 10, NULL, 0, FALSE)"
                )
            )
        finally:
            await transaction.rollback()

    assert str(old_id) not in {item["memory_id"] for item in during_snapshot["memories"]}
    assert str(old_id) in {item["memory_id"] for item in after_snapshot["memories"]}


async def test_historical_confidence_comes_from_append_only_revision_audit(db_pool):
    now = datetime.now(timezone.utc)
    memory_at = now - timedelta(days=30)
    before_revision = now - timedelta(days=1)

    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            memory_id = await _memory(
                conn, "The launch risk is moderate", memory_at, confidence=0.5
            )
            revision = _json(
                await conn.fetchval(
                    """
                    SELECT revise_memory_confidence(
                        $1,
                        '{"kind":"document","ref":"risk-review","trust":0.9}'::jsonb,
                        'supports',
                        'temporal-test'
                    )
                    """,
                    memory_id,
                )
            )
            historical = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('launch risk', $1, 10, NULL, 0, FALSE)",
                    before_revision,
                )
            )
            current = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('launch risk', CURRENT_TIMESTAMP, 10, NULL, 0, FALSE)"
                )
            )
        finally:
            await transaction.rollback()

    old = next(item for item in historical["memories"] if item["memory_id"] == str(memory_id))
    new = next(item for item in current["memories"] if item["memory_id"] == str(memory_id))
    assert old["confidence"] == 0.5
    assert new["confidence"] == pytest.approx(revision["posterior"])
    assert new["confidence"] > old["confidence"]


@pytest.mark.parametrize("inactive_status", ["archived", "invalidated"])
async def test_inactive_memory_is_historical_but_never_current(
    db_pool, inactive_status
):
    now = datetime.now(timezone.utc)
    created_at = now - timedelta(days=10)
    closed_at = now - timedelta(days=2)

    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            memory_id = await conn.fetchval(
                """
                INSERT INTO memories (
                    type, content, embedding, embedding_status, status,
                    created_at, updated_at, valid_from, source_attribution,
                    trust_level, metadata
                ) VALUES (
                    'semantic', $1,
                    array_fill(0.1::float, ARRAY[embedding_dimension()])::vector,
                    'embedded', $2::memory_status, $3, $4, $3,
                    '{"kind":"user_testimony","label":"Temporal test source"}'::jsonb,
                    0.8, '{"confidence":0.6}'::jsonb
                )
                RETURNING id
                """,
                f"The {inactive_status} temporal sentinel was once active",
                inactive_status,
                created_at,
                closed_at,
            )
            current_snapshot = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('temporal sentinel once active', $1, 10, NULL, 0, FALSE)",
                    now,
                )
            )
            before_close = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('temporal sentinel once active', $1, 10, NULL, 0, FALSE)",
                    closed_at - timedelta(seconds=1),
                )
            )
            after_close = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('temporal sentinel once active', $1, 10, NULL, 0, FALSE)",
                    closed_at + timedelta(seconds=1),
                )
            )
        finally:
            await transaction.rollback()

    assert str(memory_id) not in {
        item["memory_id"] for item in current_snapshot["memories"]
    }
    assert str(memory_id) in {
        item["memory_id"] for item in before_close["memories"]
    }
    assert str(memory_id) not in {
        item["memory_id"] for item in after_close["memories"]
    }


async def test_snapshot_falls_back_loudly_when_embedding_search_is_unavailable(
    db_pool,
):
    now = datetime.now(timezone.utc)
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            memory_id = await _memory(
                conn,
                "The lexical fallback marker is heliotrope",
                now - timedelta(days=2),
            )
            await conn.execute(
                """
                CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
                RETURNS vector[]
                LANGUAGE plpgsql
                AS $function$
                BEGIN
                    RAISE EXCEPTION 'embedding service intentionally unavailable';
                END;
                $function$
                """
            )
            result = _json(
                await conn.fetchval(
                    "SELECT temporal_memory_snapshot('lexical fallback heliotrope', $1, 10, NULL, 0, FALSE)",
                    now - timedelta(days=1),
                )
            )
        finally:
            await transaction.rollback()

    assert result["degraded"] is True
    assert result["retrieval_mode"] == "lexical"
    assert "Embedding search was unavailable" in result["degraded_reason"]
    assert str(memory_id) in {item["memory_id"] for item in result["memories"]}


async def test_dispatch_validation_and_group_sensitivity_wall(db_pool):
    now = datetime.now(timezone.utc)
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            private_id = await _memory(
                conn,
                "The private project codename is Lantern",
                now - timedelta(days=2),
                sensitivity="private",
            )
            private_result = _json(
                await conn.fetchval(
                    """
                    SELECT execute_memory_tool(
                        'recall_at_time',
                        jsonb_build_object(
                            'query', 'project codename',
                            'as_of', CURRENT_TIMESTAMP,
                            'min_score', 0,
                            'exclude_sensitive', TRUE
                        )
                    )
                    """
                )
            )
            future = _json(
                await conn.fetchval(
                    """
                    SELECT execute_memory_tool(
                        'recall_at_time',
                        jsonb_build_object(
                            'query', 'project',
                            'as_of', CURRENT_TIMESTAMP + INTERVAL '1 day'
                        )
                    )
                    """
                )
            )
            backwards = _json(
                await conn.fetchval(
                    """
                    SELECT execute_memory_tool(
                        'diff_memory_history',
                        jsonb_build_object(
                            'query', 'project',
                            'from_time', CURRENT_TIMESTAMP,
                            'to_time', CURRENT_TIMESTAMP - INTERVAL '1 day'
                        )
                    )
                    """
                )
            )
            bad_score = _json(
                await conn.fetchval(
                    """
                    SELECT execute_memory_tool(
                        'recall_at_time',
                        jsonb_build_object(
                            'query', 'project',
                            'as_of', CURRENT_TIMESTAMP,
                            'min_score', 'not-a-number'
                        )
                    )
                    """
                )
            )
            bad_types = _json(
                await conn.fetchval(
                    """
                    SELECT execute_memory_tool(
                        'recall_at_time',
                        jsonb_build_object(
                            'query', 'project',
                            'as_of', CURRENT_TIMESTAMP,
                            'memory_types', 'semantic'
                        )
                    )
                    """
                )
            )
        finally:
            await transaction.rollback()

    assert private_result["success"] is True
    assert str(private_id) not in {
        item["memory_id"] for item in private_result["output"]["memories"]
    }
    assert future == {
        "success": False,
        "error": "as_of cannot be in the future; choose now or an earlier instant",
        "error_type": "invalid_params",
    }
    assert backwards == {
        "success": False,
        "error": "from_time must be earlier than to_time",
        "error_type": "invalid_params",
    }
    assert bad_score["success"] is False
    assert bad_score["error_type"] == "invalid_params"
    assert "min_score" in bad_score["error"]
    assert bad_types == {
        "success": False,
        "error": "memory_types must be an array",
        "error_type": "invalid_params",
    }


async def test_temporal_migration_replay_preserves_history_and_prompt_cue(db_pool):
    migration = _MIGRATION.read_text(encoding="utf-8")
    now = datetime.now(timezone.utc)
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            memory_id = await _memory(
                conn, "Temporal migration sentinel", now - timedelta(days=1)
            )
            await conn.execute(migration)
            await conn.execute(migration)
            exists = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM memories WHERE id=$1)", memory_id)
            cue_count = await conn.fetchval(
                """
                SELECT (length(content) - length(replace(content, 'When the question is temporally framed', '')))
                       / length('When the question is temporally framed')
                FROM prompt_modules WHERE key='conversation'
                """
            )
        finally:
            await transaction.rollback()

    assert exists is True
    assert cue_count == 1
