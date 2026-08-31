from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


async def test_embedded_memory_invariant_is_enforced_at_write(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            memory_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, embedding_status)
                VALUES (
                    'semantic',
                    $1,
                    array_fill(0.0::float, ARRAY[embedding_dimension()])::vector,
                    'embedded'
                )
                RETURNING id
                """,
                f"zero-vector-invariant-{uuid.uuid4()}",
            )
            row = await conn.fetchrow(
                "SELECT embedding_status, vector_norm(embedding) AS norm FROM memories WHERE id = $1",
                memory_id,
            )
            constraint = await conn.fetchval("""
                SELECT pg_get_constraintdef(oid)
                FROM pg_constraint
                WHERE conrelid = 'memories'::regclass
                  AND conname = 'memories_embedded_vector_valid'
                """)
        finally:
            await tr.rollback()

    assert row["embedding_status"] == "pending"
    assert row["norm"] == 0
    assert "vector_norm(embedding)" in constraint


async def test_memory_ann_query_is_eligible_for_partial_hnsw_index(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # On an empty or tiny test database PostgreSQL reasonably prefers a
            # sequential scan. Disabling it for this EXPLAIN proves that the
            # predicate is eligible for the partial HNSW index; production use
            # is separately observable through pg_stat_user_indexes.idx_scan.
            await conn.execute("SET LOCAL enable_seqscan = off")
            query_vector = await conn.fetchval(
                "SELECT array_fill(0.1::float, ARRAY[embedding_dimension()])::vector::text"
            )
            plan_rows = await conn.fetch(
                """
                EXPLAIN (COSTS OFF)
                SELECT id
                FROM memories
                WHERE status = 'active'
                  AND embedding_status = 'embedded'
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT 5
                """,
                query_vector,
            )
            definitions = await conn.fetch("""
                SELECT proname, pg_get_functiondef(oid) AS definition
                FROM pg_proc
                WHERE pronamespace = 'public'::regnamespace
                  AND proname IN (
                      'recmem_recall_context',
                      'recmem_subconscious_vector_hits',
                      'search_similar_memories',
                      'sense_memory_availability'
                  )
                """)
        finally:
            await tr.rollback()

    plan = "\n".join(str(row[0]) for row in plan_rows)
    assert "Index Scan using idx_memories_embedding" in plan
    assert definitions
    for row in definitions:
        definition = row["definition"]
        assert "embedding <> zero_vec" not in definition
    recall = next(
        row["definition"]
        for row in definitions
        if row["proname"] == "recmem_recall_context"
    )
    assert "embedding_status = 'embedded'" in recall
