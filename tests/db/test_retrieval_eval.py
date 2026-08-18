"""Retrieval eval (#96 Batch 1d): a seeded corpus with known-relevant
targets pins the fused ranker's behavior — targets rank on their home turf,
the knowledge tier is reachable, association expansion finds off-axis
neighbors, mood colors recall, and activation boosts genuinely lift
memories. This is the guard against silent recall regression and the seed
of the emergence suite (Batch 5).
"""
from __future__ import annotations

import json

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


async def _stub(conn):
    await conn.execute(
        """
        CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
        RETURNS vector[] AS $$
            SELECT COALESCE(array_agg((
                array_fill(0.01::float, ARRAY[2 + abs(hashtext(t)) % (embedding_dimension() - 2)]) ||
                ARRAY[1.0::float] ||
                array_fill(0.01::float, ARRAY[embedding_dimension() - 3 - abs(hashtext(t)) % (embedding_dimension() - 2)])
            )::vector), ARRAY[]::vector[])
            FROM unnest(text_contents) t
        $$ LANGUAGE sql;
        """
    )


async def _seed(conn, content, *, mem_type="semantic", axis=None, metadata=None,
                importance=0.5):
    """Seed a memory on the embedding axis of `axis` (defaults to content).
    Axis text is embedded in prefixed search form so it matches recall."""
    return await conn.fetchval(
        """
        INSERT INTO memories (type, content, embedding, importance, trust_level, status, metadata)
        VALUES ($1::memory_type, $2,
                (get_embedding(ARRAY[ensure_embedding_prefix($3, 'search_query')]))[1],
                $5, 0.8, 'active', COALESCE($4::jsonb, '{}'::jsonb))
        RETURNING id
        """,
        mem_type, content, axis or content, json.dumps(metadata) if metadata else None,
        importance,
    )


async def test_targets_rank_on_home_turf_and_knowledge_is_reachable(db_pool):
    m = get_test_identifier("eval")
    q = f"target retrieval question {m}"
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub(conn)
            # Targets get a strictly higher importance than distractors: the
            # stub's hash-position vectors can collide (~1/770 per distractor,
            # random test ids), and a similarity tie would hand top-1 to an
            # arbitrary row. Strength breaks the tie toward the target.
            sem = await _seed(conn, f"semantic target {m}", axis=q, importance=0.6)
            epi = await _seed(conn, f"episodic target {m}", mem_type="episodic", axis=q, importance=0.6)
            world = await _seed(conn, f"worldview target {m}", mem_type="worldview", axis=q, importance=0.6)
            proc = await _seed(conn, f"procedural target {m}", mem_type="procedural", axis=q, importance=0.6)
            for i in range(4):
                await _seed(conn, f"distractor {i} {m}", axis=f"unrelated axis {i} {m}")

            rows = await conn.fetch(
                "SELECT tier, item_id, score FROM recmem_recall_context($1, 0, 5, 5, NULL, FALSE, 5)", q
            )
            by_tier = {}
            for r in rows:
                by_tier.setdefault(r["tier"], []).append(r["item_id"])
        finally:
            await tr.rollback()

    assert by_tier["semantic"][0] == sem      # home-turf #1
    assert by_tier["episodic"][0] == epi
    knowledge = by_tier.get("knowledge", [])
    assert world in knowledge and proc in knowledge  # unreachable pre-fusion


async def test_association_expansion_reaches_off_axis_neighbors(db_pool):
    m = get_test_identifier("eval")
    q = f"association probe {m}"
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub(conn)
            # Same tie-break as above: a hash collision must not let the
            # neighbor tie the on-axis seed's similarity and flip line 109.
            seed_mem = await _seed(conn, f"on-axis seed {m}", axis=q, importance=0.6)
            neighbor = await _seed(conn, f"off-axis neighbor {m}", axis=f"totally different axis {m}")
            await _seed(conn, f"off-axis distractor {m}", axis=f"third axis {m}")
            await conn.execute(
                """
                INSERT INTO memory_neighborhoods (memory_id, neighbors, is_stale, computed_at)
                VALUES ($1, jsonb_build_object($2::text, 0.9), FALSE, CURRENT_TIMESTAMP)
                ON CONFLICT (memory_id) DO UPDATE
                SET neighbors = EXCLUDED.neighbors, is_stale = FALSE, computed_at = CURRENT_TIMESTAMP
                """,
                seed_mem, str(neighbor),
            )
            rows = await conn.fetch(
                "SELECT item_id, score FROM recmem_recall_context($1, 0, 0, 10, NULL, FALSE, 0) "
                "WHERE tier = 'semantic'", q
            )
            scores = {r["item_id"]: r["score"] for r in rows}
        finally:
            await tr.rollback()

    assert neighbor in scores  # spreading activation pulled it into candidates
    assert scores[seed_mem] > scores[neighbor]  # direct match still wins


async def test_mood_colors_recall(db_pool):
    m = get_test_identifier("eval")
    q = f"mood probe {m}"
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub(conn)
            joyful = await _seed(conn, f"joyful equal-sim {m}", axis=q, metadata={
                "emotional_context": {"valence": 0.8, "arousal": 0.7, "intensity": 0.6,
                                      "primary_emotion": "joy"}})
            somber = await _seed(conn, f"somber equal-sim {m}", axis=q, metadata={
                "emotional_context": {"valence": -0.8, "arousal": 0.7, "intensity": 0.6,
                                      "primary_emotion": "sadness"}})
            await conn.execute(
                "SELECT set_current_affective_state($1::jsonb)",
                json.dumps({"valence": 0.8, "arousal": 0.7, "intensity": 0.6,
                            "primary_emotion": "joy"}),
            )
            rows = await conn.fetch(
                "SELECT item_id, score FROM recmem_recall_context($1, 0, 0, 5, NULL, FALSE, 0) "
                "WHERE tier = 'semantic' AND item_id = ANY($2::uuid[]) ORDER BY score DESC",
                q, [joyful, somber],
            )
        finally:
            await tr.rollback()

    assert len(rows) == 2
    assert rows[0]["item_id"] == joyful  # congruent memory surfaces first


async def test_activation_boost_lifts_ranking(db_pool):
    m = get_test_identifier("eval")
    q = f"boost probe {m}"
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub(conn)
            boosted = await _seed(conn, f"boosted equal-sim {m}", axis=q,
                                  metadata={"activation_boost": 0.8})
            plain = await _seed(conn, f"plain equal-sim {m}", axis=q)
            rows = await conn.fetch(
                "SELECT item_id FROM recmem_recall_context($1, 0, 0, 5, NULL, FALSE, 0) "
                "WHERE tier = 'semantic' AND item_id = ANY($2::uuid[]) ORDER BY score DESC",
                q, [boosted, plain],
            )
        finally:
            await tr.rollback()

    assert len(rows) == 2
    assert rows[0]["item_id"] == boosted  # incubation/reward genuinely surfaces it
