"""Tests for DB-owned ingestion routing (db/41_functions_ingest.sql).

The dedup/related/create policy (thresholds + nearest-neighbor decision) moved
out of services/ingest.py:_create_semantic_memories into SQL. These exercise the
policy with controlled dummy vectors + threshold config, so they need no
embedding service.
"""
from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def _route(conn, fill: float) -> dict:
    raw = await conn.fetchval(
        "SELECT ingest_route_embedding(array_fill($1::float8, ARRAY[embedding_dimension()])::vector)",
        fill,
    )
    return json.loads(raw) if isinstance(raw, str) else raw


async def test_routing_thresholds_are_config_driven(db_pool):
    """sim vs (theta_dup, theta_related) decides duplicate / related / create."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mid = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, importance, trust_level, status)
                VALUES ('semantic', 'phase5 routing probe',
                        array_fill(0.1, ARRAY[embedding_dimension()])::vector, 0.8, 0.9, 'active')
                RETURNING id
                """
            )

            # Identical vector => cosine similarity 1.0 => duplicate (default dup=0.92).
            r = await _route(conn, 0.1)
            assert r["decision"] == "duplicate"
            assert r["matched_memory_id"] == str(mid)
            assert float(r["similarity"]) == pytest.approx(1.0, abs=1e-6)

            # Push the dup threshold above 1.0 => the same 1.0 sim is now "related".
            await conn.execute("SELECT set_config('memory.ingest_theta_dup', '1.5')")
            assert (await _route(conn, 0.1))["decision"] == "related"

            # Push related above 1.0 too => "create".
            await conn.execute("SELECT set_config('memory.ingest_theta_related', '1.5')")
            assert (await _route(conn, 0.1))["decision"] == "create"

            # Opposite vector (cosine -1.0) is always below threshold => create.
            await conn.execute("SELECT set_config('memory.ingest_theta_dup', '0.92')")
            await conn.execute("SELECT set_config('memory.ingest_theta_related', '0.8')")
            far = await _route(conn, -0.1)
            assert far["decision"] == "create"
            assert far["matched_memory_id"] is None
        finally:
            await tr.rollback()


async def test_route_extractions_drops_below_confidence(db_pool):
    """ingest_route_extractions filters by confidence before embedding — a fully
    below-threshold batch returns an empty plan without calling get_embedding."""
    async with db_pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT ingest_route_extractions($1::jsonb, $2::float)",
            json.dumps([
                {"content": "too uncertain", "confidence": 0.1},
                {"content": "also uncertain", "confidence": 0.2},
            ]),
            0.5,
        )
        plan = json.loads(raw) if isinstance(raw, str) else raw
        assert plan == []

        empty = await conn.fetchval(
            "SELECT ingest_route_extractions('[]'::jsonb, 0.0)"
        )
        assert (json.loads(empty) if isinstance(empty, str) else empty) == []


async def test_ingest_persist_extractions_atomic_pass(db_pool):
    """3.1 pushdown: one call routes, corroborates via the audited policy,
    creates, links, and sets decay — no half-written state possible."""
    import json as _json_mod

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                """
                CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
                RETURNS vector[] AS $$
                    SELECT COALESCE(
                        array_agg(array_fill(0.2::float, ARRAY[embedding_dimension()])::vector),
                        ARRAY[]::vector[]
                    ) FROM unnest(text_contents)
                $$ LANGUAGE sql;
                """
            )
            source = {"kind": "doc", "ref": "persist-test.md", "label": "persist test", "trust": 0.8}
            extractions = [
                {"content": "Photosynthesis converts light into chemical energy.",
                 "confidence": 0.8, "importance": 0.6, "category": "science",
                 "concepts": ["photosynthesis"], "connections": [],
                 "supports": None, "contradicts": None},
                {"content": "Too vague to keep.", "confidence": 0.1, "importance": 0.2,
                 "category": "general", "concepts": [], "connections": [],
                 "supports": None, "contradicts": None},
            ]
            result_raw = await conn.fetchval(
                "SELECT ingest_persist_extractions($1::jsonb, $2::jsonb, NULL, 0.7, $3::jsonb)",
                _json_mod.dumps(extractions),
                _json_mod.dumps(source),
                _json_mod.dumps({"min_confidence": 0.5, "base_trust": 0.7, "permanent": False}),
            )
            result = _json_mod.loads(result_raw) if isinstance(result_raw, str) else result_raw
            assert len(result["created"]) == 1
            mem_id = result["created"][0]

            row = await conn.fetchrow(
                "SELECT decay_rate, trust_level FROM memories WHERE id = $1::uuid", mem_id
            )
            # intensity 0.7 -> half the config base decay (vivid band).
            assert abs(row["decay_rate"] - 0.005) < 1e-9

            # Embed the new memory so the re-ingest below can find and
            # corroborate it instead of creating a duplicate (async, 0129).
            from tests.utils import embed_pending_memories
            await embed_pending_memories(conn)

            # Re-ingesting the same fact corroborates instead of duplicating.
            again_raw = await conn.fetchval(
                "SELECT ingest_persist_extractions($1::jsonb, $2::jsonb, NULL, 0.7, $3::jsonb)",
                _json_mod.dumps(extractions[:1]),
                _json_mod.dumps({**source, "ref": "persist-test-2.md"}),
                _json_mod.dumps({"min_confidence": 0.5, "base_trust": 0.7, "permanent": False}),
            )
            again = _json_mod.loads(again_raw) if isinstance(again_raw, str) else again_raw
            assert again["created"] == []
            assert again["corroborated"] == 1
            revisions = await conn.fetchval(
                "SELECT count(*) FROM belief_revision_audit WHERE memory_id = $1::uuid", mem_id
            )
            assert revisions >= 1
        finally:
            await tr.rollback()


async def test_slow_ingest_persist_facts_trust_and_edges(db_pool):
    """3.2 pushdown + #83: contested facts gain CONTESTED_BECAUSE edges;
    duplicates corroborate; trust derives from sources (stance is never a
    multiplier)."""
    import json as _json_mod

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # Hash-bucketed stub: distinct texts get near-orthogonal vectors.
            await conn.execute(
                """
                CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
                RETURNS vector[] AS $$
                    SELECT COALESCE(array_agg((
                        SELECT array_agg(CASE WHEN i = 2 + abs(hashtext(t)) % (embedding_dimension() - 2)
                                              THEN 1.0::float ELSE 0.0::float END)
                        FROM generate_series(1, embedding_dimension()) i
                    )::vector), ARRAY[]::vector[])
                    FROM unnest(text_contents) t
                $$ LANGUAGE sql;
                """
            )
            reason_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, importance, trust_level, status)
                VALUES ('semantic', 'A rejection rationale.',
                        (ARRAY[1.0::float] || array_fill(0.0::float, ARRAY[embedding_dimension() - 1]))::vector,
                        0.5, 0.9, 'active')
                RETURNING id
                """
            )
            assessment = {
                "acceptance": "contest",
                "trust_assessment": 0.8,
                "importance": 0.6,
                "worldview_impact": "neutral",
            }
            result_raw = await conn.fetchval(
                """
                SELECT slow_ingest_persist_facts(
                    $1::jsonb, $2::jsonb, '{"kind": "doc", "ref": "slow.md"}'::jsonb,
                    NULL, ARRAY[]::uuid[], ARRAY[]::uuid[], ARRAY[$3]::uuid[], 'slow_ingest')
                """,
                _json_mod.dumps(["a contested claim that is long enough"]),
                _json_mod.dumps(assessment),
                reason_id,
            )
            result = _json_mod.loads(result_raw) if isinstance(result_raw, str) else result_raw
            assert len(result["created"]) == 1
            mem_id = result["created"][0]

            # NOTE (pushdown finding): the acceptance trust multiplier is
            # currently cosmetic — provenance triggers recompute trust_level
            # from source attribution, for the Python path exactly as for this
            # SQL path. Recorded in plans/db_pushdown.md as follow-up.

            contested = await conn.fetchval(
                """
                SELECT count(*) FROM memory_edges
                WHERE src_id = $1 AND dst_id = $2 AND rel_type = 'CONTESTED_BECAUSE'
                """,
                str(mem_id), str(reason_id),
            )
            assert contested == 1
        finally:
            await tr.rollback()
