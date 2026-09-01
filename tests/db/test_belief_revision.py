"""Belief revision (#35/#36): the residual_v1 evidence policy must be
calibrated (exact math), independence-aware, bounded, symmetric, audited,
and respectful of protected memories.
"""
from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _coerce_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


async def _stub_get_embedding(conn):
    await conn.execute(
        """
        CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
        RETURNS vector[] AS $$
            SELECT COALESCE(
                array_agg((
                    ARRAY[1.0::float] ||
                    array_fill(0.0::float, ARRAY[embedding_dimension() - 1])
                )::vector),
                ARRAY[]::vector[]
            )
            FROM unnest(text_contents)
        $$ LANGUAGE sql;
        """
    )


async def _seed_belief(conn, content: str, confidence: float = 0.5, protected: bool = False) -> str:
    meta = {"confidence": confidence}
    if protected:
        meta["protected"] = True
    return str(
        await conn.fetchval(
            """
            INSERT INTO memories (type, content, embedding, importance, trust_level, status, metadata)
            VALUES ('semantic', $1, array_fill(0.1, ARRAY[embedding_dimension()])::vector,
                    0.8, 0.3, 'active', $2::jsonb)
            RETURNING id
            """,
            content,
            json.dumps(meta),
        )
    )


async def _revise(conn, memory_id: str, ref: str, stance: str, trust: float = 0.8) -> dict:
    return _coerce_json(
        await conn.fetchval(
            "SELECT revise_memory_confidence($1::uuid, $2::jsonb, $3::text, 'test')",
            memory_id,
            json.dumps({"kind": "test_source", "ref": ref, "trust": trust}),
            stance,
        )
    )


async def test_supports_applies_residual_formula(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mid = await _seed_belief(conn, "belief: residual support", confidence=0.5)
            r = await _revise(conn, mid, "doc-a", "supports", trust=0.8)
            # 0.5 + (1 - 0.5) * 0.35 * 0.8 = 0.64
            assert r["applied"] is True
            assert r["independent"] is True
            assert abs(r["posterior"] - 0.64) < 1e-9
            stored = _coerce_json(
                await conn.fetchval(
                    "SELECT metadata->>'confidence' FROM memories WHERE id = $1::uuid", mid
                )
            )
            assert abs(float(stored) - 0.64) < 1e-9
        finally:
            await tr.rollback()


async def test_contradicts_applies_symmetric_formula(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mid = await _seed_belief(conn, "belief: residual contradiction", confidence=0.64)
            r = await _revise(conn, mid, "doc-b", "contradicts", trust=0.8)
            # 0.64 * (1 - 0.35 * 0.8) = 0.4608
            assert r["applied"] is True
            assert abs(r["posterior"] - 0.4608) < 1e-9
            # Contradicting sources never inflate the supporting evidence set.
            meta = _coerce_json(
                await conn.fetchval("SELECT metadata FROM memories WHERE id = $1::uuid", mid)
            )
            assert len(meta.get("contradicting_sources", [])) == 1
            assert not meta.get("source_references")
        finally:
            await tr.rollback()


async def test_duplicate_source_never_moves_confidence(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mid = await _seed_belief(conn, "belief: dedupe", confidence=0.5)
            first = await _revise(conn, mid, "same-ref", "supports")
            second = await _revise(conn, mid, "same-ref", "supports")
            assert first["applied"] is True
            assert second["applied"] is False
            assert second["reason"] == "duplicate_source"
            assert second["independent"] is False
            assert second["posterior"] == first["posterior"]
            # A known supporter flipping stance is still non-independent.
            flipped = await _revise(conn, mid, "same-ref", "contradicts")
            assert flipped["applied"] is False
            assert flipped["reason"] == "duplicate_source"
        finally:
            await tr.rollback()


async def test_confidence_floor_and_ceiling_hold(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mid = await _seed_belief(conn, "belief: bounded", confidence=0.10)
            for i in range(20):
                r = await _revise(conn, mid, f"contra-{i}", "contradicts", trust=1.0)
            assert r["posterior"] >= 0.05 - 1e-9

            mid2 = await _seed_belief(conn, "belief: bounded high", confidence=0.9)
            for i in range(20):
                r = await _revise(conn, mid2, f"supp-{i}", "supports", trust=1.0)
            assert r["posterior"] <= 0.99 + 1e-9
        finally:
            await tr.rollback()


async def test_independent_support_raises_trust(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mid = await _seed_belief(conn, "belief: trust resync", confidence=0.5)
            before = await conn.fetchval(
                "SELECT trust_level FROM memories WHERE id = $1::uuid", mid
            )
            r = await _revise(conn, mid, "doc-trust", "supports", trust=0.9)
            after = await conn.fetchval(
                "SELECT trust_level FROM memories WHERE id = $1::uuid", mid
            )
            assert r["applied"] is True
            assert after > before
        finally:
            await tr.rollback()


async def test_disabled_config_records_but_does_not_apply(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "UPDATE config SET value = 'false'::jsonb WHERE key = 'belief.revision_enabled'"
            )
            mid = await _seed_belief(conn, "belief: disabled", confidence=0.5)
            r = await _revise(conn, mid, "doc-disabled", "supports")
            assert r["applied"] is False
            assert r["reason"] == "disabled"
            assert r["posterior"] == r["prior"]
            # Source is still merged for provenance.
            meta = _coerce_json(
                await conn.fetchval("SELECT metadata FROM memories WHERE id = $1::uuid", mid)
            )
            assert any(s.get("ref") == "doc-disabled" for s in meta.get("source_references", []))
        finally:
            await tr.rollback()


async def test_protected_memory_is_questioned_not_rewritten(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mid = await _seed_belief(conn, "belief: protected", confidence=0.9, protected=True)
            trust_before = await conn.fetchval(
                "SELECT trust_level FROM memories WHERE id = $1::uuid", mid
            )
            contra = await _revise(conn, mid, "doc-contra", "contradicts")
            assert contra["applied"] is False
            assert contra["reason"] == "protected"
            meta = _coerce_json(
                await conn.fetchval("SELECT metadata FROM memories WHERE id = $1::uuid", mid)
            )
            # The contradiction is visible, the belief unchanged.
            assert len(meta.get("contradicting_sources", [])) == 1
            assert float(meta["confidence"]) == 0.9

            # Supports still applies, but pinned trust never recomputes.
            supp = await _revise(conn, mid, "doc-supp", "supports")
            assert supp["applied"] is True
            trust_after = await conn.fetchval(
                "SELECT trust_level FROM memories WHERE id = $1::uuid", mid
            )
            assert trust_after == trust_before
        finally:
            await tr.rollback()


async def test_every_call_writes_an_audit_row(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mid = await _seed_belief(conn, "belief: audited", confidence=0.5)
            await _revise(conn, mid, "doc-1", "supports")
            await _revise(conn, mid, "doc-1", "supports")   # duplicate
            await _revise(conn, mid, "doc-2", "contradicts")
            rows = await conn.fetch(
                "SELECT * FROM belief_revision_audit WHERE memory_id = $1::uuid ORDER BY created_at",
                mid,
            )
            assert len(rows) == 3
            assert [r["reason"] for r in rows] == ["applied", "duplicate_source", "applied"]
            for row in rows:
                assert row["record_digest_v1"] and len(row["record_digest_v1"]) == 64
                record = _coerce_json(row["record"])
                assert record["prior"] is not None
                assert record["posterior"] is not None
        finally:
            await tr.rollback()


async def test_add_memory_evidence_creates_edge_and_evidence_node(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            mid = await _seed_belief(conn, "belief: evidence edges", confidence=0.5)
            r = _coerce_json(
                await conn.fetchval(
                    "SELECT add_memory_evidence($1::uuid, 'supports', $2::jsonb, $3::text)",
                    mid,
                    json.dumps({"kind": "origin_document", "ref": "docs/origin.md", "trust": 0.9}),
                    "The origin document states this directly.",
                )
            )
            assert r["applied"] is True
            evidence_id = r["evidence_memory_id"]
            assert evidence_id
            ev_type = await conn.fetchval(
                "SELECT type::text FROM memories WHERE id = $1::uuid", evidence_id
            )
            assert ev_type == "episodic"
            edge = await conn.fetchrow(
                "SELECT * FROM memory_edges WHERE src_id = $1 AND dst_id = $2 AND rel_type = 'SUPPORTS'",
                evidence_id,
                mid,
            )
            assert edge is not None
        finally:
            await tr.rollback()


async def test_non_semantic_target_is_rejected(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            eid = str(
                await conn.fetchval(
                    """
                    INSERT INTO memories (type, content, embedding, importance, trust_level, status)
                    VALUES ('episodic', 'an event', array_fill(0.1, ARRAY[embedding_dimension()])::vector,
                            0.5, 0.9, 'active')
                    RETURNING id
                    """
                )
            )
            r = await _revise(conn, eid, "doc-x", "supports")
            assert r["applied"] is False
            assert r["reason"] == "not_semantic"
        finally:
            await tr.rollback()


async def _onehot(conn, position: int):
    """A vector with 1.0 at `position` and 0.0 elsewhere: identical positions
    give cosine similarity 1.0, distinct positions give exactly 0.0 --
    deterministic near/far fixtures for similarity-threshold tests."""
    return await conn.fetchval(
        """
        SELECT (
            array_fill(0.0::float, ARRAY[$1::int])
            || ARRAY[1.0::float]
            || array_fill(0.0::float, ARRAY[(embedding_dimension() - $1 - 1)::int])
        )::vector
        """,
        position,
    )


async def test_add_evidence_tool_surfaces_linked_candidate_on_non_semantic_target(
    db_pool,
):
    """#101: add_evidence no longer dead-ends on an episodic memory_id -- a
    semantic belief already graph-linked to it (e.g. DERIVED_FROM) is
    surfaced by name, and retrying against it succeeds outright."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            v = await _onehot(conn, 0)
            episodic_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, importance, trust_level, status)
                VALUES ('episodic', 'Eric described a mini-heartbeat trigger', $1::vector,
                        0.5, 0.9, 'active')
                RETURNING id
                """,
                v,
            )
            belief_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, importance, trust_level, status, metadata)
                VALUES ('semantic', 'Mini-heartbeats should trigger on crash alerts',
                        $1::vector, 0.6, 0.8, 'active', '{"confidence": 0.6}'::jsonb)
                RETURNING id
                """,
                v,
            )
            await conn.execute(
                "SELECT create_memory_relationship($1::uuid, $2::uuid, 'DERIVED_FROM'::graph_edge_type, '{}'::jsonb)",
                belief_id,
                episodic_id,
            )

            blocked = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('add_evidence', $1::jsonb)",
                    json.dumps(
                        {
                            "memory_id": str(episodic_id),
                            "stance": "supports",
                            "source": {"ref": "test-101"},
                        }
                    ),
                )
            )
            assert blocked["success"] is False
            assert str(belief_id) in blocked["error"]
            assert "call add_evidence again" in blocked["error"]
            assert blocked["candidates"][0]["memory_id"] == str(belief_id)
            assert blocked["candidates"][0]["match_reason"] == "linked"

            retried = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('add_evidence', $1::jsonb)",
                    json.dumps(
                        {
                            "memory_id": str(belief_id),
                            "stance": "supports",
                            "source": {"ref": "test-101"},
                        }
                    ),
                )
            )
            assert retried["success"] is True
        finally:
            await tr.rollback()


async def test_add_evidence_tool_falls_back_to_similar_candidate_above_threshold(
    db_pool,
):
    """#101: with no graph edge at all, a semantic memory close enough in
    embedding space is still surfaced (as a 'similar' match), but an
    unrelated one (orthogonal embedding) never is."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            near = await _onehot(conn, 1)
            far = await _onehot(conn, 7)
            episodic_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, importance, trust_level, status)
                VALUES ('episodic', 'an unlinked observation', $1::vector, 0.5, 0.9, 'active')
                RETURNING id
                """,
                near,
            )
            related_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, importance, trust_level, status, metadata)
                VALUES ('semantic', 'a closely related belief', $1::vector,
                        0.6, 0.8, 'active', '{"confidence": 0.55}'::jsonb)
                RETURNING id
                """,
                near,
            )
            await conn.execute(
                """
                INSERT INTO memories (type, content, embedding, importance, trust_level, status, metadata)
                VALUES ('semantic', 'a totally unrelated belief', $1::vector,
                        0.6, 0.8, 'active', '{}'::jsonb)
                """,
                far,
            )

            result = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('add_evidence', $1::jsonb)",
                    json.dumps(
                        {
                            "memory_id": str(episodic_id),
                            "stance": "supports",
                            "source": {"ref": "test-101b"},
                        }
                    ),
                )
            )
            assert result["success"] is False
            ids = [c["memory_id"] for c in result["candidates"]]
            assert ids == [str(related_id)]
            assert result["candidates"][0]["match_reason"] == "similar"
        finally:
            await tr.rollback()


async def test_add_evidence_tool_no_candidates_still_names_recovery_step(db_pool):
    """#101: with nothing linked or similar enough, the caller still gets an
    explicit next step (recall by type) rather than a bare rejection."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            v = await _onehot(conn, 2)
            episodic_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding, importance, trust_level, status)
                VALUES ('episodic', 'a lone unrelated event', $1::vector, 0.5, 0.9, 'active')
                RETURNING id
                """,
                v,
            )
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('add_evidence', $1::jsonb)",
                    json.dumps(
                        {
                            "memory_id": str(episodic_id),
                            "stance": "supports",
                            "source": {"ref": "test-101c"},
                        }
                    ),
                )
            )
            assert result["success"] is False
            assert result["candidates"] == []
            assert "memory_types=['semantic']" in result["error"]
        finally:
            await tr.rollback()
