"""HMX export pipeline against a real database (plans/hmx.md, Slice 1).

Seeds memories/episodes/edges, runs export_hmx, and pins the wire contract:
export-scoped refs, no embeddings, intent policy on sections, protected-section
digests that are stable across exports (ref independence in practice), and
supersession normalized to SUPERSEDES edges.
"""

from __future__ import annotations

import json
import uuid

import pytest

from core.memory_exchange import export_hmx, iter_hmx_jsonl, verify_mind_continuity

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_SEED_SQL = """
    INSERT INTO memories (type, content, embedding, importance, trust_level, status, metadata)
    VALUES ($1, $2, array_fill(0.1, ARRAY[embedding_dimension()])::vector,
            0.8, 0.9, 'active', $3::jsonb)
    RETURNING id
"""


async def _prepare(conn):
    await conn.execute("LOAD 'age'")
    await conn.execute("SET search_path = public, ag_catalog")


async def _seed_world(conn) -> dict:
    """A small but section-complete brain: semantic + worldview + goal memories,
    an episode with membership, a supporting edge, and a supersession."""
    ids: dict = {}
    ids["semantic"] = await conn.fetchval(
        _SEED_SQL,
        "semantic",
        f"user prefers dark roast {uuid.uuid4().hex}",
        json.dumps({}),
    )
    ids["superseded"] = await conn.fetchval(
        _SEED_SQL, "semantic", f"old fact {uuid.uuid4().hex}", json.dumps({})
    )
    ids["worldview"] = await conn.fetchval(
        _SEED_SQL,
        "worldview",
        f"honesty matters {uuid.uuid4().hex}",
        json.dumps({"category": "value", "confidence": 0.9, "stability": 0.95}),
    )
    ids["goal"] = await conn.fetchval(
        _SEED_SQL,
        "goal",
        f"learn the user's research interests {uuid.uuid4().hex}",
        json.dumps(
            {"title": "learn interests", "priority": "active", "source": "curiosity"}
        ),
    )
    await conn.execute(
        "UPDATE memories SET superseded_by = $1 WHERE id = $2",
        ids["semantic"],
        ids["superseded"],
    )
    ids["episode"] = await conn.fetchval(
        "INSERT INTO episodes (started_at, summary) VALUES (CURRENT_TIMESTAMP, 'test episode') RETURNING id"
    )
    await conn.execute(
        "INSERT INTO memory_edges (src_type, src_id, rel_type, dst_type, dst_id, weight) "
        "VALUES ('memory', $1, 'IN_EPISODE', 'episode', $2, 1.0)",
        str(ids["semantic"]),
        str(ids["episode"]),
    )
    await conn.execute(
        "INSERT INTO memory_edges (src_type, src_id, rel_type, dst_type, dst_id, weight) "
        "VALUES ('memory', $1, 'SUPPORTS', 'memory', $2, 0.8)",
        str(ids["semantic"]),
        str(ids["worldview"]),
    )
    await conn.fetchval(
        "SELECT * FROM cypher('memory_graph', $$"
        "CREATE (n:LifeChapterNode {key: 'hmx-test', name: 'HMX test chapter', "
        "started_at: '2026-07-09T00:00:00Z'}) RETURN n"
        "$$) AS (n agtype)"
    )
    ids["raw_unit"] = await conn.fetchval(
        "INSERT INTO subconscious_units "
        "(content, user_text, assistant_text, source_identity, idempotency_key) "
        "VALUES ('raw exchange', 'remember this', 'noted', 'hmx:test', $1) RETURNING id",
        f"hmx-{uuid.uuid4().hex}",
    )
    await conn.execute(
        "INSERT INTO memory_source_units (memory_id, subconscious_unit_id) VALUES ($1, $2)",
        ids["semantic"],
        ids["raw_unit"],
    )
    return ids


class TestPortExport:
    async def test_current_port_export_proves_lineage_and_protected_continuity(
        self, db_pool
    ):
        async with db_pool.acquire() as conn:
            await _prepare(conn)
            source = await export_hmx(conn, intent="port")

            verification = await verify_mind_continuity(conn, source)

        assert verification["verified"] is True
        assert verification["lineage"]["matches"] is True
        assert all(
            check["matches"]
            for check in verification["protected_sections"].values()
        )

    async def test_port_export_wire_contract(self, db_pool):
        async with db_pool.acquire() as conn:
            await _prepare(conn)
            tr = conn.transaction()
            await tr.start()
            try:
                ids = await _seed_world(conn)
                env = await export_hmx(conn, intent="port")
                export_id = env["export_id"]

                # Sections present per port policy
                for section in (
                    "memories",
                    "episodes",
                    "relationships",
                    "worldview",
                    "goals",
                    "drives",
                    "narrative",
                    "identity",
                    "in_flight_work",
                    "audit_records",
                ):
                    assert section in env["sections"], section

                # Memories: scoped refs, content hash, provenance, no embeddings,
                # and no worldview/goal rows (they ride in dedicated sections)
                memories = env["sections"]["memories"]
                blob = json.dumps(memories)
                assert "embedding" not in blob
                semantic = next(
                    m for m in memories if m["ref"] == f"{export_id}:{ids['semantic']}"
                )
                assert semantic["content_hash_v1"]
                assert semantic["provenance"]["acquisition_mode"] == "experienced"
                assert semantic["provenance"]["origin_id"] == str(ids["semantic"])
                assert all(m["type"] not in ("worldview", "goal") for m in memories)
                assert all("superseded_by" not in m for m in memories)

                # Supersession normalized to a SUPERSEDES edge
                edges = env["sections"]["relationships"]
                supersedes = [e for e in edges if e["edge_type"] == "SUPERSEDES"]
                assert any(
                    e["source_ref"] == f"{export_id}:{ids['superseded']}"
                    and e["target_ref"] == f"{export_id}:{ids['semantic']}"
                    for e in supersedes
                )

                # Episode membership resolved through the edge substrate
                episode = next(
                    e
                    for e in env["sections"]["episodes"]
                    if e["ref"] == f"{export_id}:{ids['episode']}"
                )
                assert f"{export_id}:{ids['semantic']}" in episode["memory_refs"]

                # Worldview: dedicated section with evidence refs
                belief = next(
                    w
                    for w in env["sections"]["worldview"]
                    if w["ref"] == f"{export_id}:{ids['worldview']}"
                )
                assert belief["category"] == "value"
                assert f"{export_id}:{ids['semantic']}" in belief["supporting_refs"]

                # AGE-only narrative nodes get an export-scoped ref and wire
                # field normalization even when their live property is `name`.
                chapter = next(
                    c
                    for c in env["sections"]["narrative"]["life_chapters"]
                    if c.get("key") == "hmx-test"
                )
                assert chapter["ref"].startswith(f"{export_id}:")
                assert chapter["title"] == "HMX test chapter"

                # Goals: metadata-derived structure
                goal = next(
                    g
                    for g in env["sections"]["goals"]
                    if g["ref"] == f"{export_id}:{ids['goal']}"
                )
                assert goal["title"] == "learn interests"
                assert goal["priority"] == "active"

                # Protected digests present for every protected section
                assert set(env["section_digests"]) == {
                    "identity",
                    "worldview",
                    "drives",
                    "emotional_triggers",
                    "narrative",
                    "goals",
                }

                # Statistics reflect the sections
                assert env["statistics"]["total_memories"] == len(memories)
                assert env["statistics"]["estimated_embedding_items"] > 0
                assert env["statistics"]["estimated_uncompressed_bytes"] > 0
            finally:
                await tr.rollback()

    async def test_digests_stable_across_exports(self, db_pool):
        """Two exports get different export_ids, so every ref differs — the
        protected digests must not (Phase 0 fast path depends on it)."""
        async with db_pool.acquire() as conn:
            await _prepare(conn)
            tr = conn.transaction()
            await tr.start()
            try:
                await _seed_world(conn)
                env1 = await export_hmx(conn, intent="port")
                env2 = await export_hmx(conn, intent="duplicate")
                assert env1["export_id"] != env2["export_id"]
                assert env1["section_digests"] == env2["section_digests"]
            finally:
                await tr.rollback()


class TestIntentPolicyOnExport:
    async def test_telepathy_export_carries_no_protected_sections(self, db_pool):
        async with db_pool.acquire() as conn:
            await _prepare(conn)
            tr = conn.transaction()
            await tr.start()
            try:
                await _seed_world(conn)
                env = await export_hmx(conn, intent="telepathy")
                assert set(env["sections"]) == {
                    "memories",
                    "episodes",
                    "relationships",
                    "clusters",
                }
                assert "section_digests" not in env
                assert env["export_scope"]["include_protected"] == []
            finally:
                await tr.rollback()

    async def test_optional_sections_are_emitted_only_on_explicit_opt_in(self, db_pool):
        async with db_pool.acquire() as conn:
            await _prepare(conn)
            tr = conn.transaction()
            await tr.start()
            try:
                ids = await _seed_world(conn)
                await conn.execute(
                    "INSERT INTO config (key, value) VALUES "
                    "('hmx.public_setting', '\"visible\"'::jsonb), "
                    "('hmx.api_token', '\"must-not-export\"'::jsonb), "
                    "('hmx.trust_anchor', '\"also-must-not-export\"'::jsonb) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                )

                default = await export_hmx(conn, intent="port")
                assert "raw_units" not in default["sections"]
                assert "config" not in default["sections"]

                env = await export_hmx(
                    conn,
                    intent="port",
                    include_raw_units=True,
                    include_config=True,
                )
                raw = next(
                    unit
                    for unit in env["sections"]["raw_units"]
                    if unit["ref"] == f"{env['export_id']}:{ids['raw_unit']}"
                )
                assert raw["derived_memory_refs"] == [
                    f"{env['export_id']}:{ids['semantic']}"
                ]
                assert "embedding" not in raw
                assert env["statistics"]["total_raw_units"] >= 1
                assert env["sections"]["config"]["hmx.public_setting"] == "visible"
                assert "hmx.api_token" not in env["sections"]["config"]
                assert "hmx.trust_anchor" not in env["sections"]["config"]
            finally:
                await tr.rollback()

    async def test_telepathy_opt_in_carries_warning(self, db_pool):
        async with db_pool.acquire() as conn:
            await _prepare(conn)
            tr = conn.transaction()
            await tr.start()
            try:
                await _seed_world(conn)
                env = await export_hmx(
                    conn, intent="telepathy", include_protected=["worldview"]
                )
                assert "worldview" in env["sections"]
                assert "identity" not in env["sections"]
                assert any("deliberative" in w for w in env["export_warnings"])
            finally:
                await tr.rollback()


class TestJsonlStreaming:
    async def test_jsonl_round_trip_shape(self, db_pool):
        async with db_pool.acquire() as conn:
            await _prepare(conn)
            tr = conn.transaction()
            await tr.start()
            try:
                await _seed_world(conn)
                env = await export_hmx(conn, intent="port")
                lines = [json.loads(line) for line in iter_hmx_jsonl(env)]
                assert lines[0]["record_type"] == "envelope"
                assert lines[0]["data"]["hmx_version"] == "1.7"
                assert lines[-1]["record_type"] == "footer"
                assert (
                    lines[-1]["statistics"]["total_memories"]
                    == env["statistics"]["total_memories"]
                )
                types = {line["record_type"] for line in lines}
                assert {
                    "memory",
                    "episode",
                    "relationship",
                    "worldview",
                    "goal",
                    "drive",
                } <= types
            finally:
                await tr.rollback()
