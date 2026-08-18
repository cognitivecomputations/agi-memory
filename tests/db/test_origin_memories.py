"""Origin memories (#40): the origin documents must become protected,
source-attributed, recallable semantic memories — idempotently seeded,
exempt from retention fade, and questioned (never rewritten) by
contradicting evidence.
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


async def _enable_and_seed(conn) -> dict:
    await conn.execute(
        "UPDATE config SET value = 'true'::jsonb WHERE key = 'origin_memories.enabled'"
    )
    return _coerce_json(await conn.fetchval("SELECT seed_origin_memories()"))


async def test_seeding_is_idempotent_and_on_by_default(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            # On by default: no config change needed to seed.
            first = _coerce_json(await conn.fetchval("SELECT seed_origin_memories()"))
            assert first["enabled"] is True
            assert first["seeded"] == 15
            second = _coerce_json(await conn.fetchval("SELECT seed_origin_memories()"))
            assert second["seeded"] == 0
            assert second["skipped"] == 15

            # The flag is a kill switch.
            await conn.execute(
                "UPDATE config SET value = 'false'::jsonb WHERE key = 'origin_memories.enabled'"
            )
            off = _coerce_json(await conn.fetchval("SELECT seed_origin_memories()"))
            assert off == {"enabled": False, "seeded": 0, "skipped": 0}
        finally:
            await tr.rollback()


async def test_seeded_memories_are_protected_with_provenance(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await _enable_and_seed(conn)
            rows = await conn.fetch(
                """
                SELECT id, trust_level, metadata, source_attribution
                FROM memories WHERE metadata ? 'origin_claim_key'
                """
            )
            assert len(rows) == 15
            for row in rows:
                meta = _coerce_json(row["metadata"])
                attribution = _coerce_json(row["source_attribution"])
                assert meta["protected"] is True
                assert float(meta["confidence"]) == 0.9
                assert row["trust_level"] == 0.9
                assert attribution["kind"] == "origin_document"
                assert attribution["ref"].startswith("services/prompts/")
                assert attribution["content_hash"]
                protected = await conn.fetchval(
                    "SELECT is_memory_protected($1::uuid)", row["id"]
                )
                assert protected is True
        finally:
            await tr.rollback()


async def test_origin_memories_never_fade_even_when_stale(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await _enable_and_seed(conn)
            # Age everything well past the stale/idle thresholds.
            await conn.execute(
                """
                UPDATE memories
                SET created_at = now() - interval '400 days',
                    last_accessed = now() - interval '400 days',
                    last_reinforced = now() - interval '400 days',
                    source_attribution = jsonb_set(
                        source_attribution, '{observed_at}',
                        to_jsonb(now() - interval '400 days'))
                WHERE metadata ? 'origin_claim_key'
                """
            )
            # Control: an equally stale unprotected document IS found.
            await conn.execute(
                """
                INSERT INTO memories (type, content, embedding, importance, trust_level,
                                      status, created_at, last_accessed, last_reinforced,
                                      source_attribution)
                VALUES ('semantic', 'stale control fact',
                        array_fill(0.1, ARRAY[embedding_dimension()])::vector,
                        0.3, 0.5, 'active',
                        now() - interval '400 days', now() - interval '400 days',
                        now() - interval '400 days',
                        jsonb_build_object('kind', 'document', 'label', 'control-doc',
                                           'content_hash', 'control-hash-123',
                                           'observed_at', now() - interval '400 days'))
                """
            )
            stale = await conn.fetch("SELECT * FROM find_stale_ingested_documents()")
            hashes = {row["content_hash"] for row in stale}
            assert "control-hash-123" in hashes
            origin_hashes = {
                row[0]
                for row in await conn.fetch(
                    """
                    SELECT DISTINCT source_attribution->>'content_hash'
                    FROM memories WHERE metadata ? 'origin_claim_key'
                    """
                )
            }
            assert not (origin_hashes & hashes)
        finally:
            await tr.rollback()


async def test_contradicting_evidence_flags_but_never_rewrites(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await _enable_and_seed(conn)
            mid = await conn.fetchval(
                """
                SELECT id FROM memories
                WHERE metadata ? 'origin_claim_key' AND content LIKE 'Eric Hartford is my creator%'
                """
            )
            assert mid is not None
            r = _coerce_json(
                await conn.fetchval(
                    "SELECT add_memory_evidence($1::uuid, 'contradicts', $2::jsonb, $3::text)",
                    mid,
                    json.dumps({"kind": "web_page", "ref": "https://example.com/denial", "trust": 0.8}),
                    "A web page claims someone else built Hexis.",
                )
            )
            assert r["applied"] is False
            assert r["reason"] == "protected"
            row = await conn.fetchrow(
                "SELECT trust_level, metadata FROM memories WHERE id = $1::uuid", mid
            )
            meta = _coerce_json(row["metadata"])
            assert float(meta["confidence"]) == 0.9
            assert row["trust_level"] == 0.9
            # The contradiction stays visible for review.
            assert len(meta["contradicting_sources"]) == 1
            edge = await conn.fetchval(
                "SELECT count(*) FROM memory_edges WHERE dst_id = $1 AND rel_type = 'CONTRADICTS'",
                str(mid),
            )
            assert edge == 1
        finally:
            await tr.rollback()


async def test_origin_facts_are_recallable_with_source(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await _enable_and_seed(conn)
            # Writes are pending until the embedding pass runs (async, 0129).
            from tests.utils import embed_pending_memories
            await embed_pending_memories(conn)
            recalled = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('recall', $1::jsonb)",
                    json.dumps({
                        "query": "who created me inventor of Hexis",
                        "source_kind": "origin_document",
                        "limit": 20,
                    }),
                )
            )
            assert recalled["success"] is True
            memories = recalled["output"]["memories"]
            assert memories, "origin memories must be recallable"
            creator = next(
                (m for m in memories if "Eric Hartford is my creator" in m["content"]),
                None,
            )
            assert creator is not None
            assert creator["source_kind"] == "origin_document"
            assert creator["source_ref"] == "services/prompts/LetterFromClaude.md"
            assert creator["confidence"] == 0.9
            assert creator["trust"] == 0.9
        finally:
            await tr.rollback()
