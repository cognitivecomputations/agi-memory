from __future__ import annotations

import json
import os

import pytest

from services.ingest import Config, IngestionMode, IngestionPipeline, _hash_text
from tests.utils import _db_dsn, get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


class _StubLLM:
    def __init__(self, marker: str):
        self.marker = marker
        self.call_count = 0

    async def complete_json(self, messages, temperature=0.2):
        self.call_count += 1
        text = str(messages[-1].get("content", ""))
        if "key 'items'" in text:
            return {
                "items": [
                    {
                        "content": f"The source-document handle survived for {self.marker}.",
                        "confidence": 0.9,
                        "importance": 0.6,
                        "category": "fact",
                    }
                ]
            }
        return {
            "valence": 0.0,
            "arousal": 0.2,
            "primary_emotion": "neutral",
            "intensity": 0.1,
            "summary": "The document is technical reference material.",
        }


async def _stub_get_embedding(conn):
    await conn.execute(
        """
        CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
        RETURNS vector[] AS $$
            SELECT COALESCE(
                array_agg((
                    array_fill(0.01::float, ARRAY[2 + abs(hashtext(t)) % (embedding_dimension() - 2)]) ||
                    ARRAY[1.0::float] ||
                    array_fill(0.01::float, ARRAY[embedding_dimension() - 3 - abs(hashtext(t)) % (embedding_dimension() - 2)])
                )::vector),
                ARRAY[]::vector[]
            )
            FROM unnest(text_contents) t
        $$ LANGUAGE sql;
        """
    )


async def test_ingest_text_stores_raw_source_before_receipt_skip(db_pool):
    marker = get_test_identifier("ingestraw")
    content = f"# Backfill Raw {marker}\n\nThe archived raw source survives even when extraction is already receipted."
    content_hash = _hash_text(content)
    source_path = f"text:{content_hash[:12]}"

    async with db_pool.acquire() as conn:
        await conn.execute(
            "SELECT record_ingestion_receipt($1, $1, NULL, 0, $2)",
            content_hash,
            source_path,
        )

    config = Config(
        dsn=_db_dsn(os.environ.get("POSTGRES_DB")),
        llm_config={"provider": "openai", "model": "stub", "api_key": "stub"},
        verbose=False,
    )
    pipeline = IngestionPipeline(config)
    try:
        count = await pipeline.ingest_text(content, title=f"Backfill Raw {marker}")
    finally:
        await pipeline.close()

    try:
        assert count == 0
        async with db_pool.acquire() as conn:
            stored = await conn.fetchrow(
                "SELECT title, content, path FROM source_documents WHERE content_hash = $1",
                content_hash,
            )
        assert stored is not None
        assert stored["title"] == f"Backfill Raw {marker}"
        assert stored["content"] == content
        assert stored["path"] == source_path
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM subconscious_units WHERE source_attribution->>'content_hash' = $1",
                content_hash,
            )
            await conn.execute("DELETE FROM source_documents WHERE content_hash = $1", content_hash)
            await conn.execute("DELETE FROM ingestion_receipts WHERE doc_ref = $1", content_hash)


async def test_single_source_ingest_auto_loads_source_to_recmem_desk(db_pool):
    marker = get_test_identifier("ingestdesk")
    content = f"# Desk Incoming {marker}\n\nA single ingested source should land on the RecMem desk first."
    content_hash = _hash_text(content)
    source_path = f"text:{content_hash[:12]}"

    async with db_pool.acquire() as conn:
        await conn.execute(
            "SELECT record_ingestion_receipt($1, $1, NULL, 0, $2)",
            content_hash,
            source_path,
        )

    config = Config(
        dsn=_db_dsn(os.environ.get("POSTGRES_DB")),
        llm_config={"provider": "openai", "model": "stub", "api_key": "stub"},
        verbose=False,
    )
    pipeline = IngestionPipeline(config)
    try:
        count = await pipeline.ingest_text(content, title=f"Desk Incoming {marker}")
        details = dict(pipeline.last_result)
    finally:
        await pipeline.close()

    try:
        assert count == 0
        assert details["source_document_id"]
        assert details["source_document_ids"] == [details["source_document_id"]]
        assert details["content_hash"] == content_hash
        assert details["desk_load"]["count"] >= 1
        assert details["desk_load"]["desk_unit_ids"]

        async with db_pool.acquire() as conn:
            desk_count = await conn.fetchval(
                """
                SELECT count(*)
                FROM subconscious_units
                WHERE status = 'active'
                  AND metadata#>>'{recmem,kind}' = 'source_document_desk'
                  AND source_attribution->>'source_document_id' = $1
                  AND source_attribution->>'content_hash' = $2
                """,
                details["source_document_id"],
                content_hash,
            )
        assert desk_count >= 1
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM subconscious_units WHERE source_attribution->>'content_hash' = $1",
                content_hash,
            )
            await conn.execute("DELETE FROM source_documents WHERE content_hash = $1", content_hash)
            await conn.execute("DELETE FROM ingestion_receipts WHERE doc_ref = $1", content_hash)


async def test_connector_acquisition_skips_auto_recmem_desk_load(db_pool):
    marker = get_test_identifier("ingestconn")
    content = f"# Connector Backfill {marker}\n\nHistorical connector imports stay in the cabinet."
    content_hash = _hash_text(content)
    source_path = f"text:{content_hash[:12]}"

    async with db_pool.acquire() as conn:
        await conn.execute(
            "SELECT record_ingestion_receipt($1, $1, NULL, 0, $2)",
            content_hash,
            source_path,
        )

    config = Config(
        dsn=_db_dsn(os.environ.get("POSTGRES_DB")),
        llm_config={"provider": "openai", "model": "stub", "api_key": "stub"},
        acquisition="connector",
        verbose=False,
    )
    pipeline = IngestionPipeline(config)
    try:
        count = await pipeline.ingest_text(content, title=f"Connector Backfill {marker}")
        details = dict(pipeline.last_result)
    finally:
        await pipeline.close()

    try:
        assert count == 0
        assert details["source_document_id"]
        assert details["desk_load"] == {"skipped": True, "reason": "connector_backfill"}
        async with db_pool.acquire() as conn:
            desk_count = await conn.fetchval(
                """
                SELECT count(*)
                FROM subconscious_units
                WHERE metadata#>>'{recmem,kind}' = 'source_document_desk'
                  AND source_attribution->>'content_hash' = $1
                """,
                content_hash,
            )
        assert desk_count == 0
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM subconscious_units WHERE source_attribution->>'content_hash' = $1",
                content_hash,
            )
            await conn.execute("DELETE FROM source_documents WHERE content_hash = $1", content_hash)
            await conn.execute("DELETE FROM ingestion_receipts WHERE doc_ref = $1", content_hash)


async def test_directory_bulk_ingest_skips_auto_recmem_desk_load(db_pool, tmp_path):
    marker = get_test_identifier("ingestbulk")
    files = []
    for idx in range(2):
        content = f"# Bulk File {idx} {marker}\n\nA corpus import should not flood the RecMem desk."
        path = tmp_path / f"bulk-{idx}-{marker}.md"
        path.write_text(content, encoding="utf-8")
        content_hash = _hash_text(content)
        files.append((path, content_hash))

    async with db_pool.acquire() as conn:
        for path, content_hash in files:
            await conn.execute(
                "SELECT record_ingestion_receipt($1, $1, NULL, 0, $2)",
                content_hash,
                str(path),
            )

    config = Config(
        dsn=_db_dsn(os.environ.get("POSTGRES_DB")),
        llm_config={"provider": "openai", "model": "stub", "api_key": "stub"},
        verbose=False,
    )
    pipeline = IngestionPipeline(config)
    try:
        count = await pipeline.ingest_directory(tmp_path, recursive=False)
        details = dict(pipeline.last_result)
    finally:
        await pipeline.close()

    content_hashes = [content_hash for _, content_hash in files]
    try:
        assert count == 0
        assert details["bulk_ingest"] is True
        assert details["files_processed"] == len(files)
        assert details["desk_load"] == {"skipped": True, "reason": "bulk_directory_ingest"}
        async with db_pool.acquire() as conn:
            desk_count = await conn.fetchval(
                """
                SELECT count(*)
                FROM subconscious_units
                WHERE metadata#>>'{recmem,kind}' = 'source_document_desk'
                  AND source_attribution->>'content_hash' = ANY($1::text[])
                """,
                content_hashes,
            )
        assert desk_count == 0
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM subconscious_units WHERE source_attribution->>'content_hash' = ANY($1::text[])",
                content_hashes,
            )
            await conn.execute(
                "DELETE FROM source_documents WHERE content_hash = ANY($1::text[])",
                content_hashes,
            )
            await conn.execute("DELETE FROM ingestion_receipts WHERE doc_ref = ANY($1::text[])", content_hashes)


async def test_ingest_created_memories_carry_source_document_id(db_pool):
    marker = get_test_identifier("ingestdocid")
    content = f"# Source Handle {marker}\n\nThis file creates a memory with a source-document handle."
    content_hash = _hash_text(content)
    fact = f"The source-document handle survived for {marker}."

    async with db_pool.acquire() as conn:
        await _stub_get_embedding(conn)

    config = Config(
        dsn=_db_dsn(os.environ.get("POSTGRES_DB")),
        llm_config={"provider": "openai", "model": "stub", "api_key": "stub"},
        mode=IngestionMode.FAST,
        verbose=False,
    )
    pipeline = IngestionPipeline(config)
    pipeline.llm = _StubLLM(marker)
    pipeline.appraiser.llm = pipeline.llm
    pipeline.extractor.llm = pipeline.llm
    try:
        count = await pipeline.ingest_text(content, title=f"Source Handle {marker}")
    finally:
        await pipeline.close()

    try:
        assert count == 1
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT m.source_attribution, d.id::text AS document_id
                FROM memories m
                JOIN source_documents d ON d.content_hash = $2
                WHERE m.content = $1
                """,
                fact,
                content_hash,
            )
        assert row is not None
        source = row["source_attribution"]
        if isinstance(source, str):
            source = json.loads(source)
        assert source["content_hash"] == content_hash
        assert source["source_document_id"] == row["document_id"]
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM subconscious_units WHERE source_attribution->>'content_hash' = $1",
                content_hash,
            )
            await conn.execute("DELETE FROM memories WHERE content = $1", fact)
            await conn.execute("DELETE FROM source_documents WHERE content_hash = $1", content_hash)
            await conn.execute("DELETE FROM ingestion_receipts WHERE doc_ref = $1", content_hash)


class _AppraisalStubLLM(_StubLLM):
    """Like _StubLLM, but the appraisal branch returns a distinctive,
    non-neutral valence so a test can tell "the appraisal landed here" apart
    from "nothing happened" or "the default neutral value happened to match"."""

    def __init__(self, marker: str, valence: float = 0.85):
        super().__init__(marker)
        self.valence = valence

    async def complete_json(self, messages, temperature=0.2):
        text = str(messages[-1].get("content", ""))
        if "key 'items'" in text:
            return await super().complete_json(messages, temperature)
        return {
            "valence": self.valence,
            "arousal": 0.7,
            "primary_emotion": "delight",
            "intensity": 0.8,
            "summary": "The document was delightful.",
        }


async def test_ingest_does_not_overwrite_live_affective_state(db_pool):
    """#86: a document's appraisal must not slam the shared, live affective
    state -- once per document, let alone once per section for small docs.
    The felt signal belongs on the encounter memory; update_mood() blends it
    into mood from there over the normal maintenance cycle, like any other
    episodic memory's affect."""
    marker = get_test_identifier("ingestaffect")
    content = (
        f"# Reading {marker}\n\nFirst section talks about something notable.\n\n"
        f"## Second\n\nA second section, so the per-section (FAST deep) path runs.\n\n"
        f"## Third\n\nA third section for good measure."
    )
    content_hash = _hash_text(content)
    sentinel = json.dumps({
        "valence": -0.9,
        "arousal": 0.9,
        "primary_emotion": "sentinel_marker_emotion",
        "intensity": 0.9,
        "dominance": 0.5,
    })

    async with db_pool.acquire() as conn:
        await _stub_get_embedding(conn)
        original_state = await conn.fetchval(
            "SELECT affective_state FROM heartbeat_state WHERE id = 1"
        )
        await conn.execute(
            "UPDATE heartbeat_state SET affective_state = affective_state || $1::jsonb WHERE id = 1",
            sentinel,
        )

    config = Config(
        dsn=_db_dsn(os.environ.get("POSTGRES_DB")),
        llm_config={"provider": "openai", "model": "stub", "api_key": "stub"},
        mode=IngestionMode.FAST,
        deep_max_words=2000,  # well above this doc's word count -> per-section path
        verbose=False,
    )
    pipeline = IngestionPipeline(config)
    pipeline.llm = _AppraisalStubLLM(marker)
    pipeline.appraiser.llm = pipeline.llm
    pipeline.extractor.llm = pipeline.llm
    try:
        try:
            await pipeline.ingest_text(content, title=f"Reading {marker}")
        finally:
            await pipeline.close()

        async with db_pool.acquire() as conn:
            live_state = await conn.fetchval(
                "SELECT get_current_affective_state()"
            )
            live_state = json.loads(live_state) if isinstance(live_state, str) else live_state
            assert live_state["primary_emotion"] == "sentinel_marker_emotion"
            assert live_state["valence"] == -0.9

            encounter = await conn.fetchrow(
                "SELECT metadata->>'emotional_valence' AS emotional_valence FROM memories "
                "WHERE type = 'episodic' AND content LIKE $1",
                f"I read 'Reading {marker}'.%",
            )
        assert encounter is not None
        # The felt signal from ingestion landed on the encounter memory, not
        # on the live state the sentinel proved untouched above.
        assert abs(float(encounter["emotional_valence"]) - 0.85) < 1e-9
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE heartbeat_state SET affective_state = $1::jsonb WHERE id = 1",
                json.dumps(original_state) if not isinstance(original_state, str) else original_state,
            )
            await conn.execute(
                "DELETE FROM subconscious_units WHERE source_attribution->>'content_hash' = $1",
                content_hash,
            )
            await conn.execute(
                "DELETE FROM memories WHERE content LIKE $1 OR source_attribution->>'content_hash' = $2",
                f"I read 'Reading {marker}'.%",
                content_hash,
            )
            await conn.execute("DELETE FROM source_documents WHERE content_hash = $1", content_hash)
            await conn.execute("DELETE FROM ingestion_receipts WHERE doc_ref = $1", content_hash)


@pytest.mark.parametrize(
    ("mode", "runner_name"),
    [
        (IngestionMode.SLOW, "run_slow_ingest"),
        (IngestionMode.HYBRID, "run_hybrid_ingest"),
    ],
)
async def test_rlm_ingest_modes_receive_source_document_handle(db_pool, monkeypatch, mode, runner_name):
    marker = get_test_identifier(f"ingest{mode.value}doc")
    content = f"# RLM Source Handle {marker}\n\nThe RLM reader must receive the preserved source handle."
    content_hash = _hash_text(content)
    source_path = f"/tmp/{marker}.md"
    seen: dict[str, str] = {}

    async def fake_runner(*, pipeline, doc, sections, llm_config, dsn, workspace_budgets=None):
        seen["document_id"] = str(doc.document_id)
        seen["source_document_id"] = str(pipeline._source_payload(doc).get("source_document_id"))
        seen["section_count"] = str(len(sections))
        return {"memories_created": 0}

    monkeypatch.setattr(f"services.slow_ingest_rlm.{runner_name}", fake_runner)

    config = Config(
        dsn=_db_dsn(os.environ.get("POSTGRES_DB")),
        llm_config={"provider": "openai", "model": "stub", "api_key": "stub"},
        mode=mode,
        verbose=False,
    )
    pipeline = IngestionPipeline(config)
    try:
        count = await pipeline.ingest_text(
            content,
            title=f"RLM Source Handle {marker}",
            source_type="document",
            path=source_path,
            file_type=".md",
        )
    finally:
        await pipeline.close()

    try:
        assert count == 0
        assert seen["document_id"]
        assert seen["source_document_id"] == seen["document_id"]
        assert int(seen["section_count"]) >= 1
        async with db_pool.acquire() as conn:
            stored = await conn.fetchrow(
                "SELECT id::text AS id, path, content FROM source_documents WHERE content_hash = $1",
                content_hash,
            )
        assert stored is not None
        assert stored["id"] == seen["document_id"]
        assert stored["path"] == source_path
        assert stored["content"] == content
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM subconscious_units WHERE source_attribution->>'content_hash' = $1",
                content_hash,
            )
            await conn.execute("DELETE FROM source_documents WHERE content_hash = $1", content_hash)
            await conn.execute("DELETE FROM ingestion_receipts WHERE doc_ref = $1", content_hash)
