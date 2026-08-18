"""Conscious-extraction worker step (#37): claim → chat_json → apply/fail,
gated by extraction.enabled; extracted facts become recallable memories,
empty extractions create nothing, LLM failures retry.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from services.extraction import run_conscious_extraction_step

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _coerce_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


async def _stub_get_embedding(conn):
    # Constant vectors (as in test_ingest_routing): every text is maximally
    # similar, so recall visibility is decided by status/filters/ranking —
    # what these tests actually exercise — not by hash-vector luck. One-hot
    # hash vectors made query-vs-content similarity 0, which the unified
    # ranker's min-score floor correctly rejects.
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


async def _seed_unit(conn, user_text: str, seq: int) -> str:
    result = _coerce_json(
        await conn.fetchval(
            "SELECT record_chat_turn_memory($1, 'Understood.', NULL, $2, $3::jsonb)",
            user_text,
            f"chat:extraction-test:{seq}",
            json.dumps({"importance": 0.85}),
        )
    )
    return result["raw_unit_id"]


_LLM_CONFIG = {"provider": "openai", "model": "test", "api_key": "k"}


async def test_kill_switch_disables_extraction(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "UPDATE config SET value = 'false'::jsonb WHERE key = 'extraction.enabled'"
            )
            result = await run_conscious_extraction_step(conn)
            assert result == {"skipped": True, "reason": "disabled"}
        finally:
            await tr.rollback()


async def test_extracted_fact_becomes_recallable_memory(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await conn.execute(
                "UPDATE config SET value = 'true'::jsonb WHERE key = 'extraction.enabled'"
            )
            unit_id = await _seed_unit(conn, "I am the inventor of Hexis.", 1)

            canned = {
                "facts": [{
                    "unit_id": unit_id,
                    "content": "Eric is the inventor of Hexis.",
                    "kind": "user_testimony",
                    "category": "identity",
                    "confidence": 0.7,
                }]
            }
            with patch("services.extraction.load_llm_config", new=AsyncMock(return_value=_LLM_CONFIG)), \
                 patch("services.extraction.chat_json", new=AsyncMock(return_value=(canned, "{}"))):
                result = await run_conscious_extraction_step(conn)

            assert result["created"] == 1
            # Embed the new memory before recalling it (async, 0129).
            from tests.utils import embed_pending_memories
            await embed_pending_memories(conn)
            # The fact is retrievable as knowledge in a "fresh session"
            # (recall knows nothing about the conversation that produced it).
            recalled = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('recall', $1::jsonb)",
                    json.dumps({"query": "who invented Hexis",
                                "source_kind": "user_testimony", "limit": 10}),
                )
            )
            contents = [m["content"] for m in recalled["output"]["memories"]]
            assert "Eric is the inventor of Hexis." in contents
        finally:
            await tr.rollback()


async def test_empty_extraction_creates_no_memories(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await conn.execute(
                "UPDATE config SET value = 'true'::jsonb WHERE key = 'extraction.enabled'"
            )
            unit_id = await _seed_unit(conn, "Please remember this is very important stuff.", 2)
            before = await conn.fetchval("SELECT count(*) FROM memories")

            with patch("services.extraction.load_llm_config", new=AsyncMock(return_value=_LLM_CONFIG)), \
                 patch("services.extraction.chat_json", new=AsyncMock(return_value=({"facts": []}, "{}"))):
                result = await run_conscious_extraction_step(conn)

            assert result["created"] == 0
            after = await conn.fetchval("SELECT count(*) FROM memories")
            # Only the direct episodic promotion from the turn itself exists;
            # extraction added nothing (selectivity via the empty list).
            assert after == before
            status = await conn.fetchval(
                "SELECT extraction_status FROM subconscious_units WHERE id = $1::uuid",
                unit_id,
            )
            assert status == "extracted"
        finally:
            await tr.rollback()


async def test_llm_failure_marks_units_for_retry(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await conn.execute(
                "UPDATE config SET value = 'true'::jsonb WHERE key = 'extraction.enabled'"
            )
            unit_id = await _seed_unit(conn, "Another statement of identity importance.", 3)

            with patch("services.extraction.load_llm_config", new=AsyncMock(return_value=_LLM_CONFIG)), \
                 patch("services.extraction.chat_json", new=AsyncMock(side_effect=RuntimeError("llm down"))):
                result = await run_conscious_extraction_step(conn)

            assert result["failed_units"] == 1
            row = await conn.fetchrow(
                "SELECT extraction_status, extraction_attempts, extraction_error "
                "FROM subconscious_units WHERE id = $1::uuid",
                unit_id,
            )
            assert row["extraction_status"] == "pending"  # retryable
            assert row["extraction_attempts"] == 1
            assert "llm down" in row["extraction_error"]
        finally:
            await tr.rollback()
