import json
from uuid import uuid4

import pytest

from tests.utils import _coerce_json, embed_pending_memories, get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


async def test_get_goals_by_priority_filters(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "SELECT set_config('heartbeat.max_active_goals', '10'::jsonb)"
            )

            active_id = await conn.fetchval(
                "SELECT create_goal($1, $2, $3, $4, NULL, NULL)",
                f"Active {get_test_identifier('goal')}",
                "active goal",
                "curiosity",
                "active",
            )
            queued_id = await conn.fetchval(
                "SELECT create_goal($1, $2, $3, $4, NULL, NULL)",
                f"Queued {get_test_identifier('goal')}",
                "queued goal",
                "curiosity",
                "queued",
            )

            rows = await conn.fetch("SELECT * FROM get_goals_by_priority()")
            priorities = {row["priority"] for row in rows}
            assert "active" in priorities
            assert "queued" in priorities

            queued_rows = await conn.fetch(
                "SELECT * FROM get_goals_by_priority('queued')"
            )
            assert queued_rows
            assert all(row["priority"] == "queued" for row in queued_rows)
            assert all(row["origin"] == "derived" for row in queued_rows)
            assert queued_id in {row["id"] for row in queued_rows}

            assert active_id != queued_id
        finally:
            await tr.rollback()


async def test_get_worldview_snapshot_filters_by_confidence(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            high_id = await conn.fetchval(
                "SELECT create_worldview_memory($1, $2, $3, $4, $5, $6)",
                f"High confidence {get_test_identifier('worldview')}",
                "belief",
                0.9,
                0.8,
                0.9,
                "test",
            )
            low_id = await conn.fetchval(
                "SELECT create_worldview_memory($1, $2, $3, $4, $5, $6)",
                f"Low confidence {get_test_identifier('worldview')}",
                "belief",
                0.2,
                0.5,
                0.4,
                "test",
            )
            assert high_id is not None
            assert low_id is not None

            rows = await conn.fetch("SELECT * FROM get_worldview_snapshot(10, 0.5)")
            contents = {row["content"] for row in rows}
            assert any("High confidence" in content for content in contents)
            assert all("Low confidence" not in content for content in contents)
        finally:
            await tr.rollback()


async def test_get_emotional_patterns_context_returns_entries(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            pattern = f"pattern-{get_test_identifier('emotion')}"
            mem_id = await conn.fetchval(
                """
                SELECT create_strategic_memory(
                    $1,
                    $2,
                    0.7,
                    $3::jsonb,
                    NULL,
                    0.6,
                    NULL,
                    NULL
                )
                """,
                f"Emotional pattern {pattern}",
                "emotional pattern",
                json.dumps(
                    {
                        "kind": "emotional_pattern",
                        "pattern": pattern,
                        "frequency": 3,
                        "unprocessed": True,
                    }
                ),
            )
            await conn.execute(
                """
                UPDATE memories
                SET metadata = jsonb_set(metadata, '{supporting_evidence,kind}', '\"emotional_pattern\"'::jsonb)
                WHERE id = $1::uuid
                """,
                mem_id,
            )
            kind = await conn.fetchval(
                "SELECT metadata->'supporting_evidence'->>'kind' FROM memories WHERE id = $1::uuid",
                mem_id,
            )
            assert kind == "emotional_pattern"

            result = _coerce_json(
                await conn.fetchval("SELECT get_emotional_patterns_context(5)")
            )
            assert result
            assert any(pattern in entry.get("pattern", "") for entry in result)
        finally:
            await tr.rollback()


async def test_get_subconscious_and_chat_contexts(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            query_text = f"context memory {get_test_identifier('context')}"
            await conn.fetchval(
                "SELECT create_semantic_memory($1, 0.8, ARRAY['context'], NULL, NULL, 0.6)",
                query_text,
            )
            await conn.fetchval(
                """
                SELECT create_episodic_memory(
                    $1,
                    NULL,
                    jsonb_build_object('heartbeat_id', 'hb-test'),
                    NULL,
                    0.1,
                    CURRENT_TIMESTAMP,
                    0.4
                )
                """,
                f"recent {query_text}",
            )

            # Writes are pending until the embedding pass runs (async, 0129).
            await embed_pending_memories(conn)

            subconscious = _coerce_json(
                await conn.fetchval(
                    "SELECT get_subconscious_context(5, 5, 5, 2, 2, 0, 0)"
                )
            )
            assert "recent_memories" in subconscious
            assert "emotional_state" in subconscious

            chat_ctx = _coerce_json(
                await conn.fetchval(
                    "SELECT get_chat_context($1, 5)",
                    query_text,
                )
            )
            assert "relevant_memories" in chat_ctx
            assert any(
                query_text in entry.get("content", "")
                for entry in chat_ctx["relevant_memories"]
            )

            sub_chat_ctx = _coerce_json(
                await conn.fetchval(
                    "SELECT get_subconscious_chat_context($1, 5)",
                    query_text,
                )
            )
            assert any(
                query_text in entry.get("content", "")
                for entry in sub_chat_ctx["relevant_memories"]
            )
        finally:
            await tr.rollback()


async def test_subconscious_observations_and_chat_turn_memory(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            observation = f"subconscious observation {get_test_identifier('exchange')}"
            applied = _coerce_json(
                await conn.fetchval(
                    "SELECT apply_subconscious_observations($1::jsonb)",
                    json.dumps(
                        {
                            "emotional_observations": [
                                {
                                    "pattern": observation,
                                    "importance": 0.4,
                                    "confidence": 0.7,
                                }
                            ]
                        }
                    ),
                )
            )
            assert applied["emotional"] >= 1
            obs_row = await conn.fetchrow(
                "SELECT content, type FROM memories WHERE content = $1",
                f"Emotional pattern: {observation}",
            )
            assert obs_row is not None
            assert obs_row["type"] == "strategic"

            session_id = str(uuid4())
            chat_result = _coerce_json(
                await conn.fetchval(
                    "SELECT record_chat_turn_memory($1, $2, $3, $4, $5::jsonb)",
                    "hello",
                    "hi there",
                    session_id,
                    None,
                    json.dumps(
                        {"importance": 0.4, "metadata": {"type": "conversation"}}
                    ),
                )
            )
            raw_unit_id = chat_result["raw_unit_id"]
            chat_row = await conn.fetchrow(
                "SELECT content, source_attribution, metadata FROM subconscious_units WHERE id = $1::uuid",
                raw_unit_id,
            )
            assert chat_row is not None
            assert "User: hello" in chat_row["content"]
            assert "Assistant: hi there" in chat_row["content"]
            source_attribution = _coerce_json(chat_row["source_attribution"])
            metadata = _coerce_json(chat_row["metadata"])
            assert source_attribution["kind"] == "conversation"
            assert metadata["type"] == "conversation"
            assert chat_result["direct_promoted"] is False

            promoted = _coerce_json(
                await conn.fetchval(
                    "SELECT record_chat_turn_memory($1, $2, $3, $4, $5::jsonb)",
                    "my surgery is tomorrow",
                    "I will remember that and check in carefully.",
                    session_id,
                    None,
                    json.dumps(
                        {"importance": 0.99, "metadata": {"type": "conversation"}}
                    ),
                )
            )
            memory_id = promoted["promoted_memory_id"]
            promoted_row = await conn.fetchrow(
                "SELECT content, type FROM memories WHERE id = $1::uuid",
                memory_id,
            )
            assert promoted_row is not None
            assert promoted_row["type"] == "episodic"
            assert "my surgery is tomorrow" in promoted_row["content"]
        finally:
            await tr.rollback()


async def test_get_contradictions_context_returns_pairs(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mem_a = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', $1, array_fill(0.6, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """,
                f"A {get_test_identifier('contradictions')}",
            )
            mem_b = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic', $1, array_fill(0.7, ARRAY[embedding_dimension()])::vector)
                RETURNING id
                """,
                f"B {get_test_identifier('contradictions')}",
            )

            await conn.fetchval("SELECT sync_memory_node($1::uuid)", mem_a)
            await conn.fetchval("SELECT sync_memory_node($1::uuid)", mem_b)
            await conn.execute(
                "SELECT create_memory_relationship($1::uuid, $2::uuid, 'CONTRADICTS', '{}'::jsonb)",
                mem_a,
                mem_b,
            )

            contradictions = _coerce_json(
                await conn.fetchval("SELECT get_contradictions_context(5)")
            )
            assert contradictions
            contents = {entry["content_a"] for entry in contradictions} | {
                entry["content_b"] for entry in contradictions
            }
            assert any("A" in content for content in contents)
            assert any("B" in content for content in contents)
        finally:
            await tr.rollback()
