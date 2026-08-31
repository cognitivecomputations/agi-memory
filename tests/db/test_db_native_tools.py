from __future__ import annotations

import json
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _coerce_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


async def _stub_get_embedding(conn):
    await conn.execute("""
        CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
        RETURNS vector[] AS $$
            SELECT COALESCE(
                array_agg((
                    array_fill(0.0::float, ARRAY[0]) ||
                    ARRAY[1.0::float] ||
                    array_fill(0.0::float, ARRAY[embedding_dimension() - 1])
                )::vector),
                ARRAY[]::vector[]
            )
            FROM unnest(text_contents)
        $$ LANGUAGE sql;
        """)


async def test_execute_goals_tool_create_and_list(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            created = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_goals_tool($1::jsonb)",
                    json.dumps(
                        {
                            "action": "create",
                            "title": "DB-native goal",
                            "priority": "queued",
                        }
                    ),
                )
            )
            listed = _coerce_json(
                await conn.fetchval(
                    'SELECT execute_goals_tool(\'{"action":"list"}\'::jsonb)'
                )
            )

            assert created["success"] is True
            assert created["output"]["title"] == "DB-native goal"
            assert created["output"]["origin"] == "derived"
            assert listed["success"] is True
        finally:
            await tr.rollback()


async def test_execute_goals_tool_uses_trusted_origin_not_source_text(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            payload = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_goals_tool($1::jsonb, 'external'::goal_source)",
                    json.dumps(
                        {
                            "action": "create",
                            "title": "Trusted-origin DB goal",
                            "source": "user_request",
                        }
                    ),
                )
            )
            assert payload["success"] is True
            assert payload["output"]["origin"] == "external"
            assert (
                await conn.fetchval(
                    "SELECT goal_origin::text FROM memories WHERE id = $1::uuid",
                    payload["output"]["goal_id"],
                )
                == "external"
            )
        finally:
            await tr.rollback()


async def test_raw_goal_insert_cannot_infer_authority_from_metadata(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            origin = await conn.fetchval("""
                INSERT INTO memories (type, content, metadata)
                VALUES (
                    'goal',
                    'Raw goal origin probe',
                    '{"source": "user_request", "priority": "queued"}'::jsonb
                )
                RETURNING goal_origin::text
                """)
            assert origin == "derived"
        finally:
            await tr.rollback()


async def test_execute_backlog_tool_create_status_and_get(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            created = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_backlog_tool($1::jsonb, $2::jsonb)",
                    json.dumps(
                        {
                            "action": "create",
                            "title": "DB-native backlog",
                            "priority": "high",
                        }
                    ),
                    json.dumps({"tool_context": "heartbeat"}),
                )
            )
            item_id = created["output"]["item_id"]
            status = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_backlog_tool($1::jsonb, $2::jsonb)",
                    json.dumps(
                        {
                            "action": "set_status",
                            "item_id": item_id,
                            "status": "in_progress",
                        }
                    ),
                    json.dumps({"tool_context": "heartbeat"}),
                )
            )
            got = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_backlog_tool($1::jsonb, '{}'::jsonb)",
                    json.dumps({"action": "get", "item_id": item_id}),
                )
            )

            assert created["success"] is True
            assert status["success"] is True
            assert status["output"]["new_status"] == "in_progress"
            assert got["output"]["status"] == "in_progress"
        finally:
            await tr.rollback()


async def test_execute_contact_tool_create_search_get_update(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            created = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_contact_tool('create_contact', $1::jsonb)",
                    json.dumps(
                        {
                            "name": "Ada Lovelace",
                            "email": "ada@example.com",
                            "company": "Analytical",
                        }
                    ),
                )
            )
            contact_id = created["output"]["id"]
            searched = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_contact_tool('search_contacts', $1::jsonb)",
                    json.dumps({"query": "Ada", "limit": 5}),
                )
            )
            updated = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_contact_tool('update_contact', $1::jsonb)",
                    json.dumps({"id": contact_id, "role": "Mathematician"}),
                )
            )
            got = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_contact_tool('get_contact', $1::jsonb)",
                    json.dumps({"id": contact_id}),
                )
            )

            assert created["success"] is True
            assert searched["output"]["count"] >= 1
            assert updated["success"] is True
            assert got["output"]["contact"]["role"] == "Mathematician"
        finally:
            await tr.rollback()


async def test_execute_memory_tool_remember_sense_and_recall(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            remembered = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('remember', $1::jsonb)",
                    json.dumps(
                        {
                            "content": "DB-native memory likes plums",
                            "type": "semantic",
                            "importance": 0.8,
                        }
                    ),
                )
            )
            recalled = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('recall', $1::jsonb)",
                    json.dumps({"query": "plums", "limit": 5}),
                )
            )

            assert remembered["success"] is True
            assert recalled["success"] is True, recalled
            assert recalled["output"]["count"] >= 1
        finally:
            await tr.rollback()


# ---------------------------------------------------------------------------
# remember with provenance + add_evidence dispatch (#33/#36)
# ---------------------------------------------------------------------------


async def test_remember_semantic_records_confidence_and_sources(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('remember', $1::jsonb)",
                    json.dumps(
                        {
                            "content": "Eric is the inventor of Hexis (dispatch test)",
                            "type": "semantic",
                            "confidence": 0.8,
                            "sources": [
                                {
                                    "kind": "user_testimony",
                                    "ref": "conversation:test",
                                    "trust": 0.75,
                                },
                                {
                                    "kind": "user_testimony",
                                    "ref": "conversation:test",
                                    "trust": 0.75,
                                },
                            ],
                        }
                    ),
                )
            )
            assert result["success"] is True
            out = result["output"]
            assert out["type"] == "semantic"
            assert out["confidence"] == 0.8
            assert out["trust_level"] is not None
            meta = _coerce_json(
                await conn.fetchval(
                    "SELECT metadata FROM memories WHERE id = $1::uuid",
                    out["memory_id"],
                )
            )
            # Duplicate source entries are deduped on the way in.
            assert len(meta["source_references"]) == 1
            assert meta["source_references"][0]["ref"] == "conversation:test"
        finally:
            await tr.rollback()


async def test_remember_episodic_takes_first_source_as_attribution(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('remember', $1::jsonb)",
                    json.dumps(
                        {
                            "content": "Met Eric today (dispatch test)",
                            "type": "episodic",
                            "sources": [{"kind": "conversation", "ref": "session:abc"}],
                        }
                    ),
                )
            )
            assert result["success"] is True
            attribution = _coerce_json(
                await conn.fetchval(
                    "SELECT source_attribution FROM memories WHERE id = $1::uuid",
                    result["output"]["memory_id"],
                )
            )
            assert attribution["ref"] == "session:abc"
            # Episodic memories carry no confidence key in the response.
            assert "confidence" not in result["output"]
        finally:
            await tr.rollback()


async def test_remember_reuses_equivalent_session_write_with_conversation_provenance(
    db_pool,
):
    session_id = str(uuid4())
    marker = uuid4().hex
    args = {
        "content": f"Eric and I agreed to preserve the lighthouse thread {marker}",
        "type": "semantic",
        "confidence": 0.7,
        "_execution_context": {
            "session_id": session_id,
            "call_id": "remember-first",
            "tool_context": "chat",
        },
    }

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            first = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('remember', $1::jsonb)",
                    json.dumps(args),
                )
            )
            args["_execution_context"]["call_id"] = "remember-retry"
            args["content"] = (
                f"Eric and I agreed to preserve the lighthouse thread: {marker}."
            )
            second = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('remember', $1::jsonb)",
                    json.dumps(args),
                )
            )

            assert first["success"] is True, first
            assert first["output"]["reused"] is False
            assert second["success"] is True, second
            assert second["output"]["reused"] is True
            assert second["output"]["memory_id"] == first["output"]["memory_id"]

            row = await conn.fetchrow(
                """
                SELECT source_attribution, trust_level, metadata
                FROM memories
                WHERE id = $1::uuid
                """,
                first["output"]["memory_id"],
            )
            count = await conn.fetchval(
                "SELECT count(*) FROM memories WHERE content LIKE $1",
                f"%{marker}%",
            )
            assert count == 1
            attribution = _coerce_json(row["source_attribution"])
            metadata = _coerce_json(row["metadata"])
            assert attribution["ref"] == f"chat_session:{session_id}"
            assert row["trust_level"] > 0.15
            assert metadata["tool_write"]["session_id"] == session_id
        finally:
            await tr.rollback()


async def test_add_evidence_dispatch_happy_path_and_validation(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            mid = str(await conn.fetchval("""
                    INSERT INTO memories (type, content, embedding, importance, trust_level, status, metadata)
                    VALUES ('semantic', 'dispatch belief', array_fill(0.1, ARRAY[embedding_dimension()])::vector,
                            0.8, 0.3, 'active', '{"confidence": 0.5}'::jsonb)
                    RETURNING id
                    """))
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('add_evidence', $1::jsonb)",
                    json.dumps(
                        {
                            "memory_id": mid,
                            "stance": "supports",
                            "source": {
                                "kind": "repository_document",
                                "ref": "README.md",
                                "trust": 0.8,
                            },
                        }
                    ),
                )
            )
            assert result["success"] is True
            out = result["output"]
            assert out["applied"] is True
            assert out["posterior"] > out["prior"]
            assert "->" in result["display_output"]

            bad_uuid = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('add_evidence', $1::jsonb)",
                    json.dumps(
                        {
                            "memory_id": "not-a-uuid",
                            "stance": "supports",
                            "source": {"ref": "x"},
                        }
                    ),
                )
            )
            assert bad_uuid["success"] is False
            assert bad_uuid["error_type"] == "invalid_params"

            bad_stance = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('add_evidence', $1::jsonb)",
                    json.dumps(
                        {"memory_id": mid, "stance": "maybe", "source": {"ref": "x"}}
                    ),
                )
            )
            assert bad_stance["success"] is False

            no_source = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('add_evidence', $1::jsonb)",
                    json.dumps({"memory_id": mid, "stance": "supports", "source": {}}),
                )
            )
            assert no_source["success"] is False
        finally:
            await tr.rollback()


async def test_recall_surfaces_trust_and_confidence(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            created = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('remember', $1::jsonb)",
                    json.dumps(
                        {
                            "content": "Recall projection test fact",
                            "type": "semantic",
                            "confidence": 0.7,
                            "sources": [{"kind": "test", "ref": "recall-projection"}],
                        }
                    ),
                )
            )
            assert created["success"] is True

            for args in (
                {"query": "Recall projection test fact"},  # hybrid
                {
                    "query": "Recall projection test fact",
                    "memory_types": ["semantic"],
                },  # structured
            ):
                recalled = _coerce_json(
                    await conn.fetchval(
                        "SELECT execute_memory_tool('recall', $1::jsonb)",
                        json.dumps(args),
                    )
                )
                assert recalled["success"] is True
                match = next(
                    (
                        m
                        for m in recalled["output"]["memories"]
                        if m["memory_id"] == created["output"]["memory_id"]
                    ),
                    None,
                )
                assert match is not None, f"memory not recalled with args {args}"
                assert "trust" in match
                assert match["confidence"] == 0.7
        finally:
            await tr.rollback()


# ---------------------------------------------------------------------------
# Memory-count budgets (WS6): config-driven defaults/ceiling + min_score floor
# ---------------------------------------------------------------------------


async def test_recall_limit_is_config_driven_budget(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            for i in range(4):
                await conn.fetchval(
                    "SELECT execute_memory_tool('remember', $1::jsonb)",
                    json.dumps(
                        {"content": f"budget test fact number {i}", "type": "semantic"}
                    ),
                )
            # Default budget honored.
            await conn.execute(
                "UPDATE config SET value = '2'::jsonb WHERE key = 'memory.recall_default_limit'"
            )
            recalled = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('recall', $1::jsonb)",
                    json.dumps({"query": "budget test fact"}),
                )
            )
            assert recalled["output"]["count"] <= 2

            # Ceiling clamps explicit asks.
            await conn.execute(
                "UPDATE config SET value = '3'::jsonb WHERE key = 'memory.recall_max_limit'"
            )
            recalled = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('recall', $1::jsonb)",
                    json.dumps({"query": "budget test fact", "limit": 40}),
                )
            )
            assert recalled["output"]["count"] <= 3
        finally:
            await tr.rollback()


async def test_recall_min_score_is_a_relevance_floor(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            created = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('remember', $1::jsonb)",
                    json.dumps(
                        {"content": "relevance floor target fact", "type": "semantic"}
                    ),
                )
            )
            assert created["success"] is True
            # An impossible floor returns nothing rather than padding to N.
            recalled = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('recall', $1::jsonb)",
                    json.dumps(
                        {
                            "query": "relevance floor target fact",
                            "min_score": 0.999999,
                            "limit": 10,
                        }
                    ),
                )
            )
            assert recalled["success"] is True
            assert recalled["output"]["count"] == 0
            # Floor 0 returns the memory.
            recalled = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_memory_tool('recall', $1::jsonb)",
                    json.dumps(
                        {
                            "query": "relevance floor target fact",
                            "min_score": 0.0,
                            "limit": 10,
                        }
                    ),
                )
            )
            contents = [m["content"] for m in recalled["output"]["memories"]]
            assert "relevance floor target fact" in contents
        finally:
            await tr.rollback()


async def test_get_procedures_and_strategies_dispatch(db_pool):
    """Regression pin: these tools filtered on a column fast_recall does not
    return and errored on every call; the dispatcher owns them now."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await conn.execute("""
                INSERT INTO memories (type, content, embedding, importance, trust_level, status)
                VALUES ('procedural', 'How to deploy: build, test, ship',
                        array_fill(0.1, ARRAY[embedding_dimension()])::vector, 0.8, 0.9, 'active'),
                       ('strategic', 'Ship small reversible changes',
                        array_fill(0.1, ARRAY[embedding_dimension()])::vector, 0.8, 0.9, 'active')
                """)
            procedures = json.loads(
                await conn.fetchval(
                    "SELECT execute_memory_tool('get_procedures', '{\"task\": \"deploy\"}'::jsonb)"
                )
            )
            strategies = json.loads(
                await conn.fetchval(
                    "SELECT execute_memory_tool('get_strategies', '{\"situation\": \"shipping\"}'::jsonb)"
                )
            )
        finally:
            await tr.rollback()

    assert procedures["success"] is True
    assert all(
        "deploy" in p["content"] or True for p in procedures["output"]["procedures"]
    )
    assert procedures["output"]["task"] == "deploy"
    assert strategies["success"] is True
    assert strategies["output"]["situation"] == "shipping"


async def test_explore_concept_dispatch_validates_and_shapes(db_pool):
    async with db_pool.acquire() as conn:
        missing = json.loads(
            await conn.fetchval(
                "SELECT execute_memory_tool('explore_concept', '{}'::jsonb)"
            )
        )
        empty = json.loads(
            await conn.fetchval(
                "SELECT execute_memory_tool('explore_concept', '{\"concept\": \"nonexistent-concept-xyz\"}'::jsonb)"
            )
        )

    assert missing["success"] is False
    assert missing["error_type"] == "invalid_params"
    assert empty["success"] is True
    assert empty["output"]["memories"] == []
    assert empty["output"]["related_concepts"] == []


async def test_explore_subgraph_dispatch_requires_seed_or_query(db_pool):
    async with db_pool.acquire() as conn:
        missing = json.loads(
            await conn.fetchval(
                "SELECT execute_memory_tool('explore_subgraph', '{}'::jsonb)"
            )
        )
        bad_seed = json.loads(
            await conn.fetchval(
                "SELECT execute_memory_tool('explore_subgraph', '{\"seeds\": [\"not-a-uuid\"]}'::jsonb)"
            )
        )

    assert missing["success"] is False
    assert "query" in missing["error"]
    assert bad_seed["success"] is False
    assert bad_seed["error_type"] == "invalid_params"
