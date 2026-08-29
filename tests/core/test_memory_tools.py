"""Memory tool behavior through the DB dispatcher.

execute_memory_tool (db/38) owns retrieval policy: plain-query recalls route
to the hybrid retriever (and label rows with retrieval_source), filtered
recalls route to recall_memories_structured, and source_attribution is
flattened into source_* keys. The former Python fallback was deleted.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.memory import (
    DiffMemoryHistoryHandler,
    GetStrategiesHandler,
    LoadDocumentsHandler,
    OpenDocumentHandler,
    OpenDocumentsHandler,
    QueueUserMessageHandler,
    RecallAtTimeHandler,
    RecallHandler,
    RememberHandler,
    SearchDocumentsHandler,
    SearchHistoryHandler,
)
from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _ctx(db_pool, *, is_group: bool = False) -> ToolExecutionContext:
    registry = MagicMock()
    registry.pool = db_pool
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="test-call",
        registry=registry,
        is_group=is_group,
    )


async def _cleanup(db_pool, marker: str) -> None:
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE content LIKE $1", f"%{marker}%")


class TestRecallThroughDbDispatcher:
    async def test_query_only_uses_hybrid(self, db_pool, ensure_embedding_service):
        marker = get_test_identifier("hybridrecall")
        try:
            remembered = await RememberHandler().execute(
                {
                    "content": f"The zephyr protocol codename is {marker}",
                    "type": "semantic",
                },
                _ctx(db_pool),
            )
            assert remembered.success, remembered.error

            result = await RecallHandler().execute(
                {"query": f"zephyr protocol codename {marker}", "limit": 5},
                _ctx(db_pool),
            )
            assert result.success, result.error
            hits = [m for m in result.output["memories"] if marker in m["content"]]
            assert hits, result.output
            # The plain-query path is the hybrid retriever, which labels rows.
            assert hits[0].get("retrieval_source"), hits[0]
        finally:
            await _cleanup(db_pool, marker)


class TestTemporalMemoryTools:
    async def test_temporal_tools_are_registered_with_situational_contracts(self):
        from core.tools.memory import create_memory_tools

        handlers = {handler.spec.name: handler for handler in create_memory_tools()}
        assert {"recall_at_time", "diff_memory_history"} <= handlers.keys()
        assert handlers["recall_at_time"].spec.parameters["required"] == [
            "query",
            "as_of",
        ]
        assert handlers["diff_memory_history"].spec.parameters["required"] == [
            "query",
            "from_time",
        ]
        assert "as of" in handlers["recall_at_time"].spec.description.lower()
        assert (
            "what changed" in handlers["diff_memory_history"].spec.description.lower()
        )

    async def test_handlers_expose_historical_snapshot_and_diff_with_citations(
        self, db_pool, ensure_embedding_service
    ):
        marker = get_test_identifier("temporaltools")
        now = datetime.now(timezone.utc)
        created_at = now - timedelta(days=2)
        try:
            async with db_pool.acquire() as conn:
                memory_id = await conn.fetchval(
                    """
                    INSERT INTO memories (
                        type, content, embedding, embedding_status,
                        created_at, valid_from, source_attribution, trust_level
                    ) VALUES (
                        'semantic', $1,
                        array_fill(0.1, ARRAY[embedding_dimension()])::vector,
                        'embedded', $2, $2,
                        '{"kind":"user_testimony","label":"Temporal tool test"}'::jsonb,
                        0.8
                    ) RETURNING id
                    """,
                    f"The temporal tool marker is {marker}",
                    created_at,
                )

            snapshot = await RecallAtTimeHandler().execute(
                {
                    "query": f"temporal tool marker {marker}",
                    "as_of": (now - timedelta(days=1)).isoformat(),
                    "min_score": 0,
                },
                _ctx(db_pool),
            )
            changed = await DiffMemoryHistoryHandler().execute(
                {
                    "query": f"temporal tool marker {marker}",
                    "from_time": (now - timedelta(days=3)).isoformat(),
                    "to_time": now.isoformat(),
                    "min_score": 0,
                },
                _ctx(db_pool),
            )

            assert snapshot.success, snapshot.error
            hit = next(
                item
                for item in snapshot.output["memories"]
                if item["memory_id"] == str(memory_id)
            )
            assert hit["citation_id"] == f"mem-{memory_id}"
            assert changed.success, changed.error
            assert str(memory_id) in {
                item["memory_id"] for item in changed.output["added"]
            }
        finally:
            await _cleanup(db_pool, marker)


class TestSearchHistoryTouch:
    async def test_search_history_marks_raw_turn_accessed(self, db_pool):
        marker = get_test_identifier("historytouch")
        source_identity = f"history-touch-{marker}"
        async with db_pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT recmem_ingest_turn($1, $2, NULL, $3)",
                f"desk paper {marker}",
                "noted",
                source_identity,
            )
            unit = json.loads(raw) if isinstance(raw, str) else raw
            unit_id = unit["unit_id"]

        try:
            result = await SearchHistoryHandler().execute(
                {"query": f"desk paper {marker}", "sources": ["turn"]},
                _ctx(db_pool),
            )
            assert result.success, result.error
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT access_count, last_accessed FROM subconscious_units WHERE id = $1::uuid",
                    unit_id,
                )
            assert row["access_count"] == 1
            assert row["last_accessed"] is not None
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM subconscious_units WHERE id = $1::uuid", unit_id
                )

    async def test_structured_filters_use_structured_query(
        self, db_pool, ensure_embedding_service
    ):
        marker = get_test_identifier("structuredrecall")
        kind = f"web_{marker}"
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO memories (type, content, embedding, importance, trust_level, status, source_attribution)
                    VALUES ('semantic', $1, array_fill(0.1, ARRAY[embedding_dimension()])::vector,
                            0.8, 0.9, 'active', $2::jsonb)
                    """,
                    f"Structured recall subject {marker}",
                    json.dumps({"kind": kind, "label": "Doc"}),
                )

            result = await RecallHandler().execute({"source_kind": kind}, _ctx(db_pool))
            assert result.success, result.error
            assert result.output["count"] == 1, result.output
            memory = result.output["memories"][0]
            # Filtered recalls take the structured path: no retrieval_source,
            # and source_attribution is flattened into source_* keys.
            assert "retrieval_source" not in memory
            assert memory["source_kind"] == kind
            assert memory["source_label"] == "Doc"
        finally:
            await _cleanup(db_pool, marker)

    async def test_recall_requires_query_or_filter(self, db_pool):
        result = await RecallHandler().execute({}, _ctx(db_pool))
        assert not result.success
        assert "query or one filter" in (result.error or "")


class TestQueueUserMessage:
    async def test_chat_context_delivers_to_web_inbox(self, db_pool):
        marker = get_test_identifier("chatoutbox")
        message = f"chat-visible note {marker}"
        try:
            result = await QueueUserMessageHandler().execute(
                {"message": message, "intent": "status"},
                _ctx(db_pool),
            )
            assert result.success, result.error
            assert result.output["queued"] is True
            assert result.output["delivered"] is True
            assert result.output["delivery"] == {"mode": "web_inbox"}

            async with db_pool.acquire() as conn:
                inbox = await conn.fetchrow(
                    "SELECT message, intent FROM web_inbox WHERE id = $1::uuid",
                    result.output["web_inbox_id"],
                )
                outbox_status = await conn.fetchval(
                    "SELECT status FROM outbox_messages WHERE id = $1::uuid",
                    result.output["outbox_id"],
                )
            assert inbox["message"] == message
            assert inbox["intent"] == "status"
            assert outbox_status == "published"
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM web_inbox WHERE message = $1", message)
                await conn.execute(
                    "DELETE FROM outbox_messages WHERE envelope->'payload'->>'message' = $1",
                    message,
                )


class TestProvenanceTooling:
    async def test_remember_handler_passes_session_context_for_idempotency(
        self, db_pool, ensure_embedding_service
    ):
        marker = get_test_identifier("rememberreceipt")
        context = _ctx(db_pool)
        context.session_id = str(uuid4())
        try:
            first = await RememberHandler().execute(
                {"content": f"Keep the copper thread {marker}", "type": "semantic"},
                context,
            )
            context.call_id = "test-call-retry"
            second = await RememberHandler().execute(
                {"content": f"Keep the copper thread: {marker}.", "type": "semantic"},
                context,
            )

            assert first.success, first.error
            assert second.success, second.error
            assert first.output["memory_id"] == second.output["memory_id"]
            assert second.output["reused"] is True
        finally:
            await _cleanup(db_pool, marker)

    async def test_get_strategies_schema_accepts_query_alias(self):
        handler = GetStrategiesHandler()
        assert handler.validate({"query": "recover from heartbeat failures"}) == []
        assert handler.validate({"situation": "recover from heartbeat failures"}) == []

    async def test_remember_schema_supports_sources_and_confidence(self):
        spec = RememberHandler().spec
        props = spec.parameters["properties"]
        assert "confidence" in props
        assert props["sources"]["type"] == "array"
        source_props = props["sources"]["items"]["properties"]
        assert {"kind", "ref", "label", "author", "trust"} <= set(source_props)

    async def test_add_evidence_handler_registered(self):
        from core.tools.memory import AddEvidenceHandler, create_memory_tools

        names = [handler.spec.name for handler in create_memory_tools()]
        assert "add_evidence" in names
        spec = AddEvidenceHandler().spec
        assert spec.parameters["required"] == ["memory_id", "stance", "source"]
        assert spec.is_read_only is False
        assert "prior" in spec.description


class TestBeliefHistoryTool:
    async def test_belief_history_handler_registered(self):
        from core.tools.memory import BeliefHistoryHandler, create_memory_tools

        names = [handler.spec.name for handler in create_memory_tools()]
        assert "belief_history" in names
        spec = BeliefHistoryHandler().spec
        assert spec.is_read_only is True
        assert spec.energy_cost == 0
        assert spec.parameters["required"] == ["memory_id"]
        assert "why" in spec.description.lower()


class TestSourceDocumentTools:
    async def test_search_and_open_documents(self, db_pool):
        marker = get_test_identifier("sourcedoctool")
        content_hash = f"hash-{marker}"
        content = f"Tool document {marker}\n\nThe exact arclight clause is here."
        try:
            async with db_pool.acquire() as conn:
                await conn.fetchval(
                    """
                    SELECT upsert_source_document(
                        $1, 'document', $2, $3, '.txt', $4, 8,
                        $5::jsonb, '{}'::jsonb
                    )
                    """,
                    f"Tool document {marker}",
                    content_hash,
                    f"/tmp/{marker}.txt",
                    content,
                    json.dumps(
                        {
                            "kind": "document",
                            "ref": content_hash,
                            "content_hash": content_hash,
                        }
                    ),
                )

            result = await SearchDocumentsHandler().execute(
                {"query": f"arclight {marker}"}, _ctx(db_pool)
            )
            assert result.success, result.error
            assert result.output["count"] == 1
            doc_id = result.output["documents"][0]["document_id"]

            opened = await OpenDocumentHandler().execute(
                {"document_id": doc_id}, _ctx(db_pool)
            )
            assert opened.success, opened.error
            assert opened.output["content"] == content
            assert opened.output["truncated"] is False

            batch = await OpenDocumentsHandler().execute(
                {"document_ids": [doc_id], "max_chars": 12},
                _ctx(db_pool),
            )
            assert batch.success, batch.error
            assert batch.output["count"] == 1
            assert batch.output["documents"][0]["content"] == content[:12]
            assert batch.output["documents"][0]["truncated"] is True

            loaded = await LoadDocumentsHandler().execute(
                {
                    "document_ids": [doc_id],
                    "chunk_chars": 500,
                    "reason": "test exact-source desk search",
                },
                _ctx(db_pool),
            )
            assert loaded.success, loaded.error
            assert loaded.output["count"] == 1
            desk_unit_id = loaded.output["desk_unit_ids"][0]

            desk_hit = await SearchHistoryHandler().execute(
                {"query": f"arclight {marker}", "sources": ["desk"]},
                _ctx(db_pool),
            )
            assert desk_hit.success, desk_hit.error
            assert desk_hit.output["count"] == 1
            assert desk_hit.output["results"][0]["source_kind"] == "desk"
            assert desk_hit.output["results"][0]["item_id"] == desk_unit_id
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM subconscious_units WHERE metadata#>>'{recmem,content_hash}' = $1",
                    content_hash,
                )
                await conn.execute(
                    "DELETE FROM source_documents WHERE content_hash = $1", content_hash
                )

    async def test_document_handlers_registered(self):
        from core.tools.memory import create_memory_tools

        names = [handler.spec.name for handler in create_memory_tools()]
        assert "search_documents" in names
        assert "open_document" in names
        assert "open_documents" in names
        assert "load_documents" in names
