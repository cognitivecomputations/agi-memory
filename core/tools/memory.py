"""
Hexis Tools System - Memory Tools

Tools for memory operations (recall, remember, etc.).
These wrap the existing CognitiveMemory API.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from .base import (
    OutboundSpec,
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

logger = logging.getLogger(__name__)


async def _try_db_memory_tool(
    tool_name: str, arguments: dict[str, Any], context: ToolExecutionContext
) -> ToolResult | None:
    pool = context.registry.pool if context.registry else None
    if not pool:
        return None
    db_arguments = dict(arguments)
    if tool_name == "remember":
        db_arguments["_execution_context"] = {
            "session_id": context.session_id,
            "call_id": context.call_id,
            "tool_context": context.tool_context.value,
        }
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT enrich_memory_tool_result(
                    $1::text,
                    execute_memory_tool($1::text, $2::jsonb)
                )
                """,
                tool_name,
                json.dumps(db_arguments),
            )
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(payload, dict) and "success" in payload:
            if payload.get("success"):
                return ToolResult.success_result(
                    payload.get("output"), payload.get("display_output")
                )
            try:
                error_type = ToolErrorType(
                    payload.get("error_type") or ToolErrorType.EXECUTION_FAILED.value
                )
            except ValueError:
                error_type = ToolErrorType.EXECUTION_FAILED
            return ToolResult.error_result(
                payload.get("error") or "Memory tool failed", error_type
            )
    except Exception:
        return None
    return None


class RecallHandler(ToolHandler):
    """Search memories by semantic similarity."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="recall",
            description=(
                "Search memories by semantic similarity and/or structured filters. "
                "Use this to find memories related to a topic, concept, or question. "
                "Supports filtering by source, date range, concept graph, and metadata. "
                "Can be used with just filters (no query) for targeted retrieval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what you want to remember. Optional if using filters.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Memory-count budget for this recall. Defaults and ceiling "
                            "are config-driven (memory.recall_default_limit / "
                            "memory.recall_max_limit); ask for more when the question "
                            "genuinely spans many memories."
                        ),
                        "minimum": 1,
                    },
                    "min_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "Relevance floor: drop results scoring below this instead "
                            "of relying on count alone."
                        ),
                    },
                    "memory_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "episodic",
                                "semantic",
                                "procedural",
                                "strategic",
                                "worldview",
                                "goal",
                            ],
                        },
                        "description": (
                            "Filter by memory types. Omit to search ALL types (recommended for most queries). "
                            "Types: episodic (events/experiences), semantic (facts/knowledge), "
                            "procedural (how-to), strategic (patterns/plans), "
                            "worldview (identity, values, beliefs, boundaries, interests), "
                            "goal (objectives/aspirations)."
                        ),
                    },
                    "min_importance": {
                        "type": "number",
                        "description": "Minimum importance score (0.0-1.0).",
                        "default": 0.0,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "source_path": {
                        "type": "string",
                        "description": "Filter by source path (partial match). E.g., 'hexis/db' for all DB schema memories.",
                    },
                    "source_kind": {
                        "type": "string",
                        "description": "Filter by source kind. E.g., 'code', 'conversation', 'web', 'document'.",
                    },
                    "created_after": {
                        "type": "string",
                        "description": "Only memories created after this ISO date (e.g. '2025-01-15').",
                    },
                    "created_before": {
                        "type": "string",
                        "description": "Only memories created before this ISO date.",
                    },
                    "concept": {
                        "type": "string",
                        "description": "Find memories linked to this concept in the knowledge graph.",
                    },
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = dict(arguments)
        if context.is_group:
            # Group rooms recall without private memories (#92/#96) — the
            # same wall hydrate already enforces, applied to the tool path.
            args["exclude_sensitive"] = True
        db_result = await _try_db_memory_tool("recall", args, context)
        if db_result is not None:
            return db_result
        # execute_memory_tool (db/38) owns this tool; the former Python
        # compatibility path was deleted.
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class SearchHistoryHandler(ToolHandler):
    """Run free lexical search across prior turns and consolidated memories."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_history",
            description=(
                "Search exact words, names, phrases, and operators across stored "
                "conversation turns and consolidated memories using Postgres full-text "
                "search. This is cross-session and does not require embeddings. "
                "For a pure timeline ('what happened yesterday?'), pass created_after/"
                "created_before with an empty query — a time window alone returns "
                "everything in it, newest first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Postgres web-search query. Supports quoted phrases, OR, "
                            "and minus-prefixed exclusions. Leave empty to browse a "
                            "time window chronologically."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 20,
                        "description": (
                            "Up to 50 for keyword search; time-window browsing "
                            "(no keywords) allows up to 200 preview-grain rows."
                        ),
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["turn", "memory", "desk"]},
                        "minItems": 1,
                        "uniqueItems": True,
                        "default": ["turn", "memory"],
                    },
                    "created_after": {
                        "type": "string",
                        "description": "Optional inclusive ISO-8601 lower time bound.",
                    },
                    "created_before": {
                        "type": "string",
                        "description": "Optional exclusive ISO-8601 upper time bound.",
                    },
                    "exclude_current_session": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Exclude raw turns from the current UUID session when one "
                            "is available. Consolidated memories remain searchable."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        # The session id is I/O context only Python holds; everything else —
        # validation, browse-vs-keyword limits, shaping, the truncation
        # note — is owned by execute_memory_tool (db/38).
        args = dict(arguments)
        if context.is_group:
            args["exclude_sensitive"] = True
        if args.pop("exclude_current_session", True) and context.session_id:
            try:
                args["exclude_session_id"] = str(UUID(str(context.session_id)))
            except ValueError:
                pass
        db_result = await _try_db_memory_tool("search_history", args, context)
        if db_result is not None:
            if db_result.success and context.registry and context.registry.pool:
                await self._touch_history_results(db_result.output, context)
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )

    async def _touch_history_results(
        self, output: Any, context: ToolExecutionContext
    ) -> None:
        """Advisory access marking: browsing a desk item counts as using it."""
        if not isinstance(output, dict):
            return
        raw_results = output.get("results")
        if not isinstance(raw_results, list):
            return
        raw_unit_ids: list[UUID] = []
        memory_ids: list[UUID] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            try:
                item_id = UUID(str(item.get("item_id") or ""))
            except ValueError:
                continue
            if item.get("source_kind") in ("turn", "desk"):
                raw_unit_ids.append(item_id)
            elif item.get("source_kind") == "memory":
                memory_ids.append(item_id)

        if not raw_unit_ids and not memory_ids:
            return
        try:
            async with context.registry.pool.acquire() as conn:
                if raw_unit_ids:
                    await conn.execute(
                        "SELECT touch_subconscious_units($1::uuid[])", raw_unit_ids
                    )
                if memory_ids:
                    await conn.execute("SELECT touch_memories($1::uuid[])", memory_ids)
        except Exception as exc:
            logger.warning("Failed to mark search_history results as accessed: %s", exc)


class RecallAtTimeHandler(ToolHandler):
    """Recall only memories whose validity window covered one past instant."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="recall_at_time",
            description=(
                "Answer temporally framed memory questions such as 'as of last "
                "Tuesday' or 'what did you know back then?'. Searches the durable "
                "validity and supersession record at one exact past instant, "
                "including memories that are no longer current. Returns historical "
                "confidence/trust and citation envelopes. Resolve relative dates "
                "against the current temporal context before calling."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or claim to recall at the historical instant.",
                    },
                    "as_of": {
                        "type": "string",
                        "description": "Required ISO-8601 timestamp at or before now.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Config-bounded memory result budget.",
                    },
                    "min_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "memory_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "episodic",
                                "semantic",
                                "procedural",
                                "strategic",
                                "worldview",
                                "goal",
                            ],
                        },
                        "uniqueItems": True,
                    },
                },
                "required": ["query", "as_of"],
                "additionalProperties": False,
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = dict(arguments)
        if context.is_group:
            args["exclude_sensitive"] = True
        db_result = await _try_db_memory_tool("recall_at_time", args, context)
        if db_result is not None:
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class DiffMemoryHistoryHandler(ToolHandler):
    """Compare two point-in-time snapshots and return the reasons for change."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="diff_memory_history",
            description=(
                "Answer 'what changed about X between then and now, and why?'. "
                "Compares two historical memory snapshots and returns additions, "
                "expired beliefs, supersession reasons, audited belief revisions, "
                "and explicit contradiction decisions. Use this for 'has that "
                "changed?' rather than reconstructing history from current recall."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to compare."},
                    "from_time": {
                        "type": "string",
                        "description": "Required earlier ISO-8601 timestamp.",
                    },
                    "to_time": {
                        "type": "string",
                        "description": "Later ISO-8601 timestamp; omit to compare with now.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Config-bounded result budget per snapshot.",
                    },
                    "min_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "memory_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "episodic",
                                "semantic",
                                "procedural",
                                "strategic",
                                "worldview",
                                "goal",
                            ],
                        },
                        "uniqueItems": True,
                    },
                },
                "required": ["query", "from_time"],
                "additionalProperties": False,
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = dict(arguments)
        if context.is_group:
            args["exclude_sensitive"] = True
        db_result = await _try_db_memory_tool("diff_memory_history", args, context)
        if db_result is not None:
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )

class RememberHandler(ToolHandler):
    """Store or reuse a durable memory."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="remember",
            description=(
                "Store a durable memory. Equivalent recent writes in the same "
                "chat session reuse the existing memory. Use this to save important "
                "information, "
                "events, or learnings for future recall. When the memory comes "
                "from a document, conversation, or other source, cite it in "
                "`sources` — provenance is what makes a belief revisable and "
                "explainable later."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember.",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural", "strategic"],
                        "default": "episodic",
                        "description": "Type of memory to create.",
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score (0.0-1.0).",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concepts to link this memory to.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                        "description": (
                            "How confident you are the content is true, given the "
                            "evidence (semantic memories only)."
                        ),
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "description": (
                                        "Evidence kind, e.g. user_testimony, "
                                        "repository_document, web_page, self_observation."
                                    ),
                                },
                                "ref": {
                                    "type": "string",
                                    "description": "Path, URL, or identifier of the source.",
                                },
                                "label": {"type": "string"},
                                "author": {"type": "string"},
                                "trust": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                        },
                        "description": (
                            "Where this came from. Semantic memories record every "
                            "source and derive trust from them; other types use "
                            "the first as attribution."
                        ),
                    },
                },
                "required": ["content"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        db_result = await _try_db_memory_tool("remember", arguments, context)
        if db_result is not None:
            return db_result
        # execute_memory_tool (db/38) owns this tool; the former Python
        # compatibility path was deleted.
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class AddEvidenceHandler(ToolHandler):
    """Attach evidence to an existing belief and revise its confidence."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="add_evidence",
            description=(
                "Attach new evidence to an existing semantic memory (belief) and "
                "revise its confidence through the calibrated evidence policy. "
                "Use this when something you read or were told corroborates or "
                "contradicts a belief you already hold, instead of creating a "
                "duplicate memory. Returns prior and posterior confidence so you "
                "can report honestly how much the evidence moved you; duplicate "
                "sources are merged without changing confidence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "UUID of the semantic memory the evidence bears on (from recall).",
                    },
                    "stance": {
                        "type": "string",
                        "enum": ["supports", "contradicts"],
                        "description": "Whether the evidence supports or contradicts the belief.",
                    },
                    "source": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "description": (
                                    "Evidence kind, e.g. user_testimony, "
                                    "repository_document, web_page, self_observation."
                                ),
                            },
                            "ref": {
                                "type": "string",
                                "description": "Path, URL, or identifier of the source.",
                            },
                            "label": {"type": "string"},
                            "author": {"type": "string"},
                            "trust": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "description": "The evidence source; must include at least ref or label.",
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "Optional short observation of what the evidence says; "
                            "stored as an episodic evidence memory linked to the belief."
                        ),
                    },
                },
                "required": ["memory_id", "stance", "source"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        db_result = await _try_db_memory_tool("add_evidence", arguments, context)
        if db_result is not None:
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class BeliefHistoryHandler(ToolHandler):
    """Explain why a belief is held: revision history, evidence, sources."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="belief_history",
            description=(
                "Explain why you believe something: given a semantic memory's id "
                "(from recall), returns its current confidence and trust, its truth "
                "profile (sources, reinforcement, worldview alignment), the full "
                "audited revision history (what evidence moved confidence, when, "
                "and by how much), linked supporting/contradicting evidence, and "
                "any contradicting sources. Use this when asked why you believe "
                "something or what changed your mind."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "UUID of the memory to explain (from recall).",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                        "description": "Maximum revision entries to return (newest first).",
                    },
                },
                "required": ["memory_id"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        db_result = await _try_db_memory_tool("belief_history", arguments, context)
        if db_result is not None:
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class OpenMemoryHandler(ToolHandler):
    """Graded recall's drill-down: the verbatim experience behind a memory."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="open_memory",
            description=(
                "Open a memory to its verbatim grain: given a memory's id (from "
                "recall or search_history), returns the exact source turns behind "
                "it time-ordered, the pre-summary full text if it has been gisted, "
                "and any archived originals a consolidation superseded. Recall "
                "gives you the shape of a memory; open_memory gives you the exact "
                "words — reach for it when precise wording, quotes, or the full "
                "moment matter."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "UUID of the memory to open.",
                    },
                    "max_units": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 40,
                        "description": "Maximum verbatim source turns to return.",
                    },
                },
                "required": ["memory_id"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        db_result = await _try_db_memory_tool("open_memory", arguments, context)
        if db_result is not None:
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class SearchDocumentsHandler(ToolHandler):
    """Search preserved raw source documents from ingestion."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_documents",
            description=(
                "Search exact preserved source documents from ingestion. "
                "Use this when normal recall finds a source, or when a question "
                "depends on wording in a whole file, spec, email, web page, or "
                "other ingested artifact rather than only distilled memories."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Postgres web-search query over document title, path, and full raw content.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Result offset for paging.",
                    },
                    "source_path": {
                        "type": "string",
                        "description": "Optional partial path/URL filter.",
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Optional source type filter, e.g. document, web, code, email.",
                    },
                    "created_after": {
                        "type": "string",
                        "description": "Optional inclusive ISO-8601 lower time bound.",
                    },
                    "created_before": {
                        "type": "string",
                        "description": "Optional exclusive ISO-8601 upper time bound.",
                    },
                    "snippet_chars": {
                        "type": "integer",
                        "minimum": 80,
                        "maximum": 4000,
                        "default": 500,
                    },
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.registry or not context.registry.pool:
            return ToolResult.error_result(
                "Database unavailable", ToolErrorType.EXECUTION_FAILED
            )

        args = dict(arguments)
        has_selector = any(
            str(args.get(name) or "").strip()
            for name in (
                "query",
                "source_path",
                "source_type",
                "created_after",
                "created_before",
            )
        )
        if not has_selector:
            return ToolResult.error_result(
                "Provide query or one filter (source_path, source_type, created_after, created_before).",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            limit = max(1, min(int(args.get("limit") or 10), 50))
            offset = max(0, int(args.get("offset") or 0))
            snippet_chars = max(80, min(int(args.get("snippet_chars") or 500), 4000))
        except (TypeError, ValueError):
            return ToolResult.error_result(
                "limit, offset, and snippet_chars must be integers",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            async with context.registry.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    WITH hits AS (
                        SELECT *
                        FROM search_source_documents(
                            $1::text, $2::int, $3::text, $4::text,
                            NULLIF($5::text, '')::timestamptz,
                            NULLIF($6::text, '')::timestamptz,
                            false, $7::int, $8::int, $9::boolean
                        )
                    )
                    SELECT h.document_id, h.title, h.source_type, h.path, h.file_type,
                           h.content_hash, h.word_count, h.size_bytes, h.created_at,
                           h.updated_at, h.rank, h.snippet, d.source_attribution,
                           source_citation_envelope(h.document_id) AS citation
                    FROM hits h
                    JOIN source_documents d ON d.id = h.document_id
                    """,
                    args.get("query"),
                    limit,
                    args.get("source_path"),
                    args.get("source_type"),
                    args.get("created_after"),
                    args.get("created_before"),
                    offset,
                    snippet_chars,
                    bool(args.get("exclude_sensitive") or context.is_group),
                )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)

        documents = []
        for row in rows:
            citation = row["citation"]
            source_attribution = row["source_attribution"]
            if isinstance(citation, str):
                citation = json.loads(citation)
            if isinstance(source_attribution, str):
                source_attribution = json.loads(source_attribution)
            documents.append(
                {
                    "document_id": str(row["document_id"]),
                    "citation_id": citation.get("citation_id")
                    if isinstance(citation, dict)
                    else None,
                    "citation": citation,
                    "title": row["title"],
                    "source_type": row["source_type"],
                    "path": row["path"],
                    "file_type": row["file_type"],
                    "content_hash": row["content_hash"],
                    "word_count": row["word_count"],
                    "size_bytes": row["size_bytes"],
                    "created_at": (
                        row["created_at"].isoformat() if row["created_at"] else None
                    ),
                    "updated_at": (
                        row["updated_at"].isoformat() if row["updated_at"] else None
                    ),
                    "rank": row["rank"],
                    "snippet": row["snippet"],
                    "source_attribution": source_attribution,
                    "trust_level": citation.get("trust_level")
                    if isinstance(citation, dict)
                    else None,
                }
            )
        return ToolResult.success_result(
            {
                "documents": documents,
                "count": len(documents),
                "offset": offset,
                "limit": limit,
            },
            display_output=f"Found {len(documents)} source document(s)",
        )


class OpenDocumentHandler(ToolHandler):
    """Open exact preserved raw source document content."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="open_document",
            description=(
                "Open a preserved raw source document by document_id, content_hash, "
                "or path. Omit max_chars to retrieve the full exact content; pass "
                "offset/max_chars for deliberate paging when the document is too "
                "large for the current context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "UUID returned by search_documents or open_memory.source_documents.",
                    },
                    "content_hash": {
                        "type": "string",
                        "description": "Exact content hash for the source document.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Exact or partial path/URL for the source document.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                    },
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional character budget. Omit to retrieve the full document.",
                    },
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.registry or not context.registry.pool:
            return ToolResult.error_result(
                "Database unavailable", ToolErrorType.EXECUTION_FAILED
            )

        args = dict(arguments)
        document_id = args.get("document_id")
        if document_id:
            try:
                document_id = UUID(str(document_id))
            except ValueError:
                return ToolResult.error_result(
                    "document_id must be a uuid", ToolErrorType.INVALID_PARAMS
                )

        if (
            not document_id
            and not str(args.get("content_hash") or "").strip()
            and not str(args.get("path") or "").strip()
        ):
            return ToolResult.error_result(
                "Provide document_id, content_hash, or path.",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            offset = max(0, int(args.get("offset") or 0))
            max_chars = args.get("max_chars")
            if max_chars is not None:
                max_chars = max(1, int(max_chars))
        except (TypeError, ValueError):
            return ToolResult.error_result(
                "offset and max_chars must be integers", ToolErrorType.INVALID_PARAMS
            )

        try:
            async with context.registry.pool.acquire() as conn:
                raw = await conn.fetchval(
                    """
                    SELECT enrich_source_document_payload(open_source_document(
                        $1::uuid, $2::text, $3::text, $4::int, $5::int, $6::boolean
                    ))
                    """,
                    document_id,
                    args.get("content_hash"),
                    args.get("path"),
                    offset,
                    max_chars,
                    bool(args.get("exclude_sensitive") or context.is_group),
                )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)

        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict):
            return ToolResult.error_result(
                "open_source_document returned an invalid payload",
                ToolErrorType.EXECUTION_FAILED,
            )
        if payload.get("error") == "missing_selector":
            return ToolResult.error_result(
                "Provide document_id, content_hash, or path.",
                ToolErrorType.INVALID_PARAMS,
            )
        if payload.get("error") == "not_found":
            return ToolResult.error_result(
                "Source document not found.", ToolErrorType.INVALID_PARAMS
            )

        title = str(
            payload.get("title")
            or payload.get("path")
            or payload.get("document_id")
            or "document"
        )
        suffix = " (truncated)" if payload.get("truncated") else ""
        return ToolResult.success_result(
            payload, display_output=f"Opened source document: {title}{suffix}"
        )


class OpenDocumentsHandler(ToolHandler):
    """Open a deliberate batch of preserved raw source documents."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="open_documents",
            description=(
                "Open multiple preserved raw source documents by document_ids, "
                "content_hashes, or paths. Use this when a task needs a batch of "
                "files/emails/specs from the source-document filing cabinet. Pass "
                "max_chars to page large batches deliberately."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "UUIDs returned by search_documents or open_memory.source_documents.",
                    },
                    "content_hashes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact content hashes for source documents.",
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact or partial paths/URLs for source documents.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                    },
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional per-document character budget. Omit to retrieve full documents.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=2,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.registry or not context.registry.pool:
            return ToolResult.error_result(
                "Database unavailable", ToolErrorType.EXECUTION_FAILED
            )

        args = dict(arguments)

        def _string_list(name: str) -> list[str]:
            raw = args.get(name)
            if raw is None:
                return []
            if not isinstance(raw, list):
                raise ValueError(f"{name} must be an array")
            return [str(item).strip() for item in raw if str(item).strip()]

        try:
            document_ids_raw = _string_list("document_ids")
            content_hashes = _string_list("content_hashes")
            paths = _string_list("paths")
        except ValueError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)

        if not document_ids_raw and not content_hashes and not paths:
            return ToolResult.error_result(
                "Provide document_ids, content_hashes, or paths.",
                ToolErrorType.INVALID_PARAMS,
            )

        document_ids: list[UUID] = []
        for raw_id in document_ids_raw:
            try:
                document_ids.append(UUID(raw_id))
            except ValueError:
                return ToolResult.error_result(
                    "document_ids must contain only uuids", ToolErrorType.INVALID_PARAMS
                )

        try:
            offset = max(0, int(args.get("offset") or 0))
            limit = max(1, min(int(args.get("limit") or 10), 50))
            max_chars = args.get("max_chars")
            if max_chars is not None:
                max_chars = max(1, int(max_chars))
        except (TypeError, ValueError):
            return ToolResult.error_result(
                "offset, limit, and max_chars must be integers",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            async with context.registry.pool.acquire() as conn:
                raw = await conn.fetchval(
                    """
                    SELECT open_source_documents(
                        $1::uuid[], $2::text[], $3::text[], $4::int, $5::int, $6::int, $7::boolean
                    )
                    """,
                    document_ids,
                    content_hashes,
                    paths,
                    offset,
                    max_chars,
                    limit,
                    bool(args.get("exclude_sensitive") or context.is_group),
                )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)

        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict):
            return ToolResult.error_result(
                "open_source_documents returned an invalid payload",
                ToolErrorType.EXECUTION_FAILED,
            )
        if payload.get("error") == "missing_selector":
            return ToolResult.error_result(
                "Provide document_ids, content_hashes, or paths.",
                ToolErrorType.INVALID_PARAMS,
            )

        docs = (
            payload.get("documents")
            if isinstance(payload.get("documents"), list)
            else []
        )
        truncated_count = sum(
            1 for doc in docs if isinstance(doc, dict) and doc.get("truncated")
        )
        suffix = f", {truncated_count} truncated" if truncated_count else ""
        return ToolResult.success_result(
            payload, display_output=f"Opened {len(docs)} source document(s){suffix}"
        )


class LoadDocumentsHandler(ToolHandler):
    """Load preserved source documents onto the RecMem desk."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="load_documents",
            description=(
                "Load one or more preserved raw source documents onto the RecMem "
                "desk as searchable mid-term working material. Use this when a "
                "large file/email/spec needs to stay available for on-demand "
                "desk search during reasoning; use open_document(s) for read-only "
                "inspection."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "UUIDs returned by search_documents or open_memory.source_documents.",
                    },
                    "content_hashes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact content hashes for source documents.",
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact or partial paths/URLs for source documents.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                    },
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional total character window per document.",
                    },
                    "chunk_chars": {
                        "type": "integer",
                        "minimum": 500,
                        "description": "Optional desk chunk size; defaults to memory.source_document_desk_chunk_chars.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason this source needs to be on the desk.",
                    },
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=2,
            is_read_only=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.registry or not context.registry.pool:
            return ToolResult.error_result(
                "Database unavailable", ToolErrorType.EXECUTION_FAILED
            )

        args = dict(arguments)

        def _string_list(name: str) -> list[str]:
            raw = args.get(name)
            if raw is None:
                return []
            if not isinstance(raw, list):
                raise ValueError(f"{name} must be an array")
            return [str(item).strip() for item in raw if str(item).strip()]

        try:
            document_ids_raw = _string_list("document_ids")
            content_hashes = _string_list("content_hashes")
            paths = _string_list("paths")
        except ValueError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)

        if not document_ids_raw and not content_hashes and not paths:
            return ToolResult.error_result(
                "Provide document_ids, content_hashes, or paths.",
                ToolErrorType.INVALID_PARAMS,
            )

        document_ids: list[UUID] = []
        for raw_id in document_ids_raw:
            try:
                document_ids.append(UUID(raw_id))
            except ValueError:
                return ToolResult.error_result(
                    "document_ids must contain only uuids", ToolErrorType.INVALID_PARAMS
                )

        try:
            offset = max(0, int(args.get("offset") or 0))
            limit = max(1, min(int(args.get("limit") or 10), 50))
            max_chars = args.get("max_chars")
            if max_chars is not None:
                max_chars = max(1, int(max_chars))
            chunk_chars = args.get("chunk_chars")
            if chunk_chars is not None:
                chunk_chars = max(500, int(chunk_chars))
        except (TypeError, ValueError):
            return ToolResult.error_result(
                "offset, limit, max_chars, and chunk_chars must be integers",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            async with context.registry.pool.acquire() as conn:
                raw = await conn.fetchval(
                    """
                    SELECT load_source_documents_to_recmem(
                        $1::uuid[], $2::text[], $3::text[], $4::int,
                        $5::int, $6::int, $7::int, $8::boolean, $9::text
                    )
                    """,
                    document_ids,
                    content_hashes,
                    paths,
                    offset,
                    max_chars,
                    chunk_chars,
                    limit,
                    bool(args.get("exclude_sensitive") or context.is_group),
                    args.get("reason"),
                )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)

        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict):
            return ToolResult.error_result(
                "load_source_documents_to_recmem returned an invalid payload",
                ToolErrorType.EXECUTION_FAILED,
            )
        if payload.get("error") == "missing_selector":
            return ToolResult.error_result(
                "Provide document_ids, content_hashes, or paths.",
                ToolErrorType.INVALID_PARAMS,
            )

        count = int(payload.get("count") or 0)
        return ToolResult.success_result(
            payload, display_output=f"Loaded {count} source document desk chunk(s)"
        )


class SearchDocumentChunksHandler(ToolHandler):
    """Hybrid passage-level search over durable source chunks."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_document_chunks",
            description=(
                "Passage-level search of the source-document filing cabinet: "
                "hybrid full-text + embedding retrieval over durable chunks with "
                "citable locators (page, section, sheet row). Prefer this over "
                "search_documents when you need the exact passage rather than "
                "the whole file; each hit carries rank_components explaining "
                "why it ranked."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language or web-search query over chunk text.",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Optional UUID to scope the search to one document.",
                    },
                    "source_path": {
                        "type": "string",
                        "description": "Optional partial path/URL filter.",
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Optional source type filter, e.g. document, web, email, spreadsheet.",
                    },
                    "locator_kind": {
                        "type": "string",
                        "enum": [
                            "char",
                            "page",
                            "section",
                            "sheet_row",
                            "slide",
                            "message",
                        ],
                        "description": "Optional locator-kind filter (e.g. sheet_row for spreadsheet rows).",
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Optional exact sheet name for spreadsheet chunks.",
                    },
                    "page_start": {"type": "integer", "minimum": 1},
                    "page_end": {"type": "integer", "minimum": 1},
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "snippet_chars": {
                        "type": "integer",
                        "minimum": 80,
                        "maximum": 2000,
                        "default": 400,
                    },
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.registry or not context.registry.pool:
            return ToolResult.error_result(
                "Database unavailable", ToolErrorType.EXECUTION_FAILED
            )

        args = dict(arguments)
        has_selector = (
            any(
                str(args.get(name) or "").strip()
                for name in (
                    "query",
                    "document_id",
                    "source_path",
                    "source_type",
                    "locator_kind",
                    "sheet_name",
                )
            )
            or args.get("page_start")
            or args.get("page_end")
        )
        if not has_selector:
            return ToolResult.error_result(
                "Provide query or a scope filter (document_id, source_path, source_type, locator_kind, sheet_name, page range).",
                ToolErrorType.INVALID_PARAMS,
            )

        document_id: UUID | None = None
        if str(args.get("document_id") or "").strip():
            try:
                document_id = UUID(str(args["document_id"]).strip())
            except ValueError:
                return ToolResult.error_result(
                    "document_id must be a uuid", ToolErrorType.INVALID_PARAMS
                )

        try:
            limit = max(1, min(int(args.get("limit") or 10), 50))
            offset = max(0, int(args.get("offset") or 0))
            snippet_chars = max(80, min(int(args.get("snippet_chars") or 400), 2000))
            page_start = (
                int(args["page_start"]) if args.get("page_start") is not None else None
            )
            page_end = (
                int(args["page_end"]) if args.get("page_end") is not None else None
            )
        except (TypeError, ValueError):
            return ToolResult.error_result(
                "limit, offset, snippet_chars, and page bounds must be integers",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            async with context.registry.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    WITH hits AS (
                        SELECT * FROM search_source_chunks(
                            $1::text, $2::int, $3::uuid, $4::text, $5::text,
                            $6::text, $7::text, $8::int, $9::int,
                            NULLIF($10::text, '')::timestamptz,
                            NULLIF($11::text, '')::timestamptz,
                            $12::boolean, $13::int, $14::int
                        )
                    )
                    SELECT h.*, d.source_attribution,
                           source_citation_envelope(h.document_id, h.chunk_id) AS citation
                    FROM hits h
                    JOIN source_documents d ON d.id = h.document_id
                    """,
                    args.get("query"),
                    limit,
                    document_id,
                    args.get("source_path"),
                    args.get("source_type"),
                    args.get("locator_kind"),
                    args.get("sheet_name"),
                    page_start,
                    page_end,
                    args.get("created_after"),
                    args.get("created_before"),
                    bool(args.get("exclude_sensitive") or context.is_group),
                    offset,
                    snippet_chars,
                )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)

        chunks = []
        documents = set()
        for row in rows:
            components = row["rank_components"]
            if isinstance(components, str):
                components = json.loads(components)
            locator = row["locator"]
            if isinstance(locator, str):
                locator = json.loads(locator)
            citation = row["citation"]
            source_attribution = row["source_attribution"]
            if isinstance(citation, str):
                citation = json.loads(citation)
            if isinstance(source_attribution, str):
                source_attribution = json.loads(source_attribution)
            documents.add(str(row["document_id"]))
            chunks.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["document_id"]),
                    "citation_id": citation.get("citation_id")
                    if isinstance(citation, dict)
                    else None,
                    "citation": citation,
                    "chunk_index": row["chunk_index"],
                    "title": row["title"],
                    "path": row["path"],
                    "source_type": row["source_type"],
                    "locator_kind": row["locator_kind"],
                    "locator": locator,
                    "heading_path": list(row["heading_path"] or []),
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "sheet_name": row["sheet_name"],
                    "snippet": row["snippet"],
                    "content_hash": row["content_hash"],
                    "rank": row["rank"],
                    "rank_components": components,
                    "source_attribution": source_attribution,
                    "trust_level": citation.get("trust_level")
                    if isinstance(citation, dict)
                    else None,
                }
            )
        return ToolResult.success_result(
            {"chunks": chunks, "count": len(chunks), "offset": offset, "limit": limit},
            display_output=f"Found {len(chunks)} chunk(s) across {len(documents)} document(s)",
        )


class OpenDocumentChunkHandler(ToolHandler):
    """Open exact source chunks with citation locators and scroll handles."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="open_document_chunk",
            description=(
                "Open exact passages of a preserved source document by chunk id, "
                "chunk range, or page range. Returns full chunk content with its "
                "locator (page/section/sheet row) for citation, plus "
                "prev/next_chunk_id handles for scrolling. Inspection only — use "
                "load_document_chunks to keep passages searchable on the desk."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "One chunk UUID from search_document_chunks or open_memory.source_chunks.",
                    },
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple chunk UUIDs (opened in the order given).",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Document UUID when selecting by chunk_index or page range.",
                    },
                    "chunk_start": {"type": "integer", "minimum": 0},
                    "chunk_end": {"type": "integer", "minimum": 0},
                    "page_start": {"type": "integer", "minimum": 1},
                    "page_end": {"type": "integer", "minimum": 1},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.registry or not context.registry.pool:
            return ToolResult.error_result(
                "Database unavailable", ToolErrorType.EXECUTION_FAILED
            )

        args = dict(arguments)
        raw_ids = (
            [str(args.get("chunk_id") or "").strip()]
            if str(args.get("chunk_id") or "").strip()
            else []
        )
        for item in args.get("chunk_ids") or []:
            if str(item).strip():
                raw_ids.append(str(item).strip())
        chunk_ids: list[UUID] = []
        for raw_id in raw_ids:
            try:
                chunk_ids.append(UUID(raw_id))
            except ValueError:
                return ToolResult.error_result(
                    "chunk ids must be uuids", ToolErrorType.INVALID_PARAMS
                )

        document_id: UUID | None = None
        if str(args.get("document_id") or "").strip():
            try:
                document_id = UUID(str(args["document_id"]).strip())
            except ValueError:
                return ToolResult.error_result(
                    "document_id must be a uuid", ToolErrorType.INVALID_PARAMS
                )

        if not chunk_ids and document_id is None:
            return ToolResult.error_result(
                "Provide chunk_id(s), or document_id with a chunk/page range.",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            limit = max(1, min(int(args.get("limit") or 10), 50))
            ints = {}
            for name in ("chunk_start", "chunk_end", "page_start", "page_end"):
                ints[name] = int(args[name]) if args.get(name) is not None else None
        except (TypeError, ValueError):
            return ToolResult.error_result(
                "range bounds and limit must be integers", ToolErrorType.INVALID_PARAMS
            )

        try:
            async with context.registry.pool.acquire() as conn:
                raw = await conn.fetchval(
                    """
                    SELECT enrich_source_chunk_payload(open_source_chunks(
                        $1::uuid[], $2::uuid, $3::int, $4::int, $5::int, $6::int,
                        $7::int, $8::boolean
                    ))
                    """,
                    chunk_ids or None,
                    document_id,
                    ints["chunk_start"],
                    ints["chunk_end"],
                    ints["page_start"],
                    ints["page_end"],
                    limit,
                    bool(args.get("exclude_sensitive") or context.is_group),
                )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)

        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict):
            return ToolResult.error_result(
                "open_source_chunks returned an invalid payload",
                ToolErrorType.EXECUTION_FAILED,
            )
        if payload.get("error") == "missing_selector":
            return ToolResult.error_result(
                "Provide chunk_id(s) or document_id with a range.",
                ToolErrorType.INVALID_PARAMS,
            )
        if payload.get("error") == "not_found":
            return ToolResult.error_result(
                "No matching chunks — re-run search_document_chunks; the source may have been re-ingested with new chunk ids.",
                ToolErrorType.EXECUTION_FAILED,
            )

        for chunk in payload.get("chunks") or []:
            citation_bits = [chunk.get("path") or chunk.get("title") or ""]
            if chunk.get("page_start"):
                pages = str(chunk["page_start"])
                if chunk.get("page_end") and chunk["page_end"] != chunk["page_start"]:
                    pages += f"-{chunk['page_end']}"
                citation_bits.append(f"page {pages}")
            elif chunk.get("sheet_name"):
                citation_bits.append(f"sheet {chunk['sheet_name']}")
            elif chunk.get("heading_path"):
                citation_bits.append(" > ".join(chunk["heading_path"]))
            else:
                citation_bits.append(f"chunk {chunk.get('chunk_index')}")
            chunk["citation"] = ", ".join(bit for bit in citation_bits if bit)

        count = int(payload.get("count") or 0)
        return ToolResult.success_result(
            payload, display_output=f"Opened {count} chunk(s)"
        )


class LoadDocumentChunksHandler(ToolHandler):
    """Load selected source chunks onto the RecMem desk."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="load_document_chunks",
            description=(
                "Place selected source-document chunks on the RecMem desk as "
                "searchable mid-term working material — by chunk ids, by "
                "document + query (top matching passages), or by document + "
                "page range. Give a reason; pin=true protects them from desk "
                "cleanup while actively needed. Search them later with "
                'search_history sources=["desk"].'
            ),
            parameters={
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Chunk UUIDs from search_document_chunks.",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Document UUID when loading by query or page range.",
                    },
                    "query": {
                        "type": "string",
                        "description": "With document_id: load the top matching passages for this query.",
                    },
                    "page_start": {"type": "integer", "minimum": 1},
                    "page_end": {"type": "integer", "minimum": 1},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason these passages need to stay on the desk.",
                    },
                    "pin": {
                        "type": "boolean",
                        "default": False,
                        "description": "Pin the loaded items so desk cleanup keeps them.",
                    },
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=2,
            is_read_only=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.registry or not context.registry.pool:
            return ToolResult.error_result(
                "Database unavailable", ToolErrorType.EXECUTION_FAILED
            )

        args = dict(arguments)
        chunk_ids: list[UUID] = []
        for item in args.get("chunk_ids") or []:
            if not str(item).strip():
                continue
            try:
                chunk_ids.append(UUID(str(item).strip()))
            except ValueError:
                return ToolResult.error_result(
                    "chunk_ids must contain only uuids", ToolErrorType.INVALID_PARAMS
                )

        document_id: UUID | None = None
        if str(args.get("document_id") or "").strip():
            try:
                document_id = UUID(str(args["document_id"]).strip())
            except ValueError:
                return ToolResult.error_result(
                    "document_id must be a uuid", ToolErrorType.INVALID_PARAMS
                )

        if not chunk_ids and document_id is None:
            return ToolResult.error_result(
                "Provide chunk_ids, or document_id (optionally with query or page range).",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            limit = max(1, min(int(args.get("limit") or 5), 20))
            page_start = (
                int(args["page_start"]) if args.get("page_start") is not None else None
            )
            page_end = (
                int(args["page_end"]) if args.get("page_end") is not None else None
            )
        except (TypeError, ValueError):
            return ToolResult.error_result(
                "limit and page bounds must be integers", ToolErrorType.INVALID_PARAMS
            )

        exclude_sensitive = bool(args.get("exclude_sensitive") or context.is_group)
        query = str(args.get("query") or "").strip()
        session_uuid: UUID | None = None
        if context.session_id:
            try:
                session_uuid = UUID(str(context.session_id))
            except ValueError:
                session_uuid = None

        try:
            async with context.registry.pool.acquire() as conn:
                # document_id + query: resolve the top matching passages first.
                if not chunk_ids and document_id is not None and query:
                    rows = await conn.fetch(
                        """
                        SELECT chunk_id FROM search_source_chunks(
                            $1::text, $2::int, $3::uuid, NULL, NULL, NULL, NULL,
                            $4::int, $5::int, NULL, NULL, $6::boolean
                        )
                        """,
                        query,
                        limit,
                        document_id,
                        page_start,
                        page_end,
                        exclude_sensitive,
                    )
                    chunk_ids = [row["chunk_id"] for row in rows]
                    if not chunk_ids:
                        return ToolResult.error_result(
                            "No passages matched that query in this document — try search_document_chunks with different wording.",
                            ToolErrorType.EXECUTION_FAILED,
                        )
                    document_id = None  # ids are now the selector

                raw = await conn.fetchval(
                    """
                    SELECT load_source_chunks_to_recmem(
                        $1::uuid[], $2::uuid, NULL, NULL, $3::int, $4::int,
                        $5::int, $6::boolean, $7::text, $8::uuid, $9::text, NULL, $10::boolean
                    )
                    """,
                    chunk_ids or None,
                    document_id,
                    page_start,
                    page_end,
                    limit,
                    exclude_sensitive,
                    args.get("reason"),
                    session_uuid,
                    context.tool_context.value if context.tool_context else None,
                    bool(args.get("pin")),
                )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)

        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict):
            return ToolResult.error_result(
                "load_source_chunks_to_recmem returned an invalid payload",
                ToolErrorType.EXECUTION_FAILED,
            )
        if payload.get("error") == "missing_selector":
            return ToolResult.error_result(
                "Provide chunk_ids or document_id.", ToolErrorType.INVALID_PARAMS
            )

        count = int(payload.get("count") or 0)
        return ToolResult.success_result(
            payload,
            display_output=(
                f"Loaded {count} chunk(s) onto the desk — search them with "
                'search_history sources=["desk"], list with list_desk'
            ),
        )


class SenseMemoryAvailabilityHandler(ToolHandler):
    """Quick feeling-of-knowing check before full recall."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="sense_memory_availability",
            description=(
                "Sense whether you likely have relevant memories before doing a full recall. "
                "Use this for a quick feeling-of-knowing check."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic to check memory availability for.",
                    },
                },
                "required": ["query"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,  # Free - lightweight check
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        db_result = await _try_db_memory_tool(
            "sense_memory_availability", arguments, context
        )
        if db_result is not None:
            return db_result
        # execute_memory_tool (db/38) owns this tool; the former Python
        # compatibility path was deleted.
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class ExploreConceptHandler(ToolHandler):
    """Explore memories connected to a concept."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="explore_concept",
            internal=True,  # folded into `associate` (#99); one-release alias
            description=(
                "Explore memories connected to a specific concept. Shows how different "
                "memories relate to an idea and what other concepts are connected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "The concept to explore.",
                    },
                    "include_related": {
                        "type": "boolean",
                        "description": "Also return memories linked to related concepts.",
                        "default": True,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum memories to return.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["concept"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        db_result = await _try_db_memory_tool("explore_concept", arguments, context)
        if db_result is not None:
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class AssociateHandler(ToolHandler):
    """Free association over the memory graph (agent-facing name: associate)."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="associate",
            description=(
                "Follow what something reminds you of: free association through "
                "your memory's own connections. Start from a question or from "
                "specific memories and let the associations unfold — what "
                "supports what, what contradicts, what led to what, what an "
                "idea connects to. Use this when the question is about how "
                "things relate rather than what a single memory says."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Recall seed memories matching this text (used if 'seeds' omitted).",
                    },
                    "seeds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Explicit seed memory ids (uuid). Use if you already have them.",
                    },
                    "rel_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict expansion to these edge types (e.g. ['CAUSES','SUPPORTS']). Omit for all.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Max hops from a seed.",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 4,
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Max nodes in the result.",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        db_result = await _try_db_memory_tool("explore_subgraph", arguments, context)
        if db_result is not None:
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class GetProceduresHandler(ToolHandler):
    """Retrieve procedural memories for a task."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_procedures",
            description=(
                "Retrieve procedural memories (how-to knowledge) for a specific task. "
                "Returns step-by-step instructions and prerequisites."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task you want to know how to do.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum procedures to return.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["task"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        db_result = await _try_db_memory_tool("get_procedures", arguments, context)
        if db_result is not None:
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class GetStrategiesHandler(ToolHandler):
    """Retrieve strategic memories for a situation."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_strategies",
            description=(
                "Retrieve strategic memories (patterns, heuristics, lessons learned) "
                "applicable to a situation. These are meta-level insights about what works."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "situation": {
                        "type": "string",
                        "description": "The situation you need strategic guidance for.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Alias for situation, accepted for compatibility with memory-search wording.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum strategies to return.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        db_result = await _try_db_memory_tool("get_strategies", arguments, context)
        if db_result is not None:
            return db_result
        return ToolResult.error_result(
            "execute_memory_tool dispatch failed (database unavailable or errored)",
            ToolErrorType.EXECUTION_FAILED,
        )


class CreateGoalHandler(ToolHandler):
    """Create a new goal for the agent."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="create_goal",
            description=(
                "Create a new goal for the agent to pursue. Use this for reminders, "
                "TODOs, or longer-term objectives."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short goal title.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional longer description.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["active", "queued", "backburner"],
                        "default": "queued",
                        "description": "Desired priority.",
                    },
                    "source": {
                        "type": "string",
                        "enum": [
                            "curiosity",
                            "user_request",
                            "identity",
                            "derived",
                            "external",
                        ],
                        "default": "user_request",
                        "description": "Why this goal exists.",
                    },
                },
                "required": ["title"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        title = arguments["title"]
        description = arguments.get("description")
        priority = arguments.get("priority", "queued")
        source = arguments.get("source", "user_request")

        try:
            async with context.registry.pool.acquire() as conn:
                goal_id = await conn.fetchval(
                    """
                    SELECT create_goal(
                        p_title := $1,
                        p_description := $2,
                        p_priority := $3,
                        p_source := $4,
                        p_origin := $5
                    )
                    """,
                    title,
                    description,
                    priority,
                    source,
                    context.trusted_goal_origin,
                )

            return ToolResult.success_result(
                output={
                    "goal_id": str(goal_id),
                    "title": title,
                    "priority": priority,
                    "origin": context.trusted_goal_origin,
                },
                display_output=f"Created goal: {title}",
            )

        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


class ScheduleTaskHandler(ToolHandler):
    """Create a scheduled (cron-like) task."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="schedule_task",
            description=(
                "Create a scheduled task (cron-like). Use for recurring reminders or timed actions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short task name."},
                    "description": {
                        "type": "string",
                        "description": "Optional longer description.",
                    },
                    "schedule_kind": {
                        "type": "string",
                        "enum": ["once", "interval", "daily", "weekly"],
                        "description": "Schedule type.",
                    },
                    "schedule": {
                        "type": "object",
                        "description": "Schedule details for the selected type.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name (e.g., America/Los_Angeles).",
                    },
                    "action_kind": {
                        "type": "string",
                        "enum": ["queue_user_message", "create_goal"],
                        "description": "Action to perform when the schedule fires.",
                    },
                    "action_payload": {
                        "type": "object",
                        "description": "Action payload.",
                    },
                    "max_runs": {
                        "type": "integer",
                        "description": "Optional max number of runs before auto-disable.",
                    },
                },
                "required": [
                    "name",
                    "schedule_kind",
                    "schedule",
                    "action_kind",
                    "action_payload",
                ],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        import json

        name = arguments["name"]
        schedule_kind = arguments["schedule_kind"]
        schedule = arguments.get("schedule") or {}
        action_kind = arguments["action_kind"]
        action_payload = arguments.get("action_payload") or {}
        timezone = arguments.get("timezone")
        description = arguments.get("description")
        max_runs = arguments.get("max_runs")

        try:
            async with context.registry.pool.acquire() as conn:
                task_id = await conn.fetchval(
                    """
                    SELECT create_scheduled_task(
                        $1,
                        $2,
                        $3::jsonb,
                        $4,
                        $5::jsonb,
                        $6,
                        $7,
                        'active',
                        $8,
                        'agent'
                    )
                    """,
                    name,
                    schedule_kind,
                    json.dumps(schedule),
                    action_kind,
                    json.dumps(action_payload),
                    timezone,
                    description,
                    max_runs,
                )

            return ToolResult.success_result(
                output={"task_id": str(task_id), "name": name},
                display_output=f"Scheduled task: {name}",
            )

        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


class ListScheduledTasksHandler(ToolHandler):
    """List scheduled tasks."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_scheduled_tasks",
            description="List scheduled tasks with optional filters.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Optional status filter",
                    },
                    "due_before": {
                        "type": "string",
                        "description": "Optional ISO8601 cutoff",
                    },
                    "limit": {"type": "integer", "default": 50},
                },
                "required": [],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        status = arguments.get("status")
        due_before = arguments.get("due_before")
        limit = int(arguments.get("limit", 50))

        try:
            async with context.registry.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM list_scheduled_tasks($1, $2::timestamptz, $3)",
                    status,
                    due_before,
                    limit,
                )
            tasks = [dict(row) for row in rows]
            return ToolResult.success_result(
                output={"tasks": tasks, "count": len(tasks)},
                display_output=f"Found {len(tasks)} scheduled task(s)",
            )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


class UpdateScheduledTaskHandler(ToolHandler):
    """Update a scheduled task."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="update_scheduled_task",
            description="Update a scheduled task (schedule/action/status).",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "schedule_kind": {"type": "string"},
                    "schedule": {"type": "object"},
                    "timezone": {"type": "string"},
                    "action_kind": {"type": "string"},
                    "action_payload": {"type": "object"},
                    "status": {"type": "string"},
                    "max_runs": {"type": "integer"},
                },
                "required": ["task_id"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        import json

        task_id = arguments["task_id"]
        try:
            async with context.registry.pool.acquire() as conn:
                updated = await conn.fetchval(
                    """
                    SELECT update_scheduled_task(
                        $1::uuid,
                        $2,
                        $3,
                        $4,
                        $5::jsonb,
                        $6,
                        $7,
                        $8::jsonb,
                        $9,
                        $10
                    )
                    """,
                    task_id,
                    arguments.get("name"),
                    arguments.get("description"),
                    arguments.get("schedule_kind"),
                    (
                        json.dumps(arguments.get("schedule"))
                        if arguments.get("schedule") is not None
                        else None
                    ),
                    arguments.get("timezone"),
                    arguments.get("action_kind"),
                    (
                        json.dumps(arguments.get("action_payload"))
                        if arguments.get("action_payload") is not None
                        else None
                    ),
                    arguments.get("status"),
                    arguments.get("max_runs"),
                )

            return ToolResult.success_result(
                output={"task": updated},
                display_output=f"Updated task {task_id}",
            )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


class DeleteScheduledTaskHandler(ToolHandler):
    """Disable or delete a scheduled task."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="delete_scheduled_task",
            description="Disable or delete a scheduled task.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "hard_delete": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["task_id"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        task_id = arguments["task_id"]
        hard_delete = bool(arguments.get("hard_delete", False))
        reason = arguments.get("reason")
        try:
            async with context.registry.pool.acquire() as conn:
                ok = await conn.fetchval(
                    "SELECT delete_scheduled_task($1::uuid, $2::boolean, $3)",
                    task_id,
                    hard_delete,
                    reason,
                )
            return ToolResult.success_result(
                output={"deleted": bool(ok), "task_id": task_id},
                display_output=(
                    f"Deleted task {task_id}"
                    if hard_delete
                    else f"Disabled task {task_id}"
                ),
            )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


class PonderHandler(ToolHandler):
    """Let a question simmer: file a background memory search (#98)."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="request_background_search",
            description=(
                "Let a question simmer in the back of your mind. Files a "
                "background memory search; if the subconscious finds it "
                "later, the answer rises as spontaneous recall — and "
                "reaches the user as an 'it came back to me' note when it "
                "resolves strongly. Use when something feels familiar but "
                "will not surface right now."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you are trying to remember.",
                    },
                },
                "required": ["query"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult.error_result(
                "query is required", ToolErrorType.INVALID_PARAMS
            )
        try:
            async with context.registry.pool.acquire() as conn:
                activation_id = await conn.fetchval(
                    "SELECT request_background_search($1::text)", query
                )
            return ToolResult.success_result(
                {"activation_id": str(activation_id), "query": query},
                display_output=f"Letting it simmer: {query[:80]}",
            )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


class QueueUserMessageHandler(ToolHandler):
    """Queue a message for the user."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="queue_user_message",
            description=(
                "Queue a user-facing note. In chat, this delivers immediately "
                "to the dashboard inbox; in heartbeat, it queues for the normal outbox relay."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message body for the user.",
                    },
                    "intent": {
                        "type": "string",
                        "description": "Optional intent/category (e.g. 'reminder', 'status', 'question').",
                    },
                },
                "required": ["message"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,  # Free - just queuing
            is_read_only=False,
            outbound=OutboundSpec(
                recipient_arg=None,
                body_arg="message",
                channel="outbox",
                fixed_recipient="primary_user",
                primary_recipient=True,
            ),
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        message = arguments["message"]
        intent = arguments.get("intent")

        try:
            async with context.registry.pool.acquire() as conn:
                if context.tool_context == ToolContext.CHAT:
                    raw = await conn.fetchval(
                        "SELECT queue_web_inbox_message($1::text, $2::text, 'tool')",
                        message,
                        intent,
                    )
                    result = json.loads(raw) if isinstance(raw, str) else (raw or {})
                else:
                    outbox_id = await conn.fetchval(
                        "SELECT queue_outbox_message($1::text, $2::text, 'tool')",
                        message,
                        intent,
                    )
                    result = {
                        "queued": True,
                        "delivered": False,
                        "outbox_id": str(outbox_id),
                        "delivery": {"mode": "outbox"},
                    }

            if context.tool_context == ToolContext.CHAT and result.get("delivered"):
                display = f"Delivered to inbox: {message[:50]}..."
            else:
                display = f"Queued message: {message[:50]}..."

            return ToolResult.success_result(
                output={**result, "message": message[:50]},
                display_output=display,
            )

        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


class ExploreSubgraphAliasHandler(AssociateHandler):
    """One-release alias for the old name (internal: hidden, unbound)."""

    @property
    def spec(self) -> ToolSpec:
        spec = super().spec
        spec.name = "explore_subgraph"
        spec.internal = True
        return spec


class TraceWhyHandler(ToolHandler):
    """Introspective causation: why do I think/feel this?"""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="trace_why",
            description=(
                "Ask yourself why: trace where a memory or belief came from — "
                "the chain of causes, evidence, and prior experiences behind "
                "it. Give the memory id (from recall or open_memory) and get "
                "its ancestry, nearest first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The memory/belief to trace (uuid).",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How far back to trace.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 6,
                    },
                },
                "required": ["memory_id"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        try:
            memory_id = UUID(str(arguments["memory_id"]))
        except (ValueError, KeyError):
            return ToolResult.error_result(
                "memory_id must be a uuid", ToolErrorType.INVALID_PARAMS
            )
        depth = max(1, min(int(arguments.get("depth") or 3), 6))
        try:
            async with context.registry.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT cause_id, cause_content, relationship, distance "
                    "FROM find_causal_chain($1::uuid, $2)",
                    memory_id,
                    depth,
                )
            causes = [
                {
                    "memory_id": str(r["cause_id"]),
                    "content": r["cause_content"],
                    "relationship": r["relationship"],
                    "distance": r["distance"],
                }
                for r in rows
            ]
            display = (
                f"{len(causes)} step(s) in the chain behind {str(memory_id)[:8]}"
                if causes
                else "No recorded causes behind this memory — it may be a root experience."
            )
            return ToolResult.success_result(
                {"causes": causes, "count": len(causes)}, display
            )
        except Exception as e:
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


def create_memory_tools() -> list[ToolHandler]:
    """Create memory tool handlers.

    Note: Goal management is handled by core/tools/goals.py (manage_goals).
    Schedule management is handled by core/tools/cron.py (manage_schedule).
    Those unified tools replace the individual create_goal, schedule_task, etc.
    """
    return [
        RecallHandler(),
        RecallAtTimeHandler(),
        DiffMemoryHistoryHandler(),
        SearchHistoryHandler(),
        RememberHandler(),
        AddEvidenceHandler(),
        BeliefHistoryHandler(),
        OpenMemoryHandler(),
        SearchDocumentsHandler(),
        OpenDocumentHandler(),
        OpenDocumentsHandler(),
        LoadDocumentsHandler(),
        SearchDocumentChunksHandler(),
        OpenDocumentChunkHandler(),
        LoadDocumentChunksHandler(),
        SenseMemoryAvailabilityHandler(),
        ExploreConceptHandler(),
        AssociateHandler(),
        GetProceduresHandler(),
        GetStrategiesHandler(),
        QueueUserMessageHandler(),
        PonderHandler(),
        TraceWhyHandler(),
        ExploreSubgraphAliasHandler(),
    ]
