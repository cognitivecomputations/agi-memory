"""
Hexis Tools System - Multi-Agent Council

Provides tools for multi-perspective analysis through council personas,
orchestrated deliberation, and signal aggregation from system events.

F.1 - Agent Personas/Roles (prompt_modules council.persona.*)
F.2 - Council Orchestration Tool (RunCouncilHandler)
F.3 - Signal Aggregation Tool (AggregateSignalsHandler)
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, TYPE_CHECKING

from .base import (
    ToolCategory,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# F.1 -- Council personas (DB-owned: prompt_modules council.persona.*)
# ---------------------------------------------------------------------------


async def load_council_personas(
    context: "ToolExecutionContext",
) -> dict[str, dict[str, str]]:
    """Fetch the persona catalog from get_council_personas() (db/33)."""
    pool = context.registry.pool if context.registry else None
    if pool is None:
        raise RuntimeError("Council personas require a database-backed registry")
    async with pool.acquire() as conn:
        raw = await conn.fetchval("SELECT get_council_personas()")
    personas = json.loads(raw) if isinstance(raw, str) else (raw or {})
    if not personas:
        raise RuntimeError(
            "No council personas are seeded (prompt_modules council.persona.*)"
        )
    return personas


# ---------------------------------------------------------------------------
# F.1 -- List Council Personas
# ---------------------------------------------------------------------------


class ListCouncilPersonasHandler(ToolHandler):
    """List the available council personas and their roles."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_council_personas",
            description=(
                "List the available multi-agent council personas. "
                "Each persona offers a distinct analytical perspective "
                "for structured deliberation."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
            requires_approval=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        try:
            personas_summary = await load_council_personas(context)
        except Exception as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)

        return ToolResult(
            success=True,
            output={
                "count": len(personas_summary),
                "personas": personas_summary,
            },
            energy_spent=0,
        )


# ---------------------------------------------------------------------------
# F.2 -- Run Council
# ---------------------------------------------------------------------------


class RunCouncilHandler(ToolHandler):
    """Orchestrate a multi-perspective council analysis on a topic.

    Prepares a council configuration where each selected persona provides
    their analytical lens on the given topic. The main agent can then use
    these structured perspectives to make well-rounded decisions.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="run_council",
            description=(
                "Run a multi-agent council deliberation on a topic. "
                "Spawns analysis from multiple persona perspectives "
                "(growth strategist, revenue guardian, skeptical operator, "
                "creative innovator, customer advocate), then runs a "
                "moderator synthesis pass."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The question or topic for the council to discuss.",
                    },
                    "personas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Which personas to include (keys from list_council_personas). "
                            "Defaults to all 5."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context or data for the council.",
                    },
                    "signal_limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of compacted signals to include. "
                            "Defaults to the live deliberation configuration."
                        ),
                    },
                    "stakes": {
                        "type": "string",
                        "enum": ["routine", "material", "high"],
                        "description": (
                            "How consequential the decision is. This labels the "
                            "record and does not authorize or gate any action."
                        ),
                    },
                },
                "required": ["topic"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=5,
            is_read_only=False,
            supports_parallel=False,
            optional=True,
            requires_approval=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        topic = arguments.get("topic", "").strip()
        if not topic:
            return ToolResult.error_result(
                "Parameter 'topic' is required.",
                ToolErrorType.INVALID_PARAMS,
            )

        requested_personas: list[str] | None = arguments.get("personas")
        extra_context: str = arguments.get("context", "")
        stakes = str(arguments.get("stakes") or "material").strip().lower()
        if stakes not in {"routine", "material", "high"}:
            return ToolResult.error_result(
                "Parameter 'stakes' must be routine, material, or high.",
                ToolErrorType.INVALID_PARAMS,
            )

        pool: asyncpg.Pool | None = context.registry.pool if context.registry else None
        if pool is None:
            return ToolResult.error_result(
                "Council deliberation requires a database-backed registry.",
                ToolErrorType.MISSING_CONFIG,
            )

        from services.deliberation import (
            DeliberationUnavailable,
            load_deliberation_config,
            run_adversarial_deliberation,
        )

        try:
            policy = await load_deliberation_config(pool)
        except DeliberationUnavailable as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.MISSING_CONFIG)
        configured_signal_limit = int(policy["signal_limit"])
        raw_signal_limit = arguments.get("signal_limit")
        signal_limit = (
            configured_signal_limit
            if raw_signal_limit is None
            else max(0, min(int(raw_signal_limit), 30))
        )

        # Resolve persona keys against the DB catalog
        try:
            personas = await load_council_personas(context)
        except Exception as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)
        if requested_personas:
            invalid = [p for p in requested_personas if p not in personas]
            if invalid:
                return ToolResult.error_result(
                    f"Unknown persona(s): {', '.join(invalid)}. "
                    f"Valid keys: {', '.join(sorted(personas.keys()))}",
                    ToolErrorType.INVALID_PARAMS,
                )
            max_personas = int(policy["max_personas"])
            if len(requested_personas) > max_personas:
                return ToolResult.error_result(
                    "Requested "
                    f"{len(requested_personas)} personas, but the live maximum is "
                    f"{max_personas}. Choose a smaller council.",
                    ToolErrorType.INVALID_PARAMS,
                )
            selected_keys = requested_personas
        else:
            selected_keys = list(personas.keys())[: int(policy["max_personas"])]

        signals, evidence_memory_ids, collection_warning = await self._collect_signals(
            context, limit=signal_limit
        )
        try:
            output = await run_adversarial_deliberation(
                pool,
                topic=topic,
                personas=personas,
                selected_keys=selected_keys,
                extra_context=extra_context,
                signals=signals,
                stakes=stakes,
                source_context=context.tool_context.value,
                source_session_id=context.session_id,
                heartbeat_id=context.heartbeat_id,
                call_id=context.call_id,
                evidence_memory_ids=evidence_memory_ids,
                collection_warnings=(
                    [collection_warning] if collection_warning else None
                ),
            )
        except ValueError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)
        except DeliberationUnavailable as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.MISSING_CONFIG)
        except Exception as exc:
            logger.exception("Council deliberation failed")
            return ToolResult.error_result(
                "Council deliberation failed. The partial record was preserved as "
                f"failed so it can be inspected. Cause: {exc}",
                ToolErrorType.EXECUTION_FAILED,
            )

        return ToolResult(success=True, output=output, energy_spent=5)

    async def _collect_signals(
        self,
        context: ToolExecutionContext,
        *,
        limit: int,
    ) -> tuple[list[str], list[str], str | None]:
        pool: asyncpg.Pool | None = context.registry.pool if context.registry else None
        if not pool or limit <= 0:
            return [], [], None

        entries: list[tuple[str, str | None]] = []
        category_limit = max(1, (limit + 2) // 3)
        try:
            async with pool.acquire() as conn:
                event_rows = await conn.fetch(
                    """
                    SELECT source::text, payload
                    FROM gateway_events
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    category_limit,
                )
                for row in event_rows:
                    src = row["source"]
                    payload = row["payload"]
                    if isinstance(payload, str):
                        try:
                            payload = json.loads(payload)
                        except Exception:
                            payload = {}
                    if isinstance(payload, dict):
                        keys = ", ".join(sorted(payload.keys())[:4])
                        entries.append(
                            (f"Event[{src}]: payload keys ({keys or 'none'})", None)
                        )
                    else:
                        entries.append((f"Event[{src}]", None))

                mem_rows = await conn.fetch(
                    """
                    SELECT id, content
                    FROM memories
                    WHERE type = 'episodic' AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    category_limit,
                )
                for row in mem_rows:
                    content = (row["content"] or "").strip().replace("\n", " ")
                    if content:
                        entries.append((f"Memory: {content[:180]}", str(row["id"])))

                goal_rows = await conn.fetch(
                    """
                    SELECT id, content
                    FROM memories
                    WHERE type = 'goal' AND status = 'active'
                    ORDER BY importance DESC NULLS LAST, created_at DESC
                    LIMIT $1
                    """,
                    category_limit,
                )
                for row in goal_rows:
                    content = (row["content"] or "").strip().replace("\n", " ")
                    if content:
                        entries.append((f"Goal: {content[:180]}", str(row["id"])))
        except Exception as exc:
            cause = " ".join(str(exc).split())[:300] or type(exc).__name__
            warning = (
                "Recent evidence signals could not be collected; the council ran "
                f"without that context. Cause: {cause}"
            )
            logger.warning("Council signal collection failed: %s", cause)
            return [], [], warning

        included = entries[:limit]
        evidence_ids = list(
            dict.fromkeys(memory_id for _, memory_id in included if memory_id)
        )
        return [text for text, _ in included], evidence_ids, None


# ---------------------------------------------------------------------------
# F.2a -- Durable deliberation review
# ---------------------------------------------------------------------------


class ListDeliberationsHandler(ToolHandler):
    """List recent advisory council runs and their lifecycle status."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_deliberations",
            description=(
                "List recent durable council deliberations, including completed "
                "and failed runs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return (default 20, max 100).",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["running", "completed", "failed"],
                        "description": "Optional lifecycle status filter.",
                    },
                },
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
            requires_approval=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        pool = context.registry.pool if context.registry else None
        if pool is None:
            return ToolResult.error_result(
                "Deliberation history requires a database-backed registry.",
                ToolErrorType.MISSING_CONFIG,
            )
        status = str(arguments.get("status") or "").strip().lower() or None
        if status not in {None, "running", "completed", "failed"}:
            return ToolResult.error_result(
                "Parameter 'status' must be running, completed, or failed.",
                ToolErrorType.INVALID_PARAMS,
            )
        limit = max(1, min(int(arguments.get("limit", 20) or 20), 100))
        try:
            from services.deliberation import list_deliberations

            payload = await list_deliberations(pool, limit=limit, status=status)
            return ToolResult.success_result(payload)
        except Exception as exc:
            return ToolResult.error_result(
                f"Could not list deliberations: {exc}",
                ToolErrorType.EXECUTION_FAILED,
            )


class InspectDeliberationHandler(ToolHandler):
    """Inspect one council run with all perspectives, challenges, and verdict."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="inspect_deliberation",
            description=(
                "Inspect one durable council deliberation, including its input, "
                "perspectives, challenges, dissent, and invalidation conditions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "deliberation_id": {
                        "type": "string",
                        "description": "UUID returned by run_council.",
                    }
                },
                "required": ["deliberation_id"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
            requires_approval=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        try:
            deliberation_id = str(
                uuid.UUID(str(arguments.get("deliberation_id") or ""))
            )
        except (ValueError, TypeError, AttributeError):
            return ToolResult.error_result(
                "Parameter 'deliberation_id' must be a UUID returned by run_council.",
                ToolErrorType.INVALID_PARAMS,
            )
        pool = context.registry.pool if context.registry else None
        if pool is None:
            return ToolResult.error_result(
                "Deliberation history requires a database-backed registry.",
                ToolErrorType.MISSING_CONFIG,
            )
        try:
            from services.deliberation import inspect_deliberation

            payload = await inspect_deliberation(pool, deliberation_id)
            if not payload.get("found"):
                return ToolResult.error_result(
                    f"Deliberation {deliberation_id} was not found.",
                    ToolErrorType.FILE_NOT_FOUND,
                )
            return ToolResult.success_result(payload)
        except Exception as exc:
            return ToolResult.error_result(
                f"Could not inspect deliberation: {exc}",
                ToolErrorType.EXECUTION_FAILED,
            )


# ---------------------------------------------------------------------------
# F.3 -- Aggregate Signals
# ---------------------------------------------------------------------------


class AggregateSignalsHandler(ToolHandler):
    """Aggregate recent signals from events, memories, and goals.

    Provides a consolidated 'state of affairs' snapshot combining
    gateway events, episodic memories, and active goals.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="aggregate_signals",
            description=(
                "Aggregate recent signals across gateway events, episodic "
                "memories, and active goals into a consolidated snapshot. "
                "Useful for situational awareness before council deliberation "
                "or autonomous decision-making."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": (
                            "Filter signals by domain/source "
                            "(e.g. 'email', 'calendar', 'cron', 'chat')."
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "description": "How far back to look (default 7 days).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max signals per category (default 20).",
                    },
                },
            },
            category=ToolCategory.MEMORY,
            energy_cost=3,
            is_read_only=True,
            requires_approval=False,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        pool: asyncpg.Pool | None = context.registry.pool if context.registry else None
        if not pool:
            return ToolResult.error_result(
                "Database pool not available.",
                ToolErrorType.MISSING_CONFIG,
            )
        try:
            async with pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT aggregate_signals_tool($1::jsonb)", json.dumps(arguments)
                )
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(payload, dict) and "success" in payload:
                if payload.get("success"):
                    return ToolResult.success_result(
                        payload.get("output"),
                        display_output=payload.get("display_output"),
                    )
                return ToolResult.error_result(
                    payload.get("error") or "Signal aggregation failed",
                    ToolErrorType.EXECUTION_FAILED,
                )
            return ToolResult.error_result(
                "Signal aggregation failed: unexpected payload",
                ToolErrorType.EXECUTION_FAILED,
            )
        except Exception as exc:
            logger.exception("Signal aggregation failed")
            return ToolResult.error_result(
                f"Signal aggregation failed: {exc}", ToolErrorType.EXECUTION_FAILED
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_council_tools() -> list[ToolHandler]:
    """Create the multi-agent council tools."""
    return [
        ListCouncilPersonasHandler(),
        RunCouncilHandler(),
        ListDeliberationsHandler(),
        InspectDeliberationHandler(),
        AggregateSignalsHandler(),
    ]
