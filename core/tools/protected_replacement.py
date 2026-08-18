"""Agent-facing tools for the HMX Protected Section Replacement Protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.memory_exchange import HmxPolicyError
from core.protected_replacement import (
    ACKNOWLEDGEMENT_DECISIONS,
    acknowledge_protected_replacement,
    inspect_protected_replacement,
    list_protected_replacement_audit,
    open_protected_reversion_windows,
    pending_protected_replacements,
    revert_protected_replacement,
)

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

_AGENT_CONTEXTS = {ToolContext.CHAT, ToolContext.HEARTBEAT}


def _pool(context: ToolExecutionContext):
    return context.registry.pool if context.registry else None


def _missing_pool() -> ToolResult:
    return ToolResult.error_result(
        "Protected replacement tools require an active Hexis database connection. "
        "Retry from a running Hexis chat or heartbeat.",
        ToolErrorType.MISSING_CONFIG,
    )


def _error_result(exc: Exception) -> ToolResult:
    error_type = (
        ToolErrorType.BOUNDARY_VIOLATION
        if isinstance(exc, HmxPolicyError)
        else ToolErrorType.EXECUTION_FAILED
    )
    return ToolResult.error_result(str(exc), error_type)


def _timestamp(value: Any, name: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HmxPolicyError(f"{name} must be an ISO 8601 date or timestamp") from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


class ProtectedReplacementListHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="protected_replacement_list",
            description=(
                "List protected-state replacement requests still awaiting an "
                "accept, refuse, modification, or defer decision."
            ),
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await pending_protected_replacements(conn)
            return ToolResult.success_result(
                result,
                f"Found {result.get('total', 0)} pending protected replacements",
            )
        except Exception as exc:
            return _error_result(exc)


class ProtectedReplacementInspectHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="protected_replacement_inspect",
            description=(
                "Inspect current/imported protected state, execution audit, and any "
                "open reversion window for one replacement."
            ),
            parameters={
                "type": "object",
                "properties": {"replacement_id": {"type": "string"}},
                "required": ["replacement_id"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await inspect_protected_replacement(
                    conn, str(arguments["replacement_id"])
                )
            return ToolResult.success_result(
                result,
                f"Protected replacement {result['replacement_id']} inspected",
            )
        except Exception as exc:
            return _error_result(exc)


class ProtectedReplacementReviewHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="protected_replacement_review",
            description=(
                "Decide one pending protected-state replacement. Accept atomically "
                "snapshots, audits, replaces, and verifies the section; other choices "
                "leave protected state unchanged."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "replacement_id": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": list(ACKNOWLEDGEMENT_DECISIONS),
                    },
                    "rationale": {"type": "string"},
                    "proposed_changes": {"type": "object"},
                },
                "required": ["replacement_id", "decision"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            requires_approval=False,
            is_read_only=False,
            supports_parallel=False,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await acknowledge_protected_replacement(
                    conn,
                    str(arguments["replacement_id"]),
                    decision=str(arguments["decision"]),
                    rationale=arguments.get("rationale"),
                    proposed_changes=arguments.get("proposed_changes"),
                    executor="agent_tool",
                )
            return ToolResult.success_result(
                result, f"Protected replacement {result['status']}"
            )
        except Exception as exc:
            return _error_result(exc)


class ProtectedReplacementAuditListHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="protected_replacement_audit_list",
            description=(
                "List immutable local protected replacement, verification, and "
                "reversion audit records in a bounded time range."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "since": {"type": "string"},
                    "until": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "default": 100,
                    },
                },
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            since = _timestamp(arguments.get("since"), "since")
            until = _timestamp(arguments.get("until"), "until")
            async with pool.acquire() as conn:
                result = await list_protected_replacement_audit(
                    conn,
                    since=since,
                    until=until,
                    limit=int(arguments.get("limit", 100)),
                )
            return ToolResult.success_result(
                result,
                f"Found {result['total']} local protected replacement audit records",
            )
        except Exception as exc:
            return _error_result(exc)


class ProtectedReversionListHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="protected_reversion_list",
            description=(
                "List executed protected replacements whose bounded, one-shot "
                "reversion windows are still open."
            ),
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await open_protected_reversion_windows(conn)
            return ToolResult.success_result(
                result,
                f"Found {result.get('total', 0)} open protected reversion windows",
            )
        except Exception as exc:
            return _error_result(exc)


class ProtectedReplacementRevertHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="protected_replacement_revert",
            description=(
                "Revert one executed protected replacement within its open window. "
                "Requires the replacement audit ID and an explicit rationale; refuses "
                "to overwrite protected state changed after the replacement."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "audit_id": {"type": "string"},
                    "rationale": {"type": "string", "minLength": 1},
                },
                "required": ["audit_id", "rationale"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            requires_approval=False,
            is_read_only=False,
            supports_parallel=False,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await revert_protected_replacement(
                    conn,
                    str(arguments["audit_id"]),
                    rationale=str(arguments["rationale"]),
                    actor_identity="agent_tool",
                )
            return ToolResult.success_result(
                result, f"Protected replacement {result['status']}"
            )
        except Exception as exc:
            return _error_result(exc)


def create_protected_replacement_tools() -> list[ToolHandler]:
    return [
        ProtectedReplacementListHandler(),
        ProtectedReplacementInspectHandler(),
        ProtectedReplacementReviewHandler(),
        ProtectedReplacementAuditListHandler(),
        ProtectedReversionListHandler(),
        ProtectedReplacementRevertHandler(),
    ]
