"""Continuously measure the tool surface available to each worker.

The registry says which handlers exist in this process, tool configuration says
which are enabled, and skills say which can ever reach the model.  A capability
row is healthy only when all three agree.  The probe never invokes a tool: many
tools have real-world side effects, so reachability is proved by resolving the
same registration/configuration/skill path used by the agent loop.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.tools.base import ToolContext

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 15


@dataclass(frozen=True)
class CapabilityResult:
    """One observed worker/context/tool reachability result."""

    worker_name: str
    tool_name: str
    tool_context: str
    available: bool
    reason_code: str | None = None
    reason_if_missing: str | None = None
    registry_kind: str = "default"
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["checked_at"] = self.checked_at.isoformat()
        return record


def _matches_bound_tool(tool_name: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(tool_name, pattern) if "*" in pattern else tool_name == pattern


def _reachable_tool_names(registry: Any, context: ToolContext) -> set[str]:
    """Return every registered tool that can be surfaced in this context."""
    from services.skill_runtime import (
        ALWAYS_AVAILABLE_TOOL_NAMES,
        DISCOVERY_TOOL_NAMES,
        load_available_skills,
        skill_bound_tools,
    )

    registered = set(registry.list_names())
    # The MCP server publishes its registry directly; it does not run the
    # conversational skill selector. Registration + config are its surface.
    if context == ToolContext.MCP:
        return registered
    reachable = set(DISCOVERY_TOOL_NAMES) & registered
    if context != ToolContext.HEARTBEAT:
        reachable.update(ALWAYS_AVAILABLE_TOOL_NAMES & registered)

    for skill in load_available_skills(registry, context):
        for pattern in skill_bound_tools(skill):
            reachable.update(
                name for name in registered if _matches_bound_tool(name, pattern)
            )
    return reachable


async def _catalog_tool_names(pool: Any) -> set[str]:
    """Include DB-catalogued tools so stale catalog entries become visible."""
    if pool is None:
        return set()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT name FROM tool_definitions")
        return {str(row["name"]) for row in rows}
    except Exception:
        logger.debug("Capability probe could not read the DB tool catalog", exc_info=True)
        return set()


def _classify_tool(
    registry: Any,
    config: Any,
    context: ToolContext,
    tool_name: str,
    reachable: set[str],
) -> tuple[bool, str | None, str | None]:
    handler = registry.get(tool_name)
    if handler is None:
        return (
            False,
            "handler_not_registered",
            "catalogued in the database but no handler is registered in this worker",
        )

    spec = handler.spec
    if context not in spec.allowed_contexts:
        return False, "context_denied", f"not allowed in {context.value} context"
    if not config.is_tool_enabled_for_context(spec.name, spec.category, context):
        return False, "config_disabled", f"disabled for {context.value} by tools config"
    if spec.optional and not config.is_optional_allowed(spec.name, spec.category):
        return False, "optional_not_enabled", "optional tool is not allowlisted"
    if spec.internal:
        return False, "internal_only", "internal runtime tool; intentionally not model-facing"
    if tool_name not in reachable:
        return (
            False,
            "skill_unbound",
            "registered and enabled but no loadable skill or tool floor can expose it",
        )
    return True, None, None


async def _persist_results(
    pool: Any,
    *,
    worker_name: str,
    worker_id: str | None,
    registry_kind: str,
    results: list[CapabilityResult],
) -> None:
    if pool is None:
        return
    try:
        normalized_worker_id = str(uuid.UUID(str(worker_id))) if worker_id else None
    except (TypeError, ValueError, AttributeError):
        normalized_worker_id = None
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT record_worker_capabilities($1, $2::uuid, $3, $4::jsonb)",
                worker_name,
                normalized_worker_id,
                registry_kind,
                json.dumps([result.to_record() for result in results]),
            )
    except Exception:
        # Measurement is advisory. A missing migration or brief DB failure must
        # never take a heartbeat worker down.
        logger.warning("Capability probe could not persist results", exc_info=True)


async def probe_and_record(
    pool: Any,
    registry: Any,
    *,
    worker_name: str,
    worker_id: str | None = None,
    contexts: tuple[ToolContext, ...] = tuple(ToolContext),
) -> list[CapabilityResult]:
    """Measure and persist every context/tool pair for one worker registry."""
    config = await registry.get_config(force_refresh=True)
    registered = set(registry.list_names())
    catalogued = await _catalog_tool_names(pool)
    tool_names = sorted(registered | catalogued)
    registry_kind = str(getattr(registry, "registry_kind", "default"))
    results: list[CapabilityResult] = []

    for context in contexts:
        reachable = _reachable_tool_names(registry, context)
        for tool_name in tool_names:
            available, reason_code, reason = _classify_tool(
                registry, config, context, tool_name, reachable
            )
            results.append(
                CapabilityResult(
                    worker_name=worker_name,
                    tool_name=tool_name,
                    tool_context=context.value,
                    available=available,
                    reason_code=reason_code,
                    reason_if_missing=reason,
                    registry_kind=registry_kind,
                )
            )

    await _persist_results(
        pool,
        worker_name=worker_name,
        worker_id=worker_id,
        registry_kind=registry_kind,
        results=results,
    )
    logger.info(
        "Capability probe measured worker=%s registry=%s pairs=%d available=%d",
        worker_name,
        registry_kind,
        len(results),
        sum(result.available for result in results),
    )
    return results


async def _read_interval_minutes(pool: Any) -> int:
    if pool is None:
        return _DEFAULT_INTERVAL_MINUTES
    try:
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT COALESCE(get_config_int('capability_probe.interval_minutes'), $1)",
                _DEFAULT_INTERVAL_MINUTES,
            )
        return max(1, int(value))
    except Exception:
        return _DEFAULT_INTERVAL_MINUTES


_last_run: dict[str, float] = {}
_in_flight: set[str] = set()
_schedule_lock = asyncio.Lock()


async def run_probe_if_due(
    pool: Any,
    registry: Any,
    *,
    worker_name: str,
    worker_id: str | None = None,
) -> list[CapabilityResult] | None:
    """Run immediately on first use, then at the DB-configured cadence."""
    interval_minutes = await _read_interval_minutes(pool)
    now = asyncio.get_running_loop().time()
    key = f"{worker_name}:{worker_id or 'role'}"
    async with _schedule_lock:
        if key in _in_flight:
            return None
        if now - _last_run.get(key, float("-inf")) < interval_minutes * 60:
            return None
        _in_flight.add(key)
    try:
        results = await probe_and_record(
            pool,
            registry,
            worker_name=worker_name,
            worker_id=worker_id,
        )
    except Exception:
        # A failed pass retries on the next worker tick instead of hiding the
        # failure for a full interval.
        async with _schedule_lock:
            _in_flight.discard(key)
        raise
    async with _schedule_lock:
        _last_run[key] = now
        _in_flight.discard(key)
    return results


def _reset_state_for_tests() -> None:
    _last_run.clear()
    _in_flight.clear()


__all__ = ["CapabilityResult", "probe_and_record", "run_probe_if_due"]
