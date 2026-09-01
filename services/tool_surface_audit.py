"""Fail-open writer for immutable, per-turn tool-surface decisions."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from core.tools.base import ToolContext

logger = logging.getLogger(__name__)


def hash_input_text(text: str) -> str:
    """Hash normalized input so the audit proves sameness without storing a prompt."""
    return hashlib.sha256(str(text or "").strip().encode("utf-8")).hexdigest()


def _spec_names(specs: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in specs:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
    return names


async def _audit_enabled(pool: Any) -> bool:
    try:
        async with pool.acquire() as conn:
            return bool(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('tool_surface.audit_enabled'), TRUE)"
                )
            )
    except Exception:
        return True


async def record_tool_surface_decision(
    pool: Any,
    *,
    registry: Any,
    selection: Any,
    session_id: str | None,
    surface: str,
    tool_context: ToolContext,
    query: str,
) -> str | None:
    """Append the surface the model could really call for one selection.

    ``allowed_tool_names`` is the selector's intent. The intersection with both
    model specs and local handlers is the callable truth. Persisting the delta
    catches stale DB catalog entries, disabled tools, and skill bindings whose
    handler never reached this process.
    """
    return await record_tool_surface_snapshot(
        pool,
        registry=registry,
        allowed_tool_names=set(selection.allowed_tool_names),
        active_skill_names=[str(skill.name) for skill in selection.skills],
        considered=selection.considered,
        available_skill_count=len(selection.available),
        session_id=session_id,
        surface=surface,
        tool_context=tool_context,
        input_text_hash=hash_input_text(query),
        decision_kind="selection",
    )


async def record_tool_surface_snapshot(
    pool: Any,
    *,
    registry: Any,
    allowed_tool_names: set[str],
    active_skill_names: list[str],
    considered: list[dict[str, Any]] | None,
    available_skill_count: int,
    session_id: str | None,
    surface: str,
    tool_context: ToolContext,
    input_text_hash: str,
    decision_kind: str,
) -> str | None:
    """Append one resolved surface, including mid-turn skill activation."""
    if pool is None or registry is None or not await _audit_enabled(pool):
        return None

    requested = {str(name) for name in allowed_tool_names}
    try:
        exposed_specs = _spec_names(await registry.get_specs(tool_context))
    except Exception:
        logger.debug("Could not resolve tool specs for surface audit", exc_info=True)
        exposed_specs = set()
    try:
        registered = set(registry.list_names())
    except Exception:
        logger.debug("Could not resolve registered tools for surface audit", exc_info=True)
        return None
    reachable = requested & exposed_specs & registered
    missing = requested - reachable

    try:
        async with pool.acquire() as conn:
            event_id = await conn.fetchval(
                """
                SELECT record_tool_surface_decision(
                    $1::uuid, $2, $3, $4, $5, $6::text[], $7::jsonb,
                    $8::text[], $9::text[], $10::text[], $11, $12
                )
                """,
                _uuid_or_none(session_id),
                str(surface or "chat"),
                tool_context.value,
                "skill_activation" if decision_kind == "skill_activation" else "selection",
                input_text_hash,
                sorted({str(name) for name in active_skill_names}),
                json.dumps((considered or [])[:25]),
                sorted(requested),
                sorted(reachable),
                sorted(missing),
                max(int(available_skill_count), 0),
                str(getattr(registry, "registry_kind", "default")),
            )
        return str(event_id) if event_id else None
    except Exception:
        logger.debug("Tool-surface audit unavailable (non-fatal)", exc_info=True)
        return None


def _uuid_or_none(value: str | None) -> str | None:
    import uuid

    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


__all__ = [
    "hash_input_text",
    "record_tool_surface_decision",
    "record_tool_surface_snapshot",
]
