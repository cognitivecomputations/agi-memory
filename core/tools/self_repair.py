"""Self-repair tools for inspecting and managing observed substrate defects."""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"list", "diagnose", "mark_resolved"}
_VALID_STATUSES = {"open", "diagnosed", "repair_proposed", "verified", "resolved", "ignored", "all"}


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


class SelfRepairHandler(ToolHandler):
    """Review recorded software defects and draft bounded repair plans."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="self_repair",
            description=(
                "Review software defects observed in Hexis's own substrate. "
                "Use this during heartbeat or chat when tool calls, heartbeat steps, integrations, "
                "or memory retrieval fail. Actions: list unresolved reports, diagnose a specific report, "
                "or mark a verified report resolved. This tool records and reasons about defects; it does "
                "not silently edit source code."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": sorted(_VALID_ACTIONS),
                        "description": "list, diagnose, or mark_resolved.",
                    },
                    "defect_id": {
                        "type": "string",
                        "description": "Defect report UUID for diagnose or mark_resolved.",
                    },
                    "status": {
                        "type": "string",
                        "enum": sorted(_VALID_STATUSES),
                        "default": "open",
                        "description": "Status filter for list; use all for every status.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "Maximum reports to return.",
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Human-readable resolution for mark_resolved.",
                    },
                    "verification": {
                        "type": "object",
                        "description": "Verification evidence for mark_resolved, such as tests run.",
                    },
                },
                "required": ["action"],
            },
            category=ToolCategory.CODE,
            energy_cost=2,
            is_read_only=False,
            requires_approval=False,
            supports_parallel=False,
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT, ToolContext.MCP},
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors = super().validate(arguments)
        action = str(arguments.get("action") or "")
        if action and action not in _VALID_ACTIONS:
            errors.append(f"Invalid action '{action}'. Must be one of: {', '.join(sorted(_VALID_ACTIONS))}")
        status = str(arguments.get("status") or "open")
        if status and status not in _VALID_STATUSES:
            errors.append(f"Invalid status '{status}'. Must be one of: {', '.join(sorted(_VALID_STATUSES))}")
        if action in {"diagnose", "mark_resolved"} and not arguments.get("defect_id"):
            errors.append(f"defect_id is required for action '{action}'")
        if action == "mark_resolved" and not arguments.get("resolution"):
            errors.append("resolution is required for mark_resolved")
        return errors

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        action = str(arguments.get("action") or "")
        pool = context.registry.pool if context.registry else None
        if not pool:
            return ToolResult.error_result("Database pool not available", ToolErrorType.MISSING_CONFIG)

        try:
            async with pool.acquire() as conn:
                if action == "list":
                    status = str(arguments.get("status") or "open")
                    limit = min(50, max(1, int(arguments.get("limit") or 10)))
                    raw = await conn.fetchval("SELECT list_defect_reports($1, $2)", status, limit)
                    reports = _json(raw)
                    reports = reports if isinstance(reports, list) else []
                    return ToolResult.success_result(
                        {"reports": reports, "count": len(reports), "status": status},
                        display_output=_display_reports(reports, status),
                    )

                if action == "diagnose":
                    raw = await conn.fetchval(
                        "SELECT diagnose_defect_report($1::uuid)",
                        str(arguments.get("defect_id")),
                    )
                    payload = _json(raw)
                    if isinstance(payload, dict) and payload.get("success"):
                        return ToolResult.success_result(payload, display_output=_display_diagnosis(payload))
                    return ToolResult.error_result(
                        (payload or {}).get("error") if isinstance(payload, dict) else "Defect diagnosis failed",
                        ToolErrorType.EXECUTION_FAILED,
                    )

                if action == "mark_resolved":
                    raw = await conn.fetchval(
                        "SELECT mark_defect_report_resolved($1::uuid, $2, $3::jsonb)",
                        str(arguments.get("defect_id")),
                        str(arguments.get("resolution") or ""),
                        json.dumps(arguments.get("verification") or {}),
                    )
                    payload = _json(raw)
                    if isinstance(payload, dict) and payload.get("success"):
                        return ToolResult.success_result(payload, display_output="Defect report marked resolved.")
                    return ToolResult.error_result(
                        (payload or {}).get("error") if isinstance(payload, dict) else "Could not resolve defect",
                        ToolErrorType.EXECUTION_FAILED,
                    )
        except Exception as exc:
            logger.exception("Self-repair tool failed")
            return ToolResult.error_result(f"Self-repair tool failed: {exc}", ToolErrorType.EXECUTION_FAILED)

        return ToolResult.error_result(f"Invalid action '{action}'", ToolErrorType.INVALID_PARAMS)


def _display_reports(reports: list[Any], status: str) -> str:
    if not reports:
        return f"No {status} defect reports."
    lines = [f"{len(reports)} {status} defect report(s):"]
    for item in reports[:10]:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- [{item.get('severity', 'unknown')}] {item.get('title', 'Untitled defect')} "
            f"({item.get('occurrence_count', 1)}x; {item.get('status', status)})"
        )
    return "\n".join(lines)


def _display_diagnosis(payload: dict[str, Any]) -> str:
    diagnosis = payload.get("diagnosis") if isinstance(payload.get("diagnosis"), dict) else {}
    repair = payload.get("proposed_repair") if isinstance(payload.get("proposed_repair"), dict) else {}
    lines = [
        f"Diagnosis: {diagnosis.get('hypothesis', 'Needs inspection.')}",
        f"Severity: {diagnosis.get('severity', 'unknown')} · category: {diagnosis.get('category', 'unknown')}",
    ]
    likely_files = diagnosis.get("likely_files")
    if isinstance(likely_files, list) and likely_files:
        lines.append("Likely files: " + ", ".join(str(path) for path in likely_files))
    next_steps = repair.get("next_steps")
    if isinstance(next_steps, list) and next_steps:
        lines.append("Next: " + str(next_steps[0]))
    return "\n".join(lines)


def create_self_repair_tools() -> list[ToolHandler]:
    return [SelfRepairHandler()]
