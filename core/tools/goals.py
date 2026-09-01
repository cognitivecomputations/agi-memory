"""
Hexis Tools System - Goal Management

Allows the agent to create, update, complete, and manage goals
through the standard tool_use interface.
"""

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

_VALID_PRIORITIES = {"active", "queued", "backburner", "completed", "abandoned"}
_VALID_SOURCES = {"curiosity", "user_request", "identity", "derived", "external"}
_VALID_ACTIONS = {"create", "update_priority", "add_progress", "list"}


class ManageGoalsHandler(ToolHandler):
    """Manage the agent's goals: create, reprioritize, add progress notes, or list."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="manage_goals",
            description=(
                "Manage your goals. Actions: "
                "'create' (new goal), "
                "'update_priority' (change priority), "
                "'add_progress' (record progress on a goal), "
                "'list' (view current goals)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": list(_VALID_ACTIONS),
                        "description": "The goal management action to perform.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for a new goal (required for 'create').",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description for a new goal (optional for 'create').",
                    },
                    "source": {
                        "type": "string",
                        "enum": list(_VALID_SOURCES),
                        "description": "Source of the goal (for 'create'). Default: 'curiosity'.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": list(_VALID_PRIORITIES),
                        "description": "Priority level. Used for 'create' (default 'queued') and 'update_priority'.",
                    },
                    "goal_id": {
                        "type": "string",
                        "description": "Goal ID (required for 'update_priority' and 'add_progress').",
                    },
                    "note": {
                        "type": "string",
                        "description": "Progress note (required for 'add_progress').",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the priority change (optional for 'update_priority').",
                    },
                },
                "required": ["action"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=False,
            requires_approval=False,
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        action = arguments.get("action", "")
        if action not in _VALID_ACTIONS:
            return ToolResult.error_result(
                f"Invalid action '{action}'. Must be one of: {', '.join(sorted(_VALID_ACTIONS))}",
                ToolErrorType.INVALID_PARAMS,
            )
        pool = context.registry.pool if context.registry else None
        if not pool:
            return ToolResult.error_result(
                "Database pool not available",
                ToolErrorType.MISSING_CONFIG,
            )
        # execute_goals_tool (db/38) owns validation, mutation, and result
        # shaping; the former Python compatibility path was deleted.
        try:
            async with pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT execute_goals_tool($1::jsonb, $2::goal_source)",
                    json.dumps(arguments),
                    context.trusted_goal_origin,
                )
        except Exception as exc:
            logger.exception("Goals tool failed")
            return ToolResult.error_result(
                f"Goals tool failed: {exc}", ToolErrorType.EXECUTION_FAILED
            )
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(payload, dict) and "success" in payload:
            if payload.get("success"):
                return ToolResult.success_result(
                    payload.get("output"), payload.get("display_output")
                )
            return ToolResult.error_result(
                payload.get("error") or "Goals tool failed",
                ToolErrorType(
                    payload.get("error_type") or ToolErrorType.EXECUTION_FAILED.value
                ),
            )
        return ToolResult.error_result(
            "Goals tool returned an unexpected payload",
            ToolErrorType.EXECUTION_FAILED,
        )


def create_goal_tools() -> list[ToolHandler]:
    """Create goal management tools."""
    return [ManageGoalsHandler()]
