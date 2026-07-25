"""Ambient responsibility management tool."""

from __future__ import annotations

import inspect
import json
import logging
import uuid
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

_VALID_ACTIONS = {"create", "list", "pause", "resume", "cancel", "status", "checkin", "evaluate_now"}
_VALID_KINDS = {"reminder", "monitor", "checkin", "threshold", "digest", "custom"}
_VALID_DELIVERY_MODES = {"outbox", "channel", "webhook", "silent"}


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


async def _gmail_setup_result(context: ToolExecutionContext, payload: dict[str, Any]) -> ToolResult:
    display = payload.get("display_output") or "Ambient responsibility needs Gmail setup before it can run."
    output = _json(payload.get("output"))
    output = output if isinstance(output, dict) else {}
    output["display_output"] = display
    registry = context.registry
    execute = getattr(registry, "execute", None)
    if callable(execute):
        try:
            setup_context = ToolExecutionContext(
                tool_context=context.tool_context,
                call_id=f"ambient-gmail-setup:{uuid.uuid4()}",
                heartbeat_id=context.heartbeat_id,
                session_id=context.session_id,
                energy_available=context.energy_available,
                workspace_path=context.workspace_path,
                is_group=context.is_group,
                allow_network=False,
                allow_shell=False,
                allow_file_write=False,
                allow_file_read=context.allow_file_read,
                registry=registry,
            )
            maybe = execute("gmail_setup_status", {}, setup_context)
            if inspect.isawaitable(maybe):
                maybe = await maybe
            if isinstance(maybe, ToolResult) and isinstance(maybe.output, dict):
                output["setup"] = maybe.output
                if isinstance(maybe.output.get("ui"), dict):
                    output["ui"] = maybe.output["ui"]
                display = maybe.display_output or display
                output["display_output"] = display
        except Exception:
            logger.debug("Could not attach Gmail setup UI to responsibility result", exc_info=True)
    return ToolResult.success_result(output, display_output=display)


class ManageResponsibilityHandler(ToolHandler):
    """Create and manage ambient responsibilities."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="manage_responsibility",
            description=(
                "Create, list, pause, resume, cancel, check in, or force-check ambient responsibilities. "
                "Use this for durable commitments that continue after the current chat: "
                "'let me know whenever X happens', 'watch for email from Hope', "
                "'remind me to take pills twice daily', 'tell me if I have not checked in', "
                "or 'notify me if a source crosses a threshold'. "
                "Use manage_schedule only for simple timed one-shot/recurring reminders that do not "
                "observe a changing source. Ambient responsibilities have triggers, evaluators, "
                "sources, actions, delivery, and audit history. "
                "For Gmail monitors, set sources to [{connector_id:'gmail', query:'from:hope@example.com'}] "
                "or use evaluator {connector_id:'gmail', query:'...', type:'importance'} for urgent-only checks. "
                "If a required connector is missing, create the blocked responsibility and open the connector setup flow."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": sorted(_VALID_ACTIONS),
                        "description": "create, list, pause, resume, cancel, status, checkin, or evaluate_now.",
                    },
                    "responsibility_id": {
                        "type": "string",
                        "description": "Existing responsibility ID for pause/resume/cancel/status/checkin/evaluate_now.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Human-readable title. Also used to find an existing responsibility when ID is absent.",
                    },
                    "description": {"type": "string", "description": "Optional description."},
                    "user_intent": {
                        "type": "string",
                        "description": "The user's original ongoing request, written plainly.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": sorted(_VALID_KINDS),
                        "description": "reminder, monitor, checkin, threshold, digest, or custom.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "description": "Relative priority for due checks.",
                    },
                    "trigger": {
                        "type": "object",
                        "description": (
                            "When to check. Examples: "
                            "{\"kind\":\"interval\",\"every_seconds\":60}, "
                            "{\"kind\":\"cron\",\"cron\":\"*/5 * * * *\"}, "
                            "{\"kind\":\"daily\",\"times\":[\"08:00\",\"20:00\"]}."
                        ),
                    },
                    "schedule": {
                        "type": "string",
                        "description": "Optional cron expression shorthand for trigger.kind=cron.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone for daily/cron triggers, e.g. America/New_York.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Sources to observe. Gmail example: "
                            "[{\"connector_id\":\"gmail\",\"query\":\"from:hope@example.com\",\"page_size\":10}]. "
                            "Future connectors can use slack/telegram/signal/twitter_x/health with the same shape."
                        ),
                    },
                    "evaluator": {
                        "type": "object",
                        "description": (
                            "Condition logic. Examples: {\"type\":\"importance\"} for urgent/important matching, "
                            "{\"type\":\"missing_checkin\",\"lookback_minutes\":720}, "
                            "{\"type\":\"threshold\",\"metric\":\"steps\",\"operator\":\"<\",\"value\":6000}."
                        ),
                    },
                    "actions": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Actions when the condition fires. Start with notify_user only unless a separate "
                            "authorization grants external side effects. Example: "
                            "[{\"type\":\"notify_user\",\"message\":\"Hope emailed you: {title}\"}]."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": "Convenience notify_user message for create.",
                    },
                    "delivery_mode": {
                        "type": "string",
                        "enum": sorted(_VALID_DELIVERY_MODES),
                        "description": "outbox (default), channel, webhook, or silent.",
                    },
                    "delivery_channel": {"type": "string"},
                    "delivery_topic": {"type": "string"},
                    "delivery_target_id": {"type": "string"},
                    "delivery_webhook_url": {"type": "string"},
                    "memory_policy": {
                        "type": "string",
                        "enum": ["remember", "task_scoped", "forget"],
                        "description": "Whether observations may feed memory; default task_scoped.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter for list or requested status for create.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    "label": {"type": "string", "description": "Check-in label."},
                    "note": {"type": "string", "description": "Check-in note."},
                    "occurred_at": {
                        "type": "string",
                        "description": "Optional check-in timestamp.",
                    },
                    "metadata": {"type": "object"},
                },
                "required": ["action"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=2,
            is_read_only=False,
            requires_approval=False,
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        action = str(arguments.get("action") or "")
        if action not in _VALID_ACTIONS:
            return ToolResult.error_result(
                f"Invalid action '{action}'. Must be one of: {', '.join(sorted(_VALID_ACTIONS))}",
                ToolErrorType.INVALID_PARAMS,
            )

        pool = context.registry.pool if context.registry else None
        if not pool:
            return ToolResult.error_result("Database pool not available", ToolErrorType.MISSING_CONFIG)

        try:
            async with pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT manage_ambient_responsibility_tool($1::jsonb)",
                    json.dumps(arguments, default=str),
                )
            payload = _json(raw)
            if not isinstance(payload, dict) or "success" not in payload:
                return ToolResult.error_result(
                    "Ambient responsibility tool returned an unexpected payload",
                    ToolErrorType.EXECUTION_FAILED,
                )
            if not payload.get("success"):
                error_type = payload.get("error_type") or ToolErrorType.EXECUTION_FAILED.value
                try:
                    typed_error = ToolErrorType(error_type)
                except ValueError:
                    typed_error = ToolErrorType.EXECUTION_FAILED
                return ToolResult.error_result(
                    payload.get("error") or "Ambient responsibility action failed",
                    typed_error,
                )

            output = payload.get("output")
            if action == "evaluate_now" and isinstance(output, dict):
                try:
                    from services.ambient_responsibilities import run_ambient_responsibility_step

                    evaluation = await run_ambient_responsibility_step(pool, limit=1)
                    output["evaluation"] = evaluation
                    payload["display_output"] = _display_evaluation(output, payload.get("display_output"))
                except Exception as exc:
                    logger.exception("Immediate ambient evaluation failed")
                    output["evaluation"] = {"failed": True, "error": str(exc)}
            if isinstance(output, dict):
                responsibility = output.get("responsibility") if isinstance(output.get("responsibility"), dict) else {}
                missing = output.get("missing_connectors") or responsibility.get("missing_connectors")
                if isinstance(missing, list) and any(
                    isinstance(item, dict) and item.get("connector_id") == "gmail" for item in missing
                ):
                    return await _gmail_setup_result(context, payload)
            return ToolResult.success_result(output, display_output=payload.get("display_output"))
        except Exception as exc:
            logger.exception("Ambient responsibility tool failed")
            return ToolResult.error_result(
                f"Ambient responsibility tool failed: {exc}",
                ToolErrorType.EXECUTION_FAILED,
            )


def create_responsibility_tools() -> list[ToolHandler]:
    return [ManageResponsibilityHandler()]


def _display_evaluation(output: dict[str, Any], fallback: Any) -> str:
    evaluation = output.get("evaluation")
    if not isinstance(evaluation, dict):
        return str(fallback or "Ambient check queued.")
    runs = evaluation.get("runs")
    if isinstance(runs, list) and runs:
        run = runs[0] if isinstance(runs[0], dict) else {}
        decision = run.get("decision") if isinstance(run.get("decision"), dict) else {}
        status = str(run.get("status") or "checked")
        if decision.get("notify_message"):
            return str(decision["notify_message"])
        reason = str(decision.get("reason") or status).replace("_", " ")
        return f"Ambient check {status}: {reason}."
    if evaluation.get("skipped"):
        return f"Ambient check skipped: {str(evaluation.get('reason') or 'nothing due').replace('_', ' ')}."
    return str(fallback or "Ambient check ran.")
