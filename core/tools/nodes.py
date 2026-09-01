"""Approval-gated access to explicitly paired companion nodes."""

from __future__ import annotations

import base64
import binascii
import re
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

_NODE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_CAPTURE_BYTES = 8 * 1024 * 1024


def _common_properties(operations: list[str]) -> dict[str, Any]:
    return {
        "node_id": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
            "description": "The exact paired node id from `hexis node status`.",
        },
        "operation": {"type": "string", "enum": operations},
        "timeout": {
            "type": "integer",
            "minimum": 5,
            "maximum": 120,
            "default": 30,
        },
    }


class StructuredNodeHandler(ToolHandler):
    """One typed private-host surface; every call remains approval-gated."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        operations: dict[str, str],
        properties: dict[str, Any],
        display_name: str,
    ) -> None:
        self._name = name
        self._description = description
        self._operations = operations
        self._properties = {
            **_common_properties(list(operations)),
            **properties,
        }
        self._display_name = display_name

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": self._properties,
                "required": ["node_id", "operation"],
                "additionalProperties": False,
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=4,
            requires_approval=True,
            is_read_only=False,
            supports_parallel=False,
            execution_timeout_seconds=125.0,
            allowed_contexts={
                ToolContext.CHAT,
                ToolContext.HEARTBEAT,
                ToolContext.MCP,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if context.registry is None:
            return ToolResult.error_result(
                f"{self._name} requires a database-backed tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        node_id = str(arguments.get("node_id") or "").strip().lower()
        if not _NODE_ID_RE.fullmatch(node_id):
            return ToolResult.error_result(
                "node_id must be the complete 64-character id from `hexis node status`.",
                ToolErrorType.INVALID_PARAMS,
            )
        operation = str(arguments.get("operation") or "").strip()
        action = self._operations.get(operation)
        if action is None:
            return ToolResult.error_result(
                f"operation must be one of: {', '.join(self._operations)}.",
                ToolErrorType.INVALID_PARAMS,
            )
        timeout = arguments.get("timeout", 30)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or not 5 <= timeout <= 120
        ):
            return ToolResult.error_result(
                "timeout must be an integer from 5 through 120 seconds.",
                ToolErrorType.INVALID_PARAMS,
            )
        node_arguments = {
            key: value
            for key, value in arguments.items()
            if key not in {"node_id", "operation"} and value is not None
        }
        node_arguments["timeout"] = timeout

        from services.node_gateway import request_node_invocation

        requested_by = f"{context.tool_context.value}:{context.surface or 'unknown'}"
        current = await request_node_invocation(
            context.registry.pool,
            node_id=node_id,
            action=action,
            arguments=node_arguments,
            requested_by=requested_by,
            timeout_seconds=timeout,
            metadata={
                "call_id": context.call_id,
                "session_id": context.session_id,
                "heartbeat_id": context.heartbeat_id,
                "approval_request_id": context.approval_request_id,
                "approval_channel": context.approval_channel,
            },
        )
        status = str(current.get("status") or "unknown")
        if status != "succeeded":
            reason = str(
                current.get("error")
                or current.get("reason")
                or f"Node invocation ended with status {status}."
            )
            error_type = (
                ToolErrorType.TIMEOUT
                if status == "expired"
                else ToolErrorType.EXECUTION_FAILED
            )
            return ToolResult.error_result(reason, error_type)
        result = current.get("result")
        if not isinstance(result, dict):
            result = {"value": result}
        return ToolResult.success_result(
            {
                "invocation_id": current.get("invocation_id"),
                "node_id": node_id,
                "action": action,
                "status": status,
                "result": result,
            },
            display_output=(
                f"{self._display_name} {operation.replace('_', ' ')} completed "
                f"on {node_id[:12]}…"
            ),
        )


class NodeInvokeHandler(ToolHandler):
    """Invoke one capability on a node after exact per-call approval."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="node_invoke",
            description=(
                "Run one explicitly approved capability on a paired companion node. "
                "system.run accepts only a command alias allowlisted locally on that "
                "node; it never accepts a shell string. screen.capture returns fresh "
                "visual context. Every invocation requires operator approval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "The exact id from `hexis node status`.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["system.run", "screen.capture"],
                    },
                    "command": {
                        "type": "string",
                        "description": (
                            "For system.run, the local allowlist alias—not an executable "
                            "or shell command."
                        ),
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 1000},
                        "maxItems": 40,
                        "default": [],
                        "description": (
                            "Optional invocation-time arguments. The node rejects these "
                            "unless the local alias explicitly permits them."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 120,
                        "default": 30,
                    },
                },
                "required": ["node_id", "action"],
                "additionalProperties": False,
            },
            category=ToolCategory.SHELL,
            energy_cost=4,
            requires_approval=True,
            is_read_only=False,
            supports_parallel=False,
            execution_timeout_seconds=125.0,
            allowed_contexts={
                ToolContext.CHAT,
                ToolContext.HEARTBEAT,
                ToolContext.MCP,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if context.registry is None:
            return ToolResult.error_result(
                "node_invoke requires a database-backed tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        node_id = str(arguments.get("node_id") or "").strip().lower()
        if not _NODE_ID_RE.fullmatch(node_id):
            return ToolResult.error_result(
                "node_id must be the complete 64-character id from `hexis node status`.",
                ToolErrorType.INVALID_PARAMS,
            )
        action = str(arguments.get("action") or "").strip()
        timeout = arguments.get("timeout", 30)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or not 5 <= timeout <= 120
        ):
            return ToolResult.error_result(
                "timeout must be an integer from 5 through 120 seconds.",
                ToolErrorType.INVALID_PARAMS,
            )

        node_arguments: dict[str, Any] = {"timeout": timeout}
        if action == "system.run":
            command = str(arguments.get("command") or "").strip()
            if not command:
                return ToolResult.error_result(
                    "system.run requires a local allowlist alias in command.",
                    ToolErrorType.INVALID_PARAMS,
                )
            args = arguments.get("args", [])
            if not isinstance(args, list) or not all(
                isinstance(item, str) and len(item) <= 1000 for item in args
            ):
                return ToolResult.error_result(
                    "args must contain at most 40 short strings.",
                    ToolErrorType.INVALID_PARAMS,
                )
            if len(args) > 40:
                return ToolResult.error_result(
                    "args must contain at most 40 short strings.",
                    ToolErrorType.INVALID_PARAMS,
                )
            node_arguments.update({"command": command, "args": args})
        elif action == "screen.capture":
            if arguments.get("command") or arguments.get("args"):
                return ToolResult.error_result(
                    "screen.capture does not accept command or args.",
                    ToolErrorType.INVALID_PARAMS,
                )
        else:
            return ToolResult.error_result(
                "action must be system.run or screen.capture.",
                ToolErrorType.INVALID_PARAMS,
            )

        from services.node_gateway import request_node_invocation

        requested_by = f"{context.tool_context.value}:{context.surface or 'unknown'}"
        current = await request_node_invocation(
            context.registry.pool,
            node_id=node_id,
            action=action,
            arguments=node_arguments,
            requested_by=requested_by,
            timeout_seconds=timeout,
            metadata={
                "call_id": context.call_id,
                "session_id": context.session_id,
                "heartbeat_id": context.heartbeat_id,
                "approval_request_id": context.approval_request_id,
                "approval_channel": context.approval_channel,
            },
        )
        status = str(current.get("status") or "unknown")
        if status != "succeeded":
            reason = str(
                current.get("error")
                or current.get("reason")
                or f"Node invocation ended with status {status}."
            )
            error_type = (
                ToolErrorType.TIMEOUT
                if status == "expired"
                else ToolErrorType.EXECUTION_FAILED
            )
            return ToolResult.error_result(reason, error_type)

        result = current.get("result")
        if not isinstance(result, dict):
            result = {"value": result}
        output: dict[str, Any] = {
            "invocation_id": current.get("invocation_id"),
            "node_id": node_id,
            "action": action,
            "status": status,
            "result": result,
        }
        if action != "screen.capture":
            return ToolResult.success_result(
                output,
                display_output=f"Node command completed on {node_id[:12]}…",
            )

        mime_type = str(result.get("mime_type") or "")
        encoded = str(result.get("data_base64") or "")
        try:
            capture = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return ToolResult.error_result(
                "The node returned an invalid screen capture payload.",
                ToolErrorType.EXECUTION_FAILED,
            )
        if mime_type != "image/png" or not capture or len(capture) > _MAX_CAPTURE_BYTES:
            return ToolResult.error_result(
                "The node returned a screen capture with an invalid type or size.",
                ToolErrorType.EXECUTION_FAILED,
            )
        data_url = f"data:image/png;base64,{encoded}"
        safe_result = {
            key: value for key, value in result.items() if key != "data_base64"
        }
        safe_result["visual_context_attached"] = True
        output["result"] = safe_result
        if context.tool_context == ToolContext.MCP:
            # MCP has no in-loop visual-message bridge. Preserve the capture for
            # a client that explicitly invoked the tool and can consume data URLs.
            output["result"] = {**safe_result, "data_url": data_url}
        return ToolResult(
            success=True,
            output=output,
            display_output=f"Captured the screen on node {node_id[:12]}…",
            metadata={
                "model_visual_attachments": [
                    {
                        "name": f"screen-{node_id[:12]}.png",
                        "mime_type": "image/png",
                        "data_url": data_url,
                    }
                ]
            },
        )


def create_node_tools() -> list[ToolHandler]:
    reminders = StructuredNodeHandler(
        name="apple_reminders",
        description=(
            "List or create Apple Reminders through a paired Mac. No model-authored "
            "AppleScript is accepted; each exact private-host call requires approval."
        ),
        operations={
            "list": "apple.reminders.list",
            "create": "apple.reminders.create",
        },
        properties={
            "title": {"type": "string", "maxLength": 500},
            "list_name": {"type": "string", "maxLength": 200},
            "notes": {"type": "string", "maxLength": 5000},
            "due_at": {
                "type": "string",
                "description": "ISO 8601 date-time with timezone for create.",
            },
            "include_completed": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
        },
        display_name="Apple Reminders",
    )
    notes = StructuredNodeHandler(
        name="apple_notes",
        description=(
            "Search or create Apple Notes through a paired Mac using fixed local "
            "automation. Every call requires approval."
        ),
        operations={"search": "apple.notes.search", "create": "apple.notes.create"},
        properties={
            "query": {"type": "string", "maxLength": 500},
            "title": {"type": "string", "maxLength": 500},
            "body": {"type": "string", "maxLength": 20000},
            "folder": {"type": "string", "maxLength": 200},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
        },
        display_name="Apple Notes",
    )
    calendar = StructuredNodeHandler(
        name="apple_calendar",
        description=(
            "List a bounded time window or create an Apple Calendar event through "
            "a paired Mac. Date-times must carry timezone offsets and every call "
            "requires approval."
        ),
        operations={
            "list": "apple.calendar.list",
            "create": "apple.calendar.create",
        },
        properties={
            "title": {"type": "string", "maxLength": 500},
            "start_at": {
                "type": "string",
                "description": "ISO 8601 date-time with timezone.",
            },
            "end_at": {
                "type": "string",
                "description": "ISO 8601 date-time with timezone.",
            },
            "calendar": {"type": "string", "maxLength": 200},
            "location": {"type": "string", "maxLength": 500},
            "notes": {"type": "string", "maxLength": 5000},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
        },
        display_name="Apple Calendar",
    )
    shortcuts = StructuredNodeHandler(
        name="apple_shortcuts",
        description=(
            "List or run a named Apple Shortcut on a paired Mac. Shortcut execution "
            "may have arbitrary local effects, so every exact name requires approval."
        ),
        operations={"list": "apple.shortcuts.list", "run": "apple.shortcuts.run"},
        properties={"name": {"type": "string", "maxLength": 200}},
        display_name="Apple Shortcuts",
    )
    onepassword = StructuredNodeHandler(
        name="onepassword_local",
        description=(
            "List redacted 1Password item metadata or copy one exact op:// field to "
            "the paired Mac clipboard. Secret values never cross the node gateway or "
            "enter model context; every call requires approval."
        ),
        operations={
            "list_items": "onepassword.items",
            "copy_field": "onepassword.copy",
        },
        properties={
            "query": {"type": "string", "maxLength": 200},
            "vault": {"type": "string", "maxLength": 200},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
            "secret_ref": {
                "type": "string",
                "pattern": "^op://[^/\\s]+/[^/\\s]+/[^/\\s]+$",
                "maxLength": 500,
            },
            "clipboard_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": 300,
                "default": 60,
            },
        },
        display_name="1Password",
    )
    return [NodeInvokeHandler(), reminders, notes, calendar, shortcuts, onepassword]
