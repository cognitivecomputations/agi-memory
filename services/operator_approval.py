"""Durable operator approval for exact protected tool calls.

The waiting agent loop retains the real arguments in memory. Postgres stores
only a canonical hash plus a redacted preview, receives an identity-checked
phone decision, and consumes that proof once in the tool policy gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING

from channels.presentation import MessagePresentation
from services.approval_slack_actions import build_approval_presentation

if TYPE_CHECKING:
    import asyncpg
    from core.tools.base import ToolContext

logger = logging.getLogger(__name__)

_SENSITIVE_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "private_key",
)
_FINAL_STATUSES = {"approved", "denied", "consumed", "expired"}

ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def redact_approval_arguments(value: Any, *, depth: int = 0) -> Any:
    """Return a bounded human preview without persisting secrets."""
    if depth >= 5:
        return "[nested value omitted]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:40]:
            key = str(raw_key)
            normalized = key.lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_PARTS):
                result[key] = "[redacted]"
            else:
                result[key] = redact_approval_arguments(raw_value, depth=depth + 1)
        if len(value) > 40:
            result["…"] = f"{len(value) - 40} more fields"
        return result
    if isinstance(value, (list, tuple)):
        items = [redact_approval_arguments(item, depth=depth + 1) for item in value[:20]]
        if len(value) > 20:
            items.append(f"[{len(value) - 20} more items]")
        return items
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:499] + "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]


def _approval_message(
    request_id: uuid.UUID,
    tool_name: str,
    preview: dict[str, Any],
    *,
    tool_context: str,
    surface: str,
) -> str:
    code = request_id.hex[:8]
    rendered = json.dumps(preview, ensure_ascii=False, indent=2, default=str)
    return (
        f"Hexis wants to run `{tool_name}` from {surface} ({tool_context}).\n\n"
        f"Arguments shown to you:\n```\n{rendered}\n```\n\n"
        f"Approve only if this exact action is what you want. Request: {code}"
    )


def _coerce_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def _config_int(pool: "asyncpg.Pool", key: str, fallback: int) -> int:
    try:
        async with pool.acquire() as conn:
            value = await conn.fetchval("SELECT get_config_int($1)", key)
        return int(value) if value is not None else fallback
    except Exception:
        logger.debug("Approval config lookup failed for %s", key, exc_info=True)
        return fallback


async def _config_bool(pool: "asyncpg.Pool", key: str, fallback: bool) -> bool:
    try:
        async with pool.acquire() as conn:
            value = await conn.fetchval("SELECT get_config_bool($1)", key)
        return bool(value) if value is not None else fallback
    except Exception:
        logger.debug("Approval config lookup failed for %s", key, exc_info=True)
        return fallback


async def request_operator_tool_approval(
    pool: "asyncpg.Pool",
    *,
    tool_name: str,
    arguments: dict[str, Any],
    tool_context: str,
    session_id: str | None,
    heartbeat_id: str | None,
    surface: str,
    wait_seconds: int,
) -> dict[str, Any]:
    """File, deliver, and await one approval decision."""
    request_id = uuid.uuid4()
    preview = redact_approval_arguments(arguments)
    if not isinstance(preview, dict):
        preview = {"value": preview}
    message = _approval_message(
        request_id,
        tool_name,
        preview,
        tool_context=tool_context,
        surface=surface,
    )
    presentation: MessagePresentation = build_approval_presentation(
        approval_request_id=str(request_id),
        message=message,
        interactive=await _config_bool(
            pool, "operator.approval.slack_interactive_enabled", True
        ),
    )

    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT create_operator_tool_approval_request(
                $1::uuid, $2, $3::jsonb, $4::jsonb, $5, $6, $7, $8,
                $9, $10::jsonb, $11
            )
            """,
            request_id,
            tool_name,
            json.dumps(arguments, default=str),
            json.dumps(preview, default=str),
            tool_context,
            session_id,
            heartbeat_id,
            surface,
            message,
            json.dumps(presentation.to_dict(), default=str),
            wait_seconds,
        )
    created = _coerce_object(raw)
    if not created.get("created"):
        return {
            "approved": False,
            "status": str(created.get("status") or "disabled"),
            "reason": str(created.get("reason") or "Phone approval is unavailable."),
            "next_step": created.get("next_step"),
        }
    if not created.get("routed"):
        return {
            "approved": False,
            "request_id": str(request_id),
            "status": "unrouted",
            "reason": "The protected action was not run because no operator Slack recipient is configured.",
            "next_step": created.get("next_step"),
        }

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(60, wait_seconds)
    changed = asyncio.Event()

    def _listener(_connection, _pid, _channel, payload: str) -> None:
        if payload == str(request_id):
            changed.set()

    async with pool.acquire() as conn:
        await conn.add_listener("operator_approval_decisions", _listener)
        try:
            while True:
                status_raw = await conn.fetchval(
                    "SELECT get_operator_tool_approval_status($1::uuid)", request_id
                )
                status = _coerce_object(status_raw)
                state = str(status.get("status") or "unknown")
                if state in _FINAL_STATUSES:
                    return {
                        "approved": state == "approved",
                        "request_id": str(request_id),
                        "status": state,
                        "approval_channel": status.get("decision_channel"),
                        "reason": (
                            None
                            if state == "approved"
                            else f"Operator approval ended with status: {state}."
                        ),
                    }

                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                changed.clear()
                try:
                    await asyncio.wait_for(changed.wait(), timeout=min(5.0, remaining))
                except asyncio.TimeoutError:
                    pass
        finally:
            await conn.remove_listener("operator_approval_decisions", _listener)

    async with pool.acquire() as conn:
        await conn.fetchval("SELECT expire_operator_tool_approval($1::uuid)", request_id)
    return {
        "approved": False,
        "request_id": str(request_id),
        "status": "expired",
        "reason": "The approval window expired before Hexis received a decision; the tool was not run.",
        "next_step": f"Ask Hexis to try `{tool_name}` again if you still want the action.",
    }


async def create_operator_approval_callback(
    pool: "asyncpg.Pool",
    *,
    tool_context: "ToolContext",
    session_id: str | None,
    heartbeat_id: str | None,
    surface: str,
) -> tuple[ApprovalCallback, int]:
    """Return the runtime callback and additional timeout it may consume."""
    wait_seconds = min(
        3600,
        max(60, await _config_int(pool, "operator.approval.wait_seconds", 900)),
    )
    route_ready = False
    try:
        async with pool.acquire() as conn:
            route_ready = bool(await conn.fetchval(
                """
                SELECT COALESCE(get_config_bool('operator.approval.enabled'), TRUE)
                   AND NULLIF(btrim(get_config_text('channel.slack.operator_user_id')), '') IS NOT NULL
                """
            ))
    except Exception:
        logger.debug("Approval route readiness lookup failed", exc_info=True)

    async def _callback(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return await request_operator_tool_approval(
                pool,
                tool_name=tool_name,
                arguments=arguments,
                tool_context=tool_context.value,
                session_id=session_id,
                heartbeat_id=heartbeat_id,
                surface=surface,
                wait_seconds=wait_seconds,
            )
        except Exception as exc:
            logger.warning("Operator approval request failed closed", exc_info=True)
            return {
                "approved": False,
                "status": "error",
                "reason": f"Phone approval failed: {type(exc).__name__}: {exc}",
                "next_step": "Check `hexis channels status` and the operator approval configuration, then retry.",
            }

    return _callback, wait_seconds if route_ready else 0


async def resolve_operator_approval_from_inbound(
    pool: "asyncpg.Pool",
    *,
    channel: str,
    actor: str,
    text: str,
) -> dict[str, Any]:
    """Resolve an exact or unambiguous plain-text phone decision."""
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT try_resolve_operator_tool_approval_from_inbound($1, $2, $3)",
                channel,
                actor,
                text,
            )
        return _coerce_object(raw)
    except Exception:
        logger.warning("Inbound operator approval resolution failed", exc_info=True)
        return {"matched": False, "reason": "resolution_error"}


async def run_operator_approval_escalations(
    pool: "asyncpg.Pool",
    manager: Any,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """Escalate unanswered Slack approvals to the configured iMessage handle."""
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT claim_operator_tool_approval_escalations($1)", limit
        )
    rows = raw if isinstance(raw, list) else json.loads(raw) if isinstance(raw, str) else []
    if not isinstance(rows, list):
        rows = []

    escalated = 0
    errors = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        request_id = str(row.get("id") or "")
        recipient = str(row.get("recipient") or "")
        code = request_id.replace("-", "")[:8]
        tool_name = str(row.get("tool_name") or "protected action")
        preview = json.dumps(row.get("arguments_preview") or {}, indent=2, default=str)
        message = (
            "No Slack response yet. Hexis is still waiting for approval.\n\n"
            f"Tool: {tool_name}\nArguments:\n{preview}\n\n"
            f"Reply `approve {code}` or `deny {code}`."
        )
        try:
            message_id = await manager.send("imessage", recipient, message)
            if not message_id:
                raise RuntimeError("iMessage adapter returned no message id")
            async with pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT complete_operator_tool_approval_escalation($1::uuid, $2)",
                    request_id,
                    str(message_id),
                )
            escalated += 1
        except Exception as exc:
            errors += 1
            async with pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT fail_operator_tool_approval_escalation($1::uuid, $2)",
                    request_id,
                    f"{type(exc).__name__}: {exc}"[:1000],
                )
            logger.warning("Operator approval iMessage escalation failed", exc_info=True)
    return {"claimed": len(rows), "escalated": escalated, "errors": errors}
