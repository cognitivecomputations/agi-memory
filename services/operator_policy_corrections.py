"""Identity-gated bridge to the DB-owned operator policy ledger."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


async def channel_sender_is_operator(
    pool: Any,
    *,
    channel_type: str,
    sender_id: str,
) -> bool:
    """Match a channel sender against an explicit operator identity.

    This check deliberately fails closed. An allowlist authorizes conversation;
    it does not grant authority to write standing policy.
    """
    try:
        async with pool.acquire() as conn:
            return bool(
                await conn.fetchval(
                    "SELECT channel_sender_is_operator($1::text, $2::text)",
                    channel_type,
                    sender_id,
                )
            )
    except Exception as exc:
        logger.warning(
            "Could not verify operator identity for %s sender %s; policy capture is disabled for this turn: %s",
            channel_type,
            sender_id,
            exc,
            exc_info=True,
        )
        return False


async def capture_operator_policy_correction(
    pool: Any,
    *,
    channel_type: str,
    channel_id: str | None,
    sender_id: str | None,
    sender_name: str | None,
    text: str,
    is_operator: bool,
    disposition: str = "engage",
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture one explicit standing instruction without blocking the turn.

    Classification and all durable side effects are one database transaction.
    A storage failure remains observable but does not discard the user's chat.
    """
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT capture_operator_policy_correction(
                    $1::text, $2::text, $3::text, $4::text, $5::text,
                    $6::boolean, $7::text, $8::text, $9::jsonb
                )
                """,
                channel_type,
                channel_id,
                sender_id,
                sender_name,
                text,
                bool(is_operator),
                disposition,
                reason,
                json.dumps(metadata or {}, default=str),
            )
    except Exception as exc:
        logger.warning(
            "Operator policy capture failed; the conversation will continue but the standing instruction was not persisted: %s",
            exc,
            exc_info=True,
        )
        return {
            "captured": False,
            "reason": "storage_error",
            "error": str(exc),
            "next_step": "Retry the standing instruction after checking the database migration and logs.",
        }

    result = _object(raw)
    if result:
        return result
    logger.warning("Operator policy capture returned an invalid database payload")
    return {
        "captured": False,
        "reason": "invalid_result",
        "next_step": "Check the database migration and retry the standing instruction.",
    }


async def list_operator_policies(pool: Any, *, limit: int = 50) -> dict[str, Any]:
    """Return active policies for an operator-facing control surface."""
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT list_operator_policies($1::int)",
                min(200, max(1, int(limit))),
            )
    except Exception as exc:
        logger.warning("Could not list operator policies: %s", exc, exc_info=True)
        return {
            "ok": False,
            "reason": "storage_error",
            "error": str(exc),
            "next_step": "Check the database migration and retry.",
        }
    result = _object(raw)
    if not result:
        return {
            "ok": False,
            "reason": "invalid_result",
            "next_step": "Check the database migration and retry.",
        }
    return {"ok": True, **result}


async def revoke_operator_policy(
    pool: Any,
    *,
    policy_key: str,
    actor: str = "operator",
    reason: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Revoke one active policy while retaining its immutable evidence trail."""
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT revoke_operator_policy($1::text, $2::text, $3::text, $4::jsonb)",
                policy_key,
                actor,
                reason,
                json.dumps({"event_id": event_id} if event_id else {}),
            )
    except Exception as exc:
        logger.warning(
            "Could not revoke operator policy %s: %s",
            policy_key,
            exc,
            exc_info=True,
        )
        return {
            "revoked": False,
            "reason": "storage_error",
            "error": str(exc),
            "next_step": "Check the database migration and retry; the policy remains active.",
        }
    result = _object(raw)
    if result:
        return result
    return {
        "revoked": False,
        "reason": "invalid_result",
        "next_step": "Check the database migration and retry; the policy may remain active.",
    }


async def render_operator_policy_context(pool: Any) -> str:
    """Render active policy guidance for runtimes outside the normal agent path."""
    try:
        async with pool.acquire() as conn:
            return str(
                await conn.fetchval("SELECT render_operator_policy_context()") or ""
            ).strip()
    except Exception as exc:
        logger.warning(
            "Active operator policies could not be loaded for this turn: %s",
            exc,
            exc_info=True,
        )
        return ""


__all__ = [
    "capture_operator_policy_correction",
    "channel_sender_is_operator",
    "list_operator_policies",
    "render_operator_policy_context",
    "revoke_operator_policy",
]
