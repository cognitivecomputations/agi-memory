"""Thin transport/classifier bridge over the DB-owned inbound policy."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

_FAIL_OPEN_RESULT: dict[str, Any] = {
    "disposition": "observe",
    "reason": "error_fallback",
    "ambiguous": False,
    "is_operator": False,
    "reply_allowed": False,
    "trigger_stripped_text": None,
    "session_id": None,
    "audit_id": None,
}

_SYSTEM_PROMPT = """You classify an inbound message to Hexis, an AI assistant.

Decide whether the sender is addressing Hexis, and whether the message corrects or
contradicts something Hexis recently said or did. Be conservative when uncertain.
Return JSON only:
{"addressed_to_hexis": true|false, "is_correction": true|false, "confidence": 0.0-1.0}
"""


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"expected json object, got {type(value).__name__}")


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _config_bool(value: Any, *, default: bool) -> bool:
    parsed = _strict_bool(value)
    if parsed is not None:
        return parsed
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _sender_tail(sender_id: str | None) -> str:
    value = str(sender_id or "").strip()
    return f"…{value[-4:]}" if value else "<unknown>"


async def is_disposition_enabled(pool: "asyncpg.Pool") -> bool:
    """Return the dark-by-default master flag; DB failure restores legacy gates."""
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT COALESCE(get_config_bool('channel.disposition.enabled'), FALSE)"
            )
            return _config_bool(raw, default=False)
    except Exception:
        logger.warning(
            "Inbound disposition flag read failed; retaining legacy channel gates",
            exc_info=True,
        )
        return False


async def resolve_disposition(
    pool: "asyncpg.Pool",
    *,
    channel_type: str,
    sender_id: str,
    session_id: Any,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one inbound event and optionally classify operator ambiguity.

    Policy, identity, allowlists, deterministic rules, and the audit row all
    live in Postgres. Any failure retains an observe-only result.
    """
    sender_tail = _sender_tail(sender_id)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT resolve_inbound_disposition(
                    $1::text, $2::text, $3::text, $4::text, $5::jsonb, FALSE
                )
                """,
                channel_type,
                sender_id,
                None if session_id is None else str(session_id),
                text,
                json.dumps(metadata or {}, default=str),
            )
            result = _object(raw)
            if result.get("ambiguous"):
                classifier_enabled = _config_bool(
                    await conn.fetchval(
                        "SELECT COALESCE(get_config_bool('channel.disposition.classifier_enabled'), TRUE)"
                    ),
                    default=True,
                )
                timeout_seconds = float(
                    await conn.fetchval(
                        "SELECT COALESCE(get_config_int('channel.disposition.classifier_timeout_seconds'), 10)"
                    )
                    or 10
                )
                if classifier_enabled:
                    try:
                        from core.llm_config import load_llm_config

                        llm_config = await load_llm_config(
                            conn,
                            "llm.inbound_disposition",
                            fallback_key="llm.subconscious",
                        )
                    except Exception:
                        logger.warning(
                            "Inbound classifier configuration failed; retaining deterministic observation",
                            exc_info=True,
                        )
                        classifier_enabled = False
                        llm_config = {}
                else:
                    llm_config = {}
            else:
                classifier_enabled = False
                timeout_seconds = 10.0
                llm_config = {}
    except Exception:
        logger.warning(
            "Inbound disposition SQL failed for %s/%s; retaining passive observation",
            channel_type,
            sender_tail,
            exc_info=True,
        )
        return dict(_FAIL_OPEN_RESULT)

    if result.get("ambiguous") and classifier_enabled:
        result = await _finalize_with_classifier(
            pool,
            result=result,
            channel_type=channel_type,
            sender_tail=sender_tail,
            text=text,
            llm_config=llm_config,
            timeout_seconds=max(1.0, min(timeout_seconds, 60.0)),
        )

    logger.info(
        "Inbound disposition: channel=%s sender=%s disposition=%s reason=%s",
        channel_type,
        sender_tail,
        result.get("disposition"),
        result.get("classifier_label") or result.get("reason"),
    )
    return result


async def _finalize_with_classifier(
    pool: "asyncpg.Pool",
    *,
    result: dict[str, Any],
    channel_type: str,
    sender_tail: str,
    text: str,
    llm_config: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        from core.llm_json import chat_json

        payload, _raw = await asyncio.wait_for(
            chat_json(
                llm_config=llm_config,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                f"CHANNEL: {channel_type}",
                                f"SENDER_IS_OPERATOR: {str(bool(result.get('is_operator'))).lower()}",
                                f"MESSAGE: {(text or '')[:2000]}",
                            ]
                        ),
                    },
                ],
                max_tokens=120,
                temperature=0.0,
                response_format={"type": "json_object"},
                fallback={},
            ),
            timeout=timeout_seconds,
        )
        addressed = _strict_bool(payload.get("addressed_to_hexis"))
        correction = _strict_bool(payload.get("is_correction"))
        if addressed is None or correction is None:
            raise ValueError("classifier omitted required boolean fields")
    except Exception:
        logger.warning(
            "Inbound classifier failed for %s/%s; retaining deterministic %s",
            channel_type,
            sender_tail,
            result.get("disposition"),
            exc_info=True,
        )
        return result

    if addressed:
        disposition, label = "engage", "classifier_addressed"
    elif correction and bool(result.get("is_operator")):
        disposition, label = "wake", "classifier_correction"
    else:
        disposition, label = "observe", "classifier_unaddressed"

    audit_id = result.get("audit_id")
    if audit_id is not None:
        try:
            async with pool.acquire() as conn:
                finalized = await conn.fetchval(
                    "SELECT finalize_inbound_disposition($1::bigint, $2::text, $3::text)",
                    int(audit_id),
                    disposition,
                    label,
                )
            if not finalized:
                raise RuntimeError("audit row no longer exists")
        except Exception:
            logger.warning(
                "Inbound classifier audit finalization failed for event %s; retaining deterministic %s",
                audit_id,
                result.get("disposition"),
                exc_info=True,
            )
            return result

    updated = dict(result)
    updated.update(
        disposition=disposition,
        classifier_used=True,
        classifier_label=label,
    )
    return updated


async def record_passive_observation(
    pool: "asyncpg.Pool",
    *,
    message: Any,
    disposition: dict[str, Any],
) -> dict[str, Any]:
    """Persist an observe/wake event through the canonical channel ledger."""
    metadata = dict(getattr(message, "metadata", {}) or {})
    metadata.update(
        inbound_disposition={
            "event_id": disposition.get("audit_id"),
            "disposition": disposition.get("disposition"),
            "reason": disposition.get("classifier_label") or disposition.get("reason"),
            "is_operator": bool(disposition.get("is_operator")),
        }
    )
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT record_inbound_disposition_observation(
                    $1::bigint, $2::text, $3::text, $4::text, $5::text,
                    $6::text, $7::text, $8::jsonb
                )
                """,
                disposition.get("audit_id"),
                message.channel_type,
                message.channel_id,
                message.sender_id,
                message.sender_name,
                message.content,
                message.message_id,
                json.dumps(metadata, default=str),
            )
        return _object(raw)
    except Exception as exc:
        logger.warning(
            "Passive inbound observation could not be stored for %s/%s: %s",
            message.channel_type,
            _sender_tail(message.sender_id),
            exc,
            exc_info=True,
        )
        return {
            "session_id": None,
            "message_id": None,
            "error": str(exc),
        }


__all__ = [
    "is_disposition_enabled",
    "record_passive_observation",
    "resolve_disposition",
]
