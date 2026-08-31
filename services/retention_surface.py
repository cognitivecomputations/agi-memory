"""Thin service wrappers for user-controlled memory fade decisions."""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def publish_retention_surface(conn: Any) -> dict[str, Any]:
    """Publish new asks and due factual compression reports."""
    reviews = _object(
        await conn.fetchval("SELECT publish_memory_fade_review_digest()")
    )
    compressions = _object(
        await conn.fetchval(
            "SELECT publish_retention_compression_report_if_due(FALSE)"
        )
    )
    if reviews.get("skipped") and compressions.get("skipped"):
        return {
            "skipped": True,
            "reason": "no_retention_surface_work",
            "reviews": reviews,
            "compressions": compressions,
        }
    return {"reviews": reviews, "compressions": compressions}


async def resolve_memory_fade_review_from_inbound(
    pool: Any,
    *,
    channel: str,
    actor: str,
    text: str,
) -> dict[str, Any]:
    try:
        async with pool.acquire() as conn:
            return _object(
                await conn.fetchval(
                    "SELECT try_resolve_memory_fade_review_from_inbound($1, $2, $3)",
                    channel,
                    actor,
                    text,
                )
            )
    except Exception:
        logger.warning("Inbound memory-fade resolution failed", exc_info=True)
        return {"recognized": False, "matched": False, "reason": "resolution_error"}
