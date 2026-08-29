from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from core.llm_config import load_llm_config
from core.llm_json import chat_json
from services.prompt_resources import load_contradiction_detection_prompt

logger = logging.getLogger(__name__)


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def _uuid(value: Any) -> str | None:
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


async def run_contradiction_detection_step(
    conn: Any,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run one durable, rate-limited contradiction detection batch."""

    claimed = _object(
        await conn.fetchval(
            "SELECT claim_contradiction_detection_batch(NULL, $1::boolean)", force
        )
    )
    items = claimed.get("items")
    if not isinstance(items, list) or not items:
        return {
            "skipped": True,
            "reason": str(claimed.get("reason") or "empty_queue"),
        }

    queue_ids = [
        queue_id
        for queue_id in (_uuid(item.get("queue_id")) for item in items if isinstance(item, dict))
        if queue_id is not None
    ]
    usable_items = [
        item
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("memory"), dict)
        and isinstance(item.get("candidates"), list)
        and item["candidates"]
    ]
    try:
        minimum_confidence = min(
            1.0, max(0.0, float(claimed.get("minimum_confidence", 0.78)))
        )
    except (TypeError, ValueError):
        minimum_confidence = 0.78
    if not usable_items:
        result = {"checked": len(items), "filed": 0, "reason": "no_candidates"}
        await conn.fetchval(
            "SELECT finish_contradiction_detection_batch($1::uuid[], $2::jsonb, NULL)",
            queue_ids,
            json.dumps(result),
        )
        return result

    allowed_pairs: dict[frozenset[str], str] = {}
    for item in usable_items:
        memory_id = _uuid(item["memory"].get("memory_id"))
        if memory_id is None:
            continue
        for candidate in item["candidates"]:
            if not isinstance(candidate, dict):
                continue
            candidate_id = _uuid(candidate.get("memory_id"))
            if candidate_id is not None and candidate_id != memory_id:
                allowed_pairs[frozenset((memory_id, candidate_id))] = memory_id

    try:
        llm_config = await load_llm_config(
            conn, "llm.subconscious", fallback_key="llm.heartbeat"
        )
        document, raw = await chat_json(
            llm_config=llm_config,
            messages=[
                {
                    "role": "system",
                    "content": load_contradiction_detection_prompt().strip(),
                },
                {
                    "role": "user",
                    "content": "Candidate memories (JSON):\n"
                    + json.dumps(
                        {
                            "minimum_confidence": minimum_confidence,
                            "items": usable_items,
                        },
                        default=str,
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=1800,
            response_format={"type": "json_object"},
            fallback={"contradictions": []},
        )
        if not isinstance(document, dict):
            document = {"contradictions": []}
        observations = document.get("contradictions")
        if not isinstance(observations, list):
            observations = []

        filed: list[dict[str, Any]] = []
        rejected = 0
        processed_pairs: set[frozenset[str]] = set()
        for observation in observations:
            if not isinstance(observation, dict):
                rejected += 1
                continue
            memory_a = _uuid(observation.get("memory_a"))
            memory_b = _uuid(observation.get("memory_b"))
            pair = (
                frozenset((memory_a, memory_b))
                if memory_a and memory_b and memory_a != memory_b
                else frozenset()
            )
            new_memory_id = allowed_pairs.get(pair)
            tension = str(observation.get("tension") or "").strip()
            try:
                confidence = float(observation.get("confidence"))
            except (TypeError, ValueError):
                confidence = -1.0
            if (
                new_memory_id is None
                or pair in processed_pairs
                or not tension
                or not 0.0 <= confidence <= 1.0
            ):
                rejected += 1
                continue
            processed_pairs.add(pair)
            filed_raw = await conn.fetchval(
                """
                SELECT file_contradiction_case(
                    $1::uuid, $2::uuid, $3::uuid, $4, $5::float,
                    'model', $6::jsonb
                )
                """,
                memory_a,
                memory_b,
                new_memory_id,
                tension,
                confidence,
                json.dumps({"raw_response_type": type(raw).__name__}),
            )
            filed_result = _object(filed_raw)
            if filed_result.get("created"):
                filed.append(filed_result)
            else:
                rejected += 1

        result = {
            "checked": len(items),
            "candidate_sets": len(usable_items),
            "filed": len(filed),
            "cases": filed,
            "rejected": rejected,
        }
        await conn.fetchval(
            "SELECT finish_contradiction_detection_batch($1::uuid[], $2::jsonb, NULL)",
            queue_ids,
            json.dumps(result, default=str),
        )
        return result
    except Exception as exc:
        await conn.fetchval(
            "SELECT finish_contradiction_detection_batch($1::uuid[], '{}'::jsonb, $2)",
            queue_ids,
            str(exc),
        )
        logger.warning("Contradiction detection batch failed: %s", exc, exc_info=True)
        return {"failed": True, "error": str(exc), "checked": len(items)}


async def publish_contradiction_digest(conn: Any, *, force: bool = False) -> dict[str, Any]:
    result = _object(
        await conn.fetchval(
            "SELECT publish_contradiction_digest_if_due($1::boolean)", force
        )
    )
    return result or {"skipped": True, "reason": "no_result"}


async def resolve_contradiction_from_inbound(
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
                    "SELECT try_resolve_contradiction_from_inbound($1, $2, $3)",
                    channel,
                    actor,
                    text,
                )
            )
    except Exception:
        logger.warning("Inbound contradiction resolution failed", exc_info=True)
        return {"recognized": False, "matched": False, "reason": "resolution_error"}
