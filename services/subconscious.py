from __future__ import annotations

import json
import logging
from typing import Any

from core.llm_config import load_llm_config
from core.llm_json import chat_json
from core.subconscious import get_subconscious_context
from services.prompt_resources import load_subconscious_maintenance_prompt

logger = logging.getLogger(__name__)


def _coerce_json(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


async def _build_context(conn) -> dict[str, Any]:
    raw = await get_subconscious_context(conn)
    context = _coerce_json(raw) if raw is not None else {}
    return context if isinstance(context, dict) else {}


def _bounded_context_json(context: dict[str, Any], max_chars: int = 12000) -> str:
    payload = {"task": "maintenance_review", **context}

    def encode() -> str:
        return json.dumps(payload, default=str, ensure_ascii=False)

    encoded = encode()
    for key in (
        "recent_memories",
        "emotional_triggers",
        "relationships",
        "self_model",
        "worldview",
    ):
        values = payload.get(key)
        while len(encoded) > max_chars and isinstance(values, list) and values:
            values.pop()
            encoded = encode()
    if len(encoded) > max_chars:
        for key in (
            "active_transformations",
            "transformations_ready",
            "contradictions",
        ):
            payload.pop(key, None)
            encoded = encode()
            if len(encoded) <= max_chars:
                break
    if len(encoded) > max_chars:
        for key in tuple(payload):
            if key in {"task", "emotional_state"}:
                continue
            payload.pop(key, None)
            encoded = encode()
            if len(encoded) <= max_chars:
                break
    return encoded


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


async def run_subconscious_decider(conn) -> dict[str, Any]:
    llm_config = await load_llm_config(conn, "llm.subconscious", fallback_key="llm.heartbeat")
    context = await _build_context(conn)
    user_prompt = f"Context (JSON):\n{_bounded_context_json(context)}"
    try:
        doc, raw = await chat_json(
            llm_config=llm_config,
            messages=[
                {"role": "system", "content": load_subconscious_maintenance_prompt().strip()},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1800,
            response_format={"type": "json_object"},
            fallback={},
        )
    except Exception as exc:
        return {"skipped": True, "reason": str(exc)}

    if not isinstance(doc, dict):
        doc = {}

    try:
        result_raw = await conn.fetchval(
            "SELECT apply_subconscious_decider_result($1::jsonb, $2::jsonb)",
            json.dumps(doc),
            json.dumps(_jsonable(raw)),
        )
        result = _coerce_json(result_raw)
        return dict(result) if isinstance(result, dict) else {"applied": {}, "dopamine": {}, "raw_response": raw}
    except Exception as exc:
        logger.debug("Subconscious DB result application failed: %s", exc)
        return {"skipped": True, "reason": str(exc), "raw_response": raw}
