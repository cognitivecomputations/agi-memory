"""Durable clarification-question lifecycle helpers."""

from __future__ import annotations

import asyncio
import json
from typing import Any


def _coerce_json(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


async def question_timeout_seconds(pool: Any) -> int:
    """Read the live interactive wait limit with the schema default as fallback."""
    try:
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT COALESCE(get_config_int('chat.question_timeout_s'), 300)"
            )
        return max(1, min(int(value or 300), 86400))
    except Exception:
        return 300


async def wait_for_agent_question_answer(
    pool: Any,
    question_id: str,
    *,
    timeout_seconds: int,
    poll_interval: float = 0.25,
) -> dict[str, Any]:
    """Wait for one durable answer, then claim it for the paused turn."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.01, float(timeout_seconds))
    try:
        while True:
            async with pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT claim_agent_question_answer($1::uuid)", question_id
                )
            result = _coerce_json(raw)
            if result.get("status") != "pending":
                return result
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(max(0.01, poll_interval), remaining))
    except asyncio.CancelledError:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT supersede_agent_question($1::uuid, 'turn_cancelled')",
                    question_id,
                )
        finally:
            raise

    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT timeout_agent_question($1::uuid)", question_id
        )
    return _coerce_json(raw)


async def answer_agent_question(
    pool: Any,
    question_id: str,
    *,
    answer: str | None = None,
    choice_index: int | None = None,
    channel: str,
    actor: str | None = None,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT answer_agent_question($1::uuid, $2, $3, $4, $5)",
            question_id,
            answer,
            choice_index,
            channel,
            actor,
        )
    return _coerce_json(raw)


async def resolve_agent_question_from_inbound(
    pool: Any,
    *,
    channel: str,
    channel_id: str,
    actor: str,
    text: str,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT try_resolve_agent_question_from_inbound($1, $2, $3, $4)",
            channel,
            channel_id,
            actor,
            text,
        )
    return _coerce_json(raw)


def render_channel_question(payload: dict[str, Any]) -> str:
    """Render the shared numbered fallback for text-only channel surfaces."""
    prompt = str(payload.get("prompt") or "I need your input.").strip()
    question_id = str(payload.get("id") or "")
    code = question_id.replace("-", "")[:8].upper()
    choices = [
        str(item).strip()
        for item in payload.get("choices", [])
        if str(item).strip()
    ][:4]
    allow_free_text = payload.get("allow_free_text") is not False
    lines = [f"Question {code}" if code else "Question", prompt]
    lines.extend(f"{index}. {choice}" for index, choice in enumerate(choices, 1))
    if allow_free_text:
        if choices:
            lines.append(f"{len(choices) + 1}. Other (type your answer)")
        else:
            lines.append("Type your answer.")
    if choices:
        suffix = "Reply with a number."
    else:
        suffix = "Reply with your answer."
    if code:
        suffix += f" If more than one question is waiting, include code {code}."
    lines.append(suffix)
    return "\n".join(lines)
