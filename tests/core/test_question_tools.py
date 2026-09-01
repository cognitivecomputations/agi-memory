from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.questions import AskUserHandler


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _context(db_pool, *, tool_context=ToolContext.CHAT, event_callback=None):
    registry = MagicMock()
    registry.pool = db_pool
    return ToolExecutionContext(
        tool_context=tool_context,
        call_id=f"call-{uuid.uuid4().hex[:8]}",
        session_id=str(uuid.uuid4()) if tool_context == ToolContext.CHAT else None,
        heartbeat_id=str(uuid.uuid4()) if tool_context == ToolContext.HEARTBEAT else None,
        surface="api" if tool_context == ToolContext.CHAT else "heartbeat",
        event_callback=event_callback,
        registry=registry,
    )


async def test_ask_user_emits_question_and_returns_exact_answer(db_pool):
    events: list[tuple[str, dict]] = []

    async def answer_on_event(event: str, payload: dict):
        events.append((event, payload))
        async with db_pool.acquire() as conn:
            await conn.fetchval(
                "SELECT answer_agent_question($1::uuid, NULL, 2, 'test', 'test-user')",
                payload["id"],
            )

    result = await AskUserHandler().execute(
        {
            "prompt": "Which contract?",
            "choices": ["Manning", "Hartford"],
            "allow_free_text": True,
        },
        _context(db_pool, event_callback=answer_on_event),
    )

    assert result.success is True
    assert events[0][0] == "question"
    assert events[0][1]["prompt"] == "Which contract?"
    assert result.output["status"] == "answered"
    assert result.output["answer"] == "Hartford"
    assert result.energy_spent == 0


async def test_heartbeat_ask_is_inert_and_returns_without_waiting(db_pool):
    result = await AskUserHandler().execute(
        {"prompt": "Should I prepare a review?", "choices": ["Yes", "No"]},
        _context(db_pool, tool_context=ToolContext.HEARTBEAT),
    )

    assert result.success is True
    assert result.output["status"] == "pending"
    assert result.output["message"] == "asked; not yet answered"
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, outbox_message_id FROM agent_questions WHERE id = $1::uuid",
            result.output["id"],
        )
        envelope = await conn.fetchval(
            "SELECT envelope FROM outbox_messages WHERE id = $1::uuid",
            row["outbox_message_id"],
        )
    envelope = json.loads(envelope) if isinstance(envelope, str) else envelope
    assert row["status"] == "pending"
    assert envelope["payload"]["delivery"]["question_id"] == result.output["id"]


async def test_ask_user_is_free_and_available_in_both_modes():
    spec = AskUserHandler().spec
    assert spec.energy_cost == 0
    assert spec.supports_parallel is False
    assert spec.execution_timeout_seconds is None
    assert spec.allowed_contexts == {ToolContext.CHAT, ToolContext.HEARTBEAT}
