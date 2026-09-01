"""Clarification tool: pause a live turn or file an asynchronous question."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)


def _uuid_or_none(value: Any) -> str | None:
    try:
        return str(UUID(str(value))) if value else None
    except (TypeError, ValueError, AttributeError):
        return None


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


class AskUserHandler(ToolHandler):
    """Ask for missing information without guessing."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ask_user",
            description=(
                "Ask the user one necessary clarification and use their answer in this task. "
                "In a live chat, CLI, or channel turn this pauses until they answer; during a "
                "heartbeat it files and delivers the question for a later beat. Provide at most "
                "four concise choices. Free text is available by default."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The specific question the user needs to answer.",
                        "minLength": 1,
                        "maxLength": 2000,
                    },
                    "choices": {
                        "type": "array",
                        "description": "Zero to four concise answer choices.",
                        "items": {"type": "string", "minLength": 1, "maxLength": 200},
                        "maxItems": 4,
                        "default": [],
                    },
                    "allow_free_text": {
                        "type": "boolean",
                        "description": "Allow an answer outside the listed choices.",
                        "default": True,
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=False,
            supports_parallel=False,
            internal=True,
            execution_timeout_seconds=None,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if context.registry is None:
            return ToolResult.error_result(
                "ask_user requires a database-backed tool registry",
                ToolErrorType.EXECUTION_FAILED,
            )
        prompt = str(arguments.get("prompt") or "").strip()
        choices = arguments.get("choices", [])
        allow_free_text = arguments.get("allow_free_text", True) is not False
        wait_for_answer = context.tool_context == ToolContext.CHAT

        from services.agent_questions import (
            question_timeout_seconds,
            wait_for_agent_question_answer,
        )

        timeout_seconds = await question_timeout_seconds(context.registry.pool)
        try:
            async with context.registry.pool.acquire() as conn:
                raw = await conn.fetchval(
                    """
                    SELECT create_agent_question(
                        $1::uuid, $2::uuid, $3, $4, $5::jsonb,
                        $6, $7, $8, $9::jsonb
                    )
                    """,
                    _uuid_or_none(context.session_id),
                    _uuid_or_none(context.heartbeat_id),
                    context.surface or context.tool_context.value,
                    prompt,
                    json.dumps(choices),
                    allow_free_text,
                    wait_for_answer,
                    timeout_seconds,
                    json.dumps(
                        {
                            "call_id": context.call_id,
                            "tool_context": context.tool_context.value,
                            "is_group": context.is_group,
                        }
                    ),
                )
            question = _json_object(raw)
        except Exception as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)

        question_id = str(question.get("id") or "")
        if not question_id:
            return ToolResult.error_result(
                "The question could not be filed. Try asking it in conversation text.",
                ToolErrorType.EXECUTION_FAILED,
            )

        if not wait_for_answer:
            return ToolResult.success_result(
                {
                    **question,
                    "status": "pending",
                    "message": "asked; not yet answered",
                    "next_step": "End this heartbeat cleanly. A later heartbeat will receive the answer.",
                },
                display_output="Asked through the outbox; waiting for a later heartbeat.",
            )

        await context.emit_event(
            "question",
            {
                "kind": "question",
                **question,
                "timeout_seconds": timeout_seconds,
            },
        )
        result = await wait_for_agent_question_answer(
            context.registry.pool,
            question_id,
            timeout_seconds=timeout_seconds,
        )
        if result.get("status") == "answered":
            return ToolResult.success_result(
                {
                    **result,
                    "message": "The user answered. Continue the same task using this answer.",
                },
                display_output="The user answered the clarification question.",
            )
        return ToolResult.success_result(
            {
                **result,
                "status": "timed_out",
                "answer": None,
                "message": (
                    "no answer — proceed on your best judgment and say which way you went"
                ),
            },
            display_output="No answer arrived before the clarification timeout.",
        )


def create_question_tools() -> list[ToolHandler]:
    return [AskUserHandler()]
