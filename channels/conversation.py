"""
Hexis Channel System - Conversation Handler

Routes inbound channel messages through the memory-enriched conversation
pipeline and manages per-sender session state in the database.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from .base import ChannelMessage, chunk_text

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# Session history limits, energy costs, and rate limits live in the DB
# (prepare_channel_turn / finalize_channel_turn read channel.* config).


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _attachment_payload(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return str(value)


def _channel_message_payload(msg: ChannelMessage) -> dict[str, Any]:
    return {
        "channel_type": msg.channel_type,
        "channel_id": msg.channel_id,
        "sender_id": msg.sender_id,
        "sender_name": msg.sender_name,
        "content": msg.content,
        "message_id": msg.message_id,
        "reply_to_id": msg.reply_to_id,
        "thread_id": msg.thread_id,
        "attachments": [_attachment_payload(att) for att in msg.attachments],
        "metadata": dict(msg.metadata or {}),
        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
        "is_group": msg.is_group,
    }


def _operator_context(msg: ChannelMessage) -> dict[str, Any]:
    return {
        "channel_type": msg.channel_type,
        "channel_id": msg.channel_id,
        "sender_id": msg.sender_id,
        "sender_name": msg.sender_name,
        "raw_text": msg.content,
        "metadata": {
            "message_id": msg.message_id,
            "platform_message_id": msg.message_id,
            "thread_id": msg.thread_id,
        },
        "reason": "identity_verified_channel_turn",
    }


async def _prepare_channel_turn_db(
    conn: asyncpg.Connection,
    msg: ChannelMessage,
) -> dict[str, Any]:
    raw = await conn.fetchval(
        "SELECT prepare_channel_turn($1::jsonb)",
        json.dumps(_channel_message_payload(msg), default=str),
    )
    result = _coerce_json(raw)
    return result if isinstance(result, dict) else {}


async def _finalize_channel_turn_db(
    conn: asyncpg.Connection,
    *,
    session_id: str,
    user_text: str,
    assistant_text: str,
    history: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    platform_message_id: str | None = None,
) -> dict[str, Any]:
    raw = await conn.fetchval(
        "SELECT finalize_channel_turn($1::uuid, $2::text, $3::text, $4::jsonb)",
        session_id,
        user_text,
        assistant_text,
        json.dumps(
            {
                "history": history,
                "metadata": metadata or {},
                "platform_message_id": platform_message_id,
            }
        ),
    )
    result = _coerce_json(raw)
    return result if isinstance(result, dict) else {}


# The former Python channel-turn helpers (energy check, session upsert,
# history flush, message log) were deleted: the DB owns the turn lifecycle
# via prepare_channel_turn / finalize_channel_turn /
# flush_channel_history_to_memory (db/34_functions_chat_channel.sql).


async def process_channel_message(
    msg: ChannelMessage,
    pool: asyncpg.Pool,
    *,
    max_message_length: int = 4000,
    adapter: Any | None = None,
    is_operator: bool = False,
) -> list[str]:
    """
    Process an inbound channel message through the conversation pipeline.

    1. Load/create session (conversation history per sender)
    2. Delegate to services.chat.chat_turn() for enrichment + LLM + tools + memory
    3. Update session with new history
    4. Log messages for audit
    5. Chunk response for channel limits

    Args:
        msg: Normalized channel message.
        pool: Database connection pool.
        max_message_length: Channel's max message length for chunking.

    Returns:
        List of response text chunks to send back.
    """
    from core.agent_api import db_dsn_from_env
    from core.llm_config import load_llm_config

    try:
        async with pool.acquire() as conn:
            prepared = await _prepare_channel_turn_db(conn, msg)
            if not prepared.get("allowed"):
                return [prepared.get("rejection") or "I can't respond right now."]

            session_id = str(prepared["session_id"])
            history = (
                prepared.get("history")
                if isinstance(prepared.get("history"), list)
                else []
            )

            # Load LLM config from DB
            llm_config = await load_llm_config(
                conn, "llm.chat", fallback_key="llm.heartbeat"
            )

        # Record channel event for audit trail (record-and-dispatch)
        try:
            from core.gateway import EventSource, Gateway

            gateway = Gateway(pool)
            await gateway.record(
                EventSource.CHANNEL,
                f"channel:{msg.channel_type}:{msg.channel_id}:{msg.sender_id}",
                {"message": msg.content[:500], "sender": msg.sender_name},
            )
        except Exception:
            logger.debug("Gateway record failed (non-fatal)", exc_info=True)

        # Build DSN for chat_turn (it manages its own connections)
        dsn = db_dsn_from_env()

        # Inject attachment context so the LLM knows about media
        user_content = msg.content
        if msg.attachments:
            from .media import Attachment

            descs = []
            for att in msg.attachments:
                if isinstance(att, Attachment):
                    descs.append(att.describe())
                elif isinstance(att, dict):
                    descs.append(
                        str(att.get("filename") or att.get("url", "attachment"))
                    )
            if descs:
                attachment_note = "[User attached: " + "; ".join(descs) + "]"
                user_content = (
                    f"{attachment_note}\n\n{user_content}"
                    if user_content
                    else attachment_note
                )

        # Run the conversation turn
        from services.chat import chat_turn

        # The channel_sessions UUID from prepare_channel_turn (#71): the old
        # "channel:type:id:sender" string failed the UUID parse downstream, so
        # every unit landed with session_id NULL.
        async def on_agent_event(event: Any) -> None:
            if adapter is None:
                return
            from core.agent_loop import AgentEvent

            if event.event != AgentEvent.QUESTION:
                return
            from .presentation import agent_question_presentation

            await adapter.send_presentation(
                msg.channel_id,
                agent_question_presentation(dict(event.data)),
                reply_to=msg.message_id,
                thread_id=msg.thread_id,
            )

        result = await chat_turn(
            user_message=user_content,
            history=history,
            llm_config=llm_config,
            dsn=dsn,
            session_id=session_id,
            pool=pool,
            user_label=msg.sender_name,
            is_group=msg.is_group,
            trusted_operator=is_operator,
            operator_context=_operator_context(msg),
            surface=msg.channel_type,
            on_event=on_agent_event if adapter is not None else None,
        )

        assistant_text = result.get("assistant", "")
        new_history = result.get("history", [])

        async with pool.acquire() as conn:
            await _finalize_channel_turn_db(
                conn,
                session_id=session_id,
                user_text=user_content,
                assistant_text=assistant_text,
                history=new_history,
                metadata={"channel_type": msg.channel_type},
            )

        # Chunk for channel limits
        if not assistant_text:
            return ["I processed your message but have no response to give."]

        return chunk_text(assistant_text, max_message_length)

    except Exception:
        logger.exception(
            "Error processing channel message from %s/%s",
            msg.channel_type,
            msg.sender_id,
        )
        return [
            "Sorry, I encountered an error processing your message. Please try again."
        ]


async def stream_channel_message(
    msg: ChannelMessage,
    pool: asyncpg.Pool,
    adapter: Any,
    *,
    is_operator: bool = False,
) -> str | None:
    """
    Process an inbound channel message with streaming edit-in-place delivery.

    Uses StreamCoalescer to progressively edit a message as tokens arrive.
    Falls back to process_channel_message() if streaming fails.

    Args:
        msg: Normalized channel message.
        pool: Database connection pool.
        adapter: The channel adapter (must support edit_message).

    Returns:
        The platform message ID of the final response, or None.
    """
    from channels.streaming import StreamCoalescer
    from core.agent_api import db_dsn_from_env
    from core.llm_config import load_llm_config
    from services.chat import _hydrate_chat_history, stream_chat_turn

    try:
        async with pool.acquire() as conn:
            prepared = await _prepare_channel_turn_db(conn, msg)
            if not prepared.get("allowed"):
                await adapter.send(
                    msg.channel_id,
                    prepared.get("rejection") or "I can't respond right now.",
                    reply_to=msg.message_id,
                )
                return None

            session_id = str(prepared["session_id"])
            history = (
                prepared.get("history")
                if isinstance(prepared.get("history"), list)
                else []
            )
            llm_config = await load_llm_config(
                conn, "llm.chat", fallback_key="llm.heartbeat"
            )

        # Record channel event for audit trail (record-and-dispatch)
        try:
            from core.gateway import EventSource, Gateway

            gateway = Gateway(pool)
            await gateway.record(
                EventSource.CHANNEL,
                f"channel:{msg.channel_type}:{msg.channel_id}:{msg.sender_id}",
                {
                    "message": msg.content[:500],
                    "sender": msg.sender_name,
                    "streamed": True,
                },
            )
        except Exception:
            logger.debug("Gateway record failed (non-fatal)", exc_info=True)

        dsn = db_dsn_from_env()

        # Inject attachment context
        user_content = msg.content
        if msg.attachments:
            from .media import Attachment

            descs = []
            for att in msg.attachments:
                if isinstance(att, Attachment):
                    descs.append(att.describe())
                elif isinstance(att, dict):
                    descs.append(
                        str(att.get("filename") or att.get("url", "attachment"))
                    )
            if descs:
                attachment_note = "[User attached: " + "; ".join(descs) + "]"
                user_content = (
                    f"{attachment_note}\n\n{user_content}"
                    if user_content
                    else attachment_note
                )

        coalescer = StreamCoalescer(
            adapter,
            msg.channel_id,
            reply_to=msg.message_id,
            thread_id=msg.thread_id,
        )

        collected: list[str] = []

        async def present_question(payload: dict[str, Any]) -> None:
            from .presentation import agent_question_presentation

            await adapter.send_presentation(
                msg.channel_id,
                agent_question_presentation(payload),
                reply_to=msg.message_id,
                thread_id=msg.thread_id,
            )

        async for token in stream_chat_turn(
            user_message=user_content,
            history=history,
            llm_config=llm_config,
            dsn=dsn,
            session_id=session_id,
            pool=pool,
            user_label=msg.sender_name,
            is_group=msg.is_group,
            trusted_operator=is_operator,
            operator_context=_operator_context(msg),
            surface=msg.channel_type,
            on_question=present_question,
        ):
            collected.append(token)
            await coalescer.push(token)

        message_id = await coalescer.flush()
        assistant_text = "".join(collected)

        # Update session and log
        fallback_history = [
            *history,
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ]
        new_history = await _hydrate_chat_history(pool, session_id, fallback_history)

        async with pool.acquire() as conn:
            await _finalize_channel_turn_db(
                conn,
                session_id=session_id,
                user_text=user_content,
                assistant_text=assistant_text,
                history=new_history,
                platform_message_id=message_id,
                metadata={"channel_type": msg.channel_type, "streamed": True},
            )

        return message_id

    except Exception:
        logger.exception(
            "Streaming failed for %s/%s, falling back to chunked",
            msg.channel_type,
            msg.sender_id,
        )
        # Fall back to non-streaming
        chunks = await process_channel_message(
            msg,
            pool,
            max_message_length=adapter.capabilities.max_message_length,
            adapter=adapter,
            is_operator=is_operator,
        )
        reply_to = msg.message_id
        for i, chunk in enumerate(chunks):
            try:
                await adapter.send(
                    msg.channel_id,
                    chunk,
                    reply_to=reply_to if i == 0 else None,
                    thread_id=msg.thread_id,
                )
            except Exception:
                break
        return None
