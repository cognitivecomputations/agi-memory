"""
Hexis Channel System - Channel Manager

Orchestrates multiple channel adapters, routing inbound messages to the
conversation handler and providing a unified send interface for outbound.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
from typing import Any, TYPE_CHECKING

from core.integration_reliability import bounded_text, compute_backoff_seconds

from .base import ChannelAdapter, ChannelMessage
from .commands import CommandRegistry, parse_command
from .conversation import process_channel_message, stream_channel_message
from .presentation import MessagePresentation

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

ADAPTER_RESTART_INITIAL_S = float(os.getenv("CHANNEL_ADAPTER_RESTART_INITIAL_S", "5.0"))
ADAPTER_RESTART_MAX_S = float(os.getenv("CHANNEL_ADAPTER_RESTART_MAX_S", "300.0"))
TYPING_COOLDOWN_INITIAL_S = float(
    os.getenv("CHANNEL_TYPING_COOLDOWN_INITIAL_S", "30.0")
)
TYPING_COOLDOWN_MAX_S = float(os.getenv("CHANNEL_TYPING_COOLDOWN_MAX_S", "300.0"))


def _non_recoverable_adapter_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    signals = (
        "not configured",
        "not found",
        "missing",
        "required",
        "invalid token",
        "unauthorized",
        "forbidden",
        "authentication",
        "auth",
    )
    return any(signal in text for signal in signals)


class ChannelManager:
    """
    Manages the lifecycle of channel adapters and routes messages.

    Usage:
        manager = ChannelManager(pool)
        manager.register(discord_adapter)
        manager.register(telegram_adapter)
        await manager.start_all()
        ...
        await manager.stop_all()
    """

    def __init__(
        self, pool: asyncpg.Pool, *, commands: CommandRegistry | None = None
    ) -> None:
        self._pool = pool
        self._adapters: dict[str, ChannelAdapter] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._message_tasks: set[asyncio.Task[None]] = set()
        self._conversation_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self._running = False
        self._commands = commands or CommandRegistry()
        self._typing_failures: dict[tuple[str, str], int] = {}
        self._typing_blocked_until: dict[tuple[str, str], float] = {}

    @property
    def adapters(self) -> dict[str, ChannelAdapter]:
        """Registered adapters by channel_type."""
        return dict(self._adapters)

    @property
    def pool(self) -> asyncpg.Pool:
        """Database pool shared with adapters that handle durable interactions."""
        return self._pool

    def register(self, adapter: ChannelAdapter) -> None:
        """Register a channel adapter."""
        ctype = adapter.channel_type
        if ctype in self._adapters:
            logger.warning("Channel adapter %r already registered, replacing", ctype)
        self._adapters[ctype] = adapter
        logger.info("Registered channel adapter: %s", ctype)

    async def ensure_started(self, adapter: ChannelAdapter) -> bool:
        """
        Register and start an adapter if it isn't already registered.

        Returns:
            True if the adapter was newly registered (and started if the manager
            is running), False if it already existed.
        """
        ctype = adapter.channel_type
        if ctype in self._adapters:
            return False

        self.register(adapter)

        # If the manager has already been started, ensure the new adapter
        # launches immediately.
        if self._running:
            await self._start_adapter(ctype, adapter)

        return True

    async def start_all(self) -> None:
        """Start all registered adapters."""
        self._running = True
        for ctype, adapter in self._adapters.items():
            await self._start_adapter(ctype, adapter)

    async def _start_adapter(self, ctype: str, adapter: ChannelAdapter) -> None:
        """Start a single adapter with error isolation."""
        try:
            await self._record_runtime_status(
                ctype, "starting", configured=True, running=True
            )

            async def on_message(msg: ChannelMessage) -> None:
                # Adapter receive loops must remain free to accept the reply to
                # a question that an earlier turn is awaiting. Ordinary turns
                # are still serialized per conversation below.
                message_task = asyncio.create_task(
                    self._handle_message(msg),
                    name=f"channel-message-{msg.channel_type}-{msg.message_id}",
                )
                self._message_tasks.add(message_task)
                message_task.add_done_callback(self._message_task_done)

            # Each adapter's start() runs its own event loop (blocking).
            # Wrap in a task so they run concurrently.
            task = asyncio.create_task(
                self._run_adapter(ctype, adapter, on_message),
                name=f"channel-{ctype}",
            )
            self._tasks[ctype] = task
            logger.info("Started channel adapter: %s", ctype)

        except Exception as exc:
            logger.exception("Failed to start channel adapter: %s", ctype)
            await self._record_runtime_status(
                ctype,
                "error",
                configured=True,
                running=False,
                error=str(exc),
            )

    def _message_task_done(self, task: asyncio.Task[None]) -> None:
        self._message_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Channel message task failed: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _run_adapter(
        self, ctype: str, adapter: ChannelAdapter, on_message
    ) -> None:
        """Run an adapter with restart-on-crash."""
        consecutive_failures = 0
        while self._running:
            try:
                await self._record_runtime_status(
                    ctype, "running", configured=True, running=True
                )
                await self._record_presence(
                    ctype, None, "online", metadata={"source": "channel_manager"}
                )
                await adapter.start(on_message)
                consecutive_failures = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if _non_recoverable_adapter_error(exc):
                    detail = bounded_text(exc, limit=500)
                    logger.error(
                        "Channel adapter %s paused after non-recoverable startup error: %s",
                        ctype,
                        detail,
                    )
                    await self._record_runtime_status(
                        ctype,
                        "paused",
                        configured=True,
                        running=False,
                        error=(
                            f"{detail}. Fix this channel's configuration or credentials, "
                            "then restart the channel worker."
                        ),
                        metadata={
                            "source": "channel_manager",
                            "error_kind": "configuration_or_auth",
                            "recoverable": False,
                        },
                    )
                    break
                consecutive_failures += 1
                delay_s = compute_backoff_seconds(
                    consecutive_failures,
                    initial_delay=ADAPTER_RESTART_INITIAL_S,
                    max_delay=ADAPTER_RESTART_MAX_S,
                    jitter=0.2,
                )
                logger.exception(
                    "Channel adapter %s crashed, restarting in %.1fs", ctype, delay_s
                )
                await self._record_runtime_status(
                    ctype,
                    "error",
                    configured=True,
                    running=False,
                    error=str(exc),
                    metadata={
                        "source": "channel_manager",
                        "consecutive_failures": consecutive_failures,
                        "restart_delay_s": delay_s,
                        "recoverable": True,
                    },
                )
                await asyncio.sleep(delay_s)
        await self._record_runtime_status(
            ctype, "stopped", configured=True, running=False
        )
        await self._record_presence(
            ctype, None, "offline", metadata={"source": "channel_manager"}
        )

    async def _record_runtime_status(
        self,
        channel_type: str,
        status: str,
        *,
        configured: bool,
        running: bool,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT record_channel_adapter_status($1, $2, $3, $4, $5, $6::jsonb)",
                    channel_type,
                    status,
                    configured,
                    running,
                    error,
                    json.dumps(metadata or {"source": "channel_manager"}),
                )
        except Exception:
            logger.debug(
                "Failed to record channel adapter runtime for %s",
                channel_type,
                exc_info=True,
            )

    async def _record_presence(
        self,
        channel_type: str,
        channel_id: str | None,
        presence_kind: str,
        *,
        direction: str = "system",
        sender_id: str | None = None,
        session_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int = 20,
    ) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT record_channel_presence($1, $2, $3, $4, $5, $6, $7::jsonb, $8)",
                    channel_type,
                    channel_id,
                    presence_kind,
                    direction,
                    sender_id,
                    session_key,
                    json.dumps(metadata or {"source": "channel_manager"}),
                    ttl_seconds,
                )
        except Exception:
            logger.debug(
                "Failed to record channel presence for %s", channel_type, exc_info=True
            )

    async def _handle_message(self, msg: ChannelMessage) -> None:
        """Handle an inbound message by routing to conversation handler."""
        adapter = self._adapters.get(msg.channel_type)
        if not adapter:
            logger.warning("No adapter for channel type: %s", msg.channel_type)
            return

        # Contact control is above every reply/purpose/budget decision.  It is
        # evaluated even for senders the conversation allowlist will only
        # observe, because promising STOP and then filtering it would be a
        # broken opt-out.  Ordinary inbound messages also replenish the
        # recipient's cadence budget here.
        try:
            from services.outbound_safety import handle_inbound_contact_control

            contact_control = await handle_inbound_contact_control(self._pool, msg)
        except Exception:
            contact_control = {}
            logger.warning(
                "Inbound contact-control evaluation failed for %s/%s; continuing without consuming the message",
                msg.channel_type,
                msg.sender_id,
                exc_info=True,
            )
        if contact_control.get("recognized"):
            try:
                from services.inbound_disposition import record_passive_observation

                await record_passive_observation(
                    self._pool,
                    message=msg,
                    disposition={
                        "audit_id": None,
                        "disposition": "observe",
                        "reason": f"contact_control_{contact_control.get('action')}",
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to preserve inbound contact-control source artifact",
                    exc_info=True,
                )
            if contact_control.get("acknowledge"):
                try:
                    await adapter.send(
                        msg.channel_id,
                        str(contact_control.get("acknowledgement") or "Understood."),
                        reply_to=msg.message_id,
                        thread_id=msg.thread_id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to send one-time contact-control acknowledgement"
                    )
            return

        disposition_result = await self._resolve_inbound_disposition(msg)
        if disposition_result is not None:
            disposition = str(
                disposition_result.get("disposition") or "observe"
            ).lower()
            if disposition == "drop":
                return
            if disposition != "engage" or not bool(
                disposition_result.get("reply_allowed")
            ):
                if disposition == "engage":
                    logger.warning(
                        "Inbound resolver attempted engage above its allowlist ceiling; retaining passive observation"
                    )
                    disposition_result = {
                        **disposition_result,
                        "disposition": "observe",
                        "reason": "reply_allowlist_belt",
                    }
                await self._record_disposition_observation(msg, disposition_result)
                return
            stripped = disposition_result.get("trigger_stripped_text")
            if isinstance(stripped, str):
                msg = dataclasses.replace(msg, content=stripped)
            is_operator = (
                bool(disposition_result.get("is_operator")) and not msg.is_group
            )
        else:
            # An adapter that was started in forward-all mode can outlive a
            # runtime kill-switch change. Its hint preserves the exact legacy
            # suppression until the worker is restarted.
            gate_hint = (msg.metadata or {}).get("gate_hint")
            if gate_hint:
                logger.info(
                    "Inbound disposition disabled; legacy adapter gate %s suppressed %s message",
                    gate_hint,
                    msg.channel_type,
                )
                return

            # I.3: Legacy per-channel user allowlisting.
            if not await self._check_user_allowed(msg):
                logger.debug(
                    "Ignoring message from non-allowed user %s on %s",
                    msg.sender_id,
                    msg.channel_type,
                )
                return

            from services.operator_policy_corrections import (
                channel_sender_is_operator,
            )

            is_operator = not msg.is_group and await channel_sender_is_operator(
                self._pool,
                channel_type=msg.channel_type,
                sender_id=msg.sender_id,
            )

        # Ordinary channel replies bypass both provider tools and the formal
        # outbox. Wrap this direct road per inbound turn so it still receives
        # a verified reply purpose, cross-channel STOP check, disclosure, and
        # durable ledger entry. The one-time STOP/START acknowledgement above
        # intentionally uses the raw adapter because STOP becomes effective
        # before its contractual acknowledgement is sent.
        from services.outbound_safety import GovernedReplyAdapter

        adapter = GovernedReplyAdapter(self._pool, adapter, msg)

        # Media is fetched only after the sender passes the channel allowlist.
        # The transcript then follows the same question/command/conversation
        # route as text the operator typed.
        from services.voice_notes import (
            enrich_message_with_voice_transcripts,
            is_audio_attachment,
        )

        has_audio = any(is_audio_attachment(item) for item in msg.attachments)
        if has_audio:
            try:
                msg = await enrich_message_with_voice_transcripts(
                    self._pool,
                    msg,
                    attachment_downloader=adapter.download_attachment,
                )
                voice_metadata = (msg.metadata or {}).get("voice_note") or {}
                if voice_metadata.get("fallback_note") and msg.content.strip():
                    await adapter.send(
                        msg.channel_id,
                        msg.content,
                        reply_to=msg.message_id,
                        thread_id=msg.thread_id,
                    )
                    return
            except Exception:
                logger.exception(
                    "Voice-note enrichment failed for %s", msg.channel_type
                )
                if not msg.content.strip():
                    await adapter.send(
                        msg.channel_id,
                        "I received the voice note, but transcription failed before I could read it. Open Settings → Voice notes, retry, or type the message.",
                        reply_to=msg.message_id,
                        thread_id=msg.thread_id,
                    )
                    return

        logger.info(
            "Channel message: %s/%s from %s: %s",
            msg.channel_type,
            msg.channel_id,
            msg.sender_name,
            msg.content[:80],
        )

        # A phone reply to an outstanding approval is a control-plane decision,
        # not conversation text. The DB verifies the configured operator identity
        # and requires an exact request code when more than one action is pending.
        if msg.channel_type in {"slack", "imessage"}:
            from services.operator_approval import (
                resolve_operator_approval_from_inbound,
            )

            approval = await resolve_operator_approval_from_inbound(
                self._pool,
                channel=msg.channel_type,
                actor=msg.sender_id,
                text=msg.content,
            )
            if approval.get("recognized"):
                await adapter.send(
                    msg.channel_id,
                    str(
                        approval.get("message")
                        or "Approval decision could not be recorded. Use the request code from the approval message."
                    ),
                    reply_to=msg.message_id,
                    thread_id=msg.thread_id,
                )
                return

        # A reply to ask_user resumes the exact paused turn. Session identity
        # scopes even a bare number to the person and room that saw the card.
        from services.agent_questions import resolve_agent_question_from_inbound

        try:
            question = await resolve_agent_question_from_inbound(
                self._pool,
                channel=msg.channel_type,
                channel_id=msg.channel_id,
                actor=msg.sender_id,
                text=msg.content,
            )
        except Exception:
            question = {}
            logger.debug(
                "Question reply lookup failed; treating input as conversation",
                exc_info=True,
            )
        if question.get("recognized"):
            await adapter.send(
                msg.channel_id,
                str(
                    question.get("message")
                    or "That question could not be answered. Use the number or code shown with it."
                ),
                reply_to=msg.message_id,
                thread_id=msg.thread_id,
            )
            return

        # Memory fade, learning, and contradiction choices can change active
        # recall, so only the verified operator may consume one as
        # control-plane input. A private allowlisted contact is still ordinary
        # conversation here.
        if is_operator:
            from services.retention_surface import (
                resolve_memory_fade_review_from_inbound,
            )

            fade_review = await resolve_memory_fade_review_from_inbound(
                self._pool,
                channel=msg.channel_type,
                actor=msg.sender_id,
                text=msg.content,
            )
            if fade_review.get("recognized"):
                await adapter.send(
                    msg.channel_id,
                    str(
                        fade_review.get("message")
                        or "That memory-fade decision could not be recorded. Use the review code from the message."
                    ),
                    reply_to=msg.message_id,
                    thread_id=msg.thread_id,
                )
                return

            from services.skill_improvement import resolve_learning_review_from_inbound

            learning_review = await resolve_learning_review_from_inbound(
                self._pool,
                channel=msg.channel_type,
                actor=msg.sender_id,
                text=msg.content,
            )
            if learning_review.get("recognized"):
                await adapter.send(
                    msg.channel_id,
                    str(
                        learning_review.get("message")
                        or "That learning-review decision could not be recorded. Use the item code from the review message."
                    ),
                    reply_to=msg.message_id,
                    thread_id=msg.thread_id,
                )
                return

            from services.contradictions import resolve_contradiction_from_inbound

            contradiction = await resolve_contradiction_from_inbound(
                self._pool,
                channel=msg.channel_type,
                actor=msg.sender_id,
                text=msg.content,
            )
            if contradiction.get("recognized"):
                await adapter.send(
                    msg.channel_id,
                    str(
                        contradiction.get("message")
                        or "That contradiction decision could not be recorded. Use the case code from the review message."
                    ),
                    reply_to=msg.message_id,
                    thread_id=msg.thread_id,
                )
                return

        # Automation suggestions use explicit numbered/code replies. Only a
        # private conversation can consume one as control-plane input; a "1"
        # in a group remains ordinary conversation text.
        if not msg.is_group:
            from services.automation_suggestions import (
                resolve_automation_suggestion_from_inbound,
            )

            automation = await resolve_automation_suggestion_from_inbound(
                self._pool,
                channel=msg.channel_type,
                actor=msg.sender_id,
                text=msg.content,
            )
            if automation.get("recognized"):
                await adapter.send(
                    msg.channel_id,
                    str(
                        automation.get("message")
                        or "That automation decision could not be recorded. Use the code from the suggestion message."
                    ),
                    reply_to=msg.message_id,
                    thread_id=msg.thread_id,
                )
                return

        # Check for slash commands
        parsed = parse_command(msg.content)
        if parsed:
            cmd_name, cmd_args = parsed
            if self._commands.has(cmd_name):
                response = await self._commands.execute(cmd_name, cmd_args, self._pool)
                if response:
                    try:
                        await adapter.send(
                            msg.channel_id,
                            response,
                            reply_to=msg.message_id,
                            thread_id=msg.thread_id,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to send command response for /%s", cmd_name
                        )
                return

        conversation_key = (msg.channel_type, msg.channel_id, msg.sender_id)
        conversation_lock = self._conversation_locks.setdefault(
            conversation_key, asyncio.Lock()
        )
        async with conversation_lock:
            await self._run_conversation_turn(
                msg,
                adapter,
                is_operator=is_operator,
            )

    async def _resolve_inbound_disposition(
        self, msg: ChannelMessage
    ) -> dict[str, Any] | None:
        """Return a DB-owned disposition, or None for exact legacy routing."""
        from services.inbound_disposition import (
            is_disposition_enabled,
            resolve_disposition,
        )

        if not await is_disposition_enabled(self._pool):
            return None
        metadata = dict(msg.metadata or {})
        metadata.update(
            channel_id=msg.channel_id,
            platform_message_id=msg.message_id,
            reply_to_id=msg.reply_to_id,
            thread_id=msg.thread_id,
            is_group=msg.is_group,
            has_attachments=bool(msg.attachments),
        )
        try:
            return await resolve_disposition(
                self._pool,
                channel_type=msg.channel_type,
                sender_id=msg.sender_id,
                session_id=msg.channel_id,
                text=msg.content,
                metadata=metadata,
            )
        except Exception:
            logger.warning(
                "Inbound disposition bridge raised for %s/%s; retaining passive observation",
                msg.channel_type,
                msg.sender_id,
                exc_info=True,
            )
            return {
                "disposition": "observe",
                "reason": "resolver_exception",
                "ambiguous": False,
                "is_operator": False,
                "reply_allowed": False,
                "audit_id": None,
            }

    async def _record_disposition_observation(
        self,
        msg: ChannelMessage,
        disposition: dict[str, Any],
    ) -> None:
        """Persist observe/wake without producing an unsolicited reply."""
        if bool(disposition.get("is_operator")):
            from services.operator_policy_corrections import (
                capture_operator_policy_correction,
            )

            await capture_operator_policy_correction(
                self._pool,
                channel_type=msg.channel_type,
                channel_id=msg.channel_id,
                sender_id=msg.sender_id,
                sender_name=msg.sender_name,
                text=msg.content,
                is_operator=True,
                disposition=str(disposition.get("disposition") or "observe"),
                reason=str(
                    disposition.get("classifier_label")
                    or disposition.get("reason")
                    or "inbound_disposition"
                ),
                metadata={
                    **dict(msg.metadata or {}),
                    "inbound_disposition_event_id": disposition.get("audit_id"),
                },
            )

        from services.inbound_disposition import record_passive_observation

        await record_passive_observation(
            self._pool,
            message=msg,
            disposition=disposition,
        )

    async def _run_conversation_turn(
        self,
        msg: ChannelMessage,
        adapter: ChannelAdapter,
        *,
        is_operator: bool = False,
    ) -> None:
        """Run one ordinary turn without blocking adapter message intake."""

        # Send typing indicator while processing
        if adapter.capabilities.typing_indicator and not self._typing_cooldown_active(
            msg.channel_type, msg.channel_id
        ):
            try:
                await adapter.send_typing(msg.channel_id)
                self._record_typing_success(msg.channel_type, msg.channel_id)
                await self._record_presence(
                    msg.channel_type,
                    msg.channel_id,
                    "typing",
                    direction="outbound",
                    sender_id=msg.sender_id,
                    session_key=f"{msg.channel_type}:{msg.channel_id}:{msg.sender_id}",
                    metadata={
                        "source": "channel_manager",
                        "reply_to": msg.message_id,
                        "thread_id": msg.thread_id,
                    },
                    ttl_seconds=15,
                )
            except Exception as exc:
                delay_s = self._record_typing_failure(msg.channel_type, msg.channel_id)
                logger.debug(
                    "Typing indicator failed for %s/%s; cooling down for %.1fs: %s",
                    msg.channel_type,
                    msg.channel_id,
                    delay_s,
                    bounded_text(exc, limit=200),
                )

        # Use streaming for channels that support edit_message
        if adapter.capabilities.edit_message:
            if is_operator:
                await stream_channel_message(
                    msg,
                    self._pool,
                    adapter,
                    is_operator=True,
                )
            else:
                await stream_channel_message(msg, self._pool, adapter)
        else:
            # Fall back to chunked delivery
            max_len = adapter.capabilities.max_message_length
            process_kwargs: dict[str, Any] = {
                "max_message_length": max_len,
                "adapter": adapter,
            }
            if is_operator:
                process_kwargs["is_operator"] = True
            response_chunks = await process_channel_message(
                msg,
                self._pool,
                **process_kwargs,
            )

            reply_to = msg.message_id
            for i, chunk in enumerate(response_chunks):
                try:
                    await adapter.send(
                        msg.channel_id,
                        chunk,
                        reply_to=reply_to if i == 0 else None,
                        thread_id=msg.thread_id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to send response chunk %d to %s/%s",
                        i,
                        msg.channel_type,
                        msg.channel_id,
                    )
                    break

    async def send(
        self,
        channel_type: str,
        channel_id: str,
        message: str | MessagePresentation,
        **kwargs: Any,
    ) -> str | None:
        """
        Send an outbound message to a specific channel.

        Used by heartbeat/outbox for proactive messaging.
        """
        adapter = self._adapters.get(channel_type)
        if not adapter:
            logger.error("No adapter for channel type: %s", channel_type)
            return None
        if isinstance(message, MessagePresentation):
            return await adapter.send_presentation(channel_id, message, **kwargs)
        return await adapter.send(channel_id, message, **kwargs)

    def _typing_cooldown_active(self, channel_type: str, channel_id: str) -> bool:
        key = (channel_type, channel_id)
        blocked_until = self._typing_blocked_until.get(key)
        if blocked_until is None:
            return False
        if time.monotonic() >= blocked_until:
            self._typing_blocked_until.pop(key, None)
            return False
        return True

    def _record_typing_success(self, channel_type: str, channel_id: str) -> None:
        key = (channel_type, channel_id)
        self._typing_failures.pop(key, None)
        self._typing_blocked_until.pop(key, None)

    def _record_typing_failure(self, channel_type: str, channel_id: str) -> float:
        key = (channel_type, channel_id)
        failures = self._typing_failures.get(key, 0) + 1
        self._typing_failures[key] = failures
        delay_s = compute_backoff_seconds(
            failures,
            initial_delay=TYPING_COOLDOWN_INITIAL_S,
            max_delay=TYPING_COOLDOWN_MAX_S,
            jitter=0.2,
        )
        self._typing_blocked_until[key] = time.monotonic() + delay_s
        return delay_s

    async def stop_all(self) -> None:
        """Stop all adapters gracefully."""
        self._running = False

        # Cancel all adapter tasks
        for ctype, task in self._tasks.items():
            task.cancel()

        # Wait for tasks to finish
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        # A live question can leave a message turn intentionally waiting. Stop
        # those turns too so their durable questions are superseded cleanly.
        for task in tuple(self._message_tasks):
            task.cancel()
        if self._message_tasks:
            await asyncio.gather(*tuple(self._message_tasks), return_exceptions=True)

        # Stop each adapter
        for ctype, adapter in self._adapters.items():
            try:
                await adapter.stop()
                await self._record_runtime_status(
                    ctype, "stopped", configured=True, running=False
                )
                await self._record_presence(
                    ctype, None, "offline", metadata={"source": "channel_manager"}
                )
                logger.info("Stopped channel adapter: %s", ctype)
            except Exception:
                logger.exception("Error stopping channel adapter: %s", ctype)

        self._tasks.clear()
        self._message_tasks.clear()
        self._conversation_locks.clear()
        logger.info("All channel adapters stopped")

    async def _check_user_allowed(self, msg: ChannelMessage) -> bool:
        """I.3: Check per-channel user allowlist from config.

        Config key: channel.{type}.allowed_users
        Value: JSON array of user IDs, or "*" to allow all.
        """
        try:
            async with self._pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT get_config_text($1)",
                    f"channel.{msg.channel_type}.allowed_users",
                )
            if raw is None or raw == "*":
                return True
            import json

            try:
                allowed = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                return True
            if isinstance(allowed, list):
                return msg.sender_id in [str(v) for v in allowed]
            return True
        except Exception:
            # Fail open — don't block messages if config lookup fails
            return True

    async def get_session_ttl(self, channel_type: str) -> int:
        """I.4: Get configurable session lifetime in seconds.

        Config key: channel.session_ttl or channel.{type}.session_ttl
        Default: 3600 (1 hour).
        """
        try:
            async with self._pool.acquire() as conn:
                # Check channel-specific TTL first, then global
                ttl = await conn.fetchval(
                    "SELECT get_config_int($1)",
                    f"channel.{channel_type}.session_ttl",
                )
                if ttl is not None:
                    return int(ttl)
                ttl = await conn.fetchval(
                    "SELECT get_config_int($1)",
                    "channel.session_ttl",
                )
                if ttl is not None:
                    return int(ttl)
        except Exception:
            pass
        return 3600  # Default 1 hour

    def status(self) -> list[dict[str, Any]]:
        """Return status of all registered adapters."""
        result = []
        for ctype, adapter in self._adapters.items():
            result.append(
                {
                    "channel_type": ctype,
                    "connected": adapter.is_connected,
                    "capabilities": {
                        "threads": adapter.capabilities.threads,
                        "reactions": adapter.capabilities.reactions,
                        "media": adapter.capabilities.media,
                        "typing_indicator": adapter.capabilities.typing_indicator,
                        "edit_message": adapter.capabilities.edit_message,
                        "max_message_length": adapter.capabilities.max_message_length,
                        "markdown_dialect": adapter.capabilities.markdown_dialect.value,
                    },
                }
            )
        return result
