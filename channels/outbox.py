"""
Hexis Channel System - Outbox Consumer

Subscribes to the RabbitMQ outbox queue and routes heartbeat-initiated
messages to the appropriate channel adapters. This enables proactive
messaging — the agent can reach out to users without waiting for inbound.

Delivery modes:
    - direct: use explicit target_channel + target_id from payload
    - last_active: find the sender's most recent channel session
    - broadcast: send to all active sessions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Any, TYPE_CHECKING

import requests

from core.integration_reliability import (
    IntegrationHttpError,
    bounded_text,
    format_provider_error,
    request_text_response,
)

from .presentation import (
    MessagePresentation,
    normalize_message_presentation,
    presentation_from_text,
    render_presentation,
)

if TYPE_CHECKING:
    import asyncpg
    from .manager import ChannelManager

logger = logging.getLogger(__name__)

RABBITMQ_MANAGEMENT_URL = os.getenv(
    "RABBITMQ_MANAGEMENT_URL",
    f"http://localhost:{os.getenv('RABBITMQ_MANAGEMENT_PORT', '45673')}",
).rstrip("/")
RABBITMQ_USER = os.getenv("RABBITMQ_USER") or os.getenv(
    "RABBITMQ_DEFAULT_USER", "hexis"
)
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD") or os.getenv(
    "RABBITMQ_DEFAULT_PASS", "hexis_password"
)
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_OUTBOX_QUEUE = os.getenv("RABBITMQ_OUTBOX_QUEUE", "hexis.outbox")
POLL_INTERVAL = float(os.getenv("OUTBOX_POLL_INTERVAL", "2.0"))
RECOVERED_DELIVERY_MARKER = (
    "Recovered delivery: Hexis restarted or lost confirmation during the previous "
    "send attempt, so this may duplicate an earlier message.\n\n"
)

# These envelopes are created by trusted control-plane functions specifically
# for the configured operator.  Generic ``channel_message`` envelopes are not
# on this list: an explicit channel target may be a third party and must carry
# its own backed purpose unless address resolution proves it is the operator.
_PRIMARY_USER_OUTBOX_KINDS = frozenset({"user", "operator_approval"})


def _is_primary_user_envelope(payload: dict[str, Any]) -> bool:
    return str(payload.get("_outbox_kind") or "") in _PRIMARY_USER_OUTBOX_KINDS


class ChannelOutboxConsumer:
    """
    Polls the RabbitMQ outbox queue and routes messages to channel adapters.

    Usage:
        consumer = ChannelOutboxConsumer(manager, pool)
        await consumer.start()  # blocks until stop()
    """

    def __init__(self, manager: ChannelManager, pool: asyncpg.Pool) -> None:
        self._manager = manager
        self._pool = pool
        self._running = False

    async def start(self) -> None:
        """Poll the outbox queue until stopped."""
        self._running = True
        logger.info("Outbox consumer started")
        while self._running:
            try:
                count = await self._poll()
                if count > 0:
                    logger.info("Processed %d outbox message(s)", count)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Outbox poll error")
            await asyncio.sleep(POLL_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    async def _poll(self, max_messages: int = 10) -> int:
        """Fetch and process messages from the outbox queue."""
        recovered = await self._recover_delivery_obligations()
        vhost = _vhost_path()
        try:
            resp = await _rmq_request(
                "POST",
                f"/api/queues/{vhost}/{requests.utils.quote(RABBITMQ_OUTBOX_QUEUE, safe='')}/get",
                payload={
                    "count": max_messages,
                    "ackmode": "ack_requeue_false",
                    "encoding": "auto",
                    "truncate": 50000,
                },
            )
            if resp.status_code != 200:
                return 0
            msgs = resp.json()
            if not isinstance(msgs, list):
                return 0
        except Exception:
            return 0

        processed = 0
        for msg in msgs:
            raw_payload = msg.get("payload")
            try:
                body = (
                    json.loads(raw_payload)
                    if isinstance(raw_payload, str)
                    else raw_payload
                )
            except Exception:
                continue

            if not isinstance(body, dict):
                continue

            try:
                await self._process_message(body)
                processed += 1
            except Exception:
                logger.exception(
                    "Failed to process outbox message: %s", str(body)[:200]
                )

        return processed + recovered

    async def _process_message(self, body: dict[str, Any]) -> None:
        """Route an outbox message to the appropriate channel."""
        kind = body.get("kind", "")
        payload = body.get("payload", {})
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {"content": payload}

        message, content = _resolve_payload_message(payload)
        if not content:
            return
        payload["_outbox_kind"] = kind

        delivery_mode = str(payload.get("delivery_mode") or "last_active")
        outbox_msg_id = str(body.get("id") or "")

        # I.2: Check for domain-based delivery routing from cron delivery info
        delivery_info = payload.get("delivery") or body.get("delivery")

        # Web-inbox tee: the dashboard is one more delivery endpoint hooked to
        # this queue, like any external system. Every user-bound message gets a
        # copy there (config-gated), so the UI can show it even when no chat
        # platform is configured. Silent deliveries stay silent everywhere.
        if not (
            isinstance(delivery_info, dict) and delivery_info.get("mode") == "silent"
        ):
            await self._deliver_web_inbox(body)
        # Explicit web_inbox delivery (#98): the tee above IS the delivery —
        # skip channel routing so e.g. an incubated memory never lands in a
        # group chat via last-active.
        if isinstance(delivery_info, dict) and delivery_info.get("mode") == "web_inbox":
            return
        if isinstance(delivery_info, dict) and delivery_info.get("mode") == "channel":
            # Override delivery to route to specific channel+topic
            payload["target_channel"] = delivery_info.get("channel", "")
            payload["target_id"] = delivery_info.get(
                "target_id", payload.get("target_id", "")
            )
            payload["thread_id"] = delivery_info.get("topic", "")
            delivery_mode = "direct"
        elif isinstance(delivery_info, dict) and delivery_info.get("mode") == "webhook":
            await self._deliver_webhook(content, payload, delivery_info, outbox_msg_id)
            return
        elif isinstance(delivery_info, dict) and delivery_info.get("mode") == "silent":
            # Silent: log only, no notification
            logger.info("Silent delivery (suppressed): %s", content[:100])
            return

        if delivery_mode == "direct":
            delivered = await self._deliver_direct(
                message, content, payload, outbox_msg_id
            )
            if kind == "operator_approval" and delivered:
                request_id = str(payload.get("approval_request_id") or "")
                if request_id:
                    try:
                        async with self._pool.acquire() as conn:
                            await conn.fetchval(
                                """
                                SELECT mark_operator_tool_approval_slack_delivered(
                                    $1::uuid, $2, NULL
                                )
                                """,
                                request_id,
                                str(payload.get("target_id") or ""),
                            )
                    except Exception:
                        logger.warning(
                            "Failed to mark operator approval Slack delivery",
                            exc_info=True,
                        )
        elif delivery_mode == "broadcast":
            await self._deliver_broadcast(message, content, payload, outbox_msg_id)
        else:
            # I.2: Check domain-based routing config
            domain = str(payload.get("domain") or payload.get("intent") or "")
            if domain:
                routed = await self._deliver_by_domain(
                    message, content, payload, domain, outbox_msg_id
                )
                if routed:
                    return
            # Default: last_active
            await self._deliver_last_active(message, content, payload, outbox_msg_id)

    async def _deliver_web_inbox(self, body: dict[str, Any]) -> None:
        """Tee the queue body into the web dashboard inbox (db/76).

        Advisory by design: a failure here never blocks routing to the other
        endpoints, and the gate is DB config (channel.web_inbox.enabled).
        """
        inbox_id = None
        try:
            async with self._pool.acquire() as conn:
                enabled = await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('channel.web_inbox.enabled'), TRUE)"
                )
                if not enabled:
                    return
                inbox_id = await conn.fetchval(
                    "SELECT web_inbox_deliver($1::jsonb)", json.dumps(body, default=str)
                )
        except Exception:
            logger.warning("Web inbox delivery failed (non-fatal)", exc_info=True)
            return
        if inbox_id is not None:
            try:
                from services.web_push import deliver_web_push

                await deliver_web_push(self._pool, body)
            except Exception:
                logger.warning("Web Push delivery failed (non-fatal)", exc_info=True)

    async def _deliver_by_domain(
        self,
        message: str | MessagePresentation,
        content: str,
        payload: dict,
        domain: str,
        outbox_msg_id: str,
    ) -> bool:
        """I.2: Route message based on content domain config.

        Config key: channel.routing.{domain} = JSON {"channel": "...", "target_id": "...", "topic": "..."}
        Returns True if routing was found and delivery attempted, False otherwise.
        """
        channel_type = ""
        target_id = ""
        try:
            async with self._pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT get_config_text($1)",
                    f"channel.routing.{domain}",
                )
            if not raw:
                return False
            route = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(route, dict):
                return False
            channel_type = str(route.get("channel") or "")
            target_id = str(route.get("target_id") or "")
            if not channel_type or not target_id:
                return False
            thread_id = str(route.get("topic") or "") or None
            recipient_identity = (
                str(
                    route.get("sender_id")
                    or payload.get("sender_id")
                    or payload.get("target_user")
                    or ""
                )
                or None
            )
            if await self._skip_if_unreachable(
                outbox_msg_id,
                channel_type,
                target_id,
                None,
                content,
                f"domain:{domain}",
            ):
                return True
            delivered = await self._send_with_obligation(
                outbox_msg_id=outbox_msg_id,
                channel_type=channel_type,
                channel_id=target_id,
                sender_id=recipient_identity,
                thread_id=thread_id,
                message=message,
                content=content,
                delivery_mode=f"domain:{domain}",
                payload=payload,
            )
            return delivered
        except Exception as exc:
            await self._mark_unreachable_target(channel_type, target_id, exc)
            logger.warning("Domain routing for %s failed: %s", domain, exc)
            return False

    async def _deliver_direct(
        self,
        message: str | MessagePresentation,
        content: str,
        payload: dict,
        outbox_msg_id: str,
    ) -> bool:
        """Send to an explicit channel + target, optionally with thread/topic."""
        channel_type = str(payload.get("target_channel") or "")
        target_id = str(payload.get("target_id") or "")
        if not channel_type or not target_id:
            logger.warning("Direct delivery missing target_channel/target_id")
            return False

        thread_id = str(payload.get("thread_id") or "") or None
        recipient_identity = (
            str(payload.get("sender_id") or payload.get("target_user") or "") or None
        )

        if await self._skip_if_unreachable(
            outbox_msg_id, channel_type, target_id, thread_id, content, "direct"
        ):
            return False

        try:
            return await self._send_with_obligation(
                outbox_msg_id=outbox_msg_id,
                channel_type=channel_type,
                channel_id=target_id,
                sender_id=recipient_identity,
                thread_id=thread_id,
                message=message,
                content=content,
                delivery_mode="direct",
                payload=payload,
            )
        except Exception:
            logger.debug("Direct delivery failed", exc_info=True)
            return False

    async def _deliver_last_active(
        self,
        message: str | MessagePresentation,
        content: str,
        payload: dict,
        outbox_msg_id: str,
    ) -> None:
        """Send to the sender's most recently active channel session."""
        sender_id = str(payload.get("sender_id") or payload.get("target_user") or "")

        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT resolve_last_active_target($1)", sender_id or None
            )
        row = json.loads(raw) if isinstance(raw, str) else raw

        if not row:
            logger.warning("No active session found for last_active delivery")
            return

        channel_type = row["channel_type"]
        channel_id = row["channel_id"]
        resolved_sender = row["sender_id"]

        if await self._skip_if_unreachable(
            outbox_msg_id,
            channel_type,
            channel_id,
            resolved_sender,
            content,
            "last_active",
        ):
            return

        try:
            await self._send_with_obligation(
                outbox_msg_id=outbox_msg_id,
                channel_type=channel_type,
                channel_id=channel_id,
                sender_id=resolved_sender,
                thread_id=None,
                message=message,
                content=content,
                delivery_mode="last_active",
                payload=payload,
            )
        except Exception:
            logger.debug("Last-active delivery failed", exc_info=True)

    async def _deliver_broadcast(
        self,
        message: str | MessagePresentation,
        content: str,
        payload: dict,
        outbox_msg_id: str,
    ) -> None:
        """Send to all active channel sessions."""
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval("SELECT list_broadcast_targets()")
        rows = json.loads(raw) if isinstance(raw, str) else (raw or [])

        for row in rows:
            channel_type = row["channel_type"]
            channel_id = row["channel_id"]
            sender_id = row["sender_id"]
            if await self._skip_if_unreachable(
                outbox_msg_id, channel_type, channel_id, sender_id, content, "broadcast"
            ):
                continue
            try:
                await self._send_with_obligation(
                    outbox_msg_id=outbox_msg_id,
                    channel_type=channel_type,
                    channel_id=channel_id,
                    sender_id=sender_id,
                    thread_id=None,
                    message=message,
                    content=content,
                    delivery_mode="broadcast",
                    payload=payload,
                )
            except Exception:
                logger.debug("Broadcast delivery failed", exc_info=True)

    async def _deliver_webhook(
        self,
        content: str,
        payload: dict[str, Any],
        delivery_info: dict[str, Any],
        outbox_msg_id: str,
    ) -> None:
        """Send outbox payload to a configured webhook URL."""
        url = str(delivery_info.get("url") or "").strip()
        if not url:
            logger.warning("Webhook delivery missing URL")
            return

        body = {
            "content": content,
            "payload": payload,
            "delivery": delivery_info,
            "outbox_message_id": outbox_msg_id or None,
        }
        headers = {"Content-Type": "application/json"}

        if await self._skip_if_unreachable(
            outbox_msg_id, "webhook", url, None, content, "webhook"
        ):
            return

        from services.outbound_safety import (
            finalize_outbox_outbound,
            prepare_outbox_outbound,
        )

        primary_hint = _is_primary_user_envelope(payload)
        public_hint = str(payload.get("_outbox_kind") or "") == "public"
        webhook_target = (
            "webhook:" + hashlib.sha256(url.encode("utf-8", "replace")).hexdigest()[:16]
        )
        preparation = await prepare_outbox_outbound(
            self._pool,
            request_key=f"outbox:{outbox_msg_id or 'anonymous'}:{webhook_target}",
            channel="webhook",
            recipient=webhook_target,
            identity_address=None,
            body=content,
            payload=payload,
            primary_hint=primary_hint,
            public_hint=public_hint,
        )
        if not preparation.allowed:
            await self._log_delivery(
                outbox_msg_id,
                "webhook",
                url,
                None,
                content,
                "webhook",
                False,
                preparation.error,
            )
            return
        content = str(preparation.arguments.get("message") or content)
        body["content"] = content

        try:
            await request_text_response(
                "webhook",
                "POST",
                url,
                headers=headers,
                json_body=body,
                timeout=8.0,
                attempts=3,
                max_delay=5.0,
                retry_unsafe_methods=False,
            )
            await finalize_outbox_outbound(self._pool, preparation, delivered=True)
            await self._clear_unreachable_target("webhook", url)
            await self._log_delivery(
                outbox_msg_id, "webhook", url, None, content, "webhook", True
            )
        except IntegrationHttpError as e:
            await finalize_outbox_outbound(
                self._pool,
                preparation,
                delivered=False,
                error=format_provider_error("Webhook", e),
            )
            await self._mark_unreachable_target("webhook", url, e)
            await self._log_delivery(
                outbox_msg_id,
                "webhook",
                url,
                None,
                content,
                "webhook",
                False,
                format_provider_error("Webhook", e),
            )
        except Exception as e:
            await finalize_outbox_outbound(
                self._pool, preparation, delivered=False, error=str(e)
            )
            await self._mark_unreachable_target("webhook", url, e)
            await self._log_delivery(
                outbox_msg_id, "webhook", url, None, content, "webhook", False, str(e)
            )

    async def _send_with_obligation(
        self,
        *,
        outbox_msg_id: str,
        channel_type: str,
        channel_id: str,
        sender_id: str | None,
        thread_id: str | None,
        message: str | MessagePresentation,
        content: str,
        delivery_mode: str,
        payload: dict[str, Any] | None = None,
        obligation_id: str | None = None,
        already_claimed: bool = False,
        recovered: bool = False,
    ) -> bool:
        if obligation_id is None:
            obligation = await self._record_delivery_obligation(
                outbox_msg_id,
                channel_type,
                channel_id,
                sender_id,
                thread_id,
                content,
                message,
                delivery_mode,
                payload,
            )
            if obligation.get("already_delivered"):
                return True
            obligation_id = str(obligation.get("id") or "") or None

        if obligation_id and not already_claimed:
            claimed = await self._claim_delivery_obligation(obligation_id)
            if claimed is False:
                return False

        outbound_message = message
        outbound_content = content
        if recovered:
            outbound_content = RECOVERED_DELIVERY_MARKER + content
            outbound_message = _prepend_recovery_marker(message, content)

        from services.outbound_safety import prepare_outbox_outbound

        policy_payload = payload or {}
        primary_hint = _is_primary_user_envelope(policy_payload)
        public_hint = str(policy_payload.get("_outbox_kind") or "") == "public"
        preparation = await prepare_outbox_outbound(
            self._pool,
            request_key=(
                f"outbox:{outbox_msg_id or 'anonymous'}:{obligation_id or channel_type + ':' + channel_id}"
            ),
            channel=channel_type,
            recipient=channel_id,
            identity_address=sender_id,
            body=outbound_content,
            payload=policy_payload,
            thread_reference=thread_id,
            primary_hint=primary_hint,
            public_hint=public_hint,
        )
        if not preparation.allowed:
            exc = RuntimeError(preparation.error or "Outbound policy denied delivery")
            if obligation_id:
                await self._mark_delivery_obligation_failed(obligation_id, exc)
            await self._log_delivery(
                outbox_msg_id,
                channel_type,
                channel_id,
                sender_id,
                outbound_content,
                delivery_mode,
                False,
                str(exc),
            )
            return False
        governed_content = str(preparation.arguments.get("message") or outbound_content)
        if governed_content != outbound_content:
            outbound_message = governed_content
        outbound_content = governed_content

        try:
            msg_id = await self._manager.send(
                channel_type,
                channel_id,
                outbound_message,
                thread_id=thread_id,
            )
            _require_delivery_id(msg_id, channel_type, channel_id)
            from services.outbound_safety import finalize_outbox_outbound

            await finalize_outbox_outbound(
                self._pool,
                preparation,
                delivered=True,
                provider_message_id=msg_id,
            )
            if obligation_id:
                await self._mark_delivery_obligation_delivered(obligation_id)
            await self._clear_unreachable_target(channel_type, channel_id)
            await self._log_delivery(
                outbox_msg_id,
                channel_type,
                channel_id,
                sender_id,
                outbound_content,
                delivery_mode,
                True,
            )
            return True
        except Exception as exc:
            from services.outbound_safety import finalize_outbox_outbound

            await finalize_outbox_outbound(
                self._pool,
                preparation,
                delivered=False,
                error=bounded_delivery_error(exc),
            )
            if obligation_id:
                await self._mark_delivery_obligation_failed(obligation_id, exc)
            await self._mark_unreachable_target(channel_type, channel_id, exc)
            await self._log_delivery(
                outbox_msg_id,
                channel_type,
                channel_id,
                sender_id,
                outbound_content,
                delivery_mode,
                False,
                bounded_delivery_error(exc),
            )
            return False

    async def _recover_delivery_obligations(self) -> int:
        try:
            async with self._pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT claim_recoverable_channel_deliveries($1::int)",
                    25,
                )
        except Exception:
            logger.debug(
                "Failed to claim recoverable delivery obligations", exc_info=True
            )
            return 0

        rows = _coerce_json_list(raw)
        recovered = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            channel_type = str(row.get("channel_type") or "")
            channel_id = str(row.get("channel_id") or "")
            content = str(row.get("content") or "")
            message = _message_from_wire(row.get("message"), content)
            outbox_msg_id = str(row.get("source_outbox_message_id") or "")
            delivery_mode = str(row.get("delivery_mode") or "recovered")
            sender_id = row.get("sender_id")
            thread_id = row.get("thread_id")
            if await self._skip_if_unreachable(
                outbox_msg_id,
                channel_type,
                channel_id,
                str(sender_id) if sender_id is not None else None,
                content,
                delivery_mode,
            ):
                obligation_id = str(row.get("id") or "")
                if obligation_id:
                    await self._mark_delivery_obligation_failed(
                        obligation_id,
                        RuntimeError("target is temporarily marked unreachable"),
                    )
                continue
            if await self._send_with_obligation(
                outbox_msg_id=outbox_msg_id,
                channel_type=channel_type,
                channel_id=channel_id,
                sender_id=str(sender_id) if sender_id is not None else None,
                thread_id=str(thread_id) if thread_id is not None else None,
                message=message,
                content=content,
                delivery_mode=delivery_mode,
                payload=(
                    row.get("message", {}).get("_outbound_payload", {})
                    if isinstance(row.get("message"), dict)
                    else {}
                ),
                obligation_id=str(row.get("id") or ""),
                already_claimed=True,
                recovered=bool(row.get("needs_marker")),
            ):
                recovered += 1
        return recovered

    async def _skip_if_unreachable(
        self,
        outbox_msg_id: str,
        channel_type: str,
        channel_id: str,
        sender_id: str | None,
        content: str,
        delivery_mode: str,
    ) -> bool:
        try:
            async with self._pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT should_skip_channel_target($1, $2)",
                    channel_type,
                    channel_id,
                )
        except Exception:
            logger.debug("Failed to check unreachable target registry", exc_info=True)
            return False

        info = _coerce_json_object(raw)
        if not bool(info.get("skip")):
            return False

        reason = str(info.get("reason") or "target is temporarily marked unreachable")
        suppress_until = info.get("suppress_until")
        error = "Skipped delivery: target marked unreachable"
        if suppress_until:
            error += f" until {suppress_until}"
        if reason:
            error += f" ({reason})"
        await self._log_delivery(
            outbox_msg_id,
            channel_type,
            channel_id,
            sender_id,
            content,
            delivery_mode,
            False,
            error,
        )
        return True

    async def _clear_unreachable_target(
        self, channel_type: str, channel_id: str
    ) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT clear_channel_target_unreachable($1, $2)",
                    channel_type,
                    channel_id,
                )
        except Exception:
            logger.debug("Failed to clear unreachable target marker", exc_info=True)

    async def _mark_unreachable_target(
        self,
        channel_type: str,
        channel_id: str,
        exc: BaseException,
    ) -> None:
        error_kind = _target_unreachable_error_kind(exc)
        if error_kind is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT mark_channel_target_unreachable($1, $2, $3, $4, $5, $6::jsonb)",
                    channel_type,
                    channel_id,
                    bounded_delivery_error(exc),
                    error_kind,
                    _unreachable_suppress_seconds(error_kind),
                    json.dumps({"source": "channel_outbox"}),
                )
        except Exception:
            logger.debug("Failed to mark unreachable target", exc_info=True)

    async def _record_delivery_obligation(
        self,
        outbox_msg_id: str,
        channel_type: str,
        channel_id: str,
        sender_id: str | None,
        thread_id: str | None,
        content: str,
        message: str | MessagePresentation,
        delivery_mode: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = _delivery_obligation_key(
            outbox_msg_id,
            delivery_mode,
            channel_type,
            channel_id,
            sender_id,
            thread_id,
            content,
        )
        try:
            message_wire = _message_to_wire(message)
            message_wire["_outbound_payload"] = payload or {}
            async with self._pool.acquire() as conn:
                raw = await conn.fetchval(
                    """
                    SELECT upsert_channel_delivery_obligation(
                        $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9
                    )
                    """,
                    key,
                    outbox_msg_id,
                    channel_type,
                    channel_id,
                    sender_id,
                    thread_id,
                    content,
                    json.dumps(message_wire, default=str),
                    delivery_mode,
                )
            return _coerce_json_object(raw)
        except Exception:
            logger.debug("Failed to record delivery obligation", exc_info=True)
            return {}

    async def _claim_delivery_obligation(self, obligation_id: str) -> bool | None:
        try:
            async with self._pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT claim_channel_delivery_obligation($1::uuid)",
                    obligation_id,
                )
            info = _coerce_json_object(raw)
            return bool(info.get("claimed"))
        except Exception:
            logger.debug("Failed to claim delivery obligation", exc_info=True)
            return None

    async def _mark_delivery_obligation_delivered(self, obligation_id: str) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT mark_channel_delivery_obligation_delivered($1::uuid)",
                    obligation_id,
                )
        except Exception:
            logger.debug("Failed to mark delivery obligation delivered", exc_info=True)

    async def _mark_delivery_obligation_failed(
        self,
        obligation_id: str,
        exc: BaseException,
    ) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT mark_channel_delivery_obligation_failed($1::uuid, $2, $3)",
                    obligation_id,
                    bounded_delivery_error(exc),
                    _delivery_retry_seconds(exc),
                )
        except Exception:
            logger.debug("Failed to mark delivery obligation failed", exc_info=True)

    async def _log_delivery(
        self,
        outbox_message_id: str,
        channel_type: str,
        channel_id: str,
        sender_id: str | None,
        content: str,
        delivery_mode: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Log a delivery attempt to the channel_deliveries table."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO channel_deliveries
                        (outbox_message_id, channel_type, channel_id, sender_id, content, delivery_mode, success, error)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    outbox_message_id or None,
                    channel_type,
                    channel_id,
                    sender_id,
                    content[:2000],  # Truncate for storage
                    delivery_mode,
                    success,
                    error,
                )
        except Exception:
            logger.debug("Failed to log delivery", exc_info=True)


def _vhost_path() -> str:
    if RABBITMQ_VHOST == "/":
        return "%2F"
    return requests.utils.quote(RABBITMQ_VHOST, safe="")


async def _rmq_request(
    method: str, path: str, payload: dict | None = None
) -> requests.Response:
    url = f"{RABBITMQ_MANAGEMENT_URL}{path}"
    auth = (RABBITMQ_USER, RABBITMQ_PASSWORD)

    def _do() -> requests.Response:
        return requests.request(method, url, auth=auth, json=payload, timeout=5)

    return await asyncio.to_thread(_do)


def _resolve_payload_message(
    payload: dict[str, Any],
) -> tuple[str | MessagePresentation, str]:
    """Return the outbound value and its stable plain-text audit mirror."""

    content = str(
        payload.get("content") or payload.get("message") or payload.get("text") or ""
    )
    raw_presentation = payload.get("presentation")
    if raw_presentation is None:
        return content, content

    presentation = normalize_message_presentation(raw_presentation)
    mirror = content or render_presentation(presentation)
    return presentation, mirror


def _require_delivery_id(
    message_id: str | None, channel_type: str, channel_id: str
) -> None:
    if message_id:
        return
    raise RuntimeError(
        f"{channel_type} delivery to {channel_id} did not return a platform message id."
    )


def _coerce_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def bounded_delivery_error(exc: BaseException) -> str:
    if isinstance(exc, IntegrationHttpError):
        return format_provider_error(exc.provider.title(), exc)
    return bounded_text(f"{type(exc).__name__}: {exc}", limit=500)


def _target_unreachable_error_kind(exc: BaseException) -> str | None:
    if isinstance(exc, IntegrationHttpError):
        if exc.status_code in {404, 410}:
            return "not_found"
        if exc.error_kind == "not_found":
            return "not_found"
        return None

    text = f"{type(exc).__name__}: {exc}".lower()
    signals = (
        "chat not found",
        "channel not found",
        "channel_not_found",
        "user not found",
        "user_not_found",
        "recipient not found",
        "target not found",
        "not a registered user",
        "not registered",
        "group chat was deleted",
        "bot was blocked",
        "bot blocked",
        "bot was kicked",
        "bot kicked",
        "user is deactivated",
        "cannot send messages to this user",
        "cannot message this user",
        "not in channel",
        "left the channel",
        "no such channel",
        "gone",
    )
    if any(signal in text for signal in signals):
        return "not_found"
    if "forbidden" in text and (
        "chat" in text or "bot" in text or "user" in text or "recipient" in text
    ):
        return "forbidden"
    return None


def _unreachable_suppress_seconds(error_kind: str) -> int:
    if error_kind == "forbidden":
        return 24 * 60 * 60
    if error_kind == "not_found":
        return 24 * 60 * 60
    return 60 * 60


def _delivery_retry_seconds(exc: BaseException) -> int:
    if isinstance(exc, IntegrationHttpError):
        if exc.retry_after_seconds is not None:
            return int(max(30, min(exc.retry_after_seconds, 24 * 60 * 60)))
        if exc.error_kind == "rate_limited":
            return 15 * 60
        if exc.transient:
            return 5 * 60
    if _target_unreachable_error_kind(exc):
        return 60 * 60
    return 5 * 60


def _delivery_obligation_key(
    outbox_msg_id: str,
    delivery_mode: str,
    channel_type: str,
    channel_id: str,
    sender_id: str | None,
    thread_id: str | None,
    content: str,
) -> str:
    source = (
        outbox_msg_id or hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
    )
    payload = "|".join(
        [
            source,
            delivery_mode,
            channel_type,
            channel_id,
            sender_id or "",
            thread_id or "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def _message_to_wire(message: str | MessagePresentation) -> dict[str, Any]:
    if isinstance(message, MessagePresentation):
        return {"kind": "presentation", "value": message.to_dict()}
    return {"kind": "text", "value": str(message)}


def _message_from_wire(raw: Any, fallback_text: str) -> str | MessagePresentation:
    value = _coerce_json_object(raw)
    if value.get("kind") == "presentation" and isinstance(value.get("value"), dict):
        try:
            return normalize_message_presentation(value["value"])
        except Exception:
            return fallback_text
    text = value.get("value")
    return str(text) if isinstance(text, str) else fallback_text


def _prepend_recovery_marker(
    message: str | MessagePresentation,
    content: str,
) -> str | MessagePresentation:
    if isinstance(message, MessagePresentation):
        return presentation_from_text(RECOVERED_DELIVERY_MARKER + content)
    return RECOVERED_DELIVERY_MARKER + str(message)
