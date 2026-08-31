"""Thin bridge into the DB-owned outbound communication policy.

Provider handlers and channel outbox delivery both call this module.  It does
not decide who may be contacted or how much attention costs; PostgreSQL owns
those decisions and the durable ledger.  Python only maps transport argument
shapes and injects the returned disclosure.
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass, field
from email.utils import getaddresses
from typing import Any

from core.tools.base import (
    OutboundSpec,
    ToolErrorType,
    ToolExecutionContext,
    ToolResult,
)

logger = logging.getLogger(__name__)


@dataclass
class OutboundPreparation:
    allowed: bool
    arguments: dict[str, Any]
    event_ids: list[str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    error_type: ToolErrorType = ToolErrorType.OUTBOUND_BLOCKED

    def error_result(self) -> ToolResult:
        return ToolResult.error_result(
            self.error or "Outbound communication was blocked.", self.error_type
        )


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def _error_type(value: Any) -> ToolErrorType:
    try:
        return ToolErrorType(str(value))
    except ValueError:
        return ToolErrorType.OUTBOUND_BLOCKED


def _split_recipients(value: Any, *, channel: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = [str(item) for item in value]
    else:
        raw_values = [str(value)]
    if channel == "email":
        return [
            address.strip()
            for _, address in getaddresses(raw_values)
            if address.strip()
        ]
    return [item.strip() for item in raw_values if item.strip()]


def outbound_recipients(spec: OutboundSpec, arguments: dict[str, Any]) -> list[str]:
    recipients: list[str] = []
    if spec.recipient_arg:
        recipients.extend(
            _split_recipients(arguments.get(spec.recipient_arg), channel=spec.channel)
        )
    for name in spec.additional_recipient_args:
        recipients.extend(_split_recipients(arguments.get(name), channel=spec.channel))
    if not recipients and spec.fixed_recipient:
        recipients.append(spec.fixed_recipient)
    # Preserve order while preventing duplicate budget reservations.
    return list(dict.fromkeys(recipients))


async def preflight_tool_outbound_controls(
    pool: Any,
    *,
    spec: OutboundSpec,
    arguments: dict[str, Any],
) -> OutboundPreparation:
    """Check STOP and kill switches before an approval request is filed.

    This does not reserve contact points. ``prepare_tool_outbound`` repeats the
    control checks and performs the complete authorization immediately before
    provider execution.
    """
    recipients = outbound_recipients(spec, arguments)
    if not recipients:
        return OutboundPreparation(
            allowed=False,
            arguments=dict(arguments),
            error="Outbound recipient could not be resolved from the tool arguments.",
            error_type=ToolErrorType.INVALID_PARAMS,
        )

    decisions: list[dict[str, Any]] = []
    try:
        async with pool.acquire() as conn:
            for recipient in recipients:
                raw = await conn.fetchval(
                    "SELECT check_outbound_controls($1, $2, NULL, $3, $4)",
                    spec.channel,
                    recipient,
                    spec.primary_recipient,
                    spec.public_recipient,
                )
                decision = _json(raw)
                decisions.append(decision)
                if decision.get("allowed") is not True:
                    return OutboundPreparation(
                        allowed=False,
                        arguments=dict(arguments),
                        decisions=decisions,
                        error=str(
                            decision.get("reason")
                            or "Outbound controls denied this communication."
                        ),
                        error_type=_error_type(decision.get("error_type")),
                    )
    except Exception as exc:
        return OutboundPreparation(
            allowed=False,
            arguments=dict(arguments),
            decisions=decisions,
            error=f"Outbound control preflight failed: {exc}",
            error_type=ToolErrorType.OUTBOUND_BLOCKED,
        )

    return OutboundPreparation(
        allowed=True,
        arguments=dict(arguments),
        decisions=decisions,
    )


def _append_disclosure(
    arguments: dict[str, Any],
    spec: OutboundSpec,
    decisions: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    updated = dict(arguments)
    disclosures = [
        str(item.get("disclosure") or "").strip()
        for item in decisions
        if str(item.get("disclosure") or "").strip()
    ]
    if not disclosures:
        return updated, None

    # A mixed-recipient email uses the fullest disclosure any recipient is due.
    disclosure = max(disclosures, key=len)
    body = str(updated.get(spec.body_arg) or "")
    updated[spec.body_arg] = f"{body.rstrip()}\n\n{disclosure}".strip()

    if spec.html_body_arg and updated.get(spec.html_body_arg):
        html_disclosure = "<br>".join(
            html.escape(line) for line in disclosure.splitlines()
        )
        updated[spec.html_body_arg] = (
            f"{str(updated[spec.html_body_arg]).rstrip()}"
            f'<hr><p style="font-size:small">{html_disclosure}</p>'
        )

    if spec.channel == "slack" and isinstance(updated.get("blocks"), list):
        blocks = list(updated["blocks"])
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": disclosure}],
            }
        )
        updated["blocks"] = blocks

    if (
        spec.channel in {"twitter_x", "twitter_x_dm"}
        and len(str(updated[spec.body_arg])) > 280
    ):
        return updated, (
            "The required AI disclosure would exceed Twitter/X's 280-character "
            "limit. Shorten the message; the disclosure cannot be omitted."
        )
    return updated, None


async def _finalize_ids(
    pool: Any,
    event_ids: list[str],
    *,
    delivered: bool,
    provider_message_id: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not event_ids:
        return
    async with pool.acquire() as conn:
        for event_id in event_ids:
            await conn.fetchval(
                "SELECT finalize_outbound($1::uuid, $2, $3, $4, $5::jsonb)",
                event_id,
                delivered,
                provider_message_id,
                error,
                json.dumps(metadata or {}),
            )


async def prepare_tool_outbound(
    pool: Any,
    *,
    tool_name: str,
    spec: OutboundSpec,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> OutboundPreparation:
    recipients = outbound_recipients(spec, arguments)
    if not recipients:
        return OutboundPreparation(
            allowed=False,
            arguments=dict(arguments),
            error="Outbound recipient could not be resolved from the tool arguments.",
            error_type=ToolErrorType.INVALID_PARAMS,
        )

    purpose_kind = str(arguments.get("purpose_kind") or "")
    purpose_reference = str(arguments.get("purpose_reference") or "")
    urgency = str(arguments.get("urgency") or "normal")
    thread_reference = (
        str(arguments.get(spec.thread_arg) or "") if spec.thread_arg else ""
    )
    context_doc = {
        "tool_context": context.tool_context.value,
        "call_id": context.call_id,
        "heartbeat_id": context.heartbeat_id,
        "session_id": context.session_id,
        "is_operator": context.is_operator,
        "surface": context.surface,
        "approval_request_id": context.approval_request_id,
    }
    body = str(arguments.get(spec.body_arg) or "")
    decisions: list[dict[str, Any]] = []
    event_ids: list[str] = []

    try:
        async with pool.acquire() as conn:
            for index, recipient in enumerate(recipients):
                raw = await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'tool', $2, $3, $4, NULL, $5, $6, $7, $8,
                        $9::jsonb, $10, $11, $12
                    )
                    """,
                    f"tool:{context.call_id}:{index}",
                    tool_name,
                    spec.channel,
                    recipient,
                    purpose_kind,
                    purpose_reference,
                    thread_reference or None,
                    urgency,
                    json.dumps(context_doc),
                    body[:500],
                    spec.primary_recipient,
                    spec.public_recipient,
                )
                decision = _json(raw)
                decisions.append(decision)
                event_id = decision.get("event_id")
                if decision.get("allowed") is True and event_id:
                    event_ids.append(str(event_id))
                if decision.get("allowed") is not True:
                    for reserved_id in event_ids:
                        await conn.fetchval(
                            "SELECT finalize_outbound($1::uuid, FALSE, NULL, $2, '{}'::jsonb)",
                            reserved_id,
                            "A co-recipient was denied by outbound policy.",
                        )
                    return OutboundPreparation(
                        allowed=False,
                        arguments=dict(arguments),
                        decisions=decisions,
                        error=str(
                            decision.get("reason") or "Outbound policy denied the send."
                        ),
                        error_type=_error_type(decision.get("error_type")),
                    )
    except Exception as exc:
        await _finalize_ids(
            pool,
            event_ids,
            delivered=False,
            error=f"Outbound safety evaluation failed: {exc}",
        )
        return OutboundPreparation(
            allowed=False,
            arguments=dict(arguments),
            decisions=decisions,
            error=f"Outbound safety evaluation failed: {exc}",
            error_type=ToolErrorType.OUTBOUND_BLOCKED,
        )

    updated, formatting_error = _append_disclosure(arguments, spec, decisions)
    if formatting_error:
        await _finalize_ids(pool, event_ids, delivered=False, error=formatting_error)
        return OutboundPreparation(
            allowed=False,
            arguments=updated,
            decisions=decisions,
            error=formatting_error,
            error_type=ToolErrorType.INVALID_PARAMS,
        )
    updated["_outbound_event_ids"] = event_ids
    return OutboundPreparation(
        allowed=True,
        arguments=updated,
        event_ids=event_ids,
        decisions=decisions,
    )


def _provider_message_id(result: ToolResult) -> str | None:
    if not isinstance(result.output, dict):
        return None
    for key in ("message_id", "tweet_id", "id", "ts", "outbox_id"):
        value = result.output.get(key)
        if value:
            return str(value)
    return None


async def finalize_tool_outbound(
    pool: Any,
    preparation: OutboundPreparation,
    result: ToolResult,
) -> None:
    await _finalize_ids(
        pool,
        preparation.event_ids,
        delivered=result.success,
        provider_message_id=_provider_message_id(result),
        error=result.error,
        # Provider payloads can contain message content or connector-specific
        # details.  The ledger only needs delivery state and the extracted ID.
        metadata={"provider_success": result.success},
    )
    if preparation.event_ids:
        result.metadata["outbound_event_ids"] = list(preparation.event_ids)


async def prepare_outbox_outbound(
    pool: Any,
    *,
    request_key: str,
    channel: str,
    recipient: str,
    identity_address: str | None,
    body: str,
    payload: dict[str, Any],
    thread_reference: str | None = None,
    primary_hint: bool = False,
    public_hint: bool = False,
) -> OutboundPreparation:
    context_doc = (
        payload.get("context") if isinstance(payload.get("context"), dict) else {}
    )
    purpose_kind = str(
        payload.get("purpose_kind")
        or context_doc.get("purpose_kind")
        or ("connection" if primary_hint else "")
    )
    purpose_reference = str(
        payload.get("purpose_reference")
        or context_doc.get("purpose_reference")
        or (request_key if primary_hint else "")
    )
    urgency = str(payload.get("urgency") or context_doc.get("urgency") or "normal")
    raw = None
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT authorize_outbound(
                    $1, 'outbox', NULL, $2, $3, $4, $5, $6, $7, $8,
                    $9::jsonb, $10, $11, $12
                )
                """,
                request_key,
                channel,
                recipient,
                identity_address,
                purpose_kind,
                purpose_reference,
                thread_reference,
                urgency,
                json.dumps({"outbox": True, "session_id": payload.get("session_id")}),
                body[:500],
                primary_hint,
                public_hint,
            )
        decision = _json(raw)
    except Exception as exc:
        decision = {
            "allowed": False,
            "reason": f"Outbound safety evaluation failed: {exc}",
            "error_type": "outbound_blocked",
        }
    if decision.get("allowed") is not True:
        return OutboundPreparation(
            allowed=False,
            arguments={"message": body},
            decisions=[decision],
            error=str(decision.get("reason") or "Outbound policy denied delivery."),
            error_type=_error_type(decision.get("error_type")),
        )
    event_id = str(decision.get("event_id") or "")
    spec = OutboundSpec(
        recipient_arg=None,
        body_arg="message",
        channel=channel,
        fixed_recipient=recipient,
        primary_recipient=primary_hint,
        public_recipient=public_hint,
    )
    updated, formatting_error = _append_disclosure({"message": body}, spec, [decision])
    if formatting_error:
        await _finalize_ids(pool, [event_id], delivered=False, error=formatting_error)
        return OutboundPreparation(
            allowed=False,
            arguments=updated,
            decisions=[decision],
            error=formatting_error,
            error_type=ToolErrorType.INVALID_PARAMS,
        )
    return OutboundPreparation(
        allowed=True,
        arguments=updated,
        event_ids=[event_id] if event_id else [],
        decisions=[decision],
    )


async def finalize_outbox_outbound(
    pool: Any,
    preparation: OutboundPreparation,
    *,
    delivered: bool,
    provider_message_id: str | None = None,
    error: str | None = None,
) -> None:
    await _finalize_ids(
        pool,
        preparation.event_ids,
        delivered=delivered,
        provider_message_id=provider_message_id,
        error=error,
    )


class GovernedReplyAdapter:
    """Apply the same outbound contract to replies on inbound channel turns.

    The outbox and tool dispatcher cover proactive communication. Channel
    conversations send through adapters directly, so this per-turn proxy makes
    their reply purpose, cross-channel STOP check, disclosure, and ledger entry
    equally durable. Edits reuse the disclosure attached to the original send.
    """

    def __init__(self, pool: Any, adapter: Any, message: Any) -> None:
        self._pool = pool
        self._adapter = adapter
        self._message = message
        self._send_index = 0
        self._preparations: dict[str, OutboundPreparation] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    @property
    def channel_type(self) -> str:
        return str(self._adapter.channel_type)

    @property
    def capabilities(self) -> Any:
        return self._adapter.capabilities

    @property
    def is_connected(self) -> bool:
        return bool(self._adapter.is_connected)

    async def _prepare(self, body: str) -> OutboundPreparation:
        self._send_index += 1
        message_id = str(getattr(self._message, "message_id", "") or "inbound")
        preparation = await prepare_outbox_outbound(
            self._pool,
            request_key=(
                f"channel-reply:{self.channel_type}:{message_id}:{self._send_index}"
            ),
            channel=self.channel_type,
            recipient=str(getattr(self._message, "channel_id", "") or ""),
            identity_address=str(getattr(self._message, "sender_id", "") or ""),
            body=body,
            payload={
                "purpose_kind": "reply",
                "purpose_reference": message_id,
                "urgency": "normal",
                "session_id": (
                    f"channel:{self.channel_type}:"
                    f"{getattr(self._message, 'channel_id', '')}:"
                    f"{getattr(self._message, 'sender_id', '')}"
                ),
            },
            thread_reference=message_id,
            primary_hint=False,
        )
        if not preparation.allowed:
            raise RuntimeError(
                preparation.error
                or "Outbound policy denied this channel reply. Review Outbound controls and retry."
            )
        return preparation

    async def send(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        preparation = await self._prepare(str(text))
        governed = str(preparation.arguments.get("message") or text)
        try:
            message_id = await self._adapter.send(
                channel_id,
                governed,
                reply_to=reply_to,
                thread_id=thread_id,
            )
            if not message_id:
                raise RuntimeError(
                    f"{self.channel_type} did not return a platform message id"
                )
        except Exception as exc:
            try:
                await finalize_outbox_outbound(
                    self._pool, preparation, delivered=False, error=str(exc)
                )
            except Exception:
                logger.exception("Failed to refund/finalize denied channel reply")
            raise
        try:
            await finalize_outbox_outbound(
                self._pool,
                preparation,
                delivered=True,
                provider_message_id=str(message_id),
            )
        except Exception:
            # The provider effect already happened. Do not invite an automatic
            # retry merely because ledger finalization failed.
            logger.exception("Failed to finalize delivered channel reply")
        self._preparations[str(message_id)] = preparation
        return str(message_id)

    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
    ) -> bool:
        preparation = self._preparations.get(str(message_id))
        governed = (
            _apply_prepared_disclosure(str(text), self.channel_type, preparation)
            if preparation is not None
            else str(text)
        )
        return bool(await self._adapter.edit_message(channel_id, message_id, governed))

    async def send_presentation(
        self,
        channel_id: str,
        presentation: Any,
        *,
        reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        from channels.presentation import (
            ContextBlock,
            MessagePresentation,
            render_presentation,
        )

        body = render_presentation(
            presentation,
            getattr(self.capabilities, "markdown_dialect", "plain"),
        )
        preparation = await self._prepare(body)
        disclosures = _decision_disclosures(preparation)
        governed_presentation = presentation
        if disclosures and isinstance(presentation, MessagePresentation):
            governed_presentation = MessagePresentation(
                title=presentation.title,
                tone=presentation.tone,
                blocks=(*presentation.blocks, ContextBlock(max(disclosures, key=len))),
            )
        try:
            message_id = await self._adapter.send_presentation(
                channel_id,
                governed_presentation,
                reply_to=reply_to,
                thread_id=thread_id,
            )
            if not message_id:
                raise RuntimeError(
                    f"{self.channel_type} did not return a platform message id"
                )
        except Exception as exc:
            try:
                await finalize_outbox_outbound(
                    self._pool, preparation, delivered=False, error=str(exc)
                )
            except Exception:
                logger.exception(
                    "Failed to refund/finalize denied channel presentation"
                )
            raise
        try:
            await finalize_outbox_outbound(
                self._pool,
                preparation,
                delivered=True,
                provider_message_id=str(message_id),
            )
        except Exception:
            logger.exception("Failed to finalize delivered channel presentation")
        self._preparations[str(message_id)] = preparation
        return str(message_id)


def _decision_disclosures(preparation: OutboundPreparation) -> list[str]:
    return [
        str(decision.get("disclosure") or "").strip()
        for decision in preparation.decisions
        if str(decision.get("disclosure") or "").strip()
    ]


def _apply_prepared_disclosure(
    body: str,
    channel: str,
    preparation: OutboundPreparation,
) -> str:
    spec = OutboundSpec(
        recipient_arg=None,
        body_arg="message",
        channel=channel,
        fixed_recipient="reply",
    )
    updated, formatting_error = _append_disclosure(
        {"message": body}, spec, preparation.decisions
    )
    if formatting_error:
        raise RuntimeError(formatting_error)
    return str(updated.get("message") or body)


async def handle_inbound_contact_control(pool: Any, message: Any) -> dict[str, Any]:
    metadata = dict(getattr(message, "metadata", None) or {})
    metadata.update(
        channel_id=getattr(message, "channel_id", None),
        platform_message_id=getattr(message, "message_id", None),
        sender_name=getattr(message, "sender_name", None),
    )
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT handle_inbound_contact_control($1, $2, $3, $4, $5::jsonb)",
            getattr(message, "channel_type", "unknown"),
            getattr(message, "sender_id", ""),
            getattr(message, "content", ""),
            False,
            json.dumps(metadata, default=str),
        )
    return _json(raw)
