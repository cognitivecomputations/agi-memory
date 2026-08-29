from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from core.tools.base import (
    OutboundSpec,
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)
from core.tools.messaging import SlackSendHandler
from core.tools.registry import ToolRegistry

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


class _CaptureEmailHandler(ToolHandler):
    def __init__(
        self, *, name: str = "test_outbound_email", energy_cost: int = 0
    ) -> None:
        self.arguments: dict[str, Any] | None = None
        self.name = name
        self.energy_cost = energy_cost

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Capture an email-shaped outbound call.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "body"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=self.energy_cost,
            is_read_only=False,
            outbound=OutboundSpec(recipient_arg="to", body_arg="body", channel="email"),
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        self.arguments = dict(arguments)
        return ToolResult.success_result({"message_id": "captured-provider-id"})


class _UndeclaredMessagingHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="future_provider_send",
            description="A provider effect missing its mandatory descriptor.",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.MESSAGING,
            is_read_only=False,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        return ToolResult.success_result({})


async def test_outbound_spec_adds_required_purpose_schema():
    spec = SlackSendHandler().spec
    assert spec.outbound is not None
    assert spec.parameters["properties"]["purpose_kind"]["enum"] == [
        "goal",
        "responsibility",
        "reply",
        "user_request",
        "connection",
    ]
    assert "purpose_kind" in spec.parameters["required"]
    assert "purpose_reference" in spec.parameters["required"]


async def test_registry_refuses_future_messaging_effect_without_descriptor(db_pool):
    registry = ToolRegistry(db_pool)
    with pytest.raises(ValueError, match="must declare ToolSpec.outbound"):
        registry.register(_UndeclaredMessagingHandler())
    with pytest.raises(ValueError, match="must declare ToolSpec.outbound"):
        registry.register_mcp(_UndeclaredMessagingHandler())


async def test_registry_control_preflight_is_db_backed_and_fail_closed(db_pool):
    marker = uuid4().hex
    recipient = f"preflight-{marker}@example.com"
    handler = _CaptureEmailHandler()
    registry = ToolRegistry(db_pool)
    registry.register(handler)
    arguments = {
        "to": recipient,
        "body": "This must not reach the provider after STOP.",
        "purpose_kind": "user_request",
        "purpose_reference": "current_turn",
    }

    assert await registry.preflight_outbound_controls(handler.spec, arguments) is None
    async with db_pool.acquire() as conn:
        entity = await conn.fetchval(
            "SELECT entity FROM outbound_contact_endpoints WHERE channel='email' AND address=$1",
            recipient,
        )
        await conn.execute(
            """
            INSERT INTO outbound_contact_controls(entity, blocked, blocked_at)
            VALUES ($1, true, CURRENT_TIMESTAMP)
            ON CONFLICT (entity) DO UPDATE
            SET blocked=true, blocked_at=CURRENT_TIMESTAMP
            """,
            entity,
        )
    try:
        refusal = await registry.preflight_outbound_controls(handler.spec, arguments)
        assert refusal is not None
        assert refusal.error_type == ToolErrorType.OUTBOUND_BLOCKED
        assert "opted out" in (refusal.error or "")
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM outbound_contact_controls WHERE entity=$1", entity
            )
            await conn.execute(
                "DELETE FROM outbound_contact_endpoints WHERE entity=$1", entity
            )


async def test_registry_injects_disclosure_and_finalizes_ledger(db_pool):
    marker = uuid4().hex
    recipient = f"{marker}@example.com"
    handler = _CaptureEmailHandler()
    registry = ToolRegistry(db_pool)
    registry.register(handler)
    result = await registry.execute(
        "test_outbound_email",
        {
            "to": recipient,
            "body": "A concise update.",
            "purpose_kind": "user_request",
            "purpose_reference": "current_turn",
        },
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id=f"capture:{marker}",
            session_id=f"session:{marker}",
        ),
    )
    assert result.success is True
    assert handler.arguments is not None
    assert "Reply STOP" in handler.arguments["body"]
    assert "Why you received this" in handler.arguments["body"]
    assert result.metadata["outbound_event_ids"]
    async with db_pool.acquire() as conn:
        event = await conn.fetchrow(
            "SELECT status, provider_message_id FROM outbound_events WHERE id=$1::uuid",
            result.metadata["outbound_event_ids"][0],
        )
    assert event["status"] == "delivered"
    assert event["provider_message_id"] == "captured-provider-id"


async def test_registry_denies_missing_purpose_before_handler(db_pool):
    handler = _CaptureEmailHandler()
    registry = ToolRegistry(db_pool)
    registry.register(handler)
    result = await registry.execute(
        "test_outbound_email",
        {"to": f"{uuid4().hex}@example.com", "body": "No purpose."},
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id=f"missing:{uuid4()}",
        ),
    )
    assert result.success is False
    assert result.error_type.value == "invalid_params"
    assert "purpose_kind" in (result.error or "")
    assert handler.arguments is None


async def test_assigned_goal_heavily_discounts_outbound_tool_energy(db_pool):
    marker = uuid4().hex
    tool_name = f"test_outbound_energy_{marker}"
    registry = ToolRegistry(db_pool)
    registry.register(_CaptureEmailHandler(name=tool_name, energy_cost=8))
    await registry.sync_tool_catalog(force=True)

    async with db_pool.acquire() as conn:
        catalog_row = await conn.fetchrow(
            "SELECT default_energy_cost, metadata FROM tool_definitions WHERE name=$1",
            tool_name,
        )
        assert catalog_row is not None
        assert "outbound" in _json_result(catalog_row["metadata"])
        assigned_goal = await conn.fetchval(
            """
            INSERT INTO memories(type, goal_origin, content, status, metadata)
            VALUES ('goal', 'user_request', $1, 'active', '{"priority":"active"}'::jsonb)
            RETURNING id
            """,
            f"Assigned energy goal {marker}",
        )
        derived_goal = await conn.fetchval(
            """
            INSERT INTO memories(type, goal_origin, content, status, metadata)
            VALUES ('goal', 'derived', $1, 'active', '{"priority":"active"}'::jsonb)
            RETURNING id
            """,
            f"Derived energy goal {marker}",
        )
        assert (
            await conn.fetchval(
                "SELECT goal_origin::text FROM memories WHERE id=$1", assigned_goal
            )
            == "user_request"
        )
        context = json.dumps({"tool_context": "heartbeat", "energy_available": 20})
        assigned = _json_result(
            await conn.fetchval(
                "SELECT evaluate_tool_call($1, $2::jsonb, $3::jsonb)",
                tool_name,
                json.dumps(
                    {
                        "purpose_kind": "goal",
                        "purpose_reference": str(assigned_goal),
                    }
                ),
                context,
            )
        )
        derived = _json_result(
            await conn.fetchval(
                "SELECT evaluate_tool_call($1, $2::jsonb, $3::jsonb)",
                tool_name,
                json.dumps(
                    {
                        "purpose_kind": "goal",
                        "purpose_reference": str(derived_goal),
                    }
                ),
                context,
            )
        )
        multiplier = float(
            await conn.fetchval(
                "SELECT get_config_float('outbound.assigned_goal_energy_multiplier')"
            )
        )
        await conn.execute(
            "DELETE FROM memories WHERE id = ANY($1::uuid[])",
            [assigned_goal, derived_goal],
        )
        await conn.execute("DELETE FROM tool_definitions WHERE name=$1", tool_name)

    assert assigned["allowed"] is True, assigned
    assert assigned["energy_cost"] == math.ceil(8 * multiplier)
    assert derived["allowed"] is False
    assert derived["error_type"] == "insufficient_energy"
    assert assigned["energy_cost"] < derived["energy_cost"] == 8


async def test_generic_outbox_target_is_not_treated_as_primary(db_pool):
    from channels.outbox import _is_primary_user_envelope
    from services.outbound_safety import (
        finalize_outbox_outbound,
        prepare_outbox_outbound,
    )

    marker = uuid4().hex
    recipient = f"channel-{marker}"
    assert _is_primary_user_envelope({"_outbox_kind": "channel_message"}) is False
    assert _is_primary_user_envelope({"_outbox_kind": "user"}) is True

    denied = await prepare_outbox_outbound(
        db_pool,
        request_key=f"generic-missing:{marker}",
        channel="discord",
        recipient=recipient,
        identity_address=None,
        body="No durable purpose.",
        payload={"_outbox_kind": "channel_message"},
        primary_hint=False,
    )
    assert denied.allowed is False
    assert denied.error_type.value == "purpose_required"

    async with db_pool.acquire() as conn:
        goal_id = await conn.fetchval(
            """
            INSERT INTO memories(type, goal_origin, content, status, metadata)
            VALUES ('goal', 'user_request', $1, 'active', '{"priority":"active"}'::jsonb)
            RETURNING id
            """,
            f"Outbox routing goal {marker}",
        )
    allowed = await prepare_outbox_outbound(
        db_pool,
        request_key=f"generic-goal:{marker}",
        channel="discord",
        recipient=recipient,
        identity_address=None,
        body="A backed update.",
        payload={
            "_outbox_kind": "channel_message",
            "purpose_kind": "goal",
            "purpose_reference": str(goal_id),
        },
        primary_hint=False,
    )
    assert allowed.allowed is True
    assert "Reply STOP" in allowed.arguments["message"]
    await finalize_outbox_outbound(db_pool, allowed, delivered=False, error="test")

    async with db_pool.acquire() as conn:
        entity = f"discord:{recipient.lower()}"
        await conn.execute(
            "DELETE FROM outbound_events WHERE request_key LIKE $1",
            f"%:{marker}",
        )
        await conn.execute("DELETE FROM contact_budgets WHERE entity=$1", entity)
        await conn.execute(
            "DELETE FROM outbound_contact_endpoints WHERE entity=$1", entity
        )
        await conn.execute("DELETE FROM memories WHERE id=$1", goal_id)


def _json_result(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else value


async def test_channel_manager_acknowledges_stop_once_before_conversation(db_pool):
    from channels.base import ChannelMessage
    from channels.manager import ChannelManager

    marker = uuid4().hex
    phone = f"+1555{marker[:7]}"
    async with db_pool.acquire() as conn:
        contact_id = await conn.fetchval(
            "INSERT INTO contacts(name, phone) VALUES ($1, $2) RETURNING id",
            f"Manager STOP {marker}",
            phone,
        )

    adapter = type("StopAdapter", (), {})()
    adapter.channel_type = "signal"
    adapter.send = AsyncMock(return_value="ack-1")
    manager = ChannelManager(db_pool)
    manager.register(adapter)
    message = ChannelMessage(
        channel_type="signal",
        channel_id=phone,
        sender_id=phone,
        sender_name="Recipient",
        content="STOP.",
        message_id=f"stop-{marker}",
    )
    try:
        with (
            patch(
                "services.inbound_disposition.record_passive_observation",
                new=AsyncMock(),
            ),
            patch(
                "channels.manager.stream_channel_message", new=AsyncMock()
            ) as conversation,
        ):
            await manager._handle_message(message)
            await manager._handle_message(message)

        adapter.send.assert_awaited_once()
        assert "won't contact you again" in adapter.send.await_args.args[1]
        conversation.assert_not_called()
        async with db_pool.acquire() as conn:
            control = await conn.fetchrow(
                "SELECT blocked, source_channel FROM outbound_contact_controls WHERE entity=$1",
                f"contact:{contact_id}",
            )
        assert control["blocked"] is True
        assert control["source_channel"] == "signal"
    finally:
        async with db_pool.acquire() as conn:
            entity = f"contact:{contact_id}"
            await conn.execute(
                "DELETE FROM outbox_messages WHERE source='contact_opt_out' AND envelope#>>'{payload,context,entity}'=$1",
                entity,
            )
            await conn.execute(
                "DELETE FROM outbound_contact_control_events WHERE entity=$1", entity
            )
            await conn.execute(
                "DELETE FROM outbound_contact_controls WHERE entity=$1", entity
            )
            await conn.execute("DELETE FROM contact_budgets WHERE entity=$1", entity)
            await conn.execute(
                "DELETE FROM outbound_contact_endpoints WHERE entity=$1", entity
            )
            await conn.execute("DELETE FROM contacts WHERE id=$1", contact_id)


async def test_direct_channel_reply_is_disclosed_free_and_cross_channel_stoppable(
    db_pool,
):
    from channels.base import ChannelCapabilities, ChannelMessage
    from services.outbound_safety import GovernedReplyAdapter

    marker = uuid4().hex
    slack_identity = f"U-{marker[:12]}"
    room_id = f"C-{marker[:12]}"
    async with db_pool.acquire() as conn:
        contact_id = await conn.fetchval(
            """
            INSERT INTO contacts(name, metadata)
            VALUES ($1, jsonb_build_object('channels', jsonb_build_object('slack', $2::text)))
            RETURNING id
            """,
            f"Reply contact {marker}",
            slack_identity,
        )

    class ReplyAdapter:
        channel_type = "slack"
        capabilities = ChannelCapabilities(edit_message=True)
        is_connected = True

        def __init__(self):
            self.sent: list[str] = []
            self.edited: list[str] = []

        async def send(self, channel_id, text, **kwargs):
            self.sent.append(text)
            return "provider-reply-1"

        async def edit_message(self, channel_id, message_id, text):
            self.edited.append(text)
            return True

    raw_adapter = ReplyAdapter()
    inbound = ChannelMessage(
        channel_type="slack",
        channel_id=room_id,
        sender_id=slack_identity,
        sender_name="Recipient",
        content="Can you clarify?",
        message_id=f"inbound-{marker}",
    )
    governed = GovernedReplyAdapter(db_pool, raw_adapter, inbound)
    try:
        provider_id = await governed.send(
            room_id,
            "Yes — here is the clarification.",
            reply_to=inbound.message_id,
        )
        assert provider_id == "provider-reply-1"
        assert "Reply STOP" in raw_adapter.sent[0]

        assert await governed.edit_message(
            room_id, provider_id, "Updated clarification."
        )
        assert "Reply STOP" in raw_adapter.edited[0]

        async with db_pool.acquire() as conn:
            event = await conn.fetchrow(
                """
                SELECT entity, recipient, is_reply, charged_cost, status
                FROM outbound_events
                WHERE request_key LIKE $1
                ORDER BY created_at DESC LIMIT 1
                """,
                f"channel-reply:slack:{inbound.message_id}:%",
            )
            await conn.fetchval(
                "SELECT handle_inbound_contact_control('slack', $1, 'STOP', false, '{}'::jsonb)",
                slack_identity,
            )
        assert event["entity"] == f"contact:{contact_id}"
        assert event["recipient"] == room_id.lower()
        assert event["is_reply"] is True
        assert event["charged_cost"] == 0
        assert event["status"] == "delivered"

        blocked_adapter = ReplyAdapter()
        with pytest.raises(RuntimeError, match="opted out"):
            await GovernedReplyAdapter(db_pool, blocked_adapter, inbound).send(
                room_id, "This must stay silent."
            )
        assert blocked_adapter.sent == []
    finally:
        entity = f"contact:{contact_id}"
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM outbox_messages WHERE source='contact_opt_out' AND envelope#>>'{payload,context,entity}'=$1",
                entity,
            )
            await conn.execute("DELETE FROM outbound_events WHERE entity=$1", entity)
            await conn.execute(
                "DELETE FROM outbound_contact_control_events WHERE entity=$1", entity
            )
            await conn.execute(
                "DELETE FROM outbound_contact_controls WHERE entity=$1", entity
            )
            await conn.execute("DELETE FROM contact_budgets WHERE entity=$1", entity)
            await conn.execute(
                "DELETE FROM outbound_contact_endpoints WHERE entity=$1", entity
            )
            await conn.execute("DELETE FROM contacts WHERE id=$1", contact_id)
