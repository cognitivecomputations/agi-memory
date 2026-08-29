"""Isolated manager/adapter routing tests for inbound disposition."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from channels.base import ChannelCapabilities, ChannelMessage

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


class _Adapter:
    channel_type = "mock"
    capabilities = ChannelCapabilities()
    is_connected = True

    def __init__(self):
        self.send = AsyncMock(return_value="sent")
        self.download_attachment = AsyncMock()

    async def start(self, _callback):
        return None

    async def stop(self):
        return None

    async def send_typing(self, _channel_id):
        return None


def _message(*, content="hello", metadata=None) -> ChannelMessage:
    return ChannelMessage(
        channel_type="mock",
        channel_id="room",
        sender_id="sender",
        sender_name="Sender",
        content=content,
        message_id="message-1",
        metadata=metadata or {"is_group": True},
    )


async def _manager(monkeypatch):
    from channels.manager import ChannelManager

    manager = ChannelManager(MagicMock())
    manager.register(_Adapter())
    manager._run_conversation_turn = AsyncMock()
    manager._record_disposition_observation = AsyncMock()
    monkeypatch.setattr(
        "services.agent_questions.resolve_agent_question_from_inbound",
        AsyncMock(return_value={}),
    )
    return manager


async def test_manager_routes_observe_and_wake_only_to_passive_ledger(monkeypatch):
    manager = await _manager(monkeypatch)
    for disposition in ("observe", "wake", "unexpected"):
        result = {
            "disposition": disposition,
            "reason": "test",
            "reply_allowed": True,
            "is_operator": disposition == "wake",
            "audit_id": 1,
        }
        manager._resolve_inbound_disposition = AsyncMock(return_value=result)
        msg = _message(content=f"{disposition} message")

        await manager._handle_message(msg)

        manager._record_disposition_observation.assert_awaited_with(msg, result)

    manager._run_conversation_turn.assert_not_awaited()


async def test_manager_engage_uses_sql_stripped_text_and_operator_identity(monkeypatch):
    manager = await _manager(monkeypatch)
    manager._resolve_inbound_disposition = AsyncMock(
        return_value={
            "disposition": "engage",
            "reason": "trigger_match",
            "reply_allowed": True,
            "is_operator": True,
            "trigger_stripped_text": "status please",
            "audit_id": 2,
        }
    )

    await manager._handle_message(_message(content="hexis: status please"))

    handled = manager._run_conversation_turn.await_args.args[0]
    assert handled.content == "status please"
    assert manager._run_conversation_turn.await_args.kwargs["is_operator"] is False
    manager._record_disposition_observation.assert_not_awaited()


async def test_manager_never_engages_above_resolver_allowlist_ceiling(monkeypatch):
    manager = await _manager(monkeypatch)
    manager._resolve_inbound_disposition = AsyncMock(
        return_value={
            "disposition": "engage",
            "reason": "bad_classifier",
            "reply_allowed": False,
            "is_operator": False,
            "audit_id": 3,
        }
    )
    msg = _message()

    await manager._handle_message(msg)

    manager._run_conversation_turn.assert_not_awaited()
    recorded = manager._record_disposition_observation.await_args.args[1]
    assert recorded["disposition"] == "observe"
    assert recorded["reason"] == "reply_allowlist_belt"


async def test_runtime_kill_switch_honors_forward_all_legacy_hint(monkeypatch):
    manager = await _manager(monkeypatch)
    manager._resolve_inbound_disposition = AsyncMock(return_value=None)
    manager._check_user_allowed = AsyncMock(return_value=True)

    await manager._handle_message(
        _message(metadata={"is_group": True, "gate_hint": "not_allowed_channel"})
    )

    manager._check_user_allowed.assert_not_awaited()
    manager._run_conversation_turn.assert_not_awaited()


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: __import__(
                "channels.discord_adapter", fromlist=["DiscordAdapter"]
            ).DiscordAdapter({"forward_all": "true"}),
            id="discord",
        ),
        pytest.param(
            lambda: __import__(
                "channels.telegram_adapter", fromlist=["TelegramAdapter"]
            ).TelegramAdapter({"forward_all": True}),
            id="telegram",
        ),
        pytest.param(
            lambda: __import__(
                "channels.slack_adapter", fromlist=["SlackAdapter"]
            ).SlackAdapter({"forward_all": 1}),
            id="slack",
        ),
        pytest.param(
            lambda: __import__(
                "channels.signal_adapter", fromlist=["SignalAdapter"]
            ).SignalAdapter({"forward_all": "yes"}),
            id="signal",
        ),
        pytest.param(
            lambda: __import__(
                "channels.whatsapp_adapter", fromlist=["WhatsAppAdapter"]
            ).WhatsAppAdapter({"forward_all": "on"}),
            id="whatsapp",
        ),
        pytest.param(
            lambda: __import__(
                "channels.imessage_adapter", fromlist=["IMessageAdapter"]
            ).IMessageAdapter({"forward_all": True}),
            id="imessage",
        ),
        pytest.param(
            lambda: __import__(
                "channels.matrix_adapter", fromlist=["MatrixAdapter"]
            ).MatrixAdapter({"forward_all": True}),
            id="matrix",
        ),
    ],
)
async def test_every_adapter_supports_transport_only_mode(factory):
    assert factory()._forward_all is True


async def test_imessage_forward_all_preserves_raw_blocked_message():
    from channels.imessage_adapter import IMessageAdapter

    adapter = IMessageAdapter(
        {"allowed_handles": ["allowed@example.com"]}, forward_all=True
    )
    adapter._on_message = AsyncMock()
    await adapter._handle_message(
        {
            "guid": "m1",
            "isFromMe": False,
            "text": "ambient",
            "handle": {"address": "blocked@example.com"},
            "chats": [{"guid": "room", "isGroup": False}],
            "attachments": [],
        }
    )

    forwarded = adapter._on_message.await_args.args[0]
    assert forwarded.content == "ambient"
    assert forwarded.metadata["gate_hint"] == "not_allowlisted"


async def test_signal_forward_all_preserves_raw_blocked_message():
    from channels.signal_adapter import SignalAdapter

    adapter = SignalAdapter({"allowed_numbers": ["+1000"]}, forward_all=True)
    adapter._on_message = AsyncMock()
    await adapter._handle_sse_event(
        json.dumps(
            {
                "envelope": {
                    "source": "+2000",
                    "sourceName": "Other",
                    "dataMessage": {"message": "ambient", "timestamp": 1},
                }
            }
        )
    )

    forwarded = adapter._on_message.await_args.args[0]
    assert forwarded.metadata["gate_hint"] == "not_allowlisted"


async def test_slack_forward_all_preserves_unmentioned_outside_room():
    from channels.slack_adapter import SlackAdapter

    adapter = SlackAdapter({"allowed_channels": ["C-ALLOWED"]}, forward_all=True)
    adapter._bot_user_id = "U-BOT"
    adapter._on_message = AsyncMock()
    client = MagicMock()
    client.users_info = AsyncMock(return_value={"user": {"profile": {}}})
    await adapter._handle_slack_message(
        {
            "user": "U-OTHER",
            "channel": "C-OTHER",
            "channel_type": "channel",
            "text": "ambient",
            "ts": "1.0",
        },
        client,
    )

    forwarded = adapter._on_message.await_args.args[0]
    assert forwarded.metadata["gate_hint"] == "not_allowed_channel"
    assert forwarded.metadata["is_mention"] is False


async def test_whatsapp_forward_all_preserves_blocked_sender():
    from channels.whatsapp_adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter({"allowed_numbers": ["1000"]}, forward_all=True)
    adapter._on_message = AsyncMock()
    await adapter._handle_message(
        {"id": "m1", "type": "text", "from": "2000", "text": {"body": "hi"}},
        {"2000": "Other"},
    )

    forwarded = adapter._on_message.await_args.args[0]
    assert forwarded.metadata["gate_hint"] == "not_allowlisted"


async def test_matrix_forward_all_preserves_outside_room():
    from channels.matrix_adapter import MatrixAdapter

    adapter = MatrixAdapter({"allowed_rooms": ["!allowed"]}, forward_all=True)
    adapter._user_id = "@hexis:test"
    adapter._on_message = AsyncMock()
    room = SimpleNamespace(
        room_id="!other",
        member_count=3,
        display_name="Other",
        user_name=lambda _sender: "Sender",
    )
    event = SimpleNamespace(
        sender="@sender:test", body="ambient", source={}, event_id="$event"
    )
    await adapter._handle_matrix_message(room, event)

    forwarded = adapter._on_message.await_args.args[0]
    assert forwarded.metadata["gate_hint"] == "not_allowed_room"


async def test_telegram_forward_all_preserves_unmentioned_outside_chat():
    from channels.telegram_adapter import TelegramAdapter

    adapter = TelegramAdapter({"allowed_chat_ids": ["1"]}, forward_all=True)
    adapter._bot_username = "hexis_bot"
    adapter._on_message = AsyncMock()
    message = SimpleNamespace(
        chat=SimpleNamespace(type="group", id=2),
        from_user=SimpleNamespace(id=3, full_name="Sender", username="sender"),
        text="ambient",
        caption=None,
        photo=None,
        document=None,
        voice=None,
        audio=None,
        message_thread_id=None,
        is_topic_message=False,
        message_id=4,
        reply_to_message=None,
    )
    await adapter._handle_telegram_message(SimpleNamespace(message=message))

    forwarded = adapter._on_message.await_args.args[0]
    assert forwarded.metadata["gate_hint"] == "not_allowed_chat"


async def test_discord_forward_all_preserves_outside_guild(monkeypatch):
    from channels.discord_adapter import DiscordAdapter

    class DMChannel:
        pass

    class Thread:
        pass

    monkeypatch.setitem(
        sys.modules,
        "discord",
        SimpleNamespace(DMChannel=DMChannel, Thread=Thread),
    )
    adapter = DiscordAdapter({"allowed_guilds": ["1"]}, forward_all=True)
    bot_user = SimpleNamespace(id=99, mentioned_in=lambda _message: False)
    adapter._client = SimpleNamespace(user=bot_user)
    adapter._on_message = AsyncMock()
    message = SimpleNamespace(
        author=SimpleNamespace(id=2, bot=False, display_name="Sender"),
        content="ambient",
        attachments=[],
        channel=SimpleNamespace(id=3),
        guild=SimpleNamespace(id=4),
        reference=None,
        id=5,
    )
    await adapter._handle_discord_message(message)

    forwarded = adapter._on_message.await_args.args[0]
    assert forwarded.metadata["gate_hint"] == "not_allowed_guild"
