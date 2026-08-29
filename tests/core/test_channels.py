"""
Tests for the Hexis Channel System.

Covers:
- ChannelMessage, ChannelCapabilities, chunk_text (base types)
- ChannelManager (routing, lifecycle)
- Discord adapter (normalization, allowlists, chunking)
- Telegram adapter (normalization, allowlists)
- Conversation handler (session management, message logging)
- Hook event registration
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


# ============================================================================
# Base Types
# ============================================================================


class TestChannelMessage:
    def test_defaults(self):
        from channels.base import ChannelMessage

        msg = ChannelMessage(
            channel_type="discord",
            channel_id="123",
            sender_id="456",
            sender_name="TestUser",
            content="Hello",
            message_id="789",
        )
        assert msg.channel_type == "discord"
        assert msg.content == "Hello"
        assert msg.reply_to_id is None
        assert msg.thread_id is None
        assert msg.attachments == []
        assert msg.metadata == {}
        assert isinstance(msg.timestamp, datetime)

    def test_full_fields(self):
        from channels.base import ChannelMessage

        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        msg = ChannelMessage(
            channel_type="telegram",
            channel_id="chat-1",
            sender_id="user-1",
            sender_name="Alice",
            content="Hello World",
            message_id="msg-1",
            reply_to_id="msg-0",
            thread_id="thread-1",
            attachments=[{"type": "image", "url": "http://example.com/img.png"}],
            metadata={"is_private": True},
            timestamp=ts,
        )
        assert msg.reply_to_id == "msg-0"
        assert msg.thread_id == "thread-1"
        assert len(msg.attachments) == 1
        assert msg.metadata["is_private"] is True
        assert msg.timestamp == ts

    def test_is_group_from_metadata(self):
        """is_group (#92) drives exclude_sensitive recall in group rooms:
        explicit is_group wins, is_private implies its inverse, and the
        default is a 1:1 conversation."""
        from channels.base import ChannelMessage

        def msg(metadata):
            return ChannelMessage(
                channel_type="telegram",
                channel_id="chat-1",
                sender_id="user-1",
                sender_name="Alice",
                content="Hello",
                message_id="msg-1",
                metadata=metadata,
            )

        assert msg({"is_group": True}).is_group is True
        assert msg({"is_group": False}).is_group is False
        assert msg({"is_private": True}).is_group is False
        assert msg({"is_private": False}).is_group is True
        assert msg({}).is_group is False


class TestChannelCapabilities:
    def test_defaults(self):
        from channels.base import ChannelCapabilities

        caps = ChannelCapabilities()
        assert caps.threads is False
        assert caps.reactions is False
        assert caps.media is False
        assert caps.typing_indicator is False
        assert caps.edit_message is False
        assert caps.max_message_length == 4000

    def test_discord_capabilities(self):
        from channels.base import ChannelCapabilities

        caps = ChannelCapabilities(
            threads=True,
            reactions=True,
            media=True,
            typing_indicator=True,
            edit_message=True,
            max_message_length=2000,
        )
        assert caps.threads is True
        assert caps.max_message_length == 2000


class TestChunkText:
    def test_short_text_no_chunk(self):
        from channels.base import chunk_text

        result = chunk_text("Hello", 100)
        assert result == ["Hello"]

    def test_exact_limit(self):
        from channels.base import chunk_text

        text = "x" * 100
        result = chunk_text(text, 100)
        assert result == [text]

    def test_paragraph_split(self):
        from channels.base import chunk_text

        text = "First paragraph.\n\nSecond paragraph."
        result = chunk_text(text, 25)
        assert len(result) == 2
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."

    def test_newline_split(self):
        from channels.base import chunk_text

        text = "Line one.\nLine two.\nLine three."
        result = chunk_text(text, 20)
        assert len(result) >= 2

    def test_sentence_split(self):
        from channels.base import chunk_text

        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text(text, 30)
        assert len(result) >= 2

    def test_hard_split(self):
        from channels.base import chunk_text

        text = "x" * 200
        result = chunk_text(text, 100)
        assert len(result) == 2
        assert len(result[0]) == 100
        assert len(result[1]) == 100

    def test_empty_text(self):
        from channels.base import chunk_text

        result = chunk_text("", 100)
        assert result == [""]


# ============================================================================
# Channel Manager
# ============================================================================


class _MockAdapter:
    """A mock adapter for testing the ChannelManager."""

    def __init__(self, ctype: str = "mock"):
        self._ctype = ctype
        self._connected = False
        self._on_message = None
        self._sent_messages: list[dict] = []
        self._typing_channels: list[str] = []
        self.start_called = False
        self.stop_called = False

    @property
    def channel_type(self) -> str:
        return self._ctype

    @property
    def capabilities(self):
        from channels.base import ChannelCapabilities

        return ChannelCapabilities(
            typing_indicator=True,
            max_message_length=2000,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(self, on_message):
        self.start_called = True
        self._on_message = on_message
        self._connected = True
        # Don't block - just mark as started
        # In real adapters, this would block

    async def stop(self):
        self.stop_called = True
        self._connected = False

    async def send(self, channel_id, text, *, reply_to=None, thread_id=None):
        self._sent_messages.append(
            {
                "channel_id": channel_id,
                "text": text,
                "reply_to": reply_to,
                "thread_id": thread_id,
            }
        )
        return "sent-msg-1"

    async def send_typing(self, channel_id):
        self._typing_channels.append(channel_id)


class TestChannelManager:
    def test_register_adapter(self):
        from channels.manager import ChannelManager

        manager = ChannelManager(pool=MagicMock())
        adapter = _MockAdapter("test-channel")
        manager.register(adapter)

        assert "test-channel" in manager.adapters
        assert manager.adapters["test-channel"] is adapter

    def test_register_duplicate_replaces(self):
        from channels.manager import ChannelManager

        manager = ChannelManager(pool=MagicMock())
        adapter1 = _MockAdapter("test")
        adapter2 = _MockAdapter("test")
        manager.register(adapter1)
        manager.register(adapter2)

        assert manager.adapters["test"] is adapter2

    async def test_send_routes_to_adapter(self):
        from channels.manager import ChannelManager

        manager = ChannelManager(pool=MagicMock())
        adapter = _MockAdapter("test")
        manager.register(adapter)

        result = await manager.send("test", "channel-1", "Hello!")
        assert result == "sent-msg-1"
        assert len(adapter._sent_messages) == 1
        assert adapter._sent_messages[0]["text"] == "Hello!"

    async def test_send_unknown_channel_returns_none(self):
        from channels.manager import ChannelManager

        manager = ChannelManager(pool=MagicMock())
        result = await manager.send("nonexistent", "ch-1", "Hello!")
        assert result is None

    async def test_adapter_intake_stays_open_while_a_turn_waits(self):
        from channels.base import ChannelMessage
        from channels.manager import ChannelManager

        manager = ChannelManager(pool=MagicMock())
        adapter = _MockAdapter("test")
        adapter_gate = asyncio.Event()
        callback_ready = asyncio.Event()
        received: list[ChannelMessage] = []
        turn_gate = asyncio.Event()

        async def start(on_message):
            adapter._on_message = on_message
            callback_ready.set()
            await adapter_gate.wait()

        async def handle(message):
            received.append(message)
            await turn_gate.wait()

        adapter.start = start
        manager.register(adapter)
        manager._running = True
        manager._record_runtime_status = AsyncMock()
        manager._record_presence = AsyncMock()
        manager._handle_message = handle
        await manager._start_adapter("test", adapter)
        await asyncio.wait_for(callback_ready.wait(), timeout=1)

        first = ChannelMessage(
            channel_type="test",
            channel_id="room",
            sender_id="user",
            sender_name="User",
            content="Start the task",
            message_id="m1",
        )
        reply = ChannelMessage(
            channel_type="test",
            channel_id="room",
            sender_id="user",
            sender_name="User",
            content="2",
            message_id="m2",
        )
        await asyncio.wait_for(adapter._on_message(first), timeout=1)
        await asyncio.wait_for(adapter._on_message(reply), timeout=1)
        await asyncio.sleep(0)

        assert [message.message_id for message in received] == ["m1", "m2"]
        await manager.stop_all()

    def test_status(self):
        from channels.manager import ChannelManager

        manager = ChannelManager(pool=MagicMock())
        adapter = _MockAdapter("discord")
        manager.register(adapter)

        status = manager.status()
        assert len(status) == 1
        assert status[0]["channel_type"] == "discord"
        assert status[0]["connected"] is False
        assert status[0]["capabilities"]["max_message_length"] == 2000


# ============================================================================
# Discord Adapter
# ============================================================================


class TestDiscordAdapter:
    def test_channel_type(self):
        from channels.discord_adapter import DiscordAdapter

        adapter = DiscordAdapter()
        assert adapter.channel_type == "discord"

    def test_capabilities(self):
        from channels.discord_adapter import DiscordAdapter

        adapter = DiscordAdapter()
        caps = adapter.capabilities
        assert caps.threads is True
        assert caps.reactions is True
        assert caps.media is True
        assert caps.typing_indicator is True
        assert caps.max_message_length == 2000

    def test_parse_allowlist_star(self):
        from channels.discord_adapter import DiscordAdapter

        assert DiscordAdapter._parse_allowlist("*") is None
        assert DiscordAdapter._parse_allowlist(None) is None

    def test_parse_allowlist_json(self):
        from channels.discord_adapter import DiscordAdapter

        result = DiscordAdapter._parse_allowlist('["123", "456"]')
        assert result == {"123", "456"}

    def test_parse_allowlist_list(self):
        from channels.discord_adapter import DiscordAdapter

        result = DiscordAdapter._parse_allowlist([123, 456])
        assert result == {"123", "456"}

    def test_parse_allowlist_single(self):
        from channels.discord_adapter import DiscordAdapter

        result = DiscordAdapter._parse_allowlist("not-json")
        assert result == {"not-json"}

    def test_not_connected_by_default(self):
        from channels.discord_adapter import DiscordAdapter

        adapter = DiscordAdapter()
        assert adapter.is_connected is False

    def test_token_resolution_env(self, monkeypatch):
        from channels.discord_adapter import _resolve_token

        monkeypatch.setenv("MY_TOKEN", "bot-token-value")
        result = _resolve_token({"bot_token": "MY_TOKEN"})
        assert result == "bot-token-value"

    def test_token_resolution_missing(self, monkeypatch):
        from channels.discord_adapter import _resolve_token

        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        result = _resolve_token({})
        assert result is None


# ============================================================================
# Telegram Adapter
# ============================================================================


class TestTelegramAdapter:
    def test_channel_type(self):
        from channels.telegram_adapter import TelegramAdapter

        adapter = TelegramAdapter()
        assert adapter.channel_type == "telegram"

    def test_capabilities(self):
        from channels.telegram_adapter import TelegramAdapter

        adapter = TelegramAdapter()
        caps = adapter.capabilities
        assert caps.threads is True
        assert caps.reactions is True
        assert caps.media is True
        assert caps.typing_indicator is True
        assert caps.max_message_length == 4096

    def test_parse_allowlist(self):
        from channels.telegram_adapter import TelegramAdapter

        assert TelegramAdapter._parse_allowlist("*") is None
        assert TelegramAdapter._parse_allowlist(["-100123"]) == {"-100123"}

    def test_not_connected_by_default(self):
        from channels.telegram_adapter import TelegramAdapter

        adapter = TelegramAdapter()
        assert adapter.is_connected is False

    def test_token_resolution_env(self, monkeypatch):
        from channels.telegram_adapter import _resolve_token

        monkeypatch.setenv("MY_TG_TOKEN", "123:ABC-def")
        result = _resolve_token({"bot_token": "MY_TG_TOKEN"})
        assert result == "123:ABC-def"

    def test_token_resolution_missing(self, monkeypatch):
        from channels.telegram_adapter import _resolve_token

        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        result = _resolve_token({})
        assert result is None

    async def test_send_media_prefers_local_path(self):
        from channels.media import Attachment
        from channels.telegram_adapter import TelegramAdapter

        adapter = TelegramAdapter()
        bot = MagicMock()

        class _Sent:
            message_id = 42

        bot.send_photo = AsyncMock(return_value=_Sent())
        adapter._application = MagicMock(bot=bot)

        result = await adapter.send_media(
            "123",
            Attachment(url="", local_path="/tmp/image.png", mime_type="image/png"),
            "caption",
        )

        assert result == "42"
        bot.send_photo.assert_awaited_once()
        kwargs = bot.send_photo.await_args.kwargs
        assert kwargs["photo"] == "/tmp/image.png"


# ============================================================================
# Conversation Handler (DB integration)
# ============================================================================


class TestChannelConversation:
    """Integration tests for channel conversation session management."""

    _TEST_PREFIX = "test_ch_conv_"

    async def _cleanup(self, conn, sender_id: str):
        """Delete test data by sender_id."""
        session_ids = await conn.fetch(
            "SELECT id FROM channel_sessions WHERE sender_id = $1",
            sender_id,
        )
        for row in session_ids:
            await conn.execute(
                "DELETE FROM channel_messages WHERE session_id = $1", row["id"]
            )
        await conn.execute(
            "DELETE FROM channel_sessions WHERE sender_id = $1", sender_id
        )

    @staticmethod
    async def _prepare_turn(
        conn, sender_id: str, channel_id: str, message_id: str, content: str = "Hello"
    ) -> dict:
        """Run the DB-owned turn preparation (prepare_channel_turn, db/34)."""
        raw = await conn.fetchval(
            "SELECT prepare_channel_turn($1::jsonb)",
            json.dumps(
                {
                    "channel_type": "test",
                    "channel_id": channel_id,
                    "sender_id": sender_id,
                    "sender_name": "TestBot",
                    "content": content,
                    "message_id": message_id,
                }
            ),
        )
        return json.loads(raw) if isinstance(raw, str) else raw

    async def test_session_creation(self, db_pool):
        """Test that a session is created for a new sender."""
        sender_id = f"{self._TEST_PREFIX}create_{id(self)}"

        async with db_pool.acquire() as conn:
            try:
                turn = await self._prepare_turn(
                    conn, sender_id, "test-channel-1", "test-msg-1"
                )
                assert turn["allowed"] is True
                assert turn["session_id"] is not None
                assert turn["history"] == []

                # Second call returns same session
                turn2 = await self._prepare_turn(
                    conn, sender_id, "test-channel-1", "test-msg-1b"
                )
                assert turn2["session_id"] == turn["session_id"]
            finally:
                await self._cleanup(conn, sender_id)

    async def test_session_history_update(self, db_pool):
        """Test session history is stored by finalize and retrieved by prepare."""
        sender_id = f"{self._TEST_PREFIX}hist_{id(self)}"

        async with db_pool.acquire() as conn:
            try:
                turn = await self._prepare_turn(
                    conn, sender_id, "test-channel-2", "test-msg-2"
                )

                history = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
                await conn.fetchval(
                    "SELECT finalize_channel_turn($1::uuid, $2, $3, $4::jsonb)",
                    turn["session_id"],
                    "Hello",
                    "Hi there!",
                    json.dumps({"history": history}),
                )

                turn2 = await self._prepare_turn(
                    conn, sender_id, "test-channel-2", "test-msg-2b"
                )
                retrieved = turn2["history"]
                assert len(retrieved) == 2
                assert retrieved[0]["role"] == "user"
                assert retrieved[1]["content"] == "Hi there!"
            finally:
                await self._cleanup(conn, sender_id)

    async def test_message_logging(self, db_pool):
        """Test that the DB turn lifecycle logs inbound and outbound messages."""
        sender_id = f"{self._TEST_PREFIX}log_{id(self)}"

        async with db_pool.acquire() as conn:
            try:
                turn = await self._prepare_turn(
                    conn, sender_id, "test-channel-3", "test-msg-3"
                )
                await conn.fetchval(
                    "SELECT finalize_channel_turn($1::uuid, $2, $3, $4::jsonb)",
                    turn["session_id"],
                    "Hello",
                    "Hi there!",
                    json.dumps(
                        {"history": [], "platform_message_id": "test-msg-3-out"}
                    ),
                )

                rows = await conn.fetch(
                    "SELECT direction FROM channel_messages WHERE session_id = $1::uuid ORDER BY created_at",
                    turn["session_id"],
                )
                assert [r["direction"] for r in rows] == ["inbound", "outbound"]
            finally:
                await self._cleanup(conn, sender_id)

    async def test_group_flag_reaches_chat_turn(self, db_pool, monkeypatch):
        """A group-room message threads is_group into chat_turn (#92), which
        drives hydrate(exclude_sensitive=True) so private memories stay out
        of shared rooms."""
        import services.chat
        from channels.base import ChannelMessage
        from channels.conversation import process_channel_message

        sender_id = f"{self._TEST_PREFIX}group_{id(self)}"
        chat_turn_mock = AsyncMock(return_value={"assistant": "ok", "history": []})
        monkeypatch.setattr(services.chat, "chat_turn", chat_turn_mock)

        msg = ChannelMessage(
            channel_type="test",
            channel_id="test-channel-4",
            sender_id=sender_id,
            sender_name="TestBot",
            content="Hello from the group",
            message_id="test-msg-4",
            metadata={"is_group": True},
        )
        try:
            chunks = await process_channel_message(msg, db_pool)
            assert chunks == ["ok"]
            assert chat_turn_mock.await_args.kwargs["is_group"] is True
            assert chat_turn_mock.await_args.kwargs["trusted_operator"] is False
            assert chat_turn_mock.await_args.kwargs["surface"] == "test"
        finally:
            async with db_pool.acquire() as conn:
                await self._cleanup(conn, sender_id)

    async def test_verified_operator_context_reaches_chat_turn(
        self, db_pool, monkeypatch
    ):
        import services.chat
        from channels.base import ChannelMessage
        from channels.conversation import process_channel_message

        sender_id = f"{self._TEST_PREFIX}operator_{id(self)}"
        chat_turn_mock = AsyncMock(return_value={"assistant": "ok", "history": []})
        monkeypatch.setattr(services.chat, "chat_turn", chat_turn_mock)
        msg = ChannelMessage(
            channel_type="slack",
            channel_id="private-owner",
            sender_id=sender_id,
            sender_name="Owner",
            content="Always cite sources.",
            message_id="operator-message-1",
            metadata={"is_group": False},
        )
        try:
            assert await process_channel_message(msg, db_pool, is_operator=True) == [
                "ok"
            ]
            kwargs = chat_turn_mock.await_args.kwargs
            assert kwargs["trusted_operator"] is True
            assert kwargs["operator_context"]["raw_text"] == "Always cite sources."
            assert (
                kwargs["operator_context"]["metadata"]["message_id"]
                == "operator-message-1"
            )
        finally:
            async with db_pool.acquire() as conn:
                await self._cleanup(conn, sender_id)


# ============================================================================
# Hook Event
# ============================================================================


class TestChannelHookEvent:
    def test_channel_message_received_event(self):
        from core.tools.hooks import HookEvent

        assert HookEvent.CHANNEL_MESSAGE_RECEIVED == "channel_message_received"

    async def test_hook_fires_for_channel_event(self):
        from core.tools.hooks import HookEvent, HookContext, HookOutcome, HookRegistry

        registry = HookRegistry()
        received = []

        async def on_channel_msg(ctx: HookContext):
            received.append(ctx)
            return HookOutcome.passthrough()

        registry.register_function(
            HookEvent.CHANNEL_MESSAGE_RECEIVED,
            on_channel_msg,
            source="test",
        )

        ctx = HookContext(
            event=HookEvent.CHANNEL_MESSAGE_RECEIVED,
            metadata={"channel_type": "discord", "sender": "user-1"},
        )
        await registry.run(HookEvent.CHANNEL_MESSAGE_RECEIVED, ctx)
        assert len(received) == 1
        assert received[0].metadata["channel_type"] == "discord"


# ============================================================================
# Channel Worker Config Loading
# ============================================================================


class TestChannelWorkerConfig:
    async def test_load_channel_config(self, db_pool):
        """Test loading channel config from DB."""
        from services.channel_worker import _load_channel_config

        async with db_pool.acquire() as conn:
            try:
                # Insert a test config value via set_config(key, jsonb)
                await conn.execute(
                    "SELECT set_config($1, $2::jsonb)",
                    "channel.test.bot_token",
                    '"MY_TEST_TOKEN"',
                )

                config = await _load_channel_config(conn, "test")
                assert config.get("bot_token") == "MY_TEST_TOKEN"
            finally:
                await conn.execute(
                    "DELETE FROM config WHERE key = $1",
                    "channel.test.bot_token",
                )
