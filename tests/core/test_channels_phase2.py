"""
Tests for Hexis Channel System — Phase 2 features.

Covers:
- Media module (Attachment, SSRF guard, download)
- New adapters (Slack, Signal, WhatsApp, iMessage, Matrix)
- Streaming coalescer
- Slash commands
- Outbox consumer
- Energy budgeting helpers
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


# ============================================================================
# Media Module
# ============================================================================


class TestAttachment:
    def test_from_dict_basic(self):
        from channels.media import Attachment

        att = Attachment.from_dict(
            {
                "url": "https://example.com/file.png",
                "filename": "file.png",
                "mime_type": "image/png",
                "size": 1024,
            }
        )
        assert att.url == "https://example.com/file.png"
        assert att.filename == "file.png"
        assert att.mime_type == "image/png"
        assert att.size == 1024
        assert att.local_path is None

    def test_from_dict_empty(self):
        from channels.media import Attachment

        att = Attachment.from_dict({})
        assert att.url == ""
        assert att.filename is None

    def test_to_dict_roundtrip(self):
        from channels.media import Attachment

        att = Attachment(
            url="https://example.com/file.png",
            filename="file.png",
            mime_type="image/png",
            size=2048,
            platform_id="abc",
        )
        d = att.to_dict()
        att2 = Attachment.from_dict(d)
        assert att2.url == att.url
        assert att2.filename == att.filename
        assert att2.mime_type == att.mime_type
        assert att2.size == att.size

    def test_describe_with_all_fields(self):
        from channels.media import Attachment

        att = Attachment(
            url="https://example.com/photo.jpg",
            filename="photo.jpg",
            mime_type="image/jpeg",
            size=2 * 1024 * 1024,
        )
        desc = att.describe()
        assert "photo.jpg" in desc
        assert "image/jpeg" in desc
        assert "2.0MB" in desc

    def test_describe_small_file(self):
        from channels.media import Attachment

        att = Attachment(url="https://example.com/tiny.txt", size=500)
        desc = att.describe()
        assert "500B" in desc

    def test_describe_kb_file(self):
        from channels.media import Attachment

        att = Attachment(
            url="https://example.com/mid.txt", size=5 * 1024, filename="mid.txt"
        )
        desc = att.describe()
        assert "5KB" in desc

    def test_describe_url_only(self):
        from channels.media import Attachment

        att = Attachment(url="https://example.com/unknown")
        desc = att.describe()
        assert "example.com" in desc


class TestSSRFGuard:
    def test_safe_url(self):
        from channels.media import is_safe_url

        assert is_safe_url("https://cdn.discord.com/attachments/123/file.png") is True

    def test_blocks_loopback(self):
        from channels.media import is_safe_url

        assert is_safe_url("http://127.0.0.1/secrets") is False

    def test_blocks_private_10(self):
        from channels.media import is_safe_url

        assert is_safe_url("http://10.0.0.1/internal") is False

    def test_blocks_private_172(self):
        from channels.media import is_safe_url

        assert is_safe_url("http://172.16.0.1/internal") is False

    def test_blocks_private_192(self):
        from channels.media import is_safe_url

        assert is_safe_url("http://192.168.1.1/internal") is False

    def test_blocks_link_local(self):
        from channels.media import is_safe_url

        assert is_safe_url("http://169.254.169.254/metadata") is False

    def test_blocks_metadata_hostname(self):
        from channels.media import is_safe_url

        assert is_safe_url("http://metadata.google.internal/v1/") is False

    def test_blocks_ipv6_loopback(self):
        from channels.media import is_safe_url

        assert is_safe_url("http://[::1]/secret") is False

    def test_allows_domain_names(self):
        from channels.media import is_safe_url

        assert is_safe_url("https://files.slack.com/file.pdf") is True

    def test_empty_url(self):
        from channels.media import is_safe_url

        assert is_safe_url("") is False


class TestChannelMessageAttachmentConversion:
    def test_raw_dicts_converted(self):
        from channels.base import ChannelMessage
        from channels.media import Attachment

        msg = ChannelMessage(
            channel_type="test",
            channel_id="ch1",
            sender_id="u1",
            sender_name="User",
            content="Hi",
            message_id="m1",
            attachments=[
                {"url": "https://example.com/a.png", "filename": "a.png"},
                {"url": "https://example.com/b.pdf"},
            ],
        )
        assert len(msg.attachments) == 2
        assert isinstance(msg.attachments[0], Attachment)
        assert msg.attachments[0].filename == "a.png"

    def test_attachment_instances_preserved(self):
        from channels.base import ChannelMessage
        from channels.media import Attachment

        att = Attachment(url="https://example.com/c.jpg", filename="c.jpg")
        msg = ChannelMessage(
            channel_type="test",
            channel_id="ch1",
            sender_id="u1",
            sender_name="User",
            content="Hi",
            message_id="m1",
            attachments=[att],
        )
        assert msg.attachments[0] is att


# ============================================================================
# Slack Adapter
# ============================================================================


class TestSlackAdapter:
    def test_channel_type(self):
        from channels.slack_adapter import SlackAdapter

        adapter = SlackAdapter()
        assert adapter.channel_type == "slack"

    def test_capabilities(self):
        from channels.slack_adapter import SlackAdapter

        caps = SlackAdapter().capabilities
        assert caps.threads is True
        assert caps.reactions is True
        assert caps.media is True
        assert caps.typing_indicator is True
        assert caps.edit_message is True
        assert caps.max_message_length == 4000

    def test_parse_allowlist_star(self):
        from channels.slack_adapter import SlackAdapter

        assert SlackAdapter._parse_allowlist("*") is None
        assert SlackAdapter._parse_allowlist(None) is None

    def test_parse_allowlist_json(self):
        from channels.slack_adapter import SlackAdapter

        result = SlackAdapter._parse_allowlist('["C123", "C456"]')
        assert result == {"C123", "C456"}

    def test_parse_allowlist_list(self):
        from channels.slack_adapter import SlackAdapter

        result = SlackAdapter._parse_allowlist(["C123", "C456"])
        assert result == {"C123", "C456"}

    def test_not_connected_by_default(self):
        from channels.slack_adapter import SlackAdapter

        assert SlackAdapter().is_connected is False

    async def test_operator_dm_bypasses_room_allowlist_for_approval_reply(self):
        from channels.slack_adapter import SlackAdapter

        adapter = SlackAdapter(
            {"allowed_channels": ["C-ALLOWED"], "operator_user_id": "U-OWNER"}
        )
        adapter._bot_user_id = "U-BOT"
        adapter._on_message = AsyncMock()
        client = MagicMock()
        client.users_info = AsyncMock(return_value={"user": {"profile": {}}})

        await adapter._handle_slack_message(
            {
                "user": "U-OWNER",
                "channel": "D-PRIVATE",
                "channel_type": "im",
                "text": "approve 12345678",
                "ts": "171.1",
            },
            client,
        )

        adapter._on_message.assert_awaited_once()

    def test_token_resolution_env(self, monkeypatch):
        from channels.slack_adapter import _resolve_token

        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token-value")
        result = _resolve_token({}, "bot_token", "SLACK_BOT_TOKEN")
        assert result == "xoxb-test-token-value"

    def test_token_resolution_missing(self, monkeypatch):
        from channels.slack_adapter import _resolve_token

        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        result = _resolve_token({}, "bot_token", "SLACK_BOT_TOKEN")
        assert result is None


# ============================================================================
# Signal Adapter
# ============================================================================


class TestSignalAdapter:
    def test_channel_type(self):
        from channels.signal_adapter import SignalAdapter

        adapter = SignalAdapter()
        assert adapter.channel_type == "signal"

    def test_capabilities(self):
        from channels.signal_adapter import SignalAdapter

        caps = SignalAdapter().capabilities
        assert caps.threads is False
        assert caps.reactions is True
        assert caps.media is True
        assert caps.typing_indicator is False
        assert caps.edit_message is False
        assert caps.max_message_length == 8000

    def test_parse_allowlist(self):
        from channels.signal_adapter import SignalAdapter

        assert SignalAdapter._parse_allowlist("*") is None
        assert SignalAdapter._parse_allowlist(["+1234567890"]) == {"+1234567890"}

    def test_not_connected_by_default(self):
        from channels.signal_adapter import SignalAdapter

        assert SignalAdapter().is_connected is False

    def test_token_resolution_phone(self, monkeypatch):
        from channels.signal_adapter import _resolve_token

        monkeypatch.setenv("SIGNAL_PHONE_NUMBER", "+15551234567")
        result = _resolve_token({})
        assert result == "+15551234567"

    def test_token_direct_phone(self):
        from channels.signal_adapter import _resolve_token

        result = _resolve_token({"phone_number": "+15551234567"})
        assert result == "+15551234567"

    async def test_send_uses_reliability_helper(self, monkeypatch):
        from channels.signal_adapter import SignalAdapter

        calls: list[dict[str, Any]] = []

        async def fake_request_json(provider, method, url, **kwargs):
            calls.append(
                {"provider": provider, "method": method, "url": url, "kwargs": kwargs}
            )
            return {"timestamp": "signal-ts-1"}

        monkeypatch.setattr("channels.signal_adapter.request_json", fake_request_json)

        adapter = SignalAdapter({"api_url": "http://signal.test"})
        adapter._connected = True
        adapter._phone_number = "+15551234567"

        result = await adapter.send("+15557654321", "hello")

        assert result == "signal-ts-1"
        assert calls[0]["provider"] == "signal"
        assert calls[0]["method"] == "POST"
        assert calls[0]["kwargs"]["retry_unsafe_methods"] is False


# ============================================================================
# WhatsApp Adapter
# ============================================================================


class TestWhatsAppAdapter:
    def test_channel_type(self):
        from channels.whatsapp_adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter()
        assert adapter.channel_type == "whatsapp"

    def test_capabilities(self):
        from channels.whatsapp_adapter import WhatsAppAdapter

        caps = WhatsAppAdapter().capabilities
        assert caps.threads is False
        assert caps.reactions is True
        assert caps.media is True
        assert caps.typing_indicator is True
        assert caps.edit_message is False
        assert caps.max_message_length == 4096

    def test_parse_allowlist(self):
        from channels.whatsapp_adapter import WhatsAppAdapter

        assert WhatsAppAdapter._parse_allowlist("*") is None
        assert WhatsAppAdapter._parse_allowlist(["+1234567890"]) == {"+1234567890"}

    def test_not_connected_by_default(self):
        from channels.whatsapp_adapter import WhatsAppAdapter

        assert WhatsAppAdapter().is_connected is False

    def test_token_resolution_env(self, monkeypatch):
        from channels.whatsapp_adapter import _resolve_token

        monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "EAAtest123")
        result = _resolve_token({}, "access_token", "WHATSAPP_ACCESS_TOKEN")
        assert result == "EAAtest123"

    async def test_send_uses_reliability_helper(self, monkeypatch):
        from channels.whatsapp_adapter import WhatsAppAdapter

        calls: list[dict[str, Any]] = []

        async def fake_request_json(provider, method, url, **kwargs):
            calls.append(
                {"provider": provider, "method": method, "url": url, "kwargs": kwargs}
            )
            return {"messages": [{"id": "wamid.1"}]}

        monkeypatch.setattr("channels.whatsapp_adapter.request_json", fake_request_json)

        adapter = WhatsAppAdapter()
        adapter._connected = True
        adapter._access_token = "token"
        adapter._phone_number_id = "phone-id"

        result = await adapter.send("+15557654321", "hello")

        assert result == "wamid.1"
        assert calls[0]["provider"] == "whatsapp"
        assert calls[0]["method"] == "POST"
        assert calls[0]["kwargs"]["retry_unsafe_methods"] is False


# ============================================================================
# iMessage Adapter
# ============================================================================


class TestIMessageAdapter:
    def test_channel_type(self):
        from channels.imessage_adapter import IMessageAdapter

        adapter = IMessageAdapter()
        assert adapter.channel_type == "imessage"

    def test_capabilities(self):
        from channels.imessage_adapter import IMessageAdapter

        caps = IMessageAdapter().capabilities
        assert caps.threads is False
        assert caps.reactions is True
        assert caps.media is True
        assert caps.typing_indicator is True
        assert caps.edit_message is False
        assert caps.max_message_length == 20000

    def test_parse_allowlist(self):
        from channels.imessage_adapter import IMessageAdapter

        assert IMessageAdapter._parse_allowlist("*") is None
        assert IMessageAdapter._parse_allowlist(["user@icloud.com"]) == {
            "user@icloud.com"
        }

    def test_not_connected_by_default(self):
        from channels.imessage_adapter import IMessageAdapter

        assert IMessageAdapter().is_connected is False

    async def test_operator_reply_bypasses_conversation_allowlist(self):
        from channels.imessage_adapter import IMessageAdapter

        adapter = IMessageAdapter(
            {
                "allowed_handles": ["friend@example.com"],
                "operator_recipient": "+15551234567",
            }
        )
        adapter._on_message = AsyncMock()

        await adapter._handle_message(
            {
                "guid": "message-1",
                "isFromMe": False,
                "text": "deny 12345678",
                "handle": {"address": "+15551234567"},
                "chats": [{"guid": "chat-1", "isGroup": False}],
                "attachments": [],
            }
        )

        adapter._on_message.assert_awaited_once()

    def test_config_resolution(self, monkeypatch):
        from channels.imessage_adapter import _resolve_config

        monkeypatch.setenv("IMESSAGE_PASSWORD", "secret123")
        result = _resolve_config({}, "password", "IMESSAGE_PASSWORD")
        assert result == "secret123"

    async def test_send_uses_reliability_helper(self, monkeypatch):
        from channels.imessage_adapter import IMessageAdapter

        calls: list[dict[str, Any]] = []

        async def fake_request_json(provider, method, url, **kwargs):
            calls.append(
                {"provider": provider, "method": method, "url": url, "kwargs": kwargs}
            )
            return {"data": {"guid": "imessage-guid-1"}}

        monkeypatch.setattr("channels.imessage_adapter.request_json", fake_request_json)

        adapter = IMessageAdapter({"api_url": "http://bluebubbles.test"})
        adapter._connected = True
        adapter._password = "secret"

        result = await adapter.send("chat-guid", "hello")

        assert result == "imessage-guid-1"
        assert calls[0]["provider"] == "bluebubbles"
        assert calls[0]["method"] == "POST"
        assert calls[0]["kwargs"]["retry_unsafe_methods"] is False


# ============================================================================
# Matrix Adapter
# ============================================================================


class TestMatrixAdapter:
    def test_channel_type(self):
        from channels.matrix_adapter import MatrixAdapter

        adapter = MatrixAdapter()
        assert adapter.channel_type == "matrix"

    def test_capabilities(self):
        from channels.matrix_adapter import MatrixAdapter

        caps = MatrixAdapter().capabilities
        assert caps.threads is True
        assert caps.reactions is True
        assert caps.media is True
        assert caps.typing_indicator is True
        assert caps.edit_message is True
        assert caps.max_message_length == 65536

    def test_parse_allowlist(self):
        from channels.matrix_adapter import MatrixAdapter

        assert MatrixAdapter._parse_allowlist("*") is None
        assert MatrixAdapter._parse_allowlist(["!abc:matrix.org"]) == {
            "!abc:matrix.org"
        }

    def test_not_connected_by_default(self):
        from channels.matrix_adapter import MatrixAdapter

        assert MatrixAdapter().is_connected is False

    def test_token_resolution_env(self, monkeypatch):
        from channels.matrix_adapter import _resolve_token

        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_value_12345")
        result = _resolve_token({}, "access_token", "MATRIX_ACCESS_TOKEN")
        assert result == "syt_test_token_value_12345"


# ============================================================================
# Streaming Coalescer
# ============================================================================


class _EditableAdapter:
    """Mock adapter that supports edit_message."""

    def __init__(self):
        self.sent: list[dict] = []
        self.edits: list[dict] = []

    @property
    def capabilities(self):
        from channels.base import ChannelCapabilities

        return ChannelCapabilities(edit_message=True, max_message_length=4000)

    async def send(self, channel_id, text, *, reply_to=None, thread_id=None):
        msg_id = f"msg-{len(self.sent)}"
        self.sent.append({"channel_id": channel_id, "text": text, "reply_to": reply_to})
        return msg_id

    async def edit_message(self, channel_id, message_id, text):
        self.edits.append(
            {"channel_id": channel_id, "message_id": message_id, "text": text}
        )
        return True


class TestStreamCoalescer:
    async def test_short_message_no_edit(self):
        from channels.streaming import StreamCoalescer, StreamConfig

        adapter = _EditableAdapter()
        config = StreamConfig(min_chars=10)
        coalescer = StreamCoalescer(adapter, "ch1", config=config)

        await coalescer.push("Hello")
        msg_id = await coalescer.flush()

        assert msg_id is not None
        assert len(adapter.sent) == 1
        assert adapter.sent[0]["text"] == "Hello"
        assert len(adapter.edits) == 0

    async def test_triggers_initial_send_at_threshold(self):
        from channels.streaming import StreamCoalescer, StreamConfig

        adapter = _EditableAdapter()
        config = StreamConfig(min_chars=5)
        coalescer = StreamCoalescer(adapter, "ch1", config=config)

        for char in "Hello World":
            await coalescer.push(char)

        # Should have sent initial message after 5 chars
        assert len(adapter.sent) == 1
        msg_id = await coalescer.flush()
        assert msg_id is not None

    async def test_empty_stream_returns_none(self):
        from channels.streaming import StreamCoalescer

        adapter = _EditableAdapter()
        coalescer = StreamCoalescer(adapter, "ch1")

        msg_id = await coalescer.flush()
        assert msg_id is None
        assert len(adapter.sent) == 0

    async def test_message_id_tracked(self):
        from channels.streaming import StreamCoalescer, StreamConfig

        adapter = _EditableAdapter()
        config = StreamConfig(min_chars=3)
        coalescer = StreamCoalescer(adapter, "ch1", config=config)

        await coalescer.push("Hello!")
        assert coalescer.message_id is not None


# ============================================================================
# Slash Commands
# ============================================================================


class TestSlashCommands:
    def test_parse_command(self):
        from channels.commands import parse_command

        assert parse_command("/status") == ("status", "")
        assert parse_command("/recall search query") == ("recall", "search query")
        assert parse_command("Hello") is None
        assert parse_command("") is None
        assert parse_command("/") is None

    def test_parse_command_case_insensitive(self):
        from channels.commands import parse_command

        result = parse_command("/STATUS")
        assert result == ("status", "")

    def test_registry_has_builtins(self):
        from channels.commands import CommandRegistry

        registry = CommandRegistry()
        assert registry.has("status")
        assert registry.has("recall")
        assert registry.has("goals")
        assert registry.has("energy")
        assert registry.has("help")

    def test_registry_has_unknown(self):
        from channels.commands import CommandRegistry

        registry = CommandRegistry()
        assert registry.has("nonexistent") is False

    def test_registry_list_commands(self):
        from channels.commands import CommandRegistry

        registry = CommandRegistry()
        cmds = registry.list_commands()
        names = {c.name for c in cmds}
        assert {"status", "recall", "goals", "energy", "help"} <= names

    def test_register_custom_command(self):
        from channels.commands import CommandRegistry, ChannelCommand

        registry = CommandRegistry()

        async def handler(args, pool):
            return f"Custom: {args}"

        registry.register(
            ChannelCommand(name="test", description="Test", handler=handler)
        )
        assert registry.has("test")

    async def test_execute_help(self):
        from channels.commands import CommandRegistry

        registry = CommandRegistry()
        result = await registry.execute("help", "", MagicMock())
        assert result is not None
        assert "Available Commands" in result

    async def test_execute_unknown_returns_none(self):
        from channels.commands import CommandRegistry

        registry = CommandRegistry()
        result = await registry.execute("nonexistent", "", MagicMock())
        assert result is None


# ============================================================================
# Manager Command Routing
# ============================================================================


class _MockAdapterWithEdit:
    """Mock adapter with edit support for testing manager routing."""

    def __init__(self, ctype="mock"):
        self._ctype = ctype
        self._connected = False
        self._sent: list[dict] = []

    @property
    def channel_type(self):
        return self._ctype

    @property
    def capabilities(self):
        from channels.base import ChannelCapabilities

        return ChannelCapabilities(
            typing_indicator=True,
            edit_message=True,
            max_message_length=2000,
        )

    @property
    def is_connected(self):
        return self._connected

    async def start(self, on_message):
        self._connected = True

    async def stop(self):
        self._connected = False

    async def send(self, channel_id, text, *, reply_to=None, thread_id=None):
        self._sent.append({"channel_id": channel_id, "text": text})
        return "sent-1"

    async def send_typing(self, channel_id):
        pass

    async def edit_message(self, channel_id, message_id, text):
        return True


class TestManagerCommandRouting:
    @pytest.fixture(autouse=True)
    def _allow_governed_replies(self, monkeypatch):
        """Keep manager routing tests focused on routing through the governor.

        The database-backed outbound tests prove the authorization and ledger
        behavior.  These tests deliberately use a MagicMock pool, so provide
        an allowed reply decision while retaining the real wrapper.
        """
        from services.outbound_safety import OutboundPreparation

        async def allow(*args, body: str, **kwargs):
            return OutboundPreparation(allowed=True, arguments={"message": body})

        monkeypatch.setattr("services.outbound_safety.prepare_outbox_outbound", allow)

    def test_typing_failure_enters_cooldown_until_success(self):
        from channels.manager import ChannelManager

        manager = ChannelManager(pool=MagicMock())
        assert manager._typing_cooldown_active("test", "ch1") is False

        delay = manager._record_typing_failure("test", "ch1")

        assert delay > 0
        assert manager._typing_cooldown_active("test", "ch1") is True

        manager._record_typing_success("test", "ch1")

        assert manager._typing_cooldown_active("test", "ch1") is False

    async def test_command_intercepted(self):
        """Test that /help is handled by command registry, not conversation."""
        from channels.manager import ChannelManager
        from channels.base import ChannelMessage

        pool = MagicMock()
        manager = ChannelManager(pool=pool)
        adapter = _MockAdapterWithEdit("test")
        manager.register(adapter)

        msg = ChannelMessage(
            channel_type="test",
            channel_id="ch1",
            sender_id="user1",
            sender_name="User",
            content="/help",
            message_id="m1",
        )

        await manager._handle_message(msg)

        # Should have sent a response with command list
        assert len(adapter._sent) == 1
        assert "Available Commands" in adapter._sent[0]["text"]

    async def test_private_numbered_automation_reply_is_not_sent_to_conversation(self):
        from channels.base import ChannelMessage
        from channels.manager import ChannelManager

        pool = MagicMock()
        manager = ChannelManager(pool=pool)
        manager._check_user_allowed = AsyncMock(return_value=True)
        adapter = _MockAdapterWithEdit("telegram")
        manager.register(adapter)
        msg = ChannelMessage(
            channel_type="telegram",
            channel_id="private-1",
            sender_id="user1",
            sender_name="User",
            content="1 A1B2C3D4",
            message_id="m1",
            metadata={"is_group": False},
        )

        with (
            patch(
                "services.automation_suggestions.resolve_automation_suggestion_from_inbound",
                new=AsyncMock(
                    return_value={
                        "recognized": True,
                        "matched": True,
                        "message": "Accepted Morning briefing. The scheduled task is active.",
                    }
                ),
            ) as resolve,
            patch("channels.manager.stream_channel_message") as conversation,
        ):
            await manager._handle_message(msg)

        resolve.assert_awaited_once_with(
            pool,
            channel="telegram",
            actor="user1",
            text="1 A1B2C3D4",
        )
        conversation.assert_not_called()
        assert adapter._sent[0]["text"].startswith("Accepted Morning briefing")

    async def test_verified_operator_contradiction_reply_is_not_conversation(self):
        from channels.base import ChannelMessage
        from channels.manager import ChannelManager

        pool = MagicMock()
        manager = ChannelManager(pool=pool)
        manager._check_user_allowed = AsyncMock(return_value=True)
        adapter = _MockAdapterWithEdit("signal")
        manager.register(adapter)
        msg = ChannelMessage(
            channel_type="signal",
            channel_id="private-1",
            sender_id="owner-1",
            sender_name="Owner",
            content="3 ABC12345",
            message_id="m1",
            metadata={"is_group": False},
        )

        with (
            patch(
                "services.operator_policy_corrections.channel_sender_is_operator",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "services.contradictions.resolve_contradiction_from_inbound",
                new=AsyncMock(
                    return_value={
                        "recognized": True,
                        "matched": True,
                        "message": "Recorded: both memories remain valid as context-dependent.",
                    }
                ),
            ) as resolve,
            patch("channels.manager.stream_channel_message") as conversation,
        ):
            await manager._handle_message(msg)

        resolve.assert_awaited_once_with(
            pool,
            channel="signal",
            actor="owner-1",
            text="3 ABC12345",
        )
        conversation.assert_not_called()
        assert adapter._sent[0]["text"].startswith("Recorded: both memories")

    async def test_verified_operator_memory_fade_reply_is_not_conversation(self):
        from channels.base import ChannelMessage
        from channels.manager import ChannelManager

        pool = MagicMock()
        manager = ChannelManager(pool=pool)
        manager._check_user_allowed = AsyncMock(return_value=True)
        adapter = _MockAdapterWithEdit("signal")
        manager.register(adapter)
        msg = ChannelMessage(
            channel_type="signal",
            channel_id="private-1",
            sender_id="owner-1",
            sender_name="Owner",
            content="journal ABC12345: Keep the lesson.",
            message_id="m1",
            metadata={"is_group": False},
        )

        with (
            patch(
                "services.operator_policy_corrections.channel_sender_is_operator",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "services.retention_surface.resolve_memory_fade_review_from_inbound",
                new=AsyncMock(
                    return_value={
                        "recognized": True,
                        "matched": True,
                        "message": "Journal recorded. The sources remain recoverable.",
                    }
                ),
            ) as resolve,
            patch(
                "services.skill_improvement.resolve_learning_review_from_inbound"
            ) as learning,
            patch("channels.manager.stream_channel_message") as conversation,
        ):
            await manager._handle_message(msg)

        resolve.assert_awaited_once_with(
            pool,
            channel="signal",
            actor="owner-1",
            text="journal ABC12345: Keep the lesson.",
        )
        learning.assert_not_called()
        conversation.assert_not_called()
        assert adapter._sent[0]["text"].startswith("Journal recorded")

    async def test_allowlisted_nonoperator_cannot_decide_a_contradiction(self):
        from channels.base import ChannelMessage
        from channels.manager import ChannelManager

        pool = MagicMock()
        manager = ChannelManager(pool=pool)
        manager._check_user_allowed = AsyncMock(return_value=True)
        adapter = _MockAdapterWithEdit("signal")
        manager.register(adapter)
        msg = ChannelMessage(
            channel_type="signal",
            channel_id="private-1",
            sender_id="contact-1",
            sender_name="Contact",
            content="1 ABC12345",
            message_id="m1",
            metadata={"is_group": False},
        )

        with (
            patch(
                "services.operator_policy_corrections.channel_sender_is_operator",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "services.contradictions.resolve_contradiction_from_inbound"
            ) as resolve,
            patch(
                "services.automation_suggestions.resolve_automation_suggestion_from_inbound",
                new=AsyncMock(return_value={"recognized": False}),
            ),
            patch(
                "channels.manager.stream_channel_message", new=AsyncMock()
            ) as conversation,
        ):
            await manager._handle_message(msg)

        resolve.assert_not_called()
        conversation.assert_awaited_once()

    async def test_question_reply_resumes_paused_turn_before_conversation(self):
        from channels.base import ChannelMessage
        from channels.manager import ChannelManager

        pool = MagicMock()
        manager = ChannelManager(pool=pool)
        manager._check_user_allowed = AsyncMock(return_value=True)
        adapter = _MockAdapterWithEdit("telegram")
        manager.register(adapter)
        msg = ChannelMessage(
            channel_type="telegram",
            channel_id="private-1",
            sender_id="user1",
            sender_name="User",
            content="2",
            message_id="m1",
        )

        with (
            patch(
                "services.agent_questions.resolve_agent_question_from_inbound",
                new=AsyncMock(
                    return_value={
                        "recognized": True,
                        "ok": True,
                        "answer": "Hartford",
                        "message": "Thanks — continuing with your answer.",
                    }
                ),
            ) as resolve,
            patch(
                "channels.manager.stream_channel_message", new=AsyncMock()
            ) as conversation,
        ):
            await manager._handle_message(msg)

        resolve.assert_awaited_once_with(
            pool,
            channel="telegram",
            channel_id="private-1",
            actor="user1",
            text="2",
        )
        conversation.assert_not_called()
        assert adapter._sent[0]["text"] == "Thanks — continuing with your answer."

    async def test_group_number_is_left_as_conversation_text(self):
        from channels.base import ChannelMessage
        from channels.manager import ChannelManager

        pool = MagicMock()
        manager = ChannelManager(pool=pool)
        manager._check_user_allowed = AsyncMock(return_value=True)
        adapter = _MockAdapterWithEdit("slack")
        manager.register(adapter)
        msg = ChannelMessage(
            channel_type="slack",
            channel_id="group-1",
            sender_id="user1",
            sender_name="User",
            content="1 A1B2C3D4",
            message_id="m1",
            metadata={"is_group": True},
        )

        with (
            patch(
                "services.operator_approval.resolve_operator_approval_from_inbound",
                new=AsyncMock(return_value={"recognized": False}),
            ),
            patch(
                "services.automation_suggestions.resolve_automation_suggestion_from_inbound"
            ) as resolve,
            patch(
                "channels.manager.stream_channel_message", new=AsyncMock()
            ) as conversation,
        ):
            await manager._handle_message(msg)

        resolve.assert_not_called()
        conversation.assert_awaited_once()
        routed_msg, routed_pool, routed_adapter = conversation.await_args.args
        assert routed_msg is msg
        assert routed_pool is pool
        from services.outbound_safety import GovernedReplyAdapter

        assert isinstance(routed_adapter, GovernedReplyAdapter)
        assert routed_adapter._adapter is adapter


# ============================================================================
# Outbox Consumer (unit tests)
# ============================================================================


class TestOutboxConsumer:
    @pytest.fixture(autouse=True)
    def _allow_outbound_policy(self, monkeypatch):
        """Keep transport-unit tests focused on delivery mechanics.

        The real DB policy and fail-closed outbox integration have dedicated
        tests.  A MagicMock pool cannot produce an authorize_outbound result.
        """
        from services.outbound_safety import OutboundPreparation

        async def allow(*args, body: str, **kwargs):
            return OutboundPreparation(allowed=True, arguments={"message": body})

        monkeypatch.setattr("services.outbound_safety.prepare_outbox_outbound", allow)

    async def test_deliver_direct(self):
        from channels.outbox import ChannelOutboxConsumer

        pool = MagicMock()
        pool.acquire = MagicMock()

        # Mock the context manager
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        manager = MagicMock()
        manager.send = AsyncMock(return_value="sent-1")

        consumer = ChannelOutboxConsumer(manager, pool)

        body = {
            "kind": "channel_message",
            "payload": {
                "content": "Hello from heartbeat",
                "delivery_mode": "direct",
                "target_channel": "discord",
                "target_id": "ch123",
            },
        }
        await consumer._process_message(body)

        manager.send.assert_called_once_with(
            "discord", "ch123", "Hello from heartbeat", thread_id=None
        )

    async def test_empty_content_skipped(self):
        from channels.outbox import ChannelOutboxConsumer

        manager = MagicMock()
        manager.send = AsyncMock()
        pool = MagicMock()

        consumer = ChannelOutboxConsumer(manager, pool)
        await consumer._process_message({"kind": "test", "payload": {}})

        manager.send.assert_not_called()

    async def test_delivery_channel_overrides_direct_target(self):
        from channels.outbox import ChannelOutboxConsumer

        pool = MagicMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        manager = MagicMock()
        manager.send = AsyncMock(return_value="sent-2")

        consumer = ChannelOutboxConsumer(manager, pool)
        body = {
            "kind": "channel_message",
            "payload": {
                "content": "Scheduled update",
                "delivery": {
                    "mode": "channel",
                    "channel": "telegram",
                    "target_id": "chat-42",
                    "topic": "cron-topic",
                },
            },
        }

        await consumer._process_message(body)
        manager.send.assert_called_once_with(
            "telegram", "chat-42", "Scheduled update", thread_id="cron-topic"
        )

    async def test_webhook_delivery_branch(self):
        from channels.outbox import ChannelOutboxConsumer

        manager = MagicMock()
        manager.send = AsyncMock()
        pool = MagicMock()
        consumer = ChannelOutboxConsumer(manager, pool)
        consumer._deliver_webhook = AsyncMock()

        body = {
            "kind": "channel_message",
            "id": "outbox-1",
            "payload": {
                "content": "Webhook message",
                "delivery": {"mode": "webhook", "url": "https://example.com/hook"},
            },
        }

        await consumer._process_message(body)
        consumer._deliver_webhook.assert_awaited_once()
        manager.send.assert_not_called()

    async def test_deliver_webhook_logs_success(self):
        from channels.outbox import ChannelOutboxConsumer

        manager = MagicMock()
        pool = MagicMock()
        consumer = ChannelOutboxConsumer(manager, pool)
        consumer._log_delivery = AsyncMock()

        with patch(
            "channels.outbox.request_text_response", new=AsyncMock()
        ) as mock_post:
            await consumer._deliver_webhook(
                "Hello webhook",
                {"content": "Hello webhook"},
                {"mode": "webhook", "url": "https://example.com/hook"},
                "outbox-2",
            )

        mock_post.assert_awaited_once()
        consumer._log_delivery.assert_awaited_once()
        args = consumer._log_delivery.await_args.args
        assert args[1] == "webhook"
        assert args[2] == "https://example.com/hook"
        assert args[6] is True

    async def test_adapter_none_id_logs_delivery_failure(self):
        from channels.outbox import ChannelOutboxConsumer

        pool = MagicMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        manager = MagicMock()
        manager.send = AsyncMock(return_value=None)

        consumer = ChannelOutboxConsumer(manager, pool)
        consumer._log_delivery = AsyncMock()
        body = {
            "kind": "channel_message",
            "payload": {
                "content": "This should fail",
                "delivery_mode": "direct",
                "target_channel": "discord",
                "target_id": "ch123",
            },
        }

        await consumer._process_message(body)

        args = consumer._log_delivery.await_args.args
        assert args[1] == "discord"
        assert args[6] is False
        assert "did not return a platform message id" in args[7]

    async def test_unreachable_target_skips_direct_delivery(self):
        from channels.outbox import ChannelOutboxConsumer

        pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(
            return_value={
                "skip": True,
                "reason": "chat not found",
                "suppress_until": "2026-07-24T00:00:00Z",
            }
        )
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        manager = MagicMock()
        manager.send = AsyncMock(return_value="sent-1")

        consumer = ChannelOutboxConsumer(manager, pool)
        consumer._log_delivery = AsyncMock()

        await consumer._deliver_direct(
            "hello",
            "hello",
            {"target_channel": "telegram", "target_id": "chat-404"},
            "outbox-skip",
        )

        manager.send.assert_not_called()
        args = consumer._log_delivery.await_args.args
        assert args[1] == "telegram"
        assert args[2] == "chat-404"
        assert args[6] is False
        assert "target marked unreachable" in args[7]

    async def test_unreachable_failure_marks_target(self):
        from channels.outbox import ChannelOutboxConsumer

        pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(
            side_effect=[
                {"skip": False},
                {"success": True, "failure_count": 1},
            ]
        )
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        manager = MagicMock()
        manager.send = AsyncMock(side_effect=RuntimeError("Telegram chat not found"))

        consumer = ChannelOutboxConsumer(manager, pool)
        consumer._log_delivery = AsyncMock()

        await consumer._deliver_direct(
            "hello",
            "hello",
            {"target_channel": "telegram", "target_id": "chat-404"},
            "outbox-mark",
        )

        sql_calls = [call.args[0] for call in mock_conn.fetchval.await_args_list]
        assert any("mark_channel_target_unreachable" in sql for sql in sql_calls)
        args = consumer._log_delivery.await_args.args
        assert args[6] is False
        assert "chat not found" in args[7].lower()

    async def test_transient_failure_does_not_mark_target(self):
        from channels.outbox import ChannelOutboxConsumer

        pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value={"skip": False})
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        manager = MagicMock()
        manager.send = AsyncMock(side_effect=RuntimeError("temporary provider timeout"))

        consumer = ChannelOutboxConsumer(manager, pool)
        consumer._log_delivery = AsyncMock()

        await consumer._deliver_direct(
            "hello",
            "hello",
            {"target_channel": "telegram", "target_id": "chat-1"},
            "outbox-transient",
        )

        sql_calls = [call.args[0] for call in mock_conn.fetchval.await_args_list]
        assert not any("mark_channel_target_unreachable" in sql for sql in sql_calls)
        args = consumer._log_delivery.await_args.args
        assert args[6] is False

    async def test_successful_delivery_clears_unreachable_target(self):
        from channels.outbox import ChannelOutboxConsumer

        pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(
            side_effect=[
                '{"skip": false}',
                True,
            ]
        )
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        manager = MagicMock()
        manager.send = AsyncMock(return_value="sent-1")

        consumer = ChannelOutboxConsumer(manager, pool)
        consumer._log_delivery = AsyncMock()

        await consumer._deliver_direct(
            "hello",
            "hello",
            {"target_channel": "telegram", "target_id": "chat-1"},
            "outbox-clear",
        )

        sql_calls = [call.args[0] for call in mock_conn.fetchval.await_args_list]
        assert any("clear_channel_target_unreachable" in sql for sql in sql_calls)
        args = consumer._log_delivery.await_args.args
        assert args[6] is True


# ============================================================================
# Energy Check (DB integration)
# ============================================================================


class TestChannelEnergy:
    _TEST_PREFIX = "test_energy_"

    async def _cleanup(self, conn, sender_id: str):
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

    async def test_default_free_energy(self, db_pool):
        """Default energy cost is 0 (free), so messages should always be allowed.

        The DB owns the energy/rate-limit policy (prepare_channel_turn, db/34).
        """
        sender_id = f"{self._TEST_PREFIX}free_{id(self)}"

        async with db_pool.acquire() as conn:
            try:
                raw = await conn.fetchval(
                    "SELECT prepare_channel_turn($1::jsonb)",
                    json.dumps(
                        {
                            "channel_type": "test_free",
                            "channel_id": "ch1",
                            "sender_id": sender_id,
                            "sender_name": "User",
                            "content": "Hello",
                            "message_id": "m1",
                        }
                    ),
                )
                turn = json.loads(raw) if isinstance(raw, str) else raw
                assert turn["allowed"] is True
                assert turn["cost"] == 0.0
                assert "rejection" not in turn
            finally:
                await self._cleanup(conn, sender_id)


# ============================================================================
# Base ABC Extensions
# ============================================================================


class TestBaseABCExtensions:
    async def test_default_edit_message(self):
        """Default edit_message returns False."""
        from channels.base import ChannelAdapter

        # Create a minimal concrete adapter
        class MinimalAdapter(ChannelAdapter):
            @property
            def channel_type(self):
                return "minimal"

            @property
            def capabilities(self):
                from channels.base import ChannelCapabilities

                return ChannelCapabilities()

            async def start(self, on_message):
                pass

            async def stop(self):
                pass

            async def send(self, channel_id, text, *, reply_to=None, thread_id=None):
                return None

            async def send_typing(self, channel_id):
                pass

        adapter = MinimalAdapter()
        result = await adapter.edit_message("ch1", "msg1", "new text")
        assert result is False

    async def test_default_send_media(self):
        """Default send_media returns None."""
        from channels.base import ChannelAdapter
        from channels.media import Attachment

        class MinimalAdapter(ChannelAdapter):
            @property
            def channel_type(self):
                return "minimal"

            @property
            def capabilities(self):
                from channels.base import ChannelCapabilities

                return ChannelCapabilities()

            async def start(self, on_message):
                pass

            async def stop(self):
                pass

            async def send(self, channel_id, text, *, reply_to=None, thread_id=None):
                return None

            async def send_typing(self, channel_id):
                pass

        adapter = MinimalAdapter()
        att = Attachment(url="https://example.com/file.png")
        result = await adapter.send_media("ch1", att, "caption")
        assert result is None


# ============================================================================
# Attachment in __init__ exports
# ============================================================================


class TestChannelExports:
    def test_attachment_exported(self):
        from channels import Attachment

        assert Attachment is not None

    def test_all_exports(self):
        import channels

        assert "Attachment" in channels.__all__
        assert "ChannelAdapter" in channels.__all__
        assert "ChannelMessage" in channels.__all__
        assert "ChannelManager" in channels.__all__
