from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from channels.base import ChannelCapabilities, ChannelMessage
from channels.media import Attachment
from core.integration_reliability import IntegrationHttpResponse
from services.voice_notes import SttConfig, TranscriptionResult
from services.outbound_safety import OutboundPreparation

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


class _Pool:
    class _Acquire:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    def acquire(self):
        return self._Acquire()


def _bytes_response(content: bytes = b"audio") -> IntegrationHttpResponse:
    return IntegrationHttpResponse(
        status_code=200,
        headers={"content-type": "audio/ogg"},
        text="",
        json_data=None,
        correlation_id="test",
        content=content,
    )


async def test_whatsapp_resolves_authenticated_media_id():
    from channels.whatsapp_adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter()
    adapter._access_token = "token"
    attachment = Attachment(
        url="", filename="memo.ogg", mime_type="audio/ogg", platform_id="media-1"
    )
    with (
        patch(
            "channels.whatsapp_adapter.request_json",
            AsyncMock(return_value={"url": "https://lookaside.fbsbx.com/media"}),
        ),
        patch(
            "channels.whatsapp_adapter.request_bytes_response",
            AsyncMock(return_value=_bytes_response()),
        ),
    ):
        downloaded = await adapter.download_attachment(attachment, max_size=100)
    assert downloaded.local_path and os.path.isfile(downloaded.local_path)
    os.unlink(downloaded.local_path)


async def test_imessage_trusts_explicit_bluebubbles_endpoint():
    from channels.imessage_adapter import IMessageAdapter

    adapter = IMessageAdapter({"api_url": "http://localhost:1234"})
    attachment = Attachment(
        url="http://localhost:1234/api/v1/attachment/a/download?password=chosen",
        filename="memo.m4a",
        mime_type="audio/m4a",
    )
    with patch(
        "channels.imessage_adapter.request_bytes_response",
        AsyncMock(return_value=_bytes_response()),
    ):
        downloaded = await adapter.download_attachment(attachment, max_size=100)
    assert downloaded.local_path and os.path.isfile(downloaded.local_path)
    os.unlink(downloaded.local_path)


async def test_signal_only_trusts_configured_sidecar_host():
    from channels.signal_adapter import SignalAdapter

    adapter = SignalAdapter({"api_url": "http://signal:8080"})
    attachment = Attachment(
        url="http://signal:8080/v1/attachments/a",
        filename="memo.ogg",
        mime_type="audio/ogg",
    )
    with patch(
        "channels.signal_adapter.request_bytes_response",
        AsyncMock(return_value=_bytes_response()),
    ):
        downloaded = await adapter.download_attachment(attachment, max_size=100)
    assert downloaded.local_path and os.path.isfile(downloaded.local_path)
    os.unlink(downloaded.local_path)


async def test_telegram_normalizes_voice_attachment():
    from channels.telegram_adapter import TelegramAdapter

    adapter = TelegramAdapter()
    adapter._on_message = AsyncMock()
    message = SimpleNamespace(
        chat=SimpleNamespace(id=123, type="private"),
        from_user=SimpleNamespace(id=7, full_name="Ada", username="ada"),
        text=None,
        caption=None,
        photo=None,
        document=None,
        audio=None,
        voice=SimpleNamespace(
            file_unique_id="unique",
            file_id="platform-file",
            file_size=50,
            mime_type="audio/ogg",
        ),
        message_id=9,
        reply_to_message=None,
        message_thread_id=None,
        is_topic_message=False,
    )
    await adapter._handle_telegram_message(SimpleNamespace(message=message))
    normalized = adapter._on_message.await_args.args[0]
    assert normalized.attachments[0].platform_id == "platform-file"
    assert normalized.attachments[0].mime_type == "audio/ogg"


class VoiceAdapter:
    channel_type = "telegram"
    capabilities = ChannelCapabilities(media=True, typing_indicator=False)

    def __init__(self) -> None:
        self.download_attachment = AsyncMock()
        self.send = AsyncMock(return_value="sent")


async def test_manager_routes_voice_transcript_as_ordinary_text():
    from channels.manager import ChannelManager

    adapter = VoiceAdapter()
    manager = ChannelManager(pool=_Pool())
    manager.register(adapter)  # type: ignore[arg-type]
    manager._check_user_allowed = AsyncMock(return_value=True)
    message = ChannelMessage(
        channel_type="telegram",
        channel_id="chat-1",
        sender_id="user-1",
        sender_name="Ada",
        content="",
        message_id="message-1",
        attachments=[Attachment(url="", filename="memo.ogg", mime_type="audio/ogg")],
        metadata={"is_group": True},
    )
    transcription = TranscriptionResult(
        ok=True,
        transcript="Schedule lunch tomorrow",
        outcome="transcribed",
        provider="local_whisper",
        model="base",
    )
    with (
        patch(
            "services.voice_notes.load_stt_config",
            AsyncMock(return_value=SttConfig(enabled=True)),
        ),
        patch(
            "services.voice_notes.transcribe_attachment",
            AsyncMock(return_value=transcription),
        ),
        patch("services.voice_notes._record_event", AsyncMock()),
        patch(
            "services.agent_questions.resolve_agent_question_from_inbound",
            AsyncMock(return_value={"recognized": False}),
        ),
        patch(
            "channels.manager.process_channel_message",
            AsyncMock(return_value=["Done"]),
        ) as process,
        patch(
            "services.outbound_safety.prepare_outbox_outbound",
            AsyncMock(
                return_value=OutboundPreparation(
                    allowed=True,
                    arguments={"message": "Done"},
                )
            ),
        ),
        patch(
            "services.outbound_safety.finalize_outbox_outbound",
            AsyncMock(),
        ),
    ):
        await manager._handle_message(message)
    routed = process.await_args.args[0]
    assert "Schedule lunch tomorrow" in routed.content
    adapter.send.assert_awaited_once()
