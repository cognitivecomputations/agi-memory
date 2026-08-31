from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from channels.base import ChannelMessage
from channels.media import Attachment
from services.voice_notes import (
    SttConfig,
    TranscriptionResult,
    enrich_message_with_voice_transcripts,
    is_audio_attachment,
    transcribe_attachment,
    transcribe_uploaded_voice,
)


class _Pool:
    class _Acquire:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    def acquire(self):
        return self._Acquire()


def _message(attachment: Attachment, content: str = "") -> ChannelMessage:
    return ChannelMessage(
        channel_type="telegram",
        channel_id="chat-1",
        sender_id="user-1",
        sender_name="Ada",
        content=content,
        message_id="message-1",
        attachments=[attachment],
    )


def test_audio_detection_uses_mime_or_extension():
    assert is_audio_attachment(
        Attachment(url="", filename="memo.bin", mime_type="audio/ogg")
    )
    assert is_audio_attachment(
        Attachment(url="", filename="memo.m4a", mime_type="application/octet-stream")
    )
    assert not is_audio_attachment(
        Attachment(url="", filename="memo.pdf", mime_type="application/pdf")
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_disabled_audio_only_turn_has_exact_setup_step():
    attachment = Attachment(url="", filename="memo.ogg", mime_type="audio/ogg")
    with (
        patch(
            "services.voice_notes.load_stt_config",
            AsyncMock(return_value=SttConfig(enabled=False)),
        ),
        patch("services.voice_notes._record_event", AsyncMock()) as record,
    ):
        result = await enrich_message_with_voice_transcripts(
            _Pool(), _message(attachment)
        )
    assert "Settings → Voice notes" in result.content
    assert result.metadata["voice_note"]["fallback_note"] is True
    record.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_cloud_audio_needs_recorded_disclosure():
    attachment = Attachment(url="", filename="memo.ogg", mime_type="audio/ogg")
    with (
        patch(
            "services.voice_notes.load_stt_config",
            AsyncMock(
                return_value=SttConfig(
                    enabled=True,
                    provider="openai_whisper",
                    model="whisper-1",
                    cloud_disclosure_accepted=False,
                )
            ),
        ),
        patch("services.voice_notes._record_event", AsyncMock()),
    ):
        result = await enrich_message_with_voice_transcripts(
            _Pool(), _message(attachment)
        )
    assert "cloud transcription has not been accepted" in result.content
    assert result.metadata["voice_note"]["fallback_note"] is True


@pytest.mark.asyncio(loop_scope="session")
async def test_transcript_enters_same_path_as_typed_text():
    attachment = Attachment(
        url="", filename="memo.ogg", mime_type="audio/ogg", platform_id="a-1"
    )
    transcription = TranscriptionResult(
        ok=True,
        transcript="Remind me after lunch",
        outcome="transcribed",
        provider="local_whisper",
        model="base",
        filename="memo.ogg",
        mime_type="audio/ogg",
        attachment_id="a-1",
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
    ):
        result = await enrich_message_with_voice_transcripts(
            _Pool(), _message(attachment, "and include Casey")
        )
    assert (
        result.content
        == "[Voice note transcript]\nRemind me after lunch\n\nand include Casey"
    )
    assert result.metadata["voice_note"]["transcript_count"] == 1
    assert "transcript" not in result.metadata["voice_note"]["transcripts"][0]


@pytest.mark.asyncio(loop_scope="session")
async def test_adapter_download_is_cleaned_after_transcription(tmp_path: Path):
    source = Attachment(
        url="", filename="memo.ogg", mime_type="audio/ogg", platform_id="a-1"
    )
    downloaded_path = tmp_path / "downloaded.ogg"
    downloaded_path.write_bytes(b"audio")
    downloaded = Attachment(
        url="",
        filename="memo.ogg",
        mime_type="audio/ogg",
        platform_id="a-1",
        local_path=str(downloaded_path),
    )
    transcribed = TranscriptionResult(
        ok=True,
        transcript="hello",
        outcome="transcribed",
        provider="local_whisper",
        model="base",
    )
    downloader = AsyncMock(return_value=downloaded)
    with patch(
        "services.voice_notes._transcribe_local_whisper",
        AsyncMock(return_value=transcribed),
    ):
        result = await transcribe_attachment(
            source,
            cfg=SttConfig(enabled=True),
            attachment_downloader=downloader,
        )
    assert result.ok
    assert not downloaded_path.exists()
    downloader.assert_awaited_once_with(source, max_size=25 * 1024 * 1024)


@pytest.mark.asyncio(loop_scope="session")
async def test_cloud_provider_requires_worker_key(tmp_path: Path, monkeypatch):
    path = tmp_path / "memo.ogg"
    path.write_bytes(b"audio")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = await transcribe_attachment(
        Attachment(
            url="", filename="memo.ogg", mime_type="audio/ogg", local_path=str(path)
        ),
        cfg=SttConfig(enabled=True, provider="openai_whisper", model="whisper-1"),
    )
    assert result.outcome == "failed_missing_credentials"
    assert "OPENAI_API_KEY" in str(result.error_detail)


@pytest.mark.asyncio(loop_scope="session")
async def test_pwa_recording_uses_shared_policy_and_cleans_temporary_audio():
    seen_path: Path | None = None

    async def transcribe(attachment, *, cfg):
        nonlocal seen_path
        seen_path = Path(attachment.local_path)
        assert seen_path.exists()
        assert cfg.provider == "local_whisper"
        return TranscriptionResult(
            ok=True,
            transcript="Call Morgan after lunch",
            outcome="transcribed",
            provider=cfg.provider,
            model=cfg.model,
        )

    with (
        patch(
            "services.voice_notes.load_stt_config",
            AsyncMock(return_value=SttConfig(enabled=True)),
        ),
        patch("services.voice_notes.transcribe_attachment", side_effect=transcribe),
        patch("services.voice_notes._record_event", AsyncMock()) as record,
    ):
        result = await transcribe_uploaded_voice(
            _Pool(),
            b"audio",
            filename="memo.webm",
            mime_type="audio/webm",
            channel_id="pwa-device",
        )
    assert result.transcript == "Call Morgan after lunch"
    assert seen_path is not None and not seen_path.exists()
    record.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_pwa_recording_respects_disabled_voice_choice():
    with (
        patch(
            "services.voice_notes.load_stt_config",
            AsyncMock(return_value=SttConfig(enabled=False)),
        ),
        patch("services.voice_notes._record_event", AsyncMock()),
    ):
        result = await transcribe_uploaded_voice(
            _Pool(),
            b"audio",
            filename="memo.webm",
            mime_type="audio/webm",
        )
    assert result.outcome == "skipped_disabled"
    assert "Settings → Voice notes" in str(result.error_detail)
