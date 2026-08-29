from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.tools.base import ToolContext, ToolErrorType, ToolExecutionContext
from core.tools.local_audio_analysis import (
    AnalyzeLocalAudioHandler,
    TranscribeAudioHandler,
)


def _context(tool_context: ToolContext = ToolContext.CHAT) -> ToolExecutionContext:
    return ToolExecutionContext(tool_context=tool_context, call_id="test")


def test_tool_is_optional_and_approval_gated():
    spec = AnalyzeLocalAudioHandler().spec
    assert spec.optional is True
    assert spec.requires_approval is True
    assert spec.execution_timeout_seconds == 60
    transcribe = TranscribeAudioHandler().spec
    assert transcribe.name == "transcribe"
    assert transcribe.optional is True
    assert transcribe.requires_approval is True


@pytest.mark.asyncio(loop_scope="session")
async def test_status_has_recovery_step_without_existing_job(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = await AnalyzeLocalAudioHandler().execute(
        {"action": "status", "audio_path": str(tmp_path / "memo.wav")},
        _context(),
    )
    assert result.success
    assert result.output["status"] == "missing"
    assert "Start analysis" in result.output["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_heartbeat_needs_separate_autonomous_gate(tmp_path):
    audio = tmp_path / "memo.wav"
    audio.write_bytes(b"RIFF")

    async def config(_pool, key, default):
        if key == "audio_analysis.local.allow_autonomous":
            return False
        return default

    with patch(
        "core.tools.local_audio_analysis._config", AsyncMock(side_effect=config)
    ):
        result = await AnalyzeLocalAudioHandler().execute(
            {"action": "start", "audio_path": str(audio)},
            _context(ToolContext.HEARTBEAT),
        )
    assert result.error_type == ToolErrorType.APPROVAL_REQUIRED


@pytest.mark.asyncio(loop_scope="session")
async def test_transcribe_returns_service_transcript(tmp_path):
    from services.voice_notes import TranscriptionResult

    audio = tmp_path / "memo.wav"
    audio.write_bytes(b"RIFF")
    transcription = TranscriptionResult(
        ok=True,
        transcript="hello from the recording",
        outcome="transcribed",
        provider="local_whisper",
        model="base",
    )
    with patch(
        "services.voice_notes.transcribe_attachment",
        AsyncMock(return_value=transcription),
    ):
        result = await TranscribeAudioHandler().execute(
            {"audio_path": str(audio)}, _context()
        )
    assert result.success
    assert result.output["transcript"] == "hello from the recording"
