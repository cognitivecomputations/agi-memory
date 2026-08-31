from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.speech import SpeakHandler
from services import speech


class FakePool:
    class Acquire:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    def acquire(self):
        return self.Acquire()


@pytest.mark.asyncio
async def test_speak_emits_retrievable_audio_without_inline_bytes(monkeypatch):
    handler = SpeakHandler()
    events: list[tuple[str, dict]] = []

    async def emit(event, payload):
        events.append((event, payload))

    monkeypatch.setattr(
        speech,
        "load_tts_config",
        AsyncMock(return_value=speech.TtsConfig(enabled=True)),
    )
    monkeypatch.setattr(
        speech,
        "synthesize_text",
        AsyncMock(
            return_value=speech.SynthesisResult(
                ok=True,
                audio=b"RIFFaudio",
                outcome="synthesized",
                provider="local_piper",
                model="model",
            )
        ),
    )
    monkeypatch.setattr(
        speech, "store_synthesis_output", AsyncMock(return_value="audio-id")
    )
    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="call-1",
        session_id="session-1",
        surface="api",
        event_callback=emit,
    )
    context.registry = SimpleNamespace(pool=FakePool())

    result = await handler.execute({"text": "say this"}, context)

    assert result.success is True
    assert result.output["audio_url"] == "/api/voice/audio/audio-id"
    assert "RIFF" not in repr(result.output)
    assert events == [
        (
            "ui",
            {
                "kind": "speech",
                "id": "audio-id",
                "audio_url": "/api/voice/audio/audio-id",
                "mime_type": "audio/wav",
                "provider": "local_piper",
                "model": "model",
            },
        )
    ]
