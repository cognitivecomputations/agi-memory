from __future__ import annotations

import base64
import hashlib
import uuid

import pytest

from core.agent_loop import AgentEvent, AgentEventData
from services import chat, node_gateway, speech, voice_notes


class FakeConnection:
    async def fetchrow(self, _query, *_args):
        return {
            "enabled": True,
            "max_audio_bytes": 4_194_304,
            "max_response_audio_bytes": 8_388_608,
            "already_processed": False,
        }


class FakePool:
    class Acquire:
        async def __aenter__(self):
            return FakeConnection()

        async def __aexit__(self, *_args):
            return False

    def acquire(self):
        return self.Acquire()


def _message(audio: bytes) -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "mime_type": "audio/wav",
        "audio_bytes": len(audio),
        "audio_sha256": hashlib.sha256(audio).hexdigest(),
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "detector_model": "custom-wake",
        "detector_label": "wake",
        "detector_score": 0.82,
    }


@pytest.mark.asyncio
async def test_verified_wake_turn_reuses_stt_chat_and_tts(monkeypatch):
    pool = FakePool()
    audits: list[dict] = []

    async def transcribe(received_pool, raw, **kwargs):
        assert received_pool is pool
        assert raw == b"RIFFutterance"
        assert kwargs["channel_type"] == "node_wake"
        return voice_notes.TranscriptionResult(
            ok=True, transcript="what time is it", outcome="transcribed"
        )

    async def events(**kwargs):
        assert kwargs["surface"] == "node_wake"
        assert kwargs["trusted_operator"] is False
        assert kwargs["user_message"] == "what time is it"
        yield AgentEventData(
            event=AgentEvent.TEXT_DELTA,
            data={"text": "It is noon."},
        )

    async def load_config(_conn):
        return speech.TtsConfig(enabled=True, max_chars=4000)

    async def synthesize(received_pool, text, **kwargs):
        assert received_pool is pool
        assert text == "It is noon."
        assert kwargs["source"].startswith("node_wake:")
        return speech.SynthesisResult(
            ok=True,
            audio=b"RIFFreply",
            mime_type="audio/wav",
            outcome="synthesized",
            provider="local_piper",
        )

    async def record(_pool, **kwargs):
        audits.append(kwargs)

    monkeypatch.setattr(voice_notes, "transcribe_uploaded_voice", transcribe)
    monkeypatch.setattr(chat, "stream_chat_events", events)
    monkeypatch.setattr(speech, "load_tts_config", load_config)
    monkeypatch.setattr(speech, "synthesize_text", synthesize)
    monkeypatch.setattr(node_gateway, "_record_wake_event", record)

    result = await node_gateway.process_wake_utterance(
        pool,
        node_id="a" * 64,
        node_name="Kitchen",
        message=_message(b"RIFFutterance"),
    )

    assert result["status"] == "succeeded"
    assert result["transcript"] == "what time is it"
    assert result["assistant"] == "It is noon."
    assert base64.b64decode(result["audio_base64"]) == b"RIFFreply"
    assert audits[0]["outcome"] == "completed"
    assert "what time is it" not in repr(audits[0])
    assert "It is noon" not in repr(audits[0])


@pytest.mark.asyncio
async def test_wake_audio_integrity_failure_stops_before_transcription(monkeypatch):
    pool = FakePool()
    contacted: list[bool] = []
    audits: list[dict] = []
    message = _message(b"RIFFutterance")
    message["audio_sha256"] = "0" * 64

    async def transcribe(*_args, **_kwargs):
        contacted.append(True)

    async def record(_pool, **kwargs):
        audits.append(kwargs)

    monkeypatch.setattr(voice_notes, "transcribe_uploaded_voice", transcribe)
    monkeypatch.setattr(node_gateway, "_record_wake_event", record)

    result = await node_gateway.process_wake_utterance(
        pool,
        node_id="a" * 64,
        node_name="Kitchen",
        message=message,
    )

    assert result["status"] == "failed"
    assert "integrity" in result["error"]
    assert contacted == []
    assert audits[0]["outcome"] == "failed_invalid_audio"
