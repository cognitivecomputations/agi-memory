from __future__ import annotations

import uuid

import pytest

from services import speech


class FakeConnection:
    def __init__(self, config=None) -> None:
        self.config = config or {}
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchval(self, query, *args):
        self.calls.append((str(query), args))
        if "get_config(" in str(query):
            return self.config.get(str(args[0]))
        if "INSERT INTO voice_tts_outputs" in str(query):
            return str(uuid.uuid4())
        return 0


class FakePool:
    class Acquire:
        def __init__(self, conn) -> None:
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, *_args):
            return False

    def __init__(self, config=None) -> None:
        self.conn = FakeConnection(config)

    def acquire(self):
        return self.Acquire(self.conn)


class FakeResponse:
    def __init__(self, *, status=200, chunks=None, payload=None) -> None:
        self.status_code = status
        self.headers = {"content-type": "audio/wav"}
        self._chunks = list(chunks or [])
        self._payload = payload or {"voice": {"name": "provider-voice"}}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    def json(self):
        return self._payload


class FakeClient:
    response = FakeResponse(chunks=[b"RIFF", b"audio"])

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.requests: list[tuple[str, object]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def stream(self, method, url, **kwargs):
        self.requests.append((str(method), (str(url), kwargs)))
        return self.response

    async def get(self, url):
        self.requests.append(("GET", str(url)))
        return self.response


@pytest.mark.asyncio
async def test_disabled_speech_never_contacts_provider(monkeypatch):
    pool = FakePool()
    contacted: list[bool] = []
    monkeypatch.setattr(
        speech.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: contacted.append(True),
    )

    result = await speech.synthesize_text(
        pool,
        "hello",
        source="test",
        cfg=speech.TtsConfig(enabled=False),
    )

    assert result.outcome == "skipped_disabled"
    assert contacted == []


@pytest.mark.asyncio
async def test_remote_provider_url_is_refused_before_text_leaves(monkeypatch):
    pool = FakePool()
    contacted: list[bool] = []
    monkeypatch.setattr(
        speech.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: contacted.append(True),
    )

    result = await speech.synthesize_text(
        pool,
        "private words",
        source="test",
        cfg=speech.TtsConfig(enabled=True),
        endpoint="https://speech.example.test:443",
    )

    assert result.outcome == "failed_endpoint"
    assert "refuses non-local" in str(result.error_detail)
    assert contacted == []


@pytest.mark.asyncio
async def test_local_synthesis_is_bounded_and_audit_excludes_text(monkeypatch):
    pool = FakePool()
    FakeClient.response = FakeResponse(chunks=[b"RIFF", b"wave"])
    monkeypatch.setattr(speech.httpx, "AsyncClient", FakeClient)

    result = await speech.synthesize_text(
        pool,
        "do not copy this sentence",
        source="pwa",
        cfg=speech.TtsConfig(enabled=True, model="live-model"),
        endpoint="http://127.0.0.1:42667",
    )

    assert result.ok is True
    assert result.audio == b"RIFFwave"
    assert result.model == "live-model"
    audit_calls = [
        call for call in pool.conn.calls if "record_voice_tts_event" in call[0]
    ]
    assert len(audit_calls) == 1
    assert "do not copy this sentence" not in repr(audit_calls[0])
    assert 25 in audit_calls[0][1]


@pytest.mark.asyncio
async def test_provider_audio_over_limit_fails_without_returning_partial_bytes(
    monkeypatch,
):
    pool = FakePool()
    FakeClient.response = FakeResponse(chunks=[b"1234", b"5678"])
    monkeypatch.setattr(speech.httpx, "AsyncClient", FakeClient)

    result = await speech.synthesize_text(
        pool,
        "hello",
        source="pwa",
        cfg=speech.TtsConfig(enabled=True, max_audio_bytes=6),
        endpoint="http://127.0.0.1:42667",
    )

    assert result.outcome == "failed_too_large"
    assert result.audio == b""


@pytest.mark.asyncio
async def test_ephemeral_output_stores_audio_without_input_text():
    pool = FakePool()
    result = speech.SynthesisResult(
        ok=True,
        audio=b"RIFFwave",
        outcome="synthesized",
        provider="local_piper",
        model="voice-model",
    )

    output_id = await speech.store_synthesis_output(
        pool,
        result,
        ttl_minutes=30,
        metadata={"surface": "api"},
    )

    assert uuid.UUID(output_id)
    insert = next(
        call for call in pool.conn.calls if "INSERT INTO voice_tts_outputs" in call[0]
    )
    assert b"RIFFwave" in insert[1]
    assert "surface" in repr(insert[1])
