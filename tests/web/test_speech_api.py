from __future__ import annotations

import httpx
import pytest

import apps.hexis_api as hexis_api
from services import speech


class FakeConnection:
    def __init__(self, row=None) -> None:
        self.row = row
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, query, *args):
        self.calls.append((str(query), args))
        return self.row


class FakePool:
    class Acquire:
        def __init__(self, conn) -> None:
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, *_args):
            return False

    def __init__(self, row=None) -> None:
        self.conn = FakeConnection(row)

    def acquire(self):
        return self.Acquire(self.conn)


@pytest.fixture
async def client(monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(hexis_api, "_pool", pool)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=hexis_api.app),
        base_url="http://test",
    ) as value:
        yield value, pool


@pytest.mark.asyncio
async def test_synthesize_returns_uncached_audio(client, monkeypatch):
    http, pool = client

    async def fake_synthesize(received_pool, text, *, source):
        assert received_pool is pool
        assert text == "hello aloud"
        assert source == "pwa"
        return speech.SynthesisResult(
            ok=True,
            audio=b"RIFFwave",
            mime_type="audio/wav",
            outcome="synthesized",
            provider="local_piper",
            model="voice-a",
        )

    monkeypatch.setattr(speech, "synthesize_text", fake_synthesize)

    response = await http.post(
        "/api/voice/synthesize", json={"text": "hello aloud"}
    )

    assert response.status_code == 200
    assert response.content == b"RIFFwave"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-hexis-voice-provider"] == "local_piper"


@pytest.mark.asyncio
async def test_synthesize_surfaces_disabled_configuration(client, monkeypatch):
    http, _pool = client

    async def fake_synthesize(*_args, **_kwargs):
        return speech.SynthesisResult(
            ok=False,
            outcome="skipped_disabled",
            error_detail="speech output is off; enable it in Settings → Voice",
        )

    monkeypatch.setattr(speech, "synthesize_text", fake_synthesize)

    response = await http.post("/api/voice/synthesize", json={"text": "hello"})

    assert response.status_code == 409
    assert "Settings" in response.json()["detail"]


@pytest.mark.asyncio
async def test_voice_status_comes_from_shared_service(client, monkeypatch):
    http, pool = client
    expected = {
        "stt_enabled": True,
        "tts_enabled": True,
        "talk_enabled": True,
        "talk_ready": True,
        "provider_ready": True,
    }

    async def fake_status(received_pool):
        assert received_pool is pool
        return expected

    monkeypatch.setattr(speech, "voice_status", fake_status)

    response = await http.get("/api/voice/status")

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.asyncio
async def test_tool_audio_is_opaque_uncached_and_expires(client):
    http, pool = client
    output_id = "bf6fdb27-4baf-468e-b3d4-35dfb0a5bc41"
    pool.conn.row = {"audio": b"RIFFtool", "mime_type": "audio/wav"}

    response = await http.get(f"/api/voice/audio/{output_id}")

    assert response.status_code == 200
    assert response.content == b"RIFFtool"
    assert response.headers["cache-control"] == "no-store"
    query, args = pool.conn.calls[-1]
    assert "expires_at > CURRENT_TIMESTAMP" in query
    assert args == (output_id,)

    pool.conn.row = None
    expired = await http.get(f"/api/voice/audio/{output_id}")
    assert expired.status_code == 404
    assert "expired" in expired.json()["detail"]


@pytest.mark.asyncio
async def test_tool_audio_rejects_non_opaque_identifier_before_query(client):
    http, pool = client

    response = await http.get("/api/voice/audio/not-a-uuid")

    assert response.status_code == 404
    assert pool.conn.calls == []
