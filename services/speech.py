"""Opt-in local speech synthesis shared by tools, API, and talk mode.

The provider contract is intentionally small and compatible with Piper's
official HTTP server: ``GET /info`` and ``POST /synthesize`` returning WAV.
Text is sent only to an explicitly local endpoint and never copied into the
metadata audit or ephemeral output table.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from core.integration_reliability import bounded_text
from services.voice_notes import _cfg_bool, _cfg_int, _cfg_text

_LOCAL_TTS_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "host.docker.internal"})


class SpeechError(RuntimeError):
    """A speech failure with an in-place recovery step."""


@dataclass(frozen=True)
class TtsConfig:
    enabled: bool = False
    provider: str = "local_piper"
    model: str = "en_US-lessac-medium"
    voice: str = ""
    max_chars: int = 4_000
    max_audio_bytes: int = 16 * 1024 * 1024
    timeout_seconds: int = 60
    output_ttl_minutes: int = 60


@dataclass
class SynthesisResult:
    ok: bool
    audio: bytes = b""
    mime_type: str = "audio/wav"
    outcome: str = "failed_unknown"
    provider: str = ""
    model: str = ""
    voice: str = ""
    error_detail: str | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def default_tts_url() -> str:
    """Derive the host sidecar address from the running medium."""

    host = "host.docker.internal" if Path("/.dockerenv").exists() else "127.0.0.1"
    return f"http://{host}:42667"


def tts_url() -> str:
    return str(os.getenv("HEXIS_TTS_URL") or default_tts_url()).strip().rstrip("/")


def _validated_local_endpoint(raw: str) -> str:
    try:
        parsed = urllib.parse.urlparse(raw)
        port = parsed.port
    except ValueError as exc:
        raise SpeechError(
            "HEXIS_TTS_URL is invalid. Set it to the local voice sidecar, for "
            "example http://127.0.0.1:42667."
        ) from exc
    if (
        parsed.scheme.lower() != "http"
        or (parsed.hostname or "").lower() not in _LOCAL_TTS_HOSTS
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port is None
    ):
        raise SpeechError(
            "The local speech provider refuses non-local or credential-bearing "
            "HEXIS_TTS_URL values. Use http://127.0.0.1:42667 on the host or "
            "http://host.docker.internal:42667 from Docker."
        )
    return raw.rstrip("/")


async def load_tts_config(conn: Any) -> TtsConfig:
    """Resolve synthesis policy through DB defaults plus operator overrides."""

    return TtsConfig(
        enabled=await _cfg_bool(conn, "voice.tts.enabled", False),
        provider=(await _cfg_text(conn, "voice.tts.provider", "local_piper"))
        .strip()
        .lower(),
        model=(await _cfg_text(conn, "voice.tts.model", "en_US-lessac-medium")).strip(),
        voice=(await _cfg_text(conn, "voice.tts.voice", "")).strip(),
        max_chars=max(1, await _cfg_int(conn, "voice.tts.max_chars", 4_000)),
        max_audio_bytes=max(
            1,
            await _cfg_int(conn, "voice.tts.max_audio_bytes", 16 * 1024 * 1024),
        ),
        timeout_seconds=max(5, await _cfg_int(conn, "voice.tts.timeout_seconds", 60)),
        output_ttl_minutes=max(
            1, await _cfg_int(conn, "voice.tts.output_ttl_minutes", 60)
        ),
    )


async def probe_tts_provider(
    *,
    cfg: TtsConfig,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Read sidecar truth without synthesizing or changing state."""

    if cfg.provider != "local_piper":
        return {
            "ready": False,
            "detail": f"unsupported speech provider {cfg.provider!r}",
            "provider": cfg.provider,
            "model": cfg.model,
        }
    try:
        base = _validated_local_endpoint(endpoint or tts_url())
        async with httpx.AsyncClient(timeout=min(cfg.timeout_seconds, 5)) as client:
            response = await client.get(f"{base}/info")
        if response.status_code >= 400:
            return {
                "ready": False,
                "detail": f"local voice sidecar returned HTTP {response.status_code}",
                "provider": cfg.provider,
                "model": cfg.model,
            }
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        provider_voice = None
        if isinstance(payload, dict):
            voice_info = payload.get("voice")
            if isinstance(voice_info, dict):
                provider_voice = str(voice_info.get("name") or "").strip() or None
        return {
            "ready": True,
            "detail": "local Piper-compatible sidecar is reachable",
            "provider": cfg.provider,
            "model": provider_voice or cfg.model,
            "endpoint": base,
        }
    except (SpeechError, httpx.HTTPError, OSError) as exc:
        return {
            "ready": False,
            "detail": (
                f"local voice sidecar is unavailable ({bounded_text(exc, limit=180)}). "
                "Run `hexis voice setup`, then retry in place."
            ),
            "provider": cfg.provider,
            "model": cfg.model,
        }


async def synthesize_text(
    pool: Any,
    text: str,
    *,
    source: str,
    cfg: TtsConfig | None = None,
    endpoint: str | None = None,
) -> SynthesisResult:
    """Synthesize one bounded utterance and append a metadata-only audit row."""

    utterance = str(text or "").strip()
    if cfg is None:
        async with pool.acquire() as conn:
            cfg = await load_tts_config(conn)
    if not cfg.enabled:
        result = _failure(
            cfg,
            "skipped_disabled",
            "speech output is off; enable it in Settings → Voice, then retry",
        )
        await _record_event(pool, source, len(utterance), result)
        return result
    if not utterance:
        result = _failure(cfg, "failed_empty_text", "text to speak is required")
        await _record_event(pool, source, 0, result)
        return result
    if len(utterance) > cfg.max_chars:
        result = _failure(
            cfg,
            "skipped_too_long",
            f"speech text exceeds the configured {cfg.max_chars}-character limit",
        )
        await _record_event(pool, source, len(utterance), result)
        return result
    if cfg.provider != "local_piper":
        result = _failure(
            cfg,
            "failed_unsupported_provider",
            f"speech provider {cfg.provider!r} is unsupported; choose the local provider in Settings → Voice",
        )
        await _record_event(pool, source, len(utterance), result)
        return result

    started = time.monotonic()
    try:
        base = _validated_local_endpoint(endpoint or tts_url())
    except SpeechError as exc:
        result = _failure(cfg, "failed_endpoint", str(exc), started=started)
        await _record_event(pool, source, len(utterance), result)
        return result

    body: dict[str, Any] = {"text": utterance}
    if cfg.model:
        body["voice"] = cfg.model
    if cfg.voice:
        body["speaker"] = cfg.voice
    try:
        chunks: list[bytes] = []
        received = 0
        async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
            async with client.stream(
                "POST", f"{base}/synthesize", json=body
            ) as response:
                if response.status_code >= 400:
                    result = _failure(
                        cfg,
                        "failed_provider",
                        f"local voice sidecar returned HTTP {response.status_code}",
                        started=started,
                    )
                    await _record_event(pool, source, len(utterance), result)
                    return result
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    if received > cfg.max_audio_bytes:
                        result = _failure(
                            cfg,
                            "failed_too_large",
                            f"local voice sidecar exceeded the configured {cfg.max_audio_bytes}-byte audio limit",
                            started=started,
                        )
                        await _record_event(pool, source, len(utterance), result)
                        return result
                    chunks.append(chunk)
                content_type = str(response.headers.get("content-type") or "audio/wav")
    except (httpx.HTTPError, OSError) as exc:
        result = _failure(
            cfg,
            "failed_provider",
            (
                f"local voice sidecar is unavailable ({bounded_text(exc, limit=180)}). "
                "Run `hexis voice setup`, then retry in place."
            ),
            started=started,
        )
        await _record_event(pool, source, len(utterance), result)
        return result

    audio = b"".join(chunks)
    if not audio:
        result = _failure(
            cfg,
            "failed_empty_audio",
            "local voice sidecar returned no audio",
            started=started,
        )
    else:
        result = SynthesisResult(
            ok=True,
            audio=audio,
            mime_type=content_type.split(";", 1)[0].strip() or "audio/wav",
            outcome="synthesized",
            provider=cfg.provider,
            model=cfg.model,
            voice=cfg.voice,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    await _record_event(pool, source, len(utterance), result)
    return result


async def store_synthesis_output(
    pool: Any,
    result: SynthesisResult,
    *,
    ttl_minutes: int,
    metadata: dict[str, Any] | None = None,
) -> str:
    if not result.ok or not result.audio:
        raise SpeechError(result.error_detail or "speech synthesis produced no audio")
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT purge_expired_voice_tts_outputs()")
        output_id = await conn.fetchval(
            """
            INSERT INTO voice_tts_outputs (
                expires_at, audio, mime_type, provider, model, voice, metadata
            ) VALUES (
                CURRENT_TIMESTAMP + make_interval(mins => $1), $2::bytea, $3, $4,
                NULLIF($5, ''), NULLIF($6, ''), $7::jsonb
            ) RETURNING id::text
            """,
            max(1, int(ttl_minutes)),
            result.audio,
            result.mime_type,
            result.provider,
            result.model,
            result.voice,
            json.dumps(metadata or {}),
        )
    return str(output_id)


async def voice_status(pool: Any) -> dict[str, Any]:
    async with pool.acquire() as conn:
        cfg = await load_tts_config(conn)
        stt_enabled = await _cfg_bool(conn, "voice_notes.stt.enabled", False)
        talk_enabled = await _cfg_bool(conn, "voice.talk.enabled", False)
        wake_enabled = await _cfg_bool(conn, "voice.wake.enabled", False)
        max_utterance_seconds = max(
            5, await _cfg_int(conn, "voice.talk.max_utterance_seconds", 60)
        )
    provider = (
        await probe_tts_provider(cfg=cfg)
        if cfg.enabled
        else {
            "ready": False,
            "detail": "speech output is off",
            "provider": cfg.provider,
            "model": cfg.model,
        }
    )
    return {
        "stt_enabled": stt_enabled,
        "tts_enabled": cfg.enabled,
        "talk_enabled": talk_enabled,
        "wake_enabled": wake_enabled,
        "talk_ready": bool(
            stt_enabled and cfg.enabled and talk_enabled and provider.get("ready")
        ),
        "provider": cfg.provider,
        "model": cfg.model,
        "voice": cfg.voice,
        "provider_ready": bool(provider.get("ready")),
        "detail": str(provider.get("detail") or ""),
        "max_utterance_seconds": max_utterance_seconds,
    }


def _failure(
    cfg: TtsConfig,
    outcome: str,
    detail: str,
    *,
    started: float | None = None,
) -> SynthesisResult:
    return SynthesisResult(
        ok=False,
        outcome=outcome,
        provider=cfg.provider,
        model=cfg.model,
        voice=cfg.voice,
        error_detail=bounded_text(detail, limit=500),
        duration_ms=(
            int((time.monotonic() - started) * 1000) if started is not None else None
        ),
    )


async def _record_event(
    pool: Any,
    source: str,
    input_chars: int,
    result: SynthesisResult,
) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                """
                SELECT record_voice_tts_event(
                    $1, $2, NULLIF($3, ''), NULLIF($4, ''), $5, $6,
                    $7, $8, $9, $10::jsonb
                )
                """,
                str(source or "unknown")[:100],
                result.provider or "unknown",
                result.model,
                result.voice,
                result.outcome,
                max(0, int(input_chars)),
                len(result.audio) if result.audio else None,
                result.duration_ms,
                result.error_detail,
                json.dumps(result.metadata or {}),
            )
    except Exception:
        # Synthesis already succeeded or failed. Audit is advisory and must not
        # make the caller retry a provider effect, but the service logger will
        # retain the exception for diagnosis.
        import logging

        logging.getLogger(__name__).warning(
            "Speech synthesis audit write failed", exc_info=True
        )
