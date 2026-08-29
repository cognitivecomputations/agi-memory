"""Inbound voice-note transcription for configured messaging channels.

Channel adapters own authenticated media retrieval. This service owns the
explicit STT choice, bounded transcription, transcript injection, cleanup,
and metadata-only audit trail. Outbound speech is intentionally separate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from core.integration_reliability import bounded_text

logger = logging.getLogger(__name__)

_AUDIO_MIME_EXACT = frozenset(
    {
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/aac",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/webm",
        "audio/ogg",
        "audio/opus",
        "audio/flac",
        "audio/x-caf",
        "audio/caf",
        "com.apple.coreaudio-format",
        "com.apple.m4a-audio",
        "public.audio",
    }
)
_AUDIO_EXTENSIONS = frozenset(
    {
        ".m4a",
        ".mp3",
        ".mp4",
        ".wav",
        ".webm",
        ".ogg",
        ".oga",
        ".opus",
        ".flac",
        ".aac",
        ".caf",
        ".amr",
    }
)
_LOCAL_MODELS: dict[str, Any] = {}
_AUDIO_SUFFIX_BY_MIME = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
}


@dataclass(frozen=True)
class SttConfig:
    enabled: bool = False
    provider: str = "local_whisper"
    model: str = "base"
    channels: tuple[str, ...] = ()
    max_bytes: int = 25 * 1024 * 1024
    timeout_seconds: int = 60
    language: str = ""
    prepend_marker: bool = True
    cloud_disclosure_accepted: bool = False


@dataclass
class TranscriptionResult:
    ok: bool
    transcript: str = ""
    outcome: str = "failed_unknown"
    provider: str = ""
    model: str = ""
    mime_type: str | None = None
    filename: str | None = None
    attachment_id: str | None = None
    error_detail: str | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


AttachmentDownloader = Callable[..., Awaitable[Any]]


def is_audio_attachment(attachment: Any) -> bool:
    """Return whether a normalized attachment looks like an audio recording."""

    if hasattr(attachment, "mime_type"):
        mime = str(getattr(attachment, "mime_type", None) or "").lower()
        filename = str(getattr(attachment, "filename", None) or "")
    elif isinstance(attachment, dict):
        mime = str(attachment.get("mime_type") or attachment.get("type") or "").lower()
        filename = str(attachment.get("filename") or "")
    else:
        return False
    if mime in _AUDIO_MIME_EXACT or mime.startswith("audio/"):
        return True
    if "audio" in mime and (
        "apple" in mime or "public." in mime or "coreaudio" in mime
    ):
        return True
    return Path(filename).suffix.lower() in _AUDIO_EXTENSIONS


async def load_stt_config(conn: Any) -> SttConfig:
    """Resolve STT settings through database defaults plus user overrides."""

    channels = await _cfg_str_list(conn, "voice_notes.stt.channels", [])
    return SttConfig(
        enabled=await _cfg_bool(conn, "voice_notes.stt.enabled", False),
        provider=(await _cfg_text(conn, "voice_notes.stt.provider", "local_whisper"))
        .strip()
        .lower(),
        model=(await _cfg_text(conn, "voice_notes.stt.model", "base")).strip()
        or "base",
        channels=tuple(
            str(channel).strip().lower() for channel in channels if str(channel).strip()
        ),
        max_bytes=max(
            1, await _cfg_int(conn, "voice_notes.stt.max_bytes", 25 * 1024 * 1024)
        ),
        timeout_seconds=max(
            5, await _cfg_int(conn, "voice_notes.stt.timeout_seconds", 60)
        ),
        language=(await _cfg_text(conn, "voice_notes.stt.language", "")).strip(),
        prepend_marker=await _cfg_bool(conn, "voice_notes.stt.prepend_marker", True),
        cloud_disclosure_accepted=await _cfg_bool(
            conn, "voice_notes.stt.cloud_disclosure_accepted", False
        ),
    )


async def enrich_message_with_voice_transcripts(
    pool: Any,
    msg: Any,
    *,
    attachment_downloader: AttachmentDownloader | None = None,
) -> Any:
    """Return a message whose audio is represented as typed transcript text."""

    audio_attachments = [
        attachment
        for attachment in list(getattr(msg, "attachments", None) or [])
        if is_audio_attachment(attachment)
    ]
    if not audio_attachments:
        return msg
    try:
        async with pool.acquire() as conn:
            cfg = await load_stt_config(conn)
    except Exception:
        logger.warning("Voice-note configuration could not be loaded", exc_info=True)
        return _with_audio_fallback_note(
            msg,
            "[Voice note received, but transcription settings could not be loaded. Open Settings → Voice notes, then retry or type the message.]",
        )

    channel_type = str(getattr(msg, "channel_type", "") or "").lower()
    if not cfg.enabled:
        await _record_event(
            pool, msg, audio_attachments[0], _skipped(cfg, "skipped_disabled")
        )
        return _with_audio_fallback_note(
            msg,
            "[Voice note received. Transcription is off. Open Settings → Voice notes to choose local or cloud transcription, or type the message.]",
        )
    if cfg.provider == "openai_whisper" and not cfg.cloud_disclosure_accepted:
        await _record_event(
            pool,
            msg,
            audio_attachments[0],
            _skipped(cfg, "skipped_cloud_disclosure"),
        )
        return _with_audio_fallback_note(
            msg,
            "[Voice note received, but cloud transcription has not been accepted. Open Settings → Voice notes, review the disclosure, and save your choice.]",
        )
    if cfg.channels and channel_type not in cfg.channels:
        await _record_event(
            pool,
            msg,
            audio_attachments[0],
            _skipped(cfg, "skipped_channel", metadata={"channel_type": channel_type}),
        )
        return _with_audio_fallback_note(
            msg,
            f"[Voice note received on {channel_type}, but transcription is not enabled for this channel. Open Settings → Voice notes or type the message.]",
        )

    transcripts: list[str] = []
    provenance: list[dict[str, Any]] = []
    failures: list[str] = []
    for attachment in audio_attachments:
        result = await transcribe_attachment(
            attachment,
            cfg=cfg,
            attachment_downloader=attachment_downloader,
        )
        await _record_event(pool, msg, attachment, result)
        item = {
            "attachment_id": result.attachment_id,
            "mime_type": result.mime_type,
            "filename": result.filename,
            "provider": result.provider,
            "model": result.model,
            "outcome": result.outcome,
            "duration_ms": result.duration_ms,
        }
        provenance.append(
            {key: value for key, value in item.items() if value is not None}
        )
        if result.ok and result.transcript.strip():
            transcripts.append(result.transcript.strip())
        elif result.error_detail:
            failures.append(result.error_detail)

    existing = str(getattr(msg, "content", None) or "").strip()
    parts: list[str] = []
    if transcripts:
        transcript_text = "\n\n".join(transcripts)
        if cfg.prepend_marker:
            transcript_text = f"[Voice note transcript]\n{transcript_text}"
        parts.append(transcript_text)
    elif not existing:
        detail = failures[0] if failures else "transcription was unavailable"
        parts.append(
            f"[Voice note received, but it could not be transcribed: {detail}. Retry, open Settings → Voice notes, or type the message.]"
        )
    if existing:
        parts.append(existing)

    metadata = dict(getattr(msg, "metadata", None) or {})
    metadata["voice_note"] = {
        "stt_provider": cfg.provider,
        "stt_model": cfg.model,
        "transcripts": provenance,
        "transcript_count": len(transcripts),
        "fallback_note": not transcripts and not existing,
    }
    return _replace_message(msg, content="\n\n".join(parts).strip(), metadata=metadata)


async def transcribe_attachment(
    attachment: Any,
    *,
    cfg: SttConfig,
    attachment_downloader: AttachmentDownloader | None = None,
) -> TranscriptionResult:
    """Materialize and transcribe one attachment within explicit limits."""

    from channels.media import Attachment, download_attachment

    if isinstance(attachment, dict):
        source = Attachment.from_dict(attachment)
    elif isinstance(attachment, Attachment):
        source = attachment
    else:
        return _failure(cfg, "failed_bad_attachment", "unsupported attachment type")
    common = {
        "mime_type": source.mime_type,
        "filename": source.filename,
        "attachment_id": source.platform_id,
    }
    if source.size is not None and source.size > cfg.max_bytes:
        return _failure(
            cfg,
            "skipped_too_large",
            f"audio exceeds the {cfg.max_bytes}-byte limit",
            **common,
        )

    downloaded_path: str | None = None
    materialized = source
    if not source.local_path:
        try:
            downloader = attachment_downloader or download_attachment
            materialized = await downloader(source, max_size=cfg.max_bytes)
            if materialized.local_path:
                downloaded_path = str(materialized.local_path)
        except Exception as exc:
            logger.warning("Voice-note media download failed (%s)", type(exc).__name__)
            return _failure(
                cfg,
                "failed_download",
                "the channel could not download this audio",
                **common,
            )
    if not materialized.local_path or not os.path.isfile(materialized.local_path):
        return _failure(
            cfg,
            "failed_download",
            "the channel could not download this audio",
            **common,
        )

    try:
        if os.path.getsize(materialized.local_path) > cfg.max_bytes:
            return _failure(
                cfg,
                "skipped_too_large",
                f"audio exceeds the {cfg.max_bytes}-byte limit",
                **common,
            )
        if cfg.provider == "local_whisper":
            return await _transcribe_local_whisper(materialized, cfg)
        if cfg.provider == "openai_whisper":
            return await _transcribe_openai_whisper(materialized, cfg)
        return _failure(
            cfg,
            "failed_unsupported_provider",
            f"provider {cfg.provider!r} is not supported; choose local or cloud transcription in Settings → Voice notes",
            **common,
        )
    finally:
        if downloaded_path:
            try:
                os.unlink(downloaded_path)
            except OSError:
                logger.debug(
                    "Voice-note temporary file was already removed: %s", downloaded_path
                )


async def transcribe_uploaded_voice(
    pool: Any,
    raw: bytes,
    *,
    filename: str | None,
    mime_type: str | None,
    channel_id: str | None = None,
    sender_id: str | None = None,
    channel_type: str = "web",
) -> TranscriptionResult:
    """Transcribe one foreground PWA or signed-node recording under shared policy.

    The recording is held in a unique temporary file only for the provider
    call, then removed. The metadata-only audit uses the same ledger as phone
    channel voice notes; transcript content never enters that audit.
    """

    from channels.media import Attachment, materialize_attachment_bytes

    message_id = str(uuid.uuid4())
    source = Attachment(
        url="",
        filename=Path(filename or "voice-note.webm").name,
        mime_type=(mime_type or "application/octet-stream").split(";", 1)[0],
        size=len(raw),
        platform_id=message_id,
    )
    context = SimpleNamespace(
        channel_type=channel_type,
        channel_id=channel_id,
        sender_id=sender_id,
        message_id=message_id,
    )
    async with pool.acquire() as conn:
        cfg = await load_stt_config(conn)

    if not cfg.enabled:
        result = TranscriptionResult(
            ok=False,
            outcome="skipped_disabled",
            provider=cfg.provider,
            model=cfg.model,
            mime_type=source.mime_type,
            filename=source.filename,
            attachment_id=source.platform_id,
            error_detail=(
                "transcription is off; open Settings → Voice notes and choose "
                "local or cloud transcription"
            ),
        )
        await _record_event(pool, context, source, result)
        return result
    if cfg.provider == "openai_whisper" and not cfg.cloud_disclosure_accepted:
        result = TranscriptionResult(
            ok=False,
            outcome="skipped_cloud_disclosure",
            provider=cfg.provider,
            model=cfg.model,
            mime_type=source.mime_type,
            filename=source.filename,
            attachment_id=source.platform_id,
            error_detail=(
                "cloud transcription has not been accepted; review and save the "
                "disclosure in Settings → Voice notes"
            ),
        )
        await _record_event(pool, context, source, result)
        return result
    if cfg.channels and channel_type not in cfg.channels:
        result = TranscriptionResult(
            ok=False,
            outcome="skipped_channel",
            provider=cfg.provider,
            model=cfg.model,
            mime_type=source.mime_type,
            filename=source.filename,
            attachment_id=source.platform_id,
            error_detail=(
                f"{channel_type} transcription is not in the configured voice-note channel allowlist"
            ),
            metadata={"channel_type": channel_type},
        )
        await _record_event(pool, context, source, result)
        return result
    if len(raw) > cfg.max_bytes:
        result = _failure(
            cfg,
            "skipped_too_large",
            f"audio exceeds the {cfg.max_bytes}-byte limit",
            mime_type=source.mime_type,
            filename=source.filename,
            attachment_id=source.platform_id,
        )
        await _record_event(pool, context, source, result)
        return result
    if not raw or not is_audio_attachment(source):
        result = _failure(
            cfg,
            "failed_bad_attachment",
            "the upload is not a supported audio recording",
            mime_type=source.mime_type,
            filename=source.filename,
            attachment_id=source.platform_id,
        )
        await _record_event(pool, context, source, result)
        return result

    materialized = materialize_attachment_bytes(
        source,
        raw,
        content_type=source.mime_type,
        filename=source.filename,
        prefix="hexis-pwa-voice-",
    )
    try:
        result = await transcribe_attachment(materialized, cfg=cfg)
        await _record_event(pool, context, source, result)
        return result
    finally:
        if materialized.local_path:
            try:
                os.unlink(materialized.local_path)
            except OSError:
                logger.debug(
                    "PWA voice-note temporary file was already removed: %s",
                    materialized.local_path,
                )


async def _transcribe_local_whisper(
    attachment: Any, cfg: SttConfig
) -> TranscriptionResult:
    started = time.monotonic()

    def transcribe() -> str:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "install local transcription with: pip install 'hexis[media]'"
            ) from exc
        model = _LOCAL_MODELS.get(cfg.model)
        if model is None:
            model = whisper.load_model(cfg.model)
            _LOCAL_MODELS[cfg.model] = model
        kwargs: dict[str, Any] = {}
        if cfg.language:
            kwargs["language"] = cfg.language
        result = model.transcribe(str(attachment.local_path), **kwargs)
        return str(result.get("text") or "").strip()

    try:
        transcript = await asyncio.to_thread(transcribe)
    except Exception as exc:
        return _provider_failure(attachment, cfg, bounded_text(exc, limit=240), started)
    return _success_or_empty(attachment, cfg, transcript, started)


async def _transcribe_openai_whisper(
    attachment: Any, cfg: SttConfig
) -> TranscriptionResult:
    api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return _failure(
            cfg,
            "failed_missing_credentials",
            "OPENAI_API_KEY is not set for the channel worker",
            mime_type=attachment.mime_type,
            filename=attachment.filename,
            attachment_id=attachment.platform_id,
        )
    started = time.monotonic()
    try:
        import httpx

        base_url = str(
            os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        data = {"model": cfg.model}
        if cfg.language:
            data["language"] = cfg.language
        filename = _upload_filename(attachment)
        with open(attachment.local_path, "rb") as audio_file:
            files = {
                "file": (
                    filename,
                    audio_file,
                    attachment.mime_type or "application/octet-stream",
                )
            }
            async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
                response = await client.post(
                    f"{base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                    files=files,
                )
        if response.status_code >= 400:
            return _provider_failure(
                attachment,
                cfg,
                f"cloud transcription failed with HTTP {response.status_code}",
                started,
            )
        payload = response.json()
        transcript = (
            str(payload.get("text") or "").strip() if isinstance(payload, dict) else ""
        )
    except Exception as exc:
        return _provider_failure(attachment, cfg, bounded_text(exc, limit=240), started)
    return _success_or_empty(attachment, cfg, transcript, started)


def _upload_filename(attachment: Any) -> str:
    filename = Path(
        attachment.filename or Path(attachment.local_path).name or "voice-note"
    ).name
    if Path(filename).suffix:
        return filename
    mime = str(attachment.mime_type or "").split(";", 1)[0].strip().lower()
    suffix = (
        _AUDIO_SUFFIX_BY_MIME.get(mime) or mimetypes.guess_extension(mime) or ".audio"
    )
    return f"{filename}{suffix}"


def _success_or_empty(
    attachment: Any, cfg: SttConfig, transcript: str, started: float
) -> TranscriptionResult:
    common = {
        "provider": cfg.provider,
        "model": cfg.model,
        "mime_type": attachment.mime_type,
        "filename": attachment.filename,
        "attachment_id": attachment.platform_id,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
    if not transcript:
        return TranscriptionResult(
            ok=False,
            outcome="failed_empty_transcript",
            error_detail="no speech was detected",
            **common,
        )
    return TranscriptionResult(
        ok=True, transcript=transcript, outcome="transcribed", **common
    )


def _provider_failure(
    attachment: Any, cfg: SttConfig, detail: str, started: float
) -> TranscriptionResult:
    return _failure(
        cfg,
        "failed_provider",
        detail,
        mime_type=attachment.mime_type,
        filename=attachment.filename,
        attachment_id=attachment.platform_id,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _failure(
    cfg: SttConfig, outcome: str, detail: str, **kwargs: Any
) -> TranscriptionResult:
    return TranscriptionResult(
        ok=False,
        outcome=outcome,
        provider=cfg.provider,
        model=cfg.model,
        error_detail=bounded_text(detail, limit=240),
        **kwargs,
    )


def _skipped(
    cfg: SttConfig, outcome: str, *, metadata: dict[str, Any] | None = None
) -> TranscriptionResult:
    return TranscriptionResult(
        ok=False,
        outcome=outcome,
        provider=cfg.provider,
        model=cfg.model,
        metadata=metadata or {},
    )


def _with_audio_fallback_note(msg: Any, note: str) -> Any:
    if str(getattr(msg, "content", None) or "").strip():
        return msg
    metadata = dict(getattr(msg, "metadata", None) or {})
    metadata["voice_note"] = {"fallback_note": True}
    return _replace_message(msg, content=note, metadata=metadata)


def _replace_message(msg: Any, **changes: Any) -> Any:
    try:
        return replace(msg, **changes)
    except TypeError:
        for key, value in changes.items():
            setattr(msg, key, value)
        return msg


async def _record_event(
    pool: Any, msg: Any, attachment: Any, result: TranscriptionResult
) -> None:
    """Write metadata only: transcript content never enters the audit table."""

    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                """
                SELECT record_voice_note_stt_event(
                    $1::text, $2::text, $3::text, $4::text, $5::text,
                    $6::text, $7::text, $8::text, $9::text, $10::text,
                    $11::integer, $12::text, $13::integer, $14::jsonb
                )
                """,
                str(getattr(msg, "channel_type", "") or ""),
                str(getattr(msg, "channel_id", "") or "") or None,
                str(getattr(msg, "sender_id", "") or "") or None,
                str(getattr(msg, "message_id", "") or "") or None,
                str(
                    result.attachment_id or getattr(attachment, "platform_id", "") or ""
                )
                or None,
                str(result.mime_type or getattr(attachment, "mime_type", "") or "")
                or None,
                str(result.filename or getattr(attachment, "filename", "") or "")
                or None,
                result.provider or "unknown",
                result.model or None,
                result.outcome,
                len(result.transcript) if result.transcript else None,
                result.error_detail,
                result.duration_ms,
                json.dumps(result.metadata or {}),
            )
    except Exception:
        logger.debug("Voice-note audit write failed", exc_info=True)


async def _cfg_bool(conn: Any, key: str, default: bool) -> bool:
    value = await _cfg(conn, key, default)
    return (
        value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"true", "1", "yes", "on"}
    )


async def _cfg_int(conn: Any, key: str, default: int) -> int:
    value = await _cfg(conn, key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def _cfg_text(conn: Any, key: str, default: str) -> str:
    value = await _cfg(conn, key, default)
    return str(value) if value is not None else default


async def _cfg_str_list(conn: Any, key: str, default: list[str]) -> list[str]:
    value = await _cfg(conn, key, default)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return list(default)
    return [str(item) for item in value] if isinstance(value, list) else list(default)


async def _cfg(conn: Any, key: str, default: Any) -> Any:
    try:
        value = await conn.fetchval("SELECT get_config($1)", key)
        return default if value is None else value
    except Exception:
        return default
