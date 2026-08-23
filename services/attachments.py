"""Reading an attached file in the turn it is attached.

Attaching a document to a chat message should behave the way a person
expects: the agent can read it right away and answer the question that came
with it. Durable ingestion (memories, chunks, the filing cabinet) still runs
in the background exactly as before — this module is the fast path that pulls
the text out at attach time so the conversation never waits on a queue.

The composer calls this once per file, while the user is still typing; by the
time the message is sent the text is already in hand and rides the turn's
prompt addenda.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fallbacks only — the live values come from config (ingest.attachment_*).
DEFAULT_TEXT_CHARS = 60_000
DEFAULT_READ_TIMEOUT_SECONDS = 25.0
DEFAULT_READ_MAX_BYTES = 25 * 1024 * 1024

# Kinds whose text costs far more than a composer can wait for: images ride
# the turn as visual attachments (the model sees them directly), and
# audio/video need transcription, which is a background job's work.
_DEFERRED_KINDS = {"image", "audio", "video"}


@dataclass
class AttachmentText:
    """What the fast read produced, and — when it produced nothing — why.

    `reason` is never a failure to be swallowed: the caller turns it into
    something the agent is told plainly, so a file it cannot see yet never
    gets discussed as if it could.
    """

    text: str = ""
    chars: int = 0
    truncated: bool = False
    extractor: str = ""
    warnings: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    error: str | None = None

    @property
    def readable(self) -> bool:
        return bool(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_chars": self.chars,
            "truncated": self.truncated,
            "readable": self.readable,
            "extractor": self.extractor,
            "warnings": self.warnings,
            "reason": self.reason,
            "error": self.error,
        }


def attachment_kind(filename: str | None, mime_type: str | None = None) -> str:
    """Classify an upload as image / audio / video / document.

    Extension first (the readers' own tables are the source of truth for what
    each format is), MIME type as the fallback for extensionless uploads.
    """
    from services.ingest.readers import AudioReader, ImageReader, VideoReader

    suffix = Path(filename or "").suffix.lower()
    if suffix:
        if suffix in ImageReader.IMAGE_EXTENSIONS:
            return "image"
        if suffix in AudioReader.AUDIO_EXTENSIONS:
            return "audio"
        if suffix in VideoReader.VIDEO_EXTENSIONS:
            return "video"
    mime = (mime_type or "").strip().lower()
    for prefix in ("image", "audio", "video"):
        if mime.startswith(f"{prefix}/"):
            return prefix
    return "document"


def _read_sync(data: bytes, filename: str, max_chars: int) -> AttachmentText:
    from services.ingest.readers import get_reader

    safe_name = Path(filename or "attachment.txt").name or "attachment.txt"
    with tempfile.TemporaryDirectory(prefix="hexis_attachment_") as tmpdir:
        path = Path(tmpdir) / safe_name
        path.write_bytes(data)
        result = get_reader(path).read_result(path)

    text = (result.text or "").strip()
    truncated = len(text) > max_chars
    return AttachmentText(
        text=text[:max_chars],
        chars=len(text),
        truncated=truncated,
        extractor=result.extractor_name,
        warnings=result.warnings_payload(),
        reason=None if text else "empty",
    )


async def read_attachment_text(
    data: bytes,
    filename: str | None,
    *,
    mime_type: str | None = None,
    max_chars: int = DEFAULT_TEXT_CHARS,
    timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_READ_MAX_BYTES,
) -> AttachmentText:
    """Extract an attachment's text now, within a time and size budget.

    Never raises: extraction that is too slow, too big, or unsupported comes
    back as a reason the caller can state plainly. The background ingestion
    job still reads the same bytes with no budget at all, so nothing is lost
    — only deferred.
    """
    kind = attachment_kind(filename, mime_type)
    if kind in _DEFERRED_KINDS:
        return AttachmentText(reason=kind)
    if max_bytes and len(data) > max_bytes:
        return AttachmentText(reason="too_large")

    name = filename or "attachment.txt"
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_read_sync, data, name, max_chars),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        # The worker thread finishes on its own; we just stop waiting on it.
        logger.info("Attachment read exceeded %.0fs: %s", timeout_seconds, name)
        return AttachmentText(
            reason="timeout",
            error=f"Reading {name} took longer than {timeout_seconds:.0f}s.",
        )
    except Exception as exc:
        logger.warning("Attachment read failed for %s", name, exc_info=True)
        return AttachmentText(reason="error", error=str(exc))
