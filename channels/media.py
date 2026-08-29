"""
Hexis Channel System - Media Handling

Centralized media download, normalization, and SSRF protection for channel
attachments. Each channel adapter converts platform-native attachments to the
Attachment dataclass; the media module handles safe download and caching.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from core.integration_reliability import (
    IntegrationHttpError,
    format_provider_error,
    request_bytes_response,
)

logger = logging.getLogger(__name__)

# Default maximum attachment size (10 MB)
DEFAULT_MAX_SIZE = 10 * 1024 * 1024

# IP ranges that should never be fetched (SSRF protection)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local / cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique-local
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]


@dataclass
class Attachment:
    """
    Normalized attachment from any channel.

    Adapters populate the platform-facing fields (url, filename, mime_type,
    size, platform_id).  download_attachment() populates local_path.
    """

    url: str
    filename: str | None = None
    mime_type: str | None = None
    size: int | None = None
    platform_id: str | None = None
    local_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attachment:
        """Create an Attachment from a raw dict (backward compat)."""
        return cls(
            url=str(data.get("url") or ""),
            filename=data.get("filename"),
            mime_type=data.get("mime_type") or data.get("type"),
            size=data.get("size"),
            platform_id=data.get("platform_id") or data.get("id"),
            local_path=data.get("local_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size": self.size,
            "platform_id": self.platform_id,
            "local_path": self.local_path,
        }

    def describe(self) -> str:
        """Human-readable one-liner for LLM context injection."""
        parts = []
        if self.filename:
            parts.append(self.filename)
        if self.mime_type:
            parts.append(self.mime_type)
        if self.size:
            if self.size >= 1024 * 1024:
                parts.append(f"{self.size / (1024 * 1024):.1f}MB")
            elif self.size >= 1024:
                parts.append(f"{self.size / 1024:.0f}KB")
            else:
                parts.append(f"{self.size}B")
        return ", ".join(parts) if parts else self.url


def is_safe_url(url: str) -> bool:
    """
    SSRF guard: returns False for URLs that resolve to private/internal IPs.

    Checks the hostname against blocked IP ranges.  DNS resolution is NOT
    performed here (that would be async); this catches obvious literal-IP
    cases and known-bad hostnames.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False

        # Block common metadata endpoints
        if hostname in ("metadata.google.internal", "metadata.internal"):
            return False

        # Try to parse as an IP address
        try:
            addr = ipaddress.ip_address(hostname)
            for network in _BLOCKED_NETWORKS:
                if addr in network:
                    return False
        except ValueError:
            # Not a literal IP — allow (DNS resolution happens at fetch time)
            pass

        return True
    except Exception:
        return False


async def download_attachment(
    attachment: Attachment,
    *,
    max_size: int = DEFAULT_MAX_SIZE,
    cache_dir: str | None = None,
    headers: dict[str, str] | None = None,
) -> Attachment:
    """
    Download an attachment to a local file with SSRF protection and size limits.

    Returns a new Attachment with local_path populated (or the original if
    download fails or is skipped).
    """
    if not attachment.url:
        return attachment

    if not is_safe_url(attachment.url):
        logger.warning("Blocked unsafe URL: %s", attachment.url[:100])
        return attachment

    target_dir = cache_dir or tempfile.gettempdir()
    os.makedirs(target_dir, exist_ok=True)

    try:
        request_headers = {"User-Agent": "Hexis/1.0", **(headers or {})}
        response = await request_bytes_response(
            "attachment",
            "GET",
            attachment.url,
            headers=request_headers,
            timeout=30.0,
            attempts=3,
            max_delay=10.0,
            follow_redirects=True,
            max_bytes=max_size,
        )

        raw = response.content
        return materialize_attachment_bytes(
            attachment,
            raw,
            content_type=response.headers.get("content-type"),
            filename=attachment.filename
            or _filename_from_response(response, attachment.url),
            cache_dir=target_dir,
        )

    except IntegrationHttpError as exc:
        logger.warning("%s", format_provider_error("Attachment download", exc))
        return attachment
    except Exception:
        logger.exception("Failed to download attachment: %s", attachment.url[:100])
        return attachment


def materialize_attachment_bytes(
    attachment: Attachment,
    raw: bytes,
    *,
    content_type: str | None = None,
    filename: str | None = None,
    cache_dir: str | None = None,
    prefix: str = "hexis-attachment-",
) -> Attachment:
    """Persist already-authenticated media with a safe, collision-free name."""

    target_dir = cache_dir or tempfile.gettempdir()
    os.makedirs(target_dir, exist_ok=True)
    safe_name = (
        Path(filename or attachment.filename or "attachment").name or "attachment"
    )
    suffix = Path(safe_name).suffix[:16]
    fd, filepath = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=target_dir)
    try:
        with os.fdopen(fd, "wb") as attachment_file:
            attachment_file.write(raw)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(filepath)
        except OSError:
            pass
        raise
    return Attachment(
        url=attachment.url,
        filename=safe_name,
        mime_type=attachment.mime_type or content_type,
        size=len(raw),
        platform_id=attachment.platform_id,
        local_path=filepath,
    )


def _filename_from_response(resp, url: str) -> str:
    """Derive a filename from Content-Disposition header or URL path."""
    # Try Content-Disposition
    cd = resp.headers.get("content-disposition") or resp.headers.get(
        "Content-Disposition", ""
    )
    if "filename=" in cd:
        parts = cd.split("filename=")
        if len(parts) > 1:
            name = parts[1].strip().strip('"').strip("'")
            if name:
                return name

    # Fall back to URL path
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if "/" in path:
        name = path.rsplit("/", 1)[-1]
        if "." in name:
            return name

    return "attachment"
