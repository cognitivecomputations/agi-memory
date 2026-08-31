"""
Hexis Channel System - Signal Adapter

Connects to Signal via signal-cli-rest-api sidecar.
Inbound: SSE event stream.  Outbound: HTTP REST calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Awaitable

from core.integration_reliability import (
    IntegrationHttpError,
    compute_backoff_seconds,
    format_provider_error,
    request_bytes_response,
    request_json,
)

from .base import (
    ChannelAdapter,
    ChannelCapabilities,
    ChannelMessage,
    parse_allowlist,
    resolve_forward_all,
)
from .media import Attachment

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8080"


def _resolve_token(config: dict[str, Any]) -> str | None:
    """Resolve Signal phone number from config or environment.

    Phone numbers are short strings, so we can't use the generic
    resolve_channel_token (which requires len > 20 for raw values).
    """
    phone_env = (
        config.get("phone_number")
        or config.get("phone_number_env")
        or "SIGNAL_PHONE_NUMBER"
    )
    token = os.getenv(str(phone_env)) if phone_env else None
    if token:
        return token
    raw = config.get("phone_number", "")
    if raw and str(raw).startswith("+"):
        return str(raw)
    return None


class SignalAdapter(ChannelAdapter):
    """
    Signal channel adapter via signal-cli-rest-api sidecar.

    Config keys (from DB config table):
        channel.signal.api_url: REST API base URL (default: http://localhost:8080)
        channel.signal.phone_number: Bot's registered phone number (or env var name)
        channel.signal.allowed_numbers: JSON array of phone numbers, or "*"

    Requires signal-cli-rest-api running as a sidecar service.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        forward_all: bool | None = None,
    ) -> None:
        self._config = config or {}
        self._forward_all = resolve_forward_all(self._config, forward_all)
        self._on_message: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self._connected = False
        self._api_url = str(
            self._config.get("api_url")
            or os.getenv("SIGNAL_API_URL")
            or DEFAULT_API_URL
        ).rstrip("/")
        self._phone_number: str | None = None
        self._allowed_numbers = self._parse_allowlist(
            self._config.get("allowed_numbers")
        )
        self._session = None

    @staticmethod
    def _parse_allowlist(value: Any) -> set[str] | None:
        return parse_allowlist(value)

    @property
    def channel_type(self) -> str:
        return "signal"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            threads=False,
            reactions=True,
            media=True,
            typing_indicator=False,
            edit_message=False,
            max_message_length=8000,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(
        self,
        on_message: Callable[[ChannelMessage], Awaitable[None]],
    ) -> None:
        import aiohttp

        self._phone_number = _resolve_token(self._config)
        if not self._phone_number:
            raise RuntimeError(
                "Signal phone number not found. Set SIGNAL_PHONE_NUMBER env var "
                "or configure channel.signal.phone_number in the database."
            )

        self._on_message = on_message
        self._session = aiohttp.ClientSession()
        self._connected = True
        logger.info(
            "Signal adapter started for %s via %s", self._phone_number, self._api_url
        )

        try:
            await self._listen_sse()
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    async def _listen_sse(self) -> None:
        """Listen to SSE event stream from signal-cli REST API."""
        url = f"{self._api_url}/api/v1/receive/{self._phone_number}"
        consecutive_failures = 0
        while self._connected:
            try:
                async with self._session.get(url, timeout=None) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        if resp.status in (401, 403, 404):
                            raise RuntimeError(
                                f"Signal SSE endpoint unavailable/auth failed: HTTP {resp.status}: {body[:300]}"
                            )
                        consecutive_failures += 1
                        delay_s = compute_backoff_seconds(
                            consecutive_failures,
                            initial_delay=5.0,
                            max_delay=120.0,
                            jitter=0.2,
                        )
                        logger.warning(
                            "Signal SSE connection failed: HTTP %d; reconnecting in %.1fs",
                            resp.status,
                            delay_s,
                        )
                        await asyncio.sleep(delay_s)
                        continue

                    consecutive_failures = 0
                    async for line in resp.content:
                        line_str = line.decode("utf-8", errors="replace").strip()
                        if not line_str or line_str.startswith(":"):
                            continue
                        if line_str.startswith("data:"):
                            data_str = line_str[5:].strip()
                            if data_str:
                                await self._handle_sse_event(data_str)

            except asyncio.CancelledError:
                break
            except RuntimeError:
                raise
            except Exception:
                consecutive_failures += 1
                delay_s = compute_backoff_seconds(
                    consecutive_failures,
                    initial_delay=5.0,
                    max_delay=120.0,
                    jitter=0.2,
                )
                logger.exception(
                    "Signal SSE stream error, reconnecting in %.1fs", delay_s
                )
                await asyncio.sleep(delay_s)

    async def _handle_sse_event(self, data_str: str) -> None:
        """Parse and route an SSE event from signal-cli."""
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return

        envelope = data.get("envelope", {})
        data_message = envelope.get("dataMessage")
        if not data_message:
            return

        source = envelope.get("source") or envelope.get("sourceNumber")
        if not source:
            return

        gate_hint: str | None = None
        if self._allowed_numbers is not None:
            if source not in self._allowed_numbers:
                if not self._forward_all:
                    return
                gate_hint = "not_allowlisted"

        text = data_message.get("message", "")
        group_info = data_message.get("groupInfo")
        channel_id = group_info.get("groupId") if group_info else source
        timestamp = str(data_message.get("timestamp", ""))

        # Convert attachments
        attachments: list[Attachment] = []
        for att in data_message.get("attachments", []):
            attachments.append(
                Attachment(
                    url=att.get("url", ""),
                    filename=att.get("filename"),
                    mime_type=att.get("contentType"),
                    size=att.get("size"),
                    platform_id=att.get("id"),
                )
            )

        sender_name = envelope.get("sourceName") or source

        channel_msg = ChannelMessage(
            channel_type="signal",
            channel_id=str(channel_id),
            sender_id=source,
            sender_name=sender_name,
            content=text or "",
            message_id=timestamp,
            attachments=attachments,
            metadata={
                "is_group": group_info is not None,
                "is_mention": False,
                **({"gate_hint": gate_hint} if gate_hint else {}),
            },
        )

        if self._on_message:
            await self._on_message(channel_msg)

    async def stop(self) -> None:
        self._connected = False
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def download_attachment(
        self, attachment: Attachment, *, max_size: int
    ) -> Attachment:
        """Fetch sidecar-hosted media while keeping the configured host explicit."""

        if attachment.local_path or not attachment.url:
            return attachment
        if attachment.size is not None and attachment.size > max_size:
            return attachment
        from urllib.parse import urlparse

        if urlparse(attachment.url).netloc != urlparse(self._api_url).netloc:
            return await super().download_attachment(attachment, max_size=max_size)
        response = await request_bytes_response(
            "signal-media",
            "GET",
            attachment.url,
            timeout=30.0,
            attempts=3,
            max_delay=5.0,
            follow_redirects=False,
            max_bytes=max_size,
        )
        from .media import materialize_attachment_bytes

        return materialize_attachment_bytes(
            attachment,
            response.content,
            content_type=response.headers.get("content-type"),
            prefix="hexis-signal-",
        )

    async def send(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        if not self._connected:
            logger.error("Signal session not connected")
            return None

        try:
            payload: dict[str, Any] = {
                "message": text,
                "number": self._phone_number,
                "recipients": [channel_id],
            }

            result = await request_json(
                "signal",
                "POST",
                f"{self._api_url}/api/v2/send",
                json_body=payload,
                timeout=30.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )
            if not isinstance(result, dict):
                logger.error("Signal send failed: invalid response payload")
                return None
            return str(result.get("timestamp") or result.get("id") or "")
        except IntegrationHttpError as exc:
            logger.error("%s", format_provider_error("Signal", exc))
            return None
        except Exception:
            logger.exception("Failed to send Signal message to %s", channel_id)
            return None

    async def send_typing(self, channel_id: str) -> None:
        # Signal doesn't support bot typing indicators via REST API
        pass
