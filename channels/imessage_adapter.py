"""
Hexis Channel System - iMessage Adapter

Connects to iMessage via BlueBubbles server.
Inbound: Polling REST API.  Outbound: HTTP REST calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Awaitable

from core.integration_reliability import (
    IntegrationHttpError,
    format_provider_error,
    request_bytes_response,
    request_json,
    request_text_response,
)

from .base import (
    ChannelAdapter,
    ChannelCapabilities,
    ChannelMessage,
    parse_allowlist,
    resolve_channel_token,
    resolve_forward_all,
)
from .media import Attachment

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:1234"
POLL_INTERVAL = 2  # seconds


def _resolve_config(config: dict[str, Any], key: str, env_fallback: str) -> str | None:
    """Resolve a value from config or environment."""
    return resolve_channel_token(config, key, env_fallback)


class IMessageAdapter(ChannelAdapter):
    """
    iMessage channel adapter via BlueBubbles server.

    Config keys (from DB config table):
        channel.imessage.api_url: BlueBubbles server URL (default: http://localhost:1234)
        channel.imessage.password: BlueBubbles server password
        channel.imessage.operator_recipient: phone/email allowed to decide approvals
        channel.imessage.allowed_handles: JSON array of phone/email handles, or "*"

    Requires a BlueBubbles server running on a Mac with iMessage configured.
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
            or os.getenv("IMESSAGE_API_URL")
            or DEFAULT_API_URL
        ).rstrip("/")
        self._password: str | None = None
        self._allowed_handles = self._parse_allowlist(
            self._config.get("allowed_handles")
        )
        self._operator_recipient = str(
            self._config.get("operator_recipient") or ""
        ).strip()
        self._session = None
        self._last_timestamp: int = 0  # Unix timestamp ms for polling cursor

    @staticmethod
    def _parse_allowlist(value: Any) -> set[str] | None:
        return parse_allowlist(value)

    @property
    def channel_type(self) -> str:
        return "imessage"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            threads=False,
            reactions=True,
            media=True,
            typing_indicator=True,
            edit_message=False,
            max_message_length=20000,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(
        self,
        on_message: Callable[[ChannelMessage], Awaitable[None]],
    ) -> None:
        import aiohttp

        self._password = _resolve_config(self._config, "password", "IMESSAGE_PASSWORD")
        if not self._password:
            raise RuntimeError(
                "BlueBubbles password not found. Set IMESSAGE_PASSWORD env var "
                "or configure channel.imessage.password in the database."
            )

        self._on_message = on_message
        self._session = aiohttp.ClientSession()
        self._last_timestamp = int(time.time() * 1000)  # Start from now
        self._connected = True
        logger.info("iMessage adapter started via BlueBubbles at %s", self._api_url)

        try:
            await self._poll_loop()
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    async def _poll_loop(self) -> None:
        """Poll BlueBubbles for new messages."""
        while self._connected:
            try:
                await self._poll_messages()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("iMessage poll error, retrying in %ds", POLL_INTERVAL)
            await asyncio.sleep(POLL_INTERVAL)

    async def _poll_messages(self) -> None:
        """Fetch new messages since last poll."""
        if not self._connected:
            return

        params = {
            "password": self._password,
            "limit": "50",
            "after": str(self._last_timestamp),
            "sort": "ASC",
            "with": "chats",  # Include chat metadata
        }

        try:
            data = await request_json(
                "bluebubbles",
                "GET",
                f"{self._api_url}/api/v1/message",
                params=params,
                timeout=10.0,
                attempts=3,
                max_delay=5.0,
            )
            if not isinstance(data, dict):
                logger.warning("BlueBubbles poll failed: invalid response payload")
                return

            messages = data.get("data", [])
            if not isinstance(messages, list):
                logger.warning("BlueBubbles poll failed: invalid messages payload")
                return

            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                await self._handle_message(msg)

                # Update cursor
                date_created = msg.get("dateCreated")
                if date_created and date_created > self._last_timestamp:
                    self._last_timestamp = date_created
        except IntegrationHttpError as exc:
            logger.warning("%s", format_provider_error("BlueBubbles poll", exc))
        except Exception:
            logger.exception("Error polling BlueBubbles messages")

    async def _handle_message(self, msg: dict) -> None:
        """Process a single BlueBubbles message."""
        # Skip messages sent by us (isFromMe)
        if msg.get("isFromMe"):
            return

        text = msg.get("text") or ""
        handle = msg.get("handle", {})
        sender_address = (
            handle.get("address", "") if isinstance(handle, dict) else str(handle)
        )

        if not sender_address:
            return

        # In transport-only mode the manager's SQL resolver owns policy. Keep
        # a hint so a runtime kill-switch change can still reproduce the
        # legacy drop without restarting the adapter.
        gate_hint: str | None = None
        if self._allowed_handles is not None:
            operator_reply = (
                bool(self._operator_recipient)
                and sender_address.lower() == self._operator_recipient.lower()
            )
            if sender_address not in self._allowed_handles and not operator_reply:
                if not self._forward_all:
                    return
                gate_hint = "not_allowlisted"

        # Determine chat/channel ID from the first associated chat
        chats = msg.get("chats", [])
        chat_guid = chats[0].get("guid", sender_address) if chats else sender_address

        # Convert attachments
        attachments: list[Attachment] = []
        for att in msg.get("attachments", []):
            mime_type = att.get("mimeType") or att.get("uti")
            attachments.append(
                Attachment(
                    url=(
                        f"{self._api_url}/api/v1/attachment/{att.get('guid', '')}/download?password={self._password}"
                        if att.get("guid")
                        else ""
                    ),
                    filename=att.get("transferName") or att.get("filename"),
                    mime_type=mime_type,
                    size=att.get("totalBytes"),
                    platform_id=att.get("guid"),
                )
            )

        sender_name = (
            handle.get("displayName") or sender_address
            if isinstance(handle, dict)
            else sender_address
        )

        msg_guid = msg.get("guid", str(msg.get("dateCreated", "")))

        channel_msg = ChannelMessage(
            channel_type="imessage",
            channel_id=str(chat_guid),
            sender_id=sender_address,
            sender_name=sender_name,
            content=text,
            message_id=msg_guid,
            attachments=attachments,
            metadata={
                "is_group": bool(chats and chats[0].get("isGroup")),
                "service": msg.get("service", "iMessage"),
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
        """Fetch media from the explicitly configured BlueBubbles server."""

        if attachment.local_path or not attachment.url:
            return attachment
        if attachment.size is not None and attachment.size > max_size:
            return attachment
        from urllib.parse import urlparse

        configured = urlparse(self._api_url)
        requested = urlparse(attachment.url)
        if (requested.scheme, requested.netloc) != (
            configured.scheme,
            configured.netloc,
        ):
            return await super().download_attachment(attachment, max_size=max_size)
        response = await request_bytes_response(
            "bluebubbles-media",
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
            prefix="hexis-imessage-",
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
            logger.error("iMessage session not connected")
            return None

        try:
            payload: dict[str, Any] = {
                "chatGuid": channel_id,
                "message": text,
                "method": "apple-script",
            }

            params = {"password": self._password}

            result = await request_json(
                "bluebubbles",
                "POST",
                f"{self._api_url}/api/v1/message/text",
                json_body=payload,
                params=params,
                timeout=30.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )
            if not isinstance(result, dict):
                logger.error("iMessage send failed: invalid response payload")
                return None
            data = result.get("data", {})
            return data.get("guid") or str(data.get("dateCreated", ""))
        except IntegrationHttpError as exc:
            logger.error("%s", format_provider_error("BlueBubbles", exc))
            return None
        except Exception:
            logger.exception("Failed to send iMessage to %s", channel_id)
            return None

    async def send_typing(self, channel_id: str) -> None:
        """BlueBubbles supports typing indicators via API."""
        if not self._connected:
            return
        try:
            params = {"password": self._password}
            payload = {"chatGuid": channel_id}
            await request_text_response(
                "bluebubbles",
                "POST",
                f"{self._api_url}/api/v1/chat/{channel_id}/typing",
                json_body=payload,
                params=params,
                timeout=5.0,
                attempts=1,
                retry_unsafe_methods=False,
            )
        except IntegrationHttpError:
            logger.debug("BlueBubbles typing indicator failed", exc_info=True)
        except Exception:
            logger.debug("Silent exception in IMessageAdapter", exc_info=True)

    async def send_media(
        self,
        channel_id: str,
        attachment: Attachment,
        caption: str | None = None,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        if not self._session or self._session.closed:
            return None

        file_handle = None
        try:
            import aiohttp

            params = {"password": self._password}

            data = aiohttp.FormData()
            data.add_field("chatGuid", channel_id)
            if caption:
                data.add_field("message", caption)

            if attachment.local_path:
                file_handle = open(attachment.local_path, "rb")
                data.add_field(
                    "attachment",
                    file_handle,
                    filename=attachment.filename or "attachment",
                    content_type=attachment.mime_type or "application/octet-stream",
                )
            elif attachment.url:
                # Download first, then upload
                from .media import download_attachment

                downloaded = await download_attachment(attachment)
                if downloaded.local_path:
                    file_handle = open(downloaded.local_path, "rb")
                    data.add_field(
                        "attachment",
                        file_handle,
                        filename=downloaded.filename or "attachment",
                        content_type=downloaded.mime_type or "application/octet-stream",
                    )
                else:
                    return None
            else:
                return None

            timeout = aiohttp.ClientTimeout(total=60)
            async with self._session.post(
                f"{self._api_url}/api/v1/message/attachment",
                data=data,
                params=params,
                timeout=timeout,
            ) as resp:
                if resp.status in (200, 201):
                    result = await resp.json()
                    msg_data = result.get("data", {})
                    return msg_data.get("guid")
                else:
                    body = await _read_aiohttp_text_limited(resp)
                    logger.error(
                        "iMessage send_media failed: HTTP %d: %s",
                        resp.status,
                        body[:200],
                    )
                    return None
        except Exception:
            logger.exception("Failed to send iMessage media to %s", channel_id)
            return None
        finally:
            if file_handle is not None:
                file_handle.close()


async def _read_aiohttp_text_limited(resp: Any, limit: int = 8_000) -> str:
    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.content.iter_chunked(2048):
        total += len(chunk)
        if total > limit:
            chunks.append(chunk[: max(limit - (total - len(chunk)), 0)])
            chunks.append(b"... [truncated]")
            break
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")
