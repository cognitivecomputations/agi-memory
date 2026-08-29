"""
Hexis Channel System - WhatsApp Adapter

Connects to WhatsApp via the Business Cloud API (graph.facebook.com).
Inbound: Webhook receiver (aiohttp web server).  Outbound: HTTP REST calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from typing import Any, Callable, Awaitable

from core.integration_reliability import (
    IntegrationHttpError,
    format_provider_error,
    request_bytes_response,
    request_json,
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

GRAPH_API_BASE = "https://graph.facebook.com/v21.0"
DEFAULT_WEBHOOK_PORT = 8443


def _resolve_token(config: dict[str, Any], key: str, env_fallback: str) -> str | None:
    """Resolve a token from config (env var name) or direct environment."""
    return resolve_channel_token(config, key, env_fallback)


class WhatsAppAdapter(ChannelAdapter):
    """
    WhatsApp channel adapter via Business Cloud API.

    Config keys (from DB config table):
        channel.whatsapp.access_token: Meta access token (or env var name)
        channel.whatsapp.phone_number_id: WhatsApp Business phone number ID
        channel.whatsapp.verify_token: Webhook verification token
        channel.whatsapp.webhook_port: Port for webhook server (default: 8443)
        channel.whatsapp.allowed_numbers: JSON array of phone numbers, or "*"
        channel.whatsapp.app_secret: App secret for webhook signature verification

    Requires a Meta Business app with WhatsApp API configured.
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
        self._access_token: str | None = None
        self._phone_number_id: str | None = None
        self._verify_token: str | None = None
        self._app_secret: str | None = None
        self._webhook_port = int(
            self._config.get("webhook_port")
            or os.getenv("WHATSAPP_WEBHOOK_PORT")
            or DEFAULT_WEBHOOK_PORT
        )
        self._allowed_numbers = self._parse_allowlist(
            self._config.get("allowed_numbers")
        )
        self._session = None
        self._site = None
        self._runner = None

    @staticmethod
    def _parse_allowlist(value: Any) -> set[str] | None:
        return parse_allowlist(value)

    @property
    def channel_type(self) -> str:
        return "whatsapp"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            threads=False,
            reactions=True,
            media=True,
            typing_indicator=True,
            edit_message=False,
            max_message_length=4096,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(
        self,
        on_message: Callable[[ChannelMessage], Awaitable[None]],
    ) -> None:
        from aiohttp import web

        self._access_token = _resolve_token(
            self._config, "access_token", "WHATSAPP_ACCESS_TOKEN"
        )
        if not self._access_token:
            raise RuntimeError(
                "WhatsApp access token not found. Set WHATSAPP_ACCESS_TOKEN env var "
                "or configure channel.whatsapp.access_token in the database."
            )

        self._phone_number_id = str(
            self._config.get("phone_number_id")
            or os.getenv("WHATSAPP_PHONE_NUMBER_ID")
            or ""
        )
        if not self._phone_number_id:
            raise RuntimeError(
                "WhatsApp phone_number_id not found. Set WHATSAPP_PHONE_NUMBER_ID env var "
                "or configure channel.whatsapp.phone_number_id in the database."
            )

        self._verify_token = str(
            self._config.get("verify_token")
            or os.getenv("WHATSAPP_VERIFY_TOKEN")
            or "hexis_verify"
        )
        self._app_secret = _resolve_token(
            self._config, "app_secret", "WHATSAPP_APP_SECRET"
        )

        self._on_message = on_message

        # Build webhook server
        app = web.Application()
        app.router.add_get("/webhook", self._handle_verification)
        app.router.add_post("/webhook", self._handle_webhook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self._webhook_port)
        await self._site.start()

        self._connected = True
        logger.info(
            "WhatsApp adapter started (phone_number_id=%s, webhook port=%d)",
            self._phone_number_id,
            self._webhook_port,
        )

        try:
            while self._connected:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False
            await self._cleanup()

    async def _cleanup(self) -> None:
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _handle_verification(self, request) -> Any:
        """Handle webhook verification (GET) from Meta."""
        from aiohttp import web

        mode = request.query.get("hub.mode")
        token = request.query.get("hub.verify_token")
        challenge = request.query.get("hub.challenge")

        if mode == "subscribe" and token == self._verify_token and challenge:
            logger.info("WhatsApp webhook verified")
            return web.Response(text=challenge, content_type="text/plain")

        return web.Response(status=403, text="Forbidden")

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify webhook payload signature using app secret."""
        if not self._app_secret:
            return True  # No secret configured, skip verification

        if not signature.startswith("sha256="):
            return False

        expected = hmac.new(self._app_secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)

    async def _handle_webhook(self, request) -> Any:
        """Handle inbound webhook (POST) from Meta."""
        from aiohttp import web

        body = await request.read()

        # Verify signature
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not self._verify_signature(body, signature):
            logger.warning("WhatsApp webhook signature verification failed")
            return web.Response(status=403, text="Invalid signature")

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")

        # Always respond 200 quickly to avoid retries
        asyncio.create_task(self._process_webhook_data(data))
        return web.Response(status=200, text="OK")

    async def _process_webhook_data(self, data: dict) -> None:
        """Parse and route webhook data from WhatsApp Cloud API."""
        try:
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    if change.get("field") != "messages":
                        continue

                    contacts = {
                        c["wa_id"]: c.get("profile", {}).get("name", c["wa_id"])
                        for c in value.get("contacts", [])
                    }

                    for message in value.get("messages", []):
                        await self._handle_message(message, contacts)
        except Exception:
            logger.exception("Error processing WhatsApp webhook data")

    async def _handle_message(self, message: dict, contacts: dict) -> None:
        """Handle a single inbound WhatsApp message."""
        msg_type = message.get("type")
        sender = message.get("from", "")

        if not sender:
            return

        gate_hint: str | None = None
        if self._allowed_numbers is not None:
            if sender not in self._allowed_numbers:
                if not self._forward_all:
                    return
                gate_hint = "not_allowlisted"

        text = ""
        attachments: list[Attachment] = []

        if msg_type == "text":
            text = message.get("text", {}).get("body", "")
        elif msg_type == "image":
            img = message.get("image", {})
            text = img.get("caption", "")
            attachments.append(
                Attachment(
                    url="",  # Requires media download via API
                    filename=f"image_{message.get('id', '')}.jpg",
                    mime_type=img.get("mime_type", "image/jpeg"),
                    platform_id=img.get("id"),
                )
            )
        elif msg_type == "document":
            doc = message.get("document", {})
            text = doc.get("caption", "")
            attachments.append(
                Attachment(
                    url="",
                    filename=doc.get("filename", f"doc_{message.get('id', '')}"),
                    mime_type=doc.get("mime_type"),
                    platform_id=doc.get("id"),
                )
            )
        elif msg_type == "audio":
            audio = message.get("audio", {})
            attachments.append(
                Attachment(
                    url="",
                    filename=f"audio_{message.get('id', '')}",
                    mime_type=audio.get("mime_type"),
                    platform_id=audio.get("id"),
                )
            )
        elif msg_type == "video":
            video = message.get("video", {})
            text = video.get("caption", "")
            attachments.append(
                Attachment(
                    url="",
                    filename=f"video_{message.get('id', '')}",
                    mime_type=video.get("mime_type"),
                    platform_id=video.get("id"),
                )
            )
        else:
            # Unsupported message type
            return

        if not text and not attachments:
            return

        sender_name = contacts.get(sender, sender)
        timestamp = message.get("timestamp", "")

        channel_msg = ChannelMessage(
            channel_type="whatsapp",
            channel_id=sender,  # WhatsApp is 1:1 by default
            sender_id=sender,
            sender_name=sender_name,
            content=text or "",
            message_id=message.get("id", timestamp),
            attachments=attachments,
            metadata={
                # WhatsApp group JIDs end in @g.us; a 1:1 sender is a bare
                # number or @s.whatsapp.net.
                "is_group": str(sender or "").endswith("@g.us"),
                "message_type": msg_type,
                "is_mention": False,
                **({"gate_hint": gate_hint} if gate_hint else {}),
            },
        )

        if self._on_message:
            await self._on_message(channel_msg)

    async def stop(self) -> None:
        self._connected = False
        await self._cleanup()

    async def download_attachment(
        self, attachment: Attachment, *, max_size: int
    ) -> Attachment:
        """Resolve a WhatsApp media ID, then fetch it with the same token."""

        if (
            attachment.local_path
            or not attachment.platform_id
            or not self._access_token
        ):
            return attachment
        if attachment.size is not None and attachment.size > max_size:
            return attachment
        headers = {"Authorization": f"Bearer {self._access_token}"}
        metadata = await request_json(
            "whatsapp-media",
            "GET",
            f"{GRAPH_API_BASE}/{attachment.platform_id}",
            headers=headers,
            timeout=20.0,
            attempts=3,
            max_delay=5.0,
        )
        media_url = str(metadata.get("url") or "") if isinstance(metadata, dict) else ""
        from .media import is_safe_url, materialize_attachment_bytes

        if not media_url or not is_safe_url(media_url):
            logger.warning("WhatsApp returned an unsafe or empty media URL")
            return attachment
        response = await request_bytes_response(
            "whatsapp-media",
            "GET",
            media_url,
            headers=headers,
            timeout=30.0,
            attempts=3,
            max_delay=10.0,
            follow_redirects=True,
            max_bytes=max_size,
        )
        return materialize_attachment_bytes(
            attachment,
            response.content,
            content_type=response.headers.get("content-type"),
            prefix="hexis-whatsapp-",
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
            logger.error("WhatsApp session not connected")
            return None

        try:
            payload: dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": channel_id,
                "type": "text",
                "text": {"body": text},
            }

            if reply_to:
                payload["context"] = {"message_id": reply_to}

            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            }

            url = f"{GRAPH_API_BASE}/{self._phone_number_id}/messages"
            result = await request_json(
                "whatsapp",
                "POST",
                url,
                headers=headers,
                json_body=payload,
                timeout=30.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )
            if not isinstance(result, dict):
                logger.error("WhatsApp send failed: invalid response payload")
                return None
            messages = result.get("messages", [])
            return messages[0]["id"] if messages else None
        except IntegrationHttpError as exc:
            logger.error("%s", format_provider_error("WhatsApp", exc))
            return None
        except Exception:
            logger.exception("Failed to send WhatsApp message to %s", channel_id)
            return None

    async def send_typing(self, channel_id: str) -> None:
        """Send read receipt / typing indicator via mark-as-read."""
        if not self._session or self._session.closed:
            return

        # WhatsApp Cloud API doesn't have typing indicators,
        # but we can mark messages as read which shows blue ticks.
        pass

    async def send_media(
        self,
        channel_id: str,
        attachment: Attachment,
        caption: str | None = None,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        if not self._connected:
            return None

        try:
            # Determine media type
            mime = attachment.mime_type or ""
            if mime.startswith("image/"):
                media_type = "image"
            elif mime.startswith("video/"):
                media_type = "video"
            elif mime.startswith("audio/"):
                media_type = "audio"
            else:
                media_type = "document"

            media_obj: dict[str, Any] = {}
            if attachment.platform_id:
                media_obj["id"] = attachment.platform_id
            elif attachment.url:
                media_obj["link"] = attachment.url
            else:
                return None

            if caption:
                media_obj["caption"] = caption
            if media_type == "document" and attachment.filename:
                media_obj["filename"] = attachment.filename

            payload: dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": channel_id,
                "type": media_type,
                media_type: media_obj,
            }

            if reply_to:
                payload["context"] = {"message_id": reply_to}

            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            }

            url = f"{GRAPH_API_BASE}/{self._phone_number_id}/messages"
            result = await request_json(
                "whatsapp",
                "POST",
                url,
                headers=headers,
                json_body=payload,
                timeout=30.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )
            if not isinstance(result, dict):
                logger.error("WhatsApp send_media failed: invalid response payload")
                return None
            messages = result.get("messages", [])
            return messages[0]["id"] if messages else None
        except IntegrationHttpError as exc:
            logger.error("%s", format_provider_error("WhatsApp", exc))
            return None
        except Exception:
            logger.exception("Failed to send WhatsApp media to %s", channel_id)
            return None
