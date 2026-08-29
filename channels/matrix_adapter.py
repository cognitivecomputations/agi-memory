"""
Hexis Channel System - Matrix Adapter

Connects to a Matrix homeserver via matrix-nio.
Inbound: sync_forever() callback.  Outbound: room_send() API calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Awaitable

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


def _resolve_token(config: dict[str, Any], key: str, env_fallback: str) -> str | None:
    """Resolve a token from config (env var name) or direct environment."""
    return resolve_channel_token(config, key, env_fallback)


class MatrixAdapter(ChannelAdapter):
    """
    Matrix channel adapter using matrix-nio.

    Config keys (from DB config table):
        channel.matrix.homeserver: Matrix homeserver URL (e.g. https://matrix.org)
        channel.matrix.user_id: Bot user ID (e.g. @hexis:matrix.org)
        channel.matrix.access_token: Access token for the bot user
        channel.matrix.allowed_rooms: JSON array of room IDs, or "*"

    Requires a Matrix account with an access token (not password login).
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
        self._client = None
        self._homeserver: str | None = None
        self._user_id: str | None = None
        self._allowed_rooms = self._parse_allowlist(self._config.get("allowed_rooms"))

    @staticmethod
    def _parse_allowlist(value: Any) -> set[str] | None:
        return parse_allowlist(value)

    @property
    def channel_type(self) -> str:
        return "matrix"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            threads=True,
            reactions=True,
            media=True,
            typing_indicator=True,
            edit_message=True,
            max_message_length=65536,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(
        self,
        on_message: Callable[[ChannelMessage], Awaitable[None]],
    ) -> None:
        try:
            from nio import AsyncClient, RoomMessageText, MatrixRoom
        except ImportError:
            raise RuntimeError(
                "matrix-nio is required for the Matrix adapter. "
                "Install it with: pip install matrix-nio"
            )

        self._homeserver = str(
            self._config.get("homeserver") or os.getenv("MATRIX_HOMESERVER") or ""
        )
        if not self._homeserver:
            raise RuntimeError(
                "Matrix homeserver URL not found. Set MATRIX_HOMESERVER env var "
                "or configure channel.matrix.homeserver in the database."
            )

        access_token = _resolve_token(
            self._config, "access_token", "MATRIX_ACCESS_TOKEN"
        )
        if not access_token:
            raise RuntimeError(
                "Matrix access token not found. Set MATRIX_ACCESS_TOKEN env var "
                "or configure channel.matrix.access_token in the database."
            )

        self._user_id = str(
            self._config.get("user_id") or os.getenv("MATRIX_USER_ID") or ""
        )
        if not self._user_id:
            raise RuntimeError(
                "Matrix user_id not found. Set MATRIX_USER_ID env var "
                "or configure channel.matrix.user_id in the database."
            )

        self._on_message = on_message

        client = AsyncClient(self._homeserver, self._user_id)
        client.access_token = access_token
        self._client = client

        adapter = self

        async def _on_room_message(room: MatrixRoom, event: RoomMessageText) -> None:
            await adapter._handle_matrix_message(room, event)

        client.add_event_callback(_on_room_message, RoomMessageText)

        self._connected = True
        logger.info(
            "Matrix adapter started as %s on %s", self._user_id, self._homeserver
        )

        try:
            # sync_forever blocks until cancelled
            await client.sync_forever(timeout=30000, full_state=True)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Matrix sync error")
        finally:
            self._connected = False
            if client:
                await client.close()
            self._client = None

    async def _handle_matrix_message(self, room, event) -> None:
        """Filter and normalize a Matrix room message."""
        # Ignore our own messages
        if event.sender == self._user_id:
            return

        gate_hint: str | None = None
        if self._allowed_rooms is not None:
            if room.room_id not in self._allowed_rooms:
                if not self._forward_all:
                    return
                gate_hint = "not_allowed_room"

        text = event.body or ""
        if not text:
            return

        # Extract sender display name
        sender_name = room.user_name(event.sender) or event.sender

        # Check for thread (reply-to relation)
        relates_to = (
            getattr(event, "source", {}).get("content", {}).get("m.relates_to", {})
        )
        thread_id = None
        reply_to_id = None
        if relates_to.get("rel_type") == "m.thread":
            thread_id = relates_to.get("event_id")
        in_reply_to = relates_to.get("m.in_reply_to", {})
        if in_reply_to:
            reply_to_id = in_reply_to.get("event_id")

        channel_msg = ChannelMessage(
            channel_type="matrix",
            channel_id=room.room_id,
            sender_id=event.sender,
            sender_name=sender_name,
            content=text,
            message_id=event.event_id,
            reply_to_id=reply_to_id,
            thread_id=thread_id,
            metadata={
                # Matrix has no DM flag on the event: a direct room is simply a
                # room with two members. More than that is an audience.
                "is_group": (room.member_count or 0) > 2,
                "room_name": room.display_name,
                "room_member_count": room.member_count,
                "is_mention": False,
                **({"gate_hint": gate_hint} if gate_hint else {}),
            },
        )

        if self._on_message:
            await self._on_message(channel_msg)

    async def stop(self) -> None:
        self._connected = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                logger.debug("Matrix close warning", exc_info=True)
            self._client = None

    async def send(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        if not self._client:
            logger.error("Matrix client not connected")
            return None

        try:
            content: dict[str, Any] = {
                "msgtype": "m.text",
                "body": text,
            }

            # Thread support
            if thread_id:
                content["m.relates_to"] = {
                    "rel_type": "m.thread",
                    "event_id": thread_id,
                }
                if reply_to:
                    content["m.relates_to"]["m.in_reply_to"] = {"event_id": reply_to}
            elif reply_to:
                content["m.relates_to"] = {
                    "m.in_reply_to": {"event_id": reply_to},
                }

            result = await self._client.room_send(
                room_id=channel_id,
                message_type="m.room.message",
                content=content,
            )
            return getattr(result, "event_id", None)
        except Exception:
            logger.exception("Failed to send Matrix message to %s", channel_id)
            return None

    async def send_typing(self, channel_id: str) -> None:
        if not self._client:
            return
        try:
            await self._client.room_typing(channel_id, typing_state=True, timeout=5000)
        except Exception:
            logger.debug("Silent exception in MatrixAdapter", exc_info=True)

    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
    ) -> bool:
        if not self._client:
            return False
        try:
            content: dict[str, Any] = {
                "msgtype": "m.text",
                "body": f"* {text}",
                "m.new_content": {
                    "msgtype": "m.text",
                    "body": text,
                },
                "m.relates_to": {
                    "rel_type": "m.replace",
                    "event_id": message_id,
                },
            }
            await self._client.room_send(
                room_id=channel_id,
                message_type="m.room.message",
                content=content,
            )
            return True
        except Exception:
            logger.exception("Failed to edit Matrix message %s", message_id)
            return False

    async def send_media(
        self,
        channel_id: str,
        attachment: Attachment,
        caption: str | None = None,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        if not self._client:
            return None

        try:
            # Upload the file to Matrix media repo first
            source = attachment.local_path or attachment.url
            if not source:
                return None

            mime = attachment.mime_type or "application/octet-stream"
            filename = attachment.filename or "attachment"

            if attachment.local_path:
                # Upload from local file
                with open(attachment.local_path, "rb") as f:
                    resp, _keys = await self._client.upload(
                        f,
                        content_type=mime,
                        filename=filename,
                    )
            else:
                # Download first, then upload
                from .media import download_attachment

                downloaded = await download_attachment(attachment)
                if not downloaded.local_path:
                    return None
                with open(downloaded.local_path, "rb") as f:
                    resp, _keys = await self._client.upload(
                        f,
                        content_type=mime,
                        filename=downloaded.filename or filename,
                    )

            content_uri = getattr(resp, "content_uri", None)
            if not content_uri:
                return None

            # Determine message type
            is_image = mime.startswith("image/")
            is_video = mime.startswith("video/")
            is_audio = mime.startswith("audio/")

            if is_image:
                msgtype = "m.image"
            elif is_video:
                msgtype = "m.video"
            elif is_audio:
                msgtype = "m.audio"
            else:
                msgtype = "m.file"

            content: dict[str, Any] = {
                "msgtype": msgtype,
                "body": caption or filename,
                "url": content_uri,
                "info": {
                    "mimetype": mime,
                },
            }
            if attachment.size:
                content["info"]["size"] = attachment.size

            result = await self._client.room_send(
                room_id=channel_id,
                message_type="m.room.message",
                content=content,
            )
            return getattr(result, "event_id", None)

        except Exception:
            logger.exception("Failed to send Matrix media to %s", channel_id)
            return None
