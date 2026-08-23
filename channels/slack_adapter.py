"""
Hexis Channel System - Slack Adapter

Connects to Slack via slack-bolt using Socket Mode (primary) or HTTP events.
Listens for messages and routes them through the conversation pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Awaitable

from .base import ChannelAdapter, ChannelCapabilities, ChannelMessage, parse_allowlist, resolve_channel_token
from .media import Attachment
from .presentation import MarkdownDialect

logger = logging.getLogger(__name__)


def _resolve_token(config: dict[str, Any], key: str, env_fallback: str) -> str | None:
    """Resolve a token from config (env var name) or direct environment."""
    return resolve_channel_token(config, key, env_fallback)


class SlackAdapter(ChannelAdapter):
    """
    Slack channel adapter using slack-bolt.

    Config keys (from DB config table):
        channel.slack.bot_token: env var name for xoxb-... bot token
        channel.slack.app_token: env var name for xapp-... app token (Socket Mode)
        channel.slack.allowed_channels: JSON array of channel IDs, or "*"

    Connection modes:
        - Socket Mode (primary): requires both bot_token and app_token
        - HTTP Events: fallback when only bot_token is provided (requires webhook setup)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._app = None
        self._on_message: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self._connected = False
        self._bot_user_id: str | None = None
        self._allowed_channels = self._parse_allowlist(self._config.get("allowed_channels"))

    @staticmethod
    def _parse_allowlist(value: Any) -> set[str] | None:
        """Parse an allowlist value. Returns None for '*' (allow all)."""
        return parse_allowlist(value)

    @property
    def channel_type(self) -> str:
        return "slack"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            threads=True,
            reactions=True,
            media=True,
            typing_indicator=True,
            edit_message=True,
            max_message_length=4000,
            markdown_dialect=MarkdownDialect.SLACK,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(
        self,
        on_message: Callable[[ChannelMessage], Awaitable[None]],
    ) -> None:
        try:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        except ImportError:
            raise RuntimeError(
                "slack-bolt is required for the Slack adapter. "
                "Install it with: pip install slack-bolt slack-sdk"
            )

        bot_token = _resolve_token(self._config, "bot_token", "SLACK_BOT_TOKEN")
        if not bot_token:
            raise RuntimeError(
                "Slack bot token not found. Set SLACK_BOT_TOKEN env var "
                "or configure channel.slack.bot_token in the database."
            )

        app_token = _resolve_token(self._config, "app_token", "SLACK_APP_TOKEN")

        self._on_message = on_message
        app = AsyncApp(token=bot_token)
        self._app = app

        adapter = self

        @app.event("message")
        async def handle_message_events(event, say, client):
            await adapter._handle_slack_message(event, client)

        # Get bot user ID
        try:
            auth = await app.client.auth_test()
            self._bot_user_id = auth.get("user_id")
        except Exception:
            logger.warning("Could not determine Slack bot user ID")

        self._connected = True
        logger.info("Slack connected (bot_user_id=%s)", self._bot_user_id)

        try:
            if app_token:
                # Socket Mode (preferred — bidirectional, no webhook setup)
                handler = AsyncSocketModeHandler(app, app_token)
                await handler.start_async()
            else:
                # HTTP mode (requires external webhook setup)
                logger.warning(
                    "No Slack app_token — running in HTTP mode. "
                    "Set SLACK_APP_TOKEN for Socket Mode."
                )
                # Keep running until cancelled
                while self._connected:
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False

    async def _handle_slack_message(self, event: dict, client) -> None:
        """Filter and normalize a Slack message event."""
        # Ignore bot messages and message subtypes (edits, joins, etc.)
        if event.get("bot_id") or event.get("subtype"):
            return

        user_id = event.get("user")
        if not user_id or user_id == self._bot_user_id:
            return

        text = event.get("text", "")
        channel_id = event.get("channel", "")
        ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")

        # Check channel allowlist
        if self._allowed_channels is not None:
            if channel_id not in self._allowed_channels:
                # Still respond if mentioned
                if self._bot_user_id and f"<@{self._bot_user_id}>" not in text:
                    return

        # Strip bot mention
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        if not text and not event.get("files"):
            return

        # Get user info for display name
        sender_name = user_id
        try:
            user_info = await client.users_info(user=user_id)
            profile = user_info.get("user", {}).get("profile", {})
            sender_name = profile.get("display_name") or profile.get("real_name") or user_id
        except Exception:
            logger.debug("Silent exception in SlackAdapter", exc_info=True)

        # Convert Slack file attachments
        attachments: list[Attachment] = []
        for f in event.get("files", []):
            attachments.append(Attachment(
                url=f.get("url_private_download") or f.get("url_private") or "",
                filename=f.get("name"),
                mime_type=f.get("mimetype"),
                size=f.get("size"),
                platform_id=f.get("id"),
            ))

        channel_msg = ChannelMessage(
            channel_type="slack",
            channel_id=channel_id,
            sender_id=user_id,
            sender_name=sender_name,
            content=text or "",
            message_id=ts,
            thread_id=thread_ts,
            attachments=attachments,
            metadata={
                # Slack marks 1:1 DMs as "im"; "channel", "group" and "mpim"
                # (multi-party DM) all have an audience beyond the sender.
                "is_group": str(event.get("channel_type") or "").lower() not in ("im", ""),
                "channel_type": event.get("channel_type"),
            },
        )

        if self._on_message:
            await self._on_message(channel_msg)

    async def stop(self) -> None:
        self._connected = False
        self._app = None

    async def send(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        if not self._app:
            logger.error("Slack app not connected")
            return None

        try:
            kwargs: dict[str, Any] = {
                "channel": channel_id,
                "text": text,
            }
            if thread_id:
                kwargs["thread_ts"] = thread_id

            result = await self._app.client.chat_postMessage(**kwargs)
            return result.get("ts")
        except Exception:
            logger.exception("Failed to send Slack message to %s", channel_id)
            return None

    async def send_typing(self, channel_id: str) -> None:
        # Slack doesn't have a direct typing indicator API for bots
        # in the same way Discord/Telegram do. Omit silently.
        pass

    async def edit_message(
        self, channel_id: str, message_id: str, text: str,
    ) -> bool:
        if not self._app:
            return False
        try:
            await self._app.client.chat_update(
                channel=channel_id,
                ts=message_id,
                text=text,
            )
            return True
        except Exception:
            logger.exception("Failed to edit Slack message %s", message_id)
            return False

    async def send_media(
        self,
        channel_id: str,
        attachment: Attachment,
        caption: str | None = None,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        if not self._app:
            return None
        try:
            kwargs: dict[str, Any] = {"channels": channel_id}
            if caption:
                kwargs["initial_comment"] = caption

            if attachment.local_path:
                kwargs["file"] = attachment.local_path
                kwargs["filename"] = attachment.filename or "attachment"
            elif attachment.url:
                kwargs["file"] = attachment.url
                kwargs["filename"] = attachment.filename or "attachment"
            else:
                return None

            result = await self._app.client.files_upload_v2(**kwargs)
            return result.get("file", {}).get("id")
        except Exception:
            logger.exception("Failed to send Slack media to %s", channel_id)
            return None
