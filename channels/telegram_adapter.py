"""
Hexis Channel System - Telegram Adapter

Connects to Telegram via bot token using python-telegram-bot.
Uses long-polling mode (works behind NAT, no webhook setup needed).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
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
from .presentation import MarkdownDialect

logger = logging.getLogger(__name__)


def _resolve_token(config: dict[str, Any]) -> str | None:
    """Resolve Telegram bot token from config (env var name) or environment."""
    return resolve_channel_token(config, "bot_token", "TELEGRAM_BOT_TOKEN")


class TelegramAdapter(ChannelAdapter):
    """
    Telegram channel adapter using python-telegram-bot.

    Config keys (from DB config table):
        channel.telegram.bot_token: env var name holding the bot token
        channel.telegram.allowed_chat_ids: JSON array of chat IDs, or "*"

    The bot responds to:
        - Private messages (always)
        - Group messages where the bot is mentioned (@botname)
        - Group messages in allowed chats
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        forward_all: bool | None = None,
    ) -> None:
        self._config = config or {}
        self._forward_all = resolve_forward_all(self._config, forward_all)
        self._application = None
        self._on_message: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self._connected = False
        self._bot_username: str | None = None
        self._allowed_chat_ids = self._parse_allowlist(
            self._config.get("allowed_chat_ids")
        )

    @staticmethod
    def _parse_allowlist(value: Any) -> set[str] | None:
        """Parse an allowlist value. Returns None for '*' (allow all)."""
        return parse_allowlist(value)

    @property
    def channel_type(self) -> str:
        return "telegram"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            threads=True,  # Telegram forum topics = threads
            reactions=True,
            media=True,
            typing_indicator=True,
            edit_message=True,
            max_message_length=4096,
            markdown_dialect=MarkdownDialect.TELEGRAM,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(
        self,
        on_message: Callable[[ChannelMessage], Awaitable[None]],
    ) -> None:
        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                MessageHandler,
                filters,
            )
        except ImportError:
            raise RuntimeError(
                "python-telegram-bot is required for the Telegram adapter. "
                "Install it with: pip install python-telegram-bot"
            )

        token = _resolve_token(self._config)
        if not token:
            raise RuntimeError(
                "Telegram bot token not found. Set TELEGRAM_BOT_TOKEN env var "
                "or configure channel.telegram.bot_token in the database."
            )

        self._on_message = on_message

        application = Application.builder().token(token).build()
        self._application = application

        # Register message handler
        async def handle_message(update: Update, context) -> None:
            await self._handle_telegram_message(update)

        inbound = (
            filters.TEXT
            | filters.PHOTO
            | filters.Document.ALL
            | filters.VOICE
            | filters.AUDIO
        )
        application.add_handler(
            MessageHandler(inbound & ~filters.COMMAND, handle_message)
        )

        # Get bot info
        await application.initialize()
        bot_info = await application.bot.get_me()
        self._bot_username = bot_info.username
        self._connected = True
        logger.info(
            "Telegram connected as @%s (ID: %s)",
            bot_info.username,
            bot_info.id,
        )

        try:
            # Start polling (blocking)
            await application.start()
            await application.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=["message"],
            )

            # Keep running until cancelled
            while self._connected:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False
            try:
                if application.updater and application.updater.running:
                    await application.updater.stop()
                if application.running:
                    await application.stop()
                await application.shutdown()
            except Exception:
                logger.debug("Telegram shutdown warning", exc_info=True)

    async def _handle_telegram_message(self, update) -> None:
        """Filter and normalize a Telegram message."""
        if not update.message:
            return

        message = update.message
        chat = message.chat
        user = message.from_user

        if not user:
            return

        # Accept text plus media that can be discussed or transcribed.
        has_text = bool(message.text or message.caption)
        has_media = bool(
            message.photo or message.document or message.voice or message.audio
        )
        if not has_text and not has_media:
            return

        is_private = chat.type == "private"
        raw_text = message.text or message.caption or ""
        is_mention = bool(self._bot_username and f"@{self._bot_username}" in raw_text)
        gate_hint: str | None = None

        if not is_private:
            # Check chat allowlist
            if self._allowed_chat_ids is not None:
                if str(chat.id) not in self._allowed_chat_ids:
                    # Still respond if mentioned
                    if not is_mention:
                        if not self._forward_all:
                            return
                        gate_hint = "not_allowed_chat"

        # Strip bot mention from content
        content = raw_text
        if self._bot_username:
            content = content.replace(f"@{self._bot_username}", "").strip()

        if not content and not has_media:
            return

        # Convert Telegram attachments to Attachment instances
        attachments: list[Attachment] = []
        if message.photo:
            # Telegram provides multiple sizes; pick the largest
            photo = message.photo[-1]
            attachments.append(
                Attachment(
                    url="",  # Telegram requires bot.get_file() to get the URL
                    filename=f"photo_{photo.file_unique_id}.jpg",
                    mime_type="image/jpeg",
                    size=photo.file_size,
                    platform_id=photo.file_id,
                )
            )
        if message.document:
            doc = message.document
            attachments.append(
                Attachment(
                    url="",
                    filename=doc.file_name or f"doc_{doc.file_unique_id}",
                    mime_type=doc.mime_type,
                    size=doc.file_size,
                    platform_id=doc.file_id,
                )
            )
        if message.voice:
            voice = message.voice
            attachments.append(
                Attachment(
                    url="",
                    filename=f"voice_{voice.file_unique_id}.ogg",
                    mime_type=voice.mime_type or "audio/ogg",
                    size=voice.file_size,
                    platform_id=voice.file_id,
                )
            )
        if message.audio:
            audio = message.audio
            attachments.append(
                Attachment(
                    url="",
                    filename=audio.file_name or f"audio_{audio.file_unique_id}.mp3",
                    mime_type=audio.mime_type or "audio/mpeg",
                    size=audio.file_size,
                    platform_id=audio.file_id,
                )
            )

        sender_name = user.full_name or user.username or str(user.id)

        # Extract forum topic ID if available (I.1: Telegram topic support)
        topic_id = None
        if hasattr(message, "message_thread_id") and message.message_thread_id:
            topic_id = str(message.message_thread_id)

        channel_msg = ChannelMessage(
            channel_type="telegram",
            channel_id=str(chat.id),
            sender_id=str(user.id),
            sender_name=sender_name,
            content=content or "",
            message_id=str(message.message_id),
            reply_to_id=(
                str(message.reply_to_message.message_id)
                if message.reply_to_message
                else None
            ),
            thread_id=topic_id,
            attachments=attachments,
            metadata={
                "chat_type": chat.type,
                "is_private": is_private,
                "is_group": not is_private,
                "is_mention": is_mention,
                "username": user.username,
                "topic_id": topic_id,
                "is_topic_message": getattr(message, "is_topic_message", False),
                **({"gate_hint": gate_hint} if gate_hint else {}),
            },
        )

        if self._on_message:
            await self._on_message(channel_msg)

    async def stop(self) -> None:
        self._connected = False
        if self._application:
            try:
                if self._application.updater and self._application.updater.running:
                    await self._application.updater.stop()
                if self._application.running:
                    await self._application.stop()
                await self._application.shutdown()
            except Exception:
                logger.debug("Telegram stop warning", exc_info=True)
            self._application = None

    async def download_attachment(
        self, attachment: Attachment, *, max_size: int
    ) -> Attachment:
        if attachment.local_path or not attachment.platform_id or not self._application:
            return attachment
        if attachment.size is not None and attachment.size > max_size:
            return attachment
        telegram_file = await self._application.bot.get_file(attachment.platform_id)
        filename = Path(attachment.filename or "voice-note").name
        suffix = Path(filename).suffix[:16]
        fd, path = tempfile.mkstemp(prefix="hexis-telegram-", suffix=suffix)
        os.close(fd)
        try:
            await telegram_file.download_to_drive(custom_path=path)
            size = os.path.getsize(path)
            if size > max_size:
                os.unlink(path)
                return attachment
            return Attachment(
                url=attachment.url,
                filename=filename,
                mime_type=attachment.mime_type,
                size=size,
                platform_id=attachment.platform_id,
                local_path=path,
            )
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise

    async def send(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        if not self._application or not self._application.bot:
            logger.error("Telegram bot not connected")
            return None

        try:
            kwargs: dict[str, Any] = {
                "chat_id": int(channel_id),
                "text": text,
                "parse_mode": "Markdown",
            }
            if reply_to:
                kwargs["reply_to_message_id"] = int(reply_to)
            # I.1: Telegram forum topic support — route to specific topic
            if thread_id:
                kwargs["message_thread_id"] = int(thread_id)

            sent = await self._application.bot.send_message(**kwargs)
            return str(sent.message_id)

        except Exception:
            # Retry without Markdown in case of parse errors
            try:
                kwargs.pop("parse_mode", None)
                sent = await self._application.bot.send_message(**kwargs)
                return str(sent.message_id)
            except Exception:
                logger.exception("Failed to send Telegram message to %s", channel_id)
                return None

    async def send_typing(self, channel_id: str) -> None:
        if not self._application or not self._application.bot:
            return
        try:
            await self._application.bot.send_chat_action(
                chat_id=int(channel_id),
                action="typing",
            )
        except Exception:
            logger.debug("Silent exception in TelegramAdapter", exc_info=True)

    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
    ) -> bool:
        if not self._application or not self._application.bot:
            return False
        try:
            await self._application.bot.edit_message_text(
                chat_id=int(channel_id),
                message_id=int(message_id),
                text=text,
                parse_mode="Markdown",
            )
            return True
        except Exception:
            # Retry without Markdown
            try:
                await self._application.bot.edit_message_text(
                    chat_id=int(channel_id),
                    message_id=int(message_id),
                    text=text,
                )
                return True
            except Exception:
                logger.exception("Failed to edit Telegram message %s", message_id)
                return False

    async def send_media(
        self,
        channel_id: str,
        attachment: "Attachment",
        caption: str | None = None,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        """G.3: Send media attachments (images, documents) via Telegram."""
        if not self._application or not self._application.bot:
            return None

        try:
            kwargs: dict[str, Any] = {
                "chat_id": int(channel_id),
            }
            if caption:
                kwargs["caption"] = caption[:1024]
            if reply_to:
                kwargs["reply_to_message_id"] = int(reply_to)

            mime = attachment.mime_type or ""

            if mime.startswith("image/"):
                # Send as photo
                if attachment.local_path:
                    kwargs["photo"] = attachment.local_path
                elif attachment.url:
                    kwargs["photo"] = attachment.url
                elif attachment.platform_id:
                    kwargs["photo"] = attachment.platform_id
                else:
                    return None
                sent = await self._application.bot.send_photo(**kwargs)
            elif mime.startswith("video/"):
                if attachment.local_path:
                    kwargs["video"] = attachment.local_path
                elif attachment.url:
                    kwargs["video"] = attachment.url
                elif attachment.platform_id:
                    kwargs["video"] = attachment.platform_id
                else:
                    return None
                sent = await self._application.bot.send_video(**kwargs)
            elif mime.startswith("audio/"):
                if attachment.local_path:
                    kwargs["audio"] = attachment.local_path
                elif attachment.url:
                    kwargs["audio"] = attachment.url
                elif attachment.platform_id:
                    kwargs["audio"] = attachment.platform_id
                else:
                    return None
                sent = await self._application.bot.send_audio(**kwargs)
            else:
                # Send as document
                if attachment.local_path:
                    kwargs["document"] = attachment.local_path
                elif attachment.url:
                    kwargs["document"] = attachment.url
                elif attachment.platform_id:
                    kwargs["document"] = attachment.platform_id
                else:
                    return None
                sent = await self._application.bot.send_document(**kwargs)

            return str(sent.message_id)

        except Exception:
            logger.exception("Failed to send media to Telegram %s", channel_id)
            return None
