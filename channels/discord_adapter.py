"""
Hexis Channel System - Discord Adapter

Connects to Discord via bot token using discord.py.
Listens for messages and routes them through the conversation pipeline.
"""

from __future__ import annotations

import asyncio
import logging
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
    return resolve_channel_token(config, "bot_token", "DISCORD_BOT_TOKEN")


class DiscordAdapter(ChannelAdapter):
    """
    Discord channel adapter using discord.py.

    Config keys (from DB config table):
        channel.discord.bot_token: env var name holding the bot token
        channel.discord.allowed_guilds: JSON array of guild IDs, or "*"
        channel.discord.allowed_channels: JSON array of channel IDs, or "*"

    The bot responds to:
        - Direct messages (always)
        - Channel messages where the bot is mentioned
        - Channel messages in allowed channels
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        forward_all: bool | None = None,
    ) -> None:
        self._config = config or {}
        self._forward_all = resolve_forward_all(self._config, forward_all)
        self._client = None
        self._on_message: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self._connected = False
        self._allowed_guilds = self._parse_allowlist(self._config.get("allowed_guilds"))
        self._allowed_channels = self._parse_allowlist(
            self._config.get("allowed_channels")
        )

    @staticmethod
    def _parse_allowlist(value: Any) -> set[str] | None:
        return parse_allowlist(value)

    @property
    def channel_type(self) -> str:
        return "discord"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            threads=True,
            reactions=True,
            media=True,
            typing_indicator=True,
            edit_message=True,
            max_message_length=2000,
            markdown_dialect=MarkdownDialect.MARKDOWN,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(
        self,
        on_message: Callable[[ChannelMessage], Awaitable[None]],
    ) -> None:
        try:
            import discord
        except ImportError:
            raise RuntimeError(
                "discord.py is required for the Discord adapter. "
                "Install it with: pip install discord.py"
            )

        token = _resolve_token(self._config)
        if not token:
            raise RuntimeError(
                "Discord bot token not found. Set DISCORD_BOT_TOKEN env var "
                "or configure channel.discord.bot_token in the database."
            )

        self._on_message = on_message

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.dm_messages = True

        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_ready():
            self._connected = True
            logger.info(
                "Discord connected as %s (ID: %s) in %d guilds",
                client.user.name if client.user else "?",
                client.user.id if client.user else "?",
                len(client.guilds),
            )

        @client.event
        async def on_message(message: discord.Message):
            await self._handle_discord_message(message)

        try:
            await client.start(token)
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False
            if not client.is_closed():
                await client.close()

    async def _handle_discord_message(self, message) -> None:
        """Filter and normalize a Discord message."""
        import discord

        # Ignore own messages and bot messages
        if not self._client or not self._client.user:
            return
        if message.author.id == self._client.user.id:
            return
        if message.author.bot:
            return

        # Skip empty messages
        if not message.content and not message.attachments:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mention = bool(self._client.user.mentioned_in(message))
        gate_hint: str | None = None

        if not is_dm:
            # Check guild allowlist
            if self._allowed_guilds is not None and message.guild:
                if str(message.guild.id) not in self._allowed_guilds:
                    if not self._forward_all:
                        return
                    gate_hint = "not_allowed_guild"

            # Check channel allowlist
            if self._allowed_channels is not None:
                if str(message.channel.id) not in self._allowed_channels:
                    # Still respond if mentioned
                    if not is_mention and gate_hint is None:
                        if not self._forward_all:
                            return
                        gate_hint = "not_allowed_channel"

        # Strip bot mention from content
        content = message.content
        if self._client.user:
            mention_str = f"<@{self._client.user.id}>"
            mention_nick = f"<@!{self._client.user.id}>"
            content = content.replace(mention_str, "").replace(mention_nick, "").strip()

        if not content:
            return

        # Build normalized message
        thread_id = None
        if isinstance(message.channel, discord.Thread):
            thread_id = str(message.channel.id)

        # Convert Discord attachments to Attachment instances
        attachments = [
            Attachment(
                url=a.url,
                filename=a.filename,
                mime_type=a.content_type,
                size=a.size,
                platform_id=str(a.id),
            )
            for a in message.attachments
        ]

        channel_msg = ChannelMessage(
            channel_type="discord",
            channel_id=str(message.channel.id),
            sender_id=str(message.author.id),
            sender_name=message.author.display_name,
            content=content,
            message_id=str(message.id),
            reply_to_id=str(message.reference.message_id)
            if message.reference
            else None,
            thread_id=thread_id,
            attachments=attachments,
            metadata={
                # A message with no guild arrived in a DM; anything inside a
                # guild is a room other people can read.
                "is_group": message.guild is not None,
                "guild_id": str(message.guild.id) if message.guild else None,
                "is_dm": is_dm,
                "is_mention": is_mention,
                **({"gate_hint": gate_hint} if gate_hint else {}),
            },
        )

        if self._on_message:
            await self._on_message(channel_msg)

    async def stop(self) -> None:
        self._connected = False
        if self._client and not self._client.is_closed():
            await self._client.close()
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
            logger.error("Discord client not connected")
            return None

        try:
            target_id = int(thread_id or channel_id)
            channel = self._client.get_channel(target_id)
            if channel is None:
                channel = await self._client.fetch_channel(target_id)

            reference = None
            if reply_to:
                import discord

                reference = discord.MessageReference(
                    message_id=int(reply_to),
                    channel_id=int(channel_id),
                )

            sent = await channel.send(text, reference=reference)
            return str(sent.id)

        except Exception:
            logger.exception("Failed to send Discord message to %s", channel_id)
            return None

    async def send_typing(self, channel_id: str) -> None:
        if not self._client:
            return
        try:
            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))
            await channel.typing()
        except Exception:
            logger.debug("Silent exception in DiscordAdapter", exc_info=True)

    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
    ) -> bool:
        if not self._client:
            return False
        try:
            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))
            msg = await channel.fetch_message(int(message_id))
            await msg.edit(content=text)
            return True
        except Exception:
            logger.exception("Failed to edit Discord message %s", message_id)
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
            import discord

            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))

            reference = None
            if reply_to:
                reference = discord.MessageReference(
                    message_id=int(reply_to),
                    channel_id=int(channel_id),
                )

            if attachment.local_path:
                sent = await channel.send(
                    content=caption,
                    file=discord.File(
                        attachment.local_path, filename=attachment.filename
                    ),
                    reference=reference,
                )
            else:
                # Send URL as an embed
                embed = discord.Embed()
                if attachment.mime_type and attachment.mime_type.startswith("image/"):
                    embed.set_image(url=attachment.url)
                else:
                    embed.description = (
                        f"[{attachment.filename or 'Attachment'}]({attachment.url})"
                    )
                sent = await channel.send(
                    content=caption, embed=embed, reference=reference
                )

            return str(sent.id)
        except Exception:
            logger.exception("Failed to send Discord media to %s", channel_id)
            return None
