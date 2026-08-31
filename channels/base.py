"""
Hexis Channel System - Base Types

Core abstractions for multi-channel messaging:
- ChannelMessage: Normalized message format across all platforms
- ChannelAdapter: ABC that each channel (Discord, Telegram, etc.) implements
- ChannelCapabilities: What a channel supports (threads, reactions, media)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from .presentation import MarkdownDialect, MessagePresentation, render_presentation

if TYPE_CHECKING:
    from .media import Attachment


@dataclass
class ChannelCapabilities:
    """Declares what a channel supports."""

    threads: bool = False
    reactions: bool = False
    media: bool = False
    typing_indicator: bool = False
    edit_message: bool = False
    max_message_length: int = 4000
    markdown_dialect: MarkdownDialect = MarkdownDialect.PLAIN


@dataclass
class ChannelMessage:
    """
    Normalized message from any channel.

    Every channel adapter converts platform-specific messages into this
    format before handing them to the conversation handler.
    """

    channel_type: str  # "discord", "telegram"
    channel_id: str  # Platform chat/channel ID
    sender_id: str  # Platform user ID
    sender_name: str  # Display name
    content: str  # Text content
    message_id: str  # Platform message ID (for replies)
    reply_to_id: str | None = None
    thread_id: str | None = None
    attachments: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_group(self) -> bool:
        """Whether this message arrived in a group context (#92): adapters
        record it in metadata; the conversation handler threads it into
        recall so private memories stay out of shared rooms."""
        meta = self.metadata or {}
        if "is_group" in meta:
            return bool(meta.get("is_group"))
        if "is_private" in meta:
            return not bool(meta.get("is_private"))
        return False

    def __post_init__(self) -> None:
        """Convert raw dicts in attachments to Attachment instances."""
        from .media import Attachment as Att

        converted = []
        for item in self.attachments:
            if isinstance(item, dict):
                converted.append(Att.from_dict(item))
            else:
                converted.append(item)
        self.attachments = converted


class ChannelAdapter(ABC):
    """
    Base class for channel adapters.

    Each platform (Discord, Telegram, etc.) implements this interface.
    The adapter handles:
    - Connecting to the platform API
    - Listening for inbound messages and normalizing them
    - Sending outbound messages in the platform's format
    """

    @abstractmethod
    async def start(
        self,
        on_message: Callable[[ChannelMessage], Awaitable[None]],
    ) -> None:
        """
        Start listening for messages.

        Args:
            on_message: Callback invoked for each inbound message.
                        The callback should process the message and send
                        a reply via this adapter's send() method.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening and disconnect from the platform."""
        ...

    @abstractmethod
    async def send(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        """
        Send a message to a channel.

        Args:
            channel_id: Platform-specific channel/chat ID.
            text: Message text.
            reply_to: Platform message ID to reply to.
            thread_id: Thread ID for threaded replies.

        Returns:
            Platform message ID of the sent message, or None.
        """
        ...

    @abstractmethod
    async def send_typing(self, channel_id: str) -> None:
        """Send a typing indicator to signal the bot is processing."""
        ...

    @property
    @abstractmethod
    def channel_type(self) -> str:
        """Unique channel identifier, e.g. 'discord', 'telegram'."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> ChannelCapabilities:
        """Declare what this channel supports."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the adapter is currently connected and listening."""
        return False

    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
    ) -> bool:
        """
        Edit a previously sent message.

        Override in adapters that support message editing (Discord, Telegram,
        Slack, Matrix).  Returns True on success, False if not supported.
        """
        return False

    async def send_media(
        self,
        channel_id: str,
        attachment: Attachment,
        caption: str | None = None,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        """
        Send a media attachment to a channel.

        Override in adapters that support media sending.
        Returns platform message ID, or None if not supported.
        """
        return None

    async def download_attachment(
        self,
        attachment: Attachment,
        *,
        max_size: int,
    ) -> Attachment:
        """Materialize one inbound attachment for a service such as STT.

        Adapters with authenticated or platform-id media override this. The
        portable fallback keeps SSRF and response-size checks in one place.
        """
        from .media import download_attachment

        return await download_attachment(attachment, max_size=max_size)

    async def send_presentation(
        self,
        channel_id: str,
        presentation: MessagePresentation,
        *,
        reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        """Render and deliver portable blocks, degrading through text safely.

        The first platform message ID is returned for reply/audit correlation;
        every additional chunk is still delivered in order.
        """
        text = render_presentation(presentation, self.capabilities.markdown_dialect)
        chunks = chunk_text(text, self.capabilities.max_message_length)
        first_message_id: str | None = None
        for index, chunk in enumerate(chunks):
            message_id = await self.send(
                channel_id,
                chunk,
                reply_to=reply_to if index == 0 else None,
                thread_id=thread_id,
            )
            if index == 0:
                first_message_id = message_id
        return first_message_id


def parse_allowlist(value: Any) -> set[str] | None:
    """Parse an allowlist value from config. Returns None for '*' (allow all)."""
    if value is None or value == "*":
        return None
    if isinstance(value, str):
        import json

        try:
            value = json.loads(value)
        except Exception:
            return {value}
    if isinstance(value, list):
        return {str(v) for v in value}
    return None


def resolve_forward_all(
    config: dict[str, Any],
    override: bool | None = None,
) -> bool:
    """Resolve the transport-only mode used by inbound disposition.

    The explicit constructor flag is useful to isolated adapter tests. In
    production the channel worker supplies ``config["forward_all"]`` so a
    runtime feature-flag change needs no adapter restart. The manager uses
    each message's legacy gate hint while the DB-owned layer is disabled.
    """
    if override is not None:
        return bool(override)
    value = config.get("forward_all")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def resolve_channel_token(
    config: dict[str, Any],
    key: str = "bot_token",
    env_fallback: str = "",
) -> str | None:
    """Resolve a channel token/secret from config or environment.

    Config stores the env var NAME, not the value. Falls back to treating
    the config value as a raw token if it looks like one.
    """
    import os

    token_env = config.get(key) or config.get(f"{key}_env") or env_fallback
    token = os.getenv(str(token_env)) if token_env else None
    if token:
        return token
    raw = config.get(key, "")
    if raw and len(str(raw)) > 20:
        return str(raw)
    return None


def chunk_text(text: str, max_length: int) -> list[str]:
    """
    Split text into chunks respecting the channel's max message length.

    Tries to split on paragraph boundaries, then sentence boundaries,
    then falls back to hard splits.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Try to split on double newline (paragraph)
        split_at = remaining.rfind("\n\n", 0, max_length)
        if split_at > max_length // 3:
            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
            continue

        # Try to split on single newline
        split_at = remaining.rfind("\n", 0, max_length)
        if split_at > max_length // 3:
            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
            continue

        # Try to split on sentence boundary
        for sep in (". ", "! ", "? "):
            split_at = remaining.rfind(sep, 0, max_length)
            if split_at > max_length // 3:
                chunks.append(remaining[: split_at + 1].rstrip())
                remaining = remaining[split_at + 1 :].lstrip()
                break
        else:
            # Hard split on space
            split_at = remaining.rfind(" ", 0, max_length)
            if split_at > max_length // 3:
                chunks.append(remaining[:split_at].rstrip())
                remaining = remaining[split_at:].lstrip()
            else:
                # Last resort: hard split
                chunks.append(remaining[:max_length])
                remaining = remaining[max_length:]

    return chunks
