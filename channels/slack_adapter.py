"""
Hexis Channel System - Slack Adapter

Connects to Slack via slack-bolt using Socket Mode (primary) or HTTP events.
Listens for messages and routes them through the conversation pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from .base import (
    ChannelAdapter,
    ChannelCapabilities,
    ChannelMessage,
    parse_allowlist,
    resolve_channel_token,
    resolve_forward_all,
)
from .media import Attachment
from .presentation import (
    ActionsBlock,
    ContextBlock,
    DividerBlock,
    MarkdownDialect,
    MessagePresentation,
    TextBlock,
    render_presentation,
)

if TYPE_CHECKING:
    import asyncpg

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
        channel.slack.signing_secret: env var name for HTTP interactivity verification
        channel.slack.operator_user_id: Slack U... id for private approval DMs
        channel.slack.allowed_channels: JSON array of channel IDs, or "*"

    Connection modes:
        - Socket Mode (primary): requires both bot_token and app_token
        - HTTP Events: fallback when only bot_token is provided (requires webhook setup)
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        pool: "asyncpg.Pool | None" = None,
        forward_all: bool | None = None,
    ) -> None:
        self._config = config or {}
        self._forward_all = resolve_forward_all(self._config, forward_all)
        self._pool = pool
        self._app = None
        self._on_message: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self._connected = False
        self._bot_user_id: str | None = None
        self._allowed_channels = self._parse_allowlist(
            self._config.get("allowed_channels")
        )
        self._operator_user_id = str(self._config.get("operator_user_id") or "").strip()

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
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
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

        async def handle_approval_action(ack, body, action, client):
            # Slack requires a fast acknowledgement; the durable DB decision is
            # recorded immediately after and failures stay visible in logs.
            await ack()
            await adapter._handle_operator_approval_action(body, action, client)

        app.action("operator_approval_approve")(handle_approval_action)
        app.action("operator_approval_deny")(handle_approval_action)

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
        is_mention = bool(self._bot_user_id and f"<@{self._bot_user_id}>" in text)
        is_dm = str(event.get("channel_type") or "").lower() == "im"
        gate_hint: str | None = None

        # Check channel allowlist
        if self._allowed_channels is not None:
            if channel_id not in self._allowed_channels:
                operator_dm = (
                    is_dm
                    and bool(self._operator_user_id)
                    and user_id == self._operator_user_id
                )
                # Still respond if mentioned
                if not operator_dm and self._bot_user_id and not is_mention:
                    if not self._forward_all:
                        return
                    gate_hint = "not_allowed_channel"

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
            sender_name = (
                profile.get("display_name") or profile.get("real_name") or user_id
            )
        except Exception:
            logger.debug("Silent exception in SlackAdapter", exc_info=True)

        # Convert Slack file attachments
        attachments: list[Attachment] = []
        for f in event.get("files", []):
            attachments.append(
                Attachment(
                    url=f.get("url_private_download") or f.get("url_private") or "",
                    filename=f.get("name"),
                    mime_type=f.get("mimetype"),
                    size=f.get("size"),
                    platform_id=f.get("id"),
                )
            )

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
                "is_group": str(event.get("channel_type") or "").lower()
                not in ("im", ""),
                "channel_type": event.get("channel_type"),
                "is_mention": is_mention,
                "is_dm": is_dm,
                **({"gate_hint": gate_hint} if gate_hint else {}),
            },
        )

        if self._on_message:
            await self._on_message(channel_msg)

    async def stop(self) -> None:
        self._connected = False
        self._app = None

    async def download_attachment(
        self, attachment: Attachment, *, max_size: int
    ) -> Attachment:
        from .media import download_attachment

        token = _resolve_token(self._config, "bot_token", "SLACK_BOT_TOKEN")
        if not token:
            return attachment
        return await download_attachment(
            attachment,
            max_size=max_size,
            headers={"Authorization": f"Bearer {token}"},
        )

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
            channel_id = await self._resolve_destination(channel_id)
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

    async def _resolve_destination(self, recipient: str) -> str:
        """Turn an explicit Slack user ID into a private DM channel."""
        if not recipient.startswith("U") or not self._app:
            return recipient
        result = await self._app.client.conversations_open(users=recipient)
        channel_id = str((result.get("channel") or {}).get("id") or "")
        if not channel_id:
            raise RuntimeError(f"Slack did not open a DM for operator {recipient}")
        return channel_id

    async def send_presentation(
        self,
        channel_id: str,
        presentation: MessagePresentation,
        *,
        reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        """Render portable action blocks as native Slack Block Kit."""
        if not self._app:
            logger.error("Slack app not connected")
            return None
        destination = await self._resolve_destination(channel_id)
        blocks: list[dict[str, Any]] = []
        if presentation.title:
            blocks.append(
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": presentation.title[:150]},
                }
            )
        for block in presentation.blocks:
            if isinstance(block, TextBlock):
                text = block.text[:2999]
                blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": text},
                    }
                )
            elif isinstance(block, ContextBlock):
                blocks.append(
                    {
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": block.text[:1999]}],
                    }
                )
            elif isinstance(block, DividerBlock):
                blocks.append({"type": "divider"})
            elif isinstance(block, ActionsBlock):
                native: dict[str, Any] = {
                    "type": "actions",
                    "elements": [],
                }
                if block.block_id:
                    native["block_id"] = block.block_id[:255]
                for action in block.actions:
                    button: dict[str, Any] = {
                        "type": "button",
                        "action_id": action.action_id,
                        "text": {"type": "plain_text", "text": action.label[:75]},
                        "value": action.value[:2000],
                    }
                    if action.style != "default":
                        button["style"] = action.style
                    native["elements"].append(button)
                blocks.append(native)
        kwargs: dict[str, Any] = {
            "channel": destination,
            "text": render_presentation(presentation, MarkdownDialect.SLACK),
            "blocks": blocks,
        }
        if thread_id:
            kwargs["thread_ts"] = thread_id
        try:
            result = await self._app.client.chat_postMessage(**kwargs)
            return result.get("ts")
        except Exception:
            logger.exception("Failed to send Slack presentation to %s", destination)
            return None

    async def _handle_operator_approval_action(
        self,
        body: dict[str, Any],
        action: dict[str, Any],
        client: Any,
    ) -> None:
        """Record an identity-checked decision received over Socket Mode."""
        if self._pool is None:
            logger.warning("Slack approval action ignored: database pool unavailable")
            return
        actor = str(((body.get("user") or {}).get("id")) or "")
        try:
            value = json.loads(str(action.get("value") or "{}"))
        except json.JSONDecodeError:
            logger.warning("Slack approval action has invalid value")
            return
        request_id = str(value.get("approval_request_id") or "")
        decision = str(value.get("decision") or "")
        if not request_id or decision not in {"approve", "deny"}:
            logger.warning("Slack approval action is missing an exact decision")
            return
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT record_operator_tool_approval_decision(
                    $1::uuid, $2, 'slack', $3, NULL
                )
                """,
                request_id,
                decision,
                actor,
            )
        result = (
            raw
            if isinstance(raw, dict)
            else json.loads(raw)
            if isinstance(raw, str)
            else {}
        )
        if not result.get("ok"):
            error = str(result.get("error") or "unknown error")
            logger.warning(
                "Slack approval decision rejected: %s",
                error,
            )
            if error in {"not_pending_or_expired", "approval_disabled"}:
                channel = str(((body.get("channel") or {}).get("id")) or "")
                message_ts = str(((body.get("message") or {}).get("ts")) or "")
                if channel and message_ts:
                    try:
                        await client.chat_update(
                            channel=channel,
                            ts=message_ts,
                            text=(
                                "This protected action is no longer pending. "
                                "Ask Hexis to try it again if you still want it."
                            ),
                            blocks=[],
                        )
                    except Exception:
                        logger.debug(
                            "Failed to retire stale Slack approval buttons",
                            exc_info=True,
                        )
            return
        channel = str(((body.get("channel") or {}).get("id")) or "")
        message_ts = str(((body.get("message") or {}).get("ts")) or "")
        if channel and message_ts:
            status = str(result.get("status") or decision)
            tool_name = str(result.get("tool_name") or "protected action")
            try:
                await client.chat_update(
                    channel=channel,
                    ts=message_ts,
                    text=f"{status.title()}: {tool_name}",
                    blocks=[],
                )
            except Exception:
                logger.debug(
                    "Failed to replace resolved Slack approval buttons", exc_info=True
                )

    async def send_typing(self, channel_id: str) -> None:
        # Slack doesn't have a direct typing indicator API for bots
        # in the same way Discord/Telegram do. Omit silently.
        pass

    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
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
