"""
Hexis Tools System - Messaging Integrations

Provides messaging tools for Discord, Slack, and Telegram.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from core.integration_reliability import (
    IntegrationHttpError,
    format_provider_error,
    request_json,
    request_text_response,
)

from .base import (
    OutboundSpec,
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

logger = logging.getLogger(__name__)


def _integration_tool_error_type(exc: IntegrationHttpError) -> ToolErrorType:
    if exc.error_kind == "auth_failed":
        return ToolErrorType.AUTH_FAILED
    if exc.error_kind == "rate_limited":
        return ToolErrorType.RATE_LIMITED
    if exc.error_kind == "timeout":
        return ToolErrorType.FETCH_TIMEOUT
    if exc.error_kind == "network":
        return ToolErrorType.NETWORK_ERROR
    return ToolErrorType.HTTP_ERROR


def _integration_error_result(
    provider_label: str, exc: IntegrationHttpError
) -> ToolResult:
    return ToolResult.error_result(
        format_provider_error(provider_label, exc),
        _integration_tool_error_type(exc),
    )


async def _load_db_channel_config(
    context: ToolExecutionContext,
    channel_type: str,
) -> dict[str, Any]:
    registry = context.registry
    pool = getattr(registry, "pool", None) if registry else None
    if pool is None:
        return {}
    try:
        from services.channel_worker import _load_channel_config

        async with pool.acquire() as conn:
            loaded = await _load_channel_config(conn, channel_type)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        logger.debug(
            "Failed to load %s channel config from DB", channel_type, exc_info=True
        )
        return {}


def _target_allowed(config: dict[str, Any], key: str, target: Any) -> bool:
    try:
        from channels.base import parse_allowlist

        allowed = parse_allowlist(config.get(key))
    except Exception:
        return True
    if allowed is None:
        return True
    return str(target) in allowed


class DiscordSendHandler(ToolHandler):
    """Send messages to Discord via webhook or bot API."""

    def __init__(
        self,
        config_resolver: Callable[[], dict[str, Any] | None] | None = None,
    ):
        """
        Initialize the handler.

        Args:
            config_resolver: Callable that returns Discord configuration dict with keys:
                - bot_token: Discord bot token (for bot API)
                - webhook_url: Discord webhook URL (alternative to bot)
        """
        self._config_resolver = config_resolver

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="discord_send",
            description="Send a message to a Discord channel. Use for notifications, updates, or reaching out.",
            parameters={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID (required for bot API)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message content",
                    },
                    "webhook_url": {
                        "type": "string",
                        "description": "Webhook URL (overrides default, optional)",
                    },
                    "username": {
                        "type": "string",
                        "description": "Override bot username (webhook only)",
                    },
                    "embed": {
                        "type": "object",
                        "description": "Discord embed object (optional)",
                    },
                },
                "required": ["message"],
            },
            category=ToolCategory.MESSAGING,
            energy_cost=5,
            is_read_only=False,
            requires_approval=True,
            optional=True,
            outbound=OutboundSpec(
                recipient_arg="channel_id",
                body_arg="message",
                channel="discord",
                fixed_recipient="configured_webhook",
            ),
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        config = {}
        if self._config_resolver:
            config = self._config_resolver() or {}

        message = arguments["message"]
        channel_id = arguments.get("channel_id")
        webhook_url = arguments.get("webhook_url") or config.get("webhook_url")
        username = arguments.get("username")
        embed = arguments.get("embed")
        bot_token = config.get("bot_token")

        # Prefer webhook if available
        if webhook_url:
            return await self._send_webhook(webhook_url, message, username, embed)
        elif bot_token and channel_id:
            return await self._send_bot(bot_token, channel_id, message, embed)
        else:
            return ToolResult(
                success=False,
                output=None,
                error="Discord not configured. Provide webhook_url or bot_token + channel_id",
                error_type=ToolErrorType.AUTH_FAILED,
            )

    async def _send_webhook(
        self,
        webhook_url: str,
        message: str,
        username: str | None,
        embed: dict | None,
    ) -> ToolResult:
        payload: dict[str, Any] = {"content": message}
        if username:
            payload["username"] = username
        if embed:
            payload["embeds"] = [embed]

        try:
            await request_text_response(
                "discord",
                "POST",
                webhook_url,
                json_body=payload,
                timeout=20.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )

            return ToolResult(
                success=True,
                output={"sent": True, "method": "webhook"},
                display_output=f"Discord message sent: {message[:50]}...",
            )
        except IntegrationHttpError as e:
            return _integration_error_result("Discord webhook", e)
        except Exception as e:
            logger.exception("Discord webhook error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Discord webhook failed: {str(e)}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )

    async def _send_bot(
        self,
        bot_token: str,
        channel_id: str,
        message: str,
        embed: dict | None,
    ) -> ToolResult:
        payload: dict[str, Any] = {"content": message}
        if embed:
            payload["embeds"] = [embed]

        try:
            data = await request_json(
                "discord",
                "POST",
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                json_body=payload,
                headers={"Authorization": f"Bot {bot_token}"},
                timeout=20.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )
            if not isinstance(data, dict):
                return ToolResult.error_result(
                    "Discord API returned an invalid payload.",
                    ToolErrorType.HTTP_ERROR,
                )

            return ToolResult(
                success=True,
                output={"sent": True, "method": "bot", "message_id": data.get("id")},
                display_output=f"Discord message sent to channel {channel_id}",
            )
        except IntegrationHttpError as e:
            return _integration_error_result("Discord API", e)
        except Exception as e:
            logger.exception("Discord bot error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Discord bot API failed: {str(e)}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )


class SlackSendHandler(ToolHandler):
    """Send messages to Slack via webhook or API."""

    def __init__(
        self,
        config_resolver: Callable[[], dict[str, Any] | None] | None = None,
    ):
        """
        Initialize the handler.

        Args:
            config_resolver: Callable that returns Slack configuration dict with keys:
                - bot_token: Slack bot OAuth token
                - webhook_url: Slack incoming webhook URL
        """
        self._config_resolver = config_resolver

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slack_send",
            description="Send a message to a Slack channel. Use for notifications, updates, or team communication.",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Slack channel ID or name (e.g., #general or C01234567)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message text",
                    },
                    "webhook_url": {
                        "type": "string",
                        "description": "Webhook URL (overrides default, optional)",
                    },
                    "blocks": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Slack Block Kit blocks (optional)",
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Thread timestamp to reply in thread",
                    },
                },
                "required": ["message"],
            },
            category=ToolCategory.MESSAGING,
            energy_cost=5,
            is_read_only=False,
            requires_approval=True,
            optional=True,
            outbound=OutboundSpec(
                recipient_arg="channel",
                body_arg="message",
                channel="slack",
                thread_arg="thread_ts",
                fixed_recipient="configured_webhook",
            ),
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        config = self._config_resolver() if self._config_resolver else None
        if not config:
            config = await _load_db_channel_config(context, "slack")
        config = config or {}

        message = arguments["message"]
        channel = arguments.get("channel")
        webhook_url = arguments.get("webhook_url") or config.get("webhook_url")
        blocks = arguments.get("blocks")
        thread_ts = arguments.get("thread_ts")
        bot_token = config.get("bot_token")
        try:
            from channels.slack_adapter import _resolve_token as _resolve_slack_token

            bot_token = (
                _resolve_slack_token(config, "bot_token", "SLACK_BOT_TOKEN")
                or bot_token
            )
        except Exception:
            logger.debug(
                "Slack token resolution via channel adapter failed", exc_info=True
            )

        # Prefer webhook if available
        if webhook_url:
            return await self._send_webhook(webhook_url, message, blocks)
        elif bot_token and channel:
            if not _target_allowed(config, "allowed_channels", channel):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Slack channel {channel} is not in channel.slack.allowed_channels.",
                    error_type=ToolErrorType.INVALID_PARAMS,
                )
            return await self._send_api(bot_token, channel, message, blocks, thread_ts)
        else:
            return ToolResult(
                success=False,
                output=None,
                error="Slack not configured. Provide webhook_url or bot_token + channel",
                error_type=ToolErrorType.AUTH_FAILED,
            )

    async def _send_webhook(
        self,
        webhook_url: str,
        message: str,
        blocks: list | None,
    ) -> ToolResult:
        payload: dict[str, Any] = {"text": message}
        if blocks:
            payload["blocks"] = blocks

        try:
            await request_text_response(
                "slack",
                "POST",
                webhook_url,
                json_body=payload,
                timeout=20.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )

            return ToolResult(
                success=True,
                output={"sent": True, "method": "webhook"},
                display_output=f"Slack message sent: {message[:50]}...",
            )
        except IntegrationHttpError as e:
            return _integration_error_result("Slack webhook", e)
        except Exception as e:
            logger.exception("Slack webhook error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Slack webhook failed: {str(e)}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )

    async def _send_api(
        self,
        bot_token: str,
        channel: str,
        message: str,
        blocks: list | None,
        thread_ts: str | None,
    ) -> ToolResult:
        payload: dict[str, Any] = {
            "channel": channel,
            "text": message,
        }
        if blocks:
            payload["blocks"] = blocks
        if thread_ts:
            payload["thread_ts"] = thread_ts

        try:
            data = await request_json(
                "slack",
                "POST",
                "https://slack.com/api/chat.postMessage",
                json_body=payload,
                headers={"Authorization": f"Bearer {bot_token}"},
                timeout=20.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )
            if not isinstance(data, dict):
                return ToolResult.error_result(
                    "Slack API returned an invalid payload.",
                    ToolErrorType.HTTP_ERROR,
                )
            if not data.get("ok"):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Slack API error: {data.get('error')}",
                    error_type=ToolErrorType.EXECUTION_FAILED,
                )

            return ToolResult(
                success=True,
                output={
                    "sent": True,
                    "method": "api",
                    "ts": data.get("ts"),
                    "channel": data.get("channel"),
                },
                display_output=f"Slack message sent to {channel}",
            )
        except IntegrationHttpError as e:
            return _integration_error_result("Slack API", e)
        except Exception as e:
            logger.exception("Slack API error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Slack API failed: {str(e)}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )


class TelegramSendHandler(ToolHandler):
    """Send messages via Telegram Bot API."""

    def __init__(
        self,
        config_resolver: Callable[[], dict[str, Any] | None] | None = None,
    ):
        """
        Initialize the handler.

        Args:
            config_resolver: Callable that returns Telegram configuration dict with keys:
                - bot_token: Telegram bot token from BotFather
                - default_chat_id: Default chat ID to send to
        """
        self._config_resolver = config_resolver

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="telegram_send",
            description="Send a message via Telegram. Use for notifications, alerts, or personal outreach.",
            parameters={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "Telegram chat ID (user, group, or channel)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message text (supports Markdown)",
                    },
                    "parse_mode": {
                        "type": "string",
                        "enum": ["Markdown", "MarkdownV2", "HTML"],
                        "default": "Markdown",
                        "description": "Message formatting mode",
                    },
                    "disable_notification": {
                        "type": "boolean",
                        "default": False,
                        "description": "Send silently",
                    },
                    "reply_to_message_id": {
                        "type": "integer",
                        "description": "Message ID to reply to",
                    },
                },
                "required": ["message"],
            },
            category=ToolCategory.MESSAGING,
            energy_cost=5,
            is_read_only=False,
            requires_approval=True,
            optional=True,
            outbound=OutboundSpec(
                recipient_arg="chat_id",
                body_arg="message",
                channel="telegram",
                thread_arg="reply_to_message_id",
                fixed_recipient="configured_default_chat",
            ),
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        config = self._config_resolver() if self._config_resolver else None
        if not config:
            config = await _load_db_channel_config(context, "telegram")
        config = config or {}

        bot_token = config.get("bot_token")
        try:
            from channels.telegram_adapter import (
                _resolve_token as _resolve_telegram_token,
            )

            bot_token = _resolve_telegram_token(config) or bot_token
        except Exception:
            logger.debug(
                "Telegram token resolution via channel adapter failed", exc_info=True
            )
        if not bot_token:
            return ToolResult(
                success=False,
                output=None,
                error="Telegram bot token not configured",
                error_type=ToolErrorType.AUTH_FAILED,
            )

        message = arguments["message"]
        chat_id = arguments.get("chat_id") or config.get("default_chat_id")
        if not chat_id:
            return ToolResult(
                success=False,
                output=None,
                error="No chat_id provided and no default configured",
                error_type=ToolErrorType.INVALID_PARAMS,
            )
        if not _target_allowed(config, "allowed_chat_ids", chat_id):
            return ToolResult(
                success=False,
                output=None,
                error=f"Telegram chat {chat_id} is not in channel.telegram.allowed_chat_ids.",
                error_type=ToolErrorType.INVALID_PARAMS,
            )

        parse_mode = arguments.get("parse_mode", "Markdown")
        disable_notification = arguments.get("disable_notification", False)
        reply_to = arguments.get("reply_to_message_id")

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }
        if reply_to:
            payload["reply_to_message_id"] = reply_to

        try:
            data = await request_json(
                "telegram",
                "POST",
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json_body=payload,
                timeout=20.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )
            if not isinstance(data, dict):
                return ToolResult.error_result(
                    "Telegram API returned an invalid payload.",
                    ToolErrorType.HTTP_ERROR,
                )
            if not data.get("ok"):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Telegram API error: {data.get('description')}",
                    error_type=ToolErrorType.EXECUTION_FAILED,
                )

            result_msg = data.get("result", {})
            return ToolResult(
                success=True,
                output={
                    "sent": True,
                    "message_id": result_msg.get("message_id"),
                    "chat_id": chat_id,
                },
                display_output=f"Telegram message sent to {chat_id}",
            )
        except IntegrationHttpError as e:
            return _integration_error_result("Telegram API", e)
        except Exception as e:
            logger.exception("Telegram API error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Telegram API failed: {str(e)}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )


class SignalSendHandler(ToolHandler):
    """Send messages via signal-cli-rest-api."""

    def __init__(
        self,
        config_resolver: Callable[[], dict[str, Any] | None] | None = None,
    ):
        self._config_resolver = config_resolver

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="signal_send",
            description="Send a Signal message through the configured signal-cli-rest-api sidecar.",
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Signal recipient phone number or group identifier.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message text.",
                    },
                    "api_url": {
                        "type": "string",
                        "description": "Optional signal-cli-rest-api URL override.",
                    },
                    "phone_number": {
                        "type": "string",
                        "description": "Optional sender phone-number override; normally read from channel.signal.phone_number.",
                    },
                },
                "required": ["recipient", "message"],
            },
            category=ToolCategory.MESSAGING,
            energy_cost=5,
            is_read_only=False,
            requires_approval=True,
            optional=True,
            outbound=OutboundSpec(
                recipient_arg="recipient",
                body_arg="message",
                channel="signal",
            ),
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        config = self._config_resolver() if self._config_resolver else None
        if not config:
            config = await _load_db_channel_config(context, "signal")
        config = config or {}

        recipient = str(arguments.get("recipient") or "").strip()
        message = str(arguments.get("message") or "")
        if not recipient:
            return ToolResult.error_result(
                "recipient is required.", ToolErrorType.INVALID_PARAMS
            )
        if not message.strip():
            return ToolResult.error_result(
                "message is required.", ToolErrorType.INVALID_PARAMS
            )
        if not _target_allowed(config, "allowed_numbers", recipient):
            return ToolResult.error_result(
                f"Signal recipient {recipient} is not in channel.signal.allowed_numbers.",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            from channels.signal_adapter import (
                DEFAULT_API_URL,
                _resolve_token as _resolve_signal_phone,
            )

            sender_number = str(
                arguments.get("phone_number") or ""
            ).strip() or _resolve_signal_phone(config)
            api_url = str(
                arguments.get("api_url")
                or config.get("api_url")
                or os.getenv("SIGNAL_API_URL")
                or DEFAULT_API_URL
            ).rstrip("/")
        except Exception:
            logger.debug(
                "Signal config resolution via channel adapter failed", exc_info=True
            )
            sender_number = str(
                arguments.get("phone_number") or config.get("phone_number") or ""
            ).strip()
            api_url = str(
                arguments.get("api_url")
                or config.get("api_url")
                or os.getenv("SIGNAL_API_URL")
                or "http://localhost:8080"
            ).rstrip("/")

        if not sender_number:
            return ToolResult.error_result(
                "Signal phone number not configured. Set SIGNAL_PHONE_NUMBER or channel.signal.phone_number.",
                ToolErrorType.AUTH_FAILED,
            )

        payload = {
            "message": message,
            "number": sender_number,
            "recipients": [recipient],
        }
        try:
            data = await request_json(
                "signal",
                "POST",
                f"{api_url}/api/v2/send",
                json_body=payload,
                timeout=30.0,
                attempts=3,
                max_delay=10.0,
                retry_unsafe_methods=False,
            )
            if not isinstance(data, dict):
                data = {"raw": data}
            return ToolResult.success_result(
                {
                    "sent": True,
                    "recipient": recipient,
                    "timestamp": data.get("timestamp")
                    if isinstance(data, dict)
                    else None,
                    "response": data,
                },
                display_output=f"Signal message sent to {recipient}",
            )
        except IntegrationHttpError as exc:
            return _integration_error_result("Signal API", exc)
        except Exception as exc:
            logger.exception("Signal API error")
            return ToolResult.error_result(
                f"Signal API failed: {exc}",
                ToolErrorType.EXECUTION_FAILED,
            )


def create_messaging_tools(
    discord_config_resolver: Callable[[], dict[str, Any] | None] | None = None,
    slack_config_resolver: Callable[[], dict[str, Any] | None] | None = None,
    telegram_config_resolver: Callable[[], dict[str, Any] | None] | None = None,
    signal_config_resolver: Callable[[], dict[str, Any] | None] | None = None,
) -> list[ToolHandler]:
    """
    Create messaging tool handlers.

    Args:
        discord_config_resolver: Callable that returns Discord configuration dict.
        slack_config_resolver: Callable that returns Slack configuration dict.
        telegram_config_resolver: Callable that returns Telegram configuration dict.
        signal_config_resolver: Callable that returns Signal configuration dict.

    Returns:
        List of messaging tool handlers.
    """
    return [
        DiscordSendHandler(discord_config_resolver),
        SlackSendHandler(slack_config_resolver),
        TelegramSendHandler(telegram_config_resolver),
        SignalSendHandler(signal_config_resolver),
    ]
