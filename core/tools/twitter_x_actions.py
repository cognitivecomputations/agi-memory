"""Twitter/X provider action tools."""

from __future__ import annotations

from typing import Any

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


class TwitterXPostHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="twitter_x_post",
            description="Create a new Twitter/X post using the connected Twitter/X account.",
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional connected Twitter/X account key.",
                    },
                    "text": {"type": "string", "description": "Post text."},
                },
                "required": ["text"],
            },
            category=ToolCategory.MESSAGING,
            energy_cost=5,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            outbound=OutboundSpec(
                recipient_arg=None,
                body_arg="text",
                channel="twitter_x",
                fixed_recipient="public",
                public_recipient=True,
            ),
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from core.auth.twitter_x import TwitterXOAuthError
        from services.twitter_x import TwitterXProviderError, post_twitter_x

        try:
            result = await post_twitter_x(
                account_key=arguments.get("account_key"),
                text=str(arguments.get("text") or ""),
            )
        except (TwitterXProviderError, TwitterXOAuthError) as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)
        return ToolResult.success_result(
            result,
            display_output=f"Twitter/X post sent: {result.get('tweet_id') or '(id pending)'}",
        )


class TwitterXReplyHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="twitter_x_reply",
            description="Reply to an existing Twitter/X post using the connected Twitter/X account.",
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional connected Twitter/X account key.",
                    },
                    "reply_to_tweet_id": {
                        "type": "string",
                        "description": "Tweet/Post ID to reply to.",
                    },
                    "text": {"type": "string", "description": "Reply text."},
                },
                "required": ["reply_to_tweet_id", "text"],
            },
            category=ToolCategory.MESSAGING,
            energy_cost=5,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            outbound=OutboundSpec(
                recipient_arg=None,
                body_arg="text",
                channel="twitter_x",
                thread_arg="reply_to_tweet_id",
                fixed_recipient="public",
                public_recipient=True,
            ),
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from core.auth.twitter_x import TwitterXOAuthError
        from services.twitter_x import TwitterXProviderError, reply_twitter_x

        try:
            result = await reply_twitter_x(
                account_key=arguments.get("account_key"),
                reply_to_tweet_id=str(arguments.get("reply_to_tweet_id") or ""),
                text=str(arguments.get("text") or ""),
            )
        except (TwitterXProviderError, TwitterXOAuthError) as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)
        return ToolResult.success_result(
            result,
            display_output=f"Twitter/X reply sent: {result.get('tweet_id') or '(id pending)'}",
        )


class TwitterXDMSendHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="twitter_x_dm_send",
            description="Send a Twitter/X Direct Message to a participant using the connected Twitter/X account.",
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional connected Twitter/X account key.",
                    },
                    "participant_id": {
                        "type": "string",
                        "description": "Twitter/X user ID to receive the DM.",
                    },
                    "text": {"type": "string", "description": "Direct Message text."},
                },
                "required": ["participant_id", "text"],
            },
            category=ToolCategory.MESSAGING,
            energy_cost=5,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            outbound=OutboundSpec(
                recipient_arg="participant_id",
                body_arg="text",
                channel="twitter_x_dm",
            ),
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from core.auth.twitter_x import TwitterXOAuthError
        from services.twitter_x import TwitterXProviderError, send_twitter_x_dm

        try:
            result = await send_twitter_x_dm(
                account_key=arguments.get("account_key"),
                participant_id=str(arguments.get("participant_id") or ""),
                text=str(arguments.get("text") or ""),
            )
        except (TwitterXProviderError, TwitterXOAuthError) as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)
        return ToolResult.success_result(
            result,
            display_output=f"Twitter/X DM sent to {result.get('participant_id')}.",
        )


def create_twitter_x_action_tools() -> list[ToolHandler]:
    return [
        TwitterXPostHandler(),
        TwitterXReplyHandler(),
        TwitterXDMSendHandler(),
    ]
