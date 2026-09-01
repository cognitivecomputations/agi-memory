"""Gmail provider action tools."""

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


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


class GmailSendHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="gmail_send",
            description="Send a new Gmail message using the connected Gmail account.",
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional connected Gmail account/email.",
                    },
                    "to": {
                        "type": "string",
                        "description": "Recipient email address or comma-separated addresses.",
                    },
                    "subject": {"type": "string", "description": "Email subject."},
                    "body": {"type": "string", "description": "Plain-text email body."},
                    "cc": {"type": "string", "description": "Optional CC recipients."},
                    "bcc": {
                        "type": "string",
                        "description": "Optional BCC recipients.",
                    },
                    "reply_to": {
                        "type": "string",
                        "description": "Optional Reply-To address.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=4,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            outbound=OutboundSpec(
                recipient_arg="to",
                additional_recipient_args=("cc", "bcc"),
                body_arg="body",
                channel="email",
            ),
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from core.auth.google_gmail import GmailOAuthError
        from core.tools.email import _gmail_setup_required_result
        from services.gmail_actions import GmailActionError, send_gmail_message

        try:
            result = await send_gmail_message(
                account_key=arguments.get("account_key"),
                to=str(arguments.get("to") or ""),
                subject=str(arguments.get("subject") or ""),
                body=str(arguments.get("body") or ""),
                cc=arguments.get("cc"),
                bcc=arguments.get("bcc"),
                reply_to=arguments.get("reply_to"),
            )
        except GmailOAuthError:
            return await _gmail_setup_required_result(context)
        except GmailActionError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)
        return ToolResult.success_result(
            result,
            display_output=f"Gmail sent to {', '.join(result.get('to') or [])}: {result.get('subject')}",
        )


class GmailReplyHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="gmail_reply",
            description="Reply in an existing Gmail thread using the connected Gmail account.",
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional connected Gmail account/email.",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Gmail thread ID to reply in.",
                    },
                    "to": {
                        "type": "string",
                        "description": "Recipient email address or comma-separated addresses.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Reply subject, usually Re: original subject.",
                    },
                    "body": {"type": "string", "description": "Plain-text reply body."},
                    "in_reply_to": {
                        "type": "string",
                        "description": "Optional Message-ID header being replied to.",
                    },
                    "references": {
                        "type": "string",
                        "description": "Optional References header.",
                    },
                    "cc": {"type": "string", "description": "Optional CC recipients."},
                    "bcc": {
                        "type": "string",
                        "description": "Optional BCC recipients.",
                    },
                },
                "required": ["thread_id", "to", "subject", "body"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=4,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            outbound=OutboundSpec(
                recipient_arg="to",
                additional_recipient_args=("cc", "bcc"),
                body_arg="body",
                channel="email",
                thread_arg="thread_id",
            ),
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from core.auth.google_gmail import GmailOAuthError
        from core.tools.email import _gmail_setup_required_result
        from services.gmail_actions import GmailActionError, reply_gmail_message

        try:
            result = await reply_gmail_message(
                account_key=arguments.get("account_key"),
                thread_id=str(arguments.get("thread_id") or ""),
                to=str(arguments.get("to") or ""),
                subject=str(arguments.get("subject") or ""),
                body=str(arguments.get("body") or ""),
                in_reply_to=arguments.get("in_reply_to"),
                references=arguments.get("references"),
                cc=arguments.get("cc"),
                bcc=arguments.get("bcc"),
            )
        except GmailOAuthError:
            return await _gmail_setup_required_result(context)
        except GmailActionError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)
        return ToolResult.success_result(
            result,
            display_output=f"Gmail reply sent in thread {result.get('thread_id')}.",
        )


class GmailLabelHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="gmail_label",
            description="Apply or remove Gmail labels on a message.",
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional connected Gmail account/email.",
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Gmail message ID.",
                    },
                    "add_label_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Gmail label IDs to add.",
                    },
                    "remove_label_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Gmail label IDs to remove.",
                    },
                },
                "required": ["message_id"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=2,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from core.auth.google_gmail import GmailOAuthError
        from core.tools.email import _gmail_setup_required_result
        from services.gmail_actions import GmailActionError, modify_gmail_labels

        try:
            result = await modify_gmail_labels(
                account_key=arguments.get("account_key"),
                message_id=str(arguments.get("message_id") or ""),
                add_label_ids=_string_list(arguments.get("add_label_ids")),
                remove_label_ids=_string_list(arguments.get("remove_label_ids")),
            )
        except GmailOAuthError:
            return await _gmail_setup_required_result(context)
        except GmailActionError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)
        return ToolResult.success_result(
            result,
            display_output=f"Gmail labels updated for {result.get('message_id')}.",
        )


class GmailSpamTriageHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="gmail_spam_triage",
            description="Move a Gmail message into spam, out of spam, or archive it.",
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional connected Gmail account/email.",
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Gmail message ID.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["mark_spam", "mark_not_spam", "archive"],
                        "description": "Spam triage action.",
                    },
                },
                "required": ["message_id", "action"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=2,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from core.auth.google_gmail import GmailOAuthError
        from core.tools.email import _gmail_setup_required_result
        from services.gmail_actions import GmailActionError, triage_gmail_spam

        try:
            result = await triage_gmail_spam(
                account_key=arguments.get("account_key"),
                message_id=str(arguments.get("message_id") or ""),
                action=str(arguments.get("action") or ""),
            )
        except GmailOAuthError:
            return await _gmail_setup_required_result(context)
        except GmailActionError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)
        return ToolResult.success_result(
            result,
            display_output=f"Gmail spam triage updated {result.get('message_id')}.",
        )


class GmailDeleteHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="gmail_delete",
            description=(
                "Trash or permanently delete a Gmail message using the connected Gmail account. "
                "Trash by default; set permanent only after the user explicitly requests irreversible deletion."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional connected Gmail account/email.",
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Gmail message ID.",
                    },
                    "permanent": {
                        "type": "boolean",
                        "default": False,
                        "description": "Permanently delete instead of moving to trash. Requires explicit user authorization.",
                    },
                },
                "required": ["message_id"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=3,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from core.auth.google_gmail import GmailOAuthError
        from core.tools.email import _gmail_setup_required_result
        from services.gmail_actions import GmailActionError, delete_gmail_message

        try:
            result = await delete_gmail_message(
                account_key=arguments.get("account_key"),
                message_id=str(arguments.get("message_id") or ""),
                permanent=bool(arguments.get("permanent", False)),
            )
        except GmailOAuthError:
            return await _gmail_setup_required_result(context)
        except GmailActionError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)
        verb = "permanently deleted" if result.get("permanent") else "moved to trash"
        return ToolResult.success_result(
            result,
            display_output=f"Gmail message {result.get('message_id')} {verb}.",
        )


def create_gmail_action_tools() -> list[ToolHandler]:
    return [
        GmailSendHandler(),
        GmailReplyHandler(),
        GmailLabelHandler(),
        GmailSpamTriageHandler(),
        GmailDeleteHandler(),
    ]
