"""
Hexis Tools System - Email Integration

Provides email tools for sending and reading messages.
Supports SMTP/SendGrid for sending, and Google Gmail API for reading.
"""

from __future__ import annotations

import logging
import inspect
import smtplib
import ssl
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable

from core.integration_reliability import (
    IntegrationHttpError,
    format_provider_error,
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

GMAIL_SETUP_HINT = (
    "Gmail is not connected yet. Open the structured Gmail setup flow in this "
    "conversation before reading or managing email."
)


async def _gmail_setup_required_result(context: ToolExecutionContext) -> ToolResult:
    """Return a Gmail auth failure with the setup UI attached when available."""
    display = "Gmail is not connected. Open the Gmail setup panel to approve access with Google."
    output: dict[str, Any] = {
        "connector_id": "gmail",
        "status": "needs_connection",
        "next_step": display,
    }

    registry = context.registry
    execute = getattr(registry, "execute", None)
    if callable(execute):
        try:
            status_context = ToolExecutionContext(
                tool_context=context.tool_context,
                call_id=f"gmail-setup-status:{uuid.uuid4()}",
                heartbeat_id=context.heartbeat_id,
                session_id=context.session_id,
                energy_available=context.energy_available,
                workspace_path=context.workspace_path,
                is_group=context.is_group,
                allow_network=False,
                allow_shell=False,
                allow_file_write=False,
                allow_file_read=context.allow_file_read,
                registry=registry,
            )
            maybe_result = execute("gmail_setup_status", {}, status_context)
            if inspect.isawaitable(maybe_result):
                maybe_result = await maybe_result
            if isinstance(maybe_result, ToolResult) and isinstance(
                maybe_result.output, dict
            ):
                output = dict(maybe_result.output)
                display = maybe_result.display_output or display
        except Exception:
            logger.debug(
                "Could not attach Gmail setup status to auth failure", exc_info=True
            )

    if not isinstance(output.get("ui"), dict):
        try:
            from core.tools.integrations import _gmail_connector_setup_ui

            output["ui"] = _gmail_connector_setup_ui(
                output,
                status="needs_client_secret",
                next_step=output.get("next_step") or display,
            )
        except Exception:
            logger.debug(
                "Could not build Gmail setup UI for auth failure", exc_info=True
            )

    return ToolResult(
        success=False,
        output=output,
        display_output=display,
        error=GMAIL_SETUP_HINT,
        error_type=ToolErrorType.AUTH_FAILED,
    )


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


class EmailSendHandler(ToolHandler):
    """Send email via SMTP."""

    def __init__(
        self,
        config_resolver: Callable[[], dict[str, Any] | None] | None = None,
    ):
        """
        Initialize the handler.

        Args:
            config_resolver: Callable that returns SMTP configuration dict with keys:
                - smtp_host: SMTP server hostname
                - smtp_port: SMTP server port (default: 587)
                - smtp_user: SMTP username
                - smtp_password: SMTP password
                - from_email: Default sender email
                - from_name: Default sender name (optional)
        """
        self._config_resolver = config_resolver

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="email_send",
            description="Send an email message. Use for important communications, notifications, or outreach.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body (plain text)",
                    },
                    "html_body": {
                        "type": "string",
                        "description": "Email body (HTML, optional)",
                    },
                    "cc": {
                        "type": "string",
                        "description": "CC recipients (comma-separated)",
                    },
                    "reply_to": {
                        "type": "string",
                        "description": "Reply-to address (optional)",
                    },
                },
                "required": ["to", "subject", "body"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=4,
            is_read_only=False,
            requires_approval=True,
            optional=True,
            outbound=OutboundSpec(
                recipient_arg="to",
                additional_recipient_args=("cc",),
                body_arg="body",
                html_body_arg="html_body",
                channel="email",
            ),
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        config = None
        if self._config_resolver:
            config = self._config_resolver()

        if not config:
            return ToolResult(
                success=False,
                output=None,
                error="Email configuration not set. Configure SMTP settings via 'hexis tools set-api-key email_send EMAIL_CONFIG'",
                error_type=ToolErrorType.AUTH_FAILED,
            )

        smtp_host = config.get("smtp_host")
        smtp_port = config.get("smtp_port", 587)
        smtp_user = config.get("smtp_user")
        smtp_password = config.get("smtp_password")
        from_email = config.get("from_email")
        from_name = config.get("from_name", "")

        if not all([smtp_host, smtp_user, smtp_password, from_email]):
            return ToolResult(
                success=False,
                output=None,
                error="Incomplete SMTP configuration. Required: smtp_host, smtp_user, smtp_password, from_email",
                error_type=ToolErrorType.INVALID_PARAMS,
            )

        to_email = arguments["to"]
        subject = arguments["subject"]
        body = arguments["body"]
        html_body = arguments.get("html_body")
        cc = arguments.get("cc")
        reply_to = arguments.get("reply_to")

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{from_name} <{from_email}>" if from_name else from_email
            msg["To"] = to_email

            if cc:
                msg["Cc"] = cc
            if reply_to:
                msg["Reply-To"] = reply_to

            # Attach plain text
            msg.attach(MIMEText(body, "plain"))

            # Attach HTML if provided
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            # Send via SMTP
            ssl_context = ssl.create_default_context()

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls(context=ssl_context)
                server.login(smtp_user, smtp_password)

                recipients = [to_email]
                if cc:
                    recipients.extend([e.strip() for e in cc.split(",")])

                server.sendmail(from_email, recipients, msg.as_string())

            return ToolResult(
                success=True,
                output={
                    "to": to_email,
                    "subject": subject,
                    "sent": True,
                },
                display_output=f"Email sent to {to_email}: {subject}",
            )

        except smtplib.SMTPAuthenticationError as e:
            logger.exception("SMTP auth error")
            return ToolResult(
                success=False,
                output=None,
                error=f"SMTP authentication failed: {str(e)}",
                error_type=ToolErrorType.AUTH_FAILED,
            )
        except smtplib.SMTPException as e:
            logger.exception("SMTP error")
            return ToolResult(
                success=False,
                output=None,
                error=f"SMTP error: {str(e)}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )
        except Exception as e:
            logger.exception("Email send error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to send email: {str(e)}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )


class SendGridEmailHandler(ToolHandler):
    """Send email via SendGrid API."""

    def __init__(
        self,
        api_key_resolver: Callable[[], str | None] | None = None,
        from_email: str | None = None,
    ):
        self._api_key_resolver = api_key_resolver
        self._from_email = from_email

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="email_send_sendgrid",
            description="Send an email via SendGrid API. Alternative to SMTP-based sending.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body (plain text)",
                    },
                    "html_body": {
                        "type": "string",
                        "description": "Email body (HTML, optional)",
                    },
                    "from_email": {
                        "type": "string",
                        "description": "Sender email (if different from default)",
                    },
                },
                "required": ["to", "subject", "body"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=4,
            is_read_only=False,
            requires_approval=True,
            optional=True,
            outbound=OutboundSpec(
                recipient_arg="to",
                body_arg="body",
                html_body_arg="html_body",
                channel="email",
            ),
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        api_key = None
        if self._api_key_resolver:
            api_key = self._api_key_resolver()

        if not api_key:
            return ToolResult(
                success=False,
                output=None,
                error="SendGrid API key not configured",
                error_type=ToolErrorType.AUTH_FAILED,
            )

        to_email = arguments["to"]
        subject = arguments["subject"]
        body = arguments["body"]
        html_body = arguments.get("html_body")
        from_email = arguments.get("from_email") or self._from_email

        if not from_email:
            return ToolResult(
                success=False,
                output=None,
                error="No sender email configured",
                error_type=ToolErrorType.INVALID_PARAMS,
            )

        try:
            content = [{"type": "text/plain", "value": body}]
            if html_body:
                content.append({"type": "text/html", "value": html_body})

            payload = {
                "personalizations": [{"to": [{"email": to_email}]}],
                "from": {"email": from_email},
                "subject": subject,
                "content": content,
            }

            await request_text_response(
                "sendgrid",
                "POST",
                "https://api.sendgrid.com/v3/mail/send",
                json_body=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
                attempts=3,
                max_delay=15.0,
                retry_unsafe_methods=False,
            )

            return ToolResult(
                success=True,
                output={
                    "to": to_email,
                    "subject": subject,
                    "sent": True,
                },
                display_output=f"Email sent to {to_email}: {subject}",
            )

        except IntegrationHttpError as e:
            return ToolResult(
                success=False,
                output=None,
                error=format_provider_error("SendGrid", e),
                error_type=_integration_tool_error_type(e),
            )
        except Exception as e:
            logger.exception("SendGrid send error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to send email: {str(e)}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )


class EmailListHandler(ToolHandler):
    """List recent emails from a mailbox via Gmail API."""

    def __init__(
        self,
        credentials_resolver: Callable[[], dict[str, Any] | None] | None = None,
    ):
        self._credentials_resolver = credentials_resolver

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="email_list",
            description=(
                "List recent emails from the inbox or another label. "
                "Use this for live requests like 'check my email', 'read a batch of my emails', "
                "or 'tell me whether anything urgent came in'. "
                "Returns subject, sender, date, and snippet for each message."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "default": "INBOX",
                        "description": "Gmail label to list (INBOX, SENT, STARRED, IMPORTANT, etc.)",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of emails to return",
                    },
                    "unread_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "Only return unread messages",
                    },
                },
            },
            category=ToolCategory.EMAIL,
            energy_cost=2,
            is_read_only=True,
            optional=True,
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        credentials = None
        if self._credentials_resolver:
            credentials = self._credentials_resolver()

        if not credentials:
            return await _gmail_setup_required_result(context)

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="google-api-python-client not installed. Run: pip install google-api-python-client google-auth",
                error_type=ToolErrorType.MISSING_DEPENDENCY,
            )

        label = arguments.get("label", "INBOX")
        max_results = arguments.get("max_results", 10)
        unread_only = arguments.get("unread_only", False)

        try:
            creds = Credentials.from_authorized_user_info(credentials)
            service = build("gmail", "v1", credentials=creds)

            # Build label + unread filter
            label_ids = [label]
            if unread_only:
                label_ids.append("UNREAD")

            # List message IDs
            resp = (
                service.users()
                .messages()
                .list(userId="me", labelIds=label_ids, maxResults=max_results)
                .execute()
            )

            messages = resp.get("messages", [])
            if not messages:
                return ToolResult(
                    success=True,
                    output={"emails": [], "count": 0, "label": label},
                    display_output=f"No emails in {label}"
                    + (" (unread)" if unread_only else ""),
                )

            # Fetch metadata for each message
            emails = []
            for msg_stub in messages:
                msg = (
                    service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=msg_stub["id"],
                        format="metadata",
                        metadataHeaders=["From", "Subject", "Date", "To"],
                    )
                    .execute()
                )
                headers = {
                    h["name"]: h["value"]
                    for h in msg.get("payload", {}).get("headers", [])
                }
                emails.append(
                    {
                        "id": msg["id"],
                        "thread_id": msg.get("threadId"),
                        "from": headers.get("From", ""),
                        "to": headers.get("To", ""),
                        "subject": headers.get("Subject", "(No subject)"),
                        "date": headers.get("Date", ""),
                        "snippet": msg.get("snippet", ""),
                        "unread": "UNREAD" in msg.get("labelIds", []),
                    }
                )

            display_lines = []
            for e in emails:
                marker = "*" if e["unread"] else " "
                display_lines.append(f"{marker} {e['date']}: {e['from']}")
                display_lines.append(f"  {e['subject']}")

            return ToolResult(
                success=True,
                output={"emails": emails, "count": len(emails), "label": label},
                display_output="\n".join(display_lines),
            )

        except Exception as e:
            logger.exception("Gmail list error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Gmail API error: {e}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )


class EmailReadHandler(ToolHandler):
    """Read a specific email by ID via Gmail API."""

    def __init__(
        self,
        credentials_resolver: Callable[[], dict[str, Any] | None] | None = None,
    ):
        self._credentials_resolver = credentials_resolver

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="email_read",
            description=(
                "Read the full content of a specific email by its message ID. "
                "Use email_list or email_search first to get message IDs. "
                "Setting mark_read=true changes Gmail state and requires connector action authorization outside chat."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "Gmail message ID (from email_list or email_search)",
                    },
                    "mark_read": {
                        "type": "boolean",
                        "default": False,
                        "description": "Mark the message as read after fetching. This changes Gmail state.",
                    },
                },
                "required": ["message_id"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=2,
            is_read_only=True,
            optional=True,
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        credentials = None
        if self._credentials_resolver:
            credentials = self._credentials_resolver()

        if not credentials:
            return await _gmail_setup_required_result(context)

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="google-api-python-client not installed.",
                error_type=ToolErrorType.MISSING_DEPENDENCY,
            )

        message_id = arguments["message_id"]
        mark_read = arguments.get("mark_read", False)

        try:
            creds = Credentials.from_authorized_user_info(credentials)
            service = build("gmail", "v1", credentials=creds)

            msg = (
                service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

            # Extract headers
            headers = {
                h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])
            }

            # Extract body
            body_text = _extract_body(msg.get("payload", {}))

            # Extract attachments metadata
            attachments = _extract_attachments(msg.get("payload", {}))

            # Optionally mark as read
            if mark_read and "UNREAD" in msg.get("labelIds", []):
                service.users().messages().modify(
                    userId="me",
                    id=message_id,
                    body={"removeLabelIds": ["UNREAD"]},
                ).execute()

            result = {
                "id": msg["id"],
                "thread_id": msg.get("threadId"),
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "cc": headers.get("Cc", ""),
                "subject": headers.get("Subject", "(No subject)"),
                "date": headers.get("Date", ""),
                "body": body_text[:10000] if body_text else "",
                "labels": msg.get("labelIds", []),
                "attachments": attachments,
            }

            display = f"From: {result['from']}\nTo: {result['to']}\nDate: {result['date']}\nSubject: {result['subject']}\n\n{result['body'][:2000]}"

            return ToolResult(
                success=True,
                output=result,
                display_output=display,
            )

        except Exception as e:
            logger.exception("Gmail read error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Gmail API error: {e}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )


class EmailSearchHandler(ToolHandler):
    """Search emails using Gmail's search syntax."""

    def __init__(
        self,
        credentials_resolver: Callable[[], dict[str, Any] | None] | None = None,
    ):
        self._credentials_resolver = credentials_resolver

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="email_search",
            description=(
                "Search emails using Gmail query syntax. Supports operators like "
                "from:, to:, subject:, after:, before:, has:attachment, is:unread, label:, etc. "
                "Use this for live requests to find specific messages before reading or analyzing them."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Gmail search query. Examples: "
                            "'from:alice@example.com', 'subject:invoice after:2024/01/01', "
                            "'is:unread has:attachment', 'newer_than:7d'"
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of results",
                    },
                },
                "required": ["query"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=2,
            is_read_only=True,
            optional=True,
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        credentials = None
        if self._credentials_resolver:
            credentials = self._credentials_resolver()

        if not credentials:
            return await _gmail_setup_required_result(context)

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="google-api-python-client not installed.",
                error_type=ToolErrorType.MISSING_DEPENDENCY,
            )

        query = arguments["query"]
        max_results = arguments.get("max_results", 10)

        try:
            creds = Credentials.from_authorized_user_info(credentials)
            service = build("gmail", "v1", credentials=creds)

            resp = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )

            messages = resp.get("messages", [])
            if not messages:
                return ToolResult(
                    success=True,
                    output={"results": [], "count": 0, "query": query},
                    display_output=f"No emails matching: {query}",
                )

            results = []
            for msg_stub in messages:
                msg = (
                    service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=msg_stub["id"],
                        format="metadata",
                        metadataHeaders=["From", "Subject", "Date", "To"],
                    )
                    .execute()
                )
                headers = {
                    h["name"]: h["value"]
                    for h in msg.get("payload", {}).get("headers", [])
                }
                results.append(
                    {
                        "id": msg["id"],
                        "thread_id": msg.get("threadId"),
                        "from": headers.get("From", ""),
                        "to": headers.get("To", ""),
                        "subject": headers.get("Subject", "(No subject)"),
                        "date": headers.get("Date", ""),
                        "snippet": msg.get("snippet", ""),
                        "unread": "UNREAD" in msg.get("labelIds", []),
                    }
                )

            display_lines = [f"Search results for: {query}", ""]
            for r in results:
                marker = "*" if r["unread"] else " "
                display_lines.append(f"{marker} {r['date']}: {r['from']}")
                display_lines.append(f"  {r['subject']}")

            return ToolResult(
                success=True,
                output={"results": results, "count": len(results), "query": query},
                display_output="\n".join(display_lines),
            )

        except Exception as e:
            logger.exception("Gmail search error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Gmail API error: {e}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )


def _extract_body(payload: dict[str, Any]) -> str:
    """Extract plain text body from a Gmail message payload, handling multipart."""
    import base64

    mime_type = payload.get("mimeType", "")

    # Simple text part
    if mime_type == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode(
            "utf-8", errors="replace"
        )

    # Multipart — recurse into parts, prefer text/plain
    parts = payload.get("parts", [])
    plain_text = ""
    html_text = ""

    for part in parts:
        part_mime = part.get("mimeType", "")
        if part_mime == "text/plain" and part.get("body", {}).get("data"):
            plain_text += base64.urlsafe_b64decode(part["body"]["data"]).decode(
                "utf-8", errors="replace"
            )
        elif part_mime == "text/html" and part.get("body", {}).get("data"):
            html_text += base64.urlsafe_b64decode(part["body"]["data"]).decode(
                "utf-8", errors="replace"
            )
        elif part_mime.startswith("multipart/"):
            # Nested multipart
            nested = _extract_body(part)
            if nested:
                plain_text += nested

    if plain_text:
        return plain_text
    if html_text:
        # Strip HTML tags as a fallback
        import re

        clean = re.sub(r"<[^>]+>", " ", html_text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    return ""


def _extract_attachments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract attachment metadata from a Gmail message payload."""
    attachments = []

    def _walk(part: dict[str, Any]) -> None:
        filename = part.get("filename")
        if filename:
            attachments.append(
                {
                    "filename": filename,
                    "mime_type": part.get("mimeType", ""),
                    "size": part.get("body", {}).get("size", 0),
                    "attachment_id": part.get("body", {}).get("attachmentId"),
                }
            )
        for sub in part.get("parts", []):
            _walk(sub)

    _walk(payload)
    return attachments


class IngestEmailsHandler(ToolHandler):
    """B.2: Fetch recent emails, store as episodic memories, and extract contacts.

    Composite tool that orchestrates:
    1. Gmail API fetch of recent emails
    2. Store each email as an episodic memory with source_attribution
    3. Upsert sender contacts into CRM
    """

    def __init__(
        self,
        credentials_resolver: Callable[[], dict[str, Any] | None] | None = None,
    ):
        self._credentials_resolver = credentials_resolver

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ingest_emails",
            description=(
                "Fetch recent emails from Gmail, store as episodic memories, and "
                "extract sender contacts into CRM. Use as a daily ingestion job."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of emails to ingest",
                    },
                    "label": {
                        "type": "string",
                        "default": "INBOX",
                        "description": "Gmail label to ingest from",
                    },
                    "unread_only": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only ingest unread emails",
                    },
                },
            },
            category=ToolCategory.EMAIL,
            energy_cost=5,
            is_read_only=False,
            optional=True,
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        credentials = None
        if self._credentials_resolver:
            credentials = self._credentials_resolver()

        if not credentials:
            return await _gmail_setup_required_result(context)

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="google-api-python-client not installed",
                error_type=ToolErrorType.MISSING_DEPENDENCY,
            )

        import json
        import re

        max_results = arguments.get("max_results", 20)
        label = arguments.get("label", "INBOX")
        unread_only = arguments.get("unread_only", True)
        pool = context.registry.pool

        try:
            creds = Credentials.from_authorized_user_info(credentials)
            service = build("gmail", "v1", credentials=creds)

            # 1. List recent messages
            label_ids = [label]
            if unread_only:
                label_ids.append("UNREAD")

            resp = (
                service.users()
                .messages()
                .list(userId="me", labelIds=label_ids, maxResults=max_results)
                .execute()
            )

            messages = resp.get("messages", [])
            if not messages:
                return ToolResult.success_result(
                    {
                        "emails_ingested": 0,
                        "contacts_created": 0,
                        "contacts_updated": 0,
                    },
                    display_output="No new emails to ingest",
                )

            ingested = 0
            contacts_created = 0
            contacts_updated = 0
            skipped = 0

            async with pool.acquire() as conn:
                for msg_stub in messages:
                    try:
                        # 2. Fetch full message
                        msg = (
                            service.users()
                            .messages()
                            .get(userId="me", id=msg_stub["id"], format="full")
                            .execute()
                        )
                        headers = {
                            h["name"]: h["value"]
                            for h in msg.get("payload", {}).get("headers", [])
                        }
                        sender = headers.get("From", "")
                        subject = headers.get("Subject", "(No subject)")
                        date = headers.get("Date", "")
                        to = headers.get("To", "")

                        # Extract body
                        body = _extract_body(msg.get("payload", {}))
                        snippet = body[:500] if body else msg.get("snippet", "")

                        # Check for duplicate via content hash
                        import hashlib

                        content_hash = hashlib.sha256(
                            f"{msg_stub['id']}:{subject}".encode()
                        ).hexdigest()

                        existing = await conn.fetchval(
                            "SELECT id FROM memories WHERE source_attribution->>'content_hash' = $1",
                            content_hash,
                        )
                        if existing:
                            skipped += 1
                            continue

                        # 3. Store as episodic memory
                        content = f"Email from {sender}: {subject}\n\n{snippet}"
                        source_attr = json.dumps(
                            {
                                "kind": "email",
                                "sender": sender,
                                "to": to,
                                "subject": subject,
                                "date": date,
                                "gmail_id": msg_stub["id"],
                                "content_hash": content_hash,
                            }
                        )

                        await conn.fetchval(
                            "SELECT create_episodic_memory($1, NULL, $2, NULL, 0.0, CURRENT_TIMESTAMP, 0.5, $3)",
                            content,
                            json.dumps({"type": "email_ingestion", "label": label}),
                            source_attr,
                        )
                        ingested += 1

                        # 4. Extract and upsert contact
                        match = re.match(r"(.+?)\s*<(.+?)>", sender)
                        if match:
                            name, email = (
                                match.group(1).strip().strip('"'),
                                match.group(2).strip(),
                            )
                        elif "@" in sender:
                            email = sender.strip()
                            name = email.split("@")[0]
                        else:
                            continue

                        upsert_raw = await conn.fetchval(
                            "SELECT upsert_contact($1, $2, 'email')", name, email
                        )
                        upsert = (
                            json.loads(upsert_raw)
                            if isinstance(upsert_raw, str)
                            else (upsert_raw or {})
                        )
                        if upsert.get("created"):
                            contacts_created += 1
                        elif "id" in upsert:
                            contacts_updated += 1

                    except Exception as e:
                        logger.debug(
                            "Failed to ingest email %s: %s", msg_stub.get("id"), e
                        )
                        continue

            summary = {
                "emails_ingested": ingested,
                "emails_skipped": skipped,
                "contacts_created": contacts_created,
                "contacts_updated": contacts_updated,
                "total_fetched": len(messages),
            }

            return ToolResult.success_result(
                summary,
                display_output=(
                    f"Email ingestion complete: {ingested} emails stored, "
                    f"{skipped} duplicates skipped, "
                    f"{contacts_created} new contacts, {contacts_updated} contacts updated"
                ),
            )

        except Exception as e:
            logger.exception("Email ingestion error")
            return ToolResult(
                success=False,
                output=None,
                error=f"Email ingestion failed: {e}",
                error_type=ToolErrorType.EXECUTION_FAILED,
            )


def create_email_tools(
    smtp_config_resolver: Callable[[], dict[str, Any] | None] | None = None,
    sendgrid_api_key_resolver: Callable[[], str | None] | None = None,
    sendgrid_from_email: str | None = None,
    gmail_credentials_resolver: Callable[[], dict[str, Any] | None] | None = None,
) -> list[ToolHandler]:
    """
    Create email tool handlers.

    Args:
        smtp_config_resolver: Callable that returns SMTP configuration dict.
        sendgrid_api_key_resolver: Callable that returns SendGrid API key.
        sendgrid_from_email: Default sender email for SendGrid.
        gmail_credentials_resolver: Callable that returns Google OAuth credentials dict
            for Gmail API read access.

    Returns:
        List of email tool handlers.
    """
    tools: list[ToolHandler] = [EmailSendHandler(smtp_config_resolver)]

    # Only add SendGrid if API key resolver is provided
    if sendgrid_api_key_resolver:
        tools.append(
            SendGridEmailHandler(sendgrid_api_key_resolver, sendgrid_from_email)
        )

    # Gmail read tools (always registered; auth checked at execution time)
    tools.append(EmailListHandler(gmail_credentials_resolver))
    tools.append(EmailReadHandler(gmail_credentials_resolver))
    tools.append(EmailSearchHandler(gmail_credentials_resolver))

    # B.2: Email ingestion pipeline tool
    tools.append(IngestEmailsHandler(gmail_credentials_resolver))

    return tools
