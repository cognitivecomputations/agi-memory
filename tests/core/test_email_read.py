"""Tests for Gmail read tools (email_list, email_read, email_search)."""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from core.tools.base import ToolCategory, ToolContext, ToolErrorType, ToolExecutionContext
from core.tools.email import (
    EmailListHandler,
    EmailReadHandler,
    EmailSearchHandler,
    _extract_body,
    _extract_attachments,
    create_email_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_credentials():
    return {
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "refresh_token": "test-refresh-token",
        "token": "test-token",
        "token_uri": "https://oauth2.googleapis.com/token",
    }


def _make_context():
    registry = MagicMock()
    registry.pool = MagicMock()
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="test-call",
        registry=registry,
    )


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


@pytest.fixture()
def mock_google_modules():
    """Install fake google.* modules so lazy imports succeed, then clean up."""
    mock_creds_class = MagicMock()
    mock_build = MagicMock()

    # Build module tree
    google = ModuleType("google")
    google_oauth2 = ModuleType("google.oauth2")
    google_oauth2_creds = ModuleType("google.oauth2.credentials")
    google_oauth2_creds.Credentials = mock_creds_class

    googleapiclient = ModuleType("googleapiclient")
    googleapiclient_discovery = ModuleType("googleapiclient.discovery")
    googleapiclient_discovery.build = mock_build

    google.oauth2 = google_oauth2
    google_oauth2.credentials = google_oauth2_creds

    mods = {
        "google": google,
        "google.oauth2": google_oauth2,
        "google.oauth2.credentials": google_oauth2_creds,
        "googleapiclient": googleapiclient,
        "googleapiclient.discovery": googleapiclient_discovery,
    }

    saved = {}
    for name in mods:
        if name in sys.modules:
            saved[name] = sys.modules[name]

    sys.modules.update(mods)
    yield mock_creds_class, mock_build

    # Restore
    for name in mods:
        if name in saved:
            sys.modules[name] = saved[name]
        else:
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Spec tests
# ---------------------------------------------------------------------------

class TestEmailListSpec:
    def test_spec_fields(self):
        handler = EmailListHandler()
        spec = handler.spec
        assert spec.name == "email_list"
        assert spec.category == ToolCategory.EMAIL
        assert spec.is_read_only is True
        assert spec.energy_cost == 2
        assert spec.optional is True
        assert ToolContext.HEARTBEAT in spec.allowed_contexts

    def test_spec_parameters(self):
        spec = EmailListHandler().spec
        props = spec.parameters["properties"]
        assert "label" in props
        assert "max_results" in props
        assert "unread_only" in props


class TestEmailReadSpec:
    def test_spec_fields(self):
        handler = EmailReadHandler()
        spec = handler.spec
        assert spec.name == "email_read"
        assert spec.category == ToolCategory.EMAIL
        assert spec.is_read_only is True
        assert "message_id" in spec.parameters["required"]
        assert ToolContext.HEARTBEAT in spec.allowed_contexts

    def test_spec_parameters(self):
        spec = EmailReadHandler().spec
        props = spec.parameters["properties"]
        assert "message_id" in props
        assert "mark_read" in props


class TestEmailSearchSpec:
    def test_spec_fields(self):
        handler = EmailSearchHandler()
        spec = handler.spec
        assert spec.name == "email_search"
        assert spec.category == ToolCategory.EMAIL
        assert spec.is_read_only is True
        assert "query" in spec.parameters["required"]
        assert ToolContext.HEARTBEAT in spec.allowed_contexts


# ---------------------------------------------------------------------------
# Auth failure tests
# ---------------------------------------------------------------------------

class TestAuthFailures:
    @pytest.mark.asyncio
    async def test_list_no_credentials(self):
        handler = EmailListHandler()
        result = await handler.execute({}, _make_context())
        assert not result.success
        assert result.error_type == ToolErrorType.AUTH_FAILED
        assert result.output["ui"]["kind"] == "connector_setup"
        assert result.output["ui"]["connector_id"] == "gmail"

    @pytest.mark.asyncio
    async def test_read_no_credentials(self):
        handler = EmailReadHandler()
        result = await handler.execute({"message_id": "abc"}, _make_context())
        assert not result.success
        assert result.error_type == ToolErrorType.AUTH_FAILED
        assert result.output["ui"]["kind"] == "connector_setup"

    @pytest.mark.asyncio
    async def test_search_no_credentials(self):
        handler = EmailSearchHandler()
        result = await handler.execute({"query": "test"}, _make_context())
        assert not result.success
        assert result.error_type == ToolErrorType.AUTH_FAILED
        assert result.output["ui"]["kind"] == "connector_setup"

    @pytest.mark.asyncio
    async def test_list_resolver_returns_none(self):
        handler = EmailListHandler(credentials_resolver=lambda: None)
        result = await handler.execute({}, _make_context())
        assert not result.success
        assert result.error_type == ToolErrorType.AUTH_FAILED
        assert result.output["ui"]["kind"] == "connector_setup"


# ---------------------------------------------------------------------------
# Body extraction tests
# ---------------------------------------------------------------------------

class TestExtractBody:
    def test_plain_text(self):
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _b64("Hello world")},
        }
        assert _extract_body(payload) == "Hello world"

    def test_multipart_prefers_plain(self):
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64("Plain text")},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": _b64("<b>HTML</b>")},
                },
            ],
        }
        assert _extract_body(payload) == "Plain text"

    def test_html_fallback(self):
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/html",
                    "body": {"data": _b64("<p>Hello <b>world</b></p>")},
                },
            ],
        }
        body = _extract_body(payload)
        assert "Hello" in body
        assert "world" in body
        assert "<p>" not in body  # HTML tags stripped

    def test_empty_payload(self):
        assert _extract_body({}) == ""

    def test_nested_multipart(self):
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {"data": _b64("Nested plain")},
                        },
                    ],
                },
            ],
        }
        assert _extract_body(payload) == "Nested plain"


# ---------------------------------------------------------------------------
# Attachment extraction tests
# ---------------------------------------------------------------------------

class TestExtractAttachments:
    def test_no_attachments(self):
        payload = {"mimeType": "text/plain", "body": {"data": _b64("Hi")}}
        assert _extract_attachments(payload) == []

    def test_single_attachment(self):
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64("body")}},
                {
                    "filename": "report.pdf",
                    "mimeType": "application/pdf",
                    "body": {"size": 12345, "attachmentId": "att-1"},
                },
            ],
        }
        atts = _extract_attachments(payload)
        assert len(atts) == 1
        assert atts[0]["filename"] == "report.pdf"
        assert atts[0]["size"] == 12345
        assert atts[0]["attachment_id"] == "att-1"

    def test_nested_attachment(self):
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {
                            "filename": "image.png",
                            "mimeType": "image/png",
                            "body": {"size": 5000, "attachmentId": "att-2"},
                        },
                    ],
                },
            ],
        }
        atts = _extract_attachments(payload)
        assert len(atts) == 1
        assert atts[0]["filename"] == "image.png"


# ---------------------------------------------------------------------------
# Gmail API mock tests
# ---------------------------------------------------------------------------

def _mock_gmail_service(mock_build, messages_list_return=None, messages_get_return=None):
    """Configure mock_build to return a Gmail service with canned responses."""
    service = MagicMock()
    users = MagicMock()
    messages = MagicMock()

    service.users.return_value = users
    users.messages.return_value = messages

    # .list()
    list_req = MagicMock()
    messages.list.return_value = list_req
    list_req.execute.return_value = messages_list_return or {"messages": []}

    # .get()
    get_req = MagicMock()
    messages.get.return_value = get_req
    get_req.execute.return_value = messages_get_return or {}

    # .modify()
    modify_req = MagicMock()
    messages.modify.return_value = modify_req
    modify_req.execute.return_value = {}

    mock_build.return_value = service
    return service


class TestEmailListExecution:
    @pytest.mark.asyncio
    async def test_list_empty_inbox(self, mock_google_modules):
        mock_creds_class, mock_build = mock_google_modules
        _mock_gmail_service(mock_build, messages_list_return={"messages": []})
        handler = EmailListHandler(credentials_resolver=_fake_credentials)

        result = await handler.execute({}, _make_context())

        assert result.success
        assert result.output["count"] == 0

    @pytest.mark.asyncio
    async def test_list_with_messages(self, mock_google_modules):
        mock_creds_class, mock_build = mock_google_modules
        _mock_gmail_service(
            mock_build,
            messages_list_return={"messages": [{"id": "msg-1"}, {"id": "msg-2"}]},
            messages_get_return={
                "id": "msg-1",
                "threadId": "thread-1",
                "snippet": "Test snippet",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "alice@test.com"},
                        {"name": "Subject", "value": "Test Subject"},
                        {"name": "Date", "value": "Thu, 13 Feb 2026 10:00:00 +0000"},
                        {"name": "To", "value": "bob@test.com"},
                    ]
                },
            },
        )
        handler = EmailListHandler(credentials_resolver=_fake_credentials)

        result = await handler.execute({"max_results": 5}, _make_context())

        assert result.success
        assert result.output["count"] == 2
        assert result.output["emails"][0]["from"] == "alice@test.com"
        assert result.output["emails"][0]["unread"] is True

    @pytest.mark.asyncio
    async def test_list_unread_only(self, mock_google_modules):
        mock_creds_class, mock_build = mock_google_modules
        service = _mock_gmail_service(mock_build, messages_list_return={"messages": []})
        handler = EmailListHandler(credentials_resolver=_fake_credentials)

        await handler.execute({"unread_only": True}, _make_context())

        # Verify UNREAD was included in label_ids
        call_kwargs = service.users().messages().list.call_args
        assert "UNREAD" in call_kwargs.kwargs.get("labelIds", call_kwargs[1].get("labelIds", []))


class TestEmailReadExecution:
    @pytest.mark.asyncio
    async def test_read_message(self, mock_google_modules):
        mock_creds_class, mock_build = mock_google_modules
        _mock_gmail_service(
            mock_build,
            messages_get_return={
                "id": "msg-123",
                "threadId": "thread-1",
                "labelIds": ["INBOX"],
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "From", "value": "alice@test.com"},
                        {"name": "Subject", "value": "Hello"},
                        {"name": "Date", "value": "Thu, 13 Feb 2026 10:00:00 +0000"},
                        {"name": "To", "value": "bob@test.com"},
                    ],
                    "body": {"data": _b64("Hello Bob, how are you?")},
                },
            },
        )
        handler = EmailReadHandler(credentials_resolver=_fake_credentials)

        result = await handler.execute({"message_id": "msg-123"}, _make_context())

        assert result.success
        assert result.output["id"] == "msg-123"
        assert result.output["from"] == "alice@test.com"
        assert "Hello Bob" in result.output["body"]
        assert result.output["subject"] == "Hello"

    @pytest.mark.asyncio
    async def test_read_with_mark_read(self, mock_google_modules):
        mock_creds_class, mock_build = mock_google_modules
        service = _mock_gmail_service(
            mock_build,
            messages_get_return={
                "id": "msg-123",
                "threadId": "thread-1",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "From", "value": "a@b.com"},
                        {"name": "Subject", "value": "X"},
                        {"name": "Date", "value": "Thu, 13 Feb 2026"},
                        {"name": "To", "value": "c@d.com"},
                    ],
                    "body": {"data": _b64("body")},
                },
            },
        )
        handler = EmailReadHandler(credentials_resolver=_fake_credentials)

        result = await handler.execute(
            {"message_id": "msg-123", "mark_read": True}, _make_context()
        )

        assert result.success
        # Verify modify was called to remove UNREAD
        service.users().messages().modify.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_with_attachments(self, mock_google_modules):
        mock_creds_class, mock_build = mock_google_modules
        _mock_gmail_service(
            mock_build,
            messages_get_return={
                "id": "msg-att",
                "threadId": "thread-2",
                "labelIds": ["INBOX"],
                "payload": {
                    "mimeType": "multipart/mixed",
                    "headers": [
                        {"name": "From", "value": "sender@test.com"},
                        {"name": "Subject", "value": "With attachment"},
                        {"name": "Date", "value": "Thu, 13 Feb 2026"},
                        {"name": "To", "value": "me@test.com"},
                    ],
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {"data": _b64("See attached")},
                        },
                        {
                            "filename": "doc.pdf",
                            "mimeType": "application/pdf",
                            "body": {"size": 9999, "attachmentId": "att-99"},
                        },
                    ],
                },
            },
        )
        handler = EmailReadHandler(credentials_resolver=_fake_credentials)

        result = await handler.execute({"message_id": "msg-att"}, _make_context())

        assert result.success
        assert len(result.output["attachments"]) == 1
        assert result.output["attachments"][0]["filename"] == "doc.pdf"


class TestEmailSearchExecution:
    @pytest.mark.asyncio
    async def test_search_no_results(self, mock_google_modules):
        mock_creds_class, mock_build = mock_google_modules
        _mock_gmail_service(mock_build, messages_list_return={})
        handler = EmailSearchHandler(credentials_resolver=_fake_credentials)

        result = await handler.execute({"query": "from:nobody@nowhere.com"}, _make_context())

        assert result.success
        assert result.output["count"] == 0

    @pytest.mark.asyncio
    async def test_search_with_results(self, mock_google_modules):
        mock_creds_class, mock_build = mock_google_modules
        _mock_gmail_service(
            mock_build,
            messages_list_return={"messages": [{"id": "msg-42"}]},
            messages_get_return={
                "id": "msg-42",
                "threadId": "thread-9",
                "snippet": "Found it",
                "labelIds": ["INBOX"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "found@test.com"},
                        {"name": "Subject", "value": "Match"},
                        {"name": "Date", "value": "Thu, 13 Feb 2026"},
                        {"name": "To", "value": "me@test.com"},
                    ]
                },
            },
        )
        handler = EmailSearchHandler(credentials_resolver=_fake_credentials)

        result = await handler.execute({"query": "subject:Match"}, _make_context())

        assert result.success
        assert result.output["count"] == 1
        assert result.output["results"][0]["subject"] == "Match"
        assert result.output["query"] == "subject:Match"

    @pytest.mark.asyncio
    async def test_search_passes_query_to_api(self, mock_google_modules):
        mock_creds_class, mock_build = mock_google_modules
        service = _mock_gmail_service(mock_build, messages_list_return={})
        handler = EmailSearchHandler(credentials_resolver=_fake_credentials)

        await handler.execute({"query": "from:alice has:attachment"}, _make_context())

        call_kwargs = service.users().messages().list.call_args
        assert call_kwargs.kwargs.get("q", call_kwargs[1].get("q")) == "from:alice has:attachment"


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestCreateEmailTools:
    def test_includes_gmail_read_tools(self):
        tools = create_email_tools()
        names = {t.spec.name for t in tools}
        assert "email_list" in names
        assert "email_read" in names
        assert "email_search" in names

    def test_includes_send_tools(self):
        tools = create_email_tools()
        names = {t.spec.name for t in tools}
        assert "email_send" in names

    def test_sendgrid_only_with_resolver(self):
        tools = create_email_tools()
        names = {t.spec.name for t in tools}
        assert "email_send_sendgrid" not in names

        tools_with_sg = create_email_tools(sendgrid_api_key_resolver=lambda: "key")
        names_sg = {t.spec.name for t in tools_with_sg}
        assert "email_send_sendgrid" in names_sg

    def test_total_tool_count(self):
        # 1 SMTP + 3 Gmail read + 1 ingest = 5
        tools = create_email_tools()
        assert len(tools) == 5

        # 1 SMTP + 1 SendGrid + 3 Gmail read + 1 ingest = 6
        tools_full = create_email_tools(sendgrid_api_key_resolver=lambda: "key")
        assert len(tools_full) == 6
