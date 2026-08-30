from __future__ import annotations

import base64
import json

import pytest

from core.auth.google_gmail import GMAIL_DEFAULT_CREDENTIAL_REF
from core.auth.store import save_auth
from core.auth.utils import now_ms
from services import gmail_backfill
from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


def _b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _message(message_id: str, *, subject: str, body: str, internal_ms: str = "1710000000000"):
    return {
        "id": message_id,
        "threadId": f"thread-{message_id}",
        "historyId": f"history-{message_id}",
        "internalDate": internal_ms,
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": f"Snippet for {subject}",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": "Alice <alice@example.com>"},
                {"name": "To", "value": "Eric <eric@example.com>"},
                {"name": "Date", "value": "Sat, 9 Mar 2024 12:00:00 +0000"},
            ],
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64(body)}},
                {
                    "filename": "note.txt",
                    "mimeType": "text/plain",
                    "body": {"attachmentId": f"att-{message_id}", "size": 42},
                },
            ],
        },
    }


async def _connected_gmail(pool, marker: str, account: str) -> None:
    async with pool.acquire() as conn:
        attempt = _j(await conn.fetchval(
            """
            SELECT start_connection_attempt(
                'gmail',
                '["read", "search"]'::jsonb,
                ARRAY[]::text[],
                '{}'::jsonb,
                NULL,
                NULL,
                'test',
                $1,
                CURRENT_TIMESTAMP + INTERVAL '10 minutes'
            )
            """,
            marker,
        ))
        await conn.fetchval(
            """
            SELECT complete_connection_attempt(
                $1::uuid,
                $2,
                $2,
                'integration.gmail.default',
                $3::text[],
                '["read", "search"]'::jsonb,
                '{"test": true}'::jsonb
            )
            """,
            attempt["attempt_id"],
            account,
            [
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/gmail.readonly",
            ],
        )


async def test_gmail_message_parser_preserves_full_body_and_metadata():
    source = gmail_backfill.gmail_message_to_source_item(
        _message(
            "msg-parser",
            subject="Parser contract",
            body="The complete searchable clause lives here.",
        )
    )

    assert source["provider_item_id"] == "msg-parser"
    assert source["title"] == "Parser contract"
    assert "The complete searchable clause lives here." in source["content"]
    assert source["labels"] == ["INBOX", "UNREAD"]
    assert source["participants"][0]["role"] == "from"
    assert source["attachments"][0]["filename"] == "note.txt"
    assert source["metadata"]["gmail_history_id"] == "history-msg-parser"


async def test_run_gmail_backfill_step_stores_source_documents_and_ingestion_jobs(
    db_pool, monkeypatch, tmp_path
):
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    marker = get_test_identifier("gmail-backfill")
    account = f"eric-{marker}@example.com"
    save_auth(
        GMAIL_DEFAULT_CREDENTIAL_REF,
        {
            "type": "authorized_user",
            "token": "access-token",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client",
            "client_secret": "secret",
            "expires_ms": now_ms() + 3_600_000,
            "account_email": account,
            "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
        },
    )
    await _connected_gmail(db_pool, marker, account)

    async with db_pool.acquire() as conn:
        job = _j(await conn.fetchval(
            """
            SELECT enqueue_connector_backfill_job(
                'gmail',
                $1,
                'messages',
                '{"label_ids": ["INBOX"], "max_messages": 2, "page_size": 2}'::jsonb,
                '{"test": true}'::jsonb
            )
            """,
            account,
        ))

    async def fake_gmail_get(credentials, path, *, params=None):
        if path == "/users/me/messages":
            assert params["labelIds"] == ["INBOX"]
            assert params["maxResults"] == 2
            return {"messages": [{"id": "msg-1"}, {"id": "msg-2"}], "nextPageToken": "next-page"}
        if path == "/users/me/messages/msg-1":
            return _message("msg-1", subject="First", body="First full email body.")
        if path == "/users/me/messages/msg-2":
            return _message("msg-2", subject="Second", body="Second full email body.")
        raise AssertionError(f"Unexpected Gmail path: {path}")

    monkeypatch.setattr(gmail_backfill, "_gmail_get", fake_gmail_get)
    handled = await gmail_backfill.run_gmail_backfill_step(db_pool, limit=1)

    async with db_pool.acquire() as conn:
        status = _j(await conn.fetchval("SELECT get_connector_backfill_status('gmail', $1)", account))
        items = await conn.fetch(
            """
            SELECT provider_item_id, source_document_id, ingestion_job_id, raw_metadata
            FROM connector_source_items
            WHERE account_key = $1
            ORDER BY provider_item_id
            """,
            account,
        )
        opened = _j(await conn.fetchval(
            "SELECT open_source_document($1::uuid)",
            items[0]["source_document_id"],
        ))

    assert handled == 1
    jobs_by_id = {j["job_id"]: j for j in status["jobs"]}
    original = jobs_by_id[job["job_id"]]
    assert original["status"] == "completed"
    assert original["result"]["items_stored"] == 2
    assert original["result"]["truncated"] is True
    assert status["cursors"][0]["cursor_value"]["page_token"] == "next-page"
    # #110: a truncated completion now self-continues -- a pending successor
    # job is queued so the next maintenance tick picks up at "next-page"
    # instead of a capped backfill silently completing forever.
    successors = [j for jid, j in jobs_by_id.items() if jid != job["job_id"]]
    assert len(successors) == 1
    assert successors[0]["status"] == "pending"
    assert [row["provider_item_id"] for row in items] == ["msg-1", "msg-2"]
    assert items[0]["ingestion_job_id"] is not None
    assert _j(items[0]["raw_metadata"])["gmail_history_id"] == "history-msg-1"
    assert "First full email body." in opened["content"]
    assert opened["source_attribution"]["connector_id"] == "gmail"
