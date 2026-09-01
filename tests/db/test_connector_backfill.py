from __future__ import annotations

import json

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


async def _connected_gmail(conn, marker: str) -> str:
    account = f"backfill-{marker}@example.com"
    attempt = _j(await conn.fetchval(
        """
        SELECT start_connection_attempt(
            'gmail',
            '["read", "search", "ingest"]'::jsonb,
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
            '[]'::jsonb,
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
    return account


async def test_connector_backfill_job_cursor_lifecycle(db_pool):
    marker = get_test_identifier("connector-backfill")

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            account = await _connected_gmail(conn, marker)
            job = _j(await conn.fetchval(
                """
                SELECT enqueue_connector_backfill_job(
                    'gmail', $1, 'messages',
                    '{"from": "now-30d"}'::jsonb,
                    '{"reason": "test"}'::jsonb
                )
                """,
                account,
            ))
            duplicate = _j(await conn.fetchval(
                "SELECT enqueue_connector_backfill_job('gmail', $1, 'messages')",
                account,
            ))
            assert duplicate["existing"] is True
            assert duplicate["job_id"] == job["job_id"]

            claimed = _j(await conn.fetchval("SELECT claim_connector_backfill_jobs_for('gmail', 5)"))
            assert [item["id"] for item in claimed] == [job["job_id"]]

            progress = _j(await conn.fetchval(
                """
                SELECT update_connector_backfill_progress(
                    $1::uuid,
                    '{"pages": 1, "items_seen": 2}'::jsonb,
                    '{"page_token": "abc"}'::jsonb,
                    CURRENT_TIMESTAMP
                )
                """,
                job["job_id"],
            ))
            assert progress["status"] == "in_progress"
            assert progress["cancel_requested"] is False

            completed = _j(await conn.fetchval(
                """
                SELECT complete_connector_backfill_job(
                    $1::uuid,
                    '{"items_seen": 2}'::jsonb,
                    '{"page_token": "done"}'::jsonb,
                    CURRENT_TIMESTAMP
                )
                """,
                job["job_id"],
            ))
            status = _j(await conn.fetchval(
                "SELECT get_connector_backfill_status('gmail', $1)",
                account,
            ))
        finally:
            await tr.rollback()

    assert completed["status"] == "completed"
    assert status["cursors"][0]["cursor_value"] == {"page_token": "done"}
    assert status["jobs"][0]["status"] == "completed"


async def test_enqueue_gmail_backfill_creates_ongoing_sync_responsibility(db_pool):
    """#110: a Gmail backfill previously had no ongoing-sync mechanism --
    the ambient responsibility evaluator was fully implemented but nothing
    ever created a row for it to evaluate. Enqueuing a backfill now ensures
    one exists, idempotently."""
    marker = get_test_identifier("connector-ambient")

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            account = await _connected_gmail(conn, marker)
            await conn.fetchval(
                "SELECT enqueue_connector_backfill_job('gmail', $1, 'messages')",
                account,
            )

            rows = await conn.fetch(
                """
                SELECT id, status, sources FROM ambient_responsibilities
                WHERE sources @> jsonb_build_array(
                    jsonb_build_object('connector_id', 'gmail', 'account_key', $1::text)
                )
                """,
                account.lower(),
            )
            assert len(rows) == 1
            assert rows[0]["status"] in ("active", "proposed", "blocked")

            # A second backfill for the same account must not duplicate it.
            await conn.fetchval(
                "SELECT enqueue_connector_backfill_job('gmail', $1, 'messages')",
                account,
            )
            rows_again = await conn.fetch(
                """
                SELECT id FROM ambient_responsibilities
                WHERE sources @> jsonb_build_array(
                    jsonb_build_object('connector_id', 'gmail', 'account_key', $1::text)
                )
                """,
                account.lower(),
            )
            assert len(rows_again) == 1
            assert rows_again[0]["id"] == rows[0]["id"]
        finally:
            await tr.rollback()


async def test_completed_truncated_job_reenqueues_successor(db_pool):
    """#110: a capped backfill that hit max_messages and still had a
    page_token left previously completed and just sat there forever.
    Completing with result.truncated=true must queue a successor so the
    next maintenance tick continues from the advanced cursor."""
    marker = get_test_identifier("connector-truncated")

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            account = await _connected_gmail(conn, marker)
            job = _j(await conn.fetchval(
                "SELECT enqueue_connector_backfill_job('gmail', $1, 'messages')",
                account,
            ))
            await conn.fetchval("SELECT claim_connector_backfill_jobs_for('gmail', 5)")

            completed = _j(await conn.fetchval(
                """
                SELECT complete_connector_backfill_job(
                    $1::uuid,
                    '{"items_seen": 100, "truncated": true, "next_page_token": "page-2"}'::jsonb,
                    '{"page_token": "page-2"}'::jsonb,
                    CURRENT_TIMESTAMP
                )
                """,
                job["job_id"],
            ))
            assert completed["status"] == "completed"

            status = _j(await conn.fetchval(
                "SELECT get_connector_backfill_status('gmail', $1)", account
            ))
            statuses = {j["job_id"]: j["status"] for j in status["jobs"]}
            assert statuses[job["job_id"]] == "completed"
            successors = [
                jid for jid, st in statuses.items()
                if jid != job["job_id"] and st == "pending"
            ]
            assert len(successors) == 1, statuses
            assert status["cursors"][0]["cursor_value"] == {"page_token": "page-2"}
        finally:
            await tr.rollback()


async def test_completed_non_truncated_job_has_no_successor(db_pool):
    marker = get_test_identifier("connector-finished")

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            account = await _connected_gmail(conn, marker)
            job = _j(await conn.fetchval(
                "SELECT enqueue_connector_backfill_job('gmail', $1, 'messages')",
                account,
            ))
            await conn.fetchval("SELECT claim_connector_backfill_jobs_for('gmail', 5)")
            await conn.fetchval(
                """
                SELECT complete_connector_backfill_job(
                    $1::uuid,
                    '{"items_seen": 3, "truncated": false}'::jsonb,
                    '{"page_token": null}'::jsonb,
                    CURRENT_TIMESTAMP
                )
                """,
                job["job_id"],
            )

            status = _j(await conn.fetchval(
                "SELECT get_connector_backfill_status('gmail', $1)", account
            ))
            assert len(status["jobs"]) == 1
        finally:
            await tr.rollback()


async def test_connector_backfill_pause_resume_and_cancel(db_pool):
    marker = get_test_identifier("connector-pause")

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            account = await _connected_gmail(conn, marker)
            job = _j(await conn.fetchval(
                "SELECT enqueue_connector_backfill_job('gmail', $1, 'messages')",
                account,
            ))
            paused = _j(await conn.fetchval(
                "SELECT pause_connector_backfill_job($1::uuid, 'operator pause')",
                job["job_id"],
            ))
            claimed_while_paused = _j(await conn.fetchval("SELECT claim_connector_backfill_jobs(5)"))
            resumed = _j(await conn.fetchval(
                "SELECT resume_connector_backfill_job($1::uuid)",
                job["job_id"],
            ))
            cancelled = _j(await conn.fetchval(
                "SELECT cancel_connector_backfill_job($1::uuid, 'operator cancel')",
                job["job_id"],
            ))
        finally:
            await tr.rollback()

    assert paused["status"] == "paused"
    assert claimed_while_paused == []
    assert resumed["status"] == "pending"
    assert cancelled["status"] == "cancelled"


async def test_connector_source_item_preserves_raw_artifact_and_ingestion_receipt(db_pool):
    marker = get_test_identifier("connector-source")
    content = (
        f"Subject: Source receipt {marker}\n\n"
        "This email contains the heliotrope connector backfill clause."
    )

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            account = await _connected_gmail(conn, marker)
            item = _j(await conn.fetchval(
                """
                SELECT upsert_connector_source_item(
                    'gmail',
                    $1,
                    $2,
                    'Source receipt',
                    $3,
                    'message',
                    'thread-1',
                    CURRENT_TIMESTAMP,
                    ARRAY['INBOX']::text[],
                    '[{"email": "eric@example.com", "role": "to"}]'::jsonb,
                    '[{"filename": "note.txt", "size": 12}]'::jsonb,
                    '{"gmail_history_id": "42"}'::jsonb,
                    'private',
                    TRUE
                )
                """,
                account,
                f"msg-{marker}",
                content,
            ))
            opened = _j(await conn.fetchval(
                "SELECT open_source_document($1::uuid)",
                item["document_id"],
            ))
            second = _j(await conn.fetchval(
                """
                SELECT upsert_connector_source_item(
                    'gmail', $1, $2, 'Source receipt', $3,
                    'message', 'thread-1', CURRENT_TIMESTAMP,
                    ARRAY['INBOX']::text[], '[]'::jsonb, '[]'::jsonb,
                    '{"seen_again": true}'::jsonb, 'private', TRUE
                )
                """,
                account,
                f"msg-{marker}",
                content,
            ))
            row = await conn.fetchrow(
                """
                SELECT provider_item_id, source_document_id, ingestion_job_id, sensitivity, raw_metadata
                FROM connector_source_items
                WHERE id = $1::uuid
                """,
                item["source_item_id"],
            )
            job = _j(await conn.fetchval(
                "SELECT get_ingestion_job($1::uuid)",
                item["ingestion_job_id"],
            ))
        finally:
            await tr.rollback()

    assert opened["content"] == content
    assert opened["source_attribution"]["connector_id"] == "gmail"
    assert opened["source_attribution"]["provider_item_id"] == f"msg-{marker}"
    assert opened["source_attribution"]["sensitivity"] == "private"
    assert second["source_item_id"] == item["source_item_id"]
    assert second["ingestion_job_id"] == item["ingestion_job_id"]
    assert row["provider_item_id"] == f"msg-{marker}"
    assert row["sensitivity"] == "private"
    assert _j(row["raw_metadata"])["seen_again"] is True
    assert job["payload"]["provider_item_id"] == f"msg-{marker}"
    assert job["payload"]["acquisition"] == "connector"
    assert "content" not in job
