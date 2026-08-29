"""Web inbox (db/76): the dashboard as a delivery endpoint of the outbox
queue — idempotent delivery, feed with unread count, DB-side read receipts.
"""

from __future__ import annotations

import json
import uuid

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


def _body(message: str, msg_id: str | None = None, intent: str | None = None):
    return json.dumps(
        {
            "id": msg_id or f"msg-{uuid.uuid4().hex[:10]}",
            "kind": "user",
            "payload": {"message": message, "intent": intent, "context": {}},
        }
    )


async def test_deliver_is_idempotent_by_envelope_id(db_pool):
    msg_id = f"msg-{uuid.uuid4().hex[:10]}"
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            first = await conn.fetchval(
                "SELECT web_inbox_deliver($1::jsonb)",
                _body("hello from a heartbeat", msg_id),
            )
            second = await conn.fetchval(
                "SELECT web_inbox_deliver($1::jsonb)",
                _body("hello again (redelivery)", msg_id),
            )
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM web_inbox WHERE outbox_msg_id = $1", msg_id
            )
        finally:
            await tr.rollback()
    assert first is not None
    assert second is None  # redelivery of the same envelope is a no-op
    assert count == 1


async def test_deliver_resolves_content_and_skips_empty(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            from_content = await conn.fetchval(
                "SELECT web_inbox_deliver($1::jsonb)",
                json.dumps(
                    {
                        "id": f"m-{uuid.uuid4().hex[:8]}",
                        "kind": "user",
                        "payload": {"content": "content-key text"},
                    }
                ),
            )
            empty = await conn.fetchval(
                "SELECT web_inbox_deliver($1::jsonb)",
                json.dumps(
                    {
                        "id": f"m-{uuid.uuid4().hex[:8]}",
                        "kind": "user",
                        "payload": {"context": {}},
                    }
                ),
            )
            text = await conn.fetchval(
                "SELECT message FROM web_inbox WHERE id = $1", from_content
            )
        finally:
            await tr.rollback()
    assert text == "content-key text"
    assert empty is None  # nothing user-readable, nothing to show


async def test_feed_and_read_receipts(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            first = await conn.fetchval(
                "SELECT web_inbox_deliver($1::jsonb)",
                _body("first note", intent="status"),
            )
            second = await conn.fetchval(
                "SELECT web_inbox_deliver($1::jsonb)", _body("second note")
            )
            feed = json.loads(await conn.fetchval("SELECT get_web_inbox(10)"))
            assert feed["unread"] >= 2
            newest = feed["messages"][0]
            assert newest["message"] == "second note"
            assert newest["read_at"] is None

            marked = await conn.fetchval(
                "SELECT mark_web_inbox_read(ARRAY[$1, $2]::uuid[])", first, second
            )
            assert marked == 2
            remarked = await conn.fetchval(
                "SELECT mark_web_inbox_read(ARRAY[$1]::uuid[])", first
            )
            assert remarked == 0  # receipts are recorded once
            after = json.loads(await conn.fetchval("SELECT get_web_inbox(10)"))
            assert after["unread"] == feed["unread"] - 2
        finally:
            await tr.rollback()


async def test_queue_web_inbox_message_delivers_without_channel_worker(db_pool):
    test_id = f"queue-web-{uuid.uuid4().hex[:10]}"
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            result = json.loads(
                await conn.fetchval(
                    "SELECT queue_web_inbox_message($1::text, $2::text, 'tool')",
                    f"note {test_id}",
                    "status",
                )
            )
            outbox = await conn.fetchrow(
                "SELECT status, envelope FROM outbox_messages WHERE id = $1::uuid",
                result["outbox_id"],
            )
            delivered = await conn.fetchrow(
                "SELECT message, intent FROM web_inbox WHERE id = $1::uuid",
                result["web_inbox_id"],
            )
        finally:
            await tr.rollback()

    assert result["queued"] is True
    assert result["delivered"] is True
    assert result["delivery"] == {"mode": "web_inbox"}
    assert outbox["status"] == "published"
    envelope = (
        json.loads(outbox["envelope"])
        if isinstance(outbox["envelope"], str)
        else outbox["envelope"]
    )
    assert envelope["payload"]["delivery"] == {"mode": "web_inbox"}
    assert delivered["message"] == f"note {test_id}"
    assert delivered["intent"] == "status"


async def test_outbox_consumer_tees_to_web_inbox(db_pool):
    """The channel worker's consumer delivers a copy of every user-bound
    queue body to the web endpoint — even with no chat adapters configured —
    and silent deliveries stay silent everywhere."""
    from unittest.mock import MagicMock

    from channels.outbox import ChannelOutboxConsumer

    consumer = ChannelOutboxConsumer(MagicMock(), db_pool)
    msg_id = f"msg-{uuid.uuid4().hex[:10]}"
    silent_id = f"msg-{uuid.uuid4().hex[:10]}"
    try:
        await consumer._process_message(
            {
                "id": msg_id,
                "kind": "user",
                "payload": {
                    "message": "reaching out from a heartbeat",
                    "intent": "status",
                },
            }
        )
        await consumer._process_message(
            {
                "id": silent_id,
                "kind": "user",
                "payload": {"message": "suppressed", "delivery": {"mode": "silent"}},
            }
        )
        async with db_pool.acquire() as conn:
            delivered = await conn.fetchrow(
                "SELECT message, intent FROM web_inbox WHERE outbox_msg_id = $1", msg_id
            )
            suppressed = await conn.fetchval(
                "SELECT COUNT(*) FROM web_inbox WHERE outbox_msg_id = $1", silent_id
            )
        assert delivered is not None
        assert delivered["message"] == "reaching out from a heartbeat"
        assert delivered["intent"] == "status"
        assert suppressed == 0
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM web_inbox WHERE outbox_msg_id = ANY($1::text[])",
                [msg_id, silent_id],
            )


async def test_automation_suggestion_pushes_once_after_durable_inbox_delivery(
    db_pool, monkeypatch
):
    """A redelivered queue envelope must not duplicate a phone notification."""
    from unittest.mock import AsyncMock, MagicMock

    from channels.outbox import ChannelOutboxConsumer

    push = AsyncMock(return_value=1)
    monkeypatch.setattr("services.web_push.deliver_web_push", push)
    consumer = ChannelOutboxConsumer(MagicMock(), db_pool)
    msg_id = f"automation-{uuid.uuid4().hex[:10]}"
    body = {
        "id": msg_id,
        "kind": "automation_suggestion",
        "payload": {
            "message": "A weekly review automation is ready for approval.",
            "intent": "automation_suggestion",
            "suggestion_id": str(uuid.uuid4()),
            "delivery": {"mode": "web_inbox"},
        },
    }
    try:
        await consumer._process_message(body)
        await consumer._process_message(body)

        async with db_pool.acquire() as conn:
            delivered = await conn.fetchval(
                "SELECT COUNT(*) FROM web_inbox WHERE outbox_msg_id = $1", msg_id
            )

        assert delivered == 1
        push.assert_awaited_once_with(db_pool, body)
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM web_inbox WHERE outbox_msg_id = $1",
                msg_id,
            )
