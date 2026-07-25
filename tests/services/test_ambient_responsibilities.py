from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from services.ambient_responsibilities import run_ambient_responsibility_step
from services.worker_service import MaintenanceWorker
from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _coerce_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


async def test_ambient_reminder_delivers_to_web_inbox(db_pool):
    marker = get_test_identifier("ambient-reminder")
    title = f"ambient reminder {marker}"
    message = f"ambient reminder delivered {marker}"
    worker = MaintenanceWorker()
    worker.pool = db_pool
    worker.bridge = None

    async with db_pool.acquire() as conn:
        created = _coerce_json(
            await conn.fetchval(
                "SELECT manage_ambient_responsibility_tool($1::jsonb)",
                json.dumps(
                    {
                        "action": "create",
                        "title": title,
                        "kind": "reminder",
                        "user_intent": f"Remind me: {message}",
                        "trigger": {"kind": "interval", "every_seconds": 60},
                        "message": message,
                    }
                ),
            )
        )
        responsibility_id = created["output"]["responsibility_id"]
        await conn.execute(
            "UPDATE ambient_responsibilities SET next_check_at = NOW() - INTERVAL '1 second' WHERE id = $1",
            responsibility_id,
        )

    try:
        result = await worker._run_ambient_responsibilities()
        assert result["claimed"] >= 1
        assert result["fired"] >= 1
        assert result["web_inbox_delivered"] >= 1

        async with db_pool.acquire() as conn:
            delivered = await conn.fetchrow(
                "SELECT message, intent, delivered_at FROM web_inbox WHERE message = $1",
                message,
            )
            responsibility = await conn.fetchrow(
                """
                SELECT status, last_checked_at, last_fired_at, consecutive_errors
                FROM ambient_responsibilities
                WHERE id = $1::uuid
                """,
                responsibility_id,
            )
        assert delivered["message"] == message
        assert delivered["intent"] == "ambient_responsibility"
        assert delivered["delivered_at"] <= datetime.now(timezone.utc)
        assert responsibility["status"] == "active"
        assert responsibility["last_checked_at"] is not None
        assert responsibility["last_fired_at"] is not None
        assert responsibility["consecutive_errors"] == 0
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM web_inbox WHERE message = $1", message)
            await conn.execute("DELETE FROM ambient_responsibilities WHERE title = $1", title)


async def test_recent_checkin_keeps_ambient_responsibility_silent(db_pool):
    marker = get_test_identifier("ambient-checkin")
    title = f"ambient checkin {marker}"
    message = f"take pills {marker}"

    async with db_pool.acquire() as conn:
        created = _coerce_json(
            await conn.fetchval(
                "SELECT manage_ambient_responsibility_tool($1::jsonb)",
                json.dumps(
                    {
                        "action": "create",
                        "title": title,
                        "kind": "checkin",
                        "user_intent": "Let me know if I have not checked in.",
                        "trigger": {"kind": "interval", "every_seconds": 60},
                        "evaluator": {"type": "missing_checkin", "lookback_minutes": 720},
                        "message": message,
                    }
                ),
            )
        )
        responsibility_id = created["output"]["responsibility_id"]
        await conn.fetchval(
            "SELECT manage_ambient_responsibility_tool($1::jsonb)",
            json.dumps(
                {
                    "action": "checkin",
                    "responsibility_id": responsibility_id,
                    "label": "pills",
                    "note": "done",
                }
            ),
        )
        await conn.execute(
            "UPDATE ambient_responsibilities SET next_check_at = NOW() - INTERVAL '1 second' WHERE id = $1",
            responsibility_id,
        )

    try:
        result = await run_ambient_responsibility_step(db_pool)
        assert result["claimed"] >= 1
        assert result["silent"] >= 1
        assert result["fired"] == 0
        assert result["outbox_messages"] == []

        async with db_pool.acquire() as conn:
            responsibility = await conn.fetchrow(
                """
                SELECT last_checked_at, last_fired_at, consecutive_silent
                FROM ambient_responsibilities
                WHERE id = $1::uuid
                """,
                responsibility_id,
            )
        assert responsibility["last_checked_at"] is not None
        assert responsibility["last_fired_at"] is None
        assert responsibility["consecutive_silent"] >= 1
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM ambient_responsibilities WHERE title = $1", title)


async def _connected_channel(conn, connector_id: str, marker: str, account_key: str) -> None:
    capabilities = json.dumps(["live_chat", "send", "ingest_live"])
    attempt = _coerce_json(
        await conn.fetchval(
            """
            SELECT start_connection_attempt(
                $1,
                $3::jsonb,
                ARRAY[]::text[],
                '{}'::jsonb,
                NULL,
                NULL,
                'test',
                $2,
                CURRENT_TIMESTAMP + INTERVAL '10 minutes'
            )
            """,
            connector_id,
            marker,
            capabilities,
        )
    )
    await conn.fetchval(
        """
        SELECT complete_connection_attempt(
            $1::uuid,
            $2,
            $3,
            $4,
            ARRAY[]::text[],
            $5::jsonb,
            '{"test": true}'::jsonb
        )
        """,
        attempt["attempt_id"],
        account_key,
        connector_id,
        f"config:channel.{connector_id}",
        capabilities,
    )


async def test_generic_connector_source_monitor_fires_from_new_item(db_pool):
    marker = get_test_identifier("ambient-slack")
    title = f"ambient slack monitor {marker}"
    account = f"channel:slack:{marker}"
    provider_item_id = f"ambient-msg-{marker}"
    message = f"Hope pinged you {marker}: {{title}}"

    async with db_pool.acquire() as conn:
        await _connected_channel(conn, "slack", marker, account)
        created = _coerce_json(
            await conn.fetchval(
                "SELECT manage_ambient_responsibility_tool($1::jsonb)",
                json.dumps(
                    {
                        "action": "create",
                        "title": title,
                        "kind": "monitor",
                        "user_intent": "Let me know whenever Hope messages me in Slack.",
                        "trigger": {"kind": "interval", "every_seconds": 60},
                        "sources": [
                            {
                                "connector_id": "slack",
                                "account_key": account,
                                "query": "from:Hope",
                                "page_size": 10,
                            }
                        ],
                        "actions": [{"type": "notify_user", "message": message}],
                    }
                ),
            )
        )
        responsibility_id = created["output"]["responsibility_id"]
        await conn.fetchval(
            """
            SELECT upsert_connector_source_item(
                'slack',
                $1,
                $2,
                'Need your attention',
                $3,
                'message',
                NULL,
                CURRENT_TIMESTAMP,
                ARRAY['slack']::text[],
                '[{"role": "sender", "name": "Hope", "id": "UHOPE"}]'::jsonb,
                '[]'::jsonb,
                '{"channel_id": "C123"}'::jsonb,
                'private',
                FALSE
            )
            """,
            account,
            provider_item_id,
            f"Hope: please look at this when you can {marker}",
        )
        await conn.execute(
            "UPDATE ambient_responsibilities SET next_check_at = NOW() - INTERVAL '1 second' WHERE id = $1::uuid",
            responsibility_id,
        )

    try:
        result = await run_ambient_responsibility_step(db_pool)
        assert result["claimed"] >= 1
        assert result["fired"] >= 1
        assert result["outbox_messages"]
        assert message.format(title="Need your attention") in json.dumps(result["outbox_messages"])

        async with db_pool.acquire() as conn:
            observation = await conn.fetchrow(
                """
                SELECT connector_id, provider_item_id, source_item_id, source_document_id
                FROM ambient_observations
                WHERE responsibility_id = $1::uuid
                  AND provider_item_id = $2
                """,
                responsibility_id,
                provider_item_id,
            )
        assert observation["connector_id"] == "slack"
        assert observation["source_item_id"] is not None
        assert observation["source_document_id"] is not None
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM web_inbox WHERE message LIKE $1", f"%{marker}%")
            await conn.execute("DELETE FROM ambient_responsibilities WHERE title = $1", title)
            await conn.execute("DELETE FROM connector_source_items WHERE account_key = $1", account)
            await conn.execute("DELETE FROM integration_connections WHERE account_key = $1", account)
            await conn.execute("DELETE FROM connection_attempts WHERE source_session_id = $1", marker)


async def test_blocked_connector_monitor_unblocks_after_connection(db_pool):
    marker = get_test_identifier("ambient-unblock")
    title = f"ambient blocked monitor {marker}"
    account = f"channel:telegram:{marker}"

    async with db_pool.acquire() as conn:
        created = _coerce_json(
            await conn.fetchval(
                "SELECT manage_ambient_responsibility_tool($1::jsonb)",
                json.dumps(
                    {
                        "action": "create",
                        "title": title,
                        "kind": "monitor",
                        "user_intent": "Let me know whenever Hope messages me in Telegram.",
                        "trigger": {"kind": "interval", "every_seconds": 60},
                        "sources": [{"connector_id": "telegram", "account_key": account, "query": "from:Hope"}],
                    }
                ),
            )
        )
        responsibility_id = created["output"]["responsibility_id"]
        assert created["output"]["status"] == "blocked"

        await _connected_channel(conn, "telegram", marker, account)
        refreshed = _coerce_json(await conn.fetchval("SELECT refresh_ambient_responsibility_blockers()"))
        row = await conn.fetchrow(
            "SELECT status, next_check_at, last_error FROM ambient_responsibilities WHERE id = $1::uuid",
            responsibility_id,
        )

    try:
        assert refreshed["unblocked"] >= 1
        assert row["status"] == "active"
        assert row["next_check_at"] is not None
        assert row["last_error"] is None
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM ambient_responsibilities WHERE title = $1", title)
            await conn.execute("DELETE FROM integration_connections WHERE account_key = $1", account)
            await conn.execute("DELETE FROM connection_attempts WHERE source_session_id = $1", marker)


async def test_metric_threshold_blocks_without_provider_and_fires_with_current_value(db_pool):
    marker = get_test_identifier("ambient-steps")
    missing_title = f"ambient steps missing {marker}"
    ready_title = f"ambient steps ready {marker}"

    async with db_pool.acquire() as conn:
        missing = _coerce_json(
            await conn.fetchval(
                "SELECT manage_ambient_responsibility_tool($1::jsonb)",
                json.dumps(
                    {
                        "action": "create",
                        "title": missing_title,
                        "kind": "threshold",
                        "user_intent": "Let me know if I have not taken enough steps.",
                        "trigger": {"kind": "interval", "every_seconds": 60},
                        "sources": [
                            {
                                "connector_id": "health",
                                "require_connection": False,
                                "metric": "steps",
                                "operator": "<",
                                "value": 6000,
                            }
                        ],
                        "actions": [{"type": "notify_user", "message": f"Steps are low {marker}"}],
                    }
                ),
            )
        )
        missing_id = missing["output"]["responsibility_id"]
        ready = _coerce_json(
            await conn.fetchval(
                "SELECT manage_ambient_responsibility_tool($1::jsonb)",
                json.dumps(
                    {
                        "action": "create",
                        "title": ready_title,
                        "kind": "threshold",
                        "user_intent": "Let me know if I have not taken enough steps.",
                        "trigger": {"kind": "interval", "every_seconds": 60},
                        "sources": [
                            {
                                "connector_id": "health",
                                "require_connection": False,
                                "metric": "steps",
                                "current_value": 3200,
                                "operator": "<",
                                "value": 6000,
                            }
                        ],
                        "actions": [{"type": "notify_user", "message": f"Steps are low {marker}"}],
                    }
                ),
            )
        )
        ready_id = ready["output"]["responsibility_id"]
        await conn.execute(
            """
            UPDATE ambient_responsibilities
            SET next_check_at = NOW() - INTERVAL '1 second'
            WHERE id = ANY($1::uuid[])
            """,
            [missing_id, ready_id],
        )

    try:
        result = await run_ambient_responsibility_step(db_pool, limit=10)
        assert result["blocked"] >= 1
        assert result["fired"] >= 1
        decisions = json.dumps([run["decision"] for run in result["runs"]])
        assert "metric_provider_not_configured" in decisions
        assert "threshold_crossed" in decisions
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM ambient_responsibilities WHERE title = ANY($1::text[])",
                [missing_title, ready_title],
            )
