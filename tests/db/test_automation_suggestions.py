from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "db"
    / "migrations"
    / "0207_automation_suggestions.sql"
)


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _task_spec(name: str = "Test routine") -> dict:
    return {
        "action": "create",
        "name": name,
        "description": "A test-only recurring reminder.",
        "schedule": "weekly:monday:09:15",
        "action_kind": "queue_user_message",
        "message": "Open Hexis for the test routine.",
        "delivery_mode": "outbox",
    }


async def test_proposal_is_inert_and_acceptance_creates_exactly_one_task(db_pool):
    key = f"usage:test:{uuid.uuid4().hex}"
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute("SELECT set_config('automation.suggestions.enabled', 'true'::jsonb)")
            await conn.execute(
                "SELECT set_config('agent.timezone', $1::jsonb)",
                json.dumps("America/New_York"),
            )
            before = await conn.fetchval("SELECT count(*) FROM scheduled_tasks")
            proposed = _json(
                await conn.fetchval(
                    "SELECT propose_automation('usage', $1, 'Test routine', 'Three matching asks support this cadence.', $2::jsonb)",
                    key,
                    json.dumps(_task_spec()),
                )
            )
            after_proposal = await conn.fetchval("SELECT count(*) FROM scheduled_tasks")
            row = await conn.fetchrow(
                "SELECT status, task_spec, outbox_message_id FROM automation_suggestions WHERE id = $1::uuid",
                proposed["suggestion_id"],
            )
            envelope = _json(
                await conn.fetchval(
                    "SELECT envelope FROM outbox_messages WHERE id = $1::uuid",
                    row["outbox_message_id"],
                )
            )

            accepted = _json(
                await conn.fetchval(
                    "SELECT accept_automation($1::uuid, 'web', 'test')",
                    proposed["suggestion_id"],
                )
            )
            accepted_again = _json(
                await conn.fetchval(
                    "SELECT accept_automation($1::uuid, 'web', 'test')",
                    proposed["suggestion_id"],
                )
            )
            task = await conn.fetchrow(
                "SELECT * FROM scheduled_tasks WHERE id = $1::uuid",
                accepted["scheduled_task_id"],
            )
            after_accept = await conn.fetchval("SELECT count(*) FROM scheduled_tasks")
        finally:
            await transaction.rollback()

    assert proposed["created"] is True
    assert after_proposal == before
    assert row["status"] == "pending"
    assert _json(row["task_spec"])["action"] == "create"
    assert envelope["payload"]["intent"] == "automation_suggestion"
    assert "1 " in envelope["payload"]["message"]
    assert "2 " in envelope["payload"]["message"]
    assert accepted["ok"] is True
    assert accepted_again["already_decided"] is True
    assert after_accept == before + 1
    assert task["timezone"] == "America/New_York"
    assert task["schedule_kind"] == "weekly"
    assert _json(task["schedule"]) == {"weekday": "monday", "time": "09:15"}
    assert _json(task["action_payload"])["message"] == "Open Hexis for the test routine."


async def test_dismissal_latches_by_dedup_key(db_pool):
    key = f"catalog:test:{uuid.uuid4().hex}"
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            first = _json(
                await conn.fetchval(
                    "SELECT propose_automation('catalog', $1, 'No thanks', 'Test the permanent dismissal.', $2::jsonb)",
                    key,
                    json.dumps(_task_spec("No thanks")),
                )
            )
            dismissed = _json(
                await conn.fetchval(
                    "SELECT dismiss_automation($1::uuid, 'web', 'test')",
                    first["suggestion_id"],
                )
            )
            duplicate = _json(
                await conn.fetchval(
                    "SELECT propose_automation('catalog', $1, 'No thanks again', 'This must not reopen.', $2::jsonb)",
                    key,
                    json.dumps(_task_spec("No thanks again")),
                )
            )
            count = await conn.fetchval(
                "SELECT count(*) FROM automation_suggestions WHERE dedup_key = $1", key
            )
        finally:
            await transaction.rollback()

    assert dismissed["dismissal_latched"] is True
    assert duplicate["created"] is False
    assert duplicate["status"] == "dismissed"
    assert duplicate["reason"] == "dismissal_latched"
    assert count == 1


async def test_catalog_uses_live_readiness_and_connector_preconditions(db_pool):
    account_key = f"automation-test-{uuid.uuid4().hex}"
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute("DELETE FROM automation_suggestions")
            await conn.execute("SELECT set_config('agent.is_configured', 'true'::jsonb)")
            await conn.execute("UPDATE heartbeat_state SET init_stage = 'complete' WHERE id = 1")
            catalog = _json(
                await conn.fetchval("SELECT refresh_automation_suggestion_catalog()")
            )
            before_gmail = await conn.fetchval(
                "SELECT count(*) FROM automation_suggestions WHERE dedup_key = 'connector:gmail:important-mail-monitor'"
            )
            await conn.execute(
                """
                INSERT INTO integration_connections (
                    connector_id, account_key, status, capabilities, connected_at
                ) VALUES ('gmail', $1, 'connected', '["read", "search"]'::jsonb, CURRENT_TIMESTAMP)
                """,
                account_key,
            )
            after_gmail = await conn.fetchval(
                "SELECT count(*) FROM automation_suggestions WHERE dedup_key = 'connector:gmail:important-mail-monitor'"
            )
            source = await conn.fetchval(
                "SELECT source FROM automation_suggestions WHERE dedup_key = 'connector:gmail:important-mail-monitor'"
            )
        finally:
            await transaction.rollback()

    assert catalog["created"] == 3
    assert catalog["ineligible"] >= 1
    assert before_gmail == 0
    assert after_gmail == 1
    assert source == "connector"


async def test_numbered_channel_reply_requires_code_when_ambiguous(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            first = _json(
                await conn.fetchval(
                    "SELECT propose_automation('catalog', $1, 'First', 'First test routine.', $2::jsonb)",
                    f"catalog:test:{uuid.uuid4().hex}",
                    json.dumps(_task_spec("First")),
                )
            )
            second = _json(
                await conn.fetchval(
                    "SELECT propose_automation('catalog', $1, 'Second', 'Second test routine.', $2::jsonb)",
                    f"catalog:test:{uuid.uuid4().hex}",
                    json.dumps(_task_spec("Second")),
                )
            )
            ambiguous = _json(
                await conn.fetchval(
                    "SELECT try_resolve_automation_suggestion_from_inbound('telegram', 'user-1', '1')"
                )
            )
            code = str(second["suggestion_id"]).replace("-", "")[:8]
            exact = _json(
                await conn.fetchval(
                    "SELECT try_resolve_automation_suggestion_from_inbound('telegram', 'user-1', $1)",
                    f"2 {code}",
                )
            )
            status_rows = await conn.fetch(
                "SELECT id::text AS id, status FROM automation_suggestions WHERE id = ANY($1::uuid[])",
                [first["suggestion_id"], second["suggestion_id"]],
            )
            statuses = {row["id"]: row["status"] for row in status_rows}
        finally:
            await transaction.rollback()

    assert ambiguous["recognized"] is True
    assert ambiguous["matched"] is False
    assert ambiguous["reason"] == "ambiguous_without_code"
    assert exact["matched"] is True
    assert exact["status"] == "dismissed"
    assert statuses[str(first["suggestion_id"])] == "pending"
    assert statuses[str(second["suggestion_id"])] == "dismissed"


async def test_automation_suggestion_migration_is_idempotent(db_pool):
    migration = _MIGRATION.read_text(encoding="utf-8")
    key = f"migration-sentinel:{uuid.uuid4().hex}"
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(
                """
                INSERT INTO automation_suggestions (
                    source, dedup_key, title, rationale, task_spec, status
                ) VALUES (
                    'catalog', $1, 'Sentinel', 'Must survive migration replay.',
                    $2::jsonb, 'dismissed'
                )
                """,
                key,
                json.dumps(_task_spec("Sentinel")),
            )
            await conn.execute(migration)
            await conn.execute(migration)
            row = await conn.fetchrow(
                "SELECT status, title FROM automation_suggestions WHERE dedup_key = $1",
                key,
            )
        finally:
            await transaction.rollback()

    assert row["status"] == "dismissed"
    assert row["title"] == "Sentinel"
