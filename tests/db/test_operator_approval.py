from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from services.operator_approval import request_operator_tool_approval

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "db"
    / "migrations"
    / "0206_operator_tool_approvals.sql"
)


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _configure(conn) -> None:
    await conn.execute(
        "SELECT set_config('operator.approval.enabled', 'true'::jsonb)"
    )
    await conn.execute(
        "SELECT set_config('channel.slack.operator_user_id', $1::jsonb)",
        json.dumps("U-OPERATOR"),
    )
    await conn.execute(
        "SELECT set_config('channel.imessage.operator_recipient', $1::jsonb)",
        json.dumps("+15551234567"),
    )


async def _create_request(
    conn,
    *,
    tool_name: str = "operator_test_tool",
    arguments: dict | None = None,
    session_id: str = "operator-test-session",
) -> str:
    request_id = str(uuid.uuid4())
    arguments = arguments or {"target": "exact"}
    presentation = {
        "blocks": [
            {"type": "text", "text": "Approve test action"},
            {
                "type": "actions",
                "actions": [
                    {
                        "action_id": "operator_approval_approve",
                        "label": "Approve",
                        "value": json.dumps(
                            {
                                "approval_request_id": request_id,
                                "decision": "approve",
                            }
                        ),
                        "style": "primary",
                    },
                    {
                        "action_id": "operator_approval_deny",
                        "label": "Deny",
                        "value": json.dumps(
                            {
                                "approval_request_id": request_id,
                                "decision": "deny",
                            }
                        ),
                        "style": "danger",
                    },
                ],
            },
        ]
    }
    result = _json(
        await conn.fetchval(
            """
            SELECT create_operator_tool_approval_request(
                $1::uuid, $2, $3::jsonb, $4::jsonb, 'heartbeat', $5,
                'heartbeat-test', 'heartbeat', 'Approve test action', $6::jsonb, 300
            )
            """,
            request_id,
            tool_name,
            json.dumps(arguments),
            json.dumps(arguments),
            session_id,
            json.dumps(presentation),
        )
    )
    assert result["created"] is True
    assert result["routed"] is True
    return request_id


async def _sync_slack_send(conn) -> None:
    await conn.fetchval(
        "SELECT sync_tool_definitions($1::jsonb)",
        json.dumps(
            [
                {
                    "name": "slack_send",
                    "description": "Send a Slack message",
                    "schema": {"type": "object", "properties": {}},
                    "category": "messaging",
                    "energy_cost": 3,
                    "allowed_contexts": ["chat", "heartbeat"],
                    "requires_approval": True,
                    "supports_parallel": False,
                }
            ]
        ),
    )


async def test_identity_checked_decision_and_exact_once_policy_proof(db_pool) -> None:
    arguments = {"channel": "#ops", "message": "Build failed"}
    context = {
        "tool_context": "heartbeat",
        "energy_available": 20,
        "session_id": "operator-test-session",
        "heartbeat_id": "heartbeat-test",
    }
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await _configure(conn)
            await conn.execute("SELECT set_config('tools', '{}'::jsonb)")
            await _sync_slack_send(conn)
            request_id = await _create_request(
                conn, tool_name="slack_send", arguments=arguments
            )

            unauthorized = _json(
                await conn.fetchval(
                    "SELECT record_operator_tool_approval_decision($1::uuid, 'approve', 'slack', 'U-OTHER')",
                    request_id,
                )
            )
            approved = _json(
                await conn.fetchval(
                    "SELECT record_operator_tool_approval_decision($1::uuid, 'approve', 'slack', 'U-OPERATOR')",
                    request_id,
                )
            )
            wrong = _json(
                await conn.fetchval(
                    "SELECT evaluate_tool_call('slack_send', $1::jsonb, $2::jsonb)",
                    json.dumps({**arguments, "message": "Different"}),
                    json.dumps({**context, "approval_request_id": request_id}),
                )
            )
            allowed = _json(
                await conn.fetchval(
                    "SELECT evaluate_tool_call('slack_send', $1::jsonb, $2::jsonb)",
                    json.dumps(arguments),
                    json.dumps({**context, "approval_request_id": request_id}),
                )
            )
            replay = _json(
                await conn.fetchval(
                    "SELECT evaluate_tool_call('slack_send', $1::jsonb, $2::jsonb)",
                    json.dumps(arguments),
                    json.dumps({**context, "approval_request_id": request_id}),
                )
            )
            status = await conn.fetchval(
                "SELECT status FROM operator_tool_approval_requests WHERE id = $1::uuid",
                request_id,
            )
        finally:
            await transaction.rollback()

    assert unauthorized == {"ok": False, "error": "unauthorized_actor"}
    assert approved["status"] == "approved"
    assert wrong["allowed"] is False
    assert wrong["error_type"] == "approval_required"
    assert allowed["allowed"] is True
    assert allowed["operator_approval_request_id"] == request_id
    assert allowed["connector_action"]["authorization_kind"] == (
        "operator_exact_once_approval"
    )
    assert replay["allowed"] is False
    assert replay["error_type"] == "approval_required"
    assert status == "consumed"


async def test_outbox_payload_is_actionable_but_stores_only_redacted_preview(db_pool) -> None:
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await _configure(conn)
            request_id = await _create_request(
                conn,
                arguments={"recipient": "person@example.com", "token": "[redacted]"},
            )
            row = await conn.fetchrow(
                """
                SELECT arguments_hash, arguments_preview, outbox_message_id
                FROM operator_tool_approval_requests WHERE id = $1::uuid
                """,
                request_id,
            )
            envelope = _json(
                await conn.fetchval(
                    "SELECT envelope FROM outbox_messages WHERE id = $1::uuid",
                    row["outbox_message_id"],
                )
            )
        finally:
            await transaction.rollback()

    assert len(row["arguments_hash"]) == 64
    assert _json(row["arguments_preview"])["token"] == "[redacted]"
    assert envelope["kind"] == "operator_approval"
    assert envelope["payload"]["delivery_mode"] == "direct"
    assert envelope["payload"]["target_id"] == "U-OPERATOR"
    actions = next(
        block
        for block in envelope["payload"]["presentation"]["blocks"]
        if block["type"] == "actions"
    )
    assert [action["label"] for action in actions["actions"]] == ["Approve", "Deny"]


async def test_plain_reply_requires_code_when_multiple_requests_are_pending(db_pool) -> None:
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await _configure(conn)
            first = await _create_request(conn, tool_name="first_tool")
            second = await _create_request(conn, tool_name="second_tool")
            ambiguous = _json(
                await conn.fetchval(
                    "SELECT try_resolve_operator_tool_approval_from_inbound('imessage', '+15551234567', 'approve')"
                )
            )
            exact = _json(
                await conn.fetchval(
                    "SELECT try_resolve_operator_tool_approval_from_inbound('imessage', '+15551234567', $1)",
                    f"deny {second.replace('-', '')[:8]}",
                )
            )
            first_status = await conn.fetchval(
                "SELECT status FROM operator_tool_approval_requests WHERE id = $1::uuid",
                first,
            )
            second_status = await conn.fetchval(
                "SELECT status FROM operator_tool_approval_requests WHERE id = $1::uuid",
                second,
            )
        finally:
            await transaction.rollback()

    assert ambiguous["recognized"] is True
    assert ambiguous["matched"] is False
    assert ambiguous["reason"] == "ambiguous_without_code"
    assert "approve CODE" in ambiguous["message"]
    assert exact["matched"] is True
    assert exact["status"] == "denied"
    assert first_status == "pending"
    assert second_status == "denied"


async def test_pending_slack_delivery_escalates_to_imessage_and_completes(db_pool) -> None:
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await _configure(conn)
            request_id = await _create_request(conn)
            await conn.execute(
                "UPDATE operator_tool_approval_requests SET escalate_after = CURRENT_TIMESTAMP - INTERVAL '1 second' WHERE id = $1::uuid",
                request_id,
            )
            claimed = _json(
                await conn.fetchval(
                    "SELECT claim_operator_tool_approval_escalations(10)"
                )
            )
            completed = await conn.fetchval(
                "SELECT complete_operator_tool_approval_escalation($1::uuid, 'imessage-1')",
                request_id,
            )
            row = await conn.fetchrow(
                """
                SELECT status, escalation_attempts, imessage_recipient,
                       imessage_message_id
                FROM operator_tool_approval_requests WHERE id = $1::uuid
                """,
                request_id,
            )
        finally:
            await transaction.rollback()

    assert [str(item["id"]) for item in claimed] == [request_id]
    assert completed is True
    assert row["status"] == "escalated"
    assert row["escalation_attempts"] == 1
    assert row["imessage_recipient"] == "+15551234567"
    assert row["imessage_message_id"] == "imessage-1"


async def test_operator_approval_migration_is_idempotent(db_pool) -> None:
    migration = _MIGRATION.read_text(encoding="utf-8")
    sentinel = str(uuid.uuid4())
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(
                """
                INSERT INTO operator_tool_approval_requests (
                    id, tool_name, arguments_hash, tool_context, status, expires_at
                ) VALUES ($1::uuid, 'sentinel', repeat('0', 64), 'chat', 'expired', CURRENT_TIMESTAMP)
                """,
                sentinel,
            )
            await conn.execute(migration)
            await conn.execute(migration)
            survived = await conn.fetchval(
                "SELECT count(*) FROM operator_tool_approval_requests WHERE id = $1::uuid",
                sentinel,
            )
        finally:
            await transaction.rollback()

    assert survived == 1


async def test_service_files_waits_and_wakes_on_slack_decision(db_pool) -> None:
    keys = [
        "operator.approval.enabled",
        "operator.approval.slack_interactive_enabled",
        "channel.slack.operator_user_id",
    ]
    async with db_pool.acquire() as conn:
        original_rows = await conn.fetch(
            "SELECT key, value FROM config WHERE key = ANY($1::text[])", keys
        )
        await conn.execute(
            "SELECT set_config('operator.approval.enabled', 'true'::jsonb)"
        )
        await conn.execute(
            "SELECT set_config('operator.approval.slack_interactive_enabled', 'true'::jsonb)"
        )
        await conn.execute(
            "SELECT set_config('channel.slack.operator_user_id', $1::jsonb)",
            json.dumps("U-OPERATOR"),
        )

    request_id = None
    waiter = None
    try:
        waiter = asyncio.create_task(
            request_operator_tool_approval(
                db_pool,
                tool_name="service_wait_test_tool",
                arguments={"target": "exact", "api_token": "secret"},
                tool_context="heartbeat",
                session_id="service-wait-session",
                heartbeat_id="service-wait-heartbeat",
                surface="heartbeat",
                wait_seconds=60,
            )
        )
        for _ in range(50):
            async with db_pool.acquire() as conn:
                request_id = await conn.fetchval(
                    """
                    SELECT id::text FROM operator_tool_approval_requests
                    WHERE tool_name = 'service_wait_test_tool'
                    ORDER BY created_at DESC LIMIT 1
                    """
                )
            if request_id:
                break
            await asyncio.sleep(0.05)
        assert request_id is not None

        async with db_pool.acquire() as conn:
            decision = _json(
                await conn.fetchval(
                    "SELECT record_operator_tool_approval_decision($1::uuid, 'approve', 'slack', 'U-OPERATOR')",
                    request_id,
                )
            )
        result = await asyncio.wait_for(waiter, timeout=5)

        assert decision["ok"] is True
        assert result == {
            "approved": True,
            "request_id": request_id,
            "status": "approved",
            "approval_channel": "slack",
            "reason": None,
        }
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT arguments_preview, outbox_message_id
                FROM operator_tool_approval_requests WHERE id = $1::uuid
                """,
                request_id,
            )
        assert _json(row["arguments_preview"])["api_token"] == "[redacted]"
    finally:
        if waiter is not None and not waiter.done():
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        if request_id:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM outbox_messages WHERE id = (SELECT outbox_message_id FROM operator_tool_approval_requests WHERE id = $1::uuid)",
                    request_id,
                )
                await conn.execute(
                    "DELETE FROM operator_tool_approval_requests WHERE id = $1::uuid",
                    request_id,
                )
        original = {row["key"]: row["value"] for row in original_rows}
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM config WHERE key = ANY($1::text[])", keys)
            for key, value in original.items():
                await conn.execute(
                    "INSERT INTO config (key, value) VALUES ($1, $2::jsonb)",
                    key,
                    value if isinstance(value, str) else json.dumps(value),
                )
