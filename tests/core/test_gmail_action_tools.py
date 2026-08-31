from __future__ import annotations

import json
from uuid import uuid4

import pytest

from core.auth.google_gmail import GMAIL_DEFAULT_CREDENTIAL_REF
from core.auth.store import save_auth
from core.auth.utils import now_ms
from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.registry import create_default_registry

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_gmail_send_heartbeat_requires_connector_action_policy(
    db_pool, monkeypatch, tmp_path
):
    import core.auth.store as auth_store
    import services.gmail_actions as gmail_actions

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
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
            "account_email": "eric@example.com",
            "scopes": [gmail_actions.SCOPE_SEND],
        },
    )
    registry = create_default_registry(db_pool)
    call_args = {
        "account_key": "eric@example.com",
        "to": "alice@example.com",
        "subject": "Update",
        "body": "Done.",
    }
    ctx = ToolExecutionContext(
        tool_context=ToolContext.HEARTBEAT,
        call_id="gmail-send-policy",
        session_id="gmail-send-policy",
        energy_available=20,
    )

    async with db_pool.acquire() as conn:
        await conn.execute("SELECT grant_tool_approval('gmail_send')")

    denied = await registry.execute("gmail_send", call_args, ctx)
    assert not denied.success
    assert denied.error_type.value == "approval_required"
    assert "gmail/send" in denied.error

    async def fake_send_gmail_message(**kwargs):
        assert kwargs["to"] == "alice@example.com"
        return {
            "sent": True,
            "connector_id": "gmail",
            "account_key": kwargs["account_key"],
            "message_id": "sent-1",
            "thread_id": "thread-1",
            "to": ["alice@example.com"],
            "subject": kwargs["subject"],
        }

    monkeypatch.setattr(gmail_actions, "send_gmail_message", fake_send_gmail_message)
    async with db_pool.acquire() as conn:
        goal_id = await conn.fetchval(
            """
            INSERT INTO memories(type, goal_origin, content, importance, trust_level, status, metadata)
            VALUES ('goal', 'user_request', $1, 0.7, 0.8, 'active', '{"priority":"active"}'::jsonb)
            RETURNING id
            """,
            f"Gmail send policy test {uuid4()}",
        )
        policy = _j(
            await conn.fetchval(
                """
            SELECT grant_connector_action_policy(
                'gmail',
                'send',
                'eric@example.com',
                '{"allowed_recipients": ["alice@example.com"]}'::jsonb,
                TRUE,
                FALSE,
                ARRAY['heartbeat']::text[],
                NULL,
                'gmail-send-policy',
                'test grant',
                'user'
            )
            """
            )
        )

    call_args.update(
        purpose_kind="goal",
        purpose_reference=str(goal_id),
    )

    allowed = await registry.execute("gmail_send", call_args, ctx)

    async with db_pool.acquire() as conn:
        audits = await conn.fetch(
            """
            SELECT policy_id::text, connector_id, account_key, action_kind, target, decision
            FROM connector_action_audit
            WHERE context->>'call_id' = 'gmail-send-policy'
            ORDER BY created_at DESC
            """
        )
        await conn.execute(
            "DELETE FROM connector_action_audit WHERE context->>'call_id' = 'gmail-send-policy'"
        )
        await conn.execute(
            "DELETE FROM connector_action_policies WHERE source_session_id = 'gmail-send-policy'"
        )
        await conn.execute(
            "DELETE FROM outbound_events WHERE request_key LIKE 'tool:gmail-send-policy:%'"
        )
        await conn.execute("DELETE FROM memories WHERE id = $1", goal_id)

    assert allowed.success
    assert allowed.output["message_id"] == "sent-1"
    assert audits[0]["policy_id"] == policy["policy_id"]
    assert audits[0]["connector_id"] == "gmail"
    assert audits[0]["account_key"] == "eric@example.com"
    assert audits[0]["action_kind"] == "send"
    assert audits[0]["target"] == "alice@example.com"
    assert audits[0]["decision"] == "allowed"
