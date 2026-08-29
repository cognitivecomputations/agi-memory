from __future__ import annotations

import json
from uuid import uuid4

import pytest

from core.auth.store import save_auth
from core.auth.twitter_x import TWITTER_X_DEFAULT_CREDENTIAL_REF
from core.auth.utils import now_ms
from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.registry import create_default_registry

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_twitter_x_post_heartbeat_requires_connector_action_policy(
    db_pool, monkeypatch, tmp_path
):
    import core.auth.store as auth_store
    import services.twitter_x as twitter_x

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    save_auth(
        TWITTER_X_DEFAULT_CREDENTIAL_REF,
        {
            "type": "twitter_x_oauth2",
            "token": "access-token",
            "refresh_token": "refresh-token",
            "client_id": "client",
            "expires_ms": now_ms() + 3_600_000,
            "account_key": "x:123",
            "user_id": "123",
            "username": "hexis_test",
            "scopes": [
                twitter_x.SCOPE_TWEET_READ,
                twitter_x.SCOPE_TWEET_WRITE,
                twitter_x.SCOPE_USERS_READ,
                "offline.access",
            ],
        },
    )
    registry = create_default_registry(db_pool)
    call_args = {
        "account_key": "x:123",
        "text": "Status update from Hexis test.",
    }
    ctx = ToolExecutionContext(
        tool_context=ToolContext.HEARTBEAT,
        call_id="twitter-x-post-policy",
        session_id="twitter-x-post-policy",
        energy_available=20,
    )

    async with db_pool.acquire() as conn:
        await conn.execute("SELECT grant_tool_approval('twitter_x_post')")

    denied = await registry.execute("twitter_x_post", call_args, ctx)
    assert not denied.success
    assert denied.error_type.value == "approval_required"
    assert "twitter_x/post" in denied.error

    async def fake_post_twitter_x(**kwargs):
        assert kwargs["text"].startswith("Status update from Hexis test.")
        assert "Reply STOP" in kwargs["text"]
        return {
            "sent": True,
            "connector_id": "twitter_x",
            "account_key": kwargs["account_key"],
            "tweet_id": "tweet-1",
            "text": kwargs["text"],
        }

    monkeypatch.setattr(twitter_x, "post_twitter_x", fake_post_twitter_x)
    async with db_pool.acquire() as conn:
        goal_id = await conn.fetchval(
            """
            INSERT INTO memories(type, goal_origin, content, importance, trust_level, status, metadata)
            VALUES ('goal', 'user_request', $1, 0.7, 0.8, 'active', '{"priority":"active"}'::jsonb)
            RETURNING id
            """,
            f"Twitter/X policy test {uuid4()}",
        )
        policy = _j(
            await conn.fetchval(
                """
            SELECT grant_connector_action_policy(
                'twitter_x',
                'post',
                'x:123',
                '{}'::jsonb,
                TRUE,
                FALSE,
                ARRAY['heartbeat']::text[],
                NULL,
                'twitter-x-post-policy',
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

    allowed = await registry.execute("twitter_x_post", call_args, ctx)

    async with db_pool.acquire() as conn:
        audits = await conn.fetch(
            """
            SELECT policy_id::text, connector_id, account_key, action_kind, target, decision
            FROM connector_action_audit
            WHERE context->>'call_id' = 'twitter-x-post-policy'
            ORDER BY created_at DESC
            """
        )
        await conn.execute(
            "DELETE FROM connector_action_audit WHERE context->>'call_id' = 'twitter-x-post-policy'"
        )
        await conn.execute(
            "DELETE FROM connector_action_policies WHERE source_session_id = 'twitter-x-post-policy'"
        )
        await conn.execute(
            "DELETE FROM outbound_events WHERE request_key LIKE 'tool:twitter-x-post-policy:%'"
        )
        await conn.execute("DELETE FROM memories WHERE id = $1", goal_id)

    assert allowed.success
    assert allowed.output["tweet_id"] == "tweet-1"
    assert audits[0]["policy_id"] == policy["policy_id"]
    assert audits[0]["connector_id"] == "twitter_x"
    assert audits[0]["account_key"] == "x:123"
    assert audits[0]["action_kind"] == "post"
    assert audits[0]["target"].startswith("Status update from Hexis test.")
    assert "Reply STOP" in audits[0]["target"]
    assert audits[0]["decision"] == "allowed"
