from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from core.auth.google_gmail import (
    GMAIL_CLIENT_SECRET_REF,
    GMAIL_DEFAULT_CREDENTIAL_REF,
    GMAIL_PENDING_PREFIX,
    GMAIL_REDIRECT_URI,
    has_bundled_gmail_oauth_client,
    has_hexis_gmail_oauth_client,
    save_gmail_client_secret_payload,
)
from core.auth.store import load_auth, save_auth
from core.auth.utils import now_ms
from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.integrations import (
    CompleteGmailConnectionHandler,
    ConnectGmailHandler,
    ControlGmailBackfillHandler,
    ConnectorActionPolicyStatusHandler,
    GmailBackfillStatusHandler,
    GmailSetupStatusHandler,
    GrantConnectorActionPolicyHandler,
    RevokeGmailConnectionHandler,
    RevokeConnectorActionPolicyHandler,
    StartGmailBackfillHandler,
    _gmail_connector_setup_ui,
)
from core.tools.registry import create_default_registry
from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


def _ctx(db_pool, marker: str) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id=f"call-{marker}",
        session_id=marker,
        registry=SimpleNamespace(pool=db_pool),
    )


def _client_secret() -> dict[str, object]:
    return {
        "installed": {
            "client_id": "gmail-test-client",
            "client_secret": "gmail-test-secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


async def test_gmail_setup_contexts_respect_user_presence():
    assert ToolContext.HEARTBEAT in GmailSetupStatusHandler().spec.allowed_contexts
    assert ToolContext.HEARTBEAT in GmailBackfillStatusHandler().spec.allowed_contexts

    # Starting/completing account authorization needs a present user surface.
    assert ToolContext.HEARTBEAT not in ConnectGmailHandler().spec.allowed_contexts
    assert ToolContext.HEARTBEAT not in CompleteGmailConnectionHandler().spec.allowed_contexts
    assert ToolContext.HEARTBEAT not in RevokeGmailConnectionHandler().spec.allowed_contexts
    assert ToolContext.HEARTBEAT not in StartGmailBackfillHandler().spec.allowed_contexts


async def _seed_connected_gmail(db_pool, marker: str, account: str) -> None:
    async with db_pool.acquire() as conn:
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


async def test_connect_gmail_without_client_secret_returns_next_step(monkeypatch, tmp_path):
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    for name in (
        "GOOGLE_GMAIL_CLIENT_SECRET_JSON",
        "GOOGLE_CLIENT_SECRET_JSON",
        "GOOGLE_GMAIL_CLIENT_SECRET_PATH",
        "GOOGLE_CLIENT_SECRET_PATH",
        "HEXIS_GMAIL_OAUTH_CLIENT_ID",
        "HEXIS_GMAIL_OAUTH_CLIENT_SECRET",
        "GOOGLE_GMAIL_OAUTH_CLIENT_ID",
        "GOOGLE_GMAIL_OAUTH_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    result = await ConnectGmailHandler().execute(
        {},
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="gmail-no-secret",
            registry=SimpleNamespace(pool=object()),
        ),
    )

    assert result.success
    assert result.output["status"] == "needs_client_secret"
    assert "hexis_oauth_client" in result.output["accepted_inputs"]
    assert "advanced_self_hosted" in result.output["accepted_inputs"]
    assert "client_secret_file" in result.output["accepted_inputs"]
    assert "client_secret_path" in result.output["accepted_inputs"]
    assert "Gmail sign-in is not enabled" in result.output["next_step"]
    assert "setup guide" in result.output["next_step"]
    assert result.output["setup_steps"]
    assert result.output["ui"]["kind"] == "connector_setup"
    assert result.output["ui"]["connector_id"] == "gmail"
    assert result.output["ui"]["status"] == "needs_client_secret"
    assert result.output["ui"]["capabilities"] == ["read", "search"]
    assert result.output["ui"]["credential_step"]["preferred_mode"] == "hosted_oauth_missing"
    assert result.output["ui"]["hexis_oauth_client_available"] is False
    assert result.output["ui"]["setup_steps"]
    assert result.output["ui"]["credential_step"]["modes"][0]["label"] == "Built-in Google sign-in"
    assert "advanced_self_hosted" in result.output["ui"]["accepted_inputs"]
    assert "client_secret_file" in result.output["ui"]["accepted_inputs"]


async def test_connect_gmail_uses_configured_hexis_oauth_client(db_pool, monkeypatch, tmp_path):
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    monkeypatch.setenv("HEXIS_GMAIL_OAUTH_CLIENT_ID", "hexis-hosted-client")
    monkeypatch.setenv("HEXIS_GMAIL_OAUTH_CLIENT_SECRET", "hexis-hosted-secret")
    for name in (
        "GOOGLE_GMAIL_CLIENT_SECRET_JSON",
        "GOOGLE_CLIENT_SECRET_JSON",
        "GOOGLE_GMAIL_CLIENT_SECRET_PATH",
        "GOOGLE_CLIENT_SECRET_PATH",
        "GOOGLE_GMAIL_OAUTH_CLIENT_ID",
        "GOOGLE_GMAIL_OAUTH_CLIENT_SECRET",
        "HEXIS_GMAIL_OAUTH_REDIRECT_URI",
        "GOOGLE_GMAIL_OAUTH_REDIRECT_URI",
        "HEXIS_API_URL",
        "HEXIS_API_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    marker = get_test_identifier("gmail-hosted-oauth")
    result = await ConnectGmailHandler().execute(
        {"capabilities": ["read", "search"], "source_channel": "web"},
        _ctx(db_pool, marker),
    )

    try:
        assert result.success
        assert result.output["status"] == "pending_user"
        assert result.output["ui"]["status"] == "pending_authorization"
        assert result.output["ui"]["client_secret_saved"] is True
        auth_params = parse_qs(urlparse(result.output["authorization_url"]).query)
        assert auth_params["client_id"] == ["hexis-hosted-client"]
        assert load_auth(GMAIL_CLIENT_SECRET_REF)["installed"]["client_secret"] == "hexis-hosted-secret"
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM connection_attempts WHERE source_session_id = $1", marker)


async def test_bundled_gmail_oauth_client_is_available(monkeypatch, tmp_path):
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    monkeypatch.delenv("HEXIS_GMAIL_OAUTH_DISABLE_BUNDLED", raising=False)
    for name in (
        "GOOGLE_GMAIL_CLIENT_SECRET_JSON",
        "GOOGLE_CLIENT_SECRET_JSON",
        "GOOGLE_GMAIL_CLIENT_SECRET_PATH",
        "GOOGLE_CLIENT_SECRET_PATH",
        "HEXIS_GMAIL_OAUTH_CLIENT_ID",
        "HEXIS_GMAIL_OAUTH_CLIENT_SECRET",
        "GOOGLE_GMAIL_OAUTH_CLIENT_ID",
        "GOOGLE_GMAIL_OAUTH_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    assert has_bundled_gmail_oauth_client() is True
    assert has_hexis_gmail_oauth_client() is True


async def test_save_gmail_client_secret_payload_redacts_saved_client(monkeypatch, tmp_path):
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")

    saved = save_gmail_client_secret_payload(_client_secret(), source="test_upload")

    assert saved["status"] == "client_secret_saved"
    assert saved["connector_id"] == "gmail"
    assert saved["client_secret_saved"] is True
    assert saved["client_id_hint"].endswith("t-client")
    assert saved["source"] == "test_upload"
    assert "gmail-test-secret" not in json.dumps(saved)
    assert load_auth(GMAIL_CLIENT_SECRET_REF) == _client_secret()


async def test_connect_gmail_persists_memory_and_heartbeat_policy_as_config(db_pool, monkeypatch, tmp_path):
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    for name in (
        "GOOGLE_GMAIL_CLIENT_SECRET_JSON",
        "GOOGLE_CLIENT_SECRET_JSON",
        "GOOGLE_GMAIL_CLIENT_SECRET_PATH",
        "GOOGLE_CLIENT_SECRET_PATH",
        "HEXIS_GMAIL_OAUTH_CLIENT_ID",
        "HEXIS_GMAIL_OAUTH_CLIENT_SECRET",
        "GOOGLE_GMAIL_OAUTH_CLIENT_ID",
        "GOOGLE_GMAIL_OAUTH_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    try:
        result = await ConnectGmailHandler().execute(
            {
                "capabilities": ["read", "search"],
                "memory_policy": "forget",
                "heartbeat_digest_enabled": True,
            },
            _ctx(db_pool, get_test_identifier("gmail-memory-policy")),
        )

        assert result.success
        assert result.output["status"] == "needs_client_secret"
        assert result.output["ui"]["memory_policy"] == "forget"
        assert result.output["ui"]["heartbeat_digest_enabled"] is True

        async with db_pool.acquire() as conn:
            policy = await conn.fetchval(
                "SELECT get_config_text('integrations.gmail.memory_policy')"
            )
            heartbeat_digest_enabled = await conn.fetchval(
                "SELECT get_config_bool('integrations.gmail.heartbeat_digest_enabled')"
            )
        assert policy == "forget"
        assert heartbeat_digest_enabled is True
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('integrations.gmail.memory_policy', '\"ask\"'::jsonb)"
            )
            await conn.execute(
                "SELECT set_config('integrations.gmail.heartbeat_digest_enabled', 'false'::jsonb)"
            )


async def test_gmail_setup_status_executes_through_registry(db_pool, monkeypatch, tmp_path):
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    for name in (
        "GOOGLE_GMAIL_CLIENT_SECRET_JSON",
        "GOOGLE_CLIENT_SECRET_JSON",
        "GOOGLE_GMAIL_CLIENT_SECRET_PATH",
        "GOOGLE_CLIENT_SECRET_PATH",
        "HEXIS_GMAIL_OAUTH_CLIENT_ID",
        "HEXIS_GMAIL_OAUTH_CLIENT_SECRET",
        "GOOGLE_GMAIL_OAUTH_CLIENT_ID",
        "GOOGLE_GMAIL_OAUTH_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    registry = create_default_registry(db_pool)

    result = await registry.execute(
        "gmail_setup_status",
        {},
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="gmail-status-registry",
            session_id="gmail-status-registry",
        ),
    )

    assert result.success
    assert result.output["connectors"][0]["id"] == "gmail"
    assert result.output["client_secret_saved"] is False
    assert result.output["credentials_saved"] is False
    assert result.output["ui"]["kind"] == "connector_setup"
    assert result.output["ui"]["connector_id"] == "gmail"
    assert result.output["ui"]["status"] == "needs_client_secret"


async def test_connected_gmail_setup_ui_suppresses_stale_authorization_url():
    ui = _gmail_connector_setup_ui(
        {
            "authorization_url": "https://accounts.google.com/stale",
            "attempt_id": "stale-attempt",
            "recent_attempts": [
                {
                    "connector_id": "gmail",
                    "status": "pending_user",
                    "attempt_id": "stale-attempt",
                    "authorization_url": "https://accounts.google.com/stale",
                }
            ],
        },
        connected_accounts=[
            {
                "connector_id": "gmail",
                "account_key": "eric@example.com",
                "status": "connected",
            }
        ],
        credentials_saved=True,
    )

    assert ui["status"] == "connected"
    assert ui["authorization_url"] is None
    assert ui["attempt_id"] is None
    assert ui["completion_mode"] is None
    assert ui["manual_completion_available"] is False


async def test_connect_and_complete_gmail_oauth_round_trip(db_pool, monkeypatch, tmp_path):
    import core.auth.google_gmail as google_gmail
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    for name in (
        "HEXIS_GMAIL_OAUTH_REDIRECT_URI",
        "GOOGLE_GMAIL_OAUTH_REDIRECT_URI",
        "HEXIS_API_URL",
        "HEXIS_API_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    save_auth(GMAIL_CLIENT_SECRET_REF, _client_secret())

    marker = get_test_identifier("gmail-connector")
    email = f"eric-{marker}@example.com"

    async def fake_exchange_code(**kwargs):
        assert kwargs["code"] == "oauth-code"
        assert kwargs["client_id"] == "gmail-test-client"
        assert kwargs["client_secret"] == "gmail-test-secret"
        assert kwargs["redirect_uri"] == GMAIL_REDIRECT_URI
        return {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "scope": (
                "https://www.googleapis.com/auth/userinfo.email "
                "https://www.googleapis.com/auth/gmail.readonly"
            ),
        }

    async def fake_fetch_account_email(access_token: str):
        assert access_token == "access-token"
        return email

    monkeypatch.setattr(google_gmail, "_exchange_code", fake_exchange_code)
    monkeypatch.setattr(google_gmail, "_fetch_account_email", fake_fetch_account_email)

    try:
        started = await ConnectGmailHandler().execute(
            {"capabilities": ["read", "search"], "source_channel": "cli"},
            _ctx(db_pool, marker),
        )

        assert started.success
        assert started.output["connector_id"] == "gmail"
        assert started.output["status"] == "pending_user"
        assert started.output["source_channel"] == "cli"
        assert started.output["source_session_id"] == marker
        assert "authorization_url" in started.output

        auth_params = parse_qs(urlparse(started.output["authorization_url"]).query)
        assert auth_params["client_id"] == ["gmail-test-client"]
        assert auth_params["redirect_uri"] == [GMAIL_REDIRECT_URI]

        attempt_id = started.output["attempt_id"]
        pending = load_auth(f"{GMAIL_PENDING_PREFIX}{attempt_id}")
        assert pending["capabilities"] == ["read", "search"]

        completed = await CompleteGmailConnectionHandler().execute(
            {
                "authorization_response": f"{GMAIL_REDIRECT_URI}/?code=oauth-code&state={pending['state']}",
            },
            _ctx(db_pool, marker),
        )

        assert completed.success
        assert completed.output["status"] == "connected"
        assert completed.output["account_key"] == email
        assert completed.output["capabilities"] == ["read", "search"]

        credentials = load_auth(GMAIL_DEFAULT_CREDENTIAL_REF)
        assert credentials["refresh_token"] == "refresh-token"
        assert credentials["account_email"] == email

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT account_key, status, capabilities, credential_ref
                FROM integration_connections
                WHERE connector_id = 'gmail'
                  AND account_key = $1
                """,
                email,
            )
        assert row is not None
        assert row["status"] == "connected"
        assert row["credential_ref"] == GMAIL_DEFAULT_CREDENTIAL_REF
        assert _j(row["capabilities"]) == ["read", "search"]
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM integration_connections WHERE account_key = $1", email)
            await conn.execute("DELETE FROM connection_attempts WHERE source_session_id = $1", marker)


async def test_gmail_backfill_tools_queue_status_and_control(db_pool, monkeypatch, tmp_path):
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    marker = get_test_identifier("gmail-backfill-tool")
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
    await _seed_connected_gmail(db_pool, marker, account)

    try:
        queued = await StartGmailBackfillHandler().execute(
            {
                "account_key": account,
                "query": "newer_than:7d",
                "label_ids": ["INBOX"],
                "max_messages": 5,
                "source_channel": "cli",
            },
            _ctx(db_pool, marker),
        )
        assert queued.success
        assert queued.output["status"] == "pending"
        assert queued.output["connector_id"] == "gmail"
        assert queued.output["requested_range"]["query"] == "newer_than:7d"

        status = await GmailBackfillStatusHandler().execute(
            {"account_key": account},
            _ctx(db_pool, marker),
        )
        assert status.success
        assert status.output["jobs"][0]["job_id"] == queued.output["job_id"]
        assert "1 active jobs" in status.display_output

        paused = await ControlGmailBackfillHandler().execute(
            {"job_id": queued.output["job_id"], "action": "pause", "reason": "test pause"},
            _ctx(db_pool, marker),
        )
        resumed = await ControlGmailBackfillHandler().execute(
            {"job_id": queued.output["job_id"], "action": "resume"},
            _ctx(db_pool, marker),
        )
        cancelled = await ControlGmailBackfillHandler().execute(
            {"job_id": queued.output["job_id"], "action": "cancel", "reason": "test cancel"},
            _ctx(db_pool, marker),
        )
        assert paused.output["status"] == "paused"
        assert resumed.output["status"] == "pending"
        assert cancelled.output["status"] == "cancelled"
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM connector_backfill_jobs WHERE account_key = $1", account)
            await conn.execute("DELETE FROM connector_sync_cursors WHERE account_key = $1", account)
            await conn.execute("DELETE FROM integration_connections WHERE account_key = $1", account)
            await conn.execute("DELETE FROM connection_attempts WHERE source_session_id = $1", marker)


async def test_connector_action_policy_tools_grant_status_and_revoke(db_pool):
    marker = get_test_identifier("connector-policy-tool")
    ctx = _ctx(db_pool, marker)

    try:
        granted = await GrantConnectorActionPolicyHandler().execute(
            {
                "connector_id": "slack",
                "action_kind": "send",
                "constraints": {"allowed_targets": ["#ops"], "max_per_day": 2},
                "allow_autonomous": True,
                "requires_per_action_approval": False,
                "contexts": ["heartbeat"],
                "rationale": "Test operational alert grant",
            },
            ctx,
        )
        assert granted.success
        assert granted.output["connector_id"] == "slack"
        assert granted.output["allow_autonomous"] is True

        status = await ConnectorActionPolicyStatusHandler().execute(
            {"connector_id": "slack"},
            ctx,
        )
        assert status.success
        assert any(item["policy_id"] == granted.output["policy_id"] for item in status.output["policies"])

        revoked = await RevokeConnectorActionPolicyHandler().execute(
            {"policy_id": granted.output["policy_id"], "reason": "test cleanup"},
            ctx,
        )
        assert revoked.success
        assert revoked.output["status"] == "revoked"
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM connector_action_audit WHERE context->>'session_id' = $1", marker)
            await conn.execute("DELETE FROM connector_action_policies WHERE source_session_id = $1", marker)
