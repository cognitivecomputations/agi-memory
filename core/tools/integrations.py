"""Tools for first-class personal-data connector setup."""

from __future__ import annotations

import json
import re
from typing import Any

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


_CHANNEL_CONNECTORS = {"slack", "telegram", "signal"}
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SECRET_CHANNEL_KEYS = {"bot_token", "app_token", "access_token", "password"}
_GMAIL_DEFAULT_CAPABILITIES = ["read", "search"]
_GMAIL_MEMORY_POLICY_CONFIG_KEY = "integrations.gmail.memory_policy"
_GMAIL_HEARTBEAT_DIGEST_CONFIG_KEY = "integrations.gmail.heartbeat_digest_enabled"
_GMAIL_MEMORY_POLICIES = {"remember", "forget"}
_GMAIL_SETUP_FALLBACK_STEPS = [
    "Open the Google setup page.",
    "Create or choose a project named Hexis.",
    "Enable the Gmail API for that project.",
    "Set up the app consent screen. For a personal Gmail account, choose External and add your Gmail address as a test user if Google asks.",
    "On the Credentials page, click Create credentials, choose Google's sign-in client option, set Application type to Desktop app, and name it Hexis.",
    "Download the setup file Google gives you.",
    "Upload that setup file here, then start Google sign-in.",
]


def _connector_id(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _setup_next_step(plan: dict[str, Any]) -> str:
    manifest = plan.get("setup_manifest") if isinstance(plan, dict) else {}
    if not isinstance(manifest, dict):
        manifest = {}
    step = str(manifest.get("user_next_step") or "").strip()
    if step:
        return step
    notes = manifest.get("notes")
    if isinstance(notes, list) and notes:
        return " ".join(str(item) for item in notes if item)
    return "Follow the connector setup manifest, then verify the connection."


def enrich_gmail_setup_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach runtime Gmail setup availability without exposing secrets."""
    connectors = payload.get("connectors")
    if not isinstance(connectors, list):
        return payload

    try:
        from core.auth.google_gmail import (
            GMAIL_SETUP_STEPS,
            has_env_gmail_client_secret,
            has_hexis_gmail_oauth_client,
        )

        hosted_available = has_hexis_gmail_oauth_client()
        env_available = has_env_gmail_client_secret()
        setup_steps = list(GMAIL_SETUP_STEPS)
    except Exception:
        hosted_available = False
        env_available = False
        setup_steps = list(_GMAIL_SETUP_FALLBACK_STEPS)

    for connector in connectors:
        if not isinstance(connector, dict) or connector.get("id") != "gmail":
            continue
        manifest = _json(connector.get("setup_manifest")) or {}
        if not isinstance(manifest, dict):
            manifest = {}
        connector["setup_manifest"] = {
            **manifest,
            "hosted_oauth_configured": hosted_available,
            "hexis_oauth_client_available": hosted_available,
            "env_client_secret_available": env_available,
            "setup_steps": manifest.get("setup_steps") or setup_steps,
        }
    return payload


def _gmail_capabilities_from_status(payload: dict[str, Any], fallback: Any = None) -> list[str]:
    if isinstance(fallback, list) and fallback:
        return [str(item) for item in fallback if str(item or "").strip()]
    for connector in payload.get("connectors", []):
        if not isinstance(connector, dict) or connector.get("id") != "gmail":
            continue
        manifest = _json(connector.get("setup_manifest")) or {}
        caps = manifest.get("default_capabilities")
        if isinstance(caps, list) and caps:
            return [str(item) for item in caps if str(item or "").strip()]
    return list(_GMAIL_DEFAULT_CAPABILITIES)


def _normalize_gmail_memory_policy(value: Any) -> str | None:
    if value is None:
        return None
    policy = str(value or "").strip().lower().replace("-", "_")
    if policy in {"remember", "learn", "ingest", "retain", "store"}:
        return "remember"
    if policy in {"forget", "ephemeral", "temporary", "task_scoped", "task_scoped_only"}:
        return "forget"
    return None


def _gmail_memory_policy_from_payload(payload: dict[str, Any], fallback: Any = None) -> str | None:
    return _normalize_gmail_memory_policy(
        fallback
        or payload.get("memory_policy")
        or payload.get("gmail_memory_policy")
        or payload.get("integrations.gmail.memory_policy")
    )


async def _persist_gmail_memory_policy(pool: Any, policy: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO config (key, value, description, updated_at)
            VALUES (
                $1,
                to_jsonb($2::text),
                'Controls whether Gmail reads may feed Hexis ingestion and memory by default.',
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                description = COALESCE(config.description, EXCLUDED.description),
                updated_at = EXCLUDED.updated_at
            """,
            _GMAIL_MEMORY_POLICY_CONFIG_KEY,
            policy,
        )


def _gmail_heartbeat_digest_from_payload(payload: dict[str, Any], fallback: Any = None) -> bool | None:
    if fallback is not None:
        value = fallback
    elif "heartbeat_digest_enabled" in payload:
        value = payload.get("heartbeat_digest_enabled")
    elif "autonomous_gmail_digest" in payload:
        value = payload.get("autonomous_gmail_digest")
    elif "enable_heartbeat_digest" in payload:
        value = payload.get("enable_heartbeat_digest")
    else:
        value = payload.get(_GMAIL_HEARTBEAT_DIGEST_CONFIG_KEY)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"true", "1", "yes", "y", "enable", "enabled", "allow", "allowed", "heartbeat_digest"}:
        return True
    if normalized in {"false", "0", "no", "n", "disable", "disabled", "deny", "ask_only", "manual"}:
        return False
    return None


async def _persist_gmail_heartbeat_digest_enabled(pool: Any, enabled: bool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO config (key, value, description, updated_at)
            VALUES (
                $1,
                to_jsonb($2::boolean),
                'Controls whether heartbeat may proactively check connected Gmail for digests or important messages without a live user turn.',
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                description = COALESCE(config.description, EXCLUDED.description),
                updated_at = EXCLUDED.updated_at
            """,
            _GMAIL_HEARTBEAT_DIGEST_CONFIG_KEY,
            enabled,
        )


def _gmail_connector_setup_ui(
    payload: dict[str, Any],
    *,
    status: str | None = None,
    capabilities: Any = None,
    memory_policy: Any = None,
    heartbeat_digest_enabled: Any = None,
    pending_attempt: dict[str, Any] | None = None,
    connected_accounts: list[dict[str, Any]] | None = None,
    client_secret_saved: bool | None = None,
    credentials_saved: bool | None = None,
    next_step: str | None = None,
) -> dict[str, Any]:
    """Build a transport-neutral setup card for chat/CLI surfaces."""
    connected = connected_accounts
    if connected is None:
        connected = [
            item
            for item in payload.get("connections", [])
            if isinstance(item, dict) and item.get("connector_id") == "gmail" and item.get("status") == "connected"
        ]
    pending = pending_attempt
    if pending is None:
        for item in payload.get("recent_attempts", []):
            if not isinstance(item, dict) or item.get("connector_id") != "gmail":
                continue
            if item.get("status") in {"pending_user", "awaiting_input", "pending", "in_progress", "error"}:
                pending = item
                break

    resolved_status = status
    if not resolved_status:
        has_client_secret = (
            bool(client_secret_saved)
            if client_secret_saved is not None
            else bool(payload.get("client_secret_saved"))
        )
        if connected:
            resolved_status = "connected"
        elif pending:
            resolved_status = "pending_authorization"
        elif has_client_secret:
            resolved_status = "client_secret_saved"
        else:
            resolved_status = "needs_client_secret"
    if connected:
        pending = None
        resolved_status = "connected"

    caps = _gmail_capabilities_from_status(payload, capabilities)
    resolved_memory_policy = _gmail_memory_policy_from_payload(payload, memory_policy)
    resolved_heartbeat_digest_enabled = _gmail_heartbeat_digest_from_payload(
        payload,
        heartbeat_digest_enabled,
    )
    try:
        from core.auth.google_gmail import (
            GMAIL_SETUP_DOCS_URL,
            GMAIL_SETUP_NEXT_STEP,
            GMAIL_SETUP_STEPS,
            GMAIL_SETUP_TECHNICAL_NEXT_STEP,
        )
    except Exception:
        GMAIL_SETUP_DOCS_URL = "https://console.cloud.google.com/apis/credentials"
        GMAIL_SETUP_NEXT_STEP = (
            "Gmail sign-in is not enabled for this local Hexis build yet. "
            "Use the setup guide in the panel to enable it once, then start Google sign-in."
        )
        GMAIL_SETUP_STEPS = list(_GMAIL_SETUP_FALLBACK_STEPS)
        GMAIL_SETUP_TECHNICAL_NEXT_STEP = (
            "For developers and hosted builds: configure HEXIS_GMAIL_OAUTH_CLIENT_ID and "
            "HEXIS_GMAIL_OAUTH_CLIENT_SECRET in the Hexis API environment to make built-in "
            "Google sign-in available."
        )

    docs_url = GMAIL_SETUP_DOCS_URL
    authorization_url = None
    attempt_id = None
    pending_next_step = None
    if pending and resolved_status != "connected":
        authorization_url = pending.get("authorization_url")
        attempt_id = pending.get("attempt_id") or pending.get("id")
        pending_next_step = pending.get("user_next_step") or pending.get("next_step")
    if resolved_status != "connected":
        authorization_url = authorization_url or payload.get("authorization_url")
        attempt_id = attempt_id or payload.get("attempt_id")
    try:
        from core.auth.google_gmail import has_env_gmail_client_secret, has_hexis_gmail_oauth_client

        hexis_oauth_client_available = has_hexis_gmail_oauth_client()
        env_client_secret_available = has_env_gmail_client_secret()
    except Exception:
        hexis_oauth_client_available = bool(payload.get("hexis_oauth_client_available"))
        env_client_secret_available = bool(payload.get("env_client_secret_available"))
    saved_client_secret = (
        bool(client_secret_saved) if client_secret_saved is not None else bool(payload.get("client_secret_saved"))
    )
    resolved_next_step = next_step or pending_next_step or payload.get("next_step") or payload.get("user_next_step")
    if not resolved_next_step and resolved_status in {"needs_client_secret", "setup"}:
        resolved_next_step = GMAIL_SETUP_NEXT_STEP
    credential_step = {
        "status": "saved" if saved_client_secret else "needs_client",
        "preferred_mode": (
            "saved"
            if saved_client_secret
            else "hosted_oauth"
            if hexis_oauth_client_available
            else "hosted_oauth_missing"
        ),
        "save_action": "save_gmail_client_secret",
        "modes": [
            {
                "id": "hosted_oauth",
                "label": "Built-in Google sign-in",
                "available": hexis_oauth_client_available,
                "description": (
                    "Connect Gmail with the sign-in flow already configured for this Hexis build."
                    if hexis_oauth_client_available
                    else "This local build needs one-time Google setup before built-in sign-in can be used."
                ),
            },
            {
                "id": "advanced_self_hosted",
                "label": "Walk me through setup",
                "available": True,
                "description": (
                    "Step-by-step setup for a local or self-hosted Hexis install."
                ),
            },
            {
                "id": "upload_json",
                "label": "Upload Google setup file",
                "available": True,
                "description": (
                    "Pick the downloaded Google setup file. Hexis saves it privately and never sends it to the model."
                ),
            },
            {
                "id": "path",
                "label": "Use downloaded setup file path",
                "available": True,
                "description": "CLI fallback for the downloaded setup file already on this machine.",
            },
            {
                "id": "configured_env",
                "label": "Use server-provided setup file",
                "available": env_client_secret_available,
                "description": (
                    "Use GOOGLE_GMAIL_CLIENT_SECRET_PATH/JSON or GOOGLE_CLIENT_SECRET_PATH/JSON "
                    "only after the user explicitly selects this developer mode."
                ),
            },
        ],
    }

    connected_status = resolved_status == "connected"

    return {
        "kind": "connector_setup",
        "version": 1,
        "id": f"connector_setup:gmail:{attempt_id or resolved_status}",
        "connector_id": "gmail",
        "display_name": "Gmail",
        "title": "Gmail Connected" if connected_status else "Connect Gmail",
        "status": resolved_status,
        "summary": (
            "Gmail is connected. Samantha can use only the email powers you approved."
            if connected_status
            else "Connect Gmail after you approve access with Google. Samantha only gets the email powers you choose."
        ),
        "capabilities": caps,
        "memory_policy": resolved_memory_policy,
        "memory_config_key": _GMAIL_MEMORY_POLICY_CONFIG_KEY,
        "heartbeat_digest_enabled": bool(resolved_heartbeat_digest_enabled),
        "heartbeat_digest_config_key": _GMAIL_HEARTBEAT_DIGEST_CONFIG_KEY,
        "autonomy_options": [
            {
                "id": "ask_only",
                "label": "Only when I ask",
                "description": (
                    "Samantha can read/search Gmail during a live request, but heartbeats will not check email on their own."
                ),
                "heartbeat_digest_enabled": False,
            },
            {
                "id": "heartbeat_digest",
                "label": "Allow heartbeat checks",
                "description": (
                    "Samantha may check connected Gmail during hourly heartbeats for important messages and digests."
                ),
                "heartbeat_digest_enabled": True,
            },
        ],
        "client_secret_saved": saved_client_secret,
        "credentials_saved": (
            bool(credentials_saved) if credentials_saved is not None else bool(payload.get("credentials_saved"))
        ),
        "accepted_inputs": [
            "hexis_oauth_client",
            "advanced_self_hosted",
            "client_secret_file",
            "client_secret_path",
            "use_env_client_secret",
        ],
        "hexis_oauth_client_available": hexis_oauth_client_available,
        "env_client_secret_available": env_client_secret_available,
        "credential_step": credential_step,
        "credential_step_label": (
            "connected"
            if connected_status
            else "Google sign-in ready"
            if saved_client_secret or hexis_oauth_client_available
            else "Google setup needed"
        ),
        "docs_url": docs_url,
        "setup_steps": list(GMAIL_SETUP_STEPS),
        "technical_next_step": GMAIL_SETUP_TECHNICAL_NEXT_STEP,
        "authorization_url": authorization_url,
        "attempt_id": attempt_id,
        "completion_mode": "automatic_with_manual_fallback" if attempt_id and resolved_status != "connected" else None,
        "manual_completion_available": bool(attempt_id and resolved_status != "connected"),
        "connected_accounts": connected,
        "next_step": resolved_next_step,
        "primary_action": {
            "action": "connect_gmail",
            "label": "Start Google sign-in",
            "arguments": {
                "capabilities": caps,
                **({"memory_policy": resolved_memory_policy} if resolved_memory_policy else {}),
                "heartbeat_digest_enabled": bool(resolved_heartbeat_digest_enabled),
            },
        },
        "completion_action": {
            "action": "complete_gmail",
            "label": "Complete connection",
            "arguments": {"attempt_id": attempt_id} if attempt_id else {},
        },
        "safety_note": (
            "Google controls the provider permissions. Email remembering is a Hexis memory setting. "
            "Sending, replying, labeling, spam triage, and delete actions require explicit later authorization."
        ),
    }


async def _connected_gmail_accounts(pool: Any) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT account_key, display_name, capabilities, granted_scopes, updated_at
            FROM integration_connections
            WHERE connector_id = 'gmail'
              AND status = 'connected'
            ORDER BY updated_at DESC, account_key
            """
        )
    accounts: list[dict[str, Any]] = []
    for row in rows:
        accounts.append(
            {
                "account_key": row["account_key"],
                "display_name": row["display_name"],
                "capabilities": _json(row["capabilities"]) or [],
                "granted_scopes": list(row["granted_scopes"] or []),
                "updated_at": row["updated_at"],
            }
        )
    return accounts


async def _resolve_gmail_account(pool: Any, requested: Any = None) -> str:
    account_key = str(requested or "").strip().lower()
    accounts = await _connected_gmail_accounts(pool)
    if account_key:
        if any(str(item.get("account_key") or "").lower() == account_key for item in accounts):
            return account_key
        raise ValueError(f"Gmail account is not connected: {account_key}")
    if not accounts:
        raise ValueError("Gmail is not connected. Use connect_gmail first.")
    if len(accounts) == 1:
        return str(accounts[0]["account_key"])
    choices = ", ".join(str(item["account_key"]) for item in accounts)
    raise ValueError(f"Multiple Gmail accounts are connected. Specify account_key. Connected: {choices}")


async def _connected_accounts(pool: Any, connector_id: str) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT account_key, display_name, capabilities, granted_scopes, updated_at
            FROM integration_connections
            WHERE connector_id = $1
              AND status = 'connected'
            ORDER BY updated_at DESC, account_key
            """,
            connector_id,
        )
    return [
        {
            "account_key": row["account_key"],
            "display_name": row["display_name"],
            "capabilities": _json(row["capabilities"]) or [],
            "granted_scopes": list(row["granted_scopes"] or []),
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


async def _resolve_connector_account(pool: Any, connector_id: str, requested: Any = None) -> str:
    account_key = str(requested or "").strip()
    accounts = await _connected_accounts(pool, connector_id)
    if account_key:
        if any(str(item.get("account_key") or "") == account_key for item in accounts):
            return account_key
        raise ValueError(f"{connector_id} account is not connected: {account_key}")
    if not accounts:
        raise ValueError(f"{connector_id} is not connected. Start and verify the connector setup first.")
    if len(accounts) == 1:
        return str(accounts[0]["account_key"])
    choices = ", ".join(str(item["account_key"]) for item in accounts)
    raise ValueError(f"Multiple {connector_id} accounts are connected. Specify account_key. Connected: {choices}")


async def _ensure_twitter_archive_connection(pool: Any, account_key: str, source_session_id: str | None) -> str:
    account_key = account_key.strip() or "archive:twitter_x"
    async with pool.acquire() as conn:
        existing = await conn.fetchval(
            """
            SELECT account_key
            FROM integration_connections
            WHERE connector_id = 'twitter_x'
              AND account_key = $1
              AND status = 'connected'
            """,
            account_key,
        )
        if existing:
            return str(existing)
        attempt_raw = await conn.fetchval(
            """
            SELECT start_connection_attempt(
                'twitter_x',
                '["archive_import"]'::jsonb,
                ARRAY[]::text[],
                '{"setup_kind": "local_archive_import", "auth_type": "local_export"}'::jsonb,
                NULL,
                'Archive connection will be created for local Twitter/X history import.',
                'tool',
                $1,
                CURRENT_TIMESTAMP + INTERVAL '10 minutes'
            )
            """,
            source_session_id,
        )
        attempt = _json(attempt_raw) or {}
        await conn.fetchval(
            """
            SELECT complete_connection_attempt(
                $1::uuid,
                $2,
                'Twitter/X archive',
                'local_export:twitter_x_archive',
                ARRAY[]::text[],
                '["archive_import"]'::jsonb,
                '{"verified_by": "start_connector_backfill", "setup_kind": "local_archive_import", "secret_values_stored": false}'::jsonb
            )
            """,
            attempt["attempt_id"],
            account_key,
        )
    return account_key


class IntegrationSetupStatusHandler(ToolHandler):
    """Inspect first-class connector setup state for any provider."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="integration_setup_status",
            description=(
                "Show connector setup state for Gmail, Slack, Telegram, Signal, Twitter/X, "
                "and other registered integrations. Does not expose secrets."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {
                        "type": "string",
                        "description": "Optional connector filter, such as gmail, slack, telegram, signal, or twitter_x.",
                    }
                },
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "integration_setup_status requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        connector_id = _connector_id(arguments.get("connector_id")) or None
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval("SELECT integration_status($1)", connector_id)
            runtime_raw = await conn.fetchval(
                "SELECT list_channel_adapter_status($1)",
                connector_id if connector_id in _CHANNEL_CONNECTORS else None,
            )
        payload = _json(raw) or {}
        enrich_gmail_setup_runtime(payload)
        from services.life_integrations import enrich_life_setup_status

        enrich_life_setup_status(payload, connector_id)
        payload["channel_runtime"] = _json(runtime_raw) or []
        connectors = payload.get("connectors", [])
        connected = payload.get("connections", [])
        if connector_id and connectors:
            label = connectors[0].get("display_name") or connector_id
            status = connectors[0].get("status")
            count = len([item for item in connected if item.get("status") == "connected"])
            display = f"{label}: {status}; connected accounts: {count}"
            runtime = payload["channel_runtime"]
            if runtime:
                display += f"; adapter: {runtime[0].get('status')}"
        else:
            available = [item["id"] for item in connectors if item.get("status") == "available"]
            planned = [item["id"] for item in connectors if item.get("status") == "planned"]
            display = f"Available connectors: {', '.join(available) or 'none'}"
            if planned:
                display += f"; planned: {', '.join(planned)}"
        return ToolResult.success_result(payload, display_output=display)


class StartIntegrationSetupHandler(ToolHandler):
    """Start a DB-owned setup attempt for non-Gmail connectors."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="start_integration_setup",
            description=(
                "Start a first-class connector setup attempt for manual/pairing/API-key channels "
                "such as Slack, Telegram, or Signal. Gmail sign-in uses connect_gmail; Twitter/X "
                "OAuth uses connect_twitter_x. Twitter/X archive import uses start_connector_backfill "
                "after the account/archive path is available."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {
                        "type": "string",
                        "description": "Connector id, such as slack, telegram, signal, or twitter_x.",
                    },
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional requested capabilities. Defaults come from the DB connector manifest.",
                    },
                    "source_channel": {
                        "type": "string",
                        "description": "Optional source surface, such as cli, web, slack, telegram, or signal.",
                    },
                    "source_session_id": {
                        "type": "string",
                        "description": "Optional conversation/session identifier for resuming setup.",
                    },
                    "account_key": {
                        "type": "string",
                        "description": "Optional local account key for local-export connectors. Defaults to archive:<connector_id>.",
                    },
                },
                "required": ["connector_id"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "start_integration_setup requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        connector_id = _connector_id(arguments.get("connector_id"))
        if connector_id == "gmail":
            return ToolResult.error_result(
                "Use connect_gmail for Gmail setup.",
                ToolErrorType.INVALID_PARAMS,
            )
        if connector_id == "twitter_x":
            return ToolResult.error_result(
                "Use connect_twitter_x for Twitter/X OAuth setup. Use start_connector_backfill with export_path for archive import.",
                ToolErrorType.INVALID_PARAMS,
            )
        requested = arguments.get("capabilities")
        requested_json = json.dumps(requested) if requested is not None else None

        try:
            async with context.registry.pool.acquire() as conn:
                plan_raw = await conn.fetchval(
                    "SELECT prepare_connection_attempt($1, $2::jsonb)",
                    connector_id,
                    requested_json,
                )
                plan = _json(plan_raw) or {}
                next_step = _setup_next_step(plan)
                attempt_raw = await conn.fetchval(
                    """
                    SELECT start_connection_attempt(
                        $1,
                        $2::jsonb,
                        $3::text[],
                        $4::jsonb,
                        NULL,
                        $5,
                        $6,
                        $7,
                        NULL
                    )
                    """,
                    connector_id,
                    requested_json or json.dumps(plan.get("capabilities") or []),
                    [str(item) for item in plan.get("requested_scopes", [])],
                    json.dumps({"setup_kind": "manual_channel", "auth_type": plan.get("auth_type")}),
                    next_step,
                    arguments.get("source_channel"),
                    arguments.get("source_session_id") or context.session_id,
                )
        except Exception as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)

        payload = _json(attempt_raw) or {}
        payload["setup_plan"] = plan
        payload["next_step"] = payload.get("user_next_step") or next_step
        return ToolResult.success_result(payload, display_output=payload["next_step"])


class ConfigureChannelIntegrationHandler(ToolHandler):
    """Write non-secret channel connector config through the DB catalog."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="configure_channel_integration",
            description=(
                "Configure Slack, Telegram, or Signal channel settings using DB-owned channel config. "
                "Use env var names for tokens; do not paste token values because tool calls are audited."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {
                        "type": "string",
                        "description": "Channel connector id: slack, telegram, or signal.",
                    },
                    "settings": {
                        "type": "object",
                        "description": (
                            "Channel settings accepted by apply_channel_config. Token fields must be env var names, "
                            "for example {'bot_token': 'TELEGRAM_BOT_TOKEN'}."
                        ),
                    },
                },
                "required": ["connector_id", "settings"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "configure_channel_integration requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        connector_id = _connector_id(arguments.get("connector_id"))
        if connector_id not in _CHANNEL_CONNECTORS:
            return ToolResult.error_result(
                "configure_channel_integration supports slack, telegram, and signal.",
                ToolErrorType.INVALID_PARAMS,
            )
        settings = arguments.get("settings")
        if not isinstance(settings, dict) or not settings:
            return ToolResult.error_result("settings must be a non-empty object.", ToolErrorType.INVALID_PARAMS)

        for key, value in settings.items():
            if key in _SECRET_CHANNEL_KEYS:
                if not isinstance(value, str) or not _ENV_NAME_RE.fullmatch(value):
                    return ToolResult.error_result(
                        f"{key} must be an environment variable name, not a token value.",
                        ToolErrorType.INVALID_PARAMS,
                    )

        try:
            async with context.registry.pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT apply_channel_config($1, $2::jsonb)",
                    connector_id,
                    json.dumps(settings),
                )
        except Exception as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)

        payload = _json(raw) or {}
        return ToolResult.success_result(
            payload,
            display_output=f"{connector_id} channel config applied: {', '.join(payload.get('applied', []))}.",
        )


class VerifyChannelIntegrationHandler(ToolHandler):
    """Verify channel config and mark the connector connected."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="verify_channel_integration",
            description=(
                "Verify that Slack, Telegram, or Signal channel config resolves the credentials expected "
                "by the channel worker, then mark the connector connected in the DB."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {
                        "type": "string",
                        "description": "Channel connector id: slack, telegram, or signal.",
                    },
                    "attempt_id": {
                        "type": "string",
                        "description": "Optional setup attempt ID. Defaults to latest active attempt for the connector.",
                    },
                    "account_key": {
                        "type": "string",
                        "description": "Optional redacted account key. Defaults to channel:<connector_id>.",
                    },
                },
                "required": ["connector_id"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "verify_channel_integration requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        connector_id = _connector_id(arguments.get("connector_id"))
        if connector_id not in _CHANNEL_CONNECTORS:
            return ToolResult.error_result(
                "verify_channel_integration supports slack, telegram, and signal.",
                ToolErrorType.INVALID_PARAMS,
            )

        from services.channel_worker import _is_channel_configured, _load_channel_config

        async with context.registry.pool.acquire() as conn:
            config = await _load_channel_config(conn, connector_id)
            configured = _is_channel_configured(connector_id, config)
            if not configured:
                raw_status = await conn.fetchval("SELECT integration_status($1)", connector_id)
                status = _json(raw_status) or {}
                connectors = status.get("connectors") or []
                plan = connectors[0].get("setup_manifest", {}) if connectors else {}
                next_step = plan.get("user_next_step") or "Configure the channel token env vars, then verify again."
                return ToolResult.error_result(next_step, ToolErrorType.MISSING_CONFIG)

            attempt_id = str(arguments.get("attempt_id") or "").strip()
            if not attempt_id:
                attempt_id = await conn.fetchval(
                    """
                    SELECT id::text
                    FROM connection_attempts
                    WHERE connector_id = $1
                      AND status IN ('pending_user', 'awaiting_input', 'error')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    connector_id,
                )
            if not attempt_id:
                started = _json(await conn.fetchval(
                    """
                    SELECT start_connection_attempt(
                        $1,
                        NULL,
                        ARRAY[]::text[],
                        '{"setup_kind": "verified_existing_channel"}'::jsonb,
                        NULL,
                        'Existing channel configuration verified.',
                        'chat',
                        $2,
                        NULL
                    )
                    """,
                    connector_id,
                    context.session_id,
                ))
                attempt_id = started["attempt_id"]

            plan = _json(await conn.fetchval(
                "SELECT prepare_connection_attempt($1, NULL)",
                connector_id,
            )) or {}
            account_key = str(arguments.get("account_key") or "").strip() or f"channel:{connector_id}"
            if connector_id == "signal":
                try:
                    from channels.signal_adapter import _resolve_token

                    phone = _resolve_token(config)
                    if phone:
                        account_key = phone
                except Exception:
                    pass

            completed = _json(await conn.fetchval(
                """
                SELECT complete_connection_attempt(
                    $1::uuid,
                    $2,
                    $3,
                    $4,
                    $5::text[],
                    $6::jsonb,
                    $7::jsonb
                )
                """,
                attempt_id,
                account_key,
                plan.get("display_name") or connector_id,
                f"config:channel.{connector_id}",
                [str(item) for item in plan.get("requested_scopes", [])],
                json.dumps(plan.get("capabilities") or []),
                json.dumps({
                    "verified_by": "verify_channel_integration",
                    "channel_config_keys": sorted(config.keys()),
                    "runtime": "hexis-channels",
                    "secret_values_stored": False,
                }),
            ))

        completed["next_step"] = "Start or restart hexis-channels if the adapter is not already running."
        return ToolResult.success_result(
            completed,
            display_output=f"{plan.get('display_name') or connector_id} verified. {completed['next_step']}",
        )


class GmailSetupStatusHandler(ToolHandler):
    """Inspect Gmail connector setup state without exposing secrets."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="gmail_setup_status",
            description=(
                "Show Gmail connector setup status, granted capabilities, pending sign-in attempts, "
                "and whether local credential files exist. Does not expose secrets."
            ),
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.EMAIL,
            energy_cost=1,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "Gmail setup status requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        from core.auth.google_gmail import (
            has_hexis_gmail_oauth_client,
            has_saved_gmail_client_secret,
            load_default_credentials,
        )

        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval("SELECT integration_status('gmail')")
            memory_policy = await conn.fetchval(
                "SELECT COALESCE(get_config_text($1), 'ask')",
                _GMAIL_MEMORY_POLICY_CONFIG_KEY,
            )
            heartbeat_digest_enabled = await conn.fetchval(
                "SELECT COALESCE(get_config_bool($1), FALSE)",
                _GMAIL_HEARTBEAT_DIGEST_CONFIG_KEY,
            )
        payload = _json(raw) or {}
        payload["client_secret_saved"] = has_saved_gmail_client_secret()
        payload["hexis_oauth_client_available"] = has_hexis_gmail_oauth_client()
        payload["credentials_saved"] = load_default_credentials() is not None
        payload["memory_policy"] = memory_policy
        payload["heartbeat_digest_enabled"] = bool(heartbeat_digest_enabled)

        connected = [
            item
            for item in payload.get("connections", [])
            if isinstance(item, dict) and item.get("status") == "connected"
        ]
        pending = [
            item
            for item in payload.get("recent_attempts", [])
            if isinstance(item, dict) and item.get("status") in {"pending_user", "awaiting_input", "error"}
        ]
        display = "Gmail: "
        if connected:
            accounts = ", ".join(str(item.get("account_key")) for item in connected)
            display += f"connected ({accounts})"
        elif pending:
            display += "setup pending"
        elif payload["client_secret_saved"]:
            display += "Google setup saved; no account connected"
        elif payload["hexis_oauth_client_available"]:
            display += "Google sign-in ready; no account connected"
        else:
            display += "not connected"

        payload["ui"] = _gmail_connector_setup_ui(
            payload,
            memory_policy=memory_policy,
            heartbeat_digest_enabled=heartbeat_digest_enabled,
            client_secret_saved=payload["client_secret_saved"],
            credentials_saved=payload["credentials_saved"],
        )
        return ToolResult.success_result(payload, display_output=display)


class ConnectGmailHandler(ToolHandler):
    """Start a Gmail sign-in connection attempt."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="connect_gmail",
            description=(
                "Start the Gmail connector setup flow. Use when the user asks to connect Gmail, "
                "authorize email reading/search, label or spam triage, sending/replying, or delete powers. "
                "Ask separately whether Gmail reads should be remembered; memory_policy is a Hexis config "
                "choice, not a Google permission. "
                "Ask separately whether heartbeat may proactively check Gmail while the user is away; "
                "heartbeat_digest_enabled is a Hexis autonomy config choice and defaults off. "
                "Use the built-in Google sign-in path when configured. If it is not configured, "
                "the structured setup UI walks the user through the one-time local setup."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "capabilities": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": (
                            "Gmail provider capabilities to request. Default is read/search. "
                            "Only include label/spam_triage/send/reply/delete when the user asked for those powers. "
                            "The database connector manifest validates names, aliases, and required scopes."
                        ),
                    },
                    "memory_policy": {
                        "type": "string",
                        "enum": ["remember", "forget"],
                        "description": (
                            "Hexis-side policy for whether email contents may feed ingestion/memory by default. "
                            "This is stored in config and is not sent to Google as a provider permission."
                        ),
                    },
                    "heartbeat_digest_enabled": {
                        "type": "boolean",
                        "description": (
                            "Hexis-side autonomy grant for background heartbeat Gmail checks. "
                            "Default is false. Only set true after the user explicitly authorizes "
                            "Samantha to check Gmail while they are away."
                        ),
                    },
                    "client_secret_path": {
                        "type": "string",
                        "description": "Local path to the downloaded Google setup JSON file.",
                    },
                    "use_env_client_secret": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Use GOOGLE_GMAIL_CLIENT_SECRET_PATH/JSON or GOOGLE_CLIENT_SECRET_PATH/JSON. "
                            "Only set true after the user explicitly asks to use a server-provided setup file."
                        ),
                    },
                    "use_hexis_oauth_client": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Use the product-provided Gmail sign-in client when configured. "
                            "This is the default user-friendly path and still shows Google's consent screen."
                        ),
                    },
                    "source_channel": {
                        "type": "string",
                        "description": "Optional source surface, such as cli, web, slack, telegram, or signal.",
                    },
                    "source_session_id": {
                        "type": "string",
                        "description": "Optional conversation/session identifier for resuming setup.",
                    },
                },
            },
            category=ToolCategory.EMAIL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "connect_gmail requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        from core.auth.google_gmail import GmailOAuthError, GmailOAuthStart, start_gmail_oauth

        memory_policy_raw = arguments.get("memory_policy")
        memory_policy = _normalize_gmail_memory_policy(memory_policy_raw)
        if memory_policy_raw is not None and memory_policy not in _GMAIL_MEMORY_POLICIES:
            return ToolResult.error_result(
                "memory_policy must be remember or forget.",
                ToolErrorType.INVALID_PARAMS,
            )
        if memory_policy:
            try:
                await _persist_gmail_memory_policy(context.registry.pool, memory_policy)
            except Exception as exc:
                return ToolResult.error_result(
                    f"Could not save Gmail memory policy: {exc}",
                    ToolErrorType.EXECUTION_FAILED,
                )

        heartbeat_digest_raw = None
        if "heartbeat_digest_enabled" in arguments:
            heartbeat_digest_raw = arguments["heartbeat_digest_enabled"]
        elif "autonomous_gmail_digest" in arguments:
            heartbeat_digest_raw = arguments.get("autonomous_gmail_digest")
        elif "enable_heartbeat_digest" in arguments:
            heartbeat_digest_raw = arguments.get("enable_heartbeat_digest")
        heartbeat_digest_enabled = _gmail_heartbeat_digest_from_payload(
            {},
            heartbeat_digest_raw,
        )
        if heartbeat_digest_raw is not None and heartbeat_digest_enabled is None:
            return ToolResult.error_result(
                "heartbeat_digest_enabled must be a boolean.",
                ToolErrorType.INVALID_PARAMS,
            )
        if heartbeat_digest_enabled is not None:
            try:
                await _persist_gmail_heartbeat_digest_enabled(
                    context.registry.pool,
                    heartbeat_digest_enabled,
                )
            except Exception as exc:
                return ToolResult.error_result(
                    f"Could not save Gmail heartbeat authorization: {exc}",
                    ToolErrorType.EXECUTION_FAILED,
                )

        try:
            started = await start_gmail_oauth(
                context.registry.pool,
                capabilities=arguments.get("capabilities"),
                client_secret_path=arguments.get("client_secret_path"),
                use_hexis_oauth_client=bool(arguments.get("use_hexis_oauth_client", True)),
                use_env_client_secret=bool(arguments.get("use_env_client_secret", False)),
                source_channel=arguments.get("source_channel"),
                source_session_id=arguments.get("source_session_id") or context.session_id,
            )
        except GmailOAuthError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.MISSING_CONFIG)

        if isinstance(started, dict):
            started["memory_policy"] = memory_policy
            started["heartbeat_digest_enabled"] = bool(heartbeat_digest_enabled)
            started["ui"] = _gmail_connector_setup_ui(
                started,
                status=str(started.get("status") or "needs_client_secret"),
                capabilities=arguments.get("capabilities"),
                memory_policy=memory_policy,
                heartbeat_digest_enabled=heartbeat_digest_enabled,
                client_secret_saved=bool(started.get("client_secret_saved")),
                credentials_saved=False,
                next_step=started.get("next_step"),
            )
            return ToolResult.success_result(
                started,
                display_output=started.get("next_step"),
            )

        assert isinstance(started, GmailOAuthStart)
        payload = started.attempt_payload
        payload["memory_policy"] = memory_policy
        payload["heartbeat_digest_enabled"] = bool(heartbeat_digest_enabled)
        payload["ui"] = _gmail_connector_setup_ui(
            payload,
            status="pending_authorization",
            capabilities=payload.get("requested_capabilities") or arguments.get("capabilities"),
            memory_policy=memory_policy,
            heartbeat_digest_enabled=heartbeat_digest_enabled,
            pending_attempt=payload,
            client_secret_saved=True,
            credentials_saved=False,
            next_step=payload.get("user_next_step") or payload.get("next_step"),
        )
        display = (
            "Google sign-in started for Gmail.\n"
            f"Attempt: {payload['attempt_id']}\n"
            f"{payload['authorization_url']}\n\n"
            "Approve access in Google. Hexis will finish the connection automatically "
            "when the browser returns to the local callback."
        )
        return ToolResult.success_result(payload, display_output=display)


class CompleteGmailConnectionHandler(ToolHandler):
    """Complete a pending Gmail sign-in attempt from the pasted redirect URL."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="complete_gmail_connection",
            description=(
                "Complete Gmail sign-in setup from a Google callback URL or raw authorization code. "
                "Normally the local Hexis callback completes this automatically; use this as a fallback."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "authorization_response": {
                        "type": "string",
                        "description": "Full Google callback URL or raw authorization code.",
                    },
                    "attempt_id": {
                        "type": "string",
                        "description": "Optional Gmail connection attempt ID. Defaults to latest pending Gmail attempt.",
                    },
                },
                "required": ["authorization_response"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "complete_gmail_connection requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        from core.auth.google_gmail import GmailOAuthError, complete_gmail_oauth

        try:
            completed = await complete_gmail_oauth(
                context.registry.pool,
                authorization_response=str(arguments.get("authorization_response") or ""),
                attempt_id=arguments.get("attempt_id"),
            )
        except GmailOAuthError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.AUTH_FAILED)

        output = {
            "connector_id": "gmail",
            "status": "connected",
            "account_key": completed.account_key,
            "display_name": completed.display_name,
            "credential_ref": completed.credential_ref,
            "granted_scopes": completed.granted_scopes,
            "capabilities": completed.capabilities,
        }
        try:
            async with context.registry.pool.acquire() as conn:
                memory_policy = await conn.fetchval(
                    "SELECT COALESCE(get_config_text($1), 'ask')",
                    _GMAIL_MEMORY_POLICY_CONFIG_KEY,
                )
                heartbeat_digest_enabled = await conn.fetchval(
                    "SELECT COALESCE(get_config_bool($1), FALSE)",
                    _GMAIL_HEARTBEAT_DIGEST_CONFIG_KEY,
                )
        except Exception:
            memory_policy = None
            heartbeat_digest_enabled = False
        output["memory_policy"] = memory_policy
        output["heartbeat_digest_enabled"] = bool(heartbeat_digest_enabled)
        output["ui"] = _gmail_connector_setup_ui(
            output,
            status="connected",
            capabilities=completed.capabilities,
            memory_policy=memory_policy,
            heartbeat_digest_enabled=heartbeat_digest_enabled,
            connected_accounts=[
                {
                    "account_key": completed.account_key,
                    "display_name": completed.display_name,
                    "capabilities": completed.capabilities,
                    "granted_scopes": completed.granted_scopes,
                    "status": "connected",
                }
            ],
            client_secret_saved=True,
            credentials_saved=True,
        )
        return ToolResult.success_result(
            output,
            display_output=f"Gmail connected for {completed.display_name}.",
        )


class RevokeGmailConnectionHandler(ToolHandler):
    """Disconnect Gmail locally and mark DB connection state revoked."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="revoke_gmail_connection",
            description=(
                "Disconnect Gmail from Hexis by deleting the local credential file and marking the "
                "connection revoked. The user can also remove the Google-side grant in their Google Account permissions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional Gmail account key/email. If omitted, revokes local default Gmail credentials.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional user-visible revocation reason.",
                    },
                },
            },
            category=ToolCategory.EMAIL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "revoke_gmail_connection requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        from core.auth.google_gmail import delete_default_credentials

        delete_default_credentials()
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT revoke_integration_connection('gmail', $1, $2)",
                arguments.get("account_key"),
                arguments.get("reason") or "revoked by user request",
            )
        payload = _json(raw) or {}
        payload["local_credentials_deleted"] = True
        payload["remote_revocation"] = "not_attempted"
        payload["next_step"] = (
            "Local Gmail credentials are removed. To remove provider-side access too, "
            "open your Google Account security settings and remove Hexis from third-party access."
        )
        return ToolResult.success_result(
            payload,
            display_output="Gmail disconnected locally.",
        )


class ConnectTwitterXHandler(ToolHandler):
    """Start a Twitter/X OAuth connection attempt."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="connect_twitter_x",
            description=(
                "Start the Twitter/X connector OAuth setup flow. Use when the user asks to connect "
                "Twitter/X, read posts or mentions, search recent posts, import live X history, "
                "read DMs, post, reply, or send DMs. Request only capabilities the user asked for."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Twitter/X grants to request. Default is read/search/ingest. Include dm_read, "
                            "send, or dm_send only when the user asked for those powers. The database "
                            "connector manifest validates names, aliases, and required scopes."
                        ),
                    },
                    "client_id": {
                        "type": "string",
                        "description": "X OAuth 2.0 Client ID from the X Developer Console.",
                    },
                    "client_secret": {
                        "type": "string",
                        "description": "Optional X OAuth 2.0 Client Secret for confidential apps. Do not paste unless the user explicitly chooses to store it locally.",
                    },
                    "use_env_client": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Use TWITTER_X_CLIENT_ID/X_CLIENT_ID and optional secret env vars. Only set true "
                            "after the user explicitly asks to use environment-provided client credentials."
                        ),
                    },
                    "source_channel": {
                        "type": "string",
                        "description": "Optional source surface, such as cli, web, slack, telegram, or signal.",
                    },
                    "source_session_id": {
                        "type": "string",
                        "description": "Optional conversation/session identifier for resuming setup.",
                    },
                },
            },
            category=ToolCategory.MESSAGING,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "connect_twitter_x requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        from core.auth.twitter_x import TwitterXOAuthError, TwitterXOAuthStart, start_twitter_x_oauth

        try:
            started = await start_twitter_x_oauth(
                context.registry.pool,
                capabilities=arguments.get("capabilities"),
                client_id=arguments.get("client_id"),
                client_secret=arguments.get("client_secret"),
                use_env_client=bool(arguments.get("use_env_client", False)),
                source_channel=arguments.get("source_channel"),
                source_session_id=arguments.get("source_session_id") or context.session_id,
            )
        except TwitterXOAuthError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.MISSING_CONFIG)

        if isinstance(started, dict):
            return ToolResult.success_result(
                started,
                display_output=started.get("next_step"),
            )

        assert isinstance(started, TwitterXOAuthStart)
        payload = started.attempt_payload
        display = (
            "Twitter/X authorization started.\n"
            f"Attempt: {payload['attempt_id']}\n"
            f"{payload['authorization_url']}\n\n"
            "After approving, paste the full redirected localhost URL back here."
        )
        return ToolResult.success_result(payload, display_output=display)


class CompleteTwitterXConnectionHandler(ToolHandler):
    """Complete a pending Twitter/X OAuth attempt from the pasted redirect URL."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="complete_twitter_x_connection",
            description=(
                "Complete Twitter/X OAuth setup after the user pastes the full redirected localhost URL "
                "or authorization code from X."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "authorization_response": {
                        "type": "string",
                        "description": "Full redirected URL or raw authorization code.",
                    },
                    "attempt_id": {
                        "type": "string",
                        "description": "Optional Twitter/X connection attempt ID. Defaults to latest pending Twitter/X attempt.",
                    },
                },
                "required": ["authorization_response"],
            },
            category=ToolCategory.MESSAGING,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "complete_twitter_x_connection requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        from core.auth.twitter_x import TwitterXOAuthError, complete_twitter_x_oauth

        try:
            completed = await complete_twitter_x_oauth(
                context.registry.pool,
                authorization_response=str(arguments.get("authorization_response") or ""),
                attempt_id=arguments.get("attempt_id"),
            )
        except TwitterXOAuthError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.AUTH_FAILED)

        output = {
            "connector_id": "twitter_x",
            "status": "connected",
            "account_key": completed.account_key,
            "display_name": completed.display_name,
            "credential_ref": completed.credential_ref,
            "granted_scopes": completed.granted_scopes,
            "capabilities": completed.capabilities,
        }
        return ToolResult.success_result(
            output,
            display_output=f"Twitter/X connected for {completed.display_name}.",
        )


class RevokeTwitterXConnectionHandler(ToolHandler):
    """Disconnect Twitter/X locally and mark DB connection state revoked."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="revoke_twitter_x_connection",
            description=(
                "Disconnect Twitter/X from Hexis by deleting the local credential file and marking the "
                "connection revoked. The user can also remove the X-side grant in X settings."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional Twitter/X account key. If omitted, revokes local default Twitter/X credentials.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional user-visible revocation reason.",
                    },
                },
            },
            category=ToolCategory.MESSAGING,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "revoke_twitter_x_connection requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        from core.auth.twitter_x import delete_default_credentials

        delete_default_credentials()
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT revoke_integration_connection('twitter_x', $1, $2)",
                arguments.get("account_key"),
                arguments.get("reason") or "revoked by user request",
            )
        payload = _json(raw) or {}
        payload["local_credentials_deleted"] = True
        payload["remote_revocation"] = "not_attempted"
        payload["next_step"] = (
            "Local Twitter/X credentials are removed. To remove the X-side OAuth grant too, "
            "open your X connected apps settings and revoke the app there."
        )
        return ToolResult.success_result(
            payload,
            display_output="Twitter/X disconnected locally.",
        )


class StartGmailBackfillHandler(ToolHandler):
    """Queue a DB-owned Gmail history backfill."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="start_gmail_backfill",
            description=(
                "Queue Gmail message history ingestion into raw source documents and the memory ingestion queue. "
                "Use after Gmail is connected when the user asks Hexis to learn from email, import email history, "
                "or ingest a Gmail search/label. Do not use this for live requests like 'check my email now' "
                "or 'read a batch of emails'; use email_list/email_search/email_read first. "
                "This only reads Gmail; message sending, replying, labeling, "
                "and spam actions require separate tools and explicit authorization."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Connected Gmail account/email. Required only when multiple Gmail accounts are connected.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional Gmail search query, such as newer_than:30d or from:alice@example.com.",
                    },
                    "label_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional Gmail labels to restrict the backfill, such as INBOX or SENT.",
                    },
                    "include_spam_trash": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include spam and trash in Gmail list results.",
                    },
                    "max_messages": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Maximum messages to fetch in this job chunk.",
                    },
                    "page_size": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Gmail API page size for this job.",
                    },
                    "source_channel": {
                        "type": "string",
                        "description": "Optional source surface, such as cli, web, slack, telegram, or signal.",
                    },
                    "source_session_id": {
                        "type": "string",
                        "description": "Optional conversation/session identifier.",
                    },
                },
            },
            category=ToolCategory.EMAIL,
            energy_cost=2,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "start_gmail_backfill requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        from core.auth.google_gmail import load_default_credentials

        if load_default_credentials() is None:
            payload = {
                "connector_id": "gmail",
                "status": "needs_connection",
                "next_step": "Gmail is not connected. Open the Gmail setup panel before starting an email backfill.",
            }
            payload["ui"] = _gmail_connector_setup_ui(
                payload,
                status="needs_client_secret",
                capabilities=["read", "search"],
                next_step=payload["next_step"],
            )
            return ToolResult(
                success=False,
                output=payload,
                display_output=payload["next_step"],
                error="Gmail is not connected. Open the structured Gmail setup flow before backfilling email.",
                error_type=ToolErrorType.AUTH_FAILED,
            )
        try:
            account_key = await _resolve_gmail_account(
                context.registry.pool,
                arguments.get("account_key"),
            )
        except ValueError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)

        requested_range = {
            "query": str(arguments.get("query") or "").strip() or None,
            "label_ids": arguments.get("label_ids") or [],
            "include_spam_trash": bool(arguments.get("include_spam_trash", False)),
            "max_messages": arguments.get("max_messages", 100),
            "page_size": arguments.get("page_size", 100),
        }
        metadata = {
            "source_channel": arguments.get("source_channel"),
            "source_session_id": arguments.get("source_session_id") or context.session_id,
            "queued_by_tool": "start_gmail_backfill",
        }
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT enqueue_connector_backfill_job(
                    'gmail',
                    $1,
                    'messages',
                    $2::jsonb,
                    $3::jsonb
                )
                """,
                account_key,
                json.dumps({k: v for k, v in requested_range.items() if v not in (None, [], "")}),
                json.dumps({k: v for k, v in metadata.items() if v not in (None, "")}),
            )
        payload = _json(raw) or {}
        verb = "already queued" if payload.get("existing") else "queued"
        return ToolResult.success_result(
            payload,
            display_output=(
                f"Gmail backfill {verb} for {account_key}. "
                "The maintenance worker will fetch messages and queue ingestion."
            ),
        )


class GmailBackfillStatusHandler(ToolHandler):
    """Inspect DB-owned Gmail backfill status."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="gmail_backfill_status",
            description="Show Gmail backfill jobs, cursors, and raw source-item counts.",
            parameters={
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "Optional connected Gmail account/email to filter status.",
                    },
                },
            },
            category=ToolCategory.EMAIL,
            energy_cost=1,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "gmail_backfill_status requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT get_connector_backfill_status('gmail', $1)",
                arguments.get("account_key"),
            )
        payload = _json(raw) or {}
        jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
        item_counts = payload.get("item_counts") if isinstance(payload.get("item_counts"), list) else []
        active = [job for job in jobs if isinstance(job, dict) and job.get("status") in {"pending", "in_progress", "paused"}]
        total_items = sum(int(item.get("count") or 0) for item in item_counts if isinstance(item, dict))
        return ToolResult.success_result(
            payload,
            display_output=f"Gmail backfill: {len(active)} active jobs, {total_items} source items.",
        )


class ControlGmailBackfillHandler(ToolHandler):
    """Pause, resume, or cancel a queued Gmail backfill job."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="control_gmail_backfill",
            description="Pause, resume, or cancel a Gmail backfill job by job_id.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Connector backfill job ID from gmail_backfill_status.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["pause", "resume", "cancel"],
                        "description": "Control action to apply.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional human-readable reason for pause/cancel.",
                    },
                },
                "required": ["job_id", "action"],
            },
            category=ToolCategory.EMAIL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "control_gmail_backfill requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        action = str(arguments.get("action") or "").strip().lower()
        job_id = str(arguments.get("job_id") or "").strip()
        if action not in {"pause", "resume", "cancel"}:
            return ToolResult.error_result("action must be pause, resume, or cancel.", ToolErrorType.INVALID_PARAMS)
        statement = {
            "pause": "SELECT pause_connector_backfill_job($1::uuid, $2)",
            "resume": "SELECT resume_connector_backfill_job($1::uuid)",
            "cancel": "SELECT cancel_connector_backfill_job($1::uuid, $2)",
        }[action]
        async with context.registry.pool.acquire() as conn:
            if action == "resume":
                raw = await conn.fetchval(statement, job_id)
            else:
                raw = await conn.fetchval(statement, job_id, arguments.get("reason"))
        payload = _json(raw) or {}
        return ToolResult.success_result(
            payload,
            display_output=f"Gmail backfill {action}: {payload.get('status', 'unknown')}.",
        )


class StartConnectorBackfillHandler(ToolHandler):
    """Queue DB-owned history backfill for a connected non-Gmail connector."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="start_connector_backfill",
            description=(
                "Queue historical import for connected communication connectors. Use when the user asks "
                "to import Slack, Telegram, Signal, or Twitter/X history. Slack requires a channel_id. "
                "Telegram, Signal, and Twitter/X use local export/archive paths for retro-history."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {
                        "type": "string",
                        "description": "Connector id: slack, telegram, signal, or twitter_x.",
                    },
                    "account_key": {
                        "type": "string",
                        "description": "Connected account key. Required only when multiple accounts exist.",
                    },
                    "channel_id": {
                        "type": "string",
                        "description": "Slack channel/conversation ID to import, such as C123, G123, or D123.",
                    },
                    "requested_range": {
                        "type": "object",
                        "description": "Provider-specific range options. Slack: oldest, latest, inclusive, cursor. Telegram/Signal/Twitter/X: export_path or import_path.",
                    },
                    "export_path": {
                        "type": "string",
                        "description": "Local Telegram/Signal export file or Twitter/X archive directory/JS file to import when provider APIs cannot expose retro-history.",
                    },
                    "import_path": {
                        "type": "string",
                        "description": "Alias for export_path.",
                    },
                    "export_format": {
                        "type": "string",
                        "description": "Optional export format hint, such as telegram_json, signal_json, or signal_csv.",
                    },
                    "max_messages": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 5000,
                    },
                    "page_size": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 200,
                    },
                    "inclusive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Slack range option: include messages exactly at oldest/latest.",
                    },
                    "source_channel": {
                        "type": "string",
                        "description": "Optional source surface, such as cli, web, slack, telegram, or signal.",
                    },
                    "source_session_id": {
                        "type": "string",
                        "description": "Optional conversation/session identifier.",
                    },
                },
                "required": ["connector_id"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=2,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "start_connector_backfill requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        connector_id = _connector_id(arguments.get("connector_id"))
        if connector_id == "gmail":
            return ToolResult.error_result(
                "Use start_gmail_backfill for Gmail because Gmail uses a dedicated OAuth credential path.",
                ToolErrorType.INVALID_PARAMS,
            )
        if connector_id not in {"slack", "telegram", "signal", "twitter_x"}:
            return ToolResult.error_result(
                "start_connector_backfill supports slack, telegram, signal, and twitter_x.",
                ToolErrorType.INVALID_PARAMS,
            )

        requested_range = arguments.get("requested_range")
        if requested_range is None:
            requested_range = {}
        if not isinstance(requested_range, dict):
            return ToolResult.error_result("requested_range must be an object.", ToolErrorType.INVALID_PARAMS)
        requested_range = dict(requested_range)
        for key in (
            "channel_id",
            "oldest",
            "latest",
            "cursor",
            "export_path",
            "import_path",
            "export_format",
            "stream",
            "source",
            "query",
            "pagination_token",
            "next_token",
            "user_id",
        ):
            value = arguments.get(key)
            if value not in (None, ""):
                requested_range[key] = value
        for key in ("max_messages", "page_size"):
            if arguments.get(key) is not None:
                requested_range[key] = arguments.get(key)
        if arguments.get("inclusive") is not None:
            requested_range["inclusive"] = bool(arguments.get("inclusive"))
        if connector_id == "slack" and not str(requested_range.get("channel_id") or "").strip():
            return ToolResult.error_result(
                "Slack history backfill requires channel_id because broad workspace import is too surprising.",
                ToolErrorType.INVALID_PARAMS,
            )

        has_local_export = bool(requested_range.get("export_path") or requested_range.get("import_path"))
        try:
            if connector_id == "twitter_x" and has_local_export:
                account_key = await _ensure_twitter_archive_connection(
                    context.registry.pool,
                    str(arguments.get("account_key") or "").strip() or "archive:twitter_x",
                    arguments.get("source_session_id") or context.session_id,
                )
            else:
                account_key = await _resolve_connector_account(
                    context.registry.pool,
                    connector_id,
                    arguments.get("account_key"),
                )
        except ValueError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)

        metadata = {
            "source_channel": arguments.get("source_channel"),
            "source_session_id": arguments.get("source_session_id") or context.session_id,
            "queued_by_tool": "start_connector_backfill",
        }
        max_attempts = 3 if (connector_id in {"slack", "twitter_x"} or has_local_export) else 1
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT enqueue_connector_backfill_job(
                    $1,
                    $2,
                    'messages',
                    $3::jsonb,
                    $4::jsonb,
                    $5::int
                )
                """,
                connector_id,
                account_key,
                json.dumps({k: v for k, v in requested_range.items() if v not in (None, [], "")}),
                json.dumps({k: v for k, v in metadata.items() if v not in (None, "")}),
                max_attempts,
            )
        payload = _json(raw) or {}
        verb = "already queued" if payload.get("existing") else "queued"
        estimate = payload.get("estimate") if isinstance(payload.get("estimate"), dict) else {}
        status = estimate.get("provider_status")
        estimate_text = f" Estimate: {status}." if status else ""
        return ToolResult.success_result(
            payload,
            display_output=(
                f"{connector_id} backfill {verb} for {account_key}. "
                "The maintenance worker will fetch what the provider can expose and preserve raw source items."
                f"{estimate_text}"
            ),
        )


class ConnectorBackfillStatusHandler(ToolHandler):
    """Inspect DB-owned connector backfill status for any provider."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="connector_backfill_status",
            description="Show connector backfill jobs, cursors, and raw source-item counts for any provider.",
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {
                        "type": "string",
                        "description": "Optional connector filter, such as gmail, slack, telegram, signal, or twitter_x.",
                    },
                    "account_key": {
                        "type": "string",
                        "description": "Optional provider account filter.",
                    },
                },
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "connector_backfill_status requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        connector_id = _connector_id(arguments.get("connector_id")) or None
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT get_connector_backfill_status($1, $2)",
                connector_id,
                arguments.get("account_key"),
            )
        payload = _json(raw) or {}
        jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
        item_counts = payload.get("item_counts") if isinstance(payload.get("item_counts"), list) else []
        active = [
            job for job in jobs
            if isinstance(job, dict) and job.get("status") in {"pending", "in_progress", "paused"}
        ]
        total_items = sum(int(item.get("count") or 0) for item in item_counts if isinstance(item, dict))
        label = connector_id or "connectors"
        return ToolResult.success_result(
            payload,
            display_output=f"{label} backfill: {len(active)} active jobs, {total_items} source items.",
        )


class ControlConnectorBackfillHandler(ToolHandler):
    """Pause, resume, or cancel any connector backfill job."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="control_connector_backfill",
            description="Pause, resume, or cancel a connector backfill job by job_id.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Connector backfill job ID from connector_backfill_status.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["pause", "resume", "cancel"],
                        "description": "Control action to apply.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional human-readable reason for pause/cancel.",
                    },
                },
                "required": ["job_id", "action"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "control_connector_backfill requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        action = str(arguments.get("action") or "").strip().lower()
        job_id = str(arguments.get("job_id") or "").strip()
        if action not in {"pause", "resume", "cancel"}:
            return ToolResult.error_result("action must be pause, resume, or cancel.", ToolErrorType.INVALID_PARAMS)
        statement = {
            "pause": "SELECT pause_connector_backfill_job($1::uuid, $2)",
            "resume": "SELECT resume_connector_backfill_job($1::uuid)",
            "cancel": "SELECT cancel_connector_backfill_job($1::uuid, $2)",
        }[action]
        async with context.registry.pool.acquire() as conn:
            if action == "resume":
                raw = await conn.fetchval(statement, job_id)
            else:
                raw = await conn.fetchval(statement, job_id, arguments.get("reason"))
        payload = _json(raw) or {}
        return ToolResult.success_result(
            payload,
            display_output=f"Connector backfill {action}: {payload.get('status', 'unknown')}.",
        )


class ConnectorActionPolicyStatusHandler(ToolHandler):
    """List DB-owned connector action policies."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="connector_action_policy_status",
            description=(
                "List connector action policies that authorize sends, replies, label/spam actions, "
                "or other external provider state changes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {
                        "type": "string",
                        "description": "Optional connector filter, such as gmail, slack, telegram, or email.",
                    },
                    "account_key": {
                        "type": "string",
                        "description": "Optional provider account filter.",
                    },
                    "include_inactive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include revoked/expired policies.",
                    },
                },
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "connector_action_policy_status requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT list_connector_action_policies($1, $2, $3)",
                arguments.get("connector_id"),
                arguments.get("account_key"),
                bool(arguments.get("include_inactive", False)),
            )
        policies = _json(raw) or []
        return ToolResult.success_result(
            {"policies": policies, "count": len(policies)},
            display_output=f"Connector action policies: {len(policies)}.",
        )


class GrantConnectorActionPolicyHandler(ToolHandler):
    """Grant a DB-owned connector action policy."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="grant_connector_action_policy",
            description=(
                "Create a scoped connector action policy. Use only when the user explicitly authorizes "
                "a class of actions, such as allowing heartbeat to send Slack alerts to one channel or "
                "letting Gmail reply no-thank-you to a constrained class of emails."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {
                        "type": "string",
                        "description": "Connector/provider id, such as gmail, slack, telegram, discord, or email.",
                    },
                    "action_kind": {
                        "type": "string",
                        "description": "Action kind, such as send, reply, label, spam_triage, mark_read, or delete.",
                    },
                    "account_key": {
                        "type": "string",
                        "description": "Optional provider account/email this policy applies to.",
                    },
                    "constraints": {
                        "type": "object",
                        "description": (
                            "Policy constraints. Supported keys include allowed_targets, denied_targets, "
                            "allowed_recipients, denied_recipients, and max_per_day."
                        ),
                    },
                    "allow_autonomous": {
                        "type": "boolean",
                        "default": False,
                        "description": "Allow non-chat contexts such as heartbeat to use this policy.",
                    },
                    "requires_per_action_approval": {
                        "type": "boolean",
                        "default": True,
                        "description": "Require explicit approval for each non-chat action unless false.",
                    },
                    "contexts": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["chat", "heartbeat", "mcp"]},
                        "description": "Contexts where this policy can apply.",
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "Optional ISO timestamp after which this policy expires.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Short human-readable reason for the grant.",
                    },
                    "source_session_id": {
                        "type": "string",
                        "description": "Optional source conversation/session identifier.",
                    },
                },
                "required": ["connector_id", "action_kind"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "grant_connector_action_policy requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        constraints = arguments.get("constraints")
        if constraints is None:
            constraints = {}
        if not isinstance(constraints, dict):
            return ToolResult.error_result("constraints must be an object.", ToolErrorType.INVALID_PARAMS)
        contexts = arguments.get("contexts")
        if contexts is not None and not isinstance(contexts, list):
            return ToolResult.error_result("contexts must be an array.", ToolErrorType.INVALID_PARAMS)
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT grant_connector_action_policy(
                    $1,
                    $2,
                    $3,
                    $4::jsonb,
                    $5,
                    $6,
                    $7::text[],
                    NULLIF($8, '')::timestamptz,
                    $9,
                    $10,
                    'user'
                )
                """,
                arguments.get("connector_id"),
                arguments.get("action_kind"),
                arguments.get("account_key"),
                json.dumps(constraints),
                bool(arguments.get("allow_autonomous", False)),
                bool(arguments.get("requires_per_action_approval", True)),
                [str(item) for item in contexts] if contexts is not None else None,
                str(arguments.get("expires_at") or ""),
                arguments.get("source_session_id") or context.session_id,
                arguments.get("rationale"),
            )
        payload = _json(raw) or {}
        return ToolResult.success_result(
            payload,
            display_output=(
                f"Connector action policy granted: {payload.get('connector_id')}/"
                f"{payload.get('action_kind')} ({payload.get('policy_id')})."
            ),
        )


class RevokeConnectorActionPolicyHandler(ToolHandler):
    """Revoke a DB-owned connector action policy."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="revoke_connector_action_policy",
            description="Revoke a connector action policy by policy_id.",
            parameters={
                "type": "object",
                "properties": {
                    "policy_id": {
                        "type": "string",
                        "description": "Policy ID from connector_action_policy_status.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional human-readable revocation reason.",
                    },
                },
                "required": ["policy_id"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.MCP},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result(
                "revoke_connector_action_policy requires an active tool registry.",
                ToolErrorType.EXECUTION_FAILED,
            )
        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT revoke_connector_action_policy($1::uuid, $2)",
                arguments.get("policy_id"),
                arguments.get("reason"),
            )
        payload = _json(raw) or {}
        return ToolResult.success_result(
            payload,
            display_output=f"Connector action policy revoke: {payload.get('status', 'unknown')}.",
        )


def create_integration_tools() -> list[ToolHandler]:
    return [
        IntegrationSetupStatusHandler(),
        StartIntegrationSetupHandler(),
        ConfigureChannelIntegrationHandler(),
        VerifyChannelIntegrationHandler(),
        GmailSetupStatusHandler(),
        ConnectGmailHandler(),
        CompleteGmailConnectionHandler(),
        RevokeGmailConnectionHandler(),
        ConnectTwitterXHandler(),
        CompleteTwitterXConnectionHandler(),
        RevokeTwitterXConnectionHandler(),
        StartGmailBackfillHandler(),
        GmailBackfillStatusHandler(),
        ControlGmailBackfillHandler(),
        StartConnectorBackfillHandler(),
        ConnectorBackfillStatusHandler(),
        ControlConnectorBackfillHandler(),
        ConnectorActionPolicyStatusHandler(),
        GrantConnectorActionPolicyHandler(),
        RevokeConnectorActionPolicyHandler(),
    ]
