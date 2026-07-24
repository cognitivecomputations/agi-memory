"""Twitter/X OAuth setup helpers for first-class personal-data connectors."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from core.auth.store import auth_lock, delete_auth, load_auth, save_auth
from core.auth.utils import create_state, generate_pkce, now_ms
from core.integration_reliability import (
    IntegrationHttpError,
    format_provider_error,
    request_json,
)

TWITTER_X_CONNECTOR_ID = "twitter_x"
TWITTER_X_DEFAULT_CREDENTIAL_REF = "integration.twitter_x.default"
TWITTER_X_CLIENT_REF = "integration.twitter_x.client"
TWITTER_X_PENDING_PREFIX = "integration.twitter_x.pending."

TWITTER_X_AUTHORIZE_URL = "https://x.com/i/oauth2/authorize"
TWITTER_X_TOKEN_URL = "https://api.x.com/2/oauth2/token"
TWITTER_X_ME_URL = "https://api.x.com/2/users/me"
TWITTER_X_REDIRECT_URI = "http://localhost:1"

_CLIENT_ID_ENV = ("TWITTER_X_CLIENT_ID", "X_CLIENT_ID", "TWITTER_CLIENT_ID")
_CLIENT_SECRET_ENV = ("TWITTER_X_CLIENT_SECRET", "X_CLIENT_SECRET", "TWITTER_CLIENT_SECRET")


class TwitterXOAuthError(RuntimeError):
    """Expected, user-actionable Twitter/X setup failure."""


@dataclass(frozen=True)
class TwitterXOAuthStart:
    attempt_payload: dict[str, Any]
    pending_auth_ref: str


@dataclass(frozen=True)
class TwitterXOAuthComplete:
    account_key: str
    display_name: str
    credential_ref: str
    granted_scopes: list[str]
    capabilities: list[str]
    credential_payload: dict[str, Any]


def _json_arg(value: Any) -> str:
    return json.dumps(value)


async def prepare_twitter_x_connection_attempt(pool: Any, capabilities: Any = None) -> dict[str, Any]:
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT prepare_connection_attempt('twitter_x', $1::jsonb)",
                _json_arg(capabilities) if capabilities is not None else None,
            )
    except Exception as exc:
        raise TwitterXOAuthError(str(exc)) from exc
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        raise TwitterXOAuthError("Twitter/X connector preparation returned an invalid payload.")
    return payload


def _normalize_client_config(payload: dict[str, Any]) -> dict[str, str]:
    client_id = str(payload.get("client_id") or "").strip()
    client_secret = str(payload.get("client_secret") or "").strip()
    if not client_id:
        raise TwitterXOAuthError("Twitter/X OAuth client_id is required.")
    result = {"client_id": client_id}
    if client_secret:
        result["client_secret"] = client_secret
    return result


def _load_env_client_config() -> tuple[dict[str, str] | None, str | None]:
    client_id = ""
    client_id_source = ""
    for name in _CLIENT_ID_ENV:
        value = os.getenv(name)
        if value and value.strip():
            client_id = value.strip()
            client_id_source = name
            break
    if not client_id:
        return None, None

    client_secret = ""
    client_secret_source = ""
    for name in _CLIENT_SECRET_ENV:
        value = os.getenv(name)
        if value and value.strip():
            client_secret = value.strip()
            client_secret_source = name
            break
    payload = {"client_id": client_id}
    if client_secret:
        payload["client_secret"] = client_secret
    source = client_id_source if not client_secret_source else f"{client_id_source}+{client_secret_source}"
    return payload, source


def _resolve_client_config(
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    use_env_client: bool = False,
) -> tuple[dict[str, str] | None, str | None]:
    if client_id and client_id.strip():
        payload: dict[str, Any] = {"client_id": client_id.strip()}
        if client_secret and client_secret.strip():
            payload["client_secret"] = client_secret.strip()
        config = _normalize_client_config(payload)
        save_auth(TWITTER_X_CLIENT_REF, config)
        return config, "arguments"

    stored = load_auth(TWITTER_X_CLIENT_REF)
    if isinstance(stored, dict):
        return _normalize_client_config(stored), "stored"

    if use_env_client:
        config, source = _load_env_client_config()
        if config:
            save_auth(TWITTER_X_CLIENT_REF, config)
            return config, source

    return None, None


def build_client_needed_payload() -> dict[str, Any]:
    return {
        "status": "needs_client",
        "connector_id": TWITTER_X_CONNECTOR_ID,
        "client_saved": False,
        "accepted_inputs": ["client_id", "client_secret", "use_env_client"],
        "env_options": [*_CLIENT_ID_ENV, *_CLIENT_SECRET_ENV],
        "next_step": (
            "Create or choose an X Developer app, enable user sign-in, add "
            f"{TWITTER_X_REDIRECT_URI} as the callback URI, then enter the X app client ID. "
            "Include the app secret only if X says the app is confidential. "
            "Use server-provided settings only if you explicitly want Hexis to read configured environment values."
        ),
        "docs_url": "https://docs.x.com/fundamentals/authentication/oauth-2-0/authorization-code",
    }


def load_default_credentials() -> dict[str, Any] | None:
    value = load_auth(TWITTER_X_DEFAULT_CREDENTIAL_REF)
    return value if isinstance(value, dict) else None


def delete_default_credentials() -> None:
    delete_auth(TWITTER_X_DEFAULT_CREDENTIAL_REF)


def _expiry_ms(credentials: dict[str, Any]) -> int:
    raw = credentials.get("expires_ms")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    expiry = credentials.get("expiry")
    if isinstance(expiry, str) and expiry.strip():
        try:
            parsed = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            return int(parsed.timestamp() * 1000)
        except ValueError:
            return 0
    return 0


def _expiry_iso(expires_in: int) -> str:
    expiry = datetime.fromtimestamp((now_ms() + int(expires_in) * 1000) / 1000, tz=timezone.utc)
    return expiry.isoformat().replace("+00:00", "Z")


def _token_headers(client_config: dict[str, Any]) -> dict[str, str]:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    client_secret = client_config.get("client_secret")
    client_id = client_config.get("client_id")
    if isinstance(client_id, str) and isinstance(client_secret, str) and client_secret:
        raw = f"{client_id}:{client_secret}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
    return headers


async def refresh_default_credentials_if_needed(*, leeway_ms: int = 60_000) -> dict[str, Any]:
    with auth_lock(TWITTER_X_DEFAULT_CREDENTIAL_REF):
        credentials = load_default_credentials()
        if not credentials:
            raise TwitterXOAuthError("Twitter/X credentials are not saved. Use connect_twitter_x first.")

        if _expiry_ms(credentials) > now_ms() + leeway_ms and credentials.get("token"):
            return credentials

        refresh_token = credentials.get("refresh_token")
        client_id = credentials.get("client_id")
        if not isinstance(refresh_token, str) or not refresh_token.strip():
            raise TwitterXOAuthError(
                "Saved Twitter/X credentials do not include a refresh token. "
                "Reconnect Twitter/X with offline.access."
            )
        if not isinstance(client_id, str) or not client_id.strip():
            raise TwitterXOAuthError("Saved Twitter/X credentials are missing client_id. Reconnect Twitter/X.")

        client_config = {
            "client_id": client_id,
            "client_secret": credentials.get("client_secret") or "",
        }
        data = {
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        if not client_config.get("client_secret"):
            data["client_id"] = client_id

        async def _refresh() -> dict[str, Any]:
            try:
                payload = await request_json(
                    "twitter_x_oauth",
                    "POST",
                    TWITTER_X_TOKEN_URL,
                    headers=_token_headers(client_config),
                    data=data,
                    timeout=30.0,
                    attempts=3,
                    max_delay=15.0,
                    retry_unsafe_methods=True,
                )
            except IntegrationHttpError as exc:
                raise TwitterXOAuthError(format_provider_error("Twitter/X token refresh", exc)) from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
                raise TwitterXOAuthError("Twitter/X token refresh did not return an access token.")
            return payload

        token_data = await _refresh()
        expires_in = int(token_data.get("expires_in") or 7200)
        refreshed = {
            **credentials,
            "token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token") or refresh_token,
            "expiry": _expiry_iso(expires_in),
            "expires_ms": now_ms() + expires_in * 1000,
        }
        scopes = token_data.get("scope")
        if isinstance(scopes, str) and scopes.strip():
            refreshed["scopes"] = scopes.split()
        save_auth(TWITTER_X_DEFAULT_CREDENTIAL_REF, refreshed)
        return refreshed


async def start_twitter_x_oauth(
    pool: Any,
    *,
    capabilities: Any = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    use_env_client: bool = False,
    source_channel: str | None = None,
    source_session_id: str | None = None,
) -> TwitterXOAuthStart | dict[str, Any]:
    client_config, source = _resolve_client_config(
        client_id=client_id,
        client_secret=client_secret,
        use_env_client=use_env_client,
    )
    if not client_config:
        return build_client_needed_payload()

    prepared = await prepare_twitter_x_connection_attempt(pool, capabilities)
    caps = list(prepared.get("capabilities") or prepared.get("requested_capabilities") or [])
    scopes = list(prepared.get("requested_scopes") or [])
    if not caps or not scopes:
        raise TwitterXOAuthError("Twitter/X connector preparation did not return capabilities and scopes.")

    verifier, challenge = generate_pkce()
    state = create_state()
    params = {
        "response_type": "code",
        "client_id": client_config["client_id"],
        "redirect_uri": TWITTER_X_REDIRECT_URI,
        "scope": " ".join(scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    authorization_url = f"{TWITTER_X_AUTHORIZE_URL}?{urlencode(params)}"
    next_step = (
        "Open authorization_url, approve the requested Twitter/X scopes, ignore the expected "
        "localhost connection failure, then paste the full redirected URL back into this conversation."
    )
    flow_state = {
        "pending_auth_ref": None,
        "state_hash": hashlib.sha256(state.encode("utf-8")).hexdigest(),
        "redirect_uri": TWITTER_X_REDIRECT_URI,
        "client_source": source,
        "scope_count": len(scopes),
    }

    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT start_connection_attempt(
                'twitter_x',
                $1::jsonb,
                ARRAY[]::text[],
                $2::jsonb,
                $3,
                $4,
                $5,
                $6,
                CURRENT_TIMESTAMP + INTERVAL '10 minutes'
            )
            """,
            json.dumps(caps),
            json.dumps(flow_state),
            authorization_url,
            next_step,
            source_channel,
            source_session_id,
        )
    payload = json.loads(raw) if isinstance(raw, str) else raw
    pending_ref = f"{TWITTER_X_PENDING_PREFIX}{payload['attempt_id']}"
    save_auth(
        pending_ref,
        {
            "state": state,
            "verifier": verifier,
            "redirect_uri": TWITTER_X_REDIRECT_URI,
            "client_ref": TWITTER_X_CLIENT_REF,
            "scopes": list(payload.get("requested_scopes") or scopes),
            "capabilities": list(payload.get("requested_capabilities") or caps),
            "created_ms": now_ms(),
        },
    )
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE connection_attempts
            SET flow_state = flow_state || $2::jsonb,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1::uuid
            """,
            payload["attempt_id"],
            json.dumps({"pending_auth_ref": pending_ref}),
        )
    payload["pending_auth_ref"] = pending_ref
    return TwitterXOAuthStart(attempt_payload=payload, pending_auth_ref=pending_ref)


def parse_authorization_response(value: str) -> tuple[str, str | None]:
    raw = (value or "").strip()
    if not raw:
        raise TwitterXOAuthError("Paste the full redirected URL or authorization code.")
    if not raw.startswith("http"):
        return raw, None
    parsed = urlparse(raw)
    params = parse_qs(parsed.query)
    if params.get("error"):
        raise TwitterXOAuthError(f"Twitter/X authorization failed: {params['error'][0]}")
    code = (params.get("code") or [""])[0]
    if not code:
        raise TwitterXOAuthError("The redirected URL does not contain a code parameter.")
    state = (params.get("state") or [None])[0]
    return code, state


async def _exchange_code(
    *,
    code: str,
    verifier: str,
    client_config: dict[str, str],
    redirect_uri: str,
) -> dict[str, Any]:
    data = {
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }
    if not client_config.get("client_secret"):
        data["client_id"] = client_config["client_id"]
    try:
        payload = await request_json(
            "twitter_x_oauth",
            "POST",
            TWITTER_X_TOKEN_URL,
            headers=_token_headers(client_config),
            data=data,
            timeout=30.0,
            attempts=3,
            max_delay=15.0,
            retry_unsafe_methods=True,
        )
    except IntegrationHttpError as exc:
        raise TwitterXOAuthError(format_provider_error("Twitter/X token exchange", exc)) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
        raise TwitterXOAuthError("Twitter/X token exchange did not return an access token.")
    return payload


async def _fetch_account(access_token: str) -> dict[str, str] | None:
    try:
        payload = await request_json(
            "twitter_x",
            "GET",
            TWITTER_X_ME_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params={"user.fields": "username,name,verified,protected"},
            timeout=10.0,
            attempts=2,
            max_delay=2.0,
        )
    except Exception:
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    user_id = str(data.get("id") or "").strip()
    username = str(data.get("username") or "").strip()
    if not user_id:
        return None
    display = f"@{username}" if username else f"X user {user_id}"
    return {
        "user_id": user_id,
        "username": username,
        "display_name": display,
        "account_key": f"x:{user_id}",
    }


async def complete_twitter_x_oauth(
    pool: Any,
    *,
    authorization_response: str,
    attempt_id: str | None = None,
) -> TwitterXOAuthComplete:
    if not attempt_id:
        async with pool.acquire() as conn:
            attempt_id = await conn.fetchval(
                """
                SELECT id::text
                FROM connection_attempts
                WHERE connector_id = 'twitter_x'
                  AND status IN ('pending_user', 'awaiting_input', 'error')
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
    if not attempt_id:
        raise TwitterXOAuthError("No pending Twitter/X connection attempt. Start with connect_twitter_x first.")

    pending_ref = f"{TWITTER_X_PENDING_PREFIX}{attempt_id}"
    pending = load_auth(pending_ref)
    if not isinstance(pending, dict):
        raise TwitterXOAuthError("The pending Twitter/X OAuth session expired or is missing. Start connect_twitter_x again.")

    code, returned_state = parse_authorization_response(authorization_response)
    if returned_state and returned_state != pending.get("state"):
        raise TwitterXOAuthError("OAuth state mismatch. Start a fresh Twitter/X connection attempt and use the newest browser tab.")

    client_config = load_auth(str(pending.get("client_ref") or TWITTER_X_CLIENT_REF))
    if not isinstance(client_config, dict):
        raise TwitterXOAuthError("Stored Twitter/X OAuth client configuration is missing. Start connect_twitter_x again.")
    client = _normalize_client_config(client_config)

    async with pool.acquire() as conn:
        await conn.fetchval("SELECT mark_connection_attempt_exchanging($1::uuid)", attempt_id)

    try:
        token_data = await _exchange_code(
            code=code,
            verifier=str(pending["verifier"]),
            client_config=client,
            redirect_uri=str(pending.get("redirect_uri") or TWITTER_X_REDIRECT_URI),
        )
        access_token = token_data["access_token"]
        scopes = token_data.get("scope")
        granted_scopes = scopes.split() if isinstance(scopes, str) and scopes.strip() else list(pending.get("scopes") or [])
        account = await _fetch_account(access_token)
        if not account:
            account = {
                "account_key": f"x:{attempt_id}",
                "display_name": "Twitter/X account",
                "user_id": "",
                "username": "",
            }
        expires_in = int(token_data.get("expires_in") or 7200)
        credential_payload = {
            "type": "twitter_x_oauth2",
            "client_id": client["client_id"],
            "client_secret": client.get("client_secret", ""),
            "refresh_token": token_data.get("refresh_token", ""),
            "token": access_token,
            "token_uri": TWITTER_X_TOKEN_URL,
            "scopes": granted_scopes,
            "expiry": _expiry_iso(expires_in),
            "expires_ms": now_ms() + expires_in * 1000,
            "account_key": account["account_key"],
            "user_id": account.get("user_id") or "",
            "username": account.get("username") or "",
            "display_name": account.get("display_name") or account["account_key"],
        }
        save_auth(TWITTER_X_DEFAULT_CREDENTIAL_REF, credential_payload)
        caps = list(pending.get("capabilities") or [])
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
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
                account["account_key"],
                account.get("display_name") or account["account_key"],
                TWITTER_X_DEFAULT_CREDENTIAL_REF,
                granted_scopes,
                json.dumps(caps),
                json.dumps({
                    "auth_store": "filesystem",
                    "token_uri": TWITTER_X_TOKEN_URL,
                    "user_id": account.get("user_id") or "",
                    "username": account.get("username") or "",
                }),
            )
        delete_auth(pending_ref)
        result = json.loads(raw) if isinstance(raw, str) else raw
        result_caps = list(result.get("capabilities") or caps)
        return TwitterXOAuthComplete(
            account_key=str(result["account_key"]),
            display_name=str(result.get("display_name") or result["account_key"]),
            credential_ref=TWITTER_X_DEFAULT_CREDENTIAL_REF,
            granted_scopes=granted_scopes,
            capabilities=result_caps,
            credential_payload=credential_payload,
        )
    except Exception as exc:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT mark_connection_attempt_error($1::uuid, $2)",
                attempt_id,
                str(exc)[:2000],
            )
        if isinstance(exc, TwitterXOAuthError):
            raise
        raise TwitterXOAuthError(str(exc)) from exc
