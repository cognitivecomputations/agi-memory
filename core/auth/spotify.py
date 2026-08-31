"""Spotify Authorization Code + PKCE setup and token refresh."""

from __future__ import annotations

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
    request_json_response,
)


SPOTIFY_CONNECTOR_ID = "spotify"
SPOTIFY_DEFAULT_CREDENTIAL_REF = "integration.spotify.default"
SPOTIFY_CLIENT_REF = "integration.spotify.client"
SPOTIFY_PENDING_PREFIX = "integration.spotify.pending."
SPOTIFY_CALLBACK_PATH = "/api/integrations/spotify/callback"
SPOTIFY_DEFAULT_API_BASE_URL = "http://127.0.0.1:43817"


class SpotifyOAuthError(RuntimeError):
    """Expected, user-actionable Spotify setup failure."""


@dataclass(frozen=True)
class SpotifyOAuthStart:
    attempt_payload: dict[str, Any]
    pending_auth_ref: str


@dataclass(frozen=True)
class SpotifyOAuthComplete:
    account_key: str
    display_name: str
    credential_ref: str
    granted_scopes: list[str]
    capabilities: list[str]


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _clean_loopback_base(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SpotifyOAuthError("Spotify callback base must be a complete HTTP(S) URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SpotifyOAuthError("Spotify callback base cannot contain credentials, a query, or a fragment.")
    if parsed.hostname == "localhost" or (parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1"}):
        raise SpotifyOAuthError(
            "Spotify HTTP callbacks require a loopback IP address. Use http://127.0.0.1:<port>."
        )
    return parsed._replace(path="", params="", query="", fragment="").geturl().rstrip("/")


def configured_spotify_redirect_uri() -> str:
    """Return the exact redirect URI the user must register in Spotify."""
    explicit = os.getenv("HEXIS_SPOTIFY_REDIRECT_URI")
    if explicit:
        parsed = urlparse(explicit.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SpotifyOAuthError("HEXIS_SPOTIFY_REDIRECT_URI is not a complete HTTP(S) URL.")
        if parsed.hostname in {"localhost", None}:
            raise SpotifyOAuthError(
                "Spotify requires a loopback IP address, not localhost. Use http://127.0.0.1:<port>/api/integrations/spotify/callback."
            )
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1"}:
            raise SpotifyOAuthError(
                "Spotify HTTP callbacks require a loopback IP address. Use http://127.0.0.1:<port>/api/integrations/spotify/callback."
            )
        return parsed._replace(params="", query="", fragment="").geturl().rstrip("/")
    base = os.getenv("HEXIS_API_URL") or os.getenv("HEXIS_API_BASE_URL") or SPOTIFY_DEFAULT_API_BASE_URL
    return f"{_clean_loopback_base(base)}{SPOTIFY_CALLBACK_PATH}"


async def _connector_plan(pool: Any, capabilities: Any = None) -> dict[str, Any]:
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT prepare_connection_attempt('spotify', $1::jsonb)",
                json.dumps(capabilities) if capabilities is not None else None,
            )
    except Exception as exc:
        raise SpotifyOAuthError(str(exc)) from exc
    plan = _json(raw)
    if not isinstance(plan, dict):
        raise SpotifyOAuthError("Spotify connector preparation returned an invalid payload.")
    return plan


def _manifest_endpoint(plan: dict[str, Any], key: str) -> str:
    setup = plan.get("setup_manifest") or {}
    value = str(setup.get(key) or "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise SpotifyOAuthError(f"Spotify connector manifest has an invalid {key}.")
    return value


def _resolve_client_id(*, client_id: str | None, client_id_env: str | None) -> tuple[str | None, str | None]:
    direct = str(client_id or "").strip()
    env_name = str(client_id_env or "").strip()
    if direct and env_name:
        raise SpotifyOAuthError("Choose either client_id or client_id_env, not both.")
    if direct:
        save_auth(SPOTIFY_CLIENT_REF, {"client_id": direct, "source": "user_input"})
        return direct, "user_input"
    if env_name:
        from services.life_integrations import ENV_NAME_RE

        if not ENV_NAME_RE.fullmatch(env_name):
            raise SpotifyOAuthError("client_id_env must be an environment variable name such as SPOTIFY_CLIENT_ID.")
        value = os.getenv(env_name)
        if not value:
            raise SpotifyOAuthError(
                f"{env_name} is not set in the Hexis runtime. Set it, restart Hexis, then start Spotify setup again."
            )
        client = value.strip()
        save_auth(SPOTIFY_CLIENT_REF, {"client_id": client, "source": f"env:{env_name}", "client_id_env": env_name})
        return client, f"env:{env_name}"
    stored = load_auth(SPOTIFY_CLIENT_REF)
    if isinstance(stored, dict) and isinstance(stored.get("client_id"), str):
        return stored["client_id"].strip(), "stored_user_choice"
    return None, None


def spotify_client_needed_payload(plan: dict[str, Any]) -> dict[str, Any]:
    redirect_uri = configured_spotify_redirect_uri()
    return {
        "status": "needs_client",
        "connector_id": "spotify",
        "redirect_uri": redirect_uri,
        "accepted_inputs": ["client_id", "client_id_env"],
        "next_step": (
            "Create a Spotify app, add this exact Redirect URI to the app settings, then enter the app client ID "
            f"or explicitly select its environment variable: {redirect_uri}"
        ),
        "docs_url": plan.get("docs_url"),
    }


async def start_spotify_oauth(
    pool: Any,
    *,
    capabilities: Any = None,
    client_id: str | None = None,
    client_id_env: str | None = None,
    source_channel: str | None = None,
    source_session_id: str | None = None,
) -> SpotifyOAuthStart | dict[str, Any]:
    plan = await _connector_plan(pool, capabilities)
    resolved_client_id, client_source = _resolve_client_id(client_id=client_id, client_id_env=client_id_env)
    if not resolved_client_id:
        return spotify_client_needed_payload(plan)

    scopes = list(plan.get("requested_scopes") or [])
    caps = list(plan.get("capabilities") or [])
    if not caps or not scopes:
        raise SpotifyOAuthError("Spotify connector preparation did not return capabilities and scopes.")
    verifier, challenge = generate_pkce()
    state = create_state()
    redirect_uri = configured_spotify_redirect_uri()
    authorize_url = _manifest_endpoint(plan, "authorize_url")
    token_url = _manifest_endpoint(plan, "token_url")
    api_base_url = _manifest_endpoint(plan, "api_base_url")
    authorization_params = {
        'client_id': resolved_client_id,
        'response_type': 'code',
        'redirect_uri': redirect_uri,
        'scope': ' '.join(scopes),
        'code_challenge_method': 'S256',
        'code_challenge': challenge,
        'state': state,
        'show_dialog': 'true',
    }
    authorization_url = f"{authorize_url}?{urlencode(authorization_params)}"
    next_step = (
        "Open authorization_url and approve the selected Spotify powers. Spotify returns to the local Hexis "
        "callback automatically. If it cannot, paste the full callback URL into complete_spotify_connection."
    )
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT start_connection_attempt(
                'spotify', $1::jsonb, ARRAY[]::text[], $2::jsonb,
                $3, $4, $5, $6, CURRENT_TIMESTAMP + INTERVAL '15 minutes'
            )
            """,
            json.dumps(caps),
            json.dumps({
                "setup_kind": "oauth2_authorization_code_pkce",
                "state_hash": hashlib.sha256(state.encode("utf-8")).hexdigest(),
                "redirect_uri": redirect_uri,
                "client_source": client_source,
                "secret_values_stored": False,
            }),
            authorization_url,
            next_step,
            source_channel,
            source_session_id,
        )
    payload = _json(raw) or {}
    pending_ref = f"{SPOTIFY_PENDING_PREFIX}{payload['attempt_id']}"
    save_auth(
        pending_ref,
        {
            "state": state,
            "verifier": verifier,
            "client_id": resolved_client_id,
            "redirect_uri": redirect_uri,
            "token_url": token_url,
            "api_base_url": api_base_url,
            "scopes": scopes,
            "capabilities": caps,
            "created_ms": now_ms(),
        },
    )
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE connection_attempts SET flow_state = flow_state || $2::jsonb, updated_at = CURRENT_TIMESTAMP WHERE id = $1::uuid",
            payload["attempt_id"],
            json.dumps({"pending_auth_ref": pending_ref}),
        )
    payload["pending_auth_ref"] = pending_ref
    payload["redirect_uri"] = redirect_uri
    return SpotifyOAuthStart(payload, pending_ref)


def _parse_authorization_response(value: str) -> tuple[str, str | None]:
    raw = str(value or "").strip()
    if not raw:
        raise SpotifyOAuthError("Paste the full Spotify callback URL or authorization code.")
    if not raw.startswith(("http://", "https://")):
        return raw, None
    params = parse_qs(urlparse(raw).query)
    if params.get("error"):
        raise SpotifyOAuthError(f"Spotify authorization failed: {params['error'][0]}")
    code = (params.get("code") or [""])[0]
    if not code:
        raise SpotifyOAuthError("The Spotify callback URL does not contain a code parameter.")
    return code, (params.get("state") or [None])[0]


def _expiry_iso(expires_in: int) -> str:
    value = datetime.fromtimestamp((now_ms() + int(expires_in) * 1000) / 1000, tz=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


async def _fetch_profile(api_base_url: str, access_token: str) -> dict[str, Any]:
    try:
        profile = await request_json(
            "spotify",
            "GET",
            f"{api_base_url}/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15.0,
            attempts=2,
            max_delay=3.0,
        )
    except IntegrationHttpError as exc:
        raise SpotifyOAuthError(format_provider_error("Spotify profile", exc)) from exc
    if not isinstance(profile, dict) or not profile.get("id"):
        raise SpotifyOAuthError("Spotify verification did not return a user ID.")
    return profile


async def complete_spotify_oauth(
    pool: Any,
    *,
    authorization_response: str,
    attempt_id: str | None = None,
) -> SpotifyOAuthComplete:
    if not attempt_id:
        async with pool.acquire() as conn:
            attempt_id = await conn.fetchval(
                """
                SELECT id::text FROM connection_attempts
                WHERE connector_id = 'spotify'
                  AND status IN ('pending_user', 'awaiting_input', 'error')
                ORDER BY created_at DESC LIMIT 1
                """
            )
    if not attempt_id:
        raise SpotifyOAuthError("No pending Spotify setup. Start with connect_spotify first.")
    pending_ref = f"{SPOTIFY_PENDING_PREFIX}{attempt_id}"
    pending = load_auth(pending_ref)
    if not isinstance(pending, dict):
        raise SpotifyOAuthError("The pending Spotify OAuth session is missing. Start Spotify setup again.")
    code, returned_state = _parse_authorization_response(authorization_response)
    if returned_state is not None and returned_state != pending.get("state"):
        raise SpotifyOAuthError("OAuth state mismatch. Start a fresh Spotify setup and use its newest browser tab.")

    async with pool.acquire() as conn:
        await conn.fetchval("SELECT mark_connection_attempt_exchanging($1::uuid)", attempt_id)
    try:
        token_data = await request_json(
            "spotify_oauth",
            "POST",
            str(pending["token_url"]),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "client_id": pending["client_id"],
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": pending["redirect_uri"],
                "code_verifier": pending["verifier"],
            },
            timeout=30.0,
            attempts=2,
            max_delay=5.0,
            retry_unsafe_methods=True,
        )
        if not isinstance(token_data, dict) or not isinstance(token_data.get("access_token"), str):
            raise SpotifyOAuthError("Spotify token exchange did not return an access token.")
        access_token = token_data["access_token"]
        profile = await _fetch_profile(str(pending["api_base_url"]), access_token)
        returned_scope = token_data.get("scope")
        scopes = returned_scope.split() if isinstance(returned_scope, str) else list(pending.get("scopes") or [])
        expires_in = int(token_data.get("expires_in") or 3600)
        account_key = f"spotify:{profile['id']}"
        display_name = str(profile.get("display_name") or profile["id"])
        credentials = {
            "type": "spotify_oauth2_pkce",
            "client_id": pending["client_id"],
            "access_token": access_token,
            "refresh_token": token_data.get("refresh_token") or "",
            "token_type": token_data.get("token_type") or "Bearer",
            "token_url": pending["token_url"],
            "api_base_url": pending["api_base_url"],
            "scopes": scopes,
            "expires_ms": now_ms() + expires_in * 1000,
            "expiry": _expiry_iso(expires_in),
            "account_key": account_key,
            "user_id": profile["id"],
            "display_name": display_name,
        }
        save_auth(SPOTIFY_DEFAULT_CREDENTIAL_REF, credentials)
        caps = list(pending.get("capabilities") or [])
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT complete_connection_attempt(
                    $1::uuid, $2, $3, $4, $5::text[], $6::jsonb, $7::jsonb
                )
                """,
                attempt_id,
                account_key,
                display_name,
                SPOTIFY_DEFAULT_CREDENTIAL_REF,
                scopes,
                json.dumps(caps),
                json.dumps({
                    "auth_store": "filesystem",
                    "api_base_url": pending["api_base_url"],
                    "user_id": profile["id"],
                    "product": profile.get("product"),
                    "country": profile.get("country"),
                    "secret_values_stored": False,
                }),
            )
        result = _json(raw) or {}
        delete_auth(pending_ref)
        return SpotifyOAuthComplete(
            account_key=str(result["account_key"]),
            display_name=str(result.get("display_name") or result["account_key"]),
            credential_ref=SPOTIFY_DEFAULT_CREDENTIAL_REF,
            granted_scopes=scopes,
            capabilities=list(result.get("capabilities") or caps),
        )
    except Exception as exc:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT mark_connection_attempt_error($1::uuid, $2)", attempt_id, str(exc)[:2000])
        if isinstance(exc, SpotifyOAuthError):
            raise
        if isinstance(exc, IntegrationHttpError):
            raise SpotifyOAuthError(format_provider_error("Spotify OAuth", exc)) from exc
        raise SpotifyOAuthError(str(exc)) from exc


def delete_spotify_credentials() -> None:
    delete_auth(SPOTIFY_DEFAULT_CREDENTIAL_REF)


async def refresh_spotify_credentials_if_needed(*, leeway_ms: int = 60_000) -> dict[str, Any]:
    with auth_lock(SPOTIFY_DEFAULT_CREDENTIAL_REF):
        credentials = load_auth(SPOTIFY_DEFAULT_CREDENTIAL_REF)
        if not isinstance(credentials, dict):
            raise SpotifyOAuthError("Spotify credentials are not saved. Use connect_spotify first.")
        if int(credentials.get("expires_ms") or 0) > now_ms() + leeway_ms and credentials.get("access_token"):
            return credentials
        refresh_token = str(credentials.get("refresh_token") or "").strip()
        if not refresh_token:
            raise SpotifyOAuthError("Spotify credentials have no refresh token. Reconnect Spotify.")
        try:
            token_data = await request_json(
                "spotify_oauth",
                "POST",
                str(credentials["token_url"]),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": credentials["client_id"],
                },
                timeout=30.0,
                attempts=2,
                max_delay=5.0,
                retry_unsafe_methods=True,
            )
        except IntegrationHttpError as exc:
            raise SpotifyOAuthError(format_provider_error("Spotify token refresh", exc)) from exc
        if not isinstance(token_data, dict) or not isinstance(token_data.get("access_token"), str):
            raise SpotifyOAuthError("Spotify token refresh did not return an access token.")
        expires_in = int(token_data.get("expires_in") or 3600)
        updated = {
            **credentials,
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token") or refresh_token,
            "expires_ms": now_ms() + expires_in * 1000,
            "expiry": _expiry_iso(expires_in),
        }
        if isinstance(token_data.get("scope"), str):
            updated["scopes"] = token_data["scope"].split()
        save_auth(SPOTIFY_DEFAULT_CREDENTIAL_REF, updated)
        return updated


async def spotify_api_request(
    pool: Any,
    method: str,
    path: str,
    *,
    capability: str,
    account_key: str | None = None,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    from services.life_integrations import connected_account, require_capability

    connection = await connected_account(pool, "spotify", account_key)
    require_capability(connection, capability)
    credentials = await refresh_spotify_credentials_if_needed()
    if credentials.get("account_key") != connection.get("account_key"):
        raise SpotifyOAuthError("Saved Spotify credentials do not match the selected connection. Reconnect Spotify.")
    response = await request_json_response(
        "spotify",
        method,
        f"{str(credentials['api_base_url']).rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {credentials['access_token']}"},
        params=params,
        json_body=body,
        timeout=30.0,
        attempts=3 if method.upper() == "GET" else 1,
    )
    return response.json_data if response.text else {}
