"""Gmail OAuth setup helpers for first-class personal-data connectors."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from core.auth.store import auth_lock, delete_auth, load_auth, save_auth
from core.auth.utils import create_state, generate_pkce, now_ms
from core.integration_reliability import (
    IntegrationHttpError,
    format_provider_error,
    request_json,
)

GMAIL_CONNECTOR_ID = "gmail"
GMAIL_DEFAULT_CREDENTIAL_REF = "integration.gmail.default"
GMAIL_CLIENT_SECRET_REF = "integration.gmail.client"
GMAIL_PENDING_PREFIX = "integration.gmail.pending."
GMAIL_CALLBACK_PATH = "/api/integrations/gmail/callback"
GMAIL_DEFAULT_API_BASE_URL = "http://127.0.0.1:43817"

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_GMAIL_PROFILE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"
GMAIL_REDIRECT_URI = GMAIL_DEFAULT_API_BASE_URL

_CLIENT_SECRET_ENV_JSON = ("GOOGLE_GMAIL_CLIENT_SECRET_JSON", "GOOGLE_CLIENT_SECRET_JSON")
_CLIENT_SECRET_ENV_PATH = ("GOOGLE_GMAIL_CLIENT_SECRET_PATH", "GOOGLE_CLIENT_SECRET_PATH")
_HEXIS_CLIENT_ID_ENV = ("HEXIS_GMAIL_OAUTH_CLIENT_ID", "GOOGLE_GMAIL_OAUTH_CLIENT_ID")
_HEXIS_CLIENT_SECRET_ENV = ("HEXIS_GMAIL_OAUTH_CLIENT_SECRET", "GOOGLE_GMAIL_OAUTH_CLIENT_SECRET")
_GMAIL_REDIRECT_URI_ENV = ("HEXIS_GMAIL_OAUTH_REDIRECT_URI", "GOOGLE_GMAIL_OAUTH_REDIRECT_URI")
_HEXIS_API_BASE_URL_ENV = ("HEXIS_API_URL", "HEXIS_API_BASE_URL")
_DISABLE_BUNDLED_CLIENT_ENV = "HEXIS_GMAIL_OAUTH_DISABLE_BUNDLED"
GMAIL_BUNDLED_CLIENT_SECRET_PATHS = (
    Path(__file__).resolve().with_name("gmail-credentials.json"),
    Path(__file__).resolve().parents[2] / "gmail-credentials.json",
)
GMAIL_SETUP_DOCS_URL = "https://console.cloud.google.com/apis/credentials"
GMAIL_SETUP_STEPS = [
    "Open the Google setup page.",
    "Create or choose a project named Hexis.",
    "Enable the Gmail API for that project.",
    "Set up the app consent screen. For a personal Gmail account, choose External and add your Gmail address as a test user if Google asks.",
    "On the Credentials page, click Create credentials, choose Google's sign-in client option, set Application type to Desktop app, and name it Hexis.",
    "Download the setup file Google gives you.",
    "Upload that setup file here, then start Google sign-in.",
]
GMAIL_SETUP_NEXT_STEP = (
    "Gmail sign-in is not enabled for this local Hexis build yet. "
    "Use the setup guide in the panel to enable it once, then start Google sign-in."
)
GMAIL_SETUP_TECHNICAL_NEXT_STEP = (
    "For developers and hosted builds: configure HEXIS_GMAIL_OAUTH_CLIENT_ID and "
    "HEXIS_GMAIL_OAUTH_CLIENT_SECRET in the Hexis API environment to make built-in "
    "Google sign-in available."
)


class GmailOAuthError(RuntimeError):
    """Expected, user-actionable Gmail setup failure."""


@dataclass(frozen=True)
class GmailOAuthStart:
    attempt_payload: dict[str, Any]
    pending_auth_ref: str


@dataclass(frozen=True)
class GmailOAuthComplete:
    account_key: str
    display_name: str
    credential_ref: str
    granted_scopes: list[str]
    capabilities: list[str]
    credential_payload: dict[str, Any]


def _json_arg(value: Any) -> str:
    return json.dumps(value)


async def prepare_gmail_connection_attempt(pool: Any, capabilities: Any = None) -> dict[str, Any]:
    """Ask the database to normalize capabilities and derive OAuth scopes."""
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT prepare_connection_attempt('gmail', $1::jsonb)",
                _json_arg(capabilities) if capabilities is not None else None,
            )
    except Exception as exc:
        raise GmailOAuthError(str(exc)) from exc
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        raise GmailOAuthError("Gmail connector preparation returned an invalid payload.")
    return payload


def _client_section(payload: dict[str, Any], *, require_installed: bool = False) -> dict[str, Any]:
    return _client_section_with_kind(payload, require_installed=require_installed)[1]


def _client_section_with_kind(
    payload: dict[str, Any],
    *,
    require_installed: bool = False,
) -> tuple[str, dict[str, Any]]:
    kind = "installed" if require_installed or isinstance(payload.get("installed"), dict) else "web"
    section = payload.get("installed") if kind == "installed" else payload.get("web")
    if not isinstance(section, dict):
        if require_installed:
            raise GmailOAuthError(
                "The bundled Gmail sign-in setup must be a Google Desktop app credential."
            )
        raise GmailOAuthError(
            "The Google setup file must contain an 'installed' or 'web' object. "
            "Use the Gmail setup guide to upload the downloaded setup file."
        )
    if not isinstance(section.get("client_id"), str) or not isinstance(section.get("client_secret"), str):
        raise GmailOAuthError("The Google setup file is missing client_id or client_secret.")
    return kind, section


def _clean_redirect_uri(value: str) -> str:
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        raise GmailOAuthError(f"Invalid Gmail OAuth redirect URI: {value}")
    return parsed._replace(params="", query="", fragment="").geturl().rstrip("/")


def configured_gmail_redirect_uri() -> str:
    """Return the loopback redirect URI Hexis will use for Gmail OAuth."""
    raw, _ = _first_env(_GMAIL_REDIRECT_URI_ENV)
    if raw:
        return _clean_redirect_uri(raw)

    api_base, _ = _first_env(_HEXIS_API_BASE_URL_ENV)
    if api_base:
        return _clean_redirect_uri(api_base)

    return GMAIL_REDIRECT_URI


def _select_redirect_uri(client: dict[str, Any]) -> str:
    redirects = client.get("redirect_uris")
    if isinstance(redirects, list):
        for value in redirects:
            if isinstance(value, str) and value.startswith(("http://localhost", "http://127.0.0.1", "http://[::1]")):
                return value
    return GMAIL_REDIRECT_URI


def _load_client_secret_path(path: str) -> dict[str, Any]:
    src = Path(path).expanduser()
    if not src.exists():
        raise GmailOAuthError(f"Google setup file not found: {src}")
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GmailOAuthError(f"Google setup file is not valid JSON: {src}") from exc
    if not isinstance(payload, dict):
        raise GmailOAuthError("Google setup file must contain a JSON object.")
    _client_section(payload)
    return payload


def _load_env_client_secret() -> tuple[dict[str, Any] | None, str | None]:
    for name in _CLIENT_SECRET_ENV_JSON:
        raw = os.getenv(name)
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GmailOAuthError(f"{name} is not valid JSON.") from exc
            if not isinstance(payload, dict):
                raise GmailOAuthError(f"{name} must contain a JSON object.")
            _client_section(payload)
            return payload, name
    for name in _CLIENT_SECRET_ENV_PATH:
        raw = os.getenv(name)
        if raw:
            return _load_client_secret_path(raw), name
    return None, None


def _first_env(names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip(), name
    return None, None


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_bundled_client_secret() -> tuple[dict[str, Any] | None, str | None]:
    if _truthy_env(_DISABLE_BUNDLED_CLIENT_ENV):
        return None, None
    for path in GMAIL_BUNDLED_CLIENT_SECRET_PATHS:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GmailOAuthError(f"Bundled Gmail sign-in setup is not valid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise GmailOAuthError("Bundled Gmail sign-in setup must contain a JSON object.")
        _client_section(payload, require_installed=True)
        return payload, f"bundled:{path.name}"
    return None, None


def has_bundled_gmail_oauth_client() -> bool:
    try:
        payload, _ = _load_bundled_client_secret()
        return payload is not None
    except GmailOAuthError:
        return False


def _load_hexis_env_client_secret() -> tuple[dict[str, Any] | None, str | None]:
    client_id, id_name = _first_env(_HEXIS_CLIENT_ID_ENV)
    client_secret, secret_name = _first_env(_HEXIS_CLIENT_SECRET_ENV)
    if not client_id and not client_secret:
        return None, None
    if not client_id or not client_secret:
        missing = "client id" if not client_id else "client secret"
        raise GmailOAuthError(f"Hexis Gmail OAuth client is partially configured; missing {missing}.")
    payload = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": GOOGLE_AUTH_URL,
            "token_uri": GOOGLE_TOKEN_URL,
            "redirect_uris": [configured_gmail_redirect_uri()],
        }
    }
    _client_section(payload)
    return payload, f"{id_name}+{secret_name}"


def _load_hexis_client_secret() -> tuple[dict[str, Any] | None, str | None]:
    payload, source = _load_hexis_env_client_secret()
    if payload:
        return payload, source
    return _load_bundled_client_secret()


def has_hexis_gmail_oauth_client() -> bool:
    client_id, _ = _first_env(_HEXIS_CLIENT_ID_ENV)
    client_secret, _ = _first_env(_HEXIS_CLIENT_SECRET_ENV)
    return bool(client_id and client_secret) or has_bundled_gmail_oauth_client()


def has_env_gmail_client_secret() -> bool:
    """Return whether an explicitly configured Gmail client secret env value exists.

    This is a presence check only. The setup flow must still ask the user before
    consuming environment credentials, because ambient state should not be used
    silently.
    """
    return any(os.getenv(name) for name in (*_CLIENT_SECRET_ENV_JSON, *_CLIENT_SECRET_ENV_PATH))


def has_saved_gmail_client_secret() -> bool:
    return isinstance(load_auth(GMAIL_CLIENT_SECRET_REF), dict)


def save_gmail_client_secret_payload(payload: dict[str, Any], *, source: str = "upload") -> dict[str, Any]:
    """Validate and save a Google setup file payload without returning secrets."""
    if not isinstance(payload, dict):
        raise GmailOAuthError("Google setup file upload must be a JSON object.")
    client = _client_section(payload)
    save_auth(GMAIL_CLIENT_SECRET_REF, payload)
    client_id = str(client["client_id"])
    suffix = client_id[-8:] if len(client_id) > 8 else client_id
    return {
        "status": "client_secret_saved",
        "connector_id": GMAIL_CONNECTOR_ID,
        "client_secret_saved": True,
        "client_id_hint": f"...{suffix}" if len(client_id) > len(suffix) else suffix,
        "source": source,
        "next_step": "Start Google sign-in from the setup panel.",
    }


def load_default_credentials() -> dict[str, Any] | None:
    value = load_auth(GMAIL_DEFAULT_CREDENTIAL_REF)
    return value if isinstance(value, dict) else None


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


async def refresh_default_credentials_if_needed(*, leeway_ms: int = 60_000) -> dict[str, Any]:
    """Return saved Gmail credentials, refreshing the access token when stale.

    This deliberately uses only the credential selected during Hexis Gmail setup.
    It does not fall back to ambient Google auth state.
    """
    with auth_lock(GMAIL_DEFAULT_CREDENTIAL_REF):
        credentials = load_default_credentials()
        if not credentials:
            raise GmailOAuthError("Gmail credentials are not saved. Use connect_gmail first.")

        if _expiry_ms(credentials) > now_ms() + leeway_ms and credentials.get("token"):
            return credentials

        refresh_token = credentials.get("refresh_token")
        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")
        if not all(isinstance(v, str) and v.strip() for v in (refresh_token, client_id, client_secret)):
            raise GmailOAuthError(
                "Saved Gmail credentials cannot refresh. Reconnect Gmail with connect_gmail."
            )

        async def _refresh() -> dict[str, Any]:
            try:
                data = await request_json(
                    "google_oauth",
                    "POST",
                    str(credentials.get("token_uri") or GOOGLE_TOKEN_URL),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    },
                    timeout=30.0,
                    attempts=3,
                    max_delay=15.0,
                    retry_unsafe_methods=True,
                )
            except IntegrationHttpError as exc:
                raise GmailOAuthError(format_provider_error("Google token refresh", exc)) from exc
            if not isinstance(data, dict) or not isinstance(data.get("access_token"), str):
                raise GmailOAuthError("Google token refresh did not return an access token.")
            return data

        token_data = await _refresh()
        expires_in = int(token_data.get("expires_in") or 3600)
        refreshed = {
            **credentials,
            "token": token_data["access_token"],
            "expiry": _expiry_iso(expires_in),
            "expires_ms": now_ms() + expires_in * 1000,
        }
        scopes = token_data.get("scope")
        if isinstance(scopes, str) and scopes.strip():
            refreshed["scopes"] = scopes.split()
        save_auth(GMAIL_DEFAULT_CREDENTIAL_REF, refreshed)
        return refreshed


def delete_default_credentials() -> None:
    delete_auth(GMAIL_DEFAULT_CREDENTIAL_REF)


def _resolve_client_secret(
    *,
    client_secret_path: str | None = None,
    use_hexis_oauth_client: bool = True,
    use_env_client_secret: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    if client_secret_path and client_secret_path.strip():
        payload = _load_client_secret_path(client_secret_path)
        save_auth(GMAIL_CLIENT_SECRET_REF, payload)
        return payload, "file"

    stored = load_auth(GMAIL_CLIENT_SECRET_REF)
    if isinstance(stored, dict):
        _client_section(stored)
        return stored, "stored"

    if use_hexis_oauth_client:
        payload, env_names = _load_hexis_client_secret()
        if payload:
            save_auth(GMAIL_CLIENT_SECRET_REF, payload)
            return payload, env_names or "hexis_oauth_client"

    if use_env_client_secret:
        payload, env_name = _load_env_client_secret()
        if payload:
            save_auth(GMAIL_CLIENT_SECRET_REF, payload)
            return payload, env_name

    return None, None


def build_client_secret_needed_payload() -> dict[str, Any]:
    return {
        "status": "needs_client_secret",
        "connector_id": GMAIL_CONNECTOR_ID,
        "client_secret_saved": False,
        "accepted_inputs": [
            "hexis_oauth_client",
            "advanced_self_hosted",
            "client_secret_file",
            "client_secret_path",
            "use_env_client_secret",
        ],
        "env_options": [*_CLIENT_SECRET_ENV_PATH, *_CLIENT_SECRET_ENV_JSON],
        "hexis_oauth_client_env": [*_HEXIS_CLIENT_ID_ENV, *_HEXIS_CLIENT_SECRET_ENV],
        "hexis_oauth_client_available": has_hexis_gmail_oauth_client(),
        "bundled_oauth_client_available": has_bundled_gmail_oauth_client(),
        "env_client_secret_available": has_env_gmail_client_secret(),
        "next_step": GMAIL_SETUP_NEXT_STEP,
        "technical_next_step": GMAIL_SETUP_TECHNICAL_NEXT_STEP,
        "setup_steps": list(GMAIL_SETUP_STEPS),
        "docs_url": GMAIL_SETUP_DOCS_URL,
    }


async def start_gmail_oauth(
    pool: Any,
    *,
    capabilities: Any = None,
    client_secret_path: str | None = None,
    use_hexis_oauth_client: bool = True,
    use_env_client_secret: bool = False,
    source_channel: str | None = None,
    source_session_id: str | None = None,
) -> GmailOAuthStart | dict[str, Any]:
    """Create a DB connection attempt and persisted pending PKCE state."""
    client_secret, source = _resolve_client_secret(
        client_secret_path=client_secret_path,
        use_hexis_oauth_client=use_hexis_oauth_client,
        use_env_client_secret=use_env_client_secret,
    )
    if not client_secret:
        return build_client_secret_needed_payload()

    prepared = await prepare_gmail_connection_attempt(pool, capabilities)
    caps = list(prepared.get("capabilities") or prepared.get("requested_capabilities") or [])
    scopes = list(prepared.get("requested_scopes") or [])
    if not caps or not scopes:
        raise GmailOAuthError("Gmail connector preparation did not return capabilities and scopes.")
    client_kind, client = _client_section_with_kind(client_secret)
    redirect_uri = (
        configured_gmail_redirect_uri()
        if client_kind == "installed"
        else _select_redirect_uri(client)
    )
    verifier, challenge = generate_pkce()
    state = create_state()
    params = {
        "client_id": client["client_id"],
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    authorization_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    next_step = (
        "Open authorization_url and approve the requested Gmail scopes. Hexis will finish "
        "the connection automatically when Google returns to the local callback."
    )
    flow_state = {
        "pending_auth_ref": None,
        "state_hash": hashlib.sha256(state.encode("utf-8")).hexdigest(),
        "redirect_uri": redirect_uri,
        "callback_path": GMAIL_CALLBACK_PATH,
        "client_secret_source": source,
        "scope_count": len(scopes),
    }

    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT start_connection_attempt(
                'gmail',
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
    pending_ref = f"{GMAIL_PENDING_PREFIX}{payload['attempt_id']}"
    save_auth(
        pending_ref,
        {
            "state": state,
            "verifier": verifier,
            "redirect_uri": redirect_uri,
            "client_ref": GMAIL_CLIENT_SECRET_REF,
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
    return GmailOAuthStart(attempt_payload=payload, pending_auth_ref=pending_ref)


def parse_authorization_response(value: str) -> tuple[str, str | None]:
    raw = (value or "").strip()
    if not raw:
        raise GmailOAuthError("Paste the full redirected URL or authorization code.")
    if not raw.startswith("http"):
        return raw, None
    parsed = urlparse(raw)
    params = parse_qs(parsed.query)
    if params.get("error"):
        raise GmailOAuthError(f"Google authorization failed: {params['error'][0]}")
    code = (params.get("code") or [""])[0]
    if not code:
        raise GmailOAuthError("The redirected URL does not contain a code parameter.")
    state = (params.get("state") or [None])[0]
    return code, state


async def _exchange_code(
    *,
    code: str,
    verifier: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict[str, Any]:
    try:
        data = await request_json(
            "google_oauth",
            "POST",
            GOOGLE_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
            timeout=30.0,
            attempts=3,
            max_delay=15.0,
            retry_unsafe_methods=True,
        )
    except IntegrationHttpError as exc:
        raise GmailOAuthError(format_provider_error("Google token exchange", exc)) from exc
    if not isinstance(data, dict) or not isinstance(data.get("access_token"), str):
        raise GmailOAuthError("Google token exchange did not return an access token.")
    return data


async def _fetch_account_email(access_token: str) -> str | None:
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        payload = await request_json(
            "gmail_profile",
            "GET",
            GOOGLE_GMAIL_PROFILE_URL,
            headers=headers,
            timeout=10.0,
            attempts=2,
            max_delay=2.0,
        )
        email = payload.get("emailAddress") if isinstance(payload, dict) else None
        if isinstance(email, str) and email.strip():
            return email.strip().lower()
    except Exception:
        pass
    try:
        payload = await request_json(
            "google_userinfo",
            "GET",
            GOOGLE_USERINFO_URL,
            headers=headers,
            timeout=10.0,
            attempts=2,
            max_delay=2.0,
        )
        email = payload.get("email") if isinstance(payload, dict) else None
        if isinstance(email, str) and email.strip():
            return email.strip().lower()
    except Exception:
        pass
    return None


def _expiry_iso(expires_in: int) -> str:
    expiry = datetime.fromtimestamp((now_ms() + int(expires_in) * 1000) / 1000, tz=timezone.utc)
    return expiry.isoformat().replace("+00:00", "Z")


async def complete_gmail_oauth(
    pool: Any,
    *,
    authorization_response: str,
    attempt_id: str | None = None,
) -> GmailOAuthComplete:
    code, returned_state = parse_authorization_response(authorization_response)
    if not attempt_id:
        async with pool.acquire() as conn:
            if returned_state:
                attempt_id = await conn.fetchval(
                    """
                    SELECT id::text
                    FROM connection_attempts
                    WHERE connector_id = 'gmail'
                      AND status IN ('pending_user', 'awaiting_input', 'error')
                      AND flow_state->>'state_hash' = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    hashlib.sha256(returned_state.encode("utf-8")).hexdigest(),
                )
            if not attempt_id:
                attempt_id = await conn.fetchval(
                    """
                    SELECT id::text
                    FROM connection_attempts
                    WHERE connector_id = 'gmail'
                      AND status IN ('pending_user', 'awaiting_input', 'error')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
    if not attempt_id:
        raise GmailOAuthError("No pending Gmail connection attempt. Start with connect_gmail first.")

    pending_ref = f"{GMAIL_PENDING_PREFIX}{attempt_id}"
    pending = load_auth(pending_ref)
    if not isinstance(pending, dict):
        raise GmailOAuthError("The pending Gmail OAuth session expired or is missing. Start connect_gmail again.")

    if returned_state and returned_state != pending.get("state"):
        raise GmailOAuthError("OAuth state mismatch. Start a fresh Gmail connection attempt and use the newest browser tab.")

    client_secret = load_auth(str(pending.get("client_ref") or GMAIL_CLIENT_SECRET_REF))
    if not isinstance(client_secret, dict):
        raise GmailOAuthError(
            "Stored Gmail setup is missing. Open Gmail setup again, follow the guide if needed, then start Google sign-in."
        )
    client = _client_section(client_secret)

    async with pool.acquire() as conn:
        await conn.fetchval("SELECT mark_connection_attempt_exchanging($1::uuid)", attempt_id)

    try:
        token_data = await _exchange_code(
            code=code,
            verifier=str(pending["verifier"]),
            client_id=client["client_id"],
            client_secret=client["client_secret"],
            redirect_uri=str(pending.get("redirect_uri") or GMAIL_REDIRECT_URI),
        )
        access_token = token_data["access_token"]
        scopes = token_data.get("scope")
        granted_scopes = scopes.split() if isinstance(scopes, str) and scopes.strip() else list(pending.get("scopes") or [])
        account_email = await _fetch_account_email(access_token)
        account_key = account_email or f"gmail:{attempt_id}"
        expires_in = int(token_data.get("expires_in") or 3600)
        credential_payload = {
            "type": "authorized_user",
            "client_id": client["client_id"],
            "client_secret": client["client_secret"],
            "refresh_token": token_data.get("refresh_token", ""),
            "token": access_token,
            "token_uri": GOOGLE_TOKEN_URL,
            "scopes": granted_scopes,
            "expiry": _expiry_iso(expires_in),
            "expires_ms": now_ms() + expires_in * 1000,
            "account_email": account_email,
        }
        save_auth(GMAIL_DEFAULT_CREDENTIAL_REF, credential_payload)
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
                account_key,
                account_email or "Gmail account",
                GMAIL_DEFAULT_CREDENTIAL_REF,
                granted_scopes,
                json.dumps(caps),
                json.dumps({"auth_store": "filesystem", "token_uri": GOOGLE_TOKEN_URL}),
            )
        delete_auth(pending_ref)
        result = json.loads(raw) if isinstance(raw, str) else raw
        result_caps = list(result.get("capabilities") or caps)
        return GmailOAuthComplete(
            account_key=str(result["account_key"]),
            display_name=str(result.get("display_name") or result["account_key"]),
            credential_ref=GMAIL_DEFAULT_CREDENTIAL_REF,
            granted_scopes=granted_scopes,
            capabilities=result_caps,
            credential_payload=credential_payload,
        )
    except Exception as exc:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT mark_connection_attempt_error($1::uuid, $2)",
                attempt_id,
                str(exc),
            )
        if isinstance(exc, GmailOAuthError):
            raise
        raise GmailOAuthError(str(exc)) from exc
