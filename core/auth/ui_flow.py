"""Short-lived OAuth sessions used by the browser initialization flow.

The provider modules remain responsible for OAuth protocol details and durable
credential storage.  This coordinator holds only transient PKCE/device-flow
state and returns redacted status payloads suitable for the browser.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse

from core.auth import create_state, generate_pkce
from core.auth.callback_server import run_callback_server

SUPPORTED_PROVIDERS = {
    "openai-codex",
    "chutes",
    "github-copilot",
    "qwen-portal",
    "minimax-portal",
    "google-gemini-cli",
    "google-antigravity",
}
MAX_AUTH_SESSIONS = 64

AuthFlowKind = Literal["authorization_code", "device_code"]
AuthSessionStatus = Literal[
    "awaiting_code",
    "waiting_for_user",
    "exchanging",
    "complete",
    "error",
    "expired",
]


class AuthFlowError(RuntimeError):
    """An expected, user-actionable authentication flow failure."""


@dataclass
class _AuthSession:
    id: str
    provider: str
    flow: AuthFlowKind
    status: AuthSessionStatus
    created_at: float
    expires_at: float
    authorization_url: str | None = None
    verification_uri: str | None = None
    user_code: str | None = None
    verifier: str | None = None
    state: str | None = None
    redirect_uri: str | None = None
    options: dict[str, str] = field(default_factory=dict)
    device: Any = None
    error: str | None = None
    callback_active: bool = False
    callback_cancel: threading.Event | None = field(default=None, repr=False)
    task: asyncio.Task[Any] | None = field(default=None, repr=False)


class AuthFlowCoordinator:
    """Coordinate browser-visible auth without exposing credentials."""

    def __init__(self, *, enable_callbacks: bool = True) -> None:
        self._sessions: dict[str, _AuthSession] = {}
        self._enable_callbacks = enable_callbacks

    async def start(
        self, provider: str, options: dict[str, str] | None = None
    ) -> dict[str, Any]:
        provider = _canonical_provider(provider)
        clean_options = _clean_options(options)
        self._prune_sessions()
        await self._cancel_active_provider_sessions(provider)

        try:
            if provider in {"github-copilot", "qwen-portal", "minimax-portal"}:
                session = await self._start_device_flow(provider, clean_options)
            else:
                session = self._start_authorization_code_flow(provider, clean_options)
        except AuthFlowError:
            raise
        except Exception as exc:
            raise AuthFlowError(_error_message(exc)) from exc

        self._sessions[session.id] = session
        if session.flow == "device_code":
            session.task = asyncio.create_task(self._poll_device_flow(session.id))
        else:
            callback = self._callback_target(session)
            if callback and self._enable_callbacks:
                session.callback_active = True
                session.callback_cancel = threading.Event()
                port, path = callback
                session.task = asyncio.create_task(
                    self._wait_for_callback(session.id, port, path)
                )
        return self._public_session(session)

    def status(self, provider: str) -> dict[str, Any]:
        provider = _canonical_provider(provider)
        creds = _load_credentials(provider)
        payload: dict[str, Any] = {"provider": provider, "configured": bool(creds)}
        if not creds:
            return payload

        expires_ms = getattr(creds, "expires_ms", None)
        if isinstance(expires_ms, (int, float)):
            payload["expires_in_seconds"] = int(
                (expires_ms - int(time.time() * 1000)) / 1000
            )
            payload["expires_at"] = datetime.fromtimestamp(
                expires_ms / 1000, tz=timezone.utc
            ).isoformat()
        for name in (
            "email",
            "account_id",
            "base_url",
            "project_id",
            "resource_url",
            "region",
        ):
            value = getattr(creds, name, None)
            if isinstance(value, str) and value:
                payload[name] = value
        return payload

    async def validate(self, provider: str) -> dict[str, Any]:
        """Refresh-check a stored login after an explicit user action."""
        provider = _canonical_provider(provider)
        if not _load_credentials(provider):
            return {"provider": provider, "configured": False}
        try:
            await _ensure_fresh_credentials(provider)
        except Exception as exc:
            raise AuthFlowError(
                f"The stored {provider} login could not be refreshed: {_error_message(exc)} "
                "Use Authenticate again and retry."
            ) from exc
        return self.status(provider)

    def session(self, session_id: str) -> dict[str, Any]:
        return self._public_session(self._get_session(session_id))

    async def complete(
        self, session_id: str, authorization_input: str
    ) -> dict[str, Any]:
        session = self._get_session(session_id)
        if session.flow != "authorization_code":
            raise AuthFlowError(
                "This provider completes automatically after device approval."
            )
        if session.status == "complete":
            return self._public_session(session)
        if session.status == "exchanging":
            return self._public_session(session)
        if session.status not in {"awaiting_code", "error"}:
            raise AuthFlowError(
                "This authorization attempt is no longer active. Start a new one."
            )

        code, returned_state = _parse_authorization_input(
            session.provider, authorization_input
        )
        if not code:
            raise AuthFlowError(
                "Paste the authorization code or the full redirect URL."
            )
        if returned_state and returned_state != session.state:
            raise AuthFlowError(
                "The authorization state does not match this attempt. Start again and use the newest browser tab."
            )

        await self._exchange_code(session, code)
        return self._public_session(session)

    async def close(self) -> None:
        tasks: list[asyncio.Task[Any]] = []
        for session in self._sessions.values():
            if session.callback_cancel:
                session.callback_cancel.set()
            if session.task:
                tasks.append(session.task)
                if session.flow == "device_code":
                    session.task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _start_authorization_code_flow(
        self, provider: str, options: dict[str, str]
    ) -> _AuthSession:
        verifier, challenge = generate_pkce()
        state = create_state()
        redirect_uri: str | None = None

        if provider == "openai-codex":
            from core.auth.openai_codex import (
                OPENAI_CODEX_REDIRECT_URI,
                build_authorize_url,
            )

            redirect_uri = OPENAI_CODEX_REDIRECT_URI
            authorization_url = build_authorize_url(challenge=challenge, state=state)
        elif provider == "chutes":
            from core.auth.chutes import build_authorize_url

            client_id = (
                options.get("client_id") or os.getenv("CHUTES_CLIENT_ID", "").strip()
            )
            if not client_id:
                raise AuthFlowError(
                    "Enter the OAuth client ID from your Chutes developer settings."
                )
            redirect_uri = options.get("redirect_uri") or os.getenv(
                "CHUTES_REDIRECT_URI", "http://localhost:11435/auth/callback"
            )
            options["client_id"] = client_id
            options["redirect_uri"] = redirect_uri
            authorization_url = build_authorize_url(
                challenge=challenge,
                state=state,
                client_id=client_id,
                redirect_uri=redirect_uri,
            )
        elif provider == "google-gemini-cli":
            from core.auth.google_gemini_cli import (
                GEMINI_CLI_REDIRECT_URI,
                build_authorize_url,
            )

            redirect_uri = GEMINI_CLI_REDIRECT_URI
            authorization_url = build_authorize_url(
                challenge=challenge,
                state=state,
                client_id=options.get("client_id"),
                client_secret=options.get("client_secret"),
            )
        elif provider == "google-antigravity":
            from core.auth.google_antigravity import (
                ANTIGRAVITY_REDIRECT_URI,
                build_authorize_url,
            )

            redirect_uri = ANTIGRAVITY_REDIRECT_URI
            authorization_url = build_authorize_url(
                challenge=challenge,
                state=state,
                client_id=options.get("client_id"),
                client_secret=options.get("client_secret"),
            )
        else:  # pragma: no cover - guarded by canonical provider validation
            raise AuthFlowError(f"Unsupported OAuth provider: {provider}")

        now = time.time()
        return _AuthSession(
            id=uuid.uuid4().hex,
            provider=provider,
            flow="authorization_code",
            status="awaiting_code",
            created_at=now,
            expires_at=now + 600,
            authorization_url=authorization_url,
            verifier=verifier,
            state=state,
            redirect_uri=redirect_uri,
            options=options,
        )

    async def _start_device_flow(
        self, provider: str, options: dict[str, str]
    ) -> _AuthSession:
        verifier: str | None = None
        if provider == "github-copilot":
            from core.auth.github_copilot import start_device_flow

            enterprise_domain = _normalize_domain(options.get("enterprise_domain", ""))
            if enterprise_domain:
                options["enterprise_domain"] = enterprise_domain
            domain = enterprise_domain or "github.com"
            device = await start_device_flow(domain)
            verification_uri = device.verification_uri
        elif provider == "qwen-portal":
            from core.auth.qwen_portal import start_device_flow

            device, verifier = await start_device_flow()
            verification_uri = (
                device.verification_uri_complete or device.verification_uri
            )
        else:
            from core.auth.minimax_portal import start_user_code_flow

            region = options.get("region", "global")
            if region not in {"global", "cn"}:
                raise AuthFlowError("MiniMax region must be global or cn.")
            options["region"] = region
            device, verifier = await start_user_code_flow(region)
            verification_uri = device.verification_uri

        now = time.time()
        return _AuthSession(
            id=uuid.uuid4().hex,
            provider=provider,
            flow="device_code",
            status="waiting_for_user",
            created_at=now,
            expires_at=now + max(30, int(device.expires_in)),
            verification_uri=verification_uri,
            user_code=device.user_code,
            verifier=verifier,
            options=options,
            device=device,
        )

    async def _poll_device_flow(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        try:
            if session.provider == "github-copilot":
                from core.auth.github_copilot import (
                    exchange_github_for_copilot,
                    poll_for_github_token,
                    save_credentials,
                )

                enterprise_domain = session.options.get("enterprise_domain") or None
                domain = enterprise_domain or "github.com"
                github_token = await poll_for_github_token(
                    domain,
                    session.device.device_code,
                    session.device.interval,
                    session.device.expires_in,
                )
                creds = await exchange_github_for_copilot(
                    github_token, enterprise_domain
                )
                save_credentials(creds)
            elif session.provider == "qwen-portal":
                from core.auth.qwen_portal import poll_for_token, save_credentials

                creds = await poll_for_token(
                    session.device.device_code,
                    session.verifier,
                    session.device.interval,
                    session.device.expires_in,
                )
                save_credentials(creds)
            else:
                from core.auth.minimax_portal import poll_for_token, save_credentials

                creds = await poll_for_token(
                    session.device.user_code,
                    session.verifier,
                    session.device.interval,
                    session.device.expires_in,
                    session.options.get("region", "global"),
                )
                save_credentials(creds)
            session.status = "complete"
            session.error = None
            session.options = {}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            session.status = "expired" if time.time() >= session.expires_at else "error"
            session.error = _error_message(exc)

    async def _wait_for_callback(self, session_id: str, port: int, path: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        timeout = max(5, min(180, int(session.expires_at - time.time())))
        try:
            result = await asyncio.to_thread(
                run_callback_server,
                port=port,
                callback_path=path,
                timeout_seconds=timeout,
                expected_state=session.state,
                cancel_event=session.callback_cancel,
            )
            session.callback_active = False
            if result and result.get("code") and session.status == "awaiting_code":
                await self._exchange_code(session, result["code"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            session.callback_active = False
            if session.status != "complete":
                session.error = _error_message(exc)

    async def _exchange_code(self, session: _AuthSession, code: str) -> None:
        if not session.verifier:
            raise AuthFlowError(
                "This authorization attempt is missing PKCE state. Start again."
            )
        session.status = "exchanging"
        session.error = None
        try:
            if session.provider == "openai-codex":
                from core.auth.openai_codex import (
                    exchange_authorization_code,
                    save_openai_codex_credentials,
                )

                creds = await exchange_authorization_code(
                    code=code, verifier=session.verifier
                )
                save_openai_codex_credentials(creds)
            elif session.provider == "chutes":
                from core.auth.chutes import exchange_code, save_credentials

                creds = await exchange_code(
                    code=code,
                    verifier=session.verifier,
                    client_id=session.options["client_id"],
                    redirect_uri=session.options["redirect_uri"],
                    client_secret=session.options.get("client_secret")
                    or os.getenv("CHUTES_CLIENT_SECRET"),
                )
                save_credentials(creds)
            elif session.provider == "google-gemini-cli":
                from core.auth.google_gemini_cli import complete_login, save_credentials

                creds = await complete_login(
                    code,
                    session.verifier,
                    client_id=session.options.get("client_id"),
                    client_secret=session.options.get("client_secret"),
                )
                save_credentials(creds)
            else:
                from core.auth.google_antigravity import (
                    complete_login,
                    save_credentials,
                )

                creds = await complete_login(
                    code,
                    session.verifier,
                    client_id=session.options.get("client_id"),
                    client_secret=session.options.get("client_secret"),
                )
                save_credentials(creds)
            session.status = "complete"
            session.error = None
            session.callback_active = False
            session.options = {}
            if session.callback_cancel:
                session.callback_cancel.set()
        except Exception as exc:
            session.status = "awaiting_code"
            session.error = _error_message(exc)
            raise AuthFlowError(session.error) from exc

    def _callback_target(self, session: _AuthSession) -> tuple[int, str] | None:
        if not session.redirect_uri:
            return None
        parsed = urlparse(session.redirect_uri)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}:
            return None
        port = parsed.port or 80
        return port, parsed.path or "/auth/callback"

    def _get_session(self, session_id: str) -> _AuthSession:
        session = self._sessions.get(session_id)
        if not session:
            raise AuthFlowError("Authorization attempt not found. Start a new one.")
        if (
            session.status not in {"complete", "error", "expired"}
            and time.time() >= session.expires_at
        ):
            session.status = "expired"
            session.callback_active = False
            session.options = {}
            if session.task:
                if session.callback_cancel:
                    session.callback_cancel.set()
                session.task.cancel()
        return session

    async def _cancel_active_provider_sessions(self, provider: str) -> None:
        tasks: list[asyncio.Task[Any]] = []
        for session in self._sessions.values():
            if session.provider != provider or session.status in {
                "complete",
                "error",
                "expired",
            }:
                continue
            session.status = "expired"
            session.callback_active = False
            session.options = {}
            if session.callback_cancel:
                session.callback_cancel.set()
            if session.task:
                tasks.append(session.task)
                if session.flow == "device_code":
                    session.task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _prune_sessions(self) -> None:
        if len(self._sessions) < MAX_AUTH_SESSIONS:
            return
        removable = sorted(
            (
                session
                for session in self._sessions.values()
                if session.status in {"complete", "error", "expired"}
                and (session.task is None or session.task.done())
            ),
            key=lambda session: session.created_at,
        )
        for session in removable:
            if len(self._sessions) < MAX_AUTH_SESSIONS // 2:
                break
            self._sessions.pop(session.id, None)

    def _public_session(self, session: _AuthSession) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session.id,
            "provider": session.provider,
            "flow": session.flow,
            "status": session.status,
            "expires_in_seconds": max(0, int(session.expires_at - time.time())),
            "callback_active": session.callback_active,
        }
        for key in ("authorization_url", "verification_uri", "user_code", "error"):
            value = getattr(session, key)
            if value:
                payload[key] = value
        if session.status == "complete":
            payload["credential"] = self.status(session.provider)
        return payload


def _canonical_provider(provider: str) -> str:
    value = (provider or "").strip().lower()
    if value not in SUPPORTED_PROVIDERS:
        raise AuthFlowError(f"Unsupported OAuth provider: {provider or 'missing'}")
    return value


def _clean_options(options: dict[str, str] | None) -> dict[str, str]:
    if not options:
        return {}
    return {
        str(key): str(value).strip()
        for key, value in options.items()
        if value is not None and str(value).strip()
    }


def _normalize_domain(value: str) -> str:
    domain = value.strip().lower()
    if not domain:
        return ""
    parsed = urlparse(domain if "://" in domain else f"https://{domain}")
    if (
        not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise AuthFlowError(
            "Enter only the GitHub enterprise hostname, such as github.example.com."
        )
    return parsed.hostname


def _parse_authorization_input(
    provider: str, value: str
) -> tuple[str | None, str | None]:
    from core.auth.openai_codex import parse_authorization_input
    return parse_authorization_input(value)


def _load_credentials(provider: str) -> Any:
    if provider == "openai-codex":
        from core.auth.openai_codex import load_openai_codex_credentials

        return load_openai_codex_credentials()
    import importlib

    module = importlib.import_module(
        {
            "chutes": "core.auth.chutes",
            "github-copilot": "core.auth.github_copilot",
            "qwen-portal": "core.auth.qwen_portal",
            "minimax-portal": "core.auth.minimax_portal",
            "google-gemini-cli": "core.auth.google_gemini_cli",
            "google-antigravity": "core.auth.google_antigravity",
        }[provider]
    )
    return module.load_credentials()


async def _ensure_fresh_credentials(provider: str) -> Any:
    if provider == "openai-codex":
        from core.auth.openai_codex import ensure_fresh_openai_codex_credentials

        return await ensure_fresh_openai_codex_credentials(skew_seconds=0)

    import importlib

    module = importlib.import_module(
        {
            "chutes": "core.auth.chutes",
            "github-copilot": "core.auth.github_copilot",
            "qwen-portal": "core.auth.qwen_portal",
            "minimax-portal": "core.auth.minimax_portal",
            "google-gemini-cli": "core.auth.google_gemini_cli",
            "google-antigravity": "core.auth.google_antigravity",
        }[provider]
    )
    return await module.ensure_fresh_credentials(skew_seconds=0)


def _error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or "Authentication failed. Start a new attempt and try again."


auth_flow_coordinator = AuthFlowCoordinator()
