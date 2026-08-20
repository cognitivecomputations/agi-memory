"""CLI auth sub-commands for all OAuth / token providers.

Public API
----------
``register_auth_subparsers(auth_parser, db_parent)``
    Wire all provider subparsers under ``hexis auth``.

``dispatch_auth_command(func, args, dsn)``
    Handle all ``auth_*`` func strings.  Returns an exit code.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import sys
import time
from typing import Any


def _print_err(msg: str) -> None:
    sys.stderr.write(msg + "\n")


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------

def register_auth_subparsers(
    auth_sub: argparse._SubParsersAction,  # type: ignore[type-arg]
    db_parent: argparse.ArgumentParser,
) -> None:
    """Register all provider subparsers under the ``auth`` command."""

    # ── OpenAI Codex ──────────────────────────────────────────────
    oai = auth_sub.add_parser("openai-codex", parents=[db_parent], help="ChatGPT Plus/Pro (Codex OAuth)")
    oai_sub = oai.add_subparsers(dest="openai_codex_command")
    _login = oai_sub.add_parser("login", parents=[db_parent], help="Login via browser OAuth (PKCE)")
    _login.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    _login.add_argument("--timeout-seconds", type=int, default=60, help="Callback wait timeout")
    _login.set_defaults(func="auth_openai_codex_login")
    _status = oai_sub.add_parser("status", parents=[db_parent], help="Show current OAuth status")
    _status.add_argument("--json", action="store_true", help="Output JSON")
    _status.set_defaults(func="auth_openai_codex_status")
    _logout = oai_sub.add_parser("logout", parents=[db_parent], help="Delete stored credentials")
    _logout.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    _logout.set_defaults(func="auth_openai_codex_logout")
    oai.set_defaults(func="auth_openai_codex")

    # ── Anthropic ─────────────────────────────────────────────────
    ant = auth_sub.add_parser("anthropic", parents=[db_parent], help="Anthropic Claude (setup-token)")
    ant_sub = ant.add_subparsers(dest="anthropic_command")
    _st = ant_sub.add_parser("setup-token", parents=[db_parent], help="Paste a setup token")
    _st.add_argument("--token", default=None, help="Token value (prompted if omitted)")
    _st.set_defaults(func="auth_anthropic_setup_token")
    _as = ant_sub.add_parser("status", parents=[db_parent], help="Show setup-token status")
    _as.add_argument("--json", action="store_true", help="Output JSON")
    _as.set_defaults(func="auth_anthropic_status")
    _al = ant_sub.add_parser("logout", parents=[db_parent], help="Delete stored credentials")
    _al.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    _al.set_defaults(func="auth_anthropic_logout")
    ant.set_defaults(func="auth_anthropic")

    # ── Chutes ────────────────────────────────────────────────────
    _register_pkce_provider(auth_sub, db_parent, "chutes", "Chutes AI (PKCE OAuth)", extra_login_args=[
        ("--client-id", {"default": None, "help": "Override Chutes client ID"}),
    ])

    # ── GitHub Copilot ────────────────────────────────────────────
    ghc = auth_sub.add_parser("github-copilot", parents=[db_parent], help="GitHub Copilot (device code)")
    ghc_sub = ghc.add_subparsers(dest="github_copilot_command")
    _ghcl = ghc_sub.add_parser("login", parents=[db_parent], help="Login via device code flow")
    _ghcl.add_argument("--enterprise-domain", default=None, help="GitHub Enterprise domain")
    _ghcl.add_argument("--timeout-seconds", type=int, default=300, help="Polling timeout")
    _ghcl.set_defaults(func="auth_github_copilot_login")
    _ghcs = ghc_sub.add_parser("status", parents=[db_parent], help="Show status")
    _ghcs.add_argument("--json", action="store_true", help="Output JSON")
    _ghcs.set_defaults(func="auth_github_copilot_status")
    _ghco = ghc_sub.add_parser("logout", parents=[db_parent], help="Delete stored credentials")
    _ghco.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    _ghco.set_defaults(func="auth_github_copilot_logout")
    ghc.set_defaults(func="auth_github_copilot")

    # ── Qwen Portal ───────────────────────────────────────────────
    qw = auth_sub.add_parser("qwen-portal", parents=[db_parent], help="Qwen Portal (device code)")
    qw_sub = qw.add_subparsers(dest="qwen_portal_command")
    _qwl = qw_sub.add_parser("login", parents=[db_parent], help="Login via device code flow")
    _qwl.add_argument("--timeout-seconds", type=int, default=300, help="Polling timeout")
    _qwl.set_defaults(func="auth_qwen_portal_login")
    _qws = qw_sub.add_parser("status", parents=[db_parent], help="Show status")
    _qws.add_argument("--json", action="store_true", help="Output JSON")
    _qws.set_defaults(func="auth_qwen_portal_status")
    _qwo = qw_sub.add_parser("logout", parents=[db_parent], help="Delete stored credentials")
    _qwo.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    _qwo.set_defaults(func="auth_qwen_portal_logout")
    qw.set_defaults(func="auth_qwen_portal")

    # ── MiniMax Portal ────────────────────────────────────────────
    mm = auth_sub.add_parser("minimax-portal", parents=[db_parent], help="MiniMax Portal (user-code + PKCE)")
    mm_sub = mm.add_subparsers(dest="minimax_portal_command")
    _mml = mm_sub.add_parser("login", parents=[db_parent], help="Login via user-code flow")
    _mml.add_argument("--region", choices=["global", "cn"], default="global", help="API region")
    _mml.add_argument("--timeout-seconds", type=int, default=300, help="Polling timeout")
    _mml.set_defaults(func="auth_minimax_portal_login")
    _mms = mm_sub.add_parser("status", parents=[db_parent], help="Show status")
    _mms.add_argument("--json", action="store_true", help="Output JSON")
    _mms.set_defaults(func="auth_minimax_portal_status")
    _mmo = mm_sub.add_parser("logout", parents=[db_parent], help="Delete stored credentials")
    _mmo.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    _mmo.set_defaults(func="auth_minimax_portal_logout")
    mm.set_defaults(func="auth_minimax_portal")

    # ── Google Gemini CLI ─────────────────────────────────────────
    _register_pkce_provider(auth_sub, db_parent, "google-gemini-cli", "Google Gemini CLI (Cloud Code Assist)")

    # ── Google Antigravity ────────────────────────────────────────
    _register_pkce_provider(auth_sub, db_parent, "google-antigravity", "Google Antigravity (Cloud Code Assist sandbox)")


def _register_pkce_provider(
    auth_sub: argparse._SubParsersAction,  # type: ignore[type-arg]
    db_parent: argparse.ArgumentParser,
    name: str,
    help_text: str,
    extra_login_args: list[tuple[str, dict[str, Any]]] | None = None,
) -> None:
    """Helper: register a standard PKCE provider with login/status/logout."""
    slug = name.replace("-", "_")
    p = auth_sub.add_parser(name, parents=[db_parent], help=help_text)
    p_sub = p.add_subparsers(dest=f"{slug}_command")

    login = p_sub.add_parser("login", parents=[db_parent], help="Login via browser OAuth (PKCE)")
    login.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    login.add_argument("--timeout-seconds", type=int, default=120, help="Callback wait timeout")
    login.add_argument("--manual", action="store_true", help="Manual paste flow (skip callback server)")
    for flag, kwargs in extra_login_args or []:
        login.add_argument(flag, **kwargs)
    login.set_defaults(func=f"auth_{slug}_login")

    status = p_sub.add_parser("status", parents=[db_parent], help="Show status")
    status.add_argument("--json", action="store_true", help="Output JSON")
    status.set_defaults(func=f"auth_{slug}_status")

    logout = p_sub.add_parser("logout", parents=[db_parent], help="Delete stored credentials")
    logout.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    logout.set_defaults(func=f"auth_{slug}_logout")

    p.set_defaults(func=f"auth_{slug}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_auth_command(func: str, args: Any, dsn: str) -> int | None:
    """Handle an ``auth_*`` func string. Returns exit code, or None if not handled."""
    ws = getattr(args, "wait_seconds", 30)

    # ── OpenAI Codex ──
    if func in {"auth", "auth_openai_codex"}:
        return asyncio.run(_openai_codex_status(dsn, ws, as_json=False))
    if func == "auth_openai_codex_login":
        return asyncio.run(_openai_codex_login(dsn, ws, args.no_open, args.timeout_seconds))
    if func == "auth_openai_codex_status":
        return asyncio.run(_openai_codex_status(dsn, ws, as_json=bool(getattr(args, "json", False))))
    if func == "auth_openai_codex_logout":
        return asyncio.run(_openai_codex_logout(dsn, ws, getattr(args, "yes", False)))

    # ── Anthropic ──
    if func == "auth_anthropic":
        return asyncio.run(_anthropic_status(dsn, ws, as_json=False))
    if func == "auth_anthropic_setup_token":
        return asyncio.run(_anthropic_setup_token(dsn, ws, getattr(args, "token", None)))
    if func == "auth_anthropic_status":
        return asyncio.run(_anthropic_status(dsn, ws, as_json=bool(getattr(args, "json", False))))
    if func == "auth_anthropic_logout":
        return asyncio.run(_anthropic_logout(dsn, ws, getattr(args, "yes", False)))

    # ── Chutes ──
    if func == "auth_chutes":
        return asyncio.run(_generic_oauth_status(dsn, ws, "chutes", as_json=False))
    if func == "auth_chutes_login":
        return asyncio.run(_chutes_login(dsn, ws, args))
    if func == "auth_chutes_status":
        return asyncio.run(_generic_oauth_status(dsn, ws, "chutes", as_json=bool(getattr(args, "json", False))))
    if func == "auth_chutes_logout":
        return asyncio.run(_generic_logout(dsn, ws, "chutes", getattr(args, "yes", False)))

    # ── GitHub Copilot ──
    if func == "auth_github_copilot":
        return asyncio.run(_generic_oauth_status(dsn, ws, "github-copilot", as_json=False))
    if func == "auth_github_copilot_login":
        return asyncio.run(_github_copilot_login(dsn, ws, args))
    if func == "auth_github_copilot_status":
        return asyncio.run(_generic_oauth_status(dsn, ws, "github-copilot", as_json=bool(getattr(args, "json", False))))
    if func == "auth_github_copilot_logout":
        return asyncio.run(_generic_logout(dsn, ws, "github-copilot", getattr(args, "yes", False)))

    # ── Qwen Portal ──
    if func == "auth_qwen_portal":
        return asyncio.run(_generic_oauth_status(dsn, ws, "qwen-portal", as_json=False))
    if func == "auth_qwen_portal_login":
        return asyncio.run(_qwen_portal_login(dsn, ws, args))
    if func == "auth_qwen_portal_status":
        return asyncio.run(_generic_oauth_status(dsn, ws, "qwen-portal", as_json=bool(getattr(args, "json", False))))
    if func == "auth_qwen_portal_logout":
        return asyncio.run(_generic_logout(dsn, ws, "qwen-portal", getattr(args, "yes", False)))

    # ── MiniMax Portal ──
    if func == "auth_minimax_portal":
        return asyncio.run(_generic_oauth_status(dsn, ws, "minimax-portal", as_json=False))
    if func == "auth_minimax_portal_login":
        return asyncio.run(_minimax_portal_login(dsn, ws, args))
    if func == "auth_minimax_portal_status":
        return asyncio.run(_generic_oauth_status(dsn, ws, "minimax-portal", as_json=bool(getattr(args, "json", False))))
    if func == "auth_minimax_portal_logout":
        return asyncio.run(_generic_logout(dsn, ws, "minimax-portal", getattr(args, "yes", False)))

    # ── Google Gemini CLI ──
    if func == "auth_google_gemini_cli":
        return asyncio.run(_generic_oauth_status(dsn, ws, "google-gemini-cli", as_json=False))
    if func == "auth_google_gemini_cli_login":
        return asyncio.run(_google_gemini_cli_login(dsn, ws, args))
    if func == "auth_google_gemini_cli_status":
        return asyncio.run(_generic_oauth_status(dsn, ws, "google-gemini-cli", as_json=bool(getattr(args, "json", False))))
    if func == "auth_google_gemini_cli_logout":
        return asyncio.run(_generic_logout(dsn, ws, "google-gemini-cli", getattr(args, "yes", False)))

    # ── Google Antigravity ──
    if func == "auth_google_antigravity":
        return asyncio.run(_generic_oauth_status(dsn, ws, "google-antigravity", as_json=False))
    if func == "auth_google_antigravity_login":
        return asyncio.run(_google_antigravity_login(dsn, ws, args))
    if func == "auth_google_antigravity_status":
        return asyncio.run(_generic_oauth_status(dsn, ws, "google-antigravity", as_json=bool(getattr(args, "json", False))))
    if func == "auth_google_antigravity_logout":
        return asyncio.run(_generic_logout(dsn, ws, "google-antigravity", getattr(args, "yes", False)))

    return None  # not handled


# ---------------------------------------------------------------------------
# Provider modules registry (lazy import, keyed by provider slug)
# ---------------------------------------------------------------------------

def _provider_module(provider: str):  # noqa: ANN202
    """Lazy-import the auth module for a provider."""
    _map = {
        "chutes": "core.auth.chutes",
        "github-copilot": "core.auth.github_copilot",
        "qwen-portal": "core.auth.qwen_portal",
        "minimax-portal": "core.auth.minimax_portal",
        "google-gemini-cli": "core.auth.google_gemini_cli",
        "google-antigravity": "core.auth.google_antigravity",
    }
    import importlib
    return importlib.import_module(_map[provider])


_PROVIDER_LABELS = {
    "chutes": "Chutes",
    "github-copilot": "GitHub Copilot",
    "qwen-portal": "Qwen Portal",
    "minimax-portal": "MiniMax Portal",
    "google-gemini-cli": "Google Gemini CLI",
    "google-antigravity": "Google Antigravity",
}


# ---------------------------------------------------------------------------
# Generic status / logout (works for any provider with load_credentials / delete_credentials)
# ---------------------------------------------------------------------------

async def _generic_oauth_status(dsn: str, wait_seconds: int, provider: str, *, as_json: bool) -> int:
    mod = _provider_module(provider)
    creds = mod.load_credentials()

    label = _PROVIDER_LABELS.get(provider, provider)
    if not creds:
        if as_json:
            sys.stdout.write(json.dumps({"configured": False, "provider": provider}, indent=2) + "\n")
        else:
            sys.stdout.write(f"{label}: not logged in\n")
        return 0

    now = int(time.time() * 1000)
    expires_in_s = int((creds.expires_ms - now) / 1000)
    expires_at = _dt.datetime.fromtimestamp(creds.expires_ms / 1000, tz=_dt.timezone.utc).isoformat()

    payload: dict[str, Any] = {
        "configured": True,
        "provider": provider,
        "expires_at": expires_at,
        "expires_in_seconds": expires_in_s,
    }
    # Add provider-specific fields
    for field in ("email", "account_id", "base_url", "project_id", "resource_url", "region"):
        val = getattr(creds, field, None)
        if val is not None:
            payload[field] = val

    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        parts = [f"expires_in={expires_in_s}s"]
        for field in ("email", "account_id", "base_url", "project_id", "region"):
            val = getattr(creds, field, None)
            if val:
                parts.append(f"{field}={val}")
        sys.stdout.write(f"{label}: {' '.join(parts)}\n")
    return 0


async def _generic_logout(dsn: str, wait_seconds: int, provider: str, yes: bool) -> int:
    from apps.cli_theme import console
    mod = _provider_module(provider)
    label = _PROVIDER_LABELS.get(provider, provider)

    if not yes:
        try:
            answer = input(f"Delete stored {label} credentials? Type 'yes' to confirm: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted.[/dim]")
            return 1
        if answer != "yes":
            console.print("[dim]Aborted.[/dim]")
            return 1

    mod.delete_credentials()
    console.print(f"[ok]Deleted {label} credentials.[/ok]")
    return 0


# ---------------------------------------------------------------------------
# OpenAI Codex handlers (migrated from hexis_cli.py)
# ---------------------------------------------------------------------------

def _wait_for_callback_or_paste(
    *,
    port: int,
    callback_path: str,
    timeout_seconds: int,
    expected_state: str | None,
    console: Any,
    bind_host: str = "127.0.0.1",
) -> str | None:
    """Wait for the OAuth redirect while ALSO accepting a pasted redirect URL.

    On WSL (and some VPN/firewall setups) the browser finishes the sign-in
    but can never reach the local callback server — the user is left staring
    at "This site can't be reached" with the answer sitting in the address
    bar. Reading stdin concurrently means pasting that URL works immediately
    instead of only after the callback timeout expires.

    Returns the authorization code, or None on timeout. KeyboardInterrupt
    propagates to the caller. Falls back to a plain callback wait when stdin
    isn't an interactive POSIX tty.
    """
    import os
    import select
    import threading

    from core.auth.callback_server import run_callback_server
    from core.auth.openai_codex import parse_authorization_input

    interactive = (
        os.name == "posix"
        and sys.stdin is not None
        and sys.stdin.isatty()
        and hasattr(select, "select")
    )
    if not interactive:
        result = run_callback_server(
            port=port,
            callback_path=callback_path,
            timeout_seconds=timeout_seconds,
            expected_state=expected_state,
            bind_host=bind_host,
        )
        return (result or {}).get("code")

    holder: dict[str, str] = {}
    done = threading.Event()
    cancel = threading.Event()

    def _serve() -> None:
        result = run_callback_server(
            port=port,
            callback_path=callback_path,
            timeout_seconds=timeout_seconds,
            expected_state=expected_state,
            cancel_event=cancel,
            bind_host=bind_host,
        )
        if result:
            holder.update(result)
        done.set()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + max(5, timeout_seconds)
    try:
        while time.monotonic() < deadline and not done.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not ready:
                continue
            line = sys.stdin.readline().strip()
            if not line:
                continue
            code, pasted_state = parse_authorization_input(line)
            if pasted_state and expected_state and pasted_state != expected_state:
                console.print(
                    "[warn]State mismatch — paste the redirect URL from THIS "
                    "login attempt.[/warn]"
                )
                continue
            if code:
                return code
            console.print(
                "[warn]No authorization code in that paste — copy the full "
                "redirect URL from the browser's address bar.[/warn]"
            )
    finally:
        cancel.set()
        done.wait(timeout=2.0)
    return holder.get("code")


async def _openai_codex_login(
    dsn: str,
    wait_seconds: int,
    no_open: bool,
    timeout_seconds: int,
    allow_manual_fallback: bool = True,
) -> int:
    import socket

    from apps.cli_theme import console
    from core.browser import is_wsl, open_url
    from core.auth.openai_codex import (
        OPENAI_CODEX_REDIRECT_URI,
        build_authorize_url,
        create_state,
        ensure_fresh_openai_codex_credentials,
        exchange_authorization_code,
        generate_pkce,
        parse_authorization_input,
        save_openai_codex_credentials,
    )

    # Pre-check whether port 1455 is available before starting the flow.
    port_available = True
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 1455))
    except OSError:
        port_available = False

    verifier, challenge = generate_pkce()
    state = create_state()
    auth_url = build_authorize_url(challenge=challenge, state=state)

    console.print("\n[bold]OpenAI Codex OAuth[/bold]")

    if not port_available:
        # Identify the blocking process for the error message.
        blocker = ""
        try:
            import subprocess as _sp
            out = _sp.check_output(
                ["lsof", "-i", ":1455", "-sTCP:LISTEN", "-P", "-n", "-Fp"],
                text=True, timeout=5,
            )
            pids = [line[1:] for line in out.splitlines() if line.startswith("p")]
            if pids:
                names = _sp.check_output(
                    ["ps", "-p", ",".join(pids), "-o", "pid=,comm="],
                    text=True, timeout=5,
                ).strip()
                blocker = f"\n  Blocking process:\n    {names}\n"
        except Exception:
            pass

        console.print(
            "[fail]Port 1455 is required for OpenAI Codex OAuth but is already in use.[/fail]\n"
            f"{blocker}\n"
            "  Please free the port and try again:\n"
            "    [bold]lsof -ti :1455 | xargs kill[/bold]\n"
        )
        return 1
    else:
        console.print("1. A browser window should open. Sign in to ChatGPT and approve.")
        console.print(
            "2. If the final page shows [bold]This site can't be reached[/bold] "
            "(common on WSL), the sign-in still worked — copy the full "
            "localhost:1455/... URL from the address bar and paste it here, "
            "then press Enter.\n"
        )
        console.print(f"[dim]{auth_url}[/dim]\n")

        if not no_open and not open_url(auth_url):
            console.print(
                "[warn]Could not open a browser automatically — open the URL "
                "above yourself.[/warn]"
            )

        console.print("[dim]Waiting for the browser callback (or a pasted URL)...[/dim]")
        try:
            code = _wait_for_callback_or_paste(
                port=1455,
                callback_path="/auth/callback",
                timeout_seconds=timeout_seconds,
                expected_state=state,
                console=console,
                # The WSL2 localhost relay picks up wildcard binds far more
                # reliably than 127.0.0.1-only ones; the server is
                # state-validated and lives only for the timeout window.
                bind_host="0.0.0.0" if is_wsl() else "127.0.0.1",
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Aborted.[/dim]")
            return 1

        if not code and not allow_manual_fallback:
            _print_err(
                "Authorization callback not received. Retry login in TUI or run "
                "`hexis auth openai-codex login` in a terminal."
            )
            return 1

        if not code:
            console.print(
                "[warn]No callback arrived before the timeout. If the browser "
                "finished on an unreachable localhost page, authentication "
                "succeeded anyway — only the redirect could not reach this "
                "terminal.[/warn]"
            )
            try:
                pasted = input("Paste the authorization code (or full redirect URL): ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Aborted.[/dim]")
                return 1
            parsed_code, parsed_state = parse_authorization_input(pasted)
            if parsed_state and parsed_state != state:
                _print_err("State mismatch. Ensure you pasted the redirect URL from this login attempt.")
                return 1
            code = parsed_code

    if not code:
        _print_err("Missing authorization code.")
        return 1

    console.print("[accent]Exchanging code for tokens...[/accent]")
    creds = await exchange_authorization_code(code=code, verifier=verifier)

    save_openai_codex_credentials(creds)
    await ensure_fresh_openai_codex_credentials(skew_seconds=0)

    expires_sec = max(0, int((creds.expires_ms - int(time.time() * 1000)) / 1000))
    console.print(f"[ok]Logged in.[/ok] account_id={creds.account_id} expires_in~{expires_sec}s")
    return 0


async def _openai_codex_status(dsn: str, wait_seconds: int, *, as_json: bool) -> int:
    from core.auth.openai_codex import load_openai_codex_credentials

    creds = load_openai_codex_credentials()

    if not creds:
        if as_json:
            sys.stdout.write(json.dumps({"configured": False}, indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write("not logged in\n")
        return 0

    now_ms = int(time.time() * 1000)
    expires_in_s = int((creds.expires_ms - now_ms) / 1000)
    expires_at = _dt.datetime.fromtimestamp(creds.expires_ms / 1000, tz=_dt.timezone.utc).isoformat()

    payload = {
        "configured": True,
        "account_id": creds.account_id,
        "expires_at": expires_at,
        "expires_in_seconds": expires_in_s,
    }
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"account_id={creds.account_id} expires_in={expires_in_s}s\n")
    return 0


async def _openai_codex_logout(dsn: str, wait_seconds: int, yes: bool) -> int:
    from apps.cli_theme import console
    from core.auth.openai_codex import delete_openai_codex_credentials

    if not yes:
        try:
            answer = input("Delete stored OpenAI Codex OAuth credentials? Type 'yes' to confirm: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted.[/dim]")
            return 1
        if answer != "yes":
            console.print("[dim]Aborted.[/dim]")
            return 1

    delete_openai_codex_credentials()
    console.print("[ok]Deleted OAuth credentials.[/ok]")
    return 0


# ---------------------------------------------------------------------------
# Anthropic setup-token handlers
# ---------------------------------------------------------------------------

async def _anthropic_setup_token(dsn: str, wait_seconds: int, token: str | None) -> int:
    import getpass

    from apps.cli_theme import console
    from core.auth.anthropic_setup_token import (
        AnthropicSetupTokenCredentials,
        save_credentials,
        validate_setup_token,
    )

    if not token:
        try:
            token = getpass.getpass("Paste your Anthropic setup token: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted.[/dim]")
            return 1

    error = validate_setup_token(token)
    if error:
        _print_err(f"Invalid setup token: {error}")
        return 1

    save_credentials(AnthropicSetupTokenCredentials(token=token))
    console.print("[ok]Setup token saved.[/ok]")
    return 0


async def _anthropic_status(dsn: str, wait_seconds: int, *, as_json: bool) -> int:
    from core.auth.anthropic_setup_token import (
        load_credentials as load_setup,
        read_claude_code_credentials,
    )

    cc_creds = read_claude_code_credentials()
    setup_creds = load_setup()

    sources: list[dict[str, Any]] = []

    if cc_creds:
        expires_at = cc_creds.get("expiresAt", 0)
        now = int(time.time() * 1000)
        expires_in_s = int((expires_at - now) / 1000) if expires_at else None
        sources.append({
            # Detected, but Hexis does NOT use Claude Code's login (it manages
            # its own store). Don't report it as "configured".
            "type": "claude_code",
            "configured": False,
            "used_by_runtime": False,
            "note": "detected but not used — run `hexis auth anthropic setup-token`",
            "expires_in_seconds": expires_in_s,
            "source": cc_creds.get("source", "unknown"),
            "token_preview": cc_creds.get("accessToken", "")[-6:],
        })

    if setup_creds:
        redacted = setup_creds.token[:20] + "..." if len(setup_creds.token) > 20 else "***"
        sources.append({
            "type": "setup_token",
            "configured": True,
            "token_prefix": redacted,
        })

    if not sources:
        if as_json:
            sys.stdout.write(json.dumps({"configured": False}, indent=2) + "\n")
        else:
            sys.stdout.write("Anthropic: not configured\n")
        return 0

    # "configured" for Hexis means a source the runtime actually uses.
    usable = bool(setup_creds)
    if as_json:
        sys.stdout.write(json.dumps({"configured": usable, "sources": sources}, indent=2, sort_keys=True) + "\n")
    else:
        if not usable:
            sys.stdout.write("Anthropic: no setup token stored. Run `hexis auth anthropic setup-token`.\n")
        for src in sources:
            t = src["type"]
            if t == "claude_code":
                exp = f" expires_in={src['expires_in_seconds']}s" if src.get("expires_in_seconds") is not None else ""
                sys.stdout.write(f"  Claude Code: detected but NOT used by Hexis ({src['source']}{exp})\n")
            elif t == "setup_token":
                sys.stdout.write(f"  Setup token: {src['token_prefix']}\n")
    return 0


async def _anthropic_logout(dsn: str, wait_seconds: int, yes: bool) -> int:
    from apps.cli_theme import console
    from core.auth.anthropic_setup_token import delete_credentials as delete_setup
    from core.auth.store import delete_auth

    if not yes:
        try:
            answer = input("Delete stored Anthropic credentials (setup token)? Type 'yes' to confirm: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted.[/dim]")
            return 1
        if answer != "yes":
            console.print("[dim]Aborted.[/dim]")
            return 1

    delete_setup()
    # Also clear any credential left behind by the removed OAuth login flow.
    delete_auth("oauth.anthropic")
    console.print("[ok]Deleted Anthropic credentials.[/ok]")
    return 0


# ---------------------------------------------------------------------------
# Chutes login
# ---------------------------------------------------------------------------

async def _chutes_login(dsn: str, wait_seconds: int, args: Any) -> int:
    import os

    from apps.cli_theme import console
    from core.auth import create_state, generate_pkce
    from core.auth.chutes import exchange_code, save_credentials
    from core.browser import is_wsl, open_url

    client_id = getattr(args, "client_id", None) or os.getenv("CHUTES_CLIENT_ID", "")
    if not client_id:
        _print_err(
            "Chutes OAuth needs a client id. Create one at https://chutes.ai (developer "
            "settings), then set CHUTES_CLIENT_ID (or pass --client-id). Alternatively, "
            "use Chutes as an OpenAI-compatible endpoint with an API key."
        )
        return 1

    redirect_uri = os.getenv("CHUTES_REDIRECT_URI", "http://localhost:11435/auth/callback")
    verifier, challenge = generate_pkce()
    state = create_state()

    from core.auth.chutes import build_authorize_url
    auth_url = build_authorize_url(
        challenge=challenge, state=state, client_id=client_id, redirect_uri=redirect_uri,
    )

    console.print("\n[bold]Chutes OAuth[/bold]")
    console.print(f"[dim]{auth_url}[/dim]\n")

    no_open = getattr(args, "no_open", False)
    manual = getattr(args, "manual", False)
    timeout_seconds = getattr(args, "timeout_seconds", 120)
    non_interactive = bool(getattr(args, "non_interactive", False))

    code: str | None = None
    if not manual:
        from urllib.parse import urlparse
        parsed_uri = urlparse(redirect_uri)
        port = parsed_uri.port or 80
        path = parsed_uri.path or "/auth/callback"

        if not no_open and not open_url(auth_url):
            console.print(
                "[warn]Could not open a browser automatically — open the URL "
                "above yourself.[/warn]"
            )

        try:
            code = _wait_for_callback_or_paste(
                port=port,
                callback_path=path,
                timeout_seconds=timeout_seconds,
                expected_state=state,
                console=console,
                bind_host="0.0.0.0" if is_wsl() else "127.0.0.1",
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Aborted.[/dim]")
            return 1

    if not code and non_interactive:
        _print_err(
            "Authorization callback not received. Retry and complete browser OAuth, or run "
            "`hexis auth chutes login --manual` in a terminal."
        )
        return 1

    if not code:
        try:
            pasted = input("Paste the authorization code (or full redirect URL): ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted.[/dim]")
            return 1
        from core.auth.openai_codex import parse_authorization_input
        parsed_code, parsed_state = parse_authorization_input(pasted)
        if parsed_state and parsed_state != state:
            _print_err("State mismatch.")
            return 1
        code = parsed_code

    if not code:
        _print_err("Missing authorization code.")
        return 1

    console.print("[accent]Exchanging code for tokens...[/accent]")
    creds = await exchange_code(
        code=code, verifier=verifier, client_id=client_id, redirect_uri=redirect_uri,
        client_secret=os.getenv("CHUTES_CLIENT_SECRET"),
    )

    save_credentials(creds)
    console.print(f"[ok]Logged in.[/ok] email={creds.email or 'unknown'}")
    return 0


# ---------------------------------------------------------------------------
# GitHub Copilot login
# ---------------------------------------------------------------------------

async def _github_copilot_login(dsn: str, wait_seconds: int, args: Any) -> int:
    from core.browser import open_url

    from apps.cli_theme import console
    from core.auth.github_copilot import (
        exchange_github_for_copilot,
        poll_for_github_token,
        save_credentials,
        start_device_flow,
    )

    enterprise_domain = getattr(args, "enterprise_domain", None)
    domain = enterprise_domain or "github.com"

    console.print(f"\n[bold]GitHub Copilot (device code flow)[/bold]  domain={domain}")
    device = await start_device_flow(domain)

    console.print(f"\n1. Open: [link]{device.verification_uri}[/link]")
    console.print(f"2. Enter code: [bold]{device.user_code}[/bold]\n")

    try:
        open_url(device.verification_uri)
    except Exception:
        pass

    console.print("[accent]Waiting for authorization...[/accent]")
    github_token = await poll_for_github_token(
        domain, device.device_code, device.interval, device.expires_in,
    )

    console.print("[accent]Exchanging for Copilot token...[/accent]")
    creds = await exchange_github_for_copilot(github_token, enterprise_domain)

    save_credentials(creds)
    console.print(f"[ok]Logged in.[/ok] base_url={creds.base_url}")
    return 0


# ---------------------------------------------------------------------------
# Qwen Portal login
# ---------------------------------------------------------------------------

async def _qwen_portal_login(dsn: str, wait_seconds: int, args: Any) -> int:
    from core.browser import open_url

    from apps.cli_theme import console
    from core.auth.qwen_portal import poll_for_token, save_credentials, start_device_flow

    console.print("\n[bold]Qwen Portal (device code flow)[/bold]")
    device, verifier = await start_device_flow()

    uri = device.verification_uri_complete or device.verification_uri
    console.print(f"\n1. Open: [link]{uri}[/link]")
    if not device.verification_uri_complete:
        console.print(f"2. Enter code: [bold]{device.user_code}[/bold]")
    console.print()

    try:
        open_url(uri)
    except Exception:
        pass

    console.print("[accent]Waiting for authorization...[/accent]")
    creds = await poll_for_token(device.device_code, verifier, device.interval, device.expires_in)

    save_credentials(creds)
    console.print("[ok]Logged in.[/ok]")
    return 0


# ---------------------------------------------------------------------------
# MiniMax Portal login
# ---------------------------------------------------------------------------

async def _minimax_portal_login(dsn: str, wait_seconds: int, args: Any) -> int:
    from core.browser import open_url

    from apps.cli_theme import console
    from core.auth.minimax_portal import poll_for_token, save_credentials, start_user_code_flow

    region = getattr(args, "region", "global")
    console.print(f"\n[bold]MiniMax Portal (user-code flow)[/bold]  region={region}")
    user_code_resp, verifier = await start_user_code_flow(region)

    console.print(f"\n1. Open: [link]{user_code_resp.verification_uri}[/link]")
    console.print(f"2. Enter code: [bold]{user_code_resp.user_code}[/bold]\n")

    try:
        open_url(user_code_resp.verification_uri)
    except Exception:
        pass

    console.print("[accent]Waiting for authorization...[/accent]")
    creds = await poll_for_token(
        user_code_resp.user_code, verifier, user_code_resp.interval,
        user_code_resp.expires_in, region,
    )

    save_credentials(creds)
    console.print("[ok]Logged in.[/ok]")
    return 0


# ---------------------------------------------------------------------------
# Google Gemini CLI login
# ---------------------------------------------------------------------------

async def _google_gemini_cli_login(dsn: str, wait_seconds: int, args: Any) -> int:
    from core.browser import open_url

    from apps.cli_theme import console
    from core.auth import create_state, generate_pkce
    from core.auth.callback_server import run_callback_server
    from core.auth.google_gemini_cli import (
        GEMINI_CLI_REDIRECT_URI,
        build_authorize_url,
        complete_login,
        save_credentials,
    )

    verifier, challenge = generate_pkce()
    state = create_state()
    auth_url = build_authorize_url(challenge=challenge, state=state)

    console.print("\n[bold]Google Gemini CLI OAuth[/bold]")
    console.print(f"[dim]{auth_url}[/dim]\n")

    no_open = getattr(args, "no_open", False)
    manual = getattr(args, "manual", False)
    timeout_seconds = getattr(args, "timeout_seconds", 120)
    non_interactive = bool(getattr(args, "non_interactive", False))

    code: str | None = None
    if not manual:
        if not no_open:
            try:
                open_url(auth_url)
            except Exception:
                pass
        result = run_callback_server(
            port=8085, callback_path="/oauth2callback",
            timeout_seconds=timeout_seconds, expected_state=state,
        )
        code = result.get("code") if result else None

    if not code and non_interactive:
        _print_err(
            "Authorization callback not received. Retry and complete browser OAuth, or run "
            "`hexis auth google-gemini-cli login --manual` in a terminal."
        )
        return 1

    if not code:
        try:
            pasted = input("Paste the full redirect URL: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted.[/dim]")
            return 1
        from core.auth.openai_codex import parse_authorization_input
        parsed_code, parsed_state = parse_authorization_input(pasted)
        if parsed_state and parsed_state != state:
            _print_err("State mismatch.")
            return 1
        code = parsed_code

    if not code:
        _print_err("Missing authorization code.")
        return 1

    console.print("[accent]Exchanging code and discovering project...[/accent]")
    creds = await complete_login(code, verifier)

    save_credentials(creds)
    console.print(f"[ok]Logged in.[/ok] project={creds.project_id} email={creds.email or 'unknown'}")
    return 0


# ---------------------------------------------------------------------------
# Google Antigravity login
# ---------------------------------------------------------------------------

async def _google_antigravity_login(dsn: str, wait_seconds: int, args: Any) -> int:
    from core.browser import open_url

    from apps.cli_theme import console
    from core.auth import create_state, generate_pkce
    from core.auth.callback_server import run_callback_server
    from core.auth.google_antigravity import (
        ANTIGRAVITY_REDIRECT_URI,
        build_authorize_url,
        complete_login,
        save_credentials,
    )

    verifier, challenge = generate_pkce()
    state = create_state()
    auth_url = build_authorize_url(challenge=challenge, state=state)

    console.print("\n[bold]Google Antigravity OAuth[/bold]")
    console.print(f"[dim]{auth_url}[/dim]\n")

    no_open = getattr(args, "no_open", False)
    manual = getattr(args, "manual", False)
    timeout_seconds = getattr(args, "timeout_seconds", 120)
    non_interactive = bool(getattr(args, "non_interactive", False))

    code: str | None = None
    if not manual:
        if not no_open:
            try:
                open_url(auth_url)
            except Exception:
                pass
        result = run_callback_server(
            port=51121, callback_path="/oauth-callback",
            timeout_seconds=timeout_seconds, expected_state=state,
        )
        code = result.get("code") if result else None

    if not code and non_interactive:
        _print_err(
            "Authorization callback not received. Retry and complete browser OAuth, or run "
            "`hexis auth google-antigravity login --manual` in a terminal."
        )
        return 1

    if not code:
        try:
            pasted = input("Paste the full redirect URL: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted.[/dim]")
            return 1
        from core.auth.openai_codex import parse_authorization_input
        parsed_code, parsed_state = parse_authorization_input(pasted)
        if parsed_state and parsed_state != state:
            _print_err("State mismatch.")
            return 1
        code = parsed_code

    if not code:
        _print_err("Missing authorization code.")
        return 1

    console.print("[accent]Exchanging code and discovering project...[/accent]")
    creds = await complete_login(code, verifier)

    save_credentials(creds)
    console.print(f"[ok]Logged in.[/ok] project={creds.project_id} email={creds.email or 'unknown'}")
    return 0
