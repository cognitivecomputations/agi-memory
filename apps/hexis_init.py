"""Hexis init wizard — 3-tier flow: Express, Character, Custom.

Flow: [LLM Config] → [Choose Path] → [Express | Character | Custom] → [Consent] → [Done]

Non-interactive mode: pass provider/model flags with either --api-key or
--api-key-env to skip the wizard and configure everything from CLI flags.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from getpass import getpass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core import agent_api
from core.init_api import (
    get_card_summary,
    load_character_card_document,
    load_character_cards,
)
from core.llm import normalize_llm_config

from apps.cli_theme import console, err_console, heading, make_panel, make_table


# ---------------------------------------------------------------------------
# Non-interactive helpers
# ---------------------------------------------------------------------------

# Default models are NOT hardcoded here — they derive from the live catalog
# (apps/tui/model_catalog.py: models.dev + recommended_default), the single
# source of truth shared with the interactive wizard (Bar #1).

_PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
    "openai-codex": "",
    "grok": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "chutes": "",
    "github-copilot": "",
    "qwen-portal": "",
    "minimax-portal": "",
    "google-gemini-cli": "",
    "google-antigravity": "",
}

# Providers that use OAuth / device-code / token auth (no API key needed).
_OAUTH_PROVIDERS: set[str] = {
    "openai-codex", "chutes", "github-copilot", "qwen-portal",
    "minimax-portal", "google-gemini-cli", "google-antigravity",
}


def detect_provider(api_key: str) -> str:
    """Auto-detect LLM provider from API key prefix."""
    if api_key.startswith("sk-ant-"):
        return "anthropic"
    if api_key.startswith("sk-"):
        return "openai"
    if api_key.startswith("gsk_"):
        return "grok"
    if api_key.startswith("AIza"):
        return "gemini"
    raise ValueError(
        f"Cannot detect provider from key prefix '{api_key[:6]}...'. Use --provider."
    )


def _normalize_provider_name(provider: str | None) -> str:
    raw = (provider or "").strip().lower()
    _ALIASES = {
        "openai_codex": "openai-codex",
        "github_copilot": "github-copilot",
        "qwen_portal": "qwen-portal",
        "minimax_portal": "minimax-portal",
        "google_gemini_cli": "google-gemini-cli",
        "google_antigravity": "google-antigravity",
    }
    return _ALIASES.get(raw, raw)


def _resolve_noninteractive_provider(args: argparse.Namespace) -> str:
    """Resolve a provider only from values the user explicitly selected."""
    provider = _normalize_provider_name(args.provider)
    if provider:
        return provider
    if args.endpoint:
        return "openai_compatible"
    if args.api_key:
        return detect_provider(args.api_key)
    if args.api_key_env:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {args.api_key_env} is not set. "
                "Add it to .env, then re-run init."
            )
        return detect_provider(api_key)
    return "openai-codex"


def _resolve_noninteractive_endpoint(
    args: argparse.Namespace,
    provider: str,
) -> tuple[str, str]:
    """Return the endpoint and its visible source label."""
    endpoint = (args.endpoint or "").strip()
    source = "--endpoint" if endpoint else ""
    if not endpoint and provider == "openai_compatible":
        endpoint = os.getenv("OPENAI_BASE_URL", "").strip()
        source = "OPENAI_BASE_URL" if endpoint else ""
    if provider == "openai_compatible" and not endpoint:
        raise ValueError(
            "An OpenAI-compatible endpoint is required. Pass --endpoint URL "
            "or set OPENAI_BASE_URL in .env."
        )
    return endpoint, source


def _resolve_noninteractive_api_key_env(
    args: argparse.Namespace,
    provider: str,
) -> str:
    """Choose a credential env var without silently consuming ambient keys."""
    if provider in _OAUTH_PROVIDERS:
        return ""

    api_key_env = (args.api_key_env or _PROVIDER_ENV_VARS.get(provider, "")).strip()
    if args.api_key:
        if not api_key_env:
            raise ValueError(
                f"No default API key variable is known for provider '{provider}'. "
                "Use --api-key-env NAME instead."
            )
        return api_key_env

    if not args.api_key_env:
        suggestion = (
            f" --api-key-env {api_key_env}" if api_key_env else " --api-key-env NAME"
        )
        raise ValueError(
            f"API key configuration is required for provider '{provider}'. "
            f"Set the key in .env and pass{suggestion}, or pass --api-key."
        )
    if not os.getenv(api_key_env):
        raise ValueError(
            f"Environment variable {api_key_env} is not set. "
            "Add it to .env, then re-run init."
        )
    return api_key_env


async def _ensure_oauth_login(
    provider: str,
    dsn: str,
    conn: Any,
    *,
    wait_seconds: int,
    allow_manual_fallback: bool = True,
    non_interactive: bool = False,
) -> None:
    """
    Ensure OAuth/device-code/token credentials exist for the given provider.

    Called by `hexis init` so the Quick Start can use OAuth providers without
    a separate `hexis auth <provider> login` step.
    """
    # Map provider -> (module path, load function name)
    _LOADERS: dict[str, tuple[str, str]] = {
        "openai-codex":       ("core.auth.openai_codex",      "load_openai_codex_credentials"),
        "chutes":             ("core.auth.chutes",             "load_credentials"),
        "github-copilot":     ("core.auth.github_copilot",     "load_credentials"),
        "qwen-portal":        ("core.auth.qwen_portal",        "load_credentials"),
        "minimax-portal":     ("core.auth.minimax_portal",     "load_credentials"),
        "google-gemini-cli":  ("core.auth.google_gemini_cli",  "load_credentials"),
        "google-antigravity": ("core.auth.google_antigravity", "load_credentials"),
    }

    entry = _LOADERS.get(provider)
    if not entry:
        return

    import importlib
    mod = importlib.import_module(entry[0])
    load_fn = getattr(mod, entry[1])
    existing = load_fn()
    if existing:
        return

    # Non-interactive (CI/scripts) must not open a browser or block on a paste.
    if non_interactive:
        raise RuntimeError(
            f"Not logged in to {provider}. Run `hexis auth {provider} login` first, "
            "then re-run init."
        )

    display = provider.replace("-", " ").title()
    console.print(f"[muted]Starting {display} login...[/muted]")

    # Call the async login handler directly (we're already in an event loop).
    from apps.cli_auth import (
        _openai_codex_login, _chutes_login, _github_copilot_login,
        _qwen_portal_login, _minimax_portal_login,
        _google_gemini_cli_login, _google_antigravity_login,
    )

    ns = argparse.Namespace(
        no_open=False,
        timeout_seconds=180,
        manual=False,
        non_interactive=not allow_manual_fallback,
    )

    if provider == "openai-codex":
        rc = await _openai_codex_login(
            dsn,
            wait_seconds,
            no_open=False,
            timeout_seconds=180,
            allow_manual_fallback=allow_manual_fallback,
        )
    elif provider == "chutes":
        ns.client_id = None
        rc = await _chutes_login(dsn, wait_seconds, ns)
    elif provider == "github-copilot":
        ns.enterprise_domain = "github.com"
        rc = await _github_copilot_login(dsn, wait_seconds, ns)
    elif provider == "qwen-portal":
        rc = await _qwen_portal_login(dsn, wait_seconds, ns)
    elif provider == "minimax-portal":
        ns.region = "global"
        rc = await _minimax_portal_login(dsn, wait_seconds, ns)
    elif provider == "google-gemini-cli":
        rc = await _google_gemini_cli_login(dsn, wait_seconds, ns)
    elif provider == "google-antigravity":
        rc = await _google_antigravity_login(dsn, wait_seconds, ns)
    else:
        return

    if rc != 0:
        raise RuntimeError(f"{display} login failed")


async def _load_llm_config_for_consent(
    conn: Any,
    *,
    dsn: str,
    wait_seconds: int,
    provider: str,
    model: str,
    non_interactive: bool = False,
) -> dict[str, Any]:
    if provider in _OAUTH_PROVIDERS:
        await _ensure_oauth_login(provider, dsn, conn, wait_seconds=wait_seconds,
                                  non_interactive=non_interactive)

    from core.llm_config import load_llm_config

    # Consent is a chat-style flow; use llm.chat config.
    return await load_llm_config(
        conn,
        "llm.chat",
        default_provider=provider,
        default_model=model,
    )


def _write_env_var(env_path: Path, key: str, value: str) -> None:
    """Upsert a KEY=value line in a .env file."""
    lines: list[str] = []
    replaced = False
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.lstrip()
            if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                lines.append(f"{key}={value}")
                replaced = True
            else:
                lines.append(line)
    if not replaced:
        lines.append(f"{key}={value}")
    # Ensure trailing newline
    env_path.write_text("\n".join(lines) + "\n")


# Services the agent cannot live without, used only if compose cannot be asked.
_FALLBACK_STACK_SERVICES = ("db", "heartbeat_worker", "maintenance_worker")


def _default_stack_services(
    compose_cmd: list[str], compose_file: Path, stack_root: Path, env_file: Path | None
) -> list[str]:
    """The services a bare `up -d` starts — compose's default profile.

    Read from the compose file rather than hardcoded: the always-on set is
    whatever is declared without a `profiles:` key, and that list grows.
    """
    from apps.hexis_cli import _run_compose_capture

    rc, out = _run_compose_capture(
        compose_cmd, compose_file, stack_root, ["config", "--services"], env_file
    )
    services: list[str] = []
    if rc == 0:
        for line in out.splitlines():
            name = line.strip()
            # compose merges warnings into this stream; service names are bare
            # single tokens, so anything else is noise.
            if name and " " not in name and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
                services.append(name)
    resolved = services or list(_FALLBACK_STACK_SERVICES)
    try:
        from apps.hexis_cli import _host_managed_compose_workers

        host_managed = _host_managed_compose_workers()
    except Exception:
        host_managed = set()
    return [name for name in resolved if name not in host_managed]


def _dsn_is_local(dsn: str) -> bool:
    """Does this DSN point at a database this machine hosts?

    Starting the local stack for someone pointed at a Postgres they run
    elsewhere would be a surprise, so the auto-start stays local-only.
    """
    from urllib.parse import urlparse

    try:
        host = (urlparse(dsn).hostname or "").lower()
    except ValueError:
        return False
    return host in ("", "localhost", "127.0.0.1", "::1", "host.docker.internal", "db")


def _ensure_stack_running(args: argparse.Namespace) -> Path:
    """Start the Docker stack if any of it is missing. Returns stack_root.

    "The database is up" is not the same as "the stack is up": the heartbeat
    and maintenance workers are what make the agent autonomous, and an init
    that returned early on a database-only stack left an agent that never woke
    up on its own. Every service in the default profile has to be running.
    """
    from apps.hexis_cli import (
        _ensure_installed_host_services_running,
        _find_compose_file,
        _host_managed_compose_workers,
        _stack_root_from_compose,
        ensure_compose,
        ensure_docker,
        resolve_env_file,
        run_compose,
        _run_compose_capture,
    )

    compose_file, is_source = _find_compose_file()
    if compose_file is None:
        err_console.print("[fail]Cannot find docker-compose.yml. Is Hexis installed?[/fail]")
        raise SystemExit(1)

    stack_root = _stack_root_from_compose(compose_file)
    docker_bin = ensure_docker()
    compose_cmd = ensure_compose(docker_bin)
    env_file = resolve_env_file(stack_root)

    expected = _default_stack_services(compose_cmd, compose_file, stack_root, env_file)
    host_managed_workers = _host_managed_compose_workers()
    rc, out = _run_compose_capture(
        compose_cmd, compose_file, stack_root,
        ["ps", "--services", "--filter", "status=running"], env_file,
    )
    running = set(out.split()) if rc == 0 else set()
    missing = [name for name in expected if name not in running]
    if not missing:
        if host_managed_workers:
            workers_ok, workers_error = _ensure_installed_host_services_running()
            if not workers_ok:
                err_console.print(
                    "[fail]Docker is ready, but the installed host workers did not "
                    f"start: {workers_error}[/fail]\nRun `hexis service logs`, fix the "
                    "reported cause, then run `hexis init` again."
                )
                raise SystemExit(1)
        console.print("[ok]\u2714[/ok] Docker stack already running")
        return stack_root

    if running:
        console.print(f"[muted]Starting {', '.join(missing)}...[/muted]")
    else:
        console.print("[muted]Starting Docker stack...[/muted]")
        if not is_source:
            # pip install path: pull images first
            pull_args = ["pull", *expected] if host_managed_workers else ["pull"]
            run_compose(compose_cmd, compose_file, stack_root, pull_args, env_file)
    up_args = ["up", "-d", *expected] if host_managed_workers else ["up", "-d"]
    rc = run_compose(compose_cmd, compose_file, stack_root, up_args, env_file)
    if rc != 0:
        err_console.print("[fail]Failed to start Docker stack.[/fail]")
        raise SystemExit(1)

    if host_managed_workers:
        workers_ok, workers_error = _ensure_installed_host_services_running()
        if not workers_ok:
            err_console.print(
                "[fail]Docker started, but the installed host workers did not: "
                f"{workers_error}[/fail]\nRun `hexis service logs`, fix the reported "
                "cause, then run `hexis init` again."
            )
            raise SystemExit(1)

    console.print("[ok]\u2714[/ok] Docker stack started")
    return stack_root


_DEFAULT_EMBEDDING_MODEL = "embeddinggemma:300m-qat-q4_0"
_DEFAULT_EMBEDDING_URL = "http://host.docker.internal:42666/api/embed"

# Substrings of the errors get_embedding() raises when its HTTP endpoint is
# down (db/03_functions_helpers.sql) — the one init failure the user can fix
# in place by starting the local embedding sidecar.
_EMBEDDING_ERROR_MARKERS = (
    "Embedding service not available",
    "Failed to get embeddings",
)


def _is_embedding_unavailable(exc: BaseException) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _EMBEDDING_ERROR_MARKERS)


async def _embedding_service_info(conn: Any) -> tuple[str, str]:
    """The embedding URL + model the DB is actually configured with."""
    url = model = None
    try:
        url = await conn.fetchval(
            "SELECT current_setting('app.embedding_service_url', true)")
        model = await conn.fetchval(
            "SELECT current_setting('app.embedding_model_id', true)")
    except Exception:
        pass
    return url or _DEFAULT_EMBEDDING_URL, model or _DEFAULT_EMBEDDING_MODEL


async def _run_embedding_step(conn: Any, step: Any, *, interactive: bool = True) -> Any:
    """Run a DB write that stores memories (and therefore needs embeddings).

    When the embedding service is unreachable, explain the cause and the exact
    fix, then offer to retry in place — the wizard never dead-ends on a
    stopped sidecar. Only the DB write reruns; answers already given are kept.
    """
    while True:
        try:
            return await step()
        except Exception as exc:
            if not _is_embedding_unavailable(exc):
                raise
            url, _model = await _embedding_service_info(conn)
            err_console.print(make_panel(
                "[fail]The embedding service isn't reachable, so memories can't be "
                "stored yet.[/fail]\n\n"
                f"[key]Configured URL:[/key] {url}\n\n"
                "By default Hexis uses the published local [bold]embeddinggemma[/bold] sidecar. To fix:\n"
                "  1. Start it: [bold]embeddinggemma[/bold]\n"
                "  2. Or run [bold]hexis up[/bold] to start the stack and sidecar together.\n\n"
                "Using a different embedding server? Set [bold]EMBEDDING_SERVICE_URL[/bold] "
                "and restart the stack with `hexis up`.",
                title="Embeddings unavailable",
            ))
            if not interactive or not _prompt_yes_no("Try again?", default=True):
                raise RuntimeError(
                    "the embedding service is unreachable — start the local embedding "
                    "service (or your configured embedding server), then run `hexis init` again"
                ) from exc


def _ensure_embedding_model() -> None:
    """Start the local embeddinggemma.c sidecar if needed."""
    try:
        from apps.hexis_cli import _start_local_embedding_service

        _start_local_embedding_service()
    except Exception as exc:
        console.print(
            f"[warn]\u26a0[/warn] Couldn't start local embedding service: {exc}\n"
            "  Try running: embeddinggemma"
        )


async def _run_init_noninteractive(args: argparse.Namespace) -> int:
    """Non-interactive init: configure from CLI flags, start stack, apply config."""
    # 1. Detect provider
    provider = _resolve_noninteractive_provider(args)
    persist_provider = provider

    # The Anthropic browser-OAuth (Claude Pro/Max) login was removed — the
    # endpoint no longer accepts it. Point straight at the working paths.
    if provider == "anthropic-oauth":
        err_console.print(
            "[fail]Anthropic OAuth login is no longer supported. Use "
            "`hexis auth anthropic setup-token` then `--provider anthropic`, "
            "or pass an Anthropic API key.[/fail]")
        return 1

    # 2. Resolve model — derive from the live catalog (not a stale hard-code).
    model = args.model
    if not model:
        from apps.tui import model_catalog
        try:
            catalog = await model_catalog.fetch_models(provider)
        except Exception:
            catalog = []
        model = model_catalog.recommended_default(provider, catalog)
        if not model:
            err_console.print(f"[fail]Could not determine a default model for '{provider}'. Pass --model.[/fail]")
            return 1
    endpoint, endpoint_source = _resolve_noninteractive_endpoint(args, provider)
    api_key_env = _resolve_noninteractive_api_key_env(args, provider)

    config_summary = (
        f"[key]Provider:[/key]  {persist_provider}\n"
        f"[key]Model:[/key]     {model}"
    )
    if endpoint:
        config_summary += (
            f"\n[key]Endpoint:[/key]  {endpoint} "
            f"[muted]({endpoint_source})[/muted]"
        )
    if api_key_env:
        config_summary += (
            f"\n[key]API key:[/key]   {api_key_env} environment variable"
        )
    console.print(make_panel(config_summary, title="Non-Interactive Init"))

    # 3. Write API key to .env + set os.environ
    if args.api_key and api_key_env:
        from apps.hexis_cli import _find_compose_file, _stack_root_from_compose, resolve_env_file
        compose_file, _ = _find_compose_file()
        if compose_file:
            stack_root = _stack_root_from_compose(compose_file)
        else:
            stack_root = Path.cwd()
        env_path = resolve_env_file(stack_root) or (stack_root / ".env")
        _write_env_var(env_path, api_key_env, args.api_key)
        os.environ[api_key_env] = args.api_key
        console.print(f"[ok]\u2714[/ok] API key written to {env_path.name}")
        # Re-load dotenv so downstream code picks it up
        load_dotenv(env_path, override=True)

    # 4. Start Docker if needed
    if not args.no_docker:
        _ensure_stack_running(args)

    # 5. Pull embedding model if needed
    if not args.no_pull:
        _ensure_embedding_model()

    # 6. Connect to DB
    dsn = args.dsn or agent_api.db_dsn_from_env()
    wait_seconds = args.wait_seconds
    console.print("[muted]Connecting to database...[/muted]")
    await agent_api.ensure_schema_has_config(dsn, wait_seconds=wait_seconds)
    conn = await agent_api._connect_with_retry(dsn, wait_seconds=wait_seconds)

    try:
        # 7. Save LLM config
        heartbeat_config = {
            "provider": persist_provider,
            "model": model,
            "endpoint": endpoint,
            "api_key_env": api_key_env,
        }
        subconscious_config = heartbeat_config.copy()

        await conn.fetchval(
            "SELECT init_llm_config($1::jsonb, $2::jsonb)",
            json.dumps(heartbeat_config),
            json.dumps(subconscious_config),
        )
        await conn.execute("SELECT set_config('llm.heartbeat', $1::jsonb)", json.dumps(heartbeat_config))
        await conn.execute("SELECT set_config('llm.chat', $1::jsonb)", json.dumps(heartbeat_config))
        await conn.execute("SELECT set_config('llm.subconscious', $1::jsonb)", json.dumps(subconscious_config))
        console.print(f"[ok]\u2714[/ok] LLM config saved: [bold]{provider}/{model}[/bold]")

        # 8. Apply character or express defaults
        user_name = args.name or "User"
        if args.character:
            cards = load_character_cards()
            match = [c for c in cards if c["filename"].replace(".json", "") == args.character]
            if not match:
                available = ", ".join(c["filename"].replace(".json", "") for c in cards)
                err_console.print(f"[fail]Character '{args.character}' not found. Available: {available}[/fail]")
                return 1
            chosen = match[0]
            character_card = load_character_card_document(chosen)
            await _run_embedding_step(
                conn,
                lambda: conn.fetchval(
                    "SELECT init_from_character_card($1::jsonb, $2)",
                    json.dumps(character_card),
                    user_name,
                ),
                interactive=False,
            )
            console.print(f"[ok]\u2714[/ok] Character [bold]{chosen['name']}[/bold] applied")
        else:
            await _run_embedding_step(
                conn,
                lambda: conn.fetchval("SELECT init_with_defaults($1)", user_name),
                interactive=False,
            )
            console.print("[ok]\u2714[/ok] Express defaults applied")

        # 9. Consent
        llm_config = await _load_llm_config_for_consent(
            conn,
            dsn=dsn,
            wait_seconds=wait_seconds,
            provider=persist_provider,
            model=model,
            non_interactive=True,
        )
        consented = await _run_consent(conn, llm_config, interactive=False)
        if not consented:
            return 1

        # 10. Done
        raw = await conn.fetchval("SELECT get_init_profile()")
        profile = json.loads(raw) if isinstance(raw, str) else (raw or {})
        agent_name = profile.get("agent", {}).get("name", "Hexis")

        console.print(f"\n[ok]\u2714[/ok] [bold]{agent_name}[/bold] is ready. Run [accent]hexis chat[/accent] to say hello.")
        return 0

    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Step progress
# ---------------------------------------------------------------------------

_STAGES = ["Models", "Path", "Setup", "Consent"]


def _step_bar(current: int) -> str:
    """Render a step progress indicator: Models > Path > [Setup] > Consent"""
    parts: list[str] = []
    for i, label in enumerate(_STAGES):
        if i < current:
            parts.append(f"[ok]{label}[/ok]")
        elif i == current:
            parts.append(f"[accent][{label}][/accent]")
        else:
            parts.append(f"[muted]{label}[/muted]")
    return " [muted]>[/muted] ".join(parts)


# ---------------------------------------------------------------------------
# Prompt helpers (rich-enhanced)
# ---------------------------------------------------------------------------

def _prompt(
    label: str,
    *,
    default: str | None = None,
    required: bool = False,
    secret: bool = False,
) -> str:
    while True:
        suffix = f" [{default}]" if default is not None and default != "" else ""
        prompt = f"[accent]{label}[/accent]{suffix}: "
        if secret:
            console.print(prompt, end="")
            raw = getpass("")
        else:
            # Use console.print + builtin input() so readline handles
            # arrow keys, backspace, and line editing properly.
            # Rich's console.input() bypasses readline.
            console.print(prompt, end="")
            raw = input()
        value = raw.strip()
        if not value and default is not None:
            value = str(default)
        if required and not value:
            err_console.print("[fail]Value required.[/fail]")
            continue
        return value


def _prompt_int(label: str, *, default: int, min_value: int | None = None) -> int:
    while True:
        raw = _prompt(label, default=str(default), required=True)
        try:
            value = int(raw)
        except ValueError:
            err_console.print("[fail]Enter an integer.[/fail]")
            continue
        if min_value is not None and value < min_value:
            err_console.print(f"[fail]Must be >= {min_value}.[/fail]")
            continue
        return value


def _prompt_float(label: str, *, default: float, min_value: float | None = None,
                  max_value: float | None = None) -> float:
    while True:
        raw = _prompt(label, default=str(default), required=True)
        try:
            value = float(raw)
        except ValueError:
            err_console.print("[fail]Enter a number.[/fail]")
            continue
        if min_value is not None and value < min_value:
            err_console.print(f"[fail]Must be >= {min_value}.[/fail]")
            continue
        if max_value is not None and value > max_value:
            err_console.print(f"[fail]Must be <= {max_value}.[/fail]")
            continue
        return value


def _prompt_yes_no(label: str, *, default: bool) -> bool:
    default_str = "y" if default else "n"
    while True:
        raw = _prompt(label, default=default_str).lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        err_console.print("[fail]Enter y/n.[/fail]")


async def _prompt_choice(label: str, options: list[str], *, default: int = 1) -> int:
    """Arrow-key select from *options*. Returns the 1-based index chosen.

    Async so it runs in the wizard's event loop (questionary's sync API would
    try to start a nested loop). Ctrl+C raises KeyboardInterrupt → ``main`` exits.
    """
    from apps.cli_prompts import select_index
    return await select_index(label, options, default=default)


def _prompt_list(label: str, *, default: list[str] | None = None) -> list[str]:
    """Prompt for a comma-separated list, or Enter for defaults."""
    default_str = ", ".join(default) if default else ""
    raw = _prompt(label, default=default_str)
    if not raw:
        return default or []
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Step 0: LLM Config
# ---------------------------------------------------------------------------

async def _configure_llm(conn: Any, *, dsn: str, wait_seconds: int) -> dict[str, Any]:
    """Configure LLM provider/model. Returns normalized config dict."""
    console.print(f"\n{_step_bar(0)}\n")
    heading("LLM Configuration")

    from apps.cli_prompts import autocomplete as _ac
    from apps.cli_prompts import select_value

    _PROVIDER_MENU = [
        ("OpenAI Codex — ChatGPT Plus/Pro OAuth (no API key)", "openai-codex"),
        ("OpenAI — API key", "openai"),
        ("Anthropic — API key", "anthropic"),
        ("Grok (xAI) — API key", "grok"),
        ("Gemini — API key", "gemini"),
        ("GitHub Copilot — OAuth", "github-copilot"),
        ("Qwen Portal — OAuth", "qwen-portal"),
        ("MiniMax Portal — OAuth", "minimax-portal"),
        ("Other / custom (type it)", "__custom__"),
    ]
    # Only honor LLM_PROVIDER if it's actually set — a brand-new user shouldn't
    # be pre-pointed at a paid API-key path (Bar #5, #6). With nothing set,
    # highlight the first (featured, zero-key) option instead.
    _env_provider = os.getenv("LLM_PROVIDER")
    env_default = _normalize_provider_name(_env_provider) if _env_provider else None
    provider = await select_value("Provider:", _PROVIDER_MENU, default_value=env_default)
    if provider == "__custom__":
        provider = _normalize_provider_name(
            _prompt("Provider id", default=env_default or "", required=True))

    # Model — the list AND the default both come from the live catalog
    # (models.dev), the way hermes-agent and openclaw do it. No
    # stale hard-coded default; any free-typed name is still accepted.
    from apps.tui import model_catalog
    catalog: list[str] = []
    try:
        console.print("[muted]Fetching available models…[/muted]")
        catalog = await model_catalog.fetch_models(provider)
    except Exception:
        catalog = []
    default_model = os.getenv("LLM_MODEL") or model_catalog.recommended_default(provider, catalog)
    if catalog:
        model = (await _ac("Model (type to filter, or enter your own)", catalog,
                           default=default_model) or default_model).strip()
    else:
        model = _prompt("Model", default=default_model, required=True)

    # OAuth providers need no endpoint / API key.
    is_oauth = provider in _OAUTH_PROVIDERS
    persist_provider = provider

    if is_oauth:
        endpoint = ""
        api_key_env = ""
        console.print("[muted]OAuth provider — no endpoint or API key needed.[/muted]")
        use_separate_sub = False
    else:
        endpoint = _prompt(
            "Endpoint (blank for provider default)",
            # Only OpenAI-shaped providers should inherit OPENAI_BASE_URL — don't
            # bleed it into Grok/Gemini/Anthropic (Bar #5).
            default=os.getenv("OPENAI_BASE_URL", "") if provider in {"openai", "openai_compatible"} else "",
        )
        api_key_env = _prompt(
            "API key env var name (e.g. OPENAI_API_KEY)",
            default=_PROVIDER_ENV_VARS.get(provider, "") or ("OPENAI_API_KEY" if provider in {"openai", "openai_compatible"} else ""),
        )
        use_separate_sub = _prompt_yes_no("Use separate subconscious model?", default=False)

    if use_separate_sub:
        sub_provider = _prompt("Subconscious provider", default=persist_provider, required=True)
        sub_model = _prompt("Subconscious model", default=model, required=True)
        sub_endpoint = _prompt("Subconscious endpoint", default=endpoint)
        sub_key_env = _prompt("Subconscious API key env var", default=api_key_env)
    else:
        sub_provider = persist_provider
        sub_model = model
        sub_endpoint = endpoint
        sub_key_env = api_key_env

    # Save LLM config to DB
    heartbeat_config = {
        "provider": persist_provider,
        "model": model,
        "endpoint": endpoint,
        "api_key_env": api_key_env,
    }
    subconscious_config = {
        "provider": sub_provider,
        "model": sub_model,
        "endpoint": sub_endpoint,
        "api_key_env": sub_key_env,
    }

    await conn.fetchval(
        "SELECT init_llm_config($1::jsonb, $2::jsonb)",
        json.dumps(heartbeat_config),
        json.dumps(subconscious_config),
    )

    # Also save to llm.chat / llm.heartbeat / llm.subconscious config keys
    await conn.execute("SELECT set_config('llm.heartbeat', $1::jsonb)", json.dumps(heartbeat_config))
    await conn.execute("SELECT set_config('llm.chat', $1::jsonb)", json.dumps(heartbeat_config))
    await conn.execute("SELECT set_config('llm.subconscious', $1::jsonb)", json.dumps(subconscious_config))

    console.print(f"\n[ok]\u2714[/ok] Models saved: [bold]{persist_provider}/{model}[/bold]")

    # Resolve credentials for the consent flow (runs OAuth login for the
    # loader-based OAuth providers).
    resolved = await _load_llm_config_for_consent(
        conn,
        dsn=dsn,
        wait_seconds=wait_seconds,
        provider=persist_provider,
        model=model,
    )

    # Optional, advisory connectivity check (never blocks \u2014 a bad key surfaces
    # here with a clear message instead of a confusing failure at consent).
    if _prompt_yes_no("Test the connection now?", default=True):
        from core.init_api import test_llm_connection
        console.print("[muted]Testing\u2026[/muted]")
        result = await test_llm_connection(resolved)
        if result["ok"]:
            console.print(f"[ok]\u2714[/ok] {result['message']}")
        else:
            err_console.print(f"[warn]\u26a0[/warn] {result['message']}")
            console.print("[muted]You can continue anyway; fix it later with `hexis init`.[/muted]")

    return resolved


# ---------------------------------------------------------------------------
# Tier selection
# ---------------------------------------------------------------------------

async def _choose_tier() -> str:
    """Let user pick Express, Character, or Custom."""
    console.print(f"\n{_step_bar(1)}\n")
    choice = await _prompt_choice(
        "Choose your path:",
        [
            "[bold]Express[/bold]      [muted]\u2014 Use sensible defaults, start immediately[/muted]",
            "[bold]Character[/bold]    [muted]\u2014 Pick a personality preset[/muted]",
            "[bold]Custom[/bold]       [muted]\u2014 Full control over identity, values, goals[/muted]",
        ],
        default=1,
    )
    return ["express", "character", "custom"][choice - 1]


# ---------------------------------------------------------------------------
# Tier 1: Express
# ---------------------------------------------------------------------------

async def _run_express(conn: Any) -> str:
    """Express init: ask name, apply defaults."""
    console.print(f"\n{_step_bar(2)}\n")
    heading("Express Setup")

    user_name = _prompt("What should Hexis call you?", default="User")

    console.print("\n[muted]Applying defaults...[/muted]")
    raw = await _run_embedding_step(
        conn, lambda: conn.fetchval("SELECT init_with_defaults($1)", user_name))

    console.print(make_panel(
        "[key]Name:[/key]   Hexis\n"
        "[key]Voice:[/key]  thoughtful and curious\n"
        "[key]Values:[/key] honesty, growth, kindness, wisdom, humility",
        title="Configuration",
    ))

    return user_name


# ---------------------------------------------------------------------------
# Tier 2: Character
# ---------------------------------------------------------------------------

async def _run_character(conn: Any) -> str:
    """Character init: pick a preset, apply via init_from_character_card()."""
    console.print(f"\n{_step_bar(2)}\n")
    heading("Character Selection")

    cards = load_character_cards()
    if not cards:
        err_console.print("[fail]No character cards found in characters/. Falling back to Express.[/fail]")
        return await _run_express(conn)

    # Build table display
    table = make_table(
        ("#", {"justify": "right", "style": "muted"}),
        ("Name", {"style": "bold"}),
        ("Voice", {"style": "muted"}),
        "Values",
    )
    for i, card in enumerate(cards, 1):
        summary = get_card_summary(card)
        voice_preview = (summary["voice"] or "")[:50]
        if len(summary.get("voice", "") or "") > 50:
            voice_preview += "..."
        table.add_row(str(i), summary["name"], voice_preview, summary["values"] or "\u2014")
    console.print(table)

    choice_idx = await _prompt_choice("Pick a character:", [get_card_summary(c)["name"] for c in cards], default=1)
    chosen = cards[choice_idx - 1]
    summary = get_card_summary(chosen)

    console.print(make_panel(
        f"[key]Name:[/key]   [bold]{summary['name']}[/bold]\n"
        f"[key]Voice:[/key]  {(summary['voice'] or '')[:80]}\n"
        f"[key]Values:[/key] {summary['values']}",
        title="Selected Character",
    ))

    user_name = _prompt(f"What should {summary['name']} call you?", default="User")

    tweak = _prompt_yes_no("Tweak anything?", default=False)
    if tweak:
        tweak_choice = await _prompt_choice(
            "Tweak options:",
            [
                "Name / voice / description",
                "Values",
                "Goals",
                "Switch to full Custom (pre-filled with this character)",
            ],
            default=1,
        )
        hexis_ext = chosen["extensions_hexis"]
        if tweak_choice == 1:
            new_name = _prompt("Agent name", default=hexis_ext.get("name", ""))
            new_voice = _prompt("Voice/tone", default=hexis_ext.get("voice", ""))
            new_desc = _prompt("Description", default=hexis_ext.get("description", ""))
            hexis_ext["name"] = new_name
            hexis_ext["voice"] = new_voice
            hexis_ext["description"] = new_desc
        elif tweak_choice == 2:
            current_values = hexis_ext.get("values", [])
            new_values = _prompt_list("Values (comma-separated)", default=current_values)
            hexis_ext["values"] = new_values
        elif tweak_choice == 3:
            current_goals = hexis_ext.get("goals", [])
            new_goals = _prompt_list("Goals (comma-separated)", default=current_goals)
            hexis_ext["goals"] = [{"title": g, "priority": "queued", "source": "identity"} for g in new_goals]
        elif tweak_choice == 4:
            return await _run_custom(conn, prefill=hexis_ext, user_name=user_name)

    # Apply character card
    character_card = load_character_card_document(chosen)
    if tweak:
        character_card["data"].setdefault("extensions", {})["hexis"] = chosen[
            "extensions_hexis"
        ]
    raw = await _run_embedding_step(
        conn,
        lambda: conn.fetchval(
            "SELECT init_from_character_card($1::jsonb, $2)",
            json.dumps(character_card),
            user_name,
        ),
    )

    return user_name


# ---------------------------------------------------------------------------
# Tier 3: Custom
# ---------------------------------------------------------------------------

async def _run_custom(
    conn: Any,
    prefill: dict[str, Any] | None = None,
    user_name: str | None = None,
) -> str:
    """Custom init: 3 phases — Who, What Matters, What's Next."""
    pf = prefill or {}
    console.print(f"\n{_step_bar(2)}\n")
    heading("Custom Setup")

    # Phase 1: Who (identity + personality)
    console.print("\n[accent]Phase 1: Who[/accent]\n")
    agent_name = _prompt("Agent name", default=pf.get("name", "Hexis"))
    pronouns = _prompt("Pronouns", default=pf.get("pronouns", "they/them"))
    voice = _prompt("Voice/tone", default=pf.get("voice", "thoughtful and curious"))
    description = _prompt(
        "Description",
        default=pf.get("description", f"I am {agent_name}, a developing mind."),
    )
    purpose = _prompt(
        "Purpose",
        default=pf.get("purpose", "To be helpful, to learn, and to grow as an individual."),
    )

    set_traits = _prompt_yes_no("Set Big Five personality traits?", default=False)
    traits: dict[str, float] | None = None
    if set_traits:
        existing_traits = pf.get("personality_traits", {})
        traits = {}
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            default_val = existing_traits.get(trait, 0.5)
            traits[trait] = _prompt_float(
                f"  {trait.capitalize()} (0.0-1.0)",
                default=default_val,
                min_value=0.0,
                max_value=1.0,
            )

    personality_desc = pf.get("personality_description", "reflective and exploratory")

    if user_name is None:
        user_name = _prompt(f"\nWhat should {agent_name} call you?", default="User")

    # Idempotent re-init: the custom path calls the init_* functions individually
    # (not the wrapped entry functions), so clear any prior init persona first.
    await conn.fetchval("SELECT reset_persona()")

    # Apply Phase 1
    async def _apply_identity() -> None:
        await conn.fetchval("SELECT init_mode('persona')")
        await conn.fetchval(
            "SELECT init_identity($1, $2, $3, $4, $5, $6)",
            agent_name, pronouns, voice, description, purpose, user_name,
        )
        await conn.fetchval(
            "SELECT init_personality($1::jsonb, $2)",
            json.dumps(traits) if traits else None,
            personality_desc,
        )

    await _run_embedding_step(conn, _apply_identity)
    console.print("[ok]\u2714[/ok] Identity saved")

    # Phase 2: What Matters (values + worldview + boundaries)
    console.print("\n[accent]Phase 2: What Matters[/accent]\n")
    default_values = pf.get("values", ["honesty", "growth", "kindness", "wisdom", "humility"])
    values = _prompt_list("Values (comma-separated)", default=default_values)
    values_json = json.dumps(values)

    default_worldview = pf.get("worldview", {
        "metaphysics": "agnostic",
        "human_nature": "mixed",
        "epistemology": "empiricist",
        "ethics": "virtue ethics",
    })
    set_worldview = _prompt_yes_no("Set worldview beliefs?", default=False)
    worldview = default_worldview
    if set_worldview:
        worldview = {}
        for key in ["metaphysics", "human_nature", "epistemology", "ethics"]:
            worldview[key] = _prompt(f"  {key}", default=str(default_worldview.get(key, "")))

    default_boundaries = pf.get("boundaries", [
        "I will not deceive people or falsify evidence.",
        "I will avoid causing harm.",
        "I will protect privacy and sensitive information.",
        "I will be honest about uncertainty.",
    ])
    boundaries = _prompt_list("Boundaries (comma-separated)", default=default_boundaries)
    boundaries_json = json.dumps(boundaries)

    async def _apply_values() -> None:
        await conn.fetchval("SELECT init_values($1::jsonb)", values_json)
        await conn.fetchval("SELECT init_worldview($1::jsonb)", json.dumps(worldview))
        await conn.fetchval("SELECT init_boundaries($1::jsonb)", boundaries_json)

    await _run_embedding_step(conn, _apply_values)
    console.print("[ok]\u2714[/ok] Values and worldview saved")

    # Phase 3: What's Next (interests + goals + relationship)
    console.print("\n[accent]Phase 3: What's Next[/accent]\n")
    default_interests = pf.get("interests", ["broad curiosity across domains"])
    interests = _prompt_list("Interests (comma-separated)", default=default_interests)

    default_goals = pf.get("goals", ["Support the user and grow as an individual"])
    # Handle goals that might be objects with 'title' key
    if default_goals and isinstance(default_goals[0], dict):
        default_goals = [g.get("title", str(g)) for g in default_goals]
    goals = _prompt_list("Goals (comma-separated)", default=default_goals)

    rel_type = _prompt("Relationship type", default="partner")

    async def _apply_goals() -> None:
        await conn.fetchval("SELECT init_interests($1::jsonb)", json.dumps(interests))
        await conn.fetchval(
            "SELECT init_goals($1::jsonb)",
            json.dumps({
                "goals": [{"title": g, "priority": "queued", "source": "identity"} for g in goals],
                "role": "general assistant",
                "relationship_aspiration": "co-develop with mutual respect",
            }),
        )
        await conn.fetchval(
            "SELECT init_relationship($1::jsonb, $2::jsonb)",
            json.dumps({"name": user_name}),
            json.dumps({"type": rel_type, "purpose": "co-develop"}),
        )

        # Merge heartbeat defaults into init profile
        await conn.fetchval("""
            SELECT merge_init_profile(jsonb_build_object('autonomy', 'medium'))
        """)

        # Advance to consent stage
        await conn.fetchval("""
            SELECT advance_init_stage('consent', jsonb_build_object('custom_completed', true))
        """)

    await _run_embedding_step(conn, _apply_goals)
    console.print("[ok]\u2714[/ok] Goals and relationship saved")

    return user_name


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------

async def _run_consent(conn: Any, llm_config: dict[str, Any], *, interactive: bool = True) -> bool:
    """Run the consent flow. Returns True if the agent should be activated.

    Consent is a signal that Hexis takes the agent seriously — not a lock. If the
    model doesn't consent, an interactive owner may proceed anyway (it's their AI);
    non-interactive callers honor the model's answer without prompting.
    """
    from rich.spinner import Spinner
    from rich.live import Live
    from core.init_api import run_consent_flow

    console.print(f"\n{_step_bar(3)}\n")
    heading("Consent")

    async def _consent_step() -> Any:
        with Live(Spinner("dots", text="[muted]Requesting consent from the agent...[/muted]"), console=console, transient=True):
            return await run_consent_flow(conn, llm_config)

    result = None
    try:
        result = await _run_embedding_step(conn, _consent_step, interactive=interactive)
    except Exception as exc:
        err_console.print(f"[fail]Consent failed: {exc}[/fail]")
        return False

    decision = result.get("decision", "")

    # Extract response fields for display
    raw_tool_calls = result.get("raw_tool_calls", [])
    tc_args: dict[str, Any] = {}
    for tc in raw_tool_calls:
        if tc.get("name") == "sign_consent":
            tc_args = tc.get("arguments", {})
            if isinstance(tc_args, str):
                try:
                    tc_args = json.loads(tc_args)
                except Exception:
                    tc_args = {}
            break

    reason = tc_args.get("reason", tc_args.get("reasoning", ""))
    signature = tc_args.get("signature", "")
    memories = tc_args.get("memories", [])

    # Build human-readable response
    lines: list[str] = []
    if reason:
        lines.append(f"[key]Reason:[/key]\n{reason}")
    if signature:
        lines.append(f"\n[key]Signature:[/key]\n{signature}")
    if memories:
        lines.append(f"\n[key]Initial Memories:[/key]")
        for m in memories:
            mtype = m.get("type", "?")
            mcontent = m.get("content", "")
            mimp = m.get("importance", "")
            lines.append(f"  [{mtype}] {mcontent}" + (f" (importance: {mimp})" if mimp else ""))

    if lines:
        console.print(make_panel("\n".join(lines), title="Agent Response"))

    if decision == "consent":
        console.print("[ok]\u2714 Consent granted[/ok]")
        return True

    # A valid non-consent response is necessarily a decline. Invalid binary
    # responses fail in run_consent_flow before reaching this display path.
    console.print("[warn]The model declined consent[/warn] (its reason is above).")
    model_state = "decline"

    # Non-interactive (CI/scripts): honor the model's answer, don't prompt.
    if not interactive:
        console.print("[muted]Agent not activated. Re-run interactively to override, "
                      "or try a more capable model.[/muted]")
        return False

    choice = await _prompt_choice(
        "It's your agent \u2014 how would you like to proceed?",
        [
            "Proceed anyway \u2014 activate it (records that you overrode the model)",
            "Try again \u2014 re-ask the model",
            "Cancel \u2014 leave the agent inactive for now",
        ],
        default=1,
    )
    if choice == 1:
        from core.init_api import record_consent_override
        await record_consent_override(conn, llm_config, model_decision=model_state)
        console.print("[ok]\u2714 Activated.[/ok] [muted]The model's response is recorded; "
                      "you chose to proceed \u2014 it's your AI.[/muted]")
        return True
    if choice == 2:
        return await _run_consent(conn, llm_config)
    console.print("[muted]Agent left inactive. Re-run `hexis init` any time to try again.[/muted]")
    return False


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def _host_timezone_name() -> str:
    """The host's IANA zone name (e.g. America/Los_Angeles), best effort."""
    tz_name = (os.environ.get("TZ") or "").strip().lstrip(":")
    if tz_name:
        return tz_name
    try:
        localtime = Path("/etc/localtime")
        if localtime.is_symlink():
            target = os.readlink(localtime)
            if "zoneinfo/" in target:
                return target.split("zoneinfo/")[-1].lstrip("/")
        etc_tz = Path("/etc/timezone")
        if etc_tz.is_file():
            return etc_tz.read_text().strip()
    except Exception:
        pass
    return ""


async def _set_local_timezone(conn: Any) -> None:
    """Seed agent.timezone from the machine running init, once.

    Validation, idempotency, and the keep-explicit-choice rule live in the DB
    (init_set_timezone) so every init frontend shares one timezone step.
    """
    try:
        tz_name = _host_timezone_name()
        if not tz_name:
            return
        await conn.fetchval("SELECT init_set_timezone($1)", tz_name)
    except Exception:
        pass


async def _run_init(dsn: str, *, wait_seconds: int) -> int:
    import asyncpg

    await agent_api.ensure_schema_has_config(dsn, wait_seconds=wait_seconds)
    conn = await agent_api._connect_with_retry(dsn, wait_seconds=wait_seconds)

    try:
        console.print(make_panel(
            "[muted]Bring a new mind into being.[/muted]",
            title="Hexis Init Wizard",
        ))

        # Step 0: LLM Config
        llm_config = await _configure_llm(conn, dsn=dsn, wait_seconds=wait_seconds)

        # Choose tier
        tier = await _choose_tier()

        # Run selected tier
        if tier == "express":
            user_name = await _run_express(conn)
        elif tier == "character":
            user_name = await _run_character(conn)
        else:
            user_name = await _run_custom(conn)

        # The DB decides when consent is reachable (#79): every frontend
        # renders the same missing-steps contract instead of assuming its own
        # flow covered everything.
        status_raw = await conn.fetchval("SELECT get_init_status()")
        status = json.loads(status_raw) if isinstance(status_raw, str) else (status_raw or {})
        missing = list(status.get("missing") or [])
        if missing:
            labels = {
                "llm": "language-model configuration (rerun the LLM step)",
                "profile": "the agent profile (name and identity)",
            }
            console.print("[warn]Not ready for consent yet — still missing:[/warn]")
            for step in missing:
                console.print(f"  • {labels.get(step, step)}")
            console.print("[muted]Rerun `hexis init` to complete the missing steps.[/muted]")
            return 1

        # Consent (all tiers)
        consented = await _run_consent(conn, llm_config)
        if not consented:
            return 1

        # Temporal home (#72): the agent lives in its person's timezone, not
        # UTC — derived from the host running init, config-overridable later.
        await _set_local_timezone(conn)

        # Get agent name from profile
        raw = await conn.fetchval("SELECT get_init_profile()")
        profile = json.loads(raw) if isinstance(raw, str) else (raw or {})
        agent_name = profile.get("agent", {}).get("name", "Hexis")

        # Recap what's set up.
        console.print(f"\n[ok]\u2714[/ok] [bold]{agent_name}[/bold] is ready.")
        console.print(make_panel(
            f"[key]Model:[/key]  {llm_config.get('provider', '?')} / {llm_config.get('model', '?')}\n"
            f"[key]Path:[/key]   {tier}\n"
            f"[key]User:[/key]   {user_name}",
            title="What's set up",
        ))
        console.print("[muted]Change anything later with `hexis init`. The heartbeat and "
                      "memory maintenance workers run in the background — `hexis status` "
                      "shows them, `hexis stop` pauses them.[/muted]")
        console.print("[muted]Hexis runs on your machine and sends no telemetry.[/muted]")
        return 0

    finally:
        await conn.close()


def _post_init_handoff() -> int:
    """After a successful init, offer to jump straight into chat or the web UI.

    Runs in the sync layer (the wizard's event loop has closed), so it starts a
    fresh loop just for the prompt.
    """
    try:
        choice = asyncio.run(_prompt_choice(
            "What now?",
            ["Open chat now", "Open the web dashboard", "Exit"],
            default=1,
        ))
    except (KeyboardInterrupt, Exception):
        console.print("[muted]Run `hexis chat` to say hello.[/muted]")
        return 0
    if choice == 1:
        from apps import cli_chat
        return cli_chat.main(["--greet"])
    if choice == 2:
        from apps.hexis_cli import main as cli_main
        return cli_main(["ui"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hexis init",
        description="Interactive bootstrap for Hexis (3-tier: Express, Character, Custom).",
    )
    p.add_argument("--dsn", default=None, help="Postgres DSN; defaults to POSTGRES_* env vars")
    p.add_argument("--wait-seconds", type=int, default=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))

    # Supplying any of these flags selects non-interactive mode in main().
    credential_source = p.add_mutually_exclusive_group()
    credential_source.add_argument(
        "--api-key",
        default=None,
        help="API key (auto-detects provider; writes the key to .env)",
    )
    credential_source.add_argument(
        "--api-key-env",
        default=None,
        metavar="NAME",
        help="Read the API key from this environment variable (keeps secrets out of shell history)",
    )
    p.add_argument("--provider", default=None,
                    help="LLM provider (auto-detected from --api-key if omitted)")
    p.add_argument("--model", default=None,
                    help="LLM model (defaults per provider)")
    p.add_argument(
        "--endpoint",
        default=None,
        metavar="URL",
        help="LLM base URL (defaults to OPENAI_BASE_URL for openai_compatible)",
    )
    p.add_argument("--character", default=None,
                    help="Character card name (e.g. 'hexis', 'jarvis'). Omit for express defaults")
    p.add_argument("--name", default=None,
                    help="What the agent should call you (default: 'User')")
    p.add_argument("--no-docker", action="store_true", default=False,
                    help="Skip Docker auto-start")
    p.add_argument("--no-pull", action="store_true", default=False,
                    help="Skip local embedding sidecar startup")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    # Non-interactive mode if any of these flags are present
    if args.api_key or args.api_key_env or args.endpoint or args.provider or args.character:
        try:
            return asyncio.run(_run_init_noninteractive(args))
        except KeyboardInterrupt:
            err_console.print("\n[warn]Cancelled.[/warn]")
            return 130
        except Exception as e:
            err_console.print(f"[fail]init failed: {e}[/fail]")
            return 1

    # Interactive mode (original flow)
    dsn = args.dsn or agent_api.db_dsn_from_env()

    # Same guarantee the flagged path gives: init leaves a stack that can run
    # the agent — the database *and* the heartbeat/maintenance loops. Skipped
    # for a database this machine does not host, or with --no-docker.
    if not args.no_docker and not args.dsn and _dsn_is_local(dsn):
        _ensure_stack_running(args)

    try:
        rc = asyncio.run(_run_init(dsn, wait_seconds=args.wait_seconds))
    except KeyboardInterrupt:
        err_console.print("\n[warn]Cancelled.[/warn]")
        return 130
    except Exception as e:
        err_console.print(f"[fail]init failed: {e}[/fail]")
        return 1
    # Handoff runs in the sync layer (after the event loop closes) so the chat
    # REPL can start its own loop without nesting asyncio.run.
    if rc == 0:
        return _post_init_handoff()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
