from __future__ import annotations

import json
import os
from typing import Any, Callable, Coroutine

from core.llm import normalize_llm_config, normalize_provider


DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# Per-provider config loaders (inject api_key / endpoint / auth_mode)
# ---------------------------------------------------------------------------

async def _load_openai_codex(conn, cfg: dict[str, Any]) -> None:
    from core.auth.openai_codex import ensure_fresh_openai_codex_credentials

    creds = await ensure_fresh_openai_codex_credentials()
    cfg["api_key"] = creds.access
    cfg.setdefault("endpoint", "https://chatgpt.com/backend-api")


async def _load_chutes(conn, cfg: dict[str, Any]) -> None:
    from core.auth.chutes import CHUTES_DEFAULT_ENDPOINT, ensure_fresh_credentials

    creds = await ensure_fresh_credentials()
    cfg["api_key"] = creds.access
    cfg.setdefault("endpoint", CHUTES_DEFAULT_ENDPOINT)


async def _load_github_copilot(conn, cfg: dict[str, Any]) -> None:
    from core.auth.github_copilot import ensure_fresh_credentials

    creds = await ensure_fresh_credentials()
    cfg["api_key"] = creds.access
    cfg.setdefault("endpoint", creds.base_url)


async def _load_qwen_portal(conn, cfg: dict[str, Any]) -> None:
    from core.auth.qwen_portal import QWEN_PORTAL_DEFAULT_ENDPOINT, ensure_fresh_credentials

    creds = await ensure_fresh_credentials()
    cfg["api_key"] = creds.access
    cfg.setdefault("endpoint", creds.resource_url or QWEN_PORTAL_DEFAULT_ENDPOINT)


async def _load_minimax_portal(conn, cfg: dict[str, Any]) -> None:
    from core.auth.minimax_portal import default_endpoint, ensure_fresh_credentials

    creds = await ensure_fresh_credentials()
    cfg["api_key"] = creds.access
    cfg.setdefault("endpoint", creds.resource_url or default_endpoint(creds.region))


async def _load_google_gemini_cli(conn, cfg: dict[str, Any]) -> None:
    from core.auth.google_gemini_cli import GOOGLE_CODE_ASSIST_ENDPOINT, ensure_fresh_credentials

    creds = await ensure_fresh_credentials()
    # Pack token + project as JSON for downstream parsing in llm.py
    cfg["api_key"] = json.dumps({"token": creds.access, "projectId": creds.project_id})
    cfg.setdefault("endpoint", GOOGLE_CODE_ASSIST_ENDPOINT)


async def _load_google_antigravity(conn, cfg: dict[str, Any]) -> None:
    from core.auth.google_antigravity import CODE_ASSIST_ENDPOINTS, ensure_fresh_credentials

    creds = await ensure_fresh_credentials()
    cfg["api_key"] = json.dumps({"token": creds.access, "projectId": creds.project_id})
    cfg.setdefault("endpoint", CODE_ASSIST_ENDPOINTS[0])


async def _load_anthropic_credentials(conn, cfg: dict[str, Any]) -> None:
    """Resolve Anthropic credentials from Hexis's stored setup token.

    Priority: setup token > env API key (the latter via normalize_llm_config).
    """
    from core.auth.anthropic_setup_token import resolve_anthropic_token

    token, auth_mode = await resolve_anthropic_token()
    if token:
        cfg["api_key"] = token
        if auth_mode:
            cfg["auth_mode"] = auth_mode


_PROVIDER_CONFIG_LOADERS: dict[
    str,
    Callable[..., Coroutine[Any, Any, None]],
] = {
    "openai-codex": _load_openai_codex,
    "chutes": _load_chutes,
    "github-copilot": _load_github_copilot,
    "qwen-portal": _load_qwen_portal,
    "minimax-portal": _load_minimax_portal,
    "google-gemini-cli": _load_google_gemini_cli,
    "google-antigravity": _load_google_antigravity,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configured_llm_identity(
    config: dict[str, Any] | None,
    *,
    default_provider: str = DEFAULT_LLM_PROVIDER,
    default_model: str = DEFAULT_LLM_MODEL,
) -> dict[str, str]:
    """Resolve public provider/model identity without loading credentials."""

    config = config if isinstance(config, dict) else {}
    provider = normalize_provider(str(config.get("provider") or default_provider))
    model = str(
        config.get("model")
        or ("gpt-5.2" if provider == "openai-codex" else default_model)
    )
    return {"provider": provider, "model": model}


async def load_llm_config(
    conn,
    key: str,
    *,
    default_provider: str = DEFAULT_LLM_PROVIDER,
    default_model: str = DEFAULT_LLM_MODEL,
    fallback_key: str | None = None,
) -> dict[str, Any]:
    async def _fetch(config_key: str) -> dict[str, Any] | None:
        raw = await conn.fetchval("SELECT get_config($1)", config_key)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = None
        # A JSON-null or empty row means "no override configured" (the seeds
        # use 'null'::jsonb for optional overrides) — treat it as absent so
        # fallback_key applies, instead of silently defaulting the provider.
        return raw if isinstance(raw, dict) and raw else None

    cfg = await _fetch(key)
    if cfg is None and fallback_key:
        cfg = await _fetch(fallback_key)
    if cfg is None:
        cfg = {}

    identity = configured_llm_identity(
        cfg,
        default_provider=default_provider,
        default_model=default_model,
    )
    provider = identity["provider"]
    cfg.update(identity)

    # Run the provider-specific config loader (inject api_key, endpoint, auth_mode).
    loader = _PROVIDER_CONFIG_LOADERS.get(provider)
    if loader:
        await loader(conn, cfg)
    elif provider == "anthropic" and not cfg.get("api_key"):
        await _load_anthropic_credentials(conn, cfg)

    return normalize_llm_config(cfg, default_model=default_model)


async def resolve_llm_config(
    pool_or_conn,
    key: str = "llm.chat",
    *,
    fallback_key: str | None = "llm",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around :func:`load_llm_config`.

    Accepts either an ``asyncpg.Pool`` or an ``asyncpg.Connection``.  When a
    pool is provided, a connection is acquired automatically.  This makes it
    usable from both tool handlers (which have pools) and CLI paths (which
    already hold a connection).

    Optional *overrides* are merged **after** credential resolution so that
    callers can patch fields (e.g. model) without interfering with the auth
    flow.
    """

    async def _load(conn) -> dict[str, Any]:
        cfg = await load_llm_config(conn, key, fallback_key=fallback_key)
        if overrides:
            cfg.update(overrides)
        return cfg

    # Duck-type: pools have .acquire(), connections don't.
    if hasattr(pool_or_conn, "acquire"):
        async with pool_or_conn.acquire() as conn:
            return await _load(conn)
    return await _load(pool_or_conn)
