"""Current per-provider model lists for the init wizard.

Mirrors how hermes-agent / openclaw populate their model pickers: prefer a live,
auto-updating source over a hand-maintained list, treat it as advisory, and
always allow a free-typed id.

  * cloud providers → the models.dev catalog (one no-auth JSON covering every
    provider), cached on disk ~24h and sorted newest-first
  * on any failure → a short curated fallback; the model field stays free-text
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from core.integration_reliability import request_json

MODELS_DEV_URL = "https://models.dev/api.json"
_CACHE = Path.home() / ".cache" / "hexis" / "models_dev.json"
_CACHE_TTL = 24 * 3600
_TIMEOUT = 12.0

# Hexis provider id → models.dev slug.
PROVIDER_SLUG: dict[str, str] = {
    "openai": "openai",
    "openai-codex": "openai",
    "anthropic": "anthropic",
    "grok": "xai",
    "gemini": "google",
    "chutes": "chutes",
    "github-copilot": "github-copilot",
    "qwen-portal": "alibaba",
    "minimax-portal": "minimax",
    "google-gemini-cli": "google",
    "google-antigravity": "google",
}

# Non-chat model ids to hide from the dropdown (still typeable as free text).
_NON_CHAT_RE = re.compile(
    r"embed|tts|whisper|moderation|rerank|image|audio|video|dall.?e|imagen|"
    r"veo|sora|guard|ocr|speech|transcrib",
    re.IGNORECASE,
)

# Minimal offline fallback if models.dev is unreachable.
_FALLBACK: dict[str, list[str]] = {
    "openai": ["gpt-5.2", "gpt-4o", "gpt-4o-mini"],
    "openai-codex": ["gpt-5.2", "gpt-5.2-codex"],
    "anthropic": ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"],
    "grok": ["grok-4.3", "grok-3"],
    "gemini": ["gemini-3-pro-preview", "gemini-2.5-flash"],
    "chutes": ["deepseek-ai/DeepSeek-V3-0324"],
    "github-copilot": ["gpt-4o"],
}

_MEM: dict | None = None


def _cache_read() -> dict | None:
    try:
        if _CACHE.exists() and (time.time() - _CACHE.stat().st_mtime) < _CACHE_TTL:
            return json.loads(_CACHE.read_text())
    except Exception:
        pass
    return None


def _cache_write(data: dict) -> None:
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(data))
    except Exception:
        pass


async def _models_dev() -> dict:
    global _MEM
    if _MEM is not None:
        return _MEM
    cached = _cache_read()
    if cached is not None:
        _MEM = cached
        return cached
    data = await request_json(
        "models_dev",
        "GET",
        MODELS_DEV_URL,
        timeout=_TIMEOUT,
        attempts=3,
        max_delay=5.0,
    )
    if not isinstance(data, dict):
        raise RuntimeError("models.dev returned an invalid catalog payload.")
    _cache_write(data)
    _MEM = data
    return data


def _sort_key(m: dict) -> str:
    return m.get("last_updated") or m.get("release_date") or ""


# Variant/specialty suffixes that shouldn't be the *default* pick (they're still
# in the list — you can choose them). Keeps the default a sensible flagship.
_DEFAULT_SKIP = (
    "-pro", "-mini", "-nano", "-lite", "preview", "-exp", "experimental",
    "thinking", "chat-latest", "non-reasoning", "multi-agent", "imagine",
    "-build", "deep-research", "realtime", "-audio", "-tts", "-image",
    "-high", "-low", "-search", "computer-use",
)

# Only where "newest non-variant flagship" isn't the right recommendation.
# Self-healing: ignored if not present in the live catalog.
_PREFERRED_DEFAULT: dict[str, str] = {
    "openai-codex": "gpt-5.2-codex",
}


def recommended_default(provider: str, models: list[str]) -> str:
    """Pick a sensible default model from the *live* catalog (newest-first).

    No provider's default model id is hard-coded (except the tiny self-healing
    override above). Prefers the newest non-variant flagship; falls back to the
    newest model; returns "" only if the catalog is empty.
    """
    if not models:
        return ""
    pref = _PREFERRED_DEFAULT.get(provider)
    if pref and pref in models:
        return pref
    for mid in models:  # models is newest-first
        if not any(s in mid.lower() for s in _DEFAULT_SKIP):
            return mid
    return models[0]


def chat_models(provider_block: dict) -> list[str]:
    """Extract text/chat model ids from a models.dev provider block, newest first."""
    rows: list[tuple[str, str]] = []
    for m in provider_block.get("models", {}).values():
        mid = m.get("id", "")
        if not mid or _NON_CHAT_RE.search(mid):
            continue
        modal = m.get("modalities") or {}
        outs = modal.get("output") if isinstance(modal, dict) else None
        if outs and "text" not in outs:
            continue
        rows.append((mid, _sort_key(m)))
    rows.sort(key=lambda t: t[1], reverse=True)
    seen: set[str] = set()
    ids: list[str] = []
    for mid, _ in rows:
        if mid not in seen:
            seen.add(mid)
            ids.append(mid)
    return ids


async def fetch_models(provider: str, *, endpoint: str | None = None) -> list[str]:
    """Best-effort current chat model ids for *provider* (newest first)."""
    _ = endpoint
    try:
        slug = PROVIDER_SLUG.get(provider)
        if not slug:
            return _FALLBACK.get(provider, [])
        data = await _models_dev()
        block = data.get(slug)
        if not block:
            return _FALLBACK.get(provider, [])
        return chat_models(block) or _FALLBACK.get(provider, [])
    except Exception:
        return _FALLBACK.get(provider, [])
