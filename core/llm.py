from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, AsyncIterator

import httpx

from core.integration_reliability import IntegrationHttpError, iter_sse_json_events

logger = logging.getLogger(__name__)

try:
    import openai
except Exception:  # pragma: no cover
    openai = None  # type: ignore[assignment]

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]

try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as gemini_types  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    gemini_types = None  # type: ignore[assignment]


OPENAI_COMPATIBLE = {
    "openai",
    "openai_compatible",
    "openai-chat-completions-endpoint",
    "grok",
    "chutes",
    "github-copilot",
    "qwen-portal",
}

# OpenAI Codex (ChatGPT subscription) backend
_CODEX_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api"
_CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"

# OpenAI client cache: (api_key, base_url, provider) -> client
_openai_clients: dict[tuple[str, str, str], Any] = {}

# LLM retry configuration
_LLM_MAX_RETRIES = 4
_LLM_RETRY_BACKOFF_BASE = 2  # seconds
_LLM_RETRY_MAX_WAIT = 60.0  # seconds; cap for Retry-After honoring


def _codex_http_error(resp: Any, body: bytes) -> RuntimeError:
    """Build the Codex HTTP error with retry metadata attached, so
    _retry_on_transient can recognize the status and honor Retry-After."""
    err = RuntimeError(
        f"OpenAI Codex request failed: HTTP {resp.status_code}: {body.decode('utf-8', errors='replace')}"
    )
    err.status_code = resp.status_code
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            err.retry_after = float(retry_after)
        except ValueError:
            pass
    return err


async def _retry_on_transient(
    coro_factory,
    *,
    max_retries: int = _LLM_MAX_RETRIES,
    should_retry: Any | None = None,
) -> Any:
    """Retry an LLM call on transient errors (rate limits, server errors, network).

    ``should_retry`` is an optional zero-arg predicate consulted before each
    retry — streaming callers use it to refuse replay once tokens have
    already reached the consumer.
    """
    import asyncio as _asyncio
    import random as _random

    last_exc = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            status = getattr(exc, 'status_code', None) or getattr(exc, 'status', None)
            is_transient = (
                status in (429, 500, 502, 503, 504, 529)
                or 'rate' in exc_str
                or 'overloaded' in exc_str
                or 'timeout' in exc_str
                or 'connection' in exc_str
                or isinstance(exc, (ConnectionError, TimeoutError, OSError))
            )
            if should_retry is not None and not should_retry():
                is_transient = False
            if is_transient and attempt < max_retries - 1:
                wait = _LLM_RETRY_BACKOFF_BASE ** attempt + _random.uniform(0.0, 1.0)
                retry_after = getattr(exc, 'retry_after', None)
                if retry_after:
                    wait = max(wait, float(retry_after))
                wait = min(wait, _LLM_RETRY_MAX_WAIT)
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, wait, exc,
                )
                await _asyncio.sleep(wait)
                continue
            raise
    raise last_exc  # Should not reach here, but just in case


def _get_openai_client(api_key: str | None, base_url: str | None, provider: str, default_headers: dict[str, Any] | None = None) -> Any:
    """Get or create a cached OpenAI client."""
    if openai is None:
        raise RuntimeError("openai package is required for OpenAI-compatible providers.")

    # Include headers in cache key to handle cases like github-copilot
    headers_key = tuple(sorted((default_headers or {}).items())) if default_headers else ()
    cache_key = (api_key or "", base_url or "", provider, headers_key)
    if cache_key in _openai_clients:
        return _openai_clients[cache_key]

    client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if default_headers:
        client_kwargs["default_headers"] = default_headers

    client = openai.AsyncOpenAI(**client_kwargs)
    _openai_clients[cache_key] = client
    return client


def _clear_openai_client_cache() -> None:
    """Clear the OpenAI client cache. Useful for testing."""
    _openai_clients.clear()


def _b64url_decode(raw: str) -> bytes:
    s = (raw or "").strip()
    if not s:
        return b""
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _extract_codex_account_id(token: str) -> str:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT")
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        account_id = payload.get(_CODEX_JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("Missing account id in token")
        return account_id
    except Exception as exc:
        raise ValueError("Failed to extract accountId from token") from exc


def _resolve_codex_url(base_url: str | None) -> str:
    raw = (base_url or _CODEX_DEFAULT_BASE_URL).strip()
    normalized = raw.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _messages_to_codex_responses_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Convert Chat Completions-style messages into Responses API input items in the
    structured content-part format Codex expects.
    """
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            text = _content_text(content)
            if text.strip():
                system_parts.append(text)
            continue

        if role == "user":
            input_items.append({
                "role": "user",
                "content": _content_to_responses_parts(content),
            })
            continue

        if role == "assistant":
            if content:
                input_items.append({
                    "role": "assistant",
                    "content": _content_to_responses_parts(content, assistant=True),
                })
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args)
                input_items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": args,
                })
            continue

        if role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": str(content),
            })
            continue

    instructions = "\n\n".join(p for p in system_parts if p.strip()) or None
    return instructions, input_items


async def _iter_sse_events_json(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """
    Parse SSE responses where each event is JSON in one or more `data:` lines.
    Mirrors the parsing strategy used by OpenClaw/pi-ai for Codex.
    """
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
            continue
        if line.strip() != "":
            # Ignore non-data SSE fields (event:, id:, retry:, etc.)
            continue
        if not data_lines:
            continue
        data = "\n".join(data_lines).strip()
        data_lines = []
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


async def _codex_responses_completion(
    *,
    model: str,
    endpoint: str | None,
    api_key: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
    on_text_delta: Any | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("Missing OpenAI Codex OAuth access token (api_key).")

    account_id = _extract_codex_account_id(api_key)
    url = _resolve_codex_url(endpoint)
    instructions, input_items = _messages_to_codex_responses_input(messages)
    responses_tools = _tools_to_responses(tools)

    payload: dict[str, Any] = {
        "model": model,
        "store": False,
        "stream": True,
        "input": input_items,
        "text": {"verbosity": "medium"},
        "include": ["reasoning.encrypted_content"],
    }
    if instructions:
        payload["instructions"] = instructions
    # Codex Responses API does not support temperature or max_tokens; omit both.
    _ = temperature, max_tokens  # kept for signature parity

    if responses_tools:
        payload["tools"] = responses_tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = True

    headers = {
        "Authorization": f"Bearer {api_key}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": "pi",
        "accept": "text/event-stream",
        "content-type": "application/json",
        "user-agent": f"hexis (python; {os.uname().sysname if hasattr(os, 'uname') else 'unknown'})",
    }

    import asyncio as _asyncio
    import ssl as _ssl

    _MAX_RETRIES = 3
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)

    for attempt in range(_MAX_RETRIES):
        try:
            return await _codex_responses_attempt(
                url=url, headers=headers, payload=payload,
                timeout=timeout, on_text_delta=on_text_delta,
            )
        except (_ssl.SSLError, httpx.RemoteProtocolError, httpx.ReadError) as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                await _asyncio.sleep(wait)
                continue
            raise RuntimeError(f"OpenAI Codex request failed after {_MAX_RETRIES} retries: {exc}") from exc


async def _codex_responses_attempt(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: Any,
    on_text_delta: Any | None = None,
) -> dict[str, Any]:
    """Single attempt at a Codex Responses API streaming call."""
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    # Current function call (arguments stream)
    current_fc: dict[str, Any] | None = None

    try:
        async for event in iter_sse_json_events(
            "openai-codex",
            "POST",
            url,
            headers=headers,
            json_body=payload,
            timeout=timeout,
            attempts=3,
            max_delay=_LLM_RETRY_MAX_WAIT,
            retry_unsafe_methods=True,
        ):
            event_type = event.get("type")

            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if isinstance(delta, str) and delta:
                    content_parts.append(delta)
                    if on_text_delta:
                        import asyncio

                        result = on_text_delta(delta)
                        if asyncio.iscoroutine(result):
                            await result
                continue

            if event_type == "response.output_item.added":
                item = event.get("item") or {}
                if isinstance(item, dict) and item.get("type") == "function_call":
                    current_fc = {
                        "call_id": item.get("call_id"),
                        "name": item.get("name"),
                        "args_buf": item.get("arguments") or "",
                    }
                continue

            if event_type == "response.function_call_arguments.delta":
                if current_fc is not None:
                    delta = event.get("delta", "")
                    if isinstance(delta, str) and delta:
                        current_fc["args_buf"] = (current_fc.get("args_buf") or "") + delta
                continue

            if event_type == "response.function_call_arguments.done":
                if current_fc is not None:
                    args = event.get("arguments", "")
                    if isinstance(args, str) and args:
                        current_fc["args_buf"] = args
                continue

            if event_type in {"response.done", "response.completed"}:
                break

            if event_type in {"error", "response.failed"}:
                message = (
                    event.get("message")
                    or (event.get("error") or {}).get("message")
                    or "Codex request failed"
                )
                raise RuntimeError(str(message))

            if event_type == "response.output_item.done":
                item = event.get("item") or {}
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue
                call_id = item.get("call_id") or (current_fc or {}).get("call_id")
                name = item.get("name") or (current_fc or {}).get("name")
                args_str = item.get("arguments") or (current_fc or {}).get("args_buf") or ""

                try:
                    args = json.loads(args_str) if isinstance(args_str, str) and args_str else {}
                except Exception:
                    logger.debug("Failed to parse tool arguments: %r", str(args_str)[:200])
                    args = {}
                tool_calls.append({
                    "id": call_id,
                    "name": name or "",
                    "arguments": args,
                })
                current_fc = None
                continue
    except IntegrationHttpError as exc:
        raise RuntimeError(str(exc)) from exc

    return {"content": "".join(content_parts), "tool_calls": tool_calls, "raw": None}


# ---------------------------------------------------------------------------
# Responses API capability detection
# ---------------------------------------------------------------------------

_HAS_RESPONSES_API: bool = False
if openai is not None:
    try:
        from openai.resources import responses as _responses_mod  # noqa: F401
        _HAS_RESPONSES_API = True
    except ImportError:
        pass

# Per-endpoint cache: normalized URL -> True (supported) / False (unsupported)
_endpoint_responses_support: dict[str, bool] = {}


def _endpoint_cache_key(endpoint: str | None) -> str:
    return (endpoint or "default").rstrip("/")


def _should_try_responses(endpoint: str | None) -> bool:
    if not _HAS_RESPONSES_API:
        return False
    key = _endpoint_cache_key(endpoint)
    cached = _endpoint_responses_support.get(key)
    if cached is False:
        return False
    return True


def _cache_responses_support(endpoint: str | None, supported: bool) -> None:
    _endpoint_responses_support[_endpoint_cache_key(endpoint)] = supported


def _is_responses_unsupported_error(exc: Exception) -> bool:
    """Return True if the error means the endpoint lacks Responses API support."""
    if openai is None:
        return False
    if isinstance(exc, openai.NotFoundError):
        return True
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        msg = str(exc).lower()
        if "not found" in msg or "unknown" in msg or "unsupported" in msg:
            return True
    if isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", 0) == 501:
        return True
    return False


# ---------------------------------------------------------------------------
# Responses API format converters
# ---------------------------------------------------------------------------


def _tools_to_responses(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert Chat Completions tools (nested function key) to Responses API flat format."""
    if not tools:
        return []
    result: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", {})
        result.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return result


def _messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Convert Chat Completions messages to Responses API (instructions, input_items).

    System messages → instructions parameter.
    Assistant tool_calls (nested OpenAI format) → function_call items.
    Tool result messages → function_call_output items.
    """
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            text = _content_text(content)
            if text.strip():
                system_parts.append(text)

        elif role == "user":
            input_items.append({"role": "user", "content": _content_to_responses(content)})

        elif role == "assistant":
            if content:
                input_items.append({"role": "assistant", "content": _content_to_responses(content, assistant=True)})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args)
                input_items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": args,
                })

        elif role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": content,
            })

    instructions = "\n\n".join(p for p in system_parts if p.strip()) or None
    return instructions, input_items


def _extract_responses_result(response: Any) -> dict[str, Any]:
    """Extract {content, tool_calls, raw} from a Responses API response object."""
    content = getattr(response, "output_text", "") or ""
    tool_calls: list[dict[str, Any]] = []

    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            raw_args = getattr(item, "arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                logger.debug("Failed to parse tool arguments: %r", str(raw_args)[:200])
                args = {}
            tool_calls.append({
                "id": getattr(item, "call_id", None),
                "name": getattr(item, "name", ""),
                "arguments": args,
            })

    return {"content": content, "tool_calls": tool_calls, "raw": response}


_PROVIDER_ALIASES: dict[str, str] = {
    "openai_chat_completions_endpoint": "openai-chat-completions-endpoint",
    "openai_codex": "openai-codex",
    "github_copilot": "github-copilot",
    "qwen_portal": "qwen-portal",
    "minimax_portal": "minimax-portal",
    "google_gemini_cli": "google-gemini-cli",
    "google_antigravity": "google-antigravity",
}


def normalize_provider(provider: str | None) -> str:
    if not provider:
        return "openai"
    raw = provider.strip().lower()
    return _PROVIDER_ALIASES.get(raw, raw)


def normalize_endpoint(provider: str, endpoint: str | None) -> str | None:
    # ChatGPT subscription OAuth has one registered backend. Never inherit a
    # stale OpenAI API-key endpoint from a previous provider selection.
    if provider == "openai-codex":
        return _CODEX_DEFAULT_BASE_URL
    if endpoint:
        return endpoint.strip() or None
    _DEFAULTS: dict[str, str] = {
        "grok": "https://api.x.ai/v1",
        "chutes": "https://api.chutes.ai/v1",
        "qwen-portal": "https://portal.qwen.ai/v1",
        "google-gemini-cli": "https://cloudcode-pa.googleapis.com",
        "google-antigravity": "https://cloudcode-pa.googleapis.com",
    }
    return _DEFAULTS.get(provider)


def resolve_api_key(api_key_env: str | None) -> str | None:
    if not api_key_env:
        return None
    value = api_key_env.strip()
    if not value:
        return None
    import os

    return os.getenv(value)


def normalize_llm_config(config: dict[str, Any] | None, *, default_model: str = "gpt-4o") -> dict[str, Any]:
    """Normalize a raw LLM config dict (provider aliases, env-var API keys, endpoint defaults).

    .. warning::
        This function does **not** run provider-specific credential loaders
        (OAuth token refresh for Codex, Copilot, Gemini CLI, etc.).  Entry
        points that need fully-resolved credentials should use
        :func:`core.llm_config.resolve_llm_config` or
        :func:`core.llm_config.load_llm_config` instead.
    """
    config = config or {}
    provider = normalize_provider(str(config.get("provider") or "openai"))
    model = str(config.get("model") or default_model)
    endpoint = normalize_endpoint(provider, str(config.get("endpoint") or "").strip() or None)
    api_key = config.get("api_key")
    if not api_key:
        api_key = resolve_api_key(str(config.get("api_key_env") or "").strip() or None)
    if not api_key:
        provider_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "grok": "XAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai_compatible": "OPENAI_API_KEY",
            "openai-chat-completions-endpoint": "OPENAI_API_KEY",
        }
        env_name = provider_env_map.get(provider)
        if env_name:
            api_key = os.getenv(env_name)
    result: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "api_key": api_key,
    }
    # Preserve auth_mode when set by provider-specific config loaders.
    auth_mode = config.get("auth_mode")
    if auth_mode:
        result["auth_mode"] = auth_mode
    return result


def _extract_system_prompt(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    parts, rest = _extract_system_parts(messages)
    return "\n\n".join(parts), rest


def _extract_system_parts(messages: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """System messages kept as separate parts, in order.

    Callers assemble the prompt so that everything stable comes first and
    everything per-turn comes last (see services.agent.SystemPrompt). Providers
    that can cache a prefix need that boundary preserved rather than flattened,
    so the parts stay split until the last moment.
    """
    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            text = _content_text(msg.get("content"))
            if text.strip():
                system_parts.append(text)
        else:
            rest.append(msg)
    return system_parts, rest


def _anthropic_system_blocks(parts: list[str]) -> list[dict[str, Any]] | None:
    """Anthropic `system` as content blocks, with a cache breakpoint.

    Everything except the final part is the stable prefix, so the breakpoint
    goes on the last stable block: Anthropic caches up to and including the
    marked block and re-reads only what follows. With a single part there is
    nothing to cache against, so it is sent plain.
    """
    if not parts:
        return None
    if len(parts) == 1:
        return [{"type": "text", "text": parts[0]}]
    blocks: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        block: dict[str, Any] = {"type": "text", "text": part}
        if index == len(parts) - 2:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "input_text", "output_text"}:
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content or "")


def _image_url_from_part(part: dict[str, Any]) -> str | None:
    raw = part.get("image_url")
    if isinstance(raw, dict):
        url = raw.get("url")
        return str(url) if url else None
    if isinstance(raw, str):
        return raw
    url = part.get("url")
    return str(url) if url else None


def _image_detail_from_part(part: dict[str, Any]) -> str:
    raw = part.get("detail")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    raw_image_url = part.get("image_url")
    if isinstance(raw_image_url, dict):
        raw = raw_image_url.get("detail")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return "auto"


def _data_url_parts(url: str) -> tuple[str | None, str | None]:
    if not url.startswith("data:") or "," not in url:
        return None, None
    header, data = url.split(",", 1)
    media_type = header[5:].split(";", 1)[0] or None
    return media_type, data


def _content_to_openai_chat(content: Any) -> Any:
    if not isinstance(content, list):
        return _content_text(content)
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
            continue
        if ptype in {"image_url", "input_image"}:
            url = _image_url_from_part(part)
            if url:
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": _image_detail_from_part(part),
                    },
                })
    return parts or _content_text(content)


def _content_to_responses(content: Any, *, assistant: bool = False) -> Any:
    if not isinstance(content, list):
        return _content_text(content)
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "output_text" if assistant else "input_text", "text": text})
            continue
        if not assistant and ptype in {"image_url", "input_image"}:
            url = _image_url_from_part(part)
            if url:
                parts.append({
                    "type": "input_image",
                    "image_url": url,
                    "detail": _image_detail_from_part(part),
                })
    return parts or _content_text(content)


def _content_to_responses_parts(content: Any, *, assistant: bool = False) -> list[dict[str, Any]]:
    converted = _content_to_responses(content, assistant=assistant)
    if isinstance(converted, list):
        return converted
    text = _content_text(converted)
    return [{"type": "output_text" if assistant else "input_text", "text": text}]


def _content_to_anthropic(content: Any) -> Any:
    if not isinstance(content, list):
        return _content_text(content)
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
            continue
        if ptype in {"image_url", "input_image"}:
            url = _image_url_from_part(part)
            if not url:
                continue
            media_type, data = _data_url_parts(url)
            if media_type and data:
                parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                })
    return parts or _content_text(content)


def _messages_to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        next_msg = dict(msg)
        next_msg["content"] = _content_to_anthropic(next_msg.get("content"))
        converted.append(next_msg)
    return converted


def _content_to_gemini_parts(content: Any) -> list[Any]:
    if gemini_types is None:
        return []
    if not isinstance(content, list):
        return [gemini_types.Part(text=_content_text(content))]

    parts: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append(gemini_types.Part(text=text))
            continue
        if ptype in {"image_url", "input_image"}:
            url = _image_url_from_part(part)
            if not url:
                continue
            media_type, data = _data_url_parts(url)
            if not (media_type and data):
                continue
            raw = base64.b64decode(data)
            from_bytes = getattr(gemini_types.Part, "from_bytes", None)
            if callable(from_bytes):
                parts.append(from_bytes(data=raw, mime_type=media_type))
            else:
                parts.append(gemini_types.Part(
                    inline_data=gemini_types.Blob(mime_type=media_type, data=raw)
                ))
    return parts or [gemini_types.Part(text=_content_text(content))]


def _openai_tool_calls(raw_calls: list[Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for call in raw_calls or []:
        fn = getattr(call, "function", None) or {}
        name = getattr(fn, "name", None) or fn.get("name")
        raw_args = getattr(fn, "arguments", None) or fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            logger.debug("Failed to parse tool arguments: %r", str(raw_args)[:200])
            args = {}
        tool_calls.append({"id": getattr(call, "id", None), "name": name, "arguments": args})
    return tool_calls


def _anthropic_tools(openai_tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not openai_tools:
        return []
    tools: list[dict[str, Any]] = []
    for tool in openai_tools:
        fn = tool.get("function", {})
        tools.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return tools


def _gemini_tools(openai_tools: list[dict[str, Any]] | None) -> list[Any]:
    """
    Convert OpenAI-style tools to google-genai tool declarations.

    Returns a list of google.genai.types.Tool instances (typed as Any here to
    avoid importing google-genai at type-check time).
    """
    if not openai_tools or gemini_types is None:
        return []
    decls: list[Any] = []
    for tool in openai_tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        decls.append(
            gemini_types.FunctionDeclaration(
                name=name,
                description=fn.get("description", ""),
                parameters_json_schema=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    if not decls:
        return []
    return [gemini_types.Tool(function_declarations=decls)]


def _messages_to_gemini_contents(messages: list[dict[str, Any]]) -> list[Any]:
    """
    Convert OpenAI-style message list into google-genai `contents`.

    Handles:
    - user text messages
    - assistant text messages
    - assistant tool_calls (OpenAI format) -> functionCall parts
    - tool result messages (role=tool) -> functionResponse parts
    """
    if gemini_types is None:
        return []

    # Map OpenAI tool call id -> function name so we can attach tool outputs.
    call_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            call_id = str(tc.get("id") or "")
            fn = tc.get("function") or {}
            name = str((fn.get("name") if isinstance(fn, dict) else "") or "")
            if call_id and name:
                call_id_to_name[call_id] = name

    contents: list[Any] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "user":
            contents.append(
                gemini_types.Content(
                    role="user",
                    parts=_content_to_gemini_parts(content),
                )
            )
            continue

        if role == "assistant":
            parts: list[Any] = []
            if content:
                parts.append(gemini_types.Part(text=_content_text(content)))

            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name") or "")
                if not name:
                    continue
                call_id = tc.get("id")
                raw_args: Any = fn.get("arguments", "{}")
                args: dict[str, Any] = {}
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except Exception:
                        logger.debug("Failed to parse tool arguments: %r", raw_args[:200])
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                parts.append(
                    gemini_types.Part(
                        function_call=gemini_types.FunctionCall(
                            id=str(call_id) if call_id else None,
                            name=name,
                            args=args,
                        )
                    )
                )

            if parts:
                contents.append(gemini_types.Content(role="model", parts=parts))
            continue

        if role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            fn_name = call_id_to_name.get(call_id) or ""
            if fn_name:
                contents.append(
                    gemini_types.Content(
                        role="user",
                        parts=[
                            gemini_types.Part(
                                function_response=gemini_types.FunctionResponse(
                                    id=call_id or None,
                                    name=fn_name,
                                    response={"content": str(content)},
                                )
                            )
                        ],
                    )
                )
            else:
                # Fallback: if we can't find a matching function name, inject as user text.
                contents.append(
                    gemini_types.Content(
                        role="user",
                        parts=[gemini_types.Part(text=_content_text(content))],
                    )
                )
            continue

        # Ignore other roles (system is handled separately).

    return contents


# ---------------------------------------------------------------------------
# Responses API implementation
# ---------------------------------------------------------------------------


async def _responses_completion(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Non-streaming completion via the Responses API."""
    instructions, input_items = _messages_to_responses_input(messages)
    responses_tools = _tools_to_responses(tools)

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if instructions:
        payload["instructions"] = instructions
    payload["input"] = input_items or ""
    if responses_tools:
        payload["tools"] = responses_tools
        payload["tool_choice"] = "auto"
    if response_format:
        fmt_type = response_format.get("type", "text")
        if fmt_type == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}
        elif fmt_type == "json_schema":
            payload["text"] = {"format": response_format}

    response = await client.responses.create(**payload)
    return _extract_responses_result(response)


async def _responses_stream_completion(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
    on_text_delta: Any | None = None,
) -> dict[str, Any]:
    """Streaming completion via the Responses API."""
    instructions, input_items = _messages_to_responses_input(messages)
    responses_tools = _tools_to_responses(tools)

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if instructions:
        payload["instructions"] = instructions
    payload["input"] = input_items or ""
    if responses_tools:
        payload["tools"] = responses_tools
        payload["tool_choice"] = "auto"

    content_parts: list[str] = []
    # Accumulate tool calls: item_id -> {call_id, name, arguments_parts}
    tc_accum: dict[str, dict[str, Any]] = {}

    async with client.responses.stream(**payload) as stream:
        async for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta":
                text = getattr(event, "delta", "")
                content_parts.append(text)
                if on_text_delta:
                    import asyncio
                    result = on_text_delta(text)
                    if asyncio.iscoroutine(result):
                        await result

            elif event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if item and getattr(item, "type", None) == "function_call":
                    raw_args = getattr(item, "arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        logger.debug("Failed to parse tool arguments: %r", str(raw_args)[:200])
                        args = {}
                    tc_accum[getattr(item, "id", "")] = {
                        "call_id": getattr(item, "call_id", None),
                        "name": getattr(item, "name", ""),
                        "arguments": args,
                    }

    tool_calls: list[dict[str, Any]] = []
    for tc in tc_accum.values():
        tool_calls.append({
            "id": tc["call_id"],
            "name": tc["name"],
            "arguments": tc["arguments"],
        })

    return {"content": "".join(content_parts), "tool_calls": tool_calls, "raw": None}


async def chat_completion(
    *,
    provider: str,
    model: str,
    endpoint: str | None,
    api_key: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1200,
    response_format: dict[str, Any] | None = None,
    auth_mode: str | None = None,
) -> dict[str, Any]:
    provider = normalize_provider(provider)
    endpoint = normalize_endpoint(provider, endpoint)

    if provider == "gemini":
        if genai is None or gemini_types is None:
            raise RuntimeError("google-genai package is required for Gemini provider (pip install google-genai).")
        if not api_key:
            raise RuntimeError("Gemini API key is required. Set GEMINI_API_KEY or configure api_key_env.")

        client = genai.Client(api_key=api_key)
        # Gemini 2.5+ implicitly caches repeated prefixes. SystemPrompt emits
        # the stable half first and the volatile half second; keep that order
        # intact when flattening the API's single system instruction.
        system_prompt, rest = _extract_system_prompt(messages)
        contents = _messages_to_gemini_contents(rest)
        gemini_tools = _gemini_tools(tools)

        tool_config = None
        if gemini_tools:
            tool_config = gemini_types.ToolConfig(
                function_calling_config=gemini_types.FunctionCallingConfig(
                    mode=gemini_types.FunctionCallingConfigMode.AUTO,
                )
            )

        config = gemini_types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=gemini_tools or None,
            tool_config=tool_config,
        )

        async def _do_gemini_completion():
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            content = getattr(response, "text", "") or ""
            tool_calls: list[dict[str, Any]] = []
            for call in getattr(response, "function_calls", None) or []:
                tool_calls.append({
                    "id": getattr(call, "id", None),
                    "name": getattr(call, "name", "") or "",
                    "arguments": getattr(call, "args", None) or {},
                })
            return {"content": content, "tool_calls": tool_calls, "raw": response}

        return await _retry_on_transient(_do_gemini_completion)

    if provider in OPENAI_COMPATIBLE:
        default_headers = None
        if provider == "github-copilot":
            from core.auth.github_copilot import COPILOT_REQUEST_HEADERS
            default_headers = COPILOT_REQUEST_HEADERS
        client = _get_openai_client(api_key, endpoint, provider, default_headers)

        # Try Responses API first, fall back to Chat Completions
        if _should_try_responses(endpoint):
            try:
                result = await _retry_on_transient(lambda: _responses_completion(
                    client, model, messages, tools, temperature, max_tokens, response_format,
                ))
                _cache_responses_support(endpoint, True)
                return result
            except Exception as exc:
                if _is_responses_unsupported_error(exc):
                    _cache_responses_support(endpoint, False)
                else:
                    raise

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {**msg, "content": _content_to_openai_chat(msg.get("content"))}
                for msg in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format

        async def _do_chat_completion():
            response = await client.chat.completions.create(**payload)
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = _openai_tool_calls(message.tool_calls or [])
            return {"content": content, "tool_calls": tool_calls, "raw": response}

        return await _retry_on_transient(_do_chat_completion)

    if provider == "openai-codex":
        # Wrapped like every other provider branch: a 429/5xx here was the
        # one unretried path (it killed live turns during ingestion storms).
        return await _retry_on_transient(lambda: _codex_responses_completion(
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            on_text_delta=None,
        ))

    if provider == "anthropic":
        if auth_mode == "setup-token":
            from core.providers.anthropic_http import anthropic_http_completion
            system_prompt, rest = _extract_system_prompt(messages)
            rest = _messages_to_anthropic_messages(rest)
            return await _retry_on_transient(lambda: anthropic_http_completion(
                endpoint=endpoint or "https://api.anthropic.com",
                api_key=api_key or "",
                model=model,
                messages=rest,
                tools=tools,
                auth_mode="setup-token",
                max_tokens=max_tokens,
                system_prompt=system_prompt or None,
            ))
        if anthropic is None:
            raise RuntimeError("anthropic package is required for Anthropic provider.")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        system_parts, rest = _extract_system_parts(messages)
        system_blocks = _anthropic_system_blocks(system_parts)
        rest = _messages_to_anthropic_messages(rest)
        anthropic_tools = _anthropic_tools(tools)

        async def _do_anthropic_completion():
            response = await client.messages.create(
                model=model,
                system=system_blocks,
                messages=rest,
                tools=anthropic_tools or None,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in response.content or []:
                if block.type == "text":
                    text_parts.append(block.text)
                if block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})
            return {"content": "".join(text_parts), "tool_calls": tool_calls, "raw": response}

        return await _retry_on_transient(_do_anthropic_completion)

    if provider == "minimax-portal":
        from core.providers.anthropic_http import anthropic_http_completion
        system_prompt, rest = _extract_system_prompt(messages)
        rest = _messages_to_anthropic_messages(rest)
        return await anthropic_http_completion(
            endpoint=endpoint or "https://api.minimax.io/anthropic",
            api_key=api_key or "",
            model=model,
            messages=rest,
            tools=tools,
            auth_mode="api-key",
            max_tokens=max_tokens,
            system_prompt=system_prompt or None,
        )

    if provider in {"google-gemini-cli", "google-antigravity"}:
        from core.providers.google_code_assist import google_code_assist_completion
        _api_key_data = json.loads(api_key) if api_key else {}
        access_token = _api_key_data.get("token", api_key or "")
        project_id = _api_key_data.get("projectId", "")
        return await google_code_assist_completion(
            endpoint=endpoint or "https://cloudcode-pa.googleapis.com",
            access_token=access_token,
            project_id=project_id,
            model=model,
            messages=messages,
            tools=tools,
            is_antigravity=(provider == "google-antigravity"),
        )

    raise ValueError(f"Unsupported provider: {provider}")


async def stream_chat_completion(
    *,
    provider: str,
    model: str,
    endpoint: str | None,
    api_key: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1200,
    on_text_delta: Any | None = None,
    auth_mode: str | None = None,
) -> dict[str, Any]:
    """
    Streaming chat completion that supports tools.

    Accumulates the full response while optionally calling ``on_text_delta(text)``
    for each token. Returns the same shape as ``chat_completion()``:
    ``{content, tool_calls, raw}``.

    ``on_text_delta`` can be a sync or async callable accepting a single str
    argument. It's called for each text token as it arrives.
    """
    provider = normalize_provider(provider)
    endpoint = normalize_endpoint(provider, endpoint)

    if provider == "gemini":
        if genai is None or gemini_types is None:
            raise RuntimeError("google-genai package is required for Gemini provider (pip install google-genai).")
        if not api_key:
            raise RuntimeError("Gemini API key is required. Set GEMINI_API_KEY or configure api_key_env.")

        client = genai.Client(api_key=api_key)
        # Preserve the stable-prefix/volatile-tail order for Gemini's implicit
        # caching. The complete instruction still reaches the model.
        system_prompt, rest = _extract_system_prompt(messages)
        contents = _messages_to_gemini_contents(rest)
        gemini_tools = _gemini_tools(tools)

        tool_config = None
        if gemini_tools:
            tool_config = gemini_types.ToolConfig(
                function_calling_config=gemini_types.FunctionCallingConfig(
                    mode=gemini_types.FunctionCallingConfigMode.AUTO,
                )
            )

        config = gemini_types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=gemini_tools or None,
            tool_config=tool_config,
        )

        async def _do_gemini_stream():
            # Track emitted text so we can compute deltas if the stream is cumulative.
            emitted: str = ""
            calls_by_id: dict[str, dict[str, Any]] = {}
            last_chunk: Any | None = None

            async for chunk in client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                last_chunk = chunk
                text = getattr(chunk, "text", "") or ""
                if text:
                    if text.startswith(emitted):
                        delta = text[len(emitted) :]
                        emitted = text
                    else:
                        delta = text
                        emitted += text
                    if delta and on_text_delta:
                        import asyncio

                        result = on_text_delta(delta)
                        if asyncio.iscoroutine(result):
                            await result

                for call in getattr(chunk, "function_calls", None) or []:
                    call_id = getattr(call, "id", None) or ""
                    calls_by_id[str(call_id)] = {
                        "id": call_id or None,
                        "name": getattr(call, "name", "") or "",
                        "arguments": getattr(call, "args", None) or {},
                    }

            tool_calls = [v for k, v in calls_by_id.items() if k]
            # Gemini reports cached_content_token_count on the final streamed
            # chunk. Preserve it so record_llm_usage can prove cache hits.
            return {"content": emitted, "tool_calls": tool_calls, "raw": last_chunk}

        return await _retry_on_transient(_do_gemini_stream)

    if provider in OPENAI_COMPATIBLE:
        default_headers = None
        if provider == "github-copilot":
            from core.auth.github_copilot import COPILOT_REQUEST_HEADERS
            default_headers = COPILOT_REQUEST_HEADERS
        client = _get_openai_client(api_key, endpoint, provider, default_headers)

        # Try Responses API first, fall back to Chat Completions
        if _should_try_responses(endpoint):
            try:
                result = await _retry_on_transient(lambda: _responses_stream_completion(
                    client, model, messages, tools, temperature, max_tokens, on_text_delta,
                ))
                _cache_responses_support(endpoint, True)
                return result
            except Exception as exc:
                if _is_responses_unsupported_error(exc):
                    _cache_responses_support(endpoint, False)
                else:
                    raise

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {**msg, "content": _content_to_openai_chat(msg.get("content"))}
                for msg in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async def _do_stream_completion():
            response = await client.chat.completions.create(**payload)

            content_parts: list[str] = []
            # Accumulate tool calls: index -> {id, name, arguments_parts}
            tc_accum: dict[int, dict[str, Any]] = {}

            async for event in response:
                delta = event.choices[0].delta
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if on_text_delta:
                        import asyncio
                        result = on_text_delta(delta.content)
                        if asyncio.iscoroutine(result):
                            await result
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_accum:
                            tc_accum[idx] = {
                                "id": getattr(tc_delta, "id", None),
                                "name": None,
                                "arguments_parts": [],
                            }
                        if tc_delta.id:
                            tc_accum[idx]["id"] = tc_delta.id
                        fn = getattr(tc_delta, "function", None)
                        if fn:
                            if getattr(fn, "name", None):
                                tc_accum[idx]["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                tc_accum[idx]["arguments_parts"].append(fn.arguments)

            # Build final tool calls
            tool_calls: list[dict[str, Any]] = []
            for idx in sorted(tc_accum.keys()):
                tc = tc_accum[idx]
                raw_args = "".join(tc["arguments_parts"])
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    logger.debug("Failed to parse tool arguments: %r", raw_args[:200])
                    args = {}
                tool_calls.append({"id": tc["id"], "name": tc["name"], "arguments": args})
            return {"content": "".join(content_parts), "tool_calls": tool_calls, "raw": None}

        return await _retry_on_transient(_do_stream_completion)

    if provider == "openai-codex":
        # A 429/5xx fails before any token reaches the caller and retries
        # freely; once tokens have streamed, retry is refused so the consumer
        # never sees the same text twice.
        delivered = False

        def _marking_delta(text: str):
            nonlocal delivered
            delivered = True
            if on_text_delta is not None:
                return on_text_delta(text)
            return None

        return await _retry_on_transient(
            lambda: _codex_responses_completion(
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                on_text_delta=_marking_delta if on_text_delta is not None else None,
            ),
            should_retry=lambda: not delivered,
        )

    if provider == "anthropic":
        if auth_mode == "setup-token":
            from core.providers.anthropic_http import stream_anthropic_http_completion
            system_prompt, rest = _extract_system_prompt(messages)
            rest = _messages_to_anthropic_messages(rest)
            return await _retry_on_transient(lambda: stream_anthropic_http_completion(
                endpoint=endpoint or "https://api.anthropic.com",
                api_key=api_key or "",
                model=model,
                messages=rest,
                tools=tools,
                auth_mode="setup-token",
                max_tokens=max_tokens,
                system_prompt=system_prompt or None,
                on_text_delta=on_text_delta,
            ))
        if anthropic is None:
            raise RuntimeError("anthropic package is required for Anthropic provider.")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        system_prompt, rest = _extract_system_prompt(messages)
        rest = _messages_to_anthropic_messages(rest)
        anthropic_tools = _anthropic_tools(tools)
        sdk_kwargs: dict[str, Any] = {
            "model": model,
            "messages": rest,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            sdk_kwargs["system"] = system_prompt
        if anthropic_tools:
            sdk_kwargs["tools"] = anthropic_tools

        async def _do_anthropic_stream():
            async with client.messages.stream(**sdk_kwargs) as stream:
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                current_tool: dict[str, Any] | None = None

                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "tool_use":
                                current_tool = {"id": block.id, "name": block.name, "arguments_json": ""}
                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                text_parts.append(delta.text)
                                if on_text_delta:
                                    import asyncio
                                    result = on_text_delta(delta.text)
                                    if asyncio.iscoroutine(result):
                                        await result
                            elif delta.type == "input_json_delta" and current_tool is not None:
                                current_tool["arguments_json"] += delta.partial_json
                        elif event.type == "content_block_stop":
                            if current_tool is not None:
                                raw_args = current_tool["arguments_json"]
                                try:
                                    args = json.loads(raw_args) if raw_args else {}
                                except Exception:
                                    logger.debug("Failed to parse tool arguments: %r", raw_args[:200])
                                    args = {}
                                tool_calls.append({
                                    "id": current_tool["id"],
                                    "name": current_tool["name"],
                                    "arguments": args,
                                })
                                current_tool = None

                return {"content": "".join(text_parts), "tool_calls": tool_calls, "raw": None}

        return await _retry_on_transient(_do_anthropic_stream)

    if provider == "minimax-portal":
        from core.providers.anthropic_http import stream_anthropic_http_completion
        system_prompt, rest = _extract_system_prompt(messages)
        rest = _messages_to_anthropic_messages(rest)
        return await stream_anthropic_http_completion(
            endpoint=endpoint or "https://api.minimax.io/anthropic",
            api_key=api_key or "",
            model=model,
            messages=rest,
            tools=tools,
            auth_mode="api-key",
            max_tokens=max_tokens,
            system_prompt=system_prompt or None,
            on_text_delta=on_text_delta,
        )

    if provider in {"google-gemini-cli", "google-antigravity"}:
        from core.providers.google_code_assist import stream_google_code_assist_completion
        _api_key_data = json.loads(api_key) if api_key else {}
        access_token = _api_key_data.get("token", api_key or "")
        project_id = _api_key_data.get("projectId", "")
        return await stream_google_code_assist_completion(
            endpoint=endpoint or "https://cloudcode-pa.googleapis.com",
            access_token=access_token,
            project_id=project_id,
            model=model,
            messages=messages,
            tools=tools,
            is_antigravity=(provider == "google-antigravity"),
            on_text_delta=on_text_delta,
        )

    raise ValueError(f"Unsupported provider: {provider}")


async def stream_text_completion(
    *,
    provider: str,
    model: str,
    endpoint: str | None,
    api_key: str | None,
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 1400,
) -> AsyncIterator[str]:
    provider = normalize_provider(provider)
    endpoint = normalize_endpoint(provider, endpoint)

    if provider in OPENAI_COMPATIBLE:
        client = _get_openai_client(api_key, endpoint, provider)

        async def _do_text_stream():
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for event in response:
                delta = event.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        # Note: We can't wrap a generator with retry logic the same way,
        # but we retry the initial request
        async def _start_stream():
            return _do_text_stream()

        stream = await _retry_on_transient(_start_stream)
        async for chunk in stream:
            yield chunk
        return

    if provider == "gemini":
        if genai is None or gemini_types is None:
            raise RuntimeError("google-genai package is required for Gemini provider (pip install google-genai).")
        if not api_key:
            raise RuntimeError("Gemini API key is required. Set GEMINI_API_KEY or configure api_key_env.")

        client = genai.Client(api_key=api_key)
        # Preserve the stable-prefix/volatile-tail order for Gemini's implicit
        # caching. The complete instruction still reaches the model.
        system_prompt, rest = _extract_system_prompt(messages)
        contents = _messages_to_gemini_contents(rest)
        config = gemini_types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        emitted = ""
        async for chunk in client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            text = getattr(chunk, "text", "") or ""
            if not text:
                continue
            if text.startswith(emitted):
                delta = text[len(emitted) :]
                emitted = text
            else:
                delta = text
                emitted += text
            if delta:
                yield delta
        return

    if provider == "anthropic":
        if anthropic is None:
            raise RuntimeError("anthropic package is required for Anthropic provider.")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        system_prompt, rest = _extract_system_prompt(messages)
        rest = _messages_to_anthropic_messages(rest)
        async with client.messages.stream(
            model=model,
            system=system_prompt or None,
            messages=rest,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text
        return

    if provider == "openai-codex":
        if not api_key:
            raise RuntimeError("Missing OpenAI Codex OAuth access token (api_key).")
        account_id = _extract_codex_account_id(api_key)
        url = _resolve_codex_url(endpoint)
        instructions, input_items = _messages_to_codex_responses_input(messages)
        payload: dict[str, Any] = {
            "model": model,
            "store": False,
            "stream": True,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "temperature": temperature,
        }
        if instructions:
            payload["instructions"] = instructions
        _ = max_tokens  # max token field support is inconsistent; omit unless proven.
        headers = {
            "Authorization": f"Bearer {api_key}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "pi",
            "accept": "text/event-stream",
            "content-type": "application/json",
            "user-agent": f"hexis (python; {os.uname().sysname if hasattr(os, 'uname') else 'unknown'})",
        }
        timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
        try:
            async for event in iter_sse_json_events(
                "openai-codex",
                "POST",
                url,
                headers=headers,
                json_body=payload,
                timeout=timeout,
                attempts=3,
                max_delay=_LLM_RETRY_MAX_WAIT,
                retry_unsafe_methods=True,
            ):
                event_type = event.get("type")
                if event_type == "response.output_text.delta":
                    delta = event.get("delta", "")
                    if isinstance(delta, str) and delta:
                        yield delta
                elif event_type in {"response.done", "response.completed"}:
                    return
                elif event_type in {"error", "response.failed"}:
                    message = event.get("message") or "Codex request failed"
                    raise RuntimeError(str(message))
        except IntegrationHttpError as exc:
            raise RuntimeError(str(exc)) from exc
        return

    if provider == "minimax-portal":
        from core.providers.anthropic_http import stream_anthropic_http_completion
        system_prompt, rest = _extract_system_prompt(messages)
        rest = _messages_to_anthropic_messages(rest)
        result = await stream_anthropic_http_completion(
            endpoint=endpoint or "https://api.minimax.io/anthropic",
            api_key=api_key or "",
            model=model,
            messages=rest,
            tools=None,
            auth_mode="api-key",
            max_tokens=max_tokens,
            system_prompt=system_prompt or None,
        )
        if result["content"]:
            yield result["content"]
        return

    if provider in {"google-gemini-cli", "google-antigravity"}:
        from core.providers.google_code_assist import stream_google_code_assist_completion
        _api_key_data = json.loads(api_key) if api_key else {}
        access_token = _api_key_data.get("token", api_key or "")
        project_id = _api_key_data.get("projectId", "")
        result = await stream_google_code_assist_completion(
            endpoint=endpoint or "https://cloudcode-pa.googleapis.com",
            access_token=access_token,
            project_id=project_id,
            model=model,
            messages=messages,
            tools=None,
            is_antigravity=(provider == "google-antigravity"),
        )
        if result["content"]:
            yield result["content"]
        return

    raise ValueError(f"Unsupported provider: {provider}")
