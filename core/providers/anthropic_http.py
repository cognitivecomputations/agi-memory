"""HTTP-based Anthropic Messages client.

Supports two auth modes:
- ``api-key``: Standard ``x-api-key`` header (used by MiniMax Portal).
- ``setup-token``: Bearer auth with Claude Code headers (used by Anthropic setup-token).

The official ``anthropic`` Python SDK does not support Bearer / setup-token auth,
so this module implements the Messages API via raw ``httpx`` requests.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from core.integration_reliability import IntegrationHttpError, iter_sse_json_events

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANTHROPIC_VERSION = "2023-06-01"
_ANTHROPIC_BETA = "claude-code-20250219,oauth-2025-04-20"
_CLAUDE_CODE_USER_AGENT = "claude-cli/2.1.2 (external, cli)"
# Must be verbatim: Anthropic's subscription (OAuth) endpoint validates the
# Claude-Code identity preamble. hermes-agent and openclaw both send this exact
# string; a different wording can 401.
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_headers(api_key: str, auth_mode: str) -> dict[str, str]:
    if auth_mode == "setup-token":
        return {
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": _ANTHROPIC_VERSION,
            "anthropic-beta": _ANTHROPIC_BETA,
            "user-agent": _CLAUDE_CODE_USER_AGENT,
            "x-app": "cli",
            "Content-Type": "application/json",
        }
    # api-key mode (MiniMax, standard Anthropic)
    return {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }


def _oauth_tool_name(name: str) -> str:
    """Normalize a tool name for the OAuth/subscription path.

    Anthropic's subscription billing classifier flags single-underscore
    ``mcp_`` tool names as third-party apps and 400s the request; Claude Code
    uses a double underscore (``mcp__``). Hexis emits ``mcp_<server>_<tool>``
    (see core/tools/mcp.py), so rewrite those. Matches hermes-agent.
    """
    if name.startswith("mcp_") and not name.startswith("mcp__"):
        return "mcp__" + name[len("mcp_"):]
    return name


def _tool_name_restore_map(
    tools: list[dict[str, Any]] | None, auth_mode: str
) -> dict[str, str]:
    """Map wire (on-the-wire) tool names back to originals for the OAuth path."""
    if auth_mode != "setup-token" or not tools:
        return {}
    restore: dict[str, str] = {}
    for tool in tools:
        orig = (tool.get("function") or {}).get("name", "")
        wire = _oauth_tool_name(orig)
        if wire != orig:
            restore[wire] = orig
    return restore


def _convert_tools(
    tools: list[dict[str, Any]] | None, auth_mode: str = "api-key"
) -> list[dict[str, Any]]:
    """Convert OpenAI-format tools to Anthropic Messages tool format."""
    if not tools:
        return []
    result: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if auth_mode == "setup-token":
            name = _oauth_tool_name(name)
        result.append({
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return result


def _build_system_prompt(
    system_prompt: str | list[str] | None, auth_mode: str
) -> str | list[dict[str, Any]] | None:
    """Assemble `system`, with a cache breakpoint when the parts are split.

    A list means the caller separated the stable prefix from this turn's
    volatile tail (see services.agent.SystemPrompt). Anthropic caches up to and
    including the block marked with `cache_control`, so the marker goes on the
    last stable part. A bare string has no boundary to exploit and is passed
    through unchanged.
    """
    split = isinstance(system_prompt, list)
    parts: list[str] = (
        [p for p in system_prompt if p and p.strip()]
        if split
        else ([system_prompt] if system_prompt else [])
    )

    if auth_mode == "setup-token":
        # The identity preamble is the most stable text there is; it leads.
        parts = [_CLAUDE_CODE_IDENTITY, *parts]

    if not parts:
        return None
    # Only reshape the wire format when the caller actually split the prompt.
    # A bare string has no cacheable boundary to exploit, and this is a
    # validated auth path — Anthropic checks the identity preamble on the OAuth
    # flow, so it keeps the exact shape it has always had.
    if not split or len(parts) == 1:
        return "\n\n".join(parts)
    blocks: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        block: dict[str, Any] = {"type": "text", "text": part}
        if index == len(parts) - 2:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks


def _build_request_body(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    auth_mode: str,
    max_tokens: int,
    system_prompt: str | list[str] | None,
    stream: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    sys = _build_system_prompt(system_prompt, auth_mode)
    if sys:
        body["system"] = sys
    anthropic_tools = _convert_tools(tools, auth_mode)
    if anthropic_tools:
        body["tools"] = anthropic_tools
    if stream:
        body["stream"] = True
    return body


def _parse_response(
    data: dict[str, Any], restore_map: dict[str, str] | None = None
) -> dict[str, Any]:
    """Parse a non-streaming Anthropic Messages response into {content, tool_calls, raw}."""
    restore = restore_map or {}
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "")
            tool_calls.append({
                "id": block.get("id") or str(uuid.uuid4()),
                "name": restore.get(name, name),
                "arguments": block.get("input") or {},
            })
    return {"content": "".join(text_parts), "tool_calls": tool_calls, "raw": data}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def anthropic_http_completion(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    auth_mode: str = "api-key",
    max_tokens: int = 16384,
    system_prompt: str | list[str] | None = None,
) -> dict[str, Any]:
    """Non-streaming Anthropic Messages completion via HTTP."""
    url = f"{endpoint.rstrip('/')}/v1/messages"
    headers = _build_headers(api_key, auth_mode)
    body = _build_request_body(
        model, messages, tools,
        auth_mode=auth_mode, max_tokens=max_tokens,
        system_prompt=system_prompt, stream=False,
    )

    restore_map = _tool_name_restore_map(tools, auth_mode)
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=body)
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"Anthropic HTTP request failed: HTTP {resp.status_code}: {resp.text}"
        )
    return _parse_response(resp.json(), restore_map)


async def stream_anthropic_http_completion(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    auth_mode: str = "api-key",
    max_tokens: int = 16384,
    system_prompt: str | list[str] | None = None,
    on_text_delta: Any | None = None,
) -> dict[str, Any]:
    """Streaming Anthropic Messages completion via HTTP SSE.

    Returns the same shape as ``anthropic_http_completion()``.
    Optionally calls ``on_text_delta(text)`` for each text token.
    """
    import asyncio

    url = f"{endpoint.rstrip('/')}/v1/messages"
    headers = _build_headers(api_key, auth_mode)
    body = _build_request_body(
        model, messages, tools,
        auth_mode=auth_mode, max_tokens=max_tokens,
        system_prompt=system_prompt, stream=True,
    )

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    current_tool: dict[str, Any] | None = None
    restore_map = _tool_name_restore_map(tools, auth_mode)

    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    try:
        async for event in iter_sse_json_events(
            "anthropic-http",
            "POST",
            url,
            headers=headers,
            json_body=body,
            timeout=timeout,
            attempts=3,
            max_delay=60.0,
            retry_unsafe_methods=True,
        ):
            event_type = event.get("type", "")

            if event_type == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    current_tool = {
                        "id": block.get("id") or str(uuid.uuid4()),
                        "name": block.get("name", ""),
                        "arguments_json": "",
                    }

            elif event_type == "content_block_delta":
                delta = event.get("delta") or {}
                dtype = delta.get("type", "")
                if dtype == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        text_parts.append(text)
                        if on_text_delta:
                            result = on_text_delta(text)
                            if asyncio.iscoroutine(result):
                                await result
                elif dtype == "input_json_delta" and current_tool is not None:
                    current_tool["arguments_json"] += delta.get("partial_json", "")

            elif event_type == "content_block_stop":
                if current_tool is not None:
                    raw_args = current_tool["arguments_json"]
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except Exception:
                        args = {}
                    name = current_tool["name"]
                    tool_calls.append({
                        "id": current_tool["id"],
                        "name": restore_map.get(name, name),
                        "arguments": args,
                    })
                    current_tool = None

            elif event_type == "error":
                msg = (event.get("error") or {}).get("message", "Anthropic streaming error")
                raise RuntimeError(msg)
    except IntegrationHttpError as exc:
        raise RuntimeError(str(exc)) from exc

    return {"content": "".join(text_parts), "tool_calls": tool_calls, "raw": None}
