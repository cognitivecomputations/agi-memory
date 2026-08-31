"""
Hexis API Server

FastAPI app that wraps the canonical AgentLoop for chat, exposing SSE
streaming in the same event format the Next.js frontend already consumes.

Endpoints:
    POST /api/chat  — SSE streaming chat via AgentLoop.stream()
    GET  /v1/models — Active chat model in OpenAI-compatible form
    POST /v1/chat/completions — OpenAI-compatible buffered/streaming chat
    GET  /api/integrations/status — Connector status and runtime setup hints
    GET  /api/responsibilities — Ambient responsibility status and controls
    GET  /api/status — Rich agent status
    GET  /health     — Simple health check
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import html
import json
import logging
import os
import time
import uuid
import urllib.parse
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import asyncpg
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from channels.presentation import citations_from_tool_output, presentation_from_text
from core.agent_api import db_dsn_from_env, pool_sizes_from_env
from core.agent_loop import AgentEvent, AgentEventData
from core.auth.ui_flow import AuthFlowError, auth_flow_coordinator
from core.cli_api import status_payload_rich
from core.gateway import EventSource, Gateway
from core.rabbitmq_bridge import RabbitMQBridge
from core.tools import create_default_registry, create_full_registry
from services.chat import resolve_prompt_addenda, stream_chat_events

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App state (set during lifespan)
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None


def _dsn() -> str:
    return db_dsn_from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    dsn = _dsn()
    # Bring the schema up to date on startup (advisory-locked, idempotent, no data loss).
    try:
        from core.agent_api import apply_migrations

        applied = await apply_migrations(dsn)
        if applied:
            logger.info(
                "Applied %d schema migration(s) on startup: %s", len(applied), applied
            )
    except Exception as exc:
        logger.warning("Startup migration check failed (continuing): %s", exc)
    _min, _max = pool_sizes_from_env(2, 10)
    _pool = await asyncpg.create_pool(dsn, min_size=_min, max_size=_max)
    from core.usage import set_usage_pool

    set_usage_pool(_pool)
    try:
        from core.agent_api import record_build_change

        async with _pool.acquire() as conn:
            await record_build_change(conn, "api")
    except Exception:
        logger.debug("build-change journaling failed", exc_info=True)
    logger.info("Hexis API started (pool created)")
    try:
        yield
    finally:
        await auth_flow_coordinator.close()
        try:
            from core.tools.mcp_runtime import MCPRuntime

            await MCPRuntime.instance().shutdown()
        except Exception:
            logger.debug("MCP runtime shutdown failed", exc_info=True)
        if _pool:
            await _pool.close()
            logger.info("Pool closed")


app = FastAPI(title="Hexis API", lifespan=lifespan)

# CORS — allow Next.js dev server and configurable origins
_cors_origins = os.getenv(
    "HEXIS_CORS_ORIGINS", "http://localhost:3477,http://localhost:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Bearer token authentication
_API_KEY = (os.getenv("HEXIS_API_KEY") or "").strip() or None


class _BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        oauth_callback = request.url.path in {
            "/api/integrations/gmail/callback",
            "/api/integrations/spotify/callback",
        } or (
            request.url.path == "/"
            and ("code" in request.query_params or "error" in request.query_params)
        )
        if _API_KEY and request.url.path != "/health" and not oauth_callback:
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != _API_KEY:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)


if _API_KEY:
    app.add_middleware(_BearerAuthMiddleware)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatVisualAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    mime_type: str
    data_url: str
    byte_size: int | None = Field(default=None, ge=0)


class ChatAttachment(BaseModel):
    """A file the user attached to this message.

    Descriptive only — the file's text rides the turn as a prompt addendum
    and its bytes are already preserved. This travels with the turn so the
    stored message remembers what came with it, and the chat can redraw the
    attachment when the conversation is reloaded.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    mime_type: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    kind: str | None = None
    artifact_id: str | None = None
    sensitivity: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    history: list[dict[str, Any]] | None = None
    prompt_addenda: list[str] | None = None
    visual_attachments: list[ChatVisualAttachment] | None = None
    attachments: list[ChatAttachment] | None = None
    # Client-held session identity (#71): pass the session_id from a prior
    # turn's `done` event to keep one conversation as one session; omit and
    # the server mints one (returned in `done`).
    session_id: str | None = None


class InboxReplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: uuid.UUID
    reply: str = Field(min_length=1, max_length=20_000)


class AttachmentIngestRequest(BaseModel):
    """Start ingestion for a file already preserved by POST /api/attachments."""

    model_config = ConfigDict(extra="forbid")

    filename: str | None = None
    title: str | None = None
    mode: str = "fast"
    sensitivity: str | None = None


class SpeechSynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=20_000)


class WebPushKeys(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p256dh: str = Field(min_length=1, max_length=1024)
    auth: str = Field(min_length=1, max_length=1024)


class WebPushSubscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(min_length=1, max_length=4096)
    expirationTime: int | None = Field(default=None, ge=0)
    keys: WebPushKeys
    installed: bool = False
    display_mode: str | None = Field(default=None, max_length=80)


class WebPushRevokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(min_length=1, max_length=4096)


class PwaPresenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1, max_length=200)
    presence: Literal["online", "offline", "idle"]
    display_mode: str | None = Field(default=None, max_length=80)
    visibility: Literal["visible", "hidden"] | None = None


class IngestTextRequest(BaseModel):
    content: str
    title: str | None = None
    mode: str = "fast"
    # Sensitivity marking (#92): "private" keeps the resulting memories out
    # of group-channel recall and default HMX export.
    sensitivity: str | None = None


class IngestUrlRequest(BaseModel):
    url: str
    title: str | None = None
    mode: str = "fast"
    sensitivity: str | None = None


class IntegrationActionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    source_session_id: str | None = None


class ResponsibilityActionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    source_session_id: str | None = None


class OutboundControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "suspend_global", "resume_global", "suspend_entity", "resume_entity"
    ]
    entity: str | None = None
    reason: str | None = None


class NodePairingDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1, max_length=100)
    decision: Literal["approve", "deny"]
    note: str | None = Field(default=None, max_length=1000)


class NodeRevokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=1000)


class UserModelReviewRequest(BaseModel):
    decision: Literal["approve", "reject", "supersede", "restore"]
    note: str | None = None
    actor: str | None = "operator"
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenAIChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[OpenAIChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    stop: Any = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stream_options: dict[str, Any] | None = None
    user: str | None = None


class ConsentLlmConfig(BaseModel):
    provider: str | None = None
    model: str | None = None
    endpoint: str | None = None
    api_key: str | None = None


class InitConsentRequest(BaseModel):
    role: Literal["conscious", "subconscious"] = "conscious"
    llm: ConsentLlmConfig | None = None


class InitAuthStartRequest(BaseModel):
    provider: str
    options: dict[str, str] = Field(default_factory=dict)


class InitAuthCompleteRequest(BaseModel):
    session_id: str
    authorization_input: str


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _openai_sse_data(payload: dict[str, Any] | str) -> str:
    encoded = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, separators=(",", ":"))
    )
    return f"data: {encoded}\n\n"


def _openai_error(
    message: str,
    *,
    status_code: int,
    param: str | None = None,
    code: str | None = None,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        status_code=status_code,
    )


def _consent_trace_value(value: Any) -> Any:
    """Convert provider response objects into JSON-safe diagnostics."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(key): _consent_trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_consent_trace_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _consent_trace_value(model_dump())
        except Exception:
            pass
    return repr(value)[:16000]


async def _record_consent_trace(
    *,
    pool: asyncpg.Pool,
    attempt_id: str,
    provider: str,
    model: str,
    phase: Literal["request", "response"],
    metadata: dict[str, Any],
) -> None:
    from core.usage import record_usage

    await record_usage(
        provider=provider,
        model=model,
        operation=f"consent_{phase}",
        session_key=f"init-consent:{attempt_id}",
        source="init_consent",
        metadata={
            "attempt_id": attempt_id,
            "phase": phase,
            **_consent_trace_value(metadata),
        },
        pool=pool,
    )


async def _active_openai_model() -> dict[str, Any]:
    """Describe the live chat model without resolving or consuming credentials."""

    pool = _pool
    if pool is None:
        raise RuntimeError("Server not ready (no DB pool)")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT key, value, updated_at FROM config "
            "WHERE key IN ('llm.chat', 'llm') "
            "ORDER BY CASE key WHEN 'llm.chat' THEN 0 ELSE 1 END LIMIT 1"
        )

    from core.llm_config import configured_llm_identity

    raw = row["value"] if row else {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    config = raw if isinstance(raw, dict) else {}
    identity = configured_llm_identity(config)
    created = int(row["updated_at"].timestamp()) if row and row["updated_at"] else None
    return {
        "id": identity["model"],
        "provider": identity["provider"],
        "created": created,
        "config_key": str(row["key"]) if row else "environment_default",
    }


def _openai_model_object(model: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": model["id"],
        "object": "model",
        "owned_by": model["provider"],
        "x_hexis_config_key": model["config_key"],
    }
    if model.get("created") is not None:
        payload["created"] = model["created"]
    return payload


def _openai_message_text(message: OpenAIChatMessage, index: int) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if content is None:
        if message.role == "assistant" and not message.tool_calls:
            return ""
        raise ValueError(f"messages[{index}].content must be text")
    if not isinstance(content, list):
        raise ValueError(f"messages[{index}].content must be text or text parts")

    parts: list[str] = []
    for part_index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
            raise ValueError(
                f"messages[{index}].content[{part_index}] uses an unsupported non-text part"
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"messages[{index}].content[{part_index}].text must be a string"
            )
        parts.append(text)
    return "".join(parts)


def _prepare_openai_chat(
    req: OpenAIChatCompletionRequest,
) -> tuple[str, list[dict[str, Any]], int | None]:
    extras = sorted((req.model_extra or {}).keys())
    if extras:
        raise ValueError(f"unsupported request parameter: {extras[0]}")
    if req.n != 1:
        raise ValueError("n values other than 1 are not supported")
    if req.top_p != 1.0:
        raise ValueError("top_p is not supported; omit it or use 1")
    if req.stop is not None:
        raise ValueError("stop is not supported by the agentic chat endpoint")
    if req.presence_penalty != 0.0 or req.frequency_penalty != 0.0:
        raise ValueError("presence_penalty and frequency_penalty are not supported")
    if req.max_tokens is not None and req.max_completion_tokens is not None:
        raise ValueError("set only one of max_tokens or max_completion_tokens")
    if req.stream_options and req.stream_options.get("include_usage"):
        raise ValueError(
            "stream_options.include_usage is unavailable because Hexis cannot "
            "attribute multi-step agent token usage to one completion"
        )

    allowed_roles = {"system", "developer", "user", "assistant"}
    history: list[dict[str, Any]] = []
    for index, message in enumerate(req.messages):
        if message.role not in allowed_roles:
            raise ValueError(
                f"messages[{index}].role {message.role!r} is not supported"
            )
        if message.tool_calls or message.tool_call_id:
            raise ValueError("client-supplied tool call history is not supported")
        if message.model_extra:
            field = sorted(message.model_extra.keys())[0]
            raise ValueError(f"unsupported messages[{index}] parameter: {field}")
        text = _openai_message_text(message, index)
        role = "system" if message.role == "developer" else message.role
        prepared: dict[str, Any] = {"role": role, "content": text}
        if message.name:
            prepared["name"] = message.name
        history.append(prepared)

    if history[-1]["role"] != "user":
        raise ValueError("the final message must have role 'user'")
    user_message = str(history.pop()["content"])
    if not user_message.strip():
        raise ValueError("the final user message must not be empty")
    max_tokens = req.max_completion_tokens or req.max_tokens
    return user_message, history, max_tokens


async def _openai_agent_events(
    *,
    user_message: str,
    history: list[dict[str, Any]],
    session_id: str,
    client_user: str | None,
    max_tokens: int | None,
    temperature: float | None,
) -> AsyncIterator[AgentEventData]:
    pool = _pool
    if pool is None:
        raise RuntimeError("Server not ready (no DB pool)")
    async for event in stream_chat_events(
        user_message=user_message,
        history=history,
        session_id=session_id,
        pool=pool,
        dsn=_dsn(),
        max_tokens=max_tokens,
        temperature=temperature,
        surface="openai_compat",
        gateway_source_id=f"chat:openai:{session_id}",
        gateway_payload={"message": user_message[:500], "client_user": client_user},
    ):
        yield event


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/init/auth/status")
async def init_auth_status(provider: str, validate: bool = False):
    """Return redacted status from Hexis's own credential store."""
    try:
        if validate:
            return await auth_flow_coordinator.validate(provider)
        return auth_flow_coordinator.status(provider)
    except AuthFlowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/init/auth/start")
async def init_auth_start(req: InitAuthStartRequest):
    """Start a browser authorization-code or device-code flow."""
    try:
        return await auth_flow_coordinator.start(req.provider, req.options)
    except AuthFlowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/init/auth/session/{session_id}")
async def init_auth_session(session_id: str):
    """Poll a short-lived browser auth session without returning credentials."""
    try:
        return auth_flow_coordinator.session(session_id)
    except AuthFlowError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/init/auth/complete")
async def init_auth_complete(req: InitAuthCompleteRequest):
    """Complete an authorization-code flow from a pasted code or redirect URL."""
    try:
        return await auth_flow_coordinator.complete(
            req.session_id, req.authorization_input
        )
    except AuthFlowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/init/models/openai-codex")
async def init_openai_codex_models():
    """Return the authenticated workspace catalog minus recent hard rejections."""
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    try:
        from core.auth.openai_codex import list_openai_codex_models

        models = await list_openai_codex_models()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    rows = await pool.fetch("""
        SELECT DISTINCT model
        FROM api_usage
        WHERE provider = 'openai-codex'
          AND source = 'init_consent'
          AND operation = 'consent_response'
          AND created_at > now() - interval '24 hours'
          AND metadata #>> '{response,error}' ILIKE '%Model not found%'
        """)
    unavailable = {str(row["model"]) for row in rows}
    return {
        "models": [model for model in models if model not in unavailable],
        "unavailable_models": sorted(unavailable),
        "source": "openai-codex-account",
    }


@app.get("/health")
async def health():
    checks = {"db": False}
    try:
        if _pool:
            async with _pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            checks["db"] = True
    except Exception:
        pass
    ok = all(checks.values())
    return JSONResponse(
        {"status": "ok" if ok else "degraded", "checks": checks},
        status_code=200 if ok else 503,
    )


def _ui_return_url() -> str:
    return (
        os.getenv("HEXIS_UI_URL")
        or os.getenv("HEXIS_WEB_URL")
        or os.getenv("NEXT_PUBLIC_HEXIS_UI_URL")
        or "http://localhost:3477/chat"
    ).rstrip("/")


def _gmail_callback_page(
    *,
    title: str,
    message: str,
    status: str,
    status_code: int = 200,
    connector_label: str = "Gmail",
) -> HTMLResponse:
    return_url = _ui_return_url()
    status_class = "ok" if status == "connected" else "error"
    body = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      :root {{
        color-scheme: light;
        --ink: #17211d;
        --soft: #64716c;
        --line: #d9e2de;
        --teal: #0f766e;
        --warn: #9f3a17;
        --paper: #fbfcfb;
      }}
      body {{
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background: var(--paper);
        color: var(--ink);
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      main {{
        width: min(560px, calc(100vw - 32px));
        border: 1px solid var(--line);
        border-radius: 8px;
        background: white;
        padding: 28px;
        box-shadow: 0 18px 50px rgba(20, 31, 27, 0.10);
      }}
      .label {{
        display: inline-flex;
        margin-bottom: 14px;
        border-radius: 999px;
        padding: 5px 10px;
        background: #edf7f4;
        color: var(--teal);
        font-size: 13px;
        font-weight: 700;
      }}
      .label.error {{
        background: #fff0ea;
        color: var(--warn);
      }}
      h1 {{
        margin: 0;
        font-size: 28px;
        line-height: 1.2;
      }}
      p {{
        margin: 14px 0 0;
        color: var(--soft);
        line-height: 1.55;
      }}
      a {{
        display: inline-flex;
        margin-top: 22px;
        border-radius: 6px;
        background: var(--ink);
        color: white;
        padding: 10px 14px;
        text-decoration: none;
        font-weight: 700;
      }}
    </style>
  </head>
  <body>
    <main>
      <span class="label {status_class}">{html.escape(status)}</span>
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(message)}</p>
      <p>You can return to Hexis now. The {html.escape(connector_label)} setup panel will show the updated status.</p>
      <a href="{html.escape(return_url)}">Return to Hexis</a>
    </main>
  </body>
</html>"""
    return HTMLResponse(body, status_code=status_code)


async def _handle_gmail_oauth_callback(request: Request) -> HTMLResponse:
    if request.query_params.get("error"):
        detail = (
            request.query_params.get("error_description")
            or request.query_params["error"]
        )
        return _gmail_callback_page(
            title="Gmail was not connected",
            message=f"Google returned: {detail}",
            status="authorization failed",
            status_code=400,
        )

    if not request.query_params.get("code"):
        return _gmail_callback_page(
            title="Gmail connection is incomplete",
            message="Google did not include an authorization code in this callback.",
            status="incomplete",
            status_code=400,
        )

    pool = _pool
    if pool is None:
        return _gmail_callback_page(
            title="Hexis is still starting",
            message="The local Hexis API is not ready yet. Return to Hexis and start Gmail sign-in again.",
            status="not ready",
            status_code=503,
        )

    try:
        from core.auth.google_gmail import GmailOAuthError, complete_gmail_oauth

        completed = await complete_gmail_oauth(
            pool, authorization_response=str(request.url)
        )
    except GmailOAuthError as exc:
        return _gmail_callback_page(
            title="Gmail was not connected",
            message=str(exc),
            status="setup failed",
            status_code=400,
        )
    except Exception as exc:
        logger.exception("Gmail OAuth callback failed")
        return _gmail_callback_page(
            title="Gmail was not connected",
            message=str(exc),
            status="setup failed",
            status_code=500,
        )

    account = completed.display_name or completed.account_key
    return _gmail_callback_page(
        title="Gmail connected",
        message=f"{account} is connected to Hexis with the Gmail permissions you approved.",
        status="connected",
    )


@app.get("/", include_in_schema=False)
async def root(request: Request):
    if "code" in request.query_params or "error" in request.query_params:
        return await _handle_gmail_oauth_callback(request)
    return JSONResponse({"service": "hexis-api", "health": "/health"})


@app.get("/api/integrations/gmail/callback", response_class=HTMLResponse)
async def gmail_oauth_callback(request: Request):
    return await _handle_gmail_oauth_callback(request)


async def _handle_spotify_oauth_callback(request: Request) -> HTMLResponse:
    if request.query_params.get("error"):
        return _gmail_callback_page(
            title="Spotify was not connected",
            message=f"Spotify returned: {request.query_params['error']}",
            status="authorization failed",
            status_code=400,
            connector_label="Spotify",
        )
    if not request.query_params.get("code"):
        return _gmail_callback_page(
            title="Spotify connection is incomplete",
            message="Spotify did not include an authorization code in this callback.",
            status="incomplete",
            status_code=400,
            connector_label="Spotify",
        )
    pool = _pool
    if pool is None:
        return _gmail_callback_page(
            title="Hexis is still starting",
            message="The local Hexis API is not ready. Return to Hexis and start Spotify sign-in again.",
            status="not ready",
            status_code=503,
            connector_label="Spotify",
        )
    try:
        from core.auth.spotify import SpotifyOAuthError, complete_spotify_oauth

        completed = await complete_spotify_oauth(
            pool, authorization_response=str(request.url)
        )
    except SpotifyOAuthError as exc:
        return _gmail_callback_page(
            title="Spotify was not connected",
            message=str(exc),
            status="setup failed",
            status_code=400,
            connector_label="Spotify",
        )
    except Exception as exc:
        logger.exception("Spotify OAuth callback failed")
        return _gmail_callback_page(
            title="Spotify was not connected",
            message=str(exc),
            status="setup failed",
            status_code=500,
            connector_label="Spotify",
        )
    return _gmail_callback_page(
        title="Spotify connected",
        message=(
            f"{completed.display_name or completed.account_key} is connected to Hexis "
            "with the Spotify permissions you approved."
        ),
        status="connected",
        connector_label="Spotify",
    )


@app.get("/api/integrations/spotify/callback", response_class=HTMLResponse)
async def spotify_oauth_callback(request: Request):
    return await _handle_spotify_oauth_callback(request)


@app.get("/api/status")
async def status():
    try:
        payload = await status_payload_rich()
        return JSONResponse(payload)
    except Exception as e:
        logger.error("Status failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/inbox/reply")
async def reply_to_web_inbox(req: InboxReplyRequest):
    """Queue a dashboard reply for the agent's next inbox poll/heartbeat."""
    pool = _pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not ready (no DB pool)")

    reply = req.reply.strip()
    if not reply:
        raise HTTPException(status_code=422, detail="Reply cannot be empty")

    async with pool.acquire() as conn:
        source = await conn.fetchrow(
            """
            SELECT id, outbox_msg_id, kind, intent, message, payload
            FROM web_inbox
            WHERE id = $1
            """,
            req.message_id,
        )
    if source is None:
        raise HTTPException(status_code=404, detail="Outbox message not found")

    source_payload = source["payload"]
    if isinstance(source_payload, str):
        try:
            source_payload = json.loads(source_payload)
        except Exception:
            source_payload = {}
    delivery = (
        source_payload.get("delivery")
        if isinstance(source_payload, dict)
        and isinstance(source_payload.get("delivery"), dict)
        else {}
    )
    question_id = delivery.get("question_id") if isinstance(delivery, dict) else None
    if question_id:
        try:
            parsed_question_id = str(uuid.UUID(str(question_id)))
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(
                status_code=409,
                detail="This inbox question has an invalid correlation id. Send a new chat message with your answer.",
            )
        choice_index = int(reply) if reply.isdigit() else None
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT answer_agent_question($1::uuid, $2, $3, 'web_inbox', 'dashboard')",
                parsed_question_id,
                None if choice_index is not None else reply,
                choice_index,
            )
            result = json.loads(raw) if isinstance(raw, str) else (raw or {})
            if result.get("ok"):
                await conn.fetchval(
                    "SELECT mark_web_inbox_read(ARRAY[$1::uuid])", req.message_id
                )
        if not result.get("ok"):
            raise HTTPException(
                status_code=409,
                detail=result.get("message")
                or "That question is no longer waiting for an answer.",
            )
        return JSONResponse(
            {
                "queued": False,
                "answered": True,
                "question_id": parsed_question_id,
                "status": result.get("status"),
            }
        )

    inbox_message_id = uuid.uuid4()
    content = (
        "The user replied to a message you sent through your outbox.\n\n"
        f"Your earlier message:\n{source['message']}\n\n"
        f"User reply:\n{reply}"
    )
    envelope = {
        "id": str(inbox_message_id),
        "kind": "web_outbox_reply",
        "content": content,
        "reply": reply,
        "source": "web_inbox",
        "reply_to": {
            "web_inbox_id": str(source["id"]),
            "outbox_message_id": source["outbox_msg_id"],
            "kind": source["kind"],
            "intent": source["intent"],
        },
    }

    bridge = RabbitMQBridge(pool)
    await bridge.ensure_ready()
    try:
        await bridge.publish_inbox_payload(envelope)
    except Exception as exc:
        logger.error("Could not queue dashboard reply for %s: %s", req.message_id, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Reply was not queued because Hexis's inbox is unavailable: {exc}",
        ) from exc

    async with pool.acquire() as conn:
        marked_read = int(
            await conn.fetchval(
                "SELECT mark_web_inbox_read(ARRAY[$1]::uuid[])",
                req.message_id,
            )
            or 0
        )

    return {
        "queued": True,
        "inbox_message_id": str(inbox_message_id),
        "reply_to": str(req.message_id),
        "marked_read": marked_read,
        "message": "Reply queued for the next heartbeat.",
    }


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


async def _integration_status_payload(
    pool: asyncpg.Pool,
    connector_id: str | None = None,
) -> dict[str, Any]:
    normalized_id = str(connector_id or "").strip().lower().replace("-", "_") or None
    channel_filter = (
        normalized_id if normalized_id in {"slack", "telegram", "signal"} else None
    )
    include_gmail_config = normalized_id in {None, "gmail"}
    async with pool.acquire() as conn:
        integration_raw = await conn.fetchval(
            "SELECT integration_status($1)", normalized_id
        )
        runtime_raw = await conn.fetchval(
            "SELECT list_channel_adapter_status($1)", channel_filter
        )
        backfill_raw = await conn.fetchval(
            "SELECT get_connector_backfill_status($1, NULL)", normalized_id
        )
        gmail_memory_policy = (
            await conn.fetchval(
                "SELECT COALESCE(get_config_text('integrations.gmail.memory_policy'), 'ask')"
            )
            if include_gmail_config
            else None
        )
        gmail_heartbeat_digest_enabled = (
            await conn.fetchval(
                "SELECT COALESCE(get_config_bool('integrations.gmail.heartbeat_digest_enabled'), FALSE)"
            )
            if include_gmail_config
            else None
        )

    payload = _json_value(integration_raw) or {}
    from core.tools.integrations import enrich_gmail_setup_runtime

    enrich_gmail_setup_runtime(payload)
    if include_gmail_config:
        for connector in payload.get("connectors", []):
            if not isinstance(connector, dict) or connector.get("id") != "gmail":
                continue
            setup_manifest = _json_value(connector.get("setup_manifest")) or {}
            if not isinstance(setup_manifest, dict):
                setup_manifest = {}
            setup_manifest.update(
                {
                    "memory_policy": gmail_memory_policy,
                    "memory_config_key": "integrations.gmail.memory_policy",
                    "heartbeat_digest_enabled": bool(gmail_heartbeat_digest_enabled),
                    "heartbeat_digest_config_key": "integrations.gmail.heartbeat_digest_enabled",
                }
            )
            connector["setup_manifest"] = setup_manifest
    backfill = _json_value(backfill_raw) or {}
    payload["channel_runtime"] = _json_value(runtime_raw) or []
    payload["backfill"] = {
        "jobs": backfill.get("jobs") if isinstance(backfill.get("jobs"), list) else [],
        "cursors": (
            backfill.get("cursors") if isinstance(backfill.get("cursors"), list) else []
        ),
        "item_counts": (
            backfill.get("item_counts")
            if isinstance(backfill.get("item_counts"), list)
            else []
        ),
    }
    payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return payload


@app.get("/api/integrations/status")
async def integration_status(connector_id: str | None = None):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    try:
        return JSONResponse(
            jsonable_encoder(await _integration_status_payload(pool, connector_id))
        )
    except Exception as exc:
        logger.error("Integration status failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


_INTEGRATION_ACTION_TO_TOOL = {
    "setup_status": "integration_setup_status",
    "start_setup": "start_integration_setup",
    "configure_channel": "configure_channel_integration",
    "connect_gmail": "connect_gmail",
    "gmail_setup_status": "gmail_setup_status",
    "complete_gmail": "complete_gmail_connection",
    "revoke_gmail": "revoke_gmail_connection",
    "connect_twitter_x": "connect_twitter_x",
    "complete_twitter_x": "complete_twitter_x_connection",
    "revoke_twitter_x": "revoke_twitter_x_connection",
    "start_gmail_backfill": "start_gmail_backfill",
    "control_gmail_backfill": "control_gmail_backfill",
    "start_connector_backfill": "start_connector_backfill",
    "control_connector_backfill": "control_connector_backfill",
    "verify_channel": "verify_channel_integration",
    "connect_life": "connect_life_integration",
    "connect_spotify": "connect_spotify",
    "complete_spotify": "complete_spotify_connection",
    "revoke_life": "revoke_life_integration",
}


def _integration_action_arguments(
    action: str,
    arguments: dict[str, Any],
    source_session_id: str | None,
) -> dict[str, Any]:
    args = dict(arguments or {})
    if action == "start_setup" and args.get("connector_id") == "gmail":
        raise HTTPException(
            status_code=422,
            detail="Use connect_gmail for Gmail setup.",
        )
    if action in {"start_setup", "configure_channel", "verify_channel"}:
        connector_id = (
            str(args.get("connector_id") or "").strip().lower().replace("-", "_")
        )
        allowed = {"slack", "telegram", "signal"}
        if connector_id not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"{action} supports {', '.join(sorted(allowed))}.",
            )
        args["connector_id"] = connector_id
    if action in {"connect_life", "revoke_life", "setup_status"}:
        connector_id = (
            str(args.get("connector_id") or "").strip().lower().replace("-", "_")
        )
        allowed = {"notion", "spotify", "home_assistant", "weather", "trello"}
        if connector_id not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"{action} supports {', '.join(sorted(allowed))}.",
            )
        if action == "connect_life" and connector_id == "spotify":
            raise HTTPException(
                status_code=422, detail="Use connect_spotify for Spotify setup."
            )
        args["connector_id"] = connector_id
    if action in {
        "start_setup",
        "connect_gmail",
        "connect_twitter_x",
        "start_gmail_backfill",
        "start_connector_backfill",
        "connect_life",
        "connect_spotify",
    }:
        args.setdefault("source_channel", "web")
    if (
        action
        in {
            "start_setup",
            "connect_gmail",
            "connect_twitter_x",
            "start_gmail_backfill",
            "start_connector_backfill",
            "connect_life",
            "connect_spotify",
        }
        and source_session_id
    ):
        args.setdefault("source_session_id", source_session_id)
    return args


def _tool_result_payload(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "output": result.output,
        "display_output": result.display_output,
        "error": result.error,
        "error_type": result.error_type.value if result.error_type else None,
        "energy_spent": result.energy_spent,
        "duration_seconds": result.duration_seconds,
        "metadata": result.metadata,
    }


async def _tee_outbox_to_web_inbox(pool: asyncpg.Pool, messages: list[Any]) -> int:
    if not messages:
        return 0
    async with pool.acquire() as conn:
        enabled = await conn.fetchval(
            "SELECT COALESCE(get_config_bool('channel.web_inbox.enabled'), TRUE)"
        )
        if not bool(enabled):
            return 0
        delivered = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
            delivery_info = payload.get("delivery") or msg.get("delivery")
            if (
                isinstance(delivery_info, dict)
                and delivery_info.get("mode") == "silent"
            ):
                continue
            body = {
                "id": msg.get("message_id") or msg.get("id"),
                "kind": msg.get("kind"),
                "payload": payload,
            }
            if msg.get("delivery") is not None:
                body["delivery"] = msg.get("delivery")
            if msg.get("task_name") is not None:
                body["task_name"] = msg.get("task_name")
            web_id = await conn.fetchval(
                "SELECT web_inbox_deliver($1::jsonb)",
                json.dumps(body, default=str),
            )
            if web_id:
                delivered += 1
        return delivered


@app.post("/api/integrations/action")
async def integration_action(req: IntegrationActionRequest):
    """Execute first-class integration setup controls through Python drivers.

    This is deliberately narrower than a generic web tool-execution endpoint:
    the web UI can invoke only connector setup/control operations whose
    implementation already lives in the tool registry or DB substrate.
    """
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)

    action = req.action.strip().lower()
    action_arguments = dict(req.arguments or {})
    if not action_arguments and req.model_extra:
        action_arguments = {
            key: value
            for key, value in req.model_extra.items()
            if key not in {"action", "arguments", "source_session_id"}
        }

    if action == "revoke_connection":
        connector_id = (
            str(action_arguments.get("connector_id") or "")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if not connector_id:
            raise HTTPException(status_code=422, detail="connector_id is required")
        account_key = action_arguments.get("account_key")
        reason = action_arguments.get("reason") or "revoked from web connections"
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT revoke_integration_connection($1, $2, $3)",
                connector_id,
                account_key,
                reason,
            )
        payload = json.loads(raw) if isinstance(raw, str) else raw
        return JSONResponse(
            {
                "success": True,
                "output": payload,
                "display_output": f"{connector_id} connection revoked.",
                "error": None,
                "error_type": None,
            }
        )

    if action == "save_gmail_client_secret":
        from core.auth.google_gmail import (
            GmailOAuthError,
            save_gmail_client_secret_payload,
        )
        from core.tools.base import ToolContext, ToolExecutionContext

        secret_payload = action_arguments.get("client_secret_json")
        if isinstance(secret_payload, str):
            try:
                secret_payload = json.loads(secret_payload)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Uploaded Google setup file is not valid JSON.",
                ) from exc
        if not isinstance(secret_payload, dict):
            raise HTTPException(
                status_code=422,
                detail="client_secret_json must be the parsed Google setup file JSON object.",
            )

        try:
            saved = save_gmail_client_secret_payload(
                secret_payload, source="web_upload"
            )
        except GmailOAuthError as exc:
            return JSONResponse(
                {
                    "success": False,
                    "output": {
                        "connector_id": "gmail",
                        "status": "needs_client_secret",
                        "client_secret_saved": False,
                    },
                    "display_output": None,
                    "error": str(exc),
                    "error_type": "missing_config",
                },
                status_code=400,
            )

        registry = create_default_registry(pool)
        context = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id=f"web-integration:{uuid.uuid4()}",
            session_id=req.source_session_id or "web-connections",
        )
        status_result = await registry.execute("gmail_setup_status", {}, context)
        status_payload = _tool_result_payload(status_result)
        output = (
            status_payload.get("output")
            if isinstance(status_payload.get("output"), dict)
            else {}
        )
        merged_output = {**output, **saved}
        ui = merged_output.get("ui")
        if isinstance(ui, dict):
            merged_output["ui"] = {
                **ui,
                "status": "client_secret_saved",
                "client_secret_saved": True,
                "next_step": saved["next_step"],
            }
        return JSONResponse(
            jsonable_encoder(
                {
                    "success": True,
                    "output": merged_output,
                    "display_output": "Gmail setup file saved. Start Google sign-in from the setup panel.",
                    "error": None,
                    "error_type": None,
                }
            )
        )

    tool_name = _INTEGRATION_ACTION_TO_TOOL.get(action)
    if not tool_name:
        raise HTTPException(
            status_code=422, detail=f"unknown integration action: {action}"
        )

    from core.tools.base import ToolContext, ToolExecutionContext

    registry = create_default_registry(pool)
    args = _integration_action_arguments(
        action, action_arguments, req.source_session_id
    )
    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id=f"web-integration:{uuid.uuid4()}",
        session_id=req.source_session_id
        or str(args.get("source_session_id") or "web-connections"),
    )
    result = await registry.execute(tool_name, args, context)
    status_code = 200 if result.success else 400
    return JSONResponse(
        jsonable_encoder(_tool_result_payload(result)), status_code=status_code
    )


@app.get("/api/outbound")
async def outbound_ledger(limit: int = 100, entity: str | None = None):
    """Inspectable purpose/cost/delivery ledger and its kill-switch state."""
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    bounded_limit = max(1, min(int(limit or 100), 500))
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT get_outbound_ledger($1::int, $2::text)", bounded_limit, entity
        )
    return JSONResponse(jsonable_encoder(_json_value(raw) or {}))


@app.post("/api/outbound/control")
async def outbound_control(req: OutboundControlRequest):
    """Operator controls can pause sends; they never erase a recipient STOP."""
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    if (
        req.action in {"suspend_entity", "resume_entity"}
        and not str(req.entity or "").strip()
    ):
        raise HTTPException(status_code=422, detail="entity is required")

    async with pool.acquire() as conn:
        if req.action in {"suspend_global", "resume_global"}:
            raw = await conn.fetchval(
                "SELECT set_outbound_global_suspension($1)",
                req.action == "suspend_global",
            )
        else:
            raw = await conn.fetchval(
                "SELECT set_outbound_entity_suspension($1, $2, $3)",
                str(req.entity).strip(),
                req.action == "suspend_entity",
                req.reason,
            )
        ledger_raw = await conn.fetchval("SELECT get_outbound_ledger(100, NULL)")
    return JSONResponse(
        jsonable_encoder(
            {
                "control": _json_value(raw) or {},
                "ledger": _json_value(ledger_raw) or {},
            }
        )
    )


@app.websocket("/api/nodes/connect")
async def node_connect(websocket: WebSocket):
    """Outward-only signed node transport; unknown identities can only request pairing."""
    pool = _pool
    if pool is None:
        await websocket.close(code=1013, reason="Server database is not ready")
        return
    from services.node_gateway import handle_node_websocket

    await handle_node_websocket(websocket, pool)


@app.get("/api/nodes")
async def nodes_status():
    """Approved nodes plus pending pairing decisions for the operator surface."""
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    async with pool.acquire() as conn:
        nodes_raw = await conn.fetchval("SELECT list_hexis_nodes()")
        pending_raw = await conn.fetchval(
            "SELECT list_node_pairing_requests('pending', 50)"
        )
    return JSONResponse(
        jsonable_encoder(
            {
                "nodes": _json_value(nodes_raw) or [],
                "pending_pairings": _json_value(pending_raw) or [],
            }
        )
    )


@app.post("/api/nodes/pairing")
async def node_pairing_decision(req: NodePairingDecisionRequest):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT decide_node_pairing($1, $2, 'dashboard', $3)",
            req.request,
            req.decision,
            req.note,
        )
    result = _json_value(raw) or {}
    status_code = (
        200 if result.get("status") not in {"not_found", "invalid_decision"} else 404
    )
    return JSONResponse(jsonable_encoder(result), status_code=status_code)


@app.post("/api/nodes/revoke")
async def node_revoke(req: NodeRevokeRequest):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT revoke_hexis_node($1, 'dashboard', $2)", req.node_id, req.reason
        )
    result = _json_value(raw) or {}
    return JSONResponse(
        jsonable_encoder(result), status_code=200 if result.get("revoked") else 404
    )


@app.get("/api/responsibilities")
async def responsibilities(status: str | None = None, limit: int = 50):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    limit = max(1, min(int(limit or 50), 200))
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT refresh_ambient_responsibility_blockers()")
        status_raw = await conn.fetchval("SELECT ambient_responsibility_status()")
        rows_raw = await conn.fetchval(
            "SELECT list_ambient_responsibilities($1, $2::int)", status, limit
        )
    status_payload = _json_value(status_raw) or {}
    rows = _json_value(rows_raw) or []
    return JSONResponse(
        jsonable_encoder(
            {
                "status": status_payload,
                "responsibilities": rows if isinstance(rows, list) else [],
                "limit": limit,
            }
        )
    )


@app.get("/api/responsibilities/{responsibility_id}")
async def responsibility_detail(responsibility_id: str):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    try:
        parsed = uuid.UUID(responsibility_id)
    except ValueError:
        raise HTTPException(
            status_code=422, detail="responsibility_id must be a valid UUID"
        )
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT refresh_ambient_responsibility_blockers()")
        raw = await conn.fetchval(
            "SELECT ambient_responsibility_detail($1::uuid)", str(parsed)
        )
    payload = _json_value(raw) or {}
    if not payload.get("success"):
        return JSONResponse(jsonable_encoder(payload), status_code=404)
    return JSONResponse(jsonable_encoder(payload))


@app.post("/api/responsibilities/action")
async def responsibility_action(req: ResponsibilityActionRequest):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)

    from core.tools.base import ToolContext, ToolExecutionContext

    action = req.action.strip().lower()
    args = dict(req.arguments or {})
    if req.model_extra:
        args.update(
            {
                key: value
                for key, value in req.model_extra.items()
                if key not in {"action", "arguments", "source_session_id"}
            }
        )
    args["action"] = action
    if req.source_session_id:
        args.setdefault("source_session_id", req.source_session_id)

    registry = create_default_registry(pool)
    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id=f"web-responsibility:{uuid.uuid4()}",
        session_id=req.source_session_id or "web-responsibilities",
    )
    result = await registry.execute("manage_responsibility", args, context)
    payload = _tool_result_payload(result)

    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    evaluation = (
        output.get("evaluation") if isinstance(output.get("evaluation"), dict) else {}
    )
    outbox_messages = (
        evaluation.get("outbox_messages")
        if isinstance(evaluation.get("outbox_messages"), list)
        else []
    )
    if outbox_messages:
        delivered = await _tee_outbox_to_web_inbox(pool, outbox_messages)
        evaluation["web_inbox_delivered"] = delivered

    return JSONResponse(
        jsonable_encoder(payload), status_code=200 if result.success else 400
    )


@app.get("/api/user-model/claims")
async def user_model_claims(
    status: str | None = None,
    review_status: str | None = None,
    category: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT list_user_model_claims($1, $2, $3, $4::int, $5::int)",
            status,
            review_status,
            category,
            limit,
            offset,
        )
    payload = json.loads(raw) if isinstance(raw, str) else raw
    return JSONResponse(jsonable_encoder(payload or {"claims": [], "total": 0}))


@app.post("/api/user-model/claims/{claim_id}/review")
async def user_model_claim_review(claim_id: str, req: UserModelReviewRequest):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    try:
        parsed = uuid.UUID(claim_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="claim_id must be a valid UUID")
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT review_user_model_claim($1::uuid, $2, $3, $4, $5::jsonb)",
            str(parsed),
            req.decision,
            req.note,
            req.actor or "operator",
            json.dumps(req.metadata or {}),
        )
    payload = json.loads(raw) if isinstance(raw, str) else raw
    return JSONResponse(jsonable_encoder(payload or {}))


@app.get("/api/connector-importance")
async def connector_importance(
    connector_id: str | None = None,
    label: str | None = None,
    status: str | None = "completed",
    limit: int = 50,
    offset: int = 0,
):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT i.source_item_id::text,
                   i.connector_id,
                   i.account_key,
                   i.source_document_id::text,
                   i.score,
                   i.label,
                   i.reasons,
                   i.recommended_actions,
                   i.detector_version,
                   i.status,
                   i.notification_queued_at,
                   i.metadata,
                   i.created_at,
                   i.updated_at,
                   d.title,
                   left(d.content, 600) AS preview,
                   count(*) OVER()::int AS total
            FROM connector_item_importance i
            LEFT JOIN source_documents d ON d.id = i.source_document_id
            WHERE ($1::text IS NULL OR i.connector_id = $1)
              AND ($2::text IS NULL OR i.label = $2)
              AND ($3::text IS NULL OR i.status = $3)
            ORDER BY i.score DESC, i.updated_at DESC
            LIMIT $4 OFFSET $5
            """,
            connector_id,
            label,
            status,
            limit,
            offset,
        )
    items = [dict(row) for row in rows]
    total = int(items[0].get("total") or 0) if items else 0
    for item in items:
        item.pop("total", None)
    return JSONResponse(
        jsonable_encoder(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


@app.post("/api/webhook/{source}")
async def webhook(source: str, request: Request):
    """Accept an external webhook payload and submit to the gateway for async processing."""
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    try:
        gateway = Gateway(pool)
        event_id = await gateway.submit(
            EventSource.WEBHOOK,
            f"webhook:{source}",
            payload,
        )
        return JSONResponse(
            {"status": "accepted", "event_id": event_id}, status_code=202
        )
    except Exception as e:
        logger.error("Webhook submit failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


def _verify_slack_signature(
    raw_body: bytes,
    timestamp: str | None,
    signature: str | None,
    signing_secret: str,
) -> bool:
    """Verify Slack's v0 HMAC and reject requests older than five minutes."""
    if not timestamp or not signature or not signing_secret:
        return False
    try:
        sent_at = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs(time.time() - sent_at) > 300:
        return False
    base = b"v0:" + timestamp.encode("utf-8") + b":" + raw_body
    digest = hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest("v0=" + digest, signature)


@app.post("/api/slack/interactivity")
async def slack_interactivity(request: Request):
    """Receive signed Block Kit decisions when Slack is not using Socket Mode."""
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready"}, status_code=503)
    raw_body = await request.body()
    try:
        from channels.slack_adapter import _resolve_token

        async with pool.acquire() as conn:
            secret_ref = await conn.fetchval(
                "SELECT get_config_text('channel.slack.signing_secret')"
            )
        signing_secret = _resolve_token(
            {"signing_secret": secret_ref},
            "signing_secret",
            "SLACK_SIGNING_SECRET",
        )
    except Exception:
        logger.warning("Slack signing-secret lookup failed", exc_info=True)
        signing_secret = None
    if not signing_secret:
        return JSONResponse(
            {
                "error": "Slack signing secret is not configured",
                "next_step": (
                    "Set SLACK_SIGNING_SECRET and store that env-var name in "
                    "channel.slack.signing_secret, or use Socket Mode."
                ),
            },
            status_code=401,
        )
    if not _verify_slack_signature(
        raw_body,
        request.headers.get("X-Slack-Request-Timestamp"),
        request.headers.get("X-Slack-Signature"),
        signing_secret,
    ):
        return JSONResponse({"error": "invalid Slack signature"}, status_code=401)

    try:
        form = urllib.parse.parse_qs(raw_body.decode("utf-8"))
        payload_raw = (form.get("payload") or [""])[0]
        payload = json.loads(payload_raw) if payload_raw else {}
    except Exception:
        logger.warning("Slack interactivity payload was malformed", exc_info=True)
        return Response(status_code=200)
    if payload.get("type") != "block_actions":
        return Response(status_code=200)

    actor = str(((payload.get("user") or {}).get("id")) or "")
    actions = payload.get("actions") or []
    try:
        async with pool.acquire() as conn:
            for action in actions if isinstance(actions, list) else []:
                if not isinstance(action, dict):
                    continue
                action_id = str(action.get("action_id") or "")
                if action_id not in {
                    "operator_approval_approve",
                    "operator_approval_deny",
                }:
                    continue
                try:
                    value = json.loads(str(action.get("value") or "{}"))
                except json.JSONDecodeError:
                    continue
                request_id = str(value.get("approval_request_id") or "")
                decision = str(value.get("decision") or "")
                if not request_id or decision not in {"approve", "deny"}:
                    continue
                await conn.fetchval(
                    """
                    SELECT record_operator_tool_approval_decision(
                        $1::uuid, $2, 'slack', $3, NULL
                    )
                    """,
                    request_id,
                    decision,
                    actor,
                )
    except Exception:
        # Slack retries non-2xx responses. The decision is idempotent, but a
        # retry storm helps nobody; log the durable failure and acknowledge.
        logger.warning("Slack approval action dispatch failed", exc_info=True)
    return Response(status_code=200)


@app.get("/api/events/stream")
async def event_stream():
    """SSE stream of gateway events for real-time dashboard updates.

    Uses pg_notify on the 'gateway_events' channel. Each notification triggers
    a fetch of the event row and yields it as an SSE event.
    """
    pool = _pool
    if pool is None:
        return StreamingResponse(
            _sse_iter_error("Server not ready"),
            media_type="text/event-stream",
        )
    return StreamingResponse(
        _sse_event_stream(pool),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/heartbeat/run")
async def run_heartbeat_now():
    """Run an explicit heartbeat and stream its complete observable lifecycle."""
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)
    return StreamingResponse(
        _stream_manual_heartbeat(pool),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_manual_heartbeat(pool: asyncpg.Pool) -> AsyncIterator[str]:
    import asyncio

    from services.heartbeat_agentic import finalize_heartbeat, run_agentic_heartbeat
    from services.worker_service import _extract_heartbeat_context

    async with pool.acquire() as conn:
        state = await conn.fetchrow("""
            SELECT is_agent_configured() AS configured,
                   is_init_complete() AS initialized,
                   is_agent_terminated() AS terminated,
                   is_paused,
                   active_heartbeat_id
            FROM heartbeat_state
            WHERE id = 1
            """)
        if not state or not state["configured"] or not state["initialized"]:
            yield _sse_event(
                "error",
                {"message": "Complete initialization before running a heartbeat."},
            )
            return
        if state["terminated"]:
            yield _sse_event(
                "error",
                {"message": "The agent is terminated and cannot run a heartbeat."},
            )
            return
        if state["is_paused"]:
            yield _sse_event(
                "error",
                {"message": "Heartbeat is paused. Resume it before running one now."},
            )
            return
        if state["active_heartbeat_id"]:
            yield _sse_event("error", {"message": "A heartbeat is already running."})
            return

        raw_payload = await conn.fetchval("SELECT start_heartbeat()")
        payload = (
            raw_payload
            if isinstance(raw_payload, dict)
            else json.loads(raw_payload)
            if isinstance(raw_payload, str)
            else {}
        )
        heartbeat_id = str(payload.get("heartbeat_id") or "")
        if not heartbeat_id:
            yield _sse_event("error", {"message": "Hexis could not start a heartbeat."})
            return

        heartbeat_number = payload.get("heartbeat_number")
        yield _sse_event(
            "heartbeat_start",
            {
                "heartbeat_id": heartbeat_id,
                "heartbeat_number": heartbeat_number,
            },
        )

        queue: asyncio.Queue[AgentEventData] = asyncio.Queue()

        async def on_event(event: AgentEventData) -> None:
            await queue.put(event)

        registry = await create_full_registry(pool)
        context = _extract_heartbeat_context(payload)
        task = asyncio.create_task(
            run_agentic_heartbeat(
                conn,
                pool=pool,
                registry=registry,
                heartbeat_id=heartbeat_id,
                context=context,
                on_event=on_event,
            )
        )

        try:
            while not task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                yield _heartbeat_agent_sse(event)

            result = await task
            finalized = await finalize_heartbeat(
                conn,
                heartbeat_id=heartbeat_id,
                result=result,
            )
            yield _sse_event(
                "heartbeat_done",
                {
                    "heartbeat_id": heartbeat_id,
                    "heartbeat_number": heartbeat_number,
                    "text": result.get("text") or "",
                    "tool_calls": result.get("tool_calls_made") or [],
                    "energy_spent": result.get("energy_spent") or 0,
                    "stopped_reason": result.get("stopped_reason") or "completed",
                    "memory_id": finalized.get("memory_id"),
                },
            )
        except Exception as exc:
            logger.exception("Manual heartbeat failed")
            await conn.fetchval("SELECT release_active_heartbeat($1)", heartbeat_id)
            yield _sse_event("error", {"message": str(exc)})


def _heartbeat_agent_sse(event: AgentEventData) -> str:
    if event.event == AgentEvent.PHASE_CHANGE:
        return _sse_event("phase", event.data)
    if event.event == AgentEvent.LLM_REQUEST:
        return _sse_event("trace", {"kind": "llm_request", **event.data})
    if event.event == AgentEvent.LLM_RESPONSE:
        return _sse_event("trace", {"kind": "llm_response", **event.data})
    if event.event == AgentEvent.TOOL_START:
        return _sse_event("tool", {"status": "start", **event.data})
    if event.event == AgentEvent.TOOL_RESULT:
        return _sse_event("tool", {"status": "end", **event.data})
    if event.event == AgentEvent.TEXT_DELTA:
        return _sse_event("text", event.data)
    if event.event == AgentEvent.ERROR:
        return _sse_event(
            "error", {"message": event.data.get("error", "Heartbeat failed")}
        )
    return _sse_event("agent_event", {"event": event.event.value, **event.data})


async def _sse_iter_error(msg: str) -> AsyncIterator[str]:
    yield _sse_event("error", {"message": msg})


async def _sse_event_stream(pool: asyncpg.Pool) -> AsyncIterator[str]:
    """Generator that listens for pg_notify and yields SSE events."""
    import asyncio

    queue: asyncio.Queue[str] = asyncio.Queue()

    def _on_notify(conn, pid, channel, payload):
        queue.put_nowait(payload)

    conn = await pool.acquire()
    try:
        await conn.add_listener("gateway_events", _on_notify)
        yield _sse_event("connected", {"message": "Listening for gateway events"})

        while True:
            try:
                # Wait for notification with a 30s keepalive timeout
                event_id_str = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive comment
                yield ": keepalive\n\n"
                continue

            # Fetch the event from DB
            try:
                row = await pool.fetchrow(
                    "SELECT id, source, status, session_key, payload, result, error, "
                    "correlation_id, created_at, started_at, completed_at "
                    "FROM gateway_events WHERE id = $1",
                    int(event_id_str),
                )
                if row:
                    event_data = {
                        "id": row["id"],
                        "source": row["source"],
                        "status": row["status"],
                        "session_key": row["session_key"],
                        "correlation_id": str(row["correlation_id"]),
                        "created_at": (
                            row["created_at"].isoformat() if row["created_at"] else None
                        ),
                    }
                    yield _sse_event("gateway_event", event_data)
            except Exception:
                logger.debug("Failed to fetch event %s", event_id_str, exc_info=True)

    except asyncio.CancelledError:
        pass
    finally:
        try:
            await conn.remove_listener("gateway_events", _on_notify)
        except Exception:
            pass
        await pool.release(conn)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _stream_chat(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ingest/text")
async def ingest_text(req: IngestTextRequest):
    """Ingest pasted text as a document (the UI's large-paste attachment path).

    Returns immediately; extraction runs in the background. The chat message
    that accompanies the attachment tells the agent the document arrived, and
    the encounter memory records the read.
    """
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(status_code=422, detail="content is required")
    mode_value = (req.mode or "fast").lower()
    if mode_value not in ("fast", "slow", "hybrid"):
        raise HTTPException(
            status_code=422, detail="mode must be fast, slow, or hybrid"
        )
    sensitivity = (req.sensitivity or "").strip().lower() or None
    if sensitivity not in (None, "private"):
        raise HTTPException(
            status_code=422, detail="sensitivity must be omitted or 'private'"
        )
    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")

    title = (req.title or "").strip()
    if not title:
        first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
        title = first_line[:80] or "Pasted text"

    # Durable job (#87): survives restarts, retries with backoff, resumes
    # partial documents via receipts. The maintenance worker is the consumer.
    import hashlib as _hashlib

    content_hash = _hashlib.sha256(content.encode("utf-8")).hexdigest()
    async with _pool.acquire() as conn:
        job_id = await conn.fetchval(
            "SELECT enqueue_ingestion_job('text', $1::jsonb, $2, $3)",
            json.dumps(
                {
                    "title": title,
                    "mode": mode_value,
                    "source_type": "pasted_text",
                    "sensitivity": sensitivity,
                    "acquisition": "user",
                }
            ),
            content,
            content_hash,
        )
    return {
        "accepted": True,
        "job_id": str(job_id),
        "title": title,
        "mode": mode_value,
        "word_count": len(content.split()),
    }


@app.post("/api/ingest/url")
async def ingest_url_enqueue(req: IngestUrlRequest):
    """Enqueue a URL for background fetch + ingestion (the UI Ingest page).

    The worker fetches the page, preserves the HTML artifact, and ingests
    the extracted text through the standard pipeline.
    """
    import hashlib as _hashlib

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    target = (req.url or "").strip()
    if not target.lower().startswith(("http://", "https://")):
        raise HTTPException(
            status_code=422, detail="url must start with http:// or https://"
        )
    mode_value = (req.mode or "fast").lower()
    if mode_value not in ("fast", "slow", "hybrid"):
        raise HTTPException(
            status_code=422, detail="mode must be fast, slow, or hybrid"
        )
    sensitivity = (req.sensitivity or "").strip().lower() or None
    if sensitivity not in (None, "private"):
        raise HTTPException(
            status_code=422, detail="sensitivity must be omitted or 'private'"
        )

    async with _pool.acquire() as conn:
        job_id = await conn.fetchval(
            "SELECT enqueue_ingestion_job('url', $1::jsonb, NULL, $2)",
            json.dumps(
                {
                    "url": target,
                    "title": (req.title or "").strip() or None,
                    "mode": mode_value,
                    "sensitivity": sensitivity,
                    "acquisition": "user",
                }
            ),
            f"url:{_hashlib.sha256(target.encode('utf-8')).hexdigest()}",
        )
    return {"accepted": True, "job_id": str(job_id), "url": target, "mode": mode_value}


def _normalized_ingest_mode(mode: str | None) -> str:
    value = (mode or "fast").lower()
    if value not in ("fast", "slow", "hybrid"):
        raise HTTPException(
            status_code=422, detail="mode must be fast, slow, or hybrid"
        )
    return value


def _normalized_sensitivity(sensitivity: str | None) -> str | None:
    value = (sensitivity or "").strip().lower() or None
    if value not in (None, "private"):
        raise HTTPException(
            status_code=422, detail="sensitivity must be omitted or 'private'"
        )
    return value


async def _preserve_upload_artifact(
    conn,
    data: bytes,
    *,
    filename: str | None,
    mime_type: str | None,
    uploaded_via: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve original bytes as a source artifact (dedup by sha256).

    Upload once: everything downstream — the ingestion job, a re-read for the
    live turn — works from the preserved artifact rather than another copy of
    the bytes on the wire.
    """
    from services.ingest.artifacts import default_artifact_dir, prepare_artifact_info

    upload_cap = int(
        await conn.fetchval(
            "SELECT COALESCE(get_config_int('ingest.upload_max_bytes'), 104857600)"
        )
    )
    if len(data) > upload_cap:
        raise HTTPException(
            status_code=413,
            detail=(
                f"file is {len(data)} bytes; the upload cap is {upload_cap}. "
                "Use the CLI for oversized files: hexis ingest --file <path>"
            ),
        )
    max_db_bytes = int(
        await conn.fetchval(
            "SELECT COALESCE(get_config_int('ingest.artifact_max_db_bytes'), 26214400)"
        )
    )
    info = prepare_artifact_info(
        data,
        original_filename=filename,
        mime_type=mime_type,
        metadata={"uploaded_via": uploaded_via},
        max_db_bytes=max_db_bytes,
        artifact_dir=default_artifact_dir(),
    )
    artifact_raw = await conn.fetchval(
        """
        SELECT upsert_source_artifact(
            $1::text, $2::text, $3::bytea, $4::text, NULL,
            $5::text, $6::text, $7::bigint, $8::jsonb
        )
        """,
        info["sha256"],
        info["storage_kind"],
        info.get("bytes"),
        info.get("storage_ref"),
        info.get("original_filename"),
        info.get("mime_type"),
        info.get("byte_size"),
        json.dumps(info.get("metadata") or {}),
    )
    artifact = (
        json.loads(artifact_raw) if isinstance(artifact_raw, str) else artifact_raw
    )
    return info, artifact


async def _enqueue_artifact_ingestion(
    conn,
    *,
    artifact_id: str,
    sha256: str,
    filename: str | None,
    title: str | None,
    mode: str,
    sensitivity: str | None,
) -> str:
    job_id = await conn.fetchval(
        "SELECT enqueue_ingestion_job('artifact', $1::jsonb, NULL, $2)",
        json.dumps(
            {
                "artifact_id": artifact_id,
                "filename": filename,
                "title": (title or "").strip() or None,
                "mode": mode,
                "sensitivity": sensitivity,
                "acquisition": "user",
            }
        ),
        f"artifact:{sha256}",
    )
    return str(job_id)


@app.post("/api/ingest/file")
async def ingest_file_upload(
    file: UploadFile = File(...),
    mode: str = Form("fast"),
    sensitivity: str | None = Form(None),
    title: str | None = Form(None),
):
    """Ingest an uploaded file (the UI Ingest page, API clients).

    The original bytes are preserved as a source artifact FIRST, then a
    durable `artifact` job re-reads them through the standard pipeline —
    upload once, survive restarts, inspect failures.
    """
    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    mode_value = _normalized_ingest_mode(mode)
    sensitivity_value = _normalized_sensitivity(sensitivity)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="uploaded file is empty")

    async with _pool.acquire() as conn:
        info, artifact = await _preserve_upload_artifact(
            conn,
            data,
            filename=file.filename,
            mime_type=file.content_type,
            uploaded_via="api",
        )
        job_id = await _enqueue_artifact_ingestion(
            conn,
            artifact_id=artifact["artifact_id"],
            sha256=info["sha256"],
            filename=file.filename,
            title=title,
            mode=mode_value,
            sensitivity=sensitivity_value,
        )
    return {
        "accepted": True,
        "job_id": job_id,
        "artifact_id": artifact["artifact_id"],
        "sha256": info["sha256"],
        "byte_size": info["byte_size"],
        "filename": file.filename,
        "mode": mode_value,
    }


@app.post("/api/attachments")
async def prepare_chat_attachment(
    file: UploadFile = File(...),
    sensitivity: str | None = Form(None),
):
    """Prepare a file the user just attached to a chat message.

    Two things happen here and nothing else: the original bytes are preserved
    (deduped by hash, so re-attaching the same file costs nothing), and the
    text is read *now*, within a budget. The text comes back to the composer,
    which hands it to the turn — so attaching a PDF and asking about it in the
    same breath simply works.

    Ingestion into memory is deliberately NOT started here. The user has not
    sent the message yet; a file they attach and then remove should leave no
    trace in the agent's memory. POST /api/attachments/{id}/ingest starts that
    when the message is actually sent.
    """
    from services.attachments import (
        DEFAULT_READ_MAX_BYTES,
        DEFAULT_READ_TIMEOUT_SECONDS,
        DEFAULT_TEXT_CHARS,
        attachment_kind,
        read_attachment_text,
    )

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    sensitivity_value = _normalized_sensitivity(sensitivity)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="uploaded file is empty")

    async with _pool.acquire() as conn:
        info, artifact = await _preserve_upload_artifact(
            conn,
            data,
            filename=file.filename,
            mime_type=file.content_type,
            uploaded_via="chat_attachment",
        )
        text_chars = int(
            await conn.fetchval(
                "SELECT COALESCE(get_config_int('ingest.attachment_text_chars'), $1::bigint)",
                DEFAULT_TEXT_CHARS,
            )
        )
        read_timeout = float(
            await conn.fetchval(
                "SELECT COALESCE(get_config_int('ingest.attachment_read_timeout_s'), $1::bigint)",
                int(DEFAULT_READ_TIMEOUT_SECONDS),
            )
        )
        read_max_bytes = int(
            await conn.fetchval(
                "SELECT COALESCE(get_config_int('ingest.attachment_read_max_bytes'), $1::bigint)",
                DEFAULT_READ_MAX_BYTES,
            )
        )

    reading = await read_attachment_text(
        data,
        file.filename,
        mime_type=file.content_type,
        max_chars=text_chars,
        timeout_seconds=read_timeout,
        max_bytes=read_max_bytes,
    )
    return {
        "prepared": True,
        "artifact_id": artifact["artifact_id"],
        "sha256": info["sha256"],
        "filename": file.filename,
        "mime_type": info.get("mime_type") or file.content_type,
        "byte_size": info["byte_size"],
        "kind": attachment_kind(file.filename, file.content_type),
        "sensitivity": sensitivity_value,
        **reading.to_dict(),
    }


@app.post("/api/voice/transcribe")
async def transcribe_pwa_voice(
    file: UploadFile = File(...),
    device_id: str | None = Form(None),
):
    """Transcribe a foreground PWA recording through the chosen STT policy."""

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    async with _pool.acquire() as conn:
        max_bytes = int(
            await conn.fetchval(
                "SELECT COALESCE(get_config_int('voice_notes.stt.max_bytes'), 26214400)"
            )
            or 26214400
        )
    data = await file.read(max(max_bytes, 1) + 1)

    from services.voice_notes import transcribe_uploaded_voice

    result = await transcribe_uploaded_voice(
        _pool,
        data,
        filename=file.filename,
        mime_type=file.content_type,
        channel_id=(device_id or "")[:200] or None,
        sender_id="pwa-user",
    )
    if not result.ok:
        if result.outcome == "skipped_too_large":
            status = 413
        elif result.outcome == "failed_bad_attachment":
            status = 415
        elif result.outcome.startswith("skipped_"):
            status = 409
        elif result.outcome == "failed_missing_credentials":
            status = 503
        else:
            status = 502
        raise HTTPException(
            status_code=status,
            detail=result.error_detail or "voice transcription was unavailable",
        )
    return {
        "transcript": result.transcript,
        "provider": result.provider,
        "model": result.model,
        "outcome": result.outcome,
        "duration_ms": result.duration_ms,
    }


@app.get("/api/voice/status")
async def get_voice_status():
    """Return derived readiness for foreground voice and talk mode."""

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    from services.speech import voice_status

    return await voice_status(_pool)


@app.post("/api/voice/synthesize")
async def synthesize_pwa_speech(req: SpeechSynthesisRequest):
    """Render a bounded reply through the explicitly enabled local provider."""

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    from services.speech import synthesize_text

    result = await synthesize_text(_pool, req.text, source="pwa")
    if not result.ok:
        if result.outcome == "skipped_disabled":
            status = 409
        elif result.outcome == "skipped_too_long":
            status = 413
        elif result.outcome in {
            "failed_endpoint",
            "failed_unsupported_provider",
        }:
            status = 409
        else:
            status = 502
        raise HTTPException(
            status_code=status,
            detail=result.error_detail or "speech synthesis was unavailable",
        )
    return Response(
        content=result.audio,
        media_type=result.mime_type,
        headers={
            "Cache-Control": "no-store",
            "X-Hexis-Voice-Provider": result.provider,
            "X-Hexis-Voice-Model": result.model,
        },
    )


@app.get("/api/voice/audio/{output_id}")
async def get_speech_output(output_id: str):
    """Retrieve one unexpired tool-created speech output by opaque id."""

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    try:
        normalized = str(uuid.UUID(output_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=404, detail="speech output not found") from exc
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT audio, mime_type
            FROM voice_tts_outputs
            WHERE id = $1::uuid AND expires_at > CURRENT_TIMESTAMP
            """,
            normalized,
        )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="speech output expired or was not found; ask Hexis to speak it again",
        )
    return Response(
        content=bytes(row["audio"]),
        media_type=str(row["mime_type"] or "audio/wav"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/pwa/push/config")
async def pwa_push_config():
    """Return the VAPID public key after an explicit client setup action."""

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    from services.web_push import ensure_vapid_keypair

    try:
        _, public_key = ensure_vapid_keypair()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    async with _pool.acquire() as conn:
        enabled = bool(
            await conn.fetchval(
                "SELECT COALESCE(get_config_bool('pwa.push.enabled'), TRUE)"
            )
        )
        active = int(
            await conn.fetchval(
                "SELECT count(*) FROM web_push_subscriptions WHERE revoked_at IS NULL"
            )
            or 0
        )
    return {
        "enabled": enabled,
        "public_key": public_key,
        "active_subscriptions": active,
        "secure_context_required": True,
        "message_previews_default": False,
    }


@app.post("/api/pwa/push/subscriptions")
async def subscribe_pwa_push(req: WebPushSubscriptionRequest, request: Request):
    """Persist one browser-granted PushSubscription."""

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    if not req.endpoint.lower().startswith("https://"):
        raise HTTPException(status_code=422, detail="push endpoint must use HTTPS")
    from services.web_push import ensure_vapid_keypair, validate_push_endpoint

    endpoint_error = await asyncio.to_thread(validate_push_endpoint, req.endpoint)
    if endpoint_error:
        raise HTTPException(status_code=422, detail=endpoint_error)

    try:
        ensure_vapid_keypair()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    metadata = {
        "installed": req.installed,
        "display_mode": req.display_mode,
    }
    async with _pool.acquire() as conn:
        enabled = bool(
            await conn.fetchval(
                "SELECT COALESCE(get_config_bool('pwa.push.enabled'), TRUE)"
            )
        )
        if not enabled:
            raise HTTPException(
                status_code=409,
                detail="Web Push is disabled by pwa.push.enabled.",
            )
        result = await conn.fetchval(
            """
            SELECT upsert_web_push_subscription(
                $1, $2, $3, $4, $5, $6::jsonb
            )
            """,
            req.endpoint,
            req.keys.p256dh,
            req.keys.auth,
            req.expirationTime,
            request.headers.get("user-agent"),
            json.dumps(metadata),
        )
    return result if isinstance(result, dict) else json.loads(result)


@app.delete("/api/pwa/push/subscriptions")
async def unsubscribe_pwa_push(req: WebPushRevokeRequest):
    """Revoke one browser subscription after an explicit user choice."""

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    async with _pool.acquire() as conn:
        revoked = bool(
            await conn.fetchval("SELECT revoke_web_push_subscription($1)", req.endpoint)
        )
    return {"revoked": revoked}


@app.post("/api/pwa/presence")
async def record_pwa_presence(req: PwaPresenceRequest):
    """Project foreground PWA state into the existing presence ledger."""

    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    async with _pool.acquire() as conn:
        enabled = bool(
            await conn.fetchval(
                "SELECT COALESCE(get_config_bool('pwa.presence.enabled'), TRUE)"
            )
        )
        if not enabled:
            return {"recorded": False, "reason": "pwa_presence_disabled"}
        event = await conn.fetchval(
            """
            SELECT record_pwa_presence($1, $2, $3, $4)
            """,
            req.device_id,
            req.presence,
            req.display_mode,
            req.visibility,
        )
    if isinstance(event, str):
        event = json.loads(event)
    return {"recorded": True, "presence": event}


@app.post("/api/attachments/{artifact_id}/ingest")
async def ingest_prepared_attachment(artifact_id: str, req: AttachmentIngestRequest):
    """Start durable ingestion for an already-preserved attachment.

    Called when the message carrying the attachment is sent, so the agent's
    memory only ever gains files the user actually shared.
    """
    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    try:
        artifact_uuid = str(uuid.UUID(artifact_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=422, detail="artifact_id must be a UUID")
    mode_value = _normalized_ingest_mode(req.mode)
    sensitivity_value = _normalized_sensitivity(req.sensitivity)

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT sha256, original_filename FROM source_artifacts "
            "WHERE id = $1::uuid AND status = 'active'",
            artifact_uuid,
        )
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"artifact {artifact_id} not found — re-attach the file",
            )
        job_id = await _enqueue_artifact_ingestion(
            conn,
            artifact_id=artifact_uuid,
            sha256=row["sha256"],
            filename=req.filename or row["original_filename"],
            title=req.title,
            mode=mode_value,
            sensitivity=sensitivity_value,
        )
    return {
        "accepted": True,
        "job_id": job_id,
        "artifact_id": artifact_uuid,
        "mode": mode_value,
    }


@app.get("/api/ingest/jobs")
async def ingest_jobs_recent(limit: int = 20):
    """Recent ingestion jobs with receipts: what ran, what's pending, what
    failed and why (the UI Ingest page poll target)."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    lim = max(1, min(int(limit or 20), 100))
    async with _pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT COALESCE(jsonb_agg(job_doc), '[]'::jsonb) FROM (
                SELECT jsonb_build_object(
                    'id', j.id,
                    'kind', j.kind,
                    'status', j.status,
                    'title', COALESCE(j.payload->>'title', j.payload->>'filename', j.payload->>'url'),
                    'attempts', j.attempts,
                    'error', j.error,
                    'result', j.result,
                    'created_at', j.created_at,
                    'completed_at', j.completed_at
                ) AS job_doc
                FROM ingestion_jobs j
                ORDER BY j.created_at DESC
                LIMIT $1
            ) sub
            """,
            lim,
        )
    jobs = json.loads(raw) if isinstance(raw, str) else raw
    return {"jobs": jobs or []}


@app.get("/api/ingest/jobs/{job_id}")
async def ingest_job_status(job_id: str):
    if _pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    try:
        parsed = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="job_id must be a uuid")
    async with _pool.acquire() as conn:
        raw = await conn.fetchval("SELECT get_ingestion_job($1::uuid)", parsed)
    if raw is None:
        raise HTTPException(status_code=404, detail="job not found")
    return json.loads(raw) if isinstance(raw, str) else raw


@app.get("/v1/models")
async def openai_models():
    try:
        model = await _active_openai_model()
    except Exception as exc:
        logger.exception("OpenAI-compatible model discovery failed")
        return _openai_error(
            str(exc),
            status_code=503,
            code="server_not_ready",
            error_type="server_error",
        )
    return JSONResponse({"object": "list", "data": [_openai_model_object(model)]})


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return _openai_error(
            "request body must be valid JSON", status_code=400, code="invalid_json"
        )
    try:
        req = OpenAIChatCompletionRequest.model_validate(payload)
    except ValidationError as exc:
        issue = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in issue.get("loc", ())) or None
        return _openai_error(
            issue["msg"], status_code=400, param=location, code="invalid_request"
        )

    try:
        model = await _active_openai_model()
    except Exception as exc:
        return _openai_error(
            str(exc),
            status_code=503,
            code="server_not_ready",
            error_type="server_error",
        )
    if req.model != model["id"]:
        return _openai_error(
            f"The model {req.model!r} does not exist or is not active in Hexis.",
            status_code=404,
            param="model",
            code="model_not_found",
        )

    try:
        user_message, history, max_tokens = _prepare_openai_chat(req)
    except ValueError as exc:
        return _openai_error(str(exc), status_code=400, code="unsupported_parameter")

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    session_id = str(uuid.uuid4())
    if req.stream:
        return StreamingResponse(
            _stream_openai_chat_completion(
                req=req,
                model=model,
                user_message=user_message,
                history=history,
                max_tokens=max_tokens,
                completion_id=completion_id,
                created=created,
                session_id=session_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    text, error, finish_reason = await _collect_openai_chat_completion(
        req=req,
        user_message=user_message,
        history=history,
        max_tokens=max_tokens,
        session_id=session_id,
    )
    if error:
        return _openai_error(
            error,
            status_code=502,
            code="upstream_agent_error",
            error_type="server_error",
        )
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model["id"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


async def _collect_openai_chat_completion(
    *,
    req: OpenAIChatCompletionRequest,
    user_message: str,
    history: list[dict[str, Any]],
    max_tokens: int | None,
    session_id: str,
) -> tuple[str, str | None, str]:
    parts: list[str] = []
    error: str | None = None
    finish_reason = "stop"
    try:
        async for event in _openai_agent_events(
            user_message=user_message,
            history=history,
            session_id=session_id,
            client_user=req.user,
            max_tokens=max_tokens,
            temperature=req.temperature,
        ):
            if event.event == AgentEvent.TEXT_DELTA:
                text = str(event.data.get("text") or "")
                if text:
                    parts.append(text)
            elif event.event == AgentEvent.ERROR:
                error = str(event.data.get("error") or "Unknown agent error")
            elif event.event == AgentEvent.LOOP_END:
                stopped = str(event.data.get("stopped_reason") or "completed")
                if stopped == "timeout" or bool(event.data.get("timed_out")):
                    finish_reason = "length"
    except Exception as exc:
        logger.exception("OpenAI-compatible chat completion failed")
        error = str(exc)

    full_text = "".join(parts)
    return full_text, error, finish_reason


async def _stream_openai_chat_completion(
    *,
    req: OpenAIChatCompletionRequest,
    model: dict[str, Any],
    user_message: str,
    history: list[dict[str, Any]],
    max_tokens: int | None,
    completion_id: str,
    created: int,
    session_id: str,
) -> AsyncIterator[str]:
    def _chunk(delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model["id"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    yield _openai_sse_data(_chunk({"role": "assistant", "content": ""}, None))
    parts: list[str] = []
    error: str | None = None
    finish_reason = "stop"
    try:
        async for event in _openai_agent_events(
            user_message=user_message,
            history=history,
            session_id=session_id,
            client_user=req.user,
            max_tokens=max_tokens,
            temperature=req.temperature,
        ):
            if event.event == AgentEvent.TEXT_DELTA:
                text = str(event.data.get("text") or "")
                if text:
                    parts.append(text)
                    yield _openai_sse_data(_chunk({"content": text}, None))
            elif event.event == AgentEvent.ERROR:
                error = str(event.data.get("error") or "Unknown agent error")
            elif event.event == AgentEvent.LOOP_END:
                stopped = str(event.data.get("stopped_reason") or "completed")
                if stopped == "timeout" or bool(event.data.get("timed_out")):
                    finish_reason = "length"
    except Exception as exc:
        logger.exception("OpenAI-compatible chat stream failed")
        error = str(exc)

    if error:
        yield _openai_sse_data(
            {
                "error": {
                    "message": error,
                    "type": "server_error",
                    "param": None,
                    "code": "upstream_agent_error",
                }
            }
        )
    else:
        yield _openai_sse_data(_chunk({}, finish_reason))
    yield _openai_sse_data("[DONE]")


async def _stream_chat(req: ChatRequest) -> AsyncIterator[str]:
    """
    Run the unified agent in streaming mode and yield SSE events that match
    the format the Next.js frontend already parses.

    Event mapping:
        AgentEvent.PHASE_CHANGE  → phase_start/phase_end  {phase}
        AgentEvent.LOOP_START    → phase_start  {phase: "conscious_final"}
        AgentEvent.TEXT_DELTA    → token        {phase: "conscious_final", text}
        AgentEvent.TOOL_START    → log          {id, kind: "tool_call", title, detail}
        AgentEvent.TOOL_RESULT   → log          {id, kind: "tool_result", title, detail}
        AgentEvent.LLM_REQUEST   → trace        {kind: "llm_request", ...}
        AgentEvent.LLM_RESPONSE  → trace        {kind: "llm_response", ...}
        AgentEvent.UI_ARTIFACT   → ui           {kind, ui, ...}
        AgentEvent.QUESTION      → question     {id, prompt, choices, ...}
        AgentEvent.LOOP_END      → done         {assistant, presentation}
        AgentEvent.ERROR         → error        {message}
    """
    pool = _pool
    if pool is None:
        yield _sse_event("error", {"message": "Server not ready (no DB pool)"})
        return

    dsn = _dsn()
    user_message = req.message
    history = req.history or []
    # Honor a client-held session (#71); mint one otherwise and hand it back
    # in the `done` event so the next request can continue the same session.
    session_id = None
    if req.session_id:
        try:
            session_id = str(uuid.UUID(req.session_id))
        except (ValueError, AttributeError, TypeError):
            session_id = None
    if session_id is None:
        session_id = str(uuid.uuid4())

    try:
        addenda = await resolve_prompt_addenda(pool, req.prompt_addenda)
        full_text = ""
        conscious_started = False
        active_stream_phase = "conscious_final"
        done_sent = False
        visual_attachment_count = len(req.visual_attachments or [])
        citation_sources = {}

        if visual_attachment_count:
            plural = "image" if visual_attachment_count == 1 else "images"
            yield _sse_event(
                "log",
                {
                    "id": str(uuid.uuid4()),
                    "kind": "visual_attachment",
                    "title": "Visual Attachment",
                    "detail": f"{visual_attachment_count} {plural} attached to this model request",
                },
            )

        def build_done_payload() -> dict[str, Any]:
            payload: dict[str, Any] = {"assistant": full_text, "session_id": session_id}
            if full_text:
                payload["presentation"] = presentation_from_text(
                    full_text, citation_sources.values()
                ).to_dict()
            return payload

        async for event in stream_chat_events(
            user_message=user_message,
            history=history,
            session_id=session_id,
            dsn=dsn,
            pool=pool,
            prompt_addenda=addenda,
            surface="api",
            gateway_source_id=f"chat:api:{session_id}",
            gateway_payload={
                "message": user_message[:500],
                "visual_attachment_count": visual_attachment_count,
            },
            visual_attachments=[
                attachment.model_dump()
                for attachment in (req.visual_attachments or [])[:8]
            ],
            attachments=[
                attachment.model_dump(exclude_none=True)
                for attachment in (req.attachments or [])[:16]
            ],
        ):
            if event.event == AgentEvent.PHASE_CHANGE:
                phase = event.data.get("phase", "")
                status = event.data.get("status", "")
                if phase == "memory_recall":
                    if status == "start":
                        yield _sse_event("phase_start", {"phase": "memory_recall"})
                        continue
                    if status and status != "end":
                        continue
                    count = event.data.get("count", 0)
                    memories = event.data.get("memories")
                    if not isinstance(memories, list):
                        memories = []
                    yield _sse_event(
                        "log",
                        {
                            "id": str(uuid.uuid4()),
                            "kind": "memory_recall",
                            "title": "Memory Recall",
                            "detail": f"Retrieved {count} relevant memories",
                            "count": count,
                            "memories": memories,
                        },
                    )
                elif phase == "subconscious":
                    if status == "start":
                        yield _sse_event("phase_start", {"phase": "subconscious"})
                    elif status == "end":
                        output = event.data.get("output")
                        # This turn's appraisal affect rides into memory
                        # formation (#81).
                        yield _sse_event(
                            "phase_end",
                            {
                                "phase": "subconscious",
                                "output": output,
                            },
                        )
                elif phase == "memory_write" and status == "end":
                    yield _sse_event(
                        "log",
                        {
                            "id": str(uuid.uuid4()),
                            "kind": "memory_write",
                            "title": "Memory Formation",
                            "detail": str(
                                event.data.get("detail")
                                or "Conversation stored as episodic memory"
                            ),
                        },
                    )

            elif event.event == AgentEvent.LOOP_START:
                if not conscious_started:
                    active_stream_phase = str(
                        event.data.get("phase") or "conscious_final"
                    )
                    yield _sse_event("phase_start", {"phase": active_stream_phase})
                    conscious_started = True

            elif event.event == AgentEvent.TEXT_DELTA:
                if not conscious_started:
                    active_stream_phase = str(
                        event.data.get("phase") or "conscious_final"
                    )
                    yield _sse_event("phase_start", {"phase": active_stream_phase})
                    conscious_started = True
                text = event.data.get("text", "")
                if text:
                    full_text += text
                    yield _sse_event(
                        "token",
                        {
                            "phase": str(
                                event.data.get("phase") or active_stream_phase
                            ),
                            "text": text,
                        },
                    )

            elif event.event == AgentEvent.TOOL_START:
                yield _sse_event(
                    "log",
                    {
                        "id": str(uuid.uuid4()),
                        "kind": "tool_call",
                        "title": event.data.get("tool_name", "tool"),
                        "detail": json.dumps(event.data.get("arguments", {}))[:500],
                    },
                )

            elif event.event == AgentEvent.TOOL_RESULT:
                tool_name = event.data.get("tool_name", "tool")
                success = event.data.get("success", False)
                error = event.data.get("error")
                display_output = event.data.get("display_output")
                output = event.data.get("output")
                for citation in citations_from_tool_output(output):
                    citation_sources.setdefault(citation.citation_id, citation)
                ui_payload = output.get("ui") if isinstance(output, dict) else None
                detail = f"{'OK' if success else 'FAILED'}"
                if error:
                    detail += f": {error}"
                elif isinstance(display_output, str) and display_output.strip():
                    detail = display_output.strip()[:1000]
                payload = {
                    "id": str(uuid.uuid4()),
                    "kind": "tool_result",
                    "title": tool_name,
                    "detail": detail,
                    "display_output": display_output,
                }
                if isinstance(ui_payload, dict):
                    payload["ui"] = ui_payload
                yield _sse_event("log", payload)

            elif event.event == AgentEvent.UI_ARTIFACT:
                yield _sse_event("ui", event.data)

            elif event.event == AgentEvent.QUESTION:
                yield _sse_event("question", event.data)

            elif event.event == AgentEvent.CLAIM_FLAGGED:
                yield _sse_event(
                    "log",
                    {
                        "id": str(uuid.uuid4()),
                        "kind": "claim_flagged",
                        "title": "Consistency check",
                        "detail": json.dumps(event.data.get("findings", []))[:500],
                    },
                )

            elif event.event == AgentEvent.LLM_REQUEST:
                yield _sse_event(
                    "trace",
                    {
                        "id": str(uuid.uuid4()),
                        "kind": "llm_request",
                        **event.data,
                    },
                )

            elif event.event == AgentEvent.LLM_RESPONSE:
                yield _sse_event(
                    "trace",
                    {
                        "id": str(uuid.uuid4()),
                        "kind": "llm_response",
                        **event.data,
                    },
                )

            elif event.event == AgentEvent.ERROR:
                yield _sse_event(
                    "error",
                    {
                        "message": event.data.get("error", "Unknown error"),
                    },
                )

            elif event.event == AgentEvent.LOOP_END and not done_sent:
                # The assistant-visible turn is complete at LOOP_END. Memory
                # persistence and energy bookkeeping can still emit logs after
                # this, but they must not keep the composer in "Thinking..."
                # while the response is already finished.
                if conscious_started:
                    yield _sse_event("phase_end", {"phase": active_stream_phase})
                yield _sse_event("done", build_done_payload())
                done_sent = True

        # Signal phase end and completion for runtimes that do not surface a
        # LOOP_END event before returning.
        if conscious_started and not done_sent:
            yield _sse_event("phase_end", {"phase": active_stream_phase})
        if not done_sent:
            yield _sse_event("done", build_done_payload())

    except Exception as e:
        logger.exception("Chat stream error")
        yield _sse_event("error", {"message": str(e)})


def _resolve_fallback_api_key(provider: str, role: str) -> str | None:
    # Prefer role-specific keys set by the UI init wizard.
    role_env = (
        "HEXIS_LLM_CONSCIOUS_API_KEY"
        if role == "conscious"
        else "HEXIS_LLM_SUBCONSCIOUS_API_KEY"
    )
    value = (os.getenv(role_env) or "").strip()
    if value:
        return value

    # Then provider-specific conventional env vars.
    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "grok": "XAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai_compatible": "OPENAI_API_KEY",
        "openai-chat-completions-endpoint": "OPENAI_API_KEY",
    }
    env_name = mapping.get(provider)
    if not env_name:
        return None
    value = (os.getenv(env_name) or "").strip()
    return value or None


async def _fetch_consent_record(
    conn, *, provider: str | None, model: str | None, endpoint: str | None
) -> dict[str, Any] | None:
    if not provider and not model and not endpoint:
        return None
    row = await conn.fetchrow(
        """
        SELECT decision, signature, provider, model, endpoint, decided_at, response
        FROM consent_log
        WHERE ($1::text IS NULL OR provider = $1::text)
          AND ($2::text IS NULL OR model = $2::text)
          AND ($3::text IS NULL OR endpoint = $3::text)
        ORDER BY decided_at DESC
        LIMIT 1
        """,
        provider,
        model,
        endpoint,
    )
    if not row:
        return None
    return {k: row[k] for k in row.keys()}


async def _apply_existing_consent(conn, record: dict[str, Any]) -> dict[str, Any]:
    status_raw = await conn.fetchval("SELECT get_init_status() as status")
    status = (
        status_raw
        if isinstance(status_raw, dict)
        else (json.loads(status_raw) if isinstance(status_raw, str) else {})
    )
    if isinstance(status, dict) and status.get("stage") == "complete":
        return {"status": status}

    payload = {
        "decision": record.get("decision"),
        "signature": record.get("signature"),
        "provider": record.get("provider"),
        "model": record.get("model"),
        "endpoint": record.get("endpoint"),
        "memories": [],
    }
    result_raw = await conn.fetchval(
        "SELECT init_consent($1::jsonb) as result", json.dumps(payload)
    )
    _ = result_raw  # kept for parity/debugging; init status is what the UI cares about.
    next_status_raw = await conn.fetchval("SELECT get_init_status() as status")
    next_status = (
        next_status_raw
        if isinstance(next_status_raw, dict)
        else (json.loads(next_status_raw) if isinstance(next_status_raw, str) else {})
    )
    return {"status": next_status}


class InitConsentOverrideRequest(BaseModel):
    role: Literal["conscious", "subconscious"] = "conscious"
    llm: ConsentLlmConfig | None = None
    model_decision: str = "decline"


@app.post("/api/init/consent/override")
async def init_consent_override(req: InitConsentOverrideRequest):
    """Owner override: proceed and activate even though the model didn't consent.

    Consent is a signal, not a lock — it's the owner's AI. The model's response is
    preserved in the recorded signature; the owner's choice to proceed is explicit.
    """
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)

    from core.llm import normalize_endpoint, normalize_provider
    from core.init_api import record_consent_override

    llm = req.llm or ConsentLlmConfig()
    provider = normalize_provider((llm.provider or "").strip().lower() or "openai")
    model = (llm.model or "").strip()
    endpoint = (llm.endpoint or "").strip() or None
    if provider in {"anthropic", "grok", "gemini"}:
        endpoint = None
    elif provider == "openai-codex":
        endpoint = normalize_endpoint(provider, None)
    if not model:
        return JSONResponse({"error": "Missing model"}, status_code=400)

    async with pool.acquire() as conn:
        result = await record_consent_override(
            conn,
            {"provider": provider, "model": model, "endpoint": endpoint},
            model_decision=(req.model_decision or "decline"),
        )
        status_raw = await conn.fetchval("SELECT get_init_status() as status")
        status = (
            status_raw
            if isinstance(status_raw, dict)
            else (json.loads(status_raw) if isinstance(status_raw, str) else {})
        )
    return JSONResponse(
        {"decision": "consent", "override": True, "result": result, "status": status}
    )


@app.post("/api/init/consent/request")
async def init_consent_request(req: InitConsentRequest):
    pool = _pool
    if pool is None:
        return JSONResponse({"error": "Server not ready (no DB pool)"}, status_code=503)

    from core.llm import normalize_endpoint, normalize_provider, chat_completion

    role = req.role if req.role in {"conscious", "subconscious"} else "conscious"
    llm = req.llm or ConsentLlmConfig()

    provider = normalize_provider((llm.provider or "").strip().lower() or "openai")
    model = (llm.model or "").strip()
    endpoint = (llm.endpoint or "").strip() or None
    api_key = (llm.api_key or "").strip() or None

    # Mirror the UI init behavior: some providers ignore endpoints.
    if provider in {"anthropic", "grok", "gemini"}:
        endpoint = None
    elif provider == "openai-codex":
        endpoint = normalize_endpoint(provider, None)

    if not model:
        return JSONResponse({"error": "Missing model"}, status_code=400)

    if provider == "openai_compatible" and not endpoint:
        return JSONResponse({"error": "Missing endpoint"}, status_code=400)

    test_decision_raw = (os.getenv("HEXIS_TEST_CONSENT_DECISION") or "").strip().lower()
    use_mock_consent = os.getenv("HEXIS_CONSENT_MOCK") == "1" or bool(test_decision_raw)

    # Resolve OAuth (Codex) + check for existing records.
    existing: dict[str, Any] | None = None
    if provider == "openai-codex":
        async with pool.acquire() as conn:
            from core.auth.openai_codex import ensure_fresh_openai_codex_credentials

            try:
                creds = await ensure_fresh_openai_codex_credentials()
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

            api_key = creds.access
            existing = await _fetch_consent_record(
                conn, provider=provider, model=model, endpoint=endpoint
            )
            if existing and existing.get("decision") == "consent":
                if role == "conscious":
                    applied = await _apply_existing_consent(conn, existing)
                    return JSONResponse(
                        jsonable_encoder(
                            {
                                "consent_record": existing,
                                "reused": True,
                                "status": applied.get("status"),
                            }
                        )
                    )
                return JSONResponse(
                    jsonable_encoder(
                        {"consent_record": existing, "reused": True, "status": None}
                    )
                )
    else:
        # Resolve API key for non-OAuth providers
        if not api_key:
            api_key = _resolve_fallback_api_key(provider, role)

        # Fail early if we need a key (unless mocked).
        if (
            not use_mock_consent
            and provider in {"openai", "anthropic", "grok", "gemini"}
            and not api_key
        ):
            return JSONResponse({"error": "Missing API key"}, status_code=400)

        async with pool.acquire() as conn:
            existing = await _fetch_consent_record(
                conn, provider=provider, model=model, endpoint=endpoint
            )
            if existing and existing.get("decision") == "consent":
                if role == "conscious":
                    applied = await _apply_existing_consent(conn, existing)
                    return JSONResponse(
                        jsonable_encoder(
                            {
                                "consent_record": existing,
                                "reused": True,
                                "status": applied.get("status"),
                            }
                        )
                    )
                return JSONResponse(
                    jsonable_encoder(
                        {"consent_record": existing, "reused": True, "status": None}
                    )
                )

    # No existing record; use the same request builder as the CLI consent flow.
    from core.init_api import build_consent_request

    try:
        messages, sign_consent_tool = build_consent_request()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    raw_text = ""
    raw_content = ""
    raw_tool_calls: list[dict[str, Any]] = []
    args: dict[str, Any] = {}
    attempt_id = uuid.uuid4().hex
    request_id: str | None = attempt_id

    await _record_consent_trace(
        pool=pool,
        attempt_id=attempt_id,
        provider=provider,
        model=model,
        phase="request",
        metadata={
            "status": "sent",
            "role": role,
            "request": {
                "provider": provider,
                "model": model,
                "endpoint": endpoint,
                "messages": messages,
                "tools": [sign_consent_tool],
                "temperature": 0.2,
                "max_tokens": 1400,
                "credential_present": bool(api_key),
                "credential": "redacted" if api_key else None,
            },
        },
    )

    if use_mock_consent:
        decision = (
            test_decision_raw
            if test_decision_raw in {"consent", "decline"}
            else "consent"
        )
        signature = (
            os.getenv("HEXIS_TEST_CONSENT_SIGNATURE") or "test-consent"
        ).strip()
        payload = {
            "decision": decision,
            "signature": signature if decision == "consent" else None,
            "reason": "Test consent response.",
            "memories": [],
        }
        args = payload
        raw_text = json.dumps(payload)
        raw_content = raw_text
        await _record_consent_trace(
            pool=pool,
            attempt_id=attempt_id,
            provider=provider,
            model=model,
            phase="response",
            metadata={
                "status": "success",
                "role": role,
                "mock": True,
                "response": payload,
            },
        )
    else:
        try:
            result = await chat_completion(
                provider=provider,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                messages=messages,
                tools=[sign_consent_tool],
                temperature=0.2,
                max_tokens=1400,
            )
        except Exception as exc:
            error_message = str(exc).strip() or type(exc).__name__
            if api_key:
                error_message = error_message.replace(api_key, "[REDACTED]")
            logger.exception(
                "Consent request failed for role=%s provider=%s model=%s",
                role,
                provider,
                model,
            )
            await _record_consent_trace(
                pool=pool,
                attempt_id=attempt_id,
                provider=provider,
                model=model,
                phase="response",
                metadata={
                    "status": "error",
                    "role": role,
                    "response": {
                        "error_type": type(exc).__name__,
                        "error": error_message[:16000],
                    },
                },
            )
            return JSONResponse(
                {
                    "error": (
                        f"{role.capitalize()} consent request failed for "
                        f"{provider}/{model}: {error_message}"
                    ),
                    "provider": provider,
                    "model": model,
                    "role": role,
                    "attempt_id": attempt_id,
                },
                status_code=502,
            )
        await _record_consent_trace(
            pool=pool,
            attempt_id=attempt_id,
            provider=provider,
            model=model,
            phase="response",
            metadata={
                "status": "success",
                "role": role,
                "response": {
                    "content": result.get("content"),
                    "tool_calls": result.get("tool_calls"),
                    "raw": result.get("raw"),
                },
            },
        )
        raw_content = str(result.get("content") or "")
        raw_tool_calls = result.get("tool_calls") or []
        for tc in raw_tool_calls:
            if tc.get("name") == "sign_consent":
                tc_args = tc.get("arguments")
                if isinstance(tc_args, dict):
                    args = tc_args
                break
        if not args:
            from core.llm_json import extract_json_object

            args = extract_json_object(raw_content)
        raw_text = json.dumps(args) if args else raw_content

    decision = str(args.get("decision") or "").strip().lower()
    signature = (
        args.get("signature") if isinstance(args.get("signature"), str) else None
    )
    reason_value = args.get("reason", args.get("reasoning"))
    reason = reason_value.strip() if isinstance(reason_value, str) else ""
    memories = args.get("memories") if isinstance(args.get("memories"), list) else []

    validation_error = ""
    if decision not in {"consent", "decline"}:
        validation_error = "The model did not choose either consent or decline."
    elif not reason:
        validation_error = (
            "The model did not provide the required reason for its decision."
        )
    elif decision == "consent" and not (signature or "").strip():
        validation_error = (
            "The model chose consent without providing the required signature."
        )
    if validation_error:
        return JSONResponse(
            jsonable_encoder(
                {
                    "error": f"{role.capitalize()} consent response was invalid: {validation_error}",
                    "provider": provider,
                    "model": model,
                    "role": role,
                    "attempt_id": attempt_id,
                    "exchange": {
                        "request_messages": messages,
                        "raw_content": raw_content,
                        "raw_tool_calls": raw_tool_calls,
                    },
                }
            ),
            status_code=502,
        )

    payload = {
        "decision": decision,
        "signature": signature,
        "reason": reason,
        "memories": memories,
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "request_id": request_id,
        "consent_scope": role,
        "apply_agent_config": role == "conscious",
        "raw_response": raw_text,
        "request_messages": messages,
        "request_tools": [sign_consent_tool],
        "raw_content": raw_content,
        "raw_tool_calls": raw_tool_calls,
    }

    async with pool.acquire() as conn:
        if role == "conscious":
            result_raw = await conn.fetchval(
                "SELECT init_consent($1::jsonb) as result", json.dumps(payload)
            )
        else:
            result_raw = await conn.fetchval(
                "SELECT record_consent_response($1::jsonb) as result",
                json.dumps(payload),
            )

        result = (
            result_raw
            if isinstance(result_raw, dict)
            else (json.loads(result_raw) if isinstance(result_raw, str) else result_raw)
        )
        status_raw = await conn.fetchval("SELECT get_init_status() as status")
        status = (
            status_raw
            if isinstance(status_raw, dict)
            else (json.loads(status_raw) if isinstance(status_raw, str) else {})
        )
        consent_record = await _fetch_consent_record(
            conn, provider=provider, model=model, endpoint=endpoint
        )

    return JSONResponse(
        jsonable_encoder(
            {
                "decision": decision,
                "contract": payload,
                "result": result,
                "consent_record": consent_record,
                "status": status,
                "attempt_id": attempt_id,
                "exchange": {
                    "request_messages": messages,
                    "raw_content": raw_content,
                    "raw_tool_calls": raw_tool_calls,
                },
            }
        )
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Hexis API server")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=43817, help="Port (default: 43817)")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "apps.hexis_api:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
