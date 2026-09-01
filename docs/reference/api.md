<!--
title: API Reference
summary: FastAPI HTTP endpoints for Hexis
read_when:
  - "You want to use the HTTP API"
  - "You're building an integration against the API"
section: reference
-->

# API Reference

FastAPI server providing HTTP endpoints for chat, status, and events.

## Starting the Server

```bash
hexis api [--host HOST] [--port PORT]
```

Default: `127.0.0.1:43817`

## Authentication

Optional Bearer token via `HEXIS_API_KEY` environment variable. If unset, no auth is required.

```bash
curl -H "Authorization: Bearer <token>" http://localhost:43817/api/status
```

## Endpoints

### GET /health

Health check.

**Response**: `{"status": "ok", "checks": {"db": true}}`

### GET /api/status

Rich agent status.

**Response**: Full status payload including identity, memory counts, energy level, heartbeat info.

### POST /api/chat

Streaming chat via SSE. The primary conversation endpoint.

**Request body**:
```json
{
  "message": "Hello, how are you?",
  "history": [],
  "prompt_addenda": ""
}
```

**SSE events**:

| Event | Data | Description |
|-------|------|-------------|
| `phase_start` | `{"phase": "string"}` | Processing phase started |
| `phase_end` | `{"phase": "string"}` | Processing phase completed |
| `token` | `{"phase": "string", "text": "string"}` | Streaming text delta |
| `log` | `{"id", "kind", "title", "detail"}` | Tool call/result/memory log |
| `done` | `{"assistant": "full_text"}` | Completion signal |
| `error` | `{"message": "string"}` | Error occurred |

**Log kinds**: `tool_call`, `tool_result`, `memory_recall`, `memory_write`, `claim_flagged` (consistency check: the reply claimed an action with no matching successful tool call; a visible `[Correction]` is appended to the token stream)

### GET /v1/models

OpenAI-compatible model discovery. Hexis exposes the active `llm.chat` model
from the live database configuration rather than maintaining a separate model
list.

```bash
curl http://localhost:43817/v1/models
```

The response uses the OpenAI model-list shape and includes
`x_hexis_config_key` to show which configuration selected the model.

### POST /v1/chat/completions

OpenAI-compatible access to the same canonical agent, memory, skills, and tool
loop used by `/api/chat`.

```bash
curl http://localhost:43817/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<id from /v1/models>",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Set `"stream": true` for standard `chat.completion.chunk` SSE records followed
by `data: [DONE]`. `system`, `developer`, `user`, and `assistant` text messages
are supported; the final message must be a non-empty user message.

Supported generation controls are `temperature` and either `max_tokens` or
`max_completion_tokens`. Hexis rejects unsupported controls and client-supplied
tool-call history explicitly instead of silently ignoring them. Per-completion
token usage is omitted because one Hexis turn may contain several model/tool
iterations and the runtime does not yet attribute those tokens to one response.

OpenAI Python client example:

```python
from openai import OpenAI

client = OpenAI(api_key="unused", base_url="http://localhost:43817/v1")
model = client.models.list().data[0].id
answer = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What are you working on?"}],
)
print(answer.choices[0].message.content)
```

### POST /api/webhook/{source}

Accept external webhook events (e.g., from channels or external services).

**Response**: `{"status": "accepted", "event_id": "..."}`

### GET /api/events/stream

SSE stream of gateway events. Listens on PostgreSQL `pg_notify` for real-time updates.

### Voice

| Endpoint | Description |
|----------|-------------|
| `POST /api/voice/transcribe` | Transcribe one bounded foreground upload through the selected STT policy |
| `GET /api/voice/status` | Return live STT, TTS, Talk-mode, and local-provider readiness |
| `POST /api/voice/synthesize` | Synthesize bounded text through the enabled local provider and return uncached audio |
| `GET /api/voice/audio/{output_id}` | Retrieve one unexpired tool-created output by opaque UUID |

Synthesis is off by default. The server refuses remote or credential-bearing TTS
URLs. Generated audio responses use `Cache-Control: no-store`; metadata audits do
not contain the input text or audio. See [Voice and Talk Mode](../operations/voice.md).

### GET /api/integrations/status

Returns DB-owned connector manifests, saved connection references, recent setup
attempts, channel runtime, and backfill status. An optional `connector_id` query
parameter filters the result.

### POST /api/integrations/action

Runs the narrow set of connector setup, verification, completion, revocation, and
backfill controls used by the Connections page. It is not a generic tool-execution
endpoint. Provider-state writes such as creating a Notion page or changing Home
Assistant state are not exposed here; they remain normal approval-gated tools.

### GET /api/integrations/spotify/callback

Completes a pending Spotify Authorization Code + PKCE flow and returns a small HTML
success or recovery page. Register the exact URI shown by the setup flow; local HTTP
callbacks use a loopback IP rather than `localhost`.

### POST /api/init/consent/request

Trigger the consent flow for a model.

**Request body**:
```json
{
  "role": "conscious",
  "llm": {
    "provider": "openai-codex",
    "model": "gpt-5.2"
  }
}
```

**Response**: Consent decision, contract, and recorded certificate.

## CORS

Configurable via `HEXIS_CORS_ORIGINS` env var. Default: `localhost:3477`, `localhost:3000`.

## Related

- [Web UI](../guides/web-ui.md) -- the web UI that uses this API
- [Docker Compose](../operations/docker-compose.md) -- API server port mapping
