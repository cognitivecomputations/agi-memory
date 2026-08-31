<!--
title: Environment Variables
summary: Complete .env reference for all Hexis configuration
read_when:
  - "You want to configure Hexis via environment variables"
  - "You need to see all available settings"
section: operations
-->

# Environment Variables

All environment variables used by Hexis, configured via `.env`.

## Database

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DB` | `hexis_memory` | Database name |
| `POSTGRES_USER` | `hexis_user` | Database user |
| `POSTGRES_PASSWORD` | `hexis_password` | Database password |
| `POSTGRES_HOST` | `localhost` | Database host |
| `POSTGRES_PORT` | `43815` | Host port for PostgreSQL |

## Networking

| Variable | Default | Description |
|----------|---------|-------------|
| `HEXIS_BIND_ADDRESS` | `127.0.0.1` | Bind address for all services. Keep loopback in OSS; use a tailnet or authenticated reverse proxy for remote access. |
| `HEXIS_UI_PORT` | `3477` | Host dashboard port; `hexis tunnel` derives its loopback proxy target from this value |
| `HEXIS_UI_PUBLIC_URL` | *(auto-discovered)* | Authoritative private HTTPS dashboard URL for `hexis doctor` when Tailscale discovery is unavailable |
| `HEXIS_WEB_PUSH_VAPID_PRIVATE_KEY_FILE` | `~/.hexis/web-push-vapid-private.pem` | Persistent VAPID EC private-key path; created only after explicit notification setup |

## Embedding Service

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_SERVICE_URL` | `http://host.docker.internal:42666/api/embed` | HTTP endpoint for embeddings |
| `EMBEDDING_MODEL_ID` | `embeddinggemma:300m-qat-q4_0` | Model identifier |
| `EMBEDDING_DIMENSION` | `768` | Vector dimension |

## LLM Providers

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI Platform or OpenAI-compatible API key |
| `OPENAI_BASE_URL` | Base URL for `openai_compatible`; also used by cloud voice-note transcription after the user selects it |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `XAI_API_KEY` | xAI Grok API key |

These are only needed for API-key providers. OAuth providers store credentials in the database.

## Local Audio Analysis

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face token used only to download the configured pyannote diarization model after its terms are accepted |
| `HUGGING_FACE_HUB_TOKEN` | Accepted fallback name for `HF_TOKEN` |
| `XDG_CACHE_HOME` | Optional cache root; audio-analysis artifacts default to `$XDG_CACHE_HOME/hexis/audio-analysis` |

## Local Speech Output

| Variable | Default | Description |
|----------|---------|-------------|
| `HEXIS_TTS_URL` | Host: `http://127.0.0.1:42667`; Docker API: `http://host.docker.internal:42667` | Advanced Piper-compatible endpoint override. Only credential-free local HTTP hosts are accepted. |

## RabbitMQ

| Variable | Default | Description |
|----------|---------|-------------|
| `RABBITMQ_DEFAULT_USER` | `hexis` | RabbitMQ user |
| `RABBITMQ_DEFAULT_PASS` | `hexis_password` | RabbitMQ password |
| `RABBITMQ_MANAGEMENT_PORT` | `45673` | Host-mapped RabbitMQ management port used by host workers |
| `RABBITMQ_MANAGEMENT_URL` | `http://localhost:${RABBITMQ_MANAGEMENT_PORT}` | Exact management endpoint; Docker workers override this with the container endpoint |
| `RABBITMQ_USER` | `RABBITMQ_DEFAULT_USER` | Optional worker-specific user override |
| `RABBITMQ_PASSWORD` | `RABBITMQ_DEFAULT_PASS` | Optional worker-specific password override |

## Host Worker Services

| Variable | Default | Description |
|----------|---------|-------------|
| `HEXIS_ENV_FILE` | *(unset)* | Explicit environment file loaded before worker runtime imports. `hexis service install --env-file` writes only this path into the managed unit; it never copies file values. |

Normally you choose this through `hexis service install`; do not set it to a
different file in ambient shell startup unless that difference is intentional.

## API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HEXIS_API_KEY` | *(unset)* | Bearer token for API auth. If unset, no auth required. |

## Pool Sizes

| Variable | Default | Description |
|----------|---------|-------------|
| `HEXIS_POOL_MIN_SIZE` | Varies | Minimum DB connection pool size |
| `HEXIS_POOL_MAX_SIZE` | Varies | Maximum DB connection pool size |

## Instance Management

| Variable | Default | Description |
|----------|---------|-------------|
| `HEXIS_INSTANCE` | *(unset)* | Override active instance for any command |

## Channel Credentials

| Variable | Description |
|----------|-------------|
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `SLACK_BOT_TOKEN` | Slack bot token |
| `SLACK_APP_TOKEN` | Slack app-level token (Socket Mode) |
| `SIGNAL_PHONE_NUMBER` | Signal phone number |

## OAuth Provider Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_CLI_OAUTH_CLIENT_ID` | Google Gemini CLI OAuth client ID |
| `GEMINI_CLI_OAUTH_CLIENT_SECRET` | Google Gemini CLI OAuth client secret |
| `ANTIGRAVITY_OAUTH_CLIENT_ID` | Google Antigravity OAuth client ID |
| `ANTIGRAVITY_OAUTH_CLIENT_SECRET` | Google Antigravity OAuth client secret |

## External Service API Keys

The first-class Notion, Home Assistant, and Trello setup flows accept an
environment variable **name** chosen in the Connections page; they do not silently
consume a conventional variable. Names such as `NOTION_TOKEN`,
`HOME_ASSISTANT_TOKEN`, `TRELLO_API_KEY`, and `TRELLO_TOKEN` are examples only.
Spotify likewise uses `SPOTIFY_CLIENT_ID` only when the user explicitly selects
that name during setup. Weather requires no credential.

| Variable | Default | Description |
|----------|---------|-------------|
| `HEXIS_SPOTIFY_REDIRECT_URI` | *(derived)* | Exact Spotify OAuth callback override. HTTP callbacks must use `127.0.0.1` or `::1`; HTTPS callbacks may use a private reverse-proxy URL. |
| `HEXIS_API_URL` / `HEXIS_API_BASE_URL` | `http://127.0.0.1:43817` | API base used to derive the Spotify callback when no exact redirect override is set. |

| Variable | Description |
|----------|-------------|
| `TAVILY_API_KEY` | Tavily search API key |
| `BRAVE_SEARCH_API_KEY` | Brave Search API key |
| `SEARXNG_URL` | SearXNG base URL for self-hosted no-key web search |
| `FIRECRAWL_API_KEY` | Firecrawl scraping API key |
| `HUBSPOT_API_KEY` | HubSpot CRM API key |
| `TODOIST_API_KEY` | Todoist API key |
| `ASANA_API_KEY` | Asana API key |
| `YOUTUBE_API_KEY` | YouTube Data API key |
| `TWITTER_BEARER_TOKEN` | Twitter/X API bearer token |
| `FATHOM_API_KEY` | Fathom analytics API key |
| `STABILITY_API_KEY` | Stability AI API key |
| `RUNWAY_API_KEY` | Runway ML API key |
| `SENDGRID_API_KEY` | SendGrid email API key |

## Related

- [Installation](../start/installation.md) -- initial .env setup
- [Docker Compose](docker-compose.md) -- port mappings and profiles
