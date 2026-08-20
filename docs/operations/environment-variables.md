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
| `HEXIS_BIND_ADDRESS` | `127.0.0.1` | Bind address for all services (set to `0.0.0.0` to expose) |

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
| `OPENAI_BASE_URL` | Base URL for `openai_compatible`; used as the `hexis init` endpoint default only when that provider is selected |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `XAI_API_KEY` | xAI Grok API key |

These are only needed for API-key providers. OAuth providers store credentials in the database.

## RabbitMQ

| Variable | Default | Description |
|----------|---------|-------------|
| `RABBITMQ_DEFAULT_USER` | `hexis` | RabbitMQ user |
| `RABBITMQ_DEFAULT_PASS` | `hexis_password` | RabbitMQ password |

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
