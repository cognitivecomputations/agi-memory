<!--
title: Prerequisites
summary: Software requirements for running Hexis
read_when:
  - "You want to know what to install before Hexis"
  - "You're troubleshooting a missing dependency"
section: start
-->

# Prerequisites

What you need before installing Hexis.

## Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| Docker Engine + Compose | Docker 20.10+, Compose v2+ | Runs PostgreSQL (the agent's brain). Use [Docker Desktop](https://docs.docker.com/get-docker/) or the Docker CLI with [Colima](https://github.com/abiosoft/colima) on macOS |
| [uv](https://docs.astral.sh/uv/) | Current | Installs the Hexis CLI into an isolated environment; downloads Python automatically if missing. The [install script](installation.md) sets it up for you |
| Local embedding sidecar | Current | Generates embeddings for memory storage |

Managing Python yourself instead of using uv? pipx or a virtualenv with Python 3.10+ works too — see [Installation](installation.md).

## Verify Installation

Installed is not enough — Docker's daemon and the embedding sidecar must both be **running** when you start `hexis init`:

```bash
docker --version          # Docker version 20.10+
docker info               # daemon is running (Docker Desktop, Colima, or Docker Engine)
docker compose version    # Docker Compose v2+
embeddinggemma --help
uv --version              # or: python3 --version (3.10+) if managing Python yourself
```

`hexis init` starts the local embedding sidecar and downloads the ~300M-parameter embedding model on first use. Set `EMBEDDING_SERVICE_URL` only if you are intentionally pointing Hexis at a different embedding service.

## LLM Provider

You need access to at least one LLM provider. Hexis supports:

| Provider | Auth Type | Cost |
|----------|-----------|------|
| ChatGPT (Codex OAuth) | Browser OAuth | ChatGPT Plus/Pro subscription |
| GitHub Copilot | Device code | Copilot subscription |
| Chutes | Browser OAuth | Free |
| Google Gemini CLI | Browser OAuth | Free tier available |
| Qwen Portal | Device code | Free tier available |
| MiniMax Portal | User code | Free tier available |
| OpenAI Platform | API key | Pay-per-use |
| Anthropic | API key or setup token | Pay-per-use or Claude subscription |
| Local OpenAI-compatible endpoint | Optional API key | Varies |

See [Auth Providers](../integrations/auth/index.md) for setup details on each provider.

## Optional

| Dependency | Purpose |
|------------|---------|
| [Node.js / Bun](https://bun.sh/) | Running the web UI from source |
| [RabbitMQ](https://www.rabbitmq.com/) | Included in Docker stack; only needed if running externally |
| Git | Cloning the repo for source development |

## Platform Support

Hexis runs on macOS, Linux, and Windows (via WSL2). It uses the active Docker context, so Docker Desktop and Colima both work without project-specific Compose changes.

## Next Steps

- [Installation](installation.md) -- install Hexis with uv, pipx, or pip, or from source
