<!--
title: Quickstart
summary: Get a running Hexis agent in 3 commands
read_when:
  - "You want the fastest path to a working agent"
  - "You just want to try Hexis"
section: start
-->

# Quickstart

Get a running agent in 3 commands.

## Prerequisites

- A running Docker daemon with Compose -- [Docker Desktop](https://docs.docker.com/get-docker/), or Docker CLI + [Colima](https://github.com/abiosoft/colima) on macOS
- Local embedding sidecar -- `hexis init` starts it and downloads the ~300M-parameter embedding model on first use
- For the default command below: a **ChatGPT Plus/Pro subscription** (browser OAuth, no API key). Without one, pick any provider from [Other Providers](#other-providers) instead.

## 3-Command Setup

```bash
curl -LsSf https://quixi.ai/hexis.sh | sh
hexis init --character hexis --provider openai-codex --model gpt-5.2
hexis chat
```

The install script sets up [uv](https://docs.astral.sh/uv/) if needed (uv brings its own Python), installs the `hexis` CLI into an isolated environment, and is safe to re-run (it upgrades). If `hexis` isn't found afterward, open a new terminal. Prefer `uv tool install hexis`, pipx, or pip-in-a-virtualenv? See [Installation](installation.md).

`hexis init` opens a browser window for login, starts the containers, pulls the embedding model, configures the character, and runs consent (the agent's recorded agreement to operate) -- all in one command.

**What success looks like:** init finishes with consent recorded; `hexis chat` greets you in character; `hexis status` reports a configured agent. Tell it your name, open a *new* chat, and ask -- it remembers.

**If it breaks:** `hexis doctor` diagnoses the usual suspects (Docker daemon down, embeddings unreachable, login incomplete). Then see [Troubleshooting](../operations/troubleshooting.md).

## Other Providers

**OAuth (no API key needed):**

```bash
# GitHub Copilot (device code login)
hexis init --character jarvis --provider github-copilot --model gpt-4o

# Chutes (free inference)
hexis init --character hexis --provider chutes --model deepseek-ai/DeepSeek-V3-0324

# Google Gemini CLI
hexis init --provider google-gemini-cli --model gemini-2.5-flash --character hexis

# Qwen Portal
hexis init --provider qwen-portal --model qwen-max-latest --character hexis
```

**API-key providers:**

```bash
# OpenAI Platform (auto-detect provider from key prefix)
hexis init --character jarvis --api-key sk-...

# Anthropic
hexis init --provider anthropic --model claude-sonnet-4-20250514 --api-key sk-ant-...

# Custom OpenAI-compatible endpoint
# Set OPENAI_BASE_URL and OPENAI_API_KEY in .env first (see .env.example).
hexis init --provider openai_compatible --model your-model-id --character hexis \
  --api-key-env OPENAI_API_KEY

# Express defaults (no character card)
hexis init --api-key sk-ant-...
```

`hexis init` auto-detects the provider from API key prefixes. `--api-key-env`
reads a named variable from `.env`, keeping the key out of shell history. For
all supported providers, see [Auth Providers](../integrations/auth/index.md).

## Verify It Worked

```bash
hexis status    # shows agent status, memory counts, energy level
hexis doctor    # checks Docker, DB, embedding service health
hexis demo      # proves recall, refusal, energy, and heartbeat, then rolls back
hexis maturity  # shows live capability levels and exact next steps
```

## Autonomous Heartbeat

```bash
hexis up
```

`hexis up` starts the heartbeat and maintenance workers by default. Once init and
consent are complete, the heartbeat uses a 60-minute base cadence, stretching to
90 minutes while quiet and shortening toward 15 minutes as drive urgency rises.

## What Just Happened

1. `hexis init` started the default stack (database, queue, heartbeat and maintenance workers, API, dashboard, and delivery relay), started the embedding sidecar, downloaded the embedding model if needed, configured your chosen character's identity/personality/values, and ran a consent flow where the agent agreed to begin.
2. `hexis chat` opened an interactive conversation with memory enrichment -- your messages are augmented with relevant memories, and the agent forms new memories from the conversation.

## Next Steps

- [Choose a character](../guides/character-cards.md) -- 11 presets or create your own
- [Ingest knowledge](../guides/ingestion.md) -- feed documents into memory
- [Enable the heartbeat](../guides/heartbeat.md) -- let the agent think autonomously
- [Set up messaging channels](../integrations/channels/index.md) -- Discord, Telegram, Slack, and more
- [Full installation guide](installation.md) -- .env configuration, source checkout
