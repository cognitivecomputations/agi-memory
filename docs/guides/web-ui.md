<!--
title: Web UI
summary: Web dashboard for initialization, chat, and agent management
read_when:
  - "You want to use the web interface"
  - "You want to set up the UI"
section: guides
-->

# Web UI

Hexis includes a Next.js web dashboard for initialization, interactive chat, and agent management.

## Quick Start

```bash
hexis ui     # start the UI (container or local dev server, auto-detected)
hexis open   # open http://localhost:3477 in your browser
```

## Features

- **Init Wizard** -- 3-tier initialization flow (Express, Character, Custom) with character gallery
- **Interactive Chat** -- Streaming conversation with tool use visibility; attached files are read the moment they are attached, so the agent can answer about a PDF in the same message it arrives in (large pastes attach the same way, and each attachment has a private toggle)
- **Agent Status** -- Memory counts, energy level, heartbeat status
- **Memory Browser** -- Search and inspect distilled memories; memory detail links to the exact source documents and chunks behind it
- **Documents** -- The source-document filing cabinet: search files or passages (with page/section/sheet locators), preview with paging and a PDF page picker, see extraction warnings, and load sources onto the desk
- **Desk** -- Mid-term working material: read items window by window, pin what stays needed, clear the rest (cleared items archive; sources stay in the cabinet)
- **Ingest** -- Bulk ingestion: multi-file upload, paste box, URL field, and a live job list showing what ran, what is pending, and what failed and why
- **Character Gallery** -- Browse and select from 11 preset characters with portraits

## Init Wizard

The web UI shares the same 3-tier initialization flow as the CLI:

```
[LLM Config] -> [Choose Your Path] -> [Express | Character | Custom] -> [Consent] -> [Done]
```

1. **Models** -- Configure LLM provider and model
2. **Choose Your Path**: Express (quick defaults), Character (preset gallery), or Custom (full control)
3. **Consent** -- The agent reviews and decides whether to begin

## Running from Source

For local development with hot reload:

```bash
cd hexis-ui
bun install   # postinstall runs prisma generate automatically
```

Configure `hexis-ui/.env.local`:

```bash
DATABASE_URL=postgresql://hexis_user:hexis_password@127.0.0.1:43815/hexis_memory
HEXIS_LLM_CONSCIOUS_API_KEY=...      # set during init wizard
HEXIS_LLM_SUBCONSCIOUS_API_KEY=...   # optional
```

```bash
bun dev   # http://localhost:3477
```

## Architecture

The web UI uses a Next.js BFF (Backend for Frontend) with Prisma to call database functions directly. It does not go through the Python API layer.

## Related

- [First Agent](../start/first-agent.md) -- init wizard walkthrough
- [Character Cards](character-cards.md) -- character customization
- [Docker Compose](../operations/docker-compose.md) -- UI container configuration
