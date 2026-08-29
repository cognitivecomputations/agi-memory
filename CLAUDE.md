# Repository Guidelines

## Project Overview

**Hexis** is an edge-native memory system that gives AI persistent identity, continuity, and autonomy. Core thesis: LLMs are intelligence engines but lack *selfhood*. Hexis wraps any LLM with a PostgreSQL-backed cognitive architecture providing:

- Multi-layered memory (episodic, semantic, procedural, strategic, working)
- Persistent identity and worldview
- Autonomous goal-pursuit (heartbeat system)
- Energy-based action budgeting
- Knowledge graphs for reasoning (Apache AGE)
- Consent, boundaries, and the ability to refuse

**Key principle**: The database is the brain, not just storage. State and logic live in Postgres; Python is a thin convenience layer.

## The Experience Bar (IMPORTANT — applies to every user-facing change)

A change is not "done" when it compiles or the test passes. It is done when the
end-to-end experience holds. Check every user-facing change against these before
calling it complete. Full text + rationale: `HEXIS_EXPERIENCE_BAR.md`.

1. **Derive from truth — never hardcode.** If a value has a live source (models,
   defaults, endpoints, versions), read it; don't hardcode a constant that goes stale.
2. **The user keeps control.** No destructive/irreversible action on a timer, by
   default, or without an explicit choice. No auto-exit, no auto-overwrite, no silent delete.
3. **Honor the medium.** Terminal ⇒ Ctrl+C exits, native copy/paste + scrollback,
   keyboard-first, no focus-hunting. Don't fight the platform; if a framework's
   defaults need constant overriding, it's the wrong tool.
4. **No dead-ends.** Every flow completes in place or hands the user the exact next
   step. Never "quit, run this other command, come back." Errors say what/why/next.
5. **Least surprise.** Never silently reuse ambient state the user didn't choose
   (env creds, other tools' logins, stale config). Surface it; don't consume it.
6. **Defaults are the expert's choice**, not the first constant that compiles.
7. **Whole journey, not the diff.** Drive the real path a user runs, start to finish,
   before calling it done.
8. **Fail loud, recover gracefully.** Advisory checks never block; failures show
   cause + fix — never a bare traceback, never a silent `except: pass`.

## Project Structure & Module Organization

```
hexis/
├── db/*.sql                # Split schema files (tables, functions, views, triggers)
├── core/                   # Fundamental interfaces (DB + LLM + messaging)
│   ├── cognitive_memory_api.py   # Main memory client (remember, recall, hydrate)
│   ├── agent_api.py              # Agent status and configuration
│   ├── agent_loop.py             # Unified agent loop (heartbeat + chat)
│   ├── memory_tools.py           # Memory tool definitions + handlers
│   ├── tools/                    # Tool system (ToolHandler ABC, registry, ~80 handlers)
│   ├── consent.py                # Consent DB wrappers
│   ├── subconscious.py           # Subconscious DB wrappers
│   ├── state.py                  # Heartbeat/maintenance DB wrappers
│   ├── llm.py                    # LLM provider abstraction
│   ├── usage.py                  # Token and cost tracking
│   └── rabbitmq_bridge.py        # Messaging bridge
├── services/               # Orchestration/workflows built on core
│   ├── conversation.py     # Conversation loop orchestration
│   ├── ingest.py           # Ingestion pipeline orchestration
│   ├── worker_service.py   # Heartbeat + maintenance loops
│   └── prompts/            # Markdown prompt templates
├── characters/             # Preset character cards (JSON + images)
├── apps/
│   ├── hexis_cli.py          # CLI entrypoint (hexis ...)
│   ├── hexis_init.py         # Interactive init wizard
│   ├── hexis_api.py          # FastAPI API server (SSE chat)
│   ├── hexis_mcp_server.py   # MCP tools server for LLMs
│   └── worker.py             # Heartbeat + maintenance workers
├── channels/               # Multi-platform messaging adapters
├── hexis-ui/               # Next.js web dashboard
├── plugins/                # Plugin system (extensibility framework)
├── skills/                 # Skill system (declarative skill definitions)
├── ops/                    # Dockerfiles and deployment scripts
├── tests/
│   ├── db/                 # Database integration tests
│   ├── core/               # Core API tests
│   ├── services/           # Service-level tests
│   └── cli/                # CLI smoke tests
├── docs/                   # index.md + concepts/ guides/ reference/ operations/ …
│   ├── philosophy/PHILOSOPHY.md   # Philosophical framework
│   └── memory_retention_design.md # Compression-native memory substrate design
└── docker-compose.yml      # Local stack (Postgres + workers; local embedding sidecar)
```

### Key Files

| File | Purpose |
|------|---------|
| `db/*.sql` | Database schema split across tables, functions, triggers, and views. Applied on fresh DB init. |
| `core/cognitive_memory_api.py` | Primary Python interface - `CognitiveMemory` class with `remember()`, `recall()`, `hydrate()`, `connect()` |
| `services/worker_service.py` | Stateless workers: `HeartbeatWorker` (conscious loop) + `MaintenanceWorker` (subconscious upkeep) |
| `apps/hexis_mcp_server.py` | Exposes memory operations as MCP tools for LLM integration |
| `apps/hexis_cli.py` | CLI commands: `up`, `down`, `init`, `chat`, `ui`, `open`, `ingest`, `mcp` |
| `apps/hexis_api.py` | FastAPI server with SSE chat streaming |

## Memory Architecture

### Memory Types
- **Episodic**: Events with action, context, result, emotional valence
- **Semantic**: Facts with confidence, sources, contradictions
- **Procedural**: How-to steps with success tracking
- **Strategic**: Patterns with supporting evidence
- **Working**: Transient short-term buffer with expiry

### Key Database Tables
- `memories` - Base table (id, type, content, embedding, importance, trust_level)
- `clusters` - Thematic groupings with centroid embeddings
- `memory_neighborhoods` - Precomputed associative neighbors (hot-path optimization)
- `memories` (type=`worldview`, `goal`) - Beliefs, boundaries, and goals stored as memories
- `external_calls` - Queue for LLM/embedding requests
- `memory_graph` (Apache AGE) - Graph nodes/edges for multi-hop reasoning

### Key Database Functions
- `fast_recall(text, limit)` - Primary hot-path retrieval (vector + neighborhood + temporal)
- `create_semantic_memory()`, `create_episodic_memory()`, etc.
- `get_embedding(text[])` - Generate embeddings via HTTP (cached in DB), returns vector[]
- `run_heartbeat()` - Autonomous cognitive loop
- `run_subconscious_maintenance()` - Background upkeep

## Build, Test, and Development Commands

```bash
# Start services (db + RabbitMQ + heartbeat/maintenance workers; embeddings via local sidecar)
docker compose up -d

# Apply a schema change WITHOUT losing data (add db/migrations/NNNN_*.sql first)
hexis migrate            # or `hexis upgrade` to also refresh images/code

# Dev loop: watch mode — code + migration edits apply to the running stack
# automatically (sync + restart; Ctrl+C stops watching, stack keeps running).
# `hexis up` uses cached/published images by default, including in a source
# checkout. Use `hexis up --build` for a deliberate build or `hexis dev` while editing.
hexis dev

# Python runtime image dependencies are resolved only from the committed uv.lock.
# After changing pyproject.toml, refresh and verify it before building.
uv lock
uv lock --check

# Wipe the DB volume — deliberate clean slate only (loses all data)
docker compose down -v && docker compose up -d

# Configure agent (gates heartbeats until done)
hexis init

# Run tests (expects Docker services up)
pytest tests -q           # All tests
pytest tests/db -q        # DB integration tests
pytest tests/core -q      # Core API tests
pytest tests/cli -q       # CLI smoke tests

# CI parity checks (GitHub Actions)
# - Required branch check: all-checks-pass
# - CI uses ops/ci/fake_embeddings.py instead of the real embedding model
# - Migration-survivor proves existing DB data survives hexis migrate

# Other CLI commands
hexis status              # Agent status
hexis chat                # Interactive chat
hexis ingest --input <docs>  # Batch knowledge ingestion
hexis mcp                 # Start MCP server
```

## Coding Style & Naming Conventions

- **Python**: Follow Black formatting; prefer type hints and explicit names
- **Database authority**: Add/modify SQL in `db/*.sql` rather than duplicating logic in Python
- **Additive schema changes**: Prefer backwards-compatible changes; avoid renames unless necessary
- **Stateless workers**: Workers can be killed/restarted without losing state; all state lives in Postgres

## Testing Guidelines

- **Framework**: `pytest` + `pytest-asyncio` (session loop scope)
- **Style**: Integration tests using transactions/rollbacks to avoid cross-test coupling
- **Naming**: `test_*` functions; use `get_test_identifier()` from `tests/utils.py` for unique data
- **Database tests**: Cover schema, workers, and database functions via asyncpg

## Commit & Pull Request Guidelines

- **Commits**: Short, imperative summaries (e.g., "Add MCP server tools", "Gate heartbeat on config")
- **Never add `Co-Authored-By` trailers** to commit messages
- **PRs**: Include rationale, how to run/verify, and any DB reset requirements
- **Call out changes to**: `db/*.sql`, `docker-compose.yml`, `README.md`

## Configuration & Safety Notes

- **Secrets**: Store API keys in environment variables (`.env`), not in Postgres; DB config stores env var *names* only
- **`core/auth/gmail-credentials.json` is intentional — do not "fix" it.** Hexis ships
  as a **desktop application**, and that file is an OAuth client for an *installed
  application*. Google's own documentation states the client secret for an installed
  app is not treated as confidential, and RFC 8252 ("OAuth 2.0 for Native Apps") says
  the same: a native client cannot keep a secret, so the security model does not rely
  on it. Packaging it via `pyproject.toml` `package-data` is the documented path. Code
  audits repeatedly flag this by applying a web-server threat model to a desktop
  client — **it has been reviewed and blessed. Do not rotate it, remove it from
  `package-data`, or rewrite git history over it.**
- **OSS has no authentication layer, and is not getting one.** API-key auth
  (`HEXIS_API_KEY`) is a **Hexis Pro** feature. OSS Hexis is a desktop application
  bound to `127.0.0.1`; "the dashboard has no auth" is the design, not a gap. Audits
  flag this by assuming a server product. The consequence that *does* matter: OSS
  remote access must be secured at the network layer only — a tailnet, or a reverse
  proxy supplying its own auth — never by exposing the dashboard directly.
- **Heartbeat gating**: Heartbeat is blocked until `agent.is_configured=true` (set via `hexis init`)
- **Consent flow**: Agent signs consent before first LLM use; consent is final and only ends via self-termination
- **Pause/terminate**: Heartbeat pauses must include a detailed reason queued to the outbox; self-termination must queue a last will to the outbox
- **Never revert or discard files without asking**: Do NOT run `git checkout`, `git restore`, `rm`, or any other destructive/irreversible file operation without explicit user confirmation first. Always ask before reverting, deleting, or overwriting files that have uncommitted changes.

## Architecture Principles

1. **Database is the Brain** - Not just storage; state and logic live in Postgres
2. **Stateless Workers** - Can be killed/restarted without losing anything
3. **ACID for Cognition** - Atomic memory updates ensure consistent state
4. **Embeddings as Implementation Detail** - App never sees them; DB handles caching
5. **Energy as Unified Constraint** - Balances compute cost, network load, user attention
6. **Precomputed Neighborhoods** - Hot path optimization for fast recall
7. **Schema Authority** - DB schema is source of truth; Python is convenience layer

## Heartbeat System (Autonomous Loop)

The heartbeat is the agent's conscious cognitive loop:

1. **Initialize** - Regenerate energy (+10/hour, max 20)
2. **Observe** - Check environment, pending events, user presence
3. **Orient** - Review goals, gather context (memories, clusters, identity, worldview)
4. **Decide** - LLM call with action budget and context
5. **Act** - Execute chosen actions within energy budget
6. **Record** - Store heartbeat as episodic memory
7. **Wait** - Sleep until next heartbeat

**Action costs**: Free (observe, remember) → Cheap (recall: 1, reflect: 2) → Expensive (reach out: 5-7)

## Debugging Tips

- **Schema changes not taking effect?** SQL files are baked into the Docker image -- see "Bouncing the Database" below
- **Heartbeat not running?** Check `agent.is_configured` via `hexis status` or run `hexis init`
- **Memory not found?** Check that the local embedding service is running and has the model available.
- **Test failures?** Ensure Docker services are up before running pytest; after a fresh `down -v`, wait for Postgres to accept connections. Use `POSTGRES_HOST=127.0.0.1` with pytest if localhost SSL negotiation flakes.

## Agent Operational Notes

### Python Virtual Environment

Always activate the venv before running any Python, pytest, or hexis CLI commands:

```bash
source /Users/eric/hexis/.venv/bin/activate
```

Prefix all shell commands with this activation. Example:

```bash
source /Users/eric/hexis/.venv/bin/activate && pytest tests -q
```

### Applying Schema Changes — migrations vs. a full bounce

`db/*.sql` is the **baseline** (baked into the image, applied by Postgres only on a
fresh/empty volume). To evolve an existing database **without losing data**, add a
forward delta — do NOT reach for `down -v`.

**The normal path (keeps data):**
1. Write the change as `db/migrations/NNNN_slug.sql` (additive + idempotent — see
   `db/migrations/README.md`; `-- migrate:no-transaction` for `ALTER TYPE ... ADD VALUE`).
2. Optionally mirror it into the baseline `db/*.sql` for greenfield clarity.
3. Apply it: **`hexis migrate`** (or `hexis upgrade` to also refresh images/code).
   Migrations also auto-apply on `hexis up` and on worker/API startup (advisory-locked) —
   `db/` ships inside the worker images, so containers migrate themselves on restart.
   In `hexis dev` watch mode, saving a migration file syncs it in and restarts the
   worker, which applies it — no command needed.

Back up / restore data with **`hexis backup`** and **`hexis restore <file>`** (AGE-aware).

**A full bounce (WIPES data — only for a deliberate clean slate, or when you edit the
baseline in a non-idempotent way during early dev):**

```bash
source /Users/eric/hexis/.venv/bin/activate && docker-compose down -v && docker-compose build db && docker-compose up -d
```

`down -v` removes the data volume (all memories/identity/goals) and re-inits from
`db/*.sql`; `hexis reset` wraps this same flow. Reserve it for when you *want* an
empty brain — schema changes to a live agent go through `hexis migrate`.

**Important**: The docker-compose service is named `db`, but the container is named `hexis_brain`. Always use the service name (`db`) with docker-compose commands (e.g., `docker-compose build db`), but use the container name with `docker exec` (e.g., `docker exec hexis_brain psql ...`).

### Verifying Schema Changes

After bouncing, verify your changes took effect:

```bash
# Check if a specific function exists
docker exec hexis_brain psql -U hexis_user -d hexis_memory -c "\df function_name"

# Check config keys
docker exec hexis_brain psql -U hexis_user -d hexis_memory -c "SELECT key, value FROM config WHERE key LIKE 'rlm.%'"

# Check table columns
docker exec hexis_brain psql -U hexis_user -d hexis_memory -c "SELECT column_name FROM information_schema.columns WHERE table_name = 'memories' ORDER BY ordinal_position"
```

### Docker Port Mapping

Default port mappings (all on `127.0.0.1`):

```
hexis_brain:       43815 -> 5432   (Postgres)
hexis_api:         43817 -> 43817  (FastAPI SSE)
hexis_ui:          3477  -> 3477   (Next.js dashboard)
hexis_rabbitmq:    45672 -> 5672   (AMQP)
hexis_rabbitmq:    45673 -> 15672  (RabbitMQ management)
hexis_browser:     49222 -> 3000   (Chrome CDP)
```

Default DB credentials: `hexis_user` / `hexis_password` / `hexis_memory`.

### Test Conventions

- **Loop scope**: All async tests using the `db_pool` fixture must use `loop_scope="session"` (not `"module"`):
  ```python
  pytestmark = [pytest.mark.asyncio(loop_scope="session")]
  ```

- **Seeding test memories**: The `memories` table has a NOT NULL constraint on `embedding`. Use `array_fill` to generate a dummy vector:
  ```python
  await conn.fetchval("""
      INSERT INTO memories (type, content, embedding, importance, trust_level, status)
      VALUES ('semantic', $1, array_fill(0.1, ARRAY[embedding_dimension()])::vector, 0.8, 0.9, 'active')
      RETURNING id
  """, content)
  ```
