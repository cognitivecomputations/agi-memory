<!--
title: Contributing
summary: Development setup, coding style, and contribution guidelines
read_when:
  - "You want to contribute to Hexis"
  - "You need to set up a development environment"
section: contributing
-->

# Contributing

## Development Setup

```bash
git clone https://github.com/QuixiAI/Hexis.git && cd Hexis
uv sync --locked --inexact
source .venv/bin/activate
cp .env.example .env   # edit with your settings; never commit .env
hexis up             # start services
hexis doctor         # verify health
```

No uv? A plain virtualenv works too: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`

## Coding Style

- **Python**: Follow Black formatting; prefer type hints and explicit names
- **Database authority**: Add/modify SQL in `db/*.sql` rather than duplicating logic in Python
- **Additive schema changes**: Prefer backwards-compatible changes; avoid renames unless necessary
- **Stateless workers**: Workers can be killed/restarted without losing state; all state lives in Postgres

## Project Structure

```
hexis/
├── db/*.sql          # Schema files (tables, functions, triggers, views)
├── core/             # Thin DB + LLM adapter
│   └── tools/        # ~80 tool handlers across 11 categories
├── services/         # Orchestration (conversation, ingestion, workers)
├── apps/             # CLI, API server, MCP server, workers
├── channels/         # Messaging adapters
├── characters/       # Preset character cards
├── skills/           # Declarative workflow packages
├── plugins/          # Plugin system
├── tests/            # pytest test suite
└── docs/             # Documentation
```

## Commit Guidelines

- Short, imperative summaries (e.g., "Add MCP server tools", "Gate heartbeat on config")
- Include rationale, how to run/verify, and any DB reset requirements in PR descriptions
- Call out changes to `db/*.sql`, `docker-compose.yml`, `README.md`

## Testing

See [Testing](testing.md) for test conventions, running tests, and writing new tests.

## PyPI Package

Hexis is published as the `hexis` package on PyPI.

### Publishing manually

```bash
# Bump version in pyproject.toml first, then:
python -m pip install --upgrade build twine
rm -rf dist
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

`twine` authenticates via `~/.pypirc` or the `TWINE_USERNAME`/`TWINE_PASSWORD` environment variables. Version tags are optional and do not trigger publish automation.

## Docker Images

Hexis ships 4 Docker images, all published to `ghcr.io/quixiai/`:

| Image | Dockerfile | Base | Contents |
|-------|-----------|------|----------|
| `hexis-brain` | `ops/Dockerfile.db` | `postgres:16-bullseye` | Postgres + pgvector + pgsql-http + Apache AGE + schema (`db/*.sql`) |
| `hexis-worker` | `ops/Dockerfile.worker` | `python:3.12-slim` | Heartbeat worker, maintenance worker, and API server |
| `hexis-channels` | `ops/Dockerfile.channels` | `python:3.12-slim` | Channel adapters + messaging library dependencies |
| `hexis-ui` | `ops/Dockerfile.ui` | `node:20-slim` | Next.js web dashboard (multi-stage build) |

### Building locally

```bash
# Build all images (used by docker-compose.yml for local dev)
docker compose build

# Build a single image
docker compose build db          # hexis-brain
docker compose build heartbeat_worker  # hexis-worker
docker compose build channel_worker    # hexis-channels

# Build with a tag for manual testing
docker build -f ops/Dockerfile.db -t ghcr.io/quixiai/hexis-brain:dev .
docker build -f ops/Dockerfile.worker -t ghcr.io/quixiai/hexis-worker:dev .
docker build -f ops/Dockerfile.channels -t ghcr.io/quixiai/hexis-channels:dev .
docker build -f ops/Dockerfile.ui -t ghcr.io/quixiai/hexis-ui:dev .
```

The `hexis-brain` image takes the longest to build because it compiles pgvector, pgsql-http, and Apache AGE from source.
The Python images install their runtime and channel dependencies from the
committed `uv.lock`; after changing `pyproject.toml`, run `uv lock` and commit
both files. `uv lock --check` is the local/CI drift check.

### Publishing manually

Docker image publishing is manual.

```bash
# Authenticate with GHCR
export VERSION=<version>
export GHCR_USERNAME=<github-username>
export GHCR_TOKEN=<github-token-with-write-packages>
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin

# Build versioned images
docker build -f ops/Dockerfile.db -t ghcr.io/quixiai/hexis-brain:$VERSION .
docker build -f ops/Dockerfile.worker -t ghcr.io/quixiai/hexis-worker:$VERSION .
docker build -f ops/Dockerfile.channels -t ghcr.io/quixiai/hexis-channels:$VERSION .
docker build -f ops/Dockerfile.ui -t ghcr.io/quixiai/hexis-ui:$VERSION .

# Push versioned tags
docker push ghcr.io/quixiai/hexis-brain:$VERSION
docker push ghcr.io/quixiai/hexis-worker:$VERSION
docker push ghcr.io/quixiai/hexis-channels:$VERSION
docker push ghcr.io/quixiai/hexis-ui:$VERSION

# Update latest tags
docker tag ghcr.io/quixiai/hexis-brain:$VERSION ghcr.io/quixiai/hexis-brain:latest
docker tag ghcr.io/quixiai/hexis-worker:$VERSION ghcr.io/quixiai/hexis-worker:latest
docker tag ghcr.io/quixiai/hexis-channels:$VERSION ghcr.io/quixiai/hexis-channels:latest
docker tag ghcr.io/quixiai/hexis-ui:$VERSION ghcr.io/quixiai/hexis-ui:latest
docker push ghcr.io/quixiai/hexis-brain:latest
docker push ghcr.io/quixiai/hexis-worker:latest
docker push ghcr.io/quixiai/hexis-channels:latest
docker push ghcr.io/quixiai/hexis-ui:latest
```

### Runtime vs dev compose files

- **`docker-compose.yml`** -- used for local development; has `build:` directives that build from source
- **`ops/docker-compose.runtime.yml`** -- used by `hexis up` when installed via pip; references pre-built `ghcr.io/quixiai/*:latest` images

### Applying schema changes

`db/*.sql` is the fresh-install baseline. Existing databases must evolve through
additive migrations in `db/migrations/`.

```bash
hexis migrate
```

Use `docker compose down -v` only for a deliberate clean slate; it removes all
memories, identity, and goals. CI has a migration-survivor lane that verifies
existing data survives the migration runner.

## Key Principles

1. **Database is the brain** -- state and logic live in Postgres
2. **Schema authority** -- `db/*.sql` is the source of truth
3. **Stateless workers** -- can be killed/restarted without losing anything
4. **ACID for cognition** -- atomic memory updates ensure consistent state
5. **Core is the mind; capability lives at the edges** -- before adding
   anything to `core/`, walk the footprint ladder in
   [CONTRIBUTING.md](../../CONTRIBUTING.md#where-new-capability-belongs--the-footprint-ladder)
   (extend existing → skill → gated tool → plugin → MCP → core tool last)
