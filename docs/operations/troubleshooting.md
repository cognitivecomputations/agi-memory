<!--
title: Troubleshooting
summary: Symptom-based diagnosis and fixes for common Hexis issues
read_when:
  - "Something isn't working"
  - "You're getting an error"
section: operations
-->

# Troubleshooting

Diagnose and fix common Hexis issues.

## Diagnostic Ladder

Run these commands in order to identify the problem:

```bash
hexis doctor                     # 1. Overall health check
hexis status                     # 2. Agent status
docker compose ps                # 3. Container status
docker compose logs db           # 4. Database logs
docker compose logs heartbeat_worker   # 5. Worker logs
```

## Decision Tree

```
Problem?
├── Can't start Hexis
│   ├── Docker not running → Start Docker Desktop
│   ├── Port conflict → Change POSTGRES_PORT in .env
│   └── Embedding service not running → Start the local sidecar
│
├── Database connection errors
│   ├── Container not running → docker compose up -d
│   ├── Wrong port → Check POSTGRES_PORT in .env
│   └── Extensions missing → hexis reset
│
├── Heartbeat not running
│   ├── Not configured → hexis init
│   ├── Paused → Check heartbeat_state.is_paused
│   ├── Workers not started → hexis up
│   └── No energy → Wait for regeneration
│
├── Memory search returns nothing
│   ├── Embedding service not running → Start the local sidecar
│   ├── Wrong dimension → hexis reset after fixing EMBEDDING_DIMENSION
│   └── No memories stored → Ingest content first
│
├── Schema changes not taking effect
│   └── SQL baked into image → docker compose down -v && docker compose build db && docker compose up -d
│
└── Auth issues
    ├── Provider not configured → hexis auth <provider> login
    ├── Credentials expired → Re-run login
    └── Callback won't bind → Use --manual flag
```

## Common Issues

### Upgrade Keeps Re-installing the Same Version

**Symptoms**: `hexis upgrade` pulls the images you already had; `hexis --version` doesn't change. On hexis ≤ 1.0.11 the dashboard also fails with a Prisma `libssl.so.1.1` error (a packaging bug in those UI images).

Image tags are pinned to the CLI's own version, so the stack can only move when the CLI package moves. Versions up to 1.0.11 could not update themselves when installed with uv or pipx (their tool venvs ship no `pip`), and quietly continued with the old version.

**Fixes**:
- Move the package with the tool that installed it, then upgrade the stack:

```bash
uv tool install --force hexis     # uv installs (the default)
# or: pipx install --force hexis  # pipx installs
# or: pip install -U hexis        # plain pip / virtualenv installs

hexis upgrade
```

- From 1.0.13 on, `hexis upgrade` detects the install method and does this itself — and stops loudly with the exact command if the package still can't move.

### Database Connection Errors

**Symptoms**: `connection refused`, `could not connect to server`

```bash
# Check container status
docker compose ps

# Check logs
docker compose logs db

# Verify port
docker exec hexis_brain psql -U hexis_user -d hexis_memory -c "SELECT 1"
```

**Fixes**:
- Start the container: `docker compose up -d`
- Check port conflicts: change `POSTGRES_PORT` in `.env`
- Verify extensions: `hexis doctor`

### Heartbeat Not Running

**Symptoms**: `hexis status` shows no recent heartbeats

```bash
# Check if configured
hexis status    # look for "is_configured: true"

# Check if paused
docker exec hexis_brain psql -U hexis_user -d hexis_memory \
  -c "SELECT is_paused FROM heartbeat_state WHERE id = 1"

# Check worker status
docker compose ps | grep heartbeat
```

**Fixes**:
- Run `hexis init` if not configured
- Unpause: `UPDATE heartbeat_state SET is_paused = FALSE WHERE id = 1`
- Start workers: `docker compose up -d heartbeat_worker maintenance_worker`

### Memory Search Returns Nothing

**Symptoms**: `hexis recall` returns no results

```bash
# Check embedding service
hexis doctor

# Check that the local embedding sidecar starts
embeddinggemma --help

# Check memory count
docker exec hexis_brain psql -U hexis_user -d hexis_memory \
  -c "SELECT type, count(*) FROM memories WHERE status='active' GROUP BY type"
```

**Fixes**:
- Start the sidecar: `embeddinggemma`
- Let the sidecar download `embeddinggemma-300M-qat-Q4_0.gguf` on first use
- Ingest content: `hexis ingest --file ./notes.md`

### Schema Changes Not Taking Effect

**Symptoms**: New SQL functions or columns are missing after editing `db/*.sql`

SQL files are baked into the Docker image at build time. Editing on disk does nothing to the running container.

**Fix**:

```bash
docker compose down -v && docker compose build db && docker compose up -d
```

This destroys all data. Export first if needed.

### Memory Search Performance

**Symptoms**: Slow recall queries

```bash
# Check memory health
docker exec hexis_brain psql -U hexis_user -d hexis_memory \
  -c "SELECT * FROM memory_health"
```

**Fixes**:
- Ensure maintenance worker is running (recomputes neighborhoods)
- Check HNSW index status
- Consider memory pruning for very large datasets

### Auth Provider Issues

**"Provider X is not configured"** -- Run `hexis auth <provider> login`

**"Credentials expired"** -- Re-run `hexis auth <provider> login`

**"Callback server won't bind"** -- Use `--manual` to paste the redirect URL

**`hexis doctor` shows auth warnings** -- Run `hexis auth <provider> status` to check health

### Test Failures

```bash
# Ensure services are up
hexis up
hexis doctor

# Run with explicit host (avoids SSL flakes)
POSTGRES_HOST=127.0.0.1 pytest tests -q
```

## Related

- [Docker Compose](docker-compose.md) -- profiles and services
- [Workers](workers.md) -- worker lifecycle
- [Database](database.md) -- schema management
- [Embeddings](embeddings.md) -- embedding service configuration
