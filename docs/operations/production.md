<!--
title: Production
summary: Cloud deployment and scaling for Hexis
read_when:
  - "You want to deploy Hexis to the cloud"
  - "You want to scale workers"
section: operations
-->

# Production

Deploy Hexis to a cloud environment with managed services.

## Architecture Overview

```
Managed Postgres  <--  N stateless workers  <--  App services
     (RDS/Cloud SQL)       (polling external_calls)    (CLI/API/MCP)
```

Workers are stateless -- scale them horizontally by running multiple instances. All state lives in Postgres.

## Managed PostgreSQL

Use any managed PostgreSQL service (AWS RDS, Google Cloud SQL, Azure Database, etc.):

- **Extensions required**: `pgvector`, `age` (Apache AGE), `btree_gist`, `pg_trgm`
- **Minimum version**: PostgreSQL 14+
- Check that your managed service supports Apache AGE -- not all do

### Schema Initialization

Apply schema files from `db/*.sql` in order:

```bash
for f in db/*.sql; do
  psql -h <host> -U <user> -d <database> -f "$f"
done
```

## Embedding Service

Options for production:

| Option | Pros | Cons |
|--------|------|------|
| **Local embedding sidecar on host** | Simple, fast for small scale | Single point of failure |
| **HuggingFace TEI** | Docker-based, scalable | CPU-only (float32) |
| **OpenAI Embeddings** | No infrastructure | Cost per request, latency |
| **vLLM / LiteLLM** | GPU support, OpenAI-compatible | More infrastructure |

## Workers

Run workers as long-lived processes (systemd, Docker, Kubernetes):

```bash
hexis-worker --mode heartbeat --instance production
hexis-worker --mode maintenance --instance production
```

For a single-user macOS or Linux host installed with uv, prefer the managed
desktop lifecycle instead of authoring units by hand:

```bash
hexis service install --env-file /path/to/production.env --enable-linger
hexis service status
```

`--enable-linger` applies only to Linux and is an explicit choice. Multi-user
servers and orchestrated deployments should continue to use deployment-owned
system units, containers, or Kubernetes workloads.

Key properties:
- **Stateless** -- kill and restart freely
- **Advisory locks** -- prevent double-execution
- **Horizontal scaling** -- run N heartbeat workers for N instances

## Security

- Set `HEXIS_API_KEY` for API server authentication
- Set `HEXIS_BIND_ADDRESS=127.0.0.1` and use a reverse proxy
- Store API keys in environment variables, not in the database
- Use separate database credentials per environment

## Scaling Considerations

- **Memory consolidation**: recommended every 4-6 hours (handled by maintenance worker)
- **Database optimization**: schedule during off-peak hours
- **Vector indexes**: monitor HNSW index performance with large datasets (10K+ memories)
- **Connection pooling**: use `HEXIS_POOL_MIN_SIZE` / `HEXIS_POOL_MAX_SIZE` to tune

## Related

- [Docker Compose](docker-compose.md) -- local development setup
- [Workers](workers.md) -- worker lifecycle
- [Environment Variables](environment-variables.md) -- configuration reference
