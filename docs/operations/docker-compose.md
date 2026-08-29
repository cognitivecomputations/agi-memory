<!--
title: Docker Compose
summary: Docker Compose profiles, services, ports, and overrides
read_when:
  - "You want to understand the Docker setup"
  - "You need to configure ports or profiles"
section: operations
-->

# Docker Compose

Hexis uses Docker Compose to manage PostgreSQL, workers, RabbitMQ, and optional services.

## Quick Start

```bash
hexis up                         # start the always-on brain, API, dashboard, delivery relay, and workers
hexis down                       # stop services
hexis ps                         # show running containers
hexis logs -f                    # tail logs
```

If launchd/systemd worker units have been installed with `hexis service
install`, these commands derive the Compose service list and exclude each
host-owned worker. They never start a second Docker copy. See
[Workers](workers.md) for migration and service-manager commands.

## Compose Files

| File | Used When |
|------|-----------|
| `./docker-compose.yml` | Source checkout |
| `./ops/docker-compose.runtime.yml` | pip install |

The CLI auto-detects which to use based on whether you're in a source tree.

## Profiles

| Profile | Services Added | Purpose |
|---------|---------------|---------|
| *(default)* | `db`, `rabbitmq`, `heartbeat_worker`, `maintenance_worker`, `api`, `ui`, `channel_worker` | Always-on brain, PWA/API, web-inbox relay, heartbeat, and memory maintenance |
| `active` | *(compatibility alias; core services already start by default)* | Existing channel setup commands may still pass this profile |
| `signal` | `signal-cli` | Signal messaging bridge (requires `SIGNAL_PHONE_NUMBER`) |
| `browser` | browserless chromium | Headless browser for web tools |

Combine profiles:

```bash
docker compose --profile active --profile browser up -d
```

## Port Mappings

All services bind to `127.0.0.1` by default. OSS has no application auth layer,
so do not bind them publicly. For another device, keep loopback and follow
[Secure Phone and PWA Access](secure-remote-access.md).

| Service | Container | Host Port | Internal Port |
|---------|-----------|-----------|---------------|
| PostgreSQL | `hexis_brain` | 43815 | 5432 |
| FastAPI | `hexis_api` | 43817 | 43817 |
| Web UI | `hexis_ui` | 3477 | 3477 |
| RabbitMQ AMQP | `hexis_rabbitmq` | 45672 | 5672 |
| RabbitMQ Management | `hexis_rabbitmq` | 45673 | 15672 |
| Browser CDP | `hexis_browser` | 49222 | 3000 |

If a port conflicts, set `POSTGRES_PORT` (or the relevant variable) in `.env`.

## Common Operations

```bash
# Start the default always-on stack
docker compose up -d

# Start only workers (DB already running)
docker compose up -d heartbeat_worker maintenance_worker

# Stop workers only
docker compose stop heartbeat_worker maintenance_worker

# Restart workers
docker compose restart heartbeat_worker maintenance_worker

# Deliberately rebuild after code or dependency changes
docker compose build
docker compose up -d

# View specific service logs
docker compose logs heartbeat_worker -f
docker compose logs db -f
```

## Dev Loop (watch mode)

`hexis dev` (or raw `docker compose watch`) keeps the running containers in
sync with the source tree — no manual `hexis upgrade` needed while it runs:

- Edits under `core/`, `services/`, `apps/`, `channels/`, `plugins/`, `skills/`,
  and `db/` sync into the containers and restart the affected services.
- Restarted workers apply any pending `db/migrations/*.sql` on startup, so
  schema deltas flow automatically too (never touches data).
- `pyproject.toml` or the committed `uv.lock` triggers an image rebuild. Update
  them together with `uv lock`; `uv lock --check` verifies no resolution drift.
- Ctrl+C stops watching; the stack keeps running.

This works because the worker images use an editable install rooted at `/app`
(see `ops/Dockerfile.worker`) — file sync is used instead of bind mounts, which
misbehave on macOS external drives. Outside watch mode, `hexis up` uses the
cached or published image matching the CLI version. Use `hexis up --build` for
a deliberate one-time source build; a normal start never falls through into a
surprise dependency build.

`ops/Dockerfile.worker` and `ops/Dockerfile.channels` use a digest-pinned uv
binary, `uv sync --locked`, bounded HTTP retries, and the hashes recorded in
`uv.lock`. A slow package index can delay a deliberate build, but it cannot make
the resolver select a different dependency graph.

`hexis dev` owns Docker workers so file synchronization and restart behavior
remain coherent. It refuses to start while any host worker is active; stop the
host units first with `hexis service stop`, or use the normal host-service path
outside watch mode.

## Overrides

Use `docker-compose.override.yml` for local customization:

```yaml
# Example: workers for multiple instances
services:
  worker_alice:
    extends:
      service: heartbeat_worker
    environment:
      HEXIS_INSTANCE: alice
```

## Linux Notes

The stack runs on native Linux Docker (both x86_64 and arm64 images are
published). Platform specifics:

- **`host.docker.internal`** does not resolve on native Linux Docker by
  itself; both compose files map it to the host gateway via `extra_hosts`,
  so the DB reaches the host embedding sidecar without configuration.
- **Embedding sidecar**: the `embeddinggemma` installer publishes
  `linux-x86_64` binaries (cpu/cuda/rocm/xpu variants) and `linux-arm64-cpu`
  (Graviton, Raspberry Pi, Ampere — CPU only; no CUDA/ROCm on ARM, and the
  installer auto-selects cpu there). All Linux platforms work out of the box.
- **`~/.hexis` ownership**: containers run as root, so files they write to
  the bind-mounted `~/.hexis` (OAuth token refresh) end up root-owned on
  native Linux. If host tools later fail to write there, reclaim it with
  `sudo chown -R "$USER" ~/.hexis`.
- **`hexis dev`** needs Docker Compose ≥ 2.22 (the `watch` subcommand) —
  current on any Docker CE install; distro-packaged older compose versions
  only affect watch mode, not `hexis up`.

## Default Credentials

| Service | User | Password |
|---------|------|----------|
| PostgreSQL | `hexis_user` | `hexis_password` |
| RabbitMQ | `hexis` | `hexis_password` |

Override via `POSTGRES_USER`, `POSTGRES_PASSWORD`, `RABBITMQ_DEFAULT_USER`, `RABBITMQ_DEFAULT_PASS` in `.env`.

## Related

- [Environment Variables](environment-variables.md) -- complete .env reference
- [Workers](workers.md) -- worker lifecycle
- [Database](database.md) -- schema management
