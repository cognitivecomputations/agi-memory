<!--
title: Workers
summary: Heartbeat and maintenance worker lifecycle management
read_when:
  - "You want to start, stop, or monitor workers"
  - "You want to understand worker architecture"
section: operations
-->

# Workers

Hexis has three independent background workers that drive autonomous behavior.

## Worker Types

| Worker | Purpose | Schedule |
|--------|---------|----------|
| **Heartbeat** (conscious) | Polls `external_calls`, triggers heartbeats | `should_run_heartbeat()` |
| **Maintenance** (subconscious) | Substrate upkeep, outbox/inbox bridging | `should_run_maintenance()` |
| **Channel** | Bridges messaging platforms to RabbitMQ | Persistent connections |

All workers are **stateless** -- they can be killed and restarted without losing anything. All state lives in Postgres.

## Starting and Stopping

### Host user services (recommended)

Keep PostgreSQL and RabbitMQ in Docker while launchd on macOS or the systemd
user manager on Linux owns the stateless Python workers:

```bash
hexis up
hexis service install --replace-docker-workers
hexis service status
```

The migration command installs and enables the host units before stopping the
matching Docker workers, then starts the host copies. If host startup fails,
Hexis stops the attempted host copies and restores the previous Docker workers.
It refuses to proceed when it cannot determine whether Docker workers are
running. This ensures each worker has one owner.

The CLI selects the same `.env` file as the stack. You can make that choice
explicit without copying its values into service definitions:

```bash
hexis service install --env-file /path/to/.env --replace-docker-workers
```

Service lifecycle commands are:

```bash
hexis service status [--json]
hexis service start [heartbeat maintenance channels]
hexis service stop [heartbeat maintenance channels]
hexis service restart [heartbeat maintenance channels]
hexis service logs [-f] [heartbeat maintenance channels]
hexis service uninstall [--yes]
```

The channel worker remains opt-in for host ownership; add `--channels` during
installation to migrate it too. On Linux, `--enable-linger` explicitly allows
the systemd user manager to survive logout. Without it, Hexis reports the live
linger status and leaves that system setting under your control. On macOS the
units live in `~/Library/LaunchAgents`; on Linux they live in
`~/.config/systemd/user`. `hexis service uninstall` preserves worker logs.

Once installed, `hexis up`, `down`, `start`, `stop`, `reset`, and `upgrade`
coordinate host workers and omit their Docker copies. `hexis upgrade` restarts
them onto the updated Python package after migrations. `hexis uninstall`
removes managed units before removing the CLI. Source watch mode deliberately
refuses to start while host workers are active because it owns Docker workers.

### Docker Compose alternative

```bash
# Start the default background stack
docker compose up -d

# Start specific default workers
docker compose up -d heartbeat_worker maintenance_worker

# Stop workers (containers stay)
docker compose stop heartbeat_worker maintenance_worker

# Restart
docker compose restart heartbeat_worker maintenance_worker
```

### Via CLI

```bash
hexis up       # start DB, queue, heartbeat worker, and maintenance worker
hexis start    # start workers manually if they were stopped
hexis stop     # stop workers
```

Without host units, these commands retain the Docker-only behavior. The channel
delivery relay is part of the default stack.

### Running Locally

Run workers on the host machine (connects to Postgres over TCP):

```bash
hexis worker -- --mode heartbeat
hexis worker -- --mode maintenance
hexis worker -- --mode both

# For a specific instance
hexis worker -- --instance myagent --mode heartbeat
```

Or directly:

```bash
hexis-worker --mode heartbeat
hexis-worker --mode maintenance
```

## Heartbeat Worker

The heartbeat worker drives the agent's conscious cognitive loop:

1. Checks `should_run_heartbeat()` on a polling interval
2. Opens a beat in Postgres, which regenerates/decays energy and gathers context
3. Runs the energy-bounded agent loop and its tool calls
4. Derives useful outcomes from durable tool receipts
5. Deducts exact spend, records the episode, and stores the urgency-adjusted next due time

### Prerequisites

The heartbeat won't run until:
- `agent.is_configured = true` (set by `hexis init`)
- `is_init_complete = true`
- Heartbeat is not paused

### Pausing from the DB

```sql
-- Pause (without stopping containers)
UPDATE heartbeat_state SET is_paused = TRUE WHERE id = 1;

-- Resume
UPDATE heartbeat_state SET is_paused = FALSE WHERE id = 1;
```

## Maintenance Worker

The maintenance worker handles subconscious upkeep:

- **Working memory cleanup** -- promotes or deletes expired items
- **Neighborhood recomputation** -- refreshes stale precomputed neighbors
- **Embedding cache pruning** -- cleans old cached embeddings
- **Outbox/inbox bridging** -- publishes outbox messages to RabbitMQ, ingests inbox messages
- **Conscious-episode extraction** -- sweeps recent chat turns and heartbeat episodes (`subconscious_units`) and selectively promotes salient facts into durable memories; one LLM call per batch, importance floor `extraction.min_importance`, duplicates corroborate existing beliefs. Gated by `extraction.enabled` (default on)
- **Origin-memory seeding** -- idempotently keeps the protected origin-story memories seeded (`origin_memories.enabled`, default on); a flipped flag takes effect on the next tick

Note: the workers no longer eagerly connect MCP servers at startup — MCP is
skill-gated by default (`mcp.skill_gated`) and connects on skill activation.

### Pausing

```sql
UPDATE maintenance_state SET is_paused = TRUE WHERE id = 1;
UPDATE maintenance_state SET is_paused = FALSE WHERE id = 1;
```

### Alternative Scheduling

If you don't want the maintenance worker, schedule directly:

```sql
SELECT run_subconscious_maintenance();
```

The function uses an advisory lock, so multiple schedulers won't double-run.

## Outbox and RabbitMQ

The maintenance worker bridges outbox/inbox:

- Publishes pending `outbox_messages` to `hexis.outbox` RabbitMQ queue
- Polls `hexis.inbox` and inserts messages into working memory

RabbitMQ details:
- Management UI: `http://localhost:45673`
- AMQP: `amqp://localhost:45672`
- Credentials: `hexis` / `hexis_password`

## Monitoring

```bash
hexis status                          # heartbeat number, energy, last run
hexis service status                  # launchd/systemd process state
hexis service logs -f                 # launchd/systemd worker logs
hexis logs -f                         # Docker logs
docker compose logs heartbeat_worker -f
docker compose logs maintenance_worker -f
```

## Related

- [Heartbeat guide](../guides/heartbeat.md) -- enabling autonomous behavior
- [Docker Compose](docker-compose.md) -- profiles and services
- [Troubleshooting](troubleshooting.md) -- diagnosing worker issues
