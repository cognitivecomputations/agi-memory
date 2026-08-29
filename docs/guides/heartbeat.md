<!--
title: Heartbeat
summary: Enable autonomous thinking and action for your agent
read_when:
  - "You want your agent to think on its own"
  - "You want to understand autonomous behavior"
section: guides
-->

# Heartbeat

The heartbeat is the agent's autonomous cognitive loop. When enabled, the agent wakes on its own, reviews goals, reflects on experience, and reaches out when it has something to say.

## Quick Start

```bash
# Start the always-on heartbeat and maintenance workers
hexis up

# Check status
hexis status
```

## How It Works

The heartbeat follows an OODA loop:

1. **Initialize** -- Regenerate from elapsed time, decay banked surplus, and apply the prior outcome multiplier
2. **Observe** -- Check environment, pending events, user presence
3. **Orient** -- Review goals, gather context (memories, clusters, identity, worldview)
4. **Decide** -- LLM call with action budget and context
5. **Act** -- Execute chosen actions within energy budget
6. **Record** -- Store heartbeat as episodic memory; the turn also mirrors into `subconscious_units` for the conscious-episode extraction sweep, and its final text passes the action-claim guardrail (unsupported "I did X" claims get a visible `[Correction]`)
7. **Wait** -- Schedule the next heartbeat from current drive urgency

### Energy Budget

Each action has an energy cost. The agent must decide what's worth doing within its budget:

| Cost | Actions |
|------|---------|
| **0** (free) | Observe, sense memory availability |
| **1** | Recall, remember, explore concepts |
| **2** | Web search, reflect |
| **3** | Shell, code execution |
| **5** | Send messages, slow ingest |

See [Energy Model](../reference/energy-model.md) for the complete cost table and philosophy.

The default normal reserve is 20 and the default bank capacity is 60. Energy
above the reserve decays with a 12-hour half-life. Ordinary beats spend from one
reserve; beats with actionable backlog can draw up to two reserves when that
energy has actually been banked. Useful durable outcomes improve the next
regeneration multiplier.

### Context Restrictions

The heartbeat context is more restricted than chat:

- `shell` and `write_file` are disabled by default
- Lower energy limits per tool call (default max: 5)
- The agent acts autonomously -- no user present to supervise

## Starting and Stopping

### Via Docker Compose

```bash
# Start everything (DB + RabbitMQ + workers)
docker compose up -d

# Start only workers (if DB is already running)
docker compose up -d heartbeat_worker maintenance_worker

# Stop workers (containers stay)
docker compose stop heartbeat_worker maintenance_worker

# Restart workers
docker compose restart heartbeat_worker maintenance_worker
```

### Via CLI

```bash
hexis start    # start workers
hexis stop     # stop workers
```

### Running Locally

```bash
hexis worker -- --mode heartbeat      # run heartbeat worker on host
hexis worker -- --mode maintenance    # run maintenance worker on host
```

## Pausing Without Stopping

Pause the heartbeat from the database without stopping containers:

```sql
-- Pause conscious decision-making
UPDATE heartbeat_state SET is_paused = TRUE WHERE id = 1;

-- Resume
UPDATE heartbeat_state SET is_paused = FALSE WHERE id = 1;
```

## Prerequisites

The heartbeat won't run until:

1. `agent.is_configured` is `true` (set by `hexis init`)
2. `is_init_complete` is `true`
3. The heartbeat is not paused

Once running, the database chooses the next time from live drive urgency. The
configured 60-minute interval is the baseline: quiet state waits longer (90
minutes by default), while high urgency can shorten the cadence to 15 minutes.

## Monitoring

```bash
hexis status              # shows heartbeat number, energy, last heartbeat time
hexis logs -f             # tail all worker logs
docker compose logs heartbeat_worker -f   # tail heartbeat worker specifically
```

```sql
-- Inspect recent heartbeats
SELECT heartbeat_number, started_at, narrative
FROM heartbeat_log
ORDER BY started_at DESC
LIMIT 20;
```

## Related

- [Energy Model](../reference/energy-model.md) -- energy budget mechanics and action costs
- [Heartbeat System](../concepts/heartbeat-system.md) -- architectural deep-dive
- [Workers](../operations/workers.md) -- worker lifecycle management
- [Scheduling](scheduling.md) -- schedule recurring tasks
