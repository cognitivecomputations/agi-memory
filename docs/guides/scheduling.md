<!--
title: Scheduling
summary: Set up cron tasks and active hours for your agent
read_when:
  - "You want to schedule recurring tasks"
  - "You want to configure active hours"
section: guides
-->

# Scheduling

Schedule recurring tasks and configure when your agent is active.

## Quick Start

```bash
# Create a daily task
hexis schedule create morning-briefing \
  --kind daily \
  --action queue_user_message \
  --payload '{"message": "Give me a morning briefing"}' \
  --schedule '{"time": "09:00"}' \
  --timezone "America/New_York"
```

## CLI Commands

```bash
# List scheduled tasks
hexis schedule list
hexis schedule list --status active --json

# Create a task
hexis schedule create <name> \
  --kind {once,interval,daily,weekly} \
  --action {queue_user_message,create_goal} \
  --schedule '<schedule_json>' \
  [--payload '<payload_json>'] \
  [--timezone '<timezone>'] \
  [--description '<description>']

# Delete a task
hexis schedule delete <task_id>
hexis schedule delete <task_id> --force   # hard delete
```

## Schedule Kinds

| Kind | Schedule JSON | Example |
|------|--------------|---------|
| `once` | `{"at": "2026-03-01T09:00:00"}` | One-time execution |
| `interval` | `{"seconds": 3600}` | Every hour |
| `daily` | `{"time": "09:00"}` | Every day at 9 AM |
| `weekly` | `{"day": "monday", "time": "09:00"}` | Every Monday at 9 AM |

## Actions

| Action | Payload | Description |
|--------|---------|-------------|
| `queue_user_message` | `{"message": "..."}` | Delivers the fixed message to the user through the configured outbox route |
| `create_goal` | `{"title": "...", "priority": "active"}` | Creates a new goal |

A scheduled `queue_user_message` is a reminder, not an agent invocation. It
does not dynamically run a skill, inspect Gmail, or generate a briefing. The
message should tell the user exactly what to open or ask Hexis to do next.

## Agent-Side Scheduling

The agent can also manage schedules via the `manage_schedule` tool during chat or heartbeats. This allows the agent to create, modify, and delete its own scheduled tasks.

## Automation Suggestions

Hexis can meet you halfway by proposing useful recurring reminders. A
suggestion is inert: it appears in the web inbox with **Accept** and **Not for
me**, and may also arrive over your most recently active private channel with a
numbered response and eight-character code.

- **Accept** creates exactly the stored `manage_schedule` task, using your
  current `agent.timezone` when the proposal does not specify one.
- **Not for me** is final for that suggestion's deduplication key, so a worker
  restart or later catalog scan cannot nag you again.
- If more than one suggestion is pending on a channel, include the code:
  `1 A1B2C3D4` accepts and `2 A1B2C3D4` dismisses.
- Suggestions can come from the starter catalog, a newly connected account,
  an installed skill's `blueprint:` block, or the separately opted-in
  recurring-usage review. None of those sources can schedule work directly.

The starter catalog offers morning briefing, evening wind-down, and weekly
review prompts. Connecting Gmail makes the important-mail check eligible.

## Active Hours

Configure when the agent should be active (affects heartbeat decisions, especially social actions):

```sql
-- Set active hours (agent won't initiate outreach outside these times)
SELECT set_config('agent.active_hours_start', '"09:00"'::jsonb);
SELECT set_config('agent.active_hours_end', '"22:00"'::jsonb);
SELECT set_config('agent.timezone', '"America/New_York"'::jsonb);
```

## Related

- [Heartbeat](heartbeat.md) -- how scheduled tasks integrate with the heartbeat
- [Goals and Backlog](goals-and-backlog.md) -- goal-driven scheduling
- [CLI reference](../reference/cli.md) -- full `hexis schedule` syntax
