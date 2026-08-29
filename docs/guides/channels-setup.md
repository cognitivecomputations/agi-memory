<!--
title: Channels Setup
summary: General pattern for connecting messaging platforms
read_when:
  - "You want to connect a messaging channel"
  - "You want to understand how channels work"
section: guides
-->

# Channels Setup

Connect your Hexis agent to messaging platforms (Discord, Telegram, Slack, Signal, WhatsApp, iMessage, Matrix).

## Quick Start

```bash
# Configure a channel
hexis channels setup discord

# Start additional live channel workers (part of active profile)
hexis up

# Check status
hexis channels status
```

## General Setup Pattern

All channels follow the same three-step pattern:

### 1. Configure Credentials

```bash
hexis channels setup <channel>
```

This interactive command stores the required credentials (bot token, phone number, etc.) in the database config.

### 2. Start the Channel Worker

The channel worker runs as part of the `active` Docker Compose profile:

```bash
docker compose up -d
```

Or start a specific channel:

```bash
hexis channels start --channel discord
```

### 3. Verify

```bash
hexis channels status          # show session counts per channel
hexis channels status --json   # JSON output
```

### 4. Identify the Operator (Optional)

Each setup wizard can record your exact platform user ID or phone number. In a
private conversation, that identity may create standing instructions such as
“Always cite the source.” Other allowed participants can chat normally but
cannot write, inspect, or revoke operator policy. Group messages never gain
operator authority, even when you sent them.

Slack uses `channel.slack.operator_user_id`; iMessage uses
`channel.imessage.operator_recipient`; the other channels use
`channel.<type>.operator_user_id`. Leave the prompt blank if that channel
should never manage standing policy.

## Optional: Central Inbound Routing

By default, the established adapter allowlists and reply behavior remain active. To
make inbound routing centrally auditable, enable the DB-owned policy:

```sql
SELECT set_config('channel.disposition.enabled', 'true'::jsonb);
```

The change applies to running channel workers. Allowed direct conversations continue
normally. Messages outside the live channel allowlist are observed without a reply;
empty messages drop; configured trigger words and native mentions engage; and a fresh
identity-verified operator correction can wake an unpaused heartbeat. Every decision
is visible in `inbound_disposition_events`.

Trigger and continuation gates are opt-in per channel. For example:

```sql
SELECT set_config('channel.imessage.disposition.trigger_word', '"hexis"'::jsonb);
SELECT set_config(
  'channel.imessage.disposition.continuation_window_seconds',
  '300'::jsonb
);
```

Set the master switch back to `false` at any time to restore legacy routing. No worker
restart is required in either direction.

## Outbound Safety

Every person-facing send—provider tool, formal outbox delivery, or direct channel
reply—passes the same database-owned policy. Third-party sends require a backed
purpose, draw from a per-person/per-channel attention budget, and identify the agent
and principal. Replies are free. A recipient's STOP applies immediately across every
known channel; START or UNSTOP is the only way to reverse it.

Open **Outbound** in the dashboard to inspect purposes, costs, disclosure form,
delivery outcomes, unanswered-contact counts, and budget strain. The same page can
pause all outbound communication or one person without erasing recipient opt-outs.
See [Outbound Safety](outbound-safety.md) for the full operator workflow.

## Architecture

```
Platform API  <-->  Channel Adapter  <-->  RabbitMQ  <-->  Agent Loop
```

Each adapter:
- Maintains a persistent connection to the platform
- Routes incoming messages through RabbitMQ to the agent's conversation loop
- Delivers the agent's responses back to the platform
- Tracks conversation sessions per user/channel

## Supported Channels

| Channel | Required | Notes |
|---------|----------|-------|
| Discord | Bot token | Create at discord.com/developers |
| Telegram | Bot token | Create via @BotFather |
| Slack | Bot token + App token | Create a Slack app with Socket Mode |
| Signal | Phone number | Requires `signal` Docker profile |
| WhatsApp | Phone number | WhatsApp Business API |
| iMessage | macOS + AppleScript | macOS only, no Docker |
| Matrix | Access token + homeserver | Any Matrix homeserver |

See individual channel pages under [Integrations > Channels](../integrations/channels/index.md) for per-channel setup details.

## Related

- [Channels overview](../integrations/channels/index.md) -- comparison matrix and individual channel docs
- [Outbound Safety](outbound-safety.md) -- purpose, cadence, STOP, ledger, and kill switches
- [Docker Compose](../operations/docker-compose.md) -- profiles and services
