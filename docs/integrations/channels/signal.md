<!--
title: Signal
summary: Signal messaging integration for Hexis
read_when:
  - "You want to connect your agent to Signal"
section: integrations
-->

# Signal

Connect your Hexis agent to Signal.

> **Status**: Production-ready
> **Adapter**: `channels/signal_adapter.py`
> **Library**: `aiohttp` (connects to `signal-cli-rest-api` sidecar)
> **Connection**: SSE (Server-Sent Events) stream with auto-reconnect

## Prerequisites

- A phone number registered with Signal
- The `signal` Docker Compose profile (runs `signal-cli-rest-api` as a sidecar)

## Quick Start

```bash
hexis channels setup signal
docker compose --profile signal up -d
hexis channels status
```

Note the `signal` profile -- Signal requires its own sidecar container.

## Configuration

| Config Key | Type | Description |
|------------|------|-------------|
| `channel.signal.api_url` | text | REST API base URL (default: `http://localhost:8080`) |
| `channel.signal.phone_number` | text | Bot's registered phone number |
| `channel.signal.allowed_numbers` | array | JSON array of phone numbers or `"*"` (all) |

Environment variables: `SIGNAL_API_URL`, `SIGNAL_PHONE_NUMBER`

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Direct messages | Yes | Filtered by allowed numbers |
| Group messages | Yes | Uses `groupInfo.groupId` as channel_id |
| Threads | No | Signal does not support threads |
| Reactions | Yes | |
| Media (attachments) | Yes | Requires separate API call to download |
| Typing indicator | No | Not available via REST API |
| Edit messages | No | Not supported by Signal |
| Max message length | 8,000 chars | |

## How It Works

- Connects to the `signal-cli-rest-api` sidecar via SSE stream on `/api/v1/receive/{phone_number}`
- Parses incoming JSON events from the SSE stream
- Sends outbound messages via HTTP POST to `/api/v2/send`
- **Auto-reconnect**: 5-second backoff on stream errors
- **Group detection**: Uses `groupInfo.groupId` to distinguish group messages from 1:1 chats

## Docker Profile

Signal requires its own Docker Compose profile because it runs an external sidecar:

```bash
# Start with both active and signal profiles
docker compose --profile signal up -d
```

The `signal-cli-rest-api` container handles phone number registration and the Signal protocol. Hexis communicates with it via REST API.

## Troubleshooting

- **No messages received**: Check that `signal-cli-rest-api` is running (`docker compose ps`) and the phone number is registered
- **Connection errors**: Verify `channel.signal.api_url` points to the signal-cli container (default: `http://localhost:8080`)
- **Phone number not registered**: The phone number must be registered with Signal before the adapter can receive messages
- **Stream disconnections**: The adapter auto-reconnects with a 5-second backoff; check logs for persistent errors

## Related

- [Channels overview](index.md)
- [Channels Setup guide](../../guides/channels-setup.md)
- [Docker Compose](../../operations/docker-compose.md) -- the `signal` profile
