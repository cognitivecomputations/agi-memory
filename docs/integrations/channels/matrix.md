<!--
title: Matrix
summary: Matrix messaging integration for Hexis
read_when:
  - "You want to connect your agent to Matrix"
section: integrations
-->

# Matrix

Connect your Hexis agent to Matrix rooms.

> **Status**: Production-ready
> **Adapter**: `channels/matrix_adapter.py`
> **Library**: `matrix-nio` (async Matrix client)
> **Connection**: Sync loop (`sync_forever()` with event callbacks)

## Prerequisites

- A Matrix account for the bot on any homeserver (e.g., matrix.org, self-hosted Synapse)
- Access token for the bot account

## Quick Start

```bash
hexis channels setup matrix
hexis up
hexis channels status
```

## Configuration

| Config Key | Type | Description |
|------------|------|-------------|
| `channel.matrix.homeserver` | text | Matrix homeserver URL (e.g., `https://matrix.org`) |
| `channel.matrix.user_id` | text | Bot user ID (e.g., `@hexis:matrix.org`) |
| `channel.matrix.access_token` | text | Access token for the bot |
| `channel.matrix.allowed_rooms` | array | JSON array of room IDs or `"*"` (all) |

Environment variables: `MATRIX_HOMESERVER`, `MATRIX_USER_ID`, `MATRIX_ACCESS_TOKEN`

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Room messages | Yes | Filtered by allowed rooms |
| Direct messages | Yes | |
| Threads | Yes | Uses `m.thread` relation type |
| Reactions | Yes | Emoji events |
| Media (images, videos, audio, files) | Yes | Uploaded to Matrix media repository |
| Typing indicator | Yes | Via `room_typing()` API |
| Edit messages | Yes | Via `m.replace` relation |
| Max message length | 65,536 chars | Largest of all channels |

## How It Works

- Uses `matrix-nio` `AsyncClient` with token-based authentication (not password login)
- Registers event callbacks on `RoomMessageText` events, then calls `sync_forever()`
- **Self-message filtering**: Ignores events where sender matches the bot's own user ID
- **Thread support**: Extracts thread context from `m.relates_to` with `rel_type: "m.thread"` in event source
- **Reply support**: Detects `m.in_reply_to` relations for inline replies
- **Room filtering**: Checks room ID against `allowed_rooms` allowlist
- **Metadata**: Includes room member count and sender display name

## Media Handling

- **Inbound**: Extracts media from Matrix events with MIME type and content URI
- **Outbound (upload-then-send)**: First uploads the file to the Matrix media repository via `client.upload()`, receives a `content_uri`, then sends the media event to the room
- **Type detection**: Determines media type (image, video, audio, file) from MIME type to use the correct Matrix event type (`m.image`, `m.video`, `m.audio`, `m.file`)

## Message Editing

Matrix is one of the few channels that supports message editing. The adapter sends an `m.replace` relation targeting the original event ID, which updates the message in all clients that support edits.

## Troubleshooting

- **Bot not responding**: Verify the access token is valid and the bot has joined the room (use `!invite @hexis:yourserver`)
- **Media upload failures**: Check that the homeserver accepts uploads from the bot account and the file is within size limits
- **Thread messages not appearing**: Ensure the Matrix client supports threads (Element, FluffyChat, etc.)
- **Edit not working**: Some older Matrix clients may not display edits; the original message remains unchanged for those clients
- **Sync errors**: Check that the homeserver URL is correct and reachable from the Hexis container

## Related

- [Channels overview](index.md)
- [Channels Setup guide](../../guides/channels-setup.md)
