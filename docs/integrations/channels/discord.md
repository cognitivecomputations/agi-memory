<!--
title: Discord
summary: Discord bot integration for Hexis
read_when:
  - "You want to connect your agent to Discord"
section: integrations
-->

# Discord

Connect your Hexis agent to Discord servers.

> **Status**: Production-ready
> **Adapter**: `channels/discord_adapter.py`
> **Library**: `discord.py`
> **Connection**: WebSocket (persistent gateway connection)

## Prerequisites

- A Discord bot token from the [Discord Developer Portal](https://discord.com/developers)
- Bot invited to your server with `Send Messages` and `Read Message History` permissions

## Quick Start

```bash
hexis channels setup discord
hexis up
hexis channels status
```

## Configuration

| Config Key | Type | Description |
|------------|------|-------------|
| `channel.discord.bot_token` | text | Env var name holding the bot token |
| `channel.discord.allowed_guilds` | array | JSON array of guild IDs or `"*"` (all) |
| `channel.discord.allowed_channels` | array | JSON array of channel IDs or `"*"` (all) |

Environment variable: `DISCORD_BOT_TOKEN`

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Direct messages | Yes | Always responds |
| Channel messages | Yes | Responds to mentions and allowed channels |
| Threads | Yes | Detects `discord.Thread` and extracts thread_id |
| Reactions | Yes | Emoji reactions |
| Media (images, files) | Yes | URL embeds for remote, file uploads for local |
| Typing indicator | Yes | Uses Discord's built-in `channel.typing()` |
| Edit messages | Yes | Fetches message by ID, then edits content |
| Max message length | 2,000 chars | |

## How It Works

- Maintains a persistent WebSocket connection to Discord's gateway via `discord.Client.start()`
- Receives all messages via the `on_message` event callback
- Automatically strips bot mention (`<@BOT_ID>` and `<@!BOT_ID>`) from message content
- Ignores its own messages and messages from other bots
- **Guild filtering**: Can restrict to specific guilds; still responds if directly mentioned in non-allowed guilds
- **Channel filtering**: Can restrict to specific channels within allowed guilds

## Attachment Handling

Discord attachments are normalized to `Attachment` objects with URL, filename, MIME type, size, and platform_id. The adapter can both receive attachments (via `message.attachments`) and send them (via URL embed or file upload).

## Troubleshooting

- **Bot not responding**: Check that the bot token is set and the bot is invited to the server with correct permissions
- **Missing messages in channels**: Verify `allowed_guilds` and `allowed_channels` settings; the bot needs to be mentioned or the channel must be in the allowlist
- **Permission errors**: Ensure the bot has `Send Messages`, `Read Message History`, and `View Channel` permissions in the Discord developer portal
- **Thread messages not working**: The bot needs `Send Messages in Threads` permission

## Related

- [Channels overview](index.md)
- [Channels Setup guide](../../guides/channels-setup.md)
