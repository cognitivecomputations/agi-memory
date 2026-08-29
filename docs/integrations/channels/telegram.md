<!--
title: Telegram
summary: Telegram bot integration for Hexis
read_when:
  - "You want to connect your agent to Telegram"
section: integrations
-->

# Telegram

Connect your Hexis agent to Telegram.

> **Status**: Production-ready
> **Adapter**: `channels/telegram_adapter.py`
> **Library**: `python-telegram-bot`
> **Connection**: Long polling (works behind NAT, no webhook setup needed)

## Prerequisites

- A Telegram bot token from [@BotFather](https://t.me/botfather)

## Quick Start

```bash
hexis channels setup telegram
hexis up
hexis channels status
```

## Configuration

| Config Key | Type | Description |
|------------|------|-------------|
| `channel.telegram.bot_token` | text | Env var name holding the bot token |
| `channel.telegram.allowed_chat_ids` | array | JSON array of chat IDs or `"*"` (all) |

Environment variable: `TELEGRAM_BOT_TOKEN`

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Private messages | Yes | Always responds |
| Group messages | Yes | Responds to mentions and allowed chat IDs |
| Forum topics | Yes | Extracts `message_thread_id` for forum channels |
| Reactions | Yes | Emoji reactions |
| Media (photos, videos, audio, docs) | Yes | Photo picks largest resolution (`photo[-1]`) |
| Typing indicator | Yes | Uses `send_chat_action("typing")` |
| Edit messages | Yes | `edit_message_text()` with Markdown fallback |
| Max message length | 4,096 chars | |

## How It Works

- Uses long-polling via `updater.start_polling()` -- no open ports or webhook setup required
- Drops pending messages on startup (`drop_pending_updates=True`) to avoid replaying old messages
- Strips bot mention from message content before processing
- **Markdown formatting**: Attempts to send with `parse_mode="Markdown"`; falls back to plain text on parse errors
- **Forum topics**: Automatically detects forum channels and routes messages to the correct `message_thread_id`
- **Chat type detection**: Metadata includes `chat_type` (private, group, supergroup, channel)

## Media Handling

- **Photos**: Telegram provides multiple resolutions; the adapter selects the largest (`photo[-1]`)
- **Documents**: Extracts filename, MIME type, and `file_id` (platform_id)
- **Downloading**: Full URLs require a `bot.get_file()` call; `file_id` is stored as platform_id for reference
- **Sending**: Type-specific bot methods (`send_photo`, `send_video`, `send_audio`, `send_document`)

## Troubleshooting

- **Bot not responding**: Verify the token via `hexis auth` and check channel worker logs (`docker compose logs channel_worker -f`)
- **Group messages ignored**: Add the bot to the group and mention it, or add the chat ID to `allowed_chat_ids`
- **Forum topic messages not routing**: Ensure the bot is a member of the forum and has permission to post in topics
- **Markdown formatting broken**: Some messages with special characters may cause parse errors; the adapter automatically falls back to plain text

## Related

- [Channels overview](index.md)
- [Channels Setup guide](../../guides/channels-setup.md)
