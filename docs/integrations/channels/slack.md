<!--
title: Slack
summary: Slack bot integration for Hexis
read_when:
  - "You want to connect your agent to Slack"
section: integrations
-->

# Slack

Connect your Hexis agent to Slack workspaces.

> **Status**: Production-ready
> **Adapter**: `channels/slack_adapter.py`
> **Library**: `slack-bolt` + `slack-sdk`
> **Connection**: Socket Mode (primary) or HTTP Events (fallback)

## Prerequisites

- A Slack app with Bot token (`xoxb-...`) from the [Slack API](https://api.slack.com/apps)
- App token (`xapp-...`) for Socket Mode (recommended)
- Required scopes: `chat:write`, `channels:history`, `im:history`, `im:write`

## Quick Start

```bash
hexis channels setup slack
hexis up
hexis channels status
```

## Configuration

| Config Key | Type | Description |
|------------|------|-------------|
| `channel.slack.bot_token` | text | Env var name for `xoxb-...` bot token |
| `channel.slack.app_token` | text | Env var name for `xapp-...` app token (Socket Mode) |
| `channel.slack.signing_secret` | text | Env var name for Slack's signing secret (HTTP interactivity only) |
| `channel.slack.operator_user_id` | text | Your Slack `U...` user ID for private approval DMs |
| `channel.slack.allowed_channels` | array | JSON array of channel IDs or `"*"` (all) |

Environment variables: `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, and optionally `SLACK_SIGNING_SECRET`

## Connection Modes

| Mode | Requires | Notes |
|------|----------|-------|
| **Socket Mode** (recommended) | Bot token + App token | Bidirectional, works behind firewalls, no webhook setup |
| **HTTP Events** (fallback) | Bot token + webhook URL | Requires external accessibility; used if `app_token` is missing |

Socket Mode is strongly recommended -- it works behind NAT and firewalls without any webhook configuration.

Approve/Deny buttons are received over Socket Mode automatically. If you use
HTTP interactivity instead, set Slack's Interactivity Request URL to
`https://your-secure-endpoint/api/slack/interactivity`, configure
`channel.slack.signing_secret`, and expose only that signed route through a
tightly scoped reverse proxy rather than exposing the dashboard. Hexis verifies
Slack's signature and five-minute replay window.

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Direct messages | Yes | Always responds |
| Channel messages | Yes | Responds to bot mentions |
| Threads | Yes | Extracts `thread_ts` as thread_id |
| Reactions | Yes | Emoji reactions |
| Media (files, images) | Yes | Uses `files_upload_v2()` |
| Typing indicator | No | Slack API does not support bot typing indicators (silently skipped) |
| Edit messages | Yes | Via message timestamp |
| Protected-tool approvals | Yes | Identity-checked, one-shot Approve/Deny buttons in a private DM |
| Max message length | 4,000 chars | |

## How It Works

- Uses `slack-bolt` `AsyncApp` with event-driven architecture
- Listens on `@app.event("message")` for incoming messages
- Ignores bot messages (`bot_id` present) and message subtypes (edits, joins, leaves)
- **User info**: Fetches display name asynchronously via `client.users_info(user=user_id)`
- **Thread support**: Uses Slack's `thread_ts` (timestamp) for threaded conversations
- **Channel filtering**: Responds to allowlisted channels; always responds when mentioned in non-allowed channels
- **File attachments**: Extracts `url_private_download` or `url_private` from file metadata
- **Approvals**: A protected tool call opens a private Slack DM. The exact
  arguments are shown with secrets redacted; the resulting proof is consumed
  once and only for those arguments. If iMessage escalation is configured and
  Slack is unanswered for five minutes, Hexis sends a coded approve/deny prompt
  there instead.
- **Plain replies**: The configured operator can also reply `approve CODE` or
  `deny CODE`. The code is optional only when exactly one request is pending.

## Troubleshooting

- **Bot not responding**: Verify both `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` are set
- **Socket Mode errors**: Ensure Socket Mode is enabled in your Slack app settings (Settings > Socket Mode)
- **Missing permissions**: Add `chat:write`, `channels:history`, `im:history`, and `im:write` scopes to your bot token
- **No typing indicator**: This is expected -- the Slack bot API does not support typing indicators
- **HTTP fallback warning**: If you see "Using HTTP fallback" in logs, set the `SLACK_APP_TOKEN` to enable Socket Mode

## Related

- [Channels overview](index.md)
- [Channels Setup guide](../../guides/channels-setup.md)
