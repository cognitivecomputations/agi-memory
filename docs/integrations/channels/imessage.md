<!--
title: iMessage
summary: iMessage integration for Hexis (macOS only)
read_when:
  - "You want to connect your agent to iMessage"
section: integrations
-->

# iMessage

Connect your Hexis agent to iMessage via BlueBubbles. **macOS only.**

> **Status**: Production-ready
> **Adapter**: `channels/imessage_adapter.py`
> **Library**: `aiohttp` (connects to BlueBubbles REST API)
> **Connection**: Polling (2-second interval) with cursor tracking

## Prerequisites

- macOS with iMessage signed in
- [BlueBubbles](https://bluebubbles.app/) server running on the Mac
- BlueBubbles server password

## Quick Start

```bash
hexis channels setup imessage
hexis up
hexis channels status
```

## Configuration

| Config Key | Type | Description |
|------------|------|-------------|
| `channel.imessage.api_url` | text | BlueBubbles server URL (default: `http://localhost:1234`) |
| `channel.imessage.password` | text | BlueBubbles password or env var name |
| `channel.imessage.allowed_handles` | array | JSON array of phone/email handles or `"*"` (all) |
| `channel.imessage.operator_recipient` | text | Your phone/email for unanswered Slack approval escalation |

Environment variable: `IMESSAGE_PASSWORD`

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Direct messages | Yes | Filtered by handle (phone/email) |
| Group messages | No | |
| Threads | No | iMessage does not support threads |
| Reactions | Yes | Tapback reactions via BlueBubbles |
| Media (attachments) | Yes | GUID-based download URLs |
| Typing indicator | Yes | Via BlueBubbles API |
| Edit messages | No | |
| Max message length | 20,000 chars | |

## How It Works

- Connects to the BlueBubbles REST API running on the same Mac (or accessible over the network)
- The configured operator recipient may answer approval prompts even when the
  ordinary conversation allowlist excludes that handle; replies still require
  the eight-character request code when multiple actions are pending.
- **Polling**: Queries `/api/v1/message` every 2 seconds with a timestamp cursor to fetch new messages
- **Cursor tracking**: Maintains `_last_timestamp` to avoid processing duplicates
- Skips messages with `isFromMe` flag to avoid echo loops
- **Sender identification**: Extracts phone number or email handle from message sender address
- **Chat detection**: Uses chat GUID to identify conversations
- **Outbound**: Sends messages via BlueBubbles POST API (AppleScript backend on the Mac)

## Media Handling

- **Inbound**: Constructs download URLs from attachment GUIDs; extracts MIME type from `mimeType` or `uti` fields
- **Outbound**: Uploads files via multipart/form-data to BlueBubbles; supports local file paths or URL download-then-upload
- **Attachment metadata**: Filename, MIME type, and size are normalized to the standard `Attachment` format

## Limitations

- **macOS only** -- iMessage requires macOS and a signed-in Apple ID
- **No Docker containerization** -- BlueBubbles runs on the host Mac, not in a container
- **Polling latency** -- 2-second polling interval means messages may take up to 2 seconds to be received
- **No thread support** -- iMessage does not have a threads concept

## Troubleshooting

- **Bot not responding**: Verify BlueBubbles is running and the password is correct
- **Connection refused**: Check that `api_url` points to the correct BlueBubbles server address and port
- **Duplicate messages**: This usually resolves itself -- the cursor tracking prevents reprocessing after the first poll
- **Media download failing**: Verify the attachment GUID is valid and BlueBubbles has access to the attachment file
- **Not receiving messages**: Ensure iMessage is signed in on the Mac and BlueBubbles shows "Connected" status

## Related

- [Channels overview](index.md)
- [Channels Setup guide](../../guides/channels-setup.md)
