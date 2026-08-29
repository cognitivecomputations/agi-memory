<!--
title: WhatsApp
summary: WhatsApp Business integration for Hexis
read_when:
  - "You want to connect your agent to WhatsApp"
section: integrations
-->

# WhatsApp

Connect your Hexis agent to WhatsApp via the Meta Business Cloud API.

> **Status**: Production-ready
> **Adapter**: `channels/whatsapp_adapter.py`
> **Library**: `aiohttp` (direct REST calls to Meta's Cloud API)
> **Connection**: Webhook (inbound) + REST API (outbound)

## Prerequisites

- Meta Developer account with WhatsApp Business API access
- Access token, phone number ID, and app secret from [Meta Business dashboard](https://developers.facebook.com)
- External webhook URL (WhatsApp requires inbound connectivity)

## Quick Start

```bash
hexis channels setup whatsapp
hexis up
hexis channels status
```

## Configuration

| Config Key | Type | Description |
|------------|------|-------------|
| `channel.whatsapp.access_token` | text | Meta access token or env var name |
| `channel.whatsapp.phone_number_id` | text | WhatsApp Business phone number ID |
| `channel.whatsapp.verify_token` | text | Webhook verification token (default: `hexis_verify`) |
| `channel.whatsapp.webhook_port` | int | Port for webhook server (default: `8443`) |
| `channel.whatsapp.allowed_numbers` | array | JSON array of phone numbers or `"*"` (all) |
| `channel.whatsapp.app_secret` | text | App secret for HMAC signature verification |

Environment variables: `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_APP_SECRET`

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Direct messages | Yes | Filtered by allowed numbers |
| Group messages | Yes | |
| Threads | No | |
| Reactions | Yes | |
| Media (images, videos, audio, docs) | Yes | Lazy download via platform_id |
| Typing indicator | No | Sends read receipts instead |
| Edit messages | No | |
| Max message length | 4,096 chars | |

## How It Works

- Uses Meta's WhatsApp Business Cloud API (`graph.facebook.com/v21.0`)
- Runs an `aiohttp` web server to receive webhook POST events
- **Webhook verification**: Responds to Meta's GET challenge-response during setup
- **Signature verification**: Validates inbound webhooks via HMAC-SHA256 using `app_secret` (optional but recommended)
- Responds with HTTP 200 immediately, then processes the message asynchronously
- **Reply context**: Supports reply-to via message ID reference in outbound messages
- **Number filtering**: Checks sender against `allowed_numbers` allowlist

## Media Handling

- **Inbound**: Extracts media type (image, document, audio, video) with caption and `platform_id`
- **Downloading**: Media requires a separate API call using the platform_id -- the adapter stores the ID for lazy download
- **Outbound**: Posts media with type-specific payloads and optional caption

## Setup Notes

1. Create a Meta app at [developers.facebook.com](https://developers.facebook.com)
2. Add the WhatsApp product and configure a phone number
3. Set up a webhook pointing to your Hexis instance's webhook port (default: `8443`)
4. Use the verify token configured in Hexis (default: `hexis_verify`)
5. Subscribe to the `messages` webhook field

## Troubleshooting

- **Webhook verification failing**: Check that `verify_token` matches what's configured in the Meta dashboard
- **No messages received**: Ensure the webhook URL is externally accessible (WhatsApp cannot reach `localhost`)
- **Signature errors**: Verify `app_secret` is correct; if unset, signature verification is skipped
- **Media not downloading**: Media URLs from Meta expire; use the platform_id to fetch fresh download URLs
- **Rate limits**: Meta enforces rate limits on the Business API; check the Meta dashboard for quota status

## Related

- [Channels overview](index.md)
- [Channels Setup guide](../../guides/channels-setup.md)
