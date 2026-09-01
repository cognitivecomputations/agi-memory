<!--
title: Channels
summary: Messaging platform integrations for Hexis
read_when:
  - "You want to connect your agent to a messaging platform"
  - "You want to compare channel options"
section: integrations
-->

# Channels

Connect your Hexis agent to messaging platforms. Each channel runs as an adapter that bridges the platform to RabbitMQ, where the agent processes messages.

## Channel Comparison

| Channel | Library | Connection | Auth | Docker Profile |
|---------|---------|-----------|------|----------------|
| [Discord](discord.md) | `discord.py` | WebSocket | Bot token | `active` |
| [Telegram](telegram.md) | `python-telegram-bot` | Long polling | Bot token | `active` |
| [Slack](slack.md) | `slack-bolt` | Socket Mode / HTTP | Bot + App token | `active` |
| [Signal](signal.md) | `aiohttp` | SSE stream | Phone number + sidecar | `active` + `signal` |
| [WhatsApp](whatsapp.md) | `aiohttp` | Webhook | Meta Business API | `active` |
| [iMessage](imessage.md) | `aiohttp` | Polling (2s) | BlueBubbles server | macOS only |
| [Matrix](matrix.md) | `matrix-nio` | Sync loop | Access token | `active` |

## Capabilities Matrix

| Capability | Discord | Telegram | Slack | Signal | WhatsApp | iMessage | Matrix |
|------------|---------|----------|-------|--------|----------|----------|--------|
| Direct messages | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Group messages | Yes | Yes | Yes | Yes | Yes | No | Yes |
| Threads | Yes | Yes (forums) | Yes | No | No | No | Yes |
| Reactions | Yes | Yes | Yes | Yes | Yes | Yes (tapback) | Yes |
| Media | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Typing indicator | Yes | Yes | No-op | No | No-op | Yes | Yes |
| Edit messages | Yes | Yes | Yes | No | No | No | Yes |
| Max message length | 2,000 | 4,096 | 4,000 | 8,000 | 4,096 | 20,000 | 65,536 |

## General Setup Pattern

All channels follow the same pattern:

1. **Configure credentials** -- `hexis channels setup <channel>`
2. **Start the stack** -- `hexis up` includes the channel/web-inbox worker even when no external adapter is configured
3. **Verify** -- `hexis channels status`

See [Channels Setup guide](../../guides/channels-setup.md) for the general pattern and tips.

## Architecture

```
Platform API  <-->  Channel Adapter  <-->  RabbitMQ  <-->  Agent Loop
```

Each adapter implements the `ChannelAdapter` ABC from `channels/base.py`, providing:

- `start(on_message)` -- connect and listen for inbound messages
- `stop()` -- disconnect gracefully
- `send(channel_id, text, reply_to, thread_id)` -- send a message
- `send_presentation(channel_id, presentation, ...)` -- render portable blocks
- `capabilities` -- declares supported features via `ChannelCapabilities` dataclass

Adapters run in the channel worker container, maintaining persistent connections to platform APIs and routing messages through RabbitMQ to the agent's conversation loop.

## Portable Presentation

Proactive outbox messages can carry an optional presentation envelope. The
portable contract supports ordered `text`, `context`, and `divider` blocks, an
optional title, and a semantic tone. Adapters render those blocks using their
declared text dialect: Discord Markdown, Telegram legacy Markdown, Slack
mrkdwn, or plain text for channels without a compatible rich-text send path.
Long output still uses the adapter's live `max_message_length` capability.

```json
{
  "kind": "channel_message",
  "payload": {
    "content": "Deployment ready. Derived from the live health check.",
    "presentation": {
      "title": "Deployment",
      "tone": "success",
      "blocks": [
        {"type": "text", "text": "**Ready** for review."},
        {"type": "divider"},
        {"type": "context", "text": "Derived from the live health check."}
      ]
    },
    "delivery_mode": "direct",
    "target_channel": "discord",
    "target_id": "123456789"
  }
}
```

`content` remains the canonical plain-text audit mirror. If it is omitted, the
outbox derives a plain mirror from the blocks. Invalid or unknown block types
fail with the exact block path instead of delivering a partial message. The web
chat receives the same envelope on its final SSE event; conversation history
continues to store only the canonical assistant text.

## Choosing a Channel

- **Easiest setup**: Discord or Telegram (bot token only, no external services)
- **Enterprise**: Slack (Socket Mode works behind firewalls)
- **Privacy-focused**: Signal or Matrix (self-hosted, encrypted)
- **Mobile**: WhatsApp or iMessage (phone-based messaging)
- **Self-hosted**: Matrix (bring your own homeserver)

## Related

- [Channels Setup guide](../../guides/channels-setup.md) -- general setup walkthrough
- [Workers](../../operations/workers.md) -- channel worker lifecycle
