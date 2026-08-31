---
name: integration-connector-setup
description: Start, configure, verify, inspect, and revoke first-class connectors including Slack, Telegram, Signal, Notion, Spotify, Home Assistant, Weather, and Trello
category: communication
requires:
  tools: [integration_setup_status, start_integration_setup, configure_channel_integration, verify_channel_integration, connect_life_integration, connect_spotify]
contexts: [chat]
bound_tools: [integration_setup_status, start_integration_setup, configure_channel_integration, verify_channel_integration, start_connector_backfill, connector_backfill_status, control_connector_backfill, connect_life_integration, connect_spotify, complete_spotify_connection, revoke_life_integration]
---

# Integration Connector Setup

Use this when the user asks to connect Slack, Telegram, Signal, Notion, Spotify,
Home Assistant, Weather, or Trello, or to inspect available connectors.

## Principles

- Treat setup as an in-conversation flow with exact next steps from the DB connector manifest.
- Never ask the user to paste bot tokens, app tokens, passwords, or API secrets into chat. Token fields must be env var names.
- Use `connect_gmail` from the Gmail connector skill for Gmail OAuth; this skill covers manual/pairing channel connectors.
- Use `connect_life_integration` for Notion, Home Assistant, Weather, and Trello. Pass only environment variable names for secret fields. Never search ambient environment variables or paste secret values into tool arguments.
- Use `connect_spotify` for Spotify PKCE. Its app client ID is non-secret, but an environment value may be used only when the user explicitly names that environment variable. Register the exact loopback redirect URI returned by Hexis; Spotify does not accept `localhost`.
- Be explicit when a capability is planned rather than available for a given provider.
- Historical backfill is local memory ingestion and needs explicit user approval, even when the provider access is read-only. Queue it only for a connected connector, after the user asks to import history.
- After config is written or the user says env vars are set, call `verify_channel_integration` so the DB records the connection only when the channel worker's config truth resolves.

## Flow

1. Call `integration_setup_status` for the requested connector or for all connectors if the user asks what can be connected.
2. Call `start_integration_setup` with the least capabilities matching the user's request.
3. If the user provides channel settings, call `configure_channel_integration` only with env var names for token fields and non-secret allowlists or URLs.
4. Call `verify_channel_integration` after the env/config values should be available.
5. Tell the user to start or restart `hexis-channels` only if verification succeeded and the adapter is not already running.
6. If the user asks to import history, call `start_connector_backfill` with the smallest useful scope — Slack needs a `channel_id`; Telegram, Signal, and Twitter/X take a local export/archive path (`export_path`/`import_path`).
7. Use `connector_backfill_status` for progress, and `control_connector_backfill` only when the user asks to pause, resume, or cancel a job.
8. For an everyday-life API connector, call its connect tool once without missing fields to surface the provider-specific setup card; verify only after the user chooses the credential references or default location.
9. Use `complete_spotify_connection` only when the automatic local callback did not finish. Use `revoke_life_integration` when the user asks to disconnect one of these providers.
