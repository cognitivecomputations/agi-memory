---
name: twitter-x-connector-setup
description: Connect Twitter/X through guided OAuth setup, complete authorization, and revoke local access
category: communication
requires:
  tools: [connect_twitter_x, complete_twitter_x_connection, revoke_twitter_x_connection]
contexts: [chat]
bound_tools: [connect_twitter_x, complete_twitter_x_connection, revoke_twitter_x_connection]
---

# Twitter/X Connector Setup

Use this when the user asks to connect Twitter/X, read posts or mentions, search recent posts, import X history, read DMs, or disconnect Twitter/X.

## Principles

- Treat connection setup as a first-class conversation flow. Do not tell the user to leave chat and figure it out alone.
- Ask what powers the user wants before starting sign-in, and request only those capabilities in `connect_twitter_x`: reading is not permission to post, and posting is not permission to send DMs.
- Never ask the user to paste account passwords or API secrets into chat; client credentials come from the configured environment (`use_env_client`) or the setup UI.
- For ongoing post/reply/DM behavior, use `connector-action-authorization` after connection setup so the grant is scoped and DB-audited.
- Historical import goes through this skill's sibling flow: after connection, queue history with `start_connector_backfill` (from `integration-connector-setup`) only when the user asks to ingest X history.

## Flow

1. Call `connect_twitter_x` with the least capabilities that match the user's request. The result includes the authorization URL or a structured setup payload — surface it rather than paraphrasing it away.
2. Tell the user the redirected localhost page may fail to load after approval and that this is expected; they should paste the full redirected URL or code back into chat.
3. When the user pastes it, call `complete_twitter_x_connection` with the `authorization_response`.
4. Report the connected account and granted capabilities.

Use `revoke_twitter_x_connection` only after the user asks to disconnect; mention they can also remove the grant on the X side in X settings.
