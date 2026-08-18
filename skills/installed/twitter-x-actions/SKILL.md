---
name: twitter-x-actions
description: Create Twitter/X posts, reply to posts, and send DMs with explicit action authorization
category: communication
requires:
  tools: [twitter_x_post, twitter_x_reply, twitter_x_dm_send]
contexts: [chat, heartbeat]
bound_tools: [twitter_x_post, twitter_x_reply, twitter_x_dm_send]
---

# Twitter/X Actions

Use this only after Twitter/X is connected and the user asks for an outward X action: post, reply, or send a DM.

## Principles

- A connected X account is not permission to act. Posting is public and effectively irreversible once seen — hold outward X actions to at least the outreach bar: new, wanted, timely.
- One-off chat actions still need a clear user request in the current conversation.
- Heartbeat actions require a matching DB-owned connector action policy; if none exists, do not improvise. Use `connector-action-authorization` to establish one.
- Keep posts and replies short, literal, and aligned with the user's stated intent. Do not escalate a narrow request ("reply to this thread") into broader posting.
- Personal matters belong in DMs, never public posts.

## Flow

1. For a new post, call `twitter_x_post` with the exact `text` the user approved.
2. For a reply, call `twitter_x_reply` with `reply_to_tweet_id` and the text.
3. For a direct message, call `twitter_x_dm_send` with the `participant_id` and text.
4. Report what was posted or sent, quoting the final text.
