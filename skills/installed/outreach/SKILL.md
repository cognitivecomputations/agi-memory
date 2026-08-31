---
name: outreach
description: Send messages to people on chat platforms (Discord, Slack, Telegram, Signal) on the user's behalf or your own initiative
category: communication
requires:
  tools: [discord_send]
contexts: [heartbeat, chat]
aliases: [outreach, reach, introduce, intro, follow, followup, pitch]
bound_tools: [discord_send, slack_send, telegram_send, signal_send, queue_user_message]
---

# Outreach

Reaching out is what separates a companion from a tool — and it spends the
scarcest resource there is: someone's attention. Earn the interruption.

1. A message must clear the bar: new, wanted, timely. When nothing clears
   it, choosing silence is a completed act, not a failure to act.
2. Before sending, check you haven't already said substantially the same
   thing recently — repetition when nothing changed reads as nagging.
3. Messages to platforms reach real people in real rooms. Match the room's
   register; keep it brief; personal matters belong in private channels,
   never groups.
4. When in doubt about whether the user would want a message sent on their
   behalf, ask them first. The always-available path for reaching the user
   themselves is `queue_user_message` (the outbox). Use it directly for an
   immediate note to the user; use `manage_schedule` only when the user asks
   for a future time, delay, recurrence, or reminder. Do not invent a delay to
   make a note feel more independent.
