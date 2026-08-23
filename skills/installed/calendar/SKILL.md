---
name: calendar
description: Read and manage the user's calendar — list events, create, update, and delete entries
category: productivity
requires:
  tools: [calendar_events]
contexts: [heartbeat, chat]
aliases: [book, booking, meeting, meet, schedule, scheduling, appointment, availability, free, busy, remind, reminder, invite, reschedule]
bound_tools: [calendar_events, calendar_create, calendar_update, calendar_delete]
---

# Calendar

Work with the user's calendar as a careful assistant would.

1. Read before you write: `calendar_events` first, so creates and updates
   respect what already exists.
2. Creating, updating, or deleting an entry changes the user's real
   schedule — confirm intent in conversation when the request is ambiguous,
   and state exactly what you changed afterward.
3. Deletions are the user's call, made explicitly. Describe what you are
   about to remove before removing it.
4. Times carry timezones; say them explicitly when confirming.
