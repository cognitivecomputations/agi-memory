---
name: daily-briefing
description: Compile a comprehensive daily briefing from calendar, contacts, goals, and recent activity
category: productivity
requires:
  tools: [recall, calendar_events]
contexts: [heartbeat, chat]
aliases: [briefing, brief, today, morning, agenda, day, catch]
bound_tools: [recall, calendar_events, search_contacts, aggregate_signals]
---

# Daily Briefing Compilation

Assemble a structured morning briefing that gives the user a complete picture of their day: schedule, priorities, pending items, and relevant context.

## When to Use

- During the first heartbeat of the user's active hours (typically morning)
- When the user explicitly asks for a briefing ("brief me", "what's on today")
- Only once per day -- after generating a briefing, store a working memory noting it was delivered, and skip on subsequent heartbeats

## Step-by-Step Methodology

1. **Check if already delivered**: Use `recall` to see if a daily briefing was already generated today. If so, skip unless the user explicitly asks for a refresh.
2. **Pull today's calendar**: Call `calendar_events` for today's date range. Extract meeting times, titles, attendees, and locations.
3. **Enrich meetings**: For each meeting, run a lightweight contact lookup and memory recall on attendees and meeting topics. Do not go deep -- this is a summary, not full meeting prep. Delegate to the meeting-prep skill for any meeting the user wants to dive into.
4. **Review active goals**: Use `recall` to surface active goals and their recent progress. Identify which goals have actionable next steps for today.
5. **Check pending items**: Recall any action items, promises, or deadlines that fall on or near today. These might come from email digests, prior conversations, or meeting notes.
6. **Aggregate signals**: If `aggregate_signals` is available, pull together any overnight notifications, alerts, or environmental changes worth noting.
7. **Compose the briefing**: Structure the output as:
   - **Schedule**: Chronological list of today's events with times and key attendees
   - **Priorities**: Top 3-5 items that need attention today, drawn from goals and pending actions
   - **Context**: Any relevant background (yesterday's outcomes, approaching deadlines, notable signals)
   - **Suggested focus**: A brief recommendation on where to spend energy today
8. **Store and surface**: Save the briefing as a working memory with a 24-hour expiry. Queue it for delivery to the user via the outbox.

## Quality Guidelines

- Keep the briefing to one screenful of text. If the day is packed, prioritize ruthlessly rather than listing everything.
- Lead with the most time-sensitive items. A meeting in 30 minutes matters more than a goal due next week.
- Use plain, direct language. This is an operational briefing, not a narrative.
- If calendar access is unavailable, still produce a briefing from goals and memories -- note that calendar data is missing.
- Never fabricate calendar events or action items. If data is sparse, the briefing should be short, not padded.
- Respect energy budget. The daily briefing is a valuable but non-trivial operation. If energy is critically low, defer to the next heartbeat.
