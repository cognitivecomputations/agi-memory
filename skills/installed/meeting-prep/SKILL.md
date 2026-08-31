---
name: meeting-prep
description: Prepare briefings for upcoming calendar events with attendee context and relevant memories
category: productivity
requires:
  tools: [calendar_events, search_contacts, recall]
  config: [google_calendar]
  env: [GOOGLE_CREDENTIALS_JSON]
contexts: [heartbeat, chat]
aliases: [meeting, prep, brief, agenda, attendees, before]
bound_tools: [calendar_events, meeting_prep, search_contacts, recall]
---

# Meeting Preparation Workflow

Compile a briefing for an upcoming meeting by pulling together calendar details, attendee background, relevant memories, and suggested talking points.

## When to Use

- During heartbeats when a meeting is starting within the next 30-60 minutes
- When the user asks "what do I have coming up" or "prep me for my next meeting"
- When a daily briefing skill delegates meeting context to this skill
- Before any meeting where attendee relationships or prior context would be valuable

## Step-by-Step Methodology

1. **Identify the meeting**: Use `calendar_events` to find the next upcoming event (or a specific one if the user named it). Extract the title, time, location/link, and attendee list.
2. **Look up attendees**: For each attendee, call `search_contacts` to pull their name, role, company, and relationship notes. If a contact is unknown, note them as a new face.
3. **Recall prior interactions**: Use `recall` with each attendee's name or company to surface past conversations, decisions, promises, or open items. Focus on the last 30 days of episodic memories.
4. **Recall topic context**: If the meeting title or description references a project, product, or topic, run a targeted `recall` for that subject to gather relevant semantic and strategic memories.
5. **Synthesize the briefing**: Combine all gathered context into a structured brief:
   - Meeting logistics (time, location, link)
   - Attendee profiles and relationship notes
   - Key context and recent history
   - Open items or promises to follow up on
   - Suggested talking points or questions
6. **Deliver or store**: In chat context, present the briefing directly. In heartbeat context, store it as a working memory so it surfaces at the right time.

## Quality Guidelines

- Keep briefings concise. A wall of text defeats the purpose; aim for a scannable format with headers and bullets.
- Prioritize actionable context over trivia. "You promised to send them the proposal by Friday" matters more than "they like coffee."
- If calendar credentials are unavailable, fall back to asking the user what meeting they want to prepare for.
- When attendees are unknown, note this explicitly rather than guessing. Suggest the user add them as contacts.
- Respect the energy budget: skip deep recall when energy is low and provide a lighter briefing instead.
