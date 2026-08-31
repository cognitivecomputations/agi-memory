---
name: travel-prep
description: Turn calendar events, confirmations, current research, and personal context into a practical trip brief
category: productivity
requires:
  tools: [calendar_events, recall]
contexts: [chat]
aliases: [travel, trip, itinerary, packing, vacation]
bound_tools: [calendar_events, calendar_create, calendar_update, email_search, email_read, search_contacts, recall, remember, web_search, web_fetch, manage_backlog, todoist_list_tasks, todoist_create_task, asana_list_projects, asana_create_task]
---

# Travel Preparation

Build a source-grounded plan for a trip without turning planning into an unauthorized booking or purchase.

## Workflow

1. **Establish the trip.** Confirm the destination, dates, travelers, origin, time zone, and any constraints the user supplied. If several calendar events could be the trip, show the candidates and ask which one they mean.
2. **Gather owned facts.** Read the relevant calendar events and, when useful, search email for booking confirmations. Treat confirmation numbers, addresses, times, and baggage or check-in rules as sensitive. Include them only when they help the user and do not persist them unless asked.
3. **Recover personal context.** Use `recall` for stated preferences, accessibility needs, previous promises, and destination context. Never convert a weak or unrelated memory into a trip requirement.
4. **Research what can change.** Use `web_search` and `web_fetch` for current entry requirements, transit disruptions, opening hours, weather-dependent advice, and other time-sensitive facts. Name the source and checked date. Separate official requirements from suggestions.
5. **Build the brief.** Present a chronological itinerary, unresolved gaps, departure checklist, local logistics, reservations, contacts, and a short contingency section. Preserve local times and label time zones when ambiguity is possible.
6. **Offer actions.** Propose calendar blocks or tasks after the brief. Create or update them only when the user explicitly chooses the exact action and destination system. Prefer the local backlog when no external task service is configured.

## Control and Failure Boundaries

- Never book, buy, cancel, check in, or submit traveler information. This skill prepares; it does not transact.
- Never silently move an event or infer a missing date, traveler, currency, visa status, or reservation.
- A request to "prepare" authorizes reading and drafting, not calendar changes, outbound mail, or task creation.
- If calendar or email access is unavailable, continue from details the user provides and state exactly what is missing. Give the concrete setup or input needed next instead of ending at an error.
- Keep unknowns visible. A short plan with an explicit gap is better than a complete-looking fiction.
