---
name: meeting-follow-up
description: Turn completed meetings into an accurate recap, owned actions, and user-approved follow-up drafts
category: productivity
requires:
  tools: [recall]
contexts: [chat]
aliases: [followup, recap, minutes, decisions, action-items]
bound_tools: [calendar_events, email_search, email_read, search_contacts, recall, remember, manage_backlog, gmail_reply, gmail_send, email_send, fathom_transcripts, fathom_ingest, todoist_create_task, asana_list_projects, asana_create_task]
---

# Meeting Follow-up

Convert evidence from a completed meeting into a concise record and a controlled set of next actions. Use `meeting-prep` for work before a meeting.

## Workflow

1. **Identify the meeting.** Use the user's wording and calendar details to confirm the event, date, and attendees. If the match is ambiguous, show likely events instead of choosing silently.
2. **Choose the source.** Prefer notes or a transcript supplied by the user. If Fathom is available, list likely recordings and let the user identify the right one before ingesting it. Calendar metadata alone is not evidence of what was discussed.
3. **Extract outcomes.** Separate decisions, commitments, action items, owners, due dates, open questions, and topics explicitly deferred. Quote sparingly and preserve uncertainty.
4. **Check context.** Use `recall` and relevant email threads to identify earlier promises or conflicts. Label these as prior context rather than meeting outcomes.
5. **Draft the package.** Produce a scannable recap, an owner-by-owner action list, and any requested follow-up email draft. Flag every missing owner or date rather than inventing one.
6. **Confirm mutations.** Show proposed memory, backlog, Todoist, or Asana entries before creating them unless the user's request already specified their exact content and destination. Show every outbound message and recipient before sending.
7. **Report completion.** Return what was recorded, created, or sent, including anything that failed and the exact retry or manual step.

## Control and Failure Boundaries

- "Follow up" authorizes analysis and drafting, not sending mail, assigning another person, or creating external tasks without confirmation.
- Do not mark a suggestion as a decision, a discussion point as a commitment, or an attendee as an owner without evidence.
- Do not ingest a transcript merely because a likely recording exists; ingestion changes local memory and requires a deliberate choice.
- When no notes or transcript are available, ask the user for notes or offer a recap template. Never manufacture minutes from the calendar title.
