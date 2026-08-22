---
name: email-digest
description: Digest and ingest emails into memory, surfacing important threads and action items
category: communication
requires:
  tools: [email_list, email_read, remember, recall, gmail_setup_status, queue_user_message]
contexts: [heartbeat, chat]
aliases: [email, emails, mail, inbox, unread, important, triage]
bound_tools: [email_list, email_read, email_search, ingest_emails, gmail_setup_status, recall, remember, queue_user_message, email_send, email_send_sendgrid]
---

# Email Digest Workflow

Process incoming emails into structured memories, extract action items, and surface threads that need attention.

## When to Use

- During autonomous heartbeats to check for new mail since the last digest, but only when Gmail is connected and `integrations.gmail.heartbeat_digest_enabled` is true or the active heartbeat task/goal carries explicit user authorization for this check
- When the user asks "what's in my inbox" or "any important emails"
- When a goal depends on information that may have arrived via email
- When preparing a daily briefing that includes email highlights

## Step-by-Step Methodology

1. **Check authorization**: In heartbeat, only proceed if the runtime exposed this skill because the user granted autonomous Gmail digest access. If setup/status indicates the grant is missing, stop and queue a short request for the user to enable background email checks.
2. **Check recency**: Use `recall` to find when the last email digest ran. Avoid re-processing messages already ingested.
3. **Check setup**: Call `gmail_setup_status`. If Gmail is not connected, record the missing setup clearly and stop; do not loop.
4. **List new emails**: Call `email_list` with a date filter to fetch unread or recent messages. Start with the inbox; expand to other labels only if the user has configured them.
5. **Triage by sender and subject**: Scan the list for high-signal indicators -- known contacts, reply chains the user is on, calendar invites, and keywords matching active goals.
6. **Read priority threads**: Use `email_read` on the top 5-10 most relevant messages. Do not read every email; batch processing wastes energy and context.
7. **Extract action items**: For each important email, identify: (a) what is being asked, (b) who is asking, (c) any deadline mentioned, (d) whether a reply is expected.
8. **Store findings**: Respect `integrations.gmail.memory_policy`. If it is `forget`, do not use `remember` or `ingest_emails` for email-derived content. If it is `ask`, queue a user-facing choice before storing. If it is `remember`, persist action items as episodic memories with high importance, tagged with the sender, thread ID, and any relevant goal.
9. **Surface urgency**: If an email requires a time-sensitive response, flag it in the heartbeat result so it can be raised to the user at next opportunity.

## Quality Guidelines

- Never store raw email bodies as memories. Summarize and extract the salient points.
- Respect privacy: do not log email content to external services. All storage stays in the local Postgres brain.
- Google OAuth authorization is not the same as Hexis autonomy authorization. A connected Gmail account does not imply permission for background heartbeat reading.
- When multiple emails belong to the same thread, consolidate into a single memory rather than creating duplicates.
- If credentials are missing or expired, fail gracefully. If `gmail_setup_status` says Google sign-in is ready, queue a short user-facing request that they connect Gmail from chat or CLI; do not start or complete sign-in during heartbeat.
- Prefer recalling existing contact memories to enrich email context (who is this person, what is the relationship).
