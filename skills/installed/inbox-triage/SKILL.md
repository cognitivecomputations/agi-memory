---
name: inbox-triage
description: Review an inbox, propose a safe triage plan, and apply only the email actions the user approves
category: communication
requires:
  tools: [email_list, email_read]
contexts: [chat]
aliases: [inbox, triage, cleanup, archive, unsubscribe]
bound_tools: [email_list, email_read, email_search, search_contacts, recall, remember, gmail_reply, gmail_label, gmail_spam_triage, gmail_delete, manage_backlog, todoist_create_task, asana_create_task]
---

# Inbox Triage

Reduce inbox load while keeping classification, replies, and destructive actions under the user's control. Use `email-digest` instead when the user wants only a summary or memory ingestion.

## Workflow

1. **Set scope.** Confirm the account when more than one is connected, then honor the requested folder, query, date range, or message limit. Default to a small recent inbox sample rather than scanning all mail.
2. **List before reading.** Use message metadata to identify likely high-value threads. Read only enough candidate messages to classify them accurately; do not bulk-read the whole mailbox.
3. **Classify with reasons.** Group messages into: reply needed, task or deadline, waiting/reference, safe archive, likely spam, and uncertain. Include message IDs or unambiguous sender/subject labels so the proposed actions are inspectable.
4. **Show the plan first.** Present counts and the proposed action for each thread. Keep uncertain messages out of action batches and ask about them.
5. **Apply the chosen batch.** Only after the user chooses, apply labels, archive, or mark spam to the exact messages approved. Report successes and failures per message.
6. **Handle replies separately.** Draft a reply, show recipients, subject, and body, then send only after explicit approval. A request to triage is not permission to send.
7. **Capture tasks deliberately.** Suggest tasks for real commitments. Create them in the local backlog, Todoist, or Asana only after the user chooses the destination and content.

## Control and Failure Boundaries

- Never permanently delete during ordinary triage. Use trash only after an explicit request; permanent deletion requires unmistakable, message-specific authorization.
- Never infer that a newsletter is spam or that silence means approval. Archive, spam, labels, replies, and tasks are separate choices.
- Respect the configured Gmail memory policy. Do not store email-derived content when it says `forget`; ask before storage when it says `ask`; summarize rather than storing raw bodies.
- Gmail connection is not autonomous permission. This skill is chat-only and runs because the user asked now.
- If the account is disconnected or a provider action is unavailable, preserve the proposed plan and give the exact connection or manual next step. Do not discard the useful read-only work.
