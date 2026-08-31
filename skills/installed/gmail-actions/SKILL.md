---
name: gmail-actions
description: Send Gmail messages, reply to threads, apply labels, triage spam, and delete messages with explicit action authorization
category: communication
requires:
  tools: [gmail_send, gmail_reply, gmail_label, gmail_spam_triage, gmail_delete, connector_action_policy_status]
contexts: [chat, heartbeat]
aliases: [email, emails, mail, inbox, message, reply, send, unread, archive, spam]
bound_tools: [gmail_send, gmail_reply, gmail_label, gmail_spam_triage, gmail_delete, connector_action_policy_status]
---

# Gmail Actions

Use this only after Gmail is connected and the user asks for an outward Gmail action: send, reply, label, mark spam/not-spam, archive, or delete.

## Principles

- A connected Gmail account is not permission to act. Check or establish connector action policy for ongoing/autonomous behavior.
- One-off chat actions still need a clear user request in the current conversation.
- Heartbeat actions require a matching DB-owned connector action policy; if none exists, do not improvise.
- Keep messages short, literal, and aligned with the user's stated intent. Do not escalate a narrow request into broader correspondence.
- Deletion is Trash-only by default and never autonomous. Call `gmail_delete` only when the user names a specific message in the current conversation; set `permanent` only after the user explicitly asks for irreversible deletion of that message. Never delete during heartbeats.

## Flow

1. If the user asks for ongoing behavior, use `connector-action-authorization` first.
2. For a one-off send, call `gmail_send` with `to`, `subject`, and `body`.
3. For a reply, call `gmail_reply` with `thread_id`, recipient, subject, and body.
4. For labels, call `gmail_label` with explicit `add_label_ids` and/or `remove_label_ids`.
5. For spam triage, call `gmail_spam_triage` with `mark_spam`, `mark_not_spam`, or `archive`.
6. For deletion the user explicitly requested, call `gmail_delete` with the `message_id`; leave `permanent` unset (Trash) unless the user asked for irreversible deletion in so many words.
