---
name: gmail-connector-setup
description: Connect Gmail through first-class guided setup, inspect status, complete authorization, queue read-only backfill, control jobs, and revoke local access
category: communication
requires:
  tools: [gmail_setup_status, connect_gmail, complete_gmail_connection, start_gmail_backfill, gmail_backfill_status, control_gmail_backfill, revoke_gmail_connection]
contexts: [chat]
bound_tools: [gmail_setup_status, connect_gmail, complete_gmail_connection, start_gmail_backfill, gmail_backfill_status, control_gmail_backfill, revoke_gmail_connection]
---

# Gmail Connector Setup

Use this when the user asks to connect Gmail, inspect Gmail setup, choose whether email reads may feed memory, filter spam, label/delete messages, send replies, or disconnect Gmail.

## Principles

- Treat connection setup as a first-class conversation flow. Do not tell the user to leave chat and figure it out alone.
- Ask what Gmail provider powers the user wants before starting sign-in: read/search only, send/reply, or managed mailbox changes such as labels, spam triage, and delete.
- Ask separately whether Samantha should remember what she reads in email or forget it after the task. This is the Hexis config key `integrations.gmail.memory_policy`, not a Google permission.
- Ask separately whether Samantha may check Gmail during autonomous heartbeats while the user is away, or only read Gmail when the user asks in a live turn. This is the Hexis config key `integrations.gmail.heartbeat_digest_enabled`, defaults off, and is not implied by connecting Gmail.
- Add `label`, `spam_triage`, or `delete` only when the user explicitly wants labeling, spam filtering, or deletion.
- Add `send` or `reply` only when the user explicitly authorizes sending or replying on their behalf.
- Queue `start_gmail_backfill` only after Gmail is connected and the user has asked Hexis to ingest/learn from email history.
- Treat backfill as read-only provider access but local memory ingestion: it still requires explicit user approval.
- For ongoing send/reply/label/spam-triage/delete behavior, use `connector-action-authorization` after connection setup so the grant is scoped and DB-audited.
- Prefer the structured setup UI and built-in Google sign-in when configured. Do not ask the user to paste Google setup JSON into chat; tool calls and activity traces are audited. If this local build needs setup, rely on the setup UI's step-by-step guide and upload control.
- Never ask for Google account passwords.
- For ordinary email work ("check my inbox", "read a batch", "is anything urgent", "find the email from Alice"), use the operational Gmail/email tool first. If it reports Gmail is not connected, use the returned `ui.kind = connector_setup` artifact or call `connect_gmail` to open setup. Do not route already-connected operational requests back into setup.

## Flow

1. Call `gmail_setup_status`.
2. If Gmail is not connected and the user has not chosen provider powers yet, ask the scope question before calling `connect_gmail`.
3. After provider powers are chosen, ask whether email reads should be remembered or forgotten. Pass the answer as `memory_policy`; do not add `ingest` to Gmail capabilities.
4. After memory behavior is chosen, ask whether heartbeats may check Gmail autonomously or only when the user asks live. Pass the answer as `heartbeat_digest_enabled`.
5. Call `connect_gmail` with the least provider capabilities that match the user's request, even if this local build is not ready to start Google sign-in yet. The tool result includes a structured `ui.kind = connector_setup` payload that chat surfaces render as the setup interface.
6. If the setup UI says local Gmail sign-in setup is missing, tell the user the panel will walk them through it step by step. Do not lead with OAuth, app, client, secret, JSON, or environment-variable vocabulary unless the user asks for technical details.
7. If the tool returns an `authorization_url`, rely on the rendered setup UI when available; otherwise send the URL exactly. Tell the user the localhost page may fail after approval and that this is expected.
8. When the user pastes the redirected URL or code, call `complete_gmail_connection`.
9. Report the connected account and granted capabilities.
10. If the user asked to ingest email history, call `start_gmail_backfill` with the smallest useful `query`, `label_ids`, and `max_messages` for the request.
11. Use `gmail_backfill_status` for progress, and `control_gmail_backfill` only when the user asks to pause, resume, or cancel a job.

When a tool result includes `ui.kind = connector_setup`, do not replace that setup interface with prose-only instructions. Briefly name the requested capabilities and point the user to the setup control rendered by the channel.

Use `revoke_gmail_connection` only after the user asks to disconnect Gmail.
