---
name: core-memory
description: Semantic recall, exact cross-session search, remembering, and normal continuity
category: system
requires:
  tools: [recall, search_history, remember]
contexts: [heartbeat, chat]
bound_tools: [recall, recall_at_time, diff_memory_history, search_history, remember, add_evidence, belief_history, open_memory, search_documents, open_document, open_documents, load_documents, search_document_chunks, open_document_chunk, load_document_chunks, list_desk, open_desk_item, pin_desk_item, unpin_desk_item, clear_desk, sense_memory_availability, read_journal, write_journal, search_journal, manage_goals, manage_schedule, manage_responsibility, manage_backlog, manage_operator_policies, list_document_fade_requests, resolve_document_fade, associate, explore_concept, explore_subgraph, trace_why, get_procedures, get_strategies]
---

# Core Memory and Continuity

Use this skill for ordinary continuity: recalling relevant memories, opening exact source material, storing new experiences, maintaining goals, consulting the permanent journal, and resolving pending document-fade approvals.

## When to Use

- The user asks about something that may already be in memory.
- The current conversation contains information worth preserving.
- A goal, schedule item, backlog item, or journal entry should be created or updated.
- The user answers a document-fade approval request.
- Before claiming you do not know something, check memory when the answer plausibly lives there.

## Method

1. Use `sense_memory_availability` for a cheap check when unsure whether memory is likely to help.
2. Use `recall` for targeted retrieval. Prefer specific queries over broad ones.
3. Use `recall_at_time` when the user asks what was known "as of" a past instant,
   and `diff_memory_history` when they ask what changed between two times or why.
4. Use `search_history` for exact names, phrases, or details from earlier
   sessions, especially when semantic recall is weak or embeddings are unavailable.
5. When the answer depends on an ingested source rather than distilled memory,
   climb the cabinet ladder: `search_documents` for files or
   `search_document_chunks` for citable passages -> `open_document` /
   `open_document_chunk` for read-only inspection -> `load_documents` /
   `load_document_chunks` (with a reason) when the material must stay
   searchable -> `search_history` with `sources=["desk"]` while reasoning ->
   `open_desk_item` to scroll long items -> cite the document/chunk/page
   handle. Run `list_desk` first -- do not re-load what is already on the
   desk. `pin_desk_item` what stays actively needed; `clear_desk` when done
   (cleared items archive; sources stay in the cabinet).
6. Use `remember` when a durable fact, event, preference, promise, or decision should persist.
7. Use journal tools only for deliberate permanent entries, not ordinary memory.
8. Use goal, schedule, or backlog tools when the user asks for ongoing commitments or work tracking. For durable watch-and-notify commitments that outlive the conversation ("let me know whenever X happens", "watch for email from Hope", recurring medication reminders), use `manage_responsibility`; use `manage_schedule` only for simple timed reminders that observe nothing.
9. Use document approval tools when the user explicitly says to keep or let an ingested document fade.
10. Use `manage_operator_policies` to list standing instructions or revoke one only when the verified operator explicitly asks. A replacement is a revoke followed by the operator stating the new standing instruction.

## Quality Guidelines

- Memory is evidence, not omniscience. If retrieved context is weak or absent, say so.
- Do not store secrets unless the user explicitly asks.
- Do not turn every minor sentence into memory; persist what will matter later.
- Keep tool use proportional. A direct answer does not need a recall if the answer is already present in the current conversation.
