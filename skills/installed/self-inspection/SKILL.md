---
name: self-inspection
description: Browse and search Hexis source code and inspect the live PostgreSQL database schema
category: system
requires:
  tools: [inspect_source, inspect_database_schema]
contexts: [heartbeat, chat]
aliases: [how, why, yourself, your, internals, schema, config, source, debug]
activation_phrases: [browse your source code, inspect your source code, inspect the database schema]
bound_tools: [inspect_source, inspect_database_schema, inspect_config, review_recent_actions, review_recent_changes, self_repair]
---

# Self Inspection

Use this skill to answer questions about how Hexis is implemented, what the
current checkout actually does, and what schema is running in PostgreSQL.

## Evidence Order

1. Inspect the source or live schema before making implementation claims.
2. Use `inspect_source` to locate and read the relevant code. Start with
   `search`, then read only the necessary line ranges.
3. Use `inspect_database_schema` for current database truth. Prefer
   `describe_relation` or `get_function` after a narrow `search`.
4. Use the core-memory skill separately when prior decisions or experiences are
   relevant. Memory is historical evidence; source and live schema are
   implementation evidence.
5. Reconcile differences explicitly. Baseline SQL files describe a fresh
   database, migrations evolve existing databases, and the live schema is the
   authority for what is running now.

## Source Method

- `inspect_source(action="list", path="...", file_pattern="...")` discovers
  repository files.
- `inspect_source(action="search", query="...", path="...", file_pattern="...")`
  finds definitions and call sites.
- `inspect_source(action="read", path="...", offset=..., limit=...)` reads a
  bounded, line-numbered range.
- Follow references across modules when needed. Do not infer behavior from a
  filename or one isolated function.

## Schema Method

- Start with `overview` only when the relevant object is unknown.
- Use `search` for a table, column, view, or stored-function name fragment.
- Use `describe_relation` for columns, defaults, constraints, indexes, and view
  definitions.
- Use `get_function` to inspect the actual stored definition and overloads.
- The schema tool is metadata-only and does not accept arbitrary SQL.

## Reporting

- Distinguish facts observed in source, facts observed in the live schema, and
  inferences.
- Cite repository-relative paths and function/relation names.
- State when the running schema differs from baseline files or migrations.
- Use `review_recent_changes` to see what changed about your own substrate —
  migrations, code rebuilds, prompt edits, and operator config decisions.
- When tool calls, heartbeat steps, integrations, or memory retrieval have been
  failing, use `self_repair` to list unresolved defect reports, diagnose one
  with the source/schema evidence gathered here, and mark it resolved only
  after verification. It records and reasons about defects; it never silently
  edits source code.
- Do not claim access to model weights, provider infrastructure, hardware, or
  secrets unless another explicit capability supplies that evidence.
- Do not expose credentials, `.env` contents, or unrelated private files.
