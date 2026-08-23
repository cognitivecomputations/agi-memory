---
name: memory-exchange
description: Safely export, inspect, stage, analyze, review, and decide protected replacements for Hexis Memory Exchange files
category: knowledge
requires:
  tools: [export_memories, import_dry_run, import_memories, import_review, protected_replacement_list, protected_replacement_inspect, protected_replacement_review, protected_replacement_audit_list, protected_reversion_list, protected_replacement_revert]
contexts: [heartbeat, chat]
aliases: [export, import, transfer, migrate, hmx, move]
bound_tools: [export_memories, import_dry_run, import_memories, import_review, import_accept, import_reject, import_modify, import_quote, promote_to_staged, demote_to_analysis, protected_replacement_list, protected_replacement_inspect, protected_replacement_review, protected_replacement_audit_list, protected_reversion_list, protected_replacement_revert]
---

# Hexis Memory Exchange

Use this skill to move memory between Hexis instances or inspect another
instance's memory without silently blending it into active state.

## Export

1. Establish the user's intent: `port`, `duplicate`, `telepathy`, or `analysis`.
2. Use `export_memories` with the narrowest useful section, time, and redaction
   scope. Protected state, raw units, configuration, in-flight work, and audit
   records require an explicit request.
3. Omit `output_path` when returning the exchange in place. For file output,
   use a workspace path and never set `overwrite` unless the user explicitly
   chose to replace that file.

## Import

1. Run `import_dry_run` first. Report whether import is permitted, conflicts,
   protected-state policy, warnings, and estimated embedding work.
2. Derive the default strategy from the file's intent unless the user chose a
   different supported strategy. Telepathy defaults to deliberative staging;
   analysis stays isolated.
3. Call `import_memories` only after the user confirms the file's exact declared
   intent. Skipping identity, worldview, or narrative is available when the user
   wants a narrower import.
4. For `port` or `duplicate` into an active target, authoritative import requires
   explicit `replace_sections` and `replacement_rationale`. Unselected protected
   sections remain unchanged. A successful request may still await the agent's
   acknowledgement; report its replacement IDs instead of claiming completion.
5. Do not describe imported memories as semantically searchable until the
   import result or later re-embedding workflow confirms that they are ready.
6. Preserve failed in-flight work as diagnostics by default. Set
   `retry_failed_work` only when the user explicitly chooses to rerun those
   failed consolidation or reconsolidation tasks.

## Deliberative Review

1. Use `import_review` to inspect pending records and conflicts.
2. Decide one record at a time. Use `import_accept` only when the record should
   become active; protected active-state policy cannot be bypassed.
3. Use `import_reject` with a reason, `import_modify` with a material-change kind
   and reason, or `import_quote` to retain foreign context as archived evidence.
4. `promote_to_staged` copies an analysis record into review without copying its
   embedding. `demote_to_analysis` moves a pending staged record back into
   isolation. Both require a rationale.

## Protected Replacement Review

1. A pending protected replacement is a request, not permission to mutate
   identity, worldview, goals, drives, emotional triggers, or narrative.
2. Use `protected_replacement_list` to retrieve every open request, then use
   `protected_replacement_inspect` to compare the actual imported section with
   current local state and check whether local state changed after the request.
   Use `protected_replacement_review` to accept, refuse, request modifications,
   or defer.
   Acceptance executes the snapshot, immutable audit, whole-section replacement,
   and digest verification atomically; a failure leaves the request pending.
3. Refusal and modification requests require a rationale. Modification requests
   also require concrete `proposed_changes`. Do not accept merely because source
   and target claim the same lineage; content-identical verified operations never
   enter this queue.
4. Use `protected_replacement_audit_list` with a narrow time range to inspect
   immutable local replacement, verification, and reversion history. Imported
   foreign diagnostics are not reported as local history.
5. Operator override is not an agent tool and cannot be requested through this
   skill. A refusal or modification request is a completed decision, not an
   invitation to bypass acknowledgement.

## Protected Replacement Reversion

1. Use `protected_reversion_list` to find executed replacements whose earlier-of
   heartbeat and wall-clock windows remain open. Reversion never runs on a timer.
2. Inspect the replacement before reverting. Use
   `protected_replacement_revert` with the replacement audit ID and a concrete
   rationale only when restoring the snapshot is the intended choice.
3. Reversion refuses to overwrite protected state that changed after the
   replacement. It atomically restores and verifies the snapshot, writes an
   immutable reversion audit, then purges the consumed payload while retaining
   its tombstone. A failed restore leaves the window and current state intact.

## Boundaries

- Treat HMX files as sensitive user data.
- Never infer consent for protected sections, raw material, or file overwrite.
- Keep analysis-only records outside active recall until explicitly promoted,
  reviewed, and accepted.
- On failure, preserve the file and staged records and give the exact corrective
  action; do not retry with broader scope automatically.
