---
name: operations
description: Back up the brain, check retention, and export or import configuration — the housekeeping that keeps this instance recoverable
category: system
requires:
  tools: [database_backup]
contexts: [heartbeat, chat]
aliases: [backup, backups, restore, export, import, retention, config]
bound_tools: [database_backup, backup_retention, config_export, config_import]
---

# Operations

Housekeeping for the instance itself: backups, retention checks, and configuration
export/import.

## When to Use

- **The user asks for a backup**, or asks whether one exists.
- **Before a change you cannot undo** — a migration, a bulk import, a reset. Back
  up first and say that you did.
- **Retention drift.** `backup_retention` reports what is kept and for how long;
  raise it if the answer is "nothing recent."
- **Moving or cloning an instance** — `config_export` then `config_import`.

## When Not to Use

- Routinely, on a hunch. Backups cost disk and time; the continuity drive already
  files a request when they go stale.

## Notes

- `config_export` writes configuration **keys and env-var names**, never secret
  values. Say so if the user asks what it contains.
- `config_import` overwrites live configuration. Show the user what will change
  before doing it, not after.
- These are instance-level actions. If something here fails, report the failure and
  the fix — never let a backup silently not happen.
