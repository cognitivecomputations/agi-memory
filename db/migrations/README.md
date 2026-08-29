# Schema migrations

`db/*.sql` is the **baseline** schema — Postgres applies it once, on a fresh volume.
This directory holds **forward deltas** that evolve an *existing* database **without
wiping data**. `current schema = baseline + all migrations`.

## Applying them

- Automatically on startup (`hexis up`, the workers, and the API each run the
  runner; it's advisory-locked, so they can't double-apply).
- Manually: `hexis migrate` (apply pending) · `hexis migrate --status` (list) ·
  `hexis upgrade` (pull/restart **without** wiping, then migrate).
- `hexis reset` still exists, but it's the **deliberate wipe** path (`down -v`) —
  not how you ship a schema change anymore.

Each applied file is recorded in the `schema_migrations` table and never re-run.
Its SHA-256 is checked on every migration startup and by `hexis migrate --status`;
changing an applied file fails loudly with the exact mismatched hashes and directs
the operator to restore it and write a new forward migration.

## Writing one

Name files `NNNN_short_slug.sql` (zero-padded, ordered). Keep them **additive and
idempotent** so they're safe on both fresh and existing databases:

- `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`,
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, `CREATE OR REPLACE FUNCTION`,
  `... ON CONFLICT DO NOTHING`, and guarded `DO $$ ... EXCEPTION WHEN ... $$`.
- `ALTER TYPE ... ADD VALUE IF NOT EXISTS` **cannot run in a transaction block** —
  put those in a file whose first line is `-- migrate:no-transaction`. Such files
  must be simple (`;`-separated statements, no `$$` blocks); each statement is run
  in autocommit.
- Everything else runs atomically in one transaction and rolls back on error.

Migrations that touch the AGE graph run with `LOAD 'age'` +
`search_path = ag_catalog, public` already set by the runner.

Also mirror the change into the baseline `db/*.sql` when it makes greenfield reading
clearer — the runner still applies the migration (idempotently) everywhere, so the
two never diverge in effect.
