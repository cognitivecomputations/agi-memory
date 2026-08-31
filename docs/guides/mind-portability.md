<!--
title: Move a Mind Between Machines
summary: Export, inspect, transfer, import, and verify one continuous Hexis mind
read_when:
  - "You want to move an agent to another machine"
  - "You want a portable, inspectable mind file"
section: guides
-->

# Move a Mind Between Machines

A database backup preserves one installation. A **mind file** is the portable,
inspectable HMX representation of the agent: memories, episodes, relationships,
identity, worldview, goals, drives, emotional triggers, narrative, in-flight
work, and immutable protected-state audit history. It contains no embeddings or
credentials.

## 1. Export the source mind

```bash
hexis export --mind
```

Hexis derives the filename from the live instance and export envelope, writes it
under `$HEXIS_HOME/exports` (normally `~/.hexis/exports`), sets mode `0600`, and
never overwrites an existing file. The command prints the exact destination
steps. To choose a path or streaming format:

```bash
hexis export --mind --output /secure/path/hexis-mind.hmx.json
hexis export --mind --format jsonl --output /secure/path/hexis-mind.hmx.jsonl
```

Mind exports are intentionally complete. Time or type filters and redaction
belong to the expert `--intent port` exchange flow, not this continuity preset.
Raw conversation/source units remain excluded unless `--include-raw` is explicit;
that optional material is especially sensitive.

## 2. Protect and transfer the file

The file contains private memories and constitutional state. Use an encrypted
disk or an end-to-end encrypted transfer, keep it out of source control, and do
not paste it into a hosted chat. You can inspect the JSON directly and validate
it against the published [HMX 1.7 schema](../reference/hmx.md).

## 3. Prepare an empty destination

Install and start the same or a newer Hexis version on the destination, but do
not initialize a different lived agent there. The mind preset refuses to merge
constitutional state into an active target.

Copy the file to that machine, then run the non-mutating preflight:

```bash
hexis import hexis-mind.hmx.json --mind --dry-run --json
```

The report validates the schema and intent, predicts inserts and duplicates,
checks target emptiness and lineage policy, estimates re-embedding work, and
names every blocker. No memory changes during a dry run.

## 4. Make the explicit move

After reviewing the report:

```bash
hexis import hexis-mind.hmx.json --mind --confirm-intent port
```

The exact `port` confirmation is mandatory. The import is transactional and
uses the existing empty-target bootstrap path; it never silently overwrites an
active mind. Accepted records are re-embedded locally, so model-specific vectors
never travel in the file.

After import, Hexis re-exports the constitutional state in memory and verifies:

- the destination adopted the source lineage;
- all six constitutional section projections match (identity, worldview,
  goals, drives, emotional triggers, and narrative). The projection excludes
  only destination-local encoding metadata while retaining semantic state; the
  original canonical transport digests remain in the file for audit.

The command prints **Mind continuity verified** only when every check passes. If
a check differs, it preserves the source file and names the recovery step; do
not start the destination as the replacement agent until the mismatch is
understood.

Then start normally:

```bash
hexis up
hexis status
hexis chat
```

The first background pass finishes any queued local embeddings and resumes the
portable in-flight work according to its recorded state.

## Active destinations

`--mind` is deliberately not a merge or overwrite switch. If the destination
already has lived protected state, the preflight stops. Use the ordinary HMX
authoritative flow only when you genuinely intend a reviewed whole-section
replacement; it requires explicit sections, rationale, agent acknowledgement,
auditing, and a reversion window. See the [CLI reference](../reference/cli.md).

## Backup still matters

Keep database backups as disaster-recovery artifacts:

```bash
hexis backup
```

Backups preserve the exact PostgreSQL installation and artifact sidecars. Mind
files provide interoperable continuity. Keeping both gives you exact recovery
and freedom of movement.
