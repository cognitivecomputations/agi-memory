<!--
title: Forget Well
summary: Review memory pressure, make load-bearing fade choices, and inspect factual compression receipts
read_when:
  - "You want to understand what Hexis is forgetting"
  - "Hexis asked whether to keep, journal, or compress memories"
section: guides
-->

# Forget Well

Hexis compresses ordinary episodic detail so recall stays useful as experience
accumulates. The safe default is reversible: a gist becomes the active memory and
the full source memories move to the archive. Irreversible pruning is separately
off by default.

## See the current pressure

Run:

```bash
hexis retention
```

This reports active episodic mass, its configured capacity, candidate groups,
recoverable archived originals, pending load-bearing reviews, and whether hard
pruning is enabled. To exercise the real retention functions without retaining
any change:

```bash
hexis retention dry-run
```

The dry run rolls the whole simulated rest cycle back before returning.

The dashboard's **Forgetting** page (`/forgetting`) shows the same live pressure,
low-fidelity memories, pending decisions, and completed compression receipts.

## Decide a load-bearing fade

A borderline group waits indefinitely. The review horizon can produce another
reminder, but it never selects an outcome. Choose one:

- **Keep** protects and reinforces the group. This spends one finite keep-budget
  point for the current life chapter.
- **Journal first** writes the words you deliberately want to preserve outside
  passive memory, then archives the group into a recoverable gist.
- **Let compress** archives the full group and queues its gist for summarization.

The dashboard completes each choice in place. A verified private-channel digest
also accepts only the exact replies it prints, such as `keep A1B2C3D4`,
`release A1B2C3D4`, or
`journal A1B2C3D4: Keep the lesson, not every detail.` Unknown or stale codes do
nothing and point back to the current page.

The autonomous heartbeat may spend its own finite budget to keep a memory, which
is non-destructive. A database trigger rejects heartbeat attempts to release or
journal a pending user review; the entire attempt rolls back.

## Read a compression receipt

Once summarization completes, Hexis records:

- the exact number and IDs of source memories;
- the resulting gist memory;
- the gist's stored fidelity, not a model estimate;
- a preview and timestamp.

The receipt appears on `/forgetting` and in a rate-limited outbox digest. Low
fidelity is presented as reconstruction rather than equally certain recall.

## Pause or recover

`hexis retention disable` pauses automatic rest-cycle consolidation. It does not
delete memories or archives. `hexis retention enable` first shows a rollback-only
preview and asks for confirmation.

With `retention.irreversible_pruning_enabled=false` (the default), archived source
memories remain in PostgreSQL and can be recovered operationally. Setting that
advanced database configuration to `true` authorizes hard deletion after the
configured grace window and capacity pruning; do that only after a database backup
and a deliberate review of `hexis retention --json`. There is intentionally no
timer or dashboard shortcut that silently turns it on.

## Related

- [Memory Architecture](../concepts/memory-architecture.md)
- [Memory Operations](memory-operations.md)
- [Backup and Recovery](backup-and-recovery.md)
- [Configuration Keys](../reference/config-keys.md)
