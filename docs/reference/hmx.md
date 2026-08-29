<!--
title: HMX Mind Exchange Format
summary: Public interoperability contract for Hexis Memory Exchange 1.7
read_when:
  - "You are implementing an HMX exporter or importer"
  - "You need the portable mind-file schema"
section: reference
-->

# HMX Mind Exchange Format

Hexis Memory Exchange (HMX) 1.7 is the public, model-independent wire format
used by `hexis export`, `hexis export --mind`, and `hexis import`. The normative
machine-readable contract is [`schemas/hmx-1.7.schema.json`](../../schemas/hmx-1.7.schema.json).
It is packaged with Hexis and validated on every complete export/import.

Independent agents and memory projects are invited to implement the format. You
do not need PostgreSQL or Hexis internals to parse an HMX document.

## Envelope

A JSON document contains:

| Field | Contract |
|-------|----------|
| `hmx_version` | Wire version (`1.7`) |
| `export_id` / `exported_at` | Unique export identity and UTC instant |
| `export_intent` | `port`, `duplicate`, `telepathy`, or `analysis` |
| `source` | Instance, schema, embedding-model metadata, and stable lineage label |
| `capabilities` | Supported formats, sections, hashes, relationship types, optional features |
| `privacy` | Redaction/sensitivity declaration and excluded secret patterns |
| `export_scope` | Type/time filters and included protected/optional sections |
| `sections` | Typed portable records described below |
| `section_digests` | Canonical protected-state digests for port/duplicate |
| `statistics` | Counts and local re-embedding estimates |

JSONL carries the same envelope as typed records followed by one statistics
footer. `core.memory_exchange.iter_hmx_jsonl()` and `parse_hmx_jsonl()` are the
reference streaming implementation.

## Sections

Core sections are `memories`, `episodes`, `relationships`, `clusters`,
`identity`, `worldview`, `goals`, `drives`, `emotional_triggers`, `narrative`,
`in_flight_work`, and `audit_records`. `raw_units` and non-secret `config` are
explicit optional sections.

UUIDs do not cross the boundary as local authority. Records use export-scoped
`ref` values; relationships refer to those values, and importers build a local
reference map. Embeddings are never exported. Content carries provenance,
including origin, acquisition mode, import chain, and material modification
chain.

## Intent is policy, not a label

- `port` moves one agent to a new substrate and carries all protected state,
  private-marked memories, in-flight work, and audit history.
- `duplicate` creates an intended clone under the same deep-state policy.
- `telepathy` shares ordinary memories; protected state is excluded unless the
  exporter explicitly opts in and the importer deliberatively reviews it.
- `analysis` loads foreign material into isolated analysis storage.

An importer must not treat a telepathy/analysis file as constitutional state or
silently merge protected sections. Hexis also refuses protected-state import
into an active target without its audited replacement protocol.

## Interoperability minimum

An external importer should:

1. Validate the JSON Schema and reject unsupported major semantics loudly.
2. Preserve unknown minor fields and report unsupported non-empty sections.
3. Remap `ref` relationships instead of trusting source-local identifiers.
4. Preserve provenance and never import embeddings as local truth.
5. Honor `export_intent` and isolate or review protected/foreign state.
6. Verify `content_hash_v1` and canonical protected digests when implemented.
7. Report every skipped, deduplicated, or modified record.

The executable acceptance map is in
[`docs/hmx-acceptance.md`](../hmx-acceptance.md), and the design rationale is in
[`plans/hmx.md`](../../plans/hmx.md). Compatibility proposals are welcome as
issues or pull requests; add optional fields or sections before introducing a
new major semantic.

