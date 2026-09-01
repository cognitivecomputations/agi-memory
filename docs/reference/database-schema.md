<!--
title: Database Schema
summary: Table reference and key columns for the Hexis database
read_when:
  - "You want to understand the database tables"
  - "You need to query the database directly"
section: reference
-->

# Database Schema

Key tables in the Hexis cognitive architecture. Source of truth: `db/*.sql`.

## Core Memory Tables

### memories

Primary long-term memory store. All durable knowledge, boundaries, goals, worldview, and episodic traces.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `type` | TEXT | episodic, semantic, procedural, strategic, worldview, goal |
| `content` | TEXT | Memory content |
| `embedding` | vector | Vector embedding (NOT NULL) |
| `importance` | FLOAT | 0.0-1.0 |
| `trust_level` | FLOAT | 0.0-1.0 |
| `status` | TEXT | active, archived, decayed |
| `metadata` | JSONB | Type-specific metadata; `metadata.protected=true` pins trust and exempts the memory from retention fade (contradicting evidence is flagged, never applied) |
| `created_at` | TIMESTAMPTZ | Creation time |
| `last_accessed_at` | TIMESTAMPTZ | Last retrieval |

### belief_revision_audit

Immutable audit of every confidence revision (policy `residual_v1`).

| Column | Type | Description |
|--------|------|-------------|
| `audit_id` | UUID | Primary key |
| `memory_id` | UUID | Revised memory (no FK: audits outlive deletions) |
| `stance` | TEXT | supports, contradicts |
| `evidence` | JSONB | Normalized evidence source |
| `prior_confidence` / `posterior_confidence` | FLOAT | Before/after |
| `prior_trust` / `posterior_trust` | FLOAT | Before/after |
| `applied` | BOOLEAN | Whether confidence moved |
| `reason` | TEXT | applied, duplicate_source, protected, disabled, not_semantic |
| `record` | JSONB | Full revision record |
| `record_digest_v1` | TEXT | SHA-256 of the record |

### action_claim_patterns

Data-driven patterns for the action-claim guardrail (tunable live).

| Column | Type | Description |
|--------|------|-------------|
| `claim_kind` | TEXT | memory_write, goal_backlog, scheduled, external_send, source_inspection |
| `pattern` | TEXT | POSIX regex, evaluated per sentence |
| `satisfied_by_tools` | TEXT[] | LIKE patterns over tool names (e.g. `mcp\_%`) |
| `require_arg_key` | TEXT | Argument that must be echoed in the sentence (e.g. `path`) |
| `enabled` | BOOLEAN | Pattern active |

### working_memory (UNLOGGED)

Short-lived buffer with expiry and promotion rules.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `content` | TEXT | Content |
| `embedding` | vector | Vector embedding |
| `context` | TEXT | Context tag |
| `expires_at` | TIMESTAMPTZ | Auto-expiry time |

### clusters

Thematic groupings with centroid embeddings.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `centroid_embedding` | vector | Cluster centroid |
| `label` | TEXT | Cluster label |
| `memory_count` | INT | Number of memories in cluster |

### episodes

Temporal groupings and summaries.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `time_range` | TSTZRANGE | Generated time range |
| `summary` | TEXT | Episode summary |
| `summary_embedding` | vector | Summary embedding |

### memory_neighborhoods

Precomputed associative neighbors for hot-path recall.

| Column | Type | Description |
|--------|------|-------------|
| `memory_id` | UUID | FK to memories |
| `neighbors` | JSONB | Precomputed neighbor data |
| `is_stale` | BOOLEAN | Needs recomputation |

## Source Documents & Desk

### source_documents

The filing cabinet: every ingested file/email/page preserved as normalized
text, keyed by content hash. Status `active` / `redacted` / `archived`;
redacted documents are frozen and never rehydrate.

| Column | Type | Description |
|--------|------|-------------|
| `content` | TEXT | Normalized extracted text |
| `content_hash` | TEXT | Unique dedup/handle key (sha256 of text) |
| `original_hash` | TEXT | sha256 of the original artifact bytes |
| `source_attribution` | JSONB | Provenance incl. `sensitivity` and `acquisition` (`user`/`agent`/`connector`) |

### source_document_chunks

Durable, citable slices with locators and their own embeddings for hybrid
retrieval. Keyed `UNIQUE(source_document_id, chunk_index)`; ids and
embeddings survive re-ingestion of unchanged content.

| Column | Type | Description |
|--------|------|-------------|
| `locator_kind` | TEXT | `char` / `page` / `section` / `sheet_row` / `slide` / `message` |
| `char_start`..`column_end` | INT | Exact-substring offsets plus page/sheet/row/column ranges |
| `heading_path` | TEXT[] | Markdown/DOCX heading trail |
| `embedding` | vector | Populated by the background embed queue (`embedding_status` lifecycle) |
| `chunker_version` | TEXT | Backfill marker (`hexis ingest backfill-chunks`) |

### source_artifacts

Original bytes (or a stable reference), captured before extraction and
deduped by `sha256`. `storage_kind`: `database` (BYTEA in-row), `filesystem`
(managed artifact dir), `connector`, `url`, `external`.

### source_extraction_runs

Which extractor produced a document's text: name/version, status, and
structured `warnings` (`ocr_used`, `truncated_rows`, `image_only_page`, …).
Failed runs may carry an artifact but no document — the source survives a
broken parser.

### RecMem desk

Not a separate table: desk items are `subconscious_units` rows tagged
`metadata.recmem.kind = 'source_document_desk'`, with `pinned_at`/`pinned_by`
protecting them from idle GC (redacted sources are swept regardless).

## Operational State

### config

JSON configuration for all system settings.

| Column | Type | Description |
|--------|------|-------------|
| `key` | TEXT | Config key (primary key) |
| `value` | JSONB | Config value |

### heartbeat_state / maintenance_state

Views over the `state` table projecting runtime state.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INT | Always 1 |
| `current_energy` | FLOAT | Energy currently stored in the heartbeat bank |
| `last_heartbeat_at` | TIMESTAMPTZ | Most recent start/completion timestamp |
| `next_heartbeat_at` | TIMESTAMPTZ | Database-selected adaptive due time |
| `active_heartbeat_id` | UUID | Current claim, or null when idle |
| `is_paused` | BOOLEAN | Whether the loop is paused |

### heartbeat_economy_state / heartbeat_outcomes / heartbeat_outcome_signals

Auditable heartbeat economy state. The singleton economy row anchors elapsed-time
regeneration. Each heartbeat outcome records regeneration, spend, durable outcome
counts, urgency, and the next cadence; signal rows point back to exact tool or
verified-feedback evidence.

### deliberation_sessions / deliberation_moves / deliberation_verdicts

Durable advisory council history. A session records the question, stakes, source,
selected personas, compact evidence context, and lifecycle status. Moves preserve
each perspective, targeted adversarial challenge, and synthesis with exact evidence
memory IDs. The one verdict per session retains agreements, disagreements, risks,
missing evidence, dissent, observable invalidation conditions, and an optional
episodic summary memory. These records explain advice; they do not authorize, gate,
or execute actions.

### consent_log

Durable consent contracts.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `provider` | TEXT | LLM provider |
| `model` | TEXT | Model identifier |
| `decision` | TEXT | accepted, refused |
| `signature` | TEXT | Consent signature |
| `created_at` | TIMESTAMPTZ | When consent was given |

### heartbeat_log

Heartbeat execution log.

| Column | Type | Description |
|--------|------|-------------|
| `heartbeat_number` | INT | Sequential number |
| `started_at` | TIMESTAMPTZ | Start time |
| `narrative` | TEXT | Heartbeat narrative |

## Performance Caches

### embedding_cache

Cached embeddings keyed by content hash.

### drives

Dynamic drive levels used during heartbeat decisioning.

### emotional_triggers

Pattern/embedding triggers for affect updates.

### memory_activation (UNLOGGED)

Short-lived activation tracking.

## Graph (Apache AGE)

Graph nodes and edges for multi-hop reasoning:

- **MemoryNode** -- linked to `memories` table
- **ConceptNode** -- linked to `concepts` table
- **SelfNode** -- the agent's self-representation
- **LifeChapterNode** -- narrative chapters

Edge types: `ASSOCIATED`, `TEMPORAL_NEXT`, `CAUSES`, `DERIVED_FROM`, `CONTRADICTS`, `SUPPORTS`, `INSTANCE_OF`, `PARENT_OF`, `IN_EPISODE`, `CONTESTED_BECAUSE`

## Extensions Required

| Extension | Purpose |
|-----------|---------|
| `pgvector` | Vector similarity search |
| `age` (Apache AGE) | Graph database |
| `btree_gist` | GiST index for range types |
| `pg_trgm` | Trigram text similarity |

## Key Views

| View | Description |
|------|-------------|
| `memory_health` | Aggregate statistics on memory system |
| `cluster_insights` | Cluster details ordered by size |
| `episode_summary` | Episode overview with memory counts |
| `stale_neighborhoods` | Neighborhoods needing recomputation |

## Related

- [Database API](database-api.md) -- SQL function reference
- [Database management](../operations/database.md) -- schema operations
