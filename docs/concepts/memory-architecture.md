<!--
title: Memory Architecture
summary: Multi-layered memory system with vectors, graphs, and neighborhoods
read_when:
  - "You want to understand how memory works"
  - "You want to understand the retrieval system"
section: concepts
-->

# Memory Architecture

Hexis implements a multi-layered memory system modeled after cognitive science research.

## In Brief

Five memory types (working, episodic, semantic, procedural, strategic) with vector embeddings for similarity search, graph relationships for reasoning, and precomputed neighborhoods for fast retrieval.

## The Problem

Simple RAG systems store text chunks with embeddings and retrieve by similarity. This works for knowledge retrieval but fails to capture:

- **Temporal relationships** -- what happened before/after
- **Causal chains** -- what caused what
- **Contradictions** -- when new information conflicts with existing beliefs
- **Importance decay** -- not all memories are equally important over time
- **Associative recall** -- remembering one thing triggers related memories

## How Hexis Approaches It

### Memory Types

```mermaid
graph TD
    Input[New Information] --> WM[Working Memory]
    WM --> |Consolidation| LTM[Long-Term Memory]

    subgraph "Long-Term Memory"
        LTM --> EM[Episodic Memory]
        LTM --> SM[Semantic Memory]
        LTM --> PM[Procedural Memory]
        LTM --> STM[Strategic Memory]
    end

    Query[Query/Retrieval] --> |Vector Search| LTM
    Query --> |Graph Traversal| LTM

    EM ---|Relationships| SM
    SM ---|Relationships| PM
    PM ---|Relationships| STM

    LTM --> |Decay| Archive[Archive/Removal]
    WM --> |Cleanup| Archive
```

1. **Working Memory** -- temporary buffer (UNLOGGED table for fast writes). Information enters here first. Expires automatically; important items are promoted.

2. **Episodic Memory** -- events with temporal context, actions, results, and emotional valence. Forms the agent's autobiographical timeline.

3. **Semantic Memory** -- facts with confidence scores, structured source provenance, and evidence-based belief revision. New evidence moves a belief's confidence through a calibrated, audited policy (`add_evidence` / `revise_memory_confidence`): independent corroboration closes a fraction of the remaining doubt, contradiction erodes it symmetrically, and known sources never double-count. Every change is explainable from `belief_revision_audit`. The agent's knowledge base.

4. **Procedural Memory** -- step-by-step procedures with success rate tracking. How the agent knows how to do things.

5. **Strategic Memory** -- patterns with adaptation history. High-level strategies learned from experience.

### Memory Infrastructure

**Vector embeddings** (pgvector) provide similarity-based retrieval via HNSW indexes. The `get_embedding()` function handles generation and caching.

**Graph relationships** (Apache AGE) enable multi-hop traversal: `TEMPORAL_NEXT` for narrative sequence, `CAUSES` for causal reasoning, `CONTRADICTS` for dialectical tension, `SUPPORTS` for evidence chains.

**Automatic clustering** groups memories into thematic clusters with emotional signatures and centroid embeddings.

**Precomputed neighborhoods** store associative neighbor data for each memory, enabling spreading activation without real-time graph traversal.

**Full-text history search** uses PostgreSQL GIN indexes across raw RecMem turns
and consolidated memories. It provides a free lexical fallback for exact names
and phrases even before a turn has an embedding or while an embedding provider
is unavailable.

**Memory decay** reduces importance over time with importance-weighted persistence. Permanent memories (from important ingestion) are exempt, as are **protected memories** (`metadata.protected`) — notably the origin memories seeded at consent, whose trust is pinned and which contradicting evidence can question but never silently rewrite.

**Memory formation is layered**: explicit writes (`remember`), document ingestion, and the **conscious-episode extraction** sweep — a maintenance job that reviews recent chat turns and heartbeat episodes and selectively promotes salient facts into durable memory (an importance floor gates the LLM pass; routine content yields nothing; near-duplicates corroborate existing beliefs instead of piling up).

The opt-in weekly learning review makes those changes inspectable as a diff. A
model may decide that enough changed and select records from a bounded candidate
set, but the review renders the database's exact content and evidence. Approval
keeps the change, correction creates a bitemporal replacement (semantic changes
pass through the contradiction ledger), and forgetting closes active recall
without erasing the historical account.

**Forgetting is deliberate and inspectable.** Rest-cycle consolidation can
compress ordinary episodic groups into lower-fidelity gists, but full source rows
are archived recoverably by default. Borderline or load-bearing groups become an
explicit user review: keep one with a finite chapter budget, journal the words
that should survive, or let the group compress. No expiry timer chooses. Each
completed summary records its actual source count and stored fidelity on the
Forgetting page. Irreversible pruning is a separate, off-by-default operator opt-in.

### Retrieval Model

Three performance tiers:

| Path | Method | Speed | Use Case |
|------|--------|-------|----------|
| **Lexical** | `search_cross_session_history` | Fast | Exact prior-turn or memory details without embeddings |
| **Hot** | `fast_recall` + neighborhoods + temporal | Fast | Primary retrieval |
| **Warm** | Cluster/episode lookups | Medium | Thematic search |
| **Cold** | Graph traversal (Apache AGE) | Slow | Multi-hop reasoning |

`fast_recall()` combines:
1. **Vector similarity** -- cosine distance on embeddings
2. **Neighborhood expansion** -- precomputed associative neighbors
3. **Temporal context** -- memories in the same episode get a boost

### Four Layers of Information Access

Distilled memories are one layer of a larger model. The agent works with
information the way a person works with a filing cabinet, a desk, and their
own recollection:

| Layer | What it holds | Lifetime |
|-------|---------------|----------|
| **Long-term memory** | Distilled, confidence-bearing facts and events (`memories`) | Decay/retention-managed; archived originals remain recoverable unless hard pruning is explicitly enabled |
| **Filing cabinet** | Exact preserved sources — files, emails, pages — with citable chunks (`source_documents`, `source_document_chunks`, original bytes in `source_artifacts`) | Durable; user data never auto-fades |
| **RecMem desk** | Passages deliberately loaded for multi-step reasoning; searchable, scrollable, pinnable, GC'd when idle | Mid-term |
| **Current context** | The live prompt window | One turn |

The agent climbs a retrieval ladder across these layers: recall first, follow
provenance to the exact source when wording matters, search the cabinet
(passage-grain search is hybrid lexical + vector with inspectable rank
components), load onto the desk for sustained work, scroll rather than dump,
and cite exact handles — document, chunk, page, path.

### Provenance and Contradiction Review

Recall and document tools return a stable citation envelope with the source,
trust level, and exact page/section/sheet locator. The conversation renderer
turns the model's `[^citation-id]` markers into expandable, linked footnotes;
sources below the live `memory.low_trust_threshold` are visibly marked. Trust
defaults come from `memory.source_trust_defaults`, so new source kinds can be
calibrated without changing application code.

New active semantic and worldview memories enter a durable, rate-limited
contradiction queue. PostgreSQL selects a bounded same-topic candidate set; the
model can only file pairs from that set and cannot decide which claim wins.
Confident findings become `contradiction_cases`, appear in the daily review and
the dashboard ledger, and remain inert until the operator chooses newer, older,
or context-dependent tension. A winner decision records a
`memory_supersessions` event and closes the loser's `valid_until`; it never
deletes the old row. Accepting tension retains both memories and the
`CONTRADICTS` relationship.

Evidence attached directly to one semantic belief still uses the separately
audited confidence-revision policy. Protected memories can be questioned but
are never silently rewritten.

### Point-in-Time State

The memory store retains validity rather than overwriting history. A memory's
`valid_from` / `valid_until` window and durable `memory_supersessions` events
answer which claims held at an instant; reverted supersessions reopen the old
claim after the recorded resolution. The first `belief_revision_audit` event
after that instant supplies its prior confidence and trust, reconstructing the
epistemic state that actually existed then.

`temporal_memory_snapshot()` applies those rules during retrieval.
`diff_memory_history()` compares two snapshots and joins the intervening event
ledgers, so “what changed?” and “why?” are answered from recorded evidence.
Chat activates these tools from temporal language, while the **Memory history**
dashboard exposes snapshots and diffs directly. Invalidated rows without an
explicit validity close are never treated as current, and private memories are
excluded whenever the calling context requires the sensitivity wall.

## Key Design Decisions

- **Single `memories` table** -- all memory types share one table with JSONB metadata for type-specific fields. Simpler than a table-per-type approach.
- **Neighborhoods over real-time graph traversal** -- precomputed during maintenance for hot-path speed
- **Embeddings as DB implementation detail** -- application code never sees vectors
- **UNLOGGED working memory** -- fast writes since we can afford data loss (it's temporary)

## Implementation Pointers

- Tables: `db/*_tables_memory.sql`
- Functions: `db/*_functions_memory.sql`
- Neighborhoods: `db/*_functions_maintenance.sql`
- Belief revision: `db/59_belief_revision.sql` (policy + `belief_revision_audit`)
- Provenance envelopes: `db/46i_functions_memory_provenance.sql`, `channels/presentation.py`
- Contradiction events: `db/46j_functions_contradictions.sql`, `services/contradictions.py`
- Point-in-time history: `db/46k_functions_temporal_memory.sql`, `core/tools/memory.py`
- Origin memories: `db/60_origin_memories.sql`
- Conscious-episode extraction: `db/61_functions_conscious_extraction.sql`, `services/extraction.py`
- Python client: `core/cognitive_memory_api.py`

## Related

- [Memory Types](../reference/memory-types.md) -- field-level reference
- [Memory Operations](../guides/memory-operations.md) -- practical usage
- [Database Is the Brain](database-is-the-brain.md) -- why memory is in Postgres
