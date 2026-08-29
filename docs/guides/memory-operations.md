<!--
title: Memory Operations
summary: Practical usage of remember, recall, and hydrate
read_when:
  - "You want to search the agent's memories"
  - "You want to understand how memory works in practice"
section: guides
-->

# Memory Operations

Practical guide to storing, retrieving, and using the agent's memories.

## Quick Start

```bash
# Search memories from the CLI
hexis recall "user preferences"

# Search with type filter
hexis recall "how to deploy" --type procedural

# JSON output
hexis recall "project goals" --type goal --json
```

## Core Operations

### Remember (Store)

In chat, the agent automatically forms memories from conversations. You can also store memories programmatically:

```python
from core.cognitive_memory_api import CognitiveMemory

async with CognitiveMemory.connect(DSN) as mem:
    await mem.remember("User prefers dark mode and concise responses")
```

Or directly via SQL:

```sql
SELECT create_semantic_memory('User prefers dark mode', 0.9);
```

The agent-facing `remember` tool also accepts optional provenance — a
`sources` array (`{kind, ref, label, author, trust}`) and a `confidence`
score. Semantic memories record every source and derive their trust from
them, which is what makes a belief revisable later.

### Add Evidence (Revise a Belief)

When new information bears on a belief that already exists, don't create a
duplicate — attach evidence. The `add_evidence` tool takes a `memory_id`
(from recall), a stance (`supports` or `contradicts`), and a source, then
revises the belief's confidence through the audited revision policy and
returns the prior and posterior values ("confidence 0.50 → 0.66"). Duplicate
sources are merged without moving confidence, and every change is recorded in
`belief_revision_audit`:

```sql
SELECT * FROM belief_revision_audit WHERE memory_id = '<id>' ORDER BY created_at;
```

The agent-facing mirror of that audit is the `belief_history` tool: given a
memory id it returns the belief's current state, truth profile, revision
history, evidence links, and contradicting sources — the full answer to "why
do I believe this?".

### Recall (Retrieve)

Recall uses vector similarity search augmented with precomputed neighborhoods and temporal context:

```bash
hexis recall "UI preferences" --limit 5
```

```python
async with CognitiveMemory.connect(DSN) as mem:
    memories = await mem.recall("UI preferences", limit=5)
```

```sql
SELECT * FROM fast_recall('UI preferences', 5);
```

Recall results include each memory's complete `source_attribution`, `trust`,
and stable citation envelope. Claims drawn from those results carry linked
footnotes in chat by default; low-trust sources are visibly marked rather than
presented with the same certainty as strong evidence. The agent-facing tool also accepts `min_score` (a
relevance floor — drop weak matches instead of padding to the count), and its
default/maximum counts are config-driven budgets (`memory.recall_default_limit`,
`memory.recall_max_limit`).

### Search History (Exact Cross-Session Retrieval)

Use full-text history search for exact names, phrases, operators, or details from
prior conversations. It searches active raw turns and consolidated memories in
Postgres without calling an embedding provider:

```python
async with CognitiveMemory.connect(DSN) as mem:
    results = await mem.search_history(
        '"project lantern" deployment',
        sources=["turn", "memory"],
        limit=20,
    )
```

```sql
SELECT *
FROM search_cross_session_history('"project lantern" deployment', 20);
```

The agent-facing `search_history` tool excludes raw turns from its current UUID
session by default because the live conversation is already in context. Set
`exclude_current_session` to false when the current stored turn is relevant.
Inactive memories, expired memories, and redacted or archived raw turns are
never returned.

### Recall at a Point in Time

Temporal questions use the validity record rather than trying to infer the past
from today's memories. Ask naturally in chat—“As of last Tuesday, what did you
know about the Manning deal?” or “Has that changed since June?”—and the
conversation agent selects `recall_at_time` or `diff_memory_history` from the
temporal cue. The **Memory history** dashboard provides the same journey as an
explicit snapshot or side-by-side comparison.

Point-in-time results include memories that have since been superseded, but only
when their `valid_from` / `valid_until` and supersession events prove they were
valid at the requested instant. Historical confidence and trust are rebuilt
from `belief_revision_audit`; they are not copied from the current row. A diff
returns what became valid, what expired, and the recorded supersession,
contradiction decision, or evidence-revision reason:

```sql
SELECT temporal_memory_snapshot(
    'Manning retainer',
    '2026-06-01T12:00:00Z'::timestamptz
);

SELECT diff_memory_history(
    'Manning retainer',
    '2026-06-01T12:00:00Z'::timestamptz,
    CURRENT_TIMESTAMP
);
```

If embedding retrieval is unavailable, the snapshot still runs exact lexical
history and reports that degraded mode. An empty result means the record has no
matching memory at that instant; it does not mean the record asserted the
opposite.

### Exact Sources: the Filing Cabinet and the Desk

Distilled memories never carry whole files. When exact wording matters, the
agent climbs the retrieval ladder over two additional layers:

- **Filing cabinet** — every ingested file/email/page preserved verbatim
  (`source_documents`) with durable, citable chunks
  (`source_document_chunks`). Search with `search_documents` (file grain) or
  `search_document_chunks` (passage grain — hybrid full-text + vector, each
  hit carrying a locator like page 4, sheet `Vendors` row 12, or a heading
  path, plus DB-generated `rank_components` explaining the score). Open exact
  passages with `open_document` / `open_document_chunk`; open results include
  any extraction warnings (OCR used, rows truncated).
- **RecMem desk** — mid-term working material. `load_documents` /
  `load_document_chunks` place sources on the desk with a reason; then
  `search_history` with `sources=["desk"]` searches them while reasoning.
  `list_desk` shows what is loaded, `open_desk_item` scrolls long items
  window by window, `pin_desk_item` protects what stays actively needed, and
`clear_desk` archives the rest — the cabinet copy always survives.

From the terminal the same surfaces are `hexis docs ...` and `hexis desk ...`
(see the [CLI Reference](../reference/cli.md)); in the web UI they are the
**Documents** and **Desk** pages. `open_memory` on a distilled memory returns
`source_documents` and `source_chunks` handles, so provenance is always one
hop away.

### Review Contradictions

Hexis checks new semantic and worldview memories against a bounded set of
same-topic memories once per day. A finding never changes either memory by
itself. Pending cases appear in the Conversation inbox and on the
**Contradictions** page with source, trust, and three explicit choices:

- **Newer is right** — close the older memory's validity window.
- **Older is right** — close the newer memory's validity window.
- **Both, by context** — retain both as an explicit tension.

The losing row remains queryable in history. From a verified private channel,
reply with the numbered choice and the eight-character case code shown in the
daily digest, such as `3 ABC12345`. Bare numbers and replies from non-operator
contacts stay ordinary conversation.

### Hydrate (Context Building)

Hydrate gathers a rich context package for LLM prompts -- memories, goals, identity, worldview:

```python
async with CognitiveMemory.connect(DSN) as mem:
    ctx = await mem.hydrate("How should I respond?", include_goals=True)
```

## Memory Types

| Type | Purpose | Example |
|------|---------|---------|
| **Working** | Temporary buffer with expiry | Recent conversation context |
| **Episodic** | Events with temporal context | "User asked about deployment at 3pm" |
| **Semantic** | Facts with confidence scores | "Python 3.10+ is required" |
| **Procedural** | Step-by-step procedures | "How to deploy: 1. Build... 2. Push..." |
| **Strategic** | Patterns and strategies | "User responds well to concise answers" |
| **Worldview** | Beliefs and identity | "I value honesty over comfort" |
| **Goal** | Active objectives | "Help user learn Python" |

See [Memory Types reference](../reference/memory-types.md) for full field details.

## CLI Recall Options

```bash
hexis recall <query>                    # basic search
hexis recall <query> --limit 20         # more results
hexis recall <query> --type semantic    # filter by type
hexis recall <query> --json             # JSON output
```

## How Retrieval Works

The `fast_recall()` function combines three retrieval strategies:

1. **Vector similarity** -- cosine distance on embeddings (pgvector HNSW index)
2. **Neighborhood expansion** -- precomputed associative neighbors for spreading activation
3. **Temporal context** -- memories in the same episode get a temporal boost

Results are scored, deduplicated, and ranked.

`search_cross_session_history()` is the complementary lexical path. PostgreSQL
web-search syntax supports quoted phrases, `OR`, and minus-prefixed exclusions;
the partial GIN index on active raw turns keeps this path independent of vector
generation and RecMem embedding lag.

## Working with Memories in SQL

```sql
-- Search active memories
SELECT * FROM fast_recall('what the user likes', 10);

-- Free lexical search across prior turns and consolidated memories
SELECT * FROM search_cross_session_history('"release checklist"', 20);

-- Reconstruct what was valid at one historical instant
SELECT temporal_memory_snapshot('release plan', '2026-06-01T12:00:00Z');

-- Explain what changed between two instants
SELECT diff_memory_history('release plan', '2026-06-01T12:00:00Z', CURRENT_TIMESTAMP);

-- Count memories by type
SELECT type, count(*) FROM memories WHERE status = 'active' GROUP BY type;

-- View memory health
SELECT * FROM memory_health;

-- Search working memory
SELECT * FROM search_working_memory('current task');
```

## Related

- [Ingestion](ingestion.md) -- feeding content into memory
- [Memory Types](../reference/memory-types.md) -- detailed type reference
- [Memory Architecture](../concepts/memory-architecture.md) -- how memory works architecturally
- [Database API](../reference/database-api.md) -- SQL function reference
