<!--
title: Ingestion
summary: Feed documents and knowledge into your agent's memory
read_when:
  - "You want to load documents into memory"
  - "You want to understand ingestion modes"
section: guides
-->

# Ingestion

Feed documents, files, URLs, and other content into your agent's memory.

## Quick Start

```bash
# Ingest a file
hexis ingest --file ./notes.md

# Ingest a directory (recursive)
hexis ingest --input ./documents

# Ingest a URL
hexis ingest --url https://example.com/article
```

## How It Works

Ingestion is tiered and source-preserving:

1. Preserves the raw artifact in the source-document filing cabinet.
2. Creates an **encounter memory** (the agent "encountered" this content).
3. **Appraises** the content (importance, relevance, emotional valence).
4. **Extracts** semantic knowledge based on the chosen mode, with each memory
   carrying provenance back to the preserved source document.

Content is deduplicated at two levels. Whole documents are receipt-tracked by
content hash — re-ingesting an identical document is skipped. Individual
extracted facts route through the memory dedup policy: a fact that near-matches
an existing memory **corroborates** it (the document is merged as a source, the
belief's confidence rises through the audited revision policy, and a `SUPPORTS`
edge links the reading encounter to the belief) instead of being re-stored.

## Ingestion Modes

| Mode | Behavior | Best for |
|------|----------|----------|
| `fast` | Quick chunking + fact extraction + graph linking | Default and general use |
| `slow` | Conscious mini-RLM loop per chunk | Short, high-value documents |
| `hybrid` | Fast pass, then slow-processes high-signal chunks | Large or mixed-value documents |

```bash
hexis ingest --input ./docs --mode fast
hexis ingest --file paper.pdf --mode slow
hexis ingest --input ./corpus --mode hybrid
```

`auto`, `standard`, `deep`, and `shallow` are accepted as compatibility aliases
for older scripts, but new workflows should use `fast`, `slow`, or `hybrid`.

## Conscious Ingestion

`slow` and `hybrid` use the RLM loop for conscious reading against the
agent's existing knowledge. The reader can search existing memories, compare the
new document against worldview beliefs, and mark extracted claims as accepted,
contested, or uncertain.

```bash
# Conscious slow reading of an important document
hexis ingest --file philosophy.md --mode slow

# Hybrid: fast scan, deep-read only what matters
hexis ingest --input ./research/ --mode hybrid
```

Contested content is stored with a `contested` flag and `CONTESTED_BECAUSE`
graph edges linking to the beliefs that caused rejection.

## Raw Source Documents

Every ingestion path preserves the raw source first. The agent can later browse
the filing cabinet with `search_documents`, open exact sources with
`open_document` or `open_documents`, and deliberately load substantial sources
onto the RecMem desk with `load_documents`. Desk-loaded chunks are mid-term
working material: they are searchable with `search_history` using
`sources=["desk"]`, but they are not permanent distilled memories unless the
agent explicitly stores a durable memory.

## Durable Chunks, Artifacts, and Extraction Runs

Beyond the normalized text, ingestion now preserves three more layers:

- **Source chunks** (`source_document_chunks`): stable, citable slices with
  locators — page ranges for PDFs, sheet/row ranges for spreadsheets, heading
  paths for Markdown/DOCX. Chunk ids and embeddings survive re-ingestion of
  unchanged content. The agent searches them with `search_document_chunks`
  (hybrid full-text + vector, with inspectable `rank_components`), opens exact
  passages with `open_document_chunk`, and loads them onto the desk with
  `load_document_chunks`. Pre-existing documents gain chunks via
  `hexis ingest backfill-chunks`; embeddings are generated in the background
  by the maintenance worker.
- **Original artifacts** (`source_artifacts`): the exact bytes a document was
  extracted from, captured *before* the parser runs. A failed parse never
  loses the source — the artifact is preserved with a `failed` extraction run
  so you can fix dependencies and re-ingest. Bytes up to
  `ingest.artifact_max_db_bytes` (25 MB default) live in-DB (and ride
  `hexis backup`); larger files go to `$HEXIS_ARTIFACT_DIR`
  (default `~/.hexis/artifacts`), tarred as a backup side-car.
- **Extraction runs** (`source_extraction_runs`): which extractor produced the
  text, with structured warnings — `ocr_used`, `image_only_page`,
  `truncated_rows`, `unsupported_feature`. Warnings surface in
  `open_document`, search results, `hexis docs info`, and `hexis ingest status`.

## Where Ingestion Comes From

All paths converge on the same pipeline and receipts:

1. **CLI bulk** — `hexis ingest --file/--input/--url/--github/--stdin`.
2. **Chat** — attaching a file preserves its original bytes and reads its text
   immediately (`POST /api/attachments`), so the agent can discuss it in the
   same turn; sending the message starts the durable background job
   (`POST /api/attachments/{artifact_id}/ingest`). Large pastes attach the
   same way. Nothing is ingested until the message is actually sent.
3. **Web UI** — the **Ingest** page: multi-file upload, paste box, URL field,
   and a live job list showing what ran, what is pending, and what failed and why.
4. **The agent itself** — `url_ingest`/`git_ingest` during heartbeats. Sources
   are stamped with `acquisition` provenance (`user`, `agent`, `connector`):
   user-provided sources never auto-fade (retention asks you first), while
   agent-acquired sources may be archived autonomously by the daily retention
   pass once truly idle (`retention.agent_source_*` config, ships dark behind
   `retention.enabled`).

## Input Sources

```bash
# Single file
hexis ingest --file ./notes.md

# Directory (recursive by default, --no-recursive to disable)
hexis ingest --input ./docs

# URL
hexis ingest --url https://example.com/article

# Stdin
cat ./notes.md | hexis ingest --stdin --stdin-type markdown --stdin-title "notes"
```

## Useful Flags

| Flag | Description |
|------|-------------|
| `--mode <mode>` | Choose ingestion mode |
| `--min-importance 0.6` | Floor importance for extracted memories |
| `--permanent` | No decay (memories persist indefinitely) |
| `--base-trust 0.7` | Override source trust level |
| `--no-recursive` | Don't recurse into subdirectories |
| `--quiet`, `-q` | Suppress verbose output |

## Tips

- Start with `--mode fast`. Use `slow` for known small/important content and
  `hybrid` for large sources where only some sections deserve conscious reading.
- Use `--permanent` sparingly -- it disables decay for created memories.
- Use `--base-trust` for low-trust sources (e.g., random web pages).
- The agent can also trigger ingestion autonomously during heartbeats via `fast_ingest`, `slow_ingest`, and `hybrid_ingest` tools.

## Troubleshooting

**"I ingested a big folder and recall is sparse"** -- recall returns distilled
memories, not whole files. Use `search_documents` to browse preserved raw
sources, `open_document`/`open_documents` for exact reading, or `load_documents`
to place substantial files on the RecMem desk for later exact search.

**Duplicate content skipped** -- Ingestion receipts are content-hash based. If
you need to re-ingest, change the content or clear the specific receipt. Note
that near-duplicate *facts* are not lost -- they corroborate the matched memory
(source merged, confidence revised, audited in `belief_revision_audit`).

## Related

- [Memory Operations](memory-operations.md) -- recall and search memories
- [Memory Types](../reference/memory-types.md) -- memory type reference
- [CLI reference](../reference/cli.md) -- full `hexis ingest` flags
