---
name: knowledge-ingest
description: Ingest URLs, documents, and text into the memory system as structured knowledge
category: knowledge
requires:
  tools: [url_ingest, remember]
contexts: [heartbeat, chat]
aliases: [ingest, summarize, summary, document, file, pdf, contract, read, attach]
bound_tools: [url_ingest, fast_ingest, slow_ingest, hybrid_ingest, remember, recall, search_documents, open_document, open_documents, load_documents, search_document_chunks, load_document_chunks, list_desk, git_ingest]
---

# Knowledge Base Ingestion

Transform external content -- web pages, documents, raw text -- into structured semantic memories that persist in the knowledge graph.

## When to Use

- When the user shares a URL and says "learn this" or "remember this article"
- When a research workflow finds valuable sources that should be retained long-term
- When the user pastes raw text (notes, transcripts, outlines) to be ingested
- During heartbeats when a goal involves building knowledge on a specific topic
- When importing reference material for a project or domain

## Step-by-Step Methodology

1. **Assess the source**: Before ingesting, determine what kind of content it is (article, documentation, transcript, raw notes). This guides how aggressively to summarize.
2. **Fetch and parse**: For URLs, use `url_ingest` which handles fetching, HTML-to-text conversion, and chunking. For files or raw knowledge sources, use the fast/slow/hybrid ingestion tools as appropriate.
3. **Check for duplicates**: Use `recall` with the URL or a key phrase from the content to see if it has already been ingested. Avoid storing the same source twice.
4. **Chunk intelligently**: Long content is automatically chunked by the ingestion pipeline. Each chunk becomes a separate semantic memory linked by source metadata, and the raw source artifact remains searchable with `search_documents` and retrievable with `open_document` or `open_documents`. Use `load_documents` when a large source should be deliberately placed on the RecMem desk for later exact search. Trust the pipeline's chunking; do not manually split content unless it is clearly failing.
5. **Add context**: When storing via `remember`, include metadata about the source: URL, author, date published, and why it was ingested (which goal or topic it serves).
6. **Verify ingestion**: After ingestion completes, run a quick `recall` on a key concept from the content to confirm distilled knowledge is retrievable. For exact wording or large specifications, use `search_documents`, `open_document`, or `load_documents` instead of expecting recall to carry the whole source. For passage-level verification, `search_document_chunks` should find key phrases with page/section locators; `list_desk` shows what is already loaded before you load more.
7. **Connect to goals**: If the ingested content relates to an active goal, note the connection so future heartbeats can leverage it.

## Quality Guidelines

- Prefer ingesting authoritative, primary sources over summaries or aggregators.
- Do not ingest entire websites. Be selective -- ingest the specific pages that contain the needed information.
- When ingesting long documents, let the chunking pipeline do its job. Each chunk retains a reference to the parent source.
- Always record the source URL or origin. Memories without provenance are harder to evaluate and update later.
- Respect rate limits and robots.txt when fetching URLs. If a fetch fails, note the failure and move on rather than retrying aggressively.
- For sensitive or private content (internal docs, personal notes), ensure the user understands that ingested content persists in the local database.
