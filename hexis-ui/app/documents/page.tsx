"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  Brain,
  FileText,
  Layers,
  Lock,
  Search,
  X,
} from "lucide-react";
import Link from "next/link";
import { Card } from "../components/ui/card";
import { PageHeader } from "../components/ui/page-header";
import { Spinner } from "../components/ui/spinner";

type DocRow = {
  document_id: string;
  title: string | null;
  path: string | null;
  source_type: string | null;
  file_type: string | null;
  word_count: number | null;
  size_bytes: number | null;
  updated_at: string | null;
  rank: number | null;
  snippet: string | null;
  sensitivity?: string | null;
  acquisition?: string | null;
  extraction_warnings?: { code?: string; message?: string }[];
  best_chunk_id?: string | null;
};

type ChunkRow = {
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  title: string | null;
  path: string | null;
  page_start: number | null;
  page_end: number | null;
  sheet_name: string | null;
  heading_path: string[] | null;
  snippet: string | null;
  rank: number | null;
};

type DocDetail = {
  document: {
    document_id: string;
    title?: string;
    path?: string;
    source_type?: string;
    content?: string;
    total_chars?: number;
    truncated?: boolean;
    next_offset?: number | null;
    extraction_warnings?: { code?: string; message?: string }[];
    source_attribution?: Record<string, unknown>;
  };
  chunks: {
    chunk_id: string;
    chunk_index: number;
    locator_kind: string;
    page_start: number | null;
    sheet_name: string | null;
    heading_path: string[] | null;
  }[];
  memories: { id: string; type: string; content: string }[];
};

export default function DocumentsPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"documents" | "chunks">("documents");
  const [typeFilter, setTypeFilter] = useState("");
  const [docs, setDocs] = useState<DocRow[]>([]);
  const [chunks, setChunks] = useState<ChunkRow[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<DocDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (query.trim()) params.set("q", query.trim());
      if (typeFilter) params.set("type", typeFilter);
      params.set("mode", mode);
      params.set("limit", "25");
      const res = await fetch(`/api/documents?${params}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`Failed to load documents (${res.status})`);
      const data = await res.json();
      setDocs(data.documents || []);
      setChunks(data.chunks || []);
      setTotal(data.total || (data.chunks || []).length);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents.");
    } finally {
      setLoading(false);
    }
  }, [query, typeFilter, mode]);

  useEffect(() => {
    const timer = window.setTimeout(fetchDocs, query ? 250 : 0);
    return () => window.clearTimeout(timer);
  }, [fetchDocs, query]);

  const openDetail = useCallback(async (documentId: string, pageStart?: number) => {
    setSelected(documentId);
    setDetailLoading(true);
    try {
      const params = new URLSearchParams();
      if (pageStart) {
        params.set("page_start", String(pageStart));
      }
      const res = await fetch(`/api/documents/${documentId}?${params}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`Failed to open document (${res.status})`);
      const data = await res.json();
      if (data.chunks && data.document) {
        setDetail(data);
      } else if (data.chunks) {
        // Page-range open returns raw chunk payload; merge into prior detail.
        setDetail((prev) =>
          prev
            ? {
                ...prev,
                document: {
                  ...prev.document,
                  content: (data.chunks as { content: string }[])
                    .map((c) => c.content)
                    .join("\n\n"),
                  truncated: false,
                  next_offset: null,
                },
              }
            : prev
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to open document.");
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const documentId = params.get("document");
    if (!documentId) return;
    void openDetail(documentId);
  }, [openDetail]);

  const loadMore = useCallback(async () => {
    if (!detail?.document?.next_offset || !selected) return;
    const res = await fetch(
      `/api/documents/${selected}?offset=${detail.document.next_offset}`,
      { cache: "no-store" }
    );
    if (!res.ok) return;
    const data = await res.json();
    setDetail((prev) =>
      prev
        ? {
            ...prev,
            document: {
              ...data.document,
              content: `${prev.document.content || ""}${data.document.content || ""}`,
            },
            chunks: prev.chunks,
            memories: prev.memories,
          }
        : data
    );
  }, [detail, selected]);

  const loadToDesk = useCallback(async (documentId: string) => {
    setNotice(null);
    const res = await fetch("/api/desk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "load", document_id: documentId, reason: "loaded from Documents page" }),
    });
    const data = await res.json();
    if (res.ok && !data.error) {
      setNotice(`Loaded ${data.count ?? 0} item(s) onto the desk — see the Desk page.`);
    } else {
      setNotice(`Load failed: ${data.error || res.status}`);
    }
  }, []);

  const pages = (detail?.chunks || [])
    .filter((c) => c.page_start != null)
    .map((c) => c.page_start as number);
  const uniquePages = Array.from(new Set(pages)).sort((a, b) => a - b);

  return (
    <div className="space-y-6 p-6">
      <PageHeader
        title="Documents"
        subtitle="The source-document filing cabinet: every ingested file, page, and message — preserved exactly, searchable, and loadable onto the desk."
      />

      {notice ? (
        <div className="rounded-md border border-[var(--teal)]/40 bg-[var(--teal)]/5 px-3 py-2 text-sm">
          {notice}{" "}
          <Link href="/desk" className="font-semibold text-[var(--teal)] underline">
            Open the desk →
          </Link>
        </div>
      ) : null}
      {error ? (
        <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
      ) : null}

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-64 flex-1">
          <Search size={15} className="absolute left-2.5 top-2.5 text-[var(--ink-soft)]" />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={mode === "chunks" ? "Search passages…" : "Search documents…"}
            className="w-full rounded-md border border-[var(--outline)] py-2 pl-8 pr-3 text-sm outline-none focus:border-[var(--teal)]"
          />
        </div>
        <div className="flex rounded-md border border-[var(--outline)] text-xs font-medium">
          {(["documents", "chunks"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`px-3 py-2 capitalize ${
                mode === m ? "bg-[var(--surface-strong)] text-[var(--foreground)]" : "text-[var(--ink-soft)]"
              }`}
            >
              {m === "chunks" ? "passages" : m}
            </button>
          ))}
        </div>
        <select
          value={typeFilter}
          onChange={(event) => setTypeFilter(event.target.value)}
          aria-label="Filter by source type"
          className="rounded-md border border-[var(--outline)] px-2 py-2 text-sm"
        >
          <option value="">All types</option>
          <option value="document">Documents</option>
          <option value="web">Web</option>
          <option value="email">Email</option>
          <option value="spreadsheet">Spreadsheets</option>
          <option value="code">Code</option>
          <option value="pasted_text">Pasted text</option>
        </select>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr,1.2fr]">
        <Card className="max-h-[70vh] overflow-y-auto p-2">
          {loading ? (
            <div className="flex items-center gap-2 p-4 text-sm text-[var(--ink-soft)]">
              <Spinner /> Loading…
            </div>
          ) : mode === "chunks" ? (
            chunks.length === 0 ? (
              <p className="p-4 text-sm text-[var(--ink-soft)]">
                No passages matched. Try different wording, or the documents tab.
              </p>
            ) : (
              chunks.map((c) => (
                <button
                  key={c.chunk_id}
                  type="button"
                  onClick={() => openDetail(c.document_id, c.page_start ?? undefined)}
                  className="block w-full rounded-md px-3 py-2 text-left hover:bg-[var(--surface-strong)]"
                >
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <Layers size={14} className="flex-none text-[var(--teal)]" />
                    <span className="truncate">{c.title || c.path || "Untitled"}</span>
                    <span className="flex-none text-xs text-[var(--ink-soft)]">
                      {c.page_start ? `p.${c.page_start}` : c.sheet_name ? c.sheet_name : `#${c.chunk_index}`}
                    </span>
                  </div>
                  <p className="mt-1 line-clamp-2 text-xs text-[var(--ink-soft)]">{c.snippet}</p>
                </button>
              ))
            )
          ) : docs.length === 0 ? (
            <p className="p-4 text-sm text-[var(--ink-soft)]">
              The cabinet is empty for this filter. Ingest something on the{" "}
              <Link href="/ingest" className="font-semibold text-[var(--teal)] underline">
                Ingest page
              </Link>
              .
            </p>
          ) : (
            docs.map((d) => (
              <button
                key={d.document_id}
                type="button"
                onClick={() => openDetail(d.document_id)}
                className={`block w-full rounded-md px-3 py-2 text-left hover:bg-[var(--surface-strong)] ${
                  selected === d.document_id ? "bg-[var(--surface-strong)]" : ""
                }`}
              >
                <div className="flex items-center gap-2 text-sm font-medium">
                  <FileText size={14} className="flex-none text-[var(--teal)]" />
                  <span className="truncate">{d.title || d.path || "Untitled"}</span>
                  {d.sensitivity === "private" ? (
                    <Lock size={12} className="flex-none text-[var(--teal)]" aria-label="Private" />
                  ) : null}
                  {(d.extraction_warnings || []).length > 0 ? (
                    <AlertTriangle size={12} className="flex-none text-amber-500" aria-label="Extraction warnings" />
                  ) : null}
                </div>
                <p className="mt-0.5 truncate text-xs text-[var(--ink-soft)]">
                  {d.source_type} · {d.word_count?.toLocaleString()} words
                  {d.acquisition === "agent" ? " · agent-acquired" : ""}
                </p>
                <p className="mt-1 line-clamp-2 text-xs text-[var(--ink-soft)]">{d.snippet}</p>
              </button>
            ))
          )}
          {!loading && mode === "documents" ? (
            <p className="px-3 py-2 text-xs text-[var(--ink-soft)]">{total} document(s) total</p>
          ) : null}
        </Card>

        <Card className="max-h-[70vh] overflow-y-auto p-4">
          {!selected ? (
            <p className="text-sm text-[var(--ink-soft)]">Select a document to preview it.</p>
          ) : detailLoading && !detail ? (
            <div className="flex items-center gap-2 text-sm text-[var(--ink-soft)]">
              <Spinner /> Opening…
            </div>
          ) : detail ? (
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h2 className="truncate text-base font-semibold">
                    {detail.document.title || detail.document.path}
                  </h2>
                  <p className="truncate text-xs text-[var(--ink-soft)]">{detail.document.path}</p>
                </div>
                <div className="flex flex-none items-center gap-2">
                  <button
                    type="button"
                    onClick={() => loadToDesk(detail.document.document_id)}
                    className="rounded-md bg-[var(--foreground)] px-3 py-1.5 text-xs font-semibold text-white hover:bg-[var(--teal)]"
                  >
                    Load to desk
                  </button>
                  <button
                    type="button"
                    aria-label="Close preview"
                    onClick={() => {
                      setSelected(null);
                      setDetail(null);
                    }}
                    className="rounded p-1 text-[var(--ink-soft)] hover:bg-[var(--surface-strong)]"
                  >
                    <X size={15} />
                  </button>
                </div>
              </div>

              {(detail.document.extraction_warnings || []).length > 0 ? (
                <div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                  {(detail.document.extraction_warnings || []).map((w, i) => (
                    <p key={i}>
                      <span className="font-semibold">[{w.code}]</span> {w.message}
                    </p>
                  ))}
                </div>
              ) : null}

              {uniquePages.length > 1 ? (
                <div className="flex flex-wrap items-center gap-1 text-xs">
                  <span className="text-[var(--ink-soft)]">Pages:</span>
                  {uniquePages.slice(0, 40).map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => openDetail(detail.document.document_id, p)}
                      className="rounded border border-[var(--outline)] px-1.5 py-0.5 hover:border-[var(--teal)]"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              ) : null}

              <pre className="whitespace-pre-wrap break-words rounded-md bg-[var(--surface-strong)] p-3 text-xs leading-relaxed">
                {detail.document.content}
              </pre>
              {detail.document.truncated ? (
                <button
                  type="button"
                  onClick={loadMore}
                  className="rounded-md border border-[var(--outline)] px-3 py-1.5 text-xs font-medium hover:border-[var(--teal)]"
                >
                  Load more ({detail.document.total_chars?.toLocaleString()} chars total)
                </button>
              ) : null}

              {detail.memories.length > 0 ? (
                <div>
                  <h3 className="mb-1 flex items-center gap-1.5 text-sm font-semibold">
                    <Brain size={14} className="text-[var(--teal)]" /> Memories from this source
                  </h3>
                  <ul className="space-y-1">
                    {detail.memories.map((m) => (
                      <li key={m.id} className="text-xs text-[var(--ink-soft)]">
                        <Link href={`/memories?open=${m.id}`} className="hover:text-[var(--teal)]">
                          [{m.type}] {m.content}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : null}
        </Card>
      </div>
    </div>
  );
}
