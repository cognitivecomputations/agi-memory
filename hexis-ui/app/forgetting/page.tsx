"use client";

import { ArchiveRestore, Gauge, NotebookPen, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Badge } from "../components/ui/badge";
import { PageHeader } from "../components/ui/page-header";
import { Spinner } from "../components/ui/spinner";

type Pressure = {
  episodic_mass?: number;
  capacity?: number;
  capacity_ratio?: number | null;
  candidate_groups?: number;
  archived_recoverable?: number;
  summarization_pending?: number;
};

type Compression = {
  report_id: string;
  gist_memory_id: string;
  source_count: number;
  fidelity: number;
  summary_preview: string;
  compressed_at: string;
};

type LowFidelity = {
  memory_id: string;
  content: string;
  type: string;
  fidelity: number;
};

type FadeMemory = {
  id: string;
  content: string;
  importance: number;
  strength: number;
  fidelity: number;
  load_bearing: boolean;
};

type FadeReview = {
  id: string;
  code: string;
  status: "pending" | "kept" | "released";
  decision?: "keep" | "release" | "journal";
  reason?: string;
  preview?: string;
  budget_remaining: number;
  memories: FadeMemory[];
};

type Observe = {
  enabled?: boolean;
  irreversible_pruning_enabled?: boolean;
  pressure?: Pressure;
  low_fidelity_count?: number;
  low_fidelity?: LowFidelity[];
  recent_compressions?: Compression[];
};

const tabs = ["pending", "kept", "released", "all"] as const;
type Tab = (typeof tabs)[number];

function pct(value: number | null | undefined) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

export default function ForgettingPage() {
  const [tab, setTab] = useState<Tab>("pending");
  const [observe, setObserve] = useState<Observe>({});
  const [reviews, setReviews] = useState<FadeReview[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [journaling, setJournaling] = useState<string | null>(null);
  const [journal, setJournal] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/retention?status=${tab}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Failed to load forgetting state");
      setObserve(payload.observe || {});
      setReviews(Array.isArray(payload.reviews) ? payload.reviews : []);
      setError(null);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Failed to load forgetting state");
    } finally {
      setLoading(false);
    }
  }, [tab]);

  useEffect(() => { void load(); }, [load]);

  const decide = async (review: FadeReview, decision: "keep" | "release" | "journal") => {
    setBusy(review.id);
    setNotice(null);
    setError(null);
    try {
      const response = await fetch("/api/retention/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: review.id,
          decision,
          journal_content: decision === "journal" ? journal : undefined,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.message || payload.error || "The fade decision could not be recorded");
      const compression = payload.compression;
      const sourceCount = Number(compression?.source_count || 0);
      setNotice(compression
        ? `${sourceCount} source ${sourceCount === 1 ? "memory" : "memories"} entered one recoverable gist; summarization is queued and the resulting fidelity will be reported.`
        : payload.next_step || "Retention decision recorded.");
      setJournaling(null);
      setJournal("");
      await load();
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "The fade decision could not be recorded");
    } finally {
      setBusy(null);
    }
  };

  const pressure = observe.pressure || {};
  const capacity = Number(pressure.capacity || 0);
  const ratio = pressure.capacity_ratio;

  return (
    <div className="space-y-6 p-6">
      <PageHeader
        title="Forgetting"
        subtitle="Memory pressure, honest fidelity, explicit load-bearing choices, and a factual record of what compressed."
      />

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-lg border border-[var(--outline)] bg-white p-4">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase text-[var(--ink-soft)]"><Gauge size={16} /> Episodic pressure</div>
          <p className="mt-2 text-2xl font-semibold">{Number(pressure.episodic_mass || 0).toFixed(2)}</p>
          <p className="mt-1 text-xs text-[var(--ink-soft)]">{capacity > 0 ? `${pct(ratio)} of ${capacity}` : "No capacity ceiling configured"}</p>
        </div>
        <div className="rounded-lg border border-[var(--outline)] bg-white p-4">
          <p className="text-xs font-semibold uppercase text-[var(--ink-soft)]">Low-fidelity memories</p>
          <p className="mt-2 text-2xl font-semibold">{observe.low_fidelity_count || 0}</p>
          <p className="mt-1 text-xs text-[var(--ink-soft)]">Below 75% fidelity and shown as reconstruction</p>
        </div>
        <div className="rounded-lg border border-[var(--outline)] bg-white p-4">
          <p className="text-xs font-semibold uppercase text-[var(--ink-soft)]">Near compression</p>
          <p className="mt-2 text-2xl font-semibold">{pressure.candidate_groups || 0}</p>
          <p className="mt-1 text-xs text-[var(--ink-soft)]">Candidate groups; load-bearing cases wait below</p>
        </div>
        <div className="rounded-lg border border-[var(--outline)] bg-white p-4">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase text-[var(--ink-soft)]"><ArchiveRestore size={16} /> Archived originals</div>
          <p className="mt-2 text-2xl font-semibold">{pressure.archived_recoverable || 0}</p>
          <p className="mt-1 text-xs text-[var(--ink-soft)]">{observe.irreversible_pruning_enabled ? "Hard pruning explicitly enabled" : "Recoverable; hard pruning is off"}</p>
        </div>
      </div>

      {notice ? <div className="rounded-md border border-[var(--teal)]/40 bg-[var(--teal)]/5 px-3 py-2 text-sm">{notice}</div> : null}
      {error ? <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div> : null}

      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Load-bearing decisions</h2>
            <p className="text-sm text-[var(--ink-soft)]">No timer chooses these. Keep spends a finite chapter budget; journal preserves words before compression.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {tabs.map((value) => (
              <button key={value} onClick={() => setTab(value)} className={`rounded-md px-3 py-2 text-xs font-semibold capitalize ${tab === value ? "bg-[var(--foreground)] text-white" : "border border-[var(--outline)] bg-white"}`}>{value}</button>
            ))}
          </div>
        </div>
        {loading ? <div className="py-12"><Spinner label="Loading forgetting state…" /></div> : null}
        {!loading && reviews.length === 0 ? <div className="rounded-lg border border-[var(--outline)] bg-white p-6 text-sm text-[var(--ink-soft)]">No {tab === "all" ? "recorded" : tab} fade decisions.</div> : null}
        {reviews.map((review) => (
          <article key={review.id} className="rounded-lg border border-[var(--outline)] bg-white p-4 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2"><ShieldCheck size={17} /><Badge variant={review.status === "pending" ? "warning" : "default"}>{review.status}</Badge><span className="font-mono text-xs text-[var(--ink-soft)]">{review.code}</span></div>
              <span className="text-xs text-[var(--ink-soft)]">keep budget {review.budget_remaining}</span>
            </div>
            <p className="mt-3 text-sm font-semibold">{review.preview || "Memory group awaiting a decision"}</p>
            <p className="mt-1 text-xs text-[var(--ink-soft)]">Reason: {(review.reason || "near threshold").replaceAll("_", " ")}</p>
            <div className="mt-3 space-y-2">
              {(review.memories || []).map((memory) => (
                <div key={memory.id} className="rounded-md bg-[var(--surface)] px-3 py-2 text-sm">
                  <p>{memory.content}</p>
                  <div className="mt-1 flex flex-wrap gap-3 text-xs text-[var(--ink-soft)]"><span>strength {pct(memory.strength)}</span><span>fidelity {pct(memory.fidelity)}</span>{memory.load_bearing ? <span className="font-semibold text-amber-700">load-bearing</span> : null}<Link className="font-semibold text-[var(--teal)] underline" href={`/memories?memory=${memory.id}`}>Open memory</Link></div>
                </div>
              ))}
            </div>
            {review.status === "pending" ? (
              <div className="mt-3">
                {journaling === review.id ? (
                  <div className="space-y-2 rounded-md border border-[var(--outline)] p-3">
                    <label htmlFor={`journal-${review.id}`} className="flex items-center gap-2 text-xs font-semibold"><NotebookPen size={15} /> What should remain in writing?</label>
                    <textarea id={`journal-${review.id}`} value={journal} onChange={(event) => setJournal(event.target.value)} className="min-h-20 w-full rounded-md border border-[var(--outline)] px-3 py-2 text-sm" />
                    <div className="flex gap-2"><button disabled={!journal.trim() || busy === review.id} onClick={() => void decide(review, "journal")} className="rounded-md bg-[var(--teal)] px-3 py-2 text-xs font-semibold text-white disabled:opacity-40">Journal, then compress</button><button onClick={() => { setJournaling(null); setJournal(""); }} className="rounded-md border border-[var(--outline)] px-3 py-2 text-xs font-semibold">Cancel</button></div>
                  </div>
                ) : (
                  <div className="grid gap-2 sm:grid-cols-3"><button disabled={busy === review.id} onClick={() => void decide(review, "keep")} className="rounded-md bg-[var(--teal)] px-3 py-2 text-xs font-semibold text-white disabled:opacity-40">Keep</button><button disabled={busy === review.id} onClick={() => setJournaling(review.id)} className="rounded-md border border-[var(--outline)] px-3 py-2 text-xs font-semibold disabled:opacity-40">Journal first</button><button disabled={busy === review.id} onClick={() => void decide(review, "release")} className="rounded-md border border-amber-300 px-3 py-2 text-xs font-semibold text-amber-800 disabled:opacity-40">Let compress</button></div>
                )}
              </div>
            ) : null}
          </article>
        ))}
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">What compressed</h2>
        {(observe.recent_compressions || []).length === 0 ? <div className="rounded-lg border border-[var(--outline)] bg-white p-6 text-sm text-[var(--ink-soft)]">No completed compression reports yet.</div> : null}
        {(observe.recent_compressions || []).map((item) => (
          <article key={item.report_id} className="rounded-lg border border-[var(--outline)] bg-white p-4"><div className="flex flex-wrap items-center justify-between gap-2"><span className="text-sm font-semibold">{item.source_count} source memories → one gist</span><Badge variant="teal">{pct(item.fidelity)} fidelity</Badge></div><p className="mt-2 text-sm leading-6">{item.summary_preview}</p><Link className="mt-2 inline-block text-xs font-semibold text-[var(--teal)] underline" href={`/memories?memory=${item.gist_memory_id}`}>Open compressed memory</Link></article>
        ))}
      </section>

      {(observe.low_fidelity || []).length ? (
        <section className="space-y-3"><h2 className="text-lg font-semibold">Lowest fidelity</h2>{(observe.low_fidelity || []).map((item) => <article key={item.memory_id} className="rounded-lg border border-[var(--outline)] bg-white p-4"><div className="flex items-center justify-between gap-2"><Badge variant="warning">{pct(item.fidelity)} fidelity</Badge><span className="text-xs capitalize text-[var(--ink-soft)]">{item.type}</span></div><p className="mt-2 text-sm">{item.content}</p><Link className="mt-2 inline-block text-xs font-semibold text-[var(--teal)] underline" href={`/memories?memory=${item.memory_id}`}>Open memory</Link></article>)}</section>
      ) : null}
    </div>
  );
}
