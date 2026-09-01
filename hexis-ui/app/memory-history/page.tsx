"use client";

import { ArrowRight, CalendarClock, GitCompareArrows, Search } from "lucide-react";
import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

import { Badge, MemoryTypeBadge } from "../components/ui/badge";
import { PageHeader } from "../components/ui/page-header";
import { Spinner } from "../components/ui/spinner";

type MemorySnapshotItem = {
  memory_id: string;
  content: string;
  type: string;
  score?: number | null;
  confidence?: number | null;
  trust_level?: number | null;
  valid_from?: string | null;
  valid_until?: string | null;
};

type Snapshot = {
  as_of: string;
  count: number;
  retrieval_mode: string;
  degraded?: boolean;
  degraded_reason?: string | null;
  memories: MemorySnapshotItem[];
};

type ChangeEvent = {
  event: string;
  at: string;
  reason?: string | null;
  note?: string | null;
  superseded_content?: string | null;
  replacement_content?: string | null;
  prior_confidence?: number | null;
  posterior_confidence?: number | null;
  outcome?: string | null;
};

type Diff = {
  from_time: string;
  to_time: string;
  from_snapshot: Snapshot;
  to_snapshot: Snapshot;
  added: MemorySnapshotItem[];
  expired: MemorySnapshotItem[];
  supersessions: ChangeEvent[];
  belief_revisions: ChangeEvent[];
  contradiction_decisions: ChangeEvent[];
  summary: Record<string, number>;
};

type ToolEnvelope = {
  success: boolean;
  output?: Snapshot | Diff;
  error?: string;
  display_output?: string;
};

function localInputValue(date: Date): string {
  const offset = date.getTimezoneOffset() * 60_000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 16);
}

function readableTime(value?: string | null): string {
  if (!value) return "Unknown time";
  return new Date(value).toLocaleString();
}

function percentage(value?: number | null): string {
  return value == null ? "—" : `${Math.round(Number(value) * 100)}%`;
}

function MemoryCard({ item, state }: { item: MemorySnapshotItem; state?: string }) {
  return (
    <article className="rounded-md border border-[var(--outline)] bg-white p-4">
      <div className="flex flex-wrap items-center gap-2">
        <MemoryTypeBadge type={item.type} />
        {state ? <Badge variant="accent">{state}</Badge> : null}
      </div>
      <p className="mt-3 whitespace-pre-wrap text-sm leading-6">{item.content}</p>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-xs text-[var(--ink-soft)]">
        <span>Historical confidence {percentage(item.confidence)}</span>
        <span>Historical trust {percentage(item.trust_level)}</span>
        {item.valid_from ? <span>Valid from {readableTime(item.valid_from)}</span> : null}
        {item.valid_until ? <span>Valid until {readableTime(item.valid_until)}</span> : null}
        <Link className="font-semibold text-[var(--teal)] underline" href={`/memories?memory=${item.memory_id}`}>
          Open memory
        </Link>
      </div>
    </article>
  );
}

function SnapshotView({ snapshot }: { snapshot: Snapshot }) {
  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="font-display text-xl">Known at {readableTime(snapshot.as_of)}</h2>
        <span className="text-xs text-[var(--ink-soft)]">
          {snapshot.count} result{snapshot.count === 1 ? "" : "s"} · {snapshot.retrieval_mode}
        </span>
      </div>
      {snapshot.degraded ? (
        <div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          {snapshot.degraded_reason || "Semantic retrieval was unavailable; lexical history was used."}
        </div>
      ) : null}
      {snapshot.memories.length ? snapshot.memories.map((item) => (
        <MemoryCard key={item.memory_id} item={item} />
      )) : (
        <div className="rounded-md border border-[var(--outline)] bg-white p-8 text-center text-sm text-[var(--ink-soft)]">
          The record contains no matching memory at this time. That is different from a memory stating the opposite.
        </div>
      )}
    </section>
  );
}

function EventCard({ event }: { event: ChangeEvent }) {
  const explanation = event.reason || event.note || "No explanation was recorded.";
  return (
    <li className="rounded-md border border-[var(--outline)] bg-white p-3 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-semibold capitalize">{event.event.replaceAll("_", " ")}</span>
        <span className="text-xs text-[var(--ink-soft)]">{readableTime(event.at)}</span>
      </div>
      {event.superseded_content ? <p className="mt-2 text-[var(--ink-soft)]">Before: {event.superseded_content}</p> : null}
      {event.replacement_content ? <p className="mt-1">After: {event.replacement_content}</p> : null}
      {event.prior_confidence != null || event.posterior_confidence != null ? (
        <p className="mt-1">Confidence: {percentage(event.prior_confidence)} <ArrowRight className="inline" size={13} /> {percentage(event.posterior_confidence)}</p>
      ) : null}
      <p className="mt-2 text-xs text-[var(--ink-soft)]">Why: {explanation}</p>
    </li>
  );
}

function DiffView({ diff }: { diff: Diff }) {
  const events = [
    ...(diff.supersessions || []),
    ...(diff.belief_revisions || []),
    ...(diff.contradiction_decisions || []),
  ].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  const degraded = diff.from_snapshot.degraded || diff.to_snapshot.degraded;
  return (
    <section className="space-y-5">
      <div>
        <h2 className="font-display text-xl">What changed</h2>
        <p className="mt-1 text-xs text-[var(--ink-soft)]">
          {readableTime(diff.from_time)} <ArrowRight className="inline" size={13} /> {readableTime(diff.to_time)}
        </p>
      </div>
      {degraded ? (
        <div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          {diff.from_snapshot.degraded_reason || diff.to_snapshot.degraded_reason}
        </div>
      ) : null}
      <div className="grid gap-3 sm:grid-cols-3">
        <Metric label="Added" value={diff.summary.added || 0} />
        <Metric label="Expired" value={diff.summary.expired || 0} />
        <Metric label="Recorded reasons" value={events.length} />
      </div>
      {diff.added.length ? <div className="space-y-3"><h3 className="text-sm font-semibold">Became valid</h3>{diff.added.map((item) => <MemoryCard key={item.memory_id} item={item} state="added" />)}</div> : null}
      {diff.expired.length ? <div className="space-y-3"><h3 className="text-sm font-semibold">Stopped being valid</h3>{diff.expired.map((item) => <MemoryCard key={item.memory_id} item={item} state="expired" />)}</div> : null}
      {events.length ? (
        <div><h3 className="text-sm font-semibold">Why it changed</h3><ol className="mt-3 space-y-3">{events.map((event, index) => <EventCard key={`${event.event}-${event.at}-${index}`} event={event} />)}</ol></div>
      ) : null}
      {!diff.added.length && !diff.expired.length && !events.length ? (
        <div className="rounded-md border border-[var(--outline)] bg-white p-8 text-center text-sm text-[var(--ink-soft)]">No recorded changes for this topic between these instants.</div>
      ) : null}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return <div className="rounded-md border border-[var(--outline)] bg-white p-4"><p className="text-xs uppercase text-[var(--ink-soft)]">{label}</p><p className="mt-1 font-display text-2xl">{value}</p></div>;
}

export default function MemoryHistoryPage() {
  const [mode, setMode] = useState<"snapshot" | "diff">("snapshot");
  const [query, setQuery] = useState("");
  const [asOf, setAsOf] = useState("");
  const [fromTime, setFromTime] = useState("");
  const [toTime, setToTime] = useState("");
  const [result, setResult] = useState<Snapshot | Diff | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const now = new Date();
    setAsOf(localInputValue(now));
    setToTime(localInputValue(now));
    setFromTime(localInputValue(new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000)));
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setResult(null);
    if (!query.trim()) {
      setError("Enter the topic whose history you want to inspect.");
      return;
    }
    try {
      const params = new URLSearchParams({ mode, q: query.trim() });
      if (mode === "snapshot") {
        if (!asOf) throw new Error("Choose the historical date and time.");
        params.set("as_of", new Date(asOf).toISOString());
      } else {
        if (!fromTime || !toTime) throw new Error("Choose both comparison times.");
        params.set("from_time", new Date(fromTime).toISOString());
        params.set("to_time", new Date(toTime).toISOString());
      }
      setLoading(true);
      const response = await fetch(`/api/memories/history?${params}`, { cache: "no-store" });
      const payload = await response.json() as ToolEnvelope;
      if (!response.ok || !payload.success || !payload.output) {
        throw new Error(payload.error || "Memory history could not be loaded.");
      }
      setResult(payload.output);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Memory history could not be loaded.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="mx-auto max-w-5xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <PageHeader title="Memory history" subtitle="See what Hexis knew at an exact time, or compare two points and inspect the recorded reason for every change." />

        <div className="mt-6 rounded-lg border border-[var(--outline)] bg-white p-4 sm:p-5">
          <div className="flex gap-2">
            <button disabled={loading} type="button" onClick={() => { setMode("snapshot"); setResult(null); setError(null); }} className={`flex items-center gap-2 rounded-md px-3 py-2 text-xs font-semibold disabled:opacity-50 ${mode === "snapshot" ? "bg-[var(--foreground)] text-white" : "border border-[var(--outline)]"}`}><CalendarClock size={15} /> As of a time</button>
            <button disabled={loading} type="button" onClick={() => { setMode("diff"); setResult(null); setError(null); }} className={`flex items-center gap-2 rounded-md px-3 py-2 text-xs font-semibold disabled:opacity-50 ${mode === "diff" ? "bg-[var(--foreground)] text-white" : "border border-[var(--outline)]"}`}><GitCompareArrows size={15} /> Compare two times</button>
          </div>
          <form onSubmit={submit} className="mt-4 grid gap-4">
            <label className="text-sm font-medium">Topic<input aria-label="Topic" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. Manning retainer" className="mt-1 block w-full rounded-md border border-[var(--outline)] px-3 py-2.5 text-sm outline-none focus:border-[var(--teal)]" /></label>
            {mode === "snapshot" ? (
              <label className="text-sm font-medium">As of<input aria-label="As of" type="datetime-local" value={asOf} onChange={(event) => setAsOf(event.target.value)} className="mt-1 block w-full rounded-md border border-[var(--outline)] px-3 py-2.5 text-sm" /></label>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2"><label className="text-sm font-medium">From<input aria-label="From" type="datetime-local" value={fromTime} onChange={(event) => setFromTime(event.target.value)} className="mt-1 block w-full rounded-md border border-[var(--outline)] px-3 py-2.5 text-sm" /></label><label className="text-sm font-medium">To<input aria-label="To" type="datetime-local" value={toTime} onChange={(event) => setToTime(event.target.value)} className="mt-1 block w-full rounded-md border border-[var(--outline)] px-3 py-2.5 text-sm" /></label></div>
            )}
            <button type="submit" disabled={loading} className="flex items-center justify-center gap-2 rounded-md bg-[var(--teal)] px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-50"><Search size={16} />{mode === "snapshot" ? "Recall snapshot" : "Compare history"}</button>
          </form>
        </div>

        {error ? <div className="mt-5 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div> : null}
        {loading ? <div className="py-16"><Spinner label="Reconstructing memory history…" /></div> : null}
        {!loading && result ? <div className="mt-6">{mode === "snapshot" ? <SnapshotView snapshot={result as Snapshot} /> : <DiffView diff={result as Diff} />}</div> : null}
      </div>
    </div>
  );
}
