"use client";

import { AlertTriangle, Check, GitCompareArrows, Scale } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Badge } from "../components/ui/badge";
import { PageHeader } from "../components/ui/page-header";
import { Spinner } from "../components/ui/spinner";

type MemorySide = {
  id: string;
  content: string;
  type: string;
  trust_level: number;
  source_attribution?: Record<string, unknown>;
  created_at: string;
  valid_from?: string | null;
  valid_until?: string | null;
  superseded_by?: string | null;
};

type ContradictionCase = {
  id: string;
  code: string;
  status: "pending" | "resolved" | "tension";
  outcome?: "new_right" | "old_right" | "tension";
  tension: string;
  confidence: number;
  new_memory_id?: string | null;
  memory_a: MemorySide;
  memory_b: MemorySide;
  resolution_note?: string | null;
  detected_at: string;
  resolved_at?: string | null;
};

const tabs = ["pending", "resolved", "tension", "all"] as const;
type Tab = (typeof tabs)[number];

function sourceLabel(memory: MemorySide): string {
  const source = memory.source_attribution || {};
  return String(source.label || source.path || source.ref || source.kind || "unattributed");
}

export default function ContradictionsPage() {
  const [tab, setTab] = useState<Tab>("pending");
  const [cases, setCases] = useState<ContradictionCase[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/contradictions?status=${tab}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Failed to load contradiction ledger");
      setCases(Array.isArray(payload.cases) ? payload.cases : []);
      setError(null);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Failed to load contradiction ledger");
    } finally {
      setLoading(false);
    }
  }, [tab]);

  useEffect(() => { void load(); }, [load]);

  const decide = async (
    item: ContradictionCase,
    outcome: "new_right" | "old_right" | "tension"
  ) => {
    setBusy(item.id);
    setNotice(null);
    try {
      const response = await fetch("/api/contradictions/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: item.id, outcome }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "The decision could not be recorded");
      setNotice(
        outcome === "tension"
          ? `Case ${item.code}: both memories remain valid in context.`
          : `Case ${item.code}: the losing memory remains in history with its validity window closed.`
      );
      await load();
    } catch (requestError: unknown) {
      setNotice(requestError instanceof Error ? requestError.message : "The decision could not be recorded");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <PageHeader
        title="Contradiction ledger"
        subtitle="Conflicts Hexis detected, the decision you made, and the bitemporal history preserved afterward."
      />

      <div className="flex flex-wrap gap-2">
        {tabs.map((value) => (
          <button
            key={value}
            type="button"
            onClick={() => setTab(value)}
            className={`rounded-md px-3 py-2 text-xs font-semibold capitalize ${
              tab === value ? "bg-[var(--foreground)] text-white" : "border border-[var(--outline)] bg-white"
            }`}
          >
            {value}
          </button>
        ))}
      </div>

      {notice ? <div className="rounded-md border border-[var(--teal)]/40 bg-[var(--teal)]/5 px-3 py-2 text-sm">{notice}</div> : null}
      {error ? <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div> : null}
      {loading ? <div className="py-16"><Spinner label="Loading contradiction ledger…" /></div> : null}
      {!loading && cases.length === 0 ? (
        <div className="rounded-lg border border-[var(--outline)] bg-white p-8 text-center text-sm text-[var(--ink-soft)]">
          No {tab === "all" ? "recorded" : tab} contradictions.
        </div>
      ) : null}

      <div className="space-y-4">
        {cases.map((item) => {
          const newer = item.new_memory_id === item.memory_a.id
            ? item.memory_a
            : item.new_memory_id === item.memory_b.id
              ? item.memory_b
              : Date.parse(item.memory_a.created_at) >= Date.parse(item.memory_b.created_at)
                ? item.memory_a
                : item.memory_b;
          const older = newer.id === item.memory_a.id ? item.memory_b : item.memory_a;
          return (
            <article key={item.id} className="rounded-lg border border-[var(--outline)] bg-white p-4 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  {item.status === "pending" ? <AlertTriangle size={17} className="text-amber-600" /> : item.status === "tension" ? <Scale size={17} className="text-[var(--teal)]" /> : <Check size={17} className="text-emerald-600" />}
                  <Badge variant="accent">{item.status}</Badge>
                  <span className="font-mono text-xs text-[var(--ink-soft)]">{item.code}</span>
                </div>
                <span className="text-xs text-[var(--ink-soft)]">
                  detector confidence {Math.round(item.confidence * 100)}%
                </span>
              </div>
              <p className="mt-3 text-sm font-semibold leading-6">{item.tension}</p>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {([["Newer", newer], ["Older", older]] as const).map(([label, memory]) => (
                  <div key={memory.id} className="rounded-md border border-[var(--outline)] bg-[var(--surface)] p-3">
                    <div className="flex items-center justify-between gap-2 text-xs">
                      <span className="font-semibold uppercase text-[var(--ink-soft)]">{label}</span>
                      <span>trust {Math.round(Number(memory.trust_level || 0) * 100)}%</span>
                    </div>
                    <p className="mt-2 text-sm leading-6">{memory.content}</p>
                    <div className="mt-2 flex flex-wrap gap-x-3 text-xs text-[var(--ink-soft)]">
                      <span>{sourceLabel(memory)}</span>
                      <Link className="font-semibold text-[var(--teal)] underline" href={`/memories?memory=${memory.id}`}>Open memory</Link>
                    </div>
                    {memory.valid_until ? <p className="mt-2 text-xs text-[var(--ink-soft)]">Valid until {String(memory.valid_until).slice(0, 16).replace("T", " ")}</p> : null}
                  </div>
                ))}
              </div>
              {item.status === "pending" ? (
                <div className="mt-4 grid gap-2 sm:grid-cols-3">
                  <button disabled={busy === item.id} onClick={() => void decide(item, "new_right")} className="rounded-md bg-[var(--teal)] px-3 py-2 text-xs font-semibold text-white disabled:opacity-40">Newer is right</button>
                  <button disabled={busy === item.id} onClick={() => void decide(item, "old_right")} className="rounded-md border border-[var(--outline)] px-3 py-2 text-xs font-semibold disabled:opacity-40">Older is right</button>
                  <button disabled={busy === item.id} onClick={() => void decide(item, "tension")} className="rounded-md border border-[var(--outline)] px-3 py-2 text-xs font-semibold disabled:opacity-40">Both, by context</button>
                </div>
              ) : (
                <div className="mt-4 flex items-start gap-2 rounded-md bg-[var(--surface-strong)] px-3 py-2 text-sm">
                  <GitCompareArrows size={16} className="mt-0.5 flex-none" />
                  <span>{item.resolution_note || (item.outcome === "tension" ? "Both retained as contextual tension." : "Resolved; history preserved.")}</span>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </div>
  );
}
