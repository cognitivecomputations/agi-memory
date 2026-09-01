"use client";

import { BookCheck, Brain, Lightbulb, RefreshCcw, Wrench } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Badge } from "../components/ui/badge";
import { PageHeader } from "../components/ui/page-header";
import { Spinner } from "../components/ui/spinner";

type LearningItem = {
  id: string;
  code: string;
  kind: "semantic_belief" | "new_procedure" | "revised_strategy" | "proposed_skill";
  status: "pending" | "approved" | "corrected" | "forgotten";
  title: string;
  content: string;
  source_memory_id?: string;
  source_skill_proposal_id?: string;
  skill_proposal_status?: string;
  skill_last_error?: string;
  correction?: string;
  evidence?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
};

type LearningReview = {
  id: string;
  status: "pending" | "completed";
  summary: string;
  period_start: string;
  period_end: string;
  items: LearningItem[];
};

const tabs = ["pending", "completed", "all"] as const;
type Tab = (typeof tabs)[number];

const kindLabels: Record<LearningItem["kind"], string> = {
  semantic_belief: "Belief",
  new_procedure: "Procedure",
  revised_strategy: "Strategy",
  proposed_skill: "Proposed skill",
};

function KindIcon({ kind }: { kind: LearningItem["kind"] }) {
  if (kind === "new_procedure") return <BookCheck size={17} />;
  if (kind === "revised_strategy") return <RefreshCcw size={17} />;
  if (kind === "proposed_skill") return <Wrench size={17} />;
  return <Brain size={17} />;
}

function percent(value: unknown): string | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : null;
}

export default function LearningReviewPage() {
  const [tab, setTab] = useState<Tab>("pending");
  const [reviews, setReviews] = useState<LearningReview[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [correcting, setCorrecting] = useState<string | null>(null);
  const [correction, setCorrection] = useState("");
  const [confirmForget, setConfirmForget] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/learning-review?status=${tab}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Failed to load learning review");
      setReviews(Array.isArray(payload.reviews) ? payload.reviews : []);
      setError(null);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Failed to load learning review");
    } finally {
      setLoading(false);
    }
  }, [tab]);

  useEffect(() => { void load(); }, [load]);

  const decide = async (
    item: LearningItem,
    action: "approve" | "correct" | "forget",
    confirmLoadBearing = false
  ) => {
    setBusy(item.id);
    setNotice(null);
    setError(null);
    try {
      const response = await fetch("/api/learning-review/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: item.id,
          action,
          correction: action === "correct" ? correction : undefined,
          confirm_load_bearing: confirmLoadBearing,
        }),
      });
      const payload = await response.json();
      if (payload.confirmation_required) {
        setConfirmForget(item.id);
        setNotice(payload.message || "This memory is load-bearing; confirm once more to forget it.");
        return;
      }
      if (!response.ok) throw new Error(payload.error || "The learning decision could not be recorded");
      setCorrecting(null);
      setCorrection("");
      setConfirmForget(null);
      setNotice(payload.next_step || `${kindLabels[item.kind]} ${payload.status}.`);
      await load();
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "The learning decision could not be recorded");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <PageHeader
        title="Learning review"
        subtitle="A weekly diff of grounded beliefs, procedures, strategies, and proposed skills. Nothing here changes silently."
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
      {loading ? <div className="py-16"><Spinner label="Loading learning review…" /></div> : null}
      {!loading && reviews.length === 0 ? (
        <div className="rounded-lg border border-[var(--outline)] bg-white p-8 text-center text-sm text-[var(--ink-soft)]">
          <Lightbulb className="mx-auto mb-3" size={22} />
          No {tab === "all" ? "recorded" : tab} learning reviews. Hexis only asks when the opted-in weekly pass finds enough grounded change.
        </div>
      ) : null}

      <div className="space-y-6">
        {reviews.map((review) => (
          <section key={review.id} className="rounded-lg border border-[var(--outline)] bg-white p-4 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <Badge variant={review.status === "pending" ? "warning" : "success"}>{review.status}</Badge>
                  <span className="text-xs text-[var(--ink-soft)]">
                    {String(review.period_start).slice(0, 10)} → {String(review.period_end).slice(0, 10)}
                  </span>
                </div>
                <p className="mt-3 max-w-3xl text-sm leading-6">{review.summary}</p>
              </div>
              <span className="text-xs text-[var(--ink-soft)]">{review.items.length} change{review.items.length === 1 ? "" : "s"}</span>
            </div>

            <div className="mt-4 space-y-3">
              {review.items.map((item) => {
                const evidence = item.evidence || {};
                const source = (evidence.source_attribution || {}) as Record<string, unknown>;
                const confidence = percent(evidence.confidence);
                const trust = percent(evidence.trust_level);
                return (
                  <article key={item.id} className="rounded-md border border-[var(--outline)] bg-[var(--surface)] p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <KindIcon kind={item.kind} />
                        <span className="text-xs font-semibold uppercase text-[var(--ink-soft)]">{kindLabels[item.kind]}</span>
                        <span className="font-mono text-xs text-[var(--ink-soft)]">{item.code}</span>
                      </div>
                      <Badge variant={item.status === "pending" ? "warning" : "default"}>{item.status}</Badge>
                    </div>
                    <p className="mt-2 text-sm font-semibold">{item.title}</p>
                    <p className="mt-1 text-sm leading-6">{item.content}</p>
                    <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-[var(--ink-soft)]">
                      {confidence ? <span>confidence {confidence}</span> : null}
                      {trust ? <span>trust {trust}</span> : null}
                      {source.label ? <span>{String(source.label)}</span> : null}
                      {item.source_memory_id ? (
                        <Link className="font-semibold text-[var(--teal)] underline" href={`/memories?memory=${item.source_memory_id}`}>Open memory</Link>
                      ) : null}
                      {item.kind === "proposed_skill" && item.skill_proposal_status ? (
                        <span>skill {item.skill_proposal_status}</span>
                      ) : null}
                    </div>
                    {item.skill_last_error ? <p className="mt-2 text-xs text-red-700">Application error: {item.skill_last_error}</p> : null}
                    {item.correction ? <p className="mt-2 rounded bg-white px-2 py-1 text-xs">Correction: {item.correction}</p> : null}

                    {item.status === "pending" ? (
                      <div className="mt-3 space-y-2">
                        {correcting === item.id ? (
                          <div className="space-y-2">
                            <label className="block text-xs font-semibold" htmlFor={`correction-${item.id}`}>What should this say instead?</label>
                            <textarea
                              id={`correction-${item.id}`}
                              value={correction}
                              onChange={(event) => setCorrection(event.target.value)}
                              className="min-h-20 w-full rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-sm"
                            />
                            <div className="flex gap-2">
                              <button disabled={!correction.trim() || busy === item.id} onClick={() => void decide(item, "correct")} className="rounded-md bg-[var(--teal)] px-3 py-2 text-xs font-semibold text-white disabled:opacity-40">Save correction</button>
                              <button onClick={() => { setCorrecting(null); setCorrection(""); }} className="rounded-md border border-[var(--outline)] px-3 py-2 text-xs font-semibold">Cancel</button>
                            </div>
                          </div>
                        ) : confirmForget === item.id ? (
                          <div className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm">
                            <p>This is protected or load-bearing. Forgetting removes it from active recall while preserving historical accountability.</p>
                            <div className="mt-2 flex gap-2">
                              <button disabled={busy === item.id} onClick={() => void decide(item, "forget", true)} className="rounded-md bg-red-700 px-3 py-2 text-xs font-semibold text-white disabled:opacity-40">Confirm forget</button>
                              <button onClick={() => setConfirmForget(null)} className="rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-xs font-semibold">Keep it</button>
                            </div>
                          </div>
                        ) : (
                          <div className="grid gap-2 sm:grid-cols-3">
                            <button disabled={busy === item.id} onClick={() => void decide(item, "approve")} className="rounded-md bg-[var(--teal)] px-3 py-2 text-xs font-semibold text-white disabled:opacity-40">Approve</button>
                            <button disabled={busy === item.id} onClick={() => { setCorrecting(item.id); setCorrection(""); }} className="rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-xs font-semibold disabled:opacity-40">Correct</button>
                            <button disabled={busy === item.id} onClick={() => void decide(item, "forget")} className="rounded-md border border-red-300 bg-white px-3 py-2 text-xs font-semibold text-red-700 disabled:opacity-40">Forget</button>
                          </div>
                        )}
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
