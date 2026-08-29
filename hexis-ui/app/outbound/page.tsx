"use client";

import {
  Ban,
  CircleDollarSign,
  MessageSquareOff,
  Pause,
  Play,
  RefreshCw,
  Send,
  ShieldCheck,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { PageHeader } from "../components/ui/page-header";
import { Spinner } from "../components/ui/spinner";

type OutboundEvent = {
  id: string;
  created_at: string;
  entity: string;
  entity_name: string;
  channel: string;
  recipient: string;
  purpose_kind: string | null;
  purpose_reference: string | null;
  charged_cost: number;
  status: "denied" | "authorized" | "delivered" | "failed";
  reason: string | null;
  disclosure_mode: "none" | "full" | "marker";
  urgency: string;
};

type ContactBudget = {
  entity: string;
  channel: string;
  points: number;
  max_points: number;
  regen_per_day: number;
  observed_per_week: number | null;
  reciprocity: number;
  strain: number;
  consecutive_silent: number;
  updated_at: string;
};

type ContactControl = {
  entity: string;
  blocked: boolean;
  suspended: boolean;
  reason: string | null;
  source_channel: string | null;
  updated_at: string;
};

type LedgerPayload = {
  suspended: boolean;
  events: OutboundEvent[];
  budgets: ContactBudget[];
  controls: ContactControl[];
};

export default function OutboundPage() {
  const [ledger, setLedger] = useState<LedgerPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const loadLedger = useCallback(async () => {
    try {
      const response = await fetch("/api/outbound?limit=150", { cache: "no-store" });
      const payload = (await response.json()) as LedgerPayload & { error?: string };
      if (!response.ok) throw new Error(payload.error || `Ledger request failed (${response.status})`);
      setLedger(payload);
      setError(null);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Failed to load outbound ledger.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadLedger();
    const timer = window.setInterval(loadLedger, 15000);
    return () => window.clearInterval(timer);
  }, [loadLedger]);

  const runControl = async (action: string, entity?: string) => {
    const key = `${action}:${entity || "global"}`;
    if (busy) return;
    setBusy(key);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch("/api/outbound", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, entity, reason: "dashboard_control" }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || result.error || `Control failed (${response.status})`);
      }
      setLedger(result.ledger as LedgerPayload);
      setNotice(
        action.includes("suspend")
          ? entity
            ? "Outbound contact paused for this person. Recipient STOP blocks remain unchanged."
            : "All outbound communication is paused."
          : entity
            ? "Outbound contact resumed for this person."
            : "Outbound communication resumed."
      );
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Control failed.");
    } finally {
      setBusy(null);
    }
  };

  const controlsByEntity = useMemo(
    () => new Map((ledger?.controls || []).map((item) => [item.entity, item])),
    [ledger?.controls]
  );
  const delivered = (ledger?.events || []).filter((item) => item.status === "delivered").length;
  const denied = (ledger?.events || []).filter((item) => item.status === "denied").length;
  const strained = (ledger?.budgets || []).filter((item) => item.strain > 0).length;
  const silent = (ledger?.budgets || []).filter((item) => item.consecutive_silent > 0).length;

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner label="Loading outbound ledger..." />
      </div>
    );
  }

  return (
    <div className="app-shell">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <div className="flex flex-col gap-4 border-b border-[var(--outline)] pb-5 lg:flex-row lg:items-center lg:justify-between">
          <PageHeader
            title="Outbound"
            subtitle="Purpose, attention cost, delivery, disclosure, and recipient controls"
          />
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" onClick={loadLedger} disabled={Boolean(busy)}>
              <RefreshCw size={16} className="mr-2 inline" />
              Refresh
            </Button>
            <Button
              onClick={() => runControl(ledger?.suspended ? "resume_global" : "suspend_global")}
              disabled={Boolean(busy)}
              className={ledger?.suspended ? "" : "!bg-red-700 hover:!bg-red-800"}
            >
              {ledger?.suspended ? (
                <Play size={16} className="mr-2 inline" />
              ) : (
                <Pause size={16} className="mr-2 inline" />
              )}
              {ledger?.suspended ? "Resume all outbound" : "Pause all outbound"}
            </Button>
          </div>
        </div>

        {error ? (
          <div className="mt-4 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error} Check that the Hexis API is running and migrations are current, then retry.
          </div>
        ) : null}
        {notice ? (
          <div className="mt-4 rounded-md border border-[var(--teal)]/40 bg-[var(--teal)]/5 px-3 py-2 text-sm">
            {notice}
          </div>
        ) : null}

        <Card className={`mt-5 ${ledger?.suspended ? "border-red-300 bg-red-50" : ""}`}>
          <div className="flex items-start gap-3">
            {ledger?.suspended ? <Ban className="mt-0.5 text-red-700" /> : <ShieldCheck className="mt-0.5 text-[var(--teal)]" />}
            <div>
              <p className="font-semibold">
                {ledger?.suspended ? "Outbound communication is paused" : "Outbound safeguards are active"}
              </p>
              <p className="mt-1 text-sm text-[var(--ink-soft)]">
                Recipient STOP blocks are permanent across channels until that recipient sends START or UNSTOP.
                Operator pause controls cannot clear them.
              </p>
            </div>
          </div>
        </Card>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric label="Delivered" value={delivered} icon={Send} />
          <Metric label="Policy denials" value={denied} icon={ShieldCheck} />
          <Metric label="Budgets in strain" value={strained} icon={CircleDollarSign} />
          <Metric label="Awaiting replies" value={silent} icon={MessageSquareOff} />
        </div>

        <div className="mt-6 grid gap-6 xl:grid-cols-[0.9fr_1.4fr]">
          <section>
            <h2 className="font-display text-xl font-semibold">Contact budgets</h2>
            <p className="mt-1 text-sm text-[var(--ink-soft)]">
              Replies are free. Inbound engagement restores points; urgent, backed contact may overdraft into visible strain.
            </p>
            <div className="mt-3 space-y-3">
              {(ledger?.budgets || []).length ? (
                ledger?.budgets.map((budget) => {
                  const control = controlsByEntity.get(budget.entity);
                  return (
                    <Card key={`${budget.entity}:${budget.channel}`}>
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate font-semibold">{entityLabel(budget.entity, ledger)}</p>
                          <p className="text-xs uppercase tracking-wide text-[var(--ink-soft)]">{budget.channel}</p>
                        </div>
                        {control?.blocked ? (
                          <Badge variant="error">STOP</Badge>
                        ) : control?.suspended ? (
                          <Badge variant="warning">Paused</Badge>
                        ) : budget.strain > 0 ? (
                          <Badge variant="warning">Strain {formatNumber(budget.strain)}</Badge>
                        ) : (
                          <Badge variant="success">Available</Badge>
                        )}
                      </div>
                      <div className="mt-4 h-2 overflow-hidden rounded-full bg-[var(--surface-strong)]">
                        <div
                          className="h-full rounded-full bg-[var(--teal)]"
                          style={{ width: `${budgetPercent(budget)}%` }}
                        />
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-[var(--ink-soft)]">
                        <span>{formatNumber(budget.points)} / {formatNumber(budget.max_points)} points</span>
                        <span className="text-right">+{formatNumber(budget.regen_per_day)}/day</span>
                        <span>{budget.consecutive_silent} unanswered</span>
                        <span className="text-right">reciprocity {formatNumber(budget.reciprocity)}</span>
                        <span>strain {formatNumber(budget.strain)}</span>
                        <span className="text-right">
                          observed {formatNumber(budget.observed_per_week)}/week
                        </span>
                      </div>
                      <div className="mt-4">
                        {control?.blocked ? (
                          <p className="text-xs text-red-700">Only this recipient can reverse their opt-out by sending START or UNSTOP.</p>
                        ) : (
                          <Button
                            variant="secondary"
                            className="w-full"
                            disabled={Boolean(busy)}
                            onClick={() => runControl(control?.suspended ? "resume_entity" : "suspend_entity", budget.entity)}
                          >
                            {control?.suspended ? "Resume this contact" : "Pause this contact"}
                          </Button>
                        )}
                      </div>
                    </Card>
                  );
                })
              ) : (
                <Card className="text-sm text-[var(--ink-soft)]">No third-party contact budget exists yet.</Card>
              )}
            </div>
          </section>

          <section>
            <h2 className="font-display text-xl font-semibold">Communication ledger</h2>
            <p className="mt-1 text-sm text-[var(--ink-soft)]">
              Every attempt records its purpose, cost, disclosure mode, and final outcome.
            </p>
            <div className="mt-3 overflow-hidden rounded-lg border border-[var(--outline)] bg-white">
              {(ledger?.events || []).length ? (
                <div className="divide-y divide-[var(--outline)]">
                  {ledger?.events.map((event) => (
                    <div key={event.id} className="p-4">
                      <div className="flex flex-wrap items-start justify-between gap-2">
                        <div>
                          <p className="font-semibold">{event.entity_name || event.entity}</p>
                          <p className="text-xs text-[var(--ink-soft)]">
                            {event.channel} · {shortTime(event.created_at)}
                          </p>
                        </div>
                        <StatusBadge status={event.status} />
                      </div>
                      <div className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
                        <p><span className="text-[var(--ink-soft)]">Purpose:</span> {event.purpose_kind || "missing"}</p>
                        <p><span className="text-[var(--ink-soft)]">Cost:</span> {formatNumber(event.charged_cost)} points</p>
                        <p className="truncate" title={event.purpose_reference || ""}>
                          <span className="text-[var(--ink-soft)]">Reference:</span> {event.purpose_reference || "—"}
                        </p>
                        <p><span className="text-[var(--ink-soft)]">Disclosure:</span> {event.disclosure_mode}</p>
                      </div>
                      {event.reason ? <p className="mt-3 rounded bg-red-50 px-2 py-1 text-xs text-red-700">{event.reason}</p> : null}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="p-5 text-sm text-[var(--ink-soft)]">No outbound attempts have been recorded.</p>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, icon: Icon }: { label: string; value: number; icon: typeof Send }) {
  return (
    <Card className="flex items-center gap-3">
      <div className="rounded-md bg-[var(--surface-strong)] p-2 text-[var(--teal)]"><Icon size={18} /></div>
      <div><p className="text-xs uppercase tracking-wide text-[var(--ink-soft)]">{label}</p><p className="text-xl font-semibold">{value}</p></div>
    </Card>
  );
}

function StatusBadge({ status }: { status: OutboundEvent["status"] }) {
  const variant = status === "delivered" ? "success" : status === "denied" || status === "failed" ? "error" : "warning";
  return <Badge variant={variant}>{status}</Badge>;
}

function entityLabel(entity: string, ledger: LedgerPayload | null): string {
  return ledger?.events.find((item) => item.entity === entity)?.entity_name || entity;
}

function budgetPercent(budget: ContactBudget): number {
  if (budget.max_points <= 0) return 0;
  return Math.max(0, Math.min(100, (budget.points / budget.max_points) * 100));
}

function formatNumber(value: number | null | undefined): string {
  return Number(value || 0).toFixed(2).replace(/\.00$/, "");
}

function shortTime(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}
