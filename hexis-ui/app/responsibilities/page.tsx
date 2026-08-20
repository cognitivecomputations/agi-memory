"use client";

import {
  BellRing,
  Check,
  Clock3,
  Pause,
  Play,
  RefreshCw,
  ShieldAlert,
  Trash2,
  Watch,
  type LucideIcon,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Label, Select, TextArea, TextInput } from "../components/ui/input";
import { PageHeader } from "../components/ui/page-header";
import { Spinner } from "../components/ui/spinner";

type JsonRecord = Record<string, unknown>;

type AmbientStatus = {
  enabled?: boolean;
  active?: number;
  blocked?: number;
  paused?: number;
  disabled?: number;
  due_now?: number;
  needs_setup?: number;
  next_due_at?: string | null;
};

type Responsibility = {
  id: string;
  title: string;
  kind: string;
  status: string;
  priority: string;
  user_intent: string;
  trigger: JsonRecord;
  evaluator: JsonRecord;
  sources: JsonRecord[];
  actions: JsonRecord[];
  delivery: JsonRecord;
  memory_policy: string;
  timezone: string;
  next_check_at: string | null;
  last_checked_at: string | null;
  last_fired_at: string | null;
  consecutive_errors: number;
  consecutive_silent: number;
  last_error: string | null;
  created_at: string | null;
  missing_connectors?: JsonRecord[];
};

type ResponsibilityDetail = {
  success?: boolean;
  responsibility?: Responsibility;
  latest_runs?: JsonRecord[];
  latest_observations?: JsonRecord[];
  latest_checkins?: JsonRecord[];
};

type ResponsibilitiesPayload = {
  status: AmbientStatus;
  responsibilities: Responsibility[];
};

const TEMPLATES = ["gmail", "checkin", "reminder", "custom"] as const;
type Template = (typeof TEMPLATES)[number];

const STATUS_FILTERS = ["all", "active", "blocked", "paused", "disabled"] as const;

export default function ResponsibilitiesPage() {
  const [payload, setPayload] = useState<ResponsibilitiesPayload | null>(null);
  const [detail, setDetail] = useState<ResponsibilityDetail | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filter, setFilter] = useState<(typeof STATUS_FILTERS)[number]>("all");
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [template, setTemplate] = useState<Template>("gmail");
  const [title, setTitle] = useState("");
  const [intent, setIntent] = useState("");
  const [priority, setPriority] = useState("normal");
  const [cadence, setCadence] = useState("300");
  const [gmailQuery, setGmailQuery] = useState("");
  const [urgentOnly, setUrgentOnly] = useState(false);
  const [lookbackHours, setLookbackHours] = useState("12");
  const [message, setMessage] = useState("");
  const [dailyTimes, setDailyTimes] = useState("09:00, 21:00");
  const [customJson, setCustomJson] = useState(
    JSON.stringify(
      {
        title: "Watch connector source",
        kind: "monitor",
        user_intent: "Let me know whenever a matching source item arrives.",
        trigger: { kind: "interval", every_seconds: 300 },
        sources: [{ connector_id: "slack", query: "from:hope" }],
        actions: [{ type: "notify_user" }],
      },
      null,
      2
    )
  );

  const fetchResponsibilities = useCallback(async () => {
    try {
      const search = filter === "all" ? "" : `?status=${encodeURIComponent(filter)}`;
      const response = await fetch(`/api/responsibilities${search}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`Failed to load responsibilities (${response.status})`);
      const data = (await response.json()) as ResponsibilitiesPayload;
      setPayload(data);
      setError(null);
      setSelectedId((current) =>
        current && data.responsibilities.some((item) => item.id === current)
          ? current
          : data.responsibilities[0]?.id ?? null
      );
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Failed to load responsibilities.");
    } finally {
      setLoading(false);
    }
  }, [filter]);

  const fetchDetail = useCallback(async (id: string | null) => {
    if (!id) {
      setDetail(null);
      return;
    }
    setDetailLoading(true);
    try {
      const response = await fetch(`/api/responsibilities/${encodeURIComponent(id)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`Failed to load detail (${response.status})`);
      setDetail((await response.json()) as ResponsibilityDetail);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Failed to load responsibility detail.");
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchResponsibilities();
    const timer = window.setInterval(fetchResponsibilities, 15000);
    return () => window.clearInterval(timer);
  }, [fetchResponsibilities]);

  useEffect(() => {
    fetchDetail(selectedId);
  }, [fetchDetail, selectedId]);

  const responsibilities = payload?.responsibilities || [];
  const selected =
    responsibilities.find((item) => item.id === selectedId) || detail?.responsibility || null;

  const runAction = async (action: string, args: JsonRecord = {}) => {
    const key = `${action}:${String(args.responsibility_id || args.title || "new")}`;
    if (busy) return;
    setBusy(key);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch("/api/responsibilities", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, arguments: args, source_session_id: "web-responsibilities" }),
      });
      const result = await response.json();
      if (!response.ok || result.success === false) {
        throw new Error(actionError(result, response.status));
      }
      setNotice(actionNotice(result));
      await fetchResponsibilities();
      if (args.responsibility_id) await fetchDetail(String(args.responsibility_id));
      const output = asRecord(result.output);
      const responsibility = asRecord(output.responsibility);
      if (typeof responsibility.id === "string") setSelectedId(responsibility.id);
      if (action === "create") setShowCreate(false);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Action failed.");
    } finally {
      setBusy(null);
    }
  };

  const createResponsibility = async () => {
    const args = buildCreateArgs();
    if (!args) return;
    await runAction("create", args);
  };

  const status = payload?.status || {};

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner label="Loading responsibilities..." />
      </div>
    );
  }

  return (
    <div className="app-shell">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <div className="flex flex-col gap-4 border-b border-[var(--outline)] pb-5 lg:flex-row lg:items-center lg:justify-between">
          <PageHeader
            title="Responsibilities"
            subtitle="Durable reminders, monitors, check-ins, and source watches"
          />
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="secondary" onClick={fetchResponsibilities}>
              <RefreshCw size={16} className="mr-2 inline" />
              Refresh
            </Button>
            <Button type="button" onClick={() => setShowCreate((value) => !value)}>
              <BellRing size={16} className="mr-2 inline" />
              {showCreate ? "Close" : "New"}
            </Button>
          </div>
        </div>

        {error ? (
          <div className="mt-4 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        ) : null}
        {notice ? (
          <div className="mt-4 rounded-md border border-[var(--teal)]/40 bg-[var(--teal)]/5 px-3 py-2 text-sm text-[var(--foreground)]">
            {notice}
          </div>
        ) : null}

        <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <Metric label="Active" value={String(status.active || 0)} icon={Watch} />
          <Metric label="Due now" value={String(status.due_now || 0)} icon={Clock3} />
          <Metric label="Blocked" value={String(status.blocked || 0)} icon={ShieldAlert} />
          <Metric label="Paused" value={String(status.paused || 0)} icon={Pause} />
          <Metric label="Next check" value={shortTime(status.next_due_at)} icon={BellRing} />
        </div>

        {showCreate ? (
          <Card className="mt-5">
            <div className="flex flex-wrap gap-2">
              {TEMPLATES.map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setTemplate(value)}
                  className={`rounded-md border px-3 py-2 text-sm font-medium capitalize ${
                    template === value
                      ? "border-[var(--teal)] bg-[var(--teal)]/10 text-[var(--foreground)]"
                      : "border-[var(--outline)] bg-white text-[var(--ink-soft)]"
                  }`}
                >
                  {value}
                </button>
              ))}
            </div>
            <div className="mt-5 grid gap-4 lg:grid-cols-2">
              {template !== "custom" ? (
                <>
                  <Field label="Title">
                    <TextInput value={title} onChange={(event) => setTitle(event.target.value)} placeholder={titlePlaceholder(template)} />
                  </Field>
                  <Field label="Priority">
                    <Select value={priority} onChange={(event) => setPriority(event.target.value)}>
                      {["low", "normal", "high", "urgent"].map((value) => (
                        <option key={value} value={value}>{value}</option>
                      ))}
                    </Select>
                  </Field>
                  <Field label="Intent">
                    <TextInput value={intent} onChange={(event) => setIntent(event.target.value)} placeholder={intentPlaceholder(template)} />
                  </Field>
                  <Field label="Message">
                    <TextInput value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Notification text" />
                  </Field>
                </>
              ) : null}

              {template === "gmail" ? (
                <>
                  <Field label="Gmail query">
                    <TextInput value={gmailQuery} onChange={(event) => setGmailQuery(event.target.value)} placeholder="from:hope@example.com newer_than:1d" />
                  </Field>
                  <Field label="Cadence">
                    <Select value={cadence} onChange={(event) => setCadence(event.target.value)}>
                      <option value="60">Every minute</option>
                      <option value="300">Every 5 minutes</option>
                      <option value="900">Every 15 minutes</option>
                      <option value="3600">Hourly</option>
                    </Select>
                  </Field>
                  <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={urgentOnly} onChange={(event) => setUrgentOnly(event.target.checked)} />
                    Urgent or important only
                  </label>
                </>
              ) : null}

              {template === "checkin" ? (
                <>
                  <Field label="Lookback">
                    <Select value={lookbackHours} onChange={(event) => setLookbackHours(event.target.value)}>
                      <option value="8">8 hours</option>
                      <option value="12">12 hours</option>
                      <option value="24">24 hours</option>
                      <option value="48">48 hours</option>
                    </Select>
                  </Field>
                  <Field label="Check cadence">
                    <Select value={cadence} onChange={(event) => setCadence(event.target.value)}>
                      <option value="900">Every 15 minutes</option>
                      <option value="3600">Hourly</option>
                      <option value="7200">Every 2 hours</option>
                    </Select>
                  </Field>
                </>
              ) : null}

              {template === "reminder" ? (
                <Field label="Times">
                  <TextInput value={dailyTimes} onChange={(event) => setDailyTimes(event.target.value)} placeholder="09:00, 21:00" />
                </Field>
              ) : null}

              {template === "custom" ? (
                <div className="lg:col-span-2">
                  <Field label="JSON">
                    <TextArea value={customJson} onChange={(event) => setCustomJson(event.target.value)} rows={12} />
                  </Field>
                </div>
              ) : null}
            </div>
            <div className="mt-5 flex justify-end">
              <Button type="button" disabled={Boolean(busy)} onClick={createResponsibility}>
                <Check size={16} className="mr-2 inline" />
                Create responsibility
              </Button>
            </div>
          </Card>
        ) : null}

        <div className="mt-5 flex flex-wrap gap-2">
          {STATUS_FILTERS.map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setFilter(value)}
              className={`rounded-full px-3 py-1.5 text-sm capitalize ${
                filter === value
                  ? "bg-[var(--foreground)] text-white"
                  : "bg-white text-[var(--ink-soft)] ring-1 ring-[var(--outline)]"
              }`}
            >
              {value}
            </button>
          ))}
        </div>

        <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
          <section className="space-y-3">
            {responsibilities.length === 0 ? (
              <div className="rounded-md border border-dashed border-[var(--outline)] bg-white px-4 py-10 text-center text-sm text-[var(--ink-soft)]">
                No responsibilities match this view.
              </div>
            ) : (
              responsibilities.map((item) => (
                <ResponsibilityCard
                  key={item.id}
                  item={item}
                  selected={item.id === selectedId}
                  busy={busy}
                  onSelect={() => setSelectedId(item.id)}
                  onAction={(action) => runAction(action, { responsibility_id: item.id })}
                />
              ))
            )}
          </section>

          <aside className="min-h-[420px] rounded-md border border-[var(--outline)] bg-white">
            <div className="border-b border-[var(--outline)] px-4 py-3">
              <h2 className="text-sm font-semibold">{selected?.title || "Responsibility detail"}</h2>
              <p className="mt-1 text-xs text-[var(--ink-soft)]">
                {selected ? `${selected.kind} · ${selected.status}` : "Select a responsibility."}
              </p>
            </div>
            {detailLoading ? (
              <div className="flex min-h-[260px] items-center justify-center">
                <Spinner label="Loading detail..." />
              </div>
            ) : (
              <DetailPane
                detail={detail}
                onCheckIn={(id) => runAction("checkin", { responsibility_id: id, source: "web" })}
                onCheckNow={(id) => runAction("evaluate_now", { responsibility_id: id })}
                busy={busy}
              />
            )}
          </aside>
        </div>
      </div>
    </div>
  );

  function buildCreateArgs(): JsonRecord | null {
    if (template === "custom") {
      try {
        return asRecord(JSON.parse(customJson));
      } catch {
        setError("Custom responsibility JSON is invalid.");
        return null;
      }
    }

    const baseTitle = title.trim() || titlePlaceholder(template);
    const userIntent = intent.trim() || intentPlaceholder(template);
    const notifyMessage = message.trim() || baseTitle;
    if (template === "gmail") {
      return {
        title: baseTitle,
        kind: "monitor",
        priority,
        user_intent: userIntent,
        trigger: { kind: "interval", every_seconds: Number(cadence) || 300 },
        sources: [{ connector_id: "gmail", query: gmailQuery.trim() || "in:inbox", page_size: 10 }],
        evaluator: urgentOnly ? { type: "importance", threshold: 0.85 } : {},
        actions: [{ type: "notify_user", message: notifyMessage }],
        delivery_mode: "outbox",
        memory_policy: "task_scoped",
      };
    }
    if (template === "checkin") {
      return {
        title: baseTitle,
        kind: "checkin",
        priority,
        user_intent: userIntent,
        trigger: { kind: "interval", every_seconds: Number(cadence) || 3600 },
        evaluator: { type: "missing_checkin", lookback_minutes: (Number(lookbackHours) || 12) * 60 },
        actions: [{ type: "notify_user", message: notifyMessage }],
        delivery_mode: "outbox",
      };
    }
    return {
      title: baseTitle,
      kind: "reminder",
      priority,
      user_intent: userIntent,
      trigger: {
        kind: "daily",
        times: dailyTimes.split(",").map((value) => value.trim()).filter(Boolean),
      },
      actions: [{ type: "notify_user", message: notifyMessage }],
      delivery_mode: "outbox",
    };
  }
}

function ResponsibilityCard({
  item,
  selected,
  busy,
  onSelect,
  onAction,
}: {
  item: Responsibility;
  selected: boolean;
  busy: string | null;
  onSelect: () => void;
  onAction: (action: string) => void;
}) {
  const blocked = item.status === "blocked";
  return (
    <div
      className={`w-full rounded-md border bg-white p-4 text-left transition ${
        selected ? "border-[var(--teal)] shadow-sm" : "border-[var(--outline)] hover:border-[var(--teal)]/50"
      }`}
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <button type="button" onClick={onSelect} className="min-w-0 flex-1 text-left">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold">{item.title}</span>
            <StatusBadge status={item.status} />
            <Badge variant="muted">{item.kind}</Badge>
            <Badge variant={item.priority === "urgent" || item.priority === "high" ? "warning" : "muted"}>
              {item.priority}
            </Badge>
          </div>
          <p className="mt-2 line-clamp-2 text-sm text-[var(--ink-soft)]">{item.user_intent}</p>
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--ink-soft)]">
            <span>next {shortDateTime(item.next_check_at)}</span>
            <span>checked {shortDateTime(item.last_checked_at)}</span>
            <span>fired {shortDateTime(item.last_fired_at)}</span>
            {blocked && item.last_error ? <span className="text-red-700">{item.last_error}</span> : null}
          </div>
        </button>
        <div className="flex shrink-0 flex-wrap gap-2">
          <IconButton label="Check now" disabled={Boolean(busy)} onClick={() => onAction("evaluate_now")}>
            <RefreshCw size={15} />
          </IconButton>
          {item.status === "paused" ? (
            <IconButton label="Resume" disabled={Boolean(busy)} onClick={() => onAction("resume")}>
              <Play size={15} />
            </IconButton>
          ) : (
            <IconButton label="Pause" disabled={Boolean(busy)} onClick={() => onAction("pause")}>
              <Pause size={15} />
            </IconButton>
          )}
          <IconButton label="Cancel" disabled={Boolean(busy)} onClick={() => onAction("cancel")}>
            <Trash2 size={15} />
          </IconButton>
        </div>
      </div>
    </div>
  );
}

function DetailPane({
  detail,
  busy,
  onCheckIn,
  onCheckNow,
}: {
  detail: ResponsibilityDetail | null;
  busy: string | null;
  onCheckIn: (id: string) => void;
  onCheckNow: (id: string) => void;
}) {
  const responsibility = detail?.responsibility;
  if (!responsibility) {
    return <div className="p-4 text-sm text-[var(--ink-soft)]">No responsibility selected.</div>;
  }
  const runs = detail?.latest_runs || [];
  const observations = detail?.latest_observations || [];
  const checkins = detail?.latest_checkins || [];
  return (
    <div className="space-y-5 p-4">
      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="secondary" disabled={Boolean(busy)} onClick={() => onCheckNow(responsibility.id)}>
          <RefreshCw size={15} className="mr-2 inline" />
          Check now
        </Button>
        {responsibility.kind === "checkin" ? (
          <Button type="button" disabled={Boolean(busy)} onClick={() => onCheckIn(responsibility.id)}>
            <Check size={15} className="mr-2 inline" />
            Check in
          </Button>
        ) : null}
      </div>

      <KeyValue label="Trigger" value={JSON.stringify(responsibility.trigger)} />
      <KeyValue label="Sources" value={JSON.stringify(responsibility.sources)} />
      <KeyValue label="Evaluator" value={JSON.stringify(responsibility.evaluator)} />

      <Section title="Runs" count={runs.length}>
        {runs.length ? runs.map((run) => <RunRow key={String(run.run_id)} run={run} />) : <EmptyLine text="No runs yet." />}
      </Section>
      <Section title="Observations" count={observations.length}>
        {observations.length ? observations.map((observation) => (
          <div key={String(observation.observation_id)} className="border-b border-[var(--outline)] py-3 last:border-0">
            <p className="text-sm font-medium">{asString(observation.title, "(Untitled)")}</p>
            <p className="mt-1 text-xs text-[var(--ink-soft)]">{shortDateTime(asString(observation.observed_at, null))}</p>
            <p className="mt-2 line-clamp-3 text-sm text-[var(--ink-soft)]">{asString(observation.content_preview, "")}</p>
          </div>
        )) : <EmptyLine text="No observations yet." />}
      </Section>
      <Section title="Check-ins" count={checkins.length}>
        {checkins.length ? checkins.map((checkin) => (
          <div key={String(checkin.checkin_id)} className="border-b border-[var(--outline)] py-3 text-sm last:border-0">
            <div className="flex items-center justify-between gap-3">
              <span>{asString(checkin.label, "check-in")}</span>
              <span className="text-xs text-[var(--ink-soft)]">{shortDateTime(asString(checkin.occurred_at, null))}</span>
            </div>
            {checkin.note ? <p className="mt-1 text-xs text-[var(--ink-soft)]">{String(checkin.note)}</p> : null}
          </div>
        )) : <EmptyLine text="No check-ins yet." />}
      </Section>
    </div>
  );
}

function RunRow({ run }: { run: JsonRecord }) {
  const decision = asRecord(run.decision);
  return (
    <div className="border-b border-[var(--outline)] py-3 last:border-0">
      <div className="flex items-center justify-between gap-3">
        <StatusBadge status={asString(run.status, "run")} />
        <span className="text-xs text-[var(--ink-soft)]">{shortDateTime(asString(run.started_at, null))}</span>
      </div>
      <p className="mt-2 text-sm text-[var(--ink-soft)]">
        {asString(decision.notify_message, asString(decision.reason, "No decision summary."))}
      </p>
    </div>
  );
}

function Metric({ label, value, icon: Icon }: { label: string; value: string; icon: LucideIcon }) {
  return (
    <div className="rounded-md border border-[var(--outline)] bg-white p-4">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-md bg-[var(--surface-strong)] text-[var(--teal)]">
          <Icon size={17} />
        </div>
        <div className="min-w-0">
          <p className="text-xs uppercase tracking-[0.25em] text-[var(--ink-soft)]">{label}</p>
          <p className="truncate text-lg font-semibold">{value}</p>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      {children}
    </div>
  );
}

function Section({ title, count, children }: { title: string; count: number; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-2 flex items-center gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        <Badge variant="muted">{count}</Badge>
      </div>
      <div>{children}</div>
    </section>
  );
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.25em] text-[var(--ink-soft)]">{label}</p>
      <pre className="mt-1 max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-md bg-[var(--surface-strong)] p-2 text-xs text-[var(--ink-soft)]">
        {value}
      </pre>
    </div>
  );
}

function IconButton({
  label,
  children,
  disabled,
  onClick,
}: {
  label: string;
  children: React.ReactNode;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="flex h-9 w-9 items-center justify-center rounded-md border border-[var(--outline)] bg-white text-[var(--ink-soft)] transition hover:bg-[var(--surface-strong)] hover:text-[var(--foreground)] disabled:opacity-50"
    >
      {children}
    </button>
  );
}

function StatusBadge({ status }: { status: string }) {
  const variant: "success" | "error" | "warning" | "muted" =
    status === "active" || status === "fired"
      ? "success"
      : status === "blocked" || status === "failed"
        ? "error"
        : status === "paused"
          ? "warning"
          : "muted";
  return <Badge variant={variant}>{status}</Badge>;
}

function EmptyLine({ text }: { text: string }) {
  return <p className="rounded-md border border-dashed border-[var(--outline)] px-3 py-4 text-sm text-[var(--ink-soft)]">{text}</p>;
}

function actionError(result: JsonRecord, status: number): string {
  if (typeof result.error === "string") return result.error;
  if (typeof result.detail === "string") return result.detail;
  const output = asRecord(result.output);
  if (typeof output.error === "string") return output.error;
  return `Action failed (${status})`;
}

function actionNotice(result: JsonRecord): string {
  if (typeof result.display_output === "string" && result.display_output) return result.display_output;
  const output = asRecord(result.output);
  const evaluation = asRecord(output.evaluation);
  const runs = Array.isArray(evaluation.runs) ? evaluation.runs : [];
  if (runs.length) {
    const decision = asRecord(asRecord(runs[0]).decision);
    return asString(decision.notify_message, asString(decision.reason, "Responsibility checked."));
  }
  return "Responsibility updated.";
}

function titlePlaceholder(template: Template): string {
  if (template === "gmail") return "Watch Gmail";
  if (template === "checkin") return "Medication check-in";
  if (template === "reminder") return "Twice daily reminder";
  return "Custom responsibility";
}

function intentPlaceholder(template: Template): string {
  if (template === "gmail") return "Let me know when matching Gmail messages arrive.";
  if (template === "checkin") return "Let me know if I have not checked in recently.";
  if (template === "reminder") return "Remind me twice daily.";
  return "Describe the responsibility.";
}

function asRecord(value: unknown): JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function asString(value: unknown, fallback: string | null): string {
  return typeof value === "string" && value.trim() ? value : fallback || "";
}

function shortDateTime(value: string | null | undefined): string {
  if (!value) return "never";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  return date.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function shortTime(value: string | null | undefined): string {
  if (!value) return "none";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "none";
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}
