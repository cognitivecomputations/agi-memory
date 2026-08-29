"use client";

import {
  Activity,
  Brain,
  ChevronRight,
  Clock3,
  HeartPulse,
  MessageCircle,
  Play,
  Target,
  Zap,
} from "lucide-react";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { Badge, GoalPriorityBadge, MemoryTypeBadge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { ProgressBar } from "./components/ui/progress-bar";
import { Spinner } from "./components/ui/spinner";
import { useGatewayEvents } from "./hooks/use-gateway-events";

type StatusData = {
  agent_name?: string;
  portrait_url?: string | null;
  configured?: boolean;
  energy?: number;
  max_energy?: number;
  energy_reserve?: number;
  energy_capacity?: number;
  next_regen_multiplier?: number;
  mood?: string;
  valence?: number;
  arousal?: number;
  intensity?: number;
  heartbeat_active?: boolean;
  heartbeat_paused?: boolean;
  heartbeat_count?: number;
  last_heartbeat_at?: string;
  next_heartbeat_at?: string;
  drives?: { name: string; urgency: number; hours_since: number }[];
  emotional_trend?: { hour: string; valence: number; arousal: number }[];
  goals?: { id: string; content: string; priority: string; source: string }[];
  recent_heartbeats?: {
    id: string;
    narrative: string;
    emotional_valence: number;
    created_at: string;
  }[];
  memory_health?: { type: string; count: number; avg_importance: number }[];
};

type UsageData = {
  total_cost_usd: number;
  total_tokens: number;
  total_calls: number;
  by_model: { provider: string; model: string; calls: number; tokens: number; cost_usd: number }[];
};

type HeartbeatTrace = {
  id: string;
  event: string;
  label: string;
  detail: string;
  payload: Record<string, unknown>;
};

export default function Dashboard() {
  const router = useRouter();
  const [status, setStatus] = useState<StatusData | null>(null);
  const [usage, setUsage] = useState<UsageData | null>(null);
  const [loading, setLoading] = useState(true);
  const [heartbeatRunning, setHeartbeatRunning] = useState(false);
  const [heartbeatError, setHeartbeatError] = useState<string | null>(null);
  const [heartbeatTrace, setHeartbeatTrace] = useState<HeartbeatTrace[]>([]);

  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (response.ok) setStatus(await response.json());
    } catch {
      // The page-level connection state remains visible.
    }
    try {
      const response = await fetch("/api/usage?period=30 days", { cache: "no-store" });
      if (response.ok) setUsage(await response.json());
    } catch {
      // Usage is secondary to runtime status.
    }
  }, []);

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        if (!response.ok) throw new Error("Failed to load status");
        const data = await response.json();
        setStatus(data);
        if (data.configured === false) {
          router.push("/init");
          return;
        }
        const usageResponse = await fetch("/api/usage?period=30 days", { cache: "no-store" });
        if (usageResponse.ok) setUsage(await usageResponse.json());
      } catch {
        setStatus(null);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [router]);

  useGatewayEvents(refreshStatus);

  const appendHeartbeat = (event: string, payload: Record<string, unknown>) => {
    setHeartbeatTrace((current) => [
      ...current.slice(-79),
      {
        id: crypto.randomUUID(),
        event,
        label: heartbeatEventLabel(event, payload),
        detail: heartbeatEventDetail(event, payload),
        payload,
      },
    ]);
  };

  const runHeartbeat = async () => {
    if (heartbeatRunning) return;
    setHeartbeatRunning(true);
    setHeartbeatError(null);
    setHeartbeatTrace([]);
    try {
      const response = await fetch("/api/heartbeat/run", { method: "POST" });
      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload?.error || `Heartbeat failed (${response.status})`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";
        for (const chunk of chunks) {
          const parsed = parseSseChunk(chunk);
          if (!parsed) continue;
          appendHeartbeat(parsed.event, parsed.payload);
          if (parsed.event === "error") {
            setHeartbeatError(asString(parsed.payload.message, "Heartbeat failed."));
          }
        }
      }
      await refreshStatus();
    } catch (error: unknown) {
      setHeartbeatError(error instanceof Error ? error.message : "Heartbeat failed.");
    } finally {
      setHeartbeatRunning(false);
    }
  };

  if (loading) {
    return <div className="flex min-h-screen items-center justify-center"><Spinner label="Loading overview..." /></div>;
  }

  if (!status) {
    return (
      <div className="flex min-h-screen items-center justify-center px-6">
        <div className="max-w-md rounded-lg border border-red-200 bg-white p-5">
          <h1 className="font-semibold text-red-800">Unable to reach Hexis</h1>
          <p className="mt-2 text-sm text-red-700">Start the Hexis stack, then refresh this page.</p>
        </div>
      </div>
    );
  }

  const totalMemories = (status.memory_health || []).reduce((sum, item) => sum + (item.count || 0), 0);
  const heartbeatState = status.heartbeat_paused
    ? "Paused"
    : heartbeatRunning || status.heartbeat_active
      ? "Running"
      : "Idle";

  return (
    <div className="app-shell">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <header className="flex flex-col gap-5 border-b border-[var(--outline)] pb-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            {status.portrait_url ? (
              <Image src={status.portrait_url} alt="" width={64} height={64} unoptimized className="h-16 w-16 rounded-lg object-cover" />
            ) : (
              <div className="flex h-16 w-16 items-center justify-center rounded-lg bg-[var(--foreground)] font-display text-2xl text-white">
                {(status.agent_name || "H").slice(0, 1)}
              </div>
            )}
            <div>
              <p className="text-xs font-semibold uppercase text-[var(--teal)]">Overview</p>
              <h1 className="font-display text-3xl text-[var(--foreground)]">{status.agent_name || "Hexis"}</h1>
              <p className="mt-1 text-sm text-[var(--ink-soft)]">
                {status.mood ? <span className="capitalize">{status.mood}</span> : "No emotional state"}
                {status.valence != null ? ` · valence ${signed(status.valence)}` : ""}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="secondary"
              onClick={runHeartbeat}
              disabled={heartbeatRunning || status.heartbeat_paused || status.heartbeat_active}
              title={status.heartbeat_paused ? "Resume heartbeat before running one" : "Run heartbeat now"}
            >
              {heartbeatRunning ? <Spinner className="mr-2" /> : <Play size={16} className="mr-2 inline" />}
              {heartbeatRunning ? "Running" : "Run heartbeat"}
            </Button>
            <Link href="/chat" className="inline-flex items-center rounded-lg bg-[var(--foreground)] px-4 py-2.5 text-sm font-semibold text-white hover:bg-[var(--teal)]">
              <MessageCircle size={16} className="mr-2" />
              Open conversation
            </Link>
          </div>
        </header>

        <section className="grid border-x border-b border-[var(--outline)] bg-white sm:grid-cols-2 xl:grid-cols-4">
          <Metric icon={Activity} label="Heartbeat" value={heartbeatState} detail={formatRelativeTime(status.last_heartbeat_at)} />
          <Metric icon={Zap} label="Energy" value={`${status.energy ?? 0} / ${status.energy_capacity ?? status.max_energy ?? 20}`} detail={`reserve ${status.energy_reserve ?? status.max_energy ?? 20} · ${status.heartbeat_count ?? 0} completed`} />
          <Metric icon={HeartPulse} label="Emotion" value={status.mood || "Unknown"} detail={status.arousal != null ? `arousal ${status.arousal.toFixed(2)}` : "No reading"} />
          <Metric icon={Brain} label="Memory" value={totalMemories.toLocaleString()} detail={`${status.memory_health?.length || 0} types`} />
        </section>

        <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
          <div className="space-y-6">
            {(heartbeatRunning || heartbeatTrace.length > 0 || heartbeatError) ? (
              <section className="overflow-hidden rounded-lg border border-[var(--outline)] bg-white">
                <div className="flex items-center justify-between border-b border-[var(--outline)] px-5 py-4">
                  <div>
                    <h2 className="font-semibold">Live heartbeat</h2>
                    <p className="mt-0.5 text-xs text-[var(--ink-soft)]">{heartbeatRunning ? "Running now" : "Latest manual run"}</p>
                  </div>
                  <Badge variant={heartbeatRunning ? "teal" : heartbeatError ? "error" : "success"}>
                    {heartbeatRunning ? "live" : heartbeatError ? "failed" : "complete"}
                  </Badge>
                </div>
                {heartbeatError ? <p className="border-b border-red-200 bg-red-50 px-5 py-3 text-sm text-red-700">{heartbeatError}</p> : null}
                <div className="max-h-[440px] overflow-y-auto">
                  {heartbeatTrace.map((trace, index) => (
                    <details key={trace.id} className="group border-b border-[var(--outline)] px-5 py-3 last:border-0">
                      <summary className="flex cursor-pointer list-none items-start gap-3">
                        <span className={`mt-1.5 h-2 w-2 flex-none rounded-full ${index === heartbeatTrace.length - 1 && heartbeatRunning ? "animate-pulse bg-[var(--accent)]" : "bg-[var(--teal)]"}`} />
                        <span className="min-w-0 flex-1">
                          <span className="block text-sm font-medium">{trace.label}</span>
                          <span className="mt-0.5 block truncate text-xs text-[var(--ink-soft)]">{trace.detail}</span>
                        </span>
                      </summary>
                      <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md bg-[#f5f7f5] p-3 text-xs text-[var(--foreground)]">{JSON.stringify(trace.payload, null, 2)}</pre>
                    </details>
                  ))}
                  {heartbeatRunning && heartbeatTrace.length === 0 ? <div className="p-5"><Spinner label="Starting heartbeat..." /></div> : null}
                </div>
              </section>
            ) : null}

            <section className="rounded-lg border border-[var(--outline)] bg-white">
              <div className="flex items-center justify-between border-b border-[var(--outline)] px-5 py-4">
                <div className="flex items-center gap-2"><Activity size={18} /><h2 className="font-semibold">Recent activity</h2></div>
                <span className="text-xs text-[var(--ink-soft)]">Autonomous heartbeats</span>
              </div>
              {(status.recent_heartbeats || []).length ? (
                <div>
                  {(status.recent_heartbeats || []).slice(0, 5).map((heartbeat) => (
                    <div key={heartbeat.id} className="border-b border-[var(--outline)] px-5 py-4 last:border-0">
                      <div className="flex items-start justify-between gap-4">
                        <p className="text-sm leading-6">{heartbeat.narrative || "Heartbeat completed."}</p>
                        {heartbeat.emotional_valence != null ? <Valence value={heartbeat.emotional_valence} /> : null}
                      </div>
                      <p className="mt-2 text-xs text-[var(--ink-soft)]">{formatDateTime(heartbeat.created_at)}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="px-5 py-8 text-sm text-[var(--ink-soft)]">No completed heartbeats yet.</div>
              )}
            </section>

            <section className="rounded-lg border border-[var(--outline)] bg-white">
              <div className="flex items-center justify-between border-b border-[var(--outline)] px-5 py-4">
                <div className="flex items-center gap-2"><Target size={18} /><h2 className="font-semibold">Current goals</h2></div>
                <Link href="/goals" className="flex items-center text-xs font-medium text-[var(--teal)]">Manage <ChevronRight size={14} /></Link>
              </div>
              {(status.goals || []).length ? (
                (status.goals || []).slice(0, 5).map((goal) => (
                  <div key={goal.id} className="flex items-center gap-3 border-b border-[var(--outline)] px-5 py-3 last:border-0">
                    <GoalPriorityBadge priority={goal.priority} />
                    <span className="text-sm">{goal.content}</span>
                  </div>
                ))
              ) : <div className="px-5 py-8 text-sm text-[var(--ink-soft)]">No active goals.</div>}
            </section>
          </div>

          <aside className="space-y-6">
            <section className="rounded-lg border border-[var(--outline)] bg-white p-5">
              <h2 className="font-semibold">Emotional state</h2>
              <div className="mt-4 flex items-end justify-between gap-4">
                <div>
                  <p className="text-2xl font-semibold capitalize">{status.mood || "Unknown"}</p>
                  <p className="mt-1 text-xs text-[var(--ink-soft)]">valence {status.valence != null ? signed(status.valence) : "--"}</p>
                </div>
                <div className="flex h-20 w-36 items-end gap-1">
                  {(status.emotional_trend || []).slice(0, 16).reverse().map((point) => (
                    <span
                      key={point.hour}
                      className={`min-h-1 flex-1 rounded-sm ${point.valence >= 0 ? "bg-[var(--teal)]" : "bg-[var(--accent)]"}`}
                      style={{ height: `${Math.max(5, Math.abs(point.valence || 0) * 100)}%` }}
                      title={`${point.hour}: ${signed(point.valence || 0)}`}
                    />
                  ))}
                </div>
              </div>
            </section>

            <section className="rounded-lg border border-[var(--outline)] bg-white p-5">
              <h2 className="font-semibold">Drives</h2>
              <div className="mt-4 space-y-3">
                {(status.drives || []).map((drive) => (
                  <ProgressBar key={drive.name} value={drive.urgency || 0} max={100} label={drive.name} showValue />
                ))}
              </div>
            </section>

            <section className="rounded-lg border border-[var(--outline)] bg-white p-5">
              <div className="flex items-center justify-between"><h2 className="font-semibold">Memory</h2><Link href="/memories" className="text-xs font-medium text-[var(--teal)]">Browse</Link></div>
              <p className="mt-3 text-2xl font-semibold">{totalMemories.toLocaleString()}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {(status.memory_health || []).map((item) => (
                  <span key={item.type} className="flex items-center gap-1"><MemoryTypeBadge type={item.type} /><span className="text-xs text-[var(--ink-soft)]">{item.count}</span></span>
                ))}
              </div>
            </section>

            <section className="rounded-lg border border-[var(--outline)] bg-white p-5">
              <h2 className="font-semibold">Usage</h2>
              <div className="mt-3 grid grid-cols-2 gap-4">
                <div><p className="text-xl font-semibold">${usage?.total_cost_usd.toFixed(2) || "0.00"}</p><p className="text-xs text-[var(--ink-soft)]">30 days</p></div>
                <div><p className="text-xl font-semibold">{usage?.total_calls.toLocaleString() || "0"}</p><p className="text-xs text-[var(--ink-soft)]">API calls</p></div>
              </div>
              {usage?.by_model.slice(0, 3).map((model) => (
                <div key={`${model.provider}/${model.model}`} className="mt-3 flex items-center justify-between gap-3 text-xs">
                  <span className="truncate text-[var(--ink-soft)]">{model.model}</span><span>{model.calls}</span>
                </div>
              ))}
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}

function Metric({ icon: Icon, label, value, detail }: { icon: typeof Clock3; label: string; value: string; detail: string }) {
  return (
    <div className="flex min-h-24 items-center gap-3 border-b border-[var(--outline)] p-4 last:border-b-0 sm:border-b-0 sm:border-r sm:last:border-r-0">
      <Icon size={19} className="text-[var(--teal)]" aria-hidden="true" />
      <div className="min-w-0"><p className="text-xs text-[var(--ink-soft)]">{label}</p><p className="truncate text-sm font-semibold capitalize">{value}</p><p className="truncate text-xs text-[var(--ink-soft)]">{detail}</p></div>
    </div>
  );
}

function Valence({ value }: { value: number }) {
  return <span className={`flex-none rounded-full px-2 py-1 text-xs font-medium ${value >= 0 ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"}`}>{signed(value)}</span>;
}

function parseSseChunk(chunk: string): { event: string; payload: Record<string, unknown> } | null {
  let event = "message";
  let data = "";
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    const value = JSON.parse(data);
    return { event, payload: value && typeof value === "object" ? value : { value } };
  } catch {
    return { event, payload: { value: data } };
  }
}

function heartbeatEventLabel(event: string, payload: Record<string, unknown>): string {
  if (event === "heartbeat_start") return `Heartbeat ${payload.heartbeat_number || ""} started`.trim();
  if (event === "heartbeat_done") return "Heartbeat completed";
  if (event === "phase") return `${asString(payload.phase, "Agent")} ${asString(payload.status)}`.trim();
  if (event === "trace") return asString(payload.kind) === "llm_request" ? "Model request" : "Model response";
  if (event === "tool") return `${asString(payload.tool_name, "Tool")} ${asString(payload.status)}`;
  if (event === "text") return "Agent response";
  if (event === "error") return "Heartbeat error";
  return asString(payload.event, event);
}

function heartbeatEventDetail(event: string, payload: Record<string, unknown>): string {
  if (event === "heartbeat_start") return "Manual heartbeat requested by the user";
  if (event === "phase") return `${asString(payload.phase, "Agent")} phase ${asString(payload.status, "updated")}`;
  if (event === "heartbeat_done") return `${payload.energy_spent || 0} energy · ${asString(payload.stopped_reason, "completed")}`;
  if (event === "trace") return `${asString(payload.provider)}/${asString(payload.model)}`;
  if (event === "tool") return asString(payload.display_output) || asString(payload.error) || "Tool activity";
  if (event === "text") return asString(payload.text).slice(0, 180);
  if (event === "error") return asString(payload.message, "Heartbeat failed");
  return JSON.stringify(payload).slice(0, 180);
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function formatDateTime(value?: string | null): string {
  return value ? new Date(value).toLocaleString() : "";
}

function formatRelativeTime(value?: string | null): string {
  if (!value) return "No previous heartbeat";
  const minutes = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 60000));
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  return `${hours}h ago`;
}
