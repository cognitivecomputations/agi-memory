"use client";

import { Activity, BrainCircuit, ChevronRight, Cpu, MessageCircle, Mic, Shield, Wrench } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Badge } from "../components/ui/badge";
import { PageHeader } from "../components/ui/page-header";
import { Spinner } from "../components/ui/spinner";
import { PwaSettings } from "../components/pwa/pwa-settings";

type SettingsData = {
  groups: Record<string, Record<string, unknown>>;
  llm: Record<string, unknown>;
  heartbeat: Record<string, unknown>;
  agent: Record<string, unknown>;
  tools: Record<string, unknown>;
};

const TABS = ["models", "autonomy", "tools", "voice", "app", "advanced"] as const;
type Tab = (typeof TABS)[number];

const MODEL_ROLES = [
  { key: "llm.chat", label: "Conversation", icon: MessageCircle },
  { key: "llm.heartbeat", label: "Heartbeat", icon: Activity },
  { key: "llm.subconscious", label: "Subconscious", icon: BrainCircuit },
  { key: "llm.recmem", label: "Memory maintenance", icon: Cpu },
  { key: "llm.summarization", label: "Summarization", icon: Cpu },
  { key: "llm.skill_improvement", label: "Skill improvement", icon: Wrench },
];

export default function SettingsPage() {
  const [data, setData] = useState<SettingsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("models");

  const fetchSettings = useCallback(async () => {
    try {
      const response = await fetch("/api/settings", { cache: "no-store" });
      if (!response.ok) throw new Error(`Failed to load settings (${response.status})`);
      setData(await response.json());
      setError(null);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Failed to load settings.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  useEffect(() => {
    if (window.location.hash === "#app-access") setTab("app");
  }, []);

  if (loading) return <div className="flex min-h-screen items-center justify-center"><Spinner label="Loading settings..." /></div>;
  if (!data) {
    return <div className="flex min-h-screen items-center justify-center px-6"><div className="max-w-md rounded-lg border border-red-200 bg-white p-5"><p className="text-sm text-red-700">{error || "Unable to load settings."}</p><button onClick={() => { setLoading(true); fetchSettings(); }} className="mt-4 rounded-md bg-[var(--foreground)] px-4 py-2 text-sm font-semibold text-white">Retry</button></div></div>;
  }

  const toolsConfig = asRecord(data.tools.tools);
  const contexts = asRecord(toolsConfig.context_overrides);
  const allowedActions = arrayOfStrings(data.heartbeat["heartbeat.allowed_actions"]);
  const energyReserve = numberValue(data.heartbeat["heartbeat.max_energy"], 20);
  const energyBankMultiplier = numberValue(
    data.heartbeat["heartbeat.energy_bank_multiplier"],
    3,
  );
  const energyBankCapacity = energyReserve * energyBankMultiplier;

  return (
    <div className="app-shell">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <div className="flex items-center justify-between gap-4 border-b border-[var(--outline)] pb-5">
          <PageHeader title="Settings" subtitle="Runtime configuration" />
          <Link href="/init" className="flex items-center rounded-lg border border-[var(--outline)] bg-white px-3 py-2 text-sm font-semibold hover:bg-[var(--surface-strong)]">Reconfigure <ChevronRight size={15} className="ml-1" /></Link>
        </div>

        <div className="mt-5 flex gap-1 overflow-x-auto border-b border-[var(--outline)]" role="tablist">
          {TABS.map((value) => (
            <button key={value} type="button" role="tab" aria-selected={tab === value} onClick={() => setTab(value)} className={`flex-none border-b-2 px-4 py-3 text-sm font-medium capitalize ${tab === value ? "border-[var(--teal)] text-[var(--foreground)]" : "border-transparent text-[var(--ink-soft)] hover:text-[var(--foreground)]"}`}>{value}</button>
          ))}
        </div>

        <div className="mt-6">
          {tab === "models" ? (
            <section>
              <div className="grid gap-3 md:grid-cols-2">
                {MODEL_ROLES.map(({ key, label, icon: Icon }) => {
                  const config = asRecord(data.llm[key]);
                  const inherited = Object.keys(config).length === 0;
                  return (
                    <div key={key} className="rounded-lg border border-[var(--outline)] bg-white p-4">
                      <div className="flex items-start gap-3">
                        <div className="flex h-9 w-9 items-center justify-center rounded-md bg-[var(--surface-strong)] text-[var(--teal)]"><Icon size={17} /></div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center justify-between gap-3"><h2 className="text-sm font-semibold">{label}</h2>{inherited ? <Badge variant="muted">inherited</Badge> : null}</div>
                          <p className="mt-2 truncate text-sm">{asString(config.model, "Default model")}</p>
                          <p className="mt-0.5 truncate text-xs text-[var(--ink-soft)]">{asString(config.provider, "Uses fallback configuration")}</p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          ) : null}

          {tab === "autonomy" ? (
            <section className="space-y-5">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                <SettingMetric label="Base cadence" value={`${numberValue(data.heartbeat["heartbeat.heartbeat_interval_minutes"], 60)} min`} />
                <SettingMetric label="Normal reserve" value={String(energyReserve)} />
                <SettingMetric label="Bank capacity" value={String(energyBankCapacity)} />
                <SettingMetric label="Base regeneration" value={`${numberValue(data.heartbeat["heartbeat.base_regeneration"], 10)} / hr`} />
                <SettingMetric label="Contact cooldown" value={`${numberValue(data.heartbeat["heartbeat.user_contact_cooldown_hours"], 0)} hr`} />
              </div>
              <div className="rounded-lg border border-[var(--outline)] bg-white">
                <div className="flex items-center justify-between border-b border-[var(--outline)] px-5 py-4"><h2 className="text-sm font-semibold">Allowed heartbeat actions</h2><Badge variant="teal">{allowedActions.length}</Badge></div>
                <div className="flex flex-wrap gap-2 p-5">{allowedActions.map((action) => <Badge key={action} variant="muted">{humanize(action)}</Badge>)}</div>
              </div>
              <div className="rounded-lg border border-[var(--outline)] bg-white p-5">
                <div className="flex items-center justify-between"><span className="flex items-center gap-2 text-sm font-semibold"><BrainCircuit size={17} /> Recursive reasoning</span><Badge variant={data.heartbeat["heartbeat.use_rlm"] === true ? "success" : "muted"}>{data.heartbeat["heartbeat.use_rlm"] === true ? "Enabled" : "Disabled"}</Badge></div>
              </div>
            </section>
          ) : null}

          {tab === "tools" ? (
            <section className="space-y-5">
              <div className="grid gap-4 lg:grid-cols-2">
                <PermissionPanel name="Conversation" value={asRecord(contexts.chat)} />
                <PermissionPanel name="Heartbeat" value={asRecord(contexts.heartbeat)} />
              </div>
              <div className="grid gap-4 md:grid-cols-3">
                <SettingMetric label="Globally disabled" value={String(arrayOfStrings(toolsConfig.disabled).length)} />
                <SettingMetric label="Disabled categories" value={String(arrayOfStrings(toolsConfig.disabled_categories).length)} />
                <SettingMetric label="MCP servers" value={String(Array.isArray(toolsConfig.mcp_servers) ? toolsConfig.mcp_servers.length : 0)} />
              </div>
              {arrayOfStrings(toolsConfig.disabled).length ? <div className="rounded-lg border border-[var(--outline)] bg-white p-5"><h2 className="text-sm font-semibold">Disabled tools</h2><div className="mt-3 flex flex-wrap gap-2">{arrayOfStrings(toolsConfig.disabled).map((tool) => <Badge key={tool} variant="warning">{humanize(tool)}</Badge>)}</div></div> : null}
            </section>
          ) : null}

          {tab === "voice" ? <VoicePanel /> : null}

          {tab === "app" ? <PwaSettings /> : null}

          {tab === "advanced" ? (
            <section className="space-y-3">
              {Object.entries(data.groups).sort(([a], [b]) => a.localeCompare(b)).map(([group, entries]) => (
                <details key={group} className="rounded-lg border border-[var(--outline)] bg-white">
                  <summary className="cursor-pointer px-5 py-4 text-sm font-semibold capitalize">{group} <span className="ml-2 font-normal text-[var(--ink-soft)]">{Object.keys(entries).length}</span></summary>
                  <div className="border-t border-[var(--outline)]">
                    {Object.entries(entries).map(([key, value]) => (
                      <div key={key} className="grid gap-1 border-b border-[var(--outline)] px-5 py-3 text-xs last:border-0 md:grid-cols-[260px_minmax(0,1fr)]"><span className="font-medium">{key}</span><pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words text-[var(--ink-soft)]">{formatValue(value)}</pre></div>
                    ))}
                  </div>
                </details>
              ))}
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
}

type VoiceSettings = {
  enabled: boolean;
  provider: string;
  model: string;
  language: string;
  cloud_disclosure_accepted: boolean;
  providers: Array<{ id: string; model: string }>;
};

function VoicePanel() {
  return <div className="space-y-5"><VoiceNotesPanel /><SpeechOutputPanel /></div>;
}

function VoiceNotesPanel() {
  const [settings, setSettings] = useState<VoiceSettings | null>(null);
  const [provider, setProvider] = useState("local_whisper");
  const [enabled, setEnabled] = useState(false);
  const [language, setLanguage] = useState("");
  const [cloudAcknowledged, setCloudAcknowledged] = useState(false);
  const [busy, setBusy] = useState(true);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      const response = await fetch("/api/settings/voice-notes", { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Voice-note settings could not be loaded.");
      setSettings(payload);
      setProvider(payload.provider);
      setEnabled(payload.enabled);
      setLanguage(payload.language || "");
      setCloudAcknowledged(payload.cloud_disclosure_accepted === true);
      setError(null);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Voice-note settings could not be loaded.");
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function save() {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      const response = await fetch("/api/settings/voice-notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider,
          enabled,
          language,
          cloud_acknowledged: cloudAcknowledged,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Voice-note settings could not be saved.");
      setSettings(payload);
      setMessage(enabled ? "Voice-note transcription is enabled." : "Voice-note transcription is off.");
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Voice-note settings could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  if (busy && !settings) return <Spinner label="Loading voice-note settings..." />;
  if (!settings) {
    return <div className="rounded-lg border border-red-200 bg-white p-5"><p className="text-sm text-red-700">{error}</p><button type="button" onClick={load} className="mt-4 rounded-md bg-[var(--foreground)] px-4 py-2 text-sm font-semibold text-white">Retry</button></div>;
  }

  return (
    <section className="space-y-5">
      <div className="rounded-lg border border-[var(--outline)] bg-white p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="flex gap-3"><div className="flex h-9 w-9 items-center justify-center rounded-md bg-[var(--surface-strong)] text-[var(--teal)]"><Mic size={18} /></div><div><h2 className="text-sm font-semibold">Voice-note transcription</h2><p className="mt-1 max-w-2xl text-sm text-[var(--ink-soft)]">Turn incoming audio messages into text before Hexis replies. Audio is processed only after the sender passes the channel allowlist.</p></div></div>
          <Badge variant={enabled ? "success" : "muted"}>{enabled ? "Enabled" : "Off"}</Badge>
        </div>

        <label className="mt-5 flex items-center gap-3 rounded-md border border-[var(--outline)] p-3 text-sm font-medium"><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /> Enable transcription</label>

        <fieldset className="mt-5 space-y-3">
          <legend className="text-sm font-semibold">Where audio is processed</legend>
          {settings.providers.map((option) => {
            const local = option.id === "local_whisper";
            return (
              <label key={option.id} className={`block cursor-pointer rounded-lg border p-4 ${provider === option.id ? "border-[var(--teal)] bg-[var(--teal-soft)]" : "border-[var(--outline)]"}`}>
                <span className="flex items-start gap-3"><input type="radio" name="voice-provider" value={option.id} checked={provider === option.id} onChange={() => { setProvider(option.id); if (local) setCloudAcknowledged(false); }} className="mt-1" /><span><span className="block text-sm font-semibold">{local ? "On this device" : "Cloud transcription"}</span><span className="mt-1 block text-sm text-[var(--ink-soft)]">{local ? "Audio stays local. Requires the optional Whisper media package; the first use downloads the selected model." : "Sends each voice-note file to the OpenAI-compatible endpoint configured for this installation."}</span><span className="mt-1 block text-xs text-[var(--ink-soft)]">Model: {option.model}</span></span></span>
              </label>
            );
          })}
        </fieldset>

        {provider === "local_whisper" ? <p className="mt-4 rounded-md bg-[var(--surface-strong)] p-3 text-xs text-[var(--ink-soft)]">If local transcription reports a missing dependency, install it in the Hexis environment with <code>pip install &apos;hexis[media]&apos;</code>, then retry the voice note.</p> : null}
        {provider === "openai_whisper" ? <label className="mt-4 flex items-start gap-3 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm"><input type="checkbox" checked={cloudAcknowledged} onChange={(event) => setCloudAcknowledged(event.target.checked)} className="mt-1" /><span>I understand that voice-note audio will leave this device and be sent to the configured cloud endpoint. <span className="block text-xs text-[var(--ink-soft)]">The channel worker also needs <code>OPENAI_API_KEY</code>.</span></span></label> : null}

        <label className="mt-4 block text-sm font-medium">Language hint <span className="font-normal text-[var(--ink-soft)]">(optional)</span><input value={language} onChange={(event) => setLanguage(event.target.value)} maxLength={35} placeholder="Auto-detect" className="mt-2 block w-full max-w-sm rounded-md border border-[var(--outline)] px-3 py-2 font-normal" /></label>

        {error ? <p role="alert" className="mt-4 text-sm text-red-700">{error}</p> : null}
        {message ? <p role="status" className="mt-4 text-sm text-emerald-700">{message}</p> : null}
        <button type="button" disabled={busy} onClick={save} className="mt-5 rounded-md bg-[var(--foreground)] px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">{busy ? "Saving..." : "Save voice-note settings"}</button>
      </div>
    </section>
  );
}

type VoiceOutputSettings = {
  enabled: boolean;
  provider: string;
  model: string;
  voice: string;
  talk_enabled: boolean;
  wake_enabled: boolean;
  providers: Array<{ id: string; model: string }>;
};

function SpeechOutputPanel() {
  const [settings, setSettings] = useState<VoiceOutputSettings | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [talkEnabled, setTalkEnabled] = useState(false);
  const [wakeEnabled, setWakeEnabled] = useState(false);
  const [provider, setProvider] = useState("local_piper");
  const [voice, setVoice] = useState("");
  const [providerReady, setProviderReady] = useState(false);
  const [providerDetail, setProviderDetail] = useState("");
  const [busy, setBusy] = useState(true);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      const [settingsResponse, statusResponse] = await Promise.all([
        fetch("/api/settings/voice-output", { cache: "no-store" }),
        fetch("/api/voice/status", { cache: "no-store" }),
      ]);
      const payload = await settingsResponse.json();
      const status = await statusResponse.json().catch(() => ({}));
      if (!settingsResponse.ok) throw new Error(payload.error || "Voice-output settings could not be loaded.");
      setSettings(payload);
      setEnabled(payload.enabled === true);
      setTalkEnabled(payload.talk_enabled === true);
      setWakeEnabled(payload.wake_enabled === true);
      setProvider(String(payload.provider || "local_piper"));
      setVoice(String(payload.voice || ""));
      setProviderReady(status.provider_ready === true);
      setProviderDetail(String(status.detail || ""));
      setError(null);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Voice-output settings could not be loaded.");
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  async function save() {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      const response = await fetch("/api/settings/voice-output", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled, talk_enabled: talkEnabled, wake_enabled: wakeEnabled, provider, voice }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Voice-output settings could not be saved.");
      setSettings(payload);
      setMessage(enabled ? "Local speech output is enabled." : "Speech output is off.");
      window.setTimeout(() => { void load(); }, 0);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Voice-output settings could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  if (busy && !settings) return <Spinner label="Loading speech-output settings..." />;
  if (!settings) {
    return <section className="rounded-lg border border-red-200 bg-white p-5"><p className="text-sm text-red-700">{error}</p><button type="button" onClick={load} className="mt-4 rounded-md bg-[var(--foreground)] px-4 py-2 text-sm font-semibold text-white">Retry</button></section>;
  }
  return (
    <section className="rounded-lg border border-[var(--outline)] bg-white p-5">
      <div className="flex items-start justify-between gap-4">
        <div><h2 className="text-sm font-semibold">Speech output and Talk mode</h2><p className="mt-1 max-w-2xl text-sm text-[var(--ink-soft)]">Render replies with a local Piper-compatible sidecar. Text stays on this device; ordinary text remains the accessible transcript.</p></div>
        <Badge variant={enabled && providerReady ? "success" : enabled ? "warning" : "muted"}>{enabled ? providerReady ? "Ready" : "Needs sidecar" : "Off"}</Badge>
      </div>
      <label className="mt-5 flex items-center gap-3 rounded-md border border-[var(--outline)] p-3 text-sm font-medium"><input type="checkbox" checked={enabled} onChange={(event) => { setEnabled(event.target.checked); if (!event.target.checked) { setTalkEnabled(false); setWakeEnabled(false); } }} /> Enable local speech output</label>
      <fieldset className="mt-5 space-y-3"><legend className="text-sm font-semibold">Local provider</legend>{settings.providers.map((option) => <label key={option.id} className={`block cursor-pointer rounded-lg border p-4 ${provider === option.id ? "border-[var(--teal)] bg-[var(--teal-soft)]" : "border-[var(--outline)]"}`}><span className="flex items-start gap-3"><input type="radio" name="speech-provider" checked={provider === option.id} onChange={() => setProvider(option.id)} className="mt-1" /><span><span className="block text-sm font-semibold">Piper-compatible sidecar</span><span className="mt-1 block text-sm text-[var(--ink-soft)]">Runs on loopback and returns WAV audio. Hexis refuses remote or credential-bearing provider URLs.</span><span className="mt-1 block text-xs text-[var(--ink-soft)]">Model: {option.model}</span></span></span></label>)}</fieldset>
      <label className="mt-4 block text-sm font-medium">Speaker name <span className="font-normal text-[var(--ink-soft)]">(optional; multi-speaker models only)</span><input value={voice} onChange={(event) => setVoice(event.target.value)} maxLength={100} placeholder="Use the model default" className="mt-2 block w-full max-w-sm rounded-md border border-[var(--outline)] px-3 py-2 font-normal" /></label>
      <label className="mt-4 flex items-start gap-3 rounded-md border border-[var(--outline)] p-3 text-sm"><input type="checkbox" checked={talkEnabled} disabled={!enabled} onChange={(event) => setTalkEnabled(event.target.checked)} className="mt-1" /><span><span className="font-medium">Allow foreground Talk mode</span><span className="mt-1 block text-xs text-[var(--ink-soft)]">The microphone starts only when you press Start Talk mode, stops when the page is hidden, and never runs in the background.</span></span></label>
      <label className="mt-4 flex items-start gap-3 rounded-md border border-[var(--outline)] p-3 text-sm"><input type="checkbox" checked={wakeEnabled} disabled={!enabled} onChange={(event) => setWakeEnabled(event.target.checked)} className="mt-1" /><span><span className="font-medium">Allow paired-node wake-word turns</span><span className="mt-1 block text-xs text-[var(--ink-soft)]">This server gate does not start a microphone. On the paired device, run <code>hexis node wake setup</code>, review the model license, then restart <code>hexis node run</code>. Disable local listening with <code>hexis node wake disable</code>.</span></span></label>
      <p className={`mt-4 rounded-md p-3 text-xs ${providerReady ? "bg-emerald-50 text-emerald-800" : "bg-[var(--surface-strong)] text-[var(--ink-soft)]"}`}>{providerReady ? providerDetail : `${providerDetail || "The local voice sidecar is not running."} Run hexis voice setup, then press Refresh status.`}</p>
      {error ? <p role="alert" className="mt-4 text-sm text-red-700">{error}</p> : null}
      {message ? <p role="status" className="mt-4 text-sm text-emerald-700">{message}</p> : null}
      <div className="mt-5 flex gap-2"><button type="button" disabled={busy} onClick={save} className="rounded-md bg-[var(--foreground)] px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">{busy ? "Saving..." : "Save speech settings"}</button><button type="button" disabled={busy} onClick={load} className="rounded-md border border-[var(--outline)] px-4 py-2 text-sm font-semibold">Refresh status</button></div>
    </section>
  );
}

function SettingMetric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-lg border border-[var(--outline)] bg-white p-4"><p className="text-xs text-[var(--ink-soft)]">{label}</p><p className="mt-1 text-xl font-semibold">{value}</p></div>;
}

function PermissionPanel({ name, value }: { name: string; value: Record<string, unknown> }) {
  const disabled = arrayOfStrings(value.disabled);
  return (
    <div className="rounded-lg border border-[var(--outline)] bg-white p-5">
      <div className="flex items-center justify-between"><span className="flex items-center gap-2 text-sm font-semibold"><Shield size={17} /> {name}</span><Badge variant={value.allow_all === true ? "success" : "muted"}>{value.allow_all === true ? "Broad access" : "Restricted"}</Badge></div>
      <dl className="mt-4 space-y-3 text-sm"><SettingRow label="Shell" enabled={value.allow_shell === true} /><SettingRow label="File writing" enabled={value.allow_file_write === true} />{typeof value.max_energy_per_tool === "number" ? <div className="flex justify-between"><dt className="text-[var(--ink-soft)]">Energy per tool</dt><dd>{value.max_energy_per_tool}</dd></div> : null}</dl>
      {disabled.length ? <div className="mt-4 flex flex-wrap gap-2">{disabled.map((tool) => <Badge key={tool} variant="warning">{humanize(tool)}</Badge>)}</div> : null}
    </div>
  );
}

function SettingRow({ label, enabled }: { label: string; enabled: boolean }) {
  return <div className="flex justify-between"><dt className="text-[var(--ink-soft)]">{label}</dt><dd className={enabled ? "text-emerald-700" : "text-[var(--ink-soft)]"}>{enabled ? "Allowed" : "Blocked"}</dd></div>;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" && value ? value : fallback;
}

function numberValue(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function humanize(value: string): string {
  return value.replace(/_/g, " ");
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "null";
  return typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
}
