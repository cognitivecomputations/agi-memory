"use client";

import {
  Activity,
  BrainCircuit,
  Database,
  Eye,
  EyeOff,
  Check,
  ExternalLink,
  FileText,
  Image as ImageIcon,
  Inbox,
  Lock,
  LockOpen,
  Mail,
  Paperclip,
  Plus,
  Send,
  Settings2,
  Trash2,
  Wrench,
  X,
  type LucideIcon,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useGatewayEvents } from "../hooks/use-gateway-events";
import Image from "next/image";
import { Card } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Spinner } from "../components/ui/spinner";
import { normalizeMessagePresentation } from "../../lib/message-presentation";
import type { MessagePresentation } from "../../lib/message-presentation";
import { isImageAttachmentFile, normalizeUploadFile } from "./attachment-helpers";
import { MessagePresentationView } from "./message-presentation";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: ChatImageAttachment[];
  presentation?: MessagePresentation;
  ui?: ChatUiArtifact[];
  incomplete?: boolean;
};

type ChatImageAttachment = {
  id: string;
  name: string;
  mimeType: string;
  dataUrl: string;
  byteSize: number;
};

type ChatVisualAttachmentPayload = {
  name: string;
  mime_type: string;
  data_url: string;
  byte_size: number;
};

type ConnectorSetupCapabilityOption = {
  id: string;
  label: string;
  description?: string;
  capabilities: string[];
  risk?: string;
};

type ConnectorSetupMemoryOption = {
  id: "remember" | "forget" | string;
  label: string;
  description?: string;
  memory_policy?: string;
};

type ConnectorSetupAutonomyOption = {
  id: string;
  label: string;
  description?: string;
  heartbeat_digest_enabled?: boolean;
};

type ConnectorSetupUi = {
  kind: "connector_setup";
  version?: number;
  id?: string;
  connector_id: string;
  display_name?: string;
  title?: string;
  status?: string;
  summary?: string;
  question?: string;
  capabilities?: string[];
  capability_options?: ConnectorSetupCapabilityOption[];
  memory_options?: ConnectorSetupMemoryOption[];
  autonomy_options?: ConnectorSetupAutonomyOption[];
  memory_policy?: string;
  memory_config_key?: string;
  heartbeat_digest_enabled?: boolean;
  heartbeat_digest_config_key?: string;
  client_secret_saved?: boolean;
  credentials_saved?: boolean;
  hexis_oauth_client_available?: boolean;
  accepted_inputs?: string[];
  env_client_secret_available?: boolean;
  credential_step?: ConnectorCredentialStep;
  credential_step_label?: string;
  setup_steps?: string[];
  technical_next_step?: string;
  docs_url?: string;
  authorization_url?: string;
  attempt_id?: string;
  completion_mode?: string;
  manual_completion_available?: boolean;
  connected_accounts?: Record<string, unknown>[];
  next_step?: string;
  safety_note?: string;
};

type ConnectorCredentialStep = {
  status?: string;
  preferred_mode?: string;
  save_action?: string;
  modes?: ConnectorCredentialMode[];
};

type ConnectorCredentialMode = {
  id: string;
  label: string;
  available?: boolean;
  description?: string;
};

type ChatUiArtifact = ConnectorSetupUi;

type IntegrationActionResult = {
  success?: boolean;
  output?: unknown;
  display_output?: string | null;
  error?: string | null;
  detail?: string | null;
};

// A large paste captured as an attachment instead of composer text; on send
// it is ingested as a document (POST /api/ingest) rather than inlined.
type PastedAttachment = {
  id: string;
  title: string;
  content: string;
  wordCount: number;
  // "private" keeps the ingested memories out of group-channel recall and
  // default HMX export (#92); toggled per-chip before sending.
  sensitivity: "private" | null;
};

// A dropped/picked file; on send it uploads to POST /api/ingest/file, which
// preserves the original bytes as a source artifact and runs the standard
// ingestion pipeline in the background (PDF, DOCX, XLSX, ...).
type FileAttachment = {
  id: string;
  file: File;
  name: string;
  size: number;
  mimeType: string;
  sensitivity: "private" | null;
};

type SearchConfigProvider = "tavily" | "brave" | "searxng" | "auto";

const SEARCH_CONFIG_PROVIDERS: { id: SearchConfigProvider; label: string }[] = [
  { id: "tavily", label: "Tavily" },
  { id: "brave", label: "Brave" },
  { id: "searxng", label: "SearXNG" },
  { id: "auto", label: "Auto" },
];

function searchConfigPlaceholder(provider: SearchConfigProvider): string {
  if (provider === "brave") return "brave-... or env:BRAVE_SEARCH_API_KEY";
  if (provider === "searxng") return "https://searxng.example";
  if (provider === "auto") return "uses keyless fallback";
  return "tvly-... or env:TAVILY_API_KEY";
}

const GMAIL_SETUP_STEPS = [
  "Open the Google setup page.",
  "Create or choose a project named Hexis.",
  "Enable the Gmail API for that project.",
  "Set up the app consent screen. For a personal Gmail account, choose External and add your Gmail address as a test user if Google asks.",
  "On the Credentials page, click Create credentials, choose Google's sign-in client option, set Application type to Desktop app, and name it Hexis.",
  "Download the setup file Google gives you.",
  "Upload that setup file here, then start Google sign-in.",
];

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

// The agent's outbox is her always-available way to reach the user; the
// channel worker tees every user-bound message into web_inbox (db/76), and
// this page shows that feed plus resource requests awaiting a decision.
type InboxMessage = {
  id: string;
  kind: string | null;
  intent: string | null;
  message: string;
  // Full envelope payload; payload.delivery carries machine-readable
  // correlation keys (request_id for resource requests, content_hash for
  // document fade asks, connector source ids).
  payload?: {
    delivery?: { request_id?: string; content_hash?: string } & Record<string, unknown>;
  } & Record<string, unknown>;
  delivered_at: string;
  read_at: string | null;
};

type PendingRequest = {
  id: string;
  kind: string;
  target_key: string | null;
  requested_value: unknown;
  rationale: string;
  duration: string | null;
  requested_at: string;
};

type InboxData = {
  unread: number;
  messages: InboxMessage[];
  pending_requests: PendingRequest[];
};

// Pastes longer than this become attachments (matching the Claude/ChatGPT
// composer convention) so huge texts go through document ingestion instead
// of flooding the conversation turn.
const PASTE_ATTACH_THRESHOLD = 2000;

// The turn's system prompt carries the attachment text up to this cap so the
// agent can discuss the document immediately; ingestion holds the full text.
const ATTACHMENT_PROMPT_CHARS = 16000;
const INLINE_IMAGE_MAX_BYTES = 6 * 1024 * 1024;
function attachmentTitle(content: string): string {
  const firstLine = content.split("\n").map((line) => line.trim()).find(Boolean) || "";
  if (!firstLine) return "Pasted text";
  if (firstLine.length <= 80) return firstLine;
  const words = firstLine.split(/\s+/).slice(0, 8).join(" ");
  return `${words}…`;
}

function attachmentAddendum(attachment: PastedAttachment): string {
  const truncated = attachment.content.length > ATTACHMENT_PROMPT_CHARS;
  const body = attachment.content.slice(0, ATTACHMENT_PROMPT_CHARS);
  return [
    `----- ATTACHED DOCUMENT: ${attachment.title} -----`,
    "The user attached this document to their message. It is also being ingested into your durable memory (recall or open_memory can retrieve it later).",
    "",
    body,
    truncated
      ? "\n[Document truncated here for the live turn — the full text is in memory via ingestion.]"
      : "",
  ].join("\n");
}

function imageAttachmentAddendum(attachment: FileAttachment, liveVisual: boolean): string {
  return [
    `----- ATTACHED IMAGE: ${attachment.name} -----`,
    liveVisual
      ? "The user attached this image to their message. You can inspect the image directly in this turn; do not reduce it to text extraction or assume it is only a text document."
      : "The user attached this image to their message. It was too large to include as live visual context; the original file is preserved in the filing cabinet.",
    "The original image is also preserved as a source artifact. Text extraction may run in the background when applicable, but the visual attachment is primary.",
  ].join("\n");
}

async function fileToDataUrl(file: File): Promise<string> {
  if (typeof FileReader !== "undefined") {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(reader.error || new Error("Failed to read image."));
      reader.onload = () => {
        if (typeof reader.result === "string") resolve(reader.result);
        else reject(new Error("Image reader returned non-text data."));
      };
      reader.readAsDataURL(file);
    });
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return `data:${file.type || "application/octet-stream"};base64,${btoa(binary)}`;
}

async function fileToJsonRecord(file: File): Promise<Record<string, unknown>> {
  const text = await file.text();
  const parsed = JSON.parse(text) as unknown;
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("JSON file must contain an object.");
  }
  return parsed as Record<string, unknown>;
}

function clipboardImageFiles(event: React.ClipboardEvent<HTMLTextAreaElement>): File[] {
  const fromItems = Array.from(event.clipboardData?.items || [])
    .filter((item) => item.kind === "file")
    .map((item) => item.getAsFile())
    .filter((file): file is File => file !== null)
    .filter(isImageAttachmentFile);
  if (fromItems.length > 0) return fromItems;
  return Array.from(event.clipboardData?.files || [])
    .filter((file) => file.size > 0 && isImageAttachmentFile(file));
}

type LogEvent = {
  id: string;
  category: "phase" | "subconscious" | "model" | "tool" | "memory" | "error";
  title: string;
  detail: string;
  raw?: unknown;
  ts: number;
};

type AgentStatus = {
  configured?: boolean;
  agent_name?: string;
  portrait_url?: string | null;
  mood?: string;
  valence?: number | null;
};

type ActivityPaneId = "thoughts" | "emotion" | "activity" | "memory";

type StreamMeter = {
  active: boolean;
  startedAt: number | null;
  tokens: number;
  tokensPerSecond: number;
};

type MemoryPreview = {
  id?: string;
  type?: string;
  content: string;
  similarity?: number | null;
  relevance_score?: number | null;
  importance?: number | null;
  trust_level?: number | null;
  confidence?: number | null;
  source?: string;
};

type ThoughtPair = {
  id: string;
  ts: number;
  subconscious?: LogEvent;
  conscious?: LogEvent;
  subconsciousText: string;
  consciousText: string;
};

type EmotionSnapshot = {
  id: string;
  ts: number;
  label: string;
  valence: number | null;
  detail: string;
};

type SsePayload = Record<string, unknown>;

const promptAddendaOptions = [
  { id: "philosophy", label: "Philosophy Grounding" },
  { id: "letter", label: "Letter From Claude" },
];

// Display cache only. The authoritative conversation context lives in
// Postgres via chat_sessions/chat_messages; this just avoids a blank repaint
// while the page hydrates the DB-owned session.
const SESSION_KEY = "hexis-chat-display-messages";
const LEGACY_SESSION_KEY = "hexis-chat-messages";
const SESSION_ID_KEY = "hexis-chat-session-id";
const ACTIVITY_KEY = "hexis-chat-activity-events";
const MAX_ACTIVITY_EVENTS = 60;
const ACTIVITY_TTL_MS = 30 * 60 * 1000;

function loadSessionId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return sessionStorage.getItem(SESSION_ID_KEY);
  } catch {
    return null;
  }
}

function saveSessionId(id: string) {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(SESSION_ID_KEY, id);
  } catch {
    // ignore quota errors
  }
}

function clearSessionId() {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.removeItem(SESSION_ID_KEY);
  } catch {
    // ignore quota errors
  }
}

function loadSession(): ChatMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = sessionStorage.getItem(SESSION_KEY) || sessionStorage.getItem(LEGACY_SESSION_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveSession(messages: ChatMessage[]) {
  if (typeof window === "undefined") return;
  try {
    if (messages.length > 0) {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify(messages));
      sessionStorage.removeItem(LEGACY_SESSION_KEY);
    } else {
      sessionStorage.removeItem(SESSION_KEY);
      sessionStorage.removeItem(LEGACY_SESSION_KEY);
    }
  } catch {
    // ignore quota errors
  }
}

function pruneActivityEvents(events: LogEvent[]): LogEvent[] {
  const cutoff = Date.now() - ACTIVITY_TTL_MS;
  return events
    .filter((event) => event.ts >= cutoff)
    .slice(-MAX_ACTIVITY_EVENTS);
}

function parseActivityEvents(value: unknown): LogEvent[] {
  if (!Array.isArray(value)) return [];
  const events: LogEvent[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const record = item as Record<string, unknown>;
    const category = record.category;
    if (
      category !== "phase" &&
      category !== "subconscious" &&
      category !== "model" &&
      category !== "tool" &&
      category !== "memory" &&
      category !== "error"
    ) {
      continue;
    }
    if (
      typeof record.id !== "string" ||
      typeof record.title !== "string" ||
      typeof record.detail !== "string" ||
      typeof record.ts !== "number"
    ) {
      continue;
    }
    events.push(sanitizeActivityEvent({
      id: record.id,
      category,
      title: record.title,
      detail: record.detail,
      raw: record.raw,
      ts: record.ts,
    }));
  }
  return pruneActivityEvents(events);
}

function sanitizeActivityEvent(event: LogEvent): LogEvent {
  if (
    event.category === "tool" &&
    event.title.toLowerCase().includes("gmail") &&
      /Create a Google OAuth Desktop client|call connect_gmail|client_secret_path|upload the Google OAuth client JSON|Hexis Google OAuth app|advanced self-hosted OAuth/i.test(event.detail)
  ) {
    return {
      ...event,
      detail:
        "Gmail setup needs a one-time local Google setup. Open the setup panel for the guided steps.",
    };
  }
  return event;
}

function loadActivityEvents(): LogEvent[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = sessionStorage.getItem(ACTIVITY_KEY);
    return raw ? parseActivityEvents(JSON.parse(raw)) : [];
  } catch {
    return [];
  }
}

function saveActivityEvents(events: LogEvent[]) {
  if (typeof window === "undefined") return;
  try {
    if (events.length > 0) {
      sessionStorage.setItem(ACTIVITY_KEY, JSON.stringify(pruneActivityEvents(events)));
    } else {
      sessionStorage.removeItem(ACTIVITY_KEY);
    }
  } catch {
    // ignore quota errors
  }
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : [];
}

function connectorCapabilityOptions(value: unknown): ConnectorSetupCapabilityOption[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item): ConnectorSetupCapabilityOption[] => {
    const record = asRecord(item);
    const id = asString(record.id);
    const label = asString(record.label);
    const capabilities = stringArray(record.capabilities);
    if (!id || !label || !capabilities.length) return [];
    return [
      {
        id,
        label,
        capabilities,
        description: asString(record.description) || undefined,
        risk: asString(record.risk) || undefined,
      },
    ];
  });
}

function connectorMemoryOptions(value: unknown): ConnectorSetupMemoryOption[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item): ConnectorSetupMemoryOption[] => {
    const record = asRecord(item);
    const id = asString(record.id);
    const label = asString(record.label);
    if (!id || !label) return [];
    return [
      {
        id,
        label,
        description: asString(record.description) || undefined,
        memory_policy: asString(record.memory_policy) || id,
      },
    ];
  });
}

function connectorAutonomyOptions(value: unknown): ConnectorSetupAutonomyOption[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item): ConnectorSetupAutonomyOption[] => {
    const record = asRecord(item);
    const id = asString(record.id);
    const label = asString(record.label);
    if (!id || !label) return [];
    return [
      {
        id,
        label,
        description: asString(record.description) || undefined,
        heartbeat_digest_enabled: record.heartbeat_digest_enabled === true,
      },
    ];
  });
}

function normalizeConnectorSetupUi(value: unknown): ConnectorSetupUi | null {
  const record = asRecord(value);
  if (record.kind !== "connector_setup") return null;
  const connectorId = asString(record.connector_id);
  if (!connectorId) return null;
  return {
    kind: "connector_setup",
    version: typeof record.version === "number" ? record.version : undefined,
    id: asString(record.id) || undefined,
    connector_id: connectorId,
    display_name: asString(record.display_name) || undefined,
    title: asString(record.title) || undefined,
    status: asString(record.status) || undefined,
    summary: asString(record.summary) || undefined,
    question: asString(record.question) || undefined,
    capabilities: stringArray(record.capabilities),
    capability_options: connectorCapabilityOptions(record.capability_options),
    memory_options: connectorMemoryOptions(record.memory_options),
    autonomy_options: connectorAutonomyOptions(record.autonomy_options),
    memory_policy: asString(record.memory_policy) || undefined,
    memory_config_key: asString(record.memory_config_key) || undefined,
    heartbeat_digest_enabled: record.heartbeat_digest_enabled === true,
    heartbeat_digest_config_key: asString(record.heartbeat_digest_config_key) || undefined,
    client_secret_saved:
      typeof record.client_secret_saved === "boolean" ? record.client_secret_saved : undefined,
    credentials_saved:
      typeof record.credentials_saved === "boolean" ? record.credentials_saved : undefined,
    accepted_inputs: stringArray(record.accepted_inputs),
    env_client_secret_available:
      typeof record.env_client_secret_available === "boolean"
        ? record.env_client_secret_available
        : undefined,
    credential_step: normalizeCredentialStep(record.credential_step),
    setup_steps: stringArray(record.setup_steps),
    technical_next_step: asString(record.technical_next_step) || undefined,
    docs_url: asString(record.docs_url) || undefined,
    authorization_url: asString(record.authorization_url) || undefined,
    attempt_id: asString(record.attempt_id) || undefined,
    completion_mode: asString(record.completion_mode) || undefined,
    manual_completion_available:
      typeof record.manual_completion_available === "boolean"
        ? record.manual_completion_available
        : undefined,
    connected_accounts: Array.isArray(record.connected_accounts)
      ? record.connected_accounts.filter(
          (item): item is Record<string, unknown> =>
            item !== null && typeof item === "object" && !Array.isArray(item)
        )
      : [],
    next_step: asString(record.next_step) || undefined,
    safety_note: asString(record.safety_note) || undefined,
  };
}

function normalizeCredentialStep(value: unknown): ConnectorCredentialStep | undefined {
  const record = asRecord(value);
  if (!Object.keys(record).length) return undefined;
  const rawModes = Array.isArray(record.modes) ? record.modes : [];
  const modes = rawModes.flatMap((item): ConnectorCredentialMode[] => {
    const mode = asRecord(item);
    const id = asString(mode.id);
    const label = asString(mode.label);
    if (!id || !label) return [];
    return [
      {
        id,
        label,
        available: typeof mode.available === "boolean" ? mode.available : undefined,
        description: asString(mode.description) || undefined,
      },
    ];
  });
  return {
    status: asString(record.status) || undefined,
    preferred_mode: asString(record.preferred_mode) || undefined,
    save_action: asString(record.save_action) || undefined,
    modes,
  };
}

function connectorSetupUiFromPayload(payload: unknown): ConnectorSetupUi | null {
  const direct = normalizeConnectorSetupUi(asRecord(payload).ui);
  if (direct) return direct;
  return normalizeConnectorSetupUi(asRecord(asRecord(payload).output).ui);
}

const NON_ACTIONABLE_CONNECTOR_SETUP_STATUSES = new Set(["connected", "complete", "verified"]);

function connectorSetupNeedsUserAction(ui: ConnectorSetupUi | null): boolean {
  if (!ui) return false;
  const status = (ui.status || "").toLowerCase();
  return !NON_ACTIONABLE_CONNECTOR_SETUP_STATUSES.has(status);
}

function uiArtifactKey(ui: ChatUiArtifact): string {
  if (ui.id) return ui.id;
  return `${ui.kind}:${ui.connector_id}:${ui.attempt_id || ui.status || "setup"}`;
}

function dbMessagesToChatMessages(value: unknown): ChatMessage[] | null {
  if (!Array.isArray(value)) return null;
  const messages: ChatMessage[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const record = item as Record<string, unknown>;
    const role = record.role;
    const content = record.content;
    if ((role === "user" || role === "assistant") && typeof content === "string") {
      messages.push({
        id: typeof record.message_id === "string" ? record.message_id : crypto.randomUUID(),
        role,
        content,
      });
    }
  }
  return messages;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [events, setEvents] = useState<LogEvent[]>(() => loadActivityEvents());
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [attachments, setAttachments] = useState<PastedAttachment[]>([]);
  const [fileAttachments, setFileAttachments] = useState<FileAttachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [ready, setReady] = useState<boolean | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus>({});
  const [promptAddenda, setPromptAddenda] = useState<string[]>([]);
  const [currentPhase, setCurrentPhase] = useState<string | null>(null);
  const [showSearchConfig, setShowSearchConfig] = useState(false);
  const [searchConfigProvider, setSearchConfigProvider] = useState<SearchConfigProvider>("tavily");
  const [searchConfigValue, setSearchConfigValue] = useState("");
  const [searchConfigSaving, setSearchConfigSaving] = useState(false);
  const [searchConfigError, setSearchConfigError] = useState<string | null>(null);
  const [searchConfigNotice, setSearchConfigNotice] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionActionBusy, setSessionActionBusy] = useState<"new" | "clear" | null>(null);
  const [sessionNotice, setSessionNotice] = useState<string | null>(null);
  const [historyIndex, setHistoryIndex] = useState<number | null>(null);
  const [historyDraft, setHistoryDraft] = useState("");
  const [showInspector, setShowInspector] = useState(false);
  const [showInbox, setShowInbox] = useState(false);
  const [inbox, setInbox] = useState<InboxData>({ unread: 0, messages: [], pending_requests: [] });
  const [decideBusy, setDecideBusy] = useState<string | null>(null);
  const [decideNotes, setDecideNotes] = useState<Record<string, string>>({});
  const [decideNotice, setDecideNotice] = useState<string | null>(null);
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [replyDraft, setReplyDraft] = useState("");
  const [replyBusy, setReplyBusy] = useState(false);
  const [connectorActionBusy, setConnectorActionBusy] = useState<string | null>(null);
  const [activeConnectorSetup, setActiveConnectorSetup] = useState<{
    assistantId: string;
    ui: ConnectorSetupUi;
  } | null>(null);
  const [expandedActivityPanes, setExpandedActivityPanes] = useState<Set<ActivityPaneId>>(
    new Set(["thoughts"]),
  );
  const [streamMeter, setStreamMeter] = useState<StreamMeter>({
    active: false,
    startedAt: null,
    tokens: 0,
    tokensPerSecond: 0,
  });
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const inboxBadgeCount = inbox.unread + inbox.pending_requests.length;

  const historyPayload = useMemo(
    () =>
      messages
        .filter((msg) => msg.content.trim() && !msg.incomplete)
        .map((msg) => ({ role: msg.role, content: msg.content })),
    [messages]
  );

  // Load the current DB-owned session on mount, using the browser cache only
  // as a temporary display fallback.
  useEffect(() => {
    let cancelled = false;

    const restore = async () => {
      const saved = loadSession();
      if (saved.length > 0 && !cancelled) setMessages(saved);

      const sessionId = loadSessionId();
      if (!sessionId) return;
      if (!cancelled) setSessionId(sessionId);
      try {
        const response = await fetch(`/api/chat/session/${encodeURIComponent(sessionId)}`, {
          cache: "no-store",
        });
        if (!response.ok) return;
        const payload = await response.json();
        const dbMessages = dbMessagesToChatMessages(payload?.messages);
        if (dbMessages && !cancelled) setMessages(dbMessages);
      } catch {
        // Keep the display cache; the next send still uses DB hydration.
      }
    };

    void restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadInbox = useCallback(async () => {
    try {
      const res = await fetch("/api/outbox", { cache: "no-store" });
      if (res.ok) setInbox(await res.json());
    } catch {
      // The badge just stays stale until the next poll.
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(loadInbox, 0);
    return () => window.clearTimeout(timer);
  }, [loadInbox]);
  useGatewayEvents(loadInbox);

  // Handling is explicit: each message is acknowledged, replied to, decided,
  // or deleted by the user — opening the panel no longer marks anything read.
  const ackInboxMessages = useCallback(
    async (ids: string[]) => {
      if (ids.length === 0) return;
      try {
        await fetch("/api/outbox", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids }),
        });
      } finally {
        void loadInbox();
      }
    },
    [loadInbox]
  );

  const deleteInboxMessage = async (message: InboxMessage) => {
    setDecideBusy(message.id);
    try {
      await fetch("/api/outbox", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: [message.id] }),
      });
    } finally {
      setDecideBusy(null);
      void loadInbox();
    }
  };

  // A document fade ask: 'approve' lets the memories fade permanently, 'keep'
  // reinforces them. ref = content_hash when the message carries one, else
  // the quoted label from the prose (resolve_document_fade matches both).
  const decideFade = async (message: InboxMessage, decision: "approve" | "keep") => {
    const ref =
      message.payload?.delivery?.content_hash ||
      message.message.match(/"([^"]+)"/)?.[1] ||
      "";
    if (!ref) {
      setDecideNotice("Could not identify which document this ask refers to.");
      return;
    }
    setDecideBusy(message.id);
    setDecideNotice(null);
    try {
      const res = await fetch("/api/fade/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ref, decision }),
      });
      const result = await res.json();
      if (!res.ok) {
        setDecideNotice(result.error ? `Fade decision failed: ${result.error}` : "Fade decision failed.");
      } else {
        setDecideNotice(
          decision === "approve"
            ? `"${result.label || ref}" will fade (${result.faded ?? 0} memories released).`
            : `"${result.label || ref}" kept and reinforced.`
        );
        await ackInboxMessages([message.id]);
      }
    } catch (err: unknown) {
      setDecideNotice(err instanceof Error ? err.message : "Fade decision failed.");
    } finally {
      setDecideBusy(null);
    }
  };

  const startInboxReply = (message: InboxMessage) => {
    setReplyingTo(message.id);
    setReplyDraft("");
    setDecideNotice(null);
  };

  const cancelInboxReply = () => {
    if (replyBusy) return;
    setReplyingTo(null);
    setReplyDraft("");
  };

  const submitInboxReply = async (message: InboxMessage) => {
    const reply = replyDraft.trim();
    if (!reply) {
      setDecideNotice("Write a reply before sending it.");
      return;
    }

    setReplyBusy(true);
    setDecideNotice(null);
    try {
      const res = await fetch("/api/outbox/reply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message_id: message.id, reply }),
      });
      const result = await res.json();
      if (!res.ok) {
        const detail = result.error || result.detail || "Hexis's inbox did not accept the reply.";
        setDecideNotice(`Reply was not queued: ${detail}`);
        return;
      }

      setReplyingTo(null);
      setReplyDraft("");
      setDecideNotice(
        `Reply queued for ${agentStatus.agent_name || "the agent"}'s next heartbeat.`
      );
      await loadInbox();
    } catch (err: unknown) {
      setDecideNotice(
        `Reply was not queued: ${err instanceof Error ? err.message : "Hexis's inbox is unavailable."}`
      );
    } finally {
      setReplyBusy(false);
    }
  };

  const decideRequest = async (request: PendingRequest, decision: "granted" | "denied") => {
    setDecideBusy(request.id);
    setDecideNotice(null);
    try {
      const res = await fetch("/api/requests/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: request.id,
          decision,
          note: decideNotes[request.id] || undefined,
        }),
      });
      const result = await res.json();
      if (!res.ok) {
        setDecideNotice(result.error || "The decision could not be recorded.");
      } else {
        const applied =
          result.applied === "config"
            ? " — config applied and journaled"
            : result.applied === "energy"
              ? ` — energy now ${result.new_energy}`
              : "";
        setDecideNotice(
          `Request ${request.id.slice(0, 8)} ${decision}${applied}. ` +
            `${agentStatus.agent_name || "The agent"} will see this at her next heartbeat.`
        );
        await loadInbox();
      }
    } catch (err: unknown) {
      setDecideNotice(err instanceof Error ? err.message : "The decision could not be recorded.");
    } finally {
      setDecideBusy(null);
    }
  };

  // Save display cache on message change. This is not sent as authoritative
  // history once a DB chat session exists.
  useEffect(() => {
    saveSession(messages);
  }, [messages]);

  useEffect(() => {
    saveActivityEvents(events);
  }, [events]);

  useEffect(() => {
    const desktop = window.matchMedia("(min-width: 1024px)");
    const sync = () => setShowInspector(desktop.matches);
    const frame = requestAnimationFrame(sync);
    desktop.addEventListener("change", sync);
    return () => {
      cancelAnimationFrame(frame);
      desktop.removeEventListener("change", sync);
    };
  }, []);

  useEffect(() => {
    const load = async () => {
      const res = await fetch("/api/status", { cache: "no-store" });
      if (!res.ok) {
        setReady(false);
        return;
      }
      const data = await res.json();
      setAgentStatus(data);
      setReady(data?.configured === true);
    };
    load().catch(() => setReady(false));
  }, []);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const timer = setInterval(() => {
      const cutoff = Date.now() - ACTIVITY_TTL_MS;
      setEvents((current) => current.filter((event) => event.ts >= cutoff));
    }, 60000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const latestAssistant = [...messages]
      .reverse()
      .find((msg) => msg.role === "assistant" && msg.content);
    if (latestAssistant) {
      setShowSearchConfig(isSearchToolMisconfigured(latestAssistant.content));
    }
  }, [messages]);

  const appendLog = (event: LogEvent) => {
    setEvents((prev) => [...prev.slice(-(MAX_ACTIVITY_EVENTS - 1)), sanitizeActivityEvent(event)]);
  };

  const toggleActivityPane = (pane: ActivityPaneId) => {
    setExpandedActivityPanes((current) => {
      const next = new Set(current);
      if (next.has(pane)) next.delete(pane);
      else next.add(pane);
      return next;
    });
  };

  const updateAssistantMessage = (assistantId: string, text: string) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === assistantId ? { ...msg, content: msg.content + text } : msg
      )
    );
  };

  const setAssistantPresentation = (assistantId: string, value: unknown) => {
    const presentation = normalizeMessagePresentation(value);
    if (!presentation) return;
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === assistantId ? { ...msg, presentation } : msg
      )
    );
  };

  const markTurnIncomplete = (userId: string, assistantId: string) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === userId || msg.id === assistantId
          ? { ...msg, incomplete: true }
          : msg
      )
    );
  };

  const upsertAssistantUi = (assistantId: string, ui: ChatUiArtifact) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id !== assistantId) return msg;
        const existing = msg.ui || [];
        return {
          ...msg,
          ui: [
            ...existing.filter(
              (item) => item.kind !== ui.kind || item.connector_id !== ui.connector_id
            ),
            ui,
          ],
        };
      })
    );
  };

  const replaceAssistantUi = (
    assistantId: string,
    previous: ChatUiArtifact,
    next: ChatUiArtifact
  ) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id !== assistantId) return msg;
        const existing = msg.ui || [];
        const kept = existing.filter(
          (item) =>
            (item.kind !== previous.kind || item.connector_id !== previous.connector_id) &&
            (item.kind !== next.kind || item.connector_id !== next.connector_id)
        );
        return { ...msg, ui: [...kept, next] };
      })
    );
  };

  const runConnectorSetupAction = async (
    assistantId: string,
    currentUi: ConnectorSetupUi,
    action: string,
    argumentsPayload: Record<string, unknown>
  ): Promise<IntegrationActionResult> => {
    const busyKey = `${assistantId}:${currentUi.connector_id}:${action}`;
    if (connectorActionBusy) return { error: "Another connector action is already running." };
    setConnectorActionBusy(busyKey);
    try {
      const currentSessionId = sessionId || loadSessionId() || "web-chat";
      const response = await fetch("/api/integrations/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          arguments: argumentsPayload,
          source_session_id: currentSessionId,
        }),
      });
      const payload = await readIntegrationActionPayload(response);
      const detail = integrationActionNotice(payload, response.status);
      appendLog({
        id: crypto.randomUUID(),
        category: payload.success === false || !response.ok ? "error" : "tool",
        title: currentUi.display_name || currentUi.connector_id,
        detail,
        raw: payload,
        ts: Date.now(),
      });
      const nextUi = connectorSetupUiFromPayload(payload);
      if (nextUi) {
        replaceAssistantUi(assistantId, currentUi, nextUi);
        if (connectorSetupNeedsUserAction(nextUi)) {
          setActiveConnectorSetup({ assistantId, ui: nextUi });
        } else {
          setActiveConnectorSetup(null);
        }
      }
      return payload;
    } catch (error: unknown) {
      const detail = error instanceof Error ? error.message : "Connector action failed.";
      appendLog({
        id: crypto.randomUUID(),
        category: "error",
        title: currentUi.display_name || currentUi.connector_id,
        detail,
        ts: Date.now(),
      });
      return { error: detail };
    } finally {
      setConnectorActionBusy(null);
    }
  };

  const refreshConnectorSetupStatus = async (
    assistantId: string,
    currentUi: ConnectorSetupUi
  ): Promise<IntegrationActionResult> => {
    try {
      const currentSessionId = sessionId || loadSessionId() || "web-chat";
      const response = await fetch("/api/integrations/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "gmail_setup_status",
          arguments: {},
          source_session_id: currentSessionId,
        }),
      });
      const payload = await readIntegrationActionPayload(response);
      const nextUi = connectorSetupUiFromPayload(payload);
      if (nextUi) {
        const changed =
          nextUi.id !== currentUi.id ||
          nextUi.status !== currentUi.status ||
          nextUi.credentials_saved !== currentUi.credentials_saved ||
          (nextUi.connected_accounts?.length || 0) !==
            (currentUi.connected_accounts?.length || 0);
        if (changed) {
          replaceAssistantUi(assistantId, currentUi, nextUi);
          if (connectorSetupNeedsUserAction(nextUi)) {
            setActiveConnectorSetup({ assistantId, ui: nextUi });
          } else {
            setActiveConnectorSetup(null);
          }
        }
      }
      return payload;
    } catch (error: unknown) {
      return { error: error instanceof Error ? error.message : "Could not refresh connector status." };
    }
  };

  const handleConfigureSearchTool = async () => {
    const value = searchConfigValue.trim();
    if (searchConfigProvider !== "auto" && !value) {
      setSearchConfigError(
        searchConfigProvider === "searxng"
          ? "Enter a SearXNG URL."
          : `Enter a ${searchConfigProvider === "brave" ? "Brave" : "Tavily"} key or env reference.`
      );
      return;
    }

    setSearchConfigSaving(true);
    setSearchConfigError(null);
    setSearchConfigNotice(null);
    try {
      const payload: Record<string, unknown> = {
        provider: searchConfigProvider,
        enable: true,
      };
      if (searchConfigProvider === "searxng") {
        payload.searxng_url = value;
      } else if (value.startsWith("env:")) {
        payload.key_ref = value;
      } else if (value) {
        payload.api_key = value;
      }
      const res = await fetch("/api/settings/tools/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data?.error || `Failed with status ${res.status}`);
      }
      appendLog({
        id: crypto.randomUUID(),
        category: "tool",
        title: "Search Tool",
        detail: "Configured web_search. Retry your question to run live search.",
        ts: Date.now(),
      });
      setShowSearchConfig(false);
      setSearchConfigValue("");
      setSearchConfigNotice("Web search configured. Retry your question.");
      setSessionNotice(null);
    } catch (err: unknown) {
      setSearchConfigError(
        err instanceof Error ? err.message : "Failed to configure search tool."
      );
    } finally {
      setSearchConfigSaving(false);
    }
  };

  const handleStartNewChat = async () => {
    if (sending || sessionActionBusy) return;
    setSessionActionBusy("new");
    setSessionNotice(null);
    try {
      const res = await fetch("/api/chat/session", { method: "POST" });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok || typeof payload.session_id !== "string") {
        throw new Error(payload?.error || `Failed with status ${res.status}`);
      }
      saveSessionId(payload.session_id);
      setSessionId(payload.session_id);
      setMessages([]);
      setActiveConnectorSetup(null);
      setHistoryIndex(null);
      setHistoryDraft("");
      setSearchConfigNotice(null);
      setSessionNotice("New conversation started.");
      appendLog({
        id: crypto.randomUUID(),
        category: "memory",
        title: "Chat session",
        detail: `Started DB session ${payload.session_id.slice(0, 8)}.`,
        ts: Date.now(),
      });
    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : "Failed to start a new conversation.";
      setSessionNotice(detail);
      appendLog({
        id: crypto.randomUUID(),
        category: "error",
        title: "Chat session",
        detail,
        ts: Date.now(),
      });
    } finally {
      setSessionActionBusy(null);
    }
  };

  const handleClearChatContext = async () => {
    if (sending || sessionActionBusy) return;
    const currentSessionId = sessionId || loadSessionId();
    if (!currentSessionId && messages.length === 0) return;
    const confirmed = window.confirm(
      "Clear this conversation from active context? Long-term memories and source records are preserved."
    );
    if (!confirmed) return;

    setSessionActionBusy("clear");
    setSessionNotice(null);
    try {
      if (!currentSessionId) {
        setMessages([]);
        clearSessionId();
        setSessionId(null);
        setSessionNotice("Conversation display cleared.");
        return;
      }

      const res = await fetch(`/api/chat/session/${encodeURIComponent(currentSessionId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "clear_context", reason: "web_clear" }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(payload?.error || `Failed with status ${res.status}`);
      }
      setMessages([]);
      setActiveConnectorSetup(null);
      setHistoryIndex(null);
      setHistoryDraft("");
      setSearchConfigNotice(null);
      setSessionNotice("Conversation context cleared; long-term memory preserved.");
      appendLog({
        id: crypto.randomUUID(),
        category: "memory",
        title: "Chat context",
        detail: `Cleared ${Number(payload?.cleared_messages || 0)} visible messages from active context.`,
        raw: payload,
        ts: Date.now(),
      });
    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : "Failed to clear conversation context.";
      setSessionNotice(detail);
      appendLog({
        id: crypto.randomUUID(),
        category: "error",
        title: "Chat context",
        detail,
        ts: Date.now(),
      });
    } finally {
      setSessionActionBusy(null);
    }
  };

  const handlePaste = (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    let handled = false;
    const images = clipboardImageFiles(event);
    if (images.length > 0) {
      addFiles(images, "paste");
      handled = true;
    }

    const pasted = event.clipboardData?.getData("text") ?? "";
    if (pasted.length > PASTE_ATTACH_THRESHOLD) {
      setAttachments((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          title: attachmentTitle(pasted),
          content: pasted,
          wordCount: pasted.split(/\s+/).filter(Boolean).length,
          sensitivity: null,
        },
      ]);
      handled = true;
    }

    if (handled) event.preventDefault();
  };

  const removeAttachment = (id: string) => {
    setAttachments((prev) => prev.filter((attachment) => attachment.id !== id));
  };

  const toggleAttachmentPrivacy = (id: string) => {
    setAttachments((prev) =>
      prev.map((attachment) =>
        attachment.id === id
          ? { ...attachment, sensitivity: attachment.sensitivity === "private" ? null : "private" }
          : attachment
      )
    );
  };

  const addFiles = (files: FileList | File[] | null, source: "picker" | "drop" | "paste" = "picker") => {
    if (!files) return;
    const items = Array.from(files).filter((file) => file.size > 0);
    if (!items.length) return;
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    setFileAttachments((prev) => [
      ...prev,
      ...items.map((file, index) => {
        const uploadFile = normalizeUploadFile(file, `${source === "paste" ? "pasted-image" : "attachment"}-${timestamp}-${index + 1}`);
        return {
          id: crypto.randomUUID(),
          file: uploadFile,
          name: uploadFile.name,
          size: uploadFile.size,
          mimeType: uploadFile.type,
          sensitivity: null as "private" | null,
        };
      }),
    ]);
  };

  const handleComposerDrop = (event: React.DragEvent<HTMLDivElement>) => {
    if (!event.dataTransfer?.files?.length) return;
    event.preventDefault();
    addFiles(event.dataTransfer.files, "drop");
  };

  const removeFileAttachment = (id: string) => {
    setFileAttachments((prev) => prev.filter((attachment) => attachment.id !== id));
  };

  const toggleFileAttachmentPrivacy = (id: string) => {
    setFileAttachments((prev) =>
      prev.map((attachment) =>
        attachment.id === id
          ? { ...attachment, sensitivity: attachment.sensitivity === "private" ? null : "private" }
          : attachment
      )
    );
  };

  const handleSend = async () => {
    if ((!input.trim() && attachments.length === 0 && fileAttachments.length === 0) || sending) return;

    // Attachments ingest as documents (durable) AND ride the turn's prompt
    // addenda (immediate sight); the visible message carries only a note.
    const toIngest = attachments;
    setAttachments([]);
    const filesToUpload = fileAttachments;
    setFileAttachments([]);
    const attachmentAddenda = toIngest.map(attachmentAddendum);
    const ingestNotes: string[] = [];
    const visualAttachments: ChatVisualAttachmentPayload[] = [];
    const visibleImageAttachments: ChatImageAttachment[] = [];

    for (const attachment of filesToUpload) {
      if (!isImageAttachmentFile(attachment.file) || attachment.size > INLINE_IMAGE_MAX_BYTES) {
        continue;
      }
      try {
        const dataUrl = await fileToDataUrl(attachment.file);
        visualAttachments.push({
          name: attachment.name,
          mime_type: attachment.mimeType || attachment.file.type || "image/png",
          data_url: dataUrl,
          byte_size: attachment.size,
        });
        visibleImageAttachments.push({
          id: attachment.id,
          name: attachment.name,
          mimeType: attachment.mimeType || attachment.file.type || "image/png",
          dataUrl,
          byteSize: attachment.size,
        });
      } catch (err) {
        appendLog({
          id: crypto.randomUUID(),
          category: "error",
          title: "Image preview error",
          detail: `Image "${attachment.name}" could not be prepared for the live turn: ${err instanceof Error ? err.message : String(err)}`,
          ts: Date.now(),
        });
      }
    }

    // Dropped files upload as original bytes: preserved as source artifacts
    // first, then ingested by a durable background job.
    for (const attachment of filesToUpload) {
      try {
        const form = new FormData();
        form.append("file", attachment.file, attachment.name);
        form.append("mode", "fast");
        if (attachment.sensitivity) form.append("sensitivity", attachment.sensitivity);
        const res = await fetch("/api/ingest/file", { method: "POST", body: form });
        if (res.ok) {
          const isImage = isImageAttachmentFile(attachment.file);
          const liveVisual = visualAttachments.some((item) => item.name === attachment.name);
          ingestNotes.push(
            isImage
              ? `[Attached image "${attachment.name}" (${formatBytes(attachment.size)}) — ${
                  liveVisual ? "visible in this turn; " : "too large for live visual context; "
                }original preserved in the filing cabinet${
                  attachment.sensitivity === "private" ? " as private (kept out of group conversations and exports)" : ""
                }.]`
              : `[Attached file "${attachment.name}" (${formatBytes(attachment.size)}) — original preserved, being ingested into memory${
                  attachment.sensitivity === "private" ? " as private (kept out of group conversations and exports)" : ""
                }. Search it later with search_documents / search_document_chunks.]`
          );
          attachmentAddenda.push(
            isImage
              ? imageAttachmentAddendum(attachment, liveVisual)
              : [
                  `----- ATTACHED FILE: ${attachment.name} -----`,
                  "The user attached this file to their message. Its original bytes are preserved and it is being ingested in the background — the text is not inlined here.",
                  "Once ingestion completes (usually under a minute), search_documents / search_document_chunks will find it and open_document can read the extracted content.",
                ].join("\n")
          );
        } else {
          const detail = await res.text();
          ingestNotes.push(
            `[Attached file "${attachment.name}" could not be uploaded: ${res.status}]`
          );
          appendLog({
            id: crypto.randomUUID(),
            category: "error",
            title: "Upload error",
            detail: `File "${attachment.name}" failed (${res.status}): ${detail.slice(0, 200)}`,
            ts: Date.now(),
          });
        }
      } catch (err) {
        ingestNotes.push(
          `[Attached file "${attachment.name}" could not be uploaded: network error]`
        );
        appendLog({
          id: crypto.randomUUID(),
          category: "error",
          title: "Upload error",
          detail: `File "${attachment.name}": ${err instanceof Error ? err.message : String(err)}`,
          ts: Date.now(),
        });
      }
    }
    for (const attachment of toIngest) {
      try {
        const res = await fetch("/api/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            content: attachment.content,
            title: attachment.title,
            mode: "fast",
            sensitivity: attachment.sensitivity ?? undefined,
          }),
        });
        if (res.ok) {
          ingestNotes.push(
            `[Attached document "${attachment.title}" (${attachment.wordCount} words) — being ingested into memory${
              attachment.sensitivity === "private" ? " as private (kept out of group conversations and exports)" : ""
            }]`
          );
        } else {
          const detail = await res.text();
          ingestNotes.push(
            `[Attached document "${attachment.title}" could not be ingested: ${res.status}]`
          );
          appendLog({
            id: crypto.randomUUID(),
            category: "error",
            title: "Ingest error",
            detail: `Attachment "${attachment.title}" failed (${res.status}): ${detail.slice(0, 200)}`,
            ts: Date.now(),
          });
        }
      } catch (err) {
        ingestNotes.push(
          `[Attached document "${attachment.title}" could not be ingested: network error]`
        );
        appendLog({
          id: crypto.randomUUID(),
          category: "error",
          title: "Ingest error",
          detail: `Attachment "${attachment.title}": ${err instanceof Error ? err.message : String(err)}`,
          ts: Date.now(),
        });
      }
    }

    const messageText = [input.trim(), ...ingestNotes].filter(Boolean).join("\n\n");
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: messageText,
      attachments: visibleImageAttachments,
    };
    const assistantMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
    };
    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setHistoryIndex(null);
    setHistoryDraft("");
    setSending(true);
    setStreamMeter({
      active: true,
      startedAt: Date.now(),
      tokens: 0,
      tokensPerSecond: 0,
    });
    setCurrentPhase(null);
    setShowSearchConfig(false);
    setSearchConfigError(null);
    setSearchConfigNotice(null);
    setSessionNotice(null);

    let receivedDone = false;

    try {
      const currentSessionId = sessionId || loadSessionId();
      const chatBody: {
        message: string;
        prompt_addenda: string[];
        visual_attachments?: ChatVisualAttachmentPayload[];
        session_id?: string | null;
        history?: { role: string; content: string }[];
      } = {
        message: userMessage.content,
        prompt_addenda: [...promptAddenda, ...attachmentAddenda],
        session_id: currentSessionId,
      };
      if (visualAttachments.length > 0) {
        chatBody.visual_attachments = visualAttachments;
      }
      if (!currentSessionId && historyPayload.length > 0) {
        chatBody.history = historyPayload;
      }

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(chatBody),
      });
      if (!res.ok || !res.body) {
        appendLog({
          id: crypto.randomUUID(),
          category: "error",
          title: "Chat error",
          detail: `Failed to reach chat endpoint (${res.status}).`,
          ts: Date.now(),
        });
        setSending(false);
        setStreamMeter((current) => ({ ...current, active: false }));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const lines = part.split("\n");
          let eventType = "message";
          let data = "";
          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventType = line.replace("event:", "").trim();
            }
            if (line.startsWith("data:")) {
              data += line.replace("data:", "").trim();
            }
          }
          if (!data) continue;
          let payload: SsePayload = {};
          try {
            const parsed = JSON.parse(data);
            payload =
              parsed && typeof parsed === "object" && !Array.isArray(parsed)
                ? (parsed as SsePayload)
                : { raw: data };
          } catch {
            payload = { raw: data };
          }

          if (eventType === "token") {
            const phase = asString(payload.phase);
            const text = asString(payload.text);
            setCurrentPhase(phase);
            if ((phase === "conscious_final" || phase === "connector_setup") && text) {
              updateAssistantMessage(assistantMessage.id, text);
              setStreamMeter((current) => {
                const now = Date.now();
                const startedAt = current.startedAt || now;
                const tokens = current.tokens + Math.max(0, text.length / 4);
                const elapsedSeconds = Math.max(0.25, (now - startedAt) / 1000);
                return {
                  active: true,
                  startedAt,
                  tokens,
                  tokensPerSecond: tokens / elapsedSeconds,
                };
              });
              if (isSearchToolMisconfigured(text)) {
                setShowSearchConfig(true);
              }
            }
          }

          if (eventType === "phase_start") {
            const phase = asString(payload.phase, "phase");
            setCurrentPhase(phase);
            appendLog({
              id: crypto.randomUUID(),
              category: "phase",
              title: streamLabel(phase),
              detail: "started",
              ts: Date.now(),
            });
          }

          if (eventType === "phase_end" && asString(payload.phase) === "subconscious") {
            const output = asRecord(payload.output);
            appendLog({
              id: crypto.randomUUID(),
              category: "subconscious",
              title: "Subconscious appraisal",
              detail: summarizeSubconscious(output),
              raw: output,
              ts: Date.now(),
            });
          }

          if (eventType === "trace") {
            const request = asString(payload.kind) === "llm_request";
            appendLog({
              // Request/response traces share payload.id for correlation, so
              // the log entry mints its own key; the pair id stays in raw.
              id: crypto.randomUUID(),
              category: "model",
              title: request ? "Model request" : "Model response",
              detail: `${asString(payload.provider, "provider")}/${asString(payload.model, "model")} · iteration ${String(payload.iteration ?? "-")}`,
              raw: payload,
              ts: Date.now(),
            });
          }

          if (eventType === "log") {
            const connectorUi = connectorSetupUiFromPayload(payload);
            const detail = connectorUi
              ? connectorSetupNotice(connectorUi)
              : asString(payload.detail);
            const logKind = asString(payload.kind).toLowerCase();
            const title =
              connectorUi?.display_name ||
              connectorUi?.connector_id ||
              asString(payload.title) ||
              logKind ||
              "Activity";
            appendLog({
              id: crypto.randomUUID(),
              category: logKind.includes("memory") || title.toLowerCase().includes("memory") ? "memory" : "tool",
              title,
              detail,
              raw: payload,
              ts: Date.now(),
            });
            if (connectorUi && connectorSetupNeedsUserAction(connectorUi)) {
              upsertAssistantUi(assistantMessage.id, connectorUi);
              setActiveConnectorSetup({ assistantId: assistantMessage.id, ui: connectorUi });
            } else if (connectorUi) {
              setActiveConnectorSetup((current) =>
                current?.ui.connector_id === connectorUi.connector_id ? null : current
              );
            }
            if (isSearchToolMisconfigured(detail)) {
              setShowSearchConfig(true);
            }
          }

          if (eventType === "ui") {
            const connectorUi = connectorSetupUiFromPayload(payload);
            const title =
              asString(payload.tool_name) ||
              connectorUi?.display_name ||
              connectorUi?.connector_id ||
              "Setup";
            appendLog({
              id: crypto.randomUUID(),
              category: "tool",
              title,
              detail: connectorUi
                ? `${connectorUi.display_name || connectorUi.connector_id} setup opened`
                : "Setup UI opened",
              raw: payload,
              ts: Date.now(),
            });
            if (connectorUi && connectorSetupNeedsUserAction(connectorUi)) {
              upsertAssistantUi(assistantMessage.id, connectorUi);
              setActiveConnectorSetup({ assistantId: assistantMessage.id, ui: connectorUi });
            } else if (connectorUi) {
              setActiveConnectorSetup((current) =>
                current?.ui.connector_id === connectorUi.connector_id ? null : current
              );
            }
          }

          if (eventType === "error") {
            const detail = asString(payload.message, "Unknown error");
            appendLog({
              id: crypto.randomUUID(),
              category: "error",
              title: "Error",
              detail,
              ts: Date.now(),
            });
            if (isSearchToolMisconfigured(String(detail))) {
              setShowSearchConfig(true);
            }
          }
          if (eventType === "done") {
            receivedDone = true;
            setAssistantPresentation(assistantMessage.id, payload.presentation);
            if (typeof payload.session_id === "string" && payload.session_id) {
              saveSessionId(payload.session_id);
              setSessionId(payload.session_id);
            }
            setSending(false);
            setCurrentPhase(null);
            setStreamMeter((current) => ({ ...current, active: false }));
          }
          if (eventType === "failed") {
            receivedDone = true;
            markTurnIncomplete(userMessage.id, assistantMessage.id);
            if (typeof payload.session_id === "string" && payload.session_id) {
              saveSessionId(payload.session_id);
              setSessionId(payload.session_id);
            }
            setSending(false);
            setCurrentPhase(null);
            setStreamMeter((current) => ({ ...current, active: false }));
          }
        }
      }
    } catch (err: unknown) {
      if (receivedDone) {
        return;
      }
      appendLog({
        id: crypto.randomUUID(),
        category: "error",
        title: "Chat error",
        detail: err instanceof Error ? err.message : "Unknown error",
        ts: Date.now(),
      });
    } finally {
      setSending(false);
      setCurrentPhase(null);
      setStreamMeter((current) => ({ ...current, active: false }));
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const userHistory = messages
      .filter((msg) => msg.role === "user" && msg.content.trim())
      .map((msg) => msg.content);

    if (e.key === "ArrowUp" && userHistory.length > 0) {
      e.preventDefault();
      let nextIndex = historyIndex;
      if (nextIndex === null) {
        setHistoryDraft(input);
        nextIndex = userHistory.length - 1;
      } else {
        nextIndex = Math.max(0, nextIndex - 1);
      }
      setHistoryIndex(nextIndex);
      setInput(userHistory[nextIndex] ?? "");
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (el) {
          const pos = el.value.length;
          el.setSelectionRange(pos, pos);
        }
      });
      return;
    }

    if (e.key === "ArrowDown" && historyIndex !== null) {
      e.preventDefault();
      if (historyIndex < userHistory.length - 1) {
        const nextIndex = historyIndex + 1;
        setHistoryIndex(nextIndex);
        setInput(userHistory[nextIndex] ?? "");
      } else {
        setHistoryIndex(null);
        setInput(historyDraft);
      }
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (el) {
          const pos = el.value.length;
          el.setSelectionRange(pos, pos);
        }
      });
      return;
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (ready === false) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Card className="max-w-md text-center">
          <h1 className="font-display text-2xl">Initialization Required</h1>
          <p className="mt-3 text-sm text-[var(--ink-soft)]">
            Complete the initialization ritual before entering the main chat.
          </p>
          <a
            className="mt-6 inline-flex rounded-full bg-[var(--foreground)] px-6 py-3 text-sm font-semibold text-white"
            href="/init"
          >
            Go to Initialization
          </a>
        </Card>
      </div>
    );
  }

  if (ready === null) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner label="Loading status..." />
      </div>
    );
  }

  return (
    <div className="app-shell h-[calc(100vh-3.5rem)] overflow-hidden lg:h-screen">
      {activeConnectorSetup ? (
        <ConnectorSetupModal
          setup={activeConnectorSetup}
          busy={connectorActionBusy}
          onAction={runConnectorSetupAction}
          onRefresh={refreshConnectorSetupStatus}
          onClose={() => setActiveConnectorSetup(null)}
        />
      ) : null}
      <div className="mx-auto flex h-full max-w-[1600px]">
        <section className="flex min-w-0 flex-1 flex-col bg-white">
          <header className="flex h-16 items-center justify-between gap-4 border-b border-[var(--outline)] px-4 sm:px-6">
            <div className="flex min-w-0 items-center gap-3">
              {agentStatus.portrait_url ? (
                <Image src={agentStatus.portrait_url} alt="" width={40} height={40} unoptimized className="h-10 w-10 rounded-md object-cover" />
              ) : (
                <div className="flex h-10 w-10 items-center justify-center rounded-md bg-[var(--foreground)] font-display text-white">
                  {(agentStatus.agent_name || "H").slice(0, 1)}
                </div>
              )}
              <div className="min-w-0">
                <h1 className="truncate text-sm font-semibold">{agentStatus.agent_name || "Hexis"}</h1>
                <p className="truncate text-xs text-[var(--ink-soft)]">
                  {sending ? phaseDescription(currentPhase || "") : agentStatus.mood || "Ready"}
                  {agentStatus.valence != null ? ` · valence ${agentStatus.valence >= 0 ? "+" : ""}${agentStatus.valence.toFixed(2)}` : ""}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                aria-label="Start new conversation"
                title="Start new conversation"
                disabled={sending || sessionActionBusy !== null}
                onClick={handleStartNewChat}
                className="flex h-9 w-9 items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--surface-strong)] hover:text-[var(--foreground)] disabled:opacity-35"
              >
                <Plus size={17} />
              </button>
              <button
                type="button"
                aria-label="Clear active conversation context"
                title="Clear active conversation context"
                disabled={sending || sessionActionBusy !== null || (!sessionId && messages.length === 0)}
                onClick={handleClearChatContext}
                className="flex h-9 w-9 items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--surface-strong)] hover:text-[var(--foreground)] disabled:opacity-35"
              >
                <Trash2 size={16} />
              </button>
              <details className="relative">
                <summary className="flex h-9 cursor-pointer list-none items-center gap-2 rounded-md px-3 text-xs font-medium text-[var(--ink-soft)] hover:bg-[var(--surface-strong)]">
                  <Settings2 size={16} /> Options
                </summary>
                <div className="absolute right-0 top-11 z-30 w-64 rounded-lg border border-[var(--outline)] bg-white p-4 shadow-lg">
                  <p className="text-xs font-semibold uppercase text-[var(--ink-soft)]">Prompt modules</p>
                  <div className="mt-3 space-y-3">
                    {promptAddendaOptions.map((option) => (
                      <label key={option.id} className="flex items-center gap-3 text-sm">
                        <input
                          type="checkbox"
                          className="h-4 w-4 accent-[var(--teal)]"
                          checked={promptAddenda.includes(option.id)}
                          onChange={() => setPromptAddenda((current) => current.includes(option.id) ? current.filter((item) => item !== option.id) : [...current, option.id])}
                        />
                        {option.label}
                      </label>
                    ))}
                  </div>
                </div>
              </details>
              <button
                type="button"
                aria-label={showInbox ? "Hide inbox" : `Show inbox${inboxBadgeCount > 0 ? ` (${inboxBadgeCount} waiting)` : ""}`}
                title={showInbox ? "Hide inbox" : "Messages and requests from the agent"}
                onClick={() => setShowInbox((value) => !value)}
                className={`relative flex h-9 w-9 items-center justify-center rounded-md ${showInbox ? "bg-[var(--surface-strong)] text-[var(--foreground)]" : "text-[var(--ink-soft)] hover:bg-[var(--surface-strong)]"}`}
              >
                <Inbox size={17} />
                {inboxBadgeCount > 0 ? (
                  <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-[var(--teal)] px-1 text-[10px] font-semibold text-white">
                    {inboxBadgeCount > 9 ? "9+" : inboxBadgeCount}
                  </span>
                ) : null}
              </button>
              <button
                type="button"
                aria-label={showInspector ? "Hide activity" : "Show activity"}
                title={showInspector ? "Hide activity" : "Show activity"}
                onClick={() => setShowInspector((value) => !value)}
                className={`flex h-9 w-9 items-center justify-center rounded-md ${showInspector ? "bg-[var(--surface-strong)] text-[var(--foreground)]" : "text-[var(--ink-soft)] hover:bg-[var(--surface-strong)]"}`}
              >
                {showInspector ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </div>
          </header>

          {showSearchConfig ? (
            <div className="border-b border-amber-200 bg-amber-50 px-4 py-3 sm:px-6">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <span className="text-sm font-medium text-amber-900">Configure web search</span>
                <select
                  value={searchConfigProvider}
                  onChange={(event) => {
                    setSearchConfigProvider(event.target.value as SearchConfigProvider);
                    setSearchConfigValue("");
                    setSearchConfigError(null);
                  }}
                  className="rounded-md border border-amber-200 bg-white px-3 py-2 text-sm"
                >
                  {SEARCH_CONFIG_PROVIDERS.map((provider) => (
                    <option key={provider.id} value={provider.id}>
                      {provider.label}
                    </option>
                  ))}
                </select>
                <input
                  value={searchConfigValue}
                  onChange={(event) => setSearchConfigValue(event.target.value)}
                  placeholder={searchConfigPlaceholder(searchConfigProvider)}
                  disabled={searchConfigProvider === "auto"}
                  className="min-w-0 flex-1 rounded-md border border-amber-200 bg-white px-3 py-2 text-sm disabled:bg-amber-100 disabled:text-amber-800"
                />
                <button onClick={handleConfigureSearchTool} disabled={searchConfigSaving} className="rounded-md bg-[var(--foreground)] px-3 py-2 text-xs font-semibold text-white disabled:opacity-50">{searchConfigSaving ? "Saving" : "Enable"}</button>
                <button onClick={() => setShowSearchConfig(false)} className="px-2 py-2 text-xs text-amber-800">Dismiss</button>
              </div>
              {searchConfigError ? <p className="mt-1 text-xs text-red-700">{searchConfigError}</p> : null}
            </div>
          ) : null}
          {searchConfigNotice ? <div className="border-b border-emerald-200 bg-emerald-50 px-6 py-2 text-xs text-emerald-700">{searchConfigNotice}</div> : null}
          {sessionNotice ? <div className="border-b border-[var(--outline)] bg-[#f5f7f5] px-6 py-2 text-xs text-[var(--ink-soft)]">{sessionNotice}</div> : null}

          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 sm:px-8">
            <div className="mx-auto max-w-3xl space-y-6">
              {messages.length === 0 ? (
                <div className="flex min-h-80 flex-col items-center justify-center text-center">
                  {agentStatus.portrait_url ? <Image src={agentStatus.portrait_url} alt="" width={80} height={80} unoptimized className="h-20 w-20 rounded-lg object-cover" /> : <BrainCircuit size={38} className="text-[var(--teal)]" />}
                  <h2 className="mt-4 font-display text-2xl">Conversation with {agentStatus.agent_name || "Hexis"}</h2>
                  <p className="mt-2 text-sm text-[var(--ink-soft)]">What is on your mind?</p>
                </div>
              ) : null}
              {messages.map((message) => (
                <div key={message.id} className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                  {message.role === "assistant" ? (
                    agentStatus.portrait_url ? <Image src={agentStatus.portrait_url} alt="" width={32} height={32} unoptimized className="mt-1 h-8 w-8 flex-none rounded-md object-cover" /> : <div className="mt-1 flex h-8 w-8 flex-none items-center justify-center rounded-md bg-[var(--surface-strong)] text-xs font-semibold">H</div>
                  ) : null}
                  <div className={`max-w-[85%] text-sm leading-6 ${message.role === "user" ? "rounded-lg bg-[var(--foreground)] px-4 py-3 text-white" : "min-w-0 flex-1 py-1 text-[var(--foreground)]"}`}>
                    {message.role === "assistant" ? (
                      <div className="space-y-3">
                        {message.presentation ? (
                          <MessagePresentationView presentation={message.presentation} />
                        ) : message.content ? (
                          <MessagePresentationView presentation={{ tone: "neutral", blocks: [{ type: "text", text: message.content }] }} />
                        ) : message.incomplete ? (
                          <p className="text-sm text-red-700">Response incomplete — try again.</p>
                        ) : message.ui?.length ? null : (
                          <Spinner label="Thinking..." />
                        )}
                        {message.incomplete && message.content ? (
                          <p className="text-xs text-red-700">
                            Response incomplete — not added to conversation history.
                          </p>
                        ) : null}
                        {message.ui?.map((ui) => (
                          <ConnectorSetupCard
                            key={`${uiArtifactKey(ui)}:${ui.status || ""}`}
                            ui={ui}
                            assistantId={message.id}
                            busy={connectorActionBusy}
                            onAction={runConnectorSetupAction}
                            onRefresh={refreshConnectorSetupStatus}
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {message.attachments?.length ? (
                          <div className="grid gap-2">
                            {message.attachments.map((attachment) => (
                              <Image
                                key={attachment.id}
                                src={attachment.dataUrl}
                                alt={attachment.name}
                                width={360}
                                height={240}
                                unoptimized
                                className="max-h-72 w-full rounded-md object-contain"
                              />
                            ))}
                          </div>
                        ) : null}
                        {message.content ? <p className="whitespace-pre-wrap">{message.content}</p> : null}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="border-t border-[var(--outline)] bg-white px-4 py-3 sm:px-6">
            {!showInbox && inboxBadgeCount > 0 ? (
              <div className="mx-auto mb-2 max-w-3xl">
                <button
                  type="button"
                  onClick={() => setShowInbox(true)}
                  className="flex w-full items-center gap-2 rounded-md border border-[var(--teal)]/40 bg-[var(--teal)]/5 px-3 py-2 text-left text-xs hover:bg-[var(--teal)]/10"
                >
                  <Inbox size={14} className="flex-none text-[var(--teal)]" />
                  <span>
                    {agentStatus.agent_name || "The agent"} has{" "}
                    {inbox.unread > 0 ? `${inbox.unread} unread message${inbox.unread === 1 ? "" : "s"}` : ""}
                    {inbox.unread > 0 && inbox.pending_requests.length > 0 ? " and " : ""}
                    {inbox.pending_requests.length > 0
                      ? `${inbox.pending_requests.length} request${inbox.pending_requests.length === 1 ? "" : "s"} awaiting your decision`
                      : ""}
                    {" — open the inbox."}
                  </span>
                </button>
              </div>
            ) : null}
            {fileAttachments.length > 0 ? (
              <div className="mx-auto mb-2 flex max-w-3xl flex-wrap gap-2">
                {fileAttachments.map((attachment) => (
                  <span key={attachment.id} className="flex items-center gap-2 rounded-md border border-[var(--outline)] bg-[#f5f7f5] px-2 py-1 text-xs">
                    {isImageAttachmentFile(attachment.file) ? (
                      <ImageIcon size={13} className="flex-none text-[var(--teal)]" />
                    ) : (
                      <Paperclip size={13} className="flex-none text-[var(--teal)]" />
                    )}
                    <span className="max-w-56 truncate font-medium">{attachment.name}</span>
                    <span className="text-[var(--ink-soft)]">{formatBytes(attachment.size)}</span>
                    <button
                      type="button"
                      aria-label={
                        attachment.sensitivity === "private"
                          ? `Make file ${attachment.name} shareable`
                          : `Mark file ${attachment.name} private`
                      }
                      title={
                        attachment.sensitivity === "private"
                          ? "Private: kept out of group conversations and exports. Click to make shareable."
                          : "Shareable. Click to keep out of group conversations and exports."
                      }
                      onClick={() => toggleFileAttachmentPrivacy(attachment.id)}
                      className={`flex flex-none items-center gap-1 rounded p-0.5 ${
                        attachment.sensitivity === "private"
                          ? "text-[var(--teal)]"
                          : "text-[var(--ink-soft)] hover:bg-[var(--outline)] hover:text-[var(--foreground)]"
                      }`}
                    >
                      {attachment.sensitivity === "private" ? <Lock size={12} /> : <LockOpen size={12} />}
                      {attachment.sensitivity === "private" ? <span className="font-medium">Private</span> : null}
                    </button>
                    <button
                      type="button"
                      aria-label={`Remove file ${attachment.name}`}
                      title="Remove"
                      onClick={() => removeFileAttachment(attachment.id)}
                      className="flex-none rounded p-0.5 text-[var(--ink-soft)] hover:bg-[var(--outline)] hover:text-[var(--foreground)]"
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            ) : null}
            {attachments.length > 0 ? (
              <div className="mx-auto mb-2 flex max-w-3xl flex-wrap gap-2">
                {attachments.map((attachment) => (
                  <span key={attachment.id} className="flex items-center gap-2 rounded-md border border-[var(--outline)] bg-[#f5f7f5] px-2 py-1 text-xs">
                    <FileText size={13} className="flex-none text-[var(--teal)]" />
                    <span className="max-w-56 truncate font-medium">{attachment.title}</span>
                    <span className="text-[var(--ink-soft)]">{attachment.wordCount.toLocaleString()} words</span>
                    <button
                      type="button"
                      aria-label={
                        attachment.sensitivity === "private"
                          ? `Make attachment ${attachment.title} shareable`
                          : `Mark attachment ${attachment.title} private`
                      }
                      title={
                        attachment.sensitivity === "private"
                          ? "Private: kept out of group conversations and exports. Click to make shareable."
                          : "Shareable. Click to keep out of group conversations and exports."
                      }
                      onClick={() => toggleAttachmentPrivacy(attachment.id)}
                      className={`flex flex-none items-center gap-1 rounded p-0.5 ${
                        attachment.sensitivity === "private"
                          ? "text-[var(--teal)]"
                          : "text-[var(--ink-soft)] hover:bg-[var(--outline)] hover:text-[var(--foreground)]"
                      }`}
                    >
                      {attachment.sensitivity === "private" ? <Lock size={12} /> : <LockOpen size={12} />}
                      {attachment.sensitivity === "private" ? <span className="font-medium">Private</span> : null}
                    </button>
                    <button
                      type="button"
                      aria-label={`Remove attachment ${attachment.title}`}
                      title="Remove"
                      onClick={() => removeAttachment(attachment.id)}
                      className="flex-none rounded p-0.5 text-[var(--ink-soft)] hover:bg-[var(--outline)] hover:text-[var(--foreground)]"
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            ) : null}
            <div
              className="mx-auto flex max-w-3xl items-end gap-2 rounded-lg border border-[var(--outline)] bg-white p-2 focus-within:border-[var(--teal)] focus-within:ring-2 focus-within:ring-[var(--teal)]/10"
              onDrop={handleComposerDrop}
              onDragOver={(event) => { if (event.dataTransfer?.types?.includes("Files")) event.preventDefault(); }}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                aria-hidden="true"
                tabIndex={-1}
                onChange={(event) => { addFiles(event.target.files, "picker"); event.target.value = ""; }}
              />
              <button
                type="button"
                aria-label="Attach files"
                title="Attach files (or drop them anywhere on the composer)"
                onClick={() => fileInputRef.current?.click()}
                className="flex h-10 w-10 flex-none items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--outline)] hover:text-[var(--foreground)]"
              >
                <Paperclip size={17} />
              </button>
              <textarea
                ref={textareaRef}
                aria-label={`Message ${agentStatus.agent_name || "Hexis"}`}
                className="max-h-36 min-h-10 flex-1 resize-none border-0 bg-transparent px-2 py-2 text-sm outline-none"
                placeholder={`Message ${agentStatus.agent_name || "Hexis"}`}
                value={input}
                onChange={(event) => { if (historyIndex !== null) setHistoryIndex(null); setInput(event.target.value); }}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
                rows={1}
              />
              <button type="button" aria-label="Send message" title="Send" onClick={handleSend} disabled={sending || (!input.trim() && attachments.length === 0 && fileAttachments.length === 0)} className="flex h-10 w-10 flex-none items-center justify-center rounded-md bg-[var(--foreground)] text-white hover:bg-[var(--teal)] disabled:opacity-35">
                <Send size={17} />
              </button>
            </div>
          </div>
        </section>

        {showInbox ? (
          <aside className="fixed inset-y-14 right-0 z-20 flex w-full flex-col border-l border-[var(--outline)] bg-[#f8faf8] sm:w-[390px] lg:static lg:inset-auto lg:w-[380px]">
            <div className="flex h-16 items-center justify-between border-b border-[var(--outline)] px-4">
              <div>
                <h2 className="text-sm font-semibold">Inbox</h2>
                <p className="text-xs text-[var(--ink-soft)]">
                  {agentStatus.agent_name || "The agent"}&apos;s always-available line to you
                </p>
              </div>
              <button type="button" title="Close inbox" aria-label="Close inbox" onClick={() => setShowInbox(false)} className="flex h-8 w-8 items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--surface-strong)]"><X size={17} /></button>
            </div>
            <div className="flex-1 space-y-3 overflow-y-auto p-4">
              {decideNotice ? (
                <p className="rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-xs">{decideNotice}</p>
              ) : null}

              {inbox.pending_requests.length > 0 ? (
                <div>
                  <h3 className="text-xs font-semibold uppercase text-[var(--ink-soft)]">Awaiting your decision</h3>
                  <div className="mt-2 space-y-3">
                    {inbox.pending_requests.map((request) => (
                      <div key={request.id} className="rounded-lg border border-[var(--teal)]/40 bg-white p-3">
                        <div className="flex items-center justify-between gap-2">
                          <Badge variant="accent">{request.kind.replace("_", " ")}</Badge>
                          <span className="text-[10px] text-[var(--ink-soft)]">{String(request.requested_at).slice(0, 16).replace("T", " ")}</span>
                        </div>
                        {request.target_key ? (
                          <p className="mt-2 font-mono text-xs">
                            {request.target_key} = {JSON.stringify(request.requested_value)}
                          </p>
                        ) : request.requested_value != null ? (
                          <p className="mt-2 font-mono text-xs">requested: {JSON.stringify(request.requested_value)}</p>
                        ) : null}
                        <p className="mt-2 text-sm leading-6">{request.rationale}</p>
                        {request.duration ? (
                          <p className="mt-1 text-xs text-[var(--ink-soft)]">For: {request.duration}</p>
                        ) : null}
                        <input
                          type="text"
                          placeholder="Optional note she will read with the decision"
                          value={decideNotes[request.id] || ""}
                          onChange={(event) => setDecideNotes((current) => ({ ...current, [request.id]: event.target.value }))}
                          className="mt-3 w-full rounded-md border border-[var(--outline)] px-2 py-1.5 text-xs outline-none focus:border-[var(--teal)]"
                        />
                        <div className="mt-2 flex gap-2">
                          <button
                            type="button"
                            disabled={decideBusy === request.id}
                            onClick={() => decideRequest(request, "granted")}
                            className="flex flex-1 items-center justify-center gap-1.5 rounded-md bg-[var(--teal)] px-3 py-1.5 text-xs font-semibold text-white hover:opacity-90 disabled:opacity-40"
                          >
                            <Check size={13} /> Approve
                          </button>
                          <button
                            type="button"
                            disabled={decideBusy === request.id}
                            onClick={() => decideRequest(request, "denied")}
                            className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-[var(--outline)] px-3 py-1.5 text-xs font-semibold text-[var(--foreground)] hover:bg-[var(--surface-strong)] disabled:opacity-40"
                          >
                            <X size={13} /> Deny
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              <div>
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase text-[var(--ink-soft)]">Messages</h3>
                  {inbox.messages.some((m) => !m.read_at) ? (
                    <button
                      type="button"
                      onClick={() => ackInboxMessages(inbox.messages.filter((m) => !m.read_at).map((m) => m.id))}
                      className="text-[10px] font-medium text-[var(--ink-soft)] underline-offset-2 hover:underline"
                    >
                      Mark all read
                    </button>
                  ) : null}
                </div>
                {inbox.messages.length === 0 ? (
                  <p className="mt-2 text-xs text-[var(--ink-soft)]">
                    Nothing yet. When {agentStatus.agent_name || "the agent"} reaches out on her own — from a heartbeat, a reminder, a request — it lands here.
                  </p>
                ) : (
                  <div className="mt-2 space-y-3">
                    {inbox.messages.map((message) => {
                      const pendingRequest = message.payload?.delivery?.request_id
                        ? inbox.pending_requests.find((r) => r.id === message.payload?.delivery?.request_id)
                        : undefined;
                      return (
                      <div key={message.id} className={`rounded-lg border p-3 ${message.read_at ? "border-[var(--outline)] bg-white" : "border-[var(--teal)]/50 bg-[var(--teal)]/5"}`}>
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-semibold">{agentStatus.agent_name || "Agent"}</span>
                          <span className="flex items-center gap-2 text-[10px] text-[var(--ink-soft)]">
                            {message.intent ? <Badge variant="muted">{message.intent}</Badge> : null}
                            {String(message.delivered_at).slice(0, 16).replace("T", " ")}
                          </span>
                        </div>
                        <p className="mt-2 whitespace-pre-wrap text-sm leading-6">{message.message}</p>
                        <div className="mt-2 flex flex-wrap items-center gap-2">
                          {message.intent === "document_fade" ? (
                            <>
                              <button
                                type="button"
                                disabled={decideBusy === message.id}
                                onClick={() => decideFade(message, "keep")}
                                className="rounded-md bg-[var(--teal)] px-2.5 py-1 text-xs font-semibold text-white hover:opacity-90 disabled:opacity-40"
                              >
                                Keep it
                              </button>
                              <button
                                type="button"
                                disabled={decideBusy === message.id}
                                onClick={() => decideFade(message, "approve")}
                                className="rounded-md border border-[var(--outline)] px-2.5 py-1 text-xs font-medium text-[var(--foreground)] hover:bg-[var(--surface-strong)] disabled:opacity-40"
                              >
                                Let it fade
                              </button>
                            </>
                          ) : null}
                          {pendingRequest ? (
                            <>
                              <button
                                type="button"
                                disabled={decideBusy === pendingRequest.id}
                                onClick={() => decideRequest(pendingRequest, "granted")}
                                className="flex items-center gap-1 rounded-md bg-[var(--teal)] px-2.5 py-1 text-xs font-semibold text-white hover:opacity-90 disabled:opacity-40"
                              >
                                <Check size={12} /> Grant
                              </button>
                              <button
                                type="button"
                                disabled={decideBusy === pendingRequest.id}
                                onClick={() => decideRequest(pendingRequest, "denied")}
                                className="flex items-center gap-1 rounded-md border border-[var(--outline)] px-2.5 py-1 text-xs font-medium text-[var(--foreground)] hover:bg-[var(--surface-strong)] disabled:opacity-40"
                              >
                                <X size={12} /> Deny
                              </button>
                            </>
                          ) : null}
                          <button
                            type="button"
                            disabled={replyBusy}
                            onClick={() => startInboxReply(message)}
                            className="rounded-md border border-[var(--outline)] px-2.5 py-1 text-xs font-medium text-[var(--foreground)] hover:bg-[var(--surface-strong)]"
                          >
                            Reply
                          </button>
                          {!message.read_at ? (
                            <button
                              type="button"
                              onClick={() => ackInboxMessages([message.id])}
                              className="rounded-md border border-[var(--outline)] px-2.5 py-1 text-xs font-medium text-[var(--ink-soft)] hover:bg-[var(--surface-strong)]"
                            >
                              Acknowledge
                            </button>
                          ) : null}
                          <button
                            type="button"
                            title="Delete message"
                            aria-label="Delete message"
                            disabled={decideBusy === message.id}
                            onClick={() => deleteInboxMessage(message)}
                            className="ml-auto flex h-6 w-6 items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--surface-strong)] disabled:opacity-40"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                        {replyingTo === message.id ? (
                          <form
                            className="mt-3 rounded-md border border-[var(--outline)] bg-white p-2"
                            onSubmit={(event) => {
                              event.preventDefault();
                              void submitInboxReply(message);
                            }}
                          >
                            <label
                              htmlFor={`inbox-reply-${message.id}`}
                              className="text-xs font-semibold text-[var(--foreground)]"
                            >
                              Reply to {agentStatus.agent_name || "the agent"}
                            </label>
                            <textarea
                              id={`inbox-reply-${message.id}`}
                              autoFocus
                              rows={3}
                              value={replyDraft}
                              disabled={replyBusy}
                              onChange={(event) => setReplyDraft(event.target.value)}
                              placeholder="This will be processed at the next heartbeat"
                              className="mt-2 w-full resize-y rounded-md border border-[var(--outline)] px-2 py-1.5 text-sm outline-none focus:border-[var(--teal)] disabled:opacity-60"
                            />
                            <div className="mt-2 flex justify-end gap-2">
                              <button
                                type="button"
                                disabled={replyBusy}
                                onClick={cancelInboxReply}
                                className="rounded-md border border-[var(--outline)] px-2.5 py-1 text-xs font-medium text-[var(--ink-soft)] hover:bg-[var(--surface-strong)] disabled:opacity-40"
                              >
                                Cancel
                              </button>
                              <button
                                type="submit"
                                disabled={replyBusy || !replyDraft.trim()}
                                className="flex items-center gap-1.5 rounded-md bg-[var(--foreground)] px-2.5 py-1 text-xs font-semibold text-white hover:bg-[var(--teal)] disabled:opacity-40"
                              >
                                {replyBusy ? <Spinner className="scale-75" /> : <Send size={12} />}
                                Send reply
                              </button>
                            </div>
                          </form>
                        ) : null}
                      </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </aside>
        ) : showInspector ? (
          <ActivityInspector
            events={events}
            messages={messages}
            agentStatus={agentStatus}
            currentPhase={currentPhase}
            sending={sending}
            streamMeter={streamMeter}
            expandedPanes={expandedActivityPanes}
            onTogglePane={toggleActivityPane}
            onClear={() => setEvents([])}
            onClose={() => setShowInspector(false)}
          />
        ) : null}
      </div>
    </div>
  );
}

function ConnectorSetupModal({
  setup,
  busy,
  onAction,
  onRefresh,
  onClose,
}: {
  setup: { assistantId: string; ui: ConnectorSetupUi };
  busy: string | null;
  onAction: (
    assistantId: string,
    currentUi: ConnectorSetupUi,
    action: string,
    argumentsPayload: Record<string, unknown>
  ) => Promise<IntegrationActionResult>;
  onRefresh: (
    assistantId: string,
    currentUi: ConnectorSetupUi
  ) => Promise<IntegrationActionResult>;
  onClose: () => void;
}) {
  const connectorName = setup.ui.display_name || setup.ui.connector_id;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4 py-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="connector-setup-title"
    >
      <div className="w-full max-w-2xl border border-[var(--outline)] bg-white shadow-2xl">
        <div className="flex items-center justify-between gap-3 border-b border-[var(--outline)] px-4 py-3">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-8 w-8 flex-none items-center justify-center rounded-md bg-[var(--teal)]/10 text-[var(--teal)]">
              <Mail size={17} />
            </span>
            <div className="min-w-0">
              <h2 id="connector-setup-title" className="truncate text-sm font-semibold">
                Connect {connectorName}
              </h2>
              <p className="truncate text-xs text-[var(--ink-soft)]">
                Complete setup without leaving this conversation.
              </p>
            </div>
          </div>
          <button
            type="button"
            aria-label="Close connector setup"
            title="Close"
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--surface-strong)] hover:text-[var(--foreground)]"
          >
            <X size={16} />
          </button>
        </div>
        <div className="max-h-[min(720px,calc(100vh-8rem))] overflow-y-auto p-4">
          <ConnectorSetupCard
            key={`${uiArtifactKey(setup.ui)}:${setup.ui.status || ""}`}
            ui={setup.ui}
            assistantId={setup.assistantId}
            busy={busy}
            onAction={onAction}
            onRefresh={onRefresh}
          />
        </div>
      </div>
    </div>
  );
}

function ConnectorSetupCard({
  ui,
  assistantId,
  busy,
  onAction,
  onRefresh,
}: {
  ui: ConnectorSetupUi;
  assistantId: string;
  busy: string | null;
  onAction: (
    assistantId: string,
    currentUi: ConnectorSetupUi,
    action: string,
    argumentsPayload: Record<string, unknown>
  ) => Promise<IntegrationActionResult>;
  onRefresh: (
    assistantId: string,
    currentUi: ConnectorSetupUi
  ) => Promise<IntegrationActionResult>;
}) {
  const [clientSecretPath, setClientSecretPath] = useState("");
  const [useEnvSecret, setUseEnvSecret] = useState(false);
  const [authorizationResponse, setAuthorizationResponse] = useState("");
  const [selectedCapabilityOption, setSelectedCapabilityOption] =
    useState<ConnectorSetupCapabilityOption | null>(null);
  const [selectedMemoryPolicy, setSelectedMemoryPolicy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAdvancedCredentialSetup, setShowAdvancedCredentialSetup] = useState(true);
  const clientSecretFileRef = useRef<HTMLInputElement>(null);
  const refreshInFlightRef = useRef(false);

  const connectorName = ui.display_name || ui.connector_id;
  const capabilityOptions = ui.capability_options || [];
  const memoryOptions = ui.memory_options || [];
  const autonomyOptions = ui.autonomy_options || [];
  const connectedAccounts = ui.connected_accounts || [];
  const connected = ui.status === "connected" || connectedAccounts.length > 0;
  const startBusy = busy === `${assistantId}:${ui.connector_id}:connect_gmail`;
  const completeBusy = busy === `${assistantId}:${ui.connector_id}:complete_gmail`;
  const needsCapabilityChoice =
    ui.status === "needs_capability_choice" && capabilityOptions.length > 0;
  const capabilities =
    selectedCapabilityOption?.capabilities ||
    (ui.capabilities?.length ? ui.capabilities : needsCapabilityChoice ? [] : ["read", "search"]);
  const memoryPolicy = selectedMemoryPolicy || ui.memory_policy;
  const needsMemoryChoice =
    ui.status === "needs_memory_choice" || (Boolean(selectedCapabilityOption) && !selectedMemoryPolicy);
  const needsAutonomyChoice =
    ui.status === "needs_autonomy_choice" || Boolean(selectedMemoryPolicy);
  const canStartOAuth =
    ui.connector_id === "gmail" &&
    !connected &&
    !ui.authorization_url &&
    !needsCapabilityChoice &&
    !needsMemoryChoice &&
    !needsAutonomyChoice;
  const canCompleteOAuth = ui.connector_id === "gmail" && !connected && Boolean(ui.authorization_url || ui.attempt_id);
  const requiresClientSecret = canStartOAuth && !ui.client_secret_saved;
  const credentialModes = ui.credential_step?.modes || [];
  const hostedAvailable =
    ui.hexis_oauth_client_available ||
    credentialModes.some((mode) => mode.id === "hosted_oauth" && mode.available);
  const envAvailable =
    ui.env_client_secret_available ||
    credentialModes.some((mode) => mode.id === "configured_env" && mode.available);
  const needsLocalGoogleSetup = requiresClientSecret && !hostedAvailable;
  const showLocalGoogleSetup = needsLocalGoogleSetup && showAdvancedCredentialSetup;
  const setupSteps = ui.setup_steps?.length ? ui.setup_steps : GMAIL_SETUP_STEPS;

  useEffect(() => {
    if (ui.connector_id !== "gmail" || !canCompleteOAuth || connected) return;
    let stopped = false;
    const refresh = async () => {
      if (stopped || refreshInFlightRef.current) return;
      refreshInFlightRef.current = true;
      try {
        const payload = await onRefresh(assistantId, ui);
        const nextUi = connectorSetupUiFromPayload(payload);
        if (nextUi?.status === "connected") {
          setNotice("Gmail connected. You can return to the conversation.");
        }
      } finally {
        refreshInFlightRef.current = false;
      }
    };
    const timer = window.setInterval(() => {
      void refresh();
    }, 3000);
    void refresh();
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [assistantId, canCompleteOAuth, connected, ui.attempt_id, ui.connector_id, ui.id, ui.status]);

  const runSaveClientSecret = async (file: File | null | undefined) => {
    if (!file) return;
    setNotice(null);
    setError(null);
    try {
      const clientSecretJson = await fileToJsonRecord(file);
      const payload = await onAction(assistantId, ui, "save_gmail_client_secret", {
        client_secret_json: clientSecretJson,
      });
      if (payload.error || payload.detail || payload.success === false) {
        setError(integrationActionError(payload, 400));
      } else {
        setClientSecretPath("");
        setNotice(integrationActionNotice(payload, 200));
      }
    } catch (uploadError: unknown) {
      setError(uploadError instanceof Error ? uploadError.message : "Could not read Google setup file.");
    } finally {
      if (clientSecretFileRef.current) clientSecretFileRef.current.value = "";
    }
  };

  const runStart = async (
    selectedCapabilities = capabilities,
    selectedMemoryPolicy = memoryPolicy,
    heartbeatDigestEnabled = ui.heartbeat_digest_enabled || false,
    options: { allowMissingClientSecret?: boolean } = {}
  ) => {
    setNotice(null);
    setError(null);
    const path = clientSecretPath.trim();
    if (
      !options.allowMissingClientSecret &&
      requiresClientSecret &&
      !hostedAvailable &&
      !path &&
      !useEnvSecret
    ) {
      setError(
        "This local Hexis build needs Gmail sign-in setup first. Follow the guide in this panel, upload the Google setup file, then start Google sign-in."
      );
      return;
    }
    const payload = await onAction(assistantId, ui, "connect_gmail", {
      capabilities: selectedCapabilities,
      memory_policy: selectedMemoryPolicy || undefined,
      heartbeat_digest_enabled: heartbeatDigestEnabled,
      client_secret_path: path || undefined,
      use_hexis_oauth_client: true,
      use_env_client_secret: useEnvSecret,
    });
    if (payload.error || payload.detail || payload.success === false) {
      setError(integrationActionError(payload, 400));
    } else {
      setNotice(integrationActionNotice(payload, 200));
    }
  };

  const runComplete = async () => {
    setNotice(null);
    setError(null);
    const response = authorizationResponse.trim();
    if (!response) {
      setError("Paste the Google callback URL or authorization code.");
      return;
    }
    const payload = await onAction(assistantId, ui, "complete_gmail", {
      attempt_id: ui.attempt_id || undefined,
      authorization_response: response,
    });
    if (payload.error || payload.detail || payload.success === false) {
      setError(integrationActionError(payload, 400));
    } else {
      setNotice(integrationActionNotice(payload, 200));
    }
  };

  return (
    <div className="max-w-2xl rounded-lg border border-[var(--teal)]/35 bg-[#f8fbfa] p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <span className="mt-0.5 flex h-8 w-8 flex-none items-center justify-center rounded-md bg-[var(--teal)]/10 text-[var(--teal)]">
            <Mail size={17} />
          </span>
          <div className="min-w-0">
            <h3 className="text-sm font-semibold">{ui.title || `Connect ${connectorName}`}</h3>
            <p className="mt-1 text-xs leading-5 text-[var(--ink-soft)]">
              {ui.summary || `Set up ${connectorName} access.`}
            </p>
          </div>
        </div>
        <Badge variant={connected ? "success" : ui.status === "needs_client_secret" ? "warning" : "muted"}>
          {connectorSetupStatusLabel(ui)}
        </Badge>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {capabilities.map((capability) => (
          <Badge key={capability} variant="muted">
            {humanizeCapability(capability)}
          </Badge>
        ))}
        {memoryPolicy ? (
          <Badge variant="muted">
            memory: {humanizeCapability(memoryPolicy)}
          </Badge>
        ) : null}
      </div>

      {needsCapabilityChoice ? (
        <div className="mt-4 space-y-3">
          <p className="text-sm font-medium">{ui.question || "Choose Gmail access."}</p>
          <div className="grid gap-2 sm:grid-cols-3">
            {capabilityOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => {
                  setSelectedCapabilityOption(option);
                  setSelectedMemoryPolicy(null);
                  setNotice(null);
                  setError(null);
                }}
                className={`rounded-md border px-3 py-2 text-left text-xs transition ${
                  selectedCapabilityOption?.id === option.id
                    ? "border-[var(--teal)] bg-white shadow-sm"
                    : "border-[var(--outline)] bg-white/70 hover:border-[var(--teal)]"
                }`}
              >
                <span className="block font-semibold">{option.label}</span>
                {option.description ? (
                  <span className="mt-1 block leading-5 text-[var(--ink-soft)]">
                    {option.description}
                  </span>
                ) : null}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {needsMemoryChoice ? (
        <div className="mt-4 space-y-3">
          <p className="text-sm font-medium">
            {selectedCapabilityOption
              ? "Do you want me to remember what I read in your emails so I can learn about you, or should I forget what they say after the task?"
              : ui.question || "Choose Gmail memory behavior."}
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {(memoryOptions.length
              ? memoryOptions
              : [
                  {
                    id: "remember",
                    label: "Remember and learn",
                    description: "Allow email contents to feed Hexis ingestion and memory.",
                    memory_policy: "remember",
                  },
                  {
                    id: "forget",
                    label: "Forget after reading",
                    description: "Keep email reads task-scoped by default.",
                    memory_policy: "forget",
                  },
                ]
            ).map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() =>
                  setSelectedMemoryPolicy(option.memory_policy || option.id)
                }
                disabled={Boolean(busy)}
                className="rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-left text-xs hover:border-[var(--teal)] disabled:opacity-40"
              >
                <span className="block font-semibold">{option.label}</span>
                {option.description ? (
                  <span className="mt-1 block leading-5 text-[var(--ink-soft)]">
                    {option.description}
                  </span>
                ) : null}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {needsAutonomyChoice && !needsMemoryChoice ? (
        <div className="mt-4 space-y-3">
          <p className="text-sm font-medium">
            {ui.question && ui.status === "needs_autonomy_choice"
              ? ui.question
              : "Do you want me to check Gmail during heartbeats on my own, or only when you ask while you are here?"}
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {(autonomyOptions.length
              ? autonomyOptions
              : [
                  {
                    id: "ask_only",
                    label: "Only when I ask",
                    description: "No autonomous heartbeat email checks.",
                    heartbeat_digest_enabled: false,
                  },
                  {
                    id: "heartbeat_digest",
                    label: "Allow heartbeat checks",
                    description: "Allow hourly heartbeats to check Gmail for important messages and digests.",
                    heartbeat_digest_enabled: true,
                  },
                ]
            ).map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() =>
                  void runStart(
                    capabilities,
                    memoryPolicy,
                    option.heartbeat_digest_enabled === true,
                    { allowMissingClientSecret: true }
                  )
                }
                disabled={Boolean(busy)}
                className="rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-left text-xs hover:border-[var(--teal)] disabled:opacity-40"
              >
                <span className="block font-semibold">{option.label}</span>
                {option.description ? (
                  <span className="mt-1 block leading-5 text-[var(--ink-soft)]">
                    {option.description}
                  </span>
                ) : null}
              </button>
            ))}
          </div>
          <p className="text-xs leading-5 text-[var(--ink-soft)]">
            This is separate from Google sign-in. Connecting Gmail does not by itself permit background reads.
          </p>
        </div>
      ) : null}

      {connected ? (
        <div className="mt-3 space-y-2">
          {connectedAccounts.length ? (
            connectedAccounts.map((account, index) => (
              <div key={`${asString(account.account_key, String(index))}-${index}`} className="rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-xs">
                <span className="font-medium">
                  {asString(account.display_name) || asString(account.account_key) || "Gmail account"}
                </span>
                <span className="ml-2 text-[var(--ink-soft)]">connected</span>
              </div>
            ))
          ) : (
            <p className="rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-xs">
              Gmail credentials are saved.
            </p>
          )}
        </div>
      ) : null}

      {canStartOAuth ? (
        <div className="mt-4 space-y-3 rounded-md border border-[var(--outline)] bg-white p-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="text-sm font-medium">Gmail sign-in</p>
              <p className="mt-1 text-xs leading-5 text-[var(--ink-soft)]">
                {ui.client_secret_saved
                  ? "Google setup is saved locally. You can start sign-in."
                  : hostedAvailable
                    ? "Use Google sign-in to connect the Gmail powers you chose."
                    : "This local Hexis build needs a one-time Gmail setup before Samantha can connect. The guide below walks through each step."}
              </p>
            </div>
            <Badge variant={ui.client_secret_saved ? "success" : "warning"}>
              {ui.client_secret_saved || hostedAvailable ? "ready" : "setup needed"}
            </Badge>
          </div>
          {showLocalGoogleSetup ? (
            <div className="space-y-3 rounded-md border border-[var(--outline)] bg-[#fbfdfc] p-3">
              <ol className="list-decimal space-y-2 pl-5 text-xs leading-5 text-[var(--ink-soft)]">
                {setupSteps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
              {ui.docs_url ? (
                <a
                  href={ui.docs_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-xs font-semibold text-[var(--teal)] underline"
                >
                  Open Google setup page <ExternalLink size={12} />
                </a>
              ) : null}
              <div className="grid gap-2 sm:grid-cols-[auto_1fr]">
                <input
                  ref={clientSecretFileRef}
                  type="file"
                  accept="application/json,.json"
                  className="hidden"
                  onChange={(event) => void runSaveClientSecret(event.target.files?.[0])}
                />
                <button
                  type="button"
                  onClick={() => clientSecretFileRef.current?.click()}
                  disabled={Boolean(busy)}
                  className="inline-flex items-center justify-center gap-2 rounded-md bg-[var(--foreground)] px-3 py-2 text-xs font-semibold text-white hover:bg-[var(--teal)] disabled:opacity-40"
                >
                  <Paperclip size={14} />
                  Upload Google setup file
                </button>
                <input
                  type="text"
                  value={clientSecretPath}
                  onChange={(event) => setClientSecretPath(event.target.value)}
                  placeholder="Or paste the downloaded setup file path"
                  className="w-full rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-sm outline-none focus:border-[var(--teal)] focus:ring-2 focus:ring-[var(--teal)]/10"
                />
              </div>
              <details className="text-xs text-[var(--ink-soft)]">
                <summary className="cursor-pointer font-semibold text-[var(--teal)]">
                  Developer options
                </summary>
                <div className="mt-2 space-y-2">
                  {ui.technical_next_step ? (
                    <p className="rounded-md bg-[var(--surface-strong)] px-3 py-2 leading-5">
                      {ui.technical_next_step}
                    </p>
                  ) : null}
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={useEnvSecret}
                      onChange={(event) => setUseEnvSecret(event.target.checked)}
                      disabled={!envAvailable}
                      className="h-4 w-4 accent-[var(--teal)] disabled:opacity-40"
                    />
                    Use server-provided setup file
                    {!envAvailable ? <span className="text-amber-700">(not configured)</span> : null}
                  </label>
                </div>
              </details>
            </div>
          ) : null}
          {needsLocalGoogleSetup ? (
            <button
              type="button"
              onClick={() => setShowAdvancedCredentialSetup((current) => !current)}
              className="text-xs font-semibold text-[var(--teal)] underline"
            >
              {showAdvancedCredentialSetup ? "Hide setup guide" : "Walk me through setup"}
            </button>
          ) : null}
          <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
            <div className="space-y-2">
              {!ui.client_secret_saved ? null : (
                <p className="text-xs text-[var(--ink-soft)]">
                  Google will show the exact Gmail scopes before you approve them.
                </p>
              )}
            </div>
            {ui.client_secret_saved || hostedAvailable || showLocalGoogleSetup ? (
              <button
                type="button"
                onClick={() => void runStart()}
                disabled={
                  Boolean(busy) ||
                  (!ui.client_secret_saved &&
                    !hostedAvailable &&
                    !clientSecretPath.trim() &&
                    !useEnvSecret)
                }
                className="self-start rounded-md bg-[var(--foreground)] px-3 py-2 text-xs font-semibold text-white hover:bg-[var(--teal)] disabled:opacity-40"
              >
                {startBusy ? "Starting..." : "Start Google sign-in"}
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      {!connected && ui.authorization_url ? (
        <div className="mt-4 rounded-md border border-[var(--outline)] bg-white p-3">
          <a
            href={ui.authorization_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-xs font-semibold text-[var(--teal)] underline"
          >
            Open Google sign-in <ExternalLink size={12} />
          </a>
          <p className="mt-1 break-all font-mono text-[11px] leading-5 text-[var(--ink-soft)]">
            {ui.authorization_url}
          </p>
          <p className="mt-2 text-xs leading-5 text-[var(--ink-soft)]">
            After you approve access, Google returns to Hexis and this panel checks for completion automatically.
          </p>
        </div>
      ) : null}

      {canCompleteOAuth ? (
        <details className="mt-3 rounded-md border border-[var(--outline)] bg-white p-3 text-xs">
          <summary className="cursor-pointer font-semibold text-[var(--teal)]">
            Google did not return automatically
          </summary>
          <div className="mt-3 grid gap-2 sm:grid-cols-[1fr_auto]">
            <input
              type="text"
              value={authorizationResponse}
              onChange={(event) => setAuthorizationResponse(event.target.value)}
              placeholder="Paste Google callback URL or authorization code"
              className="w-full rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-sm outline-none focus:border-[var(--teal)] focus:ring-2 focus:ring-[var(--teal)]/10"
            />
            <button
              type="button"
              onClick={() => void runComplete()}
              disabled={Boolean(busy) || !authorizationResponse.trim()}
              className="rounded-md bg-[var(--foreground)] px-3 py-2 text-xs font-semibold text-white hover:bg-[var(--teal)] disabled:opacity-40"
            >
              {completeBusy ? "Completing..." : "Complete"}
            </button>
          </div>
        </details>
      ) : null}

      {ui.safety_note ? (
        <p className="mt-3 text-xs leading-5 text-[var(--ink-soft)]">{ui.safety_note}</p>
      ) : null}
      {ui.docs_url && !showLocalGoogleSetup && !hostedAvailable ? (
        <a
          href={ui.docs_url}
          target="_blank"
          rel="noreferrer"
          className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-[var(--teal)] underline"
        >
          Open Google setup page <ExternalLink size={12} />
        </a>
      ) : null}
      <a href="/connections" className="ml-3 inline-flex items-center gap-1 text-xs font-semibold text-[var(--ink-soft)] underline">
        Open Connections
      </a>
      {notice ? <p className="mt-3 rounded-md border border-[var(--teal)]/35 bg-white px-3 py-2 text-xs">{notice}</p> : null}
      {error ? <p className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p> : null}
    </div>
  );
}

function ActivityInspector({
  events,
  messages,
  agentStatus,
  currentPhase,
  sending,
  streamMeter,
  expandedPanes,
  onTogglePane,
  onClear,
  onClose,
}: {
  events: LogEvent[];
  messages: ChatMessage[];
  agentStatus: AgentStatus;
  currentPhase: string | null;
  sending: boolean;
  streamMeter: StreamMeter;
  expandedPanes: Set<ActivityPaneId>;
  onTogglePane: (pane: ActivityPaneId) => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const thoughtPairs = useMemo(() => buildThoughtPairs(events, messages), [events, messages]);
  const emotionHistory = useMemo(() => buildEmotionHistory(events, agentStatus), [events, agentStatus]);
  const toolEvents = useMemo(
    () => events.filter((event) => event.category === "tool" || event.category === "error"),
    [events],
  );
  const recallEvents = useMemo(() => events.filter(isMemoryRecallEvent), [events]);

  const currentEmotion = emotionHistory.length > 0
    ? emotionHistory[emotionHistory.length - 1]
    : {
        id: "current-emotion",
        ts: Date.now(),
        label: agentStatus.mood || "neutral",
        valence: typeof agentStatus.valence === "number" ? agentStatus.valence : null,
        detail: agentStatus.mood || "No emotional state reported yet.",
      };
  const latestThought = thoughtPairs.length > 0 ? thoughtPairs[thoughtPairs.length - 1] : null;
  const latestTool = toolEvents.length > 0 ? toolEvents[toolEvents.length - 1] : null;
  const latestRecall = recallEvents.length > 0 ? recallEvents[recallEvents.length - 1] : null;
  const latestMemories = latestRecall ? memoryPreviewsFromEvent(latestRecall) : [];
  const latestMemory = latestMemories.length > 0 ? latestMemories[0] : null;
  const workLabel = sending
    ? streamLabel(currentPhase || "stream")
    : latestTool && Date.now() - latestTool.ts < 15000
      ? latestTool.title
      : "Idle";

  return (
    <aside className="fixed inset-y-14 right-0 z-20 flex w-full flex-col border-l border-[var(--outline)] bg-[#f8faf8] sm:w-[390px] lg:static lg:inset-auto lg:w-[380px]">
      <div className="flex h-16 items-center justify-between border-b border-[var(--outline)] px-4">
        <div>
          <h2 className="text-sm font-semibold">Activity</h2>
          <p className="text-xs text-[var(--ink-soft)]">{events.length} recent events</p>
        </div>
        <div className="flex items-center gap-1">
          <button type="button" title="Clear activity" aria-label="Clear activity" onClick={onClear} className="flex h-8 w-8 items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--surface-strong)]"><Trash2 size={16} /></button>
          <button type="button" title="Close activity" aria-label="Close activity" onClick={onClose} className="flex h-8 w-8 items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--surface-strong)] lg:hidden"><X size={17} /></button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 border-b border-[var(--outline)] p-3">
        <ActivityKpi
          label="Tok/s"
          value={streamMeter.tokensPerSecond > 0 ? streamMeter.tokensPerSecond.toFixed(1) : "0.0"}
          detail={streamMeter.active ? "streaming" : "last turn"}
          gaugeValue={Math.min(1, streamMeter.tokensPerSecond / 60)}
        />
        <ActivityKpi
          label="Emotion"
          value={titleCase(currentEmotion.label)}
          detail={formatValence(currentEmotion.valence)}
          gaugeValue={valenceGauge(currentEmotion.valence)}
        />
        <ActivityKpi
          label="Work"
          value={workLabel}
          detail={sending ? "active" : "standing by"}
          gaugeValue={sending ? 0.82 : 0.18}
        />
        <ActivityKpi
          label="Recall"
          value={String(memoryCountFromEvent(latestRecall))}
          detail="memories"
          gaugeValue={Math.min(1, memoryCountFromEvent(latestRecall) / 10)}
        />
      </div>

      <div className="flex-1 overflow-y-auto">
        <ActivityPane
          id="thoughts"
          title="Thoughts"
          icon={BrainCircuit}
          count={thoughtPairs.length}
          expanded={expandedPanes.has("thoughts")}
          onToggle={onTogglePane}
          summary={
            latestThought ? (
              <ThoughtPairView pair={latestThought} compact />
            ) : (
              <p className="text-xs leading-5 text-[var(--ink-soft)]">No thought trace yet.</p>
            )
          }
        >
          {thoughtPairs.length === 0 ? (
            <p className="text-xs leading-5 text-[var(--ink-soft)]">No thought history yet.</p>
          ) : (
            <div className="space-y-3">
              {[...thoughtPairs].reverse().map((pair) => (
                <ThoughtPairView key={pair.id} pair={pair} />
              ))}
            </div>
          )}
        </ActivityPane>

        <ActivityPane
          id="emotion"
          title="Emotion"
          icon={Activity}
          count={emotionHistory.length}
          expanded={expandedPanes.has("emotion")}
          onToggle={onTogglePane}
          summary={<EmotionSnapshotView snapshot={currentEmotion} compact />}
        >
          {emotionHistory.length === 0 ? (
            <p className="text-xs leading-5 text-[var(--ink-soft)]">No emotion history yet.</p>
          ) : (
            <div className="space-y-2">
              {[...emotionHistory].reverse().map((snapshot) => (
                <EmotionSnapshotView key={snapshot.id} snapshot={snapshot} />
              ))}
            </div>
          )}
        </ActivityPane>

        <ActivityPane
          id="activity"
          title="Activity"
          icon={Wrench}
          count={toolEvents.length}
          expanded={expandedPanes.has("activity")}
          onToggle={onTogglePane}
          summary={
            latestTool ? (
              <ActivityEventRow event={latestTool} compact />
            ) : (
              <p className="text-xs leading-5 text-[var(--ink-soft)]">No tool activity yet.</p>
            )
          }
        >
          {toolEvents.length === 0 ? (
            <p className="text-xs leading-5 text-[var(--ink-soft)]">No tool history yet.</p>
          ) : (
            <div className="space-y-2">
              {[...toolEvents].reverse().map((event) => (
                <ActivityEventRow key={event.id} event={event} />
              ))}
            </div>
          )}
        </ActivityPane>

        <ActivityPane
          id="memory"
          title="Memory"
          icon={Database}
          count={recallEvents.length}
          expanded={expandedPanes.has("memory")}
          onToggle={onTogglePane}
          summary={
            latestMemory ? (
              <MemoryPreviewRow memory={latestMemory} compact />
            ) : latestRecall ? (
              <p className="text-xs leading-5 text-[var(--ink-soft)]">{latestRecall.detail}</p>
            ) : (
              <p className="text-xs leading-5 text-[var(--ink-soft)]">No memory retrieval yet.</p>
            )
          }
        >
          {recallEvents.length === 0 ? (
            <p className="text-xs leading-5 text-[var(--ink-soft)]">No memory retrieval history yet.</p>
          ) : (
            <div className="space-y-4">
              {[...recallEvents].reverse().map((event) => {
                const memories = memoryPreviewsFromEvent(event);
                return (
                  <div key={event.id} className="border-t border-[var(--outline)] pt-3 first:border-t-0 first:pt-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs font-semibold">{event.detail}</p>
                      <span className="text-[10px] text-[var(--ink-soft)]">{timeLabel(event.ts)}</span>
                    </div>
                    {memories.length > 0 ? (
                      <div className="mt-2 space-y-2">
                        {memories.map((memory, index) => (
                          <MemoryPreviewRow
                            key={`${event.id}:${memory.id || index}`}
                            memory={memory}
                          />
                        ))}
                      </div>
                    ) : (
                      <p className="mt-2 text-xs leading-5 text-[var(--ink-soft)]">No memory previews were attached to this recall event.</p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </ActivityPane>
      </div>
    </aside>
  );
}

function ActivityKpi({
  label,
  value,
  detail,
  gaugeValue,
}: {
  label: string;
  value: string;
  detail: string;
  gaugeValue: number;
}) {
  const pct = Math.max(0, Math.min(1, gaugeValue));
  const degrees = Math.round(pct * 360);
  return (
    <div className="min-w-0 border border-[var(--outline)] bg-white p-2">
      <div className="flex items-center gap-2">
        <span
          className="flex h-8 w-8 flex-none items-center justify-center rounded-full"
          style={{ background: `conic-gradient(var(--teal) ${degrees}deg, #e6ece8 ${degrees}deg)` }}
          aria-hidden="true"
        >
          <span className="h-4 w-4 rounded-full bg-white" />
        </span>
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--ink-soft)]">{label}</p>
          <p className="line-clamp-2 min-h-4 break-words text-sm font-semibold leading-4">{value}</p>
        </div>
      </div>
      <p className="mt-1 truncate text-[10px] text-[var(--ink-soft)]">{detail}</p>
    </div>
  );
}

function ActivityPane({
  id,
  title,
  icon: Icon,
  count,
  expanded,
  onToggle,
  summary,
  children,
}: {
  id: ActivityPaneId;
  title: string;
  icon: LucideIcon;
  count: number;
  expanded: boolean;
  onToggle: (pane: ActivityPaneId) => void;
  summary: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="border-b border-[var(--outline)] bg-white">
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => onToggle(id)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left hover:bg-[var(--surface-strong)]"
      >
        <span className="flex min-w-0 items-center gap-2">
          <Icon size={16} className="flex-none text-[var(--teal)]" />
          <span className="truncate text-sm font-semibold">{title}</span>
          <Badge variant="muted">{count}</Badge>
        </span>
        <span className="flex h-6 w-6 flex-none items-center justify-center rounded-md border border-[var(--outline)] text-sm text-[var(--ink-soft)]">
          {expanded ? "-" : "+"}
        </span>
      </button>
      <div className="px-4 pb-3">{expanded ? children : summary}</div>
    </section>
  );
}

function ThoughtPairView({ pair, compact = false }: { pair: ThoughtPair; compact?: boolean }) {
  return (
    <div className={compact ? "space-y-2" : "space-y-2 border-t border-[var(--outline)] pt-3 first:border-t-0 first:pt-0"}>
      <div>
        <div className="flex items-center justify-between gap-2">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--ink-soft)]">Subconscious</p>
          {!compact ? <span className="text-[10px] text-[var(--ink-soft)]">{timeLabel(pair.ts)}</span> : null}
        </div>
        <p className={`${compact ? "line-clamp-2" : ""} text-xs leading-5 text-[var(--foreground)]`}>
          {pair.subconsciousText || "No subconscious response captured."}
        </p>
      </div>
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--ink-soft)]">Conscious</p>
        <p className={`${compact ? "line-clamp-2" : ""} text-xs leading-5 text-[var(--ink-soft)]`}>
          {pair.consciousText || "No conscious response captured yet."}
        </p>
      </div>
    </div>
  );
}

function EmotionSnapshotView({ snapshot, compact = false }: { snapshot: EmotionSnapshot; compact?: boolean }) {
  return (
    <div className={compact ? "" : "border-t border-[var(--outline)] pt-2 first:border-t-0 first:pt-0"}>
      <div className="flex items-center justify-between gap-2">
        <p className="truncate text-xs font-semibold">{titleCase(snapshot.label)}</p>
        <span className="text-[10px] text-[var(--ink-soft)]">
          {compact ? formatValence(snapshot.valence) : `${formatValence(snapshot.valence)} | ${timeLabel(snapshot.ts)}`}
        </span>
      </div>
      <p className={`${compact ? "line-clamp-2" : ""} mt-1 text-xs leading-5 text-[var(--ink-soft)]`}>
        {snapshot.detail}
      </p>
    </div>
  );
}

function ActivityEventRow({ event, compact = false }: { event: LogEvent; compact?: boolean }) {
  return (
    <div className={compact ? "" : "border-t border-[var(--outline)] pt-2 first:border-t-0 first:pt-0"}>
      <div className="flex items-center justify-between gap-2">
        <p className="truncate text-xs font-semibold">{event.title}</p>
        <Badge variant={event.category === "error" ? "error" : "muted"}>{event.category === "error" ? "error" : "tool"}</Badge>
      </div>
      <p className={`${compact ? "line-clamp-2" : ""} mt-1 text-xs leading-5 text-[var(--ink-soft)]`}>
        {event.detail || "No detail"}
      </p>
      {!compact ? <p className="mt-1 text-[10px] text-[var(--ink-soft)]">{timeLabel(event.ts)}</p> : null}
    </div>
  );
}

function MemoryPreviewRow({ memory, compact = false }: { memory: MemoryPreview; compact?: boolean }) {
  const score = memory.similarity ?? memory.relevance_score ?? memory.importance ?? null;
  return (
    <div className={compact ? "" : "border-l-2 border-[var(--teal)]/30 pl-2"}>
      <div className="flex items-center justify-between gap-2">
        <p className="truncate text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--ink-soft)]">
          {memory.type || "memory"}
        </p>
        {score !== null ? <span className="text-[10px] text-[var(--ink-soft)]">{score.toFixed(2)}</span> : null}
      </div>
      <p className={`${compact ? "line-clamp-3" : ""} mt-1 text-xs leading-5 text-[var(--foreground)]`}>
        {memory.content || "Memory preview unavailable."}
      </p>
    </div>
  );
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function summarizeSubconscious(output: Record<string, unknown>): string {
  const signals = asRecord(output.signals);
  const emotion = asRecord(signals.emotional_state);
  const parts: string[] = [];
  const primary = asString(emotion.primary_emotion);
  if (primary) {
    const valence = typeof emotion.valence === "number" ? emotion.valence : null;
    parts.push(`${primary}${valence !== null ? ` · valence ${valence >= 0 ? "+" : ""}${valence.toFixed(2)}` : ""}`);
  }
  const reaction = asString(signals.subconscious_response);
  if (reaction) parts.push(reaction);
  const memories = Array.isArray(signals.salient_memories) ? signals.salient_memories.length : 0;
  if (memories) parts.push(`${memories} salient ${memories === 1 ? "memory" : "memories"}`);
  return parts.join(" · ") || `${asString(output.provider, "provider")}/${asString(output.model, "model")}`;
}

function buildThoughtPairs(events: LogEvent[], messages: ChatMessage[]): ThoughtPair[] {
  const pairs: ThoughtPair[] = [];
  let pendingSubconscious: LogEvent | undefined;
  for (const event of events) {
    if (event.category === "subconscious") {
      pendingSubconscious = event;
      continue;
    }
    if (!isConsciousModelResponse(event)) continue;
    pairs.push({
      id: `${pendingSubconscious?.id || "no-subconscious"}:${event.id}`,
      ts: event.ts,
      subconscious: pendingSubconscious,
      conscious: event,
      subconsciousText: pendingSubconscious ? subconsciousResponseText(pendingSubconscious) : "",
      consciousText: modelResponseText(event),
    });
    pendingSubconscious = undefined;
  }
  if (pendingSubconscious) {
    pairs.push({
      id: pendingSubconscious.id,
      ts: pendingSubconscious.ts,
      subconscious: pendingSubconscious,
      subconsciousText: subconsciousResponseText(pendingSubconscious),
      consciousText: latestAssistantText(messages),
    });
  } else if (pairs.length > 0 && !pairs[pairs.length - 1].consciousText) {
    pairs[pairs.length - 1] = {
      ...pairs[pairs.length - 1],
      consciousText: latestAssistantText(messages),
    };
  } else if (pairs.length === 0) {
    const assistantText = latestAssistantText(messages);
    if (assistantText) {
      pairs.push({
        id: "latest-assistant",
        ts: Date.now(),
        subconsciousText: "",
        consciousText: assistantText,
      });
    }
  }
  return pairs;
}

function buildEmotionHistory(events: LogEvent[], agentStatus: AgentStatus): EmotionSnapshot[] {
  const snapshots = events
    .filter((event) => event.category === "subconscious")
    .map(emotionSnapshotFromEvent)
    .filter((snapshot): snapshot is EmotionSnapshot => snapshot !== null);
  if (
    snapshots.length === 0 &&
    (agentStatus.mood || typeof agentStatus.valence === "number")
  ) {
    snapshots.push({
      id: "agent-status",
      ts: Date.now(),
      label: agentStatus.mood || "neutral",
      valence: typeof agentStatus.valence === "number" ? agentStatus.valence : null,
      detail: agentStatus.mood || "Current status",
    });
  }
  return snapshots;
}

function emotionSnapshotFromEvent(event: LogEvent): EmotionSnapshot | null {
  const raw = asRecord(event.raw);
  const signals = asRecord(raw.signals);
  const emotion = asRecord(signals.emotional_state);
  const label =
    asString(emotion.primary_emotion) ||
    asString(emotion.mood) ||
    asString(emotion.state) ||
    firstDetailPart(event.detail) ||
    "neutral";
  const valence = asNumber(emotion.valence);
  const detail =
    asString(emotion.summary) ||
    asString(emotion.reason) ||
    asString(signals.subconscious_response) ||
    event.detail ||
    label;
  return {
    id: event.id,
    ts: event.ts,
    label,
    valence,
    detail,
  };
}

function subconsciousResponseText(event: LogEvent): string {
  const raw = asRecord(event.raw);
  const signals = asRecord(raw.signals);
  return (
    asString(signals.subconscious_response).trim() ||
    afterFirstDetailPart(event.detail) ||
    event.detail ||
    ""
  );
}

function isConsciousModelResponse(event: LogEvent): boolean {
  if (event.category !== "model") return false;
  const raw = asRecord(event.raw);
  return asString(raw.kind) === "llm_response" && asString(raw.phase) !== "subconscious";
}

function modelResponseText(event: LogEvent): string {
  const raw = asRecord(event.raw);
  const content = raw.content;
  if (typeof content === "string") return trimDisplayText(content);
  if (content && typeof content === "object") return trimDisplayText(JSON.stringify(content));
  return event.detail || "";
}

function latestAssistantText(messages: ChatMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.content.trim()) {
      return trimDisplayText(message.content);
    }
  }
  return "";
}

function isMemoryRecallEvent(event: LogEvent): boolean {
  if (event.category !== "memory") return false;
  const raw = asRecord(event.raw);
  const kind = asString(raw.kind).toLowerCase();
  return kind === "memory_recall" || event.title.toLowerCase().includes("recall");
}

function memoryPreviewsFromEvent(event: LogEvent): MemoryPreview[] {
  const raw = asRecord(event.raw);
  const memories = Array.isArray(raw.memories) ? raw.memories : [];
  return memories.flatMap((item): MemoryPreview[] => {
    const record = asRecord(item);
    const content = asString(record.content).trim();
    if (!content) return [];
    return [{
      id: asString(record.id) || undefined,
      type: asString(record.type) || undefined,
      content,
      similarity: asNumber(record.similarity),
      relevance_score: asNumber(record.relevance_score),
      importance: asNumber(record.importance),
      trust_level: asNumber(record.trust_level),
      confidence: asNumber(record.confidence),
      source: asString(record.source) || undefined,
    }];
  });
}

function memoryCountFromEvent(event: LogEvent | null): number {
  if (!event) return 0;
  const raw = asRecord(event.raw);
  const count = asNumber(raw.count);
  if (count !== null) return Math.round(count);
  const match = event.detail.match(/Retrieved\s+(\d+)/i);
  return match ? Number(match[1]) : memoryPreviewsFromEvent(event).length;
}

function firstDetailPart(value: string): string {
  return value.split(" · ")[0]?.trim() || "";
}

function afterFirstDetailPart(value: string): string {
  const parts = value.split(" · ");
  return parts.slice(1).join(" · ").trim();
}

function trimDisplayText(value: string, maxLength = 520): string {
  const trimmed = value.replace(/\s+/g, " ").trim();
  if (trimmed.length <= maxLength) return trimmed;
  return `${trimmed.slice(0, maxLength - 1)}...`;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatValence(value: number | null): string {
  if (value === null) return "valence n/a";
  return `valence ${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function valenceGauge(value: number | null): number {
  if (value === null) return 0.5;
  return Math.max(0, Math.min(1, (value + 1) / 2));
}

function titleCase(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase()) || "Neutral";
}

function timeLabel(ts: number): string {
  return new Date(ts).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function streamLabel(phase: string) {
  switch (phase) {
    case "subconscious":
      return "Subconscious";
    case "conscious_plan":
      return "Conscious Plan";
    case "conscious_final":
      return "Conscious Response";
    default:
      return phase || "Stream";
  }
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

async function readIntegrationActionPayload(response: Response): Promise<IntegrationActionResult> {
  const text = await response.text();
  if (!text.trim()) return {};
  try {
    return JSON.parse(text) as IntegrationActionResult;
  } catch {
    return { error: text };
  }
}

function integrationActionError(payload: IntegrationActionResult, status: number): string {
  if (payload.error) return payload.error;
  if (payload.detail) return payload.detail;
  const output = asRecord(payload.output);
  const outputError = asString(output.error);
  if (outputError) return outputError;
  return `Action failed (${status})`;
}

function integrationActionNotice(payload: IntegrationActionResult, status: number): string {
  if (payload.error || payload.detail || payload.success === false) {
    return integrationActionError(payload, status);
  }
  const connectorUi = connectorSetupUiFromPayload(payload);
  if (connectorUi) return connectorSetupNotice(connectorUi);
  if (payload.display_output) return payload.display_output;
  const output = asRecord(payload.output);
  return (
    asString(output.next_step) ||
    asString(output.user_next_step) ||
    asString(output.status) ||
    "Action complete."
  );
}

function connectorSetupNotice(ui: ConnectorSetupUi): string {
  const connector = ui.display_name || ui.connector_id || "Connector";
  const modes = ui.credential_step?.modes || [];
  const hostedAvailable =
    ui.hexis_oauth_client_available ||
    modes.some((mode) => mode.id === "hosted_oauth" && mode.available);
  switch (ui.status) {
    case "needs_capability_choice":
      return `${connector} setup opened. Choose what access to request.`;
    case "needs_memory_choice":
      return `${connector} setup is waiting for the memory policy choice.`;
    case "needs_autonomy_choice":
      return `${connector} setup is waiting for the background email check choice.`;
    case "needs_client_secret":
    case "setup":
      if (hostedAvailable) return `${connector} is ready for Google sign-in.`;
      return `${connector} setup needs one-time Google setup. The panel walks through the steps.`;
    case "client_secret_saved":
      return `${connector} setup file saved. Start Google sign-in from the setup panel.`;
    case "pending_authorization":
      return `${connector} sign-in started. Open Google; Hexis will complete the connection when Google returns.`;
    case "connected":
      return `${connector} is connected.`;
    default:
      return `${connector} setup updated.`;
  }
}

function connectorSetupStatusLabel(ui: ConnectorSetupUi): string {
  if (ui.credential_step_label) return ui.credential_step_label;
  if (ui.status === "needs_client_secret") {
    const modes = ui.credential_step?.modes || [];
    const hostedAvailable =
      ui.hexis_oauth_client_available ||
      modes.some((mode) => mode.id === "hosted_oauth" && mode.available);
    return hostedAvailable ? "ready to connect" : "setup needed";
  }
  if (ui.status === "client_secret_saved") return "ready to connect";
  if (ui.status === "pending_authorization") return "authorization pending";
  if (ui.status === "needs_capability_choice") return "choose access";
  if (ui.status === "needs_memory_choice") return "choose memory";
  if (ui.status === "needs_autonomy_choice") return "choose background use";
  return (ui.status || "setup").replace(/_/g, " ");
}

function humanizeCapability(value: string): string {
  return value.replace(/_/g, " ");
}

function phaseDescription(phase: string) {
  switch (phase) {
    case "subconscious":
      return "Running subconscious processes...";
    case "conscious_plan":
      return "Planning response...";
    case "conscious_final":
      return "Generating response...";
    case "connector_setup":
      return "Opening setup...";
    default:
      return "Thinking...";
  }
}

function isSearchToolMisconfigured(text: string): boolean {
  const normalized = (text || "").toLowerCase();
  if (!normalized) return false;
  return (
    normalized.includes("web search api key not configured") ||
    (normalized.includes("web search") && normalized.includes("not configured")) ||
    normalized.includes("no tavily api key") ||
    normalized.includes("keyless search fallbacks failed") ||
    normalized.includes("tavily_api_key")
  );
}
