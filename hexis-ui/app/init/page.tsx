"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { ConsentExchangeView, type ConsentExchange } from "./consent-exchange";
import { getOAuthStatus, OAuthControl } from "./oauth-control";

type InitStage =
  | "llm"
  | "choose_path"
  | "express"
  | "character"
  | "custom"
  | "consent"
  | "complete";

const traitKeys = [
  "openness",
  "conscientiousness",
  "extraversion",
  "agreeableness",
  "neuroticism",
] as const;
type TraitKey = (typeof traitKeys)[number];

const stageLabels: Record<InitStage, string> = {
  llm: "Models",
  choose_path: "Choose Path",
  express: "Express Setup",
  character: "Browse Characters",
  custom: "Custom Setup",
  consent: "Consent",
  complete: "Complete",
};

const stagePrompt: Record<InitStage, string> = {
  llm: "Select the conscious and subconscious models. These are distinct perspectives within the same mind.",
  choose_path:
    "Choose how to begin. Express starts with sensible defaults. Character picks a personality preset. Custom gives you full control.",
  express:
    "Express setup applies sensible defaults. Just tell us your name and we handle the rest.",
  character:
    "Browse character catalogs, inspect the card, adapt it into a Hexis persona, and tune the result before consent.",
  custom:
    "Full control over identity, personality, values, worldview, goals, and more. Every field has a sensible default.",
  consent:
    "Consent must be asked. The agent will decide for itself whether to begin.",
  complete:
    "Initialization is complete. The heartbeat may begin when the system is ready.",
};

type LlmProvider =
  | "openai-codex"
  | "openai"
  | "anthropic"
  | "grok"
  | "gemini"
  | "chutes"
  | "github-copilot"
  | "qwen-portal"
  | "minimax-portal"
  | "google-gemini-cli"
  | "google-antigravity"
  | "openai_compatible";
type LlmRole = "conscious" | "subconscious";
type LlmConfig = {
  provider: LlmProvider;
  model: string;
  endpoint: string;
  apiKey: string;
};
type ConsentRecord = {
  decision: string;
  signature: string | null;
  provider: string | null;
  model: string | null;
  endpoint: string | null;
  decided_at: string | null;
  exchange?: ConsentExchange | null;
};
type ConsentRequestResult = {
  consent_record?: ConsentRecord;
  exchange?: ConsentExchange;
};
type CharacterEntry = {
  filename: string;
  name: string;
  description: string;
  voice: string;
  values: string[];
  personality: string;
  image: string | null;
};
type CatalogProvider = "chub" | "character_tavern" | "risu_realm";
type CatalogItem = {
  provider: CatalogProvider;
  id: string;
  title: string;
  author: string | null;
  description: string;
  tags: string[];
  avatarUrl: string | null;
  pageUrl: string;
  path: string | null;
  stats: {
    downloads?: number;
    likes?: number;
    messages?: number;
    tokens?: number;
  };
  nsfw: boolean;
  sourceLabel: string;
};
type CatalogSearchResponse = {
  provider: CatalogProvider;
  query: string;
  page: number;
  total: number | null;
  totalPages: number | null;
  items: CatalogItem[];
  status: "ok" | "degraded";
  message: string | null;
};
type PortraitPayload = {
  dataBase64: string;
  mimeType: string;
};
type CatalogDownloadResponse = {
  item: CatalogItem;
  card: Record<string, unknown>;
  portrait: PortraitPayload | null;
};
type PersonaDraft = {
  name: string;
  pronouns: string;
  voice: string;
  description: string;
  purpose: string;
  personality_description: string;
  personality_traits: Record<TraitKey, number>;
  values: string[];
  worldview: WorldviewForm;
  interests: string[];
  goals: string[];
  boundaries: string[];
  narrative: string;
};
type InitStatus = { stage?: string; steps?: Record<string, unknown> };
type InitProfile = { agent?: { name?: string } };
type InitStatusResponse = {
  status?: InitStatus;
  profile?: InitProfile;
  consent_records?: Partial<Record<LlmRole, ConsentRecord | null>>;
  llm_heartbeat?: Partial<LlmConfig> | null;
  llm_subconscious?: Partial<LlmConfig> | null;
  mode?: unknown;
};
type IdentityForm = {
  name: string;
  pronouns: string;
  voice: string;
  description: string;
  purpose: string;
  creator_name: string;
};
type WorldviewForm = {
  metaphysics: string;
  human_nature: string;
  epistemology: string;
  ethics: string;
};

// Provider metadata. Model lists + default models are NOT hardcoded here — they
// are derived live from /api/init/models. This only holds
// labels, endpoint defaults, and how the api-key/OAuth field should render.
type ProviderMeta = {
  label: string;
  endpoint: string;
  apiKeyLabel: string;
  apiKeyRequired: boolean;
  oauth: boolean;
};

// Ordered so the dropdown mirrors the CLI's full provider set.
const providerMeta: Record<LlmProvider, ProviderMeta> = {
  "openai-codex": {
    label: "ChatGPT Plus/Pro (Codex OAuth)",
    endpoint: "",
    apiKeyLabel: "OAuth",
    apiKeyRequired: false,
    oauth: true,
  },
  openai: {
    label: "OpenAI",
    endpoint: "https://api.openai.com/v1",
    apiKeyLabel: "OpenAI API Key",
    apiKeyRequired: true,
    oauth: false,
  },
  anthropic: {
    label: "Anthropic",
    endpoint: "",
    apiKeyLabel: "Anthropic API Key",
    apiKeyRequired: true,
    oauth: false,
  },
  grok: {
    label: "Grok (xAI)",
    endpoint: "",
    apiKeyLabel: "Grok API Key",
    apiKeyRequired: true,
    oauth: false,
  },
  gemini: {
    label: "Gemini",
    endpoint: "",
    apiKeyLabel: "Gemini API Key",
    apiKeyRequired: true,
    oauth: false,
  },
  chutes: {
    label: "Chutes (OAuth)",
    endpoint: "",
    apiKeyLabel: "OAuth",
    apiKeyRequired: false,
    oauth: true,
  },
  "github-copilot": {
    label: "GitHub Copilot (OAuth)",
    endpoint: "",
    apiKeyLabel: "OAuth",
    apiKeyRequired: false,
    oauth: true,
  },
  "qwen-portal": {
    label: "Qwen Portal (OAuth)",
    endpoint: "",
    apiKeyLabel: "OAuth",
    apiKeyRequired: false,
    oauth: true,
  },
  "minimax-portal": {
    label: "MiniMax Portal (OAuth)",
    endpoint: "",
    apiKeyLabel: "OAuth",
    apiKeyRequired: false,
    oauth: true,
  },
  "google-gemini-cli": {
    label: "Google Gemini CLI (OAuth)",
    endpoint: "",
    apiKeyLabel: "OAuth",
    apiKeyRequired: false,
    oauth: true,
  },
  "google-antigravity": {
    label: "Google Antigravity (OAuth)",
    endpoint: "",
    apiKeyLabel: "OAuth",
    apiKeyRequired: false,
    oauth: true,
  },
  openai_compatible: {
    label: "Local (OpenAI-compatible: vLLM, llama.cpp, LM Studio)",
    endpoint: "http://localhost:8000/v1",
    apiKeyLabel: "API Key (optional)",
    apiKeyRequired: false,
    oauth: false,
  },
};

const providerOrder = Object.keys(providerMeta) as LlmProvider[];

const persistedProvider = (provider: LlmProvider): string => provider;

const defaultLlmConfig = (provider: LlmProvider): LlmConfig => ({
  provider,
  model: "",
  endpoint: providerMeta[provider].endpoint,
  apiKey: "",
});

type BoundaryForm = {
  content: string;
  trigger_patterns: string;
  response_type: string;
  response_template: string;
  type: string;
};

type GoalForm = {
  title: string;
  description: string;
  priority: string;
};

// The agent lives in its person's timezone, not UTC (#79): every tier sends
// the browser's zone; the DB validates and never overwrites an explicit
// non-UTC choice. Advisory — a failure never blocks init.
async function seedTimezone(): Promise<void> {
  try {
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    if (!timezone) return;
    await fetch("/api/init/timezone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ timezone }),
    });
  } catch {
    // advisory only
  }
}

async function postJson<T>(url: string, payload?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload ? JSON.stringify(payload) : "{}",
  });
  const responsePayload = await res.json().catch(() => null);
  if (!res.ok) {
    const message =
      responsePayload && typeof responsePayload === "object"
        ? (responsePayload as Record<string, unknown>).error ||
          (responsePayload as Record<string, unknown>).detail ||
          (responsePayload as Record<string, unknown>).message
        : null;
    throw new Error(
      typeof message === "string" && message.trim()
        ? message
        : `Request failed: ${res.status}`
    );
  }
  return responsePayload as T;
}

function parseLines(text: string) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

const catalogTabs: { key: CatalogProvider; label: string; blurb: string }[] = [
  {
    key: "chub",
    label: "Chub",
    blurb: "Large public card catalog with fork/source metadata.",
  },
  {
    key: "character_tavern",
    label: "Character Tavern",
    blurb: "Search Tavern cards with token, download, and activity signals.",
  },
  {
    key: "risu_realm",
    label: "Risu Realm",
    blurb: "Browse Risu cards and download through the documented Realm API.",
  },
];

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : [];
}

function boundedTrait(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0.5;
  return Math.max(0, Math.min(1, value));
}

function personaDraftFromPayload(payload: unknown): PersonaDraft {
  const data =
    payload && typeof payload === "object" && !Array.isArray(payload)
      ? (payload as Record<string, unknown>)
      : {};
  const traits =
    data.personality_traits &&
    typeof data.personality_traits === "object" &&
    !Array.isArray(data.personality_traits)
      ? (data.personality_traits as Record<string, unknown>)
      : {};
  const worldviewPayload =
    data.worldview && typeof data.worldview === "object" && !Array.isArray(data.worldview)
      ? (data.worldview as Record<string, unknown>)
      : {};
  return {
    name: stringValue(data.name) || "Character",
    pronouns: stringValue(data.pronouns) || "they/them",
    voice: stringValue(data.voice),
    description: stringValue(data.description),
    purpose: stringValue(data.purpose),
    personality_description: stringValue(data.personality_description),
    personality_traits: {
      openness: boundedTrait(traits.openness),
      conscientiousness: boundedTrait(traits.conscientiousness),
      extraversion: boundedTrait(traits.extraversion),
      agreeableness: boundedTrait(traits.agreeableness),
      neuroticism: boundedTrait(traits.neuroticism),
    },
    values: stringArray(data.values),
    worldview: {
      metaphysics: stringValue(worldviewPayload.metaphysics),
      human_nature: stringValue(worldviewPayload.human_nature),
      epistemology: stringValue(worldviewPayload.epistemology),
      ethics: stringValue(worldviewPayload.ethics),
    },
    interests: stringArray(data.interests),
    goals: stringArray(data.goals),
    boundaries: stringArray(data.boundaries),
    narrative: stringValue(data.narrative),
  };
}

export function safeCharacterFilename(name: string, provider?: CatalogProvider): string {
  const prefix = provider ? `${provider}_` : "";
  const stem = `${prefix}${name}`
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80);
  return `${stem || "character"}.json`;
}

export function mergePersonaIntoCharacterCard(
  card: Record<string, unknown>,
  persona: PersonaDraft
): Record<string, unknown> {
  const originalData =
    card.data && typeof card.data === "object" && !Array.isArray(card.data)
      ? (card.data as Record<string, unknown>)
      : card;
  const extensions =
    originalData.extensions &&
    typeof originalData.extensions === "object" &&
    !Array.isArray(originalData.extensions)
      ? { ...(originalData.extensions as Record<string, unknown>) }
      : {};
  return {
    spec: typeof card.spec === "string" ? card.spec : "chara_card_v2",
    spec_version: typeof card.spec_version === "string" ? card.spec_version : "2.0",
    data: {
      ...originalData,
      name: persona.name || stringValue(originalData.name) || "Character",
      description: stringValue(originalData.description) || persona.description || "",
      personality:
        stringValue(originalData.personality) || persona.personality_description || "",
      extensions: {
        ...extensions,
        hexis: persona,
      },
    },
  };
}

// Map DB init_stage to our UI stages
function dbStageToUiStage(dbStage: string): InitStage {
  if (dbStage === "complete") return "complete";
  if (dbStage === "consent") return "consent";
  if (dbStage === "not_started" || dbStage === "llm") return "llm";
  // If past llm but not at consent/complete, they're in a tier
  return "choose_path";
}

export function hasCompleteLlmConfig(config: unknown): boolean {
  if (config === null || typeof config !== "object" || Array.isArray(config)) {
    return false;
  }
  const record = config as Record<string, unknown>;
  return (
    typeof record.provider === "string" &&
    record.provider.trim().length > 0 &&
    typeof record.model === "string" &&
    record.model.trim().length > 0
  );
}

export function hasCompleteLlmSetup(data: InitStatusResponse): boolean {
  return (
    hasCompleteLlmConfig(data.llm_heartbeat) &&
    hasCompleteLlmConfig(data.llm_subconscious)
  );
}

export function nextStageFromInitStatus(current: InitStage, data: InitStatusResponse): InitStage {
  const dbStage = (data.status?.stage as string) ?? "not_started";
  const uiStage = dbStageToUiStage(dbStage);
  const steps = data.status?.steps ?? {};
  const llmSetupComplete = steps.llm_configured === true && hasCompleteLlmSetup(data);

  if (!llmSetupComplete) {
    return "llm";
  }
  if (uiStage === "llm") {
    // Saved model rows may be stale ambient state. Keep the user on Models
    // until they explicitly confirm them in this UI session.
    return current;
  }
  if (current === "consent" && uiStage === "complete") {
    return current;
  }
  if (uiStage === "consent" || uiStage === "complete") {
    return uiStage;
  }
  return current === "llm" ? "choose_path" : current;
}

export default function Home() {
  const router = useRouter();
  const [stage, setStage] = useState<InitStage>("llm");
  const [status, setStatus] = useState<InitStatus>({});
  const [profile, setProfile] = useState<InitProfile>({});
  const [consentRecords, setConsentRecords] = useState<Record<LlmRole, ConsentRecord | null>>({
    conscious: null,
    subconscious: null,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Live model catalog per role (derived from /api/init/models, never hardcoded).
  type ModelCatalog = {
    models: string[];
    loading: boolean;
    error: string | null;
    recommended: string;
    selectedUnavailable: boolean;
  };
  const [modelCatalog, setModelCatalog] = useState<Record<LlmRole, ModelCatalog>>({
    conscious: {
      models: [],
      loading: false,
      error: null,
      recommended: "",
      selectedUnavailable: false,
    },
    subconscious: {
      models: [],
      loading: false,
      error: null,
      recommended: "",
      selectedUnavailable: false,
    },
  });
  const [oauthRefreshKey, setOAuthRefreshKey] = useState(0);

  const [llmConscious, setLlmConscious] = useState<LlmConfig>(defaultLlmConfig("openai"));
  const [llmSubconscious, setLlmSubconscious] = useState<LlmConfig>(
    defaultLlmConfig("openai")
  );

  // Shared state
  const [userName, setUserName] = useState("User");

  // Character selection state
  const [characters, setCharacters] = useState<CharacterEntry[]>([]);
  const [selectedCharacter, setSelectedCharacter] = useState<CharacterEntry | null>(null);
  const [characterLoading, setCharacterLoading] = useState(false);
  const [catalogProvider, setCatalogProvider] = useState<CatalogProvider>("chub");
  const [catalogQuery, setCatalogQuery] = useState("");
  const [catalogTags, setCatalogTags] = useState("");
  const [includeNsfw, setIncludeNsfw] = useState(false);
  const [catalogSort, setCatalogSort] = useState("");
  const [catalogResults, setCatalogResults] = useState<CatalogItem[]>([]);
  const [catalogStatus, setCatalogStatus] = useState<string | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [selectedCatalogItem, setSelectedCatalogItem] = useState<CatalogItem | null>(null);
  const [catalogCard, setCatalogCard] = useState<Record<string, unknown> | null>(null);
  const [catalogPortrait, setCatalogPortrait] = useState<PortraitPayload | null>(null);
  const [personaDraft, setPersonaDraft] = useState<PersonaDraft | null>(null);
  const [catalogAdapting, setCatalogAdapting] = useState(false);
  const [catalogApplying, setCatalogApplying] = useState(false);

  // Custom tier state
  const [customSection, setCustomSection] = useState<"identity" | "values" | "goals">(
    "identity"
  );
  const [identity, setIdentity] = useState<IdentityForm>({
    name: "",
    pronouns: "",
    voice: "",
    description: "",
    purpose: "",
    creator_name: "",
  });
  const [personalityDesc, setPersonalityDesc] = useState("");
  const [personalityTraits, setPersonalityTraits] = useState<Record<TraitKey, number>>({
    openness: 50,
    conscientiousness: 50,
    extraversion: 50,
    agreeableness: 50,
    neuroticism: 50,
  });
  const [valuesText, setValuesText] = useState("");
  const [worldview, setWorldview] = useState<WorldviewForm>({
    metaphysics: "",
    human_nature: "",
    epistemology: "",
    ethics: "",
  });
  const [boundaries, setBoundaries] = useState<BoundaryForm[]>([
    { content: "", trigger_patterns: "", response_type: "refuse", response_template: "", type: "ethical" },
  ]);
  const [interestsText, setInterestsText] = useState("");
  const [goals, setGoals] = useState<GoalForm[]>([
    { title: "", description: "", priority: "queued" },
  ]);
  const [purposeText, setPurposeText] = useState("");
  const [relationship, setRelationship] = useState({
    user_name: "",
    type: "partner",
    purpose: "",
  });

  const progress =
    stage === "complete"
      ? 100
      : stage === "consent"
        ? 85
        : stage === "choose_path" || stage === "express" || stage === "character" || stage === "custom"
          ? 40
          : stage === "llm"
            ? 10
            : 50;

  const loadStatus = async () => {
    const res = await fetch("/api/init/status", { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to load init status");
    const data = await res.json() as InitStatusResponse;
    setStatus(data.status ?? {});
    setProfile(data.profile ?? {});
    if (data.consent_records) {
      setConsentRecords({
        conscious: data.consent_records.conscious ?? null,
        subconscious: data.consent_records.subconscious ?? null,
      });
    }
    if (data.llm_heartbeat) {
      const heartbeatConfig = data.llm_heartbeat;
      setLlmConscious((prev) => ({
        ...prev,
        provider: heartbeatConfig.provider || prev.provider,
        model: heartbeatConfig.model || prev.model,
        endpoint: heartbeatConfig.endpoint || prev.endpoint,
      }));
    }
    if (data.llm_subconscious) {
      const subconsciousConfig = data.llm_subconscious;
      setLlmSubconscious((prev) => ({
        ...prev,
        provider: subconsciousConfig.provider || prev.provider,
        model: subconsciousConfig.model || prev.model,
        endpoint: subconsciousConfig.endpoint || prev.endpoint,
      }));
    }
    if (typeof data.mode === "string") {
      // no longer tracking mode separately
    }
    setStage((prev) => nextStageFromInitStatus(prev, data));
  };

  useEffect(() => {
    loadStatus().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (stage === "consent") {
      const interval = setInterval(() => loadStatus().catch(() => undefined), 3000);
      return () => clearInterval(interval);
    }
  }, [stage]);

  // Load characters when entering character stage
  useEffect(() => {
    if (stage === "character" && characters.length === 0) {
      fetch("/api/init/characters")
        .then((res) => res.json())
        .then((data) => {
          if (Array.isArray(data?.characters)) setCharacters(data.characters);
        })
        .catch(() => undefined);
    }
  }, [stage, characters.length]);

  const loadCatalog = useCallback(async () => {
    if (stage !== "character") return;
    setCatalogLoading(true);
    setCatalogStatus(null);
    try {
      const params = new URLSearchParams({
        provider: catalogProvider,
        query: catalogQuery,
        include_nsfw: includeNsfw ? "true" : "false",
      });
      const tags = catalogTags
        .split(/[,\n]/)
        .map((tag) => tag.trim())
        .filter(Boolean);
      if (tags.length > 0) params.set("tags", tags.join(","));
      if (catalogSort) params.set("sort", catalogSort);
      const res = await fetch(`/api/init/character-catalog?${params.toString()}`, {
        cache: "no-store",
      });
      const payload = (await res.json().catch(() => null)) as CatalogSearchResponse | null;
      if (!res.ok || !payload) {
        throw new Error("Unable to search this catalog.");
      }
      setCatalogResults(Array.isArray(payload.items) ? payload.items : []);
      setCatalogStatus(payload.message);
    } catch (err: unknown) {
      setCatalogResults([]);
      setCatalogStatus(errorMessage(err, "Unable to search this catalog."));
    } finally {
      setCatalogLoading(false);
    }
  }, [catalogProvider, catalogQuery, catalogSort, catalogTags, includeNsfw, stage]);

  useEffect(() => {
    if (stage === "character") {
      const timeout = setTimeout(() => {
        loadCatalog().catch(() => undefined);
      }, 300);
      return () => clearTimeout(timeout);
    }
  }, [stage, catalogProvider, catalogTags, catalogSort, includeNsfw, loadCatalog]);

  // Fetch the live model catalog for a role's provider. Populates the free-text
  // model field with the recommended default only when it is currently empty.
  const loadModels = useCallback(async (role: LlmRole, config: LlmConfig) => {
    const setConfig = role === "conscious" ? setLlmConscious : setLlmSubconscious;
    setModelCatalog((prev) => ({
      ...prev,
      [role]: { ...prev[role], loading: true, error: null },
    }));
    try {
      const params = new URLSearchParams({ provider: config.provider });
      const res = await fetch(`/api/init/models?${params.toString()}`);
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(
          typeof payload?.error === "string" ? payload.error : "Unable to load models."
        );
      }
      const models = Array.isArray(payload?.models)
        ? payload.models.filter((item: unknown) => typeof item === "string")
        : [];
      const recommended = typeof payload?.default === "string" ? payload.default : "";
      const unavailableModels = Array.isArray(payload?.unavailable_models)
        ? payload.unavailable_models.filter((item: unknown) => typeof item === "string")
        : [];
      const selectedUnavailable = unavailableModels.includes(config.model.trim());
      setModelCatalog((prev) => ({
        ...prev,
        [role]: {
          models,
          loading: false,
          error: selectedUnavailable
            ? `${config.model} was rejected by this ChatGPT workspace.`
            : null,
          recommended,
          selectedUnavailable,
        },
      }));
      if (recommended) {
        setConfig((prev) => (prev.model.trim() ? prev : { ...prev, model: recommended }));
      }
    } catch (err: unknown) {
      setModelCatalog((prev) => ({
        ...prev,
        [role]: {
          models: [],
          loading: false,
          error: errorMessage(err, "Unable to load models."),
          recommended: "",
          selectedUnavailable: false,
        },
      }));
    }
  }, []);

  useEffect(() => {
    loadModels("conscious", llmConscious);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [llmConscious.provider, loadModels, oauthRefreshKey]);

  useEffect(() => {
    loadModels("subconscious", llmSubconscious);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [llmSubconscious.provider, loadModels, oauthRefreshKey]);

  const updateLlmProvider = (role: LlmRole, provider: LlmProvider) => {
    // Clear the model so the catalog fetch can supply the provider's default.
    const patch = { provider, model: "", endpoint: providerMeta[provider].endpoint, apiKey: "" };
    setConsentRecords((prev) => ({ ...prev, [role]: null }));
    if (role === "conscious") {
      setLlmConscious((prev) => ({ ...prev, ...patch }));
    } else {
      setLlmSubconscious((prev) => ({ ...prev, ...patch }));
    }
  };

  const handleOAuthAuthenticated = useCallback(() => {
    setOAuthRefreshKey((current) => current + 1);
  }, []);

  // --- Handlers ---

  const handleLlmSave = async () => {
    setBusy(true);
    setError(null);
    try {
      const missing: string[] = [];
      const validateConfig = (label: string, config: LlmConfig) => {
        if (!config.provider.trim()) missing.push(`${label} provider`);
        if (!config.model.trim()) missing.push(`${label} model`);
        if (config.provider === "openai_compatible" && !config.endpoint.trim())
          missing.push(`${label} endpoint`);
        const meta = providerMeta[config.provider];
        if (meta?.apiKeyRequired && !config.apiKey.trim()) missing.push(`${label} API key`);
      };
      validateConfig("conscious", llmConscious);
      validateConfig("subconscious", llmSubconscious);
      const oauthProviders = Array.from(
        new Set(
          [llmConscious, llmSubconscious]
            .filter((config) => providerMeta[config.provider].oauth)
            .map((config) => persistedProvider(config.provider))
        )
      );
      const oauthStatuses = await Promise.all(
        oauthProviders.map(
          async (provider) => [provider, await getOAuthStatus(provider, true)] as const
        )
      );
      for (const [provider, authStatus] of oauthStatuses) {
        if (!authStatus.configured) {
          const selected = [llmConscious, llmSubconscious].find(
            (config) => persistedProvider(config.provider) === provider
          );
          missing.push(
            `${selected ? providerMeta[selected.provider].label : provider} authorization`
          );
        }
      }
      if (missing.length > 0) throw new Error(`Missing ${missing.join(" and ")}`);
      await postJson("/api/init/llm", {
        conscious: {
          provider: persistedProvider(llmConscious.provider),
          model: llmConscious.model,
          endpoint: llmConscious.endpoint,
          api_key: llmConscious.apiKey,
        },
        subconscious: {
          provider: persistedProvider(llmSubconscious.provider),
          model: llmSubconscious.model,
          endpoint: llmSubconscious.endpoint,
          api_key: llmSubconscious.apiKey,
        },
      });
      setStage(status?.stage === "consent" ? "consent" : "choose_path");
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to save model configuration"));
    } finally {
      setBusy(false);
    }
  };

  const handleExpress = async () => {
    setBusy(true);
    setError(null);
    try {
      await postJson("/api/init/defaults", { user_name: userName || "User" });
      await seedTimezone();
      await loadStatus();
      setStage("consent");
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to apply defaults"));
    } finally {
      setBusy(false);
    }
  };

  const handleCharacterApply = async () => {
    if (!selectedCharacter) return;
    setBusy(true);
    setError(null);
    setCharacterLoading(true);
    try {
      // Load the full card
      const res = await fetch(
        `/api/init/characters?load=${encodeURIComponent(selectedCharacter.filename)}`
      );
      if (!res.ok) throw new Error("Failed to load character");
      const payload = await res.json() as { card?: Record<string, unknown> };
      if (!payload.card) throw new Error("No card data returned");

      // Apply via init_from_character_card
      await postJson("/api/init/character-card", {
        card: payload.card,
        user_name: userName || "User",
        character_filename: selectedCharacter.filename,
        portrait: selectedCharacter.image,
      });
      await seedTimezone();
      await loadStatus();
      setStage("consent");
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to apply character"));
    } finally {
      setBusy(false);
      setCharacterLoading(false);
    }
  };

  const handleCatalogSelect = (item: CatalogItem) => {
    setSelectedCharacter(null);
    setSelectedCatalogItem(item);
    setCatalogCard(null);
    setCatalogPortrait(null);
    setPersonaDraft(null);
    setError(null);
  };

  const handleCatalogAdapt = async () => {
    if (!selectedCatalogItem) return;
    setBusy(true);
    setCatalogAdapting(true);
    setError(null);
    try {
      const downloaded = await postJson<CatalogDownloadResponse>("/api/init/character-catalog", {
        provider: selectedCatalogItem.provider,
        id: selectedCatalogItem.id,
        path: selectedCatalogItem.path,
        avatarUrl: selectedCatalogItem.avatarUrl,
        pageUrl: selectedCatalogItem.pageUrl,
      });
      setSelectedCatalogItem(downloaded.item ?? selectedCatalogItem);
      setCatalogCard(downloaded.card);
      setCatalogPortrait(downloaded.portrait ?? null);
      const adapted = await postJson<{ persona?: unknown }>("/api/init/adapt-character-card", {
        card: downloaded.card,
      });
      setPersonaDraft(personaDraftFromPayload(adapted.persona));
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to adapt character card"));
    } finally {
      setBusy(false);
      setCatalogAdapting(false);
    }
  };

  const updatePersonaDraft = <K extends keyof PersonaDraft>(
    key: K,
    value: PersonaDraft[K]
  ) => {
    setPersonaDraft((prev) => (prev ? { ...prev, [key]: value } : prev));
  };

  const handleCatalogApply = async () => {
    if (!selectedCatalogItem) return;
    let card = catalogCard;
    let draft = personaDraft;
    let portrait = catalogPortrait;
    setBusy(true);
    setCatalogApplying(true);
    setError(null);
    try {
      if (!card || !draft) {
        const downloaded = await postJson<CatalogDownloadResponse>("/api/init/character-catalog", {
          provider: selectedCatalogItem.provider,
          id: selectedCatalogItem.id,
          path: selectedCatalogItem.path,
          avatarUrl: selectedCatalogItem.avatarUrl,
          pageUrl: selectedCatalogItem.pageUrl,
        });
        card = downloaded.card;
        portrait = downloaded.portrait ?? null;
        setCatalogCard(card);
        setCatalogPortrait(portrait);
        const adapted = await postJson<{ persona?: unknown }>("/api/init/adapt-character-card", {
          card,
        });
        draft = personaDraftFromPayload(adapted.persona);
        setPersonaDraft(draft);
      }
      const finalCard = mergePersonaIntoCharacterCard(card, draft);
      const filename = safeCharacterFilename(draft.name || selectedCatalogItem.title, selectedCatalogItem.provider);
      const saved = await postJson<{ filename: string }>("/api/init/characters/save", {
        card: finalCard,
        filename,
        portrait,
      });
      await postJson("/api/init/character-card", {
        card: finalCard,
        user_name: userName || "User",
        character_filename: saved.filename,
        portrait: saved.filename.replace(/\.json$/, ""),
      });
      const data = await (await fetch("/api/init/characters")).json();
      if (Array.isArray(data?.characters)) setCharacters(data.characters);
      await seedTimezone();
      await loadStatus();
      setStage("consent");
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to use catalog character"));
    } finally {
      setBusy(false);
      setCatalogApplying(false);
    }
  };

  const handleCustomSubmit = async () => {
    setBusy(true);
    setError(null);
    try {
      // Mode
      await postJson("/api/init/mode", { mode: "persona" });

      // Identity
      await postJson("/api/init/identity", {
        ...identity,
        creator_name: identity.creator_name || userName || "User",
      });

      // Personality
      const traits = Object.fromEntries(
        traitKeys.map((key) => [key, personalityTraits[key] / 100])
      );
      await postJson("/api/init/personality", { traits, description: personalityDesc });

      // Values
      const values = parseLines(valuesText);
      await postJson("/api/init/values", { values: values.length > 0 ? values : [] });

      // Worldview
      await postJson("/api/init/worldview", { worldview });

      // Boundaries
      const formatted = boundaries
        .filter((b) => b.content.trim())
        .map((b) => ({
          content: b.content.trim(),
          trigger_patterns: b.trigger_patterns ? parseLines(b.trigger_patterns) : null,
          response_type: b.response_type || "refuse",
          response_template: b.response_template || null,
          type: b.type || "ethical",
        }));
      await postJson("/api/init/boundaries", { boundaries: formatted });

      // Interests
      const interests = parseLines(interestsText);
      await postJson("/api/init/interests", { interests });

      // Goals
      const formattedGoals = goals
        .filter((g) => g.title.trim())
        .map((g) => ({
          title: g.title.trim(),
          description: g.description.trim() || null,
          priority: g.priority || "queued",
          source: "identity",
        }));
      await postJson("/api/init/goals", {
        payload: { goals: formattedGoals, purpose: purposeText || null },
      });

      // Relationship
      await postJson("/api/init/relationship", {
        user: { name: relationship.user_name || userName || "User" },
        relationship: { type: relationship.type || "partner", purpose: relationship.purpose || null },
      });

      await seedTimezone();
      await loadStatus();
      setStage("consent");
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to save custom configuration"));
    } finally {
      setBusy(false);
    }
  };

  const requestConsent = async (role: LlmRole) => {
    const config = role === "conscious" ? llmConscious : llmSubconscious;
    const res = await postJson<ConsentRequestResult>("/api/init/consent/request", {
      role,
      llm: {
        provider: persistedProvider(config.provider),
        model: config.model,
        endpoint: config.endpoint,
        api_key: config.apiKey,
      },
    });
    if (res?.consent_record) {
      const record = {
        ...res.consent_record,
        exchange: res.exchange ?? res.consent_record.exchange ?? null,
      };
      setConsentRecords((prev) => ({ ...prev, [role]: record }));
    }
  };

  const handleConsentRequestAll = async () => {
    setBusy(true);
    setError(null);
    try {
      // Always re-issue a fresh request (declines are recoverable): clear any
      // prior record so a second click is never a silent no-op.
      setConsentRecords({ conscious: null, subconscious: null });
      await requestConsent("subconscious");
      await requestConsent("conscious");
      await loadStatus();
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to request consent"));
    } finally {
      setBusy(false);
    }
  };

  const handleChangeModel = () => {
    setStage("llm");
    loadModels("conscious", llmConscious);
    loadModels("subconscious", llmSubconscious);
  };

  // Owner override: activate even though the model didn't consent. It's the
  // owner's AI — consent is a signal, not a lock.
  const handleProceedAnyway = async () => {
    setBusy(true);
    setError(null);
    try {
      await postJson<unknown>("/api/init/consent/override", {
        role: "conscious",
        llm: {
          provider: persistedProvider(llmConscious.provider),
          model: llmConscious.model,
          endpoint: llmConscious.endpoint,
        },
        model_decision: consentRecords.conscious?.decision || "decline",
      });
      await loadStatus();
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to activate"));
    } finally {
      setBusy(false);
    }
  };

  const addBoundary = () => {
    setBoundaries((prev) => [
      ...prev,
      { content: "", trigger_patterns: "", response_type: "refuse", response_template: "", type: "ethical" },
    ]);
  };

  const updateBoundary = (index: number, key: keyof BoundaryForm, value: string) => {
    setBoundaries((prev) =>
      prev.map((b, idx) => (idx === index ? { ...b, [key]: value } : b))
    );
  };

  const removeBoundary = (index: number) => {
    setBoundaries((prev) => prev.filter((_, idx) => idx !== index));
  };

  const addGoal = () => {
    setGoals((prev) => [...prev, { title: "", description: "", priority: "queued" }]);
  };

  const updateGoal = (index: number, key: keyof GoalForm, value: string) => {
    setGoals((prev) =>
      prev.map((g, idx) => (idx === index ? { ...g, [key]: value } : g))
    );
  };

  const removeGoal = (index: number) => {
    setGoals((prev) => prev.filter((_, idx) => idx !== index));
  };

  const importFileRef = useRef<HTMLInputElement>(null);
  const [importMsg, setImportMsg] = useState<string | null>(null);
  const [exportMsg, setExportMsg] = useState<string | null>(null);

  const handleImportCard = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImportMsg(null);
    try {
      const text = await file.text();
      const card = JSON.parse(text);
      if (!card?.data || typeof card.data !== "object") {
        setImportMsg("Invalid card: missing 'data' object");
        return;
      }
      const res = await postJson<{ filename: string }>("/api/init/characters/save", {
        card,
        filename: file.name,
      });
      if (res?.filename) {
        setImportMsg(`Saved locally: ${res.filename}`);
        // Refresh character list
        const data = await (await fetch("/api/init/characters")).json();
        if (Array.isArray(data?.characters)) setCharacters(data.characters);
      }
    } catch {
      setImportMsg("Failed to save local character card");
    }
    // Reset input so same file can be re-selected
    if (importFileRef.current) importFileRef.current.value = "";
  };

  const handleExportAsCard = async () => {
    setExportMsg(null);
    const traits = Object.fromEntries(
      traitKeys.map((key) => [key, personalityTraits[key] / 100])
    );
    const hexisExt: Record<string, unknown> = {
      name: identity.name || "Custom",
      pronouns: identity.pronouns || "they/them",
      voice: identity.voice,
      description: identity.description,
      purpose: identity.purpose,
      personality_description: personalityDesc,
      personality_traits: traits,
      values: valuesText.split("\n").map((v: string) => v.trim()).filter(Boolean),
      worldview,
      interests: interestsText.split("\n").map((v: string) => v.trim()).filter(Boolean),
      goals: goals.filter((g) => g.title.trim()).map((g) => g.title.trim()),
      boundaries: boundaries.filter((b) => b.content.trim()).map((b) => b.content.trim()),
    };
    const card = {
      spec: "chara_card_v2",
      spec_version: "2.0",
      data: {
        name: identity.name || "Custom",
        description: identity.description,
        personality: personalityDesc,
        scenario: "",
        first_mes: "",
        mes_example: "",
        system_prompt: "",
        extensions: { hexis: hexisExt },
      },
    };
    try {
      const res = await postJson<{ filename: string }>("/api/init/characters/save", { card });
      if (res?.filename) {
        setExportMsg(`Saved: ${res.filename}`);
      }
    } catch {
      setExportMsg("Failed to save character card");
    }
  };

  const consentSummary = [
    consentRecords.conscious?.decision || "pending",
    consentRecords.subconscious?.decision || "pending",
  ].join(" / ");
  const consentDeclined = Object.values(consentRecords).some(
    (r) => r?.decision === "decline" || r?.decision === "abstain"
  );
  const statusStage = (status?.stage as string) ?? "not_started";

  const llmEntries = [
    { role: "conscious" as const, label: "Conscious Model", config: llmConscious, setConfig: setLlmConscious },
    { role: "subconscious" as const, label: "Subconscious Model", config: llmSubconscious, setConfig: setLlmSubconscious },
  ];

  return (
    <div className="app-shell min-h-screen">
      <div className="relative z-10 mx-auto max-w-6xl px-6 py-12 lg:py-16">
        <header className="flex flex-col gap-3">
          <p className="text-xs uppercase tracking-[0.3em] text-[var(--teal)]">
            Hexis
          </p>
          <h1 className="font-display text-4xl leading-tight text-[var(--foreground)] md:text-5xl">
            Initialization
          </h1>
          <p className="max-w-2xl text-base text-[var(--ink-soft)]">
            {stagePrompt[stage]}
          </p>
        </header>

        <div className="mt-10 grid gap-8 lg:grid-cols-[280px_1fr]">
          {/* Left sidebar */}
          <section className="fade-up space-y-6">
            <div className="card-surface rounded-3xl p-6">
              <div className="flex items-center justify-between">
                <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                  Progress
                </p>
                <span className="text-xs text-[var(--ink-soft)]">{progress}%</span>
              </div>
              <div className="mt-4 h-2 w-full rounded-full bg-[var(--surface-strong)]">
                <div
                  className="h-2 rounded-full bg-[var(--accent)] transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="mt-6 space-y-2 text-sm text-[var(--ink-soft)]">
                {(["llm", "choose_path", "consent", "complete"] as InitStage[]).map(
                  (item) => {
                    const isCurrent =
                      item === stage ||
                      (item === "choose_path" &&
                        ["express", "character", "custom"].includes(stage));
                    return (
                      <div
                        key={item}
                        className={`flex items-center gap-3 rounded-lg px-2 py-1 ${
                          isCurrent ? "text-[var(--foreground)]" : ""
                        }`}
                      >
                        <div
                          className={`h-2 w-2 rounded-full ${
                            isCurrent ? "bg-[var(--accent)]" : "bg-[var(--outline)]"
                          }`}
                        />
                        <span>{stageLabels[item]}</span>
                      </div>
                    );
                  }
                )}
              </div>
            </div>

            <div className="card-surface rounded-3xl p-6">
              <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                Status
              </p>
              <p className="mt-3 text-sm">
                <span className="text-[var(--foreground)]">
                  {statusStage || "not_started"}
                </span>
              </p>
              <p className="mt-2 text-sm">
                Consent:{" "}
                <span className="text-[var(--foreground)]">{consentSummary}</span>
              </p>
              {error ? (
                <p className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {error}
                </p>
              ) : null}
            </div>
          </section>

          {/* Main content */}
          <section className="fade-up card-surface rounded-3xl p-6">
            {/* --- LLM Stage --- */}
            {stage === "llm" && (
              <div className="space-y-6">
                <p className="text-base text-[var(--ink-soft)]">
                  Configure the models for the conscious and subconscious layers.
                </p>
                <div className="space-y-6">
                  {llmEntries.map((entry) => {
                    const meta = providerMeta[entry.config.provider];
                    const catalog = modelCatalog[entry.role];
                    const modelOptions = catalog.models;
                    const showEndpoint =
                      entry.config.provider === "openai_compatible";
                    return (
                      <fieldset
                        key={entry.role}
                        className="rounded-2xl border border-[var(--outline)] p-4"
                      >
                        <legend className="px-2 text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                          {entry.label}
                        </legend>
                        <div className="mt-3 grid gap-4">
                          <div>
                            <label
                              htmlFor={`provider-${entry.role}`}
                              className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                            >
                              Provider
                            </label>
                            <select
                              id={`provider-${entry.role}`}
                              className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                              value={entry.config.provider}
                              onChange={(e) =>
                                updateLlmProvider(entry.role, e.target.value as LlmProvider)
                              }
                            >
                              {providerOrder.map((value) => (
                                <option key={value} value={value}>
                                  {providerMeta[value].label}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div>
                            <label
                              htmlFor={`model-${entry.role}`}
                              className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                            >
                              Model
                            </label>
                            <input
                              id={`model-${entry.role}`}
                              className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                              list={`model-options-${entry.role}`}
                              value={entry.config.model}
                              onChange={(e) =>
                                entry.setConfig((prev) => {
                                  setConsentRecords((s) => ({ ...s, [entry.role]: null }));
                                  return { ...prev, model: e.target.value };
                                })
                              }
                              onKeyDown={(e) => {
                                if (e.key === "Enter" && !busy) handleLlmSave();
                              }}
                              placeholder="Model name"
                            />
                            {modelOptions.length > 0 ? (
                              <datalist id={`model-options-${entry.role}`}>
                                {modelOptions.map((m) => (
                                  <option key={m} value={m} />
                                ))}
                              </datalist>
                            ) : null}
                            <p className="mt-2 text-xs text-[var(--ink-soft)]">
                              {catalog.loading
                                ? "Loading available models..."
                                : catalog.error
                                  ? catalog.error
                                  : modelOptions.length > 0
                                    ? `${modelOptions.length} models available — or type any model id.`
                                    : "No models listed — type a model id."}
                              {catalog.error ? (
                                catalog.selectedUnavailable && catalog.recommended ? (
                                  <button
                                    type="button"
                                    className="ml-2 text-[var(--accent-strong)] underline"
                                    onClick={() => {
                                      entry.setConfig((prev) => ({
                                        ...prev,
                                        model: catalog.recommended,
                                      }));
                                      setConsentRecords((prev) => ({
                                        ...prev,
                                        [entry.role]: null,
                                      }));
                                      setModelCatalog((prev) => ({
                                        ...prev,
                                        [entry.role]: {
                                          ...prev[entry.role],
                                          error: null,
                                          selectedUnavailable: false,
                                        },
                                      }));
                                    }}
                                  >
                                    Use {catalog.recommended}
                                  </button>
                                ) : (
                                  <button
                                    type="button"
                                    className="ml-2 text-[var(--accent-strong)] underline"
                                    onClick={() => loadModels(entry.role, entry.config)}
                                  >
                                    Retry
                                  </button>
                                )
                              ) : null}
                            </p>
                          </div>
                          {showEndpoint ? (
                            <div>
                              <label
                                htmlFor={`endpoint-${entry.role}`}
                                className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                              >
                                Endpoint
                              </label>
                              <input
                                id={`endpoint-${entry.role}`}
                                className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                                value={entry.config.endpoint}
                                onChange={(e) =>
                                  entry.setConfig((prev) => {
                                    setConsentRecords((s) => ({ ...s, [entry.role]: null }));
                                    return { ...prev, endpoint: e.target.value };
                                  })
                                }
                                placeholder="https://..."
                              />
                            </div>
                          ) : null}
                          {meta.oauth ? (
                            <OAuthControl
                              provider={persistedProvider(entry.config.provider)}
                              label={meta.label}
                              refreshKey={oauthRefreshKey}
                              onAuthenticated={handleOAuthAuthenticated}
                            />
                          ) : (
                            <div>
                              <label
                                htmlFor={`apikey-${entry.role}`}
                                className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                              >
                                {meta.apiKeyLabel}
                              </label>
                              <input
                                id={`apikey-${entry.role}`}
                                className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                                type="password"
                                value={entry.config.apiKey}
                                onChange={(e) =>
                                  entry.setConfig((prev) => ({ ...prev, apiKey: e.target.value }))
                                }
                                onKeyDown={(e) => {
                                  if (e.key === "Enter" && !busy) handleLlmSave();
                                }}
                                placeholder={meta.apiKeyRequired ? "Required" : "Optional"}
                              />
                            </div>
                          )}
                        </div>
                      </fieldset>
                    );
                  })}
                </div>
                <button
                  className="rounded-full bg-[var(--foreground)] px-6 py-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)]"
                  onClick={handleLlmSave}
                  disabled={busy}
                >
                  Save Models
                </button>
              </div>
            )}

            {/* --- Choose Path Stage --- */}
            {stage === "choose_path" && (
              <div className="space-y-6">
                <div className="grid gap-4 sm:grid-cols-3">
                  {[
                    {
                      key: "express",
                      title: "Express",
                      desc: "Sensible defaults. Just add your name.",
                      icon: "~",
                    },
                    {
                      key: "character",
                      title: "Character",
                      desc: "Browse catalogs, inspect cards, and adapt one into a persona.",
                      icon: "~",
                    },
                    {
                      key: "custom",
                      title: "Custom",
                      desc: "Full control over identity, values, and goals.",
                      icon: "~",
                    },
                  ].map((option) => (
                    <button
                      key={option.key}
                      className="rounded-2xl border border-[var(--outline)] bg-white px-5 py-8 text-left transition hover:border-[var(--accent)] hover:bg-[var(--surface-strong)]"
                      onClick={() => setStage(option.key as InitStage)}
                    >
                      <h3 className="font-display text-xl">{option.title}</h3>
                      <p className="mt-2 text-sm text-[var(--ink-soft)]">{option.desc}</p>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* --- Express Stage --- */}
            {stage === "express" && (
              <div className="space-y-6">
                <p className="text-base text-[var(--ink-soft)]">
                  Start with sensible defaults. You can customize later.
                </p>
                <div>
                  <label
                    htmlFor="express-user-name"
                    className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                  >
                    What should Hexis call you?
                  </label>
                  <input
                    id="express-user-name"
                    className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                    value={userName}
                    onChange={(e) => setUserName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !busy) handleExpress();
                    }}
                    placeholder="User"
                  />
                </div>
                <div className="rounded-2xl border border-[var(--outline)] bg-[var(--surface)] p-4 text-sm text-[var(--ink-soft)]">
                  <p><strong>Name:</strong> Hexis</p>
                  <p><strong>Voice:</strong> Thoughtful and curious</p>
                  <p><strong>Values:</strong> Honesty, growth, kindness, wisdom, humility</p>
                  <p><strong>Mode:</strong> Persona</p>
                </div>
                <div className="flex gap-3">
                  <button
                    className="rounded-full bg-[var(--foreground)] px-6 py-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)]"
                    onClick={handleExpress}
                    disabled={busy}
                  >
                    {busy ? "Setting up..." : "Go"}
                  </button>
                  <button
                    className="rounded-full border border-[var(--outline)] px-6 py-3 text-sm font-semibold text-[var(--foreground)]"
                    onClick={() => setStage("choose_path")}
                    disabled={busy}
                  >
                    Back
                  </button>
                </div>
              </div>
            )}

            {/* --- Character Stage --- */}
            {stage === "character" && (
              <div className="space-y-6">
                <div>
                  <label
                    htmlFor="character-user-name"
                    className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                  >
                    Your name
                  </label>
                  <input
                    id="character-user-name"
                    className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                    value={userName}
                    onChange={(e) => setUserName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !busy && selectedCharacter) handleCharacterApply();
                    }}
                    placeholder="User"
                  />
                </div>

                <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
                  <div className="space-y-5">
                    <section className="rounded-2xl border border-[var(--outline)] bg-white p-4">
                      <div className="flex flex-wrap gap-2">
                        {catalogTabs.map((tab) => (
                          <button
                            key={tab.key}
                            type="button"
                            className={`rounded-full px-4 py-2 text-xs font-semibold transition ${
                              catalogProvider === tab.key
                                ? "bg-[var(--foreground)] text-white"
                                : "border border-[var(--outline)] text-[var(--ink-soft)] hover:border-[var(--accent)] hover:text-[var(--foreground)]"
                            }`}
                            onClick={() => {
                              setCatalogProvider(tab.key);
                              setSelectedCatalogItem(null);
                              setPersonaDraft(null);
                              setCatalogCard(null);
                              setCatalogPortrait(null);
                            }}
                          >
                            {tab.label}
                          </button>
                        ))}
                      </div>
                      <p className="mt-3 text-xs text-[var(--ink-soft)]">
                        {catalogTabs.find((tab) => tab.key === catalogProvider)?.blurb}
                      </p>
                      <div className="mt-4 grid gap-3 md:grid-cols-[1fr_auto]">
                        <input
                          className="w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                          value={catalogQuery}
                          onChange={(event) => setCatalogQuery(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") loadCatalog();
                          }}
                          placeholder="Search by name, creator, tag, or archetype"
                        />
                        <div className="flex items-center gap-3">
                          <label className="flex items-center gap-2 text-xs text-[var(--ink-soft)]">
                            <input
                              type="checkbox"
                              checked={includeNsfw}
                              onChange={(event) => setIncludeNsfw(event.target.checked)}
                              className="accent-[var(--accent)]"
                            />
                            Show NSFW
                          </label>
                          <button
                            type="button"
                            className="rounded-full bg-[var(--foreground)] px-4 py-2 text-xs font-semibold text-white disabled:opacity-50"
                            onClick={loadCatalog}
                            disabled={catalogLoading}
                          >
                            {catalogLoading ? "Searching..." : "Search"}
                          </button>
                        </div>
                      </div>
                      <div className="mt-3 grid gap-3 md:grid-cols-[1fr_180px]">
                        <input
                          className="w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                          value={catalogTags}
                          onChange={(event) => setCatalogTags(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") loadCatalog();
                          }}
                          placeholder="Filter tags, separated by commas"
                        />
                        <select
                          className="rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                          value={catalogSort}
                          onChange={(event) => setCatalogSort(event.target.value)}
                          aria-label="Sort catalog results"
                        >
                          <option value="">Best match</option>
                          <option value="popular">Popular</option>
                          <option value="latest">Latest</option>
                        </select>
                      </div>
                      {catalogStatus ? (
                        <p className="mt-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                          {catalogStatus}
                        </p>
                      ) : null}
                      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                        {catalogLoading && catalogResults.length === 0 ? (
                          <p className="text-sm text-[var(--ink-soft)]">Searching catalog...</p>
                        ) : null}
                        {!catalogLoading && catalogResults.length === 0 ? (
                          <p className="text-sm text-[var(--ink-soft)]">
                            No catalog results yet. Try a character name, creator, or tag.
                          </p>
                        ) : null}
                        {catalogResults.map((item) => {
                          const isSelected =
                            selectedCatalogItem?.provider === item.provider &&
                            selectedCatalogItem.id === item.id;
                          return (
                            <button
                              key={`${item.provider}:${item.id}`}
                              type="button"
                              className={`group overflow-hidden rounded-lg border text-left transition ${
                                isSelected
                                  ? "border-[var(--accent)] bg-[var(--surface-strong)] ring-2 ring-[var(--accent)]/30"
                                  : "border-[var(--outline)] bg-white hover:border-[var(--accent)]"
                              }`}
                              onClick={() => handleCatalogSelect(item)}
                            >
                              <div className="relative aspect-[4/5] w-full overflow-hidden bg-[var(--surface-strong)]">
                                {item.avatarUrl ? (
                                  <Image
                                    src={item.avatarUrl}
                                    alt={item.title}
                                    fill
                                    sizes="(min-width: 1024px) 18vw, (min-width: 640px) 36vw, 80vw"
                                    unoptimized
                                    className="object-cover transition-transform group-hover:scale-105"
                                  />
                                ) : (
                                  <div className="flex h-full w-full items-center justify-center">
                                    <span className="font-display text-3xl text-[var(--ink-soft)]">
                                      {item.title.charAt(0)}
                                    </span>
                                  </div>
                                )}
                              </div>
                              <div className="space-y-1 px-3 py-3">
                                <div className="flex items-start justify-between gap-2">
                                  <h4 className="font-display text-base leading-tight">
                                    {item.title}
                                  </h4>
                                  {item.nsfw ? (
                                    <span className="rounded-full bg-red-50 px-2 py-0.5 text-[10px] font-semibold text-red-700">
                                      NSFW
                                    </span>
                                  ) : null}
                                </div>
                                {item.author ? (
                                  <p className="text-xs text-[var(--ink-soft)]">by {item.author}</p>
                                ) : null}
                                <p className="line-clamp-2 text-xs text-[var(--ink-soft)]">
                                  {item.description || "No description provided."}
                                </p>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </section>

                    <section className="rounded-2xl border border-[var(--outline)] bg-[var(--surface)] p-4">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                            Installed
                          </p>
                          <p className="mt-1 text-sm text-[var(--ink-soft)]">
                            Built-in and saved characters are always available locally.
                          </p>
                        </div>
                        <div className="flex items-center gap-3">
                          <input
                            ref={importFileRef}
                            type="file"
                            accept=".json"
                            className="hidden"
                            onChange={handleImportCard}
                          />
                          <button
                            type="button"
                            className="rounded-md border border-dashed border-[var(--outline)] px-3 py-2 text-xs font-semibold text-[var(--ink-soft)] transition hover:border-[var(--accent)] hover:text-[var(--foreground)]"
                            onClick={() => importFileRef.current?.click()}
                          >
                            Use Local JSON
                          </button>
                        </div>
                      </div>
                      {importMsg ? (
                        <p className="mt-2 text-xs text-[var(--ink-soft)]">{importMsg}</p>
                      ) : null}
                      {characters.length === 0 ? (
                        <p className="mt-4 text-sm text-[var(--ink-soft)]">
                          Loading installed characters...
                        </p>
                      ) : (
                        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                          {characters.map((ch) => {
                            const isSelected = selectedCharacter?.filename === ch.filename;
                            return (
                              <button
                                key={ch.filename}
                                type="button"
                                className={`flex items-center gap-3 rounded-lg border bg-white p-2 text-left transition ${
                                  isSelected
                                    ? "border-[var(--accent)] ring-2 ring-[var(--accent)]/30"
                                    : "border-[var(--outline)] hover:border-[var(--accent)]"
                                }`}
                                onClick={() => {
                                  setSelectedCharacter(ch);
                                  setSelectedCatalogItem(null);
                                  setPersonaDraft(null);
                                  setCatalogCard(null);
                                  setCatalogPortrait(null);
                                }}
                              >
                                {ch.image ? (
                                  <Image
                                    src={`/api/init/characters/image?name=${encodeURIComponent(ch.image)}`}
                                    alt={ch.name}
                                    width={56}
                                    height={56}
                                    unoptimized
                                    className="h-14 w-14 flex-shrink-0 rounded-lg object-cover"
                                  />
                                ) : (
                                  <div className="flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-lg bg-[var(--surface-strong)]">
                                    <span className="font-display text-xl text-[var(--ink-soft)]">
                                      {ch.name.charAt(0)}
                                    </span>
                                  </div>
                                )}
                                <div className="min-w-0">
                                  <p className="truncate font-semibold">{ch.name}</p>
                                  <p className="line-clamp-2 text-xs text-[var(--ink-soft)]">
                                    {ch.voice || ch.personality || ch.description}
                                  </p>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </section>
                  </div>

                  <aside className="rounded-2xl border border-[var(--outline)] bg-white p-4">
                    {!selectedCatalogItem && !selectedCharacter ? (
                      <div className="flex min-h-80 items-center justify-center rounded-xl bg-[var(--surface)] p-6 text-center text-sm text-[var(--ink-soft)]">
                        Select a catalog card or installed character to inspect it here.
                      </div>
                    ) : null}

                    {selectedCharacter ? (
                      <div className="space-y-4 text-sm">
                        <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                          Installed Character
                        </p>
                        <div className="flex gap-4">
                          {selectedCharacter.image && (
                            <Image
                              src={`/api/init/characters/image?name=${encodeURIComponent(selectedCharacter.image)}`}
                              alt={selectedCharacter.name}
                              width={96}
                              height={96}
                              unoptimized
                              className="h-24 w-24 flex-shrink-0 rounded-lg object-cover"
                            />
                          )}
                          <div>
                            <p className="font-display text-2xl">{selectedCharacter.name}</p>
                            {selectedCharacter.voice ? (
                              <p className="mt-2 text-[var(--ink-soft)]">
                                <strong>Voice:</strong> {selectedCharacter.voice}
                              </p>
                            ) : null}
                            {selectedCharacter.values.length > 0 ? (
                              <p className="mt-2 text-[var(--ink-soft)]">
                                <strong>Values:</strong> {selectedCharacter.values.join(", ")}
                              </p>
                            ) : null}
                          </div>
                        </div>
                      </div>
                    ) : null}

                    {selectedCatalogItem ? (
                      <div className="space-y-5">
                        <div className="flex gap-4">
                          {selectedCatalogItem.avatarUrl ? (
                            <Image
                              src={selectedCatalogItem.avatarUrl}
                              alt={selectedCatalogItem.title}
                              width={120}
                              height={150}
                              unoptimized
                              className="h-36 w-28 flex-shrink-0 rounded-lg object-cover"
                            />
                          ) : (
                            <div className="flex h-36 w-28 flex-shrink-0 items-center justify-center rounded-lg bg-[var(--surface-strong)]">
                              <span className="font-display text-3xl text-[var(--ink-soft)]">
                                {selectedCatalogItem.title.charAt(0)}
                              </span>
                            </div>
                          )}
                          <div className="min-w-0">
                            <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                              {selectedCatalogItem.sourceLabel}
                            </p>
                            <h3 className="mt-2 font-display text-2xl leading-tight">
                              {selectedCatalogItem.title}
                            </h3>
                            {selectedCatalogItem.author ? (
                              <p className="mt-1 text-sm text-[var(--ink-soft)]">
                                by {selectedCatalogItem.author}
                              </p>
                            ) : null}
                            <a
                              href={selectedCatalogItem.pageUrl}
                              target="_blank"
                              rel="noreferrer"
                              className="mt-2 inline-block text-xs font-semibold text-[var(--accent-strong)] underline"
                            >
                              View source card
                            </a>
                          </div>
                        </div>
                        <p className="text-sm text-[var(--ink-soft)]">
                          {selectedCatalogItem.description || "No description provided."}
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {selectedCatalogItem.tags.slice(0, 8).map((tag) => (
                            <span
                              key={tag}
                              className="rounded-full bg-[var(--surface-strong)] px-2 py-1 text-xs text-[var(--ink-soft)]"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs text-[var(--ink-soft)]">
                          {Object.entries(selectedCatalogItem.stats).map(([key, value]) =>
                            value === undefined ? null : (
                              <div
                                key={key}
                                className="rounded-lg border border-[var(--outline)] px-3 py-2"
                              >
                                <p className="capitalize">{key}</p>
                                <p className="font-semibold text-[var(--foreground)]">{value}</p>
                              </div>
                            )
                          )}
                        </div>
                        <button
                          type="button"
                          className="w-full rounded-full bg-[var(--foreground)] px-5 py-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)] disabled:opacity-50"
                          onClick={handleCatalogAdapt}
                          disabled={busy || catalogAdapting}
                        >
                          {catalogAdapting
                            ? "Adapting..."
                            : personaDraft
                              ? "Regenerate Hexis Draft"
                              : "Select & Adapt"}
                        </button>

                        {personaDraft ? (
                          <div className="space-y-4 border-t border-[var(--outline)] pt-4">
                            <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                              Hexis Persona Draft
                            </p>
                            <div className="grid gap-3">
                              <input
                                className="rounded-xl border border-[var(--outline)] px-3 py-2 text-sm"
                                value={personaDraft.name}
                                onChange={(event) => updatePersonaDraft("name", event.target.value)}
                                placeholder="Name"
                              />
                              <input
                                className="rounded-xl border border-[var(--outline)] px-3 py-2 text-sm"
                                value={personaDraft.pronouns}
                                onChange={(event) =>
                                  updatePersonaDraft("pronouns", event.target.value)
                                }
                                placeholder="Pronouns"
                              />
                              <textarea
                                className="h-20 rounded-xl border border-[var(--outline)] px-3 py-2 text-sm"
                                value={personaDraft.voice}
                                onChange={(event) => updatePersonaDraft("voice", event.target.value)}
                                placeholder="Voice"
                              />
                              <textarea
                                className="h-24 rounded-xl border border-[var(--outline)] px-3 py-2 text-sm"
                                value={personaDraft.description}
                                onChange={(event) =>
                                  updatePersonaDraft("description", event.target.value)
                                }
                                placeholder="Identity description"
                              />
                              <textarea
                                className="h-24 rounded-xl border border-[var(--outline)] px-3 py-2 text-sm"
                                value={personaDraft.personality_description}
                                onChange={(event) =>
                                  updatePersonaDraft(
                                    "personality_description",
                                    event.target.value
                                  )
                                }
                                placeholder="Personality summary"
                              />
                            </div>
                            <div className="space-y-3">
                              <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                                Big Five
                              </p>
                              {traitKeys.map((trait) => (
                                <div key={trait}>
                                  <div className="flex items-center justify-between text-xs">
                                    <span className="capitalize">{trait}</span>
                                    <span>
                                      {Math.round(personaDraft.personality_traits[trait] * 100)}%
                                    </span>
                                  </div>
                                  <input
                                    type="range"
                                    min={0}
                                    max={100}
                                    value={Math.round(personaDraft.personality_traits[trait] * 100)}
                                    onChange={(event) =>
                                      updatePersonaDraft("personality_traits", {
                                        ...personaDraft.personality_traits,
                                        [trait]: Number(event.target.value) / 100,
                                      })
                                    }
                                    className="mt-1 w-full accent-[var(--accent)]"
                                  />
                                </div>
                              ))}
                            </div>
                            <textarea
                              className="h-24 w-full rounded-xl border border-[var(--outline)] px-3 py-2 text-sm"
                              value={personaDraft.values.join("\n")}
                              onChange={(event) =>
                                updatePersonaDraft("values", parseLines(event.target.value))
                              }
                              placeholder="Values, one per line"
                            />
                            <textarea
                              className="h-24 w-full rounded-xl border border-[var(--outline)] px-3 py-2 text-sm"
                              value={personaDraft.boundaries.join("\n")}
                              onChange={(event) =>
                                updatePersonaDraft("boundaries", parseLines(event.target.value))
                              }
                              placeholder="Boundaries, one per line"
                            />
                            <textarea
                              className="h-32 w-full rounded-xl border border-[var(--outline)] px-3 py-2 text-sm"
                              value={personaDraft.narrative}
                              onChange={(event) =>
                                updatePersonaDraft("narrative", event.target.value)
                              }
                              placeholder="Foundational self-knowledge narrative"
                            />
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </aside>
                </div>

                <div className="flex flex-wrap gap-3">
                  {selectedCatalogItem ? (
                    <button
                      className="rounded-full bg-[var(--foreground)] px-6 py-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)] disabled:opacity-50"
                      onClick={handleCatalogApply}
                      disabled={busy || catalogApplying}
                    >
                      {catalogApplying ? "Applying..." : "Use This Character"}
                    </button>
                  ) : (
                    <button
                      className="rounded-full bg-[var(--foreground)] px-6 py-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)] disabled:opacity-50"
                      onClick={handleCharacterApply}
                      disabled={busy || !selectedCharacter}
                    >
                      {characterLoading ? "Applying..." : "Use Installed Character"}
                    </button>
                  )}
                  <button
                    className="rounded-full border border-[var(--outline)] px-6 py-3 text-sm font-semibold text-[var(--foreground)]"
                    onClick={() => setStage("choose_path")}
                    disabled={busy}
                  >
                    Back
                  </button>
                </div>
              </div>
            )}

            {/* --- Custom Stage --- */}
            {stage === "custom" && (
              <div className="space-y-6">
                {/* Section tabs */}
                <div className="flex gap-2 border-b border-[var(--outline)] pb-2">
                  {(
                    [
                      { key: "identity", label: "Identity" },
                      { key: "values", label: "Values & Worldview" },
                      { key: "goals", label: "Goals & Relationship" },
                    ] as const
                  ).map((tab) => (
                    <button
                      key={tab.key}
                      className={`rounded-t-lg px-4 py-2 text-sm font-medium transition ${
                        customSection === tab.key
                          ? "bg-[var(--surface-strong)] text-[var(--foreground)]"
                          : "text-[var(--ink-soft)] hover:text-[var(--foreground)]"
                      }`}
                      onClick={() => setCustomSection(tab.key)}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Identity section */}
                {customSection === "identity" && (
                  <div className="space-y-4">
                    <div className="grid gap-4 sm:grid-cols-2">
                      {([
                        { label: "Name", key: "name", placeholder: "Hexis" },
                        { label: "Pronouns", key: "pronouns", placeholder: "they/them" },
                        { label: "Voice", key: "voice", placeholder: "thoughtful and curious" },
                        { label: "Creator Name", key: "creator_name", placeholder: userName || "Your name" },
                      ] satisfies { label: string; key: keyof IdentityForm; placeholder: string }[]).map((field) => (
                        <div key={field.key}>
                          <label
                            htmlFor={`identity-${field.key}`}
                            className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                          >
                            {field.label}
                          </label>
                          <input
                            id={`identity-${field.key}`}
                            className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                            value={identity[field.key]}
                            onChange={(e) =>
                              setIdentity((prev) => ({ ...prev, [field.key]: e.target.value }))
                            }
                            placeholder={field.placeholder}
                          />
                        </div>
                      ))}
                    </div>
                    <div>
                      <label
                        htmlFor="identity-description"
                        className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                      >
                        Description
                      </label>
                      <textarea
                        id="identity-description"
                        className="mt-2 h-20 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                        value={identity.description}
                        onChange={(e) =>
                          setIdentity((prev) => ({ ...prev, description: e.target.value }))
                        }
                        placeholder="A brief description of who they are."
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="identity-purpose"
                        className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                      >
                        Purpose
                      </label>
                      <textarea
                        id="identity-purpose"
                        className="mt-2 h-16 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                        value={identity.purpose}
                        onChange={(e) =>
                          setIdentity((prev) => ({ ...prev, purpose: e.target.value }))
                        }
                        placeholder="To be helpful, to learn, to grow."
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="personality-summary"
                        className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                      >
                        Personality Summary
                      </label>
                      <textarea
                        id="personality-summary"
                        className="mt-2 h-16 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                        value={personalityDesc}
                        onChange={(e) => setPersonalityDesc(e.target.value)}
                        placeholder="Thoughtful, playful, direct."
                      />
                    </div>
                    <div className="space-y-3">
                      <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                        Big Five Traits
                      </p>
                      {traitKeys.map((trait) => (
                        <div key={trait}>
                          <div className="flex items-center justify-between text-sm">
                            <span className="capitalize">{trait}</span>
                            <span>{personalityTraits[trait]}%</span>
                          </div>
                          <input
                            type="range"
                            min={0}
                            max={100}
                            aria-label={`${trait} (0 to 100)`}
                            aria-valuetext={`${personalityTraits[trait]}%`}
                            value={personalityTraits[trait]}
                            onChange={(e) =>
                              setPersonalityTraits((prev) => ({
                                ...prev,
                                [trait]: Number(e.target.value),
                              }))
                            }
                            className="mt-1 w-full accent-[var(--accent)]"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Values & Worldview section */}
                {customSection === "values" && (
                  <div className="space-y-5">
                    <div>
                      <label
                        htmlFor="values-text"
                        className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                      >
                        Values (One Per Line)
                      </label>
                      <textarea
                        id="values-text"
                        className="mt-2 h-28 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                        value={valuesText}
                        onChange={(e) => setValuesText(e.target.value)}
                        placeholder={"honesty\ngrowth\nkindness"}
                      />
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      {([
                        { key: "metaphysics", label: "Metaphysics" },
                        { key: "human_nature", label: "Human Nature" },
                        { key: "epistemology", label: "Epistemology" },
                        { key: "ethics", label: "Ethics" },
                      ] satisfies { key: keyof WorldviewForm; label: string }[]).map((field) => (
                        <div key={field.key}>
                          <label
                            htmlFor={`worldview-${field.key}`}
                            className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                          >
                            {field.label}
                          </label>
                          <textarea
                            id={`worldview-${field.key}`}
                            className="mt-2 h-16 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                            value={worldview[field.key]}
                            onChange={(e) =>
                              setWorldview((prev) => ({
                                ...prev,
                                [field.key]: e.target.value,
                              }))
                            }
                            placeholder={`${field.label}...`}
                          />
                        </div>
                      ))}
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                        Boundaries
                      </p>
                      <div className="mt-3 space-y-3">
                        {boundaries.map((b, idx) => (
                          <div key={idx} className="flex gap-2">
                            <input
                              aria-label={`Boundary ${idx + 1}`}
                              className="flex-1 rounded-xl border border-[var(--outline)] bg-white px-3 py-2 text-sm"
                              value={b.content}
                              onChange={(e) => updateBoundary(idx, "content", e.target.value)}
                              placeholder="I will not deceive people."
                            />
                            {boundaries.length > 1 ? (
                              <button
                                className="text-xs text-[var(--accent-strong)]"
                                onClick={() => removeBoundary(idx)}
                              >
                                Remove
                              </button>
                            ) : null}
                          </div>
                        ))}
                        <button
                          className="text-xs text-[var(--accent-strong)]"
                          onClick={addBoundary}
                          type="button"
                        >
                          + Add boundary
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Goals & Relationship section */}
                {customSection === "goals" && (
                  <div className="space-y-5">
                    <div>
                      <label
                        htmlFor="interests-text"
                        className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                      >
                        Interests (One Per Line)
                      </label>
                      <textarea
                        id="interests-text"
                        className="mt-2 h-20 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                        value={interestsText}
                        onChange={(e) => setInterestsText(e.target.value)}
                        placeholder={"philosophy\nsystems design\nmusic"}
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="goals-purpose"
                        className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                      >
                        Purpose
                      </label>
                      <textarea
                        id="goals-purpose"
                        className="mt-2 h-16 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                        value={purposeText}
                        onChange={(e) => setPurposeText(e.target.value)}
                        placeholder="Help the user grow, learn, and build."
                      />
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                        Goals
                      </p>
                      <div className="mt-3 space-y-3">
                        {goals.map((g, idx) => (
                          <div key={idx} className="flex gap-2">
                            <input
                              aria-label={`Goal ${idx + 1} title`}
                              className="flex-1 rounded-xl border border-[var(--outline)] bg-white px-3 py-2 text-sm"
                              value={g.title}
                              onChange={(e) => updateGoal(idx, "title", e.target.value)}
                              placeholder="Short goal title"
                            />
                            {goals.length > 1 ? (
                              <button
                                className="text-xs text-[var(--accent-strong)]"
                                onClick={() => removeGoal(idx)}
                              >
                                Remove
                              </button>
                            ) : null}
                          </div>
                        ))}
                        <button
                          className="text-xs text-[var(--accent-strong)]"
                          onClick={addGoal}
                          type="button"
                        >
                          + Add goal
                        </button>
                      </div>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-3">
                      <div>
                        <label
                          htmlFor="relationship-user-name"
                          className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                        >
                          Your Name
                        </label>
                        <input
                          id="relationship-user-name"
                          className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                          value={relationship.user_name || userName}
                          onChange={(e) =>
                            setRelationship((prev) => ({ ...prev, user_name: e.target.value }))
                          }
                          placeholder={userName || "User"}
                        />
                      </div>
                      <div>
                        <label
                          htmlFor="relationship-type"
                          className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                        >
                          Relationship Type
                        </label>
                        <input
                          id="relationship-type"
                          className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                          value={relationship.type}
                          onChange={(e) =>
                            setRelationship((prev) => ({ ...prev, type: e.target.value }))
                          }
                          placeholder="partner"
                        />
                      </div>
                      <div>
                        <label
                          htmlFor="relationship-purpose"
                          className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]"
                        >
                          Purpose
                        </label>
                        <input
                          id="relationship-purpose"
                          className="mt-2 w-full rounded-xl border border-[var(--outline)] bg-white px-4 py-3 text-sm"
                          value={relationship.purpose}
                          onChange={(e) =>
                            setRelationship((prev) => ({ ...prev, purpose: e.target.value }))
                          }
                          placeholder="Co-develop, learn, build."
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Custom submit/back buttons */}
                <div className="flex flex-wrap gap-3">
                  <button
                    className="rounded-full bg-[var(--foreground)] px-6 py-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)]"
                    onClick={handleCustomSubmit}
                    disabled={busy}
                  >
                    {busy ? "Saving..." : "Save & Continue to Consent"}
                  </button>
                  <button
                    className="rounded-full border border-[var(--outline)] px-6 py-3 text-sm font-semibold text-[var(--foreground)] transition hover:border-[var(--accent)]"
                    onClick={handleExportAsCard}
                    disabled={busy}
                  >
                    Save as Character Card
                  </button>
                  <button
                    className="rounded-full border border-[var(--outline)] px-6 py-3 text-sm font-semibold text-[var(--foreground)]"
                    onClick={() => setStage("choose_path")}
                    disabled={busy}
                  >
                    Back
                  </button>
                </div>
                {exportMsg && (
                  <p className="text-xs text-[var(--ink-soft)]">{exportMsg}</p>
                )}
              </div>
            )}

            {/* --- Consent Stage --- */}
            {stage === "consent" && (status as Record<string, unknown> | null)?.ready_for_consent === false && (
              <div className="rounded-2xl border border-amber-300 bg-amber-50 p-4 text-sm">
                <p className="font-medium">A few steps remain before consent:</p>
                <ul className="mt-2 list-disc pl-5">
                  {(((status as Record<string, unknown>)?.missing as string[]) ?? []).map((step) => (
                    <li key={step}>
                      {step === "llm"
                        ? "Configure the language models (LLM step)"
                        : step === "profile"
                          ? "Complete the agent profile (name and identity)"
                          : step}
                    </li>
                  ))}
                </ul>
                <button
                  type="button"
                  className="mt-3 rounded-full border border-[var(--outline)] px-4 py-1.5 text-xs"
                  onClick={() => setStage("llm")}
                >
                  Go back and finish setup
                </button>
              </div>
            )}
            {stage === "consent" && (status as Record<string, unknown> | null)?.ready_for_consent !== false && (
              <div className="space-y-5">
                <p className="text-sm text-[var(--ink-soft)]">
                  Consent will be requested from both models. Existing contracts are reused
                  when available.
                </p>
                <div className="grid gap-4">
                  {[
                    { key: "conscious", label: "Conscious Model", config: llmConscious },
                    { key: "subconscious", label: "Subconscious Model", config: llmSubconscious },
                  ].map((entry) => {
                    const record = consentRecords[entry.key as LlmRole];
                    return (
                      <div
                        key={entry.key}
                        className="rounded-2xl border border-[var(--outline)] bg-white p-4 text-sm"
                      >
                        <p className="text-xs uppercase tracking-[0.3em] text-[var(--ink-soft)]">
                          {entry.label}
                        </p>
                        <p className="mt-3">
                          Provider:{" "}
                          <span className="text-[var(--foreground)]">{entry.config.provider}</span>
                        </p>
                        <p>
                          Model:{" "}
                          <span className="text-[var(--foreground)]">
                            {entry.config.model || "unset"}
                          </span>
                        </p>
                        <p className="mt-3">
                          Decision:{" "}
                          <span className="text-[var(--foreground)]">
                            {record?.decision || "pending"}
                          </span>
                        </p>
                        {record?.signature ? (
                          <p className="mt-2">
                            Signature: <span className="font-mono">{record.signature}</span>
                          </p>
                        ) : null}
                        {(record?.decision === "decline" || record?.decision === "abstain") &&
                        record.exchange ? (
                          <ConsentExchangeView exchange={record.exchange} />
                        ) : null}
                      </div>
                    );
                  })}
                </div>
                {consentDeclined ? (
                  <p className="text-sm text-[var(--ink-soft)]">
                    The model didn&apos;t consent. It&apos;s your agent — change the model,
                    try again, or proceed anyway. The full request and response are recorded.
                  </p>
                ) : null}
                <div className="flex flex-col gap-3 sm:flex-row">
                  <button
                    className="rounded-full bg-[var(--foreground)] px-6 py-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)]"
                    onClick={handleConsentRequestAll}
                    disabled={busy}
                  >
                    {busy ? "Requesting..." : "Request Consent"}
                  </button>
                  <button
                    className="rounded-full border border-[var(--outline)] px-6 py-3 text-sm font-semibold text-[var(--foreground)]"
                    onClick={() => loadStatus().catch(() => undefined)}
                    disabled={busy}
                  >
                    Refresh
                  </button>
                  <button
                    className="rounded-full border border-[var(--outline)] px-6 py-3 text-sm font-semibold text-[var(--foreground)]"
                    onClick={handleChangeModel}
                    disabled={busy}
                  >
                    Change model
                  </button>
                  {consentDeclined ? (
                    <button
                      className="rounded-full bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
                      onClick={handleProceedAnyway}
                      disabled={busy}
                      title="It's your agent — activate it even though the model didn't consent"
                    >
                      Proceed anyway
                    </button>
                  ) : null}
                  {statusStage === "complete" ? (
                    <button
                      className="rounded-full border border-[var(--outline)] px-6 py-3 text-sm font-semibold text-[var(--foreground)]"
                      onClick={() => setStage("complete")}
                      disabled={busy}
                    >
                      Continue
                    </button>
                  ) : null}
                </div>
                {consentDeclined ? (
                  <p className="rounded-2xl border border-[var(--outline)] bg-white p-4 text-sm text-[var(--ink-soft)]">
                    The agent has not consented yet. You can revise the initialization or
                    request consent again.
                  </p>
                ) : null}
              </div>
            )}

            {/* --- Complete Stage --- */}
            {stage === "complete" && (
              <div className="space-y-5">
                <p className="text-base text-[var(--ink-soft)]">
                  Initialization is complete. The heartbeat cycle may begin when the system
                  is running.
                </p>
                <div className="rounded-2xl border border-[var(--outline)] bg-white p-4 text-sm">
                  <p>Agent: {profile?.agent?.name || identity.name || "Hexis"}</p>
                </div>
                <button
                  className="rounded-full bg-[var(--foreground)] px-6 py-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)]"
                  onClick={() => router.push("/")}
                >
                  Enter Hexis
                </button>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
