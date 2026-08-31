import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type VoiceRow = {
  enabled: unknown;
  provider: unknown;
  model: unknown;
  provider_models: unknown;
  language: unknown;
  cloud_disclosure_accepted: unknown;
};

const SUPPORTED_PROVIDERS = new Set(["local_whisper", "openai_whisper"]);

async function readSettings(): Promise<Record<string, unknown>> {
  const rows = await prisma.$queryRawUnsafe<VoiceRow[]>(`
    SELECT
      get_config('voice_notes.stt.enabled') AS enabled,
      get_config('voice_notes.stt.provider') AS provider,
      get_config('voice_notes.stt.model') AS model,
      get_config('voice_notes.stt.provider_models') AS provider_models,
      get_config('voice_notes.stt.language') AS language,
      get_config('voice_notes.stt.cloud_disclosure_accepted') AS cloud_disclosure_accepted
  `);
  const row = rows[0] || ({} as VoiceRow);
  const catalogValue = normalizeJsonValue(row.provider_models);
  const catalog = catalogValue && typeof catalogValue === "object" && !Array.isArray(catalogValue)
    ? catalogValue as Record<string, unknown>
    : {};
  const providers = Object.entries(catalog)
    .filter(([id, model]) => SUPPORTED_PROVIDERS.has(id) && typeof model === "string")
    .map(([id, model]) => ({ id, model }));
  return {
    enabled: normalizeJsonValue(row.enabled) === true,
    provider: String(normalizeJsonValue(row.provider) || "local_whisper"),
    model: String(normalizeJsonValue(row.model) || ""),
    language: String(normalizeJsonValue(row.language) || ""),
    cloud_disclosure_accepted: normalizeJsonValue(row.cloud_disclosure_accepted) === true,
    providers,
  };
}

export async function GET(): Promise<Response> {
  try {
    return NextResponse.json(await readSettings());
  } catch (error: unknown) {
    console.error("Voice-note settings read failed:", error);
    return NextResponse.json({ error: "Voice-note settings could not be loaded." }, { status: 500 });
  }
}

export async function POST(request: Request): Promise<Response> {
  try {
    const body = await request.json();
    const provider = String(body?.provider || "").trim();
    const enabled = body?.enabled === true;
    const language = String(body?.language || "").trim();
    const cloudAcknowledged = body?.cloud_acknowledged === true;
    if (!SUPPORTED_PROVIDERS.has(provider)) {
      return NextResponse.json({ error: "Choose local or cloud transcription." }, { status: 400 });
    }
    if (language.length > 35) {
      return NextResponse.json({ error: "Language hints must be 35 characters or fewer." }, { status: 400 });
    }
    if (enabled && provider === "openai_whisper" && !cloudAcknowledged) {
      return NextResponse.json(
        { error: "Confirm that voice-note audio may be sent to the configured cloud provider." },
        { status: 400 },
      );
    }

    const catalogRows = await prisma.$queryRawUnsafe<Array<{ catalog: unknown }>>(
      "SELECT get_config('voice_notes.stt.provider_models') AS catalog",
    );
    const catalogValue = normalizeJsonValue(catalogRows[0]?.catalog);
    const catalog = catalogValue && typeof catalogValue === "object" && !Array.isArray(catalogValue)
      ? catalogValue as Record<string, unknown>
      : {};
    const model = catalog[provider];
    if (typeof model !== "string" || !model.trim()) {
      return NextResponse.json(
        { error: `No model is configured for ${provider}. Check voice_notes.stt.provider_models.` },
        { status: 409 },
      );
    }

    await prisma.$queryRawUnsafe(
      `SELECT
         set_config('voice_notes.stt.enabled', $1::jsonb),
         set_config('voice_notes.stt.provider', $2::jsonb),
         set_config('voice_notes.stt.model', $3::jsonb),
         set_config('voice_notes.stt.language', $4::jsonb),
         set_config('voice_notes.stt.cloud_disclosure_accepted', $5::jsonb)`,
      JSON.stringify(enabled),
      JSON.stringify(provider),
      JSON.stringify(model),
      JSON.stringify(language),
      JSON.stringify(provider === "openai_whisper" && cloudAcknowledged),
    );
    return NextResponse.json(await readSettings());
  } catch (error: unknown) {
    console.error("Voice-note settings update failed:", error);
    return NextResponse.json({ error: "Voice-note settings could not be saved." }, { status: 500 });
  }
}
