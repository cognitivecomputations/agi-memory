import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type VoiceOutputRow = {
  enabled: unknown;
  provider: unknown;
  model: unknown;
  provider_models: unknown;
  voice: unknown;
  talk_enabled: unknown;
  wake_enabled: unknown;
};

const SUPPORTED_PROVIDERS = new Set(["local_piper"]);

async function readSettings(): Promise<Record<string, unknown>> {
  const rows = await prisma.$queryRawUnsafe<VoiceOutputRow[]>(`
    SELECT
      get_config('voice.tts.enabled') AS enabled,
      get_config('voice.tts.provider') AS provider,
      get_config('voice.tts.model') AS model,
      get_config('voice.tts.provider_models') AS provider_models,
      get_config('voice.tts.voice') AS voice,
      get_config('voice.talk.enabled') AS talk_enabled,
      get_config('voice.wake.enabled') AS wake_enabled
  `);
  const row = rows[0] || ({} as VoiceOutputRow);
  const catalogValue = normalizeJsonValue(row.provider_models);
  const catalog = catalogValue && typeof catalogValue === "object" && !Array.isArray(catalogValue)
    ? catalogValue as Record<string, unknown>
    : {};
  const providers = Object.entries(catalog)
    .filter(([id, model]) => SUPPORTED_PROVIDERS.has(id) && typeof model === "string")
    .map(([id, model]) => ({ id, model }));
  return {
    enabled: normalizeJsonValue(row.enabled) === true,
    provider: String(normalizeJsonValue(row.provider) || "local_piper"),
    model: String(normalizeJsonValue(row.model) || ""),
    voice: String(normalizeJsonValue(row.voice) || ""),
    talk_enabled: normalizeJsonValue(row.talk_enabled) === true,
    wake_enabled: normalizeJsonValue(row.wake_enabled) === true,
    providers,
  };
}

export async function GET(): Promise<Response> {
  try {
    return NextResponse.json(await readSettings());
  } catch (error: unknown) {
    console.error("Voice-output settings read failed:", error);
    return NextResponse.json({ error: "Voice-output settings could not be loaded." }, { status: 500 });
  }
}

export async function POST(request: Request): Promise<Response> {
  try {
    const body = await request.json();
    const provider = String(body?.provider || "").trim();
    const enabled = body?.enabled === true;
    const talkEnabled = body?.talk_enabled === true;
    const wakeEnabled = body?.wake_enabled === true;
    const voice = String(body?.voice || "").trim();
    if (!SUPPORTED_PROVIDERS.has(provider)) {
      return NextResponse.json({ error: "Choose the local speech provider." }, { status: 400 });
    }
    if (talkEnabled && !enabled) {
      return NextResponse.json({ error: "Enable speech output before enabling Talk mode." }, { status: 400 });
    }
    if (wakeEnabled && !enabled) {
      return NextResponse.json({ error: "Enable speech output before allowing wake-word turns." }, { status: 400 });
    }
    if (voice.length > 100) {
      return NextResponse.json({ error: "Voice names must be 100 characters or fewer." }, { status: 400 });
    }
    const catalogRows = await prisma.$queryRawUnsafe<Array<{ catalog: unknown }>>(
      "SELECT get_config('voice.tts.provider_models') AS catalog",
    );
    const catalogValue = normalizeJsonValue(catalogRows[0]?.catalog);
    const catalog = catalogValue && typeof catalogValue === "object" && !Array.isArray(catalogValue)
      ? catalogValue as Record<string, unknown>
      : {};
    const model = catalog[provider];
    if (typeof model !== "string" || !model.trim()) {
      return NextResponse.json(
        { error: `No model is configured for ${provider}. Check voice.tts.provider_models.` },
        { status: 409 },
      );
    }
    await prisma.$queryRawUnsafe(
      `SELECT
         set_config('voice.tts.enabled', $1::jsonb),
         set_config('voice.tts.provider', $2::jsonb),
         set_config('voice.tts.model', $3::jsonb),
         set_config('voice.tts.voice', $4::jsonb),
         set_config('voice.talk.enabled', $5::jsonb),
         set_config('voice.wake.enabled', $6::jsonb)`,
      JSON.stringify(enabled),
      JSON.stringify(provider),
      JSON.stringify(model),
      JSON.stringify(voice),
      JSON.stringify(talkEnabled),
      JSON.stringify(wakeEnabled),
    );
    return NextResponse.json(await readSettings());
  } catch (error: unknown) {
    console.error("Voice-output settings update failed:", error);
    return NextResponse.json({ error: "Voice-output settings could not be saved." }, { status: 500 });
  }
}
