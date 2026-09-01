import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";
import { access, readFile, readdir } from "fs/promises";
import os from "os";
import path from "path";

type DbRow = Record<string, unknown>;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Failed to fetch status";
}

export async function GET() {
  try {
    // Aggregate status from DB views
    const [
      cogHealthRows,
      heartbeatRows,
      driveRows,
      emotionRows,
      trendRows,
      goalRows,
      recentHbRows,
      memHealthRows,
      configuredRows,
      profileRows,
      economyRows,
    ] = await Promise.all([
      prisma.$queryRawUnsafe<DbRow[]>("SELECT * FROM cognitive_health LIMIT 1"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT * FROM heartbeat_state LIMIT 1"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT * FROM drive_status"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT * FROM current_emotional_state LIMIT 1"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT * FROM emotional_trend ORDER BY hour DESC LIMIT 24"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT * FROM active_goals"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT * FROM recent_heartbeats LIMIT 5"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT * FROM memory_health"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT is_agent_configured() AS configured"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT get_init_profile() AS profile"),
      prisma.$queryRawUnsafe<DbRow[]>("SELECT heartbeat_economy_status() AS economy"),
    ]);

    const cogHealth = normalize(cogHealthRows);
    const heartbeat = normalize(heartbeatRows);
    const emotion = normalize(emotionRows);
    const profile = normalizeJsonValue(profileRows[0]?.profile);
    const profileRecord = asRecord(profile);
    const profileAgent = asRecord(profileRecord.agent);
    const economy = asRecord(normalizeJsonValue(economyRows[0]?.economy));

    const agentName =
      stringValue(profileAgent.name) ||
      stringValue(cogHealth.identity) ||
      "Hexis";
    const portraitStem = await resolvePortraitStem(agentName, profile);

    return NextResponse.json({
      agent_name: agentName,
      portrait_url: portraitStem
        ? `/api/init/characters/image?name=${encodeURIComponent(portraitStem)}`
        : null,
      configured: configuredRows[0]?.configured ?? false,

      // Energy
      energy: toNum(heartbeat.current_energy ?? cogHealth.energy),
      max_energy: toNum(economy.reserve_energy ?? cogHealth.max_energy ?? 20),
      energy_reserve: toNum(economy.reserve_energy ?? cogHealth.max_energy ?? 20),
      energy_capacity: toNum(economy.bank_capacity ?? cogHealth.max_energy ?? 20),
      next_regen_multiplier: toNum(economy.next_regen_multiplier ?? 1),

      // Heartbeat
      heartbeat_active: !!heartbeat.active_heartbeat_id,
      heartbeat_paused: heartbeat.is_paused ?? false,
      heartbeat_count: toNum(heartbeat.heartbeat_count),
      last_heartbeat_at: heartbeat.last_heartbeat_at ?? null,
      next_heartbeat_at: heartbeat.next_heartbeat_at ?? null,

      // Mood
      mood: cogHealth.primary_emotion ?? emotion.primary_emotion ?? null,
      valence: toNum(emotion.valence),
      arousal: toNum(emotion.arousal),
      dominance: toNum(emotion.dominance),
      intensity: toNum(emotion.intensity),

      // Drives
      drives: driveRows.map((d) => ({
        name: d.drive_name ?? d.name,
        urgency: toNum(d.urgency_percent),
        hours_since: toNum(d.hours_since_satisfied),
      })),

      // Emotional trend (24h hourly)
      emotional_trend: trendRows.map((t) => ({
        hour: t.hour,
        valence: toNum(t.avg_valence),
        arousal: toNum(t.avg_arousal),
      })),

      // Goals
      goals: goalRows.map((g) => ({
        id: g.id,
        content: g.title ?? g.description ?? "Untitled goal",
        priority: g.is_blocked ? "blocked" : "active",
        source: g.source,
        progress_count: toNum(g.progress_count),
        last_touched: g.last_touched,
      })),

      // Recent heartbeats
      recent_heartbeats: recentHbRows.map((h) => ({
        id: h.memory_id ?? h.heartbeat_id,
        narrative: h.content ?? h.narrative,
        emotional_valence: toNum(h.emotional_valence),
        created_at: h.started_at,
      })),

      // Memory health
      memory_health: memHealthRows.map((m) => ({
        type: m.type,
        count: toNum(m.total_memories),
        avg_importance: toNum(m.avg_importance),
      })),
    });
  } catch (error: unknown) {
    console.error("Status API error:", error);
    return NextResponse.json(
      { error: errorMessage(error) },
      { status: 500 }
    );
  }
}

async function resolvePortraitStem(agentName: string, profile: unknown): Promise<string | null> {
  const configured = asRecord(asRecord(profile).agent).portrait;
  if (typeof configured === "string" && configured.trim()) return configured.trim();

  const dirs = [
    process.env.HEXIS_CHARACTERS_DIR,
    path.join(os.homedir(), ".hexis", "characters"),
    path.resolve(process.cwd(), "..", "characters"),
  ].filter((value): value is string => Boolean(value));

  for (const dir of dirs) {
    try {
      const files = await readdir(dir);
      for (const filename of files.filter((name) => name.endsWith(".json")).sort()) {
        const card = asRecord(JSON.parse(await readFile(path.join(dir, filename), "utf-8")));
        const data = asRecord(card.data);
        const hexis = asRecord(asRecord(data.extensions).hexis);
        const name = hexis.name ?? data.name;
        if (typeof name !== "string" || name.toLowerCase() !== agentName.toLowerCase()) continue;
        const stem = filename.replace(/\.json$/, "");
        await access(path.join(dir, `${stem}.jpg`));
        return stem;
      }
    } catch {
      continue;
    }
  }
  return null;
}

function normalize(rows: unknown): DbRow {
  if (Array.isArray(rows) && rows.length > 0) return asRecord(rows[0]);
  return {};
}

function asRecord(value: unknown): DbRow {
  return value !== null && typeof value === "object" ? value as DbRow : {};
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function toNum(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
