import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

const SENSITIVE_CONFIG_KEY_SUBSTRINGS = ["api_key", "token", "secret", "password"];

function isSensitiveConfigKey(key: string): boolean {
  const k = (key || "").toLowerCase();
  if (k.startsWith("oauth.")) return true;
  if (k.includes("channel.") && (k.includes("token") || k.includes("secret"))) return true;
  if (k.includes("user.contact")) return true;
  return SENSITIVE_CONFIG_KEY_SUBSTRINGS.some((s) => k.includes(s));
}

function isSensitiveFieldName(name: string): boolean {
  const n = (name || "").toLowerCase();
  if (n === "api_key_env") return false;
  if (n === "api_key") return true;
  if (n === "access" || n === "refresh" || n === "id_token") return true;
  if (n === "destinations") return true;
  if (n.includes("password") || n.includes("secret") || n.includes("token")) return true;
  if (n.includes("api_key") && !n.endsWith("_env")) return true;
  return false;
}

function redactDeep(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((v) => redactDeep(v));
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      out[k] = isSensitiveFieldName(k) ? "***" : redactDeep(v);
    }
    return out;
  }
  return value;
}

function redactValue(key: string, value: unknown): unknown {
  if (!isSensitiveConfigKey(key)) {
    // Still deep-redact common secret-shaped fields to prevent accidental leaks.
    return redactDeep(value);
  }
  if (value && typeof value === "object") {
    return redactDeep(value);
  }
  return "***";
}

export async function GET() {
  try {
    const rows = await prisma.$queryRawUnsafe<Array<{ key: string; value: unknown }>>(
      `SELECT key, value FROM config ORDER BY key`
    );

    // Group by prefix
    const groups: Record<string, Record<string, unknown>> = {};
    for (const row of rows) {
      const val = redactValue(String(row.key || ""), normalizeJsonValue(row.value));
      const prefix = row.key.split(".")[0] || "other";
      if (!groups[prefix]) groups[prefix] = {};
      groups[prefix][row.key] = val;
    }

    // Extract specific sections for easier UI consumption
    const llm = groups["llm"] || {};
    const heartbeat = groups["heartbeat"] || {};
    const agent = groups["agent"] || {};
    const tools = groups["tools"] || {};

    return NextResponse.json({
      groups,
      llm,
      heartbeat,
      agent,
      tools,
    });
  } catch (error: unknown) {
    console.error("Settings API error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error && error.message
            ? error.message
            : "Failed to fetch settings",
      },
      { status: 500 }
    );
  }
}
