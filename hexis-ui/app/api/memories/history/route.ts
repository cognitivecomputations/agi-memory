import { NextRequest, NextResponse } from "next/server";

import { normalizeJsonValue } from "@/lib/db";
import { prisma } from "@/lib/prisma";

type DbRow = { result?: unknown };
type ToolEnvelope = {
  success?: boolean;
  output?: unknown;
  display_output?: string | null;
  error?: string;
  error_type?: string;
};

function normalizedInstant(value: string | null): string | null {
  if (!value?.trim()) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toISOString();
}

export async function GET(request: NextRequest) {
  const params = new URL(request.url).searchParams;
  const mode = params.get("mode") || "snapshot";
  const query = (params.get("q") || "").trim();

  if (!new Set(["snapshot", "diff"]).has(mode)) {
    return NextResponse.json({ error: "mode must be snapshot or diff" }, { status: 422 });
  }
  if (!query) {
    return NextResponse.json({ error: "A memory topic is required" }, { status: 422 });
  }

  const toolName = mode === "snapshot" ? "recall_at_time" : "diff_memory_history";
  const args: Record<string, unknown> = { query };
  if (mode === "snapshot") {
    const asOf = normalizedInstant(params.get("as_of"));
    if (!asOf) {
      return NextResponse.json(
        { error: "as_of must be a valid date and time" },
        { status: 422 }
      );
    }
    args.as_of = asOf;
  } else {
    const fromTime = normalizedInstant(params.get("from_time"));
    const toTime = normalizedInstant(params.get("to_time"));
    if (!fromTime || !toTime) {
      return NextResponse.json(
        { error: "from_time and to_time must be valid dates and times" },
        { status: 422 }
      );
    }
    args.from_time = fromTime;
    args.to_time = toTime;
  }

  try {
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT execute_memory_tool($1::text, $2::jsonb) AS result",
      toolName,
      JSON.stringify(args)
    );
    const result = normalizeJsonValue(rows[0]?.result) as ToolEnvelope | null;
    if (!result || typeof result !== "object") {
      return NextResponse.json(
        { error: "The memory history service returned an invalid response" },
        { status: 500 }
      );
    }
    return NextResponse.json(result, { status: result.success === false ? 422 : 200 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to load memory history";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
