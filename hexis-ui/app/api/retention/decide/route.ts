import { NextRequest, NextResponse } from "next/server";
import { normalizeJsonValue } from "@/lib/db";
import { prisma } from "@/lib/prisma";

type DbRow = Record<string, unknown>;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const id = typeof body?.id === "string" ? body.id.trim() : "";
    const decision = typeof body?.decision === "string" ? body.decision : "";
    const journalContent = typeof body?.journal_content === "string" && body.journal_content.trim()
      ? body.journal_content.trim()
      : null;
    if (!UUID_RE.test(id)) {
      return NextResponse.json({ error: "id must be a UUID" }, { status: 422 });
    }
    if (!new Set(["keep", "release", "journal"]).has(decision)) {
      return NextResponse.json(
        { error: "decision must be 'keep', 'release', or 'journal'" },
        { status: 422 }
      );
    }
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT decide_memory_fade_review($1::uuid, $2, $3, 'web', 'dashboard') AS result",
      id,
      decision,
      journalContent
    );
    const result = (normalizeJsonValue(rows[0]?.result) || {}) as DbRow;
    return NextResponse.json(result, { status: result.ok === true ? 200 : 409 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to record fade decision";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
