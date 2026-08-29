import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type DbRow = Record<string, unknown>;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const id = typeof body?.id === "string" ? body.id.trim() : "";
    const outcome = typeof body?.outcome === "string" ? body.outcome : "";
    const note = typeof body?.note === "string" && body.note.trim() ? body.note.trim() : null;
    if (!id) {
      return NextResponse.json({ error: "id is required" }, { status: 422 });
    }
    if (!UUID_RE.test(id)) {
      return NextResponse.json({ error: "id must be a UUID" }, { status: 422 });
    }
    if (!new Set(["new_right", "old_right", "tension"]).has(outcome)) {
      return NextResponse.json(
        { error: "outcome must be 'new_right', 'old_right', or 'tension'" },
        { status: 422 }
      );
    }
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT decide_contradiction($1::uuid, $2, $3, 'web', 'dashboard') AS result",
      id,
      outcome,
      note
    );
    const result = normalizeJsonValue(rows[0]?.result) || {};
    const ok = typeof result === "object" && result !== null && (result as DbRow).ok === true;
    return NextResponse.json(result, { status: ok ? 200 : 409 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to decide contradiction";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
