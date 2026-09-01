import { NextRequest, NextResponse } from "next/server";
import { normalizeJsonValue } from "@/lib/db";
import { prisma } from "@/lib/prisma";

type DbRow = Record<string, unknown>;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const id = typeof body?.id === "string" ? body.id.trim() : "";
    const action = typeof body?.action === "string" ? body.action : "";
    const correction = typeof body?.correction === "string" && body.correction.trim()
      ? body.correction.trim()
      : null;
    const confirmLoadBearing = body?.confirm_load_bearing === true;
    if (!UUID_RE.test(id)) {
      return NextResponse.json({ error: "id must be a UUID" }, { status: 422 });
    }
    if (!new Set(["approve", "correct", "forget"]).has(action)) {
      return NextResponse.json(
        { error: "action must be 'approve', 'correct', or 'forget'" },
        { status: 422 }
      );
    }
    if (action === "correct" && !correction) {
      return NextResponse.json({ error: "correction is required" }, { status: 422 });
    }
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT decide_learning_review_item($1::uuid, $2, $3, 'web', 'dashboard', $4::boolean) AS result",
      id,
      action,
      correction,
      confirmLoadBearing
    );
    const result = (normalizeJsonValue(rows[0]?.result) || {}) as DbRow;
    const ok = result.ok === true;
    const status = ok ? 200 : result.confirmation_required === true ? 409 : 400;
    return NextResponse.json(result, { status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to record learning decision";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
