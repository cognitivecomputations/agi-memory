import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type DbRow = Record<string, unknown>;

/** Accept or permanently dismiss one inert automation suggestion. */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const id = typeof body?.id === "string" ? body.id.trim() : "";
    const decision = typeof body?.decision === "string" ? body.decision : "";
    if (!id) {
      return NextResponse.json({ error: "id is required" }, { status: 422 });
    }
    if (decision !== "accept" && decision !== "dismiss") {
      return NextResponse.json(
        { error: "decision must be 'accept' or 'dismiss'" },
        { status: 422 }
      );
    }

    const sql =
      decision === "accept"
        ? "SELECT accept_automation($1::uuid, 'web', 'dashboard') AS result"
        : "SELECT dismiss_automation($1::uuid, 'web', 'dashboard') AS result";
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(sql, id);
    const result = normalizeJsonValue(rows[0]?.result) || {};
    const ok = result && typeof result === "object" && (result as DbRow).ok === true;
    return NextResponse.json(result, { status: ok ? 200 : 409 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to decide automation";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
