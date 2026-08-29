import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type DbRow = Record<string, unknown>;

/** Approve or deny one exact signed companion-node identity. */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const pairingRequest =
      typeof body?.request === "string" ? body.request.trim() : "";
    const decision = typeof body?.decision === "string" ? body.decision : "";
    const note =
      typeof body?.note === "string" && body.note.trim() ? body.note.trim() : null;
    if (!pairingRequest) {
      return NextResponse.json({ error: "request is required" }, { status: 422 });
    }
    if (decision !== "approve" && decision !== "deny") {
      return NextResponse.json(
        { error: "decision must be 'approve' or 'deny'" },
        { status: 422 }
      );
    }
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT decide_node_pairing($1, $2, 'dashboard', $3) AS result",
      pairingRequest,
      decision,
      note
    );
    const result = normalizeJsonValue(rows[0]?.result) || {};
    const status =
      result && typeof result === "object" ? String((result as DbRow).status || "") : "";
    return NextResponse.json(result, {
      status: status === "approved" || status === "denied" ? 200 : 409,
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to decide node pairing";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
