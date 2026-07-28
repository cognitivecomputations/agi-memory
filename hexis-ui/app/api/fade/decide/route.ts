import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type DbRow = Record<string, unknown>;

/**
 * Decide a document fade ask from the dashboard inbox: body
 * { ref: string, decision: "approve" | "keep" }.
 * `ref` is the content_hash carried in the message's payload.delivery
 * (fuzzy label matching also works — resolve_document_fade handles both).
 * 'approve' permanently fades the document's memories; anything else keeps
 * and reinforces them — the DB treats keep as the safe default.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const ref = typeof body?.ref === "string" ? body.ref.trim() : "";
    const decision = typeof body?.decision === "string" ? body.decision : "";
    if (!ref) {
      return NextResponse.json({ error: "ref is required" }, { status: 422 });
    }
    if (decision !== "approve" && decision !== "keep") {
      return NextResponse.json(
        { error: "decision must be 'approve' or 'keep'" },
        { status: 422 }
      );
    }
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT resolve_document_fade($1, $2) AS result",
      ref,
      decision
    );
    const result = normalizeJsonValue(rows[0]?.result) || {};
    const status = result && typeof result === "object" && "error" in result ? 404 : 200;
    return NextResponse.json(result, { status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to decide fade";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
