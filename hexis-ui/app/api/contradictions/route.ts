import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type DbRow = Record<string, unknown>;

export async function GET(request: NextRequest) {
  try {
    const status = new URL(request.url).searchParams.get("status") || "all";
    if (!new Set(["all", "pending", "resolved", "tension"]).has(status)) {
      return NextResponse.json({ error: "invalid status" }, { status: 422 });
    }
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT list_contradiction_cases($1, 200) AS cases",
      status
    );
    const cases = normalizeJsonValue(rows[0]?.cases) || [];
    return NextResponse.json({ cases });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to load contradictions";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
