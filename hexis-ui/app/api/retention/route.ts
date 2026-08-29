import { NextRequest, NextResponse } from "next/server";
import { normalizeJsonValue } from "@/lib/db";
import { prisma } from "@/lib/prisma";

type DbRow = Record<string, unknown>;

export async function GET(request: NextRequest) {
  try {
    const status = new URL(request.url).searchParams.get("status") || "pending";
    if (!new Set(["pending", "kept", "released", "all"]).has(status)) {
      return NextResponse.json({ error: "invalid status" }, { status: 422 });
    }
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT retention_observe_packet(10) AS observe, list_memory_fade_reviews($1, 50) AS reviews",
      status
    );
    return NextResponse.json({
      observe: normalizeJsonValue(rows[0]?.observe) || {},
      reviews: normalizeJsonValue(rows[0]?.reviews) || [],
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to load retention state";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
