import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type GoalRow = {
  id: string;
  content: string;
  metadata: unknown;
  created_at: unknown;
};

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const rows = await prisma.$queryRawUnsafe<GoalRow[]>(
      `SELECT id, type, content, importance, metadata, created_at, last_accessed
       FROM memories
       WHERE id = $1::uuid AND type = 'goal'`,
      id
    );

    if (rows.length === 0) {
      return NextResponse.json({ error: "Goal not found" }, { status: 404 });
    }

    const g = rows[0];
    const normalizedMeta = normalizeJsonValue(g.metadata);
    const meta = normalizedMeta && typeof normalizedMeta === "object" && !Array.isArray(normalizedMeta)
      ? normalizedMeta as Record<string, unknown>
      : {};

    return NextResponse.json({
      id: g.id,
      title: meta.title || g.content,
      description: meta.description,
      source: meta.source,
      priority: meta.priority,
      progress: meta.progress || [],
      last_touched: meta.last_touched,
      created_at: g.created_at,
    });
  } catch (error: unknown) {
    console.error("Goal detail error:", error);
    return NextResponse.json(
      { error: errorMessage(error, "Failed to fetch goal") },
      { status: 500 }
    );
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await req.json() as Record<string, unknown>;

    if (body.priority) {
      await prisma.$queryRawUnsafe(
        `SELECT change_goal_priority($1::uuid, $2::goal_priority, $3)`,
        id,
        body.priority,
        body.reason || null
      );
    }

    if (body.progress_note) {
      await prisma.$queryRawUnsafe(
        `SELECT add_goal_progress($1::uuid, $2)`,
        id,
        body.progress_note
      );
    }

    return NextResponse.json({ ok: true });
  } catch (error: unknown) {
    console.error("Update goal error:", error);
    return NextResponse.json(
      { error: errorMessage(error, "Failed to update goal") },
      { status: 500 }
    );
  }
}
