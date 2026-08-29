import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type DbRecord = Record<string, unknown>;

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export async function GET() {
  try {
    const [activeRows, backlogRows] = await Promise.all([
      prisma.$queryRawUnsafe("SELECT * FROM active_goals"),
      prisma.$queryRawUnsafe("SELECT * FROM goal_backlog"),
    ]);

    const goals = (activeRows as DbRecord[]).map((g) => ({
      id: g.id,
      title: g.title,
      description: g.description,
      source: g.source,
      priority: "active",
      last_touched: g.last_touched,
      progress_count: Number(g.progress_count || 0),
      is_blocked: g.is_blocked ?? false,
      created_at: g.created_at,
    }));

    // Add queued and backburner goals from backlog
    for (const row of backlogRows as DbRecord[]) {
      const priority = typeof row.priority === "string" ? row.priority : "queued";
      if (priority === "active") continue; // already included above
      const normalizedItems = normalizeJsonValue(row.goals);
      const items = Array.isArray(normalizedItems) ? normalizedItems : [];
      for (const rawItem of items) {
        if (!rawItem || typeof rawItem !== "object" || Array.isArray(rawItem)) continue;
        const item = rawItem as DbRecord;
        goals.push({
          id: item.id,
          title: item.title,
          description: null,
          source: item.source,
          priority,
          last_touched: null,
          progress_count: 0,
          is_blocked: false,
          created_at: null,
        });
      }
    }

    return NextResponse.json({ goals });
  } catch (error: unknown) {
    console.error("Goals API error:", error);
    return NextResponse.json(
      { error: errorMessage(error, "Failed to fetch goals") },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { title, description, source, priority } = body;

    if (!title?.trim()) {
      return NextResponse.json({ error: "Title is required" }, { status: 400 });
    }

    const rows = await prisma.$queryRawUnsafe<Array<{ id: string }>>(
      `SELECT create_goal($1, $2, $3::goal_source, $4::goal_priority) AS id`,
      title.trim(),
      description?.trim() || null,
      source || "user_request",
      priority || "queued"
    );

    return NextResponse.json({ id: rows[0]?.id }, { status: 201 });
  } catch (error: unknown) {
    console.error("Create goal error:", error);
    return NextResponse.json(
      { error: errorMessage(error, "Failed to create goal") },
      { status: 500 }
    );
  }
}
