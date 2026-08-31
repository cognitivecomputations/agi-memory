import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { tool_name, enabled } = body;

    if (!tool_name) {
      return NextResponse.json({ error: "tool_name is required" }, { status: 400 });
    }

    // Update tool enabled status in config
    const key = `tools.${tool_name}.enabled`;
    const value = JSON.stringify(enabled !== false);

    await prisma.$queryRawUnsafe(
      `INSERT INTO config (key, value) VALUES ($1, $2::jsonb)
       ON CONFLICT (key) DO UPDATE SET value = $2::jsonb`,
      key,
      value
    );

    return NextResponse.json({ ok: true, key, enabled: enabled !== false });
  } catch (error: unknown) {
    console.error("Tool toggle error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error && error.message
            ? error.message
            : "Failed to update tool",
      },
      { status: 500 }
    );
  }
}
