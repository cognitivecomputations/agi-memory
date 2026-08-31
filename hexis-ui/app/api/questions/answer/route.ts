import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type DbRow = Record<string, unknown>;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/** Answer the durable question currently pausing a chat turn. */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const id = typeof body?.id === "string" ? body.id.trim() : "";
    const answer = typeof body?.answer === "string" ? body.answer.trim() : null;
    const choiceIndex = Number.isInteger(body?.choice_index)
      ? Number(body.choice_index)
      : null;
    if (!id) {
      return NextResponse.json({ error: "id is required" }, { status: 422 });
    }
    if (!UUID_RE.test(id)) {
      return NextResponse.json(
        { error: "question id must be a UUID" },
        { status: 422 }
      );
    }
    if (choiceIndex === null && !answer) {
      return NextResponse.json(
        { error: "answer or choice_index is required" },
        { status: 422 }
      );
    }

    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT answer_agent_question($1::uuid, $2, $3, 'web', 'dashboard') AS result",
      id,
      answer,
      choiceIndex
    );
    const result = normalizeJsonValue(rows[0]?.result) || {};
    const ok = result && typeof result === "object" && (result as DbRow).ok === true;
    return NextResponse.json(result, { status: ok ? 200 : 409 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to answer question";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
