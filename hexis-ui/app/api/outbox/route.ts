import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { normalizeJsonValue } from "@/lib/db";

type DbRow = Record<string, unknown>;

/**
 * The web inbox: this dashboard's view of the async messaging abstraction.
 * The agent's outbox (RabbitMQ hexis.outbox) is her always-available way to
 * reach the user; the channel worker tees every user-bound message into
 * web_inbox (db/76), and this route serves that feed plus any resource
 * requests, automation suggestions, and memory contradictions awaiting an
 * operator decision.
 */
export async function GET() {
  try {
    const [inboxRows, requestRows, automationRows, contradictionRows, nodePairingRows] = await Promise.all([
      prisma.$queryRawUnsafe<DbRow[]>("SELECT get_web_inbox(30) AS feed"),
      prisma.$queryRawUnsafe<DbRow[]>(
        "SELECT list_resource_requests('pending', 20) AS requests"
      ),
      prisma.$queryRawUnsafe<DbRow[]>(
        "SELECT list_automation_suggestions('pending', 20) AS automations"
      ),
      prisma.$queryRawUnsafe<DbRow[]>(
        "SELECT list_contradiction_cases('pending', 20) AS contradictions"
      ),
      prisma.$queryRawUnsafe<DbRow[]>(
        "SELECT list_node_pairing_requests('pending', 20) AS node_pairings"
      ),
    ]);
    const feed = normalizeJsonValue(inboxRows[0]?.feed) || {
      unread: 0,
      messages: [],
    };
    const pendingRequests = normalizeJsonValue(requestRows[0]?.requests) || [];
    const pendingAutomations = normalizeJsonValue(automationRows[0]?.automations) || [];
    const pendingContradictions = normalizeJsonValue(contradictionRows[0]?.contradictions) || [];
    const pendingNodePairings = normalizeJsonValue(nodePairingRows[0]?.node_pairings) || [];
    return NextResponse.json({
      unread: Number(feed.unread || 0),
      messages: feed.messages || [],
      pending_requests: pendingRequests,
      pending_automations: pendingAutomations,
      pending_contradictions: pendingContradictions,
      pending_node_pairings: pendingNodePairings,
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to load inbox";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

function cleanIds(raw: unknown): string[] {
  return Array.isArray(raw)
    ? raw.filter((id: unknown) => typeof id === "string" && id.length > 0)
    : [];
}

/** Mark inbox messages read: body { ids: string[] }. */
export async function POST(request: NextRequest) {
  try {
    const ids = cleanIds((await request.json())?.ids);
    if (ids.length === 0) {
      return NextResponse.json({ marked: 0 });
    }
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT mark_web_inbox_read(ARRAY(SELECT jsonb_array_elements_text($1::jsonb))::uuid[]) AS marked",
      JSON.stringify(ids)
    );
    return NextResponse.json({ marked: Number(rows[0]?.marked || 0) });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to mark read";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/** Delete inbox messages (explicit dismissal): body { ids: string[] }. */
export async function DELETE(request: NextRequest) {
  try {
    const ids = cleanIds((await request.json())?.ids);
    if (ids.length === 0) {
      return NextResponse.json({ removed: 0 });
    }
    const rows = await prisma.$queryRawUnsafe<DbRow[]>(
      "SELECT delete_web_inbox(ARRAY(SELECT jsonb_array_elements_text($1::jsonb))::uuid[]) AS removed",
      JSON.stringify(ids)
    );
    return NextResponse.json({ removed: Number(rows[0]?.removed || 0) });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to delete";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
