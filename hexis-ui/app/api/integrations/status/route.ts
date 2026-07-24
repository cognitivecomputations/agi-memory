import { NextResponse } from "next/server";

import { normalizeJsonValue } from "@/lib/db";
import { prisma } from "@/lib/prisma";
import {
  errorMessage as formatErrorMessage,
  hexisApiHeaders,
  jsonProxyResponse,
  resolveHexisApiUrl,
} from "@/lib/python-api";

export const runtime = "nodejs";

type Row = Record<string, unknown>;

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function errorMessage(error: unknown): string {
  return formatErrorMessage(error, "Failed to fetch integration status");
}

export async function GET(): Promise<Response> {
  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/integrations/status"), {
      method: "GET",
      headers: hexisApiHeaders(),
      cache: "no-store",
    });
    if (upstream.ok) {
      const payload = await upstream.text();
      return jsonProxyResponse(upstream, payload);
    }
    console.warn("Integration status upstream returned", upstream.status);
  } catch (error: unknown) {
    console.warn("Integration status upstream unavailable; falling back to DB:", error);
  }

  try {
    const rows = await prisma.$queryRawUnsafe<Row[]>(`
      SELECT
        integration_status(NULL) AS integration,
        list_channel_adapter_status(NULL) AS channel_runtime,
        get_connector_backfill_status(NULL, NULL) AS backfill
    `);
    const row = rows[0] ?? {};
    const integration = asRecord(normalizeJsonValue(row.integration));
    const backfill = asRecord(normalizeJsonValue(row.backfill));

    return NextResponse.json({
      connectors: asArray(integration.connectors),
      connections: asArray(integration.connections),
      recent_attempts: asArray(integration.recent_attempts),
      channel_runtime: asArray(normalizeJsonValue(row.channel_runtime)),
      backfill: {
        jobs: asArray(backfill.jobs),
        cursors: asArray(backfill.cursors),
        item_counts: asArray(backfill.item_counts),
      },
      generated_at: new Date().toISOString(),
    });
  } catch (error: unknown) {
    console.error("Integration status API error:", error);
    return NextResponse.json({ error: errorMessage(error) }, { status: 500 });
  }
}

export async function POST(request: Request): Promise<Response> {
  let bodyText = "";
  try {
    bodyText = await request.text();
  } catch (error: unknown) {
    return NextResponse.json(
      { error: errorMessage(error) || "Failed to read integration action body." },
      { status: 400 }
    );
  }

  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/integrations/action"), {
      method: "POST",
      headers: hexisApiHeaders({ "Content-Type": "application/json" }),
      body: bodyText,
    });
    const payload = await upstream.text();
    return jsonProxyResponse(upstream, payload);
  } catch (error: unknown) {
    console.error("Integration setup API error:", error);
    return NextResponse.json(
      { error: `Integration upstream unreachable: ${errorMessage(error)}` },
      { status: 502 }
    );
  }
}
