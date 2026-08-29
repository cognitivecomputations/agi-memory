import { NextResponse } from "next/server";

import {
  errorMessage,
  hexisApiHeaders,
  jsonProxyResponse,
  resolveHexisApiUrl,
} from "@/lib/python-api";

export const runtime = "nodejs";

export async function GET(request: Request): Promise<Response> {
  const search = new URL(request.url).search;
  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/outbound", search), {
      headers: hexisApiHeaders(),
      cache: "no-store",
    });
    return jsonProxyResponse(upstream, await upstream.text());
  } catch (error: unknown) {
    return NextResponse.json(
      { error: `Outbound ledger upstream unreachable: ${errorMessage(error, "Unknown error")}` },
      { status: 502 }
    );
  }
}

export async function POST(request: Request): Promise<Response> {
  let bodyText = "";
  try {
    bodyText = await request.text();
  } catch (error: unknown) {
    return NextResponse.json(
      { error: errorMessage(error, "Failed to read outbound control body.") },
      { status: 400 }
    );
  }
  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/outbound/control"), {
      method: "POST",
      headers: hexisApiHeaders({ "Content-Type": "application/json" }),
      body: bodyText,
    });
    return jsonProxyResponse(upstream, await upstream.text());
  } catch (error: unknown) {
    return NextResponse.json(
      { error: `Outbound control upstream unreachable: ${errorMessage(error, "Unknown error")}` },
      { status: 502 }
    );
  }
}
