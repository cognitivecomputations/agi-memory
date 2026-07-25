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
    const upstream = await fetch(resolveHexisApiUrl("/api/responsibilities", search), {
      headers: hexisApiHeaders(),
      cache: "no-store",
    });
    const payload = await upstream.text();
    return jsonProxyResponse(upstream, payload);
  } catch (error: unknown) {
    return NextResponse.json(
      { error: `Responsibilities upstream unreachable: ${errorMessage(error, "Unknown error")}` },
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
      { error: errorMessage(error, "Failed to read responsibility action body.") },
      { status: 400 }
    );
  }

  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/responsibilities/action"), {
      method: "POST",
      headers: hexisApiHeaders({ "Content-Type": "application/json" }),
      body: bodyText,
    });
    const payload = await upstream.text();
    return jsonProxyResponse(upstream, payload);
  } catch (error: unknown) {
    return NextResponse.json(
      { error: `Responsibilities upstream unreachable: ${errorMessage(error, "Unknown error")}` },
      { status: 502 }
    );
  }
}
