import { NextResponse } from "next/server";

import {
  errorMessage,
  hexisApiHeaders,
  jsonProxyResponse,
  resolveHexisApiUrl,
} from "@/lib/python-api";

export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  let bodyText = "";
  try {
    bodyText = await request.text();
  } catch (error: unknown) {
    return NextResponse.json(
      { error: errorMessage(error, "Failed to read inbox reply body.") },
      { status: 400 }
    );
  }

  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/inbox/reply"), {
      method: "POST",
      headers: hexisApiHeaders({ "Content-Type": "application/json" }),
      body: bodyText,
    });
    return jsonProxyResponse(upstream, await upstream.text());
  } catch (error: unknown) {
    console.error("Inbox reply API error:", error);
    return NextResponse.json(
      {
        error: `Inbox reply upstream unreachable: ${errorMessage(
          error,
          "Failed to reach Hexis API"
        )}`,
      },
      { status: 502 }
    );
  }
}
