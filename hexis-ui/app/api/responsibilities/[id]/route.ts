import { NextResponse } from "next/server";

import {
  errorMessage,
  hexisApiHeaders,
  jsonProxyResponse,
  resolveHexisApiUrl,
} from "@/lib/python-api";

export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
): Promise<Response> {
  const { id } = await params;
  try {
    const upstream = await fetch(
      resolveHexisApiUrl(`/api/responsibilities/${encodeURIComponent(id)}`),
      {
        headers: hexisApiHeaders(),
        cache: "no-store",
      }
    );
    const payload = await upstream.text();
    return jsonProxyResponse(upstream, payload);
  } catch (error: unknown) {
    return NextResponse.json(
      { error: `Responsibility upstream unreachable: ${errorMessage(error, "Unknown error")}` },
      { status: 502 }
    );
  }
}
