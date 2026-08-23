import { errorMessage, hexisApiHeaders, jsonProxyResponse, resolveHexisApiUrl } from "@/lib/python-api";

export const runtime = "nodejs";

/**
 * Starts durable ingestion for an attachment the composer already prepared.
 * Called when the message carrying the file is sent, so the agent's memory
 * only gains files the user actually shared.
 */

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
): Promise<Response> {
  const { id } = await params;
  let body: unknown = {};
  try {
    body = await request.json();
  } catch {
    body = {};
  }

  try {
    const upstream = await fetch(
      resolveHexisApiUrl(`/api/attachments/${encodeURIComponent(id)}/ingest`),
      {
        method: "POST",
        headers: hexisApiHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body ?? {}),
      }
    );
    const payload = await upstream.text();
    return jsonProxyResponse(upstream, payload);
  } catch (err: unknown) {
    const message = errorMessage(err, "Unknown error");
    return Response.json(
      { error: `Attachment upstream unreachable: ${message}` },
      { status: 502 }
    );
  }
}
