import { errorMessage, hexisApiHeaders, jsonProxyResponse, resolveHexisApiUrl } from "@/lib/python-api";

export const runtime = "nodejs";

/**
 * Chat attachment preparation proxy.
 *
 * A file attached in the composer lands here as multipart form data and is
 * forwarded to the Python `hexis-api` server (`POST /api/attachments`), which
 * preserves the original bytes and reads the text immediately so the agent can
 * discuss the file in the same turn. Ingestion into memory is a separate call,
 * made only when the message is actually sent.
 */

export async function POST(request: Request): Promise<Response> {
  let form: FormData;
  try {
    form = await request.formData();
  } catch (err: unknown) {
    const message = errorMessage(err, "Failed to read upload.");
    return Response.json({ error: message || "Failed to read upload." }, { status: 400 });
  }

  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/attachments"), {
      method: "POST",
      headers: hexisApiHeaders(),
      body: form,
    });
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
