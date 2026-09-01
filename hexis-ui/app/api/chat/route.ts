import {
  errorMessage,
  hexisApiHeaders,
  resolveHexisApiUrl,
  sseError,
  sseProxyResponse,
} from "@/lib/python-api";

export const runtime = "nodejs";

/**
 * Canonical chat implementation lives in the Python FastAPI server (`hexis-api`).
 *
 * This route is a thin streaming proxy so the browser can call same-origin
 * `/api/chat` while all LLM/tool logic remains in Python.
 */

export async function POST(request: Request): Promise<Response> {
  let bodyText = "";
  try {
    bodyText = await request.text();
  } catch (err: unknown) {
    const message = errorMessage(err, "Failed to read request body.");
    return sseError(message || "Failed to read request body.");
  }

  const url = resolveHexisApiUrl("/api/chat");

  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: "POST",
      headers: hexisApiHeaders({
        "Content-Type": request.headers.get("content-type") || "application/json",
        Accept: "text/event-stream",
      }),
      body: bodyText,
      signal: request.signal,
    });
  } catch (err: unknown) {
    const message = errorMessage(err, "Unknown error");
    // Keep status=200 so the UI's SSE parser can surface the error payload.
    return sseError(
      `Failed to reach Hexis API at ${url}: ${message || "Unknown error"}`
    );
  }

  return sseProxyResponse(upstream);
}
