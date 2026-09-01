import {
  errorMessage,
  hexisApiHeaders,
  jsonProxyResponse,
  resolveHexisApiUrl,
} from "@/lib/python-api";

export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/voice/status"), {
      cache: "no-store",
      headers: hexisApiHeaders(),
    });
    return jsonProxyResponse(upstream, await upstream.text());
  } catch (error: unknown) {
    return Response.json(
      {
        error: errorMessage(
          error,
          "Voice status could not reach the Hexis API. Run hexis doctor, then retry.",
        ),
      },
      { status: 502 },
    );
  }
}
