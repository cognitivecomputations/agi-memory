import {
  errorMessage,
  hexisApiHeaders,
  jsonProxyResponse,
  resolveHexisApiUrl,
} from "@/lib/python-api";

export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/voice/transcribe"), {
      method: "POST",
      cache: "no-store",
      headers: hexisApiHeaders(),
      body: await request.formData(),
    });
    return jsonProxyResponse(upstream, await upstream.text());
  } catch (error: unknown) {
    return Response.json(
      {
        error: errorMessage(
          error,
          "Voice transcription could not reach the Hexis API. Run hexis doctor, then retry.",
        ),
      },
      { status: 502 },
    );
  }
}
