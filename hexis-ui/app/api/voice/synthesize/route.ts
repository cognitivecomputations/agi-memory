import { errorMessage, hexisApiHeaders, resolveHexisApiUrl } from "@/lib/python-api";

export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/voice/synthesize"), {
      method: "POST",
      cache: "no-store",
      headers: {
        ...hexisApiHeaders(),
        "Content-Type": "application/json",
      },
      body: await request.text(),
    });
    const body = await upstream.arrayBuffer();
    return new Response(body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "application/octet-stream",
        "Cache-Control": "no-store",
      },
    });
  } catch (error: unknown) {
    return Response.json(
      {
        error: errorMessage(
          error,
          "Speech synthesis could not reach the Hexis API. Run hexis doctor, then retry.",
        ),
      },
      { status: 502 },
    );
  }
}
