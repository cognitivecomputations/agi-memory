import { errorMessage, hexisApiHeaders, resolveHexisApiUrl } from "@/lib/python-api";

export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
): Promise<Response> {
  const { id } = await params;
  try {
    const upstream = await fetch(
      resolveHexisApiUrl(`/api/voice/audio/${encodeURIComponent(id)}`),
      { cache: "no-store", headers: hexisApiHeaders() },
    );
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
          "Speech audio could not reach the Hexis API. Ask Hexis to speak it again.",
        ),
      },
      { status: 502 },
    );
  }
}
