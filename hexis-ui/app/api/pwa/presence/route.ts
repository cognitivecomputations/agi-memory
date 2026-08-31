import {
  errorMessage,
  hexisApiHeaders,
  jsonProxyResponse,
  resolveHexisApiUrl,
} from "@/lib/python-api";

export async function POST(request: Request): Promise<Response> {
  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/pwa/presence"), {
      method: "POST",
      cache: "no-store",
      headers: hexisApiHeaders({ "Content-Type": "application/json" }),
      body: await request.text(),
    });
    return jsonProxyResponse(upstream, await upstream.text());
  } catch (error: unknown) {
    return Response.json(
      {
        error: errorMessage(error, "PWA presence could not reach the Hexis API."),
      },
      { status: 502 },
    );
  }
}
