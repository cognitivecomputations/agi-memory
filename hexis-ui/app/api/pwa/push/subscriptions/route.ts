import {
  errorMessage,
  hexisApiHeaders,
  jsonProxyResponse,
  resolveHexisApiUrl,
} from "@/lib/python-api";

async function proxy(request: Request, method: "POST" | "DELETE"): Promise<Response> {
  try {
    const upstream = await fetch(
      resolveHexisApiUrl("/api/pwa/push/subscriptions"),
      {
        method,
        cache: "no-store",
        headers: hexisApiHeaders({ "Content-Type": "application/json" }),
        body: await request.text(),
      },
    );
    return jsonProxyResponse(upstream, await upstream.text());
  } catch (error: unknown) {
    return Response.json(
      {
        error: errorMessage(
          error,
          "Notification settings could not reach the Hexis API. Run hexis doctor, then retry.",
        ),
      },
      { status: 502 },
    );
  }
}

export async function POST(request: Request): Promise<Response> {
  return proxy(request, "POST");
}

export async function DELETE(request: Request): Promise<Response> {
  return proxy(request, "DELETE");
}
