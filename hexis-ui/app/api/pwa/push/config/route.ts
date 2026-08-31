import {
  errorMessage,
  hexisApiHeaders,
  jsonProxyResponse,
  resolveHexisApiUrl,
} from "@/lib/python-api";

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  try {
    const upstream = await fetch(resolveHexisApiUrl("/api/pwa/push/config"), {
      cache: "no-store",
      headers: hexisApiHeaders(),
    });
    return jsonProxyResponse(upstream, await upstream.text());
  } catch (error: unknown) {
    return Response.json(
      {
        error: errorMessage(
          error,
          "Notification setup could not reach the Hexis API. Run hexis doctor, then retry.",
        ),
      },
      { status: 502 },
    );
  }
}
