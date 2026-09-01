import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "./route";

describe("/api/outbox/reply", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("proxies a correlated reply to the Hexis API", async () => {
    const fetchMock = vi.fn(async () =>
      new Response('{"queued":true}', {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const body = {
      message_id: "11111111-1111-4111-8111-111111111111",
      reply: "Yes, please.",
    };
    const response = await POST(
      new Request("http://localhost/api/outbox/reply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ queued: true });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:43817/api/inbox/reply",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(body),
      })
    );
  });

  it("returns an actionable error when the Hexis API is unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("ECONNREFUSED");
      })
    );

    const response = await POST(
      new Request("http://localhost/api/outbox/reply", {
        method: "POST",
        body: JSON.stringify({ message_id: crypto.randomUUID(), reply: "Hello" }),
      })
    );
    const body = await response.json();

    expect(response.status).toBe(502);
    expect(body.error).toContain("Inbox reply upstream unreachable");
    expect(body.error).toContain("ECONNREFUSED");
  });

  it("preserves an upstream queue failure for the dashboard", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json(
          {
            detail:
              "Reply was not queued because Hexis's inbox is unavailable: queue unavailable",
          },
          { status: 503 }
        )
      )
    );

    const response = await POST(
      new Request("http://localhost/api/outbox/reply", {
        method: "POST",
        body: JSON.stringify({
          message_id: crypto.randomUUID(),
          reply: "Hello",
        }),
      })
    );

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      detail:
        "Reply was not queued because Hexis's inbox is unavailable: queue unavailable",
    });
  });
});
