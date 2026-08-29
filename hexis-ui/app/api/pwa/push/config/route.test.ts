import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "./route";

describe("/api/pwa/push/config", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("keeps VAPID key handling behind the Python API", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ enabled: true, public_key: "public" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const response = await GET();
    expect(response.status).toBe(200);
    expect(fetchMock.mock.calls[0][0]).toBe("http://127.0.0.1:43817/api/pwa/push/config");
    expect(fetchMock.mock.calls[0][1].cache).toBe("no-store");
  });
});
