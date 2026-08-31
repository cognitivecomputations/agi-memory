import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "./route";

describe("/api/voice/synthesize", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("forwards text and preserves the audio response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(new Uint8Array([82, 73, 70, 70]), {
        status: 200,
        headers: { "Content-Type": "audio/wav" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const response = await POST(
      new Request("http://local/api/voice/synthesize", {
        method: "POST",
        body: JSON.stringify({ text: "hello" }),
      }),
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toBe("audio/wav");
    expect(fetchMock.mock.calls[0][0]).toBe("http://127.0.0.1:43817/api/voice/synthesize");
    expect(fetchMock.mock.calls[0][1].body).toBe('{"text":"hello"}');
  });
});
