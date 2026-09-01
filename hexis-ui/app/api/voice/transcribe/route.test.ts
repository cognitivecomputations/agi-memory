import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "./route";

describe("/api/voice/transcribe", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("forwards recorded bytes to the Python voice policy endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ transcript: "hello" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const form = new FormData();
    form.append("file", new Blob(["audio"], { type: "audio/webm" }), "memo.webm");
    const request = { formData: vi.fn().mockResolvedValue(form) } as unknown as Request;
    const response = await POST(request);
    expect(response.status).toBe(200);
    expect(fetchMock.mock.calls[0][0]).toBe("http://127.0.0.1:43817/api/voice/transcribe");
    expect(fetchMock.mock.calls[0][1].body).toBeInstanceOf(FormData);
  });
});
