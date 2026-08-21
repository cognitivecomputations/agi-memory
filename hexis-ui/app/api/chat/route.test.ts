import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "./route";

describe("/api/chat", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("forwards upstream SSE chunks without buffering the response", async () => {
    let releaseSecondChunk: (() => void) | undefined;
    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(
              encoder.encode("event: reasoning\ndata: {}\n\n")
            );
            releaseSecondChunk = () => {
              controller.enqueue(
                encoder.encode(
                  'event: token\ndata: {"phase":"conscious_final","text":"Hello"}\n\n'
                )
              );
              controller.close();
            };
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } }
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(
      new Request("http://localhost/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "Hello" }),
      })
    );
    const reader = response.body?.getReader();
    expect(reader).toBeDefined();

    const first = await reader!.read();
    expect(new TextDecoder().decode(first.value)).toBe(
      "event: reasoning\ndata: {}\n\n"
    );

    let secondResolved = false;
    const secondRead = reader!.read().then((chunk) => {
      secondResolved = true;
      return chunk;
    });
    await Promise.resolve();
    expect(secondResolved).toBe(false);

    releaseSecondChunk?.();
    const second = await secondRead;
    expect(new TextDecoder().decode(second.value)).toContain(
      '"text":"Hello"'
    );
    expect(response.headers.get("Content-Type")).toBe("text/event-stream");
  });

  it("forwards configured API authentication to the stream endpoint", async () => {
    vi.stubEnv("HEXIS_API_KEY", "dashboard-api-key");
    const fetchMock = vi.fn(async () =>
      new Response("event: done\ndata: {}\n\n", {
        headers: { "Content-Type": "text/event-stream" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await POST(
      new Request("http://localhost/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "Hello" }),
      })
    );

    const requestInit = fetchMock.mock.calls[0][1];
    const headers = new Headers(requestInit?.headers);
    expect(headers.get("Authorization")).toBe("Bearer dashboard-api-key");
    expect(headers.get("Accept")).toBe("text/event-stream");
  });
});
