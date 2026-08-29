import { afterEach, describe, expect, it, vi } from "vitest";

import { GET, POST } from "./route";

describe("/api/outbound", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("proxies ledger queries to the Python policy API", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({ suspended: false, events: [], budgets: [], controls: [] }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await GET(new Request("http://localhost/api/outbound?limit=25"));

    expect(response.status).toBe(200);
    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://127.0.0.1:43817/api/outbound?limit=25",
    );
    expect(fetchMock.mock.calls[0][1].cache).toBe("no-store");
  });

  it("forwards one-click control actions without changing their meaning", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({ control: { suspended: true }, ledger: { suspended: true } }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const body = { action: "suspend_entity", entity: "contact:42" };

    const response = await POST(
      new Request("http://localhost/api/outbound", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    );

    expect(response.status).toBe(200);
    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://127.0.0.1:43817/api/outbound/control",
    );
    expect(fetchMock.mock.calls[0][1].method).toBe("POST");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual(body);
  });
});
