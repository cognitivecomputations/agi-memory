import { afterEach, describe, expect, it, vi } from "vitest";

import { prisma } from "@/lib/prisma";
import { GET } from "./route";

vi.mock("@/lib/prisma", () => ({
  prisma: { $queryRawUnsafe: vi.fn() },
}));

const query = vi.mocked(prisma.$queryRawUnsafe);

describe("/api/memories/history", () => {
  afterEach(() => query.mockReset());

  it("uses the DB-owned point-in-time dispatcher", async () => {
    query.mockResolvedValueOnce([
      {
        result: {
          success: true,
          output: { as_of: "2026-06-01T12:00:00Z", memories: [] },
        },
      },
    ]);

    const response = await GET(
      new Request(
        "http://localhost/api/memories/history?mode=snapshot&q=Manning&as_of=2026-06-01T12:00:00Z"
      )
    );

    expect(response.status).toBe(200);
    expect((await response.json()).success).toBe(true);
    expect(query).toHaveBeenCalledWith(
      "SELECT execute_memory_tool($1::text, $2::jsonb) AS result",
      "recall_at_time",
      JSON.stringify({
        query: "Manning",
        as_of: "2026-06-01T12:00:00.000Z",
      })
    );
  });

  it("returns DB validation failures as a correctable request", async () => {
    query.mockResolvedValueOnce([
      {
        result: {
          success: false,
          error: "from_time must be earlier than to_time",
          error_type: "invalid_params",
        },
      },
    ]);

    const response = await GET(
      new Request(
        "http://localhost/api/memories/history?mode=diff&q=Manning&from_time=2026-08-01T00:00:00Z&to_time=2026-06-01T00:00:00Z"
      )
    );

    expect(response.status).toBe(422);
    expect(await response.json()).toMatchObject({
      success: false,
      error: "from_time must be earlier than to_time",
    });
  });

  it("rejects an unknown view before querying", async () => {
    const response = await GET(
      new Request("http://localhost/api/memories/history?mode=erase&q=Manning")
    );

    expect(response.status).toBe(422);
    expect(await response.json()).toEqual({ error: "mode must be snapshot or diff" });
    expect(query).not.toHaveBeenCalled();
  });
});
