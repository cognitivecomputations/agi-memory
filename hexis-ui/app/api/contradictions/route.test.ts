import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({
  prisma: { $queryRawUnsafe: vi.fn() },
}));

const query = vi.mocked(prisma.$queryRawUnsafe);

describe("/api/contradictions", () => {
  afterEach(() => query.mockReset());

  it("returns the requested durable ledger view", async () => {
    query.mockResolvedValueOnce([
      { cases: [{ id: "case-1", status: "resolved", outcome: "new_right" }] },
    ]);

    const response = await GET(
      new Request("http://localhost/api/contradictions?status=resolved")
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      cases: [{ id: "case-1", status: "resolved", outcome: "new_right" }],
    });
    expect(query).toHaveBeenCalledWith(
      "SELECT list_contradiction_cases($1, 200) AS cases",
      "resolved"
    );
  });

  it("rejects unknown ledger states without querying", async () => {
    const response = await GET(
      new Request("http://localhost/api/contradictions?status=deleted")
    );

    expect(response.status).toBe(422);
    expect(await response.json()).toEqual({ error: "invalid status" });
    expect(query).not.toHaveBeenCalled();
  });
});
