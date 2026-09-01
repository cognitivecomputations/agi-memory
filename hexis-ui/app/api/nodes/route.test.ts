import { afterEach, describe, expect, it, vi } from "vitest";
import type { NextRequest } from "next/server";

import { POST } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({
  prisma: { $queryRawUnsafe: vi.fn() },
}));

const query = vi.mocked(prisma.$queryRawUnsafe);

describe("/api/nodes", () => {
  afterEach(() => query.mockReset());

  it("approves an exact pending signed identity", async () => {
    query.mockResolvedValueOnce([
      { result: { ok: true, status: "approved", node_id: "a".repeat(64) } },
    ]);
    const response = await POST(
      new Request("http://localhost/api/nodes", {
        method: "POST",
        body: JSON.stringify({ request: "A1B2C3D4", decision: "approve" }),
      }) as NextRequest
    );
    expect(response.status).toBe(200);
    expect(query).toHaveBeenCalledWith(
      "SELECT decide_node_pairing($1, $2, 'dashboard', $3) AS result",
      "A1B2C3D4",
      "approve",
      null
    );
  });

  it("rejects an invalid decision before querying", async () => {
    const response = await POST(
      new Request("http://localhost/api/nodes", {
        method: "POST",
        body: JSON.stringify({ request: "A1B2C3D4", decision: "later" }),
      }) as NextRequest
    );
    expect(response.status).toBe(422);
    expect(query).not.toHaveBeenCalled();
  });
});
