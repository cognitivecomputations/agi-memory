import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({
  prisma: { $queryRawUnsafe: vi.fn() },
}));

const query = vi.mocked(prisma.$queryRawUnsafe);
const ID = "11111111-1111-4111-8111-111111111111";

describe("/api/contradictions/decide", () => {
  afterEach(() => query.mockReset());

  it.each(["new_right", "old_right", "tension"])(
    "records the explicit %s decision through the database transition",
    async (outcome) => {
      query.mockResolvedValueOnce([
        { result: { ok: true, case_id: ID, status: "resolved", outcome } },
      ]);

      const response = await POST(
        new Request("http://localhost/api/contradictions/decide", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: ID, outcome, note: "Operator decision" }),
        })
      );

      expect(response.status).toBe(200);
      expect(await response.json()).toMatchObject({ ok: true, outcome });
      expect(query).toHaveBeenCalledWith(
        "SELECT decide_contradiction($1::uuid, $2, $3, 'web', 'dashboard') AS result",
        ID,
        outcome,
        "Operator decision"
      );
    }
  );

  it("rejects malformed ids and outcomes before querying", async () => {
    const malformed = await POST(
      new Request("http://localhost/api/contradictions/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: "not-a-uuid", outcome: "new_right" }),
      })
    );
    const unknownOutcome = await POST(
      new Request("http://localhost/api/contradictions/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ID, outcome: "delete_old" }),
      })
    );

    expect(malformed.status).toBe(422);
    expect(await malformed.json()).toEqual({ error: "id must be a UUID" });
    expect(unknownOutcome.status).toBe(422);
    expect(query).not.toHaveBeenCalled();
  });

  it("keeps stale decision conflicts actionable", async () => {
    query.mockResolvedValueOnce([
      { result: { ok: false, error: "case_not_found" } },
    ]);
    const response = await POST(
      new Request("http://localhost/api/contradictions/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ID, outcome: "tension" }),
      })
    );

    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({ ok: false, error: "case_not_found" });
  });
});
