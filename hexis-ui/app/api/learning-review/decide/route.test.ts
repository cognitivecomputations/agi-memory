import { afterEach, describe, expect, it, vi } from "vitest";
import { POST } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({ prisma: { $queryRawUnsafe: vi.fn() } }));
const query = vi.mocked(prisma.$queryRawUnsafe);
const ID = "11111111-1111-4111-8111-111111111111";

describe("/api/learning-review/decide", () => {
  afterEach(() => query.mockReset());

  it("records an explicit correction", async () => {
    query.mockResolvedValueOnce([{ result: { ok: true, status: "corrected" } }]);
    const response = await POST(new Request("http://localhost/api/learning-review/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: ID, action: "correct", correction: "The call is Friday." }),
    }));
    expect(response.status).toBe(200);
    expect(query).toHaveBeenCalledWith(
      "SELECT decide_learning_review_item($1::uuid, $2, $3, 'web', 'dashboard', $4::boolean) AS result",
      ID,
      "correct",
      "The call is Friday.",
      false
    );
  });

  it("returns an actionable conflict for load-bearing forgetting", async () => {
    query.mockResolvedValueOnce([{ result: {
      ok: false,
      confirmation_required: true,
      error: "load_bearing_confirmation_required",
      message: "Nothing changed.",
    } }]);
    const response = await POST(new Request("http://localhost/api/learning-review/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: ID, action: "forget" }),
    }));
    expect(response.status).toBe(409);
    expect(await response.json()).toMatchObject({ confirmation_required: true });
  });
});
