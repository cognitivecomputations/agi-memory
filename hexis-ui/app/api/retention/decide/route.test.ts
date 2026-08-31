import { afterEach, describe, expect, it, vi } from "vitest";
import { POST } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({ prisma: { $queryRawUnsafe: vi.fn() } }));
const query = vi.mocked(prisma.$queryRawUnsafe);
const ID = "11111111-1111-4111-8111-111111111111";

describe("/api/retention/decide", () => {
  afterEach(() => query.mockReset());

  it("journals only after the explicit request", async () => {
    query.mockResolvedValueOnce([{ result: { ok: true, decision: "journal", status: "released" } }]);
    const response = await POST(new Request("http://localhost/api/retention/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: ID, decision: "journal", journal_content: "Keep the lesson." }),
    }));
    expect(response.status).toBe(200);
    expect(query).toHaveBeenCalledWith(
      "SELECT decide_memory_fade_review($1::uuid, $2, $3, 'web', 'dashboard') AS result",
      ID,
      "journal",
      "Keep the lesson."
    );
  });

  it("keeps finite-budget refusal actionable", async () => {
    query.mockResolvedValueOnce([{ result: { ok: false, error: "no_retention_budget", message: "Nothing changed." } }]);
    const response = await POST(new Request("http://localhost/api/retention/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: ID, decision: "keep" }),
    }));
    expect(response.status).toBe(409);
    expect(await response.json()).toMatchObject({ error: "no_retention_budget" });
  });
});
