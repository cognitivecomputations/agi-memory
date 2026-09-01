import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({
  prisma: {
    $queryRawUnsafe: vi.fn(),
  },
}));

const query = vi.mocked(prisma.$queryRawUnsafe);
const ID = "11111111-1111-4111-8111-111111111111";

describe("/api/automations/decide", () => {
  afterEach(() => query.mockReset());

  it("accepts a suggestion through the atomic database transition", async () => {
    query.mockResolvedValueOnce([
      {
        result: {
          ok: true,
          status: "accepted",
          suggestion_id: ID,
          scheduled_task_id: "22222222-2222-4222-8222-222222222222",
        },
      },
    ]);

    const response = await POST(
      new Request("http://localhost/api/automations/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ID, decision: "accept" }),
      })
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({ ok: true, status: "accepted" });
    expect(query).toHaveBeenCalledWith(
      "SELECT accept_automation($1::uuid, 'web', 'dashboard') AS result",
      ID
    );
  });

  it("rejects unknown decisions before touching the database", async () => {
    const response = await POST(
      new Request("http://localhost/api/automations/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ID, decision: "later" }),
      })
    );

    expect(response.status).toBe(422);
    expect(await response.json()).toEqual({
      error: "decision must be 'accept' or 'dismiss'",
    });
    expect(query).not.toHaveBeenCalled();
  });

  it("returns a conflict with the database recovery instruction", async () => {
    query.mockResolvedValueOnce([
      {
        result: {
          ok: false,
          error: "already_accepted",
          next_step: "Cancel or pause the scheduled task if you no longer want it.",
        },
      },
    ]);

    const response = await POST(
      new Request("http://localhost/api/automations/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ID, decision: "dismiss" }),
      })
    );

    expect(response.status).toBe(409);
    expect(await response.json()).toMatchObject({
      ok: false,
      error: "already_accepted",
    });
  });
});
