import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({
  prisma: { $queryRawUnsafe: vi.fn() },
}));

const query = vi.mocked(prisma.$queryRawUnsafe);
const ID = "11111111-1111-4111-8111-111111111111";

describe("/api/questions/answer", () => {
  afterEach(() => query.mockReset());

  it("answers a listed choice through the atomic database transition", async () => {
    query.mockResolvedValueOnce([
      { result: { ok: true, id: ID, status: "answered", answer: "Hartford" } },
    ]);
    const response = await POST(
      new Request("http://localhost/api/questions/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ID, choice_index: 2 }),
      })
    );
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({ ok: true, answer: "Hartford" });
    expect(query).toHaveBeenCalledWith(
      "SELECT answer_agent_question($1::uuid, $2, $3, 'web', 'dashboard') AS result",
      ID,
      null,
      2
    );
  });

  it("requires either a choice or free text", async () => {
    const response = await POST(
      new Request("http://localhost/api/questions/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ID }),
      })
    );
    expect(response.status).toBe(422);
    expect(query).not.toHaveBeenCalled();
  });

  it("rejects a malformed question id before querying", async () => {
    const response = await POST(
      new Request("http://localhost/api/questions/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: "not-a-question", choice_index: 1 }),
      })
    );
    expect(response.status).toBe(422);
    expect(await response.json()).toEqual({ error: "question id must be a UUID" });
    expect(query).not.toHaveBeenCalled();
  });

  it("keeps a stale question conflict actionable", async () => {
    query.mockResolvedValueOnce([
      {
        result: {
          ok: false,
          status: "timed_out",
          error: "question_timed_out",
          message: "That question timed out.",
        },
      },
    ]);
    const response = await POST(
      new Request("http://localhost/api/questions/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ID, answer: "The Hartford one" }),
      })
    );
    expect(response.status).toBe(409);
    expect(await response.json()).toMatchObject({ error: "question_timed_out" });
  });
});
