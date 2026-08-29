import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({
  prisma: {
    $queryRawUnsafe: vi.fn(),
  },
}));

const query = vi.mocked(prisma.$queryRawUnsafe);

describe("/api/outbox", () => {
  afterEach(() => query.mockReset());

  it("returns every pending decision surface with the message feed", async () => {
    query
      .mockResolvedValueOnce([{ feed: { unread: 1, messages: [{ id: "message-1" }] } }])
      .mockResolvedValueOnce([{ requests: [{ id: "request-1" }] }])
      .mockResolvedValueOnce([
        {
          automations: [
            {
              id: "automation-1",
              title: "Morning briefing",
              status: "pending",
            },
          ],
        },
      ])
      .mockResolvedValueOnce([
        {
          contradictions: [
            {
              id: "contradiction-1",
              code: "ABC12345",
              status: "pending",
            },
          ],
        },
      ])
      .mockResolvedValueOnce([
        {
          node_pairings: [
            {
              id: "pairing-1",
              code: "A1B2C3D4",
              name: "Kitchen Mac",
              status: "pending",
            },
          ],
        },
      ]);

    const response = await GET();

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      unread: 1,
      messages: [{ id: "message-1" }],
      pending_requests: [{ id: "request-1" }],
      pending_automations: [
        { id: "automation-1", title: "Morning briefing", status: "pending" },
      ],
      pending_contradictions: [
        { id: "contradiction-1", code: "ABC12345", status: "pending" },
      ],
      pending_node_pairings: [
        {
          id: "pairing-1",
          code: "A1B2C3D4",
          name: "Kitchen Mac",
          status: "pending",
        },
      ],
    });
  });
});
