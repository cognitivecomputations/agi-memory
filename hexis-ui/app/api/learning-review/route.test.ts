import { afterEach, describe, expect, it, vi } from "vitest";
import { GET } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({ prisma: { $queryRawUnsafe: vi.fn() } }));
const query = vi.mocked(prisma.$queryRawUnsafe);

describe("/api/learning-review", () => {
  afterEach(() => query.mockReset());

  it("lists the durable weekly diff", async () => {
    query.mockResolvedValueOnce([{ reviews: [{ id: "review", status: "pending", items: [] }] }]);
    const response = await GET(new Request("http://localhost/api/learning-review?status=pending"));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ reviews: [{ id: "review", status: "pending", items: [] }] });
    expect(query).toHaveBeenCalledWith("SELECT list_learning_reviews($1, 50) AS reviews", "pending");
  });

  it("rejects unknown states before querying", async () => {
    const response = await GET(new Request("http://localhost/api/learning-review?status=deleted"));
    expect(response.status).toBe(422);
    expect(query).not.toHaveBeenCalled();
  });
});
