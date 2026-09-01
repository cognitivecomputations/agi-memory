import { afterEach, describe, expect, it, vi } from "vitest";
import { GET } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({ prisma: { $queryRawUnsafe: vi.fn() } }));
const query = vi.mocked(prisma.$queryRawUnsafe);

describe("/api/retention", () => {
  afterEach(() => query.mockReset());

  it("returns pressure, fidelity, compression, and reviews together", async () => {
    query.mockResolvedValueOnce([{ observe: { pressure: { episodic_mass: 4 } }, reviews: [] }]);
    const response = await GET(new Request("http://localhost/api/retention?status=pending"));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ observe: { pressure: { episodic_mass: 4 } }, reviews: [] });
  });

  it("rejects an unknown review state", async () => {
    const response = await GET(new Request("http://localhost/api/retention?status=expired"));
    expect(response.status).toBe(422);
    expect(query).not.toHaveBeenCalled();
  });
});
