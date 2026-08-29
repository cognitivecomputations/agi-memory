import { afterEach, describe, expect, it, vi } from "vitest";

import { GET, POST } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({
  prisma: { $queryRawUnsafe: vi.fn() },
}));

const query = vi.mocked(prisma.$queryRawUnsafe);
const settingsRow = {
  enabled: false,
  provider: "local_whisper",
  model: "base",
  provider_models: { local_whisper: "base", openai_whisper: "whisper-1" },
  language: "",
  cloud_disclosure_accepted: false,
};

describe("/api/settings/voice-notes", () => {
  afterEach(() => query.mockReset());

  it("returns the provider catalog from database configuration", async () => {
    query.mockResolvedValueOnce([settingsRow]);
    const response = await GET();
    expect(response.status).toBe(200);
    expect((await response.json()).providers).toEqual([
      { id: "local_whisper", model: "base" },
      { id: "openai_whisper", model: "whisper-1" },
    ]);
  });

  it("requires explicit disclosure before cloud transcription is enabled", async () => {
    const response = await POST(new Request("http://localhost", {
      method: "POST",
      body: JSON.stringify({ provider: "openai_whisper", enabled: true }),
    }));
    expect(response.status).toBe(400);
    expect(query).not.toHaveBeenCalled();
  });

  it("derives the selected model from the live provider catalog", async () => {
    query
      .mockResolvedValueOnce([{ catalog: settingsRow.provider_models }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ ...settingsRow, enabled: true }]);
    const response = await POST(new Request("http://localhost", {
      method: "POST",
      body: JSON.stringify({ provider: "local_whisper", enabled: true }),
    }));
    expect(response.status).toBe(200);
    expect(query.mock.calls[1]).toContain(JSON.stringify("base"));
  });
});
