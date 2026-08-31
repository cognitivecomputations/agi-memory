import { afterEach, describe, expect, it, vi } from "vitest";

import { GET, POST } from "./route";
import { prisma } from "@/lib/prisma";

vi.mock("@/lib/prisma", () => ({
  prisma: { $queryRawUnsafe: vi.fn() },
}));

const query = vi.mocked(prisma.$queryRawUnsafe);
const settingsRow = {
  enabled: false,
  provider: "local_piper",
  model: "en_US-lessac-medium",
  provider_models: { local_piper: "en_US-lessac-medium", remote_tts: "remote" },
  voice: "",
  talk_enabled: false,
  wake_enabled: false,
};

describe("/api/settings/voice-output", () => {
  afterEach(() => query.mockReset());

  it("derives the supported provider catalog from database configuration", async () => {
    query.mockResolvedValueOnce([settingsRow]);

    const response = await GET();

    expect(response.status).toBe(200);
    expect((await response.json()).providers).toEqual([
      { id: "local_piper", model: "en_US-lessac-medium" },
    ]);
  });

  it("refuses Talk mode unless speech output is enabled", async () => {
    const response = await POST(new Request("http://localhost", {
      method: "POST",
      body: JSON.stringify({
        provider: "local_piper",
        enabled: false,
        talk_enabled: true,
        wake_enabled: false,
      }),
    }));

    expect(response.status).toBe(400);
    expect(query).not.toHaveBeenCalled();
  });

  it("refuses the wake server gate unless speech output is enabled", async () => {
    const response = await POST(new Request("http://localhost", {
      method: "POST",
      body: JSON.stringify({
        provider: "local_piper",
        enabled: false,
        talk_enabled: false,
        wake_enabled: true,
      }),
    }));

    expect(response.status).toBe(400);
    expect(query).not.toHaveBeenCalled();
  });

  it("derives the model from the live provider catalog", async () => {
    query
      .mockResolvedValueOnce([{ catalog: settingsRow.provider_models }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ ...settingsRow, enabled: true, talk_enabled: true }]);

    const response = await POST(new Request("http://localhost", {
      method: "POST",
      body: JSON.stringify({
        provider: "local_piper",
        enabled: true,
        talk_enabled: true,
        wake_enabled: true,
        voice: "speaker-a",
      }),
    }));

    expect(response.status).toBe(200);
    expect(query.mock.calls[1]).toContain(JSON.stringify("en_US-lessac-medium"));
    expect(query.mock.calls[1]).toContain(JSON.stringify("speaker-a"));
    expect(query.mock.calls[1].at(-1)).toBe(JSON.stringify(true));
  });
});
