import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import SettingsPage from "./page";

describe("Settings voice-note choice", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("surfaces the cloud disclosure and saves the explicit choice", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url === "/api/settings") {
        return Response.json({ groups: {}, llm: {}, heartbeat: {}, agent: {}, tools: {} });
      }
      if (url === "/api/settings/voice-notes" && init?.method === "POST") {
        return Response.json({
          enabled: true,
          provider: "openai_whisper",
          model: "whisper-1",
          language: "",
          cloud_disclosure_accepted: true,
          providers: [
            { id: "local_whisper", model: "base" },
            { id: "openai_whisper", model: "whisper-1" },
          ],
        });
      }
      return Response.json({
        enabled: false,
        provider: "local_whisper",
        model: "base",
        language: "",
        cloud_disclosure_accepted: false,
        providers: [
          { id: "local_whisper", model: "base" },
          { id: "openai_whisper", model: "whisper-1" },
        ],
      });
    }) as unknown as typeof fetch);

    render(<SettingsPage />);
    fireEvent.click(await screen.findByRole("tab", { name: "voice" }));
    fireEvent.click(await screen.findByText("Cloud transcription"));
    fireEvent.click(screen.getByLabelText("Enable transcription"));
    fireEvent.click(screen.getByLabelText(/I understand that voice-note audio/));
    fireEvent.click(screen.getByRole("button", { name: "Save voice-note settings" }));

    await screen.findByText("Voice-note transcription is enabled.");
    const post = requests.find((request) => request.init?.method === "POST");
    expect(JSON.parse(String(post?.init?.body))).toMatchObject({
      provider: "openai_whisper",
      enabled: true,
      cloud_acknowledged: true,
    });
    await waitFor(() => expect(screen.getByText("Enabled")).toBeInTheDocument());
  });
});
