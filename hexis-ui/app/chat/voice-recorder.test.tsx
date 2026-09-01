import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { VoiceRecorder } from "./voice-recorder";

describe("VoiceRecorder", () => {
  afterEach(() => {
    Object.defineProperty(window, "isSecureContext", { configurable: true, value: true });
  });

  it("gives the exact HTTPS recovery path before requesting a microphone", () => {
    Object.defineProperty(window, "isSecureContext", { configurable: true, value: false });
    render(<VoiceRecorder onTranscript={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Record voice message" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Settings → App");
    expect(screen.getByRole("alert")).toHaveTextContent("Tailscale HTTPS");
  });
});
