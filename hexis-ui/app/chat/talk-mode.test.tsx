import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TalkMode } from "./talk-mode";

describe("TalkMode", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    Object.defineProperty(window, "isSecureContext", { value: true, configurable: true });
  });

  it("keeps microphone access off until explicit start", () => {
    const getUserMedia = vi.fn();
    Object.defineProperty(navigator, "mediaDevices", {
      value: { getUserMedia },
      configurable: true,
    });
    render(<TalkMode onUtterance={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Start Talk mode" })).toBeInTheDocument();
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("gives the exact settings step before requesting a microphone", async () => {
    Object.defineProperty(window, "isSecureContext", { value: true, configurable: true });
    class Recorder {}
    vi.stubGlobal("MediaRecorder", Recorder);
    const getUserMedia = vi.fn();
    Object.defineProperty(navigator, "mediaDevices", {
      value: { getUserMedia },
      configurable: true,
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ stt_enabled: false }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    render(<TalkMode onUtterance={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Start Talk mode" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Settings → Voice"));
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("refuses a remote-device microphone without HTTPS", async () => {
    Object.defineProperty(window, "isSecureContext", { value: false, configurable: true });
    render(<TalkMode onUtterance={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Start Talk mode" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("hexis tunnel start");
  });
});
