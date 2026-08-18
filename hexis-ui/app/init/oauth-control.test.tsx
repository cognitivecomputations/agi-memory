import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { OAuthControl } from "./oauth-control";

describe("OAuthControl", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("completes an authorization-code flow entirely in the UI", async () => {
    const onAuthenticated = vi.fn();
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ provider: "openai-codex", configured: false }))
      .mockResolvedValueOnce(
        jsonResponse({
          session_id: "session-1",
          provider: "openai-codex",
          flow: "authorization_code",
          status: "awaiting_code",
          expires_in_seconds: 600,
          authorization_url: "https://example.test/authorize",
          callback_active: false,
        })
      )
      .mockResolvedValueOnce(
        jsonResponse({
          session_id: "session-1",
          provider: "openai-codex",
          flow: "authorization_code",
          status: "complete",
          expires_in_seconds: 590,
          credential: {
            provider: "openai-codex",
            configured: true,
            email: "person@example.test",
          },
        })
      );

    render(
      <OAuthControl
        provider="openai-codex"
        label="OpenAI Codex"
        refreshKey={0}
        onAuthenticated={onAuthenticated}
      />
    );

    expect(await screen.findByText("Authentication required")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Start authorization" }));

    const link = await screen.findByRole("link", { name: "Open authorization page" });
    expect(link).toHaveAttribute("href", "https://example.test/authorize");
    fireEvent.change(screen.getByLabelText("Authorization code or redirect URL"), {
      target: { value: "code#state" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Complete authorization" }));

    expect(await screen.findByText("Authorization complete.")).toBeVisible();
    await waitFor(() => expect(onAuthenticated).toHaveBeenCalledWith("openai-codex"));
    expect(fetchMock.mock.calls[2][0]).toBe("/api/init/auth/complete");
  });

  it("surfaces an actionable provider error in place", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ provider: "chutes", configured: false }))
      .mockResolvedValueOnce(
        jsonResponse(
          { detail: "Enter the OAuth client ID from your Chutes developer settings." },
          400
        )
      );

    render(
      <OAuthControl
        provider="chutes"
        label="Chutes"
        refreshKey={0}
        onAuthenticated={vi.fn()}
      />
    );

    fireEvent.click(await screen.findByRole("button", { name: "Start authorization" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Enter the OAuth client ID from your Chutes developer settings."
    );
  });
});

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}
