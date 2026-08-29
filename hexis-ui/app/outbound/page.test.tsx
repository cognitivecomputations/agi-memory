import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import OutboundPage from "./page";

const ledger = {
  suspended: false,
  events: [
    {
      id: "event-1",
      created_at: "2026-08-28T12:00:00Z",
      entity: "contact:42",
      entity_name: "Alex Example",
      channel: "email",
      recipient: "alex@example.com",
      purpose_kind: "goal",
      purpose_reference: "goal-1",
      charged_cost: 1.5,
      status: "delivered",
      reason: null,
      disclosure_mode: "full",
      urgency: "normal",
    },
  ],
  budgets: [
    {
      entity: "contact:42",
      channel: "email",
      points: -1,
      max_points: 6,
      regen_per_day: 0.25,
      observed_per_week: 1,
      reciprocity: 0.8,
      strain: 1,
      consecutive_silent: 3,
      updated_at: "2026-08-28T12:00:00Z",
    },
  ],
  controls: [
    {
      entity: "contact:42",
      blocked: true,
      suspended: false,
      reason: "recipient_opt_out",
      source_channel: "email",
      updated_at: "2026-08-28T12:00:00Z",
    },
  ],
};

describe("Outbound ledger", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("makes silence, strain, purpose, and recipient STOP visible", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(Response.json(ledger)),
    );

    render(<OutboundPage />);

    expect((await screen.findAllByText("Alex Example")).length).toBeGreaterThan(0);
    expect(screen.getByText("STOP")).toBeInTheDocument();
    expect(screen.getByText("3 unanswered")).toBeInTheDocument();
    expect(screen.getByText("strain 1")).toBeInTheDocument();
    expect(screen.getByText("Purpose:").parentElement).toHaveTextContent("goal");
    expect(
      screen.getByText(/Only this recipient can reverse their opt-out/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Resume this contact" }),
    ).not.toBeInTheDocument();
  });

  it("offers an immediate global kill switch and reflects the result", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ ...ledger, controls: [], budgets: [] }))
      .mockResolvedValueOnce(
        Response.json({
          control: { suspended: true },
          ledger: { ...ledger, suspended: true, controls: [], budgets: [] },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    render(<OutboundPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Pause all outbound" }),
    );

    await screen.findByText("Outbound communication is paused");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
      action: "suspend_global",
    });
  });
});
