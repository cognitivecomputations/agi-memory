import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import MemoryHistoryPage from "./page";

describe("memory history view", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("reconstructs a point-in-time snapshot and explains an empty record", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json({
        success: true,
        output: {
          as_of: "2026-06-01T12:00:00Z",
          count: 1,
          retrieval_mode: "hybrid",
          degraded: false,
          memories: [{
            memory_id: "11111111-1111-4111-8111-111111111111",
            content: "The Manning retainer was monthly.",
            type: "semantic",
            confidence: 0.7,
            trust_level: 0.8,
            valid_from: "2026-05-01T12:00:00Z",
          }],
        },
      })) as unknown as typeof fetch
    );

    render(<MemoryHistoryPage />);
    fireEvent.change(screen.getByLabelText("Topic"), { target: { value: "Manning retainer" } });
    await waitFor(() => expect(screen.getByLabelText("As of")).toHaveValue());
    fireEvent.click(screen.getByRole("button", { name: "Recall snapshot" }));

    expect(await screen.findByText("The Manning retainer was monthly.")).toBeInTheDocument();
    expect(screen.getByText("Historical trust 80%")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open memory" })).toHaveAttribute(
      "href",
      "/memories?memory=11111111-1111-4111-8111-111111111111"
    );
  });

  it("shows the recorded reason behind a history diff", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json({
        success: true,
        output: {
          from_time: "2026-06-01T12:00:00Z",
          to_time: "2026-08-01T12:00:00Z",
          from_snapshot: { as_of: "2026-06-01T12:00:00Z", count: 1, retrieval_mode: "hybrid", memories: [] },
          to_snapshot: { as_of: "2026-08-01T12:00:00Z", count: 1, retrieval_mode: "hybrid", memories: [] },
          added: [],
          expired: [],
          supersessions: [{
            event: "supersession",
            at: "2026-07-01T12:00:00Z",
            reason: "Signed contract changed the payment cadence",
            superseded_content: "Monthly",
            replacement_content: "Quarterly",
          }],
          belief_revisions: [],
          contradiction_decisions: [],
          summary: { added: 0, expired: 0 },
        },
      })) as unknown as typeof fetch
    );

    render(<MemoryHistoryPage />);
    fireEvent.click(screen.getByRole("button", { name: "Compare two times" }));
    fireEvent.change(screen.getByLabelText("Topic"), { target: { value: "Manning retainer" } });
    await waitFor(() => expect(screen.getByLabelText("From")).toHaveValue());
    fireEvent.click(screen.getByRole("button", { name: "Compare history" }));

    expect(await screen.findByText("Why: Signed contract changed the payment cadence")).toBeInTheDocument();
    expect(screen.getByText("Before: Monthly")).toBeInTheDocument();
    expect(screen.getByText("After: Quarterly")).toBeInTheDocument();
  });
});
