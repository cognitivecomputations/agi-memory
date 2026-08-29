import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ForgettingPage from "./page";

const review = {
  id: "11111111-1111-4111-8111-111111111111",
  code: "ABC12345",
  status: "pending",
  reason: "near_protection_threshold",
  preview: "A meaningful project chapter is nearing compression",
  budget_remaining: 2,
  memories: [{
    id: "22222222-2222-4222-8222-222222222222",
    content: "We learned why the first launch failed.",
    importance: 0.8,
    strength: 0.38,
    fidelity: 0.92,
    load_bearing: true,
  }],
};

const observe = {
  irreversible_pruning_enabled: false,
  pressure: {
    episodic_mass: 7.5,
    capacity: 10,
    capacity_ratio: 0.75,
    candidate_groups: 2,
    archived_recoverable: 14,
  },
  low_fidelity_count: 1,
  low_fidelity: [{
    memory_id: "33333333-3333-4333-8333-333333333333",
    content: "A reconstructed recollection.",
    type: "episodic",
    fidelity: 0.4,
  }],
  recent_compressions: [{
    report_id: "44444444-4444-4444-8444-444444444444",
    gist_memory_id: "55555555-5555-4555-8555-555555555555",
    source_count: 3,
    fidelity: 0.7,
    summary_preview: "The launch taught us to verify the migration first.",
    compressed_at: "2026-08-28T12:00:00Z",
  }],
};

describe("forgetting", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows pressure and requires an explicit journal decision before compression", async () => {
    const decisions: Record<string, unknown>[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === "/api/retention/decide") {
        decisions.push(JSON.parse(String(init?.body)));
        return Response.json({
          ok: true,
          decision: "journal",
          compression: { source_count: 1, originals_recoverable: true },
        });
      }
      return Response.json({ observe, reviews: [review] });
    }) as unknown as typeof fetch);

    render(<ForgettingPage />);

    expect(await screen.findByText("We learned why the first launch failed.")).toBeInTheDocument();
    expect(screen.getByText("75% of 10")).toBeInTheDocument();
    expect(screen.getByText("Recoverable; hard pruning is off")).toBeInTheDocument();
    expect(screen.getByText("load-bearing")).toBeInTheDocument();
    expect(decisions).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Journal first" }));
    fireEvent.change(screen.getByLabelText("What should remain in writing?"), {
      target: { value: "Keep the lesson, not every launch detail." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Journal, then compress" }));

    await waitFor(() => expect(decisions[0]).toEqual({
      id: review.id,
      decision: "journal",
      journal_content: "Keep the lesson, not every launch detail.",
    }));
    expect(screen.getByText(/1 source memory entered one recoverable gist/)).toBeInTheDocument();
  });

  it("reports exact source count and stored fidelity after compression", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ observe, reviews: [] })) as unknown as typeof fetch);

    render(<ForgettingPage />);

    expect(await screen.findByText("3 source memories → one gist")).toBeInTheDocument();
    expect(screen.getByText("70% fidelity")).toBeInTheDocument();
    expect(screen.getByText("The launch taught us to verify the migration first.")).toBeInTheDocument();
  });
});
