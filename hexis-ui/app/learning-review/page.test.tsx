import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import LearningReviewPage from "./page";

const memoryItem = {
  id: "11111111-1111-4111-8111-111111111111",
  code: "ABC12345",
  kind: "semantic_belief",
  status: "pending",
  title: "Belief learned",
  content: "The planning call is on Thursday.",
  source_memory_id: "22222222-2222-4222-8222-222222222222",
  evidence: {
    confidence: 0.8,
    trust_level: 0.9,
    source_attribution: { label: "Planning transcript" },
  },
};

const review = {
  id: "33333333-3333-4333-8333-333333333333",
  status: "pending",
  summary: "I learned one scheduling detail worth checking with you.",
  period_start: "2026-08-21T12:00:00Z",
  period_end: "2026-08-28T12:00:00Z",
  items: [memoryItem],
};

describe("learning review", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows grounded evidence and records a correction in place", async () => {
    const decisions: Record<string, unknown>[] = [];
    let pending = true;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === "/api/learning-review/decide") {
        decisions.push(JSON.parse(String(init?.body)));
        pending = false;
        return Response.json({
          ok: true,
          status: "corrected",
          next_step: "The prior version remains queryable in memory history.",
        });
      }
      return Response.json({ reviews: pending ? [review] : [] });
    }) as unknown as typeof fetch);

    render(<LearningReviewPage />);
    expect(await screen.findByText("The planning call is on Thursday.")).toBeInTheDocument();
    expect(screen.getByText("Planning transcript")).toBeInTheDocument();
    expect(screen.getByText("confidence 80%")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open memory" })).toHaveAttribute(
      "href",
      "/memories?memory=22222222-2222-4222-8222-222222222222"
    );
    expect(decisions).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Correct" }));
    fireEvent.change(screen.getByLabelText("What should this say instead?"), {
      target: { value: "The planning call is on Friday." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save correction" }));

    await waitFor(() => expect(decisions[0]).toMatchObject({
      id: memoryItem.id,
      action: "correct",
      correction: "The planning call is on Friday.",
    }));
    expect(screen.getByText(/prior version remains queryable/)).toBeInTheDocument();
  });

  it("asks again before forgetting a load-bearing memory", async () => {
    const decisions: Record<string, unknown>[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === "/api/learning-review/decide") {
        const decision = JSON.parse(String(init?.body));
        decisions.push(decision);
        if (!decision.confirm_load_bearing) {
          return Response.json({
            ok: false,
            confirmation_required: true,
            message: "This memory is protected or load-bearing. Nothing changed.",
          }, { status: 409 });
        }
        return Response.json({ ok: true, status: "forgotten", next_step: "Left active recall." });
      }
      return Response.json({ reviews: [review] });
    }) as unknown as typeof fetch);

    render(<LearningReviewPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Forget" }));
    expect(await screen.findByText(/Nothing changed/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Confirm forget" })).toBeInTheDocument();
    expect(decisions).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Confirm forget" }));
    await waitFor(() => expect(decisions[1]).toMatchObject({
      action: "forget",
      confirm_load_bearing: true,
    }));
  });
});
