import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ContradictionsPage from "./page";


const contradiction = {
  id: "11111111-1111-4111-8111-111111111111",
  code: "ABC12345",
  status: "pending",
  tension: "The payment cadence conflicts.",
  confidence: 0.94,
  new_memory_id: null,
  memory_a: {
    id: "22222222-2222-4222-8222-222222222222",
    content: "The retainer is monthly.",
    type: "semantic",
    trust_level: 0.9,
    source_attribution: { label: "June call" },
    created_at: "2026-06-01T12:00:00Z",
  },
  memory_b: {
    id: "33333333-3333-4333-8333-333333333333",
    content: "The retainer is quarterly.",
    type: "semantic",
    trust_level: 0.95,
    source_attribution: { label: "Signed contract" },
    created_at: "2026-08-01T12:00:00Z",
  },
  detected_at: "2026-08-02T12:00:00Z",
};

describe("contradiction ledger", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows evidence and changes nothing until the operator chooses", async () => {
    const decisions: Record<string, unknown>[] = [];
    let pending = true;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input) === "/api/contradictions/decide") {
          decisions.push(JSON.parse(String(init?.body)));
          pending = false;
          return Response.json({ ok: true, status: "resolved", outcome: "new_right" });
        }
        return Response.json({ cases: pending ? [contradiction] : [] });
      }) as unknown as typeof fetch
    );

    render(<ContradictionsPage />);

    expect(await screen.findByText("The payment cadence conflicts.")).toBeInTheDocument();
    expect(screen.getByText("The retainer is quarterly.")).toBeInTheDocument();
    expect(screen.getByText("Signed contract")).toBeInTheDocument();
    expect(screen.getByText("detector confidence 94%")).toBeInTheDocument();
    expect(decisions).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Newer is right" }));

    await waitFor(() => {
      expect(decisions).toEqual([
        { id: contradiction.id, outcome: "new_right" },
      ]);
      expect(screen.getByText(/losing memory remains in history/)).toBeInTheDocument();
    });
  });
});
