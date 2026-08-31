import { render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ResponsibilitiesPage from "./page";

describe("ResponsibilitiesPage polling", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === "/api/responsibilities") {
          return Response.json({
            status: { active: 1 },
            responsibilities: [
              {
                id: "responsibility-1",
                title: "Check the source",
                kind: "monitor",
                status: "active",
                priority: "normal",
                user_intent: "Watch for changes.",
                trigger: {},
                evaluator: {},
                sources: [],
                actions: [],
                delivery: {},
                memory_policy: "task_scoped",
                timezone: "UTC",
                next_check_at: null,
                last_checked_at: null,
                last_fired_at: null,
                consecutive_errors: 0,
                consecutive_silent: 0,
                last_error: null,
                created_at: null,
              },
            ],
          });
        }
        if (url === "/api/responsibilities/responsibility-1") {
          return Response.json({ responsibility: null });
        }
        return Response.json({}, { status: 404 });
      }) as unknown as typeof fetch
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("selects the first result without restarting the list poll", async () => {
    render(<ResponsibilitiesPage />);

    await waitFor(() => {
      expect(document.body).toHaveTextContent("Check the source");
    });
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "/api/responsibilities/responsibility-1",
        { cache: "no-store" }
      );
    });

    const listCalls = vi
      .mocked(fetch)
      .mock.calls.filter(([input]) => String(input) === "/api/responsibilities");
    expect(listCalls).toHaveLength(1);
  });
});
