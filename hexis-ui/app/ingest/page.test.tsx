import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import IngestPage from "./page";

describe("IngestPage polling", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("loads once and waits for the idle polling interval", async () => {
    const fetchMock = vi.fn(async () => Response.json({ jobs: [] }));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    render(<IngestPage />);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenLastCalledWith("/api/ingest/jobs?limit=25", {
      cache: "no-store",
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(14_999);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not overlap a slow jobs request", async () => {
    let finishRequest: ((response: Response) => void) | undefined;
    const fetchMock = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          finishRequest = resolve;
        })
    );
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    render(<IngestPage />);
    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      finishRequest?.(Response.json({ jobs: [] }));
      await Promise.resolve();
    });
  });
});
