import { describe, expect, it } from "vitest";

import { normalizeMessagePresentation } from "./message-presentation";

describe("normalizeMessagePresentation", () => {
  it("preserves ordered portable blocks", () => {
    expect(
      normalizeMessagePresentation({
        title: "Deployment",
        tone: "success",
        blocks: [
          { type: "text", text: "Ready" },
          { type: "divider" },
          { type: "context", text: "Live evidence" },
          {
            type: "citation",
            citation_id: "mem-123",
            label: "A memory",
            href: "/memories?memory=123",
            trust_level: 0.42,
            low_trust: true,
            locator: { page_start: 2 },
          },
        ],
      })
    ).toEqual({
      title: "Deployment",
      tone: "success",
      blocks: [
        { type: "text", text: "Ready" },
        { type: "divider" },
        { type: "context", text: "Live evidence" },
        {
          type: "citation",
          citation_id: "mem-123",
          label: "A memory",
          href: "/memories?memory=123",
          trust_level: 0.42,
          low_trust: true,
          locator: { page_start: 2 },
        },
      ],
    });
  });

  it("rejects an unknown block instead of rendering partial content", () => {
    expect(
      normalizeMessagePresentation({
        blocks: [
          { type: "text", text: "Visible" },
          { type: "buttons", buttons: [] },
        ],
      })
    ).toBeUndefined();
  });

  it("uses a neutral tone when an older client receives a new tone", () => {
    expect(
      normalizeMessagePresentation({
        tone: "future-tone",
        blocks: [{ type: "text", text: "Visible" }],
      })?.tone
    ).toBe("neutral");
  });

  it("rejects citation links outside the local memory surfaces", () => {
    expect(
      normalizeMessagePresentation({
        blocks: [{
          type: "citation",
          citation_id: "source-1",
          label: "Unsafe",
          href: "javascript:alert(1)",
          trust_level: 1,
        }],
      })
    ).toBeUndefined();
  });
});
