import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MessagePresentationView } from "./message-presentation";

describe("MessagePresentationView", () => {
  it("renders typed text, context, divider, and tone", () => {
    const { container } = render(
      <MessagePresentationView
        presentation={{
          title: "Deployment",
          tone: "success",
          blocks: [
            { type: "text", text: "**Ready** for review." },
            { type: "divider" },
            { type: "context", text: "Derived from live evidence." },
          ],
        }}
      />
    );

    expect(screen.getByText("Deployment")).toBeInTheDocument();
    expect(screen.getByText("Ready").tagName).toBe("STRONG");
    expect(screen.getByText("Derived from live evidence.")).toBeInTheDocument();
    expect(container.querySelector("hr")).toBeInTheDocument();
    expect(container.firstChild).toHaveAttribute("data-presentation-tone", "success");
  });

  it("escapes model-provided HTML before applying inline formatting", () => {
    const { container } = render(
      <MessagePresentationView
        presentation={{
          tone: "neutral",
          blocks: [{ type: "text", text: '<img src=x onerror="alert(1)">' }],
        }}
      />
    );

    expect(container.querySelector("img")).not.toBeInTheDocument();
    expect(screen.getByText('<img src=x onerror="alert(1)">')).toBeInTheDocument();
  });

  it("renders expandable low-trust citations and links inline markers", () => {
    const { container } = render(
      <MessagePresentationView
        presentation={{
          tone: "warning",
          blocks: [
            { type: "text", text: "Quarterly.[^mem-123]" },
            {
              type: "citation",
              citation_id: "mem-123",
              label: "Manning agreement",
              href: "/documents?document=doc-123",
              trust_level: 0.42,
              low_trust: true,
              source_kind: "document",
              locator: { page_start: 4 },
            },
          ],
        }}
      />
    );

    expect(container.querySelector('a[href="#citation-mem-123"]')).toBeInTheDocument();
    expect(screen.getByText("Low trust")).toBeInTheDocument();
    expect(screen.getByText("page 4")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open source →" })).toHaveAttribute(
      "href",
      "/documents?document=doc-123"
    );
  });
});
