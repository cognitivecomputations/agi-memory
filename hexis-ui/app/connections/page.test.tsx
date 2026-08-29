import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ConnectionsPage from "./page";

const notionConnector = {
  id: "notion",
  display_name: "Notion",
  category: "productivity",
  auth_type: "api_key",
  status: "available",
  capability_manifest: {
    search: { label: "Search shared pages", scope_kind: "read", status: "available" },
    read: { label: "Read shared pages", scope_kind: "read", status: "available" },
    create: { label: "Create pages", scope_kind: "write", status: "available" },
  },
  setup_manifest: {
    default_capabilities: ["search", "read"],
    user_next_step: "Create a Notion integration and choose its token environment variable.",
    credential_fields: [
      {
        name: "token_env",
        label: "Notion token environment variable",
        secret: true,
        example: "NOTION_TOKEN",
      },
    ],
  },
  docs_url: "https://developers.notion.com/docs/create-a-notion-integration",
};

const statusPayload = {
  connectors: [notionConnector],
  connections: [],
  recent_attempts: [],
  channel_runtime: [],
  backfill: { jobs: [], cursors: [], item_counts: [] },
};

describe("Wave B connections setup", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("collects environment references, not secret values, and sends selected capabilities", async () => {
    const posts: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        if (init?.method === "POST") {
          posts.push(JSON.parse(String(init.body)));
          return Response.json({
            success: true,
            output: {
              status: "connected",
              next_step: "Notion verified.",
              ui: { kind: "connector_setup", connector_id: "notion", status: "connected" },
            },
          });
        }
        return Response.json(statusPayload);
      }) as unknown as typeof fetch
    );

    render(<ConnectionsPage />);
    const envInput = await screen.findByLabelText("Notion token environment variable");
    expect(screen.getByText(/Environment variable name only/)).toBeInTheDocument();
    fireEvent.change(envInput, { target: { value: "MY_NOTION_TOKEN" } });
    fireEvent.click(screen.getByRole("checkbox", { name: /Create pages/ }));
    fireEvent.click(screen.getByRole("button", { name: "Verify Notion" }));

    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toEqual({
      action: "connect_life",
      arguments: {
        connector_id: "notion",
        token_env: "MY_NOTION_TOKEN",
        capabilities: ["search", "read", "create"],
      },
      source_session_id: "web-connections",
    });
    expect(JSON.stringify(posts[0])).not.toContain("secret-value");
  });
});
