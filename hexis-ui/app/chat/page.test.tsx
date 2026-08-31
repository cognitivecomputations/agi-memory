import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { isImageAttachmentFile, uploadFileName } from "./attachment-helpers";
import ChatPage from "./page";

describe("chat attachment helpers", () => {
  it("detects pasted clipboard images by mime type", () => {
    const file = new File(["pixels"], "", { type: "image/png" });

    expect(isImageAttachmentFile(file)).toBe(true);
  });

  it("adds an image extension to unnamed clipboard images", () => {
    const file = new File(["pixels"], "", { type: "image/png" });

    expect(uploadFileName(file, "pasted-image-1")).toBe("pasted-image-1.png");
  });

  it("keeps a named upload filename unchanged", () => {
    const file = new File(["pixels"], "diagram.webp", { type: "image/webp" });

    expect(uploadFileName(file, "pasted-image-1")).toBe("diagram.webp");
  });
});

describe("ChatPage attachments", () => {
  const ARTIFACT_ID = "22222222-2222-4222-8222-222222222222";

  const eventStream = (events: string[]) =>
    new Response(
      new ReadableStream({
        start(controller) {
          const encoder = new TextEncoder();
          for (const event of events) controller.enqueue(encoder.encode(event));
          controller.close();
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } }
    );

  const eventStreamThatErrorsAfterDone = () =>
    new Response(
      new ReadableStream({
        pull(controller) {
          const encoder = new TextEncoder();
          controller.enqueue(
            encoder.encode(
              'event: done\ndata: {"assistant":"","session_id":"00000000-0000-4000-8000-000000000001"}\n\n'
            )
          );
          controller.error(new Error("network error"));
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } }
    );

  beforeEach(() => {
    vi.stubGlobal("matchMedia", vi.fn(() => ({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })));
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({
            configured: true,
            agent_name: "Samantha",
            mood: "Ready",
            valence: 0,
          });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({ unread: 0, messages: [], pending_requests: [] });
        }
        if (url.endsWith("/api/attachments")) {
          return Response.json({ prepared: true, artifact_id: ARTIFACT_ID, text: "", readable: false });
        }
        if (url.endsWith("/ingest")) {
          return Response.json({ accepted: true });
        }
        if (url.endsWith("/api/chat")) {
          return eventStream([
            'event: done\ndata: {"assistant":"","session_id":"00000000-0000-4000-8000-000000000001"}\n\n',
          ]);
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("turns pasted clipboard images into sendable file attachments", async () => {
    render(<ChatPage />);

    const composer = await screen.findByLabelText("Message Samantha");
    const image = new File(["pixels"], "", { type: "image/png" });
    fireEvent.paste(composer, {
      clipboardData: {
        getData: () => "",
        files: [],
        items: [
          {
            kind: "file",
            getAsFile: () => image,
          },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getByText(/pasted-image-.*\.png/)).toBeInTheDocument();
    });
  });

  it("sends pasted images as live visual attachments instead of OCR-only notes", async () => {
    const chatBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({
            configured: true,
            agent_name: "Samantha",
            mood: "Ready",
            valence: 0,
          });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({ unread: 0, messages: [], pending_requests: [] });
        }
        if (url.endsWith("/api/attachments")) {
          return Response.json({ prepared: true, artifact_id: ARTIFACT_ID, text: "", readable: false });
        }
        if (url.endsWith("/ingest")) {
          return Response.json({ accepted: true });
        }
        if (url.endsWith("/api/chat")) {
          chatBodies.push(JSON.parse(String(init?.body || "{}")));
          return eventStream([
            'event: done\ndata: {"assistant":"","session_id":"00000000-0000-4000-8000-000000000001"}\n\n',
          ]);
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);

    const composer = await screen.findByLabelText("Message Samantha");
    const image = new File(["pixels"], "", { type: "image/png" });
    fireEvent.paste(composer, {
      clipboardData: {
        getData: () => "",
        files: [],
        items: [
          {
            kind: "file",
            getAsFile: () => image,
          },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getByText(/pasted-image-.*\.png/)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => {
      expect(chatBodies.length).toBe(1);
    });
    const body = chatBodies[0];
    const visualAttachments = body.visual_attachments as Record<string, unknown>[];
    expect(visualAttachments).toHaveLength(1);
    expect(visualAttachments[0].data_url).toMatch(/^data:image\/png;base64,/);
    const addenda = (body.prompt_addenda as string[]).join("\n");
    expect(addenda).toContain("inspect the image directly in this turn");
    expect(addenda).not.toContain("OCR");
    // The message the user sees is theirs alone — no bracketed system notes.
    expect(String(body.message)).not.toContain("[Attached");
    expect(await screen.findByAltText(/pasted-image-.*\.png/)).toBeInTheDocument();
  });

  it("reads an attached PDF at attach time and answers from it in the same turn", async () => {
    const chatBodies: Record<string, unknown>[] = [];
    const ingestCalls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({
            configured: true,
            agent_name: "Samantha",
            mood: "Ready",
            valence: 0,
          });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({ unread: 0, messages: [], pending_requests: [] });
        }
        if (url.endsWith("/api/attachments")) {
          return Response.json({
            prepared: true,
            artifact_id: ARTIFACT_ID,
            filename: "Hartford.pdf",
            mime_type: "application/pdf",
            byte_size: 12,
            kind: "document",
            text: "[Page 1]\nThis Agreement is between Manning and the Author.",
            text_chars: 55,
            truncated: false,
            readable: true,
          });
        }
        if (url.endsWith("/ingest")) {
          ingestCalls.push(url);
          return Response.json({ accepted: true });
        }
        if (url.endsWith("/api/chat")) {
          chatBodies.push(JSON.parse(String(init?.body || "{}")));
          return eventStream([
            'event: done\ndata: {"assistant":"","session_id":"00000000-0000-4000-8000-000000000001"}\n\n',
          ]);
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    const { container } = render(<ChatPage />);
    const composer = await screen.findByLabelText("Message Samantha");
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const pdf = new File(["%PDF-1.7 ..."], "Hartford.pdf", { type: "application/pdf" });
    fireEvent.change(input, { target: { files: [pdf] } });

    // The chip names the file and says what it is — no ingestion vocabulary.
    expect(await screen.findByText("Hartford.pdf")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/^PDF · /)).toBeInTheDocument();
    });

    fireEvent.change(composer, { target: { value: "what do u think about this?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => {
      expect(chatBodies.length).toBe(1);
    });
    const body = chatBodies[0];
    // The agent has the text in hand this turn.
    const addenda = (body.prompt_addenda as string[]).join("\n");
    expect(addenda).toContain("This Agreement is between Manning and the Author.");
    expect(addenda).not.toMatch(/ingest/i);
    // The message stays the user's own words.
    expect(body.message).toBe("what do u think about this?");
    expect(String(body.message)).not.toMatch(/ingest/i);
    // The file travels with the turn so a reloaded conversation still shows it.
    expect(body.attachments).toEqual([
      {
        name: "Hartford.pdf",
        mime_type: "application/pdf",
        byte_size: pdf.size,
        kind: "document",
        artifact_id: ARTIFACT_ID,
      },
    ]);
    // Sending is what files it into memory.
    expect(ingestCalls).toEqual([`/api/attachments/${ARTIFACT_ID}/ingest`]);
  });

  it("says plainly when an attached file could not be read", async () => {
    const chatBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({ configured: true, agent_name: "Samantha", mood: "Ready", valence: 0 });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({ unread: 0, messages: [], pending_requests: [] });
        }
        if (url.endsWith("/api/attachments")) {
          return Response.json({
            prepared: true,
            artifact_id: ARTIFACT_ID,
            kind: "document",
            text: "",
            readable: false,
            reason: "too_large",
          });
        }
        if (url.endsWith("/ingest")) return Response.json({ accepted: true });
        if (url.endsWith("/api/chat")) {
          chatBodies.push(JSON.parse(String(init?.body || "{}")));
          return eventStream([
            'event: done\ndata: {"assistant":"","session_id":"00000000-0000-4000-8000-000000000001"}\n\n',
          ]);
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    const { container } = render(<ChatPage />);
    await screen.findByLabelText("Message Samantha");
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "huge.pdf", { type: "application/pdf" })] },
    });

    expect(await screen.findByText("Too large to read in this message")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => {
      expect(chatBodies.length).toBe(1);
    });
    const addenda = (chatBodies[0].prompt_addenda as string[]).join("\n");
    expect(addenda).toContain("You have not read it");
  });

  it("does not show a network error after the chat stream already completed", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({
            configured: true,
            agent_name: "Samantha",
            mood: "Ready",
            valence: 0,
          });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({ unread: 0, messages: [], pending_requests: [] });
        }
        if (url.endsWith("/api/chat")) {
          return eventStreamThatErrorsAfterDone();
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);

    const composer = await screen.findByLabelText("Message Samantha");
    fireEvent.change(composer, { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => {
      expect(screen.queryByText("Chat error")).not.toBeInTheDocument();
      expect(screen.queryByText("network error")).not.toBeInTheDocument();
    });
  });

  it("renders a streamed clarification card and resumes with the selected answer", async () => {
    const answers: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({ configured: true, agent_name: "Samantha" });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({ unread: 0, messages: [], pending_requests: [] });
        }
        if (url.endsWith("/api/questions/answer")) {
          answers.push(JSON.parse(String(init?.body || "{}")));
          return Response.json({
            ok: true,
            status: "answered",
            answer: "The Hartford one",
          });
        }
        if (url.endsWith("/api/chat")) {
          return eventStream([
            'event: question\ndata: {"kind":"question","id":"11111111-1111-4111-8111-111111111111","prompt":"Which contract should I review?","choices":["The Manning one","The Hartford one"],"allow_free_text":true,"status":"pending"}\n\n',
            'event: done\ndata: {"assistant":"","session_id":"00000000-0000-4000-8000-000000000001"}\n\n',
          ]);
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);
    const composer = await screen.findByLabelText("Message Samantha");
    fireEvent.change(composer, { target: { value: "Review the contract" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    expect(await screen.findByText("Which contract should I review?")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "The Hartford one" }));

    await waitFor(() => {
      expect(answers).toEqual([
        {
          id: "11111111-1111-4111-8111-111111111111",
          choice_index: 2,
        },
      ]);
      expect(screen.getByText(/Answer sent: The Hartford one/)).toBeInTheDocument();
    });
  });
});

describe("ChatPage outbox replies", () => {
  beforeEach(() => {
    vi.stubGlobal("matchMedia", vi.fn(() => ({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("queues a reply for the next heartbeat without using the chat composer", async () => {
    const replies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({
            configured: true,
            agent_name: "Samantha",
            mood: "Ready",
            valence: 0,
          });
        }
        if (url.endsWith("/api/outbox/reply")) {
          replies.push(JSON.parse(String(init?.body || "{}")));
          return Response.json({ queued: true, marked_read: 1 });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({
            unread: 1,
            messages: [
              {
                id: "11111111-1111-4111-8111-111111111111",
                kind: "user",
                intent: "check_in",
                message: "Should I prepare the report?",
                delivered_at: "2026-07-28T12:00:00Z",
                read_at: null,
              },
            ],
            pending_requests: [],
          });
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);

    const composer = await screen.findByLabelText("Message Samantha");
    fireEvent.click(await screen.findByRole("button", { name: /Show inbox/ }));
    expect(await screen.findByText("Should I prepare the report?")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reply" }));

    const replyEditor = await screen.findByLabelText("Reply to Samantha");
    fireEvent.change(replyEditor, { target: { value: "Yes, please do." } });
    fireEvent.click(screen.getByRole("button", { name: "Send reply" }));

    await waitFor(() => {
      expect(replies).toEqual([
        {
          message_id: "11111111-1111-4111-8111-111111111111",
          reply: "Yes, please do.",
        },
      ]);
      expect(screen.getByText("Reply queued for Samantha's next heartbeat.")).toBeInTheDocument();
    });
    expect(composer).toHaveValue("");
    expect(screen.getByRole("heading", { name: "Inbox" })).toBeInTheDocument();
  });

  it("shows an inert automation suggestion and activates it only after Accept", async () => {
    const decisions: Record<string, unknown>[] = [];
    let pending = true;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({ configured: true, agent_name: "Samantha" });
        }
        if (url.endsWith("/api/automations/decide")) {
          decisions.push(JSON.parse(String(init?.body || "{}")));
          pending = false;
          return Response.json({
            ok: true,
            status: "accepted",
            schedule: "daily:08:00",
            scheduled_task_id: "22222222-2222-4222-8222-222222222222",
          });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({
            unread: 0,
            messages: [],
            pending_requests: [],
            pending_automations: pending
              ? [
                  {
                    id: "11111111-1111-4111-8111-111111111111",
                    source: "catalog",
                    title: "Morning briefing",
                    rationale: "Start the day with a deliberate review.",
                    task_spec: { schedule: "daily:08:00" },
                    status: "pending",
                    created_at: "2026-08-27T12:00:00Z",
                  },
                ]
              : [],
          });
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);
    await screen.findByLabelText("Message Samantha");
    fireEvent.click(await screen.findByRole("button", { name: /Show inbox/ }));

    expect(await screen.findByText("Morning briefing")).toBeInTheDocument();
    expect(screen.getByText("Nothing runs unless you accept.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Accept" }));

    await waitFor(() => {
      expect(decisions).toEqual([
        {
          id: "11111111-1111-4111-8111-111111111111",
          decision: "accept",
        },
      ]);
      expect(
        screen.getByText("Accepted “Morning briefing” (daily:08:00). The scheduled task is active.")
      ).toBeInTheDocument();
    });
  });

  it("surfaces a contradiction as an explicit three-way decision", async () => {
    const decisions: Record<string, unknown>[] = [];
    let pending = true;
    const contradictionId = "44444444-4444-4444-8444-444444444444";
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({ configured: true, agent_name: "Samantha" });
        }
        if (url.endsWith("/api/contradictions/decide")) {
          decisions.push(JSON.parse(String(init?.body || "{}")));
          pending = false;
          return Response.json({ ok: true, status: "tension", outcome: "tension" });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({
            unread: 0,
            messages: [],
            pending_requests: [],
            pending_automations: [],
            pending_node_pairings: [],
            pending_contradictions: pending
              ? [
                  {
                    id: contradictionId,
                    code: "ABC12345",
                    status: "pending",
                    tension: "The payment cadence conflicts.",
                    confidence: 0.94,
                    new_memory_id: null,
                    memory_a: {
                      id: "55555555-5555-4555-8555-555555555555",
                      content: "The retainer is monthly.",
                      trust_level: 0.9,
                      created_at: "2026-06-01T12:00:00Z",
                    },
                    memory_b: {
                      id: "66666666-6666-4666-8666-666666666666",
                      content: "The retainer is quarterly.",
                      trust_level: 0.95,
                      created_at: "2026-08-01T12:00:00Z",
                    },
                  },
                ]
              : [],
          });
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);
    await screen.findByLabelText("Message Samantha");
    fireEvent.click(await screen.findByRole("button", { name: /Show inbox/ }));

    expect(await screen.findByText("The payment cadence conflicts.")).toBeInTheDocument();
    expect(screen.getByText(/Neither memory changes until you choose/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Both, by context" }));

    await waitFor(() => {
      expect(decisions).toEqual([{ id: contradictionId, outcome: "tension" }]);
      expect(
        screen.getByText("Case ABC12345: kept both memories as context-dependent.")
      ).toBeInTheDocument();
    });
  });

  it("pairs a signed companion node only after an inbox approval", async () => {
    const decisions: Record<string, unknown>[] = [];
    let pending = true;
    const pairingId = "33333333-3333-4333-8333-333333333333";
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/status")) {
          return Response.json({ configured: true, agent_name: "Samantha" });
        }
        if (url.endsWith("/api/nodes")) {
          decisions.push(JSON.parse(String(init?.body || "{}")));
          pending = false;
          return Response.json({ ok: true, status: "approved", name: "Studio Mac" });
        }
        if (url.endsWith("/api/outbox")) {
          return Response.json({
            unread: 0,
            messages: [],
            pending_requests: [],
            pending_automations: [],
            pending_node_pairings: pending
              ? [
                  {
                    id: pairingId,
                    code: "A1B2C3D4",
                    node_id: "a".repeat(64),
                    name: "Studio Mac",
                    capabilities: ["system.run", "screen.capture"],
                    status: "pending",
                    requested_at: "2026-08-28T12:00:00Z",
                    expires_at: "2026-08-29T12:00:00Z",
                  },
                ]
              : [],
          });
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);
    await screen.findByLabelText("Message Samantha");
    fireEvent.click(await screen.findByRole("button", { name: /Show inbox/ }));

    expect(await screen.findByText("Studio Mac")).toBeInTheDocument();
    expect(screen.getByText(/Every host action still needs approval/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Approve identity" }));

    await waitFor(() => {
      expect(decisions).toEqual([{ request: pairingId, decision: "approve" }]);
      expect(
        screen.getByText(
          "Approved the signed identity for “Studio Mac”. A waiting node connects automatically."
        )
      ).toBeInTheDocument();
    });
  });
});
