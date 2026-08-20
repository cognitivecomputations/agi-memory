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
});
