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
        if (url.endsWith("/api/ingest/file")) {
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
        if (url.endsWith("/api/ingest/file")) {
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
    expect(String(body.message)).toContain("visible in this turn");
    expect(String(body.message)).not.toContain("OCR");
    expect(await screen.findByAltText(/pasted-image-.*\.png/)).toBeInTheDocument();
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

  it("marks a partial failed response as incomplete", async () => {
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
        if (url.endsWith("/api/chat")) {
          chatBodies.push(JSON.parse(String(init?.body || "{}")));
          if (chatBodies.length > 1) {
            return eventStream([
              [
                "event: done",
                'data: {"assistant":"ok","session_id":"00000000-0000-4000-8000-000000000001"}',
                "",
                "",
              ].join("\n"),
            ]);
          }
          return eventStream([
            'event: token\ndata: {"phase":"conscious_final","text":"partial"}\n\n',
            'event: error\ndata: {"message":"stream disconnected"}\n\n',
            [
              "event: failed",
              'data: {"assistant":"partial","incomplete":true}',
              "",
              "",
            ].join("\n"),
          ]);
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);

    const composer = await screen.findByLabelText("Message Samantha");
    fireEvent.change(composer, { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    expect(
      await screen.findByText(
        "Response incomplete: stream disconnected — not added to conversation history. You can retry."
      )
    ).toBeInTheDocument();

    fireEvent.change(composer, { target: { value: "again" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => expect(chatBodies).toHaveLength(2));
    expect(chatBodies[1].history).toBeUndefined();
  });

  it("shows live reasoning activity before answer tokens arrive", async () => {
    let finishStream: (() => void) | undefined;
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
          return new Response(
            new ReadableStream({
              start(controller) {
                const encoder = new TextEncoder();
                controller.enqueue(encoder.encode("event: reasoning\ndata: {}\n\n"));
                finishStream = () => {
                  controller.enqueue(
                    encoder.encode(
                      'event: token\ndata: {"phase":"conscious_final","text":"Visible answer"}\n\n'
                    )
                  );
                  controller.enqueue(
                    encoder.encode(
                      'event: done\ndata: {"assistant":"Visible answer","session_id":"00000000-0000-4000-8000-000000000001"}\n\n'
                    )
                  );
                  controller.close();
                };
              },
            }),
            { headers: { "Content-Type": "text/event-stream" } }
          );
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);

    const composer = await screen.findByLabelText("Message Samantha");
    fireEvent.change(composer, { target: { value: "think first" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    expect(await screen.findByText(/Reasoning\.\.\./)).toBeInTheDocument();
    expect(screen.queryByText("Visible answer")).not.toBeInTheDocument();

    finishStream?.();

    expect(await screen.findByText("Visible answer")).toBeInTheDocument();
  });

  it("marks a cleanly truncated stream incomplete instead of leaving a spinner", async () => {
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
          return eventStream([
            'event: token\ndata: {"phase":"conscious_final","text":"partial"}\n\n',
          ]);
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);

    const composer = await screen.findByLabelText("Message Samantha");
    fireEvent.change(composer, { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    expect(
      await screen.findByText(
        "Response incomplete: The response stream ended before Hexis finished — not added to conversation history. You can retry."
      )
    ).toBeInTheDocument();
    expect(screen.queryByText("Thinking...")).not.toBeInTheDocument();
  });

  it("excludes an HTTP-failed turn from the next request history", async () => {
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
        if (url.endsWith("/api/chat")) {
          chatBodies.push(JSON.parse(String(init?.body || "{}")));
          if (chatBodies.length === 1) {
            return Response.json({ error: "upstream unavailable" }, { status: 502 });
          }
          return eventStream([
            'event: done\ndata: {"assistant":"ok","session_id":"00000000-0000-4000-8000-000000000001"}\n\n',
          ]);
        }
        return Response.json({});
      }) as unknown as typeof fetch
    );

    render(<ChatPage />);

    const composer = await screen.findByLabelText("Message Samantha");
    fireEvent.change(composer, { target: { value: "first" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    expect(
      await screen.findByText(
        "Response incomplete — Failed to reach chat endpoint (502). You can retry."
      )
    ).toBeInTheDocument();

    fireEvent.change(composer, { target: { value: "second" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => expect(chatBodies).toHaveLength(2));
    expect(chatBodies[1].history).toBeUndefined();
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
