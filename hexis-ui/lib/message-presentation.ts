export type MessagePresentationTone =
  | "neutral"
  | "info"
  | "success"
  | "warning"
  | "danger";

export type MessagePresentationBlock =
  | { type: "text"; text: string }
  | { type: "context"; text: string }
  | { type: "divider" }
  | {
      type: "citation";
      citation_id: string;
      label: string;
      href: string;
      trust_level: number;
      low_trust: boolean;
      source_kind?: string;
      locator?: Record<string, unknown>;
      memory_id?: string;
      document_id?: string;
      chunk_id?: string;
    };

export type MessagePresentation = {
  title?: string;
  tone: MessagePresentationTone;
  blocks: MessagePresentationBlock[];
};

const tones = new Set<MessagePresentationTone>([
  "neutral",
  "info",
  "success",
  "warning",
  "danger",
]);

export function normalizeMessagePresentation(
  value: unknown
): MessagePresentation | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const record = value as Record<string, unknown>;
  if (!Array.isArray(record.blocks)) return undefined;

  const blocks: MessagePresentationBlock[] = [];
  for (const valueBlock of record.blocks) {
    if (!valueBlock || typeof valueBlock !== "object" || Array.isArray(valueBlock)) {
      return undefined;
    }
    const block = valueBlock as Record<string, unknown>;
    if (block.type === "divider") {
      blocks.push({ type: "divider" });
      continue;
    }
    if (
      (block.type === "text" || block.type === "context") &&
      typeof block.text === "string" &&
      block.text.trim()
    ) {
      blocks.push({ type: block.type, text: block.text });
      continue;
    }
    if (block.type === "citation") {
      const citationId = typeof block.citation_id === "string" ? block.citation_id : "";
      const label = typeof block.label === "string" ? block.label.trim() : "";
      const href = typeof block.href === "string" ? block.href : "";
      const trustLevel = typeof block.trust_level === "number" ? block.trust_level : NaN;
      if (
        !/^[A-Za-z0-9:_-]+$/.test(citationId) ||
        !label ||
        !(href.startsWith("/memories?") || href.startsWith("/documents?")) ||
        !Number.isFinite(trustLevel) ||
        trustLevel < 0 ||
        trustLevel > 1
      ) {
        return undefined;
      }
      const locator = block.locator;
      if (locator != null && (typeof locator !== "object" || Array.isArray(locator))) {
        return undefined;
      }
      blocks.push({
        type: "citation",
        citation_id: citationId,
        label,
        href,
        trust_level: trustLevel,
        low_trust: block.low_trust === true,
        ...(typeof block.source_kind === "string" ? { source_kind: block.source_kind } : {}),
        ...(locator ? { locator: locator as Record<string, unknown> } : {}),
        ...(typeof block.memory_id === "string" ? { memory_id: block.memory_id } : {}),
        ...(typeof block.document_id === "string" ? { document_id: block.document_id } : {}),
        ...(typeof block.chunk_id === "string" ? { chunk_id: block.chunk_id } : {}),
      });
      continue;
    }
    return undefined;
  }

  const title = typeof record.title === "string" && record.title.trim()
    ? record.title
    : undefined;
  if (!title && blocks.length === 0) return undefined;
  const tone = typeof record.tone === "string" && tones.has(record.tone as MessagePresentationTone)
    ? (record.tone as MessagePresentationTone)
    : "neutral";
  return { title, tone, blocks };
}
