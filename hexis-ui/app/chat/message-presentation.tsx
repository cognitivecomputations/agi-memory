import type { MessagePresentation } from "../../lib/message-presentation";

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(text: string) {
  if (!text) return null;

  const parts: React.ReactNode[] = [];
  const lines = text.split("\n");

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (line.startsWith("```")) {
      const codeLines: string[] = [];
      let closingIndex = index + 1;
      while (
        closingIndex < lines.length &&
        !lines[closingIndex].startsWith("```")
      ) {
        codeLines.push(lines[closingIndex]);
        closingIndex++;
      }
      parts.push(
        <pre
          key={`code-${index}`}
          className="my-2 overflow-x-auto rounded-md bg-[var(--surface-strong)] p-3 text-xs"
        >
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      index = closingIndex;
      continue;
    }

    const formatted = escapeHtml(line)
      .replace(
        /`([^`]+)`/g,
        '<code class="rounded bg-[var(--surface-strong)] px-1.5 py-0.5 text-xs">$1</code>'
      )
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(
        /\[\^([A-Za-z0-9:_-]+)\]/g,
        '<sup><a class="font-mono text-[10px] text-[var(--teal)] underline" href="#citation-$1">[$1]</a></sup>'
      );

    parts.push(
      <span key={`line-${index}`}>
        <span dangerouslySetInnerHTML={{ __html: formatted }} />
        {index < lines.length - 1 && <br />}
      </span>
    );
  }

  return <>{parts}</>;
}

function citationLocator(locator?: Record<string, unknown>): string | null {
  if (!locator) return null;
  const pageStart = Number(locator.page_start || 0);
  const pageEnd = Number(locator.page_end || 0);
  if (pageStart > 0) {
    return `page ${pageStart}${pageEnd > 0 && pageEnd !== pageStart ? `–${pageEnd}` : ""}`;
  }
  if (typeof locator.sheet_name === "string" && locator.sheet_name) {
    const rowStart = Number(locator.row_start || 0);
    const rowEnd = Number(locator.row_end || 0);
    const rows = rowStart > 0
      ? `, row ${rowStart}${rowEnd > 0 && rowEnd !== rowStart ? `–${rowEnd}` : ""}`
      : "";
    return `sheet ${locator.sheet_name}${rows}`;
  }
  if (Array.isArray(locator.heading_path) && locator.heading_path.length > 0) {
    return locator.heading_path.map(String).filter(Boolean).join(" › ");
  }
  return locator.chunk_index != null ? `chunk ${String(locator.chunk_index)}` : null;
}

export function MessagePresentationView({
  presentation,
}: {
  presentation: MessagePresentation;
}) {
  return (
    <div className="space-y-3" data-presentation-tone={presentation.tone}>
      {presentation.title ? (
        <div className="font-semibold">{presentation.title}</div>
      ) : null}
      {presentation.blocks.map((block, index) => {
        if (block.type === "divider") {
          return <hr key={`divider-${index}`} className="border-[var(--outline)]" />;
        }
        if (block.type === "context") {
          return (
            <div key={`context-${index}`} className="text-xs text-[var(--ink-soft)]">
              {renderMarkdown(block.text)}
            </div>
          );
        }
        if (block.type === "citation") {
          const locator = citationLocator(block.locator);
          const trust = Math.round(block.trust_level * 100);
          return (
            <details
              id={`citation-${block.citation_id}`}
              key={`citation-${block.citation_id}`}
              className={`rounded-md border px-3 py-2 text-xs ${
                block.low_trust
                  ? "border-amber-300 bg-amber-50 text-amber-950"
                  : "border-[var(--outline)] bg-[var(--surface)] text-[var(--ink-soft)]"
              }`}
            >
              <summary className="cursor-pointer list-none font-medium text-[var(--foreground)]">
                <span className="font-mono">[{block.citation_id}]</span>{" "}
                {block.label}
                {block.low_trust ? (
                  <span className="ml-2 rounded bg-amber-200 px-1.5 py-0.5 text-[10px] font-semibold uppercase">
                    Low trust
                  </span>
                ) : null}
              </summary>
              <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1">
                {block.source_kind ? <span>{block.source_kind}</span> : null}
                {locator ? <span>{locator}</span> : null}
                <span>trust {trust}%</span>
                <a className="font-semibold text-[var(--teal)] underline" href={block.href}>
                  Open source →
                </a>
              </div>
            </details>
          );
        }
        return <div key={`text-${index}`}>{renderMarkdown(block.text)}</div>;
      })}
    </div>
  );
}
