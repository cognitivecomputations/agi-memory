"use client";

import {
  FileArchive,
  FileAudio,
  FileCode,
  FileImage,
  FileJson,
  FileSpreadsheet,
  FileText,
  FileVideo,
  Presentation,
  type LucideIcon,
} from "lucide-react";
import Image from "next/image";
import type { ReactNode } from "react";

export type AttachmentKind = "image" | "audio" | "video" | "document";

type Tone = { icon: LucideIcon; className: string };

// The chip says what the file *is* at a glance: a familiar icon in a familiar
// color, the way every desktop and mail client has for thirty years.
const TONES: Record<string, Tone> = {
  pdf: { icon: FileText, className: "bg-red-50 text-[#c5221f]" },
  doc: { icon: FileText, className: "bg-blue-50 text-[#2b579a]" },
  sheet: { icon: FileSpreadsheet, className: "bg-emerald-50 text-[#217346]" },
  slides: { icon: Presentation, className: "bg-orange-50 text-[#c43e1c]" },
  image: { icon: FileImage, className: "bg-[var(--surface-strong)] text-[var(--teal)]" },
  audio: { icon: FileAudio, className: "bg-violet-50 text-violet-700" },
  video: { icon: FileVideo, className: "bg-violet-50 text-violet-700" },
  code: { icon: FileCode, className: "bg-[var(--surface-strong)] text-[var(--teal)]" },
  data: { icon: FileJson, className: "bg-[var(--surface-strong)] text-[var(--teal)]" },
  archive: { icon: FileArchive, className: "bg-amber-50 text-amber-700" },
  text: { icon: FileText, className: "bg-[var(--surface-strong)] text-[var(--ink-soft)]" },
};

const EXTENSION_LABELS: Record<string, [label: string, tone: keyof typeof TONES]> = {
  ".pdf": ["PDF", "pdf"],
  ".doc": ["Word document", "doc"],
  ".docx": ["Word document", "doc"],
  ".odt": ["Document", "doc"],
  ".rtf": ["Rich text", "doc"],
  ".epub": ["EPUB", "doc"],
  ".xls": ["Spreadsheet", "sheet"],
  ".xlsx": ["Spreadsheet", "sheet"],
  ".csv": ["CSV", "sheet"],
  ".tsv": ["TSV", "sheet"],
  ".ppt": ["Presentation", "slides"],
  ".pptx": ["Presentation", "slides"],
  ".key": ["Presentation", "slides"],
  ".md": ["Markdown", "text"],
  ".markdown": ["Markdown", "text"],
  ".txt": ["Text", "text"],
  ".log": ["Log", "text"],
  ".eml": ["Email", "text"],
  ".mbox": ["Mailbox", "text"],
  ".json": ["JSON", "data"],
  ".yaml": ["YAML", "data"],
  ".yml": ["YAML", "data"],
  ".xml": ["XML", "data"],
  ".html": ["HTML", "code"],
  ".htm": ["HTML", "code"],
  ".tex": ["LaTeX", "code"],
  ".ipynb": ["Notebook", "code"],
  ".zip": ["Archive", "archive"],
  ".tar": ["Archive", "archive"],
  ".gz": ["Archive", "archive"],
  ".7z": ["Archive", "archive"],
  ".rar": ["Archive", "archive"],
};

// Extensionless uploads (pasted screenshots, connector payloads) still know
// their MIME type, so it gets the second look before we fall back to "File".
const MIME_LABELS: Record<string, [label: string, tone: keyof typeof TONES]> = {
  "application/pdf": ["PDF", "pdf"],
  "application/msword": ["Word document", "doc"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
    "Word document",
    "doc",
  ],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ["Spreadsheet", "sheet"],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
    "Presentation",
    "slides",
  ],
  "application/zip": ["Archive", "archive"],
  "application/json": ["JSON", "data"],
  "text/plain": ["Text", "text"],
  "text/markdown": ["Markdown", "text"],
  "text/csv": ["CSV", "sheet"],
  "text/html": ["HTML", "code"],
};

function extensionOf(name: string): string {
  const index = name.lastIndexOf(".");
  if (index <= 0) return "";
  return name.slice(index).toLowerCase();
}

function kindFromMime(mimeType?: string | null): AttachmentKind {
  const mime = (mimeType || "").toLowerCase();
  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("audio/")) return "audio";
  if (mime.startsWith("video/")) return "video";
  return "document";
}

/** What to show for a file: its human type name and its icon treatment. */
export function attachmentDescriptor(
  name: string,
  mimeType?: string | null,
  kind?: AttachmentKind | null
): { label: string; icon: LucideIcon; className: string } {
  const extension = extensionOf(name);
  const mapped = EXTENSION_LABELS[extension];
  if (mapped) return { label: mapped[0], ...TONES[mapped[1]] };

  const byMime = MIME_LABELS[(mimeType || "").toLowerCase().split(";")[0].trim()];
  if (byMime) return { label: byMime[0], ...TONES[byMime[1]] };

  const resolved = kind || kindFromMime(mimeType);
  if (resolved === "image") {
    return {
      label: extension ? `${extension.slice(1).toUpperCase()} image` : "Image",
      ...TONES.image,
    };
  }
  if (resolved === "audio") return { label: "Audio", ...TONES.audio };
  if (resolved === "video") return { label: "Video", ...TONES.video };
  if (extension) return { label: extension.slice(1).toUpperCase(), ...TONES.code };
  return { label: "File", ...TONES.text };
}

export type AttachmentCardStatus = "preparing" | "ready" | "error";

export type AttachmentCardProps = {
  name: string;
  mimeType?: string | null;
  kind?: AttachmentKind | null;
  status?: AttachmentCardStatus;
  /** Replaces the type label — an error, or a detail worth saying instead. */
  note?: string | null;
  /** Appended to the type label ("PDF · 754.4 KB"). */
  detail?: string | null;
  thumbnailUrl?: string | null;
  actions?: ReactNode;
  className?: string;
};

/**
 * One attached file, drawn the same way in the composer and in the
 * conversation: icon, name, what it is. The spinner is the only difference
 * between a file still being read and one that is ready.
 */
export function AttachmentCard({
  name,
  mimeType,
  kind,
  status = "ready",
  note,
  detail,
  thumbnailUrl,
  actions,
  className = "",
}: AttachmentCardProps) {
  const { label, icon: Icon, className: toneClass } = attachmentDescriptor(name, mimeType, kind);
  const subtitle =
    status === "preparing"
      ? "Reading…"
      : note || (detail ? `${label} · ${detail}` : label);

  return (
    <div
      className={`flex w-72 max-w-full items-center gap-3 rounded-xl border border-[var(--outline)] bg-white px-3 py-2.5 text-left shadow-[0_1px_2px_var(--shadow)] ${className}`}
    >
      <div
        className={`flex h-10 w-10 flex-none items-center justify-center overflow-hidden rounded-lg ${toneClass}`}
      >
        {status === "preparing" ? (
          <span
            role="status"
            aria-label={`Reading ${name}`}
            className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent"
          />
        ) : thumbnailUrl ? (
          <Image
            src={thumbnailUrl}
            alt=""
            width={40}
            height={40}
            unoptimized
            className="h-10 w-10 object-cover"
          />
        ) : (
          <Icon size={20} aria-hidden="true" />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium leading-5 text-[var(--foreground)]" title={name}>
          {name}
        </p>
        <p
          className={`truncate text-xs leading-4 ${
            status === "error" ? "text-red-700" : "text-[var(--ink-soft)]"
          }`}
          title={subtitle}
        >
          {subtitle}
        </p>
      </div>
      {actions ? <div className="flex flex-none items-center gap-0.5">{actions}</div> : null}
    </div>
  );
}
