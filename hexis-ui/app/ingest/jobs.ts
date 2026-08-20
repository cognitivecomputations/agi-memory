export type IngestJobResult = {
  memories_created?: number;
  [key: string]: unknown;
};

export type IngestJob = {
  id: string;
  kind: string;
  status: string;
  title: string | null;
  attempts: number;
  error: string | null;
  result: IngestJobResult | null;
  progress: Record<string, unknown> | null;
  payload: Record<string, unknown> | null;
  created_at: string;
  updated_at: string | null;
  completed_at: string | null;
  cancel_requested: boolean;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function numberValue(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function booleanValue(value: unknown): boolean {
  return value === true;
}

export function isActiveIngestJob(job: Pick<IngestJob, "status">): boolean {
  return job.status === "pending" || job.status === "in_progress";
}

export function normalizeIngestJob(
  raw: unknown,
  fallback: Partial<IngestJob> = {}
): IngestJob | null {
  const row = asRecord(raw);
  if (!row) return null;

  const payload = asRecord(row.payload) ?? fallback.payload ?? null;
  const result = asRecord(row.result) as IngestJobResult | null;
  const progress = asRecord(row.progress) ?? fallback.progress ?? null;
  const title =
    stringValue(row.title) ??
    stringValue(payload?.title) ??
    stringValue(payload?.filename) ??
    stringValue(payload?.url) ??
    fallback.title ??
    null;
  const id = stringValue(row.id) ?? fallback.id;
  if (!id) return null;

  return {
    id,
    kind: stringValue(row.kind) ?? fallback.kind ?? "text",
    status: stringValue(row.status) ?? fallback.status ?? "pending",
    title,
    attempts: numberValue(row.attempts, fallback.attempts ?? 0),
    error: stringValue(row.error) ?? fallback.error ?? null,
    result: result ?? fallback.result ?? null,
    progress,
    payload,
    created_at:
      stringValue(row.created_at) ??
      fallback.created_at ??
      new Date().toISOString(),
    updated_at: stringValue(row.updated_at) ?? fallback.updated_at ?? null,
    completed_at: stringValue(row.completed_at) ?? fallback.completed_at ?? null,
    cancel_requested: booleanValue(row.cancel_requested) || fallback.cancel_requested === true,
  };
}

export function mergeIngestJobs(
  recent: IngestJob[],
  exact: IngestJob[],
  limit = 25
): IngestJob[] {
  const byId = new Map<string, IngestJob>();
  for (const job of recent) byId.set(job.id, job);
  for (const job of exact) byId.set(job.id, job);
  return Array.from(byId.values())
    .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
    .slice(0, limit);
}

export function retainActiveTrackedJobIds(
  current: string[],
  missingIds: ReadonlySet<string>,
  exactById: ReadonlyMap<string, Pick<IngestJob, "status">>
): string[] {
  const next = current.filter((id) => {
    if (missingIds.has(id)) return false;
    const job = exactById.get(id);
    return job ? isActiveIngestJob(job) : true;
  });

  // React compares state by reference. Preserve it when polling found no
  // changes so consumers cannot accidentally turn a fetch effect into a
  // render/fetch feedback loop.
  return next.length === current.length ? current : next;
}
