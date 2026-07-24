export type JsonRecord = Record<string, unknown>;

export type IntegrationConnector = {
  id: string;
  display_name: string;
  category: string;
  auth_type: string;
  status: string;
  capability_manifest: JsonRecord;
  setup_manifest: JsonRecord;
  docs_url: string | null;
};

export type IntegrationConnection = {
  id: string;
  connector_id: string;
  account_key: string;
  display_name: string | null;
  status: string;
  credential_ref: string | null;
  granted_scopes: string[];
  capabilities: string[];
  source_channel: string | null;
  source_session_id: string | null;
  last_error: string | null;
  connected_at: string | null;
  last_verified_at: string | null;
  revoked_at: string | null;
  updated_at: string | null;
};

export type ConnectionAttempt = {
  attempt_id: string;
  connector_id: string;
  account_key: string | null;
  status: string;
  requested_capabilities: string[];
  requested_scopes: string[];
  authorization_url: string | null;
  user_next_step: string | null;
  source_channel: string | null;
  source_session_id: string | null;
  credential_ref: string | null;
  error: string | null;
  expires_at: string | null;
  completed_at: string | null;
  created_at: string | null;
  updated_at: string | null;
};

export type ChannelRuntime = {
  channel_type: string;
  status: string;
  configured: boolean;
  running: boolean;
  worker_id: string | null;
  pid: number | null;
  last_checked_at: string | null;
  last_started_at: string | null;
  last_stopped_at: string | null;
  last_error: string | null;
  metadata: JsonRecord;
  updated_at: string | null;
};

export type BackfillJob = {
  job_id: string;
  connector_id: string;
  account_key: string;
  cursor_key: string;
  status: string;
  attempts: number;
  max_attempts: number;
  progress: JsonRecord;
  result: JsonRecord;
  estimate?: JsonRecord;
  error: string | null;
  cancel_requested: boolean;
  pause_requested: boolean;
  updated_at: string | null;
  completed_at: string | null;
};

export type SourceItemCount = {
  connector_id: string;
  account_key: string;
  item_kind: string;
  status: string;
  count: number;
  latest_item_at: string | null;
};

export type IntegrationStatusData = {
  connectors: IntegrationConnector[];
  connections: IntegrationConnection[];
  recent_attempts: ConnectionAttempt[];
  channel_runtime: ChannelRuntime[];
  backfill: {
    jobs: BackfillJob[];
    cursors: JsonRecord[];
    item_counts: SourceItemCount[];
  };
  generated_at?: string;
};

export type ConnectorSummary = {
  connector: IntegrationConnector;
  connections: IntegrationConnection[];
  activeConnections: IntegrationConnection[];
  recentAttempts: ConnectionAttempt[];
  activeAttempts: ConnectionAttempt[];
  runtime: ChannelRuntime | null;
  backfillJobs: BackfillJob[];
  sourceItemCount: number;
  state: "connected" | "pending" | "running" | "available" | "planned" | "disabled" | "error";
};

export function asRecord(value: unknown): JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : {};
}

export function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
}

export function capabilityKeys(manifest: JsonRecord): string[] {
  return Object.keys(manifest).sort((a, b) => a.localeCompare(b));
}

export function summarizeConnectors(data: IntegrationStatusData): ConnectorSummary[] {
  const runtimeByChannel = new Map(
    data.channel_runtime.map((runtime) => [runtime.channel_type, runtime])
  );

  return data.connectors.map((connector) => {
    const connections = data.connections.filter((item) => item.connector_id === connector.id);
    const activeConnections = connections.filter((item) => item.status === "connected");
    const recentAttempts = data.recent_attempts.filter(
      (item) => item.connector_id === connector.id
    );
    const activeAttempts = recentAttempts.filter((item) =>
      ["pending_user", "awaiting_input", "pending", "in_progress"].includes(item.status)
    );
    const runtime = runtimeByChannel.get(connector.id) ?? null;
    const backfillJobs = data.backfill.jobs.filter(
      (item) => item.connector_id === connector.id
    );
    const sourceItemCount = data.backfill.item_counts
      .filter((item) => item.connector_id === connector.id)
      .reduce((total, item) => total + item.count, 0);

    let state: ConnectorSummary["state"] = "available";
    if (connector.status === "planned") state = "planned";
    else if (connector.status === "disabled") state = "disabled";
    else if (connections.some((item) => item.status === "error") || runtime?.status === "error") {
      state = "error";
    } else if (activeConnections.length > 0) {
      state = "connected";
    } else if (activeAttempts.length > 0) {
      state = "pending";
    } else if (runtime?.running) {
      state = "running";
    }

    return {
      connector,
      connections,
      activeConnections,
      recentAttempts,
      activeAttempts,
      runtime,
      backfillJobs,
      sourceItemCount,
      state,
    };
  });
}

export function stateLabel(state: ConnectorSummary["state"]): string {
  return state.replace(/_/g, " ");
}
