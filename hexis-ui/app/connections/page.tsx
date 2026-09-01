"use client";

import {
  CheckCircle2,
  Clock3,
  ExternalLink,
  Mail,
  MessageCircle,
  Paperclip,
  Plug,
  RadioTower,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Label, TextInput } from "../components/ui/input";
import { PageHeader } from "../components/ui/page-header";
import { Spinner } from "../components/ui/spinner";
import {
  BackfillJob,
  ConnectorSummary,
  IntegrationStatusData,
  asRecord,
  capabilityKeys,
  stateLabel,
  stringArray,
  summarizeConnectors,
} from "./status";

type IntegrationActionResult = {
  success?: boolean;
  output?: unknown;
  display_output?: string | null;
  error?: string | null;
  detail?: string | null;
};

type IntegrationActionHandler = (
  busyKey: string,
  action: string,
  argumentsPayload?: Record<string, unknown>
) => Promise<void>;

const GMAIL_SETUP_STEPS = [
  "Open the Google setup page.",
  "Create or choose a project named Hexis.",
  "Enable the Gmail API for that project.",
  "Set up the app consent screen. For a personal Gmail account, choose External and add your Gmail address as a test user if Google asks.",
  "On the Credentials page, click Create credentials, choose Google's sign-in client option, set Application type to Desktop app, and name it Hexis.",
  "Download the setup file Google gives you.",
  "Upload that setup file here, then start Google sign-in.",
];

export default function ConnectionsPage() {
  const [data, setData] = useState<IntegrationStatusData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch("/api/integrations/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`Failed to load connections (${response.status})`);
      setData((await response.json()) as IntegrationStatusData);
      setError(null);
    } catch (requestError: unknown) {
      setError(
        requestError instanceof Error ? requestError.message : "Failed to load connections."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const runAction = useCallback<IntegrationActionHandler>(
    async (busyKey, action, argumentsPayload = {}) => {
      if (actionBusy) return;
      setActionBusy(busyKey);
      setNotice(null);
      setError(null);
      try {
        const response = await fetch("/api/integrations/status", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action,
            arguments: argumentsPayload,
            source_session_id: "web-connections",
          }),
        });
        const payload = await readActionPayload(response);
        if (!response.ok || payload.success === false) {
          throw new Error(actionError(payload, response.status));
        }
        setNotice(actionNotice(payload));
        await fetchStatus();
      } catch (requestError: unknown) {
        setError(requestError instanceof Error ? requestError.message : "Action failed.");
      } finally {
        setActionBusy(null);
      }
    },
    [actionBusy, fetchStatus]
  );

  useEffect(() => {
    fetchStatus();
    const timer = window.setInterval(fetchStatus, 15000);
    return () => window.clearInterval(timer);
  }, [fetchStatus]);

  const summaries = useMemo(() => (data ? summarizeConnectors(data) : []), [data]);
  const connectedCount = summaries.filter((item) => item.activeConnections.length > 0).length;
  const pendingCount = summaries.filter((item) => item.activeAttempts.length > 0).length;
  const runningCount = summaries.filter((item) => item.runtime?.running).length;
  const sourceItemCount = summaries.reduce((total, item) => total + item.sourceItemCount, 0);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner label="Loading connections..." />
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div className="border-b border-[var(--outline)] pb-5">
        <PageHeader
          title="Connections"
          subtitle="Connector setup, channel runtime, and backfill status"
        />
      </div>

      {error ? (
        <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      ) : null}
      {notice ? (
        <div className="rounded-md border border-[var(--teal)]/40 bg-[var(--teal)]/5 px-3 py-2 text-sm text-[var(--foreground)]">
          {notice}
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="Connected" value={String(connectedCount)} icon={CheckCircle2} />
        <Metric label="Pending setup" value={String(pendingCount)} icon={Clock3} />
        <Metric label="Running channels" value={String(runningCount)} icon={RadioTower} />
        <Metric label="Source items" value={String(sourceItemCount)} icon={Mail} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        {summaries.map((summary) => (
          <ConnectorCard
            key={summary.connector.id}
            summary={summary}
            actionBusy={actionBusy}
            onAction={runAction}
          />
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.25fr,0.75fr]">
        <BackfillPanel
          jobs={data?.backfill.jobs || []}
          actionBusy={actionBusy}
          onAction={runAction}
        />
        <SourceItemsPanel summaries={summaries} />
      </div>
    </div>
  );
}

function ConnectorCard({
  summary,
  actionBusy,
  onAction,
}: {
  summary: ConnectorSummary;
  actionBusy: string | null;
  onAction: IntegrationActionHandler;
}) {
  const { connector, runtime } = summary;
  const setupManifest = asRecord(connector.setup_manifest);
  const defaultCapabilities = stringArray(setupManifest.default_capabilities);
  const nextStep = asString(setupManifest.user_next_step);
  const capabilities = capabilityKeys(connector.capability_manifest);
  const canStartSetup =
    ["slack", "telegram", "signal", "twitter_x"].includes(connector.id) &&
    summary.activeConnections.length === 0 &&
    summary.activeAttempts.length === 0 &&
    connector.status === "available";
  const isChannelConnector = ["slack", "telegram", "signal"].includes(connector.id);

  return (
    <Card className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <ConnectorIcon id={connector.id} />
            <h2 className="truncate text-sm font-semibold">{connector.display_name}</h2>
          </div>
          <p className="mt-1 text-xs text-[var(--ink-soft)]">
            {humanize(connector.category)} · {authTypeLabel(connector)}
          </p>
        </div>
        <Badge variant={stateVariant(summary.state)} className="capitalize">
          {stateLabel(summary.state)}
        </Badge>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {(defaultCapabilities.length ? defaultCapabilities : capabilities).map((capability) => (
          <Badge key={capability} variant="muted">
            {humanize(capability)}
          </Badge>
        ))}
      </div>

      <div className="space-y-2 text-sm">
        {summary.activeConnections.length ? (
          summary.activeConnections.map((connection) => (
            <div
              key={connection.id}
              className="rounded-md border border-[var(--outline)] px-3 py-2"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="min-w-0 truncate font-medium">
                  {connection.display_name || connection.account_key}
                </span>
                <Badge variant="success">{connection.status}</Badge>
              </div>
              <p className="mt-1 truncate text-xs text-[var(--ink-soft)]">
                {connection.capabilities.map(humanize).join(", ") || "Connected"}
              </p>
              <div className="mt-2 flex justify-end">
                <Button
                  type="button"
                  variant="ghost"
                  disabled={Boolean(actionBusy)}
                  onClick={() => {
                    if (
                      window.confirm(
                        `Disconnect ${connection.display_name || connection.account_key}?`
                      )
                    ) {
                      const revokeAction =
                        connector.id === "gmail"
                          ? "revoke_gmail"
                          : connector.id === "twitter_x"
                            ? "revoke_twitter_x"
                            : ["notion", "spotify", "home_assistant", "weather", "trello"].includes(connector.id)
                              ? "revoke_life"
                              : "revoke_connection";
                      void onAction(
                        `${connector.id}:revoke:${connection.id}`,
                        revokeAction,
                        {
                          connector_id: connector.id,
                          account_key: connection.account_key,
                          reason: "revoked from web connections",
                        }
                      );
                    }
                  }}
                  className="px-2.5 py-1 text-xs"
                >
                  {actionBusy === `${connector.id}:revoke:${connection.id}`
                    ? "Disconnecting..."
                    : "Disconnect"}
                </Button>
              </div>
            </div>
          ))
        ) : (
          <p className="text-sm text-[var(--ink-soft)]">No account connected.</p>
        )}

        {summary.activeAttempts.map((attempt) => (
          <div
            key={attempt.attempt_id}
            className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="min-w-0 truncate font-medium">
                {attempt.requested_capabilities.map(humanize).join(", ") || "Setup"}
              </span>
              <Badge variant="warning">{attempt.status}</Badge>
            </div>
            {attempt.user_next_step ? (
              <p className="mt-1 text-xs text-amber-800">{attempt.user_next_step}</p>
            ) : null}
            {attempt.authorization_url ? (
              <a
                href={attempt.authorization_url}
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-[var(--teal)] underline"
              >
                Open authorization <ExternalLink size={12} />
              </a>
            ) : null}
          </div>
        ))}
      </div>

      {runtime ? (
        <div className="rounded-md border border-[var(--outline)] px-3 py-2 text-xs">
          <div className="flex items-center justify-between gap-3">
            <span className="text-[var(--ink-soft)]">Runtime</span>
            <Badge variant={runtime.running ? "success" : runtime.status === "error" ? "error" : "muted"}>
              {runtime.status}
            </Badge>
          </div>
          {runtime.last_error ? (
            <p className="mt-1 text-red-700">{runtime.last_error}</p>
          ) : (
            <p className="mt-1 text-[var(--ink-soft)]">
              {runtime.configured ? "Configured" : "Not configured"}
              {runtime.updated_at ? ` · ${formatDate(runtime.updated_at)}` : ""}
            </p>
          )}
        </div>
      ) : null}

      {summary.activeConnections.length === 0 && summary.activeAttempts.length === 0 && nextStep ? (
        <p className="text-xs text-[var(--ink-soft)]">{nextStep}</p>
      ) : null}

      {connector.id === "gmail" ? (
        <GmailControls summary={summary} actionBusy={actionBusy} onAction={onAction} />
      ) : null}

      {isChannelConnector ? (
        <ChannelControls
          summary={summary}
          canStartSetup={canStartSetup}
          actionBusy={actionBusy}
          onAction={onAction}
        />
      ) : null}

      {connector.id === "twitter_x" ? (
        <TwitterXControls
          summary={summary}
          actionBusy={actionBusy}
          onAction={onAction}
        />
      ) : null}

      {["notion", "spotify", "home_assistant", "weather", "trello"].includes(connector.id) ? (
        <LifeConnectorControls
          summary={summary}
          actionBusy={actionBusy}
          onAction={onAction}
        />
      ) : null}

      {connector.id === "twitter_x" ? (
        <ArchiveImportControls
          summary={summary}
          actionBusy={actionBusy}
          onAction={onAction}
        />
      ) : null}

      {connector.docs_url ? (
        <a
          href={connector.docs_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 text-xs font-semibold text-[var(--teal)] underline"
        >
          Provider docs <ExternalLink size={12} />
        </a>
      ) : null}
    </Card>
  );
}

function LifeConnectorControls({
  summary,
  actionBusy,
  onAction,
}: {
  summary: ConnectorSummary;
  actionBusy: string | null;
  onAction: IntegrationActionHandler;
}) {
  const { connector } = summary;
  const setup = asRecord(connector.setup_manifest);
  const defaults = stringArray(setup.default_capabilities);
  const availableCapabilities = Object.entries(asRecord(connector.capability_manifest))
    .filter(([, value]) => asString(asRecord(value).status) !== "planned")
    .map(([name]) => name);
  const rawFields = Array.isArray(setup.credential_fields) ? setup.credential_fields : [];
  const fields = rawFields
    .map(asRecord)
    .filter((field) => Boolean(asString(field.name)));
  const [values, setValues] = useState<Record<string, string>>({});
  const [capabilities, setCapabilities] = useState<string[]>(defaults);
  const [authorizationResponse, setAuthorizationResponse] = useState("");
  const connected = summary.activeConnections.length > 0;
  const pending = summary.activeAttempts.find((attempt) => attempt.authorization_url) || null;
  const connectAction = connector.id === "spotify" ? "connect_spotify" : "connect_life";
  const busyKey = `${connector.id}:connect`;
  const completeBusyKey = `${connector.id}:complete`;

  const requiredFieldNames = fields
    .map((field) => asString(field.name))
    .filter((name): name is string => Boolean(name));
  const spotifyHasClientChoice =
    connector.id !== "spotify" || Boolean(values.client_id?.trim() || values.client_id_env?.trim());
  const configured =
    connector.id === "spotify"
      ? spotifyHasClientChoice
      : requiredFieldNames.every((name) => Boolean(values[name]?.trim()));

  const start = () => {
    const payload: Record<string, unknown> = {
      connector_id: connector.id,
      capabilities,
    };
    for (const [name, value] of Object.entries(values)) {
      if (value.trim()) payload[name] = value.trim();
    }
    void onAction(busyKey, connectAction, payload);
  };

  return (
    <div className="space-y-3 rounded-md border border-[var(--outline)] bg-[var(--surface)] p-3">
      <div>
        <p className="text-xs font-semibold">{connector.display_name} setup</p>
        <p className="mt-1 text-xs leading-5 text-[var(--ink-soft)]">
          {asString(setup.user_next_step) || "Choose the least access needed, then verify the provider connection."}
        </p>
      </div>

      {!connected ? (
        <>
          <fieldset className="space-y-2">
            <legend className="text-xs font-semibold">Capabilities</legend>
            <div className="grid gap-2 sm:grid-cols-2">
              {availableCapabilities.map((capability) => {
                const detail = asRecord(asRecord(connector.capability_manifest)[capability]);
                const checked = capabilities.includes(capability);
                return (
                  <label key={capability} className="flex items-start gap-2 rounded-md border border-[var(--outline)] bg-white px-2.5 py-2 text-xs">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(event) =>
                        setCapabilities((current) =>
                          event.target.checked
                            ? [...current.filter((item) => item !== capability), capability]
                            : current.filter((item) => item !== capability)
                        )
                      }
                      className="mt-0.5 h-4 w-4 accent-[var(--teal)]"
                    />
                    <span>
                      <span className="block font-medium">{asString(detail.label) || humanize(capability)}</span>
                      <span className="text-[var(--ink-soft)]">{humanize(asString(detail.scope_kind) || "access")}</span>
                    </span>
                  </label>
                );
              })}
            </div>
          </fieldset>

          <div className="grid gap-2 sm:grid-cols-2">
            {fields.map((field) => {
              const name = asString(field.name) || "";
              const isEnvironmentReference = name.endsWith("_env");
              const inputId = `life-connector-${connector.id}-${name}`;
              const descriptionId = isEnvironmentReference ? `${inputId}-description` : undefined;
              return (
                <div key={name} className="text-xs font-medium">
                  <Label htmlFor={inputId} className="normal-case tracking-normal text-[var(--foreground)]">
                    {asString(field.label) || humanize(name)}
                  </Label>
                  <TextInput
                    id={inputId}
                    value={values[name] || ""}
                    onChange={(event) =>
                      setValues((current) => ({ ...current, [name]: event.target.value }))
                    }
                    placeholder={asString(field.example) || (isEnvironmentReference ? "ENVIRONMENT_VARIABLE_NAME" : "")}
                    className="mt-1"
                    autoComplete="off"
                    aria-describedby={descriptionId}
                  />
                  {isEnvironmentReference ? (
                    <span id={descriptionId} className="mt-1 block font-normal text-[var(--ink-soft)]">
                      Environment variable name only — never paste the secret value.
                    </span>
                  ) : null}
                </div>
              );
            })}
          </div>

          {connector.id === "spotify" ? (
            <p className="text-xs text-[var(--ink-soft)]">
              Choose one client-ID field. Add the exact redirect URI returned after starting setup to your Spotify app.
            </p>
          ) : null}

          <Button
            type="button"
            disabled={Boolean(actionBusy) || !configured || capabilities.length === 0}
            onClick={start}
          >
            {actionBusy === busyKey
              ? "Verifying..."
              : connector.id === "spotify"
                ? "Start Spotify sign-in"
                : `Verify ${connector.display_name}`}
          </Button>
        </>
      ) : (
        <p className="text-xs text-[var(--ink-soft)]">
          Connected provider actions are available in chat and MCP through the {humanize(connector.id)} skill.
        </p>
      )}

      {pending?.authorization_url ? (
        <div className="space-y-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs">
          <a
            href={pending.authorization_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 font-semibold text-[var(--teal)] underline"
          >
            Open Spotify sign-in <ExternalLink size={12} />
          </a>
          <p className="text-amber-900">
            Hexis refreshes this page automatically after Spotify returns to the local callback.
          </p>
          <details>
            <summary className="cursor-pointer font-semibold">Callback did not complete</summary>
            <div className="mt-2 flex gap-2">
              <TextInput
                value={authorizationResponse}
                onChange={(event) => setAuthorizationResponse(event.target.value)}
                placeholder="Paste the full Spotify callback URL"
              />
              <Button
                type="button"
                disabled={Boolean(actionBusy) || !authorizationResponse.trim()}
                onClick={() =>
                  void onAction(completeBusyKey, "complete_spotify", {
                    attempt_id: pending.attempt_id,
                    authorization_response: authorizationResponse.trim(),
                  })
                }
              >
                {actionBusy === completeBusyKey ? "Completing..." : "Complete"}
              </Button>
            </div>
          </details>
        </div>
      ) : null}
    </div>
  );
}

function TwitterXControls({
  summary,
  actionBusy,
  onAction,
}: {
  summary: ConnectorSummary;
  actionBusy: string | null;
  onAction: IntegrationActionHandler;
}) {
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");
  const [useEnvClient, setUseEnvClient] = useState(false);
  const [authorizationResponse, setAuthorizationResponse] = useState("");
  const [includeDmRead, setIncludeDmRead] = useState(false);
  const [includePost, setIncludePost] = useState(false);
  const [includeDmSend, setIncludeDmSend] = useState(false);
  const [stream, setStream] = useState("timeline");
  const [query, setQuery] = useState("");
  const [maxItems, setMaxItems] = useState("100");

  const setupManifest = asRecord(summary.connector.setup_manifest);
  const defaultCapabilities = stringArray(setupManifest.default_capabilities);
  const capabilities = [
    ...(defaultCapabilities.length ? defaultCapabilities : ["read", "search", "ingest"]),
    ...(includeDmRead ? ["dm_read"] : []),
    ...(includePost ? ["send"] : []),
    ...(includeDmSend ? ["dm_send"] : []),
  ];
  const pendingAttempt =
    summary.activeAttempts[0] ||
    summary.recentAttempts.find((attempt) =>
      ["pending_user", "awaiting_input", "in_progress"].includes(attempt.status)
    );
  const connection = summary.activeConnections.find((item) =>
    item.credential_ref !== "local_export:twitter_x_archive"
  ) || null;
  const connectBusy = actionBusy === "twitter_x:connect";
  const completeBusy = actionBusy === "twitter_x:complete";
  const liveBusy = actionBusy === "twitter_x:live";

  return (
    <div className="space-y-3 rounded-md border border-[var(--outline)] px-3 py-3">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-xs font-semibold text-[var(--ink-soft)]">
          Twitter/X actions
        </h3>
        <Badge variant="muted">{capabilities.map(humanize).join(", ")}</Badge>
      </div>

      {!connection ? (
        <div className="grid gap-2 md:grid-cols-[1fr_auto]">
          <div className="space-y-2">
            <TextInput
              value={clientId}
              onChange={(event) => setClientId(event.target.value)}
              placeholder="X app client ID"
              className="py-2 text-xs"
            />
            <TextInput
              type="password"
              value={clientSecret}
              onChange={(event) => setClientSecret(event.target.value)}
              placeholder="Optional X app secret"
              className="py-2 text-xs"
            />
            <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs text-[var(--ink-soft)]">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={includeDmRead}
                  onChange={(event) => setIncludeDmRead(event.target.checked)}
                  className="h-4 w-4 accent-[var(--teal)]"
                />
                DM read
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={includePost}
                  onChange={(event) => setIncludePost(event.target.checked)}
                  className="h-4 w-4 accent-[var(--teal)]"
                />
                Post/reply
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={includeDmSend}
                  onChange={(event) => setIncludeDmSend(event.target.checked)}
                  className="h-4 w-4 accent-[var(--teal)]"
                />
                DM send
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={useEnvClient}
                  onChange={(event) => setUseEnvClient(event.target.checked)}
                  className="h-4 w-4 accent-[var(--teal)]"
                />
                Use server-provided X app
              </label>
            </div>
          </div>
          <Button
            type="button"
            variant="secondary"
            disabled={Boolean(actionBusy)}
            onClick={() =>
              void onAction("twitter_x:connect", "connect_twitter_x", {
                capabilities,
                client_id: clientId.trim() || undefined,
                client_secret: clientSecret.trim() || undefined,
                use_env_client: useEnvClient,
              })
            }
            className="self-start px-3 py-2 text-xs"
          >
            {connectBusy ? "Starting..." : "Connect"}
          </Button>
        </div>
      ) : (
        <div className="grid gap-2 md:grid-cols-[9rem_1fr_7rem_auto]">
          <select
            value={stream}
            onChange={(event) => setStream(event.target.value)}
            className="rounded-md border border-[var(--outline)] bg-white px-2 py-2 text-xs"
            aria-label="Twitter/X import stream"
          >
            <option value="timeline">Timeline</option>
            <option value="mentions">Mentions</option>
            <option value="search">Search</option>
            <option value="dms">DMs</option>
          </select>
          <TextInput
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search query"
            disabled={stream !== "search"}
            className="py-2 text-xs"
          />
          <TextInput
            type="number"
            min={1}
            max={5000}
            value={maxItems}
            onChange={(event) => setMaxItems(event.target.value)}
            aria-label="Max items"
            className="py-2 text-xs"
          />
          <Button
            type="button"
            variant="secondary"
            disabled={Boolean(actionBusy) || (stream === "search" && query.trim().length === 0)}
            onClick={() =>
              void onAction("twitter_x:live", "start_connector_backfill", {
                connector_id: "twitter_x",
                account_key: connection.account_key,
                requested_range: {
                  stream,
                  query: stream === "search" ? query.trim() : undefined,
                  max_messages: parseMaxMessages(maxItems, 5000),
                },
              })
            }
            className="px-3 py-2 text-xs"
          >
            {liveBusy ? "Queuing..." : "Import"}
          </Button>
        </div>
      )}

      {pendingAttempt ? (
        <div className="grid gap-2 md:grid-cols-[1fr_auto]">
          <TextInput
            value={authorizationResponse}
            onChange={(event) => setAuthorizationResponse(event.target.value)}
            placeholder="Paste redirected URL or authorization code"
            className="py-2 text-xs"
          />
          <Button
            type="button"
            variant="secondary"
            disabled={Boolean(actionBusy) || authorizationResponse.trim().length === 0}
            onClick={() =>
              void onAction("twitter_x:complete", "complete_twitter_x", {
                attempt_id: pendingAttempt.attempt_id,
                authorization_response: authorizationResponse.trim(),
              })
            }
            className="px-3 py-2 text-xs"
          >
            {completeBusy ? "Completing..." : "Complete"}
          </Button>
        </div>
      ) : null}
    </div>
  );
}

function GmailControls({
  summary,
  actionBusy,
  onAction,
}: {
  summary: ConnectorSummary;
  actionBusy: string | null;
  onAction: IntegrationActionHandler;
}) {
  const clientSecretFileRef = useRef<HTMLInputElement>(null);
  const [clientSecretPath, setClientSecretPath] = useState("");
  const [useEnvSecret, setUseEnvSecret] = useState(false);
  const [authorizationResponse, setAuthorizationResponse] = useState("");
  const [backfillQuery, setBackfillQuery] = useState("newer_than:30d");
  const [maxMessages, setMaxMessages] = useState("100");
  const [accessTier, setAccessTier] = useState<"read_only" | "write" | "manage">("read_only");
  const [memoryPolicy, setMemoryPolicy] = useState<"remember" | "forget">("remember");
  const [heartbeatDigestEnabled, setHeartbeatDigestEnabled] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [showAdvancedCredentialSetup, setShowAdvancedCredentialSetup] = useState(true);
  const [uploadedSetupFileSaved, setUploadedSetupFileSaved] = useState(false);

  const capabilities =
    accessTier === "manage"
      ? ["read", "search", "send", "reply", "label", "spam_triage", "delete"]
      : accessTier === "write"
        ? ["read", "search", "send", "reply"]
        : ["read", "search"];
  const pendingAttempt =
    summary.activeAttempts[0] ||
    summary.recentAttempts.find((attempt) =>
      ["pending_user", "awaiting_input", "in_progress"].includes(attempt.status)
    );
  const connection = summary.activeConnections[0] || null;
  const connectBusy = actionBusy === "gmail:connect";
  const completeBusy = actionBusy === "gmail:complete";
  const backfillBusy = actionBusy === "gmail:backfill";
  const saveClientBusy = actionBusy === "gmail:save-client";
  const setupManifest = asRecord(summary.connector.setup_manifest);
  const setupManifestSteps = stringArray(setupManifest.setup_steps);
  const setupSteps = setupManifestSteps.length ? setupManifestSteps : GMAIL_SETUP_STEPS;
  const hostedConfigured = setupManifest.hosted_oauth_configured === true;
  const envConfigured = setupManifest.env_client_secret_available === true;
  const storedHeartbeatDigestEnabled = setupManifest.heartbeat_digest_enabled === true;
  const showLocalGoogleSetup = !hostedConfigured && showAdvancedCredentialSetup;
  const technicalNextStep = asString(setupManifest.technical_next_step);
  const hasGmailSignInSetup =
    hostedConfigured || uploadedSetupFileSaved || clientSecretPath.trim().length > 0 || (useEnvSecret && envConfigured);

  const saveClientSecretFile = async (file: File | null | undefined) => {
    if (!file || actionBusy) return;
    setUploadError(null);
    try {
      const parsed = await fileToJsonRecord(file);
      await onAction("gmail:save-client", "save_gmail_client_secret", {
        client_secret_json: parsed,
      });
      setClientSecretPath("");
      setUploadedSetupFileSaved(true);
    } catch (error: unknown) {
      setUploadError(error instanceof Error ? error.message : "Could not read Google setup file.");
    } finally {
      if (clientSecretFileRef.current) clientSecretFileRef.current.value = "";
    }
  };

  return (
    <div className="space-y-3 rounded-md border border-[var(--outline)] px-3 py-3">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-xs font-semibold text-[var(--ink-soft)]">
          Gmail actions
        </h3>
        <Badge variant="muted">{capabilities.map(humanize).join(", ")}</Badge>
      </div>

      {!connection ? (
        <div className="space-y-3">
          <div className="grid gap-2 lg:grid-cols-3">
            {[
              {
                id: "read_only",
                label: "Read/search",
                detail: "Browse and search mail when asked.",
              },
              {
                id: "write",
                label: "Send/reply",
                detail: "Adds draft, send, and reply powers.",
              },
              {
                id: "manage",
                label: "Manage/delete",
                detail: "Adds labels, spam triage, trash, and delete.",
              },
            ].map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => setAccessTier(option.id as "read_only" | "write" | "manage")}
                className={`rounded-md border px-3 py-2 text-left text-xs ${
                  accessTier === option.id
                    ? "border-[var(--teal)] bg-white shadow-sm"
                    : "border-[var(--outline)] bg-white/70 hover:border-[var(--teal)]"
                }`}
              >
                <span className="block font-semibold">{option.label}</span>
                <span className="mt-1 block leading-5 text-[var(--ink-soft)]">
                  {option.detail}
                </span>
              </button>
            ))}
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {[
              {
                id: "remember",
                label: "Remember and learn",
                detail: "Email reads may feed ingestion and user-model evidence.",
              },
              {
                id: "forget",
                label: "Forget after reading",
                detail: "Email reads stay task-scoped by default.",
              },
            ].map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => setMemoryPolicy(option.id as "remember" | "forget")}
                className={`rounded-md border px-3 py-2 text-left text-xs ${
                  memoryPolicy === option.id
                    ? "border-[var(--teal)] bg-white shadow-sm"
                    : "border-[var(--outline)] bg-white/70 hover:border-[var(--teal)]"
                }`}
              >
                <span className="block font-semibold">{option.label}</span>
                <span className="mt-1 block leading-5 text-[var(--ink-soft)]">
                  {option.detail}
                </span>
              </button>
            ))}
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {[
              {
                id: "ask_only",
                label: "Only when I ask",
                detail: "No autonomous heartbeat email checks.",
                enabled: false,
              },
              {
                id: "heartbeat_digest",
                label: "Allow heartbeat checks",
                detail: "Hourly heartbeats may check Gmail for important messages and digests.",
                enabled: true,
              },
            ].map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => setHeartbeatDigestEnabled(option.enabled)}
                className={`rounded-md border px-3 py-2 text-left text-xs ${
                  heartbeatDigestEnabled === option.enabled
                    ? "border-[var(--teal)] bg-white shadow-sm"
                    : "border-[var(--outline)] bg-white/70 hover:border-[var(--teal)]"
                }`}
              >
                <span className="block font-semibold">{option.label}</span>
                <span className="mt-1 block leading-5 text-[var(--ink-soft)]">
                  {option.detail}
                </span>
              </button>
            ))}
          </div>
          <div className="grid gap-2 md:grid-cols-[auto_1fr_auto]">
            {showLocalGoogleSetup ? (
              <>
                <div className="md:col-span-3 space-y-3 rounded-md border border-[var(--outline)] bg-white px-3 py-3">
                  <p className="text-xs font-semibold text-[var(--foreground)]">Gmail sign-in setup</p>
                  <p className="text-xs leading-5 text-[var(--ink-soft)]">
                    This local Hexis build needs a one-time Gmail setup before Samantha can connect.
                    Follow these steps, then upload the setup file Google gives you.
                  </p>
                  <ol className="list-decimal space-y-2 pl-5 text-xs leading-5 text-[var(--ink-soft)]">
                    {setupSteps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ol>
                  {summary.connector.docs_url ? (
                    <a
                      href={summary.connector.docs_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-xs font-semibold text-[var(--teal)] underline"
                    >
                      Open Google setup page <ExternalLink size={12} />
                    </a>
                  ) : null}
                </div>
                <input
                  ref={clientSecretFileRef}
                  type="file"
                  accept="application/json,.json"
                  className="hidden"
                  onChange={(event) => void saveClientSecretFile(event.target.files?.[0])}
                />
                <Button
                  type="button"
                  variant="secondary"
                  disabled={Boolean(actionBusy)}
                  onClick={() => clientSecretFileRef.current?.click()}
                  className="gap-2 px-3 py-2 text-xs"
                >
                  <Paperclip size={14} />
                  {saveClientBusy ? "Saving..." : "Upload Google setup file"}
                </Button>
                <TextInput
                  value={clientSecretPath}
                  onChange={(event) => setClientSecretPath(event.target.value)}
                  placeholder="Or paste the downloaded setup file path"
                  className="py-2 text-xs"
                />
              </>
            ) : (
              <div className="md:col-span-2 rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-xs leading-5 text-[var(--ink-soft)]">
                Use Google sign-in to connect the Gmail powers you selected.
              </div>
            )}
            <Button
              type="button"
              variant="secondary"
              disabled={Boolean(actionBusy) || !hasGmailSignInSetup}
              onClick={() =>
                void onAction("gmail:connect", "connect_gmail", {
                  capabilities,
                  memory_policy: memoryPolicy,
                  heartbeat_digest_enabled: heartbeatDigestEnabled,
                  client_secret_path: clientSecretPath.trim() || undefined,
                  use_hexis_oauth_client: true,
                  use_env_client_secret: useEnvSecret,
                })
              }
              className="px-3 py-2 text-xs"
            >
              {connectBusy ? "Starting..." : "Start Google sign-in"}
            </Button>
          </div>
          {!hostedConfigured ? (
            <button
              type="button"
              onClick={() => setShowAdvancedCredentialSetup((current) => !current)}
              className="text-left text-xs font-semibold text-[var(--teal)] underline"
            >
              {showAdvancedCredentialSetup ? "Hide setup guide" : "Walk me through setup"}
            </button>
          ) : null}
          {showLocalGoogleSetup ? (
            <details className="text-xs text-[var(--ink-soft)]">
              <summary className="cursor-pointer font-semibold text-[var(--teal)]">
                Developer options
              </summary>
              <div className="mt-2 space-y-2">
                {technicalNextStep ? (
                  <p className="rounded-md bg-[var(--surface-strong)] px-3 py-2 leading-5">
                    {technicalNextStep}
                  </p>
                ) : null}
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={useEnvSecret}
                    onChange={(event) => setUseEnvSecret(event.target.checked)}
                    disabled={!envConfigured}
                    className="h-4 w-4 accent-[var(--teal)]"
                  />
                  Use server-provided setup file
                  {!envConfigured ? <span className="text-amber-700">(not configured)</span> : null}
                </label>
              </div>
            </details>
          ) : null}
          {uploadError ? <p className="text-xs text-red-700">{uploadError}</p> : null}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-xs leading-5 text-[var(--ink-soft)]">
            Heartbeat email checks are {storedHeartbeatDigestEnabled ? "enabled" : "off"}. Gmail connection does not by itself allow background reading.
          </div>
          <div className="grid gap-2 md:grid-cols-[1fr_8rem_auto]">
            <TextInput
              value={backfillQuery}
              onChange={(event) => setBackfillQuery(event.target.value)}
              placeholder="Gmail search query"
              className="py-2 text-xs"
            />
            <TextInput
              type="number"
              min={1}
              max={500}
              value={maxMessages}
              onChange={(event) => setMaxMessages(event.target.value)}
              aria-label="Max messages"
              className="py-2 text-xs"
            />
            <Button
              type="button"
              variant="secondary"
              disabled={Boolean(actionBusy)}
              onClick={() =>
                void onAction("gmail:backfill", "start_gmail_backfill", {
                  account_key: connection.account_key,
                  query: backfillQuery.trim() || undefined,
                  max_messages: parseMaxMessages(maxMessages),
                })
              }
              className="px-3 py-2 text-xs"
            >
              {backfillBusy ? "Queuing..." : "Import"}
            </Button>
          </div>
        </div>
      )}

      {pendingAttempt ? (
        <details className="rounded-md border border-[var(--outline)] bg-white px-3 py-2 text-xs">
          <summary className="cursor-pointer font-semibold text-[var(--teal)]">
            Google did not return automatically
          </summary>
          <p className="mt-2 leading-5 text-[var(--ink-soft)]">
            This page refreshes Gmail status automatically. Use this fallback only if Google leaves you on a callback page that did not complete.
          </p>
          <div className="mt-2 grid gap-2 md:grid-cols-[1fr_auto]">
            <TextInput
              value={authorizationResponse}
              onChange={(event) => setAuthorizationResponse(event.target.value)}
              placeholder="Paste Google callback URL or authorization code"
              className="py-2 text-xs"
            />
            <Button
              type="button"
              variant="secondary"
              disabled={Boolean(actionBusy) || authorizationResponse.trim().length === 0}
              onClick={() =>
                void onAction("gmail:complete", "complete_gmail", {
                  attempt_id: pendingAttempt.attempt_id,
                  authorization_response: authorizationResponse.trim(),
                })
              }
              className="px-3 py-2 text-xs"
            >
              {completeBusy ? "Completing..." : "Complete"}
            </Button>
          </div>
        </details>
      ) : null}
    </div>
  );
}

function ArchiveImportControls({
  summary,
  actionBusy,
  onAction,
}: {
  summary: ConnectorSummary;
  actionBusy: string | null;
  onAction: IntegrationActionHandler;
}) {
  const connectorId = summary.connector.id;
  const connection = summary.activeConnections[0] || null;
  const [archivePath, setArchivePath] = useState("");
  const [historyMax, setHistoryMax] = useState("1000");
  const busy = actionBusy === `${connectorId}:history`;

  return (
    <div className="space-y-3 rounded-md border border-[var(--outline)] px-3 py-3">
      <div className="grid gap-2 md:grid-cols-[1fr_7rem_auto]">
        <TextInput
          value={archivePath}
          onChange={(event) => setArchivePath(event.target.value)}
          placeholder="Twitter/X archive directory or JS file"
          className="py-2 text-xs"
        />
        <TextInput
          type="number"
          min={1}
          max={5000}
          value={historyMax}
          onChange={(event) => setHistoryMax(event.target.value)}
          aria-label="Max items"
          className="py-2 text-xs"
        />
        <Button
          type="button"
          variant="secondary"
          disabled={Boolean(actionBusy) || archivePath.trim().length === 0}
          onClick={() =>
            void onAction(`${connectorId}:history`, "start_connector_backfill", {
              connector_id: connectorId,
              account_key: connection?.account_key,
              max_messages: parseMaxMessages(historyMax, 5000),
              export_path: archivePath.trim(),
            })
          }
          className="px-3 py-2 text-xs"
        >
          {busy ? "Queuing..." : "Import archive"}
        </Button>
      </div>
    </div>
  );
}

function ChannelControls({
  summary,
  canStartSetup,
  actionBusy,
  onAction,
}: {
  summary: ConnectorSummary;
  canStartSetup: boolean;
  actionBusy: string | null;
  onAction: IntegrationActionHandler;
}) {
  const connectorId = summary.connector.id;
  const [settings, setSettings] = useState<Record<string, string>>({});
  const [historyTarget, setHistoryTarget] = useState("");
  const [historyMax, setHistoryMax] = useState("100");
  const setupManifest = asRecord(summary.connector.setup_manifest);
  const settingKeys = channelSettingKeys(setupManifest, connectorId);
  const envVars = stringArray(setupManifest.env_vars);
  const startAttempt = summary.activeAttempts[0] || null;
  const startBusy = actionBusy === `${connectorId}:start`;
  const configureBusy = actionBusy === `${connectorId}:configure`;
  const verifyBusy = actionBusy === `${connectorId}:verify`;
  const historyBusy = actionBusy === `${connectorId}:history`;
  const connection = summary.activeConnections[0] || null;
  const configuredSettings = Object.fromEntries(
    settingKeys
      .map((key) => [key, settings[key]?.trim() || ""] as const)
      .filter(([, value]) => value.length > 0)
  );
  const hasSettings = Object.keys(configuredSettings).length > 0;

  return (
    <div className="space-y-3 rounded-md border border-[var(--outline)] px-3 py-3">
      <div className="grid gap-2 md:grid-cols-2">
        {settingKeys.map((settingKey, index) => (
          <TextInput
            key={settingKey}
            value={settings[settingKey] || ""}
            onChange={(event) =>
              setSettings((current) => ({
                ...current,
                [settingKey]: event.target.value,
              }))
            }
            placeholder={`${humanize(settingKey)} (${settingPlaceholder(
              settingKey,
              envVars,
              index
            )})`}
            className="py-2 text-xs"
          />
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {canStartSetup ? (
          <Button
            type="button"
            variant="secondary"
            disabled={Boolean(actionBusy)}
            onClick={() =>
              void onAction(`${connectorId}:start`, "start_setup", {
                connector_id: connectorId,
              })
            }
            className="px-3 py-1.5 text-xs"
          >
            {startBusy ? "Starting..." : "Start setup"}
          </Button>
        ) : null}
        <Button
          type="button"
          variant="secondary"
          disabled={Boolean(actionBusy) || !hasSettings}
          onClick={() =>
            void onAction(`${connectorId}:configure`, "configure_channel", {
              connector_id: connectorId,
              settings: configuredSettings,
            })
          }
          className="px-3 py-1.5 text-xs"
        >
          {configureBusy ? "Saving..." : "Save config"}
        </Button>
        <Button
          type="button"
          variant="ghost"
          disabled={Boolean(actionBusy)}
          onClick={() =>
            void onAction(`${connectorId}:verify`, "verify_channel", {
              connector_id: connectorId,
              attempt_id: startAttempt?.attempt_id,
            })
          }
          className="px-3 py-1.5 text-xs"
        >
          {verifyBusy ? "Checking..." : "Verify config"}
        </Button>
      </div>
      {connection ? (
        <div className="grid gap-2 border-t border-[var(--outline)] pt-3 md:grid-cols-[1fr_7rem_auto]">
          <TextInput
            value={historyTarget}
            onChange={(event) => setHistoryTarget(event.target.value)}
            placeholder={connectorId === "slack" ? "Slack channel ID" : "Local export/archive path"}
            className="py-2 text-xs"
          />
          <TextInput
            type="number"
            min={1}
            max={5000}
            value={historyMax}
            onChange={(event) => setHistoryMax(event.target.value)}
            aria-label="Max messages"
            className="py-2 text-xs"
          />
          <Button
            type="button"
            variant="secondary"
            disabled={Boolean(actionBusy) || historyTarget.trim().length === 0}
            onClick={() =>
              void onAction(`${connectorId}:history`, "start_connector_backfill", {
                connector_id: connectorId,
                account_key: connection.account_key,
                max_messages: parseMaxMessages(historyMax, 5000),
                ...(connectorId === "slack"
                  ? { channel_id: historyTarget.trim() }
                  : { export_path: historyTarget.trim() }),
              })
            }
            className="px-3 py-2 text-xs"
          >
            {historyBusy ? "Queuing..." : "Import history"}
          </Button>
        </div>
      ) : null}
    </div>
  );
}

function BackfillPanel({
  jobs,
  actionBusy,
  onAction,
}: {
  jobs: BackfillJob[];
  actionBusy: string | null;
  onAction: IntegrationActionHandler;
}) {
  return (
    <Card className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Backfill jobs</h2>
        <Badge variant="muted">{jobs.length}</Badge>
      </div>
      {jobs.length === 0 ? (
        <p className="text-sm text-[var(--ink-soft)]">No recent backfill jobs.</p>
      ) : (
        <div className="space-y-2">
          {jobs.slice(0, 8).map((job) => {
            const controlBusy = actionBusy?.startsWith(`backfill:${job.job_id}:`) ?? false;
            const canControl =
              !["completed", "failed", "cancelled"].includes(job.status);
            const controlAction =
              job.connector_id === "gmail" ? "control_gmail_backfill" : "control_connector_backfill";
            const pauseOrResume =
              job.status === "paused" || job.pause_requested ? "resume" : "pause";
            return (
              <div
                key={job.job_id}
                className="grid gap-2 rounded-md border border-[var(--outline)] px-3 py-2 text-sm md:grid-cols-[1fr_auto]"
              >
                <div className="min-w-0">
                  <p className="truncate font-medium">
                    {job.connector_id} · {job.account_key}
                  </p>
                  <p className="truncate text-xs text-[var(--ink-soft)]">
                    {job.cursor_key}
                    {job.updated_at ? ` · ${formatDate(job.updated_at)}` : ""}
                  </p>
                  {job.estimate?.provider_status ? (
                    <p className="mt-1 truncate text-xs text-[var(--ink-soft)]">
                      {String(job.estimate.provider_status)}
                      {job.estimate.estimated_items != null
                        ? ` · ~${String(job.estimate.estimated_items)} items`
                        : ""}
                      {job.estimate.cost_class ? ` · ${String(job.estimate.cost_class)}` : ""}
                    </p>
                  ) : null}
                  {job.error ? <p className="mt-1 text-xs text-red-700">{job.error}</p> : null}
                </div>
                <div className="flex flex-wrap items-center gap-2 md:justify-end">
                  {job.pause_requested ? <Badge variant="warning">pause requested</Badge> : null}
                  {job.cancel_requested ? <Badge variant="error">cancel requested</Badge> : null}
                  <Badge variant={jobVariant(job.status)}>{job.status}</Badge>
                  {canControl ? (
                    <>
                      <Button
                        type="button"
                        variant="ghost"
                        disabled={Boolean(actionBusy)}
                        onClick={() =>
                          void onAction(
                            `backfill:${job.job_id}:${pauseOrResume}`,
                            controlAction,
                            {
                              job_id: job.job_id,
                              action: pauseOrResume,
                              reason: `${pauseOrResume} requested from web connections`,
                            }
                          )
                        }
                        className="px-2.5 py-1 text-xs"
                      >
                        {controlBusy ? "Working..." : humanize(pauseOrResume)}
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        disabled={Boolean(actionBusy)}
                        onClick={() => {
                          if (window.confirm(`Cancel ${job.connector_id} backfill job ${job.job_id}?`)) {
                            void onAction(
                              `backfill:${job.job_id}:cancel`,
                              controlAction,
                              {
                                job_id: job.job_id,
                                action: "cancel",
                                reason: "cancelled from web connections",
                              }
                            );
                          }
                        }}
                        className="px-2.5 py-1 text-xs"
                      >
                        Cancel
                      </Button>
                    </>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}

function SourceItemsPanel({ summaries }: { summaries: ConnectorSummary[] }) {
  const rows = summaries.filter((summary) => summary.sourceItemCount > 0);
  return (
    <Card className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Preserved sources</h2>
        <Badge variant="muted">{rows.length}</Badge>
      </div>
      {rows.length === 0 ? (
        <p className="text-sm text-[var(--ink-soft)]">No connector source items yet.</p>
      ) : (
        <div className="space-y-2">
          {rows.map((summary) => (
            <div
              key={summary.connector.id}
              className="flex items-center justify-between gap-3 rounded-md border border-[var(--outline)] px-3 py-2 text-sm"
            >
              <span className="min-w-0 truncate">{summary.connector.display_name}</span>
              <span className="font-semibold">{summary.sourceItemCount}</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

function Metric({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string;
  icon: typeof CheckCircle2;
}) {
  return (
    <Card className="flex items-center justify-between gap-3">
      <div>
        <p className="text-xs text-[var(--ink-soft)]">{label}</p>
        <p className="mt-1 text-2xl font-semibold">{value}</p>
      </div>
      <Icon size={20} className="text-[var(--teal)]" />
    </Card>
  );
}

function ConnectorIcon({ id }: { id: string }) {
  if (id === "gmail") return <Mail size={16} className="text-[var(--teal)]" />;
  if (["slack", "telegram", "signal"].includes(id)) {
    return <MessageCircle size={16} className="text-[var(--teal)]" />;
  }
  return <Plug size={16} className="text-[var(--teal)]" />;
}

function stateVariant(state: ConnectorSummary["state"]) {
  if (state === "connected" || state === "running") return "success";
  if (state === "pending" || state === "planned") return "warning";
  if (state === "error") return "error";
  return "muted";
}

function jobVariant(status: string) {
  if (status === "completed") return "success";
  if (status === "failed" || status === "cancelled") return "error";
  if (status === "paused") return "warning";
  return "muted";
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function humanize(value: string): string {
  return value.replace(/_/g, " ");
}

function authTypeLabel(connector: { id: string; auth_type: string }): string {
  if (connector.id === "gmail") return "Google sign-in";
  if (connector.auth_type === "oauth2") return "account sign-in";
  return humanize(connector.auth_type);
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

async function readActionPayload(response: Response): Promise<IntegrationActionResult> {
  const text = await response.text();
  if (!text.trim()) return {};
  try {
    return JSON.parse(text) as IntegrationActionResult;
  } catch {
    return { error: text };
  }
}

async function fileToJsonRecord(file: File): Promise<Record<string, unknown>> {
  const parsed = JSON.parse(await file.text()) as unknown;
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("JSON file must contain an object.");
  }
  return parsed as Record<string, unknown>;
}

function actionError(payload: IntegrationActionResult, status: number): string {
  if (payload.error) return payload.error;
  if (payload.detail) return payload.detail;
  const output = asRecord(payload.output);
  const outputError = asString(output.error);
  if (outputError) return outputError;
  return `Action failed (${status})`;
}

function actionNotice(payload: IntegrationActionResult): string {
  const connectorUi = connectorSetupUiFromPayload(payload);
  if (connectorUi) return connectorSetupNotice(connectorUi);
  if (payload.display_output) return payload.display_output;
  const output = asRecord(payload.output);
  return (
    asString(output.next_step) ||
    asString(output.user_next_step) ||
    asString(output.status) ||
    "Action complete."
  );
}

function connectorSetupUiFromPayload(payload: IntegrationActionResult): Record<string, unknown> | null {
  const output = asRecord(payload.output);
  const ui = asRecord(output.ui);
  return ui.kind === "connector_setup" ? ui : null;
}

function connectorSetupNotice(ui: Record<string, unknown>): string {
  const connector = asString(ui.display_name) || asString(ui.connector_id) || "Connector";
  const status = asString(ui.status) || "";
  if (status === "needs_client_secret" || status === "setup") {
    const rawModes = asRecord(ui.credential_step).modes;
    const modes = Array.isArray(rawModes) ? rawModes.map(asRecord) : [];
    const builtInAvailable =
      ui.hexis_oauth_client_available === true ||
      modes.some((mode) => mode.id === "hosted_oauth" && mode.available === true);
    if (builtInAvailable) return `${connector} is ready for Google sign-in.`;
    return `${connector} setup needs one-time Google setup. The panel walks through the steps.`;
  }
  if (status === "client_secret_saved") {
    return `${connector} setup file saved. Start Google sign-in from the setup panel.`;
  }
  if (status === "pending_authorization") {
    return `${connector} sign-in started. Open the provider authorization page; Hexis will complete the connection when it returns.`;
  }
  if (status === "connected") return `${connector} is connected.`;
  return `${connector} setup updated.`;
}

function parseMaxMessages(value: string, max = 500): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return 100;
  return Math.min(max, Math.max(1, parsed));
}

function channelSettingKeys(
  setupManifest: Record<string, unknown>,
  connectorId: string
): string[] {
  const derived = stringArray(setupManifest.config_keys)
    .map((key) => key.split(".").pop() || key)
    .filter((key, index, all) => key.length > 0 && all.indexOf(key) === index);
  if (derived.length > 0) return derived;
  if (connectorId === "slack") return ["bot_token", "app_token", "allowed_channels"];
  if (connectorId === "telegram") return ["bot_token", "allowed_chat_ids"];
  if (connectorId === "signal") return ["phone_number", "api_url", "allowed_numbers"];
  return [];
}

function settingPlaceholder(key: string, envVars: string[], index: number): string {
  if (key.includes("token")) {
    return envVars.find((item) => item.includes("TOKEN")) || "ENV_VAR_NAME";
  }
  if (key.includes("phone")) {
    return envVars.find((item) => item.includes("PHONE")) || "+15551234567";
  }
  if (key.includes("api_url")) {
    return envVars.find((item) => item.includes("API_URL")) || "http://localhost:8080";
  }
  if (key.startsWith("allowed")) return "* or comma-separated IDs";
  return envVars[index] || "value";
}
