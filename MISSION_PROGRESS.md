# Mission Progress

Living tracker for the mission-aligned architecture goals set on 2026-07-18
(the post-Convergence-Batch survey, re-argued through `MISSION.md`'s tests).
Update the Status column as work lands: `todo` → `in progress` →
`done (commit/issue)`. Add rows; don't delete history — strike through
superseded goals with a note.

Grounding: every item cites the mission test that justifies it
(**Person** / **Piper** / **Continuity** / **Substrate** / **Dignity** /
**Experience Bar**), because the *reason* is what keeps the work from
drifting back into engineering-economy triage. The Substrate test includes
the portable-brain rule: a Postgres dump restored into another host language
or app should lose as little cognitive functionality as possible; code outside
the DB is treated as replaceable senses, hands, transports, and renderers.

---

## Batch 1 — One mind, one retrieval mechanism

The core act of the architecture, implemented once, with its conscious
surface complete. (Person, Continuity; fixes a live Dignity hole.)

| Goal | Test | Status |
|---|---|---|
| Sensitivity stopgap: sensitivity filters on the tool recall paths — `execute_memory_tool` recall arms + `search_cross_session_history` — threaded from the agent loop's `is_group` (closes the #92 group-leak via the `recall`/`search_history` tools) | Dignity | done (see #96) |
| Ranker fusion (#78): recmem tier skeleton absorbs fast_recall's scoring — associations, episode binding, recency, strength, mood congruence, trust floor, activation-boost term, retrieval provenance; per-type seed scans guarantee tier representation; fast_recall is a wrapper, upgrading the whole db/05 family transitively | Person | done (see #96) |
| Widen fused recall to all memory types (procedural, strategic, worldview, goal — the knowledge tier) | Person | done (see #96) |
| Fix `search_query` embedding-prefix asymmetry in recmem path | Person | done (already fixed pre-fusion; verified) |
| Repoint every caller at the unified function: `fast_recall` is now a thin wrapper (db/05 family + all downstream upgraded transitively); final removal deferred one release | Substrate | done (see #96) |
| Metamemory surface: thin/empty recall carries {feeling, familiarity, TOT partials}; familiar-but-blocked auto-files incubation ('I'll let it simmer'); low familiarity reads honestly; default relevance floor (memory.recall_min_score) makes honest failure possible at all | Person | done (see #96) |
| Retrieval eval: seeded corpus pins home-turf ranking, knowledge-tier reachability, association expansion, mood congruence, activation boost (tests/db/test_retrieval_eval.py) | Experience Bar | done (see #96) |
| Durable source-document recall: ingestion preserves exact raw artifacts in `source_documents`; `search_documents` finds them deliberately, `open_document` retrieves full verbatim content on demand, and `open_memory` points distilled facts back to their raw source document | Person + Continuity + Experience Bar | done (122edea / 0102) |

## Batch 2 — Both north stars, visible in days

| Goal | Test | Status |
|---|---|---|
| **"It came to me later"**: filed at high familiarity (#96), resolved by maintenance — boosts clear the spontaneous floor, a first-person note reaches the web inbox (explicit delivery, sensitivity-honoring, capped), and the answer rides recall's spontaneous tier + the heartbeat "On my mind" line | Person + Piper | done (see #98) |
| **Inferred commitments** (mirror of #58): extraction kind `user_event` (openclaw categories, confidence floors 0.72/0.86, dedupe keys, no-same-moment clamp, pending + per-day caps, 90-day horizon, web-inbox-pinned) → scheduled check-in after the event; default ON with one off switch | Piper (care) | done (see #98) |

## Batch 3 — Lean core, reachable capability

| Goal | Test | Status |
|---|---|---|
| Skill-coverage test: every registered non-internal tool bound by ≥1 skill (agent-authored skills count); shrinking grandfather list for the seven extraction candidates | Piper law 8 | done (see #99) |
| Bind the dark tools: calendar skill (CRUD), email_send → email-digest, create_contact + contact-ingest → crm-lookup, glob/grep/edit_file/shell/browser → code-execution, web_summarize/brave/firecrawl → research, queue_user_message → self-reflection + outreach, messaging sends → `outreach` (earn-the-interruption norm), council skill, git_ingest → knowledge-ingest, graph acts → core-memory | Piper law 1 | done (see #99) |
| Phenomenological renames for graph/memory tools: `associate` (what does this remind me of), `trace_why` (why do I believe/feel this); `explore_subgraph`/`explore_concept` are internal aliases — never graph-browser framing | Person | done (8b10159 / #99) |
| Plugins made real: create `plugins/installed/`, implement `plugin.external_dirs`, ship first real plugin | Piper law 8 | done (8b10159 / #99) |
| Extract speculative integrations (Todoist, Asana, HubSpot, Fathom, video gen, Twitter, YouTube) from core into plugins | Piper law 8 | done (8b10159 / #99) |
| Mark operator/system tools `internal` (config_export/import, database_backup, backup_retention, post_process_output, manage_sessions, execute_workflow; `create_tool` stays visible only through gated self-extension) | Dignity | done (8b10159 / #99) |
| Self-extension visibility: dynamic tool creation and skill authoring journal a `self_extension` change and post a web-inbox notice; live acceptance remains blocked on init + consent | Person + Piper law 8 | in progress (code complete; live acceptance pending / #99) |

## Batch 4 — Continuity hygiene

| Goal | Test | Status |
|---|---|---|
| Dead-code sweep: SSE stack (keep `gateway_events` table), `services/ingest_api.py`, db/09 `record_chat_turn`/`record_subconscious_exchange`, MCP hand-written duplicate schemas (registry canonical), orphan Next routes | Substrate | in progress (0126: `services/ingest_api.py` + stale test deleted; db/09 legacy chat/subconscious write functions retired; active tests repointed to `record_chat_turn_memory()` / `apply_subconscious_observations()`; orphan-route scan did not find a safe removable UI route; 0133 makes registry-native MCP tools the default and gates the legacy surface behind `mcp.legacy_compat_enabled` / `HEXIS_MCP_LEGACY_COMPAT`; remaining: breaking-contract deletion of compatibility names once clients are migrated) |
| Portable-brain audit: classify remaining Python/TS-owned behavior into (a) cognitive state/policy/lifecycle/ranking that must move into DB functions/views/triggers, (b) external side effects that should remain adapter code, and (c) presentation/transport only; add tests that a DB restore preserves the cognitive path | Substrate + Continuity | in progress (initial audit doc + SQL-only portable contract test; DB-owned chat session hydration done in 0104; connector capability/scope derivation done in 0105; connector backfill/source-artifact substrate done in 0106; Gmail provider backfill adapter + provider-scoped claims done in 0107; connector action authorization policy/audit done in 0108; live channel source artifacts done in 0110; web connector status/action surface delegates mutations to the Python tool/DB layer; web DSN/session-history audit tightened; chat orchestration now flows through `services/chat.py`; Next API routes share one Python-proxy helper instead of route-local fetch logic; connector cognition substrate landed in 0124; next: shrink remaining transport-local behavior and broaden provider adapters) |
| Web-chat history becomes DB-owned (her memory of a conversation and the record of it = same substrate; no localStorage-only history) | Continuity | done (0104: `chat_sessions`/`chat_messages`, SQL hydrate/clear/record functions, API/CLI/TUI/channel active-context hydration; web chat hydrates visible transcript from `hydrate_chat_session()`, stops sending browser cached transcript once a DB session exists, and exposes explicit DB-backed new-session / clear-context controls) |
| One DSN resolution shared by all clients (UI reads the same instance registry as Python — no split-brain on instance switch) | Continuity | done (UI Prisma resolver reads `~/.hexis/instances.json` with `HEXIS_INSTANCE`/current-instance precedence, `hexis ui` passes the selected DSN/instance to the Next and API processes, and runtime compose sets the same explicit UI DB URL as the API/db stack) |
| Collapse chat orchestration onto `services/chat.py`; move the RLM gate where both web and channel paths pass through | Substrate | done (web/OpenAI-compatible chat now stream through `services.chat.stream_chat_events`; persistence, prompt addenda, gateway audit, RLM gate, and memory write event are canonicalized there) |
| Wire `/api/ingest/jobs/{id}` polling into the web ingest flow | Experience Bar | done (f77c494) |

## Batch 5 — Life-channel ingestion and authorized agency

Hexis has to live where the user lives and compound around a real person,
not wait for manually curated notes. Connectors are consented senses and
hands: they ingest the user's existing life channels, preserve raw
provenance, distill durable memories, and act only inside explicit authority
boundaries.

Reference bar from `.reference/hermes-agent` and `.reference/openclaw`:
setup is an onboarded, cross-channel product path. Hermes' Google Workspace
script is deliberately agent-driven so the same flow works from CLI,
Telegram, Discord, etc.; Hermes also shares slash commands across CLI/gateway
and gives Slack a native command manifest. OpenClaw treats onboarding,
daemon/gateway setup, channel pairing, status, and plugin setup descriptors as
first-class surfaces. Hexis currently has adapters, docs, and interactive CLI
setup, but not the shared setup broker Samantha can invoke while already
talking to the user.

| Goal | Test | Status |
|---|---|---|
| First-class personal-data connectors: Gmail, Slack, Telegram, Signal, and Twitter/X live as plugin-backed channels with scoped setup, clear capability manifests, account identity, revocation, and separate grants for read/search/send/delete/label/admin actions | Piper law 2 + Dignity + Experience Bar | in progress (Gmail connector manifest/status/revoke + OAuth credential store wired; DB derives Gmail capability aliases/scopes; Gmail read/search/ingest backfill worker wired; DB action grants/audit cover send/reply/label/spam/delete policy; Gmail send/reply/label/spam-triage effect tools wired through saved OAuth credentials; Slack/Telegram/Signal live-channel manifests + setup/verify tools wired; Slack historical backfill is available; Telegram/Signal local export imports are available for retro-history; Twitter/X local archive import is available with source preservation; Twitter/X OAuth setup, live timeline/mentions/search/DM read backfill, and post/reply/DM effect tools are wired through DB connector manifests and action policy; remaining: real-provider smoke validation, access-tier/rate/cost calibration, and any future streaming/quote/delete features / 0103+0105+0107+0108+0109+0111+0124+0126+0130+0138) |
| Conversation-native connection setup: if the user says "connect my Gmail/Slack/Telegram/Signal/Twitter" in CLI chat, web chat, or any existing integration, Samantha can start the relevant setup flow in that same conversation, explain scopes, request only the needed user action, hand off OAuth URLs / QR codes / app manifests / env-secret prompts in-channel, and verify the connection before returning to chat | Piper law 2 + Experience Bar + Dignity | in progress (Gmail `gmail-connector-setup` skill + `connect_gmail`/`complete_gmail_connection` tools support in-chat OAuth handoff; DB owns Gmail requested-scope derivation; `start_gmail_backfill`/`gmail_backfill_status`/`control_gmail_backfill` expose ingest job setup and control; generic `integration-connector-setup` skill starts/configures/verifies Slack/Telegram/Signal manual channel setup from chat; generic `start_connector_backfill`/`connector_backfill_status`/`control_connector_backfill` expose non-Gmail history import control; web `/connections` reads DB state and proxies setup/configure/verify/OAuth/revoke/backfill/archive-import actions through Python; Twitter/X archive import has a first-class local-export setup path; Twitter/X OAuth start/complete/revoke and capability-scoped setup are first-class; remaining: live-provider manual verification and broader provider sizing calibration / 0103+0105+0107+0111+0124+0130+0138) |
| Setup broker substrate: one DB-owned `integration_connectors`/`connection_attempts` layer records provider manifests, required scopes, setup state, account identity, current channel/session, redacted errors, revocation status, and restart/worker requirements so CLI, UI, and channel adapters share one source of truth | Substrate + Continuity | done (b33bfae / 0103+0105: `integration_connectors`, `integration_connections`, `connection_attempts`, status/start/complete/error/revoke functions, Gmail seed, manifest-driven capability/scope derivation) |
| Massive channel backfill: each connector supports incremental ingest with receipts, cursoring, retry/resume, dedupe, cost/progress visibility, and no silent ambient credential reuse | Piper law 3 + Experience Bar | in progress (0106 DB substrate: `connector_backfill_jobs`, `connector_sync_cursors`, pause/resume/cancel/retry/progress/status; 0107 Gmail provider-scoped worker uses saved Hexis OAuth credentials, token refresh, chunked Gmail pages, and DB source-item receipts; 0124 Slack `conversations.history` worker stores raw source artifacts and ingestion receipts; 0126 provider-specific estimates/cost classes are recorded with jobs and shown in UI; Telegram/Signal JSON/CSV local export imports preserve historical records when bot APIs cannot expose retro-history; 0130 Twitter/X archive importer preserves tweets and direct messages from exported archive files; 0138 Twitter/X live worker reads timeline, mentions, recent search, and DM events through saved OAuth credentials; remaining: provider-tier calibration and real-account smoke testing) |
| Raw message/source preservation: ingested emails, chats, threads, attachments, and posts are stored as exact source artifacts with content hashes, channel/account/thread/message IDs, participants, timestamps, labels, sensitivity, and redaction status before any distillation | Continuity + Substrate + Dignity | in progress (0106 DB substrate: `connector_source_items` -> `source_documents` with provider IDs/thread/participants/attachments/labels/sensitivity + ingestion job link; 0107 Gmail adapter preserves headers, labels, participants, attachment metadata, snippet, and full extracted body before ingestion; 0110 live channel messages preserve exact source documents + inbound ingestion jobs from a DB trigger; 0124 Slack history messages preserve channel/timestamp/thread/subtype/file metadata before ingestion; 0126 Telegram/Signal export imports preserve local historical source artifacts with deterministic source IDs; 0130 Twitter/X archive import preserves tweet/dm source artifacts with deterministic source IDs; 0138 Twitter/X live tweets/mentions/search results/DM events preserve raw source artifacts before ingestion; remaining: attachment/media expansion and provider smoke tests) |
| User-model synthesis: heartbeat/consolidation turns channel history into evidence-backed beliefs about preferences, likes, dislikes, relationships, routines, commitments, and judgment patterns; claims point back to openable source artifacts instead of becoming untraceable prompt lore | Person + Piper law 3 + Continuity | in progress (0124 DB substrate: `user_model_source_progress`, `user_model_claims`, `upsert_user_model_claim`, `record_user_model_synthesis`; 0126 hybrid LLM-backed synthesis adds preference/relationship/commitment/routine/judgment-pattern/identity extraction, contradiction + supersession metadata, semantic-memory evidence links, pending-review state, `list_user_model_claims()` / `review_user_model_claim()`, and a web User Model review surface; 0131 approved active claims now render into prompt context and focused tests cover false-positive test-filler rejection plus prompt inclusion; trajectory evals now cover evidence accumulation, review gating, quoted/ephemeral false positives, LLM-stub supersession, and prompt-context freshness; remaining: broader corpus tuning and live-model eval calibration) |
| Notification/action layer: important-item detection, spam triage, summaries, reminders, replies, texts, and cross-channel interventions run through explicit per-action consent or preauthorized policy, with audit logs and reversible/pauseable controls | Piper law 1 + Piper law 4 + Dignity | in progress (0108 DB substrate: connector action tool map, scoped policies, constraints, autonomous/context gates, revoke/list functions, policy evaluation in `evaluate_tool_call`, and connector action audit via `record_tool_execution`; 0109 Gmail send/reply/label/spam-triage provider tools consume the policy substrate; 0124 connector importance detector records scores/reasons/actions and queues web-inbox notifications over threshold; Slack/Telegram sends now resolve DB channel config and pass through connector action policy; 0125 Signal send tool/action map consumes the same policy substrate; 0126 richer deterministic + LLM-assisted importance scoring emits reasons and suggested action routes into DB/UI; 0138 Twitter/X post/reply/DM send effect tools consume the same policy substrate; 0185+0186+0187 ambient responsibilities add DB-owned reminders, recurring check-ins, source monitors, generic connector source watches, threshold evaluators, run/observation/check-in audit, immediate force-checks, connector blocker refresh/unblock, worker delivery to web inbox/RabbitMQ, chat/CLI tool access, and a web Responsibilities control surface; remaining: real-provider smoke validation, richer detector eval, and broader health/wearable provider adapters) |
| Connector setup UX: CLI and web flows are peers of the conversational broker, not separate instructions; all surfaces show scopes, accounts, backfill size, expected cost/time, job progress, pause/resume/revoke controls, worker/restart status, and exact next steps when a provider blocks access | Experience Bar + Dignity | in progress (Gmail conversation tools expose OAuth setup, queued backfill, status, pause/resume/cancel, and revoke; generic integration setup tools expose status/start/configure/verify for Slack/Telegram/Signal with non-secret env-var config discipline; channel worker/manager writes DB runtime status for configured/running/error/missing-dependency visibility; `connector-action-authorization` exposes grant/status/revoke for action policies; web `/connections` status surface now shows manifests, accounts, attempts, runtime, backfill jobs, preserved-source counts, plus start/configure/verify/revoke/Gmail OAuth/import/backfill pause-resume-cancel controls; 0126 adds provider estimates plus Slack channel / Telegram export / Signal export history import controls; 0130 adds provider-specific Twitter/X archive dry-run sizing and archive import controls; 0138 adds Twitter/X OAuth, live import, revoke, capability selection, and action wiring; 0187 suppresses already-connected setup modals and adds `/responsibilities` for ongoing monitors/check-ins/reminders; remaining: broader sizing calibration and live-provider verification) |

## Batch 6 — The reward loop, proven by emergence

| Goal | Test | Status |
|---|---|---|
| Dopamine wiring: fire spikes from `satisfy_drive`, goal completion, `decide_resource_request` grants, `record_backup_completed` | Person | done (0126/0128: DB reward substrate + hooks from `satisfy_drive()`, `apply_goal_changes()` completion/abandonment, `decide_resource_request()` grants/modifications, and `record_backup_completed()`; tested) |
| **Social reward**: positive-valence appraisal directed at her → RPE spike (the strongest human reward) | Person | done (0126/0128: `record_social_reward()` plus DB-owned `apply_appraisal_reward_effects()` called from inline appraisal for high-confidence positive social emotions; tested) |
| Emergence eval suite: seeded scenarios asserting signatures appear — TOT events occur, mood measurably colors recall, open goals boost related retrieval (Zeigarnik), spaced reinforcement beats massed | Person (the standing test) | done (0132: seeded evals cover TOT partial activation, mood-colored recall, open-goal/Zeigarnik boost, and spaced reinforcement; reinforcement events are persisted in Postgres instead of inferred from counters alone) |

## Batch 7 — Graph as subconscious substrate

| Goal | Test | Status |
|---|---|---|
| `reconcile_graph()`: diff memory_edges vs AGE, repair, journal drift count (#93) — dual-store becomes checked invariant | Substrate | done (0126: `graph_reconciliation_runs` + `reconcile_graph()` record dangling/repair/backfill counts; tested) |
| Causal-ancestor chains + contradiction *paths* rendered into context (directional; mind the `find_causal_chain` direction wrinkle) | Person | done (0126/0127: `memory_graph_paths()` / `memory_context_paths()` plus `HydratedContext.context_paths` and DB renderer section `## Causal/Contradiction Paths`; tested) |
| Graph-adjacency signal joins the fused ranker's association tier | Person | done (0126: `memory.recall_graph_adjacency_weight` and typed `memory_edges` adjacency candidate signal in `recmem_recall_context()`; tested) |

## Batch 8 — Conduct norms (prompt modules)

| Goal | Test | Status |
|---|---|---|
| Execute-verify-report: "I'll do that" is not doing it — do it, then report | Piper law 1 | done (446e4f1 / 0095) |
| Steering-reduction as extraction criterion: prioritize memories that prevent future corrections and reminders | Piper law 3 | done (446e4f1 / 0095) |
| Silence discipline: proactive messages clear an interruption bar; similar messages dedupe; choosing silence is a recorded, valid act | Piper law 4 | done (446e4f1 / 0095) |
| Human-scale conversational inference: local cues calibrate register without becoming durable relationship/user-preference memories; test scaffolding fades unless explicitly made durable; overloaded `partner` init memory is purpose-qualified as co-development | Person + Continuity | done (446e4f1 / 0092–0094) |
| Cross-session relationship affect: recent prior turns and unresolved relationship injuries carry across new chat sessions; direct degradation/threats create evidence-linked unresolved grievance memories that shape trust/tone until sincere repair | Person + Continuity + Dignity | done (0142–0144) |
| Web-session recent recall: web-created chat sessions participate in recent carryover, and both streaming/non-streaming chat feed carryover into appraisal so a new UI conversation does not make her deny recently available context | Continuity + Experience Bar | done (0145) |

## Batch 9 — Small mechanics

| Goal | Test | Status |
|---|---|---|
| Config-defaults registry: each default lives in exactly one row; `get_config_*` falls back to it (ends the 5-copies-of-`heartbeat.max_energy` drift risk) | Substrate | done (446e4f1 / 0096–0101: heartbeat + maintenance defaults moved; init/status/channel paths read registry; broad feature seed sweep moved into `config_defaults`; existing active config rows preserved as overrides) |
| Baseline file renumbering (duplicate 28/32) in one mechanical commit | Substrate | done (0136 workspace: duplicate plain numeric baseline prefixes removed for 28/32, plus the older 74 collision, while preserving lexical execution order) |
| Graduated appraisal depth: appraisal intensity scales with stimulus salience (#67 budgets as the hook) — attention allocation as psychology and cost control | Person + Piper | done (0135: DB-owned `appraisal_depth_for_stimulus()` assigns skim/normal/deep salience budgets and inline appraisal consumes its max-token/context limits) |
| Chat energy, the human way: tools in chat cost energy; conversation interacts with drives — connection satisfied by good interaction; cost-vs-restore governed by a character-card temperament dial | Person + Piper | done (0134: `apply_chat_turn_energy_effects()` charges base/tool energy, restores on positive social interaction, satisfies connection, and reads temperament multipliers from profile/config) |
| Presence polish on channels: typing indicators, presence beacons | Piper law 5 | done (0136: `channel_presence_events`, DB summary function, and channel manager online/offline/typing beacons) |
| Async embedding lifecycle for `memories` (adopt the units pattern: nullable + `embedding_status`; no HTTP inside transactions) | Substrate | done (0129/0137: `create_memory()` defers embeddings with pending status; worker claims batches, writes vectors, records model/attempts/failures, and lexical recall keeps pending memories visible until vectors arrive) |
| Published embeddinggemma default: Hexis resolves the installed `embeddinggemma` binary, uses the upstream default port 42666, migrates only exact old default URLs, and no longer hard-codes a source-checkout binary path | Experience Bar + Substrate | done (0146) |

---

## Settled decisions (do not relitigate)

- **AGE stays.** memory_edges = write-path truth; AGE = the graph query
  engine, subconscious substrate. The fix direction is activation, not
  retirement.
- **No cypher/graph-query tools for the agent.** Conscious memory tools
  mirror human phenomenology only (remember, feeling-of-knowing, TOT,
  incubation, association, reminiscence).
- **`working_memory` table folds into units.** The mechanism (attention
  buffer) lives as hydrated context + recent-units tier; the RabbitMQ inbox
  reroutes through `recmem_ingest_turn` so incoming messages are
  *experienced*, not shelved.
- **Chat energy is not simply metered.** Temperament-valenced, per above.
- **Mechanisms, never quirks.** Implement the architecture the quirks point
  at; the quirks must emerge. `docs/_archive/reference-quirks.md` is the
  evidence ledger.
- **OpenAI-compat endpoints stay** (an integration surface, law 2), to be
  documented as supported.
- **Portable brain rule.** Durable cognitive behavior lives in Postgres
  wherever technically reasonable. Application code should be swappable
  adapter code: senses, hands, transports, renderers, and effect drivers.

## Completed

*(move rows here with date + commit/issue as they land)*

- 2026-07-18 — MISSION.md written (purpose, distillation method, emergence
  test, second north star, six tests) — this file's grounding.
- 2026-07-19 — DB-owned active chat session substrate (`0104_db_owned_chat_sessions`):
  services, API, CLI, TUI, and channel chat hydrate short-term context from
  Postgres; `/clear` hides active-context messages while preserving long-term
  memories.
- 2026-07-19 — Connector capability/scope derivation moved into Postgres
  (`0105_connector_capability_derivation`): Gmail aliases/defaults/planned
  rejection and least-scope OAuth requests derive from `integration_connectors`.
- 2026-07-19 — Connector backfill/source-item substrate moved into Postgres
  (`0106_connector_backfill_substrate`): provider cursors, backfill job
  lifecycle, raw connector source-item receipts, source-document preservation,
  and ingestion-job linkage are DB-owned.
- 2026-07-19 — Gmail provider backfill wired on top of the DB substrate
  (`0107_connector_backfill_provider_claim`): provider-scoped job claims,
  saved-credential token refresh, Gmail page fetch/body extraction, raw
  source-document preservation, ingestion-job enqueueing, and chat-facing
  queue/status/pause/resume/cancel tools.
- 2026-07-19 — Connector action authorization moved into Postgres
  (`0108_connector_action_authorization`): tool-to-action mapping, scoped
  policies, context/autonomous gates, target/recipient/daily-limit constraints,
  grant/list/revoke functions, `evaluate_tool_call` enforcement, tool-execution
  action audit, and chat-facing policy tools.
- 2026-07-19 — Gmail provider effect tools wired to the action policy substrate
  (`0109_gmail_action_tools`): `gmail_send`, `gmail_reply`, `gmail_label`, and
  `gmail_spam_triage` use saved Hexis OAuth credentials, scope checks, account
  mismatch refusal, Gmail REST side effects, DB policy enforcement, and action
  audit.
- 2026-07-20 — Live channel source artifacts moved into Postgres
  (`0110_channel_source_artifacts`): every `channel_messages` insert now creates
  a raw `source_documents` artifact, a `channel_source_items` receipt, sensitivity
  metadata from channel privacy/group flags, and a configurable inbound ingestion
  job link.
- 2026-07-20 — Channel connector manifests and setup broker widened
  (`0111_channel_connector_manifests`): Slack, Telegram, and Signal are
  first-class available connector manifests for live channel setup; Twitter/X was
  cataloged for later live work; generic chat tools can inspect, start, configure
  non-secret channel settings, and verify channel-worker configuration into
  `integration_connections`.
- 2026-07-20 — Channel adapter runtime status moved into Postgres
  (`0112_channel_adapter_runtime_status`): channel workers and managers record
  not-configured/configured/starting/running/stopped/error/missing-dependency
  state in `channel_adapter_runtime`; setup status now surfaces adapter runtime
  state from the same DB substrate.
- 2026-07-20 — Source-document desk intake (`b9a72fe` /
  `0123_ingest_auto_desk`): single
  user/agent source ingests now land on the RecMem desk immediately, while bulk
  corpus imports and connector backfills remain in the filing cabinet until
  pulled deliberately.
- 2026-07-20 — Web ingest exact job tracking (`f77c494`): file, text, and URL
  submissions now surface immediate job receipts and poll
  `/api/ingest/jobs/{id}` through the Next proxy, so accepted work stays
  visible even when the recent job list has not caught up yet.
- 2026-07-20 — Web connector setup surface: `/connections` and
  `/api/integrations/status` read the DB-owned integration setup substrate,
  channel runtime status, and connector backfill/source-item status into one
  first-class UI surface; Slack, Telegram, and Signal can start DB-owned setup
  attempts from the page.
- 2026-07-20 — Web connector action controls: `/connections` now invokes setup,
  channel config save, verify, Gmail OAuth start/complete, revoke, Gmail import,
  and Gmail backfill pause/resume/cancel through the Python action endpoint
  instead of duplicating connector mutation logic in Next.js.
- 2026-07-20 — Web instance/session audit: the Next Prisma client now resolves
  the active database from the same instance registry Python uses, `hexis ui`
  launches API and UI with the same selected instance/DSN, and web chat
  rehydrates its visible transcript from DB-owned `chat_sessions` instead of
  treating browser storage as conversation authority.
- 2026-07-20 — Web chat session controls: `/api/chat/session` creates a
  DB-owned web session on demand, `/api/chat/session/{id}` delegates context
  clearing to `clear_chat_session_context()`, and the chat header exposes
  explicit new-conversation / clear-context controls without deleting the
  preserved conversation record.
- 2026-07-20 — Chat/connector cognition consolidation (`0124_connector_cognition`):
  web and OpenAI-compatible chat stream through canonical `services/chat.py`
  orchestration; Next API routes share a thin Python-proxy helper; Slack
  historical backfill preserves raw source items and ingestion receipts; generic
  non-Gmail backfill tools expose queue/status/control; connector source items
  feed DB-owned user-model claims and importance records, with high-importance
  items queued to the web inbox.
- 2026-07-20 — Non-Gmail send adapters tightened (`0125_signal_send_action_map`):
  Slack and Telegram send tools now resolve DB-owned channel config and enforce
  DB allowlists before provider I/O; Signal has a first-class `signal_send`
  tool and connector action-policy map entry.
- 2026-07-20 — Portable-brain cognition v2 (`0126_portable_brain_cognition_v2`):
  retired stale ingest/legacy write paths, added hybrid LLM-backed user-model
  synthesis with review/supersession/contradiction handling, Telegram/Signal
  export imports, provider backfill estimates, richer connector importance
  scoring/action suggestions, graph reconciliation/path functions, graph-adjacent
  recall scoring, and reward/RPE/social-reward substrate.
- 2026-07-21 — Twitter/X live connector (`0138_twitter_x_live_connector`):
  OAuth 2.0 PKCE setup, capability-scoped manifests, saved-token refresh,
  timeline/mentions/recent-search/DM live backfill, raw source preservation,
  archive-import coexistence, post/reply/DM send effect tools, DB action-policy
  mapping, and `/connections` controls are wired. Remaining work is provider-tier
  cost/rate calibration, real-account smoke testing, and optional future X
  features such as streaming, quote, delete, and media expansion.
- 2026-07-20 — Graph path prompt surfacing (`0127_graph_path_prompt_context`):
  hydrated memory context now carries causal/contradiction/support paths from
  the DB and renders them in chat context beside the knowledge subgraph.
- 2026-07-20 — Reward hooks (`0128_reward_event_hooks`): drive satisfaction,
  goal completion/abandonment, resource-request grants, backup completion, and
  high-confidence positive social appraisals now write reward/RPE events through
  DB-owned functions without changing public return contracts.
