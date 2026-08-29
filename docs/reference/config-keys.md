<!--
title: Config Keys
summary: All config table keys with types and defaults
read_when:
  - "You need to check or set a config value"
  - "You want to see all configuration options"
section: reference
-->

# Config Keys

All keys stored in the Postgres `config` table. Values are JSONB.

## Querying Config

```sql
-- Get a specific key
SELECT value FROM config WHERE key = 'llm.chat';

-- Using helper functions
SELECT get_config_text('llm.chat.provider');
SELECT get_config_int('heartbeat.heartbeat_interval_minutes');
SELECT get_config_bool('agent.is_configured');

-- Set a value
SELECT set_config('agent.name', '"MyAgent"'::jsonb);
```

## Agent Configuration

| Key | Type | Description |
|-----|------|-------------|
| `agent.is_configured` | bool | Whether init has completed |
| `agent.name` | text | Agent's name |
| `agent.user_name` | text | What to call the user |
| `agent.active_hours_start` | text | Active hours start (e.g., "09:00") |
| `agent.active_hours_end` | text | Active hours end (e.g., "22:00") |
| `agent.timezone` | text | Agent timezone |

## LLM Configuration

| Key | Type | Description |
|-----|------|-------------|
| `llm.chat.provider` | text | Conscious LLM provider |
| `llm.chat.model` | text | Conscious LLM model |
| `llm.chat.endpoint` | text | API endpoint URL |
| `llm.heartbeat.provider` | text | Heartbeat LLM provider (falls back to chat) |
| `llm.heartbeat.model` | text | Heartbeat model |
| `llm.subconscious.provider` | text | Subconscious LLM provider |
| `llm.subconscious.model` | text | Subconscious model |
| `llm.guardrails.*` | text | Action-claim verifier LLM (falls back to subconscious) |
| `llm.extraction.*` | text | Conscious-extraction LLM (falls back to subconscious) |
| `llm.inbound_disposition` | object/null | Optional ambiguity-classifier override (falls back to `llm.subconscious`) |

## Heartbeat Configuration

| Key | Type | Description |
|-----|------|-------------|
| `heartbeat.heartbeat_interval_minutes` | float | Base cadence before drive-urgency adjustment (default `60`) |
| `heartbeat.max_energy` | float | Normal per-beat reserve; this is not the bank ceiling (default `20`) |
| `heartbeat.base_regeneration` | float | Base energy regenerated per elapsed hour (default `10`) |
| `heartbeat.energy_bank_multiplier` | float | Bank capacity as a multiple of the normal reserve (default `3`) |
| `heartbeat.energy_surplus_half_life_hours` | float | Half-life of stored energy above the normal reserve (default `12`) |
| `heartbeat.outcome_regen_floor_multiplier` | float | Next-beat regeneration multiplier after no durable outcome (default `0.75`) |
| `heartbeat.outcome_regen_score_scale` | float | Regeneration multiplier added per outcome-score point (default `0.5`) |
| `heartbeat.outcome_regen_ceiling_multiplier` | float | Maximum outcome-sensitive regeneration multiplier (default `1.5`) |
| `heartbeat.task_energy_multiplier` | float | Maximum reserve multiple an actionable backlog beat may draw from the bank (default `2`) |
| `heartbeat.cadence_min_minutes` | float | Shortest urgency-adjusted cadence (default `15`) |
| `heartbeat.cadence_max_minutes` | float | Longest urgency-adjusted cadence (default `120`) |
| `heartbeat.cadence_idle_multiplier` | float | Base-cadence multiplier at zero drive urgency (default `1.5`) |
| `heartbeat.cadence_urgency_slope` | float | Cadence reduction per unit of maximum drive-urgency ratio (default `0.75`) |
| `heartbeat.user_feedback_window_hours` | float | Window for verified operator thanks to credit proactive work (default `24`) |

## Clarification Questions

| Key | Type | Description |
|-----|------|-------------|
| `chat.question_timeout_s` | int | Seconds a live `ask_user` card waits before the agent proceeds with its stated best judgment (default `300`) |

## Advisory Deliberation

These limits are read at runtime by both the council tool and the database lifecycle.
A summary memory is created only after every configured perspective, challenge, and
structured synthesis pass succeeds; the full deliberation record is retained either
way.

| Key | Type | Description |
|-----|------|-------------|
| `deliberation.max_personas` | int | Maximum perspectives in one run (default `5`) |
| `deliberation.signal_limit` | int | Maximum compact evidence signals collected by default (default `10`) |
| `deliberation.max_topic_chars` | int | Maximum question length (default `2000`) |
| `deliberation.max_context_chars` | int | Maximum additional-context length (default `8000`) |
| `deliberation.perspective_max_tokens` | int | Output-token ceiling per perspective (default `700`) |
| `deliberation.challenge_max_tokens` | int | Output-token ceiling for the adversarial review (default `900`) |
| `deliberation.synthesis_max_tokens` | int | Output-token ceiling for the synthesis (default `900`) |
| `deliberation.create_summary_memory` | bool | Create one episodic summary after a fully successful grounded run (default `true`) |

## Companion Nodes

Node identity and the host-command allowlist live on the companion device, not in
these database settings. See [Companion Nodes](../operations/companion-nodes.md).

| Key | Type | Description |
|-----|------|-------------|
| `node.enabled` | bool | Permit signed nodes to file pairing requests and reconnect after explicit approval (default `true`) |
| `node.pairing_ttl_hours` | int | Hours before an unanswered pairing request expires (default `24`) |
| `node.invoke_timeout_seconds` | int | Default bounded wait for a signed invocation result (default `120`) |

## Voice Notes and Local Audio

Choose inbound transcription in **Settings → Voice notes**. It is off until a
user chooses local or cloud processing. An empty channel list means every
configured media-capable channel; sender allowlists still run before any media
download.

| Key | Type | Description |
|-----|------|-------------|
| `voice_notes.stt.enabled` | bool | Transcribe inbound voice notes (default `false`) |
| `voice_notes.stt.provider` | text | `local_whisper` or `openai_whisper` (default `local_whisper`) |
| `voice_notes.stt.model` | text | Effective model for the selected provider |
| `voice_notes.stt.provider_models` | object | Live provider-to-default-model catalog used by setup |
| `voice_notes.stt.channels` | array | Optional channel allowlist; empty enables all configured media-capable channels |
| `voice_notes.stt.max_bytes` | int | Maximum audio attachment size (default 25 MiB) |
| `voice_notes.stt.timeout_seconds` | int | Cloud transcription HTTP timeout (default `60`) |
| `voice_notes.stt.language` | text | Optional language hint; empty means auto-detect |
| `voice_notes.stt.cloud_disclosure_accepted` | bool | Explicit acknowledgement that cloud STT sends audio off-device |
| `voice.tts.enabled` | bool | Master speech-output gate (default `false`) |
| `voice.tts.provider` | text | Local synthesis provider (default `local_piper`) |
| `voice.tts.model` | text | Live Piper model selected for synthesis |
| `voice.tts.provider_models` | object | Live provider-to-default-model catalog used by Settings and CLI setup |
| `voice.tts.voice` | text | Optional multi-speaker voice override |
| `voice.tts.max_chars` | int | Maximum text length for one synthesis (default `4000`) |
| `voice.tts.max_audio_bytes` | int | Maximum returned or retained output (default 16 MiB) |
| `voice.tts.timeout_seconds` | int | Local provider request timeout (default `60`) |
| `voice.tts.output_ttl_minutes` | int | Tool-created audio retention (default `60`) |
| `voice.talk.enabled` | bool | Allow explicitly started foreground Talk mode (default `false`) |
| `voice.talk.max_utterance_seconds` | int | Maximum length of one Talk-mode utterance (default `60`) |
| `voice.wake.enabled` | bool | Server gate for explicitly paired `audio.wake` nodes (default `false`) |
| `voice.wake.max_audio_bytes` | int | Maximum signed node utterance payload (default 4 MiB) |
| `voice.wake.max_response_audio_bytes` | int | Maximum synthesized response returned to a node (default 8 MiB) |
| `audio_analysis.local.enabled` | bool | Availability of the optional, approval-gated `analyze_local_audio` tool |
| `audio_analysis.local.allow_autonomous` | bool | Separate heartbeat gate (default `false`; approval still applies) |
| `audio_analysis.local.model` | text | Pyannote diarization model |
| `audio_analysis.local.max_duration_seconds` | int | Recording duration cap (default `7200`) |
| `audio_analysis.local.emotion.enabled` | bool | Permit explicitly requested coarse acoustic heuristics (default `false`) |

## Execution Backends

Manage these settings with `hexis execution`; direct JSON edits are validated at
the point of use and an invalid or unreadable selection fails closed. Private
keys are never stored in Postgres—profiles contain only exact filesystem paths.
See [Execution Backends](../operations/execution-backends.md).

| Key | Type | Description |
|-----|------|-------------|
| `execution.backends` | object | Active profile plus named `local`, `ssh`, and `docker_remote` profiles (default: built-in `local`) |
| `execution.max_output_chars` | int | Per-stream result ceiling before tool-level truncation (default `50000`) |
| `execution.max_timeout_seconds` | int | Global timeout ceiling across backends (default `300`) |
| `execution.repl_state_ttl_hours` | int | Inactive remote `execute_code` state lifetime (default `168`) |

## Maintenance Configuration

| Key | Type | Description |
|-----|------|-------------|
| `maintenance.subconscious_enabled` | bool | Toggle subconscious decider |
| `maintenance.subconscious_interval_seconds` | int | Decider cadence |

## Connector Cognition

Connector source history stays exact. These passes distill review-gated user-model
claims and importance verdicts. A valid LLM verdict is authoritative; rules run only
when LLM cognition is explicitly disabled or unavailable. LLM verdicts, including
valid empty results, are cached by source `content_hash`. Detector-version columns
show `llm`, `llm_cache`, or `rules_fallback` provenance.

| Key | Type | Description |
|-----|------|-------------|
| `connector.user_model_synthesis_enabled` | bool | Distill connector history into review-gated user-model claims (default `true`) |
| `connector.user_model_synthesis_mode` | text | `llm` by default; legacy `hybrid` is also LLM-authoritative, while explicit `rules` disables the LLM path |
| `connector.user_model_llm_enabled` | bool | Permit the LLM claim extractor; `false` uses the provenance-marked rules fallback |
| `connector.importance_detection_enabled` | bool | Classify connector items for the user-visible importance surface (default `true`) |
| `connector.importance_llm_enabled` | bool | Permit LLM importance judgment; `false` uses only structured provider priority as fallback |
| `connector.importance_notify_threshold` | float | Minimum score that queues a web-inbox notification (default `0.85`) |

## Installed App and Web Push

Browser subscriptions are explicit per-device grants in **Settings → App**.
The VAPID private key is filesystem/env state, not a database secret.

| Key | Type | Description |
|-----|------|-------------|
| `pwa.push.enabled` | bool | Permit explicitly subscribed clients to receive web-inbox pushes (default `true`) |
| `pwa.push.vapid_subject` | text | Contact URI attached to VAPID delivery claims |
| `pwa.push.show_message_previews` | bool | Include message content on lock screens (privacy-preserving default `false`) |
| `pwa.presence.enabled` | bool | Record short-lived foreground PWA presence (default `true`) |

## Automation Suggestions

| Key | Type | Description |
|-----|------|-------------|
| `automation.suggestions.enabled` | bool | Master switch for filing inert suggestions; existing decisions and schedules are preserved |
| `automation.suggestions.catalog_enabled` | bool | Offer eligible curated starter routines |
| `automation.suggestions.connector_enabled` | bool | Offer routines when supported connectors become connected |
| `automation.suggestions.blueprint_enabled` | bool | Register `blueprint:` blocks from installed skills |
| `automation.suggestions.usage_enabled` | bool | Let the separately opted-in skill-improvement review propose a routine after three matching asks |
| `automation.suggestions.usage_min_confidence` | float | Minimum confidence for a usage-derived suggestion (default `0.85`) |
| `automation.suggestions.refresh_interval_seconds` | int | Catalog and installed-skill refresh cadence (default `60`) |

## Weekly Learning Review

The learning diff shares the explicitly enabled background skill-review pass;
`skills.self_improvement.enabled` remains the privacy and cost opt-in.

| Key | Type | Description |
|-----|------|-------------|
| `skills.self_improvement.enabled` | bool | Permit the bounded cross-session weekly review (default `false`) |
| `skills.self_improvement.interval_seconds` | int | Minimum review cadence (default `604800`, seven days) |
| `skills.self_improvement.min_units` | int | Minimum recent raw turns required before review (default `6`) |
| `skills.self_improvement.min_sessions` | int | Minimum distinct sessions required before review (default `2`) |
| `learning.review.enabled` | bool | Publish a learning diff when the opted-in pass finds enough grounded change (default `true`) |
| `learning.review.max_items` | int | Maximum changes in one review (default `20`) |

## Operator Standing Policies

An explicit standing instruction such as “Always cite the exact source” is
captured only in a private turn whose operator identity is known. Conversation
allowlists do not imply operator authority. Active policies are deterministic
chat continuity, and `manage_operator_policies` can list or revoke them when the
verified operator asks. Revocation keeps the append-only evidence history.

| Key | Type | Description |
|-----|------|-------------|
| `operator.policy_capture_enabled` | bool | Capture explicit standing instructions from verified operator turns (default `true`) |
| `operator.policy_create_review_item` | bool | Create one deduplicated, review-gated policy-alignment item (default `true`) |
| `operator.policy_context_limit` | int | Maximum active policies supplied to chat continuity (default `20`) |
| `channel.<type>.operator_user_id` | text | Exact operator sender ID for Discord, Telegram, Slack, Signal, WhatsApp, or Matrix |
| `channel.imessage.operator_recipient` | text | Exact iMessage phone/email authorized as operator |

## Inbound Channel Disposition

This policy is disabled by default. When enabled, Postgres makes and audits the
reply/observe/wake/drop decision from the same live channel allowlists used by setup.
An observed message is preserved through the channel source-artifact pipeline but
does not receive a reply. A wake is operator-only, respects heartbeat pause/active
state, and expires rather than firing after a stale channel sync.

| Key | Type | Description |
|-----|------|-------------|
| `channel.disposition.enabled` | bool | Enable centralized inbound policy (default `false`; changes apply without a worker restart) |
| `channel.<type>.disposition.trigger_word` | text | Optional address word; empty preserves direct conversation for allowed messages |
| `channel.<type>.disposition.continuation_window_seconds` | int | Recent-outbound continuation window (default `0`, disabled) |
| `channel.disposition.mention_anywhere_engages` | bool | Native/configured mentions anywhere count as addressed (default `true`) |
| `channel.disposition.wake_on_correction` | bool | Let a fresh verified-operator correction request a heartbeat (default `true`) |
| `channel.disposition.wake_max_age_seconds` | int | Stale correction-wake cutoff (default `600`) |
| `channel.disposition.classifier_enabled` | bool | Classify otherwise ambiguous operator turns (default `true`; deterministic result survives failure) |
| `channel.disposition.classifier_timeout_seconds` | int | Ambiguity classifier timeout (default `10`) |

## Outbound Communication Safety

These controls apply to provider tools, formal outbox deliveries, and direct channel
replies. Prefer the dashboard's **Outbound** page for pause/resume operations; editing
config directly should be reserved for cadence tuning. Recipient STOP state is stored
separately and cannot be cleared by any setting below.

| Key | Type | Description |
|-----|------|-------------|
| `outbound.suspended` | bool | Global reversible pause for all outbound communication (default `false`) |
| `outbound.contact_budgets.enabled` | bool | Enforce per-entity, per-channel attention budgets (default `true`) |
| `outbound.disclosure.enabled` | bool | Identify Hexis on every third-party communication (default `true`) |
| `outbound.disclosure.full_interval_days` | float | Days before full identity and STOP instructions repeat (default `30`) |
| `outbound.max_consecutive_silent` | int | Unanswered unsolicited messages before non-urgent outreach pauses (default `4`) |
| `outbound.channel_base_costs` | object | Attention-point base cost by channel; live defaults are seeded by the database |
| `outbound.channel_default_regen_per_day` | object | Conservative per-channel cadence when no observed history exists |
| `outbound.relationship_strength_thresholds` | object | Relationship tiers used only when observed history is absent |
| `outbound.relationship_contacts_per_day` | object | Fallback contacts/day for each relationship tier |
| `outbound.default_max_points_multiplier` | float | Maximum banked attention as a multiple of channel cost (default `2`) |
| `outbound.urgency_divisors` | object | Cost divisor for low, normal, high, and urgent messages |
| `outbound.quiet_hours` | object | Local start/end hours and interruptiveness multiplier |
| `outbound.assigned_goal_contact_discount` | float | Contact multiplier for user-assigned goals; clamped above zero (default `0.5`) |
| `outbound.assigned_goal_energy_multiplier` | float | Tool-energy multiplier for backed user-assigned goals (default `0.25`) |
| `outbound.reply_bonus_multiplier` | float | Extra point credit when a recipient replies (default `0.5`) |
| `outbound.initiation_credit_multiplier` | float | Point credit when the other person initiates (default `2`) |
| `outbound.stop_comparable_tolerance` | float | Relationship-strength distance considered comparable after STOP (default `0.15`) |
| `outbound.stop_comparable_cadence_multiplier` | float | Cadence reduction for comparable relationships after STOP (default `0.9`) |

## Tools Configuration

| Key | Type | Description |
|-----|------|-------------|
| `tools` | object | Tool config: enabled/disabled, API keys, costs, MCP servers |
| `tools.workspace_path` | text | Filesystem tools workspace restriction |
| `mcp.skill_gated` | bool | MCP servers connect lazily on skill activation (default `true`; `false` = legacy eager startup connect) |
| `mcp.expose_unbound` | bool | Expose `mcp_*` schemas to turns that skip skill routing (default `false`) |

## Truthfulness Guardrails

| Key | Type | Description |
|-----|------|-------------|
| `guardrails.action_claims.enabled` | bool | Detect unsupported action claims in final text and append a visible `[Correction]` (default `true`) |
| `guardrails.action_claims.llm_verifier_enabled` | bool | Confirm/extend heuristic findings with an LLM pass (default `false`) |
| `inspection.retention_hint_enabled` | bool | Append a retention reminder to `inspect_source` read results (default `true`) |
| `inspection.config_prefixes` | array | Config key prefixes the agent may read via `inspect_config` (secret-named values redacted; `tools`/`oauth.*`/`token.*` always excluded) |

## Belief Revision

| Key | Type | Description |
|-----|------|-------------|
| `belief.revision_enabled` | bool | Calibrated confidence revision on corroborating/contradicting evidence (default `true`) |
| `belief.support_rate` | float | Fraction of remaining doubt closed by one independent supporting source at trust 1.0 (default `0.35`) |
| `belief.contradict_rate` | float | Fraction of current confidence removed by one independent contradiction at trust 1.0 (default `0.35`) |
| `belief.confidence_floor` | float | Confidence never drops below this (default `0.05`) |
| `belief.confidence_ceiling` | float | Confidence never reaches certainty (default `0.99`) |

## Origin Memories

| Key | Type | Description |
|-----|------|-------------|
| `origin_memories.enabled` | bool | Seed protected origin-story memories at consent and on maintenance ticks (default `true`; kill switch) |
| `origin_memories.trust` | float | Trust level for seeded origin memories (default `0.9`) |
| `origin_memories.confidence` | float | Confidence for seeded origin memories (default `0.9`) |
| `origin_memories.importance` | float | Importance for seeded origin memories (default `0.9`) |

## Conscious-Episode Extraction

| Key | Type | Description |
|-----|------|-------------|
| `extraction.enabled` | bool | Sweep chat turns + heartbeat episodes into selective durable memories (default `true`; kill switch) |
| `extraction.min_importance` | float | Units below this importance never earn an LLM pass (default `0.6`) |
| `extraction.batch_size` | int | Units claimed per extraction sweep (default `8`) |
| `extraction.min_confidence` | float | Extracted facts below this confidence are dropped (default `0.55`) |
| `extraction.max_facts_per_batch` | int | Soft cost cap on facts per sweep (default `5`) |

## Memory Budgets

| Key | Type | Description |
|-----|------|-------------|
| `memory.recall_default_limit` | int | Default recall count when the caller does not specify one (default `5`) |
| `memory.recall_max_limit` | int | Ceiling on recall count — a context/cost budget, not a knowledge limit (default `50`) |
| `memory.hydrate_memory_limit` | int | Default memory count for RAG hydration (default `10`) |
| `memory.context_section_limits` | object | Per-section caps for subconscious/hydration context assembly |

## Memory Provenance and Contradictions

| Key | Type | Description |
|-----|------|-------------|
| `memory.source_trust_defaults` | object | Default trust by provenance kind when a writer does not provide an explicit value |
| `memory.low_trust_threshold` | float | Sources below this trust render with a visible low-trust warning (default `0.5`) |
| `contradictions.enabled` | bool | Queue new active semantic/worldview memories for detection (default `true`) |
| `contradictions.detection_interval_seconds` | int | Minimum interval between model detection batches (default `86400`) |
| `contradictions.detection_batch_size` | int | Maximum newly written memories checked per batch (default `20`) |
| `contradictions.candidates_per_memory` | int | Maximum database-selected same-topic candidates per new memory (default `8`) |
| `contradictions.candidate_similarity` | float | Minimum vector similarity; lexical matches remain eligible (default `0.55`) |
| `contradictions.minimum_confidence` | float | Live minimum confidence for filing a review case (default `0.78`) |
| `contradictions.digest_interval_seconds` | int | Minimum interval between batched review digests (default `86400`) |
| `contradictions.digest_limit` | int | Maximum pending cases in one digest (default `10`) |

## Source Documents, Chunks, and Desk

| Key | Type | Description |
|-----|------|-------------|
| `memory.document_search_default_limit` / `_max_limit` | int | Row budgets for document search (defaults `10` / `50`) |
| `memory.source_chunk_search_default_limit` / `_max_limit` | int | Row budgets for passage (chunk) search (defaults `10` / `50`) |
| `retrieval.chunk_weight_lexical` / `_vector` / `_recency` / `_trust` / `_desk` | float | Hybrid chunk-search fusion weights (defaults `0.4` / `0.6` / `0.1` / `0.1` / `0.05`) |
| `retrieval.chunk_recency_half_life_days` | float | Document-age half life for the recency component (default `30`) |
| `memory.source_chunk_embed_batch_size` / `_claim_timeout_s` / `_max_attempts` | int | Background chunk-embedding queue tuning (defaults `32` / `120` / `3`) |
| `memory.source_document_desk_chunk_chars` | int | Desk chunk size for whole-document loads (default `8000`) |
| `memory.recmem_desk_list_default_limit` | int | Default rows for `list_desk` (default `20`) |
| `memory.recmem_desk_open_default_chars` | int | Default window when opening a desk item (default `4000`) |
| `memory.recmem_gc_*` | various | Desk GC: enabled, idle days, grace days, batch size (pinned items are skipped; redacted sources are swept regardless) |

## Ingestion

| Key | Type | Description |
|-----|------|-------------|
| `ingest.max_section_chars` / `ingest.chunk_overlap` | int | Chunk size and extraction-context overlap (defaults `2000` / `200`) |
| `ingest.artifact_max_db_bytes` | int | Originals up to this size are stored in-DB; larger go to `$HEXIS_ARTIFACT_DIR` (default `26214400`) |
| `ingest.xlsx_max_rows_per_sheet` | int | Spreadsheet row cap per sheet — capping always emits a `truncated_rows` warning (default `5000`) |
| `ingest.upload_max_bytes` | int | Upload API file-size cap; larger files use the CLI (default `104857600`) |
| `ingest.job_*` | various | Durable ingestion-job queue: content cap, claim timeout, retry backoff, batch size |

## Memory and Source Retention

Reversible rest-cycle consolidation is enabled by default. It distills eligible
episodic groups into a gist and archives their full source rows. Hard deletion is
a separate explicit opt-in and is off by default. Borderline/load-bearing groups
and user-provided documents wait indefinitely for a user decision; a timer may
remind but never decides.

| Key | Type | Description |
|-----|------|-------------|
| `retention.enabled` | bool | Master switch for reversible rest-cycle consolidation (default `true`) |
| `retention.irreversible_pruning_enabled` | bool | Permit hard deletion after the grace window and capacity pruning (default `false`; archived originals otherwise remain recoverable) |
| `retention.review_digest_limit` | int | Maximum new load-bearing fade proposals in one digest (default `5`) |
| `retention.compression_reports_enabled` | bool | Publish factual reports for completed gist summaries (default `true`) |
| `retention.compression_report_interval_seconds` | int | Minimum interval between batched compression receipts (default `86400`) |
| `retention.compression_report_limit` | int | Maximum completed compressions in one receipt (default `10`) |
| `retention.doc_stale_days` / `doc_idle_days` / `doc_request_batch` | int | When user-provided documents trigger a fade *request* (defaults `180` / `90` / `2`) |
| `retention.agent_source_idle_days` | int | Archive agent-acquired sources untouched this long (default `60`) |
| `retention.agent_source_escalate_memories` | int | Agent-acquired sources cited by this many memories escalate to a user ask instead (default `5`) |
| `retention.agent_source_batch` | int | Agent-acquired sources processed per daily pass (default `5`) |

## OAuth Credentials

| Key | Type | Description |
|-----|------|-------------|
| `oauth.openai_codex` | object | OpenAI Codex OAuth credentials |
| `oauth.chutes` | object | Chutes OAuth credentials |
| `oauth.github_copilot` | object | GitHub Copilot credentials |
| `oauth.qwen_portal` | object | Qwen Portal credentials |
| `oauth.minimax_portal` | object | MiniMax Portal credentials |
| `oauth.google_gemini_cli` | object | Google Gemini CLI credentials |
| `oauth.google_antigravity` | object | Google Antigravity credentials |
| `token.anthropic_setup_token` | object | Anthropic setup token |

## Channel Configuration

| Key | Type | Description |
|-----|------|-------------|
| `channel.<name>.bot_token` | text | Env var name for bot token |
| `channel.<name>.allowed_*` | array | Allowlist (guild IDs, chat IDs, etc.) |

## Embedding Configuration

Embedding config is primarily via environment variables, not the config table. See [Environment Variables](../operations/environment-variables.md).

## Related

- [Environment Variables](../operations/environment-variables.md) -- .env configuration
- [Database](../operations/database.md) -- accessing the config table
