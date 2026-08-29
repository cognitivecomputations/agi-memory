<!--
title: Database API
summary: Public SQL function contract for the Hexis cognitive architecture
read_when:
  - "You want to call database functions directly"
  - "You're building an integration against the DB"
section: reference
-->

# Database API

The application layer should treat these SQL functions as its public contract. Any language can implement an app layer by calling these functions.

## Memory Creation

| Function | Description |
|----------|-------------|
| `create_memory(type, content, importance, trust_level, metadata)` | Create any memory type |
| `create_episodic_memory(content, importance, trust_level, metadata)` | Create episodic memory |
| `create_semantic_memory(content, confidence, category[], related_concepts[], source_references, importance, source_attribution, trust_level)` | Create semantic memory; trust is computed from confidence + sources when not pinned |
| `create_procedural_memory(content, steps, prerequisites)` | Create procedural memory |
| `create_strategic_memory(content, pattern, evidence)` | Create strategic memory |
| `add_to_working_memory(content, context)` | Add to working memory buffer |

All creation functions generate embeddings via `get_embedding()` and create graph nodes.

## Belief Revision

| Function | Description |
|----------|-------------|
| `revise_memory_confidence(memory_id, evidence, stance, context)` | Calibrated confidence update (residual_v1 policy); independence-aware; every call writes a `belief_revision_audit` row |
| `add_memory_evidence(memory_id, stance, source, note, evidence_memory_id, context)` | Revision + source merge + SUPPORTS/CONTRADICTS edge from an evidence node; returns prior/posterior |
| `sync_memory_trust(memory_id)` | Recompute semantic trust from confidence + sources; early-returns for `metadata.protected` memories (pinned trust) |

## Origin Memories & Conscious Extraction

| Function | Description |
|----------|-------------|
| `origin_memory_claims()` | Curated origin-story claims (from the LetterFromClaude/philosophy prompt modules) |
| `seed_origin_memories()` | Idempotently seed the claims as protected semantic memories (config-gated) |
| `record_heartbeat_episode_unit(agent_turns)` | Mirror a finished heartbeat turn into `subconscious_units` |
| `claim_conscious_extraction_batch(limit)` | Claim pending conscious episodes above the importance floor |
| `apply_conscious_extraction(unit_ids, extractions)` | Persist extracted facts (route through dedup: duplicates corroborate) |
| `fail_conscious_extraction(unit_ids, error)` | Retry bookkeeping (3 attempts, then parked) |

## Truthfulness Guardrail

| Function | Description |
|----------|-------------|
| `detect_unsupported_action_claims(turn_id, text)` | Flag prose claims of actions with no matching successful tool call in the turn (patterns live in `action_claim_patterns`) |

## Self-State Mirrors

| Function | Description |
|----------|-------------|
| `get_belief_history(memory_id, limit)` | The full story of a belief: state, truth profile, audited revisions newest-first, evidence edges, contradicting sources |
| `inspect_agent_config(prefix)` | Allowlisted, redacted view of the agent's own config (`inspection.config_prefixes`; hard-excludes `tools`, `oauth.*`, `token.*`) |
| `get_recent_actions(hours, limit, context)` | Windowed verbatim action log from `tool_executions` (metadata only, failures included) |

## Memory Retrieval

| Function | Description |
|----------|-------------|
| `fast_recall(query_text, limit)` | Primary hot-path retrieval (vector + neighborhoods + temporal) |
| `search_cross_session_history(query, limit, sources, after, before, exclude_session)` | Free Postgres FTS across active raw turns and memories |
| `search_similar_memories(query, limit, types)` | Similarity search with type filter |
| `search_working_memory(query)` | Search working memory buffer |
| `memory_citation_envelope(memory_id)` | Stable memory citation with provenance, trust, locator, and local target |
| `source_citation_envelope(document_id, chunk_id)` | Stable document/chunk citation with exact locator |

## Contradiction Events and Temporal Revision

| Function | Description |
|----------|-------------|
| `claim_contradiction_detection_batch(limit, force)` | Atomically claim a rate-limited batch with database-selected candidate pairs and the live confidence threshold |
| `finish_contradiction_detection_batch(queue_ids, result, error)` | Complete or durably retry a claimed batch |
| `file_contradiction_case(memory_a, memory_b, new_memory_id, tension, confidence, detected_by, metadata)` | File one confidence-gated, inert review case |
| `list_contradiction_cases(status, limit)` | Return pending, resolved, tension, or complete ledger views |
| `decide_contradiction(case_id, outcome, note, channel, actor)` | Apply an explicit `new_right`, `old_right`, or `tension` decision |
| `record_supersession(old_id, replacement_id, reason, actor, ...)` | Preserve revision lineage and close the old memory's validity window without deleting it |
| `publish_contradiction_digest_if_due(force)` | Queue one bounded daily review message without deciding any case |
| `memory_was_valid_at(memory_id, as_of)` | Test the explicit validity window, including active and reverted supersession intervals |
| `memory_epistemic_state_as_of(memory_id, as_of)` | Reconstruct confidence and trust at an instant from the append-only revision audit |
| `temporal_memory_snapshot(query, as_of, limit, types, min_score, exclude_sensitive)` | Hybrid point-in-time recall; degrades loudly to lexical retrieval if embeddings are unavailable |
| `diff_memory_history(query, from_time, to_time, ...)` | Compare two snapshots and return additions, expirations, supersessions, belief revisions, contradiction decisions, and reasons |

## Weekly Learning Review

| Function | Description |
|----------|-------------|
| `create_learning_review(period_start, period_end, summary, memory_ids, skill_proposal_ids, metadata)` | Derive one durable review and outbox digest from database-owned memory and proposal records |
| `list_learning_reviews(status, limit)` | Return complete review cards with evidence and skill application state |
| `decide_learning_review_item(item_id, action, correction, channel, actor, confirm_load_bearing)` | Approve, correct, or forget one item; protected forgetting requires reconfirmation |
| `try_resolve_learning_review_from_inbound(channel, actor, text)` | Resolve an exact coded verified-operator reply |
| `claim_approved_learning_skill_application()` | Atomically claim one explicitly approved skill for ownership-checked application |
| `finish_learning_skill_application(item_id, status, error)` | Record application success or visible bounded retry state |

## Deliberate Forgetting

| Function | Description |
|----------|-------------|
| `retention_status()` | Inspect live episodic mass, capacity, archived originals, candidate groups, pending reviews, and irreversible-pruning posture |
| `retention_observe_packet(limit)` | Return pressure, low-fidelity reconstructions, and factual recent compression receipts for Observe/UI surfaces |
| `list_memory_fade_reviews(status, limit)` | Return pending or decided load-bearing reviews with exact source memories, strength, fidelity, and keep budget |
| `decide_memory_fade_review(review_id, decision, journal_content, channel, actor)` | Apply one explicit `keep`, `release`, or `journal` choice; release/journal archives recoverable sources and queues gist summarization |
| `publish_memory_fade_review_digest()` | Queue one bounded outbox ask for unpublished reviews without deciding them |
| `publish_retention_compression_report_if_due(force)` | Report exact completed source counts, stored fidelity, and gist previews |
| `try_resolve_memory_fade_review_from_inbound(channel, actor, text)` | Resolve an exact coded reply from a verified private operator |
| `run_retention_gc()` | Apply retention cleanup; pending reviews never expire into a decision and hard deletion requires explicit `retention.irreversible_pruning_enabled=true` |

## Heartbeat and Maintenance

| Function | Description |
|----------|-------------|
| `should_run_heartbeat()` | Check the database-selected adaptive due time |
| `should_run_maintenance()` | Check if maintenance is due |
| `run_heartbeat()` | Open heartbeat, gather context, return external call payloads |
| `execute_heartbeat_actions_batch(heartbeat_id, actions)` | Apply actions, return outbox payloads |
| `apply_heartbeat_decision(...)` | Apply a single heartbeat decision |
| `apply_external_call_result(call_payload, output)` | Feed LLM/embedding results back |
| `complete_heartbeat(...)` | Finalize legacy action state and log the heartbeat |
| `run_subconscious_maintenance()` | Run all maintenance tasks |
| `start_heartbeat()` | Regenerate/decay banked energy, open the outcome ledger, and initialize a beat |
| `finalize_heartbeat_economy(...)` | Deduct exact spend, derive receipt-backed outcomes, and schedule the next beat |
| `heartbeat_economy_status()` | Inspect reserve, bank capacity, regeneration multiplier, and latest outcome |

## State and Config

| Function | Description |
|----------|-------------|
| `get_state(key)` | Get runtime state value |
| `set_state(key, value)` | Set runtime state value |
| `get_config_text(key)` | Get config value as text |
| `get_config_int(key)` | Get config value as integer |
| `get_config_float(key)` | Get config value as float |
| `get_config_bool(key)` | Get config value as boolean |
| `set_config(key, value)` | Set config value |

## Advisory Deliberation

| Function | Description |
|----------|-------------|
| `get_deliberation_config()` | Return the live persona, evidence, context, token, and summary-memory limits |
| `begin_deliberation(...)` | Open a bounded durable council session |
| `record_deliberation_move(...)` | Idempotently record one perspective, challenge, or synthesis move |
| `complete_deliberation(...)` | Atomically store the advisory verdict and optional grounded episodic summary |
| `fail_deliberation(session_id, error)` | Preserve an unsuccessful run and its cause |
| `list_deliberations(limit, status)` | List recent lifecycle records with recommendation and degraded flag |
| `inspect_deliberation(session_id)` | Return the full session, ordered moves, and verdict |

The lifecycle is audit-only: no function in this subsystem grants permission,
blocks a tool, or performs the recommended action. Service-level partial failures
complete with `metadata.degraded=true`; unexpected orchestration failures use the
`failed` session status.

## Consent

| Function | Description |
|----------|-------------|
| `request_consent(...)` | Returns external call payload for consent request |
| `record_consent(...)` | Record consent decision |

Consent is permanent; refusal is handled by pause/termination, not revocation.

## Companion Nodes

| Function | Description |
|----------|-------------|
| `register_node_handshake(node_id, public_key, name, capabilities, metadata)` | File or resume exact-identity pairing; added capabilities require a new decision |
| `decide_node_pairing(request, decision, actor, note)` | Approve or deny a pending identity/capability set |
| `list_node_pairing_requests(status, limit)` | List pairing decisions for operator review |
| `list_hexis_nodes()` | Inspect paired nodes, advertised capabilities, and connection state |
| `mark_node_connection(node_id, public_key, connection_id, online, metadata)` | Acquire, heartbeat, or release signed single-session connection ownership |
| `revoke_hexis_node(node_id, actor, reason)` | Irreversibly revoke an identity and cancel its queued work |
| `create_node_invocation(node_id, action, arguments, requested_by, timeout_seconds, metadata)` | Queue one bounded, advertised action from the fixed capability vocabulary |
| `claim_node_invocation(node_id)` | Atomically claim the next invocation for a connected node |
| `complete_node_invocation(invocation_id, node_id, success, result, error, result_signature)` | Persist the signed terminal result or failure |
| `get_node_invocation(invocation_id)` | Inspect one queued or completed invocation |

Structured node actions are limited to Apple Reminders, Notes, Calendar,
Shortcuts, redacted/local-only 1Password operations, `system.run`, and
`screen.capture`. The database refuses actions the exact paired node did not
advertise.

## Embeddings

| Function | Description |
|----------|-------------|
| `get_embedding(text[])` | Generate embeddings via HTTP (cached in `embedding_cache`) |
| `embedding_dimension()` | Return configured embedding dimension |
| `check_embedding_service_health()` | Check if embedding service is reachable |

## Maintenance Functions

| Function | Description |
|----------|-------------|
| `cleanup_working_memory()` | Delete expired working memory items |
| `batch_recompute_neighborhoods()` | Refresh stale precomputed neighbors |
| `cleanup_embedding_cache()` | Prune old cached embeddings |

## Graph Operations

| Function | Description |
|----------|-------------|
| `link_memory_to_concept(memory_id, concept_name)` | Link memory to concept (creates if needed) |
| `ensure_current_life_chapter()` | Update narrative life chapter |

## Character and Identity

| Function | Description |
|----------|-------------|
| `init_from_character_card(card_json)` | Initialize identity from character card |

## Design Principles

1. **DB functions return JSON payloads** for external calls -- the app layer executes them
2. **External call results** are fed back via `apply_external_call_result()`
3. **Outbox payloads** are published by the app layer (e.g., via RabbitMQ)
4. **The DB does not store queues** -- transport logic stays outside
5. **Advisory locks** prevent double-execution of maintenance tasks

## Related

- [Database Schema](database-schema.md) -- table reference
- [Memory Types](memory-types.md) -- memory type details
- [Database Is the Brain](../concepts/database-is-the-brain.md) -- architectural philosophy
