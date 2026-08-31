<!--
title: Tools Reference
summary: Complete tool catalog with categories, costs, and parameters
read_when:
  - "You want to see all available tools"
  - "You want to find a tool's energy cost"
section: reference
-->

# Tools Reference

Complete catalog of Hexis tools organized by category.

## Tool Categories

| Category | Factory Function | Tools |
|----------|-----------------|-------|
| Memory | `create_memory_tools()` | recall, recall_at_time, diff_memory_history, search_history, remember, add_evidence, belief_history, open_memory, search_documents, open_document, open_documents, load_documents, search_document_chunks, open_document_chunk, load_document_chunks, sense_memory_availability, explore_concept, get_procedures, get_strategies, queue_user_message, + type-specific creators |
| Desk | `create_desk_tools()` | list_desk, open_desk_item, pin_desk_item, unpin_desk_item, clear_desk |
| Self-Inspection | `create_self_inspection_tools()` | inspect_source, inspect_database_schema, inspect_config, review_recent_actions |
| Web | `create_web_tools()` | web_search, web_fetch, web_summarize |
| Filesystem | `create_filesystem_tools()` | read_file, write_file, edit_file, glob, grep, list_directory |
| Shell | `create_shell_tools()` | shell, safe_shell, run_script |
| Code | `create_code_execution_tools()` | code_execution |
| Browser | `create_browser_tools()` | browser |
| Calendar | `create_calendar_tools()` | calendar_events, calendar_create, calendar_update, calendar_delete, meeting_prep |
| Email | `create_email_tools()` | email_send, email_send_sendgrid, email_list, email_read, email_search, email_forward |
| Messaging | `create_messaging_tools()` | discord_send, slack_send, telegram_send |
| Contacts | `create_contact_tools()` | search_contacts, get_contact, create_contact, update_contact, merge_contacts, ingest_contacts_from_email, ingest_contacts_from_calendar |
| Ingest | `create_ingest_tools()` | fast_ingest, slow_ingest, hybrid_ingest, git_ingest, url_ingest |
| Goals | `create_goal_tools()` | manage_goals |
| Backlog | `create_backlog_tools()` | manage_backlog |
| Cron | `create_cron_tools()` | manage_schedule |
| Sessions | `create_session_tools()` | manage_sessions |
| Image/Video | `create_image_gen_tools()`, `create_video_gen_tools()` | generate_image, generate_video |
| Council | `create_council_tools()` | list_council_personas, run_council, list_deliberations, inspect_deliberation, aggregate_signals |
| Usage | `create_usage_tools()` | query_usage |
| Backup | `create_backup_tools()` | database_backup, backup_retention, config_export, config_import |
| Humanizer | `create_humanizer_tools()` | humanize_text, post_process_output |
| External | Various | todoist_*, asana_*, hubspot_*, youtube_*, twitter_search, brave_search, firecrawl_scrape, fathom_* |
| Everyday Life | `core.tools.life_integrations` | notion_*, spotify_*, home_assistant_*, weather_forecast, trello_* |
| Companion Node | `create_node_tools()` | node_invoke, apple_reminders, apple_notes, apple_calendar, apple_shortcuts, onepassword_local |
| Speech | `create_speech_tools()` | speak |
| Workflow | `create_workflow_tools()` | workflow |
| Dynamic | `create_dynamic_tools()` | create_tool |

## Energy Cost Table

| Cost | Tools |
|------|-------|
| **0** | search_history, search_documents, search_document_chunks, list_desk, pin_desk_item, unpin_desk_item, clear_desk, sense_memory_availability, belief_history, inspect_config, review_recent_actions, queue_user_message, get_contact, list_council_personas, list_deliberations, inspect_deliberation, manage_sessions (list/get) |
| **1** | recall, recall_at_time, diff_memory_history, remember, add_evidence, open_memory, open_document, open_document_chunk, open_desk_item, explore_concept, get_procedures, get_strategies, read_file, glob, grep, list_directory, manage_goals, manage_backlog, manage_schedule, search_contacts, query_usage, hubspot_*, youtube_*, humanize_text, backup_retention, config_export, connector setup/revoke, spotify_playback_state, home_assistant_states, weather_forecast, trello_list_boards, trello_list_cards, speak |
| **2** | open_documents, load_documents, load_document_chunks, web_search, web_fetch, calendar_events, email_list, email_read, email_search, fast_ingest, write_file, edit_file, safe_shell, todoist_create, todoist_complete, asana_create, twitter_search, brave_search, fathom_transcripts, merge_contacts, workflow, post_process_output, config_import, notion_search, notion_get_page, notion_query_data_source, spotify_search |
| **3** | shell, run_script, code_execution, calendar_create, calendar_update, calendar_delete, hybrid_ingest, url_ingest, generate_image, firecrawl_scrape, ingest_contacts_*, database_backup, aggregate_signals, notion_create_page, spotify_control_playback, home_assistant_call_service, trello_create_card, trello_update_card |
| **4** | web_summarize, browser, email_send, email_send_sendgrid, git_ingest, meeting_prep, fathom_ingest, node_invoke, apple_reminders, apple_notes, apple_calendar, apple_shortcuts, onepassword_local |
| **5** | discord_send, slack_send, telegram_send, slow_ingest, email_forward, run_council, create_tool |
| **8** | generate_video |

## Memory Tool Notes

- `remember` accepts optional `confidence` (0–1, semantic memories) and
  `sources` (array of `{kind, ref, label, author, trust}`); semantic memories
  record every source and derive `trust_level` from them.
- `add_evidence` (`{memory_id, stance: supports|contradicts, source, note?}`)
  attaches evidence to an existing semantic memory and revises its confidence
  through the audited belief-revision policy; it returns prior and posterior
  confidence. Duplicate sources merge without moving confidence.
- `recall` accepts `min_score` (relevance floor) and returns `trust` and
  `confidence` per memory. Its `limit` default and ceiling are config-driven
  (`memory.recall_default_limit`, `memory.recall_max_limit`).
- `recall_at_time` searches only memories whose durable validity window covered
  the requested ISO-8601 instant and reconstructs confidence/trust from the
  append-only audit. `diff_memory_history` compares two such snapshots and
  returns additions, expirations, supersessions, belief revisions, and
  contradiction decisions with their recorded reasons. The conversation prompt
  selects them from cues such as “as of,” “back then,” and “has that changed”;
  users do not need to know the tool names.
- `search_documents` browses preserved raw source artifacts; `open_document`
  and `open_documents` read exact files without writing memory. `load_documents`
  deliberately places source chunks on the RecMem desk so later
  `search_history` calls can search them with `sources=["desk"]`.
- `search_document_chunks` is the passage-grain cabinet search: hybrid
  full-text + vector over durable chunks, returning citable locators (page,
  section, sheet row) and DB-generated `rank_components` (degrades to
  lexical-only when the embedding service is down). `open_document_chunk`
  opens exact passages with a ready-made `citation` string and
  prev/next handles; `load_document_chunks` puts selected passages on the
  desk (by ids, by document+query, or by page range; `pin` keeps them
  through cleanup).
- Desk tools: `list_desk` shows what is loaded (check before re-loading),
  `open_desk_item` scrolls an item window by window, `pin_desk_item` /
  `unpin_desk_item` control cleanup protection, and `clear_desk` archives
  items (never deletes; the sources remain in the filing cabinet).
- `list_skills` reports each skill's status (`usable` / `needs_setup` /
  `unavailable`) with the exact next step; `use_skill` activates a skill and,
  for MCP-bound skills, lazily connects the server and unlocks only
  manifest-bound tools.
- `inspect_source` read results include a retention reminder (inspection is
  in-context only; nothing is remembered without an explicit write).
- `belief_history` explains why a belief is held: current confidence/trust,
  truth profile, the audited revision history, evidence links, and
  contradicting sources — one call answers "what changed my mind?".
- `inspect_config` reads the agent's own settings (allowlisted prefixes via
  `inspection.config_prefixes`; `tools`/`oauth.*`/`token.*` always excluded;
  secret-named values redacted).
- `review_recent_actions` is the verbatim tool audit log (successes and
  failures with energy/timing; never the stored output blobs).
- `run_council` is a durable advisory write: it gathers bounded evidence, runs
  parallel perspectives followed by an adversarial review and structured synthesis,
  and preserves degraded or failed runs. `list_deliberations` and
  `inspect_deliberation` reopen that history for free. Council advice never grants
  approval or executes the proposed action.
- `speak` is an optional chat-only tool. It sends bounded text only to the
  configured local Piper-compatible endpoint and emits an expiring, uncached
  conversation audio player. Its audit excludes both input text and audio.
- Companion-node tools always require approval for the exact node, operation, and
  arguments. Apple actions use fixed source-controlled local automation;
  `onepassword_local` returns redacted metadata or a receipt for a node-local
  clipboard copy, never a secret. See [Companion Nodes](../operations/companion-nodes.md).
- `shell`, `safe_shell`, `run_script`, and `execute_code` retain their public
  schemas while the database-selected execution profile places work locally,
  on an exact SSH workspace, or in an ephemeral remote container. A configured
  remote failure is reported and never silently re-routed locally. See
  [Execution Backends](../operations/execution-backends.md).
- In energy-budgeted turns, every tool result ends with an
  `[energy: spent/budget spent]` footer, and the heartbeat system prompt
  carries a Tool Energy Costs table derived from the live ToolSpec costs.

## Context Permissions

| Context | Default |
|---------|---------|
| **Chat** | All tools enabled (user present) |
| **Heartbeat** | Restricted: shell, write_file disabled; max 5 energy per call |
| **MCP** | Memory tools only |

## Tool Handler Pattern

All tools implement the `ToolHandler` ABC:

```python
class ToolHandler(ABC):
    @property
    def spec(self) -> dict:
        """OpenAI function calling spec"""

    async def execute(self, arguments: dict, context: ToolContext) -> str:
        """Execute the tool and return result"""
```

Tools get the DB pool via `context.registry.pool` at execution time.

## Related

- [Tools Configuration](../guides/tools-configuration.md) -- enabling/disabling tools
- [Energy Model](energy-model.md) -- energy budget mechanics
- [Plugin System](plugin-system.md) -- adding custom tools via plugins
