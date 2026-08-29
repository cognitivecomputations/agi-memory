<!--
title: Skills
summary: Use built-in skills and create custom workflow packages
read_when:
  - "You want to use or create skills"
  - "You want to understand the skills system"
section: guides
-->

# Skills

Skills are declarative, composable workflows that bundle tool sequences, prompts, and configuration into reusable packages.

Skills are also the agent's **capability catalog**: the model sees skill
discovery tools plus the tools bound by active skills, and it answers "can I
do X?" from `list_skills`. Each catalog entry carries a tri-state status —
`usable`, `needs_setup` (with the exact next step, e.g. "Set
GITHUB_PERSONAL_ACCESS_TOKEN in the service environment"), or `unavailable` —
so skills with unmet requirements are listed with what's missing, never
silently hidden.

## Quick Start

```bash
hexis skills list                    # list installed skills
hexis skills info daily-briefing     # show skill details
```

## Included Skills

The runtime catalog is the source of truth; use `hexis skills list` to see every
bundled, user, plugin, and configured MCP skill available in this installation.
Hexis includes these workflows among others:

| Skill | Category | Description |
|-------|----------|-------------|
| `core-memory` | system | Core memory recall/remember workflow (active by default) |
| `daily-briefing` | productivity | Morning summary of calendar, email, goals, priorities |
| `meeting-prep` | productivity | Pre-meeting research, attendee context, agenda prep |
| `email-digest` | communication | Summarize and triage unread email |
| `research` | research | Multi-source research with memory integration |
| `twitter-research` | research | Twitter/X trend and account analysis |
| `youtube-analytics` | analytics | Channel and video performance analysis |
| `crm-lookup` | productivity | Contact and deal lookup across CRM sources |
| `cost-report` | system | Usage and cost analysis across LLM providers |
| `knowledge-ingest` | knowledge | Guided knowledge ingestion with mode selection |
| `self-reflection` | system | Guided self-reflection and worldview review |
| `self-inspection` | system | Browse own source tree and live schema |
| `skill-authoring` | system | Author and revise agent-owned skills |
| `memory-exchange` | system | Export/import memories between agents |
| `github-issues` | productivity | GitHub issues via a skill-bound MCP server |
| `image-gen` | creative | Image generation with prompt refinement |
| `humanizer` | communication | Detect and rewrite AI-patterned text |
| `travel-prep` | productivity | Build a sourced itinerary, checklist, and travel brief without transacting |
| `inbox-triage` | communication | Propose safe inbox actions and apply only the user's chosen batch |
| `meeting-follow-up` | productivity | Extract decisions and draft controlled post-meeting actions |
| `weekly-review` | productivity | Reconcile goals, tasks, commitments, and the week ahead |
| `expense-capture` | productivity | Save structured, source-linked local expense records |
| `personal-notes` | knowledge | Work with mounted Obsidian vaults and Bear Markdown exports |
| `council` | knowledge | Run durable multi-perspective deliberation for consequential choices, preserving challenges, dissent, and review conditions |
| `notion` | productivity | Search, read, query, and approval-gated page creation in a connected Notion workspace |
| `spotify` | creative | Search Spotify, inspect playback, and control it with explicit approval |
| `home-assistant` | productivity | Read entity states and call exact services with explicit approval |
| `weather` | research | Get current conditions and up to sixteen forecast days from Open-Meteo |
| `trello` | productivity | Read boards/cards and create or update cards with explicit approval |

The `council` skill is advisory. `run_council` records bounded perspective,
challenge, and synthesis passes in Postgres; `list_deliberations` finds prior
runs and `inspect_deliberation` reopens the evidence, dissent, missing evidence,
and conditions that should trigger review. A failed or degraded pass remains
inspectable and is labeled as such. It never authorizes, blocks, or executes an
action.

## Managing Skills

```bash
hexis skills list                    # list all skills
hexis skills info <name>             # show skill details
hexis skills install ./my-skill      # install custom skill
hexis skills uninstall <name>        # remove a skill
```

## Skill Format

Each skill is a directory with a `SKILL.md` file:

```yaml
---
name: my-skill
description: What this skill does
category: research          # research | productivity | communication | knowledge | analytics | creative | system | other
contexts: [chat, heartbeat] # where the skill can run
requires:
  tools: [web_search, recall]
  config: [tavily]          # provider names whose credentials must be configured
bound_tools: [web_search, recall, remember]
---

## Instructions

Markdown instructions the agent follows when executing this skill.
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique skill identifier |
| `description` | Yes | One-line description |
| `category` | Yes | Skill category |
| `contexts` | Yes | Where it can run: `chat`, `heartbeat`, or both |
| `requires.tools` | No | Native tools that must exist for the skill to load (`mcp_*` entries are ignored here — MCP tools exist only after activation) |
| `requires.config` | No | Provider config required (e.g. `tavily`) |
| `requires.env` / `requires.bins` | No | Environment variables / binaries the skill needs (drive the `needs_setup` status) |
| `bound_tools` | No | Tools this skill exposes to the model while active; supports globs like `mcp_github_*` |
| `mcp` | No | MCP server binding — see below |
| `blueprint` | No | One inert automation suggestion registered when the installed skill is discovered; it never schedules automatically |
| `provenance` | No | Ownership metadata reserved for managed agent-authored skills |

### Suggesting a routine from a skill

An installed skill may carry one consent-first automation blueprint. The flat
form works with Hexis's dependency-free frontmatter parser:

```yaml
blueprint:
  dedup_key: blueprint:my-skill-weekly-prompt
  title: Weekly project check
  rationale: A Friday prompt makes it easier to close the week's open loops.
  schedule: weekly:friday:16:00
  action_kind: queue_user_message
  message: Weekly project check — open Hexis and review the project's open loops.
  delivery_mode: outbox
```

`title`, `rationale`, and a valid schedule are required. The maintenance worker
registers the block as a pending suggestion. Installation never accepts it or
creates a scheduled task.

### Binding an MCP server

A skill can declare an MCP server as its transport. The server connects
lazily when the skill is activated, and only `bound_tools` become callable:

```yaml
mcp:
  server: github
  command: npx
  args: ["-y", "@modelcontextprotocol/server-github"]
  env_requires: [GITHUB_PERSONAL_ACCESS_TOKEN]   # env var NAMES only, never values
bound_tools: [mcp_github_create_issue, mcp_github_search_issues]
```

If `command` is omitted, the binding resolves against a server of that name in
the tools config (`hexis tools add-mcp`). See
[MCP Integration](mcp-integration.md) for the full model and
`skills/installed/github-issues/SKILL.md` for the reference implementation.

## Custom Skills

Custom skills go in `~/.hexis/skills/` or `skills/installed/` (for bundled skills).

User-authored skills under `~/.hexis/skills/` remain user-owned. The
`author_skill` agent tool writes only beneath
`~/.hexis/skills/agent-authored/` and adds structured `provenance` frontmatter.
An update is allowed only when the existing file proves it is managed by
`author_skill`; unmarked files and symlinked targets are refused without being
modified. Skills created by older Hexis versions with the exact legacy
provenance footer are upgraded to structured provenance on their next approved
update.

### Creating a Custom Skill

1. Create a directory: `~/.hexis/skills/my-skill/`
2. Add `SKILL.md` with frontmatter and instructions
3. Install: `hexis skills install ~/.hexis/skills/my-skill`

### Example

```markdown
---
name: weekly-review
description: Weekly review of goals, accomplishments, and plans
category: productivity
contexts: [chat, heartbeat]
requires:
  tools: [recall, manage_goals]
bound_tools: [recall, manage_goals, remember]
---

## Instructions

1. Recall all active goals and recent episodic memories from the past week.
2. Summarize accomplishments and progress toward each goal.
3. Identify blocked goals and suggest next steps.
4. Propose goals for the coming week based on patterns.
```

## Related

- [Tools Configuration](tools-configuration.md) -- ensuring required tools are enabled
- [Scheduling](scheduling.md) -- scheduling skill execution
