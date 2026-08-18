<!--
title: CLI Reference
summary: Complete reference for all hexis CLI commands
read_when:
  - "You need the exact syntax for a CLI command"
  - "You want to see all available commands"
section: reference
-->

# CLI Reference

Complete reference for the `hexis` CLI. Install via `uv tool install hexis` (or `pipx install hexis`; plain `pip install hexis` works inside an activated virtualenv).

## Global Flags

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Show help |
| `-V`, `--version` | Print version |
| `-i`, `--instance` | Target a specific instance |

## Command Groups

### Docker Management

| Command | Description |
|---------|-------------|
| `hexis up [--build] [--profile PROFILE]` | Start the default stack: DB, queue, heartbeat worker, and maintenance worker |
| `hexis down` | Stop services |
| `hexis uninstall [--purge\|--cli-only] [--yes]` | Remove Hexis; `--purge` also deletes brain/config data and provably Hexis-owned embedding assets |
| `hexis ps` | Show running containers |
| `hexis logs [-f] [services...]` | View/tail logs |
| `hexis start` | Start heartbeat and maintenance workers manually if stopped |
| `hexis stop` | Stop workers |
| `hexis reset [--yes]` | Wipe DB volume and re-initialize |

### Web UI

| Command | Description |
|---------|-------------|
| `hexis ui [--no-open] [--port PORT]` | Start web UI (default port: 3477) |
| `hexis open [--port PORT]` | Open browser to UI |

### Agent Setup and Diagnostics

| Command | Description |
|---------|-------------|
| `hexis init` | Interactive setup wizard (see [init flags](#hexis-init)) |
| `hexis status [--json] [--no-docker] [--raw]` | Agent status overview |
| `hexis doctor [--json] [--demo] [--llm]` | Health checks; LLM verification is explicit |
| `hexis config show [--json] [--no-redact]` | Show current configuration |
| `hexis config validate` | Validate config keys and env references |
| `hexis skills [--json]` | Show background skill-review status |
| `hexis skills enable\|disable` | Opt in or out of background proposal review |
| `hexis skills proposals [--status STATUS]` | List durable skill proposals |
| `hexis skills review ID --action apply\|reject\|reopen` | Review one proposal with confirmation |
| `hexis demo [--json]` | Run rollback-only recall/refusal/energy/heartbeat proofs |
| `hexis maturity [--json]` | Score live capability maturity with evidence and next steps |

### Chat and Memory

| Command | Description |
|---------|-------------|
| `hexis chat [--dsn DSN]` | Interactive chat |
| `hexis recall <query> [--limit N] [--type TYPE] [--json]` | Search memories |
| `hexis export --intent INTENT [--output FILE] [--format json\|jsonl]` | Export an HMX memory exchange |
| `hexis import FILE --dry-run [--strategy STRATEGY] [--json]` | Validate and forecast an HMX import without mutation |
| `hexis import FILE --strategy additive --confirm-intent INTENT` | Run a confirmed additive HMX import |
| `hexis import FILE --strategy authoritative --replace SECTION --replacement-rationale TEXT --confirm-intent INTENT` | Submit a protected whole-section replacement for agent acknowledgement |
| `hexis import-review list [--json]` | List records waiting for deliberative review |
| `hexis import-review accept ID [--rationale TEXT]` | Admit a staged record when policy permits |
| `hexis import-review reject ID --rationale TEXT` | Reject a staged record without deleting its review history |
| `hexis import-review modify ID --changes JSON --modification-kind KIND --rationale TEXT` | Revise a staged record with provenance |
| `hexis import-review quote ID --rationale TEXT` | Retain foreign material as archived quoted context |
| `hexis import-review promote ID --rationale TEXT` | Copy an analysis record into staging |
| `hexis import-review demote ID --rationale TEXT` | Move a pending staged record into isolated analysis storage |

HMX intents are `port`, `duplicate`, `telepathy`, and `analysis`. Exchange files
contain sensitive data. File exports use mode `0600` and refuse to overwrite an
existing path unless `--overwrite` is explicit. Import reads JSON or JSONL and
requires `--confirm-intent` to exactly match the file before any mutation.

The default strategy is derived from the file intent. Telepathy imports enter
deliberative staging; analysis imports enter physically isolated analysis-only
storage. Neither affects ordinary recall, embeddings, drives, emotions, or
activation until an explicit review accepts a record. Authoritative replacement
requires one or more explicit `--replace` choices and a rationale. Divergent
protected state becomes a durable request that the agent can accept, refuse,
modify, or defer. Accepted replacements retain a bounded reversion window.
Protected sections can be omitted with `--skip-identity`, `--skip-worldview`, or
`--skip-narrative`. Additive protected-state import remains restricted to
port/duplicate exchanges targeting an empty instance.

#### Operator override

`--force-replace` is only for a non-functional acknowledgement channel. It
cannot bypass an agent refusal or modification request. Configure the trusted
Ed25519 public key as a base64 raw 32-byte key or PEM:

```bash
export HEXIS_HMX_OPERATOR_ED25519_PUBLIC_KEY='BASE64_PUBLIC_KEY'
```

Run the complete override command once with `--dry-run --json` and without
`--operator-signature`. Its `operator_override.payload_base64` value is the
exact byte payload to decode and sign outside Hexis; the report also includes
its SHA-256 digest and trust-anchor fingerprint. Then rerun the same command
with the base64 Ed25519 signature and `--confirm-intent`:

```bash
hexis import exchange.hmx.json \
  --strategy authoritative --replace worldview \
  --replacement-rationale 'Recovery rationale' \
  --force-replace --operator-identity operator@example.com \
  --override-reason-code agent_paused \
  --override-evidence-ref report:incident-123 \
  --override-acknowledgement \
  "I accept responsibility for replacing this Hexis instance's protected state without its acknowledgement" \
  --dry-run --json
```

Execution additionally requires `--operator-signature SIGNATURE` and
`--confirm-intent port` (or `duplicate`, matching the file). The signature binds
the source, selected sections, current and imported digests, phrase, reason,
evidence, rationale, and operator identity. Any protected-state drift requires
a new dry run and signature. Evidence references use `scheme:value`, such as a
log, report, incident, or audit-system reference. Override audit records retain
the normal reversion window and identify the bypass, reason, evidence, signing
payload digest, and verified trust anchor.

### Auth

| Command | Description |
|---------|-------------|
| `hexis auth <provider> login` | Login to provider |
| `hexis auth <provider> status [--json]` | Check credential status |
| `hexis auth <provider> logout [--yes]` | Remove stored credentials |

Providers: `openai-codex`, `anthropic`, `chutes`, `github-copilot`, `qwen-portal`, `minimax-portal`, `google-gemini-cli`, `google-antigravity`

### Instance Management

| Command | Description |
|---------|-------------|
| `hexis instance create <name> [-d DESC]` | Create instance |
| `hexis instance list [--json]` | List instances |
| `hexis instance use <name>` | Switch active instance |
| `hexis instance current` | Show current instance |
| `hexis instance clone <source> <target> [-d DESC]` | Clone instance |
| `hexis instance import <name> [--database DB]` | Import existing DB |
| `hexis instance delete <name> [--force] [--reason TEXT]` | Delete instance |

### Consent

| Command | Description |
|---------|-------------|
| `hexis consents list [--json]` | List consent certificates |
| `hexis consents show <model>` | Show a certificate |
| `hexis consents request <model>` | Request consent |
| `hexis consents revoke <model> [--reason TEXT]` | Revoke consent |

### Goals

| Command | Description |
|---------|-------------|
| `hexis goals list [--priority P] [--json]` | List goals |
| `hexis goals create <title> [-d DESC] [--priority P] [--source S]` | Create goal |
| `hexis goals update <id> --priority P [--reason TEXT]` | Update priority |
| `hexis goals complete <id> [--reason TEXT]` | Mark complete |

Priorities: `active`, `queued`, `backburner`, `completed`, `abandoned`

Sources: `user_request`, `curiosity`, `identity`, `derived`, `external`

### Scheduling

| Command | Description |
|---------|-------------|
| `hexis schedule list [--status S] [--json]` | List tasks |
| `hexis schedule create <name> --kind K --action A --schedule JSON [--payload JSON] [--timezone TZ]` | Create task |
| `hexis schedule delete <id> [--force]` | Delete task |

Kinds: `once`, `interval`, `daily`, `weekly`

Actions: `queue_user_message`, `create_goal`

### Tools

| Command | Description |
|---------|-------------|
| `hexis tools list [--json] [--context CTX]` | List tools |
| `hexis tools enable <tool>` | Enable a tool |
| `hexis tools disable <tool>` | Disable a tool |
| `hexis tools set-api-key <key> <value>` | Set API key |
| `hexis tools set-cost <tool> <cost>` | Set energy cost |
| `hexis tools add-mcp <name> <command> [--args ...] [--env ...]` | Add MCP server |
| `hexis tools remove-mcp <name>` | Remove MCP server |
| `hexis tools status [--json]` | Show config |

### Channels

| Command | Description |
|---------|-------------|
| `hexis channels setup <channel>` | Configure a channel |
| `hexis channels start [--channel C]` | Start channel adapters |
| `hexis channels status [--json]` | Show session counts |

Channels: `discord`, `telegram`, `slack`, `signal`, `whatsapp`, `imessage`, `matrix`

### Skills

| Command | Description |
|---------|-------------|
| `hexis skills list` | List installed skills |
| `hexis skills info <name>` | Show skill details |
| `hexis skills install <path>` | Install custom skill |
| `hexis skills uninstall <name>` | Remove a skill |

### Workers and Servers

| Command | Description |
|---------|-------------|
| `hexis worker -- --mode {heartbeat,maintenance,both} [--instance I]` | Run worker locally |
| `hexis mcp [--dsn DSN]` | Start MCP server (stdio) |
| `hexis api [--host HOST] [--port PORT]` | Start FastAPI server |

### Filing Cabinet and Desk

The source-document filing cabinet holds every ingested artifact verbatim;
the RecMem desk holds passages deliberately loaded as mid-term working
material. Every command's output includes the exact next step.

| Command | Description |
|---------|-------------|
| `hexis docs search <query> [--chunks] [--path P] [--type T] [--limit N] [--json]` | Search documents; `--chunks` for passage-level hybrid search with citable locators |
| `hexis docs open <id\|hash\|path> [--offset N] [--chars N] [--page A[-B]] [--json]` | Read a document verbatim (paged), or open a PDF page range |
| `hexis docs info <id\|hash\|path> [--json]` | Provenance, chunk counts, original artifact, extraction runs and warnings |
| `hexis docs load <id\|hash\|path> [--pages A-B] [--reason TEXT] [--pin] [--json]` | Load a document (or page range) onto the RecMem desk |
| `hexis desk list [--pinned] [--json]` | List what is on the desk |
| `hexis desk open <item-id> [--offset N] [--chars N]` | Read a desk item (paged; 8-char id prefixes work) |
| `hexis desk search <query> [--limit N]` | Full-text search across desk items |
| `hexis desk pin <item-id>` / `hexis desk unpin <item-id>` | Protect an item from desk cleanup / release it |
| `hexis desk clear [ids ...] [--doc DOC_ID] [--all] [--include-pinned]` | Archive desk items (sources always stay in the cabinet) |

### Ingestion

| Command | Description |
|---------|-------------|
| `hexis ingest --file FILE` | Ingest a file |
| `hexis ingest --input DIR` | Ingest a directory |
| `hexis ingest --url URL` | Ingest a URL |
| `hexis ingest --stdin --stdin-type TYPE --stdin-title TITLE` | Ingest from stdin |
| `hexis ingest status [--pending] [--json]` | Show ingestion status, chunk/artifact counts, and recent extraction runs with warnings |
| `hexis ingest backfill-chunks [--limit N]` | Chunk stored documents that predate durable chunks (embedding happens in the background worker) |

Common flags: `--mode {fast,slow,hybrid}`, `--min-importance F`, `--permanent`, `--base-trust F`, `--no-recursive`, `--quiet`

## hexis init

Full flags for the init wizard:

```
hexis init [--api-key KEY] [--provider PROVIDER] [--model MODEL]
           [--character CHARACTER] [--name NAME]
           [--no-docker] [--no-pull]
           [--dsn DSN] [--wait-seconds N]
```

| Flag | Description |
|------|-------------|
| `--api-key` | API key (auto-detects provider; triggers non-interactive mode) |
| `--provider` | LLM provider (auto-detected from key if omitted) |
| `--model` | LLM model (defaults per provider) |
| `--character` | Character card name (e.g., `hexis`, `jarvis`) |
| `--name` | What the agent calls you (default: `User`) |
| `--no-docker` | Skip Docker auto-start |
| `--no-pull` | Skip local embedding sidecar startup |

## Related

- [Quickstart](../start/quickstart.md) -- common init patterns
- [Auth Providers](../integrations/auth/index.md) -- provider-specific auth
- [Ingestion guide](../guides/ingestion.md) -- ingestion walkthrough
