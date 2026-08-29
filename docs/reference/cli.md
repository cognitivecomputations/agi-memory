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
| `hexis up [--build] [--profile PROFILE]` | Start the default stack from cached/published images; `--build` explicitly builds a source checkout |
| `hexis down` | Stop services |
| `hexis uninstall [--purge\|--cli-only] [--yes]` | Remove Hexis; `--purge` also deletes brain/config data and provably Hexis-owned embedding assets |
| `hexis ps` | Show running containers |
| `hexis logs [-f] [services...]` | View/tail logs |
| `hexis start` | Start heartbeat and maintenance workers manually if stopped |
| `hexis stop` | Stop workers |
| `hexis reset [--yes]` | Wipe DB volume and re-initialize |

When host worker services are installed, stack lifecycle commands exclude the
matching Docker workers and control the host owner instead.

### Host Worker Services

| Command | Description |
|---------|-------------|
| `hexis service install [--channels] [--env-file PATH] [--no-start] [--enable-linger] [--replace-docker-workers]` | Install heartbeat and maintenance as launchd/systemd user services; optionally migrate channels and explicitly replace running Docker workers |
| `hexis service status [--json]` | Show installed, enabled, and active state from the host service manager |
| `hexis service start\|stop\|restart [heartbeat maintenance channels all]` | Control installed services; defaults to all installed units |
| `hexis service logs [-f] [--lines N] [heartbeat maintenance channels all]` | Read the launchd files or systemd user journal; Ctrl+C exits follow mode |
| `hexis service uninstall [services...] [--yes]` | Stop and remove only Hexis-managed unit files; preserve logs |

Installation records the selected environment-file path, working directory,
Python executable, and optional instance under `~/.hexis/host-services.json`
with mode `0600`. Environment values and provider secrets are not copied into
the unit or state file. The migration flag is required if matching Docker
workers are already running, preventing accidental duplicate owners.

### Private Remote Access

| Command | Description |
|---------|-------------|
| `hexis tunnel start [--port PORT] [--no-start-stack] [--json]` | Start the local stack if needed, then create a tailnet-only Tailscale Serve route to the loopback dashboard |
| `hexis tunnel status [--port PORT] [--json]` | Inspect local bind posture, Tailscale connection, exact Serve target, Funnel state, and route ownership without changing anything |
| `hexis tunnel stop [--json]` | Remove only the root Serve/Funnel handler recorded as Hexis-owned; preserve the local stack, brain data, and unrelated Serve paths |

OSS has no application authentication layer. The command never enables Funnel,
never widens `HEXIS_BIND_ADDRESS`, and never replaces or adopts ambient Tailscale
configuration. The phone or computer must first be approved in the same tailnet.
See [Secure Phone and PWA Access](../operations/secure-remote-access.md).

### Local Speech

| Command | Description |
|---------|-------------|
| `hexis voice` | Inspect local speech status without changing state |
| `hexis voice setup [-y] [--wait-seconds S] [--json]` | Confirm/install optional Piper support, derive the live configured model, and start it |
| `hexis voice start [--wait-seconds S] [--json]` | Start the configured model without changing packages |
| `hexis voice status [--json]` | Show readiness, endpoint, model, ownership, state, and log path |
| `hexis voice stop [--json]` | Stop only the exact process recorded as Hexis-owned |

Speech is disabled by default. The sidecar binds only to loopback; Hexis refuses
remote or credential-bearing provider endpoints and never adopts an ambient
process. See [Voice and Talk Mode](../operations/voice.md).

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
| `hexis skills [--json]` | Show weekly learning/skill-review status |
| `hexis skills enable\|disable` | Opt in or out of the bounded weekly learning and proposal pass |
| `hexis skills proposals [--status STATUS]` | List durable skill proposals |
| `hexis skills review ID --action apply\|reject\|reopen` | Review one proposal with confirmation |
| `hexis demo [--json]` | Run rollback-only recall/refusal/energy/heartbeat proofs |
| `hexis maturity [--json]` | Score live capability maturity with evidence and next steps |

`hexis doctor` also reports whether the dashboard has a trusted HTTPS route for
the installable app, continuous tool reachability, and the immutable
tool-surface audit. Workers derive reachability from the registry, live tool
configuration, and loadable skills every 15 minutes by default. A warning includes
the broken worker/context/tool path or the exact command needed to start measurement;
these advisory checks never stop the heartbeat.

### Chat and Memory

| Command | Description |
|---------|-------------|
| `hexis chat [--dsn DSN]` | Interactive chat |
| `hexis recall <query> [--limit N] [--type TYPE] [--json]` | Search memories |
| `hexis retention [--json]` | Show memory pressure, recoverable archives, review backlog, and hard-pruning posture |
| `hexis retention dry-run [--json]` | Run one rollback-only real rest cycle and show its exact diff |
| `hexis retention enable` | Preview and confirm reversible rest-cycle consolidation |
| `hexis retention disable` | Pause automatic rest-cycle consolidation without deleting anything |
| `hexis export --intent INTENT [--output FILE] [--format json\|jsonl]` | Export an HMX memory exchange |
| `hexis export --mind [--output FILE] [--format json\|jsonl]` | Export a complete private port file; defaults to `$HEXIS_HOME/exports` |
| `hexis import FILE --mind --dry-run --json` | Validate an empty-target mind move without mutation |
| `hexis import FILE --mind --confirm-intent port` | Import a mind into an empty target and verify lineage/protected-state continuity |
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

`--mind` is the safe first-class port preset. Export includes all portable and
protected sections, private-marked memories, in-flight work, and audit history;
it rejects filters or redaction that would make the file only a partial mind.
Import selects the empty-target additive path, retains the normal dry-run and
exact-intent confirmation gates, refuses active-target replacement, then verifies
the source lineage and all constitutional section projections while preserving
the exact transport digests for audit. See
[Move a Mind Between Machines](../guides/mind-portability.md).

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

### Companion Nodes

| Command | Description |
|---------|-------------|
| `hexis node init --name NAME` | Create this device's signed identity; refuses to overwrite one |
| `hexis node status [--local-only] [--json]` | Show truth-derived local capabilities and policy, paired nodes, and pending requests without exposing the private key |
| `hexis node allow ALIAS [--allow-args] [--replace] -- EXECUTABLE [FIXED_ARGS...]` | Pin one direct executable/argv policy under a local alias |
| `hexis node disallow ALIAS` | Remove one local command alias |
| `hexis node run [--gateway URL] [--once]` | Connect outward, pair in place, and serve approved invocations |
| `hexis node wake setup [--model NAME_OR_PATH] [--threshold F] [--device NAME] [-y] [--accept-model-license]` | Install optional local detection, explicitly select/license a model, and enable it in node policy without opening the microphone |
| `hexis node wake status [--json]` | Read local wake configuration without opening the microphone |
| `hexis node wake disable` | Disable wake capture for future node runs without deleting identity or models; Ctrl+C stops a current run |
| `hexis node pairing [list] [--json]` | List pending signed identities |
| `hexis node pairing approve\|deny REQUEST [--note TEXT]` | Decide a request by UUID or exact short code |
| `hexis node invoke NODE_ID system.run --command ALIAS [--arg VALUE...] [--timeout S] [--yes]` | Explicitly run one locally allowed alias |
| `hexis node invoke NODE_ID screen.capture [--output FILE] [--overwrite] [--timeout S] [--yes]` | Explicitly request a screen capture; refuses to replace a file without `--overwrite` |
| `hexis node revoke NODE_ID [--reason TEXT] [--yes]` | Permanently revoke that signed identity and cancel queued work |

The daemon never opens an inbound port. Pairing approval, capability-escalation
approval, the agent tool-approval gate, node-local command policy, and the
server's wake gate are separate boundaries. See
[Companion Nodes](../operations/companion-nodes.md) for setup, remote-access
constraints, permissions, and recovery.

### Execution Backends

| Command | Description |
|---------|-------------|
| `hexis execution status [--json]` | Show the selected profile and local prerequisites without connecting remotely |
| `hexis execution add-ssh NAME --host HOST --user USER --workspace PATH --identity-file PATH --known-hosts-file PATH [--port N] [--replace]` | Save an exact SSH profile without activating or connecting to it |
| `hexis execution add-docker NAME --docker-host ssh://USER@HOST[:PORT] --image IMAGE --workspace PATH --identity-file PATH --known-hosts-file PATH [--network none\|bridge] [--replace]` | Save an ephemeral remote-Docker profile with no implicit pull |
| `hexis execution test [NAME] [--json] [--timeout S]` | Explicitly connect and verify workspace, shell, and Python availability |
| `hexis execution use NAME` | Select a profile for new `shell`, `safe_shell`, `run_script`, and `execute_code` calls |
| `hexis execution remove NAME` | Remove an inactive profile while preserving remote files and state volumes |

Selection is database-owned and live. A remote failure never falls back to the
local worker. See [Execution Backends](../operations/execution-backends.md).

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
