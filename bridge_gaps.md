# Bridging the Gaps — from cognitive architecture to practical assistant

**Date:** 2026-08-21 · **Basis:** local read of `.reference/openclaw` (OpenClaw,
TS/Node, 52 skills, 158 extensions, native apps for macOS/iOS/Android/Linux/Windows)
and `.reference/hermes-agent` (Nous Research Hermes, Python, 95 tool modules,
6 execution backends), cross-read against this tree.

**Companion:** `positioning.md` — what Hexis should be *best in the world at*, as
opposed to this document, which is what it needs to reach parity. Two findings there
are engineering work rather than strategy and belong on this plan's radar: the
**invocation problem** (§4 there — capabilities fire on the agent's own state, not on
the user's vocabulary, so `bridge_gaps.md` Tier 0's semantic-selection fix is
necessary but not sufficient), and **dormant detectors** (`find_contradictions()` has
produced zero findings across 323 memories; 252 of 323 memories share one computed
default `trust_level`).

## The finding in one line

Hexis is not short on capability — 51 tool modules, 25 skills, 8 channel adapters,
sub-agents, cron, a document cabinet, a real memory architecture. It is short on the
**last mile between "has a tool for that" and "handles it for you."**

The tell is in the docs. `docs/concepts/` has seven files and every one is about
*being someone*: consent-and-boundaries, identity-and-worldview, memory-architecture,
heartbeat-system, self-development. OpenClaw's `docs/automation/` has twelve and every
one is about *getting your work done*: tasks, cron-jobs, hooks, standing-orders,
taskflow, webhook, poll, gmail-pubsub.

Both are real products. Only one of them takes something off your plate today.

## What this plan does not change

The differentiators stay. Layered memory with precomputed neighborhoods and an AGE
graph, a document cabinet with page/section locators, energy as a real budget, and an
agent that can refuse — neither competitor has any of it, and none of the work below
trades it away.

Nor do we copy Hermes's autonomy posture. Hermes writes skill files on its own;
`services/skill_improvement.py` deliberately "never writes skill files," only
proposals. **That stays.** Every mechanism below is proposal-then-consent, which is
not a compromise here — it is the reason two of these gaps close cheaply, because
`resource_requests` already implements exactly that pattern.

## Principles for every item

1. **Extend, don't parallel.** Suggestions produce `scheduled_tasks` rows through the
   existing `create_scheduled_task()`. There is no second job engine, no second inbox, no
   second consent surface.
2. **The agent proposes; the person decides.** Nothing below ever acts on a timer or
   by default.
3. **A "no" is remembered.** Every proposal surface latches dismissals so the same ask
   is never re-offered.
4. **Derive from truth** (Experience Bar #1). Do not offer a Gmail digest to someone
   with no Gmail connected.
5. **Degrade loudly, never silently.** A question that can't be asked, a suggestion
   that can't be built, a voice backend that isn't installed — each says so.

---

# Tier 0 — capabilities that exist and cannot be reached

**Added 2026-08-21 after probing the live tool catalog and running the production
skill-selection code offline.** This tier was not in the original analysis. It is
now first, because the fixes are cheaper than anything below and they unlock
capability already paid for.

## How the gate works

`services/skill_runtime.py:select_skills()` decides, per turn, which skills are
active. `allowed_tool_names` is then `DISCOVERY_TOOL_NAMES` (4 tools) plus the bound
tools of the selected skills — and `core/agent_loop.py:527` hard-refuses anything
else with *"tool not available in the active skill set."*

Selection is: the defaults (`{core-memory}` in chat), plus up to 3 more chosen by
`_score_skill()` — **literal token overlap**, 5 points for a token matching the
skill's *name*, 3 for its description, 1 for its body — with
`AUTO_ACTIVATE_SCORE_THRESHOLD = 5`. Five points effectively means **the user has to
say the skill's name.**

## What the probe found

Running the real `_score_skill` / `_passes_specialized_gate` / selection path over ten
ordinary assistant requests: **seven of ten activated `core-memory` and nothing else.**

| request | best non-default score | outcome |
|---|---|---|
| "did I get anything important in email?" | 2 | no email tools |
| "book time with Sarah next week" | **0** | no calendar tools |
| "remind me to call Bob at 4pm tomorrow" | 1 | no calendar tools |
| "add milk to my shopping list" | 3 | nothing |
| "who is Manning and what do we owe them?" | 4 | no contacts |
| "keep an eye on the deploy and tell me if it breaks" | 4 | nothing |
| "what did we decide about pricing last month?" | 3 | memory only |

**The always-on floor in chat is 36 tools of the 150 defined** — memory, desk,
journal, goals, backlog, schedule. Everything else waits behind a skill that mostly
does not activate:

- **email** (13 tools) — `email_list`, `email_read`, `email_search`, `gmail_*`
- **calendar** (5) — `calendar_events`, `calendar_create`, `meeting_prep`
- **contacts** (7) — `search_contacts`, `get_contact`
- **web** (6) — **`web_search`, `web_fetch`, `browser`**
- **files/shell** (10), **messaging** (7)

The agent cannot search the web by default.

## Four distinct causes

**1. Vocabulary mismatch — the user's word is not the system's word.** Every email
skill is named `gmail-*`. A user asking about "email" scores 0 on the name tokens
`{gmail, actions}`. `email-digest` would match — but it is `contexts: [heartbeat]`,
so it cannot load in chat at all. Same for `daily-briefing`.

**2. Lexical matching where semantic matching is already available.** "book time with
Sarah next week" scores **0** against the `calendar` skill: not one of `{book, sarah,
week}` appears in its name or description. Hexis has an embedding service, a cached
`get_embedding()`, and pgvector — the whole substrate for semantic selection — and
the selector uses `str.split()`.

**3. Ten tools are bound to no skill at all**, so no `use_skill` call can ever unlock
them. They are unreachable in every turn, in every context:

```
manage_sessions      ← sub-agents / delegation
explore_concept      ← graph-walk over memory
explore_subgraph
execute_workflow
database_backup      backup_retention
config_export        config_import
post_process_output  connect_twitter_x
```

`manage_sessions` is the delegation capability §"not gaps" credits Hexis with having.
It is real, it is tested, and **the agent has never been able to call it.**

**4. The escape hatch is uphill.** The prompt says "Use skills first" and lists 22
index lines. But nothing signals that *this* request needs one, so the model weighs a
two-hop detour (`use_skill` → retry) against answering from the 36 tools it already
has. It answers from memory. The refusal message only appears *after* a wrong guess.

The floor makes this vivid: `manage_schedule` is always on, `calendar_create` is not.
The agent can schedule its own future work and cannot put anything on your calendar.

## Fixes, cheapest first

**0.1 Widen the floor.** Add `web_search`, `web_fetch`, `calendar_events`,
`search_contacts`, `get_contact`, `email_search`, `email_list` to the always-on set —
read-only, low-energy, and the ones every assistant reaches for. Nothing here is
destructive; the gate earns its keep on `email_send` and `shell`, not on reading.
*One line in `DEFAULT_SKILL_NAMES` plus a small always-on skill. ~1 hour.*

**0.2 Bind or float the ten orphans.** Each goes into a skill or into the floor.
`manage_sessions` belongs in a new `delegation` skill; `explore_concept` /
`explore_subgraph` belong in `core-memory`; the backup/config tools want an
`operations` skill. *~2 hours, and it turns delegation on for the first time.*

**0.3 Add `aliases:` to skill frontmatter**, scored exactly like name tokens.
`gmail-actions` gets `[email, mail, inbox]`; `calendar` gets `[book, meeting,
schedule, appointment, availability]`; `crm-lookup` gets `[who, company, account]`.
*~3 hours including a pass over all 25 skills.*

**0.4 Make heartbeat-only skills chat-reachable.** `daily-briefing` and
`email-digest` have no reason to be unavailable when a person asks for them directly.
*~30 minutes.*

**0.5 Semantic selection.** Replace token overlap with embedding similarity over
skill name + description, using the embedding service already in the stack. Keep the
lexical score as a floor so an exact name match always wins. This is the real fix —
0.3 is a stopgap for the same problem. *~1 day.*

**0.6 Instrument it.** Nothing today records which skills were considered, what they
scored, and what was refused for not being active. Log the selection decision per
turn and the `not_available_in_active_skills` refusals. Without this, the regression
is invisible — as it has been. *~2 hours, and it should land first so 0.1–0.5 can be
measured.*

Then **port** `capability_probe.py` + `tool_surface_audit.py` (§11.4·8) so this stops
being a one-off audit: a per-worker × per-tool reachability probe and an immutable
record of every tool-surface decision. The findings above were produced by hand once;
these keep producing them.

**Total: about two days for all six.** Compare against every other tier in this
document. This is the cheapest capability increase available, because none of it
builds a capability — it stops hiding the ones already built.

# Tier 1 — the two that change daily usefulness most

Both are small. Both fit machinery that already exists. Do these first.

## 1. Automation suggestions — the agent proposes routines

**Gap.** Hermes ships `cron/suggestions.py` + `cron/blueprint_catalog.py`: 14 curated
starter automations (Morning briefing, Important-mail monitor, Bills & renewals
reminder, Habit check-in, Weekly meal plan, Evening wind-down, On-this-day discovery)
surfaced as one-tap accept, sourced from a catalog, a skill's `blueprint:` block, a
usage review that noticed a recurring ask, or a freshly connected account. Their own
docstring: *"Suggestions never auto-create jobs; acceptance is always explicit
(consent-first)."* Dismissals latch by `dedup_key`.

Hexis has `manage_schedule` and `scheduled_tasks` (db/00:1037, db/19). The user just
has to know to ask.

**Why this is first.** It is the highest-leverage item on the list and the cheapest,
because the consent model it needs is the one Hexis was built around. It converts the
agent from something you operate into something that meets you halfway.

**Build.**

`db/migrations/0199_automation_suggestions.sql`:

```sql
CREATE TABLE IF NOT EXISTS automation_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL CHECK (source IN ('catalog','blueprint','usage','connector')),
    dedup_key TEXT NOT NULL UNIQUE,      -- a "no" here is final
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,             -- why this, why now, in the agent's voice
    task_spec JSONB NOT NULL,            -- verbatim manage_schedule 'create' arguments
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','accepted','dismissed')),
    scheduled_task_id UUID REFERENCES scheduled_tasks(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMPTZ
);
```

Functions alongside it: `propose_automation(source, dedup_key, title, rationale,
task_spec)` — a no-op returning the existing row if the key was ever dismissed;
`accept_automation(id)` — calls `create_scheduled_task()` with the stored spec and links the
row; `dismiss_automation(id)`; `list_automation_suggestions(status)`.

**Sources, in build order:**

- **catalog** — a seeded starter set in `db/*.sql`, gated on what is actually
  configured. Each entry declares its precondition (`requires: gmail_connected`,
  `requires: calendar_connected`, `requires: none`) and is only proposed when it
  holds. Start with the ones that need nothing: morning briefing (the
  `daily-briefing` skill already exists), evening wind-down, weekly review.
- **connector** — `services/connector_setup.py` emits the obvious automations for a
  surface the moment it finishes connecting. Connecting Gmail should immediately offer
  the important-mail monitor.
- **usage** — `services/skill_improvement.py` already runs an opt-in background review
  over recent turns. Extend it to emit a suggestion when it sees the same ask three
  times ("you've asked for the standings every Monday").
- **blueprint** — a `blueprint:` block in a skill's YAML frontmatter (the format in
  `skills/installed/*/SKILL.md` already carries `requires:`, `contexts:`,
  `bound_tools:`). Installing a skill registers a suggestion instead of scheduling
  anything.

**Surfaces.** The web inbox already renders `pending_requests` with a decide endpoint
(`hexis-ui/app/api/requests/decide`, and the chat page's "requests awaiting your
decision" panel). Add suggestions to the same panel with Accept / Not for me. On
channels, deliver through the outbox with numbered replies.

**Effort:** ~2 days for the table, functions, accept/dismiss surfaces, and the
no-precondition catalog. Another day per additional source.

## 2. `ask_user` — a question the agent can actually ask mid-task

**Gap.** Hermes's `tools/clarify_tool.py` presents up to four choices plus an always-
appended "Other (type your answer)," and the platform layer renders it natively:
arrow-key picker in the CLI, numbered list on Telegram, buttons on Discord. The turn
blocks on the answer.

Hexis's only equivalent is `queue_user_message` — it drops a note in the outbox for
the *next heartbeat*. That is a message, not a question. An assistant that cannot ask
"the Manning contract or the Hartford one?" has to guess, and guessing is where trust
dies.

**Build.**

`db/migrations/0200_agent_questions.sql`:

```sql
CREATE TABLE IF NOT EXISTS agent_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID,
    surface TEXT NOT NULL,               -- chat | cli | heartbeat | <channel>
    prompt TEXT NOT NULL,
    choices JSONB NOT NULL DEFAULT '[]'::jsonb,   -- <= 4; "Other" is appended by the UI
    allow_free_text BOOLEAN NOT NULL DEFAULT TRUE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','answered','timed_out','superseded')),
    answer TEXT,
    asked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    answered_at TIMESTAMPTZ
);
```

**Tool:** `ask_user` in `core/tools/` — **energy cost 0.** Asking must never be
rationed; it is strictly cheaper than acting on a wrong guess.

**The dual mode is the crux, and it is Hexis-shaped.** Two behaviors from one tool:

- **Someone is present** (chat, CLI, an active channel thread) → the tool awaits the
  answer and the turn continues with it. Bounded by a new
  `chat.question_timeout_s` config (default 300). On timeout the tool returns *"no
  answer — proceed on your best judgment and say which way you went,"* never an error.
- **Nobody is present** (heartbeat) → file the question, deliver it through the
  outbox, and return *"asked; not yet answered"* so the beat ends cleanly. The answer
  arrives through the inbox and the agent picks it up on a later beat. This is the
  existing `queue_user_message` path, now with structure.

**Transports:**

- Chat SSE: a `question` event in `apps/hexis_api.py`; the UI renders a choice card.
  `ConnectorSetupCard` in `hexis-ui/app/chat/page.tsx` is already this shape —
  generalize it rather than writing a second one.
- CLI: `questionary` is already a dependency (`pyproject.toml`), so the arrow-key
  picker is nearly free.
- Channels: render in `channels/presentation.py` as a numbered list; parse a bare
  `2` in the reply back to the choice. Discord gets real buttons later.

**Effort:** ~3 days including all three transports.

---

# Tier 2 — the face and the hands

**Revised 2026-08-21.** An earlier draft folded presence, local-app skills, and voice
into one `hexis-node` daemon. That was one mechanism too few. There are two, and they
split cleanly:

- **The face — a PWA.** How the agent reaches *you*: a real installed app on desktop,
  Android, and iOS, with push notifications and microphone capture.
- **The hands — `hexis-node`.** How the agent reaches *your machine*: `osascript`,
  the 1Password CLI, screenshots. A browser sandbox can never do this.

The PWA is the cheaper half and delivers most of the user-visible value, so it goes
first. The node narrows to host commands only and loses its UI ambitions entirely.

**Why a PWA is the right call here specifically, and not just the cheap one:** a PWA
cannot execute in the background, which is normally disqualifying for an assistant.
Hexis is the case where it does not matter — **the agent already runs server-side.**
The heartbeat lives in a worker container around the clock; the client is a window
onto something always-on that *pushes* to it. OpenClaw needs native apps because its
nodes turn the device into a peripheral. Hexis does not need that to ship a client.

## 3a. The PWA — one app for desktop, Android, and iOS

**Gap.** Hexis has a Next.js dashboard at `:3477` and nothing installable. OpenClaw
ships `apps/` for android, ios, linux, macos plus a Windows Hub; Hermes ships a
desktop app and a TUI.

**Do not port a second app.** `~/samantha-pwa` is Vite + React 19; `hexis-ui` is Next
16.1.3 App Router and already carries streaming chat, attachments, the inbox, the
activity panel, and connector cards. Two chat UIs is the wrong outcome. Harvest the
**PWA layer** from that reference — the part nobody wants to write twice:

- `public/icons/` — 11 icon sizes and 7 iOS splash screens, already generated.
- `public/sw.js` — a service worker whose `push` handler already calls
  `showNotification` (`public/sw.js:171`).
- `public/manifest.webmanifest` — notably `display_override:
  ["window-controls-overlay", "standalone"]` and an `edge_side_panel` block, which is
  what makes it a real desktop window rather than a browser tab.
- `src/components/WebRTC.jsx` + `VUMeter.jsx` — `getUserMedia` capture and a live
  level meter, i.e. the entire UX for voice input.

Next 16 provides `app/manifest.ts` natively, so the work is: manifest route, service
worker in `public/`, the icon set, an install prompt, and a push subscription
endpoint.

**What this covers that the node was going to:**

- **§5.1 voice-in, in full.** `getUserMedia` in the foreground → record → POST to
  `transcribe`. No node required.
- **§4 presence**, on desktop and Android; partially on iOS.
- **Web Push.** This matters more than it looks: Tier 1's automation suggestions are
  worthless if they sit in a web inbox nobody has open. Push is what makes the
  agent's proposals *arrive*.

**What it will never cover — state this rather than discovering it later:** host
commands, wake words, and background listening. On iOS additionally: push requires
"Add to Home Screen" (16.4+), there is no Web Share Target, no Siri, and storage is
evicted after ~7 days unused if the app is not installed.

**Hard prerequisite: HTTPS.** Service workers, Web Push, and `getUserMedia` all
require a secure context. `http://localhost:3477` qualifies, so development is fine —
**a phone hitting `http://192.168.x.x:3477` does not.** No service worker, no push,
no microphone. See §8; that is the real work item, not the manifest.

**Effort:** ~3 days, most of it harvested. Gated on §8.

## 3b. `hexis-node` — the hands, and only the hands

**Gap.** OpenClaw's nodes are companion devices (macOS/iOS/watchOS/Android) that
connect to the Gateway with `role: "node"` and expose a command surface —
`canvas.*`, `camera.*`, `device.*`, `notifications.*`, `system.*` — invoked via
`node.invoke`, with signed device identity and explicit pairing approval. Their macOS
menu bar app runs in node mode. Hexis has a web dashboard and a CLI.

**Scope, now that the PWA has the face.** The node is headless. No chat window, no
canvas, no notifications surface — the PWA does those. What is left is the set of
things a browser sandbox is permanently barred from.

**Build.** A small daemon shipped with the CLI (`hexis node run`), connecting outward
to the gateway over the existing RabbitMQ/WS plumbing in `core/rabbitmq_bridge.py`
and `core/gateway.py`. Outward-only: no inbound port, no firewall rule.

Command surface, in build order:

1. `system.*` — run an allowlisted host command. This is the whole point: `osascript`
   for Apple Reminders/Notes/Calendar, the 1Password CLI, `shortcuts`.
2. `screen.capture` — visual context the browser cannot take.
3. `audio.*` — only if a wake word is ever wanted. The PWA covers foreground voice,
   so this is no longer on the critical path.

**Pairing is the security boundary and must not be skipped.** Follow OpenClaw:
the node presents a signed identity, the gateway files a pairing request, and the
operator approves it — through the same decision surface as everything else. A node
that can run host commands is the most dangerous thing in the system; it gets the
strictest consent.

**Effort:** ~1 week for the daemon, pairing, and `system.*` + `notifications.*`.

## 4. Everyday-life skills — point the skill surface outward

**Gap.** OpenClaw's 52 skills: `apple-notes`, `apple-reminders`, `things-mac`,
`obsidian`, `notion`, `spotify-player`, `sonoscli`, `openhue`, `weather`, `trello`,
`1password`, `peekaboo`, `tmux`, `himalaya`. Hexis's 25: `core-memory`,
`self-reflection`, `self-inspection`, `council`, `cost-report`, `skill-authoring`,
`memory-exchange`, plus sales-shaped ones (`crm-lookup`, `outreach`). Hexis can
reason about its own belief revisions and cannot add milk to your reminders.

**Build, in three waves by what they need:**

- **Wave A — skill-only, tools already exist.** Nothing new to build; these are
  `SKILL.md` files over `calendar_*`, `email_*`, `search_contacts`, `web_search`,
  `github`, `todoist_*`, `asana_*`. Travel prep, inbox triage, meeting follow-ups,
  weekly review, expense capture. **Start here — it is a day of writing, not
  engineering, and it is the cheapest visible win in this document.**
- **Wave B — API-backed tools.** Notion, Spotify, Home Assistant, weather, Trello.
  Each is a `ToolHandler` in `core/tools/` plus a connector-setup flow of the kind
  `services/connector_setup.py` already runs for Gmail.
- **Wave C — needs the node (§3b).** Apple Reminders/Notes/Calendar via `osascript`,
  1Password CLI, Shortcuts, screenshots. Filesystem-backed ones (Obsidian, Bear) work
  today if the vault is mounted — do those in Wave A.

**Effort:** Wave A ~2 days. Wave B ~1 day per integration. Wave C follows §3.

## 5. Voice

**Gap.** OpenClaw has wake words, Talk Mode, and three TTS paths (`sherpa-onnx-tts`,
`macos-mlx-tts`, azure-speech/deepgram). Hermes has `voice_mode.py`, `tts_tool.py`,
`transcription_tools.py`, and voice-memo transcription. In Hexis the only audio code
in the tree is `services/ingest/readers.py`, for ingesting audio *files*. Grep for
"voice" and you get persona voice — the writing style.

**Build, layered, each layer useful alone:**

1. **`transcribe` tool** — a voice memo sent on Telegram/WhatsApp/Signal becomes text.
   Works today with no node and no PWA, because the audio arrives as a file through
   the channel adapters. **Cheapest real voice win; do it first.** **Port, do not
   build** — `voice_notes.py` + `local_audio_analysis.py` in Alex's fork (§11.4·7)
   already implement this pipeline. The PWA (§3a) then
   reuses the same endpoint for in-app capture via `getUserMedia`.
2. **`speak` tool** — TTS out, following the `embeddinggemma` sidecar precedent: a
   self-published binary, no third-party runtime. See `docs/operations/embeddings.md`
   for the shape this should take.
3. **Talk mode** — continuous listen/respond. Foreground-only in the PWA (§3a),
   which is enough for a conversation you started; always-on needs the node (§3b).
4. **Wake word** — needs the node, always-on, last, and only once 1–3 are solid.

**Effort:** step 1 ~2 days. Steps 2–4 are each roughly a week and gated on §3.

---

# Tier 3 — reach and footprint

## 6. Install and stay-alive footprint

**Gap.** OpenClaw: `npm i -g openclaw` then `openclaw onboard --install-daemon`,
which registers a launchd/systemd **user service** so the assistant stays running.
Hermes: `curl | bash`, bundling uv, Python, Node, ripgrep, ffmpeg, and a portable Git.
Hexis: Docker, compose, five services, and an image build.

**This is not theoretical.** On 2026-08-20 an install of this very repo produced a
configured, consented agent whose heartbeat never fired, because `hexis init` returned
early on a database-only stack (fixed in `d1d1485`). The retry then failed because the
worker image could not finish `pip install` over a PyPI serving 37 kB/s — pip walked
every `tiktoken` release, never resolved `regex`, and reported `ResolutionImpossible`.
Neither competitor has a build step in the install path, so neither has this failure
mode.

**Build, cheapest first:**

1. **Take the build off the install path.** `hexis up` in a source checkout currently
   builds by default (`apps/hexis_cli.py`, the `--no-build` opt-out). Invert it:
   prefer the published image, build only on `--build` or when the tree provably
   differs. One afternoon; removes the whole failure above.
2. **Make the image build deterministic.** `ops/Dockerfile.worker` runs a bare
   `pip install .` with no lockfile and no timeout tuning, which is why a slow index
   became a fake dependency conflict. Move to `uv` with a committed lock. A slow
   network should be slow, never wrong.
3. **Workers as host processes.** Postgres genuinely needs the container (AGE +
   pgvector). The heartbeat and maintenance workers do not — they are stateless
   Python by design. Install them as launchd/systemd user services from the same uv
   tool that installs the CLI. This removes the worker image from the critical path
   entirely and matches how both competitors stay alive.

**Effort:** step 1 an afternoon, step 2 ~2 days, step 3 ~1 week.

## 7. Runs where it needs to

**Gap.** Hermes `tools/environments/` has six backends — `local.py`, `docker.py`,
`ssh.py`, `singularity.py`, `modal.py`, `daytona.py` — with serverless hibernate/wake,
so the agent costs nearly nothing idle and you talk to it from Telegram while it works
on a cloud VM. Hexis's `shell`, `run_script`, and `code_execution` run in the worker
container, on one machine.

**Build.** An execution-backend abstraction behind the existing tools, so the tool
contract does not change: `local` (today), then `ssh`, then `docker-remote`. Modal and
Daytona only if someone asks.

**Priority: last.** It is the largest piece of work here and the one fewest users will
notice. Listed for completeness, not urgency.

---

## 8. Remote access — the prerequisite nobody scheduled

**Gap.** Every port in `docker-compose.yml` binds `${HEXIS_BIND_ADDRESS:-127.0.0.1}`,
and `docs/operations/` contains nothing on tunnels, TLS, or reaching Hexis from
another device. The agent runs on one machine and can only be reached from that
machine's browser.

This has been invisible because the dashboard has always been a localhost tool. The
PWA makes it blocking: **no HTTPS means no service worker, no push, no microphone**,
so "install Hexis on your phone" is not a thing that can happen until this is solved.
It is also what stands between a user and the one-line pitch both competitors lead
with — *talk to it from Telegram while it works on a cloud VM*.

**Build, in ascending order of commitment:**

1. **Document the Tailscale path first.** It is the honest 80% answer: a tailnet gives
   a stable hostname, a real certificate via `tailscale cert`, and no public exposure
   at all. A page in `docs/operations/` and a `hexis doctor` check that says whether
   the dashboard is reachable over HTTPS. *~1 day, and it unblocks §3a immediately.*
2. **A `hexis tunnel` command** wrapping the same, so the path is one command instead
   of a runbook.
3. **Pairing and posture for devices, not public exposure.** OpenClaw defaults DMs
   from unknown senders to a pairing handshake rather than processing them, and ships
   a *Gateway exposure runbook*. Hexis wants the pairing half — device approval for
   nodes and PWA installs (§3a, §3b) — plus a `hexis doctor` check that fails loudly
   on a risky configuration.

**There is no step 4, and this is a permanent constraint rather than a backlog item.**
API-key authentication is a **Hexis Pro** feature; **OSS has no auth layer and is not
getting one.** So OSS remote access is *only ever* network-layer: a tailnet, or a
reverse proxy that brings its own authentication. Binding the dashboard to a public
interface is not premature — it is out of bounds, and `hexis doctor` should say so in
those terms rather than merely warning.

**Effort:** step 1 ~1 day. Steps 2–3 ~1 week combined.

## 9. The heartbeat — the differentiator, audited

**Added 2026-08-21** after reading `db/07`, `db/39`, `execute_heartbeat_action`, and
the live energy config. The heartbeat is the thing no competitor has: an agent that
acts without being asked. Neither OpenClaw's cron nor Hermes's scheduler is the same
animal — those run *jobs you defined*; this one *decides*. It is also the part of the
system most worth getting right, and it has one live bug and four design limits.

**What to leave alone.** The DB emits intentions and Python is a dumb executor, so a
worker killed mid-beat loses nothing — the state transition already committed. The
action space is a Postgres ENUM with costs in config, not a prompt convention, so the
agent cannot invent an action and the cost table is tunable data. Energy is a
structural guarantee against spam rather than an instruction that might be ignored.
Boundary checks run *before* dispatch on `reach_out_public` and `synthesize`. None of
this should be traded for anything in the reference implementations.

### 9.1 Seven offered actions have no implementation — **bug, fix first**

`heartbeat.allowed_actions` offers 35 actions with configured energy costs.
`execute_heartbeat_action` implements 28. These seven are advertised, priced, and dead:

```
debate_internally (cost 2)   inquire_deep (cost 6)   study        (cost 2)
meditate          (cost 1)   fast_ingest  (cost 2)   hybrid_ingest (cost 3)
slow_ingest       (cost 5)
```

Choosing one returns `{"success": false, "error": "Unknown action: study"}`. It fails
loudly and does not charge energy — that part is correct — but the beat's entire LLM
call is spent deciding to do something impossible, and the beat may end having done
nothing.

**20% of the advertised action space is a trap**, and it includes `inquire_deep`, the
second-most-expensive action in the system. The three `*_ingest` entries are the
strangest: the tools exist and are reachable, and the wiring simply stops at the
heartbeat.

This is the same pathology as Tier 0 — advertised capability with no path to running —
and it belongs to the same fix-it-first bucket.

**Fix:** implement them or remove them from `allowed_actions`. Either direction closes
it. Add a startup assertion that every entry in `allowed_actions` has a handler branch,
so this cannot silently return. *~half a day either way.*

For `debate_internally` specifically there is a third option: **give it a body.**
Alex's `deliberation.py` implements adversarial conjecture–attack–verdict reasoning
(§11.4·9), which is what that action was always supposed to mean. It needs its
`independence_engine` / `prediction_journal` / fragility dependencies stripped first.

### 9.2 Energy saturates and the surplus is destroyed

`base_regeneration = 10`, `max_energy = 20`, interval 60 minutes. **Energy is full
after two hours.** An agent idle overnight wakes at 20, exactly as if it had rested
since 2am — ten hours of regeneration discarded.

The cost is not waste, it is *expressiveness*: nothing costing more than 20 can exist,
so **no ambition spanning more than one beat is representable.** The most expensive
thing the agent can conceive of is `inquire_deep` twice. An economy shaped like this
can only express errands.

**Fix:** let energy bank past the cap with decay, or introduce a project that draws
down across several beats. *~2 days, and it is what makes long-horizon autonomy
possible at all.*

### 9.3 Regeneration is time-based, so nothing rewards usefulness

+10/hr whether the last beat resolved a contradiction or picked `observe` and went back
to sleep. The budget constrains *volume* and never steers toward *value* — and `rest`
(cost 0) competes against thirty-four ways to look busy.

**Fix:** couple some fraction of regen, or the cap, to outcomes — a beat that produced
a durable memory, resolved a contradiction, or was thanked by the user regenerates
better than one that did not. *~2 days, and it turns energy from a rate limit into a
gradient.*

### 9.4 Fixed cadence ignores state it already computes

Every beat costs the same LLM call at 3am with nothing pending as at 9am after forty
unread messages land. `urgency_ratio` and `urgent_drives` are **already assembled into
the heartbeat context** and are not consulted when scheduling the next beat.

**Fix:** modulate `next_heartbeat_at` by the urgency already computed. Cheaper and more
relevant, with the inputs sitting in the same object. *~1 day.*

### 9.5 Near-synonymous actions

`contemplate`, `meditate`, `study`, `debate_internally`, `reflect` — five ways to think,
chosen from one flat list. It is doubtful the model distinguishes them reliably.

Three of the five are dead per §9.1, so this partly resolves itself: implement what is
genuinely distinguishable and delete the rest. **Now that beats are running, this is
measurable** — log the action distribution, and any action never chosen in a hundred
beats is answering the question for us. *~1 day, after §9.1.*

### 9.6 Two gates that do not know about each other

`services/agent.py:825` builds the heartbeat's skill-selection query as
`json.dumps(heartbeat_context)[:4000]` and runs the Tier 0 lexical matcher over it.
Skills are therefore chosen by keyword-matching a JSON dump, while actions are chosen
by the model from a typed enum, and **nothing reconciles the two.** The agent can pick
an action whose tools the selector happened not to activate; today their agreement is
coincidental.

**Fix:** derive the heartbeat's allowed tool set from the *chosen action*, not from
lexical overlap on a serialized context. An action is a far better predictor of the
tools a beat needs than a JSON dump is. *~2 days, and it depends on Tier 0.*

## 10. Communication cadence — contact points and the permission slip

**Added 2026-08-21.** An assistant with Slack, email and a phone number is one badly
calibrated loop away from being the most annoying entity in your life. Nothing in the
system currently prevents that: `heartbeat.user_contact_cooldown_hours = 4` is defined
in `db/00_tables.sql:664` and **referenced by no code**. Cadence today is entirely the
model's discretion, informed by one line of prompt ("time since last user interaction")
and a global counter that governs one person.

The goal is not a rate limit. It is **the cadence of an engaged human** — which is not
one number, because an engaged human contacts their partner hourly, a colleague on
Tuesdays, and a former coworker at Christmas, over different media, for different
reasons.

### 10.1 Three separate questions, three separate mechanisms

The mistake to avoid is collapsing these into one budget. They answer different
questions and they fail differently:

| Question | Mechanism | Failure if missing |
|---|---|---|
| *May I contact this person at all?* | **Purpose gate** (pass/fail) | the agent chit-chats at strangers |
| *How much of their attention am I spending?* | **Contact points** (price) | the agent floods people it has a reason to reach |
| *How much of my own capacity does this cost?* | **Energy** (existing) | the agent burns budget on busywork |

An assigned goal changes the first and the third. **It does not change the second** —
see §10.4.

### 10.2 The purpose gate

**Every outbound communication must carry a purpose, with exactly one exception.**

- **Third parties** — contact requires an instrumental purpose traceable to a goal, a
  responsibility, a thread the person themselves opened, or an explicit user request.
  "Checking in" on someone who is not the primary user is not a purpose. There is no
  relationship-maintenance budget for third parties, because the agent does not have
  relationships with them — **the user does**, and spending someone else's social
  capital is not the agent's to spend.
- **The primary user** — connection is itself a legitimate purpose. Reaching out
  because it has been four days and something is thin is exactly what an engaged
  person does, and it is the whole premise of the product. This is the one place a
  *relational* rather than *instrumental* reason passes the gate.

**On the inbound half, port rather than build.** Alex's `inbound_disposition.py`
(§11.4·6) already implements operator detection, trigger words, allowlists, drop rules
and ambiguity flagging, with all of the policy in PL/pgSQL — which is where this plan
wants it anyway.

Implementation: the purpose is a required, recorded field on the outbound action —
not a prompt convention. `reach_out_user` and `reach_out_public` take
`purpose_kind ∈ {goal, responsibility, reply, user_request, connection}` plus a
reference, and `connection` is only valid when the recipient is the primary user. A
missing or unbacked purpose fails the action loudly, the way an unknown action already
does.

### 10.3 Contact points, per relationship and per channel

A ledger shaped like energy, so it reads as native rather than bolted on:

```sql
CREATE TABLE contact_budgets (
    entity            TEXT NOT NULL,      -- matches the graph ConceptNode name
    channel           TEXT NOT NULL,      -- 'slack' | 'email' | 'sms' | ...
    points            FLOAT NOT NULL DEFAULT 1,
    regen_per_day     FLOAT NOT NULL,     -- the cadence dial
    max_points        FLOAT NOT NULL,     -- cannot bank a month into one afternoon
    observed_per_week FLOAT,              -- measured from history, see 10.3.4
    reciprocity       FLOAT NOT NULL DEFAULT 1.0,
    strain            FLOAT NOT NULL DEFAULT 0,
    last_outbound_at  TIMESTAMPTZ,
    last_inbound_at   TIMESTAMPTZ,
    PRIMARY KEY (entity, channel)
);
```

Relationship strength already exists — `update_trust` writes
`(SelfNode)-[:ASSOCIATED {kind:'relationship', strength}]->(ConceptNode)` into the AGE
graph via `upsert_self_concept_edge`. Strength maps to `regen_per_day`: partner ~3/day,
close friend ~1/day, colleague ~1/weekday, acquaintance ~1/week, dormant ~1/quarter.

**10.3.1 The channel is part of the identity of the act.** Email is long and
infrequent; Slack is short and constant; SMS is intimate and interruptive. The same
message costs differently by medium, which is why `channel` is in the primary key
rather than a modifier. Rough starting shape:

| channel | base cost | typical regen | norm it encodes |
|---|---|---|---|
| Slack / chat | 1 | high | cheap, frequent, short |
| Email | 3 | low | expensive, considered, long |
| SMS / phone | 5 | very low | reserved for things that matter |

A consequence worth stating: **the budget should shape the message, not only gate it.**
An agent with one email point and three Slack points should write one considered email
rather than four fragments — the medium's norm is part of what it is deciding.

**10.3.2 Replies are free.** Only *unsolicited* contact draws down. An agent that will
not answer you because it is out of points is a worse product than one that is
occasionally chatty; unresponsiveness is the failure people actually resent. Budget the
initiation, never the response.

**10.3.3 Reciprocity is what makes it self-correcting.** Points are spent by reaching
out and **replenished by the other person engaging back**:

- reach out, no reply → spent, nothing returned
- reach out, they reply → refund plus a bonus
- they initiate → large credit

The budget therefore learns the *real* cadence from behavior rather than from a
declared strength. Label someone a close friend who never replies and the system
throttles anyway — which is exactly what an engaged human does. Without this, a
mislabeled relationship stays mislabeled forever.

**10.3.4 Bootstrap from observed history, do not guess.** Hexis already ingests channel
history into `channel_source_items` and `connector_source_items`. Measure the *user's
own* cadence per person per channel and seed `regen_per_day` from it. If Eric messages
Sarah three times a day on Slack and Bob monthly by email, the agent inherits that
rhythm without anyone declaring anything. This is Experience Bar #1 applied to
etiquette: **the user is the reference implementation.**

**10.3.5 Price by intrusiveness.** `cost = base(channel) × time_of_day × ÷ urgency`. A
DM at 2am costs several points; the same message at 2pm costs one. Urgency must be able
to drive cost near zero — an agent that will not say your flight is cancelled because
it is out of points is broken. **The budget governs chatter, never signal.**

**10.3.6 Overdraft is a signal, not a wall.** Allow going negative for genuinely urgent
contact, record it as `strain`, and suppress non-urgent outreach until it is repaid.
You *can* call a friend at 3am; you owe them afterward. That is worth modelling because
it is true.

### 10.4 Assigned goals are a permission slip

**The idea.** Work in pursuit of a goal the user assigned should not cost what
self-directed work costs. An assigned goal is pre-authorization — the user already
said yes, so the agent should not have to re-purchase permission every beat.

**What it should discount, and what it must not.**

- **Energy: yes, discount heavily.** Elective action is what energy is for — it is the
  brake on self-directed drift. Work the user explicitly asked for should not compete
  against it. A 75–100% discount on goal-attributable actions is right, and it makes
  the economy legible: *energy is the price of acting on your own initiative.*
- **Purpose gate: yes, passes automatically.** A goal reference *is* a purpose.
- **Contact points: no — discount at most, never waive.** This is the one place I would
  push back on "essentially free." **The recipient's inbox does not care why you are
  writing.** You can pursue a perfectly legitimate goal and still exhaust someone by
  messaging them six times about it. Purpose legitimises *whether*; it does not pay for
  *how much*. A modest discount (say 50%) is defensible; zero is how the agent becomes
  the thing everyone mutes.

**What it needs first.** There is no reliable assigned-vs-self-generated flag today.
Goals are memories of `type='goal'` with free-form `metadata->>'origin'` — every goal
on this instance reads `origin: initialization`, `source: curiosity`. A `memory_origin`
enum exists in `db/07` (`user_request`, `identity`, `derived`, `external`) and goals do
not use it.

**Fix:** constrain goal `origin` to that vocabulary and write `user_request` when a goal
arrives from a user turn. Until that flag is trustworthy, the permission slip cannot be
implemented, because the system cannot tell whose idea something was. *~1 day, and it is
a prerequisite for everything in §10.4.*

**The resulting shape** is a clean sentence the agent can be told and a user can
understand:

> *Doing what you asked is cheap. Deciding for yourself costs energy. Spending someone
> else's attention costs contact points, no matter whose idea it was.*

### 10.5 Disclosure and the opt-out — non-negotiable on every third-party message

Every outbound message to anyone who is **not the primary user** carries an
identification and a way out:

```
— Samantha, Eric's Hexis AI. Reply STOP to excommunicate me.
```

**This is not politeness, it is a compliance surface.** Disclosure obligations for AI
systems interacting with people are arriving in real jurisdictions (the EU AI Act's
transparency provisions, California's bot-disclosure law), and `STOP` is the
established opt-out convention for SMS. Shipping autonomous outbound messaging without
both is a liability, not a rough edge. *Confirm the specifics with counsel before
shipping — this plan is not legal advice.*

**10.5.1 Name the principal, not just the software.** "a Hexis AI" tells the recipient
what is writing; it does not tell them *for whom*. The interesting question in a
stranger's head is always "who is this actually from," and answering it is both more
honest and more effective. Prefer `— Samantha, Eric's Hexis AI`.

**10.5.2 Enforce at tool dispatch, never in the prompt — and there are two outbound
paths, not one.** A model asked to remember a disclaimer will omit it under pressure on
turn 40, the same reasoning as provenance-by-default in `positioning.md` §4.0.a. But
*where* to enforce is the part an earlier draft of this section got wrong.

**The outbox is not the main road.** It is the formal, asynchronous, email-shaped
channel. **Most communication goes out through tool calls**, and those tools bypass
the channel layer entirely — `core/tools/messaging.py` posts straight to
`https://slack.com/api/chat.postMessage`, `https://discord.com/api/v10/channels/…`,
`https://api.telegram.org/bot…/sendMessage`, and a local Signal API. A footer appended
in `channels/base.py:send()` would cover the minority path and miss the majority.

The live catalog has thirteen outbound tools today:

```
slack_send  discord_send  telegram_send  signal_send
email_send  email_send_sendgrid  gmail_send  gmail_reply
twitter_x_dm_send  twitter_x_post  twitter_x_reply
queue_user_message  (+ connector actions, which grow)
```

**Do not patch thirteen call sites.** Patching each one guarantees the fourteenth
leaks, and connector actions are added continuously.

**The design: declare outbound-ness in the ToolSpec, enforce in the dispatcher.** Give
`ToolSpec` an optional `outbound` descriptor naming which argument carries the
recipient and which carries the message body:

```python
outbound=OutboundSpec(recipient_arg="channel_id", body_arg="message", channel="slack")
```

Then one middleware in the dispatch path — beside the energy check and the approval
callback that already run there (`core/agent_loop.py:548`) — does all of it in order:

1. **STOP gate** — recipient excommunicated? refuse, above everything else.
2. **Purpose gate** (§10.2) — is a purpose present and backed?
3. **Contact budget** (§10.3) — can this be afforded, at this hour, on this channel?
4. **Footer injection** — append the channel-appropriate disclosure to `body_arg`.

The same middleware wraps `channels/outbox.py` for the formal path, so both roads pass
the same four checks.

**Make omission impossible, not merely discouraged.** A startup assertion: every tool
whose handler performs network I/O to a messaging provider must declare `outbound`, or
the registry refuses to load. This is the same shape as the `allowed_actions` handler
assertion in §9.1 — the failure mode this whole plan keeps rediscovering is *capability
that exists with no enforcement path attached*, and an assertion is how you stop
rediscovering it.

**10.5.3 The channel decides the form.** A 160-character SMS segment is money; Slack
has small-text formatting; email has a signature convention.

| channel | form |
|---|---|
| SMS | `— Samantha (AI). Reply STOP to opt out.` — short, STOP literal, counts against the segment |
| Slack / Discord | small-text footer, full form |
| Email | signature block, full form, plus a one-line "why you received this" |

**10.5.4 STOP has to actually work — this is the part that matters.** A promised
opt-out that is not wired is worse than no disclaimer at all, because it converts a
minor annoyance into a broken promise.

- Inbound matching is case-insensitive and accepts the family: `STOP`, `UNSUBSCRIBE`,
  `OPT OUT`, `EXCOMMUNICATE`, and a bare `STOP.` with punctuation.
- The block is **immediate, permanent, and cross-channel** — keyed to the entity, not
  the address. Someone who says STOP on SMS is not to be emailed.
- It is a hard gate in the outbound path, above the purpose gate and the contact
  budget. No goal, no urgency, and no user instruction routes around it. **An assigned
  goal is not a permission slip through someone else's refusal.**
- Acknowledge once, then go silent: `Understood — I won't contact you again.`
- `START` / `UNSTOP` reverses it, because people mistype and circumstances change.
- Every STOP is recorded with timestamp, channel, and the message that triggered it.

**10.5.5 A STOP is news for the user, not just a flag.** If someone excommunicates the
agent, **the human's relationship is what took the damage**, and they need to know
immediately — who, which channel, and the message that caused it. The outbox is the
right road for this one: it is formal, asynchronous, and meant to be read rather than
glanced at. Not a row in a table nobody opens. It is also the single best signal
that the cadence model in §10.3 is miscalibrated, and should feed back into
`regen_per_day` for every comparable relationship.

**10.5.6 Full form on first contact, marker afterwards.** Repeating the whole STOP line
on every message reads as spam and trains people to ignore it. Full disclosure on first
contact with a person on a channel, then a short `— Samantha (AI)` marker, with the
full form re-shown on any new thread, after a long gap, and at a configurable interval.
The identification never disappears; only the instructions compress.

**10.5.7 Never to the primary user.** They configured the agent, signed its consent,
and know exactly what it is. A disclaimer on every message home would be absurd.

**Effort:** ~3 days for the `outbound` descriptor, the dispatch middleware, the STOP
gate, the footer, and the ledger entries — and it
**ships in the same change as §10.3**, never after. Autonomous outbound without a
working opt-out is not a feature to be added to later.

### 10.6 Non-negotiables

- **A kill switch.** Anything that autonomously messages other people needs one-click
  suspension, per-person and globally.
- **A ledger view.** Every outbound message, its purpose, its cost, and the budget it
  drew from — inspectable after the fact. Autonomous outreach the user cannot audit is
  not a feature, it is a liability.
- **Silence must be observable.** `consecutive_silent` already exists on ambient
  responsibilities; the same idea belongs here. An agent that has reached out four
  times with no reply should be visibly aware of it, not merely throttled by it.

**Effort:** ~1 day for the goal-origin flag, ~3 days for the ledger and purpose gate,
~2 days for reciprocity and the history bootstrap, ~3 days for the outbound
descriptor, disclosure and the STOP gate, ~1 day for the kill switch and ledger view.
**~10 days total**, and it should not
ship in halves — a purpose gate without a budget still floods, a budget without a kill
switch is not something to point at anyone's colleagues, and a disclaimer promising an
opt-out that is not wired is worse than sending nothing at all.

## 11. Mining Alex's fork

**Added 2026-08-21.** `~/hexis-alex` (`Lazarus-AI/hexis-pro`) is a private fork with
**140 services to this tree's 35, 530 `db/*.sql` to 98, and 79 tool modules to 52.**
Alex has agreed that anything outside his proprietary architecture may be merged into
the OSS tree.

### 11.1 The boundary

Off-limits is the RCR-derived architecture and its subsystems: **human model /
endpoint profile** (`sigma_model`, capacity C, operating posture), **allocentric
engine** (agent modelling, feedforward cancellation, residual), **validation and
outcome tracking** (decision-episode review, information-determined action),
**agency window detector** (K estimation, timing gate), **fragility monitor**
(rolling `R_eff`, `F(t)`, correlation collapse), and the **environmental channel
processor** (N-channel ingest, `R_eff` estimation).

Everything else is fair game, including borderline cases. Only the named subsystems
and their concepts are excluded.

`clearwing_*` is out too, for a different reason: `docs/clearwing_hexis_fork.md`
states that *"Hexis does not push ClearWing changes to open-source upstream."* It is a
separate Lazarus product with its own MIT-attribution boundary, unrelated to the
architecture above.

### 11.2 The mechanical test — run it on every candidate

The fence is not a directory. `tool_sigma_gate.py` shows `sigma_model` threaded into
tool gating itself, so a module that looks generic can still drag proprietary
subsystems across an import. Every candidate gets grepped before it is touched:

```bash
grep -nE 'sigma_model|sigma_axes|agency_window|allocentric|branchial|independence_engine|fragility|operator_model|prediction_journal|guardian_|R_eff|r_eff|k_scheduler|hyperspace' services/<candidate>.py
```

Clean → port. Hits → port the idea, strip the dependency, keep Alex's file as the
spec rather than the source.

### 11.3 The three buckets

Applying that test across all 105 fork-only services:

- **70 are permitted** — no reference to any excluded subsystem. **Permitted is not
  recommended**; see §11.4 for the nine worth taking and §11.7 for what to decline.
- **19 would need deps stripped** — port the idea, keep his file as the spec.
- **20 are the thing itself** — excluded by definition.

```
PORT AFTER STRIPPING: agent_acquisition_dispatcher calibration_digest co_design_loop
  code_cognition comms_salience constructor_controller deliberation
  deliberation_evidence_budget deliberation_runtime_budget epistemic_hygiene
  known_unknowns local_taxonomy memory_architect memory_architect_reviews
  off_band_context personal_hexis_ingest tool_channel_registry watchdog worker_identity

EXCLUDE: agency_window branchial_cohesion endpoint_allocentric
  evidence_channel_acquisition evidence_fragility external_signal_router guardian_*
  hyperspace_projection independence_engine(_shadow) k_scheduler operator_model
  prediction_journal sigma_axes sigma_model tool_sigma_gate  (+ all clearwing_*)
```

### 11.4 What is worth taking — permission is not a recommendation

**70 modules are permitted. I would take nine.** Not all of Alex's ideas belong here;
his fork serves a different product with a different mission. Each candidate below is
argued against `MISSION.md`'s six tests, and §11.7 lists what I would decline and why.

**Tier 1 — mechanism of the mind.** The mission's highest category: *"a subsystem that
looks redundant by engineering economy may be load-bearing psychology."*

1. **`retention.py` + `scene_consolidation.py` + `incubation.py`**
   *Person Test, dead centre.* The mission states it outright: *"People forget.
   Consolidation, compression, and fading are how a finite mind stays coherent.
   Retention is a feature, not a defect."* And it names **"sleeping on it"** and free
   association as conscious acts of memory Hexis should offer — `incubation.py` is
   spontaneous recall, the mechanism behind that phenomenon. This is the strongest
   alignment in the entire fork, and it closes `positioning.md` §4.6.

2. **`memory_supersessions.py`**
   *Substrate + Continuity.* Promotes belief-revision lineage off `memories.metadata`
   into a real side-table. *"People know things because of where they learned them"* —
   supersession is provenance extended through time. Unlocks `positioning.md` §4.3,
   where the bitemporal columns already exist and nothing writes them.

3. **`belief_propagation.py`**
   *Person Test.* When a belief changes, what rests on it should move too. That is how
   a mind works and it is absent here today. It is also the plumbing half of
   contradiction-as-an-event (`positioning.md` §4.2), whose detector currently produces
   nothing.

**Tier 2 — earning her keep.** The second north star.

4. **`operator_approval.py` + `approval_slack_actions.py`**
   *Dignity + Law 2 + Law 7.* The human keeps authority; approval becomes answerable
   from a phone rather than only a terminal — *"live where the user lives."* Closes the
   fail-open gate in §11.5, which is the most consequential defect in this plan.

5. **`operator_policy_corrections.py`**
   *Law 3, Compound.* *"The most valuable memory is the one that means you never have
   to say it twice."* A correction ledger is that law's implementation, and it is what
   `positioning.md` §4.5 needs.

6. **`inbound_disposition.py`**
   *Law 4, Earn the interruption.* Operator detection, trigger words, allowlists, drop
   rules — and all of the policy in PL/pgSQL, which satisfies the Substrate Test as
   written. Serves §10 from the inbound side.

7. **`voice_notes.py` + `local_audio_analysis.py`**
   *Law 5 + Law 2.* *"Be the someone worth talking to at 2am"* is hard to do in a text
   box. Closes §5.1 with work already done.

**Tier 3 — keeping ourselves honest.** Lower ceiling, but each answers a defect this
plan found by hand.

8. **`capability_probe.py` + `tool_surface_audit.py`**
   *Law 1 + Law 7.* You cannot *do* if the tools are unreachable, and Tier 0 shows they
   often are. Continuous measurement of what §0 found manually, plus visible state
   rather than hidden magic. **Port the idea, not the line count** — 791 lines is sized
   for his fleet; this tree needs a fraction of it.

9. **`deliberation.py`** *(strip deps first)*
   *Person Test.* `debate_internally` is a heartbeat action that is priced, offered,
   and **unimplemented** (§9.1). Internal dialectic is a real mental act, and this
   would give a dead action a body. Also the nearest thing to a reason for `run_council`
   to exist. Requires clean-rooming away from `independence_engine`, `prediction_journal`
   and `fragility`.

### 11.5 The gap this exists to close

`core/agent_loop.py:550` reads:

```python
if spec and spec.requires_approval and cfg.on_approval:
```

**When `on_approval` is `None` the check is skipped and the tool runs.**
`apps/cli_chat.py:461` is the only caller in this tree that supplies one — the
heartbeat (`services/heartbeat_agentic.py:88`) does not, and neither does the API chat
path (`services/chat.py:301`).

**51 of 150 tools are marked `requires_approval`** — including `slack_send`,
`telegram_send`, `email_send`, `gmail_send`, `gmail_delete`, `twitter_x_post`,
`shell`, and `write_file`. Every one of them executes unattended today with the flag
set and nothing reading it.

Two fixes, and both are wanted:

1. **Fail closed.** Absent a callback, an approval-required tool refuses and files a
   request rather than proceeding. *One line, today.*
2. **Give it a callback worth having** — port `operator_approval.py` +
   `approval_slack_actions.py` (§11.4·4, sequencing item 2): Slack → iMessage
   escalation with Block Kit approve/deny, so approval is answerable from a phone
   instead of only from a terminal.

This is the fifth instance in this plan of one pathology: **a mechanism that exists
with nothing enforcing it.** Dead heartbeat actions (§9.1), tools bound to no
reachable skill (Tier 0), outbound tools that would slip the gate (§10.5.2), a
cooldown config referenced by no code (§10), and now an approval flag nobody reads.
Every one of them should end with an assertion, not a comment.

### 11.6 What a port actually costs

Alex's tree is database-as-brain taken further than this one — most services are thin
async wrappers over PL/pgSQL that holds the real logic. `inbound_disposition` is 394
lines of Python over 584 lines of SQL; `belief_propagation` is 191 over 513;
`operator_approval` is 173 over 1,181.

**So a port is rarely a file copy.** It is a Python module, one or more `db/*.sql`
files, a migration to bring an existing database forward, and a check that the SQL
does not reference tables this tree lacks. Budget **1–3 days per subsystem**, not an
afternoon — and prefer taking few things properly over many things partially.

### 11.7 Permitted, and declined

Listed with reasons, because "we could" is not "we should."

**Conflicts with what Hexis is.** `recursor_dispatch`, `recursor_dispatcher`,
`recursor_ledger`, `agent_acquisition_dispatcher`, `constructor_controller` —
orchestration and throughput machinery. `MISSION.md`: *"**Not an agent-orchestration
framework.** Autonomy exists so the person can pursue their own goals and tend their
own life — not to maximize task throughput."* These are good code serving a different
thesis.

**Another product's surface.** `osint_daily_summary`, `linkedin_ingest`,
`gdelt_adapter`, `matter_os_bridge`, `personal_hexis_ingest`, `personal_hexis_render`,
`feed_generator`, `feed_slack_actions`, `code_cognition`, `ui_perception`,
`hexis_read_bridge`. Law 8 — every capability pays rent. These pay rent in Alex's
product, not in this one.

**Merges wearing acquisition's clothes.** `conversation.py`, `consent.py`,
`user_model.py`, `hmx.py`, `conscious_extraction.py`, `ingest.py`. This tree already
implements every one of these concepts, in `core/` or as its own package. Taking his
versions is a reconciliation of two divergent implementations, not a new capability —
higher risk, and only worth it for a specific defect his version fixes. **My earlier
draft listed these as "port freely," which was wrong.**

**Infrastructure without a named problem here.** `worker_identity`, `cluster_health`,
`connectivity`, `zombie_remediation`, `schema_loader`, `tooling`, `trigger_payload`,
`llm_catalog_refresh`. Possibly fine; none earns a slot on a recommendation list
without a defect in *this* tree that it closes.

**One genuine open question — self-authored skills.** `skill_synthesizer.py`,
`skill_synthesis_validator.py`, `constructed_tools.py` let the agent write its own
skills. This tree deliberately does not: `services/skill_improvement.py` *"never writes
skill files"*, only reviewable proposals — a Dignity Test decision about who holds
authority.

But `MISSION.md` Law 6 says her skills should reshape around the user *"including
authoring her own new skills from experience."* **The mission endorses the thing the
code declines to do.** That is a real contradiction, not something to resolve by
picking whichever source is nearer to hand. Worth settling deliberately — and if
self-authoring wins, Alex's validator is the piece that makes it survivable.

# Sequencing

| # | Item | Effort | Unblocks |
|---|------|--------|----------|
| −1 | **Approval gate fails closed (§11.5)** | **~1h** | **51 tools stop firing unattended** |
| 0 | **Tier 0 — reachability (§0.1–0.6)** | **~2d** | **turns on capability already built** |
| 0b | **Port** `capability_probe` + `tool_surface_audit` (§11.4·8) | ~3d | Tier 0 stops being a one-off audit |
| 1 | Wave A everyday skills (§4) | ~2d | visible value immediately, no new code |
| 2 | **Port** `operator_approval` + Slack actions (§11.4·4) | ~2d | approval answerable from a phone |
| 3 | Automation suggestions (§1) | ~2d | the agent starts proposing |
| 4 | `ask_user` (§2) | ~3d | the agent stops guessing |
| 5 | **Dead heartbeat actions (§9.1)** | **~0.5d** | **20% of the action space stops being a trap** |
| 5b | Build off the install path (§6.1) | ~0.5d | no more failed installs |
| 6 | **Port** `voice_notes` + `local_audio_analysis` (§11.4·7) | ~1d | *replaces* the §5.1 `transcribe` build |
| 7 | Tailscale/HTTPS path documented (§8.1) | ~1d | **unblocks the PWA** |
| 8 | **PWA layer on `hexis-ui` (§3a)** | **~3d** | installable client + push + mic |
| 9 | **Port** `retention` + `scene_consolidation` + `incubation` (§11.4·1) | ~4d | *replaces* the `positioning.md` §4.6 build |
| 10 | **Port** `memory_supersessions` (§11.4·2) | ~2d | unblocks `positioning.md` §4.3 bitemporal |
| 11 | **Port** `belief_propagation` (§11.4·3) | ~2d | plumbing half of `positioning.md` §4.2 |
| 12 | **Port** `operator_policy_corrections` (§11.4·5) | ~2d | *replaces* the `positioning.md` §4.5 build |
| 13 | Deterministic image build (§6.2) | ~2d | — |
| 14 | Goal origin flag (§10.4) | ~1d | prerequisite for the permission slip |
| 15 | **Port** `inbound_disposition` (§11.4·6) | ~2d | §10's inbound half, policy already in SQL |
| 16 | Contact points + purpose gate + STOP (§10) | ~10d | outbound to third parties becomes safe |
| 17 | Wave B skills (§4) | ~1d each | — |
| 18 | Action/tool gate reconciliation (§9.6) | ~2d | after Tier 0 |
| 19 | **Port** `deliberation`, deps stripped (§11.4·9) | ~4d | gives `debate_internally` a body |
| 20 | `hexis-node` + pairing (§3b) | ~1w | §4 Wave C, wake word |
| 21 | Heartbeat cadence + economy (§9.2–9.4) | ~1w | long-horizon autonomy |
| 22 | Workers as host services (§6.3) | ~1w | — |
| 23 | `hexis tunnel` + exposure posture (§8.2–3) | ~1w | — |
| 24 | Voice out / talk / wake (§5.2–4) | ~3w | — |
| 25 | Execution backends (§7) | ~2w | — |

Tier 0 comes before all of it: shipping new skills (item 1) into a selector that
will not activate them is building on sand.

Items 7–8 are deliberately adjacent: the HTTPS day is worthless on its own and the
PWA is impossible without it, so they ship as one change.

**Nine of these are ports, not builds** (§11.4), and four of them *replace* work this
plan had costed as new: `voice_notes` for §5.1, the retention trio for
`positioning.md` §4.6, `operator_policy_corrections` for §4.5, and `capability_probe`
for Tier 0's instrumentation. Ports are cheaper than builds but not free — each is a
Python module plus SQL plus a migration (§11.6), so they are costed at 1–4 days, not
at zero.

Items 0–8 are about four weeks and cover the gap that actually matters: an assistant
that proposes work, asks when unsure, speaks the languages you already use, and
installs without a build. Everything after is depth.

# Definition of done

Not "the tests pass." Per `HEXIS_EXPERIENCE_BAR.md` #7, drive the real path:

- A fresh install on a clean machine reaches a **beating heartbeat** with one command
  and no build.
- The ten-request probe in Tier 0 is a **regression test**: each request activates a
  skill that can actually serve it, and no request lands on `core-memory` alone.
- Every action in `heartbeat.allowed_actions` has a handler, asserted at startup — and
  the action distribution over a hundred beats is logged and reviewed, so a dead or
  never-chosen action is visible rather than inferred.
- Over a month with Slack and email connected, **nobody the agent contacted asks it to
  stop** — and every message it sent can be shown with its purpose, its cost, and the
  relationship budget it came from.
- Every third-party message identifies the agent and its principal, and a reply of
  `STOP` on any channel silences it everywhere, permanently, within one message — with
  the user told who and why.
- Within the first day, the agent **proposes** at least one automation the user did
  not ask for, and the user can accept it in one click.
- The agent **asks** a clarifying question mid-task at least once, and the answer
  changes what it does.
- A voice memo sent from a phone gets a useful reply.
- The agent is **installed on a phone** as an app, and a proposal it makes arrives as
  a push notification rather than waiting in a web inbox nobody opens.
- Every one of those is refusable, and refusing it once means never being asked again.
