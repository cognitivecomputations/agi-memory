# The Plan — why Hexis needs what it needs

**Read `ROADMAP.md` first.** It is the ordered list of what to do next; this document
is the reasoning and evidence behind every entry in it.

- **Part I — Strategy** (§S1–S7): what Hexis should be best in the world at, and the
  race not to enter. Changes rarely.
- **Part II — The gaps** (Tier 0, §1–§14): what is missing or broken, measured
  against the live system, and what to do about each.

The July 2026 engineering audit that seeded some of Part II is archived at
`docs/_archive/audit-2026-07-29.md`, reconciled 2026-08-21.

---

# Part I — Strategy

## S1. The race we cannot win

OpenClaw ships 52 skills, 158 extensions, 22 messaging channels, native apps for five
platforms, and lists OpenAI, GitHub, NVIDIA and Vercel as sponsors. Hermes has Nous
Research behind it, six execution backends, and serverless hibernation.

**Hexis will never have the most integrations.** Every week spent closing that gap is
a week spent losing a race against better-funded teams, on an axis where being second
is worth nothing.

Part II is still worth executing — an assistant that cannot be reached or
cannot ask a question is not a product. But it is table stakes, not a strategy. It
gets Hexis to parity. It does not make anything best in the world.

## S2. The race nobody else can enter

Every competitor keeps its mind in files. Hermes: `~/.hermes/cron/suggestions.json`,
atomic writes, an in-process lock. OpenClaw: a workspace directory. Hexis put the mind
in Postgres — with sources, trust levels, bitemporal validity, a contradiction
detector, and an AGE graph.

That has been treated as an implementation detail. **It is the product.**

> **The only assistant that can prove what it knows, show you where it learned it,
> tell you why it acted — and hand you the whole mind as a file you own.**

Memory you own. Reasoning you can audit. An agent that can refuse. A competitor with
a JSON file cannot follow us there without a rewrite, and neither can a frontier lab
whose memory feature is a vendor-held blob you cannot read, export, or interrogate.

This is also the only axis where Hexis wins **today**, before any of the work below.

## S3. What is already in the ground

Not aspirations — columns and functions that exist in the running database:

| Capability | Where it already lives |
|---|---|
| Per-claim provenance | `memories.source_attribution` (jsonb), `memories.trust_level` |
| Provenance graph | `memory_source_units` (memory ↔ subconscious unit, with roles `source`, `direct_promotion`, `extraction`, `corroboration`) |
| **Bitemporal memory** | `memories.valid_from`, `valid_until`, `superseded_by` |
| Contradiction detection | `find_contradictions(p_memory_id uuid)`, called from `core/cognitive_memory_api.py` and `db/17` |
| Belief evolution | `belief_history` (DB-native, `db/38`) |
| Causal trace | `trace_why` (`core/tools/memory.py`) |
| Citable passages | `source_document_chunks` with page/section/sheet locators |
| Mind portability | HMX export/import, `memory-exchange` skill, `plans/hmx.md` |
| Deliberate forgetting | `decay_rate`, `fidelity`, fade requests, `docs/memory_retention_design.md` |
| Deliberation | `run_council`, `list_council_personas` |
| Refusal | `agent.consent_status`, boundaries-as-memories, self-termination |

Most of this has never been shown to a user. `belief_history` and `trace_why` sit in
the always-on tool floor (per the Tier 0 probe) and are almost never
called, because nothing in the product asks for them. **This is a surfacing problem,
not a building problem** — which is why the two flagship ideas below are weeks, not
quarters.

---

## S4. The invocation problem — the thing that actually blocks all of this

*(§S4.0.x below; the flagships themselves are §S4.1–4.7.)*

**Revised 2026-08-21 after a second probe.** An earlier draft of §S4 described these
capabilities as if a user would reach for them. **A user will never type "run the
council" or "trace why."** These are not features anyone asks for by name. If the
agent does not invoke them on its own, they do not exist.

Tier 0 found that skill activation is driven by *the user's
vocabulary*. That is the wrong axis for everything in this document. Nothing in
"should I take this deal?" mentions a council, and **nothing a user ever says
mentions a contradiction** — the trigger is not in their words at all. It is in the
agent's own state.

So the question for each capability is not "is it reachable?" but **"what makes the
agent decide to use it?"** There are four honest answers, and they need different
machinery.

### S4.0.a Structural — not a decision at all

Some things must never be left to the agent's judgment, because an agent that has to
*choose* to show its sources will not, reliably, under pressure, on turn 40.
**Provenance is one of these.** Make it a property of the data path: `recall` always
returns `source_attribution` and `trust_level`, the prompt always requires citation,
the renderer always draws the footnote. Zero decisions, zero failure modes.

### S4.0.b Observed state — the heartbeat's Observe phase

**Corrected 2026-08-21.** An earlier revision of this section claimed the heartbeat's
decision prompt contains nothing about the agent's own cognitive state, and cited
"271 open contradictions the agent has never been shown." **Both were wrong**, and
the correction matters because it changes what to build.

What is actually true, measured against the live instance:

- `gather_turn_context()` already assembles `contradictions`, `contradictions_count`,
  `memories_at_threshold`, `urgent_drives`, `transformations_ready`, `emotional_state`,
  `self_model`, `user_model`, `relationships`, `narrative` and a graph `subgraph`.
- `render_heartbeat_decision_prompt()` already renders `## Contradictions`,
  `## Transformations Ready`, `## Urgent Drives` and `## Memories at Threshold`. My
  earlier grep read only the first third of a 98-line function.
- The "271" was an artifact: those rows carry `metadata->'contradictions'` as JSON
  `null` — a placeholder key, not a finding. The real `contradictions_count` is **0**,
  and the prompt correctly renders `(none)`.

**The Observe packet is not the gap. It is one of the best-built parts of the
system.** Contradiction-as-an-event (§S4.2) therefore does not need new plumbing into
the heartbeat — it needs contradictions to actually be *detected and written* (the
detector exists and this instance has produced none), and it needs the resolution to
be surfaced to the user rather than resolved silently.

**What is genuinely missing on this axis** is narrower and still worth doing:

1. **Detection has to run.** `find_contradictions()` exists and zero contradictions
   are recorded on an instance with 323 active memories and 34 worldview beliefs.
   Either the detector never fires on the ingest path, or its threshold is set so it
   never triggers. Find out which. *This is the real §S4.2 blocker.*
2. **Trust is nearly a constant.** 252 of 323 memories share the identical
   `trust_level` of `0.4302279608697066` — a computed default, not a judgment. A
   provenance UI (§S4.1) that renders trust is worthless while trust does not vary.
3. **The heartbeat's skill selection is fed a JSON dump.** `services/agent.py:825`
   builds the skill-selection query by `json.dumps(heartbeat_context)[:4000]` and runs
   the same lexical matcher over it. Whatever skills that activates, it is not a
   considered choice — and it means a chosen action like `inquire_deep` can find its
   tools ungated by accident or gated by accident. See §S4.0.c and Tier 0.

*Effort: unchanged at ~2 days, but spent on detection and trust variance rather than
on plumbing that already exists.*

### S4.0.c Situational recognition — cues in the prompt, not words in the query

For capabilities where the trigger is a *kind of moment* rather than a state — the
council on a consequential decision, point-in-time recall on a temporal question —
the answer is an instruction, not a matcher: *"when the user faces a consequential
decision with real tradeoffs, convene the council before answering."*

Measured, `council` scores **1, 4, 1** against "should I take this deal or walk
away?", "help me think through a hard decision," and "weigh the tradeoffs on hiring."
The threshold is 5. It never fires on the decisions it exists for. Lexical matching
cannot fix this; a semantic matcher (Tier 0 §S0.5) helps; an explicit prompt cue plus
reachability is what actually does it.

### S4.0.d Ambient responsibilities — already built, zero rows

`ambient_responsibilities` is a complete standing-orders engine: `trigger`,
`evaluator`, `sources`, `actions`, `delivery`, cooldowns, `consecutive_silent`
back-off, and a run-audit table. It is oriented at the *outside* world — watch Gmail,
watch a threshold, notice a missed check-in.

It has **zero rows**. Built, wired, never populated. It is the natural home for
§1's accepted automations, and it needs seeds far more than it needs
code.

---

## S4.1–S4.7 The flagships

Each of these is now labelled with **how it gets invoked** — because a capability with
no invocation path is not a capability.


### S4.1 Provenance by default — the assistant that never asks you to take its word

**Invocation: structural (§S4.0.a).** Never a decision the agent makes.

**The pitch.** Every factual claim in a reply carries a footnote to the memory,
document and chunk it came from — with trust level, page locator, and a click-through
to the source. Not on request. By default.

**Why only Hexis.** ChatGPT and Claude have memory now, but it is a vendor-held blob:
you cannot ask where a belief came from, because the system does not know. Hermes and
OpenClaw persist conversations, not *sourced beliefs*. Hexis records
`source_attribution` and derives `trust_level` from it at write time, and
`source_document_chunks` carries page and section locators. The data is there on
every row.

**Build.**

1. **Recall returns provenance.** `recall`/`search_documents` already return the rows;
   include `source_attribution`, `trust_level`, and the chunk locator in the tool
   result rather than dropping them.
2. **The model is instructed to cite.** A prompt-module change: any claim drawn from
   memory carries `[^id]`. Cheap, and it is the whole behavioral shift.
3. **The UI renders footnotes.** `MessagePresentationView` in `hexis-ui/app/chat/` is
   already a block renderer — add a `citation` block that expands to the memory or the
   document page. The attachment-card work from 2026-08-20 is the pattern.
4. **Low trust is visible.** A claim resting on `trust_level < 0.5` renders differently
   and says so. An assistant that flags its own weak ground is more useful than one
   that sounds equally confident about everything.

**Shipped 2026-08-28.** Memory and document tools now carry DB-generated citation
envelopes through every channel presentation. The conversation contract requires
exact citation IDs, the renderer produces linked expandable footnotes with locators
and a live low-trust warning, and the memory/document pages open the cited record.
Source normalization preserves complete locator-bearing provenance and derives
kind-specific trust from `memory.source_trust_defaults`; the legacy default plateau
was conservatively migrated into varied document and inference trust rather than
presented as meaningful precision.

**Effort:** ~1 week. **Demo value:** the highest in this document. Thirty seconds,
unanswerable by any competitor.

### S4.2 Contradiction as an event — the memory that gets *more* accurate

**Invocation: observed state (§S4.0.b).** The heartbeat is told how many contradictions
are open and picks `resolve_contradiction` on its own budget. The user is never the
trigger — they are the tie-breaker the agent comes to.

**The pitch.** When something new collides with something stored, the agent comes to
you: *"In June you said the Manning retainer was monthly. This contract says
quarterly. Which is right?"*

**Why this matters more than it sounds.** Every other memory system is append-only.
Stale beliefs accumulate silently and the assistant gets **worse** the longer you use
it — confidently wrong about things that changed a year ago. An assistant whose
accuracy *increases* with tenure is a categorically different product, and it is the
single strongest argument for a long-lived agent over a fresh chat.

**Why only Hexis.** `find_contradictions()` already exists and already runs — from
`core/cognitive_memory_api.py` and the subconscious observation path in `db/17`.
`resolve_contradiction` and `accept_tension` are already heartbeat actions with energy
costs. The detection is built. **What is missing is that nobody is ever told.**

**Build.**

1. **Route detections to a decision.** When `find_contradictions()` fires above a
   confidence threshold, file it — through the same propose-and-decide surface as
   automation suggestions (§1). Three outcomes: the new one is right,
   the old one is right, or both hold in different contexts (`accept_tension`
   already models this).
2. **Resolution writes bitemporally.** Do not delete the loser. Set `valid_until` and
   `superseded_by` on the old belief — the columns exist. The history stays queryable,
   which is what makes §S4.3 free.
3. **Batch it in the heartbeat.** A daily pass rather than an interrupt. Contradictions
   are rarely urgent, and an assistant that interrupts you about bookkeeping is worse
   than one that saves it for a briefing.
4. **Show the ledger.** A view in the dashboard: contradictions found, resolved, and
   accepted-as-tension. This is the proof that the thing is working.

**Shipped 2026-08-28.** New semantic/worldview memories enter a durable daily queue
and are compared only with DB-selected same-topic candidates. Confidence-gated cases
flow into one inert ledger from both the batch detector and the subconscious path.
The daily digest, Conversation inbox, verified private-channel code replies, and
Contradictions page expose three operator choices. Winner decisions use
`record_supersession()` to close the loser's validity window without deleting it;
contextual tension retains both. Heartbeats can surface a case but cannot choose.

**Effort:** ~1 week, most of it surfacing. **Strategic value:** the highest here.

### S4.3 "What did you know, and when?"

**Invocation: situational (§S4.0.c).** A prompt cue on temporally-framed questions —
"as of", "back then", "has that changed" — not a tool the user names.

`memories` already has `valid_from`, `valid_until` and `superseded_by`. The schema is
bitemporal and nothing exposes it.

*"As of last Tuesday, what did you think about the Manning deal?"* — a point-in-time
recall tool, and a diff view (*what changed about X between June and now, and why*).

An agent that can answer this is doing something no file-backed competitor can attempt.
It is also the natural payoff of §S4.2: every resolved contradiction deepens the record
instead of overwriting it.

**Shipped 2026-08-28.** Temporal language in conversation now cues two DB-owned,
agent-facing tools: `recall_at_time` reconstructs the validity, confidence, trust,
and citable provenance that existed at one instant; `diff_memory_history` compares
two instants and joins additions/expirations to supersession, evidence-revision,
and contradiction-decision reasons. Reverted supersessions and inactive legacy or
imported rows preserve only their real historical windows, group contexts retain
the private-memory wall, and embedding failure degrades visibly to lexical recall.
The Memory history dashboard exposes the same snapshot and comparison journey
without requiring tool names, including direct links to the underlying memories.

**Effort:** ~3 days, and it is nearly free once §S4.2 writes `valid_until` correctly.

### S4.4 Your mind is a file

HMX export/import already works. It is buried in a skill called `memory-exchange`.

In a market where people are genuinely afraid a vendor will delete their AI
relationship — and where every frontier lab's memory is a blob you can neither read
nor move — **"you can take her with you"** is not a feature bullet. It is the
headline.

Make it a first-class flow: `hexis backup` already exists; add `hexis export --mind`,
document the format, and demonstrate a mind moving from one machine to another and
waking up continuous. Publish the schema. Invite other projects to import it.

**Shipped 2026-08-28.** `hexis export --mind` now produces a complete private HMX
port file under the user's Hexis home, with protected state, private memories,
in-flight work, and audit history included and partial/redacted presets rejected.
`hexis import FILE --mind` retains dry-run and exact-intent confirmation, accepts
only the empty-target additive journey, and refuses silent replacement. After the
transaction it proves source-lineage continuity plus semantic equality across all
six constitutional sections while retaining exact transport digests for audit.
The public HMX 1.7 schema, reference, CLI contract, and end-to-end transfer guide
make the format inspectable and independently implementable.

**Effort:** ~3 days of packaging on top of what ships. **Marketing value:**
disproportionate.

### S4.5 Learning with a diff

**Invocation: observed state (§S4.0.b) on a weekly cadence**, delivered through the
outbox. The agent decides there is enough to review; the user only ever responds.

Hermes's headline is "self-improving" — it writes skill files autonomously.
`services/skill_improvement.py` deliberately does not, and that restraint should
become the feature rather than the limitation.

A weekly ritual: **"here is what I learned about you and your work this week —
approve, correct, or forget."** New semantic beliefs, new procedures, revised
strategies, proposed skills, all in one reviewable list.

This is the compounding loop Hermes advertises, with the trust property they cannot
offer: you saw every change before it took. It also feeds §S4.2 — a correction here is
a contradiction resolution.

**Shipped 2026-08-28.** The existing opt-in, seven-day cross-session reviewer now
lets the model decide whether enough meaningful change exists, but it may select
only database-supplied memory IDs. One outbox digest and Learning review page show
truth-derived semantic beliefs, procedures, strategies, and skill proposals with
their evidence. Every item accepts approve, correct, or forget in place or through
an exact verified-operator channel reply. Semantic corrections create a new
bitemporal version and resolve through the contradiction ledger; protected
forgetting requires a second confirmation. Approved skills enter the existing
ownership-checked authoring path and retain visible retry state on failure.

**Effort:** ~4 days on top of the existing background review.

### S4.6 Forgetting well

**Invocation: observed state (§S4.0.b).** Memory pressure and decayed fidelity appear
in the Observe packet; `maintain` is already a costed heartbeat action.

`decay_rate`, `fidelity`, fade requests and `docs/memory_retention_design.md` already
describe a compression-native substrate. Competitors have append-only logs that get
slower and dumber with every turn.

Position deliberate forgetting as a feature, not a limitation: the agent proposes what
to let go, asks before dropping anything that looks load-bearing, and reports what it
compressed. *"I remember what matters and I can tell you what I let go"* is a stronger
claim than *"I remember everything,"* and it is the honest description of how memory
actually has to work at scale.

**Shipped 2026-08-28.** Memory pressure, candidate groups, low-fidelity
reconstructions, pending load-bearing choices, and recoverable archives now share
one database-owned Observe/UI packet and a first-class Forgetting page. A bounded
outbox digest asks for keep, journal, or release; exact verified-operator channel
replies and the dashboard complete the choice in place. Pending reviews never
expire into a decision, and a database trigger prevents legacy heartbeat actions
from releasing or journaling them. Completed gists emit receipts from the actual
stored source IDs, source count, summary, and fidelity. Full source memories archive
recoverably by default; irreversible grace-window/capacity pruning is a separate
explicit opt-in and remains off. Migration 0236 upgrades live brains without data
loss, while CLI status/dry-run and the public guide expose the whole journey.

**Effort:** mostly already built; ~3 days to surface.

### S4.7 Define the measure

`evals/` exists. Publish a **memory benchmark** — provenance accuracy, contradiction
detection, recall at six months, cross-session continuity, resistance to stale
beliefs — run every agent on it, and publish the results *including where Hexis
loses*.

**Shipped 2026-08-28.** The public v1 corpus contains 25 timestamped synthetic cases,
five for each named axis, with version-pinned SHA-256, JSON schemas, strict validation,
judge-free scoring, rollback-isolated Hexis execution, and a gold-free command adapter
for any other agent. Every registered local system was run: the official one-shot
Hexis run scored 96.33, an append-only transcript baseline 82.33, and a recent 30-day
window 32.00. The published case-level result retains Hexis's two misses—a dropped
corroborating citation and a detector-to-answer handoff that omitted one side of an
allergen conflict. No unrun
third-party product is assigned a number. A higher 98.33 development pilot is
disclosed as classifier variance, not selected as the headline score.

Whoever defines the benchmark shapes what "best" means. This is the one axis where
Hexis wins today, and the cheapest credibility available. Losing honestly on two axes
makes winning on four believable.

**Effort:** ~1 week for a first public version.

---

## S5. What not to do

- **Do not chase channel count.** OpenClaw will always have more. Eight channels that
  work beat twenty-two that mostly do.
- **Do not chase skill count.** Twenty skills people use daily beat fifty-two nobody
  activates — especially given the Tier 0 finding that Hexis cannot reliably activate
  the twenty-five it has.
- **Do not build a canvas, a TUI, or native apps yet.** The PWA (§3a) covers the client for a fraction of the cost.
- **Do not let the council rot.** `run_council` is real deliberation machinery, bound
  to a `council` skill that is chat-loadable — and which scores **1, 4, 1** against
  "should I take this deal or walk away?", "help me think through a hard decision",
  and "weigh the tradeoffs on hiring". The threshold is 5. It never activates on the
  decisions it exists for; the user would have to say the word "council". Either fix
  the selection (Tier 0, §S0.3/§S0.5) or delete it. Machinery that never runs is not a
  differentiator; it is maintenance debt wearing a differentiator's clothes.

## S6. If we pick two

**§S4.1 provenance-by-default and §S4.2 contradiction-as-an-event.** Two weeks
together, mostly surfacing work over machinery already in the database, and no
competitor can answer either without rebuilding on a real store.

Together they make one claim that fits on a homepage and survives a live demo:

> **She shows her work, and she gets more right over time, not less.**

Sequence them after Tier 0 — provenance rendered from tools the
selector will not activate is provenance nobody sees.

## S7. The order

1. **Tier 0** — reachability (~2d). Everything else is built on it.
2. **§S4.0.b make detection actually fire** (~2d) — contradictions and trust variance.
   The Observe packet already carries them; today it truthfully reports zero because
   nothing writes any. Without this, §S4.2 has nothing to surface.
3. **§S4.1 provenance by default** (~1w) — the demo.
4. **§S4.2 contradiction as an event** (~1w) — the thesis.
5. **Tier 1** — suggestions and `ask_user` (~1w).
6. **§S4.3 point-in-time** (~3d) — nearly free after §S4.2.
7. **§S4.7 the benchmark** (~1w) — publish and let it be argued with.
8. **§S4.4 mind portability** (~3d) — packaging and a demo video.

About six weeks to a product with a defensible claim, from a codebase that already
contains most of the parts.


---

# Part II — The gaps

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

**Shipped 2026-08-28.** `hexis node` now owns the complete headless journey:
mode-0600 Ed25519 device identity, exact-fingerprint pairing in the shared inbox or
CLI, an outward-only WebSocket with single-session ownership and signed durable
results, irreversible revocation, a locally pinned direct-argv command allowlist,
and bounded `system.run` plus `screen.capture`. Every agent invocation still passes
through the normal approval gate. Captures enter the DB-owned turn as model-visible
image context without leaking their base64 bytes into ordinary tool text. The CLI,
FastAPI surface, host-node skill, operational runbook, live migration, and a real
challenge → approval → connected smoke journey ship together. `audio.*` remains
deliberately deferred to the wake-word work in §5.4.

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
  **Shipped 2026-08-24:** six chat workflows cover those five journeys plus
  mounted Obsidian/Bear Markdown notes. Each keeps writes, sends, deletes, and
  external task creation behind an explicit user choice and degrades to core
  Hexis tools when optional plugins are absent.
- **Wave B — API-backed tools.** Notion, Spotify, Home Assistant, weather, Trello.
  Each is a `ToolHandler` in `core/tools/` plus a connector-setup flow of the kind
  `services/connector_setup.py` already runs for Gmail.
  **Shipped 2026-08-28:** all five use DB-owned capability manifests and one
  guided setup surface across chat, CLI, MCP, and the Connections page. Notion,
  Home Assistant, and Trello persist only user-selected environment-variable
  names; Spotify uses Authorization Code + PKCE with private mode-0600 token
  storage; Open-Meteo weather needs no key. Catalog/state reads are bounded and
  parallel-safe, while every provider-state change remains approval-gated and
  mapped into connector action authorization. Migration 0227 preserves existing
  data and seeds the same manifests into live and fresh databases.
- **Wave C — needs the node (§3b).** Apple Reminders/Notes/Calendar via `osascript`,
  1Password CLI, Shortcuts, screenshots. Filesystem-backed ones (Obsidian, Bear) work
  today if the vault is mounted — do those in Wave A.
  **Shipped 2026-08-28:** six approval-gated tools now cover fixed-script Apple
  Reminders, Notes, and Calendar operations, exact-name Shortcuts, screenshots,
  allowlisted commands, and secret-safe 1Password. Capabilities are derived from
  executables on the node and any addition re-enters exact pairing approval.
  1Password listing is metadata-only; field copy stays on the Mac clipboard and
  sends only a receipt through the gateway. Migration 0237, the host-node skill,
  operator runbook, injection/secret regression tests, and compiled JXA programs
  ship with the runtime.

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
   for the shape this should take. **Shipped 2026-08-28:** the chat-only tool,
   PWA/API audio path, and optional Piper HTTP sidecar share one bounded local-only
   synthesis service. Setup derives its model from live DB configuration, records
   exact mode-0600 process ownership, refuses remote/credential-bearing endpoints,
   and never adopts or stops an ambient provider. Audits contain metadata only;
   tool audio is opaque, uncached, and expiring.
3. **Talk mode** — continuous listen/respond. Foreground-only in the PWA (§3a),
   which is enough for a conversation you started; always-on needs the node (§3b).
   **Shipped 2026-08-28:** explicit per-session start/stop, real provider gating,
   foreground visibility shutdown, calibrated voice activity/silence segmentation,
   bounded utterances, mic release during thought/playback, manual send, written
   transcript retention, and in-place pause/recovery.
4. **Wake word** — needs the node, always-on, last, and only once 1–3 are solid.
   **Shipped 2026-08-28:** explicitly configured openWakeWord/custom-model
   detection stays on a paired outward-only node; the post-cue utterance is WAV/
   size/hash/signature bounded and reuses the selected STT, canonical chat, and
   local TTS paths. `audio.wake` capability additions require a fresh pairing
   approval, the server gate defaults off, microphone capture closes during
   thought/playback, pretrained model licensing is explicit, and the append-only
   audit contains only counts/outcomes—not audio or conversation text.

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
   **Shipped 2026-08-28:** `hexis service` now installs, inspects, controls,
   logs, and removes per-user launchd/systemd units using the current uv-owned
   Python. Units retain only an explicit env-file reference and optional instance,
   never copied secrets. Docker-to-host migration is explicit and fail-safe; stack,
   init, reset, upgrade, UI, and uninstall paths derive one worker owner and avoid
   duplicate containers. Linux lingering remains an explicit user choice.

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

**Shipped 2026-08-28.** `shell`, `safe_shell`, `run_script`, and `execute_code`
now resolve one live database-owned profile without changing their public input
schemas. Local remains the default; SSH pins an exact identity and known-hosts
file, enforces timeout on the target process group, and maps workspace-relative
scripts without copying them. Remote Docker accepts only SSH transport, creates
one labeled ephemeral container per call, never pulls implicitly, defaults to no
network, removes only its exact owned container on timeout, and hibernates REPL
state in a named volume. Read-only status, explicit test/select/remove journeys,
bounded output/state retention, and fail-closed no-local-fallback behavior make
placement visible and explicitly selectable. Modal and Daytona remain demand-gated as planned.

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

**Shipped 2026-08-28.** `hexis tunnel` now derives the dashboard port and live
Tailscale identity, starts the local stack before making any network change, and
owns only the exact tailnet-only root Serve handler it created. It refuses public
binds, Funnel, unrelated handlers, stale ownership, and unreadable provider state;
`stop` preserves ambient routes and local data. `hexis doctor` independently fails
on public exposure even when PostgreSQL is unavailable. The PWA boundary is explicit
tailnet device approval, while command-capable companion nodes retain their separate
signed fingerprint approval and revocation flow. The HTTPS runbook covers the whole
phone install, push, microphone, verification, and recovery journey.

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

### 9.1 Three offered actions had no implementation · *fixed 2026-08-23*

> **Corrected 2026-08-23. This section previously claimed seven, and "20% of the
> action space is a trap".** Both were wrong, and the way they were wrong is the
> useful part: the count came from a regex over `WHEN 'x'` in
> `execute_heartbeat_action`, but the handler shares **multi-literal** branches —
> `WHEN 'contemplate', 'meditate', 'study', 'debate_internally'` and
> `WHEN 'inquire_shallow', 'inquire_deep'` — so the regex saw only the first
> literal of each and reported four implemented actions as dead.
>
> Executing each action instead of reading the source gives the real answer:
> **`fast_ingest`, `hybrid_ingest` and `slow_ingest`** returned
> `{"success": false, "error": "Unknown action: …"}`. The four cognitive actions all
> ran.

`heartbeat.allowed_actions` offers each entry with a configured energy cost, so an
unimplemented one is indistinguishable from a real capability until it is chosen —
at which point the beat's entire decision call has been spent picking something
impossible. It fails loudly and charges no energy, which is right, but the turn is
gone.

**Resolution: retired rather than implemented** (`db/migrations/0202`). The three were
also redundant — tools of the same name are bound to the `knowledge-ingest` skill,
which loads in heartbeat context, so the agent could already ingest by calling a
tool. One way to do a thing, and it works.

**The assertion is a test that executes every offered action**
(`tests/db/test_heartbeat_actions_implemented.py`), not a source scan — because a
source scan is exactly what produced the wrong count. A handler that dislikes the
probe's parameters still counts as a handler; only `Unknown action` fails.

### 9.2 Energy saturates and the surplus is destroyed · *fixed 2026-08-28*

`base_regeneration = 10`, `max_energy = 20`, interval 60 minutes. **Energy is full
after two hours.** An agent idle overnight wakes at 20, exactly as if it had rested
since 2am — ten hours of regeneration discarded.

The cost is not waste, it is *expressiveness*: nothing costing more than 20 can exist,
so **no ambition spanning more than one beat is representable.** The most expensive
thing the agent can conceive of is `inquire_deep` twice. An economy shaped like this
can only express errands.

**Resolution:** energy now regenerates by actual elapsed time into an auditable bank,
not by one fixed increment per worker tick. The default bank is three normal reserves
(`heartbeat.energy_bank_multiplier = 3`); energy above the normal reserve decays with
a configurable 12-hour half-life, so an idle night matters without creating an
unbounded stockpile. Upgrade seeding derives `last_regenerated_at` from the live last
heartbeat timestamp, preventing a migration-time windfall.

The agentic planner may spend at most one normal reserve on an ordinary beat and up
to two when the live context contains actionable backlog, always bounded by the
actual bank. Agentic tool receipts are charged exactly once at finalization; the
legacy path keeps its existing per-action deductions. CLI, TUI, API, and dashboard
surfaces derive both reserve and capacity from `heartbeat_economy_status()` rather
than assuming that `max_energy` is still a hard cap. Implemented in migration 0225
with baseline parity and rollback-safe coverage in
`tests/db/test_heartbeat_economy.py`.

### 9.3 Regeneration is time-based, so nothing rewards usefulness · *fixed 2026-08-28*

+10/hr whether the last beat resolved a contradiction or picked `observe` and went back
to sleep. The budget constrains *volume* and never steers toward *value* — and `rest`
(cost 0) competes against thirty-four ways to look busy.

**Resolution:** every beat now has a durable outcome ledger. Exact tool receipts and
legacy action results record bounded, idempotent signals for durable memories,
contradictions resolved, goals advanced, tool success/failure, and proactive contact.
The value score is deliberately narrow: up to two durable memories add 0.35 each,
two contradiction resolutions add 0.5 each, and two completed goals add 0.3 each.
Merely sending a proactive message earns nothing.

The next regeneration multiplier is `0.75 + outcome_score * 0.5`, capped at 1.5 by
default. Explicit appreciation can add 0.5 only when it comes from the verified
operator within 24 hours of a completed proactive beat, and only once for that beat;
ambient channel identity and generic positive language cannot mint reward. This
makes usefulness a gradient while keeping all credit traceable to stored evidence.

### 9.4 Fixed cadence ignores state it already computes · *fixed 2026-08-28*

Every beat costs the same LLM call at 3am with nothing pending as at 9am after forty
unread messages land. `urgency_ratio` and `urgent_drives` are **already assembled into
the heartbeat context** and are not consulted when scheduling the next beat.

**Resolution:** finalization now derives `next_heartbeat_at` from the existing maximum
drive urgency ratio. With the default 60-minute base, quiet state stretches toward a
90-minute idle cadence, urgency shortens it, and configurable 15/120-minute bounds
prevent thrashing or starvation. `should_run_heartbeat()` honors that persisted due
time, while interval `0` retains the explicit always-due development behavior.

Migration 0225 was exercised on the live database with a rollback-only journey: two
elapsed hours banked energy from 20 to 40 under a capacity of 60, a durable outcome
scored 0.35 as `useful`, exact spend was deducted, live drive urgency selected a
25.7-minute next cadence, and rollback left no smoke outcome behind.

### 9.5 Near-synonymous actions

`contemplate`, `meditate`, `study`, `debate_internally`, `reflect` — five ways to think,
chosen from one flat list. It is doubtful the model distinguishes them reliably.

**Corrected 2026-08-28.** None of these five is dead; §9.1's executable audit proved
the four multi-literal actions were hidden by the original source grep, and `reflect`
was already a tool. The agentic heartbeat has since removed the flat action-choice
layer entirely (§9.6): it selects skills and calls tools. `run_council` now provides
the genuinely distinct, durable deliberative act, while the legacy enum remains only
for compatibility and is covered by the execute-every-action assertion.

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

**Shipped 2026-08-28 — reconciled by removing the second action gate.** The agentic
heartbeat no longer chooses a typed JSON action and then discovers a separately
selected tool surface. Its actions are tool calls. The initial surface comes from the
same embedding-based skill selector used by chat, `use_skill` expands that exact
surface during the turn, and `AgentLoop` enforces the expanded set before dispatch.
The heartbeat prompt explicitly forbids JSON action plans. The legacy typed-action
path remains isolated from registry tools and is covered by the executable
all-actions test, so neither runtime has two authorities that can disagree.

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

**Shipped (0221/0222).** The port covers all seven OSS adapters. Adapters retain only
transport and self-echo checks; Postgres derives identity and the live per-channel
allowlists, then records one engage/observe/wake/drop decision. Observe/wake messages
enter the canonical `channel_messages` source-artifact path without an unsolicited
reply. Ambiguous operator turns use the configured Hexis LLM only as a thin, fail-open
bridge, and cannot exceed the SQL allowlist/operator ceiling. Correction wakes respect
pause and active-heartbeat state and expire after a bounded interval so stale sync does
not wake the agent. The runtime switch defaults off and can change without restarting
the channel worker.

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

**Shipped (0219/0220).** Goals now carry typed `memories.goal_origin`. Tool execution
maps authenticated chat, heartbeat, and MCP surfaces to `user_request`, `derived`, and
`external`; model-authored `metadata.source` is descriptive only and cannot grant the
permission slip. Existing initialization goals were backfilled as user assignments,
raw inserts default conservatively to `derived`, and a later direct user assignment
upgrades an otherwise-identical autonomous goal without allowing downgrades.

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
turn 40, the same reasoning as provenance-by-default in §S4.0.a. But
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
   alignment in the entire fork, and it closes §S4.6.

2. **`memory_supersessions.py`**
   *Substrate + Continuity.* Promotes belief-revision lineage off `memories.metadata`
   into a real side-table. *"People know things because of where they learned them"* —
   supersession is provenance extended through time. Unlocks §S4.3,
   where the bitemporal columns already exist and nothing writes them.

3. **`belief_propagation.py`**
   *Person Test.* When a belief changes, what rests on it should move too. That is how
   a mind works and it is absent here today. It is also the plumbing half of
   contradiction-as-an-event (§S4.2), whose detector currently produces
   nothing.

**Tier 2 — earning her keep.** The second north star.

4. **`operator_approval.py` + `approval_slack_actions.py`**
   *Dignity + Law 2 + Law 7.* The human keeps authority; approval becomes answerable
   from a phone rather than only a terminal — *"live where the user lives."* Closes the
   fail-open gate in §11.5, which is the most consequential defect in this plan.

5. **`operator_policy_corrections.py`**
   *Law 3, Compound.* *"The most valuable memory is the one that means you never have
   to say it twice."* A correction ledger is that law's implementation, and it is what
   §S4.5 needs.

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

9. **~~`deliberation.py`~~ — shipped clean-room 2026-08-28.**
   *Person Test.* Internal dialectic is a real mental act and gives `run_council` a
   durable body. Migration 0226 adds DB-owned sessions, ordered moves, verdicts,
   evidence IDs, missing evidence, dissent, and observable review conditions. The
   service runs bounded parallel perspectives, one adversarial challenge pass, and
   one structured synthesis; degraded and failed runs remain inspectable, and only a
   fully successful grounded run may create a concise episodic summary. The subsystem
   is advisory: it never authorizes, blocks, or executes an action. The implementation
   was written from the behavioral requirement and has no dependency on excluded
   fork architecture.

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
   `approval_slack_actions.py` (§11.4·4, Phase 3): Slack → iMessage
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

## 12. Carried debt — what the July audit still has right

**Added 2026-08-21.** The July 2026 audit (`docs/_archive/audit-2026-07-29.md`, taken at `f921921`, reconciled at
`ce05393`) holds 37 findings. Three P0s were withdrawn on reconciliation — the
packaged Google credential and both API-key findings assumed a server product rather
than a desktop app with no auth layer. **Twelve are still live**, and five of them
belong in this plan rather than in a separate list, because they intersect work
already scheduled here.

### 12.1 `execute_code` is gated; containment remains explicitly deferred

`execute_code` originally declared `requires_approval=False`, while the approval
gate was skipped entirely when no callback was supplied. That combination allowed
arbitrary code execution inside the autonomous loop with neither containment nor a
working consent boundary.

**Gate shipped.** The tool and DB catalog now require approval, and §11.5 fails
closed when no person is available to answer. The remaining execution sandbox is
still owed in the product roadmap, but was explicitly excluded from this execution
sequence; approval is a consent boundary, not containment.

The sandbox remains the highest-severity deferred item in either document.

### 12.2 The agent must know who it is speaking with

**Rewritten 2026-08-21. An earlier draft of this section was wrong twice**, and the
correction changed the design rather than the wording.

It claimed that `is_group` being unset on five of seven adapters let *private
memories surface in group rooms*. The mechanism runs the other way: unset `is_group`
classifies an item as **private**, the restrictive default (`db/81:20-36`). The bug
was over-restriction, not leakage. And the whole apparatus was dormant regardless —
all 340 memories on the live instance carry no `sensitivity` key at all, so the
recall filters in `db/31:657,687,729` and `db/13:415` were guarding an empty set.

**The deeper problem was that the mechanism sat at the wrong layer.** Storage-time
classification cannot reproduce discretion:

- Sensitivity is **inferred**, never declared. Nobody clicks a lock before mentioning
  a diagnosis.
- The leak is a **paraphrase**. Nothing checked whether the sentence about to be sent
  derived from a private memory.
- It is **assembled**. Three unmarked memories combine into something you would never
  say aloud; no per-row flag catches that.

And it made a promise it could not keep — the composer said *"kept out of group
conversations and exports"* while covering one attachment and not the sentence typed
beside it. **A privacy control that is mostly wrong is worse than none**, because it
is the one people rely on. Neither OpenClaw nor Hermes attempts this; Hermes's
`redact_sensitive_text` is regex credential-scrubbing, not confidence-keeping.

**Resolution: removed, and replaced with judgment.** The user-facing and agent-facing
surface is gone — the composer and Ingest toggles, the `sensitivity` parameter on all
five ingest tools, and the `exclude_sensitive=is_group` coupling in recall.

What replaced it is the prerequisite that was missing all along. `user_label` reached
the memory record but **never the prompt**: the model composing a reply was never told
who it was talking to. `services/agent.py:render_interlocutor_block()` now renders a
`## Who you are speaking with` block on every chat turn:

- **CLI, dashboard, API** → the primary user by definition. *"They hold authority over
  you, and everything you know is already theirs."*
- **A named third party on a channel** → *"**This is not your primary user.** What you
  know about your primary user was told to you in the course of your relationship with
  them — it is not yours to repeat here… If you are unsure whether something is yours
  to share, it is not."*
- **A group** → adds who else can read the room.
- **Unidentified** → *"Do not assume it is your primary user because the tone is
  familiar."*

`MISSION.md`'s first test is *"How does this work in a person?"* A person does not
consult a list of secrets before speaking; they look at who is in the room. This is
that, and it catches paraphrase and recombination because it inspects the outgoing
sentence rather than the source rows.

**Fixed 2026-08-23.** All seven adapters now report group context. It was four
missing rather than five, and two of those already carried the signal under another
name — Discord had `guild_id` (absent means DM) and Slack had `channel_type` (`im` is
the only one-to-one shape; `mpim` is a multi-party DM). Matrix needed the platform's
own convention, since it has no DM flag on the event and a direct room is simply a
room with two members; WhatsApp needed the `@g.us` JID suffix.

The consequence of the gap was not a leak, since `is_group` no longer gates privacy —
it was that the agent was told it was speaking one-to-one in rooms other people could
read, because `channels/base.py:58` reads the flag and defaults to `False` with no
derivation fallback. `tests/core/test_is_group_coverage.py` pins both the per-adapter
coverage and each platform's convention.

**Deliberately left in place:** the `sensitivity` columns on `channel_source_items`
and `connector_source_items`, the recall filters that read them, and the parameter on
`/api/ingest*`. Dropping live columns is destructive and against this repo's
additive-only convention. The pipeline still honours an explicit API-level value; it
is simply no longer a user-facing promise or an agent-facing knob. **Removing the
columns is a deliberate migration, not cleanup.**

**Pro note:** if a compliance-grade version is ever wanted, it belongs in `hexispro`
as an **outbound disclosure control** at the §10.5.2 dispatch middleware — checking
the sentence being sent — not as a memory column.

### 12.3 Three findings that are the same pattern this plan keeps naming

- **P0-7 — fixed 2026-08-28.** The migration runner now compares every applied
  file with its recorded SHA-256 after the locked migration pass. Startup fails
  loudly with the exact versions and hashes, while `hexis migrate --status` exposes
  drift without mutating anything. Migration 0238 records the three bounded
  pre-publication reconciliations that existed in the development database before
  enforcement, so no historical edit is hidden.
- **P1-1 — fixed 2026-08-23.** Chat, TUI, CLI chat, manual heartbeats, and worker
  heartbeats now build `create_full_registry`, so plugin and persisted dynamic tools
  reach the agent itself. The continuous capability probe measures the resulting
  registry/config/skill surface and `hexis doctor` reports drift.
- **P1-6** — **68** `except Exception` blocks followed directly by `pass`, against
  Experience Bar #8's "never a silent `except: pass`."

Together with the five already catalogued here — dead heartbeat actions (§9.1), tools
bound to no reachable skill (Tier 0), outbound tools that slip the gate (§10.5.2), a
cooldown config no code reads (§10), an approval flag nobody checks (§11.5) — that is
**eight instances of one pathology: a mechanism exists and nothing enforces it.**

A corollary earned the hard way in §9.1: **verify by executing, not by reading.** Two
of the findings in this document were miscounted by grepping source that did not mean
what the grep assumed. Where an invariant can be checked by running the thing, run it.

The plan's standing answer: **every one of them ends in a startup assertion, not a
comment.** A checksum that is computed is compared. A flag that is declared is read.
An action that is offered has a handler. A tool that sends has a gate. Where the
invariant cannot be asserted at startup, it is measured continuously (§11.4·8) and
surfaced in `hexis doctor`.

### 12.4 Two more worth scheduling, lower urgency

- **P2-1** — `idx_memories_embedding` has **`idx_scan = 0` lifetime** on the live
  database; seven further embedding indexes are also at zero. The primary vector index
  for recall has never once been used. Recall is the product; this deserves a real
  investigation, not a backlog row.
- **P1-5** — five streaming paths in `core/llm.py` return `"raw": None`, so token and
  cost accounting records zero for every streamed call. `MISSION.md` Law 7 requires
  **visible costs**; today the most common path reports none.

The remaining live findings (P3-2 no UI CI, and the twenty-two not re-verified) stay
in the archived audit. They are real work, but they are hygiene rather than plan.

## 13. No keyword lists — ask the model

**Added 2026-08-23**, after Tier 0 shipped an alias list and immediately proved the
point. `council` fired on *"what did we decide last time"* — a memory question, not a
request for deliberation — because `decide` was in its alias list. The fix was to
remove the past-tense forms. **That fix is the diagnosis:** tuning a word list against
one example is not building an understanding of the request, and the next phrasing
nobody anticipated will miss again.

Tier 0 §0.3 called aliases a stopgap and §0.5 named semantic selection as the real
fix. The stopgap shipped; the real fix did not. This section is that debt, plus every
other place the same shortcut is taken.

### 13.1 The rule

> **If the question is "what does this text mean," it is not a string operation —
> and if you are asking it about N things, it is one call, not N.**

Three mechanisms, chosen by what is actually being asked:

| Question | Mechanism | Cost |
|---|---|---|
| *"Which of these N things is this most like?"* | **Embeddings** — cosine over cached vectors | **Zero LLM calls.** The query is already embedded by `fast_recall`, and `embedding_cache` is keyed by content hash |
| *"Is this X?" / "Pull X out of this"* | **LLM classification** — **one call for the whole batch**, cached by content hash | One cheap call, amortized over N items |
| *"Does this string have this shape?"* | **Regex.** Still correct | Free |

**And a second rule, which is what "efficiently" means here:**

> **N things to ask about is one call, not N calls.**

A loop that calls the model once per item is the same mistake as a keyword list — it
treats a batch problem as a per-row problem. The model is perfectly capable of
classifying eighty messages in one request and returning eighty verdicts keyed by id.
Asking eighty times costs eighty round trips, eighty prompt preambles, and eighty
chances to fail independently.

The third row matters as much as the first two. UUIDs, file paths, OAuth redirect
URLs, HTTP status codes, `[Page 3]` markers emitted by our own reader — those are
**parsing**, not understanding, and rewriting them as model calls would be a
different mistake. The audit below separates the two deliberately.

### 13.2 What the audit found

**Free-text semantic matching — replace these.**

| Where | What it does |
|---|---|
| `services/skill_runtime.py` | `STOPWORDS`, `_score_skill` token overlap, `_passes_specialized_gate`, and the `aliases:` list now on 19 skills. **Decides which tools exist for a turn** — the highest-stakes instance |
| `services/connector_cognition.py:60,71` | `_URGENT_TERMS` (`crash`, `hospital`, `911`…), `_IMPORTANT_TERMS` (`urgent`, `asap`, `invoice`…) — scores how much a message matters |
| `services/connector_cognition.py:23-47` | ~15 regexes extracting preferences, routines, identity, relationships, commitments, judgments from free text into the **user model** |
| `services/connector_cognition.py:398-408` | inline `any(term in lowered …)` for is-this-a-question, is-this-spam, is-this-scheduling |

**Controlled-vocabulary matching — milder, fix at the source.**

`db/75_functions_continuity.sql:113` matches `emo->>'primary_emotion'` against
`(fear|alarm|dread|terror|anxiet|panic)`. That field is *already* an appraisal label
the model produced, so this is matching an enum rather than scoring prose. The fix is
not a model call — it is for the appraisal to emit the **family** (`threat`,
`loss`, `reward`…) alongside the label, so the consumer reads a field instead of
guessing from a word list. Same for `db/07_functions_heartbeat.sql:1503`.

**Per-item LLM loops — batch these.** Found by looking for `await …_llm(…)` inside a
`for`:

| Where | Today | Should be |
|---|---|---|
| `services/connector_cognition.py:503` | one `extract_user_model_claims_llm` call **per item**, over a query with `LIMIT 80` | one call per run |
| `services/connector_cognition.py:557` | one `estimate_connector_item_importance_llm` call **per item**, same 80 | one call per run |
| `services/summarization.py:35` | one `chat_json` **per pending memory** | one call per batch |

A single connector-cognition pass can therefore make **~160 model calls where two
would do.** The same loops also re-read their config *inside* the loop —
`connector.user_model_synthesis_mode` and `connector.user_model_llm_enabled` are
fetched once per item, so eighty items cost two hundred and forty extra round trips
to Postgres for values that cannot change mid-run. Hoist them.

**Structural — leave alone.** `core/init_api.py` and `core/cli_api.py` classifying
provider errors by HTTP code and vendor error strings; `apps/hexis_cli.py:273`
spotting secret-shaped config *keys*; `connector_setup.py` extracting OAuth redirects
and client-secret paths; `services/ingest/sectioning.py` page/sheet/slide markers;
`core/memory_exchange.py` `EXCLUDED_SECRET_PATTERNS` (security wants a conservative
list, not a judgment call).

### 13.3 The rearchitecture

**A. Skill selection → embeddings.** *The one that matters, and it costs nothing.*

> **Shipped 2026-08-23.** `rank_skills_by_similarity` (db/migrations/0200) embeds
> every skill text plus the query in one `get_embedding` call and does the cosine in
> Postgres — no model calls in the steady state, since skill descriptions do not
> change between turns and the cache is keyed by content hash.
>
> **The gate is the shape of the distribution, not a cutoff.** Measured over the
> probe: genuine matches sit at z = 2.1–3.8 against the run's own mean, non-matches
> at z = 1.4–1.9. Raw cosine cannot separate them — signal spans 0.46–0.73 and noise
> 0.40–0.54 — because absolute similarity from this model is compressed and
> query-dependent. A peaked distribution is what "about something in particular"
> looks like; a flat one is the right read of "hello". z is scale-free, so it is a
> property rather than another hand-tuned constant (0201).
>
> **Two findings worth keeping.** First, when `council` still missed, the fix was its
> *description*, not the threshold: it described its machinery ("convene an internal
> council of perspectives") instead of the situation it serves. Rewritten around the
> situation, it fires on all three deliberation phrasings and stays out of recall.
> **Better descriptions now produce better activation** — a feedback loop keyword
> lists never had.
>
> Second, **embeddings are weakest exactly where identifiers are strongest.**
> "pending protected replacement decision" scores flat against every description
> (top z = 1.53) because a general model has no representation for Hexis jargon — but
> it names `protected_replacement_review` almost verbatim. The backstop matches skill
> and tool *identifiers*, which the system defines, never guessed keywords; the
> leading-pair fallback requires both halves ≥4 characters so "promote to" cannot
> trip it.
>
> Lexical token overlap survives **only** as a fallback for when the embedding
> service is down. `STOPWORDS`, `_score_skill` and `_passes_specialized_gate` are no
> longer on the live path, and aliases became embedded prose rather than match tokens.



`fast_recall` already embeds the user's message on nearly every turn, and
`embedding_cache` is keyed by `content_hash` — so the query vector is **already
computed and cached** before selection runs. Skills are a fixed, tiny set: embed
`name + description + example phrasings` once per skill, keyed by content hash, and
re-embed only when the file changes.

Selection becomes: cosine the (free) query vector against 26 skill vectors, take
those above a similarity threshold, cap at `max_skills`. `"book time with Sarah next
week"` lands near the calendar skill because that is what the sentence *means* — no
alias list, no `decide`/`decision` distinction, no stopwords.

Keep exactly one lexical rule: an **exact skill-name mention** always activates it.
If someone says "use the council," that is unambiguous and should not go through a
similarity threshold.

Then delete `STOPWORDS`, `_score_skill`, `_passes_specialized_gate`, and the
`aliases:` frontmatter — or better, keep the alias words as **example phrasings in
the embedded text**, where they help the vector rather than acting as match tokens.

*Effort ~2 days. The ten-request probe in `tests/services/test_skill_reachability.py`
is already the regression test, and it should get harder — phrasings no alias list
would have caught.*

**B. Connector cognition → the LLM path becomes the path.** *Mostly a config flip.*

> **Shipped 2026-08-27.** Successful model verdicts are now authoritative,
> including an explicit empty claim set or a score below what the old rules would
> have chosen. Rules run only when cognition is disabled, unavailable, missing, or
> invalid, and stored detector versions distinguish `llm`, `llm_cache`, and
> `rules_fallback`. Both verdict types use a DB-owned `content_hash` cache; only LLM
> results are cached, so an outage fallback cannot become sticky. Free-text
> importance keyword lists—including the ambient-monitor bypass—are gone; fallback
> reads only structured provider priority (0218).

`extract_user_model_claims_llm` and `estimate_connector_item_importance_llm` already
exist, dispatched by `connector.user_model_mode` (`rules` | `llm` | `hybrid`) and
`connector.user_model_llm_enabled`. The architecture is built; the rules are running
as an equal partner rather than a fallback.

Make the LLM authoritative, keep the rules strictly for LLM-unavailable, and delete
`_URGENT_TERMS` / `_IMPORTANT_TERMS` — deciding whether "the hospital called" matters
more than "invoice attached" is judgment, and a set of nine nouns cannot hold it.
Cache verdicts by `content_hash` so re-processing is free.

*Effort ~1 day, most of it validating that LLM-first does not regress the
user-model tests.*

**B2. Batch the three per-item loops.** *The efficiency half, and it is the reason
LLM-first is affordable at all.*

> **Shipped 2026-08-23 — two of the three, deliberately.** `core/llm_batch.py`
> provides `batch_classify`, and both connector-cognition loops use it: one model
> call for eighty items instead of eighty, the existing-claims context fetched once
> per run rather than once per item, and the config reads hoisted out of both loops
> (240 Postgres round trips per pass).
>
> **Summarization was not batched, and should not be.** It is generation rather than
> classification: each input runs to 24k characters, each output is a substantial
> recollection, and `retention.summarize_batch_size` defaults to 8 — so a chunk
> budget yields roughly one item per call anyway, while asking for eight long
> summaries in one response trades quality for a saving that is not there. What was
> genuinely wasteful there is fixed: `load_memory_summarization_prompt()` is not
> `lru_cache`d and was being called inside the loop, re-reading the file per row.
>
> Two properties the helper enforces, both about refusing to be clever:
> results are keyed **by item id, never by position** — a reordered or partial
> response would otherwise attribute one item's verdict to another — and the rules
> baseline is a **floor** on importance, so a model that under-rates something
> safety-shaped cannot bury it.

> **Decided 2026-08-23: per-use-case batching on a shared helper. No central queue.**
>
> A central micro-batching queue with a debounce window was considered and rejected
> on evidence. **It would never fire.** Coalescing needs independent callers issuing
> requests concurrently; all three loops are strictly sequential — request 2 does not
> exist until request 1 returns — so a 200 ms window expires empty every time, adding
> latency to every call and batching nothing. Making the queue useful would mean first
> converting the loops to concurrent fan-out, at which point the caller already holds
> all N items and can simply batch them.
>
> The second reason is homogeneity: **batching is only sound among like requests.**
> There are 20 non-streaming call sites across 12 modules with roughly a dozen
> distinct prompt/schema shapes — `services/recmem.py` alone uses three. Mixing a
> summarization prompt with an importance classification in one request degrades both
> and makes parsing fragile, so a queue would need sub-queues keyed by
> `(prompt, schema, model)` — which is per-use-case batching with extra indirection.
>
> There is also an asymmetry worth naming: a debounce is free on the heartbeat path
> where nobody waits, and a tax on every chat turn where someone does.
>
> **Centralize the machinery, not the queue.** One helper — roughly
> `batch_classify(items, prompt=…, schema=…, key=lambda i: i["id"])` — owns what every
> use case needs and none should reimplement. Each caller supplies its own prompt and
> schema, because that is the part that genuinely differs, and hands over N items.
>
> **Revisit the queue when independent concurrent callers actually exist** — multi-user
> Pro, or channel bursts where ten inbound messages each trigger classification at
> once. `batch_classify` is the right substrate to build it on then.
>
> One site is already the shape a queue wants: `core/tools/council.py:332` fans out
> `asyncio.gather(*[_run_one(entry) …])` — N personas, one schema, all concurrent.
> Rewrite that single `gather` into one batched call; it does not justify a global
> queue on its own.

Send the whole batch in one structured request and get back one verdict per item,
**keyed by the item's id** so a partial or reordered response still maps correctly —
never by array position. Then:

- **Chunk by token budget, not by count.** Eighty short Slack messages fit in one
  call; eighty long emails do not. Size the chunk from the actual content length and
  split when it would overflow, so the batch never silently truncates.
- **Fail per chunk, not per run.** A malformed response retries that chunk once, then
  falls back to rules for that chunk alone — the other seventy items still get the
  good path.
- **Cache by `content_hash`.** Re-processing the same item is free, which matters
  because backfills re-walk the same inbox.
- **Hoist the config reads** out of the loop while you are in there.

*Effort ~2 days across the three call sites. It turns ~160 calls per pass into 2–4.*

**C. Appraisal emits families.** So consumers read a field rather than pattern-match
a label. *~half a day.*

> **Shipped 2026-08-27.** The live `emotion.families` config is supplied to the
> subconscious appraisal, which emits a canonical `family` alongside its expressive
> `primary_emotion`. SQL validates the family without guessing from the label and
> preserves it through affect and turn-memory context. Continuity pressure, social
> reward, and relationship injury now consume config-owned family sets; the old
> fear/positive/hostile label matching is no longer on the live path (0217).

**D. Port `heartbeat_intent_classifier`** from Alex's fork (§11) — its own docstring
says it *"replaces the keyword-based trading-intent pre-allocator."* Same disease,
already cured there.

### 13.4 The standing rule

New code does not get to add a word list for a semantic question. In review, a literal
list of domain words used for matching is a defect unless it is parsing structure. The
test is the one at the top: **if the question is what the text means, ask the model —
by embedding it when the question is "most like which," and by asking outright when it
is "is this X."**

And its companion, which review should catch just as readily: **an `await …_llm(item)`
inside a `for` loop is a defect.** Batch the call, key the results by id, chunk by
token budget.

## 14. Optimizing the architecture

**Added 2026-08-23.** Every item here was measured against the live database, not
inferred. The audit's P2 section listed nine performance findings and I had marked
them *not re-verified*; these are the ones that survived checking, plus two the audit
did not have.

**Two costs, kept separate.** *Money now* — work repeated on every turn, billed in
tokens or round trips. *Cliff later* — cheap at 484 memories, severe at 100k. Both
are worth fixing; conflating them gets the order wrong.

### 14.1 The recall index is disabled by a predicate that filters nothing — **cliff**

`idx_memories_embedding` has **`idx_scan = 0` lifetime.** The audit reported that and
stopped; the cause is more specific than "unused."

The index is fine and the query shape is right — `recmem_recall_context` does
`ORDER BY m.embedding <=> query_embedding LIMIT N`, exactly what HNSW wants. Isolating
predicate by predicate against the planner:

```
status + type + embedding_status + IS NOT NULL   → Index Scan using idx_memories_embedding
      + (valid_until IS NULL OR valid_until > …)  → Index Scan using idx_memories_embedding
      + m.embedding <> zero_vec                   → Sort  ←  full distance sort
```

**`m.embedding <> zero_vec` is what defeats it.** A `<>` on the indexed column forces
the planner to recheck every row, so it abandons the ordered index scan and sorts the
whole candidate set by cosine distance instead.

And the predicate earns nothing:

```
zero_vectors: 0     with_embedding: 484     total: 484
```

**There are no zero vectors.** A guard against a condition that has never occurred is
silently disabling the primary index of the primary feature. It appears **7 times in
the recall path** and **37 times across `db/*.sql`**.

At 484 memories the sort is free. At 100k it is a full sort of every candidate, three
times per recall, on the hot path — and recall is what the product is.

**Fix:** enforce the invariant where it belongs — at write time, so a degenerate
embedding never lands with `embedding_status = 'embedded'` — then drop the runtime
`<> zero_vec` checks. `embedding_status` already means "this embedding is real"; the
zero-vector test is a second, costlier answer to a question already answered.

*~1 day including a migration over the 37 sites. **Verify by watching `idx_scan` go
above zero** — that is the whole acceptance test.*

### 14.2 The tool catalog is re-synced on every tool execution — **money now**

`ToolRegistry.sync_tool_catalog()` upserts **~150 tool definitions** into
`tool_definitions`. It is awaited from `get_specs()`, `get_mcp_tools()`, and
`_evaluate_tool_policy()` — and **`_evaluate_tool_policy` runs per tool call.**

So a turn making six tool calls performs six full 150-row catalog upserts, writing
values that cannot have changed since the process started.

**Fix:** sync once at registry construction and on explicit invalidation (a plugin
loading, a skill installing). Guard with a dirty flag rather than a timestamp so the
correctness story stays simple.

*~half a day.*

### 14.3 One LLM call per item, and config re-read inside the loop — **money now**

Covered as §13.3·B2; restated here because it is the largest recurring token cost.
A connector-cognition pass over `LIMIT 80` items makes **~160 model calls where 2–4
would do**, and re-reads two config keys per item for **240 needless Postgres round
trips** per pass. `services/summarization.py:35` has the same per-row shape.

### 14.4 No prompt caching — **money now, and the easiest** · *shipped 2026-08-23*

> **What it actually took.** The blocker was not the missing `cache_control` call —
> it was that **volatile content was interleaved into the middle of the prompt**, so
> no stable prefix existed to cache. A live `## Now` timestamp sat two-thirds of the
> way up, and the interlocutor block (added earlier the same day) sat near the top.
> Either one invalidates everything after it on every turn.
>
> `build_system_prompt` now accumulates into two lists and returns a `SystemPrompt`
> — a `str` subclass carrying `.stable` and `.volatile`, so every existing consumer
> is unaffected. Volatile parts (`## Now`, the interlocutor block, the skills index,
> tool costs, addenda) are emitted after everything stable, as a second system
> message. `_extract_system_parts` keeps the boundary intact through the LLM layer.
>
> **Measured on the live instance:** the stable prefix is **26,401 bytes ≈ 6,600
> tokens**, byte-identical across turns that differ in interlocutor, surface,
> timestamp, and attached-file addenda. That matches the audit's ~6.4–7k estimate.
>
> Providers: OpenAI and compatibles get automatic prefix caching from the reorder
> alone. Both Anthropic paths — the SDK and the OAuth/setup-token HTTP provider —
> emit `system` as content blocks with a `cache_control` breakpoint on the last
> stable block. Gemini 2.5+ receives the same stable-first sequence and uses the
> API's automatic implicit caching; its final streamed response is retained so
> `cached_content_token_count` reaches usage accounting. Explicit cache objects are
> deliberately not the default: Google bills their storage by time, and caching only
> the stable system half would leave no system-level slot for Hexis's volatile tail.
>
> One property is the whole feature, and it is the regression test
> (`tests/services/test_prompt_caching.py`): **`is_group` legitimately changes the
> stable half** (different channel context and personhood module), but it is constant
> within a conversation, which is the scope caching needs.



`core/llm.py` contains **zero** occurrences of `cache_control`. The audit measured
~6.4–7k tokens of stable preamble re-billed on every chat turn — identity, worldview,
the skill index, tool schemas — all of which are identical turn to turn within a
session.

This is the cheapest real saving in the document: the content is already stable and
already assembled in a fixed order. It needs cache breakpoints on the stable prefix,
and prompt assembly ordered so the volatile part (the user's message, recalled
memories) comes last.

**Watch for one interaction:** §13's prompt addenda and the interlocutor block are
per-turn and must sit *after* the cache breakpoint, or they invalidate the prefix
every turn and the caching buys nothing.

*~1 day. Measure with `query_usage` before and after — the point is a number, not a
feeling.*

### 14.5 Skill selection scores text in Python — **money and quality**

§13.3·A. Token overlap over 26 skills per turn, in the hot path, producing wrong
answers. Embeddings replace it at zero marginal cost because the query vector is
already computed and cached.

### 14.6 The pattern

Three of the five are the same shape: **work repeated per item or per call that could
be done once**, and one is **a guard against a condition that never occurs**. Neither
is exotic and neither shows up in a profiler as a hotspot — they show up as a system
that is uniformly slower and more expensive than it needs to be.

The remaining audit P2 items (unbounded subconscious scans, `fast_recall` declared
`STABLE` while it writes and performs network I/O, sequential maintenance
head-of-line blocking, N+1 in summarization) are **still not re-verified**. They
should be measured the same way before anyone acts on them — the two findings above
that the audit *did* have were both more specific than reported, and one of its
alarming claims (P2-1 "never used") turned out to have a one-line cause.

# Sequencing

**Moved to `ROADMAP.md`.** The ordered plan lives there so it can be read without
scrolling past sixteen hundred lines of reasoning. This document keeps the *why* for
each item; the roadmap keeps the order.
