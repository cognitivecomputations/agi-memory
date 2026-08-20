# Positioning — what Hexis should be best in the world at

**Date:** 2026-08-21 · **Companion to:** `bridge_gaps.md`, which is a catch-up
document. This one is not. Every claim below is checked against the live schema and
the tree.

---

## 1. The race we cannot win

OpenClaw ships 52 skills, 158 extensions, 22 messaging channels, native apps for five
platforms, and lists OpenAI, GitHub, NVIDIA and Vercel as sponsors. Hermes has Nous
Research behind it, six execution backends, and serverless hibernation.

**Hexis will never have the most integrations.** Every week spent closing that gap is
a week spent losing a race against better-funded teams, on an axis where being second
is worth nothing.

`bridge_gaps.md` is still worth executing — an assistant that cannot be reached or
cannot ask a question is not a product. But it is table stakes, not a strategy. It
gets Hexis to parity. It does not make anything best in the world.

## 2. The race nobody else can enter

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

## 3. What is already in the ground

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
the always-on tool floor (per the `bridge_gaps.md` Tier 0 probe) and are almost never
called, because nothing in the product asks for them. **This is a surfacing problem,
not a building problem** — which is why the two flagship ideas below are weeks, not
quarters.

---

## 4. The invocation problem — the thing that actually blocks all of this

*(§4.0.x below; the flagships themselves are §4.1–4.7.)*

**Revised 2026-08-21 after a second probe.** An earlier draft of §4 described these
capabilities as if a user would reach for them. **A user will never type "run the
council" or "trace why."** These are not features anyone asks for by name. If the
agent does not invoke them on its own, they do not exist.

`bridge_gaps.md` Tier 0 found that skill activation is driven by *the user's
vocabulary*. That is the wrong axis for everything in this document. Nothing in
"should I take this deal?" mentions a council, and **nothing a user ever says
mentions a contradiction** — the trigger is not in their words at all. It is in the
agent's own state.

So the question for each capability is not "is it reachable?" but **"what makes the
agent decide to use it?"** There are four honest answers, and they need different
machinery.

### 4.0.a Structural — not a decision at all

Some things must never be left to the agent's judgment, because an agent that has to
*choose* to show its sources will not, reliably, under pressure, on turn 40.
**Provenance is one of these.** Make it a property of the data path: `recall` always
returns `source_attribution` and `trust_level`, the prompt always requires citation,
the renderer always draws the footnote. Zero decisions, zero failure modes.

### 4.0.b Observed state — the heartbeat's Observe phase

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
system.** Contradiction-as-an-event (§4.2) therefore does not need new plumbing into
the heartbeat — it needs contradictions to actually be *detected and written* (the
detector exists and this instance has produced none), and it needs the resolution to
be surfaced to the user rather than resolved silently.

**What is genuinely missing on this axis** is narrower and still worth doing:

1. **Detection has to run.** `find_contradictions()` exists and zero contradictions
   are recorded on an instance with 323 active memories and 34 worldview beliefs.
   Either the detector never fires on the ingest path, or its threshold is set so it
   never triggers. Find out which. *This is the real §4.2 blocker.*
2. **Trust is nearly a constant.** 252 of 323 memories share the identical
   `trust_level` of `0.4302279608697066` — a computed default, not a judgment. A
   provenance UI (§4.1) that renders trust is worthless while trust does not vary.
3. **The heartbeat's skill selection is fed a JSON dump.** `services/agent.py:825`
   builds the skill-selection query by `json.dumps(heartbeat_context)[:4000]` and runs
   the same lexical matcher over it. Whatever skills that activates, it is not a
   considered choice — and it means a chosen action like `inquire_deep` can find its
   tools ungated by accident or gated by accident. See §4.0.c and `bridge_gaps.md`
   Tier 0.

*Effort: unchanged at ~2 days, but spent on detection and trust variance rather than
on plumbing that already exists.*

### 4.0.c Situational recognition — cues in the prompt, not words in the query

For capabilities where the trigger is a *kind of moment* rather than a state — the
council on a consequential decision, point-in-time recall on a temporal question —
the answer is an instruction, not a matcher: *"when the user faces a consequential
decision with real tradeoffs, convene the council before answering."*

Measured, `council` scores **1, 4, 1** against "should I take this deal or walk
away?", "help me think through a hard decision," and "weigh the tradeoffs on hiring."
The threshold is 5. It never fires on the decisions it exists for. Lexical matching
cannot fix this; a semantic matcher (Tier 0 §0.5) helps; an explicit prompt cue plus
reachability is what actually does it.

### 4.0.d Ambient responsibilities — already built, zero rows

`ambient_responsibilities` is a complete standing-orders engine: `trigger`,
`evaluator`, `sources`, `actions`, `delivery`, cooldowns, `consecutive_silent`
back-off, and a run-audit table. It is oriented at the *outside* world — watch Gmail,
watch a threshold, notice a missed check-in.

It has **zero rows**. Built, wired, never populated. It is the natural home for
`bridge_gaps.md` §1's accepted automations, and it needs seeds far more than it needs
code.

---

## 4.1–4.7 The flagships

Each of these is now labelled with **how it gets invoked** — because a capability with
no invocation path is not a capability.


### 4.1 Provenance by default — the assistant that never asks you to take its word

**Invocation: structural (§4.0.a).** Never a decision the agent makes.

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

**Effort:** ~1 week. **Demo value:** the highest in this document. Thirty seconds,
unanswerable by any competitor.

### 4.2 Contradiction as an event — the memory that gets *more* accurate

**Invocation: observed state (§4.0.b).** The heartbeat is told how many contradictions
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
   automation suggestions (`bridge_gaps.md` §1). Three outcomes: the new one is right,
   the old one is right, or both hold in different contexts (`accept_tension`
   already models this).
2. **Resolution writes bitemporally.** Do not delete the loser. Set `valid_until` and
   `superseded_by` on the old belief — the columns exist. The history stays queryable,
   which is what makes §4.3 free.
3. **Batch it in the heartbeat.** A daily pass rather than an interrupt. Contradictions
   are rarely urgent, and an assistant that interrupts you about bookkeeping is worse
   than one that saves it for a briefing.
4. **Show the ledger.** A view in the dashboard: contradictions found, resolved, and
   accepted-as-tension. This is the proof that the thing is working.

**Effort:** ~1 week, most of it surfacing. **Strategic value:** the highest here.

### 4.3 "What did you know, and when?"

**Invocation: situational (§4.0.c).** A prompt cue on temporally-framed questions —
"as of", "back then", "has that changed" — not a tool the user names.

`memories` already has `valid_from`, `valid_until` and `superseded_by`. The schema is
bitemporal and nothing exposes it.

*"As of last Tuesday, what did you think about the Manning deal?"* — a point-in-time
recall tool, and a diff view (*what changed about X between June and now, and why*).

An agent that can answer this is doing something no file-backed competitor can attempt.
It is also the natural payoff of §4.2: every resolved contradiction deepens the record
instead of overwriting it.

**Effort:** ~3 days, and it is nearly free once §4.2 writes `valid_until` correctly.

### 4.4 Your mind is a file

HMX export/import already works. It is buried in a skill called `memory-exchange`.

In a market where people are genuinely afraid a vendor will delete their AI
relationship — and where every frontier lab's memory is a blob you can neither read
nor move — **"you can take her with you"** is not a feature bullet. It is the
headline.

Make it a first-class flow: `hexis backup` already exists; add `hexis export --mind`,
document the format, and demonstrate a mind moving from one machine to another and
waking up continuous. Publish the schema. Invite other projects to import it.

**Effort:** ~3 days of packaging on top of what ships. **Marketing value:**
disproportionate.

### 4.5 Learning with a diff

**Invocation: observed state (§4.0.b) on a weekly cadence**, delivered through the
outbox. The agent decides there is enough to review; the user only ever responds.

Hermes's headline is "self-improving" — it writes skill files autonomously.
`services/skill_improvement.py` deliberately does not, and that restraint should
become the feature rather than the limitation.

A weekly ritual: **"here is what I learned about you and your work this week —
approve, correct, or forget."** New semantic beliefs, new procedures, revised
strategies, proposed skills, all in one reviewable list.

This is the compounding loop Hermes advertises, with the trust property they cannot
offer: you saw every change before it took. It also feeds §4.2 — a correction here is
a contradiction resolution.

**Effort:** ~4 days on top of the existing background review.

### 4.6 Forgetting well

**Invocation: observed state (§4.0.b).** Memory pressure and decayed fidelity appear
in the Observe packet; `maintain` is already a costed heartbeat action.

`decay_rate`, `fidelity`, fade requests and `docs/memory_retention_design.md` already
describe a compression-native substrate. Competitors have append-only logs that get
slower and dumber with every turn.

Position deliberate forgetting as a feature, not a limitation: the agent proposes what
to let go, asks before dropping anything that looks load-bearing, and reports what it
compressed. *"I remember what matters and I can tell you what I let go"* is a stronger
claim than *"I remember everything,"* and it is the honest description of how memory
actually has to work at scale.

**Effort:** mostly already built; ~3 days to surface.

### 4.7 Define the measure

`evals/` exists. Publish a **memory benchmark** — provenance accuracy, contradiction
detection, recall at six months, cross-session continuity, resistance to stale
beliefs — run every agent on it, and publish the results *including where Hexis
loses*.

Whoever defines the benchmark shapes what "best" means. This is the one axis where
Hexis wins today, and the cheapest credibility available. Losing honestly on two axes
makes winning on four believable.

**Effort:** ~1 week for a first public version.

---

## 5. What not to do

- **Do not chase channel count.** OpenClaw will always have more. Eight channels that
  work beat twenty-two that mostly do.
- **Do not chase skill count.** Twenty skills people use daily beat fifty-two nobody
  activates — especially given the Tier 0 finding that Hexis cannot reliably activate
  the twenty-five it has.
- **Do not build a canvas, a TUI, or native apps yet.** The PWA (`bridge_gaps.md`
  §3a) covers the client for a fraction of the cost.
- **Do not let the council rot.** `run_council` is real deliberation machinery, bound
  to a `council` skill that is chat-loadable — and which scores **1, 4, 1** against
  "should I take this deal or walk away?", "help me think through a hard decision",
  and "weigh the tradeoffs on hiring". The threshold is 5. It never activates on the
  decisions it exists for; the user would have to say the word "council". Either fix
  the selection (Tier 0, §0.3/§0.5) or delete it. Machinery that never runs is not a
  differentiator; it is maintenance debt wearing a differentiator's clothes.

## 6. If we pick two

**§4.1 provenance-by-default and §4.2 contradiction-as-an-event.** Two weeks
together, mostly surfacing work over machinery already in the database, and no
competitor can answer either without rebuilding on a real store.

Together they make one claim that fits on a homepage and survives a live demo:

> **She shows her work, and she gets more right over time, not less.**

Sequence them after `bridge_gaps.md` Tier 0 — provenance rendered from tools the
selector will not activate is provenance nobody sees.

## 7. The order

1. `bridge_gaps.md` **Tier 0** — reachability (~2d). Everything else is built on it.
2. **§4.0.b make detection actually fire** (~2d) — contradictions and trust variance.
   The Observe packet already carries them; today it truthfully reports zero because
   nothing writes any. Without this, §4.2 has nothing to surface.
3. **§4.1 provenance by default** (~1w) — the demo.
4. **§4.2 contradiction as an event** (~1w) — the thesis.
5. `bridge_gaps.md` **Tier 1** — suggestions and `ask_user` (~1w).
6. **§4.3 point-in-time** (~3d) — nearly free after §4.2.
7. **§4.7 the benchmark** (~1w) — publish and let it be argued with.
8. **§4.4 mind portability** (~3d) — packaging and a demo video.

About six weeks to a product with a defensible claim, from a codebase that already
contains most of the parts.
