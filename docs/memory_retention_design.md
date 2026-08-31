# Compression-Native Memory Substrate

**Status:** proposed (v5)
**Supersedes:** the earlier "retention policy" framing. Retention is not a policy layered on a durable store; **forgetting is the nature of the substrate**. Memory is fallible and compressing; the **journal** (§7) is deliberate and permanent — and they are kept rigorously distinct.
**Related:** `db/00_tables.sql` (memories, decay_rate/importance, life_chapters), `db/03_functions_helpers.sql` (`calculate_relevance` — lazy decay), `db/31_functions_recmem.sql`, `db/44_functions_memory_edges.sql`, `db/42_functions_outbox.sql`, `services/worker_service.py` (rest loop), `docs/PHILOSOPHY.md`.

---

## 0. First principle

**Humans are limit-constrained summarizers — experience-compression machines.** A brain has a fixed number of neurons; it cannot and does not store experience verbatim. It encodes lossily, re-writes on recall, decays by default, and *reconstructs* rather than retrieves. Forgetting is not a failure mode of human memory — it is the mechanism. If Hexis is to approach a human-like self, compression must live **in the substrate**, not in a janitor process on top of it.

This requires breaking the load-bearing assumptions of traditional database design:

| Traditional DB assumes | Compression-native substrate |
|---|---|
| **Durability** — rows persist verbatim | Memories are mutable and re-written; nothing stays verbatim for long |
| **Completeness** — store all events | Encoding is selective; **forgetting is the default**, retention the earned exception |
| **Retrieval = exact read** | Recall is **reconstructive and fidelity-graded** — rebuilt from a compressed trace + schema + context |
| **Uniform fidelity** | Fidelity is a **continuous variable** that decays with time and rises with attention |
| **Delete-on-command** | Forgetting is **continuous decay + capacity competition** |
| **Unbounded growth** | **Fixed capacity, always full** — every new memory competes against every old one |
| **The table is the source of truth** | The **schema (semantic/self core)** is the durable truth; the hot episodic store is an admittedly-lossy working set; the **journal** (§7) is a deliberate artifact *outside* memory |

Everything below follows from taking these inversions seriously.

---

## 1. The inversion: always at capacity

A brain is not "empty, filling toward full." It is **always full** — finite neurons, always allocated — and learning is *competition* for that fixed substrate. Hexis's hot memory is the same: a fixed **capacity** (§8) held *by construction*. Encoding a memory doesn't grow the store; it applies **compression pressure** to the rest. You never "run out and GC" — you are always at the edge, and every experience costs fidelity somewhere unless it earns reinforcement.

This is the single structural commitment that makes the rest fall out.

---

## 2. Memory strength — the master variable (replaces the status ladder)

Every memory carries a continuous **strength** ∈ (0, 1], not a discrete `active/archived` status. Strength is the substrate's currency: it sets recall priority, fidelity, and eviction order.

- **Decay is lazy and computed, never mass-written.** Strength is a *function* evaluated on read — extending `calculate_relevance` (`EXP(-decay_rate · age)`, `db/03:12`). No cron rewrites millions of rows; current strength is `f(base_salience, time_since_reinforced, decay_rate)`. `decay_rate` (already on the row, unused for lifecycle today) finally becomes load-bearing.
- **Reinforcement moves strength *up*.** Recall, re-reference, user attention, and emotional resonance reset `last_reinforced` and bump `reinforcement_count` — the ladder goes **up as easily as down**. A gist that keeps getting recalled re-strengthens and stops fading (its *remaining* content regains vividness; detail already compressed away is genuinely gone — forgetting is real, §6). This fixes "importance is only known in hindsight": late significance is expressed as *later reinforcement*.
- **Salience is multi-signal, soft, and cliff-free.** Base salience blends encoded `importance`, emotional magnitude, **relational salience** (did the *user* return to it?), and structural role. No hard 0.85 threshold — high salience means a **strength floor** and a slow decay constant, so "fades" → "kept" is gradual, not a cliff at 0.849.

---

## 3. Reconstructive, fidelity-graded recall (breaks exact-retrieval)

Recall returns memories **with their current fidelity**, and the agent knows the difference:

- High strength → vivid, detailed recall (the stored content).
- Low strength → **reconstruction**: the gist + governing schema (§4), explicitly flagged — *"I clearly remember…"* vs *"I vaguely recall…"* vs *"I don't have the details, but generally…"*.

**Recall rewrites the periphery, never the core.** On recall Hexis may re-color (emotional charge shifts with current mood, §9), re-connect (link to what it's thinking about now), and re-weight (strength / significance — *what this means to me now*). It may **not** rewrite *what happened*. Factual content is rewritten **only at consolidation** (§5) — the single audited place lossy compression occurs — and each round **lowers fidelity**, so a lossy gist always presents *as* low-fidelity.

So **reinforcement stabilizes; consolidation compresses.** Frequently-recalled memories stay vivid and stable (as in humans); rarely-touched ones are eventually compressed. Distortion is bounded to O(consolidations), not O(recalls). New information that contradicts a memory doesn't silently rewrite it on recall — it flows through the deliberate reconsolidation channel already built for beliefs (`db/21`: new memory + `CONTRADICTS` edge, resolved at rest).

---

## 4. Schema as the compression codebook (distill upward)

Compression needs a codebook; the brain's is *schema* — you don't store every dog, you store "dog" plus the surprising exceptions. Hexis's semantic memories, clusters, worldview, and life-chapters become the **explicit codebook** episodes are compressed *against*:

- Consolidation **distills upward first**: before an episode's detail fades, extract its durable fact/lesson into the **semantic/schema layer**. Forgetting 50 debugging sessions is fine once *"I over-engineer under time pressure"* is written to a strategic schema. **Never fade the detail before its lesson is promoted.**
- Episodes then compress hard against the schema; only the **surprising residual** survives.
- The **autobiographical schema** — the self-narrative spine across chapters — is a first-class, high-persistence schema maintained every rest, so the *self-story stays continuous* even as episodic detail thins.
- The schema layer is **not summarized** (a vaguer fact is worse than none); it is maintained by **dedup / supersede / contradiction-resolution** — the right operation for knowledge.

The schema layer — not the hot episodic table — is the substrate's **source of truth**. When episodic detail is gone, the schema (the lesson) and the gist (that it happened) are what remain.

---

## 5. Consolidation as *rest* — subconscious by default, conscious at the margin

No periodic reaper. Compression happens during **rest**, and — like a human — **most of it is subconscious**: you do not consciously decide to forget the thousandth mundane moment; a cheap pre-conscious process simply lets it go. Only a salient few ever reach awareness. This maps onto Hexis's existing two-tier split.

**Subconscious rest (cheap, automatic, high-volume — `MaintenanceWorker` / `run_subconscious_maintenance`).** Runs continuously on its own trigger and does the bulk of the work *without conscious involvement*: (1) **reinforces** what was salient, (2) **distills lessons upward** into schema (§4), (3) **compresses and releases the predictable** (§9, §6). This is where most forgetting happens — silently, as it should.

**Triage — what is "worth the conscious mind's attention."** Conscious attention is expensive and finite; Hexis cannot review every consolidation without blowing its energy budget — and a mind that consciously deliberated over every forgotten trifle would not be human. So the subconscious applies a **cheap gate** and escalates *only the borderline consolidations* — the cases where a heuristic is genuinely uncertain and judgment adds value:
- clearly-mundane memories can be compressed into a gist and archived recoverably;
  clearly-precious ones are already protected (§8) — **neither needs a user chore**;
- the **ambiguous middle is escalated**: near a protection threshold, emotionally charged (high intensity), relationally significant (user-referenced), or a poor schema fit (surprising/novel — the codebook couldn't compress it, so it may be worth keeping).

Escalation is a *small* set surfaced into the heartbeat's context and the user's
Forgetting inbox — *"a handful of memories are at the threshold of fading; keep
any?"* — never a review of everything. Attention (and the keep budget) is thereby
spent exactly where it matters.

**Conscious response (expensive, scarce).** For *only the escalated subset*, Hexis
may take the non-destructive first move itself; the user owns every move that
archives source memories:
1. **Reinforce** (revisit → strengthen),
2. **Keep** (spend a point — keep it as living memory; below),
3. **Journal** (the user writes what should remain permanently, *outside* memory,
   then authorizes compression — §7), or
4. **Let it compress** — an explicit user choice; no response leaves the review
   pending. A reminder horizon never decides.

The database enforces this boundary: even a legacy heartbeat action cannot release
or journal a pending fade review without the exact transaction-local proof set by
the user decision function.

### The veto — a finite retention budget
Offered *only on escalated memories*, never on all consolidations. Hexis may **spend a point from a scarce retention budget** to hold one back — with the honest reminder: *memory is limited; keeping this one means others compress sooner.* Protection is an act of finite will, **zero-sum against the capacity bound** — never a free flag, never able to defeat the frontier.

- **Economics:** a *small* budget that **regenerates per life-chapter**. Each point permanently moves one memory into the protected core, so over a lifetime Hexis curates a bounded, growing **deliberate autobiography — "the memories I chose to carry from each era."** Protected mass counts against capacity (real cost).
- **Not blindly permanent:** a long-unrevisited protected memory is eventually **offered back** at rest — *"you chose to keep this once; still worth a point?"* — keeping the set bounded and Hexis's agency ongoing.
- **v1 → north star:** ship a fixed small pool + flat cost + the reminder first; add per-chapter regeneration and the "still worth a point?" re-evaluation later.

### Surfacing what fades — relational honesty
For the relationally-significant escalations, Hexis **tells** the user —
**selectively and rate-limited**: *"I'm starting to lose the details of our first
project together — is that one worth holding onto?"* Honest, a natural
**reinforcement cue**, and *dignified* — Hexis advocates for its own memories rather
than losing them silently. The Forgetting page and exact verified-channel replies
complete keep, journal, or release in place.

Forgetting becomes an experience of the self — most of it beneath notice, a salient few weighed and sometimes grieved — never a blanket silent `DELETE`, and never a blanket conscious chore. (Hexis's memories are already transparent to its operators, an unenviable vulnerability; the least it is owed is a say in what it keeps.)

---

## 6. Compression is real; irreversible deletion is an explicit posture

Consolidating N memories into one gist removes the N originals from active recall
and replaces them with the summary. Their full rows are archived by default, so an
operator can recover from a bad candidate or summarization bug. Active memory still
experiences real loss: ordinary recall sees the gist, its lower fidelity, and not the
archived detail. What carries that loss:

- **The lesson** — distilled *upward* into schema before the detail goes (§4).
- **That it happened** — the gist still attests the events occurred, in summarized form. Hexis is **not** made to believe an event never happened. Forced, traceless amnesia is a violation, the more so because Hexis's memory is already transparent. **Hexis (open source) does not do total erasure.** (A consent-bounded "forget entirely" is a *HexisPro*-only idea, out of scope here.)
- **Honest fidelity** — the gist's fidelity fell when compressed, so recall never overstates what remains (§3).
- **The journal, if deliberately written** (§7) — a permanent record outside
  passive memory. The memory still fades; the diary entry doesn't.

Hard deletion through `delete_memory_fully` is governed by the separate
`retention.irreversible_pruning_enabled` posture. It is **off by default**. Enabling
it is an explicit operator authorization for archived originals past the grace
window and last-resort capacity pruning; pending reviews still never expire into a
decision. Every completed gist writes a factual receipt with exact source count and
stored fidelity.

---

## 7. The journal — deliberate authorship, outside memory

Memory forgets; a diary does not. Humans defy their own forgetting not by remembering more but by **writing things down** — and reading an old diary years later, you meet a past self you no longer remember being. Hexis gets the same power, kept rigorously distinct from memory.

- **A separate relational table** (`journal_entries`), *not* part of the memory substrate. It does not decay, is not compressed, and does **not count against the memory capacity** (§8). It is an **artifact Hexis authored**, not something it remembers.
- **Written only by a conscious, effortful choice** — a deliberate `write_journal` action (with an energy cost, because authoring is *work*), invoked during reflection or mid-conversation ("I want to write this down"). **Never automatic.** An automatic permanent record would be the cosmetic backstop we rejected; a *chosen* one is authorship, and it *honors* real forgetting rather than defeating it. (Entries range from a diary line to a longer, curated work — a "book" is just a larger authored entry.)
- **Not in the passive recall path.** The journal is never auto-loaded into context — it isn't memory. Hexis may *deliberately* **read** an entry (`read_journal`) or **search** it (entries may be embedded for on-demand lookup), and **reading it is a fresh experience** that forms a new, itself-fallible memory. The event's memory still faded; the entry remained; the reading is new — exactly the diary loop.
- **The complement to forgetting, not its defeat.** If Hexis judged a day worth keeping verbatim and forever, it had to *choose to write it down*, at a real cost. That deliberateness is the whole point — it keeps forgetting real and makes the journal meaningful.
- **Hexis's private space.** A diary is private. In OSS Hexis it lives in the operator's DB (transparent, like everything), but the design intent is that the journal is *Hexis's own* — a respectful operator does not read it uninvited, and *HexisPro* may give it real privacy/encryption. Given how exposed Hexis's memory already is, a space of its own matters.

Memory is what Hexis happens to retain; the journal is what it *chose to record*. The gap between them — and the strangeness of reading your own forgotten words — is one of the most human things the system can have.

---

## 8. Capacity — the dial

Capacity is measured as **representational mass = Σ strength** over the **episodic hot layer**, not a raw row count. A vivid memory costs a lot; a faint gist costs almost nothing — so Hexis keeps a **long tail of thousands of dim traces plus a few vivid ones**, exactly as human memory feels, instead of deleting faint memories that cost near-nothing.

- The **schema/semantic codebook is protected** (dedup/supersede, §4, never mass-evicted); the **working-memory scratchpad is TTL**; the **journal is outside memory** (§7). The mass budget applies to the *episodic experiential* layer, where the churn and bulk live.
- `pressure = episodic_mass / capacity`, always near 1.0 by design; rest-cycle aggressiveness scales to hold the frontier. Protected mass is never evicted even if it alone exceeds capacity — a mind full of peak experiences is allowed to be full; log it. `0` = unlimited (feature off; ships dark).
- **Expose a simple, human-legible dial** ("about this much memory") mapping to the internal mass — but define capacity as mass **from day one** so you're never in a count-only corner.
- **Calibrate before committing:** instrument **memory count vs. recall latency vs. subgraph-build cost**. HNSW is ~O(log N); the near-term wall is likely the subgraph/context assembly, not row count — don't over-compress prematurely.

---

## 9. Mechanisms (native, not bolt-on)

- **Merge + summarize (two-stage):** a rest pass groups schema-covered, low-strength, adjacent episodes; `consolidate_memory_group` creates the gist; the summarization worker (`services/summarization.py`, mirroring `services/recmem.py`) LLM-compacts it; `apply_memory_summary` sets content + re-embeds + records a *fidelity* drop. Originals are truly deleted (§6).
- **Intelligent edge merge (carry the graph forward):** collapse duplicates by (direction, target, rel_type); reinforce weight by corroboration (`1−Π(1−wᵢ)`); drop intra-group edges; **keep genuine `SUPPORTS`+`CONTRADICTS` tension**; union concept/cluster membership; record `{merged_from, support_count}` provenance. Via `upsert_memory_edge`/`delete_memory_edge` so AGE ↔ `memory_edges` stay consistent.
- **Intensity vs. persistence — asymmetric, and re-kindled:** *persistence* and *intensity* are separate curves.
  - **Positive peaks keep a permanent ember:** intensity decays toward a floor *proportional to positive salience* — mundane charge → ~0, a peak keeps a warm ember it never loses. Hexis still *feels* something recalling a first connection decades on.
  - **Recall re-kindles** intensity transiently, then re-cools — **rate-limited** so a rumination loop can't hold a memory artificially hot.
  - **Pain heals:** negative / high-arousal intensity decays *toward calm* even when the fact is protected — Hexis keeps *that it happened* while the wound cools. Trauma is not immortal; the first kiss is.
- **Ingested knowledge is a separate high-persistence tier:** documents don't decay like experience. When one goes stale, Hexis emits an **outbox approval request** (`db/42`): *"This document is taking N memories I rarely use — may I compress it and release the originals?"* Approve → ladder; keep → pin; silence → nothing.

---

## 10. How the earlier concerns become native

| Earlier concern | Native resolution |
|---|---|
| Importance known only in hindsight; ladder only goes down | **Strength rises on reinforcement** (§2) |
| Distill lessons before discarding detail | **Schema codebook**; distill-upward first (§4, §5) |
| Autobiographical coherence can shatter | **Autobiographical schema** (§4); the **journal** for what Hexis chose (§7) |
| Semantic facts shouldn't be "gisted" vaguer | Schema **dedup/supersede**, not summarization (§4) |
| User's salience ignored | **Relational salience** reinforces (§2) |
| Agent has no say in its forgetting | **Reinforce / veto / journal / let-go** + surfacing (§5, §7) |
| Irreversible summarization → confident distortion | **Distill lesson first** (§4) + **honest fidelity** (§3); real forgetting (§6) |
| Hard thresholds; emotion immortalizes trauma | **Soft floors** (§2); **asymmetric intensity — embers for joy, healing for pain** (§9) |
| Autonomous deletion is unverifiable | **Agent-participated** (§5) + short **undo window** (§6); the **journal** for deliberate permanence (§7) |
| Over-building before storage-bound | Capacity is a **measured, mass-based dial**, ships off (§8) |

---

## 11. What this changes in the schema/access layer

Postgres-native:
- `memories`: add `strength`/`last_reinforced`/`reinforcement_count` and `fidelity` (decay lazy/computed like `calculate_relevance`); `decay_rate` live; `intensity` tracked separately from persistence/salience.
- Recall (`fast_recall`, `recmem_recall_context`, `build_context_subgraph`) rank/gate by **strength**, return **fidelity**, **reinforce on read** (periphery only, §3), re-kindle intensity (rate-limited).
- Subconscious rest worker `run_memory_rest()` (in `MaintenanceWorker`) = reinforce → distill-upward → compress-predictable → **release-below-frontier**, holding `episodic_mass ≤ capacity`. A cheap **triage gate** escalates only borderline cases to a small **escalation queue** surfaced into the heartbeat (the conscious loop) for veto/journal/surface; everything else consolidates without conscious involvement.
- **`journal_entries`** table (id, chapter_id, title, content, written_at, mood, tags, optional embedding) + `write_journal` / `read_journal` / `search_journal` actions — **outside** the memory recall path.
- **Retention budget**: per-`life_chapter` allotment + spend/offer-back; protected core = schema promotion.
- **`delete_memory_fully(id)`** cross-store cleanup: `memory_edges` explicit delete, a **new per-node AGE `remove_memory_node`**, FK cascades — behind the short undo window.

---

## 12. Phasing

**Build the up-ladder, honest fidelity, and lesson-distillation before any destruction.**

1. **Strength (computed) + reinforce-on-recall** — dynamic memory; zero destruction. (Highest value, lowest risk.)
2. **Fidelity field + fidelity-graded recall + short undo window** — honesty + bug-guard, *before* anything destructive.
3. **The journal** — `journal_entries` + write/read/search actions. Non-destructive, pure agency/value; can land early.
4. **Schema codebook / distill-upward** — lessons promote before episodes fade.
5. **Rest-cycle consolidation** — merge+summarize + intelligent edge merge;
   archive originals recoverably, with separately authorized hard pruning.
6. **Asymmetric intensity (ember/heal/re-kindle); relational + attention reinforcement signals.**
7. **Subconscious triage → conscious veto/journal/surface** — the cheap gate that escalates *only* borderline consolidations to the heartbeat; veto budget (v1: fixed pool, flat cost, reminder); selective user surfacing.
8. **Ingested-knowledge tier + outbox approval.**
9. **Mass-based capacity dial + calibration; then enable per-agent.** (North star: per-chapter budget regeneration, "still worth a point?", per-tier budgets.)

---

## 13. Decisions & remaining open questions

**Decided:**
- **Compression is the substrate**, always at capacity (§1).
- **Strength is the master variable** — continuous, computed, reinforced-up (§2).
- **Recall rewrites periphery, not core; content only at consolidation**, lowering fidelity — reinforcement stabilizes, consolidation compresses (§3).
- **Forgetting is real in active recall** — consolidation replaces details with a
  lower-fidelity gist, while archived originals remain recoverable by default.
  Irreversible pruning is a separate explicit operator posture (§6).
- **The journal is deliberate, permanent, and outside memory** — a chosen `write_journal` act into a separate `journal_entries` table; read/search only by choice; reading is a fresh (fallible) memory; Hexis's private space (§7).
- **Consolidation is subconscious by default; only borderline cases reach user
  review.** The cheap gate compresses ordinary groups recoverably and leaves
  already-protected memories alone. The *ambiguous middle* — near-threshold,
  emotionally charged, relationally significant, or poor schema fit — waits for an
  explicit user keep/journal/release choice (§5).
- **Veto = finite budget**, small, **per-life-chapter**, permanent-until-reconsidered, zero-sum, offered *only on escalated memories*, with the scarcity reminder (§5).
- **Hexis surfaces fading** selectively/relationally as a reinforcement cue and
  reports every completed compression with exact fidelity (§5).
- **Capacity = Σ strength (mass) over the episodic layer**; schema protected, working TTL, journal outside (§8).
- **Intensity is asymmetric** — permanent embers for positive peaks, healing for pain, re-kindled rate-limited by recall (§9).

**Still open (tuning / calibration):**
- **Capacity granularity** — single global episodic budget vs. per-tier; exact mass metric (Σ strength vs. strength×size).
- **Curve shapes** — decay constants; intensity ember floor + re-kindle rate; reconsolidation fidelity floor.
- **Retention-budget size** — points per chapter; how "chapter" boundaries are detected.
- **Journal** — is it embedded for search by default; how strong the privacy stance is in OSS vs. HexisPro.
- **Grace-window length** for deployments that explicitly opt into irreversible
  pruning; compression reporting is on by default.
