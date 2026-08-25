# Hexis Roadmap

**The single ordered list.** What to work on, in order, and what each thing is for.
Open this one; the others are reference.

| Document | Answers |
|---|---|
| **`ROADMAP.md`** *(this file)* | **What next?** |
| `PLAN.md` | *Why does this item exist?* — Part I strategy (§S1–S7), Part II the gaps and evidence (§1–§14) |
| `docs/_archive/audit-2026-07-29.md` | The July 2026 audit, reconciled. Evidence, not a plan |

Section references like §14.1 point into `PLAN.md`.

---

**Phases, each with a reason it precedes the next.** An earlier flat list had been
patched five times and the order had drifted — batching, a cost paid every day, sat at
the bottom while a scaling cliff sat near the top.

## Phase 0 — Safety · *mostly shipped*

Things the system can currently do harm with. Nothing else matters while these are open.

| Item | Effort | Status |
|---|---|---|
| Approval gate fails closed (§11.5) | ~1h | **done** — `a306dda` |
| Tier 0 reachability (§0.1–0.6) | ~2d | **done** — `a306dda`, defaults-only 7/10 → 2/11 |
| `execute_code` `requires_approval=True` (§12.1) | — | **done** — `a306dda` |
| **`execute_code` sandbox (§12.1)** | **~2d** | **owed.** The flag stops it unattended; nothing yet contains it when approved |

## Phase 1 — Stop paying for nothing · ~3.5d · **shipped 2026-08-23**

Independent of each other, each under two days, and **every item below runs cheaper
and faster once they land** — including our own development loop. This is why they
precede the feature work rather than following it.

| Item | Effort | Why now |
|---|---|---|
| ~~Prompt caching (§14.4)~~ | ~1d | **done** — 6,600-token prefix cacheable across OpenAI, Anthropic, and Gemini 2.5+ |
| ~~Batch the per-item LLM loops (§13.3·B2)~~ | ~2d | **done** — connector cognition batched; summarization deliberately not |
| ~~Tool-catalog sync once, not per call (§14.2)~~ | ~0.5d | **done** — `3c4c7d3` |

## Phase 2 — Correct before bigger · ~5.5d · **shipped 2026-08-23**

Debt that gets *multiplied* by everything built on top of it. Adding twenty skills to
a selector that scores word overlap means twenty more skills that do not activate.

| Item | Effort | Why here |
|---|---|---|
| ~~Semantic skill selection (§13.3·A)~~ | ~2d | **done** — z-score gate + identifier backstop; lexical is fallback only |
| ~~Dead heartbeat actions (§9.1)~~ | ~0.5d | **done** — three, not seven; retired as redundant |
| ~~`is_group` on all seven adapters (§12.2)~~ | ~1d | **done** — four were missing; two already had the signal unnamed |
| ~~Port `capability_probe` + `tool_surface_audit` (§11.4·8)~~ | ~3d | **done** — continuous worker/context/tool snapshots + immutable surface decisions |

## Phase 3 — Become useful · ~10d

The first phase a user would notice. Ordered so each makes the next more valuable.

| Item | Effort | Why in this order |
|---|---|---|
| ~~Wave A everyday skills (§4)~~ | ~2d | **done** — six consent-safe workflows over existing core and optional plugin tools |
| Port `operator_approval` + Slack actions (§11.4·4) | ~2d | makes Phase 0's fail-closed *livable* — approve from a phone |
| Automation suggestions (§1) | ~2d | the agent starts proposing instead of waiting |
| `ask_user` (§2) | ~3d | it stops guessing when it does not know |
| Port `voice_notes` (§11.4·7) | ~1d | voice in, no client needed |

## Phase 4 — Reach · ~7.5d

| Item | Effort | Note |
|---|---|---|
| Build off the install path (§6.1) | ~0.5d | no more failed installs |
| Tailscale/HTTPS documented (§8.1) | ~1d | **ships with the PWA or not at all** |
| PWA layer on `hexis-ui` (§3a) | ~3d | installable client, push, mic |
| Deterministic image build (§6.2) | ~2d | a slow index stops meaning a wrong one |

## Phase 5 — Depth · ~13d

The Part I thesis (§S4). Mostly ports (§11.4), because Alex already built them.

| Item | Effort | Note |
|---|---|---|
| Drop `<> zero_vec`; enforce at write (§14.1) | ~1d | **first in this phase** — memory volume grows from here |
| Port `retention` + `scene_consolidation` + `incubation` (§11.4·1) | ~4d | forgetting, consolidation, spontaneous recall |
| Port `memory_supersessions` (§11.4·2) | ~2d | unblocks bitemporal recall |
| Port `belief_propagation` (§11.4·3) | ~2d | contradiction-as-event, plumbing half |
| Port `operator_policy_corrections` (§11.4·5) | ~2d | never say it twice |
| Appraisal emits emotion families (§13.3·C) | ~0.5d | retires the SQL emotion regexes |
| Connector cognition: LLM-first (§13.3·B) | ~1d | retires `_URGENT_TERMS` / `_IMPORTANT_TERMS` |

## Phase 6 — Outbound · ~14d

The riskiest thing in the plan, deliberately last of the feature work. It messages
real people, and it is gated behind knowing who is in the room (Phase 2) and being
able to approve from a phone (Phase 3).

| Item | Effort | Note |
|---|---|---|
| Goal origin flag (§10.4) | ~1d | prerequisite for the permission slip |
| Port `inbound_disposition` (§11.4·6) | ~2d | the inbound half, policy already in SQL |
| Contact points + purpose gate + STOP (§10) | ~10d | **ships whole or not at all** |
| Action/tool gate reconciliation (§9.6) | ~2d | the two gates stop disagreeing |

## Phase 7 — Long tail

Real work, no urgency. Ordered by ratio, not by ambition.

`hexis-node` + pairing (§3b, ~1w) · heartbeat cadence and economy (§9.2–9.4, ~1w) ·
port `deliberation` (§11.4·9, ~4d) · Wave B skills (§4, ~1d each) · workers as host
services (§6.3, ~1w) · `hexis tunnel` + exposure posture (§8.2–3, ~1w) · voice out,
talk, wake (§5.2–4, ~3w) · execution backends (§7, ~2w)

## The shape of it

**Phases 1–3 are about three weeks** and take Hexis from "architecturally interesting"
to "worth having around." Phases 4–6 are another six, and are where it becomes
something you would put in front of another person.

Two rules the phase boundaries encode:

1. **Fix multipliers before adding to them.** A selector that mis-activates, a gate
   that disagrees with itself, and a per-item LLM loop all get worse in proportion to
   what is built on them.
2. **Nothing that messages a third party ships before everything that makes it safe.**
   Phase 6 depends on Phase 2 and Phase 3, and §10 ships whole — a purpose gate
   without a budget still floods, and a disclaimer promising an opt-out that is not
   wired is worse than sending nothing.

**Nine of these are ports, not builds** (§11.4), and four *replace* work this plan had
costed as new: `voice_notes` for §5.1, the retention trio for §S4.6,
`operator_policy_corrections` for §S4.5, and `capability_probe` for Tier 0's
instrumentation. Ports are cheaper than builds but not free — each is a Python module
plus SQL plus a migration (§11.6), so they are costed at 1–4 days, not at zero. Prefer
taking few properly over many partially.

**A note on the HTTPS day.** It is worthless alone and the PWA is impossible without
it, so Phase 4 ships those two as one change rather than as two items.

# Definition of done

Not "the tests pass." Per `HEXIS_EXPERIENCE_BAR.md` #7, drive the real path:

- A fresh install on a clean machine reaches a **beating heartbeat** with one command
  and no build.
- The ten-request probe in Tier 0 is a **regression test**: each request activates a
  skill that can actually serve it, and no request lands on `core-memory` alone.
- Every action in `heartbeat.allowed_actions` has a handler — asserted by *executing*
  each one, since the source scan that first counted them was wrong — and the action
  distribution over a hundred beats is logged and reviewed, so a dead or never-chosen
  action is visible rather than inferred.
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
- No code path executes model-authored code without both a sandbox and a gate, and the
  agent is told who it is speaking with on every turn — never inferring the principal
  from tone.
- Every invariant this plan relies on is asserted at startup or measured continuously
  — no declared flag, computed checksum, offered action, or sending tool is left with
  nothing enforcing it.
- Every one of those is refusable, and refusing it once means never being asked again.
