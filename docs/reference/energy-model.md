<!--
title: Energy Model
summary: Energy budget mechanics, intent classes, and cost philosophy
read_when:
  - "You want to understand the energy budget"
  - "You want to know why actions have costs"
section: reference
-->

# Energy Model

The energy system makes action **intentional**, not efficient.

## Philosophy

Energy represents the *situational cost* of acting in the world:

- **Irreversibility** -- can this be undone?
- **Social exposure** -- does this affect others?
- **Commitment** -- does this bind the agent to a course?
- **User attention** -- does this demand someone's time?
- **Identity impact** -- does this shape who the agent is?

Energy does **not** represent compute cost, latency, API pricing, or system resources. A tweet costs almost nothing computationally but costs a lot in identity exposure and irreversibility. That asymmetry is the point.

**Knowing is cheap. Acting on the world -- especially publicly -- should feel expensive.**

## Core Mechanics

- Energy regenerates from elapsed wall time when a heartbeat starts. The base rate is 10/hour.
- `heartbeat.max_energy` is the normal reserve (20 by default), not a hard storage cap.
- Unused energy banks up to three times the reserve by default. Surplus above the reserve decays with a 12-hour half-life.
- Ordinary heartbeats may spend the normal reserve. A heartbeat with actionable backlog may draw up to two reserves, bounded by what is actually banked.
- Actions consume energy at execution time. Agentic tool spend is deducted once when the beat finalizes.
- The agent may rest to preserve energy. The budget remains global (no separate pools).

### Outcome Gradient

The prior completed beat changes the next regeneration rate. A beat with no
durable result gets the configured floor multiplier (0.75 by default). Durable
memories, resolved contradictions, advanced goals, and explicit thanks from a
verified operator increase its outcome score; the multiplier is capped at 1.5.
Tool receipts and outcome signals are stored in Postgres, so a model-written
summary cannot award itself credit.

Operator thanks only credit a recent heartbeat that actually made proactive
contact. Untrusted channel participants and vague positive language do not affect
the economy.

### Adaptive Cadence

The next heartbeat is scheduled from the live maximum drive-urgency ratio. With
the defaults, a quiet agent waits 90 minutes; increasing urgency shortens that
toward a 15-minute floor. `should_run_heartbeat()` honors the resulting
`next_heartbeat_at`, so the worker remains a stateless poller.

### Base Cost Dimensions

| Dimension | Weight |
|-----------|--------|
| Consequence magnitude | High |
| Reversibility | High |
| Social/identity exposure | High |
| Attention demanded from others | Medium |
| Commitment implied | Medium |

## Intent Classes

Every action is tagged with one intent class:

| Intent | Meaning | Examples |
|--------|---------|----------|
| **cognition** | Internal reasoning, reflection | reflect, synthesize |
| **exploration** | Information gathering | inquire_shallow, inquire_deep |
| **maintenance** | Memory upkeep, structural coherence | maintain, connect |
| **social** | Direct interaction with user or others | reach_out_user, reach_out_public |
| **meta** | Rest, self-regulation, pause, termination | rest, pause_heartbeat, terminate |

Intent classes do **not** change costs or impose separate budgets. They exist for awareness and reflection. The agent sees summaries like:

```
Energy spent this heartbeat:
- cognition: 6
- exploration: 3
- social: 0
```

This lets the agent notice patterns: "I've been thinking but not reaching out." "I've been exploring but not consolidating."

## MCP Tool Cost Assignment

Discovered MCP tools are assigned costs automatically:

| Category | Default Cost | Heartbeat Allowed |
|----------|-------------|-------------------|
| Read / search | 0.5-1.0 | Yes |
| Draft / create local | 1.0-2.0 | Yes |
| Modify / update | 2.0-3.0 | Context-dependent |
| Send private | 3.0-5.0 | No (default) |
| Send public | 6.0-8.0 | No |
| Delete / destroy | 5.0-7.0 | No |

### Context Modifiers

```
effective_cost = base_cost x context_multipliers
```

- **First use of tool**: x1.5 (uncertainty premium)
- **High error rate**: x1.5 (unreliability)
- **Late night social action**: x2.0 (don't wake people)
- **Recent similar action**: x1.2 (diminishing returns)

Multipliers are bounded -- the agent cannot game costs to zero.

## Cost Discomfort

When the agent repeatedly considers an action, believes it's high-value, but defers due to cost -- that's cost discomfort. It's not an error; it's a meaningful internal experience.

During reflection, the agent may record: "I have repeatedly deferred reaching out despite high confidence it would help." These become episodic and strategic memories.

## Cost Change Proposals

The agent chooses actions. The system chooses costs. This separation is deliberate -- if the agent could rewrite costs, it would minimize costs for things it wants to do. That's not agency; it's self-hacking.

But the agent can:
1. Notice costs feel wrong
2. Accumulate evidence over multiple heartbeats
3. Propose changes with justification
4. Have proposals reviewed by the user

Proposals are arguments, not instructions. Constraints:
- Must include evidence
- Suggest ranges, not exact values
- No immediate effect
- Public/destructive actions have hard cost floors

## Safety Invariants

1. The agent cannot directly modify energy costs
2. Costs do not change automatically
3. Public or destructive actions are never cheap (hard floor)
4. Heartbeat autonomy is more restricted than chat
5. Energy remains a single, shared bank and budget (no fragmentation)
6. Outcome credit comes from durable receipts or verified operator feedback, never self-report

## Related

- [Tools Reference](tools.md) -- complete energy cost table
- [Heartbeat System](../concepts/heartbeat-system.md) -- how energy fits into the heartbeat
- [Tools Configuration](../guides/tools-configuration.md) -- overriding costs
