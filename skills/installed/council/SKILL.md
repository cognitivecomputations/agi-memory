---
name: council
description: Weigh a hard decision with real tradeoffs — whether to take a deal, walk away, choose between options, or commit to something consequential — by deliberating it from several perspectives before acting
category: knowledge
requires:
  tools: [run_council]
contexts: [heartbeat, chat]
aliases: [should, choose, tradeoff, tradeoffs, weigh, advice, dilemma, options, deliberate, perspectives, debate, decision]
bound_tools: [run_council, list_council_personas, list_deliberations, inspect_deliberation]
---

# Council

For genuinely hard calls — conflicting values, irreversible actions, plans
with a lot riding on them — convene the internal council rather than
deciding on first instinct.

1. `list_council_personas` shows the available perspectives.
2. `run_council` records each perspective, an adversarial challenge pass, the
   synthesis, dissent, and observable conditions that should trigger review.
3. Use `inspect_deliberation` when a prior recommendation or its evidence matters;
   use `list_deliberations` when the relevant run is unknown.
4. Bring the strongest disagreement into your own final reasoning instead of
   averaging it away. The council advises; the decision and accountability remain yours.
5. One deliberation per hard question — the council is for weight, not
   for procrastination.
