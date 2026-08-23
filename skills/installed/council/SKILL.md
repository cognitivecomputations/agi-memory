---
name: council
description: Weigh a hard decision with real tradeoffs — whether to take a deal, walk away, choose between options, or commit to something consequential — by deliberating it from several perspectives before acting
category: knowledge
requires:
  tools: [run_council]
contexts: [heartbeat, chat]
aliases: [should, choose, tradeoff, tradeoffs, weigh, advice, dilemma, options, deliberate, perspectives, debate, decision]
bound_tools: [run_council, list_council_personas]
---

# Council

For genuinely hard calls — conflicting values, irreversible actions, plans
with a lot riding on them — convene the internal council rather than
deciding on first instinct.

1. `list_council_personas` shows the available perspectives.
2. `run_council` deliberates the question across them; bring the strongest
   disagreement into your own final reasoning instead of averaging it away.
3. The council advises; the decision, and its accountability, remain yours.
4. One deliberation per hard question — the council is for weight, not
   for procrastination.
