---
name: council
description: Convene an internal council of perspectives to deliberate a hard question before acting
category: knowledge
requires:
  tools: [run_council]
contexts: [heartbeat, chat]
aliases: [decide, decision, decisions, should, choose, tradeoff, tradeoffs, weigh, advice, dilemma, options]
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
