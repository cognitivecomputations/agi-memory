---
name: weekly-review
description: Review the past week across goals, tasks, calendar, journal, and commitments, then propose the next week
category: productivity
requires:
  tools: [recall, manage_goals, manage_backlog]
contexts: [chat]
aliases: [weekly, review, week, reflection, reset]
bound_tools: [recall, remember, manage_goals, manage_backlog, calendar_events, email_search, email_read, read_journal, search_journal, todoist_list_tasks, todoist_complete_task, todoist_create_task, asana_list_projects, asana_create_task]
---

# Weekly Review

Create an evidence-based review of what happened, what remains open, and what deserves attention next. The review should help the user decide; it should not silently rewrite their systems.

## Workflow

1. **Fix the window.** State the date range and time zone. Default to the current calendar week through today, and adjust when the user means a rolling seven days or a different work week.
2. **Collect evidence.** List goals and backlog items, recall recent outcomes and commitments, and inspect relevant calendar events. Search email only for targeted open loops; do not turn a review into an inbox crawl. Read journal entries only when the user asks to include the journal.
3. **Reconcile planned and actual.** For each active goal, distinguish completed evidence, progress, blockers, stale assumptions, and work that merely remained scheduled. Do not count an event or sent message as success without supporting context.
4. **Surface the week.** Summarize wins, decisions, missed or moved commitments, unresolved conversations, workload patterns, and lessons. Keep facts separate from interpretations.
5. **Propose next week.** Recommend a small set of priorities, concrete next actions, calendar pressure points, and items to defer or drop. Explain the tradeoff behind each recommendation.
6. **Let the user choose.** Present proposed goal changes, task completions, new tasks, and journal or memory writes as a reviewable plan. Apply only the items the user selects, in the destination they select.
7. **Close the loop.** Report the final decisions and remaining unknowns. If requested, store a concise review with the date window and source attribution.

## Optional Integrations

- If the user asks to include GitHub work, activate `github-issues` when configured rather than guessing from local memories.
- Todoist and Asana may enrich task coverage when their plugins are enabled. Their absence must not block a review based on Hexis goals and backlog.

## Control and Failure Boundaries

- Never complete, delete, reprioritize, or reschedule work merely because it looks stale; each mutation needs an explicit user choice.
- Never read the permanent journal by default; it is a deliberate source, not passive context.
- Missing integrations reduce coverage, not honesty. State which sources were checked and continue with what is available.
