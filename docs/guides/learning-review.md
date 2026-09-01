<!--
title: Review What Hexis Learned
summary: Approve, correct, or forget grounded weekly learning changes
read_when:
  - "You want to inspect what Hexis learned this week"
  - "You want learning without silent behavior changes"
-->

# Review What Hexis Learned

Hexis can turn its existing opt-in background improvement pass into one weekly
learning diff. The review can contain:

- new semantic beliefs;
- new procedures;
- revised strategies; and
- proposed reusable skills.

The model decides whether the week contains enough meaningful change to merit
attention. It can select only IDs from the bounded database-supplied change
window; the displayed content, type, source, trust, and confidence are read back
from PostgreSQL rather than accepted from generated prose.

## Enable the weekly pass

The pass examines bounded recent conversation evidence with the configured LLM,
so it remains opt-in:

```bash
hexis skills enable
```

The default cadence is seven days and requires evidence from more than one
session. A week with only routine residue produces no message.

## Respond in place

When enough changed, Hexis places one digest in the outbox and the dashboard
**Learning review** page. Every item has three choices:

- **Approve** keeps a learning active. A proposed skill enters the existing
  ownership-checked application queue only after this choice.
- **Correct** records explicit user testimony. Belief corrections create a new
  version, file the old and new versions in the contradiction ledger, resolve
  the user-selected correction, and leave the prior version queryable in
  Memory history.
- **Forget** removes the learning from active recall while retaining the
  accountable historical window. If the memory is protected or load-bearing,
  the first click changes nothing and Hexis asks for a second confirmation.

From a verified private operator channel, reply with the exact item code shown
in the digest:

```text
approve A1B2C3D4
correct A1B2C3D4: The planning call is on Friday.
forget A1B2C3D4
```

Only the configured operator can consume these replies as control-plane
decisions. Other messages remain ordinary conversation.

## Failure and recovery

Skill application never loses the approval or overwrites an unrelated skill.
The worker uses the same proposal provenance and authoring validation as manual
review. If writing fails, the item shows the error and remains retryable after a
bounded delay. Memory corrections and forgetting are transactional: a failed
transition leaves the original learning untouched.
