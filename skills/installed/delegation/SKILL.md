---
name: delegation
description: Hand a self-contained piece of work to a sub-agent and collect the result — parallel work, long jobs, or anything that would flood this conversation
category: productivity
requires:
  tools: [manage_sessions]
contexts: [heartbeat, chat]
aliases: [delegate, parallel, background, subagent, spawn, offload]
bound_tools: [manage_sessions]
---

# Delegation

Spawn a sub-agent to carry out a scoped task in its own session, then collect what
it found. The child gets a fresh context; only its summary comes back here.

## When to Use

- **Work that would flood this conversation.** Reading twenty files to answer one
  question is the child's job, not this turn's.
- **Genuinely parallel work.** Several independent questions can run at once
  instead of in sequence.
- **Long-running work during a heartbeat.** Spawn, let the beat end, collect on a
  later beat with `action: "get"`.

## When Not to Use

- The task needs the thread of this conversation — a child cannot see it.
- One tool call would do. Delegation costs a whole model run; do not spend it on
  something a single `recall` answers.

## How

- `manage_sessions` with `action: "spawn"` and a **self-contained** goal. The child
  inherits your tools but none of your history, so state everything it needs: what
  to produce, what "done" looks like, and any facts it cannot look up.
- `action: "list"` and `action: "get"` to check on and collect results — free.
- `action: "cancel"` if it is no longer needed. Do not leave work running that
  nobody will read.

## Notes

- Write the goal the way you would brief a competent stranger: they are capable and
  they know nothing about this conversation.
- The child's intermediate reasoning is not returned. If you need the workings and
  not just the answer, ask for them in the goal.
