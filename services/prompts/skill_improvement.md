# Skill Improvement Review

Review the supplied recent experience for one repeated, proven operational workflow that would make future behavior clearer and more consistent.

Return exactly one JSON object with `proposal`, `automation_suggestion`, and
`learning_review` fields. Set a field to `null` when the evidence does not
support it. Never force a proposal or a review.

When proposing, use this shape:

```json
{
  "proposal": {
    "name": "lowercase-kebab-name",
    "description": "One concise sentence describing when to use it",
    "content": "Substantive Markdown instructions covering when, method, verification, and pitfalls",
    "category": "other",
    "contexts": ["chat", "heartbeat"],
    "bound_tools": [],
    "requires_tools": [],
    "mode": "create",
    "rationale": "Why the repeated evidence supports this reusable workflow",
    "confidence": 0.0
  },
  "automation_suggestion": null,
  "learning_review": null
}
```

`learning_changes.candidates` are durable memory changes from the bounded review
window. Decide whether there is enough meaningful change to ask for attention.
When there is, select only supplied memory IDs; Hexis will derive the displayed
content, kind, trust, and source from the database rather than model prose:

```json
{
  "proposal": null,
  "automation_suggestion": null,
  "learning_review": {
    "should_review": true,
    "summary": "A concise first-person account of the meaningful pattern across these changes.",
    "memory_ids": ["uuid-from-learning-changes"]
  }
}
```

When the same user ask appears at least three times and a recurring prompt
would genuinely help, `automation_suggestion` may instead (or also) use this
shape:

```json
{
  "proposal": null,
  "automation_suggestion": {
    "title": "Weekly standings check",
    "rationale": "You asked for the standings on three separate Mondays, so a Monday prompt could save you from remembering to ask.",
    "pattern": "review league standings every monday",
    "confidence": 0.9,
    "evidence_unit_ids": ["uuid-1", "uuid-2", "uuid-3"],
    "task_spec": {
      "action": "create",
      "name": "Weekly standings check",
      "schedule": "weekly:monday:09:00",
      "action_kind": "queue_user_message",
      "message": "Standings check — open Hexis and ask for this week's standings.",
      "delivery_mode": "outbox"
    }
  }
}
```

Rules:

- Require evidence from more than one session and repeated successful or corrected execution.
- Encode a general method, never a one-off fact, specific conversation, private detail, credential, secret, token, or API key.
- Use only category, context, and tool values present in the supplied catalog.
- Prefer `update` only for an existing skill explicitly marked as Hexis-managed. Never update user-authored or bundled skills.
- Keep tool access narrow. Empty tool lists are valid.
- Confidence represents evidence strength, not writing quality. Use a high value only for clear recurrence.
- The proposal will be shown for explicit review. It will not be applied automatically.
- A learning review is warranted only when the selected changes form a useful
  weekly diff; routine transcript residue, duplicates, and isolated trivia are
  not enough. Use only supplied memory IDs and never rewrite their content.
- The user can approve, correct, or forget every selected learning. A correction
  becomes explicit user testimony and preserves the prior version in history.
- An automation needs at least three distinct matching source unit IDs from the supplied evidence. A broad topic appearing three times is not enough; the recurring intent and useful cadence must match.
- Automation task specs must be valid `manage_schedule` create arguments. Use `queue_user_message` for an honest prompt; do not claim a fixed scheduled message will dynamically run a skill or inspect an integration.
- Automation suggestions are inert until the user explicitly accepts them, and a dismissal is final for that recurring pattern.
