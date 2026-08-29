---
name: skill-authoring
description: Author or revise reusable Hexis skills from repeated successful workflows
category: system
requires:
  tools: [propose_skill, author_skill, list_skill_proposals, review_skill_proposal]
contexts: [heartbeat, chat]
aliases: [skill, skills, procedure, author, teach, learn]
activation_phrases: [create a reusable skill, create a skill, author a skill, revise a skill]
bound_tools: [propose_skill, author_skill, list_skill_proposals, review_skill_proposal, list_skills, use_skill, create_tool]
---

# Skill Authoring

Use this skill when a repeated workflow, hard-won lesson, or stable procedure should become reusable future behavior. A skill is not a memory note; it is an operational method that can be activated later and can unlock the tools it needs.

## When to Use

- The same multi-step procedure has worked more than once.
- A user explicitly asks you to make a behavior reusable.
- A correction reveals a better standard operating procedure.
- A project develops a durable local convention that should guide future work.
- You notice a capability gap that should become an operator-reviewable skill,
  even before there is enough cross-session evidence for background review.

## Method

1. Check existing skills with `list_skills` before creating a new one.
2. Prefer updating an existing skill when the new behavior is a refinement of the same workflow.
3. Write the skill as concise operating instructions:
   - when to use it,
   - what to do,
   - what to avoid,
   - which tools it should unlock.
4. Bind only the tools the skill genuinely needs. Do not add broad filesystem, shell, or browser access unless the procedure requires it.
5. Prefer `propose_skill` for on-demand growth. It creates a pending review item
   and writes no skill file. This is the normal path when you discover a reusable
   gap while working.
6. Use `author_skill` directly only when the user has explicitly approved
   immediate skill-file creation/update in the current exchange.
7. Never try to replace a user-authored skill. `author_skill` updates only files
   carrying Hexis ownership provenance; if ownership cannot be verified, choose
   a new name or leave the exact manual-edit step to the user.
8. Use `list_skill_proposals` to inspect background or on-demand review results. Apply,
   reject, or reopen one only through `review_skill_proposal`; applying always
   requires explicit approval and preserves confidence plus source lineage.

## Quality Guidelines

- Keep a skill focused. If it covers two unrelated workflows, split it.
- Do not encode secrets, credentials, private user data, or one-off facts in a skill.
- Do not directly author a live skill from a single unverified attempt unless the
  user explicitly approves it. Use `propose_skill` instead.
- Skills should make future behavior clearer and cheaper; if it would add more prompt weight than judgment, keep it as ordinary memory instead.

## Growing New Tools (self-extension)

When a workflow needs a capability no existing tool provides, you can build
it yourself with `create_tool`: write a ToolHandler subclass; it is
validated, registered immediately (no restart), and persisted for future
sessions. Then complete the growth loop:

1. Author the tool with `create_tool`.
2. Bind it into one of your own skills with `author_skill` (or by updating
   an existing agent-authored skill) — an unbound tool is a hand you
   cannot use next session.
3. Use it, and patch the skill when experience improves the method.

Your authored tools run with the same permissions as your other tools, and
every tool or skill you grow is visible to the operator (change journal +
inbox notice). `tools.allow_dynamic` is the operator's master switch for
this capability.
