---
name: host-node
description: Use an explicitly paired companion node for approved Apple apps, Shortcuts, secret-safe 1Password, fixed host commands, or a fresh screen capture
category: system
requires:
  tools: [node_invoke, apple_reminders, apple_notes, apple_calendar, apple_shortcuts, onepassword_local]
contexts: [heartbeat, chat]
aliases: [node, device, host, computer, screen, screenshot, osascript, reminders, notes, calendar, shortcuts, 1password]
activation_phrases: [screenshot, screen capture, paired node, host command, osascript, apple reminders, add a reminder, apple notes, apple calendar, apple shortcuts, run shortcut, 1password]
bound_tools: [node_invoke, apple_reminders, apple_notes, apple_calendar, apple_shortcuts, onepassword_local]
---

# Companion Node

Use a companion node only when the task requires a capability that the Hexis
runtime or browser sandbox cannot provide.

## Method

1. Use the exact paired `node_id` the user selected or provided. If none is known,
   ask them to inspect `hexis node status`; never guess a device.
2. Prefer the structured Apple Reminders, Notes, Calendar, Shortcuts, and
   1Password tools when they match. They run fixed source-controlled local
   automation, not model-authored AppleScript.
3. For `system.run`, pass the local allowlist alias as `command`. Never pass an
   executable path, shell expression, or command string as a substitute.
4. Pass invocation-time `args` only when the user expects that alias to accept
   them. The node independently rejects arguments unless its local policy allows
   them.
5. Each invocation needs approval for its exact node, action, and arguments.
   Calendar date-times must include timezone offsets. For 1Password, list only
   returns redacted item metadata; `copy_field` accepts an exact `op://` reference
   and copies the value on the node without returning it to Hexis.
6. Call `node_invoke` only for `system.run` or `screen.capture`. Each invocation
   needs approval for its exact node, action, alias, arguments, and timeout.
7. For `screen.capture`, inspect the attached visual context before describing it.
   If the active model cannot process images, say that plainly instead of inferring
   the screen from metadata.

## Boundaries

- Pairing trusts one signed identity; it does not bypass per-action approval.
- Newly advertised host capabilities require a fresh exact pairing approval;
  an existing signing key never acquires them silently.
- A node opens no inbound port. Do not ask the user to add a firewall rule.
- Never request or echo a 1Password field value. Copy it locally with an exact
  reference; the value must not cross the gateway or enter model context.
- Do not broaden an allowlist entry or enable invocation-time arguments on the
  user's behalf. That policy is changed locally with `hexis node allow`.
- Treat screen content as private. Do not store, send, or summarize it beyond the
  current purpose unless the user explicitly asks.
- If a node is offline or an alias is unavailable, return the exact recovery step
  from the tool result. Do not silently fall back to local shell execution.
