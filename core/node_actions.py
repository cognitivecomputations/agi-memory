"""One authoritative capability vocabulary for companion nodes."""

from __future__ import annotations


NODE_INVOCATION_ACTIONS = frozenset(
    {
        "system.run",
        "screen.capture",
        "apple.reminders.list",
        "apple.reminders.create",
        "apple.notes.search",
        "apple.notes.create",
        "apple.calendar.list",
        "apple.calendar.create",
        "apple.shortcuts.list",
        "apple.shortcuts.run",
        "onepassword.items",
        "onepassword.copy",
    }
)

NODE_CAPABILITIES = NODE_INVOCATION_ACTIONS | {"audio.wake"}
MAX_NODE_CAPABILITIES = len(NODE_CAPABILITIES)

APPLE_NODE_ACTIONS = frozenset(
    action for action in NODE_INVOCATION_ACTIONS if action.startswith("apple.")
)
ONEPASSWORD_NODE_ACTIONS = frozenset(
    action for action in NODE_INVOCATION_ACTIONS if action.startswith("onepassword.")
)
