"""Portable approval controls that Slack renders as Block Kit buttons."""

from __future__ import annotations

import json

from channels.presentation import (
    ActionButton,
    ActionsBlock,
    ContextBlock,
    MessagePresentation,
    TextBlock,
)


def _action_value(request_id: str, decision: str) -> str:
    return json.dumps(
        {"approval_request_id": request_id, "decision": decision},
        separators=(",", ":"),
    )


def build_approval_presentation(
    *,
    approval_request_id: str,
    message: str,
    interactive: bool = True,
) -> MessagePresentation:
    """Build one exact approve/deny decision surface.

    The action identifiers and values survive the transactional outbox and its
    recovery ledger. Slack renders buttons; other surfaces render the labels as
    text without pretending they are interactive.
    """
    code = approval_request_id.replace("-", "")[:8]
    blocks = [
        TextBlock(message),
        ContextBlock(
            f"Request {code} expires automatically. Approval is exact and one-shot."
        ),
    ]
    if interactive:
        blocks.append(
            ActionsBlock(
                block_id=f"operator_approval:{approval_request_id}",
                actions=(
                    ActionButton(
                        action_id="operator_approval_approve",
                        label="Approve",
                        value=_action_value(approval_request_id, "approve"),
                        style="primary",
                    ),
                    ActionButton(
                        action_id="operator_approval_deny",
                        label="Deny",
                        value=_action_value(approval_request_id, "deny"),
                        style="danger",
                    ),
                ),
            )
        )
    return MessagePresentation(
        title="Protected action needs approval",
        tone="warning",
        blocks=tuple(blocks),
    )
