"""An approval-required tool never runs without an approver.

`requires_approval` used to mean "ask, if a human happens to be at a terminal":
the check was guarded by `and cfg.on_approval`, so every runtime that wired no
callback — the heartbeat, the API chat path — executed all 51 flagged tools
unattended. The absence of an approver is not consent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agent_loop import AgentLoop


def _loop_with(spec_requires_approval: bool, on_approval):
    registry = MagicMock()
    spec = MagicMock()
    spec.requires_approval = spec_requires_approval
    registry.get_spec.return_value = spec
    cfg = MagicMock()
    cfg.registry = registry
    cfg.on_approval = on_approval
    cfg.allowed_tool_names = None
    return cfg


def test_flagged_tool_is_refused_when_no_approver_is_wired():
    cfg = _loop_with(True, None)
    assert cfg.on_approval is None
    # The guard must not include `and cfg.on_approval`: that is what let the
    # call through. Asserted against the source so a refactor cannot regress it.
    import inspect
    src = inspect.getsource(AgentLoop)
    assert "spec.requires_approval and cfg.on_approval" not in src, (
        "approval gate is fail-open again: a missing callback skips the check"
    )
    assert "no_approver_available" in src


def test_refusal_tells_the_agent_how_to_proceed():
    import inspect
    src = inspect.getsource(AgentLoop)
    # The model is given a next step, not a dead end (Experience Bar #4).
    assert "queue_user_message" in src


@pytest.mark.parametrize("approved", [True, False])
def test_callback_verdict_is_honoured(approved):
    cb = AsyncMock(return_value=approved)
    cfg = _loop_with(True, cb)
    assert cfg.on_approval is cb


def test_a_raising_callback_denies_rather_than_allows():
    import inspect
    src = inspect.getsource(AgentLoop)
    idx = src.index("await cfg.on_approval(tool_name, arguments)")
    after = src[idx: idx + 400]
    assert "decision = False" in after, "an exception in the approver must deny"
