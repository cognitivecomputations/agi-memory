from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.operator_policies import ManageOperatorPoliciesHandler
from services import operator_policy_corrections as service
from services.chat import _operator_policy_capture_notice, _trusted_operator_turn

pytestmark = pytest.mark.asyncio


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_args):
        return False


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


async def test_capture_returns_db_owned_result_and_passes_identity():
    conn = SimpleNamespace(
        fetchval=AsyncMock(
            return_value=json.dumps(
                {
                    "captured": True,
                    "outcome": "created",
                    "policy_key": "operator.standing.abc",
                }
            )
        )
    )
    result = await service.capture_operator_policy_correction(
        _Pool(conn),
        channel_type="slack",
        channel_id="C1",
        sender_id="U1",
        sender_name="Owner",
        text="Always cite sources.",
        is_operator=True,
        metadata={"message_id": "M1"},
    )

    assert result["captured"] is True
    args = conn.fetchval.await_args.args
    assert args[6] is True
    assert json.loads(args[-1]) == {"message_id": "M1"}


async def test_capture_storage_error_is_observable_but_non_blocking(caplog):
    conn = SimpleNamespace(
        fetchval=AsyncMock(side_effect=RuntimeError("schema missing"))
    )
    result = await service.capture_operator_policy_correction(
        _Pool(conn),
        channel_type="api",
        channel_id=None,
        sender_id=None,
        sender_name="Owner",
        text="Always cite sources.",
        is_operator=True,
    )

    assert result["captured"] is False
    assert result["reason"] == "storage_error"
    assert "schema missing" in result["error"]
    assert "not persisted" in caplog.text
    notice = _operator_policy_capture_notice(result)
    assert "was not saved" in notice
    assert result["next_step"] in notice


async def test_channel_identity_verification_fails_closed(caplog):
    conn = SimpleNamespace(fetchval=AsyncMock(side_effect=RuntimeError("db offline")))
    assert (
        await service.channel_sender_is_operator(
            _Pool(conn), channel_type="slack", sender_id="U1"
        )
        is False
    )
    assert "policy capture is disabled for this turn" in caplog.text


@pytest.mark.parametrize(
    ("surface", "is_group", "explicit", "expected"),
    [
        ("api", False, None, True),
        ("cli", False, None, True),
        ("slack", False, None, False),
        ("slack", False, True, True),
        ("api", True, True, False),
    ],
)
async def test_operator_authority_is_transport_owned(
    surface, is_group, explicit, expected
):
    assert (
        _trusted_operator_turn(
            surface=surface,
            is_group=is_group,
            trusted_operator=explicit,
        )
        is expected
    )


async def test_policy_tool_rejects_unverified_or_group_turns():
    handler = ManageOperatorPoliciesHandler()
    registry = SimpleNamespace(pool=object())

    for context in (
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="unverified",
            registry=registry,
        ),
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="group",
            registry=registry,
            is_operator=True,
            is_group=True,
        ),
    ):
        result = await handler.execute({"action": "list"}, context)
        assert result.success is False
        assert "identity-verified operator turn" in str(result.error)


async def test_policy_tool_lists_and_revokes_for_verified_operator(monkeypatch):
    handler = ManageOperatorPoliciesHandler()
    registry = SimpleNamespace(pool=object())
    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="call-1",
        registry=registry,
        is_operator=True,
        surface="api",
    )
    list_mock = AsyncMock(
        return_value={
            "ok": True,
            "count": 1,
            "policies": [{"policy_key": "operator.standing.abc"}],
        }
    )
    revoke_mock = AsyncMock(
        return_value={
            "revoked": True,
            "policy_key": "operator.standing.abc",
        }
    )
    monkeypatch.setattr(service, "list_operator_policies", list_mock)
    monkeypatch.setattr(service, "revoke_operator_policy", revoke_mock)

    listed = await handler.execute({"action": "list"}, context)
    revoked = await handler.execute(
        {
            "action": "revoke",
            "policy_key": "operator.standing.abc",
            "reason": "preference changed",
        },
        context,
    )

    assert listed.success is True
    assert listed.output["count"] == 1
    assert revoked.success is True
    revoke_mock.assert_awaited_once_with(
        registry.pool,
        policy_key="operator.standing.abc",
        actor="operator:api",
        reason="preference changed",
        event_id="call-1",
    )
