from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.tools.base import ToolContext
from services.tool_surface_audit import hash_input_text, record_tool_surface_decision


class _Connection:
    def __init__(self):
        self.calls = []

    async def fetchval(self, query, *args):
        self.calls.append((query, args))
        if "get_config_bool" in query:
            return True
        return "00000000-0000-0000-0000-000000000123"


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return False


class _Pool:
    def __init__(self):
        self.conn = _Connection()

    def acquire(self):
        return _Acquire(self.conn)


class _Registry:
    registry_kind = "full"

    def list_names(self):
        return ["list_skills", "callable"]

    async def get_specs(self, context):  # noqa: ARG002
        return [
            {"type": "function", "function": {"name": "list_skills"}},
            {"type": "function", "function": {"name": "callable"}},
            {"type": "function", "function": {"name": "stale_db_only"}},
        ]


def test_hash_input_text_normalizes_surrounding_whitespace():
    assert hash_input_text("  hello  ") == hash_input_text("hello")
    assert hash_input_text("hello") != hash_input_text("goodbye")


@pytest.mark.asyncio
async def test_audit_records_requested_reachable_delta():
    pool = _Pool()
    selection = SimpleNamespace(
        allowed_tool_names={"list_skills", "callable", "missing"},
        skills=[SimpleNamespace(name="core-memory")],
        considered=[{"name": "calendar", "score": 0.5}],
        available=[SimpleNamespace(name="core-memory"), SimpleNamespace(name="calendar")],
    )

    event_id = await record_tool_surface_decision(
        pool,
        registry=_Registry(),
        selection=selection,
        session_id=None,
        surface="chat",
        tool_context=ToolContext.CHAT,
        query="private words stay out of the row",
    )

    assert event_id == "00000000-0000-0000-0000-000000000123"
    _query, args = pool.conn.calls[-1]
    assert args[7] == ["callable", "list_skills", "missing"]
    assert args[8] == ["callable", "list_skills"]
    assert args[9] == ["missing"]
    assert args[4] == hash_input_text("private words stay out of the row")
    assert "private words" not in str(args)
