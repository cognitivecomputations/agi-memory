from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.tools.base import ToolCategory, ToolContext, ToolExecutionContext
from core.tools.responsibilities import ManageResponsibilityHandler, create_responsibility_tools

def _ctx(pool) -> ToolExecutionContext:
    registry = MagicMock()
    registry.pool = pool
    registry.execute = None
    return ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="test-call", registry=registry)


@pytest.mark.asyncio(loop_scope="session")
async def test_manage_responsibility_tool_creates_gmail_monitor_blocked_when_missing(db_pool):
    title = "ambient gmail monitor blocked test"
    try:
        result = await ManageResponsibilityHandler().execute(
            {
                "action": "create",
                "title": title,
                "kind": "monitor",
                "user_intent": "Let me know whenever Hope emails me.",
                "trigger": {"kind": "interval", "every_seconds": 60},
                "sources": [{"connector_id": "gmail", "query": "from:hope"}],
                "actions": [{"type": "notify_user", "message": "Hope emailed: {title}"}],
            },
            _ctx(db_pool),
        )
        assert result.success, result.error
        assert result.output["status"] == "blocked"
        assert result.output["missing_connectors"][0]["connector_id"] == "gmail"
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM ambient_responsibilities WHERE title = $1", title)


def test_manage_responsibility_spec_is_first_class():
    spec = ManageResponsibilityHandler().spec
    assert spec.name == "manage_responsibility"
    assert spec.category == ToolCategory.MEMORY
    assert "let me know whenever" in spec.description
    assert "sources" in spec.parameters["properties"]
    assert "evaluator" in spec.parameters["properties"]


def test_create_responsibility_tools_factory():
    tools = create_responsibility_tools()
    assert len(tools) == 1
    assert tools[0].spec.name == "manage_responsibility"
