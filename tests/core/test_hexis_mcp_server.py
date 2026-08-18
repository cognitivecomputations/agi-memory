"""MCP server listing and dispatch contracts."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from apps.hexis_mcp_server import _dispatch_server_tool, _list_server_tools
from core.tools import ToolContext


def _is_error(result) -> bool:
    """CallToolResult grew snake_case attributes (isError -> is_error) in newer
    mcp SDKs; accept both so the suite runs against either generation."""
    return result.is_error if hasattr(result, "is_error") else result.isError


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


class _RegistryResult:
    success = True

    def to_model_output(self):
        return '{"registry":true}'


class _RegistryFailure:
    success = False

    def to_model_output(self):
        return "Error: policy denied"


class _Registry:
    def __init__(self, result=None):
        self.execute = AsyncMock(return_value=result or _RegistryResult())
        self.get_mcp_tools = AsyncMock(
            return_value=[
                {
                    "name": "recall",
                    "description": "duplicate legacy name",
                    "inputSchema": {"type": "object"},
                },
                {
                    "name": "session_list",
                    "description": "List sessions",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            ]
        )


async def test_mcp_listing_uses_registry_native_tools_by_default(monkeypatch):
    monkeypatch.delenv("HEXIS_MCP_LEGACY_COMPAT", raising=False)
    registry = _Registry()

    tools, registry_names = await _list_server_tools(registry)
    names = [tool.name for tool in tools]

    assert names.count("recall") == 1
    assert "hydrate" not in names
    assert "session_list" in names
    assert registry_names == {"recall", "session_list"}
    registry.get_mcp_tools.assert_awaited_once_with(ToolContext.MCP)


async def test_mcp_listing_can_opt_into_legacy_compat(monkeypatch):
    monkeypatch.setenv("HEXIS_MCP_LEGACY_COMPAT", "1")
    registry = _Registry()

    tools, registry_names = await _list_server_tools(registry)
    names = [tool.name for tool in tools]

    assert names.count("recall") == 1
    assert "hydrate" in names
    assert "session_list" in names
    assert registry_names == {"recall", "session_list"}


async def test_mcp_legacy_dispatch_requires_compat_flag(monkeypatch):
    monkeypatch.setenv("HEXIS_MCP_LEGACY_COMPAT", "1")
    client = AsyncMock()
    client.get_health.return_value = {"status": "healthy", "score": 0.9}

    result = await _dispatch_server_tool(client, _Registry(), set(), "get_health", {})

    assert _is_error(result) is False
    assert json.loads(result.content[0].text) == {"score": 0.9, "status": "healthy"}
    client.get_health.assert_awaited_once_with()


async def test_mcp_registry_dispatch_refreshes_before_list_and_uses_mcp_context():
    registry = _Registry()
    known_names: set[str] = set()

    result = await _dispatch_server_tool(
        AsyncMock(), registry, known_names, "session_list", {"limit": 2}
    )

    assert _is_error(result) is False
    assert result.content[0].text == '{"registry":true}'
    assert known_names == {"recall", "session_list"}
    arguments = registry.execute.await_args.args
    assert arguments[0] == "session_list"
    assert arguments[1] == {"limit": 2}
    assert arguments[2].tool_context is ToolContext.MCP
    assert arguments[2].call_id


async def test_mcp_unknown_tool_sets_protocol_error_flag():
    result = await _dispatch_server_tool(
        AsyncMock(), _Registry(), set(), "does_not_exist", {}
    )

    assert _is_error(result) is True
    assert "Unknown tool 'does_not_exist'" in result.content[0].text


async def test_mcp_registry_failure_sets_protocol_error_flag():
    registry = _Registry(_RegistryFailure())

    result = await _dispatch_server_tool(
        AsyncMock(), registry, {"session_list"}, "session_list", {}
    )

    assert _is_error(result) is True
    assert result.content[0].text == "Error: policy denied"
