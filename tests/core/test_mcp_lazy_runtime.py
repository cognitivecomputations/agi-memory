"""Lazy MCP transport (#41): servers connect on skill activation, connections
are shared and idempotent, only manifest-bound tools register, failures carry
what/why/next, and unbound mcp_* schemas never reach model context.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.config import MCPServerConfig, ToolsConfig
from core.tools.mcp_runtime import MCPRuntime
from core.tools.skills import UseSkillHandler

pytestmark = pytest.mark.asyncio(loop_scope="session")


_STUB_SERVER = textwrap.dedent(
    """
    import json, sys

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {"capabilities": {"tools": {}}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "echo", "description": "Echo text",
                 "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
                {"name": "secret", "description": "Tool the manifest does not bind",
                 "inputSchema": {"type": "object", "properties": {}}},
            ]}})
        elif method == "tools/call":
            params = msg.get("params") or {}
            args = params.get("arguments") or {}
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": "echo:" + str(args.get("text", ""))}],
                "isError": False,
            }})
    """
)


@pytest.fixture()
def stub_server_config(tmp_path: Path) -> MCPServerConfig:
    script = tmp_path / "stub_mcp_server.py"
    script.write_text(_STUB_SERVER)
    return MCPServerConfig(name="stub", command=sys.executable, args=[str(script)])


@pytest.fixture()
async def runtime():
    rt = MCPRuntime()
    yield rt
    await rt.shutdown()


async def test_ensure_connected_is_idempotent_and_shared(runtime, stub_server_config):
    first = await runtime.ensure_connected(stub_server_config)
    assert first["connected"] is True
    tool_names = {t["name"] for t in first["tools"]}
    assert tool_names == {"echo", "secret"}

    second = await runtime.ensure_connected(stub_server_config)
    assert second["connected"] is True
    assert runtime.connected_servers() == ["stub"]


async def test_register_into_is_bounded_to_manifest_tools(runtime, stub_server_config):
    await runtime.ensure_connected(stub_server_config)
    registry = MagicMock()
    registered: dict[str, object] = {}
    registry.get_spec.side_effect = lambda name: registered.get(name)
    registry.register_mcp.side_effect = lambda h: registered.__setitem__(h.spec.name, h.spec)

    names = runtime.register_into(registry, "stub", ["mcp_stub_echo", "recall"])
    assert names == ["mcp_stub_echo"]
    # The unbound server tool never becomes a handler.
    assert "mcp_stub_secret" not in registered

    # Globs work (implicit mcp-<server> skills bind mcp_<server>_*).
    glob_names = runtime.register_into(registry, "stub", ["mcp_stub_*"])
    assert set(glob_names) == {"mcp_stub_echo", "mcp_stub_secret"}


async def test_connect_failure_reports_what_why_next(runtime):
    bad = MCPServerConfig(name="broken", command="definitely-not-a-real-binary-xyz", args=[])
    result = await runtime.ensure_connected(bad)
    assert result["connected"] is False
    assert "command not found" in result["error"]
    assert result["next_step"]


def _skill_dir_with_manifest(tmp_path: Path, script: Path, env_requires: str = "") -> Path:
    skills_dir = tmp_path / "skills" / "stub-echo"
    skills_dir.mkdir(parents=True)
    lines = [
        "---",
        "name: stub-echo",
        "description: Echo things through the stub MCP server.",
        "contexts: [chat]",
        "mcp:",
        "  server: stub",
        f"  command: {sys.executable}",
        f'  args: ["{script}"]',
    ]
    if env_requires:
        lines.append(f"  env_requires: [{env_requires}]")
    lines += [
        "bound_tools: [mcp_stub_echo]",
        "---",
        "",
        "# Stub Echo",
        "",
        "Call mcp_stub_echo with text.",
    ]
    (skills_dir / "SKILL.md").write_text("\n".join(lines) + "\n")
    return tmp_path / "skills"


def _registry_for_skills(extra_dir: Path) -> MagicMock:
    registry = MagicMock()
    handlers: dict[str, object] = {}
    registry.list_names.side_effect = lambda: list(handlers)
    registry.get_spec.side_effect = lambda name: handlers.get(name)
    registry.register_mcp.side_effect = lambda h: handlers.__setitem__(h.spec.name, h.spec)
    registry.get_config = AsyncMock(return_value=ToolsConfig())
    registry.extra_skill_dirs = [str(extra_dir)]
    return registry


@pytest.fixture()
def fresh_singleton():
    saved = MCPRuntime._instance
    MCPRuntime._instance = None
    yield
    instance = MCPRuntime._instance
    MCPRuntime._instance = saved
    if instance is not None:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.create_task(instance.shutdown())
        except Exception:
            pass


async def test_use_skill_activates_lazily_and_unlocks_bound_tools(
    tmp_path, stub_server_config, fresh_singleton
):
    script = Path(stub_server_config.args[0])
    skills_root = _skill_dir_with_manifest(tmp_path, script)
    registry = _registry_for_skills(skills_root)
    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT, call_id="t1", registry=registry
    )

    # Pre-activation: no mcp_* handler exists anywhere.
    assert not any(n.startswith("mcp_") for n in registry.list_names())

    result = await UseSkillHandler().execute({"name": "stub-echo"}, context)
    assert result.success
    assert result.output["status"] == "activated"
    assert "mcp_stub_echo" in result.output["bound_tools"]
    assert registry.get_spec("mcp_stub_echo") is not None
    # Only the manifest-bound tool registered.
    assert registry.get_spec("mcp_stub_secret") is None

    await MCPRuntime.instance().shutdown()


async def test_use_skill_needs_setup_is_not_a_dead_end(
    tmp_path, stub_server_config, fresh_singleton, monkeypatch
):
    monkeypatch.delenv("HEXIS_STUB_TOKEN", raising=False)
    script = Path(stub_server_config.args[0])
    skills_root = _skill_dir_with_manifest(tmp_path, script, env_requires="HEXIS_STUB_TOKEN")
    registry = _registry_for_skills(skills_root)
    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT, call_id="t2", registry=registry
    )

    result = await UseSkillHandler().execute({"name": "stub-echo"}, context)
    assert result.success  # instructions still delivered
    assert result.output["status"] == "needs_setup"
    assert "HEXIS_STUB_TOKEN" in result.output["next_step"]
    assert result.output["instructions"]
    # No server started, no tools unlocked.
    assert not any(n.startswith("mcp_") for n in registry.list_names())


async def test_use_skill_resolves_stored_github_credential_instead_of_dead_ending(
    tmp_path, stub_server_config, fresh_singleton, monkeypatch
):
    """#102: a required env var missing from the process environment isn't
    necessarily unresolvable -- Hexis's own stored GitHub credential should
    be injected into just this server's subprocess env instead of dead-
    ending into "set X in the service environment"."""
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(
        "core.auth.github_pat.resolve_github_token",
        lambda: ("github_pat_fake123", "octocat"),
    )
    script = Path(stub_server_config.args[0])
    skills_root = _skill_dir_with_manifest(
        tmp_path, script, env_requires="GITHUB_PERSONAL_ACCESS_TOKEN"
    )
    registry = _registry_for_skills(skills_root)
    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT, call_id="t3", registry=registry
    )

    result = await UseSkillHandler().execute({"name": "stub-echo"}, context)
    assert result.success
    assert result.output["status"] == "activated"
    assert "mcp_stub_echo" in result.output["bound_tools"]

    await MCPRuntime.instance().shutdown()


async def test_use_skill_github_needs_setup_points_to_setup_token_command(
    tmp_path, stub_server_config, fresh_singleton, monkeypatch
):
    """#102: with nothing stored and no env var, the next_step must explain
    itself and point at `hexis auth github setup-token` -- never the raw
    "set GITHUB_PERSONAL_ACCESS_TOKEN in the service environment" dead end,
    and never a request to paste the token into chat."""
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(
        "core.auth.github_pat.resolve_github_token", lambda: (None, "")
    )
    script = Path(stub_server_config.args[0])
    skills_root = _skill_dir_with_manifest(
        tmp_path, script, env_requires="GITHUB_PERSONAL_ACCESS_TOKEN"
    )
    registry = _registry_for_skills(skills_root)
    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT, call_id="t4", registry=registry
    )

    result = await UseSkillHandler().execute({"name": "stub-echo"}, context)
    assert result.output["status"] == "needs_setup"
    assert "hexis auth github setup-token" in result.output["next_step"]
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" not in result.output["next_step"]
    assert not any(n.startswith("mcp_") for n in registry.list_names())
