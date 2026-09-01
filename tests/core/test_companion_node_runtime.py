from __future__ import annotations

import base64
import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent_loop import AgentLoop, AgentLoopConfig
from core.node_daemon import execute_node_action, node_gateway_url
from core.node_identity import (
    initialize_node_identity,
    load_node_identity,
    remove_node_command,
    set_node_command,
    verify_signature,
)
from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.nodes import NodeInvokeHandler

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def test_signed_identity_is_private_and_detects_tampering(tmp_path: Path) -> None:
    target = tmp_path / "hexis" / "node.json"
    identity = initialize_node_identity(name="Test Mac", path=target)
    payload = {"challenge": "fresh", "node_id": identity.node_id}

    assert verify_signature(identity.public_key, payload, identity.sign(payload))
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert load_node_identity(target) == identity

    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["node_id"] = "0" * 64
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_node_identity(target)


async def test_local_allowlist_never_silently_overwrites_and_never_uses_a_shell(
    tmp_path: Path,
) -> None:
    target = tmp_path / "node.json"
    initialize_node_identity(name="Runner", path=target)
    fixed = [sys.executable, "-c", "print('fixed-only')"]
    identity = set_node_command(
        "fixed",
        fixed,
        allow_args=False,
        path=target,
    )
    with pytest.raises(FileExistsError, match="--replace"):
        set_node_command("fixed", ["/bin/false"], allow_args=False, path=target)

    denied = await execute_node_action(
        identity,
        "system.run",
        {"command": "fixed", "args": ["unexpected"]},
    )
    assert denied["success"] is False
    assert "permits no invocation-time arguments" in denied["error"]

    completed = await execute_node_action(
        identity,
        "system.run",
        {"command": "fixed", "args": [], "timeout": 10},
    )
    assert completed["success"] is True
    assert completed["result"]["stdout"] == "fixed-only\n"
    assert completed["result"]["command"] == "fixed"

    removed = remove_node_command("fixed", path=target)
    assert removed.commands == {}


async def test_node_rejects_hand_edited_relative_executable_and_bad_timeout(
    tmp_path: Path,
) -> None:
    target = tmp_path / "node.json"
    initialize_node_identity(name="Runner", path=target)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["commands"] = {"edited": {"argv": ["echo"], "allow_args": False}}
    target.write_text(json.dumps(raw), encoding="utf-8")
    os.chmod(target, 0o600)

    relative = await execute_node_action(
        load_node_identity(target),
        "system.run",
        {"command": "edited", "args": []},
    )
    assert relative["success"] is False
    assert "absolute, runnable executable" in relative["error"]

    identity = set_node_command(
        "valid",
        [sys.executable, "-c", "print('ok')"],
        allow_args=False,
        path=target,
    )
    malformed = await execute_node_action(
        identity,
        "system.run",
        {"command": "valid", "args": [], "timeout": {"bad": True}},
    )
    assert malformed == {
        "success": False,
        "error": "Node command timeout must be a whole number of seconds.",
    }


async def test_gateway_url_is_outward_websocket_and_preserves_a_prefix() -> None:
    assert node_gateway_url("http://127.0.0.1:43817") == (
        "ws://127.0.0.1:43817/api/nodes/connect"
    )
    assert node_gateway_url("https://hexis.example/base/") == (
        "wss://hexis.example/base/api/nodes/connect"
    )
    with pytest.raises(ValueError, match="http"):
        node_gateway_url("file:///tmp/socket")


def _tool_context(pool: object) -> ToolExecutionContext:
    registry = MagicMock()
    registry.pool = pool
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="call-node",
        session_id="11111111-1111-4111-8111-111111111111",
        is_operator=True,
        action_approved=True,
        approval_request_id="22222222-2222-4222-8222-222222222222",
        approval_channel="dashboard",
        registry=registry,
    )


async def test_node_tool_is_approval_gated_and_strips_image_bytes_from_model_text() -> (
    None
):
    handler = NodeInvokeHandler()
    assert handler.spec.requires_approval is True
    assert handler.spec.is_read_only is False
    assert handler.spec.category.value == "shell"

    node_id = "a" * 64
    encoded = base64.b64encode(b"png").decode("ascii")
    terminal = {
        "invocation_id": "33333333-3333-4333-8333-333333333333",
        "status": "succeeded",
        "result": {
            "mime_type": "image/png",
            "bytes": 3,
            "data_base64": encoded,
        },
    }
    with patch(
        "services.node_gateway.request_node_invocation",
        AsyncMock(return_value=terminal),
    ) as invoke:
        result = await handler.execute(
            {"node_id": node_id, "action": "screen.capture", "timeout": 30},
            _tool_context(object()),
        )

    assert result.success is True
    assert "data_base64" not in json.dumps(result.output)
    assert result.output["result"]["visual_context_attached"] is True
    visual = result.metadata["model_visual_attachments"][0]
    assert visual["data_url"] == f"data:image/png;base64,{encoded}"
    assert invoke.await_args.kwargs["metadata"]["approval_request_id"].startswith(
        "2222"
    )


async def test_agent_loop_appends_tool_screen_as_model_visible_context(db_pool) -> None:
    async with db_pool.acquire() as conn:
        started = await conn.fetchval(
            "SELECT start_agent_turn('chat', 'look', NULL, '{\"messages\":[]}'::jsonb)"
        )
    started = json.loads(started) if isinstance(started, str) else started
    registry = MagicMock()
    registry.pool = db_pool
    loop = AgentLoop(
        AgentLoopConfig(
            tool_context=ToolContext.CHAT,
            system_prompt="test",
            llm_config={},
            registry=registry,
            pool=db_pool,
        )
    )
    loop._turn_id = started["turn_id"]
    data_url = "data:image/png;base64," + base64.b64encode(b"png").decode("ascii")
    await loop._append_tool_visual_context(
        {
            "model_visual_attachments": [
                {"name": "screen.png", "mime_type": "image/png", "data_url": data_url}
            ]
        }
    )
    async with db_pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT messages FROM agent_turns WHERE id=$1::uuid", started["turn_id"]
        )
        await conn.execute(
            "DELETE FROM agent_turns WHERE id=$1::uuid", started["turn_id"]
        )
    messages = json.loads(raw) if isinstance(raw, str) else raw
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][1]["type"] == "input_image"
    assert messages[-1]["content"][1]["image_url"] == data_url


async def test_node_config_path_honors_xdg(monkeypatch, tmp_path: Path) -> None:
    from core.node_identity import node_config_path

    monkeypatch.setenv("XDG_CONFIG_HOME", os.fspath(tmp_path))
    assert node_config_path() == tmp_path / "hexis" / "node.json"


async def test_screen_request_activates_the_host_node_skill(db_pool) -> None:
    from core.tools.registry import create_default_registry
    from services.skill_runtime import select_skills

    registry = create_default_registry(db_pool)
    selection = await select_skills(
        registry,
        ToolContext.CHAT,
        query="Take a fresh screenshot of my paired Mac so you can inspect the screen.",
    )
    assert "host-node" in {skill.name for skill in selection.skills}
    assert "node_invoke" in selection.allowed_tool_names


@pytest.mark.parametrize(
    ("query", "expected_tool"),
    [
        ("Add a reminder to buy milk on my paired Mac.", "apple_reminders"),
        (
            "Copy my GitHub password from 1Password on the paired Mac.",
            "onepassword_local",
        ),
    ],
)
async def test_everyday_node_requests_activate_the_exact_structured_tool(
    db_pool,
    query: str,
    expected_tool: str,
) -> None:
    from core.tools.registry import create_default_registry
    from services.skill_runtime import select_skills

    registry = create_default_registry(db_pool)
    selection = await select_skills(
        registry,
        ToolContext.CHAT,
        query=query,
    )

    assert "host-node" in {skill.name for skill in selection.skills}
    assert expected_tool in selection.allowed_tool_names
