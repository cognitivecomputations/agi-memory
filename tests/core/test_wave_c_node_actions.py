from __future__ import annotations

import json
import platform
import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core import node_life
from core.node_actions import MAX_NODE_CAPABILITIES, NODE_CAPABILITIES
from core.node_life import detect_life_capabilities, execute_life_action
from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.nodes import create_node_tools


pytestmark = [pytest.mark.asyncio]


def _context() -> ToolExecutionContext:
    registry = MagicMock()
    registry.pool = object()
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="wave-c-call",
        session_id="11111111-1111-4111-8111-111111111111",
        is_operator=True,
        action_approved=True,
        approval_request_id="22222222-2222-4222-8222-222222222222",
        approval_channel="dashboard",
        registry=registry,
    )


async def test_capability_vocabulary_covers_every_structured_surface():
    assert MAX_NODE_CAPABILITIES == len(NODE_CAPABILITIES)
    assert {
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
        "screen.capture",
    } <= NODE_CAPABILITIES


@pytest.mark.skipif(platform.system() != "Darwin", reason="JXA compiler is macOS-only")
async def test_source_controlled_jxa_programs_compile_on_macos(tmp_path):
    compiler = shutil.which("osacompile")
    assert compiler, "macOS node support requires osacompile"
    scripts = [
        node_life._REMINDERS_LIST_JXA,
        node_life._REMINDERS_CREATE_JXA,
        node_life._NOTES_SEARCH_JXA,
        node_life._NOTES_CREATE_JXA,
        node_life._CALENDAR_LIST_JXA,
        node_life._CALENDAR_CREATE_JXA,
    ]
    for index, script in enumerate(scripts):
        compiled = subprocess.run(
            [
                compiler,
                "-l",
                "JavaScript",
                "-e",
                script,
                "-o",
                str(tmp_path / f"wave-c-{index}.scpt"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compiled.returncode == 0, compiled.stderr


async def test_host_capabilities_are_derived_from_real_local_executables(monkeypatch):
    monkeypatch.setattr(node_life.platform, "system", lambda: "Darwin")
    paths = {
        "osascript": "/usr/bin/osascript",
        "shortcuts": "/usr/bin/shortcuts",
        "op": "/usr/local/bin/op",
        "pbcopy": "/usr/bin/pbcopy",
    }
    monkeypatch.setattr(node_life.shutil, "which", paths.get)

    capabilities = detect_life_capabilities()

    assert "apple.reminders.list" in capabilities
    assert "apple.shortcuts.run" in capabilities
    assert "onepassword.items" in capabilities
    assert "onepassword.copy" in capabilities


async def test_reminder_uses_fixed_script_and_passes_user_text_only_as_argv(
    monkeypatch,
):
    calls: list[tuple[list[str], int]] = []

    async def runner(argv, *, timeout):
        calls.append((argv, timeout))
        return 0, json.dumps({"created": True, "title": argv[-4]}), ""

    monkeypatch.setattr(node_life.shutil, "which", lambda name: f"/usr/bin/{name}")
    dangerous = 'Call Eric"; Application("Finder").quit(); //'
    result = await execute_life_action(
        "apple.reminders.create",
        {"title": dangerous, "list_name": "Work"},
        timeout=30,
        runner=runner,
    )

    assert result["success"] is True
    argv, timeout = calls[0]
    assert argv[:4] == ["/usr/bin/osascript", "-l", "JavaScript", "-e"]
    assert dangerous not in argv[4]
    assert dangerous in argv[6:]
    assert timeout == 30


async def test_calendar_requires_truthful_timezone_and_order():
    runner = AsyncMock()

    missing_zone = await execute_life_action(
        "apple.calendar.list",
        {"start_at": "2026-08-28T09:00:00", "end_at": "2026-08-28T10:00:00"},
        timeout=30,
        runner=runner,
    )
    backwards = await execute_life_action(
        "apple.calendar.create",
        {
            "title": "Review",
            "start_at": "2026-08-28T10:00:00-04:00",
            "end_at": "2026-08-28T09:00:00-04:00",
        },
        timeout=30,
        runner=runner,
    )

    assert "timezone" in missing_zone["error"]
    assert "after start_at" in backwards["error"]
    runner.assert_not_awaited()


async def test_onepassword_item_listing_never_returns_fields_or_secrets(monkeypatch):
    monkeypatch.setattr(
        node_life.shutil,
        "which",
        lambda name: "/usr/local/bin/op" if name == "op" else None,
    )
    provider = [
        {
            "id": "item-1",
            "title": "Example Login",
            "category": "LOGIN",
            "vault": {"id": "vault-1", "name": "Personal"},
            "updated_at": "2026-08-28T12:00:00Z",
            "fields": [{"label": "password", "value": "super-secret"}],
        }
    ]
    runner = AsyncMock(return_value=(0, json.dumps(provider), ""))

    result = await execute_life_action(
        "onepassword.items",
        {"query": "example", "limit": 10},
        timeout=30,
        runner=runner,
    )

    encoded = json.dumps(result)
    assert result["result"]["secrets_included"] is False
    assert result["result"]["items"][0]["title"] == "Example Login"
    assert "super-secret" not in encoded
    assert "fields" not in encoded


async def test_onepassword_copy_keeps_secret_local(monkeypatch):
    copied: list[bytes] = []

    class Process:
        def __init__(self, argv):
            self.argv = argv
            self.returncode = 0

        async def communicate(self, input=None):
            if self.argv[0].endswith("op"):
                return b"super-secret", b""
            copied.append(input or b"")
            return b"", b""

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def subprocess(*argv, **_kwargs):
        return Process(argv)

    class ClosedTask:
        def add_done_callback(self, _callback):
            return None

    def close_task(coro, **_kwargs):
        coro.close()
        return ClosedTask()

    paths = {
        "op": "/usr/local/bin/op",
        "pbcopy": "/usr/bin/pbcopy",
        "pbpaste": "/usr/bin/pbpaste",
    }
    monkeypatch.setattr(node_life.shutil, "which", paths.get)
    monkeypatch.setattr(node_life.asyncio, "create_subprocess_exec", subprocess)
    monkeypatch.setattr(node_life.asyncio, "create_task", close_task)

    result = await execute_life_action(
        "onepassword.copy",
        {"secret_ref": "op://Personal/Example/password", "clipboard_seconds": 60},
        timeout=30,
        runner=AsyncMock(),
    )

    assert result["success"] is True
    assert copied == [b"super-secret"]
    assert "super-secret" not in json.dumps(result)
    assert result["result"]["secret_transmitted_to_hexis"] is False


async def test_structured_tools_are_approval_gated_and_map_exact_actions():
    handlers = {handler.spec.name: handler for handler in create_node_tools()}
    for name in (
        "apple_reminders",
        "apple_notes",
        "apple_calendar",
        "apple_shortcuts",
        "onepassword_local",
    ):
        assert handlers[name].spec.requires_approval is True
        assert handlers[name].spec.is_read_only is False

    terminal = {
        "invocation_id": "33333333-3333-4333-8333-333333333333",
        "status": "succeeded",
        "result": {"reminders": []},
    }
    with patch(
        "services.node_gateway.request_node_invocation",
        AsyncMock(return_value=terminal),
    ) as request:
        result = await handlers["apple_reminders"].execute(
            {
                "node_id": "a" * 64,
                "operation": "list",
                "list_name": "Work",
                "limit": 10,
            },
            _context(),
        )

    assert result.success is True
    assert request.await_args.kwargs["action"] == "apple.reminders.list"
    assert request.await_args.kwargs["arguments"] == {
        "list_name": "Work",
        "limit": 10,
        "timeout": 30,
    }
    assert request.await_args.kwargs["metadata"]["approval_request_id"].startswith(
        "2222"
    )
