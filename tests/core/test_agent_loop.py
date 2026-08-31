"""
Tests for the unified AgentLoop.

Covers: basic text, single/multi tool calls, multi-iteration,
self-correction, energy budget, timeout, max iterations, approval
callbacks, config overrides, event emission, streaming, error handling.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent_loop import (
    AgentEvent,
    AgentEventData,
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    _to_openai_tool_call,
)
from core.tools.base import (
    OutboundSpec,
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
)
from core.tools.config import ContextOverrides, ToolsConfig

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


# ============================================================================
# DB wiring
# ============================================================================
#
# The loop is now DB-authoritative: agent_turns owns the message log, energy
# accounting and stop decisions. These tests therefore run against the real
# temp test DB (mocking only the LLM + registry), rather than mocking the
# turn-state away. The autouse fixture publishes the pool so the helper
# factories below can hand it to every AgentLoop under test.

_DB_POOL: Any = None


@pytest.fixture(autouse=True)
def _wire_db_pool(db_pool):
    global _DB_POOL
    _DB_POOL = db_pool
    yield
    _DB_POOL = None


# ============================================================================
# Helpers
# ============================================================================


def _make_llm_config() -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": "gpt-4o",
        "endpoint": None,
        "api_key": "test-key",
    }


def _text_response(text: str) -> dict[str, Any]:
    """Simulate an LLM response with text only (no tool calls)."""
    return {"content": text, "tool_calls": [], "raw": None}


def _tool_response(
    text: str,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Simulate an LLM response with tool calls."""
    return {"content": text, "tool_calls": tool_calls, "raw": None}


def _tool_call(
    name: str, arguments: dict[str, Any], call_id: str | None = None
) -> dict[str, Any]:
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:8]}",
        "name": name,
        "arguments": arguments,
    }


def _mock_registry(
    *,
    tool_specs: list[dict[str, Any]] | None = None,
    execute_results: dict[str, ToolResult] | None = None,
    spec_map: dict[str, ToolSpec] | None = None,
    config: ToolsConfig | None = None,
) -> MagicMock:
    """Create a mock ToolRegistry (real DB pool for authoritative turn-state)."""
    registry = MagicMock()
    registry.pool = _DB_POOL

    # get_specs returns OpenAI function specs
    registry.get_specs = AsyncMock(return_value=tool_specs or [])

    # get_spec returns ToolSpec for a given name
    def _get_spec(name: str):
        if spec_map:
            return spec_map.get(name)
        return None

    registry.get_spec = MagicMock(side_effect=_get_spec)

    # execute returns ToolResult
    execute_results = execute_results or {}

    async def _execute(
        name: str, arguments: dict, context: ToolExecutionContext
    ) -> ToolResult:
        if name in execute_results:
            return execute_results[name]
        return ToolResult.success_result({"echo": name, "args": arguments})

    registry.execute = AsyncMock(side_effect=_execute)
    registry.preflight_outbound_controls = AsyncMock(return_value=None)

    # get_config returns ToolsConfig
    config = config or ToolsConfig()
    registry.get_config = AsyncMock(return_value=config)

    return registry


def _make_config(
    registry: MagicMock | None = None,
    **overrides: Any,
) -> AgentLoopConfig:
    """Build an AgentLoopConfig with sensible defaults."""
    reg = registry or _mock_registry()
    defaults = dict(
        tool_context=ToolContext.CHAT,
        system_prompt="You are a test assistant.",
        llm_config=_make_llm_config(),
        registry=reg,
        pool=reg.pool,
        timeout_seconds=10.0,
    )
    defaults.update(overrides)
    return AgentLoopConfig(**defaults)


# ============================================================================
# Unit: basic text response
# ============================================================================


class TestBasicText:
    @patch("core.agent_loop.chat_completion")
    async def test_text_only_response(self, mock_llm):
        mock_llm.return_value = _text_response("Hello, world!")
        config = _make_config()
        agent = AgentLoop(config)
        result = await agent.run("Hi")

        assert result.text == "Hello, world!"
        assert result.iterations == 1
        assert result.energy_spent == 0
        assert result.stopped_reason == "completed"
        assert result.timed_out is False
        assert len(result.tool_calls_made) == 0
        assert len(result.messages) == 3  # system + user + assistant

    @patch("core.agent_loop.chat_completion")
    async def test_empty_text_response(self, mock_llm):
        mock_llm.return_value = _text_response("")
        config = _make_config()
        agent = AgentLoop(config)
        result = await agent.run("Hi")

        assert result.text == ""
        assert result.stopped_reason == "completed"

    @patch("core.agent_loop.chat_completion")
    async def test_history_included(self, mock_llm):
        mock_llm.return_value = _text_response("Got it.")
        config = _make_config()
        agent = AgentLoop(config)
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        result = await agent.run("Follow-up", history=history)

        # system + 2 history + user + assistant = 5
        assert len(result.messages) == 5
        assert result.messages[1]["content"] == "Previous question"


# ============================================================================
# Unit: tool calls
# ============================================================================


class TestToolCalls:
    @patch("core.agent_loop.chat_completion")
    async def test_single_tool_call(self, mock_llm):
        """LLM calls one tool, gets result, then responds with text."""
        recall_result = ToolResult.success_result({"memories": ["Python is great"]})
        recall_result.energy_spent = 1
        registry = _mock_registry(
            tool_specs=[
                {
                    "type": "function",
                    "function": {
                        "name": "recall",
                        "description": "Recall",
                        "parameters": {},
                    },
                }
            ],
            execute_results={"recall": recall_result},
        )

        # First call: tool call; Second call: text response
        mock_llm.side_effect = [
            _tool_response(
                "Let me search.", [_tool_call("recall", {"query": "Python"})]
            ),
            _text_response("Python is great!"),
        ]

        config = _make_config(registry=registry)
        agent = AgentLoop(config)
        result = await agent.run("Tell me about Python")

        assert result.text == "Python is great!"
        assert result.iterations == 2
        assert result.energy_spent == 1
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0]["name"] == "recall"
        assert result.tool_calls_made[0]["success"] is True

    @patch("core.agent_loop.chat_completion")
    async def test_multiple_tool_calls_in_one_response(self, mock_llm):
        """LLM makes two tool calls in a single response."""
        recall_result = ToolResult.success_result({"data": "recalled"})
        recall_result.energy_spent = 1
        search_result = ToolResult.success_result({"results": ["found"]})
        search_result.energy_spent = 2
        registry = _mock_registry(
            execute_results={"recall": recall_result, "web_search": search_result},
        )

        mock_llm.side_effect = [
            _tool_response(
                "Searching...",
                [
                    _tool_call("recall", {"query": "test"}),
                    _tool_call("web_search", {"query": "test"}),
                ],
            ),
            _text_response("Here's what I found."),
        ]

        config = _make_config(registry=registry)
        agent = AgentLoop(config)
        result = await agent.run("Find info")

        assert result.iterations == 2
        assert result.energy_spent == 3  # 1 + 2
        assert len(result.tool_calls_made) == 2

    @patch("core.agent_loop.chat_completion")
    async def test_multi_iteration_tool_calls(self, mock_llm):
        """LLM calls tools, gets results, calls more tools, then responds."""
        r1 = ToolResult.success_result("first")
        r1.energy_spent = 1
        r2 = ToolResult.success_result("second")
        r2.energy_spent = 1
        registry = _mock_registry(execute_results={"tool_a": r1, "tool_b": r2})

        mock_llm.side_effect = [
            _tool_response("Step 1", [_tool_call("tool_a", {})]),
            _tool_response("Step 2", [_tool_call("tool_b", {})]),
            _text_response("All done."),
        ]

        config = _make_config(registry=registry)
        agent = AgentLoop(config)
        result = await agent.run("Do things")

        assert result.text == "All done."
        assert result.iterations == 3
        assert result.energy_spent == 2
        assert len(result.tool_calls_made) == 2

    @patch("core.agent_loop.chat_completion")
    async def test_use_skill_unlocks_bound_tools_for_next_iteration(self, mock_llm):
        """Skill-first mode starts narrow, then `use_skill` expands the exposed tools."""
        use_skill_spec = ToolSpec(
            name="use_skill",
            description="Activate a skill",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
            category=ToolCategory.EXTERNAL,
            energy_cost=0,
        )
        list_skills_spec = ToolSpec(
            name="list_skills",
            description="List skills",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.EXTERNAL,
            energy_cost=0,
        )
        web_search_spec = ToolSpec(
            name="web_search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
        )
        all_specs = [
            list_skills_spec.to_openai_function(),
            use_skill_spec.to_openai_function(),
            web_search_spec.to_openai_function(),
        ]
        registry = _mock_registry(
            tool_specs=all_specs,
            spec_map={
                "use_skill": use_skill_spec,
                "list_skills": list_skills_spec,
                "web_search": web_search_spec,
            },
            execute_results={
                "use_skill": ToolResult.success_result(
                    {"name": "research", "bound_tools": ["web_search"]}
                ),
                "web_search": ToolResult.success_result({"results": ["found"]}),
            },
        )
        seen_tool_names: list[list[str]] = []

        async def _llm(**kwargs):
            seen_tool_names.append(
                [spec["function"]["name"] for spec in (kwargs.get("tools") or [])]
            )
            if len(seen_tool_names) == 1:
                return _tool_response(
                    "Need research.", [_tool_call("use_skill", {"name": "research"})]
                )
            if len(seen_tool_names) == 2:
                return _tool_response(
                    "Searching.", [_tool_call("web_search", {"query": "postgres age"})]
                )
            return _text_response("Done.")

        mock_llm.side_effect = _llm
        config = _make_config(
            registry=registry,
            allowed_tool_names={"list_skills", "use_skill"},
        )
        agent = AgentLoop(config)
        result = await agent.run("Research postgres age")

        assert result.text == "Done."
        assert seen_tool_names[0] == ["list_skills", "use_skill"]
        assert "web_search" in seen_tool_names[1]
        assert [c["name"] for c in result.tool_calls_made] == [
            "use_skill",
            "web_search",
        ]

    @patch("core.agent_loop.chat_completion")
    async def test_self_correction(self, mock_llm):
        """Tool returns error, LLM sees it and adjusts approach."""
        fail_result = ToolResult.error_result(
            "File not found", ToolErrorType.FILE_NOT_FOUND
        )
        fail_result.energy_spent = 1
        ok_result = ToolResult.success_result("content of file.txt")
        ok_result.energy_spent = 1
        registry = _mock_registry(execute_results={"read_file": fail_result})

        call_count = 0

        async def _execute(name, args, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fail_result
            return ok_result

        registry.execute = AsyncMock(side_effect=_execute)

        mock_llm.side_effect = [
            _tool_response(
                "Reading file.", [_tool_call("read_file", {"path": "/bad/path"})]
            ),
            _tool_response(
                "Trying again.", [_tool_call("read_file", {"path": "/good/path"})]
            ),
            _text_response("Found the file content."),
        ]

        config = _make_config(registry=registry)
        agent = AgentLoop(config)
        result = await agent.run("Read a file")

        assert result.text == "Found the file content."
        assert result.iterations == 3
        assert len(result.tool_calls_made) == 2
        assert result.tool_calls_made[0]["success"] is False
        assert result.tool_calls_made[1]["success"] is True


# ============================================================================
# Unit: energy budget
# ============================================================================


class TestEnergyBudget:
    @patch("core.agent_loop.chat_completion")
    async def test_energy_exhausted_stops_loop(self, mock_llm):
        """Loop stops when energy budget is exhausted."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 5
        registry = _mock_registry(execute_results={"expensive_tool": tool_result})

        mock_llm.side_effect = [
            _tool_response("Using tool.", [_tool_call("expensive_tool", {})]),
            # Second iteration: energy check will stop before LLM call
        ]

        config = _make_config(registry=registry, energy_budget=5)
        agent = AgentLoop(config)
        result = await agent.run("Do expensive thing")

        assert result.stopped_reason == "energy"
        assert result.energy_spent == 5

    @patch("core.agent_loop.chat_completion")
    async def test_energy_remaining_passed_to_context(self, mock_llm):
        """Remaining energy is set on ToolExecutionContext."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 3
        registry = _mock_registry(execute_results={"tool_a": tool_result})

        captured_ctx = None

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return tool_result

        registry.execute = AsyncMock(side_effect=_capture_execute)

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool_a", {})]),
            _text_response("Done."),
        ]

        config = _make_config(registry=registry, energy_budget=10)
        agent = AgentLoop(config)
        await agent.run("Test")

        assert captured_ctx is not None
        assert captured_ctx.energy_available == 10  # Full budget at first call

    @patch("core.agent_loop.chat_completion")
    async def test_unlimited_energy_for_chat(self, mock_llm):
        """Chat mode (energy_budget=None) has no energy limit."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 100
        registry = _mock_registry(execute_results={"tool": tool_result})

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _text_response("Done."),
        ]

        config = _make_config(registry=registry, energy_budget=None)
        agent = AgentLoop(config)
        result = await agent.run("Test")

        assert result.stopped_reason == "completed"
        assert result.energy_spent == 100

    @patch("core.agent_loop.chat_completion")
    async def test_energy_none_context(self, mock_llm):
        """When energy_budget is None, context.energy_available is None."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0
        registry = _mock_registry(execute_results={"tool": tool_result})

        captured_ctx = None

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return tool_result

        registry.execute = AsyncMock(side_effect=_capture_execute)

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _text_response("Done."),
        ]

        config = _make_config(registry=registry, energy_budget=None)
        agent = AgentLoop(config)
        await agent.run("Test")

        assert captured_ctx is not None
        assert captured_ctx.energy_available is None


# ============================================================================
# Unit: limits
# ============================================================================


class TestLimits:
    @patch("core.agent_loop.chat_completion")
    async def test_max_iterations(self, mock_llm):
        """Loop stops at max_iterations."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0
        registry = _mock_registry(execute_results={"tool": tool_result})

        # Keep returning tool calls — loop should stop at 3 iterations
        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _tool_response("More", [_tool_call("tool", {})]),
            _tool_response("Even more", [_tool_call("tool", {})]),
        ]

        config = _make_config(registry=registry, max_iterations=3)
        agent = AgentLoop(config)
        result = await agent.run("Loop forever")

        assert result.stopped_reason == "max_iterations"
        assert result.iterations == 3

    @patch("core.agent_loop.chat_completion")
    async def test_timeout(self, mock_llm):
        """Loop returns with timed_out when timeout is exceeded."""

        async def _slow_llm(**kwargs):
            await asyncio.sleep(5)
            return _text_response("Too slow")

        mock_llm.side_effect = _slow_llm

        config = _make_config(timeout_seconds=0.1)
        agent = AgentLoop(config)
        result = await agent.run("Slow request")

        assert result.timed_out is True
        assert result.stopped_reason == "timeout"


# ============================================================================
# Unit: approval
# ============================================================================


class TestApproval:
    @patch("core.agent_loop.chat_completion")
    async def test_stop_preflight_precedes_approval_request(self, mock_llm):
        """A recipient STOP cannot trigger one more operator notification."""
        approve = AsyncMock(return_value=True)
        spec = ToolSpec(
            name="provider_send",
            description="Person-facing provider send.",
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
            category=ToolCategory.MESSAGING,
            requires_approval=True,
            is_read_only=False,
            outbound=OutboundSpec(
                recipient_arg="recipient",
                body_arg="message",
                channel="test",
            ),
        )
        registry = _mock_registry(spec_map={"provider_send": spec})
        registry.preflight_outbound_controls.return_value = ToolResult.error_result(
            "Recipient has opted out of all Hexis contact.",
            ToolErrorType.OUTBOUND_BLOCKED,
        )
        mock_llm.side_effect = [
            _tool_response(
                "Sending.",
                [
                    _tool_call(
                        "provider_send",
                        {
                            "recipient": "blocked@example.com",
                            "message": "This must stay silent.",
                            "purpose_kind": "user_request",
                            "purpose_reference": "current_turn",
                        },
                    )
                ],
            ),
            _text_response("The recipient has opted out, so I did not ask or send."),
        ]

        agent = AgentLoop(
            _make_config(registry=registry, on_approval=approve)
        )
        result = await agent.run("Send the update")

        registry.preflight_outbound_controls.assert_awaited_once()
        approve.assert_not_awaited()
        registry.execute.assert_not_awaited()
        assert result.tool_calls_made[0]["error"] == "outbound_control_blocked"

    @patch("core.agent_loop.chat_completion")
    async def test_approval_callback_called(self, mock_llm):
        """Approval callback is invoked for tools that require approval."""
        approval_calls = []

        async def _approve(name: str, args: dict) -> bool:
            approval_calls.append((name, args))
            return True

        spec = ToolSpec(
            name="dangerous_tool",
            description="Needs approval",
            parameters={"type": "object"},
            category=ToolCategory.SHELL,
            requires_approval=True,
        )
        tool_result = ToolResult.success_result("executed")
        tool_result.energy_spent = 0
        registry = _mock_registry(
            spec_map={"dangerous_tool": spec},
            execute_results={"dangerous_tool": tool_result},
        )

        mock_llm.side_effect = [
            _tool_response("Running.", [_tool_call("dangerous_tool", {"cmd": "ls"})]),
            _text_response("Done."),
        ]

        config = _make_config(registry=registry, on_approval=_approve)
        agent = AgentLoop(config)
        result = await agent.run("Run a command")

        assert len(approval_calls) == 1
        assert approval_calls[0][0] == "dangerous_tool"
        assert result.stopped_reason == "completed"

    @patch("core.agent_loop.chat_completion")
    async def test_exact_phone_approval_proof_reaches_tool_policy(self, mock_llm):
        """A durable decision carries its one-shot request id into execution."""
        request_id = str(uuid.uuid4())

        async def _approve(name: str, args: dict) -> dict[str, Any]:
            return {
                "approved": True,
                "request_id": request_id,
                "approval_channel": "slack",
                "status": "approved",
            }

        spec = ToolSpec(
            name="dangerous_tool",
            description="Needs approval",
            parameters={"type": "object"},
            category=ToolCategory.SHELL,
            requires_approval=True,
        )
        captured_ctx = None
        registry = _mock_registry(spec_map={"dangerous_tool": spec})

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return ToolResult.success_result("executed")

        registry.execute = AsyncMock(side_effect=_capture_execute)
        mock_llm.side_effect = [
            _tool_response("Running.", [_tool_call("dangerous_tool", {"cmd": "ls"})]),
            _text_response("Done."),
        ]

        agent = AgentLoop(_make_config(registry=registry, on_approval=_approve))
        await agent.run("Run a command")

        assert captured_ctx is not None
        assert captured_ctx.action_approved is True
        assert captured_ctx.approval_request_id == request_id
        assert captured_ctx.approval_channel == "slack"

    @patch("core.agent_loop.chat_completion")
    async def test_approval_denied(self, mock_llm):
        """When approval is denied, tool is not executed and denial message is sent to LLM."""

        async def _deny(name: str, args: dict) -> bool:
            return False

        spec = ToolSpec(
            name="dangerous_tool",
            description="Needs approval",
            parameters={"type": "object"},
            category=ToolCategory.SHELL,
            requires_approval=True,
        )
        registry = _mock_registry(spec_map={"dangerous_tool": spec})

        mock_llm.side_effect = [
            _tool_response(
                "Running.",
                [_tool_call("dangerous_tool", {"cmd": "rm -rf /"}, call_id="call_1")],
            ),
            _text_response("OK, I won't do that."),
        ]

        config = _make_config(registry=registry, on_approval=_deny)
        agent = AgentLoop(config)
        result = await agent.run("Delete everything")

        # Tool should NOT have been executed
        registry.execute.assert_not_awaited()

        # Should have a denial message in the messages
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "denied" in tool_msgs[0]["content"].lower()

        # Denial recorded
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0]["denied"] is True

    @patch("core.agent_loop.chat_completion")
    async def test_no_approver_refuses_rather_than_executing(self, mock_llm):
        """With no approver wired, an approval-required tool is refused.

        This reverses an earlier documented behaviour ("no callback → execute
        normally"). The absence of someone to ask is not consent: under the old
        rule every runtime that wired no callback — the heartbeat, the API chat
        path — ran all 51 approval-flagged tools unattended, including
        slack_send, email_send, gmail_delete, shell and write_file.
        """
        spec = ToolSpec(
            name="dangerous_tool",
            description="Needs approval",
            parameters={"type": "object"},
            category=ToolCategory.SHELL,
            requires_approval=True,
        )
        tool_result = ToolResult.success_result("ran")
        tool_result.energy_spent = 0
        registry = _mock_registry(
            spec_map={"dangerous_tool": spec},
            execute_results={"dangerous_tool": tool_result},
        )

        mock_llm.side_effect = [
            _tool_response("Go.", [_tool_call("dangerous_tool", {})]),
            _text_response("Ran it."),
        ]

        config = _make_config(registry=registry, on_approval=None)
        agent = AgentLoop(config)
        result = await agent.run("Run it")

        # Refused, not executed — and the model is told how to proceed.
        registry.execute.assert_not_awaited()
        assert result.stopped_reason == "completed"
        refused = [c for c in agent._tool_calls_made if c.get("error") == "no_approver_available"]
        assert len(refused) == 1
        assert refused[0]["name"] == "dangerous_tool"


# ============================================================================
# Unit: context propagation
# ============================================================================


class TestContextPropagation:
    @patch("core.agent_loop.chat_completion")
    async def test_session_and_heartbeat_ids(self, mock_llm):
        """Session and heartbeat IDs propagate to execution context."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0
        registry = _mock_registry(execute_results={"tool": tool_result})

        captured_ctx = None

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return tool_result

        registry.execute = AsyncMock(side_effect=_capture_execute)

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _text_response("Done."),
        ]

        config = _make_config(
            registry=registry,
            session_id="sess-123",
            heartbeat_id="hb-456",
            tool_context=ToolContext.HEARTBEAT,
        )
        agent = AgentLoop(config)
        await agent.run("Test")

        assert captured_ctx is not None
        assert captured_ctx.session_id == "sess-123"
        assert captured_ctx.heartbeat_id == "hb-456"
        assert captured_ctx.tool_context == ToolContext.HEARTBEAT

    @patch("core.agent_loop.chat_completion")
    async def test_config_overrides_applied(self, mock_llm):
        """allow_shell and allow_file_write from ToolsConfig are propagated."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0
        tools_config = ToolsConfig(
            context_overrides={
                ToolContext.CHAT: ContextOverrides(
                    allow_shell=True,
                    allow_file_write=True,
                ),
            },
            workspace_path="/test/workspace",
        )
        registry = _mock_registry(
            execute_results={"tool": tool_result},
            config=tools_config,
        )

        captured_ctx = None

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return tool_result

        registry.execute = AsyncMock(side_effect=_capture_execute)

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _text_response("Done."),
        ]

        config = _make_config(registry=registry)
        agent = AgentLoop(config)
        await agent.run("Test")

        assert captured_ctx is not None
        assert captured_ctx.allow_shell is True
        assert captured_ctx.allow_file_write is True
        assert captured_ctx.workspace_path == "/test/workspace"


# ============================================================================
# Unit: events
# ============================================================================


class TestEvents:
    @patch("core.agent_loop.chat_completion")
    async def test_event_order_text_only(self, mock_llm):
        """Events: LOOP_START → TEXT_DELTA → LOOP_END for text-only."""
        events: list[AgentEvent] = []

        async def _capture(e: AgentEventData) -> None:
            events.append(e.event)

        mock_llm.return_value = _text_response("Hello!")
        config = _make_config(on_event=_capture)
        agent = AgentLoop(config)
        await agent.run("Hi")

        assert events == [
            AgentEvent.LOOP_START,
            AgentEvent.LLM_REQUEST,
            AgentEvent.LLM_RESPONSE,
            AgentEvent.TEXT_DELTA,
            AgentEvent.LOOP_END,
        ]

    @patch("core.agent_loop.chat_completion")
    async def test_event_order_with_tool(self, mock_llm):
        """Events: LOOP_START → TEXT_DELTA → TOOL_START → TOOL_RESULT → TEXT_DELTA → LOOP_END."""
        events: list[AgentEvent] = []

        async def _capture(e: AgentEventData) -> None:
            events.append(e.event)

        tool_result = ToolResult.success_result("found it")
        tool_result.energy_spent = 1
        registry = _mock_registry(execute_results={"recall": tool_result})

        mock_llm.side_effect = [
            _tool_response("Searching.", [_tool_call("recall", {"q": "test"})]),
            _text_response("Here you go."),
        ]

        config = _make_config(registry=registry, on_event=_capture)
        agent = AgentLoop(config)
        await agent.run("Find something")

        assert events == [
            AgentEvent.LOOP_START,
            AgentEvent.LLM_REQUEST,
            AgentEvent.LLM_RESPONSE,
            AgentEvent.TEXT_DELTA,  # "Searching."
            AgentEvent.TOOL_START,
            AgentEvent.TOOL_RESULT,
            AgentEvent.LLM_REQUEST,
            AgentEvent.LLM_RESPONSE,
            AgentEvent.TEXT_DELTA,  # "Here you go."
            AgentEvent.LOOP_END,
        ]

    @patch("core.agent_loop.chat_completion")
    async def test_event_data_contains_tool_info(self, mock_llm):
        """TOOL_START and TOOL_RESULT events contain tool name and details."""
        event_data: list[AgentEventData] = []

        async def _capture(e: AgentEventData) -> None:
            event_data.append(e)

        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 2
        registry = _mock_registry(execute_results={"recall": tool_result})

        mock_llm.side_effect = [
            _tool_response(
                "", [_tool_call("recall", {"query": "test"}, call_id="call_abc")]
            ),
            _text_response("Done."),
        ]

        config = _make_config(registry=registry, on_event=_capture)
        agent = AgentLoop(config)
        await agent.run("Test")

        tool_starts = [e for e in event_data if e.event == AgentEvent.TOOL_START]
        tool_results = [e for e in event_data if e.event == AgentEvent.TOOL_RESULT]

        assert len(tool_starts) == 1
        assert tool_starts[0].data["tool_name"] == "recall"

        assert len(tool_results) == 1
        assert tool_results[0].data["tool_name"] == "recall"
        assert tool_results[0].data["success"] is True
        assert tool_results[0].data["energy_spent"] == 2

    @patch("core.agent_loop.chat_completion")
    async def test_event_callback_error_does_not_crash(self, mock_llm):
        """If event callback raises, loop continues."""

        async def _bad_callback(e: AgentEventData) -> None:
            raise RuntimeError("callback error")

        mock_llm.return_value = _text_response("Works.")
        config = _make_config(on_event=_bad_callback)
        agent = AgentLoop(config)
        result = await agent.run("Test")

        assert result.text == "Works."
        assert result.stopped_reason == "completed"


# ============================================================================
# Unit: streaming
# ============================================================================


class TestStreaming:
    @patch("core.agent_loop.stream_chat_completion")
    async def test_stream_yields_events(self, mock_stream_llm):
        """stream() yields AgentEventData objects."""

        async def _fake_stream(**kwargs):
            cb = kwargs.get("on_text_delta")
            if cb:
                await cb("Streamed!")
            return _text_response("Streamed!")

        mock_stream_llm.side_effect = _fake_stream
        config = _make_config()
        agent = AgentLoop(config)

        events = []
        async for event in agent.stream("Hi"):
            events.append(event)

        event_types = [e.event for e in events]
        assert AgentEvent.LOOP_START in event_types
        assert AgentEvent.TEXT_DELTA in event_types
        assert AgentEvent.LOOP_END in event_types

    @patch("core.agent_loop.stream_chat_completion")
    async def test_stream_text_delta_content(self, mock_stream_llm):
        """TEXT_DELTA events contain the text content."""

        # Simulate stream_chat_completion calling on_text_delta per-token
        async def _fake_stream(**kwargs):
            cb = kwargs.get("on_text_delta")
            if cb:
                await cb("Hello ")
                await cb("stream!")
            return _text_response("Hello stream!")

        mock_stream_llm.side_effect = _fake_stream
        config = _make_config()
        agent = AgentLoop(config)

        text_events = []
        async for event in agent.stream("Hi"):
            if event.event == AgentEvent.TEXT_DELTA:
                text_events.append(event)

        # Two token-level deltas
        assert len(text_events) == 2
        assert text_events[0].data["text"] == "Hello "
        assert text_events[1].data["text"] == "stream!"

    @patch("core.agent_loop.stream_chat_completion")
    async def test_stream_with_tools(self, mock_stream_llm):
        """stream() yields tool events during tool-use cycles."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 1
        registry = _mock_registry(execute_results={"recall": tool_result})

        call_count = 0

        async def _fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            cb = kwargs.get("on_text_delta")
            if call_count == 1:
                if cb:
                    await cb("Searching.")
                return _tool_response("Searching.", [_tool_call("recall", {"q": "x"})])
            else:
                if cb:
                    await cb("Found.")
                return _text_response("Found.")

        mock_stream_llm.side_effect = _fake_stream
        config = _make_config(registry=registry)
        agent = AgentLoop(config)

        event_types = []
        async for event in agent.stream("Find it"):
            event_types.append(event.event)

        assert AgentEvent.TOOL_START in event_types
        assert AgentEvent.TOOL_RESULT in event_types


# ============================================================================
# Unit: error handling
# ============================================================================


class TestErrorHandling:
    @patch("core.agent_loop.chat_completion")
    async def test_llm_error_returns_error_result(self, mock_llm):
        """LLM call failure returns error stopped_reason."""
        mock_llm.side_effect = RuntimeError("API error")
        config = _make_config()
        agent = AgentLoop(config)
        result = await agent.run("Fail please")

        assert result.stopped_reason == "error"
        assert result.iterations == 1
        assert result.timed_out is False

    @patch("core.agent_loop.chat_completion")
    async def test_tool_error_visible_to_llm(self, mock_llm):
        """Tool execution error is sent back to LLM as tool message."""
        fail_result = ToolResult.error_result(
            "Permission denied", ToolErrorType.PERMISSION_DENIED
        )
        fail_result.energy_spent = 1
        registry = _mock_registry(execute_results={"write_file": fail_result})

        mock_llm.side_effect = [
            _tool_response(
                "Writing.", [_tool_call("write_file", {"path": "/x"}, call_id="call_1")]
            ),
            _text_response("Failed to write."),
        ]

        config = _make_config(registry=registry)
        agent = AgentLoop(config)
        result = await agent.run("Write a file")

        # Verify error was passed back in messages
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "Permission denied" in tool_msgs[0]["content"]

        # LLM saw the error and responded
        assert result.text == "Failed to write."

    @patch("core.agent_loop.chat_completion")
    async def test_empty_tool_calls_list(self, mock_llm):
        """Empty tool_calls list treated as no tool calls."""
        mock_llm.return_value = {"content": "No tools.", "tool_calls": [], "raw": None}
        config = _make_config()
        agent = AgentLoop(config)
        result = await agent.run("Test")

        assert result.stopped_reason == "completed"
        assert len(result.tool_calls_made) == 0


# ============================================================================
# Unit: message format
# ============================================================================


class TestMessageFormat:
    @patch("core.agent_loop.chat_completion")
    async def test_provider_messages_strip_internal_receipt_metadata(self, mock_llm):
        mock_llm.return_value = _text_response("Understood.")
        receipt = {"name": "remember", "success": True}
        history = [
            {
                "role": "assistant",
                "content": "Stored it.",
                "metadata": {"action_receipts": [receipt]},
            }
        ]

        result = await AgentLoop(_make_config()).run("Continue", history=history)

        sent_messages = mock_llm.call_args.kwargs["messages"]
        assert all("metadata" not in message for message in sent_messages)
        stored_history = [
            message for message in result.messages if message.get("role") == "assistant"
        ]
        assert stored_history[0]["metadata"]["action_receipts"] == [receipt]

    @patch("core.agent_loop.chat_completion")
    async def test_assistant_message_includes_tool_calls(self, mock_llm):
        """Assistant message with tool calls includes them in OpenAI format."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0
        registry = _mock_registry(execute_results={"tool": tool_result})

        call_id = "call_test123"
        mock_llm.side_effect = [
            _tool_response(
                "Using tool.", [_tool_call("tool", {"x": 1}, call_id=call_id)]
            ),
            _text_response("Done."),
        ]

        config = _make_config(registry=registry)
        agent = AgentLoop(config)
        result = await agent.run("Test")

        # Find the assistant message with tool_calls
        assistant_msgs = [
            m
            for m in result.messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_msgs) == 1

        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc["id"] == call_id
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "tool"
        # arguments should be a JSON string
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"x": 1}

    def test_to_openai_tool_call_serializes_arguments(self):
        """_to_openai_tool_call converts dict arguments to JSON string."""
        result = _to_openai_tool_call(
            {
                "id": "call_1",
                "name": "test",
                "arguments": {"key": "value"},
            }
        )
        assert result["function"]["arguments"] == '{"key": "value"}'
        assert result["type"] == "function"

    def test_to_openai_tool_call_preserves_string_arguments(self):
        """If arguments is already a string, preserve it."""
        result = _to_openai_tool_call(
            {
                "id": "call_1",
                "name": "test",
                "arguments": '{"key": "value"}',
            }
        )
        assert result["function"]["arguments"] == '{"key": "value"}'

    def test_to_openai_tool_call_generates_id(self):
        """Missing ID gets a generated UUID."""
        result = _to_openai_tool_call({"name": "test", "arguments": {}})
        assert result["id"] is not None
        assert len(result["id"]) > 0


# ============================================================================
# Unit: result dataclass
# ============================================================================


class TestAgentLoopResult:
    def test_defaults(self):
        result = AgentLoopResult(
            text="hello",
            messages=[],
            tool_calls_made=[],
            iterations=1,
            energy_spent=0,
        )
        assert result.timed_out is False
        assert result.stopped_reason == "completed"

    def test_timeout_result(self):
        result = AgentLoopResult(
            text="",
            messages=[],
            tool_calls_made=[],
            iterations=0,
            energy_spent=0,
            timed_out=True,
            stopped_reason="timeout",
        )
        assert result.timed_out is True
        assert result.stopped_reason == "timeout"


# ============================================================================
# Unit: config dataclass
# ============================================================================


class TestAgentLoopConfig:
    def test_defaults(self):
        registry = _mock_registry()
        config = AgentLoopConfig(
            tool_context=ToolContext.CHAT,
            system_prompt="test",
            llm_config=_make_llm_config(),
            registry=registry,
            pool=registry.pool,
        )
        assert config.energy_budget is None
        assert config.max_iterations is None
        assert config.timeout_seconds == 300.0
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.session_id is None
        assert config.heartbeat_id is None
        assert config.on_event is None
        assert config.on_approval is None


# ============================================================================
# Unit: event dataclass
# ============================================================================


class TestAgentEventData:
    def test_defaults(self):
        event = AgentEventData(event=AgentEvent.LOOP_START)
        assert event.data == {}
        assert event.timestamp > 0

    def test_with_data(self):
        event = AgentEventData(
            event=AgentEvent.TEXT_DELTA,
            data={"text": "hello"},
        )
        assert event.data["text"] == "hello"
        assert event.event == AgentEvent.TEXT_DELTA


# ============================================================================
# Integration: real registry with mocked LLM
# ============================================================================


class TestIntegrationWithRegistry:
    @patch("core.agent_loop.chat_completion")
    async def test_chat_mode_with_real_registry(self, mock_llm, db_pool):
        """Chat mode uses real registry, unlimited energy."""
        from core.tools.registry import create_default_registry

        registry = create_default_registry(db_pool)

        mock_llm.side_effect = [
            _tool_response(
                "Let me recall.",
                [_tool_call("recall", {"query": "test"})],
            ),
            _text_response("I found something."),
        ]

        config = AgentLoopConfig(
            tool_context=ToolContext.CHAT,
            system_prompt="Test assistant.",
            llm_config=_make_llm_config(),
            registry=registry,
            pool=db_pool,
            energy_budget=None,
            timeout_seconds=30.0,
        )
        agent = AgentLoop(config)
        result = await agent.run("What do you know about test?")

        assert result.iterations == 2
        assert result.stopped_reason == "completed"
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0]["name"] == "recall"

    @patch("core.agent_loop.chat_completion")
    async def test_heartbeat_mode_with_energy(self, mock_llm, db_pool):
        """Heartbeat mode with energy budget stops when energy is used up."""
        from core.tools.registry import create_default_registry

        registry = create_default_registry(db_pool)

        mock_llm.side_effect = [
            _tool_response(
                "Recalling.",
                [_tool_call("recall", {"query": "goals"})],
            ),
            # After recall (cost 1), LLM tries another tool
            _tool_response(
                "Reflecting.",
                [_tool_call("reflect", {"query": "status"})],
            ),
            _text_response("Reflected."),
        ]

        config = AgentLoopConfig(
            tool_context=ToolContext.HEARTBEAT,
            system_prompt="Heartbeat agent.",
            llm_config=_make_llm_config(),
            registry=registry,
            pool=db_pool,
            energy_budget=20,
            timeout_seconds=30.0,
            heartbeat_id="hb-test-123",
        )
        agent = AgentLoop(config)
        result = await agent.run("Run heartbeat cycle")

        assert result.iterations >= 1
        assert result.energy_spent >= 0

    @patch("core.agent_loop.chat_completion")
    async def test_tool_specs_filtered_by_context(self, mock_llm, db_pool):
        """Registry returns context-appropriate tool specs."""
        from core.tools.registry import create_default_registry

        registry = create_default_registry(db_pool)

        mock_llm.return_value = _text_response("Done.")

        config = AgentLoopConfig(
            tool_context=ToolContext.CHAT,
            system_prompt="Test.",
            llm_config=_make_llm_config(),
            registry=registry,
            pool=db_pool,
            timeout_seconds=10.0,
        )
        agent = AgentLoop(config)
        await agent.run("Hi")

        # Verify tools were passed to LLM
        call_args = mock_llm.call_args
        tools = call_args.kwargs.get("tools") or []
        tool_names = [t["function"]["name"] for t in tools]

        # recall should be available in chat context
        assert "recall" in tool_names


# ============================================================================
# Unit: continuation nudge (Gap 5)
# ============================================================================


class TestContinuationNudge:
    @patch("core.agent_loop.chat_completion")
    async def test_no_continuation_by_default(self, mock_llm):
        """max_continuations=0 (default) exits immediately when no tool calls."""
        mock_llm.return_value = _text_response("Done.")
        config = _make_config()
        agent = AgentLoop(config)
        result = await agent.run("Hi")

        assert result.stopped_reason == "completed"
        assert result.continuations_used == 0
        assert result.iterations == 1

    @patch("core.agent_loop.chat_completion")
    async def test_nudge_injects_prompt_and_continues(self, mock_llm):
        """Continuation nudge injects prompt as user message and re-enters loop."""
        mock_llm.side_effect = [
            _text_response("I'm done."),  # First: no tool calls → nudge
            _text_response("Verified."),  # Second: after nudge, responds again
        ]

        config = _make_config(
            continuation_prompt="Did you verify your work?",
            max_continuations=1,
        )
        agent = AgentLoop(config)
        result = await agent.run("Do something")

        assert result.text == "Verified."
        assert result.continuations_used == 1
        assert result.iterations == 2
        # The nudge prompt should be in the messages
        user_msgs = [m for m in result.messages if m.get("role") == "user"]
        assert any("verify" in m["content"].lower() for m in user_msgs)

    @patch("core.agent_loop.chat_completion")
    async def test_nudge_leads_to_tool_calls(self, mock_llm):
        """After nudge, LLM may decide to call tools (self-correction)."""
        tool_result = ToolResult.success_result("test passed")
        tool_result.energy_spent = 1
        registry = _mock_registry(execute_results={"run_test": tool_result})

        mock_llm.side_effect = [
            _text_response("I think I'm done."),  # No tools → nudge
            _tool_response(
                "Let me verify.", [_tool_call("run_test", {})]
            ),  # After nudge: tool
            _text_response("Tests pass, all good."),  # Final response
        ]

        config = _make_config(
            registry=registry,
            continuation_prompt="Check your work.",
            max_continuations=1,
        )
        agent = AgentLoop(config)
        result = await agent.run("Build feature")

        assert result.text == "Tests pass, all good."
        assert result.continuations_used == 1
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0]["name"] == "run_test"
        assert result.iterations == 3

    @patch("core.agent_loop.chat_completion")
    async def test_max_continuations_respected(self, mock_llm):
        """Stops nudging after max_continuations even if LLM keeps producing text-only."""
        mock_llm.side_effect = [
            _text_response("Attempt 1."),  # nudge 1
            _text_response("Attempt 2."),  # nudge 2
            _text_response("Attempt 3."),  # no more nudges, exits
        ]

        config = _make_config(
            continuation_prompt="Try again.",
            max_continuations=2,
        )
        agent = AgentLoop(config)
        result = await agent.run("Do something")

        assert result.continuations_used == 2
        assert result.iterations == 3
        assert result.text == "Attempt 3."
        assert result.stopped_reason == "completed"

    @patch("core.agent_loop.chat_completion")
    async def test_energy_budget_enforced_after_nudge(self, mock_llm):
        """Energy check still fires after continuation nudge."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 8
        registry = _mock_registry(execute_results={"tool": tool_result})

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),  # Spends 8 energy
            _text_response("Maybe done."),  # nudge fires
            # After nudge, energy check at top of loop: 8 >= 10 → no, continues
            # But if another tool call spends more... let's just test text response
            _text_response("Confirmed."),
        ]

        config = _make_config(
            registry=registry,
            energy_budget=10,
            continuation_prompt="Verify.",
            max_continuations=1,
        )
        agent = AgentLoop(config)
        result = await agent.run("Test")

        assert result.continuations_used == 1
        # Energy was 8, budget was 10, so loop continued after nudge
        assert result.stopped_reason == "completed"
        assert result.energy_spent == 8

    @patch("core.agent_loop.chat_completion")
    async def test_continuation_event_emitted(self, mock_llm):
        """CONTINUATION event is emitted when nudge fires."""
        events: list[AgentEventData] = []

        async def _capture(e: AgentEventData) -> None:
            events.append(e)

        mock_llm.side_effect = [
            _text_response("Done."),
            _text_response("Verified."),
        ]

        config = _make_config(
            continuation_prompt="Check.",
            max_continuations=1,
            on_event=_capture,
        )
        agent = AgentLoop(config)
        await agent.run("Test")

        cont_events = [e for e in events if e.event == AgentEvent.CONTINUATION]
        assert len(cont_events) == 1
        assert cont_events[0].data["continuation_number"] == 1
        assert cont_events[0].data["max_continuations"] == 1


# ============================================================================
# Unit: runtime ContextOverrides (Gap 4)
# ============================================================================


class TestRuntimeContextOverrides:
    @patch("core.agent_loop.chat_completion")
    async def test_runtime_overrides_enable_shell(self, mock_llm):
        """Runtime context_overrides grants allow_shell."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0
        registry = _mock_registry(execute_results={"tool": tool_result})

        captured_ctx = None

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return tool_result

        registry.execute = AsyncMock(side_effect=_capture_execute)

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _text_response("Done."),
        ]

        config = _make_config(
            registry=registry,
            context_overrides=ContextOverrides(allow_shell=True),
        )
        agent = AgentLoop(config)
        await agent.run("Test")

        assert captured_ctx is not None
        assert captured_ctx.allow_shell is True
        assert captured_ctx.allow_file_write is False  # Not granted

    @patch("core.agent_loop.chat_completion")
    async def test_runtime_overrides_enable_file_write(self, mock_llm):
        """Runtime context_overrides grants allow_file_write."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0
        registry = _mock_registry(execute_results={"tool": tool_result})

        captured_ctx = None

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return tool_result

        registry.execute = AsyncMock(side_effect=_capture_execute)

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _text_response("Done."),
        ]

        config = _make_config(
            registry=registry,
            context_overrides=ContextOverrides(allow_file_write=True),
        )
        agent = AgentLoop(config)
        await agent.run("Test")

        assert captured_ctx is not None
        assert captured_ctx.allow_file_write is True
        assert captured_ctx.allow_shell is False  # Not granted

    @patch("core.agent_loop.chat_completion")
    async def test_no_runtime_overrides_means_db_only(self, mock_llm):
        """context_overrides=None means only DB config applies (default behavior)."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0
        registry = _mock_registry(execute_results={"tool": tool_result})

        captured_ctx = None

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return tool_result

        registry.execute = AsyncMock(side_effect=_capture_execute)

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _text_response("Done."),
        ]

        # No context_overrides (default None)
        config = _make_config(registry=registry)
        agent = AgentLoop(config)
        await agent.run("Test")

        assert captured_ctx is not None
        # Default ToolsConfig has no overrides → both False
        assert captured_ctx.allow_shell is False
        assert captured_ctx.allow_file_write is False

    @patch("core.agent_loop.chat_completion")
    async def test_runtime_overrides_additive_cannot_revoke(self, mock_llm):
        """Runtime overrides are additive — they can't revoke DB-granted permissions."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0

        # DB config grants shell
        tools_config = ToolsConfig(
            context_overrides={
                ToolContext.CHAT: ContextOverrides(allow_shell=True),
            },
        )
        registry = _mock_registry(
            execute_results={"tool": tool_result},
            config=tools_config,
        )

        captured_ctx = None

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return tool_result

        registry.execute = AsyncMock(side_effect=_capture_execute)

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _text_response("Done."),
        ]

        # Runtime overrides: allow_shell=False (trying to revoke)
        config = _make_config(
            registry=registry,
            context_overrides=ContextOverrides(allow_shell=False),
        )
        agent = AgentLoop(config)
        await agent.run("Test")

        assert captured_ctx is not None
        # DB granted allow_shell=True, runtime can't revoke it
        assert captured_ctx.allow_shell is True

    @patch("core.agent_loop.chat_completion")
    async def test_allow_all_enables_both(self, mock_llm):
        """allow_all=True grants both allow_shell and allow_file_write."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 0
        registry = _mock_registry(execute_results={"tool": tool_result})

        captured_ctx = None

        async def _capture_execute(name, args, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return tool_result

        registry.execute = AsyncMock(side_effect=_capture_execute)

        mock_llm.side_effect = [
            _tool_response("Go", [_tool_call("tool", {})]),
            _text_response("Done."),
        ]

        config = _make_config(
            registry=registry,
            context_overrides=ContextOverrides(allow_all=True),
        )
        agent = AgentLoop(config)
        await agent.run("Test")

        assert captured_ctx is not None
        assert captured_ctx.allow_shell is True
        assert captured_ctx.allow_file_write is True


# ============================================================================
# Unit: planning phases (Gap 1)
# ============================================================================


class TestPlanningPhases:
    @patch("core.agent_loop.chat_completion")
    async def test_no_planning_by_default(self, mock_llm):
        """enable_planning=False (default) goes straight to execute loop."""
        mock_llm.return_value = _text_response("Done.")
        config = _make_config()
        agent = AgentLoop(config)
        result = await agent.run("Hi")

        assert result.stopped_reason == "completed"
        assert result.plan_text == ""
        assert result.phases_completed == []
        assert result.iterations == 1

    @patch("core.agent_loop.chat_completion")
    async def test_plan_phase_produces_plan_text(self, mock_llm):
        """Plan phase calls LLM with tools=None and stores plan_text."""
        mock_llm.side_effect = [
            _text_response("Plan: 1. Search 2. Summarize"),  # Plan phase (no tools)
            _text_response("Executing the plan."),  # Execute phase
            _text_response("All looks good."),  # Verify phase
        ]

        config = _make_config(enable_planning=True)
        agent = AgentLoop(config)
        result = await agent.run("Do research")

        assert result.plan_text == "Plan: 1. Search 2. Summarize"
        assert "plan" in result.phases_completed
        assert "execute" in result.phases_completed
        assert "verify" in result.phases_completed

        # Plan phase should have been called with tools=None
        first_call = mock_llm.call_args_list[0]
        assert first_call.kwargs.get("tools") is None

    @patch("core.agent_loop.chat_completion")
    async def test_verify_phase_catches_incomplete_work(self, mock_llm):
        """Verify phase LLM can call tools (corrections happen here)."""
        tool_result = ToolResult.success_result("test output: 1 failure")
        tool_result.energy_spent = 1
        fix_result = ToolResult.success_result("fixed")
        fix_result.energy_spent = 1
        registry = _mock_registry(
            execute_results={"run_test": tool_result, "fix_code": fix_result}
        )

        mock_llm.side_effect = [
            _text_response("Plan: run tests then fix."),  # Plan
            _text_response("Done writing code."),  # Execute: text only
            # Verify phase:
            _tool_response(
                "Let me check.", [_tool_call("run_test", {})]
            ),  # Verify: calls tool
            _tool_response(
                "Fixing.", [_tool_call("fix_code", {})]
            ),  # Verify: calls fix
            _text_response("All tests pass now."),  # Verify: done
        ]

        config = _make_config(registry=registry, enable_planning=True)
        agent = AgentLoop(config)
        result = await agent.run("Write feature")

        assert result.text == "All tests pass now."
        assert len(result.tool_calls_made) == 2
        assert result.phases_completed == ["plan", "execute", "verify"]

    @patch("core.agent_loop.chat_completion")
    async def test_verify_text_only_completes(self, mock_llm):
        """If verify phase LLM produces text only, loop completes normally."""
        mock_llm.side_effect = [
            _text_response("Plan: just say hello."),  # Plan
            _text_response("Hello!"),  # Execute
            _text_response("Looks correct."),  # Verify: text only
        ]

        config = _make_config(enable_planning=True)
        agent = AgentLoop(config)
        result = await agent.run("Test")

        assert result.text == "Looks correct."
        assert result.stopped_reason == "completed"
        assert result.phases_completed == ["plan", "execute", "verify"]

    @patch("core.agent_loop.chat_completion")
    async def test_plan_phase_error_returns_error(self, mock_llm):
        """LLM error during plan phase returns stopped_reason='error'."""
        mock_llm.side_effect = RuntimeError("API rate limit")

        config = _make_config(enable_planning=True)
        agent = AgentLoop(config)
        result = await agent.run("Test")

        assert result.stopped_reason == "error"
        assert "plan" in result.phases_completed

    @patch("core.agent_loop.chat_completion")
    async def test_execute_energy_exhaustion_skips_verify(self, mock_llm):
        """If execute phase runs out of energy, verify is skipped."""
        tool_result = ToolResult.success_result("ok")
        tool_result.energy_spent = 10
        registry = _mock_registry(execute_results={"tool": tool_result})

        mock_llm.side_effect = [
            _text_response("Plan: use expensive tool."),  # Plan
            _tool_response("Go", [_tool_call("tool", {})]),  # Execute: spends 10
            # Energy check: 10 >= 10 -> energy exhausted, skip verify
        ]

        config = _make_config(
            registry=registry,
            energy_budget=10,
            enable_planning=True,
        )
        agent = AgentLoop(config)
        result = await agent.run("Test")

        assert result.stopped_reason == "energy"
        assert "plan" in result.phases_completed
        assert "execute" in result.phases_completed
        assert "verify" not in result.phases_completed

    @patch("core.agent_loop.chat_completion")
    async def test_phase_change_events_emitted(self, mock_llm):
        """PHASE_CHANGE events are emitted for each phase."""
        events: list[AgentEventData] = []

        async def _capture(e: AgentEventData) -> None:
            events.append(e)

        mock_llm.side_effect = [
            _text_response("My plan."),
            _text_response("Executed."),
            _text_response("Verified."),
        ]

        config = _make_config(enable_planning=True, on_event=_capture)
        agent = AgentLoop(config)
        await agent.run("Test")

        phase_events = [e for e in events if e.event == AgentEvent.PHASE_CHANGE]
        phases = [e.data["phase"] for e in phase_events]
        assert phases == ["plan", "execute", "verify"]

    @patch("core.agent_loop.chat_completion")
    async def test_custom_planning_and_verify_prompts(self, mock_llm):
        """Custom planning_prompt and verify_prompt are used."""
        mock_llm.side_effect = [
            _text_response("Custom plan."),
            _text_response("Executed."),
            _text_response("Custom verify."),
        ]

        config = _make_config(
            enable_planning=True,
            planning_prompt="Make a detailed plan for this task.",
            verify_prompt="Did everything work correctly?",
        )
        agent = AgentLoop(config)
        result = await agent.run("Do it")

        # Check that custom prompts ended up in messages
        user_msgs = [m["content"] for m in result.messages if m.get("role") == "user"]
        assert "Make a detailed plan for this task." in user_msgs
        assert "Did everything work correctly?" in user_msgs

    @patch("core.agent_loop.chat_completion")
    async def test_execute_with_tools_then_verify(self, mock_llm):
        """Full flow: plan -> execute with tools -> verify."""
        tool_result = ToolResult.success_result("file written")
        tool_result.energy_spent = 2
        registry = _mock_registry(execute_results={"write_file": tool_result})

        mock_llm.side_effect = [
            _text_response("Plan: write config file."),  # Plan
            _tool_response(
                "Writing.", [_tool_call("write_file", {"path": "/x"})]
            ),  # Execute: tool
            _text_response("File written."),  # Execute: done
            _text_response("File exists, looks good."),  # Verify
        ]

        config = _make_config(registry=registry, enable_planning=True)
        agent = AgentLoop(config)
        result = await agent.run("Create config")

        assert result.text == "File exists, looks good."
        assert len(result.tool_calls_made) == 1
        assert result.plan_text == "Plan: write config file."
        assert result.phases_completed == ["plan", "execute", "verify"]
        assert result.energy_spent == 2


# ============================================================================
# Action-claim guardrail (#38)
# ============================================================================


class TestActionClaimGuardrail:
    @patch("core.agent_loop.chat_completion")
    async def test_unsupported_claim_gets_correction(self, mock_llm):
        """A prose claim of a memory write with no tool call is corrected."""
        mock_llm.return_value = _text_response(
            "I've stored that as a memory for next time."
        )
        events: list[AgentEventData] = []

        async def on_event(event: AgentEventData) -> None:
            events.append(event)

        config = _make_config(on_event=on_event)
        agent = AgentLoop(config)
        result = await agent.run("Please remember this")

        assert "[Correction]" in result.text
        assert "memory_write" in result.text
        flagged = [e for e in events if e.event == AgentEvent.CLAIM_FLAGGED]
        assert len(flagged) == 1
        assert flagged[0].data["findings"][0]["kind"] == "memory_write"
        # The correction rides TEXT_DELTA so streaming clients see it inline.
        corrections = [
            e
            for e in events
            if e.event == AgentEvent.TEXT_DELTA and e.data.get("correction")
        ]
        assert len(corrections) == 1

    @patch("core.agent_loop.chat_completion")
    async def test_supported_claim_untouched(self, mock_llm):
        """The same claim with an actual successful remember call stays clean."""
        remember_result = ToolResult.success_result({"memory_id": "m-1"})
        registry = _mock_registry(execute_results={"remember": remember_result})
        mock_llm.side_effect = [
            _tool_response("Storing.", [_tool_call("remember", {"content": "x"})]),
            _text_response("I've stored that as a memory for next time."),
        ]
        config = _make_config(registry=registry)
        agent = AgentLoop(config)
        result = await agent.run("Please remember this")

        assert "[Correction]" not in result.text
        assert result.text == "I've stored that as a memory for next time."
        assert result.turn_id is not None
        assert result.tool_results[0]["output"] == {"memory_id": "m-1"}

    @patch("core.agent_loop.chat_completion")
    async def test_visible_intermediate_claim_is_checked(self, mock_llm):
        """Claims streamed before a later tool iteration cannot evade auditing."""
        recall_result = ToolResult.success_result({"results": []})
        registry = _mock_registry(execute_results={"recall": recall_result})
        mock_llm.side_effect = [
            _tool_response(
                "I've stored that as a memory for next time.",
                [_tool_call("recall", {"query": "the agreement"})],
            ),
            _text_response("I found no matching memory."),
        ]
        config = _make_config(registry=registry)
        result = await AgentLoop(config).run("Did you keep the agreement?")

        assert "I've stored that" in result.visible_text
        assert "[Correction]" in result.text
        assert "memory_write" in result.text

    @patch("core.agent_loop.chat_completion")
    async def test_kill_switch_disables_guardrail(self, mock_llm):
        mock_llm.return_value = _text_response(
            "I've stored that as a memory for next time."
        )
        config = _make_config()
        agent = AgentLoop(config)
        async with _DB_POOL.acquire() as conn:
            await conn.execute(
                "UPDATE config SET value = 'false'::jsonb WHERE key = 'guardrails.action_claims.enabled'"
            )
        try:
            result = await agent.run("Please remember this")
            assert "[Correction]" not in result.text
        finally:
            async with _DB_POOL.acquire() as conn:
                await conn.execute(
                    "UPDATE config SET value = 'true'::jsonb WHERE key = 'guardrails.action_claims.enabled'"
                )

    @patch("core.llm_json.chat_json", new_callable=AsyncMock)
    @patch("core.agent_loop.chat_completion")
    async def test_llm_verifier_can_clear_findings(self, mock_llm, mock_chat_json):
        """With the verifier enabled, an empty confirmed list clears the flag."""
        mock_llm.return_value = _text_response(
            "I've stored that as a memory for next time."
        )
        mock_chat_json.return_value = ({"confirmed": [], "additional": []}, "{}")
        config = _make_config()
        agent = AgentLoop(config)
        prior_receipt = {
            "name": "remember",
            "success": True,
            "arguments": {"content": "a prior fact"},
            "result": {"memory_id": str(uuid.uuid4())},
        }
        async with _DB_POOL.acquire() as conn:
            await conn.execute(
                "UPDATE config SET value = 'true'::jsonb WHERE key = 'guardrails.action_claims.llm_verifier_enabled'"
            )
        try:
            with patch(
                "core.llm_config.load_llm_config", new_callable=AsyncMock
            ) as mock_cfg:
                mock_cfg.return_value = {
                    "provider": "openai",
                    "model": "test",
                    "api_key": "k",
                }
                result = await agent.run(
                    "Please remember this",
                    history=[
                        {
                            "role": "assistant",
                            "content": "I stored an earlier fact.",
                            "metadata": {"action_receipts": [prior_receipt]},
                        }
                    ],
                )
            assert "[Correction]" not in result.text
            verifier_messages = mock_chat_json.await_args.kwargs["messages"]
            payload = json.loads(verifier_messages[1]["content"])
            assert payload["prior_action_receipts"] == [prior_receipt]
        finally:
            async with _DB_POOL.acquire() as conn:
                await conn.execute(
                    "UPDATE config SET value = 'false'::jsonb WHERE key = 'guardrails.action_claims.llm_verifier_enabled'"
                )
