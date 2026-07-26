"""Tests for services.hexis_rlm -- RLM heartbeat integration."""

import asyncio
import json
import re

import pytest

from services.hexis_rlm import (
    find_code_blocks,
    find_final_answer,
    format_execution_result,
    format_iteration,
    resolve_final_answer,
)
from services.rlm_repl import HexisLocalREPL, REPLResult


class TestParsing:
    """Unit tests for vendored parsing utilities."""

    def test_find_code_blocks_single(self):
        text = "Here is code:\n```repl\nprint('hello')\n```\nDone."
        blocks = find_code_blocks(text)
        assert len(blocks) == 1
        assert "print('hello')" in blocks[0]

    def test_find_code_blocks_multiple(self):
        text = "```repl\nx = 1\n```\ntext\n```repl\ny = 2\n```"
        blocks = find_code_blocks(text)
        assert len(blocks) == 2

    def test_find_code_blocks_none(self):
        text = "No code here. ```python\nfoo\n```"
        blocks = find_code_blocks(text)
        assert blocks == []

    def test_find_final_direct(self):
        text = "I'm done reasoning.\nFINAL({\"reasoning\": \"test\", \"actions\": []})"
        answer = find_final_answer(text)
        assert answer is not None
        parsed = json.loads(answer)
        assert parsed["reasoning"] == "test"

    def test_find_final_var(self):
        repl = HexisLocalREPL()
        repl.setup(context_payload=None)
        try:
            repl.execute_code("my_answer = 'the result'")
            text = "FINAL_VAR(my_answer)"
            answer = find_final_answer(text, repl)
            assert answer == "the result"
        finally:
            repl.cleanup()

    def test_missing_final_var_is_not_final_answer(self):
        repl = HexisLocalREPL()
        repl.setup(context_payload=None)
        try:
            text = "FINAL_VAR(response)"
            resolution = resolve_final_answer(text, repl)
            assert resolution.answer is None
            assert resolution.error is not None
            assert "Variable 'response' not found" in resolution.error
            assert find_final_answer(text, repl) is None
        finally:
            repl.cleanup()

    def test_find_final_none(self):
        text = "Still thinking..."
        assert find_final_answer(text) is None

    def test_format_execution_result_stdout(self):
        result = REPLResult(stdout="hello\n", stderr="", execution_time=0.1, local_vars={"x": "int"})
        formatted = format_execution_result(result)
        assert "hello" in formatted
        assert "x" in formatted

    def test_format_execution_result_stderr(self):
        result = REPLResult(stdout="", stderr="ZeroDivisionError: division by zero", execution_time=0.1, local_vars={})
        formatted = format_execution_result(result)
        assert "ZeroDivisionError" in formatted

    def test_format_iteration_basic(self):
        code = "x = 1\nprint(x)"
        result = REPLResult(stdout="1\n", stderr="", execution_time=0.05, local_vars={"x": "int"})
        messages = format_iteration("I'll run some code:\n```repl\nx = 1\nprint(x)\n```", [code], [result])
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"
        assert "REPL output" in messages[1]["content"]


class TestMemoryEnv:
    """Unit tests for RLM memory environment."""

    def test_workspace_budgets_enforce_count(self):
        from unittest.mock import MagicMock
        from services.rlm_memory_env import RLMMemoryEnv, RLMWorkspace, WorkspaceBudgets
        from core.memory_repo import MemoryRepo

        mock_repo = MagicMock(spec=MemoryRepo)
        # Return 30 memories to exceed the budget of 5
        mock_repo.fetch_by_ids.return_value = [
            {"id": str(i), "content": f"memory {i}", "type": "semantic"}
            for i in range(30)
        ]

        budgets = WorkspaceBudgets(max_loaded_memories=5, max_loaded_chars=100_000)
        workspace = RLMWorkspace(budgets=budgets)
        env = RLMMemoryEnv(mock_repo, workspace)

        env.memory_fetch([str(i) for i in range(30)])

        # After budget enforcement, should have at most max_loaded_memories
        assert len(workspace.loaded_memories) <= budgets.max_loaded_memories
        # Notes should have been populated by auto-summarize
        assert workspace.metrics.summarize_events >= 1

    def test_workspace_drop(self):
        from services.rlm_memory_env import RLMMemoryEnv, RLMWorkspace
        from unittest.mock import MagicMock
        from core.memory_repo import MemoryRepo

        mock_repo = MagicMock(spec=MemoryRepo)
        workspace = RLMWorkspace()
        workspace.loaded_memories = [
            {"id": "a", "content": "mem a"},
            {"id": "b", "content": "mem b"},
        ]
        env = RLMMemoryEnv(mock_repo, workspace)

        env.workspace_drop("loaded_memories", keep_ids=["a"])
        assert len(workspace.loaded_memories) == 1
        assert workspace.loaded_memories[0]["id"] == "a"

        workspace.loaded_documents = [{"document_id": "doc-1", "content": "doc"}]
        workspace.notes = "note"
        env.workspace_drop("all")
        assert workspace.loaded_memories == []
        assert workspace.loaded_documents == []
        assert workspace.notes == ""

    def test_workspace_status(self):
        from services.rlm_memory_env import RLMMemoryEnv, RLMWorkspace
        from unittest.mock import MagicMock
        from core.memory_repo import MemoryRepo

        mock_repo = MagicMock(spec=MemoryRepo)
        workspace = RLMWorkspace()
        workspace.loaded_memories = [{"id": "a", "content": "hello"}]
        workspace.notes = "some notes"
        workspace.memory_stubs = [{"id": "a"}]
        env = RLMMemoryEnv(mock_repo, workspace)

        status = env.workspace_status()
        assert status["loaded_memories_count"] == 1
        assert status["notes_chars"] == len("some notes")
        assert status["stubs_count"] == 1

    def test_source_document_workspace_syscalls(self):
        from services.rlm_memory_env import RLMMemoryEnv, RLMWorkspace
        from unittest.mock import MagicMock
        from core.memory_repo import MemoryRepo

        mock_repo = MagicMock(spec=MemoryRepo)
        mock_repo.search_documents.return_value = [
            {"document_id": "doc-1", "title": "With specification", "snippet": "bounded context"}
        ]
        mock_repo.fetch_documents.return_value = {
            "documents": [
                {"document_id": "doc-1", "title": "With specification", "content": "bounded context"}
            ],
            "count": 1,
        }
        mock_repo.load_documents_to_desk.return_value = {
            "count": 1,
            "desk_unit_ids": ["unit-1"],
        }

        workspace = RLMWorkspace()
        env = RLMMemoryEnv(mock_repo, workspace)

        stubs = env.document_search("With specification")
        fetched = env.document_fetch(document_ids=["doc-1"])
        loaded = env.document_load_to_desk(document_ids=["doc-1"], reason="need exact spec")

        assert stubs == mock_repo.search_documents.return_value
        assert fetched["count"] == 1
        assert loaded["desk_unit_ids"] == ["unit-1"]
        assert workspace.document_stubs == stubs
        assert workspace.loaded_documents[0]["document_id"] == "doc-1"
        status = env.workspace_status()
        assert status["document_stubs_count"] == 1
        assert status["loaded_documents_count"] == 1
        assert status["metrics"]["document_search_count"] == 1
        assert status["metrics"]["document_fetch_count"] == 1
        assert status["metrics"]["document_load_count"] == 1

        env.workspace_drop("loaded_documents", keep_ids=["missing"])
        assert workspace.loaded_documents == []

    def test_source_chunk_and_desk_syscalls(self):
        from services.rlm_memory_env import RLMMemoryEnv, RLMWorkspace
        from unittest.mock import MagicMock
        from core.memory_repo import MemoryRepo

        mock_repo = MagicMock(spec=MemoryRepo)
        mock_repo.search_document_chunks.return_value = [
            {"chunk_id": "chunk-1", "document_id": "doc-1", "chunk_index": 3,
             "snippet": "the retention window", "rank_components": {"lexical": 1.0}}
        ]
        mock_repo.fetch_document_chunks.return_value = {
            "chunks": [{"chunk_id": "chunk-1", "content": "the retention window is 90 days"}],
            "count": 1,
        }
        mock_repo.load_document_chunks_to_desk.return_value = {
            "count": 1, "desk_unit_ids": ["unit-9"],
        }
        mock_repo.list_desk.return_value = [
            {"desk_unit_id": "unit-9", "chunk_id": "chunk-1", "pinned": False}
        ]
        mock_repo.fetch_desk_item.return_value = {
            "desk_unit_id": "unit-9", "content": "the retention window is 90 days",
            "truncated": False,
        }
        mock_repo.pin_desk_item.return_value = {"desk_unit_id": "unit-9", "pinned": True}

        workspace = RLMWorkspace()
        env = RLMMemoryEnv(mock_repo, workspace)

        stubs = env.document_chunk_search("retention window", document_id="doc-1")
        fetched = env.document_chunk_fetch(["chunk-1"])
        loaded = env.document_chunk_load_to_desk(["chunk-1"], reason="multi-step", pin=True)
        desk = env.desk_list()
        item = env.desk_fetch("unit-9")
        pin = env.desk_pin("unit-9")

        assert workspace.document_chunk_stubs == stubs
        assert fetched["count"] == 1
        assert loaded["desk_unit_ids"] == ["unit-9"]
        assert workspace.desk_stubs == desk
        assert item["content"].startswith("the retention window")
        assert pin["pinned"] is True
        # Fetched chunk + desk content both count toward the char budget.
        assert workspace.loaded_documents and len(workspace.loaded_documents) == 2

        status = env.workspace_status()
        assert status["document_chunk_stubs_count"] == 1
        assert status["desk_stubs_count"] == 1
        assert status["metrics"]["document_chunk_search_count"] == 1
        assert status["metrics"]["document_chunk_fetch_count"] == 1
        assert status["metrics"]["document_chunk_load_count"] == 1
        assert status["metrics"]["desk_list_count"] == 1
        assert status["metrics"]["desk_fetch_count"] == 1
        assert status["metrics"]["desk_pin_count"] == 1

        # All six syscalls are exposed to the REPL namespace.
        fns = env.get_repl_functions()
        for name in ("document_chunk_search", "document_chunk_fetch",
                     "document_chunk_load_to_desk", "desk_list", "desk_fetch", "desk_pin"):
            assert name in fns


class TestReplToolBridge:
    """Unit tests for the sync-to-async tool bridge."""

    def _handler(self, *, name: str, contexts: set | None = None):
        from core.tools.base import ToolCategory, ToolContext, ToolHandler, ToolResult, ToolSpec

        class Handler(ToolHandler):
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name=name,
                    description=f"{name} description",
                    parameters={
                        "type": "object",
                        "properties": {"task": {"type": "string"}},
                        "required": ["task"],
                    },
                    category=ToolCategory.MEMORY,
                    energy_cost=2,
                    allowed_contexts=contexts or {ToolContext.HEARTBEAT},
                )

            async def execute(self, arguments, context):
                return ToolResult.success_result({"ok": True})

        return Handler()

    @pytest.mark.asyncio(loop_scope="module")
    async def test_list_tools_uses_enabled_registry_and_includes_schema(self):
        from core.tools.base import ToolContext
        from core.tools.repl_bridge import ReplToolBridge

        enabled_handler = self._handler(name="get_procedures")

        class Registry:
            async def get_enabled_tools(self, context):
                assert context == ToolContext.HEARTBEAT
                return [enabled_handler]

            def list_all(self):
                raise AssertionError("enabled registry path should be used")

        bridge = ReplToolBridge(Registry(), asyncio.get_running_loop())

        tools = await asyncio.to_thread(bridge.list_tools)

        assert tools == [{
            "name": "get_procedures",
            "description": "get_procedures description",
            "energy_cost": 2,
            "category": "memory",
            "is_read_only": True,
            "requires_approval": False,
            "supports_parallel": True,
            "optional": False,
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
            "allowed_contexts": ["heartbeat"],
        }]

    def test_list_tools_fallback_does_not_require_registry_handlers_attr(self):
        from core.tools.base import ToolContext
        from core.tools.repl_bridge import ReplToolBridge

        heartbeat_handler = self._handler(name="self_repair")
        chat_only_handler = self._handler(name="chat_only", contexts={ToolContext.CHAT})

        class Registry:
            def list_all(self):
                return [chat_only_handler, heartbeat_handler]

        loop = asyncio.new_event_loop()
        loop.close()
        bridge = ReplToolBridge(Registry(), loop)

        tools = bridge.list_tools()

        assert [tool["name"] for tool in tools] == ["self_repair"]
        assert tools[0]["parameters"]["required"] == ["task"]

    def test_tool_call_record_to_action_taken(self):
        from core.tools.repl_bridge import ToolCallRecord
        from core.tools.base import ToolResult

        result = ToolResult.success_result({"memories": 3})
        result.energy_spent = 2
        record = ToolCallRecord(tool_name="recall", arguments={"query": "test"}, result=result)
        record.end_time = record.start_time + 1.5

        action = record.to_action_taken()
        assert action["action"] == "recall"
        assert action["source"] == "rlm_repl"
        assert action["result"]["success"] is True
        assert action["result"]["energy_spent"] == 2

    def test_call_records_to_actions_taken(self):
        from core.tools.repl_bridge import ToolCallRecord, call_records_to_actions_taken
        from core.tools.base import ToolResult

        r1 = ToolResult.success_result({"ok": True})
        r1.energy_spent = 1
        r2 = ToolResult.success_result({"ok": True})
        r2.energy_spent = 3

        records = [
            ToolCallRecord(tool_name="recall", arguments={}, result=r1),
            ToolCallRecord(tool_name="reflect", arguments={}, result=r2),
            ToolCallRecord(tool_name="failed", arguments={}, result=None),  # No result
        ]

        actions = call_records_to_actions_taken(records)
        # Should skip the one without result
        assert len(actions) == 2
        assert actions[0]["action"] == "recall"
        assert actions[1]["action"] == "reflect"


class TestLegacyPathUnchanged:
    """Verify that legacy heartbeat path is not affected when RLM is off."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_start_heartbeat_default_kind(self, db_pool):
        """With heartbeat.use_rlm=false, start_heartbeat produces kind=heartbeat_decision."""
        async with db_pool.acquire() as conn:
            # Ensure RLM is off
            await conn.execute("SELECT set_config('heartbeat.use_rlm', 'false'::jsonb)")
            result = await conn.fetchval("SELECT run_heartbeat()")
            if result is None:
                pytest.skip("Heartbeat not ready (possibly not configured)")

            data = json.loads(result) if isinstance(result, str) else result
            if not isinstance(data, dict):
                pytest.skip("Unexpected heartbeat result format")

            calls = data.get("external_calls", [])
            if calls:
                call_input = calls[0].get("input", {})
                assert call_input.get("kind") == "heartbeat_decision"
