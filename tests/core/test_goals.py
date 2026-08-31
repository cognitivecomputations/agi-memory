"""
Tests for core/tools/goals.py — ManageGoalsHandler.

Covers: create, update_priority, add_progress, list actions,
validation, missing params, and DB integration.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from core.tools.base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
)
from core.tools.goals import ManageGoalsHandler, create_goal_tools

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


# ============================================================================
# Helpers
# ============================================================================


def _make_exec_context(
    *,
    pool=None,
    registry=None,
    tool_context: ToolContext = ToolContext.HEARTBEAT,
) -> ToolExecutionContext:
    """Build a ToolExecutionContext with optional mocked registry."""
    reg = registry or MagicMock()
    if pool:
        reg.pool = pool
    ctx = ToolExecutionContext(
        tool_context=tool_context,
        call_id=str(uuid.uuid4()),
    )
    ctx.registry = reg
    return ctx


# ============================================================================
# Unit: spec
# ============================================================================


class TestGoalsSpec:
    def test_spec_name(self):
        handler = ManageGoalsHandler()
        assert handler.spec.name == "manage_goals"

    def test_spec_category(self):
        handler = ManageGoalsHandler()
        assert handler.spec.category == ToolCategory.MEMORY

    def test_spec_allowed_contexts(self):
        handler = ManageGoalsHandler()
        assert ToolContext.HEARTBEAT in handler.spec.allowed_contexts
        assert ToolContext.CHAT in handler.spec.allowed_contexts
        assert ToolContext.MCP in handler.spec.allowed_contexts

    def test_spec_energy_cost(self):
        handler = ManageGoalsHandler()
        assert handler.spec.energy_cost == 1

    def test_create_goal_tools_returns_list(self):
        tools = create_goal_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], ManageGoalsHandler)

    @pytest.mark.parametrize(
        ("tool_context", "expected"),
        [
            (ToolContext.CHAT, "user_request"),
            (ToolContext.HEARTBEAT, "derived"),
            (ToolContext.MCP, "external"),
        ],
    )
    def test_goal_origin_comes_from_execution_context(self, tool_context, expected):
        ctx = _make_exec_context(tool_context=tool_context)
        assert ctx.trusted_goal_origin == expected


# ============================================================================
# Unit: validation
# ============================================================================


class TestGoalsValidation:
    async def test_invalid_action_rejected(self):
        handler = ManageGoalsHandler()
        registry = MagicMock()
        registry.pool = MagicMock()
        ctx = _make_exec_context(registry=registry)

        result = await handler.execute({"action": "invalid_action"}, ctx)
        assert not result.success
        assert result.error_type == ToolErrorType.INVALID_PARAMS

    async def test_missing_action_rejected(self):
        handler = ManageGoalsHandler()
        registry = MagicMock()
        registry.pool = MagicMock()
        ctx = _make_exec_context(registry=registry)

        result = await handler.execute({}, ctx)
        assert not result.success
        assert result.error_type == ToolErrorType.INVALID_PARAMS

    async def test_create_requires_title(self, db_pool):
        handler = ManageGoalsHandler()
        ctx = _make_exec_context(registry=MagicMock(pool=db_pool))

        result = await handler.execute({"action": "create"}, ctx)
        assert not result.success
        assert result.error_type == ToolErrorType.INVALID_PARAMS

    async def test_create_blank_title_rejected(self, db_pool):
        handler = ManageGoalsHandler()
        ctx = _make_exec_context(registry=MagicMock(pool=db_pool))

        result = await handler.execute({"action": "create", "title": "   "}, ctx)
        assert not result.success

    async def test_update_priority_requires_goal_id(self, db_pool):
        handler = ManageGoalsHandler()
        ctx = _make_exec_context(registry=MagicMock(pool=db_pool))

        result = await handler.execute(
            {"action": "update_priority", "priority": "active"}, ctx
        )
        assert not result.success
        assert result.error_type == ToolErrorType.INVALID_PARAMS

    async def test_update_priority_requires_valid_priority(self, db_pool):
        handler = ManageGoalsHandler()
        ctx = _make_exec_context(registry=MagicMock(pool=db_pool))

        result = await handler.execute(
            {"action": "update_priority", "goal_id": "abc", "priority": "mega_urgent"},
            ctx,
        )
        assert not result.success
        assert result.error_type == ToolErrorType.INVALID_PARAMS

    async def test_add_progress_requires_goal_id(self, db_pool):
        handler = ManageGoalsHandler()
        ctx = _make_exec_context(registry=MagicMock(pool=db_pool))

        result = await handler.execute(
            {"action": "add_progress", "note": "some progress"}, ctx
        )
        assert not result.success
        assert result.error_type == ToolErrorType.INVALID_PARAMS

    async def test_add_progress_requires_note(self, db_pool):
        handler = ManageGoalsHandler()
        ctx = _make_exec_context(registry=MagicMock(pool=db_pool))

        result = await handler.execute(
            {"action": "add_progress", "goal_id": "abc"}, ctx
        )
        assert not result.success
        assert result.error_type == ToolErrorType.INVALID_PARAMS

    async def test_no_pool_returns_error(self):
        handler = ManageGoalsHandler()
        registry = MagicMock()
        registry.pool = None
        ctx = _make_exec_context(registry=registry)

        result = await handler.execute({"action": "list"}, ctx)
        assert not result.success
        assert result.error_type == ToolErrorType.MISSING_CONFIG


# ============================================================================
# Integration: DB operations
# ============================================================================


class TestGoalsIntegration:
    async def test_create_goal(self, db_pool):
        handler = ManageGoalsHandler()
        registry = MagicMock()
        registry.pool = db_pool
        ctx = _make_exec_context(pool=db_pool, registry=registry)

        result = await handler.execute(
            {
                "action": "create",
                "title": "Test Goal from pytest",
                "description": "Integration test goal",
                "source": "curiosity",
                "priority": "queued",
            },
            ctx,
        )
        assert result.success
        output = result.output
        assert "goal_id" in output
        assert output["title"] == "Test Goal from pytest"
        assert output["priority"] == "queued"
        assert output["origin"] == "derived"

        # Cleanup: goals are stored in the 'memories' table with type='goal'
        async with db_pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT goal_origin::text FROM memories WHERE id = $1::uuid",
                    output["goal_id"],
                )
                == "derived"
            )
            await conn.execute(
                "DELETE FROM memories WHERE id = $1::uuid", output["goal_id"]
            )

    async def test_model_source_cannot_forge_user_assignment(self, db_pool):
        handler = ManageGoalsHandler()
        registry = MagicMock(pool=db_pool)
        title = f"Autonomous origin {uuid.uuid4()}"
        ctx = _make_exec_context(registry=registry, tool_context=ToolContext.HEARTBEAT)

        result = await handler.execute(
            {"action": "create", "title": title, "source": "user_request"},
            ctx,
        )

        try:
            assert result.success
            assert result.output["origin"] == "derived"
            async with db_pool.acquire() as conn:
                assert (
                    await conn.fetchval(
                        "SELECT goal_origin::text FROM memories WHERE id = $1::uuid",
                        result.output["goal_id"],
                    )
                    == "derived"
                )
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM memories WHERE id = $1::uuid", result.output["goal_id"]
                )

    async def test_user_assignment_upgrades_duplicate_autonomous_goal(self, db_pool):
        handler = ManageGoalsHandler()
        registry = MagicMock(pool=db_pool)
        title = f"Origin upgrade {uuid.uuid4()}"
        heartbeat_ctx = _make_exec_context(
            registry=registry, tool_context=ToolContext.HEARTBEAT
        )
        chat_ctx = _make_exec_context(registry=registry, tool_context=ToolContext.CHAT)

        autonomous = await handler.execute(
            {"action": "create", "title": title}, heartbeat_ctx
        )
        assigned = await handler.execute(
            {"action": "create", "title": title, "source": "curiosity"}, chat_ctx
        )

        try:
            assert autonomous.success and assigned.success
            assert autonomous.output["goal_id"] == assigned.output["goal_id"]
            async with db_pool.acquire() as conn:
                assert (
                    await conn.fetchval(
                        "SELECT goal_origin::text FROM memories WHERE id = $1::uuid",
                        assigned.output["goal_id"],
                    )
                    == "user_request"
                )
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM memories WHERE id = $1::uuid",
                    assigned.output["goal_id"],
                )

    async def test_create_goal_invalid_source_defaults(self, db_pool):
        handler = ManageGoalsHandler()
        registry = MagicMock()
        registry.pool = db_pool
        ctx = _make_exec_context(pool=db_pool, registry=registry)

        result = await handler.execute(
            {
                "action": "create",
                "title": "Goal with bad source",
                "source": "nonexistent_source",
            },
            ctx,
        )
        assert result.success
        # Invalid source defaults to "curiosity"

        # Cleanup
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM memories WHERE id = $1::uuid", result.output["goal_id"]
            )

    async def test_list_goals(self, db_pool):
        handler = ManageGoalsHandler()
        registry = MagicMock()
        registry.pool = db_pool
        ctx = _make_exec_context(pool=db_pool, registry=registry)

        # Create a goal first
        create_result = await handler.execute(
            {"action": "create", "title": "Listable Goal", "priority": "active"},
            ctx,
        )
        assert create_result.success
        goal_id = create_result.output["goal_id"]

        try:
            # List goals
            list_result = await handler.execute({"action": "list"}, ctx)
            assert list_result.success
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM memories WHERE id = $1::uuid", goal_id)

    async def test_list_goals_by_priority(self, db_pool):
        handler = ManageGoalsHandler()
        registry = MagicMock()
        registry.pool = db_pool
        ctx = _make_exec_context(pool=db_pool, registry=registry)

        # Create a queued goal
        create_result = await handler.execute(
            {"action": "create", "title": "Queued Goal", "priority": "queued"},
            ctx,
        )
        assert create_result.success
        goal_id = create_result.output["goal_id"]

        try:
            list_result = await handler.execute(
                {"action": "list", "priority": "queued"}, ctx
            )
            assert list_result.success
            output = list_result.output
            assert "goals" in output
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM memories WHERE id = $1::uuid", goal_id)

    async def test_update_priority(self, db_pool):
        handler = ManageGoalsHandler()
        registry = MagicMock()
        registry.pool = db_pool
        ctx = _make_exec_context(pool=db_pool, registry=registry)

        # Create a goal
        create_result = await handler.execute(
            {"action": "create", "title": "Priority Test Goal", "priority": "queued"},
            ctx,
        )
        assert create_result.success
        goal_id = create_result.output["goal_id"]

        try:
            # Update priority
            result = await handler.execute(
                {
                    "action": "update_priority",
                    "goal_id": goal_id,
                    "priority": "active",
                    "reason": "Promoting to active",
                },
                ctx,
            )
            assert result.success
            assert result.output["new_priority"] == "active"
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM memories WHERE id = $1::uuid", goal_id)

    async def test_add_progress(self, db_pool):
        handler = ManageGoalsHandler()
        registry = MagicMock()
        registry.pool = db_pool
        ctx = _make_exec_context(pool=db_pool, registry=registry)

        # Create a goal
        create_result = await handler.execute(
            {"action": "create", "title": "Progress Test Goal", "priority": "active"},
            ctx,
        )
        assert create_result.success
        goal_id = create_result.output["goal_id"]

        try:
            result = await handler.execute(
                {
                    "action": "add_progress",
                    "goal_id": goal_id,
                    "note": "Made some progress on this goal",
                },
                ctx,
            )
            assert result.success
            assert result.output["goal_id"] == goal_id
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM memories WHERE id = $1::uuid", goal_id)
