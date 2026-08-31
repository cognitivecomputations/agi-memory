"""
Tests for heartbeat agentic features in services/heartbeat_agentic.py.

Covers: backlog task detection, checkpoint context extraction, energy boost,
system prompt augmentation, always-on planning/continuation, permission
gating, and finalization auto-checkpoint.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.heartbeat_agentic import (
    build_heartbeat_system_prompt,
    finalize_heartbeat,
    run_agentic_heartbeat,
)

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


# ============================================================================
# Helpers
# ============================================================================


def _mock_registry(tool_names: list[str] | None = None) -> MagicMock:
    registry = MagicMock()
    registry.pool = MagicMock()
    specs = [
        {"type": "function", "function": {"name": n, "description": f"{n} tool", "parameters": {}}}
        for n in (tool_names or ["recall", "remember", "manage_goals", "manage_backlog"])
    ]
    registry.get_specs = AsyncMock(return_value=specs)
    registry.get_spec = MagicMock(return_value=None)
    registry.execute = AsyncMock()
    registry.get_config = AsyncMock(return_value=MagicMock(
        get_context_overrides=MagicMock(return_value=MagicMock(
            allow_shell=False, allow_file_write=False
        )),
        workspace_path=None,
    ))
    return registry


def _mock_conn(prompt: str = "[heartbeat decision prompt]") -> AsyncMock:
    """A connection mock: heartbeat_agentic_plan gets a contract-faithful
    plan computed from the context (real semantics pinned by the SQL tests);
    everything else returns the rendered prompt string."""
    conn = AsyncMock()

    async def fetchval(query, *args):
        if "attach_answered_agent_questions" in query:
            ctx = json.loads(args[0]) if args and isinstance(args[0], str) else {}
            return {**ctx, "answered_questions": []}
        if "render_answered_agent_questions" in query:
            return ""
        if "heartbeat_agentic_plan" in query:
            ctx = json.loads(args[0]) if args and isinstance(args[0], str) else {}
            backlog = ctx.get("backlog") or {}
            counts = backlog.get("counts") or {}
            actionable = backlog.get("actionable") or []
            has_tasks = bool(actionable) or (
                (counts.get("todo") or 0) + (counts.get("in_progress") or 0) > 0
            )
            energy = (ctx.get("energy") or {}).get("current", 20)
            parts = []
            if has_tasks:
                cp = [
                    f"### Resuming: {i.get('title', 'Untitled')}\n"
                    f"- Last step: {i.get('checkpoint', {}).get('step', 'unknown')}\n"
                    f"- Progress: {i.get('checkpoint', {}).get('progress', '')}\n"
                    f"- Next action: {i.get('checkpoint', {}).get('next_action', '')}"
                    for i in actionable
                    if i.get("status") == "in_progress" and i.get("checkpoint")
                ]
                if cp:
                    parts.append("## Checkpoint Resume\n\n" + "\n\n".join(cp))
            return {
                "context": ctx,
                "has_backlog_tasks": has_tasks,
                "energy_budget": energy * 2 if has_tasks else energy,
                "timeout_seconds": 300.0 if has_tasks else 120.0,
                "max_tokens": 4096 if has_tasks else 2048,
                "allow_shell": has_tasks,
                "allow_file_write": has_tasks,
                "prompt_suffix": "\n\n".join(parts) or None,
            }
        if "enforce_heartbeat_plan_energy" in query:
            plan = json.loads(args[0]) if args and isinstance(args[0], str) else {}
            energy = (plan.get("context", {}).get("energy") or {})
            available = float(energy.get("current", 0))
            reserve = float(energy.get("max", 20))
            limit = reserve * (2 if plan.get("has_backlog_tasks") else 1)
            return {
                **plan,
                "energy_budget": min(available, limit),
                "energy_available": available,
                "energy_reserve": reserve,
                "energy_bank_capacity": reserve * 3,
            }
        return prompt

    conn.fetchval = AsyncMock(side_effect=fetchval)
    return conn


def _base_context(**overrides: Any) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "agent": {"objectives": ["Test"], "guardrails": [], "tools": [], "budget": {}},
        "environment": {
            "timestamp": "2025-01-15T12:00:00Z",
            "day_of_week": "Wednesday",
            "hour_of_day": 12,
            "time_since_user_hours": 1.0,
            "pending_events": 0,
        },
        "goals": {"counts": {"active": 0, "queued": 0}, "active": [], "queued": [], "issues": []},
        "recent_memories": [],
        "identity": [],
        "worldview": [],
        "self_model": [],
        "narrative": {},
        "urgent_drives": [],
        "emotional_state": {},
        "relationships": [],
        "contradictions": [],
        "emotional_patterns": [],
        "active_transformations": [],
        "transformations_ready": [],
        "energy": {"current": 15, "max": 20},
        "allowed_actions": [],
        "action_costs": {},
        "backlog": {},
        "heartbeat_number": 42,
    }
    ctx.update(overrides)
    return ctx


# ============================================================================
# Unit: heartbeat_agentic_plan (SQL owns the gate, scaling, permissions)
# ============================================================================


class TestHeartbeatPlanSQL:
    async def _plan(self, db_pool, ctx):
        async with db_pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT heartbeat_agentic_plan($1::jsonb)", json.dumps(ctx)
            )
        return json.loads(raw) if isinstance(raw, str) else raw

    async def test_no_backlog_baseline_resources(self, db_pool):
        plan = await self._plan(db_pool, _base_context(backlog={}))
        assert plan["has_backlog_tasks"] is False
        assert plan["energy_budget"] == 15
        assert plan["timeout_seconds"] == 120
        assert plan["max_tokens"] == 2048
        assert plan["allow_shell"] is False
        assert plan["allow_file_write"] is False

    async def test_backlog_scales_and_grants(self, db_pool):
        ctx = _base_context(backlog={
            "counts": {"todo": 1},
            "actionable": [{"title": "Task", "status": "todo"}],
        })
        plan = await self._plan(db_pool, ctx)
        assert plan["has_backlog_tasks"] is True
        assert plan["energy_budget"] == 30  # 15 * 2
        assert plan["timeout_seconds"] == 300
        assert plan["max_tokens"] == 4096
        assert plan["allow_shell"] is True
        assert plan["allow_file_write"] is True

    async def test_done_only_backlog_stays_baseline(self, db_pool):
        ctx = _base_context(backlog={"counts": {"todo": 0, "in_progress": 0, "done": 4}})
        plan = await self._plan(db_pool, ctx)
        assert plan["has_backlog_tasks"] is False

    async def test_checkpoint_fragment_in_suffix(self, db_pool):
        ctx = _base_context(backlog={
            "counts": {"in_progress": 1},
            "actionable": [{
                "title": "Deploy",
                "status": "in_progress",
                "checkpoint": {"step": "2", "progress": "built", "next_action": "Push"},
            }],
        })
        plan = await self._plan(db_pool, ctx)
        assert "Checkpoint Resume" in plan["prompt_suffix"]
        assert "Deploy" in plan["prompt_suffix"]
        assert "Push" in plan["prompt_suffix"]

    async def test_empty_checkpoint_dict_ignored(self, db_pool):
        ctx = _base_context(backlog={
            "actionable": [{"title": "Task", "status": "in_progress", "checkpoint": {}}],
        })
        plan = await self._plan(db_pool, ctx)
        assert not (plan.get("prompt_suffix") or "")

    async def test_context_enriched_with_pending_summaries(self, db_pool):
        plan = await self._plan(db_pool, _base_context())
        enriched = plan["context"]
        assert "pending_import_review" in enriched
        assert "pending_skill_proposals" in enriched
        assert "pending_protected_replacements" in enriched
        assert "open_protected_reversions" in enriched


# ============================================================================
# Unit: build_heartbeat_system_prompt with has_backlog_tasks
# ============================================================================


class TestBuildSystemPromptBacklog:
    async def test_no_backlog_no_task_prompt(self):
        prompt = await build_heartbeat_system_prompt(None, has_backlog_tasks=False)
        assert "Task Mode" not in prompt

    async def test_backlog_includes_task_prompt(self):
        prompt = await build_heartbeat_system_prompt(None, has_backlog_tasks=True)
        assert "Task Mode" in prompt
        assert "PICK" in prompt
        assert "CHECKPOINT" in prompt

    async def test_backlog_with_registry(self):
        registry = _mock_registry()
        prompt = await build_heartbeat_system_prompt(registry, has_backlog_tasks=True)
        assert "Task Mode" in prompt
        assert "manage_backlog" in prompt

    async def test_default_is_no_backlog(self):
        prompt = await build_heartbeat_system_prompt()
        assert "Task Mode" not in prompt


# ============================================================================
# Unit: run_agentic_heartbeat resource scaling
# ============================================================================


class TestRunAgenticHeartbeatScaling:
    @patch("services.heartbeat_agentic.run_agent")
    async def test_backlog_can_draw_up_to_double_reserve(self, mock_run_agent):
        mock_run_agent.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )

        ctx = _base_context(backlog={
            "counts": {"todo": 1},
            "actionable": [{"title": "Task", "status": "todo"}],
        })
        ctx["energy"]["current"] = 50

        result = await run_agentic_heartbeat(
            _mock_conn(), pool=MagicMock(), registry=_mock_registry(),
            heartbeat_id="hb-tm-001", context=ctx,
        )

        call_kwargs = mock_run_agent.call_args[1]
        assert call_kwargs["energy_budget"] == 40  # two 20-point reserves
        assert result["has_backlog_tasks"] is True

    @patch("services.heartbeat_agentic.run_agent")
    async def test_no_backlog_normal_energy(self, mock_run_agent):
        mock_run_agent.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )

        ctx = _base_context(backlog={"counts": {}, "actionable": []})
        ctx["energy"]["current"] = 10

        result = await run_agentic_heartbeat(
            _mock_conn(), pool=MagicMock(), registry=_mock_registry(),
            heartbeat_id="hb-tm-002", context=ctx,
        )

        call_kwargs = mock_run_agent.call_args[1]
        assert call_kwargs["energy_budget"] == 10  # unchanged
        assert result["has_backlog_tasks"] is False

    @patch("services.heartbeat_agentic.run_agent")
    async def test_backlog_extends_timeout(self, mock_run_agent):
        mock_run_agent.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )

        ctx = _base_context(backlog={
            "counts": {"todo": 1},
            "actionable": [{"title": "Task", "status": "todo"}],
        })

        await run_agentic_heartbeat(
            _mock_conn(), pool=MagicMock(), registry=_mock_registry(),
            heartbeat_id="hb-tm-003", context=ctx,
        )

        call_kwargs = mock_run_agent.call_args[1]
        assert call_kwargs["timeout_seconds"] == 300.0

    @patch("services.heartbeat_agentic.run_agent")
    async def test_backlog_increases_max_tokens(self, mock_run_agent):
        mock_run_agent.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )

        ctx = _base_context(backlog={
            "counts": {"todo": 1},
            "actionable": [{"title": "Task", "status": "todo"}],
        })

        await run_agentic_heartbeat(
            _mock_conn(), pool=MagicMock(), registry=_mock_registry(),
            heartbeat_id="hb-tm-004", context=ctx,
        )

        call_kwargs = mock_run_agent.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    @patch("services.heartbeat_agentic.run_agent")
    async def test_checkpoint_context_appended_to_user_message(self, mock_run_agent):
        mock_run_agent.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )

        ctx = _base_context(backlog={
            "counts": {"in_progress": 1},
            "actionable": [{
                "title": "Deploy",
                "status": "in_progress",
                "checkpoint": {"step": "2", "progress": "Built", "next_action": "Push"},
            }],
        })

        await run_agentic_heartbeat(
            _mock_conn(), pool=MagicMock(), registry=_mock_registry(),
            heartbeat_id="hb-tm-005", context=ctx,
        )

        # The user_message passed to run_agent should include checkpoint context
        call_kwargs = mock_run_agent.call_args[1]
        user_msg = call_kwargs["user_message"]
        assert "Checkpoint Resume" in user_msg
        assert "Deploy" in user_msg
        assert "Push" in user_msg


# ============================================================================
# Integration: finalize_heartbeat auto-checkpoint
# ============================================================================


class TestFinalizeAutoCheckpoint:
    async def test_auto_checkpoint_on_timeout(self, db_pool):
        """In-progress items without checkpoints get auto-checkpointed on timeout."""
        async with db_pool.acquire() as conn:
            # Create an in-progress backlog item
            item_id = await conn.fetchval(
                """
                INSERT INTO public.backlog (title, status, priority)
                VALUES ('Auto-checkpoint test', 'in_progress', 'high')
                RETURNING id
                """
            )

            try:
                result = await finalize_heartbeat(
                    conn,
                    heartbeat_id=str(uuid.uuid4()),
                    result={
                        "text": "Ran out of time.",
                        "tool_calls_made": [{"name": "shell"}],
                        "energy_spent": 20,
                        "stopped_reason": "timeout",
                        "has_backlog_tasks": True,
                    },
                )

                assert result["completed"] is True
                assert result["has_backlog_tasks"] is True

                # Verify checkpoint was set
                row = await conn.fetchrow(
                    "SELECT checkpoint FROM public.backlog WHERE id = $1", item_id
                )
                checkpoint = json.loads(row["checkpoint"])
                assert checkpoint["step"] == "interrupted"
                assert "timeout" in checkpoint["progress"]
            finally:
                await conn.execute("DELETE FROM public.backlog WHERE id = $1", item_id)

    async def test_no_auto_checkpoint_when_already_checkpointed(self, db_pool):
        """Items with existing checkpoints are not overwritten."""
        async with db_pool.acquire() as conn:
            original_cp = json.dumps({"step": "step 3", "progress": "good", "next_action": "verify"})
            item_id = await conn.fetchval(
                """
                INSERT INTO public.backlog (title, status, priority, checkpoint)
                VALUES ('Already checkpointed', 'in_progress', 'high', $1::jsonb)
                RETURNING id
                """,
                original_cp,
            )

            try:
                await finalize_heartbeat(
                    conn,
                    heartbeat_id=str(uuid.uuid4()),
                    result={
                        "text": "Timed out.",
                        "tool_calls_made": [],
                        "energy_spent": 10,
                        "stopped_reason": "timeout",
                        "has_backlog_tasks": True,
                    },
                )

                row = await conn.fetchrow(
                    "SELECT checkpoint FROM public.backlog WHERE id = $1", item_id
                )
                checkpoint = json.loads(row["checkpoint"])
                assert checkpoint["step"] == "step 3"  # unchanged
            finally:
                await conn.execute("DELETE FROM public.backlog WHERE id = $1", item_id)

    async def test_no_auto_checkpoint_on_normal_completion(self, db_pool):
        """No auto-checkpoint when heartbeat completes normally."""
        async with db_pool.acquire() as conn:
            item_id = await conn.fetchval(
                """
                INSERT INTO public.backlog (title, status, priority)
                VALUES ('Normal completion', 'in_progress', 'normal')
                RETURNING id
                """
            )

            try:
                await finalize_heartbeat(
                    conn,
                    heartbeat_id=str(uuid.uuid4()),
                    result={
                        "text": "All done.",
                        "tool_calls_made": [],
                        "energy_spent": 5,
                        "stopped_reason": "completed",
                        "has_backlog_tasks": True,
                    },
                )

                row = await conn.fetchrow(
                    "SELECT checkpoint FROM public.backlog WHERE id = $1", item_id
                )
                assert row["checkpoint"] is None  # not auto-checkpointed
            finally:
                await conn.execute("DELETE FROM public.backlog WHERE id = $1", item_id)

    async def test_no_auto_checkpoint_without_backlog_tasks(self, db_pool):
        """No auto-checkpoint when has_backlog_tasks is False."""
        async with db_pool.acquire() as conn:
            item_id = await conn.fetchval(
                """
                INSERT INTO public.backlog (title, status, priority)
                VALUES ('No task mode', 'in_progress', 'normal')
                RETURNING id
                """
            )

            try:
                await finalize_heartbeat(
                    conn,
                    heartbeat_id=str(uuid.uuid4()),
                    result={
                        "text": "Timed out.",
                        "tool_calls_made": [],
                        "energy_spent": 10,
                        "stopped_reason": "timeout",
                        "has_backlog_tasks": False,
                    },
                )

                row = await conn.fetchrow(
                    "SELECT checkpoint FROM public.backlog WHERE id = $1", item_id
                )
                assert row["checkpoint"] is None
            finally:
                await conn.execute("DELETE FROM public.backlog WHERE id = $1", item_id)

    async def test_auto_checkpoint_on_energy_exhausted(self, db_pool):
        """Auto-checkpoint also triggers on energy_exhausted."""
        async with db_pool.acquire() as conn:
            item_id = await conn.fetchval(
                """
                INSERT INTO public.backlog (title, status, priority)
                VALUES ('Energy exhausted test', 'in_progress', 'urgent')
                RETURNING id
                """
            )

            try:
                await finalize_heartbeat(
                    conn,
                    heartbeat_id=str(uuid.uuid4()),
                    result={
                        "text": "Out of energy.",
                        "tool_calls_made": [{"name": "shell"}, {"name": "recall"}],
                        "energy_spent": 40,
                        "stopped_reason": "energy_exhausted",
                        "has_backlog_tasks": True,
                    },
                )

                row = await conn.fetchrow(
                    "SELECT checkpoint FROM public.backlog WHERE id = $1", item_id
                )
                checkpoint = json.loads(row["checkpoint"])
                assert checkpoint["step"] == "interrupted"
                assert "energy_exhausted" in checkpoint["progress"]
            finally:
                await conn.execute("DELETE FROM public.backlog WHERE id = $1", item_id)


# ============================================================================
# Unit: prompt_resources task mode loader
# ============================================================================


class TestTaskModePromptLoader:
    def test_load_task_mode_prompt(self):
        from services.prompt_resources import load_heartbeat_task_mode_prompt
        prompt = load_heartbeat_task_mode_prompt()
        assert "Task Mode" in prompt
        assert "PICK" in prompt
        assert "EXECUTE" in prompt
        assert "VERIFY" in prompt
        assert "CHECKPOINT" in prompt


# ============================================================================
# Integration: always-on planning/continuation, gated permissions
# ============================================================================


class TestHeartbeatAgentLoopWiring:
    """Verify that planning and continuation are always on, permissions are gated.

    Planning and continuation are now configured inside services.agent.run_agent(),
    so we mock at that level. Permission gating is tested by inspecting kwargs
    passed from run_agentic_heartbeat → run_agent.
    """

    @patch("services.agent.run_subconscious_appraisal", new_callable=AsyncMock)
    @patch("services.agent.AgentLoop")
    @patch("services.agent.load_llm_config")
    async def test_planning_always_enabled(self, mock_load_config, mock_agent_class, mock_sub):
        """Planning is on even without backlog tasks."""
        from services.agent import SubconsciousOutput
        mock_sub.return_value = SubconsciousOutput()
        mock_load_config.return_value = {
            "provider": "openai", "model": "gpt-4o", "endpoint": None, "api_key": "t",
        }
        mock_agent = AsyncMock()
        mock_agent.run.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )
        mock_agent_class.return_value = mock_agent

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        ctx = _base_context(backlog={"counts": {}, "actionable": []})

        await run_agentic_heartbeat(
            _mock_conn(), pool=mock_pool, registry=_mock_registry(),
            heartbeat_id="hb-always-plan", context=ctx,
        )

        config_arg = mock_agent_class.call_args[0][0]
        assert config_arg.enable_planning is True

    @patch("services.agent.run_subconscious_appraisal", new_callable=AsyncMock)
    @patch("services.agent.AgentLoop")
    @patch("services.agent.load_llm_config")
    async def test_continuation_always_enabled(self, mock_load_config, mock_agent_class, mock_sub):
        """Continuation nudge is on even without backlog tasks."""
        from services.agent import SubconsciousOutput
        mock_sub.return_value = SubconsciousOutput()
        mock_load_config.return_value = {
            "provider": "openai", "model": "gpt-4o", "endpoint": None, "api_key": "t",
        }
        mock_agent = AsyncMock()
        mock_agent.run.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )
        mock_agent_class.return_value = mock_agent

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        ctx = _base_context(backlog={"counts": {}, "actionable": []})

        await run_agentic_heartbeat(
            _mock_conn(), pool=mock_pool, registry=_mock_registry(),
            heartbeat_id="hb-always-cont", context=ctx,
        )

        config_arg = mock_agent_class.call_args[0][0]
        assert config_arg.continuation_prompt is not None
        assert config_arg.max_continuations >= 1

    @patch("services.heartbeat_agentic.run_agent")
    async def test_backlog_grants_permissions(self, mock_run_agent):
        """Backlog tasks grant shell + file_write permissions."""
        mock_run_agent.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )

        ctx = _base_context(backlog={
            "counts": {"todo": 1},
            "actionable": [{"title": "Write script", "status": "todo"}],
        })

        await run_agentic_heartbeat(
            _mock_conn(), pool=MagicMock(), registry=_mock_registry(),
            heartbeat_id="hb-perms", context=ctx,
        )

        call_kwargs = mock_run_agent.call_args[1]
        overrides = call_kwargs["context_overrides"]
        assert overrides is not None
        assert overrides.allow_shell is True
        assert overrides.allow_file_write is True

    @patch("services.heartbeat_agentic.run_agent")
    async def test_no_backlog_no_permissions(self, mock_run_agent):
        """Without backlog tasks, no elevated permissions are granted."""
        mock_run_agent.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )

        ctx = _base_context(backlog={"counts": {}, "actionable": []})

        await run_agentic_heartbeat(
            _mock_conn(), pool=MagicMock(), registry=_mock_registry(),
            heartbeat_id="hb-no-perms", context=ctx,
        )

        call_kwargs = mock_run_agent.call_args[1]
        assert call_kwargs.get("context_overrides") is None

    @patch("services.agent.run_subconscious_appraisal", new_callable=AsyncMock)
    @patch("services.agent.AgentLoop")
    @patch("services.agent.load_llm_config")
    async def test_backlog_gets_more_continuations(self, mock_load_config, mock_agent_class, mock_sub):
        """Backlog tasks get 2 continuations, empty backlog gets 1."""
        from services.agent import SubconsciousOutput
        mock_sub.return_value = SubconsciousOutput()
        mock_load_config.return_value = {
            "provider": "openai", "model": "gpt-4o", "endpoint": None, "api_key": "t",
        }
        mock_agent = AsyncMock()
        mock_agent.run.return_value = MagicMock(
            text="Done.", tool_calls_made=[], iterations=1,
            energy_spent=0, timed_out=False, stopped_reason="completed",
        )
        mock_agent_class.return_value = mock_agent

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        # With backlog
        ctx_tasks = _base_context(backlog={
            "counts": {"todo": 1},
            "actionable": [{"title": "Task", "status": "todo"}],
        })
        await run_agentic_heartbeat(
            _mock_conn(), pool=mock_pool, registry=_mock_registry(),
            heartbeat_id="hb-cont-tasks", context=ctx_tasks,
        )
        config_tasks = mock_agent_class.call_args[0][0]

        # Without backlog
        ctx_empty = _base_context(backlog={"counts": {}, "actionable": []})
        await run_agentic_heartbeat(
            _mock_conn(), pool=mock_pool, registry=_mock_registry(),
            heartbeat_id="hb-cont-empty", context=ctx_empty,
        )
        config_empty = mock_agent_class.call_args[0][0]

        assert config_tasks.max_continuations == 2
        assert config_empty.max_continuations == 1
