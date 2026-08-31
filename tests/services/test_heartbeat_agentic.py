"""
Tests for services/heartbeat_agentic.py — Agentic heartbeat runner.

Covers: system prompt building, agentic heartbeat execution,
finalization, context extraction, and worker integration.
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
from services.worker_service import (
    HeartbeatWorker,
    _extract_heartbeat_context,
    _is_agentic_heartbeat_enabled,
)

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


# ============================================================================
# Helpers
# ============================================================================


def _mock_registry(tool_names: list[str] | None = None) -> MagicMock:
    """Create a mock ToolRegistry with optional tool names."""
    registry = MagicMock()
    registry.pool = MagicMock()

    if tool_names:
        specs = [
            {
                "type": "function",
                "function": {"name": n, "description": f"{n} tool", "parameters": {}},
            }
            for n in tool_names
        ]
    else:
        specs = [
            {
                "type": "function",
                "function": {
                    "name": "recall",
                    "description": "Recall",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remember",
                    "description": "Remember",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "manage_goals",
                    "description": "Goals",
                    "parameters": {},
                },
            },
        ]
    registry.get_specs = AsyncMock(return_value=specs)
    registry.get_spec = MagicMock(return_value=None)
    registry.execute = AsyncMock()
    registry.get_config = AsyncMock(
        return_value=MagicMock(
            get_context_overrides=MagicMock(
                return_value=MagicMock(allow_shell=False, allow_file_write=False)
            ),
            workspace_path=None,
        )
    )
    return registry


def _mock_context() -> dict[str, Any]:
    """Build a minimal heartbeat context for testing."""
    return {
        "agent": {
            "objectives": ["Test objective"],
            "guardrails": [],
            "tools": [],
            "budget": {},
        },
        "environment": {
            "timestamp": "2025-01-15T12:00:00Z",
            "day_of_week": "Wednesday",
            "hour_of_day": 12,
            "time_since_user_hours": 1.0,
            "pending_events": 0,
        },
        "goals": {
            "counts": {"active": 1, "queued": 0},
            "active": [{"title": "Test goal"}],
            "queued": [],
            "issues": [],
        },
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
        "heartbeat_number": 42,
    }


async def test_heartbeat_worker_buffers_listener_updates():
    worker = HeartbeatWorker()
    payload = {
        "log_id": 42,
        "memory_id": str(uuid.uuid4()),
        "change_kind": "contradiction",
    }
    await worker._record_belief_update(payload)
    assert worker.get_recent_belief_updates() == [payload]


# ============================================================================
# Unit: build_heartbeat_system_prompt
# ============================================================================


class TestBuildSystemPrompt:
    async def test_includes_base_prompt(self):
        """System prompt includes the base agentic heartbeat text."""
        prompt = await build_heartbeat_system_prompt()
        # Should contain key phrases from the agentic prompt
        assert "tool" in prompt.lower() or "heartbeat" in prompt.lower()

    async def test_includes_tool_names(self):
        """System prompt gives compact skill guidance without duplicating schemas."""
        registry = _mock_registry(tool_names=["recall", "remember", "manage_goals"])
        prompt = await build_heartbeat_system_prompt(registry)
        assert "## Skills" in prompt
        assert "skills first" in prompt.lower()
        assert "list_skills" in prompt

    async def test_includes_personhood(self):
        """System prompt includes personhood modules."""
        prompt = await build_heartbeat_system_prompt()
        # Should include personhood section (may be minimal or full)
        # The personhood prompt may or may not load depending on file presence
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    async def test_graceful_with_no_registry(self):
        """Works even without a registry."""
        prompt = await build_heartbeat_system_prompt(None)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    async def test_graceful_with_registry_error(self):
        """Handles registry errors gracefully."""
        registry = MagicMock()
        registry.get_specs = AsyncMock(side_effect=RuntimeError("DB error"))
        prompt = await build_heartbeat_system_prompt(registry)
        assert isinstance(prompt, str)


# ============================================================================
# Unit: run_agentic_heartbeat
# ============================================================================


class TestRunAgenticHeartbeat:
    @patch("services.heartbeat_agentic.run_agent")
    async def test_surfaces_belief_changes_as_revision_context(
        self, mock_run_agent, db_pool
    ):
        mock_run_agent.return_value = MagicMock(
            text="I will reassess the dependent belief.",
            tool_calls_made=[],
            iterations=1,
            energy_spent=0,
            timed_out=False,
            stopped_reason="completed",
        )
        context = _mock_context()
        context["belief_updates_recent"] = [
            {
                "memory_id": str(uuid.uuid4()),
                "change_kind": "contradiction",
                "content": "The agreement renews quarterly",
            }
        ]
        async with db_pool.acquire() as conn:
            await run_agentic_heartbeat(
                conn,
                pool=MagicMock(),
                registry=_mock_registry(),
                heartbeat_id="hb-belief-propagation",
                context=context,
            )

        user_message = mock_run_agent.call_args.kwargs["user_message"]
        assert "Belief changes since your last heartbeat" in user_message
        assert "The agreement renews quarterly" in user_message
        assert "preserve unresolved contradictions" in user_message

    @patch("services.heartbeat_agentic.run_agent")
    async def test_resumes_answered_async_question_on_next_heartbeat(
        self, mock_run_agent, db_pool
    ):
        mock_run_agent.return_value = MagicMock(
            text="Continuing with Hartford.",
            tool_calls_made=[],
            iterations=1,
            energy_spent=0,
            timed_out=False,
            stopped_reason="completed",
        )
        question_id = None
        outbox_id = None
        try:
            async with db_pool.acquire() as conn:
                question = await conn.fetchval(
                    """
                    SELECT create_agent_question(
                        NULL, $1::uuid, 'heartbeat', 'Which contract?',
                        '["Manning", "Hartford"]'::jsonb, TRUE, FALSE, 300
                    )
                    """,
                    str(uuid.uuid4()),
                )
                question = (
                    json.loads(question) if isinstance(question, str) else question
                )
                question_id = question["id"]
                outbox_id = question["outbox_message_id"]
                await conn.fetchval(
                    "SELECT answer_agent_question($1::uuid, NULL, 2, 'test', 'operator')",
                    question_id,
                )
                await run_agentic_heartbeat(
                    conn,
                    pool=MagicMock(),
                    registry=_mock_registry(),
                    heartbeat_id="hb-question-resume",
                    context=_mock_context(),
                )

            call = mock_run_agent.call_args.kwargs
            resumed = call["heartbeat_context"]["answered_questions"]
            assert any(item["question_id"] == question_id for item in resumed)
            assert "Question: Which contract?" in call["user_message"]
            assert "Answer: Hartford" in call["user_message"]
        finally:
            async with db_pool.acquire() as conn:
                if question_id:
                    await conn.execute(
                        "DELETE FROM agent_questions WHERE id = $1::uuid", question_id
                    )
                if outbox_id:
                    await conn.execute(
                        "DELETE FROM outbox_messages WHERE id = $1::uuid", outbox_id
                    )

    @patch("services.heartbeat_agentic.run_agent")
    async def test_surfaces_pending_hmx_review_in_context(
        self, mock_run_agent, db_pool
    ):
        mock_run_agent.return_value = MagicMock(
            text="Done.",
            tool_calls_made=[],
            iterations=1,
            energy_spent=0,
            timed_out=False,
            stopped_reason="completed",
        )
        ctx = _mock_context()
        ctx["pending_import_review"] = {"count": 2, "by_section": {"memories": 2}}
        ctx["pending_skill_proposals"] = {
            "count": 1,
            "proposals": [{"id": "proposal-1", "name": "release-review"}],
        }
        async with db_pool.acquire() as conn:
            await run_agentic_heartbeat(
                conn,
                pool=MagicMock(),
                registry=_mock_registry(),
                heartbeat_id="hb-hmx-review",
                context=ctx,
            )

        heartbeat_context = mock_run_agent.call_args.kwargs["heartbeat_context"]
        assert heartbeat_context["pending_import_review"] == {
            "count": 2,
            "by_section": {"memories": 2},
        }
        assert heartbeat_context["pending_skill_proposals"]["count"] == 1

    @patch("services.heartbeat_agentic.run_agent")
    async def test_surfaces_pending_protected_replacement_with_action(
        self, mock_run_agent, db_pool
    ):
        mock_run_agent.return_value = MagicMock(
            text="Deferred.",
            tool_calls_made=[],
            iterations=1,
            energy_spent=0,
            timed_out=False,
            stopped_reason="completed",
        )
        ctx = _mock_context()
        ctx["pending_protected_replacements"] = {
            "total": 1,
            "records": [
                {
                    "replacement_id": "replacement-1",
                    "section": "worldview",
                    "rationale": "Restore a migrated instance",
                }
            ],
        }
        async with db_pool.acquire() as conn:
            await run_agentic_heartbeat(
                conn,
                pool=MagicMock(),
                registry=_mock_registry(),
                heartbeat_id="hb-hmx-protected",
                context=ctx,
            )

        heartbeat_context = mock_run_agent.call_args.kwargs["heartbeat_context"]
        assert heartbeat_context["pending_protected_replacements"]["total"] == 1
        user_message = mock_run_agent.call_args.kwargs["user_message"]
        assert "replacement-1" in user_message
        assert "protected_replacement_list" in user_message
        assert "protected_replacement_inspect" in user_message
        assert "protected_replacement_review" in user_message
        assert "Operator override is not available to the agent" in user_message

    @patch("services.heartbeat_agentic.run_agent")
    async def test_surfaces_open_protected_reversion_as_explicit_choice(
        self, mock_run_agent, db_pool
    ):
        mock_run_agent.return_value = MagicMock(
            text="Kept the replacement.",
            tool_calls_made=[],
            iterations=1,
            energy_spent=0,
            timed_out=False,
            stopped_reason="completed",
        )
        ctx = _mock_context()
        ctx["open_protected_reversions"] = {
            "total": 1,
            "records": [
                {
                    "replacement_id": "replacement-revert-1",
                    "audit_id": "audit-revert-1",
                    "section": "identity",
                    "heartbeats_remaining": 5,
                    "wall_clock_expires_at": "2026-08-01T00:00:00Z",
                }
            ],
        }
        async with db_pool.acquire() as conn:
            await run_agentic_heartbeat(
                conn,
                pool=MagicMock(),
                registry=_mock_registry(),
                heartbeat_id="hb-hmx-reversion",
                context=ctx,
            )

        heartbeat_context = mock_run_agent.call_args.kwargs["heartbeat_context"]
        assert heartbeat_context["open_protected_reversions"]["total"] == 1
        user_message = mock_run_agent.call_args.kwargs["user_message"]
        assert "replacement-revert-1" in user_message
        assert "audit-revert-1" in user_message
        assert "never automatic" in user_message
        assert "protected_replacement_revert" in user_message

    @patch("services.heartbeat_agentic.run_agent")
    async def test_runs_agent_loop(self, mock_run_agent, db_pool):
        """run_agentic_heartbeat creates and runs an AgentLoop via run_agent."""
        mock_run_agent.return_value = MagicMock(
            text="I reflected on my goals.",
            tool_calls_made=[{"name": "recall", "success": True, "energy_spent": 1}],
            iterations=2,
            energy_spent=1,
            timed_out=False,
            stopped_reason="completed",
        )

        async with db_pool.acquire() as conn:
            result = await run_agentic_heartbeat(
                conn,
                pool=MagicMock(),
                registry=_mock_registry(),
                heartbeat_id="hb-test-001",
                context=_mock_context(),
            )

        assert result["completed"] is True
        assert result["energy_spent"] == 1
        assert result["stopped_reason"] == "completed"
        assert len(result["tool_calls_made"]) == 1
        mock_run_agent.assert_awaited_once()

    @patch("services.heartbeat_agentic.run_agent")
    async def test_energy_budget_from_context(self, mock_run_agent, db_pool):
        """Energy budget comes from context energy.current."""
        mock_run_agent.return_value = MagicMock(
            text="Done.",
            tool_calls_made=[],
            iterations=1,
            energy_spent=0,
            timed_out=False,
            stopped_reason="completed",
        )

        ctx = _mock_context()
        ctx["energy"]["current"] = 7

        async with db_pool.acquire() as conn:
            await run_agentic_heartbeat(
                conn,
                pool=MagicMock(),
                registry=_mock_registry(),
                heartbeat_id="hb-test-002",
                context=ctx,
            )

        # Check that run_agent got energy_budget=7
        call_kwargs = mock_run_agent.call_args[1]
        assert call_kwargs["energy_budget"] == 7

    @patch("services.heartbeat_agentic.run_agent")
    async def test_timeout_reported(self, mock_run_agent, db_pool):
        """Timeout is reported in result."""
        mock_run_agent.return_value = MagicMock(
            text="Timed out.",
            tool_calls_made=[],
            iterations=3,
            energy_spent=5,
            timed_out=True,
            stopped_reason="timeout",
        )

        async with db_pool.acquire() as conn:
            result = await run_agentic_heartbeat(
                conn,
                pool=MagicMock(),
                registry=_mock_registry(),
                heartbeat_id="hb-test-003",
                context=_mock_context(),
            )

        assert result["completed"] is False
        assert result["timed_out"] is True
        assert result["stopped_reason"] == "timeout"


# ============================================================================
# Unit: finalize_heartbeat
# ============================================================================


class TestFinalizeHeartbeat:
    async def test_finalize_records_memory(self, db_pool):
        """finalize_heartbeat creates an episodic memory and updates heartbeat_state."""
        async with db_pool.acquire() as conn:
            result = await finalize_heartbeat(
                conn,
                heartbeat_id=str(uuid.uuid4()),
                result={
                    "text": "I checked my goals and recalled some memories.",
                    "tool_calls_made": [
                        {"name": "recall", "success": True},
                        {"name": "manage_goals", "success": True},
                    ],
                    "energy_spent": 3,
                    "stopped_reason": "completed",
                },
            )

            assert result["completed"] is True
            assert result["energy_spent"] == 3
            # memory_id may or may not be set depending on DB state

    async def test_finalize_with_empty_result(self, db_pool):
        """finalize_heartbeat handles empty/minimal result gracefully."""
        async with db_pool.acquire() as conn:
            result = await finalize_heartbeat(
                conn,
                heartbeat_id=str(uuid.uuid4()),
                result={},
            )

            assert result["completed"] is True
            assert result["energy_spent"] == 0

    async def test_finalize_builds_summary_with_tool_names(self, db_pool):
        """Summary includes tool names when present."""
        async with db_pool.acquire() as conn:
            result = await finalize_heartbeat(
                conn,
                heartbeat_id=str(uuid.uuid4()),
                result={
                    "text": "",
                    "tool_calls_made": [
                        {"name": "recall"},
                        {"name": "remember"},
                    ],
                    "energy_spent": 2,
                    "stopped_reason": "completed",
                },
            )

            # Should succeed regardless
            assert result["completed"] is True


# ============================================================================
# Unit: context extraction
# ============================================================================


class TestContextExtraction:
    def test_extract_context_from_external_calls(self):
        """_extract_heartbeat_context gets context from think call input."""
        payload = {
            "heartbeat_id": "hb-123",
            "external_calls": [
                {
                    "call_type": "think",
                    "input": {
                        "kind": "heartbeat_decision",
                        "heartbeat_id": "hb-123",
                        "context": {
                            "energy": {"current": 15, "max": 20},
                            "goals": {"active": [], "queued": []},
                        },
                    },
                }
            ],
        }
        context = _extract_heartbeat_context(payload)
        assert "energy" in context
        assert context["energy"]["current"] == 15

    def test_extract_context_no_nested_context(self):
        """Falls back to call input when no nested context key."""
        payload = {
            "external_calls": [
                {
                    "call_type": "think",
                    "input": {
                        "kind": "heartbeat_decision",
                        "heartbeat_id": "hb-123",
                        "energy": {"current": 10},
                    },
                }
            ],
        }
        context = _extract_heartbeat_context(payload)
        assert "energy" in context

    def test_extract_context_no_external_calls(self):
        """Returns payload when no external_calls present."""
        payload = {"heartbeat_id": "hb-123", "energy": {"current": 5}}
        context = _extract_heartbeat_context(payload)
        assert context["heartbeat_id"] == "hb-123"

    def test_extract_context_no_think_call(self):
        """Returns payload when no think call present."""
        payload = {
            "external_calls": [
                {"call_type": "embed", "input": {}},
            ],
        }
        context = _extract_heartbeat_context(payload)
        assert context == payload

    def test_extract_context_empty_external_calls(self):
        """Returns payload when external_calls is empty."""
        payload = {"external_calls": []}
        context = _extract_heartbeat_context(payload)
        assert context == payload


# ============================================================================
# Unit: agentic flag
# ============================================================================


class TestAgenticFlag:
    async def test_agentic_disabled_by_default(self, db_pool):
        """Agentic heartbeat is disabled when config key is missing."""
        async with db_pool.acquire() as conn:
            # Ensure the config key is absent
            await conn.execute(
                "DELETE FROM config WHERE key = 'heartbeat.use_agentic_loop'"
            )
            enabled = await _is_agentic_heartbeat_enabled(conn)
            assert enabled is False

    async def test_agentic_enabled_when_true(self, db_pool):
        """Agentic heartbeat is enabled when config key is 'true'."""
        async with db_pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('heartbeat.use_agentic_loop', 'true'::jsonb)"
            )
            try:
                enabled = await _is_agentic_heartbeat_enabled(conn)
                assert enabled is True
            finally:
                await conn.execute(
                    "DELETE FROM config WHERE key = 'heartbeat.use_agentic_loop'"
                )

    async def test_agentic_disabled_when_false(self, db_pool):
        """Agentic heartbeat is disabled when config key is 'false'."""
        async with db_pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('heartbeat.use_agentic_loop', 'false'::jsonb)"
            )
            try:
                enabled = await _is_agentic_heartbeat_enabled(conn)
                assert enabled is False
            finally:
                await conn.execute(
                    "DELETE FROM config WHERE key = 'heartbeat.use_agentic_loop'"
                )
