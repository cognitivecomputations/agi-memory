"""HMX onboarding interviews (#97): the persistence + binding-execution layer
in db/49a_hmx_onboarding_interview.sql. A use case fully defined in one HMX
file can carry a set of questions that bind answers to actions via existing
machinery -- these tests cover the state machine (start/answer/skip/resume/
completion) and the 'remember' binding executor end to end, plus an
unimplemented binding failing loud rather than silently doing nothing.
"""
from __future__ import annotations

import json

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


def _interview(marker: str, **overrides) -> dict:
    spec = {
        "version": 1,
        "questions": [
            {
                "id": "name",
                "type": "choice",
                "prompt": "What should I call you?",
                "options": [
                    {"label": "Boss", "value": "boss"},
                    {"label": "Friend", "value": "friend"},
                ],
                "binds": {
                    "action": "remember",
                    "params_template": {
                        "content": f"The user {marker} wants to be called {{{{answer}}}}",
                        "type": "semantic",
                    },
                },
                "required": True,
            },
            {
                "id": "extra",
                "type": "freeform",
                "prompt": "Anything else?",
                "binds": {
                    "action": "remember",
                    "params_template": {"content": f"Extra note {marker}: {{{{answer}}}}"},
                },
                "skippable": True,
            },
        ],
    }
    spec.update(overrides)
    return spec


async def test_start_hmx_interview_is_idempotent(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = get_test_identifier("hmxinterview")
            export_id = f"exp_{marker[:16]:0<16}"
            spec = _interview(marker)

            first = _j(
                await conn.fetchval(
                    "SELECT start_hmx_interview($1, $2::jsonb)", export_id, json.dumps(spec)
                )
            )
            assert first["status"] == "pending"
            assert first["export_id"] == export_id
            assert len(first["questions"]) == 2

            second = _j(
                await conn.fetchval(
                    "SELECT start_hmx_interview($1, $2::jsonb)", export_id, json.dumps(spec)
                )
            )
            assert second["id"] == first["id"]
        finally:
            await tr.rollback()


async def test_required_question_completes_interview_and_applies_remember_binding(
    db_pool,
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = get_test_identifier("hmxinterview")
            export_id = f"exp_{marker[:16]:0<16}"
            spec = _interview(marker)
            await conn.fetchval(
                "SELECT start_hmx_interview($1, $2::jsonb)", export_id, json.dumps(spec)
            )

            result = _j(
                await conn.fetchval(
                    "SELECT record_hmx_interview_answer($1, 'name', 'answered', $2::jsonb)",
                    export_id,
                    json.dumps("friend"),
                )
            )
            assert result["success"] is True
            assert result["binding_result"]["success"] is True
            # 'name' is the only required question, so the interview is
            # already complete even with 'extra' unanswered.
            assert result["interview_status"] == "completed"

            mem = await conn.fetchrow(
                "SELECT content, source_attribution FROM memories WHERE content = $1",
                f"The user {marker} wants to be called friend",
            )
            assert mem is not None
            source = _j(mem["source_attribution"])
            assert source["kind"] == "hmx_interview"
            assert source["ref"] == export_id
            assert source["label"] == "name"

            state = _j(await conn.fetchval("SELECT get_hmx_interview_state($1)", export_id))
            assert state["status"] == "completed"
            assert len(state["answers"]) == 1
        finally:
            await tr.rollback()


async def test_skipping_a_question_records_no_binding(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = get_test_identifier("hmxinterview")
            export_id = f"exp_{marker[:16]:0<16}"
            spec = _interview(marker)
            await conn.fetchval(
                "SELECT start_hmx_interview($1, $2::jsonb)", export_id, json.dumps(spec)
            )

            result = _j(
                await conn.fetchval(
                    "SELECT record_hmx_interview_answer($1, 'extra', 'skipped')", export_id
                )
            )
            assert result["success"] is True
            assert result["binding_result"] is None

            count = await conn.fetchval(
                "SELECT count(*) FROM memories WHERE content LIKE $1",
                f"Extra note {marker}:%",
            )
            assert count == 0
        finally:
            await tr.rollback()


async def test_unimplemented_binding_action_fails_loud(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = get_test_identifier("hmxinterview")
            export_id = f"exp_{marker[:16]:0<16}"
            spec = _interview(
                marker,
                questions=[
                    {
                        "id": "goal_q",
                        "type": "freeform",
                        "prompt": "What's a goal I should track?",
                        "binds": {"action": "create_goal", "params_template": {"title": "{{answer}}"}},
                    }
                ],
            )
            await conn.fetchval(
                "SELECT start_hmx_interview($1, $2::jsonb)", export_id, json.dumps(spec)
            )

            result = _j(
                await conn.fetchval(
                    "SELECT record_hmx_interview_answer($1, 'goal_q', 'answered', $2::jsonb)",
                    export_id,
                    json.dumps("ship the release"),
                )
            )
            assert result["success"] is True  # recording the answer itself succeeded
            assert result["binding_result"]["success"] is False
            assert result["binding_result"]["error_type"] == "not_implemented"
            assert "create_goal" in result["binding_result"]["error"]
        finally:
            await tr.rollback()


async def test_unanswered_required_question_leaves_interview_pending(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = get_test_identifier("hmxinterview")
            export_id = f"exp_{marker[:16]:0<16}"
            spec = _interview(marker)
            await conn.fetchval(
                "SELECT start_hmx_interview($1, $2::jsonb)", export_id, json.dumps(spec)
            )

            # Answer only the non-required question.
            result = _j(
                await conn.fetchval(
                    "SELECT record_hmx_interview_answer($1, 'extra', 'answered', $2::jsonb)",
                    export_id,
                    json.dumps("nothing else"),
                )
            )
            assert result["interview_status"] == "pending"

            state = _j(await conn.fetchval("SELECT get_hmx_interview_state($1)", export_id))
            assert state["status"] == "pending"
        finally:
            await tr.rollback()


async def test_answering_unknown_question_id_is_a_clear_error(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = get_test_identifier("hmxinterview")
            export_id = f"exp_{marker[:16]:0<16}"
            spec = _interview(marker)
            await conn.fetchval(
                "SELECT start_hmx_interview($1, $2::jsonb)", export_id, json.dumps(spec)
            )

            result = _j(
                await conn.fetchval(
                    "SELECT record_hmx_interview_answer($1, 'nonexistent', 'answered', $2::jsonb)",
                    export_id,
                    json.dumps("x"),
                )
            )
            assert result["success"] is False
            assert "unknown question id" in result["error"]
        finally:
            await tr.rollback()


async def test_answering_with_no_interview_started_is_a_clear_error(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            result = _j(
                await conn.fetchval(
                    "SELECT record_hmx_interview_answer('exp_0000000000000000', 'name', 'answered', $1::jsonb)",
                    json.dumps("x"),
                )
            )
            assert result["success"] is False
            assert "no interview in progress" in result["error"]
        finally:
            await tr.rollback()
