"""core/hmx_interview.py: the CLI-facing orchestration for HMX onboarding
interviews (#97). The DB does persistence + binding execution
(tests/db/test_hmx_onboarding_interview.py); this covers the question-loop
logic -- skip handling, the required-question re-ask loop, and the
non-interactive/no-section/already-completed short-circuits -- against a
mocked connection and mocked prompts.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from core.hmx_interview import maybe_run_cli_interview

pytestmark = pytest.mark.asyncio


def _document(**interview_overrides):
    interview = {
        "version": 1,
        "questions": [
            {
                "id": "name",
                "type": "choice",
                "prompt": "What should I call you?",
                "options": [{"label": "Boss", "value": "boss"}, {"label": "Friend", "value": "friend"}],
                "binds": {"action": "remember", "params_template": {}},
                "skippable": True,
            },
            {
                "id": "extra",
                "type": "freeform",
                "prompt": "Anything else?",
                "binds": {"action": "remember", "params_template": {}},
                "skippable": True,
            },
        ],
    }
    interview.update(interview_overrides)
    return {"export_id": "exp_1111111111111111", "onboarding_interview": interview}


def _mock_conn(*, state_after_start, per_question_results):
    """conn.fetchval dispatches on the SQL text's first word after SELECT."""

    async def fetchval(query, *args):
        if "start_hmx_interview" in query:
            return json.dumps(state_after_start)
        if "record_hmx_interview_answer" in query:
            question_id = args[1]
            return json.dumps(per_question_results[question_id])
        if "get_hmx_interview_state" in query:
            return json.dumps(state_after_start | {"status": "completed"})
        raise AssertionError(f"unexpected query: {query}")

    conn = AsyncMock()
    conn.fetchval.side_effect = fetchval
    return conn


async def test_no_interview_section_is_a_no_op():
    conn = AsyncMock()
    outcome = await maybe_run_cli_interview(conn, {"export_id": "exp_x"}, interactive=True)
    assert outcome.ran is False
    conn.fetchval.assert_not_called()


async def test_non_interactive_starts_but_does_not_prompt():
    conn = _mock_conn(
        state_after_start={"status": "pending", "answers": []},
        per_question_results={},
    )
    doc = _document()
    with patch("core.hmx_interview.select_value") as select_mock:
        outcome = await maybe_run_cli_interview(conn, doc, interactive=False)
    assert outcome.ran is False
    assert doc["export_id"] in outcome.note
    select_mock.assert_not_called()
    conn.fetchval.assert_called_once()  # only start_hmx_interview


async def test_already_completed_interview_is_skipped():
    conn = _mock_conn(
        state_after_start={"status": "completed", "answers": []},
        per_question_results={},
    )
    with patch("core.hmx_interview.select_value") as select_mock:
        outcome = await maybe_run_cli_interview(conn, _document(), interactive=True)
    assert outcome.ran is False
    assert "already completed" in outcome.note
    select_mock.assert_not_called()


async def test_choice_and_freeform_questions_answered_end_to_end():
    conn = _mock_conn(
        state_after_start={"status": "pending", "answers": []},
        per_question_results={
            "name": {
                "success": True,
                "binding_result": {"success": True},
                "interview_status": "pending",
            },
            "extra": {
                "success": True,
                "binding_result": {"success": True},
                "interview_status": "completed",
            },
        },
    )
    with (
        patch("core.hmx_interview.select_value", new=AsyncMock(return_value="friend")),
        patch("core.hmx_interview.prompt_text", new=AsyncMock(return_value="nothing else")),
    ):
        outcome = await maybe_run_cli_interview(conn, _document(), interactive=True)

    assert outcome.ran is True
    assert outcome.completed is True
    assert outcome.answered == 2
    assert outcome.skipped == 0
    assert outcome.failed_bindings == 0

    # The choice answer's value (not its label) is what gets recorded.
    answer_calls = [c for c in conn.fetchval.call_args_list if "record_hmx_interview_answer" in c.args[0]]
    assert len(answer_calls) == 2
    assert json.loads(answer_calls[0].args[3]) == "friend"
    assert json.loads(answer_calls[1].args[3]) == "nothing else"


async def test_skippable_freeform_question_skips_on_empty_answer():
    conn = _mock_conn(
        state_after_start={"status": "pending", "answers": []},
        per_question_results={
            "name": {"success": True, "binding_result": None, "interview_status": "pending"},
        },
    )
    doc = _document(
        questions=[
            {
                "id": "name",
                "type": "freeform",
                "prompt": "What should I call you?",
                "binds": {"action": "remember", "params_template": {}},
                "skippable": True,
            }
        ]
    )
    with patch("core.hmx_interview.prompt_text", new=AsyncMock(return_value="")):
        outcome = await maybe_run_cli_interview(conn, doc, interactive=True)

    assert outcome.skipped == 1
    assert outcome.answered == 0
    skip_calls = [c for c in conn.fetchval.call_args_list if "record_hmx_interview_answer" in c.args[0]]
    assert len(skip_calls) == 1
    assert "'skipped'" in skip_calls[0].args[0]
    assert skip_calls[0].args[2] == "name"


async def test_required_freeform_question_re_asks_until_answered():
    conn = _mock_conn(
        state_after_start={"status": "pending", "answers": []},
        per_question_results={
            "name": {"success": True, "binding_result": {"success": True}, "interview_status": "completed"},
        },
    )
    doc = _document(
        questions=[
            {
                "id": "name",
                "type": "freeform",
                "prompt": "What should I call you?",
                "binds": {"action": "remember", "params_template": {}},
                "required": True,
                "skippable": False,
            }
        ]
    )
    prompt_mock = AsyncMock(side_effect=["", "  ", "Friend"])
    with patch("core.hmx_interview.prompt_text", new=prompt_mock):
        outcome = await maybe_run_cli_interview(conn, doc, interactive=True)

    assert prompt_mock.await_count == 3
    assert outcome.answered == 1
    assert outcome.skipped == 0


async def test_failed_binding_is_counted_and_reported():
    conn = _mock_conn(
        state_after_start={"status": "pending", "answers": []},
        per_question_results={
            "name": {
                "success": True,
                "binding_result": {"success": False, "error": "nope", "error_type": "not_implemented"},
                "interview_status": "completed",
            },
        },
    )
    doc = _document(
        questions=[
            {
                "id": "name",
                "type": "choice",
                "prompt": "Pick one",
                "options": [{"label": "A", "value": "a"}],
                "binds": {"action": "create_goal", "params_template": {}},
                "skippable": True,
            }
        ]
    )
    with patch("core.hmx_interview.select_value", new=AsyncMock(return_value="a")):
        outcome = await maybe_run_cli_interview(conn, doc, interactive=True)

    assert outcome.answered == 1
    assert outcome.failed_bindings == 1


async def test_already_answered_questions_are_not_re_asked():
    conn = _mock_conn(
        state_after_start={
            "status": "pending",
            "answers": [{"question_id": "name", "status": "answered"}],
        },
        per_question_results={
            "extra": {"success": True, "binding_result": {"success": True}, "interview_status": "completed"},
        },
    )
    with (
        patch("core.hmx_interview.select_value", new=AsyncMock()) as select_mock,
        patch("core.hmx_interview.prompt_text", new=AsyncMock(return_value="ok")),
    ):
        outcome = await maybe_run_cli_interview(conn, _document(), interactive=True)

    select_mock.assert_not_called()  # 'name' (choice) already answered
    assert outcome.answered == 1  # only 'extra' asked
