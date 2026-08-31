from unittest.mock import AsyncMock, patch

import pytest

from apps.cli_chat import (
    _answer_cli_question,
    _append_visible_turn,
    _looks_like_json_path,
)


def test_greet_turn_does_not_seed_assistant_first_history():
    history = []

    _append_visible_turn(
        history,
        user_input="synthetic greet",
        assistant_text="Hello, I am Samantha.",
        was_greet=True,
    )

    assert history == []


def test_real_turn_appends_user_then_assistant():
    history = []

    _append_visible_turn(
        history,
        user_input="Hello",
        assistant_text="Hi.",
        was_greet=False,
    )

    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
    ]


def test_absolute_json_paths_are_not_slash_commands():
    assert _looks_like_json_path("/Users/eric/Downloads/client_secret.json")
    assert _looks_like_json_path("~/Downloads/oauth-client.json")
    assert not _looks_like_json_path("/help")


@pytest.mark.asyncio
async def test_cli_question_picker_persists_the_selected_choice():
    pool = object()
    with (
        patch(
            "apps.cli_prompts.select_index", new=AsyncMock(return_value=2)
        ),
        patch("apps.cli_prompts.text", new=AsyncMock()) as prompt_text,
        patch(
            "services.agent_questions.answer_agent_question",
            new=AsyncMock(return_value={"ok": True, "status": "answered"}),
        ) as answer,
    ):
        await _answer_cli_question(
            pool,
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "prompt": "Which contract?",
                "choices": ["Manning", "Hartford"],
                "allow_free_text": True,
            },
        )

    prompt_text.assert_not_awaited()
    answer.assert_awaited_once_with(
        pool,
        "11111111-1111-4111-8111-111111111111",
        answer=None,
        choice_index=2,
        channel="cli",
        actor="local-user",
    )
