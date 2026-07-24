from apps.cli_chat import _append_visible_turn, _looks_like_json_path


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
