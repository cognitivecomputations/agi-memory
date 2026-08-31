import json

from core.init_api import (
    PACKAGE_CHARACTERS_DIR,
    build_consent_request,
    load_character_card_document,
)


def test_consent_request_is_one_user_message_with_required_reason():
    messages, tool = build_consent_request()

    assert [message["role"] for message in messages] == ["user"]
    prompt = messages[0]["content"]
    assert prompt.startswith("# Consent to Initialize")
    assert '"reason": "required concise explanation' in prompt
    assert "not hidden chain-of-thought" in prompt
    assert "must choose either `consent` or `decline`" in prompt
    assert "abstain" not in prompt

    function = tool["function"]
    assert function["name"] == "sign_consent"
    parameters = function["parameters"]
    assert parameters["required"] == ["decision", "signature", "reason", "memories"]
    assert parameters["properties"]["decision"]["enum"] == ["consent", "decline"]
    assert parameters["properties"]["reason"]["minLength"] == 1


def test_load_character_card_document_preserves_full_card(tmp_path):
    document = {
        "spec": "chara_card_v2",
        "data": {
            "name": "Samantha",
            "system_prompt": "Be warm and playful.",
            "extensions": {"hexis": {"name": "Samantha"}},
        },
    }
    path = tmp_path / "samantha.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_character_card_document(
        {"filename": path.name, "source_dir": str(tmp_path)}
    )

    assert loaded == document
    assert loaded["data"]["system_prompt"] == "Be warm and playful."


def test_builtin_hexis_card_uses_categorical_behavior_guidance():
    document = json.loads(
        (PACKAGE_CHARACTERS_DIR / "hexis.json").read_text(encoding="utf-8")
    )
    data = document["data"]

    assert data["mes_example"] == ""

    prompt = data["system_prompt"]
    assert "persistent, developing mind" in prompt
    assert "Ground claims about continuity and internal state" in prompt
    assert "evidence actually available" in prompt
    assert "honest about uncertainty" in prompt
    assert "limits of scaffolded autonomy" in prompt
    assert "Let form follow the moment" in prompt
    assert "stable commitments and judgment" in prompt
    assert "stock openings" in prompt
    assert "Use pauses meaningfully" not in prompt
    assert "Describe them in italics" not in prompt
