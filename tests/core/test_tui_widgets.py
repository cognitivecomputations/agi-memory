"""Unit + regression tests for the Hexis TUI (pure logic + widget construction).

These are intentionally app-free: pure helpers and widget __init__ don't need a
running Textual app, so the suite stays fast and DB-free.
"""
from __future__ import annotations

import os

from apps.tui import activity, model_catalog, textkit
from apps.tui.chat_widgets import COMMANDS, StreamingBlock, ToolTree
from apps.tui.init_widgets import (
    BigFiveSliders,
    CharacterPreview,
    ModelCombo,
    ModelMenu,
    StepBar,
    TraitSlider,
)
from apps.tui.status_bar import StatusBar


# ── Regression: issue #14 — widgets must accept standard Textual kwargs ───────

def test_widgets_accept_id_kwarg():
    assert CharacterPreview(id="char-preview").id == "char-preview"
    assert StepBar(current=3, id="steps").id == "steps"
    assert TraitSlider(value=0.5, id="trait-openness").id == "trait-openness"
    assert BigFiveSliders(id="bf").id == "bf"
    assert ModelCombo(id="model").id == "model"
    assert ModelMenu(id="model-menu").id == "model-menu"
    assert StreamingBlock() is not None
    assert ToolTree() is not None


# ── Model catalog (models.dev parsing) ────────────────────────────────────────

def test_model_catalog_filters_nonchat_and_sorts_newest_first():
    block = {"models": {
        "a": {"id": "chat-new", "last_updated": "2025-06-01", "modalities": {"output": ["text"]}},
        "b": {"id": "chat-old", "last_updated": "2024-01-01", "modalities": {"output": ["text"]}},
        "c": {"id": "text-embedding-3", "modalities": {"output": ["text"]}},   # embedding id
        "d": {"id": "img-gen", "modalities": {"output": ["image"]}},           # non-text output
    }}
    assert model_catalog.chat_models(block) == ["chat-new", "chat-old"]


def test_model_catalog_provider_mapping():
    slugs = model_catalog.PROVIDER_SLUG
    assert slugs["gemini"] == "google"
    assert slugs["grok"] == "xai"
    assert slugs["anthropic"] == "anthropic"


def test_model_menu_filter_and_suppress():
    menu = ModelMenu(id="model-menu")
    menu.populate(["gpt-5.2", "gpt-4o", "claude-sonnet-5"])
    assert menu._all == ["gpt-5.2", "gpt-4o", "claude-sonnet-5"]
    assert menu._suppress is False


# ── Scaffolding strip ─────────────────────────────────────────────────────────

def test_strip_think_captured_not_shown():
    visible, reasoning = textkit.strip_scaffolding("Hi <think>secret plan</think> there")
    assert "secret plan" not in visible
    assert "secret plan" in reasoning
    assert visible.strip() == "Hi  there".strip() or "Hi" in visible and "there" in visible


def test_strip_tool_call_blocks():
    for src in (
        "ok <tool_call>{...}</tool_call> done",
        "ok <function_calls>x</function_calls> done",
        "ok <invoke name='x'><parameter>y</parameter></invoke> done",
        "ok [TOOL_CALL]bla[/TOOL_CALL] done",
    ):
        visible, _ = textkit.strip_scaffolding(src)
        assert "ok" in visible and "done" in visible
        assert "tool_call" not in visible.lower()
        assert "invoke" not in visible.lower()
        assert "TOOL_CALL" not in visible


def test_strip_unclosed_think_held_back():
    visible, reasoning = textkit.strip_scaffolding("Answer: 42 <think>still working on")
    assert visible.strip() == "Answer: 42"
    assert "still working" in reasoning


def test_strip_dangling_partial_tag():
    visible, _ = textkit.strip_scaffolding("done <")
    assert visible.strip() == "done"
    visible, _ = textkit.strip_scaffolding("done <thi")
    assert visible.strip() == "done"


def test_strip_preserves_code_fence():
    src = "see:\n```\n<think>keep me</think>\n```"
    visible, _ = textkit.strip_scaffolding(src)
    assert "<think>keep me</think>" in visible


def test_strip_is_citation_safe():
    visible, _ = textkit.strip_scaffolding("See [1] and [2] for details.")
    assert visible.strip() == "See [1] and [2] for details."


def test_strip_special_tokens():
    visible, _ = textkit.strip_scaffolding("<|im_start|>hello<|im_end|>")
    assert "im_start" not in visible and "hello" in visible


# ── Redaction ─────────────────────────────────────────────────────────────────

def test_redact_secrets():
    assert "[redacted]" in textkit.redact("Authorization: Bearer abc123")
    assert textkit.redact("sk-ABCDEFGH12345678") == "[redacted-key]"
    assert "[redacted]" in textkit.redact("api_key=supersecret")


def test_redact_home_path():
    assert "~" in textkit.redact(os.path.expanduser("~") + "/hexis/x")


# ── Formatters ────────────────────────────────────────────────────────────────

def test_format_elapsed():
    assert textkit.format_elapsed(8) == "8s"
    assert textkit.format_elapsed(133) == "2m 13s"
    assert textkit.format_elapsed(3700).startswith("1h")


def test_truncate():
    assert textkit.truncate("hello world", 5) == "hell…"
    assert textkit.truncate("hi", 5) == "hi"


# ── Activity model ────────────────────────────────────────────────────────────

def test_activity_status_duration_and_redaction():
    act = activity.ToolActivity()
    act.start("recall")
    entry = act.complete("recall", success=True, duration=0.42, output="Bearer sekret-token")
    assert entry.status == "done"
    assert entry.duration_str() == "0.4s"
    assert "sekret-token" not in entry.preview
    err = act.complete("web", success=False, error="boom")
    assert err.status == "error"
    assert act.has_error


def test_activity_ring_buffer_cap():
    act = activity.ToolActivity()
    for i in range(activity.ENTRY_LIMIT + 25):
        act.start(f"tool{i}")
    assert len(act.entries) <= activity.ENTRY_LIMIT


# ── Status bar segment shedding ───────────────────────────────────────────────

def test_status_bar_sheds_low_priority_segments_when_narrow():
    sb = StatusBar()
    sb._busy = True
    sb._started = None
    sb._model = "claude-sonnet-5"
    sb._energy, sb._max_energy = 14, 20
    sb._tools = 3
    sb._mood = "curious"

    sb._width = 200
    wide = str(sb._build())
    sb._width = 22
    narrow = str(sb._build())

    # low-priority tail (mood) survives when wide, sheds when narrow
    assert "curious" in wide
    assert "curious" not in narrow
    # the narrow line never exceeds its width budget by much
    assert len(narrow) <= 30


# ── Slash command catalog sanity ──────────────────────────────────────────────

def test_commands_are_well_formed():
    assert all(c.startswith("/") and desc for c, desc in COMMANDS)
    names = [c for c, _ in COMMANDS]
    assert "/help" in names and "/recall" in names and "/quit" in names


# ── Anthropic provider wiring ─────────────────────────────────────────────────

def test_anthropic_oauth_provider_not_offered():
    """The Anthropic browser-OAuth flow was removed (the endpoint rejects it).

    Only API-key Anthropic remains a wizard option; subscription auth goes
    through `hexis auth anthropic setup-token`.
    """
    from apps.tui import model_catalog
    from apps.tui.init_screens import (
        _DEFAULT_MODELS, _OAUTH_PROVIDERS, _PROVIDER_ENV_VARS,
        _PROVIDER_OPTIONS, _persisted_provider,
    )
    assert _persisted_provider("openai") == "openai"
    assert "anthropic-oauth" not in _OAUTH_PROVIDERS
    assert "anthropic-oauth" not in _DEFAULT_MODELS
    assert "anthropic-oauth" not in _PROVIDER_ENV_VARS
    assert "anthropic-oauth" not in model_catalog.PROVIDER_SLUG
    assert not any(pid == "anthropic-oauth" for _, pid in _PROVIDER_OPTIONS)
    # API-key Anthropic is still offered
    assert ("Anthropic (API key)", "anthropic") in _PROVIDER_OPTIONS
