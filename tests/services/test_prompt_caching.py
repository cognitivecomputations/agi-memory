"""The system prompt has a stable prefix that providers can cache.

Providers bill a stable prefix once and reuse it — OpenAI automatically, Anthropic
via an explicit `cache_control` breakpoint. Both need every volatile part to come
*after* everything stable. The prompt previously interleaved them: a live `## Now`
timestamp sat two-thirds of the way up, so the prefix changed on every single turn
and nothing could ever be reused.
"""

from __future__ import annotations

import json

import pytest

from core.llm import _anthropic_system_blocks, _extract_system_parts
from core.providers.anthropic_http import _build_system_prompt
from services.agent import SystemPrompt


class TestSystemPromptCarrier:
    def test_it_is_still_the_whole_prompt(self):
        """Every existing consumer treats this as a plain string."""
        sp = SystemPrompt("STABLE", "VOLATILE")
        assert isinstance(sp, str)
        assert sp == "STABLE\n\nVOLATILE"
        assert str(sp) == "STABLE\n\nVOLATILE"

    def test_it_carries_the_boundary(self):
        sp = SystemPrompt("STABLE", "VOLATILE")
        assert sp.stable == "STABLE"
        assert sp.volatile == "VOLATILE"

    def test_an_empty_volatile_half_does_not_leave_a_trailing_gap(self):
        assert SystemPrompt("STABLE", "") == "STABLE"


class TestTheCacheableProperty:
    """The one property that matters: the prefix must not change between turns."""

    def _prompt(self, *, now: str, who: str, skills: str) -> SystemPrompt:
        stable = "IDENTITY\n\nWORLDVIEW\n\nPERSONA"
        volatile = f"## Now\n{now}\n\n{who}\n\n{skills}"
        return SystemPrompt(stable, volatile)

    def test_prefix_is_byte_identical_across_turns_that_differ_completely(self):
        first = self._prompt(now="2026-08-23 09:00", who="Eric", skills="calendar")
        second = self._prompt(now="2026-08-23 17:42", who="Sarah Chen", skills="research")

        assert first.stable == second.stable, "the cacheable prefix must not vary"
        assert first.volatile != second.volatile, "the turn-specific half must vary"
        assert first != second


class TestSystemPartsSurviveTheLLMLayer:
    def test_parts_are_kept_separate_rather_than_flattened(self):
        messages = [
            {"role": "system", "content": "STABLE"},
            {"role": "system", "content": "VOLATILE"},
            {"role": "user", "content": "hi"},
        ]
        parts, rest = _extract_system_parts(messages)
        assert parts == ["STABLE", "VOLATILE"]
        assert [m["role"] for m in rest] == ["user"]

    def test_blank_system_messages_are_dropped(self):
        parts, _ = _extract_system_parts(
            [{"role": "system", "content": "  "}, {"role": "system", "content": "REAL"}]
        )
        assert parts == ["REAL"]


class TestAnthropicBreakpoint:
    def test_the_marker_lands_on_the_last_stable_block(self):
        blocks = _anthropic_system_blocks(["STABLE", "VOLATILE"])
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in blocks[1], "the volatile tail is never cached"

    def test_a_single_part_has_nothing_to_cache_against(self):
        blocks = _anthropic_system_blocks(["ONLY"])
        assert blocks == [{"type": "text", "text": "ONLY"}]

    def test_no_system_message_stays_none(self):
        assert _anthropic_system_blocks([]) is None

    @pytest.mark.parametrize("auth_mode", ["api-key", "setup-token"])
    def test_the_oauth_path_caches_too(self, auth_mode):
        """Both Anthropic paths cache, or OAuth users silently get nothing."""
        result = _build_system_prompt(["STABLE", "VOLATILE"], auth_mode)
        assert isinstance(result, list)
        marked = [b for b in result if "cache_control" in b]
        assert len(marked) == 1
        assert marked[0]["text"] == "STABLE"
        assert result[-1]["text"] == "VOLATILE"
        assert "cache_control" not in result[-1]

    def test_a_plain_string_is_passed_through_unchanged(self):
        """No split means no cacheable boundary — so do not reshape the wire.

        The OAuth flow is a validated path: Anthropic checks the identity
        preamble. A prompt with nothing to cache keeps exactly the shape it has
        always had rather than becoming blocks for no gain.
        """
        from core.providers.anthropic_http import _CLAUDE_CODE_IDENTITY

        assert _build_system_prompt("ONE", "api-key") == "ONE"
        oauth = _build_system_prompt("ONE", "setup-token")
        assert isinstance(oauth, str)
        assert oauth.startswith(_CLAUDE_CODE_IDENTITY)
