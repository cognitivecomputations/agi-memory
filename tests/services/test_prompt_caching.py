"""The system prompt has a stable prefix that providers can cache.

Providers bill a stable prefix once and reuse it — OpenAI and Gemini 2.5+
automatically, Anthropic via an explicit `cache_control` breakpoint. All need every
volatile part to come *after* everything stable. The prompt previously interleaved
them: a live `## Now` timestamp sat two-thirds of the way up, so the prefix changed
on every single turn and nothing could ever be reused.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core import llm
from core.llm import _anthropic_system_blocks, _extract_system_parts
from core.providers.anthropic_http import _build_system_prompt
from core.usage import extract_usage
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


class TestGeminiPrefixCaching:
    @pytest.mark.asyncio
    async def test_public_api_keeps_stable_prefix_before_volatile_tail(self):
        response = MagicMock(text="ok", function_calls=[])
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(return_value=response)
        messages = [
            {"role": "system", "content": "STABLE"},
            {"role": "system", "content": "VOLATILE"},
            {"role": "user", "content": "hello"},
        ]

        with patch.object(llm.genai, "Client", return_value=client):
            await llm.chat_completion(
                provider="gemini",
                model="gemini-2.5-flash",
                endpoint=None,
                api_key="test",
                messages=messages,
            )

        config = client.aio.models.generate_content.await_args.kwargs["config"]
        assert config.system_instruction == "STABLE\n\nVOLATILE"

    @pytest.mark.asyncio
    async def test_stream_preserves_final_usage_metadata(self):
        first = SimpleNamespace(text="Hel", function_calls=[])
        last = SimpleNamespace(
            text="lo",
            function_calls=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=7000,
                candidates_token_count=20,
                cached_content_token_count=6400,
            ),
        )

        async def chunks():
            yield first
            yield last

        client = MagicMock()
        client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda **_: chunks()
        )
        messages = [
            {"role": "system", "content": "STABLE"},
            {"role": "system", "content": "VOLATILE"},
            {"role": "user", "content": "hello"},
        ]

        with patch.object(llm.genai, "Client", return_value=client):
            result = await llm.stream_chat_completion(
                provider="gemini",
                model="gemini-2.5-flash",
                endpoint=None,
                api_key="test",
                messages=messages,
            )

        config = client.aio.models.generate_content_stream.call_args.kwargs["config"]
        assert config.system_instruction == "STABLE\n\nVOLATILE"
        assert result["content"] == "Hello"
        assert result["raw"] is last
        assert extract_usage("gemini", result["raw"]) == {
            "input_tokens": 7000,
            "output_tokens": 20,
            "cache_read_tokens": 6400,
            "cache_write_tokens": 0,
        }

    @pytest.mark.asyncio
    async def test_code_assist_does_not_override_provider_prompt_assembly(self):
        messages = [
            {"role": "system", "content": "STABLE"},
            {"role": "system", "content": "VOLATILE"},
            {"role": "user", "content": "hello"},
        ]

        with patch(
            "core.providers.google_code_assist.google_code_assist_completion",
            new_callable=AsyncMock,
            return_value={"content": "ok", "tool_calls": [], "raw": None},
        ) as completion:
            await llm.chat_completion(
                provider="google-gemini-cli",
                model="gemini-2.5-flash",
                endpoint=None,
                api_key=json.dumps({"token": "test", "projectId": "project"}),
                messages=messages,
            )

        kwargs = completion.await_args.kwargs
        assert kwargs["messages"] == messages
        assert "system_prompt" not in kwargs
