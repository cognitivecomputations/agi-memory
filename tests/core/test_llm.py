import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core import llm

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def _reset_responses_cache():
    """Clear the per-endpoint Responses API capability cache and OpenAI client cache between tests."""
    llm._endpoint_responses_support.clear()
    llm._clear_openai_client_cache()
    yield
    llm._endpoint_responses_support.clear()
    llm._clear_openai_client_cache()


def test_normalize_provider_variants():
    assert llm.normalize_provider(None) == "openai"
    assert llm.normalize_provider("OpenAI") == "openai"
    assert llm.normalize_provider("openai_chat_completions_endpoint") == "openai-chat-completions-endpoint"


@pytest.mark.asyncio(loop_scope="session")
async def test_close_async_stream_falls_back_to_response():
    class _Stream:
        pass

    response = MagicMock()
    response.aclose = AsyncMock()
    stream = _Stream()
    stream.response = response

    await llm._close_async_stream(stream)

    response.aclose.assert_awaited_once_with()


def test_normalize_provider_new_aliases():
    assert llm.normalize_provider("github_copilot") == "github-copilot"
    assert llm.normalize_provider("qwen_portal") == "qwen-portal"
    assert llm.normalize_provider("minimax_portal") == "minimax-portal"
    assert llm.normalize_provider("google_gemini_cli") == "google-gemini-cli"
    assert llm.normalize_provider("google_antigravity") == "google-antigravity"
    # Already-hyphenated should pass through
    assert llm.normalize_provider("github-copilot") == "github-copilot"
    assert llm.normalize_provider("chutes") == "chutes"


def test_normalize_endpoint_defaults():
    assert llm.normalize_endpoint("openai", "  ") is None
    assert llm.normalize_endpoint("openai", " https://example.com ") == "https://example.com"


def test_normalize_endpoint_new_providers():
    assert llm.normalize_endpoint("chutes", None) == "https://api.chutes.ai/v1"
    assert llm.normalize_endpoint("qwen-portal", None) == "https://portal.qwen.ai/v1"
    assert llm.normalize_endpoint("google-gemini-cli", None) == "https://cloudcode-pa.googleapis.com"
    assert llm.normalize_endpoint("google-antigravity", None) == "https://cloudcode-pa.googleapis.com"
    assert llm.normalize_endpoint("openai-codex", None) == "https://chatgpt.com/backend-api"
    assert (
        llm.normalize_endpoint("openai-codex", "https://api.openai.com/v1")
        == "https://chatgpt.com/backend-api"
    )


def test_normalize_llm_config_preserves_auth_mode():
    cfg = llm.normalize_llm_config({
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "auth_mode": "setup-token",
    })
    assert cfg["auth_mode"] == "setup-token"


def test_normalize_llm_config_no_auth_mode():
    cfg = llm.normalize_llm_config({"provider": "openai", "model": "gpt-4o"})
    assert "auth_mode" not in cfg


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "abc123")
    assert llm.resolve_api_key("TEST_API_KEY") == "abc123"
    assert llm.resolve_api_key("  ") is None


def test_normalize_llm_config_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "envkey")
    normalized = llm.normalize_llm_config({"provider": "openai", "model": "gpt-4o"})
    assert normalized["api_key"] == "envkey"


def test_openai_compatible_does_not_consume_ambient_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")

    normalized = llm.normalize_llm_config({
        "provider": "openai_compatible",
        "model": "local-model",
        "endpoint": "http://localhost:11434/v1",
    })

    assert normalized["api_key"] is None


def test_keyless_openai_compatible_uses_internal_sdk_key(monkeypatch):
    captured: dict[str, object] = {}
    client = object()

    def fake_async_openai(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm.openai, "AsyncOpenAI", fake_async_openai)

    result = llm._get_openai_client(
        None,
        "http://localhost:11434/v1",
        "openai_compatible",
    )

    assert result is client
    assert captured == {
        "api_key": llm._LOCAL_OPENAI_COMPATIBLE_API_KEY,
        "base_url": "http://localhost:11434/v1",
    }


def test_extract_system_prompt():
    messages = [
        {"role": "system", "content": "A"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "B"},
    ]
    system, rest = llm._extract_system_prompt(messages)  # noqa: SLF001
    assert system == "A\n\nB"
    assert rest == [{"role": "user", "content": "hi"}]


def test_openai_tool_call_parsing():
    class DummyFn:
        def __init__(self):
            self.name = "recall"
            self.arguments = '{"query":"hi"}'

    class DummyCall:
        def __init__(self):
            self.id = "call-1"
            self.function = DummyFn()

    parsed = llm._openai_tool_calls([DummyCall()])  # noqa: SLF001
    assert parsed == [{"id": "call-1", "name": "recall", "arguments": {"query": "hi"}}]


def test_anthropic_tools_conversion():
    tools = [
        {"function": {"name": "recall", "description": "desc", "parameters": {"type": "object"}}}
    ]
    out = llm._anthropic_tools(tools)  # noqa: SLF001
    assert out == [{"name": "recall", "description": "desc", "input_schema": {"type": "object"}}]


@pytest.mark.asyncio(loop_scope="session")
async def test_chat_completion_unsupported_provider():
    with pytest.raises(ValueError):
        await llm.chat_completion(
            provider="unknown",
            model="x",
            endpoint=None,
            api_key=None,
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_stream_text_completion_unsupported_provider():
    with pytest.raises(ValueError):
        async for _ in llm.stream_text_completion(
            provider="unknown",
            model="x",
            endpoint=None,
            api_key=None,
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass


@pytest.mark.asyncio(loop_scope="session")
async def test_chat_completion_requires_openai_package(monkeypatch):
    monkeypatch.setattr(llm, "openai", None)
    with pytest.raises(RuntimeError):
        await llm.chat_completion(
            provider="openai",
            model="x",
            endpoint=None,
            api_key=None,
            messages=[{"role": "user", "content": "hi"}],
        )


# ---------------------------------------------------------------------------
# Responses API: format converters
# ---------------------------------------------------------------------------


class TestToolsToResponses:
    def test_nested_to_flat(self):
        tools = [
            {"type": "function", "function": {"name": "recall", "description": "Search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
        ]
        out = llm._tools_to_responses(tools)  # noqa: SLF001
        assert out == [{
            "type": "function",
            "name": "recall",
            "description": "Search",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }]

    def test_empty_and_none(self):
        assert llm._tools_to_responses(None) == []  # noqa: SLF001
        assert llm._tools_to_responses([]) == []  # noqa: SLF001


class TestMessagesToResponsesInput:
    def test_basic_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        instructions, items = llm._messages_to_responses_input(messages)  # noqa: SLF001
        assert instructions == "You are helpful"
        assert items == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_multimodal_user_message(self):
        image_url = "data:image/png;base64,aW1hZ2U="
        messages = [
            {"role": "system", "content": "You are helpful"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What do you see?"},
                    {"type": "input_image", "image_url": image_url},
                ],
            },
        ]
        instructions, items = llm._messages_to_responses_input(messages)  # noqa: SLF001
        assert instructions == "You are helpful"
        assert items == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What do you see?"},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ]

    def test_codex_multimodal_user_message(self):
        image_url = "data:image/jpeg;base64,aW1hZ2U="
        instructions, items = llm._messages_to_codex_responses_input([  # noqa: SLF001
            {"role": "system", "content": "You are helpful"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ])
        assert instructions == "You are helpful"
        assert items == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image."},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ]

    def test_anthropic_multimodal_user_message(self):
        messages = llm._messages_to_anthropic_messages([  # noqa: SLF001
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image."},
                    {"type": "input_image", "image_url": "data:image/webp;base64,aW1hZ2U="},
                ],
            }
        ])
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/webp",
                            "data": "aW1hZ2U=",
                        },
                    },
                ],
            }
        ]

    def test_with_tool_calls(self):
        """Test assistant messages with OpenAI-format tool_calls (as stored by agent_loop)."""
        messages = [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": "I'll search.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "recall", "arguments": '{"query":"cats"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "Found 3 results"},
        ]
        instructions, items = llm._messages_to_responses_input(messages)  # noqa: SLF001
        assert instructions is None
        assert items == [
            {"role": "user", "content": "search for cats"},
            {"role": "assistant", "content": "I'll search."},
            {"type": "function_call", "call_id": "call-1", "name": "recall", "arguments": '{"query":"cats"}'},
            {"type": "function_call_output", "call_id": "call-1", "output": "Found 3 results"},
        ]

    def test_multiple_system_messages(self):
        messages = [
            {"role": "system", "content": "Part A"},
            {"role": "system", "content": "Part B"},
            {"role": "user", "content": "hi"},
        ]
        instructions, _ = llm._messages_to_responses_input(messages)  # noqa: SLF001
        assert instructions == "Part A\n\nPart B"

    def test_no_system(self):
        messages = [{"role": "user", "content": "hi"}]
        instructions, _ = llm._messages_to_responses_input(messages)  # noqa: SLF001
        assert instructions is None

    def test_dict_arguments_serialized(self):
        """If arguments are a dict (not a string), they get JSON-serialized."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "foo", "arguments": {"key": "val"}},
                    }
                ],
            }
        ]
        _, items = llm._messages_to_responses_input(messages)  # noqa: SLF001
        fc = items[0]
        assert fc["arguments"] == '{"key": "val"}'


class TestExtractResponsesResult:
    def test_text_only(self):
        resp = MagicMock()
        resp.output_text = "Hello world"
        resp.output = []
        result = llm._extract_responses_result(resp)  # noqa: SLF001
        assert result["content"] == "Hello world"
        assert result["tool_calls"] == []
        assert result["raw"] is resp

    def test_with_tool_calls(self):
        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.call_id = "call-42"
        fc_item.name = "recall"
        fc_item.arguments = '{"query":"test"}'

        resp = MagicMock()
        resp.output_text = ""
        resp.output = [fc_item]
        result = llm._extract_responses_result(resp)  # noqa: SLF001
        assert result["tool_calls"] == [
            {"id": "call-42", "name": "recall", "arguments": {"query": "test"}}
        ]

    def test_bad_json_args(self):
        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.call_id = "c1"
        fc_item.name = "broken"
        fc_item.arguments = "{invalid json"

        resp = MagicMock()
        resp.output_text = ""
        resp.output = [fc_item]
        result = llm._extract_responses_result(resp)  # noqa: SLF001
        assert result["tool_calls"][0]["arguments"] == {}


# ---------------------------------------------------------------------------
# Responses API: capability cache
# ---------------------------------------------------------------------------


class TestCapabilityCache:
    def test_should_try_no_sdk(self, monkeypatch):
        monkeypatch.setattr(llm, "_HAS_RESPONSES_API", False)
        assert llm._should_try_responses(None) is False  # noqa: SLF001

    def test_should_try_cached_false(self, monkeypatch):
        monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)
        llm._cache_responses_support("http://example.com/v1", False)  # noqa: SLF001
        assert llm._should_try_responses("http://example.com/v1") is False  # noqa: SLF001

    def test_should_try_cached_true(self, monkeypatch):
        monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)
        llm._cache_responses_support("http://example.com/v1", True)  # noqa: SLF001
        assert llm._should_try_responses("http://example.com/v1") is True  # noqa: SLF001

    def test_should_try_unknown(self, monkeypatch):
        monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)
        assert llm._should_try_responses("http://new-endpoint.com/v1") is True  # noqa: SLF001

    def test_cache_key_normalization(self):
        assert llm._endpoint_cache_key(None) == "default"  # noqa: SLF001
        assert llm._endpoint_cache_key("http://x.com/v1/") == "http://x.com/v1"  # noqa: SLF001


# ---------------------------------------------------------------------------
# Responses API: error classification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    def test_404_is_unsupported(self):
        exc = MagicMock(spec=llm.openai.NotFoundError)
        exc.__class__ = llm.openai.NotFoundError
        # isinstance checks need real instances; use a real-ish approach
        err = llm.openai.NotFoundError.__new__(llm.openai.NotFoundError)
        assert llm._is_responses_unsupported_error(err) is True  # noqa: SLF001

    def test_501_is_unsupported(self):
        err = llm.openai.APIStatusError.__new__(llm.openai.APIStatusError)
        object.__setattr__(err, "status_code", 501)
        assert llm._is_responses_unsupported_error(err) is True  # noqa: SLF001

    def test_auth_error_not_unsupported(self):
        err = llm.openai.AuthenticationError.__new__(llm.openai.AuthenticationError)
        assert llm._is_responses_unsupported_error(err) is False  # noqa: SLF001

    def test_generic_exception_not_unsupported(self):
        assert llm._is_responses_unsupported_error(RuntimeError("boom")) is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# Responses API: integration (mocked client)
# ---------------------------------------------------------------------------


def _make_responses_result(text="Hello", tool_calls=None):
    """Build a fake Responses API response object."""
    resp = MagicMock()
    resp.output_text = text
    items = []
    for tc in (tool_calls or []):
        item = MagicMock()
        item.type = "function_call"
        item.call_id = tc["id"]
        item.name = tc["name"]
        item.arguments = json.dumps(tc["arguments"])
        items.append(item)
    resp.output = items
    return resp


def _make_chat_completions_result(text="Hello", tool_calls=None):
    """Build a fake Chat Completions response object."""
    msg = MagicMock()
    msg.content = text
    if tool_calls:
        tcs = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            fn = MagicMock()
            fn.name = tc["name"]
            fn.arguments = json.dumps(tc["arguments"])
            mock_tc.function = fn
            tcs.append(mock_tc)
        msg.tool_calls = tcs
    else:
        msg.tool_calls = []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio(loop_scope="session")
async def test_chat_completion_uses_responses_api(monkeypatch):
    """When Responses API is available and supported, use it."""
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)

    fake_resp = _make_responses_result("I am responding")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=fake_resp)

    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **kw: mock_client)

    result = await llm.chat_completion(
        provider="openai",
        model="gpt-4o",
        endpoint=None,
        api_key="test",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result["content"] == "I am responding"
    assert result["tool_calls"] == []
    mock_client.responses.create.assert_awaited_once()
    # Should be cached as supported
    assert llm._endpoint_responses_support.get("default") is True


@pytest.mark.asyncio(loop_scope="session")
async def test_chat_completion_fallback_on_404(monkeypatch):
    """When Responses API returns 404, fall back to Chat Completions."""
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)

    # Responses API raises NotFoundError
    not_found = llm.openai.NotFoundError.__new__(llm.openai.NotFoundError)
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(side_effect=not_found)

    # Chat Completions succeeds
    chat_resp = _make_chat_completions_result("fallback response")
    mock_client.chat.completions.create = AsyncMock(return_value=chat_resp)

    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **kw: mock_client)

    result = await llm.chat_completion(
        provider="openai",
        model="gpt-4o",
        endpoint=None,
        api_key="test",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result["content"] == "fallback response"
    # Should be cached as unsupported
    assert llm._endpoint_responses_support.get("default") is False


@pytest.mark.asyncio(loop_scope="session")
async def test_chat_completion_skips_responses_after_cached_false(monkeypatch):
    """Once cached as unsupported, Responses API is never attempted."""
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)
    llm._cache_responses_support(None, False)

    mock_client = MagicMock()
    chat_resp = _make_chat_completions_result("direct")
    mock_client.chat.completions.create = AsyncMock(return_value=chat_resp)
    mock_client.responses.create = AsyncMock(side_effect=AssertionError("should not be called"))

    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **kw: mock_client)

    result = await llm.chat_completion(
        provider="openai",
        model="gpt-4o",
        endpoint=None,
        api_key="test",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result["content"] == "direct"
    mock_client.responses.create.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_compatible_completion_skips_responses_probe(monkeypatch):
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)

    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(
        side_effect=AssertionError("Responses API must not be used")
    )
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_chat_completions_result("compatible response")
    )
    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **_kwargs: mock_client)

    result = await llm.chat_completion(
        provider="openai_compatible",
        model="local-model",
        endpoint="http://local.test/v1",
        api_key="local-key",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result["content"] == "compatible response"
    mock_client.responses.create.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_chat_completion_real_error_propagates(monkeypatch):
    """Non-404 errors from Responses API propagate (not silently caught)."""
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)

    auth_err = llm.openai.AuthenticationError.__new__(llm.openai.AuthenticationError)
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(side_effect=auth_err)

    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **kw: mock_client)

    with pytest.raises(llm.openai.AuthenticationError):
        await llm.chat_completion(
            provider="openai",
            model="gpt-4o",
            endpoint=None,
            api_key="bad",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_chat_completion_with_tool_calls_via_responses(monkeypatch):
    """Responses API returns tool calls, verify they're normalized correctly."""
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)

    fake_resp = _make_responses_result(
        text="",
        tool_calls=[{"id": "call-99", "name": "recall", "arguments": {"query": "test"}}],
    )
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=fake_resp)
    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **kw: mock_client)

    result = await llm.chat_completion(
        provider="openai",
        model="gpt-4o",
        endpoint=None,
        api_key="test",
        messages=[{"role": "user", "content": "recall something"}],
        tools=[{"type": "function", "function": {"name": "recall", "parameters": {"type": "object"}}}],
    )
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["id"] == "call-99"
    assert result["tool_calls"][0]["name"] == "recall"
    assert result["tool_calls"][0]["arguments"] == {"query": "test"}


@pytest.mark.asyncio(loop_scope="session")
async def test_stream_completion_uses_responses_api(monkeypatch):
    """Streaming path also tries Responses API first."""
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)

    # Build a fake streaming context manager
    text_event = MagicMock()
    text_event.type = "response.output_text.delta"
    text_event.delta = "streamed"

    reasoning_event = MagicMock()
    reasoning_event.type = "response.reasoning_summary_text.delta"
    reasoning_event.delta = "thinking"

    fc_item = MagicMock()
    fc_item.type = "function_call"
    fc_item.id = "item-1"
    fc_item.call_id = "call-1"
    fc_item.name = "recall"
    fc_item.arguments = '{"q":"hi"}'

    done_event = MagicMock()
    done_event.type = "response.output_item.done"
    done_event.item = fc_item

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def __aiter__(self):
            yield reasoning_event
            yield text_event
            yield done_event

    mock_client = MagicMock()
    mock_client.responses.stream = MagicMock(return_value=FakeStream())
    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **kw: mock_client)

    deltas = []
    reasoning_deltas = []
    result = await llm.stream_chat_completion(
        provider="openai",
        model="gpt-4o",
        endpoint=None,
        api_key="test",
        messages=[{"role": "user", "content": "hi"}],
        on_text_delta=lambda t: deltas.append(t),
        on_reasoning_delta=lambda t: reasoning_deltas.append(t),
    )
    assert result["content"] == "streamed"
    assert deltas == ["streamed"]
    assert reasoning_deltas == ["thinking"]
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "recall"
    assert llm._endpoint_responses_support.get("default") is True


@pytest.mark.asyncio(loop_scope="session")
async def test_compatible_stream_uses_chat_and_separates_reasoning(monkeypatch):
    """Generic compatible servers use their promised Chat Completions stream."""
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)
    requests: list[tuple[str, dict]] = []

    async def fake_sse(_provider, _method, url, **kwargs):
        requests.append((url, kwargs["json_body"]))
        yield {"choices": [{"delta": {"reasoning": "thinking"}}]}
        yield {
            "choices": [{
                "delta": {"content": "answer"},
                "finish_reason": "stop",
            }]
        }

    monkeypatch.setattr(llm, "iter_sse_json_events", fake_sse)
    monkeypatch.setattr(
        llm.openai,
        "AsyncOpenAI",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("SDK streaming must not be used")
        ),
    )

    text_deltas: list[str] = []
    reasoning_deltas: list[str] = []
    result = await llm.stream_chat_completion(
        provider="openai_compatible",
        model="local-model",
        endpoint="http://local.test/v1",
        api_key="local-key",
        messages=[{"role": "user", "content": "hi"}],
        on_text_delta=text_deltas.append,
        on_reasoning_delta=reasoning_deltas.append,
    )

    assert result["content"] == "answer"
    assert text_deltas == ["answer"]
    assert reasoning_deltas == ["thinking"]
    assert requests[0][0] == "http://local.test/v1/chat/completions"
    assert requests[0][1]["stream"] is True


@pytest.mark.asyncio(loop_scope="session")
async def test_compatible_stream_propagates_midstream_error_after_partial_text(
    monkeypatch,
):
    calls = 0
    delivered: list[str] = []

    async def fake_sse(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        yield {"choices": [{"delta": {"content": "partial"}}]}
        raise ConnectionError("stream disconnected")

    monkeypatch.setattr(llm, "iter_sse_json_events", fake_sse)

    with pytest.raises(ConnectionError, match="stream disconnected"):
        await llm.stream_chat_completion(
            provider="openai_compatible",
            model="local-model",
            endpoint="http://local.test/v1",
            api_key="local-key",
            messages=[{"role": "user", "content": "hi"}],
            on_text_delta=delivered.append,
        )

    assert calls == 1
    assert delivered == ["partial"]


@pytest.mark.asyncio(loop_scope="session")
async def test_compatible_stream_accumulates_tool_call_deltas(monkeypatch):
    async def fake_sse(*_args, **_kwargs):
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "function": {"name": "lookup", "arguments": '{"query":'},
                    }]
                },
            }]
        }
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '"Libre WebUI"}'},
                    }]
                },
                "finish_reason": "tool_calls",
            }]
        }

    monkeypatch.setattr(llm, "iter_sse_json_events", fake_sse)

    result = await llm.stream_chat_completion(
        provider="openai_compatible",
        model="local-model",
        endpoint="http://local.test/v1",
        api_key=None,
        messages=[{"role": "user", "content": "look it up"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )

    assert result["tool_calls"] == [{
        "id": "call-1",
        "name": "lookup",
        "arguments": {"query": "Libre WebUI"},
    }]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "invalid_index",
    ["bad", -1, 1.5, True, "²", "9" * 5000],
)
async def test_compatible_stream_rejects_invalid_tool_call_index(
    monkeypatch,
    invalid_index,
):
    async def fake_sse(*_args, **_kwargs):
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": invalid_index,
                        "id": "call-1",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }]
                },
                "finish_reason": "tool_calls",
            }]
        }

    monkeypatch.setattr(llm, "iter_sse_json_events", fake_sse)

    with pytest.raises(RuntimeError, match="invalid index"):
        await llm.stream_chat_completion(
            provider="openai_compatible",
            model="local-model",
            endpoint="http://local.test/v1",
            api_key=None,
            messages=[{"role": "user", "content": "look it up"}],
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_compatible_stream_rejects_clean_truncated_text(monkeypatch):
    async def fake_sse(*_args, **_kwargs):
        yield {"choices": [{"delta": {"content": "partial"}}]}

    monkeypatch.setattr(llm, "iter_sse_json_events", fake_sse)

    with pytest.raises(RuntimeError, match="before a finish signal"):
        await llm.stream_chat_completion(
            provider="openai_compatible",
            model="local-model",
            endpoint="http://local.test/v1",
            api_key=None,
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_compatible_stream_rejects_truncated_tool_arguments(monkeypatch):
    async def fake_sse(*_args, **_kwargs):
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "function": {"name": "lookup", "arguments": '{"query":'},
                    }]
                }
            }]
        }

    monkeypatch.setattr(llm, "iter_sse_json_events", fake_sse)

    with pytest.raises(RuntimeError, match="before a finish signal"):
        await llm.stream_chat_completion(
            provider="openai_compatible",
            model="local-model",
            endpoint="http://local.test/v1",
            api_key=None,
            messages=[{"role": "user", "content": "look it up"}],
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_compatible_stream_rejects_malformed_terminal_tool_arguments(monkeypatch):
    async def fake_sse(*_args, **_kwargs):
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "function": {"name": "lookup", "arguments": '{"query":'},
                    }]
                },
                "finish_reason": "tool_calls",
            }]
        }

    monkeypatch.setattr(llm, "iter_sse_json_events", fake_sse)

    with pytest.raises(RuntimeError, match="Malformed streamed arguments"):
        await llm.stream_chat_completion(
            provider="openai_compatible",
            model="local-model",
            endpoint="http://local.test/v1",
            api_key=None,
            messages=[{"role": "user", "content": "look it up"}],
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_responses_stream_does_not_replay_after_visible_delta(monkeypatch):
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)
    calls = 0
    delivered: list[str] = []

    text_event = MagicMock()
    text_event.type = "response.output_text.delta"
    text_event.delta = "partial"

    class FailingStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def __aiter__(self):
            yield text_event
            raise ConnectionError("responses stream disconnected")

    def fake_stream(**_kwargs):
        nonlocal calls
        calls += 1
        return FailingStream()

    mock_client = MagicMock()
    mock_client.responses.stream = fake_stream
    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **_kwargs: mock_client)

    with pytest.raises(ConnectionError, match="responses stream disconnected"):
        await llm.stream_chat_completion(
            provider="openai",
            model="gpt-4o",
            endpoint=None,
            api_key="test",
            messages=[{"role": "user", "content": "hi"}],
            on_text_delta=delivered.append,
        )

    assert calls == 1
    assert delivered == ["partial"]


@pytest.mark.asyncio(loop_scope="session")
async def test_responses_stream_does_not_replay_after_reasoning_delivery(monkeypatch):
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)
    calls = 0
    reasoning_deltas: list[str] = []

    reasoning_event = MagicMock()
    reasoning_event.type = "response.reasoning_summary_text.delta"
    reasoning_event.delta = "private thought"

    class FailingStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def __aiter__(self):
            yield reasoning_event
            raise ConnectionError("responses stream disconnected")

    def fake_stream(**_kwargs):
        nonlocal calls
        calls += 1
        return FailingStream()

    mock_client = MagicMock()
    mock_client.responses.stream = fake_stream
    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **_kwargs: mock_client)

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(ConnectionError, match="responses stream disconnected"):
            await llm.stream_chat_completion(
                provider="openai",
                model="gpt-4o",
                endpoint=None,
                api_key="test",
                messages=[{"role": "user", "content": "hi"}],
                on_reasoning_delta=reasoning_deltas.append,
            )

    assert calls == 1
    assert reasoning_deltas == ["private thought"]


@pytest.mark.asyncio(loop_scope="session")
async def test_stream_completion_fallback_on_404(monkeypatch):
    """Streaming path falls back to Chat Completions on 404."""
    monkeypatch.setattr(llm, "_HAS_RESPONSES_API", True)

    not_found = llm.openai.NotFoundError.__new__(llm.openai.NotFoundError)
    mock_client = MagicMock()
    mock_client.responses.stream = MagicMock(side_effect=not_found)

    # Chat Completions streaming fallback
    delta_obj = MagicMock()
    delta_obj.content = "fallback"
    delta_obj.tool_calls = None
    choice = MagicMock()
    choice.delta = delta_obj
    event = MagicMock()
    event.choices = [choice]

    async def fake_stream(**kw):
        async def gen():
            yield event
        return gen()

    mock_client.chat.completions.create = fake_stream
    monkeypatch.setattr(llm.openai, "AsyncOpenAI", lambda **kw: mock_client)

    result = await llm.stream_chat_completion(
        provider="openai",
        model="gpt-4o",
        endpoint=None,
        api_key="test",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result["content"] == "fallback"
    assert llm._endpoint_responses_support.get("default") is False


def test_strip_leading_divider_removes_opening_rule():
    assert llm.strip_leading_divider("---\nHere is the reply.") == "Here is the reply."
    assert llm.strip_leading_divider("***\nHere is the reply.") == "Here is the reply."
    assert llm.strip_leading_divider("___\nHere is the reply.") == "Here is the reply."
    assert llm.strip_leading_divider("- - -\nHere is the reply.") == "Here is the reply."


def test_strip_leading_divider_removes_stacked_rules_and_whitespace():
    assert llm.strip_leading_divider("\n---\n\n---\n\nReply.") == "Reply."


def test_strip_leading_divider_keeps_internal_rules():
    content = "First section.\n\n---\n\nSecond section."
    assert llm.strip_leading_divider(content) == content


def test_strip_leading_divider_is_a_no_op_on_clean_content():
    for content in ("", "Plain reply.", "# Heading\n\nBody.", "-- not a rule"):
        assert llm.strip_leading_divider(content) == content


def test_strip_leading_divider_returns_empty_when_output_was_only_a_rule():
    assert llm.strip_leading_divider("---") == ""
    assert llm.strip_leading_divider("\n---\n") == ""


def test_strip_leading_divider_does_not_eat_a_list_item():
    content = "- first item\n- second item"
    assert llm.strip_leading_divider(content) == content
