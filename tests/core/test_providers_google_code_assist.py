"""Tests for core.providers.google_code_assist — Cloud Code Assist SSE client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.providers.google_code_assist import (
    _build_headers,
    _build_request_body,
    _build_url,
    _convert_messages,
    _convert_tools,
    _parse_candidates,
    google_code_assist_completion,
)

pytestmark = pytest.mark.core


def test_convert_messages_basic():
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    system, contents = _convert_messages(messages)
    assert system == "Be helpful."
    assert len(contents) == 2
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Hello"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "Hi there!"


def test_convert_messages_multimodal_user_content():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
            ],
        },
    ]
    system, contents = _convert_messages(messages)
    assert system is None
    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "What is in this image?"},
                {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
            ],
        }
    ]


def test_convert_messages_tool_calls():
    messages = [
        {"role": "user", "content": "search for X"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "tc_1",
                "function": {"name": "recall", "arguments": '{"query": "X"}'},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_1",
            "name": "recall",
            "content": '{"result": "found it"}',
        },
    ]
    system, contents = _convert_messages(messages)
    assert system is None
    assert len(contents) == 3
    # Assistant with tool call
    assert "functionCall" in contents[1]["parts"][0]
    assert contents[1]["parts"][0]["functionCall"]["name"] == "recall"
    # Tool result
    assert "functionResponse" in contents[2]["parts"][0]


def test_convert_tools_basic():
    tools = [{
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search memory",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }]
    result = _convert_tools(tools)
    assert result is not None
    assert len(result) == 1
    decls = result[0]["functionDeclarations"]
    assert len(decls) == 1
    assert decls[0]["name"] == "recall"


def test_convert_tools_empty():
    assert _convert_tools(None) is None
    assert _convert_tools([]) is None


def test_request_body_preserves_stable_prefix_and_volatile_tail():
    body = _build_request_body(
        [
            {"role": "system", "content": "STABLE"},
            {"role": "system", "content": "VOLATILE"},
            {"role": "user", "content": "hello"},
        ],
        None,
        system_prompt=None,
        project_id="project",
    )

    assert body["systemInstruction"] == {"parts": [{"text": "STABLE\n\nVOLATILE"}]}


def test_parse_candidates_text():
    data = {
        "candidates": [{
            "content": {
                "parts": [{"text": "Hello "}, {"text": "world"}],
            },
        }],
    }
    text, tool_calls = _parse_candidates(data)
    assert text == "Hello world"
    assert tool_calls == []


def test_parse_candidates_function_call():
    data = {
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "recall",
                        "args": {"query": "hello"},
                    },
                }],
            },
        }],
    }
    text, tool_calls = _parse_candidates(data)
    assert text == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "recall"
    assert tool_calls[0]["arguments"] == {"query": "hello"}
    assert tool_calls[0]["id"]  # Should have a generated UUID


def test_build_headers_gemini():
    h = _build_headers("token123", "proj-abc", is_antigravity=False)
    assert h["Authorization"] == "Bearer token123"
    assert h["X-Goog-User-Project"] == "proj-abc"
    assert "vscode" not in h["X-Goog-Api-Client"]


def test_build_headers_antigravity():
    h = _build_headers("token123", "proj-abc", is_antigravity=True)
    assert h["Authorization"] == "Bearer token123"
    assert "vscode" in h["X-Goog-Api-Client"]


def test_build_url_stream():
    url = _build_url("https://cloudcode-pa.googleapis.com", "gemini-2.5-flash", stream=True)
    assert "streamGenerateContent" in url
    assert "alt=sse" in url


def test_build_url_non_stream():
    url = _build_url("https://cloudcode-pa.googleapis.com", "gemini-2.5-flash", stream=False)
    assert "generateContent" in url
    assert "alt=sse" not in url


@pytest.mark.asyncio
async def test_google_code_assist_completion():
    """Test non-streaming completion with mocked httpx."""
    response_body = {
        "candidates": [{
            "content": {
                "parts": [{"text": "Response from Gemini"}],
            },
        }],
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_body

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("core.providers.google_code_assist.httpx.AsyncClient", return_value=mock_client):
        result = await google_code_assist_completion(
            endpoint="https://cloudcode-pa.googleapis.com",
            access_token="token123",
            project_id="proj-abc",
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello"}],
            tools=None,
        )

    assert result["content"] == "Response from Gemini"
    assert result["tool_calls"] == []
