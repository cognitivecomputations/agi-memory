"""OpenAI-compatible HTTP journeys over the canonical Hexis agent stream."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from openai import AsyncOpenAI

import apps.hexis_api as web_module
from apps.hexis_api import app
from core.agent_loop import AgentEvent, AgentEventData

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


@pytest.fixture(scope="module")
async def http_client(db_pool):
    original_pool = web_module._pool
    web_module._pool = db_pool
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client
    web_module._pool = original_pool


async def _active_model_id(client: httpx.AsyncClient) -> str:
    response = await client.get("/v1/models")
    assert response.status_code == 200
    return response.json()["data"][0]["id"]


def _agent_stream(captured: dict):
    async def stream(*args, **kwargs):
        captured.update(kwargs)
        yield AgentEventData(event=AgentEvent.LOOP_START)
        yield AgentEventData(
            event=AgentEvent.TEXT_DELTA,
            data={"text": "Hello from "},
        )
        yield AgentEventData(
            event=AgentEvent.TEXT_DELTA,
            data={"text": "Hexis."},
        )
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "completed", "timed_out": False},
        )

    return stream


async def test_models_derive_from_live_chat_config(http_client, db_pool):
    async with db_pool.acquire() as conn:
        configured = await conn.fetchval("SELECT get_config('llm.chat')")
    if isinstance(configured, str):
        configured = json.loads(configured)

    response = await http_client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"] == [
        {
            "id": configured["model"],
            "object": "model",
            "owned_by": configured["provider"],
            "created": payload["data"][0]["created"],
            "x_hexis_config_key": "llm.chat",
        }
    ]
    assert isinstance(payload["data"][0]["created"], int)


async def test_openai_client_buffered_completion_preserves_history_and_controls(
    http_client,
):
    model_id = await _active_model_id(http_client)
    captured: dict = {}

    with patch.object(web_module, "stream_chat_events", _agent_stream(captured)):
        sdk = AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
        completion = await sdk.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "developer", "content": "Keep the answer concise."},
                {"role": "assistant", "content": "Understood."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Introduce yourself."}],
                },
            ],
            max_completion_tokens=123,
            temperature=0.2,
        )

    assert completion.object == "chat.completion"
    assert completion.model == model_id
    assert completion.choices[0].message.content == "Hello from Hexis."
    assert completion.choices[0].finish_reason == "stop"
    assert captured["user_message"] == "Introduce yourself."
    assert captured["history"] == [
        {"role": "system", "content": "Keep the answer concise."},
        {"role": "assistant", "content": "Understood."},
    ]
    assert captured["max_tokens"] == 123
    assert captured["temperature"] == 0.2
    assert captured["surface"] == "openai_compat"
    assert captured["gateway_source_id"] == f"chat:openai:{captured['session_id']}"


async def test_openai_client_streams_role_text_finish_and_done(http_client):
    model_id = await _active_model_id(http_client)
    captured: dict = {}

    with patch.object(web_module, "stream_chat_events", _agent_stream(captured)):
        sdk = AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
        stream = await sdk.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Stream this."}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == (
        "Hello from Hexis."
    )
    assert chunks[-1].choices[0].finish_reason == "stop"


async def test_openai_buffered_completion_rejects_error_stop_without_error_event(
    http_client,
):
    model_id = await _active_model_id(http_client)

    async def error_stream(*_args, **_kwargs):
        yield AgentEventData(event=AgentEvent.TEXT_DELTA, data={"text": "partial"})
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "error", "timed_out": False},
        )

    with patch.object(web_module, "stream_chat_events", error_stream):
        response = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Fail safely."}],
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_agent_error"
    assert response.json()["error"]["message"] == "Agent loop ended with an error."


async def test_openai_stream_reports_error_stop_without_normal_finish(http_client):
    model_id = await _active_model_id(http_client)

    async def error_stream(*_args, **_kwargs):
        yield AgentEventData(event=AgentEvent.TEXT_DELTA, data={"text": "partial"})
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "error", "timed_out": False},
        )

    with patch.object(web_module, "stream_chat_events", error_stream):
        response = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Fail safely."}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert '"code":"upstream_agent_error"' in response.text
    assert '"finish_reason":"stop"' not in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize(
    ("body_update", "status", "code"),
    [
        ({"model": "not-the-active-model"}, 404, "model_not_found"),
        ({"tools": []}, 400, "unsupported_parameter"),
        (
            {"messages": [{"role": "assistant", "content": "done"}]},
            400,
            "unsupported_parameter",
        ),
        ({"stream_options": {"include_usage": True}}, 400, "unsupported_parameter"),
    ],
)
async def test_openai_errors_are_structured_and_actionable(
    http_client, body_update, status, code
):
    model_id = await _active_model_id(http_client)
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    body.update(body_update)

    response = await http_client.post("/v1/chat/completions", json=body)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["message"]
