from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from core.integration_reliability import (
    IntegrationHttpError,
    iter_sse_json_events,
    parse_retry_after,
    request_bytes_response,
    request_json_response,
)


def test_parse_retry_after_seconds_and_http_date():
    assert parse_retry_after("3") == 3.0
    now = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Thu, 23 Jul 2026 12:00:05 GMT", now=now) == 5.0


@pytest.mark.asyncio(loop_scope="session")
async def test_request_json_retries_safe_methods_with_retry_after():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "2"}, text="try later")
        return httpx.Response(200, json={"ok": True})

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    response = await request_json_response(
        "test_provider",
        "GET",
        "https://example.test/resource",
        transport=httpx.MockTransport(handler),
        attempts=3,
        sleep=sleep,
    )

    assert response.json_data == {"ok": True}
    assert calls == 2
    assert sleeps == [2.0]


@pytest.mark.asyncio(loop_scope="session")
async def test_request_json_does_not_retry_unsafe_post_by_default():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="created maybe")

    with pytest.raises(IntegrationHttpError) as raised:
        await request_json_response(
            "test_provider",
            "POST",
            "https://example.test/send",
            json_body={"message": "hello"},
            transport=httpx.MockTransport(handler),
            attempts=3,
        )

    assert calls == 1
    assert raised.value.error_kind == "provider_transient"
    assert raised.value.status_code == 503


@pytest.mark.asyncio(loop_scope="session")
async def test_request_json_bounds_provider_error_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="x" * 12_000)

    with pytest.raises(IntegrationHttpError) as raised:
        await request_json_response(
            "test_provider",
            "GET",
            "https://example.test/nope",
            transport=httpx.MockTransport(handler),
            attempts=1,
        )

    assert len(raised.value.response_body) < 8_100
    assert "[truncated]" in raised.value.response_body


@pytest.mark.asyncio(loop_scope="session")
async def test_request_bytes_enforces_max_size():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 12)

    with pytest.raises(IntegrationHttpError) as raised:
        await request_bytes_response(
            "test_provider",
            "GET",
            "https://example.test/file",
            transport=httpx.MockTransport(handler),
            max_bytes=8,
        )

    assert raised.value.error_kind == "response_too_large"


@pytest.mark.asyncio(loop_scope="session")
async def test_iter_sse_json_retries_before_first_event():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "1"}, text="warmup")
        return httpx.Response(200, text='data: {"ok": true}\n\n')

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    events = [
        event
        async for event in iter_sse_json_events(
            "test_provider",
            "POST",
            "https://example.test/stream",
            json_body={"prompt": "hello"},
            transport=httpx.MockTransport(handler),
            attempts=3,
            sleep=sleep,
        )
    ]

    assert events == [{"ok": True}]
    assert calls == 2
    assert sleeps == [1.0]


class _FailAfterChunk(httpx.AsyncByteStream):
    def __init__(self, chunk: bytes) -> None:
        self._chunk = chunk

    async def __aiter__(self):
        yield self._chunk
        raise httpx.RemoteProtocolError("stream interrupted")


@pytest.mark.asyncio(loop_scope="session")
async def test_iter_sse_json_does_not_retry_after_emitted_event():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            stream=_FailAfterChunk(b'data: {"partial": true}\n\n'),
        )

    events: list[dict] = []
    with pytest.raises(IntegrationHttpError) as raised:
        async for event in iter_sse_json_events(
            "test_provider",
            "POST",
            "https://example.test/stream",
            json_body={"prompt": "hello"},
            transport=httpx.MockTransport(handler),
            attempts=3,
        ):
            events.append(event)

    assert events == [{"partial": True}]
    assert calls == 1
    assert raised.value.error_kind == "network"


@pytest.mark.asyncio(loop_scope="session")
async def test_iter_sse_json_strict_mode_rejects_malformed_payload():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: {not-json}\n\n")

    with pytest.raises(ValueError, match="Malformed SSE JSON"):
        _ = [
            event
            async for event in iter_sse_json_events(
                "test_provider",
                "POST",
                "https://example.test/stream",
                transport=httpx.MockTransport(handler),
                attempts=1,
                strict_json=True,
            )
        ]


@pytest.mark.asyncio(loop_scope="session")
async def test_iter_sse_json_non_strict_skips_malformed_payload():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='data: {not-json}\n\ndata: {"ok": true}\n\n',
        )

    events = [
        event
        async for event in iter_sse_json_events(
            "test_provider",
            "POST",
            "https://example.test/stream",
            transport=httpx.MockTransport(handler),
            attempts=1,
        )
    ]

    assert events == [{"ok": True}]


@pytest.mark.asyncio(loop_scope="session")
async def test_iter_sse_json_strict_mode_rejects_malformed_trailing_payload():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: {not-json}")

    with pytest.raises(ValueError, match="Malformed SSE JSON"):
        _ = [
            event
            async for event in iter_sse_json_events(
                "test_provider",
                "POST",
                "https://example.test/stream",
                transport=httpx.MockTransport(handler),
                attempts=1,
                strict_json=True,
            )
        ]
