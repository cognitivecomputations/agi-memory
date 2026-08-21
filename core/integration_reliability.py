"""Shared reliability primitives for external integrations.

Reference projects handle provider I/O as a substrate concern rather than
letting every connector invent timeout, retry, rate-limit, and error-body
behavior. This module gives Hexis the same central contract for HTTP-backed
integrations.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx

MAX_ERROR_BODY_CHARS = 8_000
DEFAULT_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
SAFE_RETRY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@dataclass(frozen=True)
class IntegrationHttpResponse:
    """Successful provider response with loaded body and normalized headers."""

    status_code: int
    headers: dict[str, str]
    text: str
    json_data: Any
    correlation_id: str
    content: bytes = b""


class IntegrationHttpError(RuntimeError):
    """Structured provider failure.

    The human-readable message is safe to surface in logs/UI; raw provider
    bodies are bounded so an integration failure cannot dump megabytes into the
    activity pane.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        error_kind: str,
        status_code: int | None = None,
        response_body: str = "",
        retry_after_seconds: float | None = None,
        attempts: int = 1,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.error_kind = error_kind
        self.status_code = status_code
        self.response_body = response_body
        self.retry_after_seconds = retry_after_seconds
        self.attempts = attempts
        self.correlation_id = correlation_id

    @property
    def transient(self) -> bool:
        return self.error_kind in {"timeout", "network", "rate_limited", "provider_transient"}


def bounded_text(value: Any, *, limit: int = MAX_ERROR_BODY_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated]"


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse RFC 9110 Retry-After seconds or HTTP-date into seconds."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = float(text)
        if seconds >= 0:
            return seconds
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (target.astimezone(timezone.utc) - reference).total_seconds())


def parse_rate_limit_reset(headers: Mapping[str, str], *, now: datetime | None = None) -> float | None:
    raw = _header(headers, "x-rate-limit-reset")
    if not raw:
        return None
    try:
        reset = float(raw)
    except ValueError:
        return None
    if reset > 10_000_000_000:
        reset = reset / 1000.0
    reference = now or datetime.now(timezone.utc)
    return max(0.0, reset - reference.timestamp())


def classify_http_status(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request"
    if status_code == 401:
        return "auth_failed"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 429:
        return "rate_limited"
    if status_code in DEFAULT_RETRY_STATUSES or status_code >= 500:
        return "provider_transient"
    return "provider_error"


def compute_backoff_seconds(
    attempt: int,
    *,
    initial_delay: float = 0.5,
    max_delay: float = 30.0,
    factor: float = 2.0,
    jitter: float = 0.2,
    random_fn: Callable[[], float] | None = None,
) -> float:
    base = min(float(max_delay), float(initial_delay) * (float(factor) ** max(int(attempt) - 1, 0)))
    if jitter <= 0:
        return max(0.0, base)
    rand = random_fn or random.random
    spread = max(0.0, min(float(jitter), 1.0))
    multiplier = 1.0 + ((rand() * 2.0 - 1.0) * spread)
    return max(0.0, min(float(max_delay), base * multiplier))


async def request_json_response(
    provider: str,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    data: Any | None = None,
    json_body: Any | None = None,
    timeout: float | httpx.Timeout = 30.0,
    attempts: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_statuses: set[int] | frozenset[int] = DEFAULT_RETRY_STATUSES,
    retry_unsafe_methods: bool = False,
    follow_redirects: bool = False,
    correlation_id: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> IntegrationHttpResponse:
    response = await _request_response(
        provider,
        method,
        url,
        headers=headers,
        params=params,
        data=data,
        json_body=json_body,
        timeout=timeout,
        attempts=attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        retry_statuses=retry_statuses,
        retry_unsafe_methods=retry_unsafe_methods,
        follow_redirects=follow_redirects,
        correlation_id=correlation_id,
        transport=transport,
        sleep=sleep,
    )
    if not response.text:
        object.__setattr__(response, "json_data", {})
        return response
    try:
        data = response.json_data if response.json_data is not None else _loads_json(response.text)
    except ValueError as exc:
        raise IntegrationHttpError(
            f"{provider} returned invalid JSON.",
            provider=provider,
            error_kind="invalid_response",
            status_code=response.status_code,
            response_body=bounded_text(response.text),
            attempts=1,
            correlation_id=response.correlation_id,
        ) from exc
    object.__setattr__(response, "json_data", data)
    return response


async def request_json(
    provider: str,
    method: str,
    url: str,
    **kwargs: Any,
) -> Any:
    return (await request_json_response(provider, method, url, **kwargs)).json_data


async def request_text_response(
    provider: str,
    method: str,
    url: str,
    **kwargs: Any,
) -> IntegrationHttpResponse:
    return await _request_response(provider, method, url, **kwargs)


async def request_bytes_response(
    provider: str,
    method: str,
    url: str,
    *,
    max_bytes: int | None = None,
    **kwargs: Any,
) -> IntegrationHttpResponse:
    response = await _request_response(
        provider,
        method,
        url,
        max_response_bytes=max_bytes,
        **kwargs,
    )
    if max_bytes is not None and len(response.content) > max_bytes:
        raise IntegrationHttpError(
            f"{provider} response exceeded maximum size.",
            provider=provider,
            error_kind="response_too_large",
            status_code=response.status_code,
            response_body=f"{len(response.content)} bytes exceeds {int(max_bytes)} bytes",
            attempts=1,
            correlation_id=response.correlation_id,
        )
    return response


async def iter_sse_json_events(
    provider: str,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    data: Any | None = None,
    json_body: Any | None = None,
    timeout: float | httpx.Timeout = 30.0,
    attempts: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_statuses: set[int] | frozenset[int] = DEFAULT_RETRY_STATUSES,
    retry_unsafe_methods: bool = True,
    follow_redirects: bool = False,
    correlation_id: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    strict_json: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Yield JSON objects from a server-sent-event stream.

    Streaming retries are deliberately narrow: retry before any event is
    emitted, never after. Once the caller may have shown text or acted on a
    tool delta, replaying the POST could duplicate visible output.
    """

    method_upper = method.upper()
    cid = (correlation_id or f"hexis-{uuid4().hex[:16]}").strip()
    request_headers = dict(headers or {})
    request_headers.setdefault("X-Hexis-Correlation-Id", cid)
    max_attempts = max(1, int(attempts or 1))
    can_retry_method = retry_unsafe_methods or method_upper in SAFE_RETRY_METHODS

    for attempt in range(1, max_attempts + 1):
        emitted = False
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=follow_redirects,
                transport=transport,
            ) as client:
                async with client.stream(
                    method_upper,
                    url,
                    headers=request_headers,
                    params=params,
                    data=data,
                    json=json_body,
                ) as response:
                    normalized_headers = {
                        str(k).lower(): str(v) for k, v in response.headers.items()
                    }
                    if response.status_code < 200 or response.status_code >= 300:
                        body = await _read_response_bytes_limited(
                            response,
                            MAX_ERROR_BODY_CHARS * 4,
                        )
                        retry_after = parse_retry_after(_header(normalized_headers, "retry-after"))
                        if retry_after is None:
                            retry_after = parse_rate_limit_reset(normalized_headers)
                        error = IntegrationHttpError(
                            f"{provider} HTTP {response.status_code} ({classify_http_status(response.status_code)}) calling {url}",
                            provider=provider,
                            error_kind=classify_http_status(response.status_code),
                            status_code=response.status_code,
                            response_body=bounded_text(body.decode("utf-8", errors="replace")),
                            retry_after_seconds=retry_after,
                            attempts=attempt,
                            correlation_id=cid,
                        )
                        if (
                            can_retry_method
                            and response.status_code in retry_statuses
                            and attempt < max_attempts
                        ):
                            await sleep(
                                min(
                                    max_delay,
                                    retry_after
                                    if retry_after is not None
                                    else compute_backoff_seconds(
                                        attempt,
                                        initial_delay=initial_delay,
                                        max_delay=max_delay,
                                    ),
                                )
                            )
                            continue
                        raise error

                    data_lines: list[str] = []
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data_lines.append(line[5:].strip())
                            continue
                        if line.strip() != "":
                            continue
                        obj = _parse_sse_json_payload(data_lines, strict=strict_json)
                        data_lines = []
                        if obj is None:
                            continue
                        emitted = True
                        yield obj

                    obj = _parse_sse_json_payload(data_lines, strict=strict_json)
                    if obj is not None:
                        emitted = True
                        yield obj
                    return
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            if emitted or not can_retry_method or attempt >= max_attempts:
                error_kind = "timeout" if isinstance(exc, httpx.TimeoutException) else "network"
                raise IntegrationHttpError(
                    f"{provider} {error_kind} calling {url}: {exc}",
                    provider=provider,
                    error_kind=error_kind,
                    response_body=bounded_text(exc),
                    attempts=attempt,
                    correlation_id=cid,
                ) from exc
            await sleep(
                compute_backoff_seconds(
                    attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                )
            )


def format_provider_error(provider_label: str, exc: IntegrationHttpError) -> str:
    bits = [f"{provider_label} request failed"]
    if exc.status_code is not None:
        bits.append(f"HTTP {exc.status_code}")
    bits.append(exc.error_kind.replace("_", " "))
    if exc.retry_after_seconds is not None:
        bits.append(f"retry after {exc.retry_after_seconds:.0f}s")
    if exc.response_body:
        bits.append(exc.response_body)
    if exc.correlation_id:
        bits.append(f"correlation_id={exc.correlation_id}")
    return ": ".join(bits)


async def _request_response(
    provider: str,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    data: Any | None = None,
    json_body: Any | None = None,
    timeout: float | httpx.Timeout = 30.0,
    attempts: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_statuses: set[int] | frozenset[int] = DEFAULT_RETRY_STATUSES,
    retry_unsafe_methods: bool = False,
    follow_redirects: bool = False,
    correlation_id: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_response_bytes: int | None = None,
) -> IntegrationHttpResponse:
    method_upper = method.upper()
    cid = (correlation_id or f"hexis-{uuid4().hex[:16]}").strip()
    request_headers = dict(headers or {})
    request_headers.setdefault("X-Hexis-Correlation-Id", cid)
    max_attempts = max(1, int(attempts or 1))
    can_retry_method = retry_unsafe_methods or method_upper in SAFE_RETRY_METHODS

    last_error: IntegrationHttpError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=follow_redirects,
                transport=transport,
            ) as client:
                response = await _execute_request(
                    client,
                    provider,
                    method_upper,
                    url,
                    headers=request_headers,
                    params=params,
                    data=data,
                    json_body=json_body,
                    max_response_bytes=max_response_bytes,
                    attempt=attempt,
                    correlation_id=cid,
                )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            error_kind = "timeout" if isinstance(exc, httpx.TimeoutException) else "network"
            last_error = IntegrationHttpError(
                f"{provider} {error_kind} calling {url}: {exc}",
                provider=provider,
                error_kind=error_kind,
                response_body=bounded_text(exc),
                attempts=attempt,
                correlation_id=cid,
            )
            if can_retry_method and attempt < max_attempts:
                await sleep(compute_backoff_seconds(attempt, initial_delay=initial_delay, max_delay=max_delay))
                continue
            raise last_error from exc

        normalized_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
        text = bounded_text(response.text)
        if 200 <= response.status_code < 300:
            return IntegrationHttpResponse(
                status_code=response.status_code,
                headers=normalized_headers,
                text=response.text,
                json_data=None,
                correlation_id=cid,
                content=response.content,
            )

        retry_after = parse_retry_after(_header(normalized_headers, "retry-after"))
        if retry_after is None:
            retry_after = parse_rate_limit_reset(normalized_headers)
        error_kind = classify_http_status(response.status_code)
        last_error = IntegrationHttpError(
            f"{provider} HTTP {response.status_code} ({error_kind}) calling {url}",
            provider=provider,
            error_kind=error_kind,
            status_code=response.status_code,
            response_body=text,
            retry_after_seconds=retry_after,
            attempts=attempt,
            correlation_id=cid,
        )
        if (
            can_retry_method
            and response.status_code in retry_statuses
            and attempt < max_attempts
        ):
            delay = retry_after
            if delay is None:
                delay = compute_backoff_seconds(
                    attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                )
            else:
                delay = min(max(0.0, delay), max_delay)
            await sleep(delay)
            continue
        raise last_error

    if last_error is not None:
        raise last_error
    raise IntegrationHttpError(
        f"{provider} request failed before execution.",
        provider=provider,
        error_kind="unknown",
        attempts=max_attempts,
        correlation_id=cid,
    )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return headers.get(name.lower()) or headers.get(name) or None


def _loads_json(text: str) -> Any:
    import json

    return json.loads(text)


async def _execute_request(
    client: httpx.AsyncClient,
    provider: str,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    params: Mapping[str, Any] | None,
    data: Any | None,
    json_body: Any | None,
    max_response_bytes: int | None,
    attempt: int,
    correlation_id: str,
) -> httpx.Response:
    if max_response_bytes is None:
        return await client.request(
            method,
            url,
            headers=headers,
            params=params,
            data=data,
            json=json_body,
        )

    limit = max(0, int(max_response_bytes))
    async with client.stream(
        method,
        url,
        headers=headers,
        params=params,
        data=data,
        json=json_body,
    ) as response:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > limit:
                raise IntegrationHttpError(
                    f"{provider} response exceeded maximum size.",
                    provider=provider,
                    error_kind="response_too_large",
                    status_code=response.status_code,
                    response_body=f"{total} bytes exceeds {limit} bytes",
                    attempts=attempt,
                    correlation_id=correlation_id,
                )
            chunks.append(chunk)
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=b"".join(chunks),
            request=response.request,
            extensions=response.extensions,
        )


async def _read_response_bytes_limited(
    response: httpx.Response,
    limit: int,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            chunks.append(chunk[: max(limit - (total - len(chunk)), 0)])
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_sse_json_payload(
    data_lines: list[str],
    *,
    strict: bool = False,
) -> dict[str, Any] | None:
    if not data_lines:
        return None
    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        return None
    try:
        obj = _loads_json(data)
    except ValueError:
        if strict:
            raise ValueError("Malformed SSE JSON payload.")
        return None
    return obj if isinstance(obj, dict) else None
