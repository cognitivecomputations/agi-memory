"""Low-latency listener for the durable belief-update stream.

Postgres owns classification, retention, and the replayable log. LISTEN only
reduces latency; every received event also gets a database delivery receipt so
the worker path is observable rather than an in-memory best effort.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_NOTIFY_CHANNEL = "belief_updates"
BeliefHandler = Callable[[dict[str, Any]], Awaitable[None]]


def _json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


class BeliefUpdateListener:
    def __init__(self, pool: Any, *, worker_id: str) -> None:
        if pool is None:
            raise ValueError("a database pool is required for belief propagation")
        normalized = str(worker_id or "").strip()
        if not normalized:
            raise ValueError("worker_id is required for belief propagation")
        self.pool = pool
        self.worker_id = normalized
        self._handlers: list[BeliefHandler] = []
        self._conn: Any | None = None
        self._channel = DEFAULT_NOTIFY_CHANNEL
        self._started = False
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def channel(self) -> str:
        return self._channel

    def add_handler(self, handler: BeliefHandler) -> None:
        if handler not in self._handlers:
            self._handlers.append(handler)

    async def _settings(self) -> tuple[str, bool]:
        async with self.pool.acquire() as conn:
            raw_channel, raw_subscribers = await conn.fetchrow("""
                SELECT
                    COALESCE(get_config_text('belief.propagation_notify_channel'), 'belief_updates'),
                    COALESCE(get_config('belief.propagation_subscribers'), '["heartbeat"]'::jsonb)
                """)
        channel = str(raw_channel or DEFAULT_NOTIFY_CHANNEL).strip()
        if not channel:
            channel = DEFAULT_NOTIFY_CHANNEL
        subscribers = _json(raw_subscribers)
        subscribed = not isinstance(subscribers, list) or (
            self.worker_id in subscribers or "all" in subscribers
        )
        return channel, subscribed

    def _on_notify(
        self, _connection: Any, _pid: int, _channel: str, payload: str
    ) -> None:
        parsed = _json(payload)
        event = parsed if isinstance(parsed, dict) else {"raw": str(parsed)}
        task = asyncio.create_task(self._dispatch(event))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _dispatch(self, event: dict[str, Any]) -> None:
        log_id = event.get("log_id")
        if log_id is not None:
            try:
                async with self.pool.acquire() as conn:
                    await conn.fetchval(
                        "SELECT record_belief_update_delivery($1::bigint, $2, $3::jsonb)",
                        int(log_id),
                        self.worker_id,
                        json.dumps({"channel": self._channel}),
                    )
            except Exception:
                logger.warning(
                    "Belief update %s reached %s but its delivery receipt failed",
                    log_id,
                    self.worker_id,
                    exc_info=True,
                )

        results = await asyncio.gather(
            *(handler(event) for handler in list(self._handlers)),
            return_exceptions=True,
        )
        for handler, result in zip(self._handlers, results, strict=False):
            if isinstance(result, BaseException):
                logger.error(
                    "Belief update handler %r failed for log %s",
                    handler,
                    log_id,
                    exc_info=(type(result), result, result.__traceback__),
                )

    async def start(self) -> bool:
        if self._started:
            return True
        try:
            self._channel, subscribed = await self._settings()
            if not subscribed:
                logger.info(
                    "Worker %s is not configured as a belief-update subscriber",
                    self.worker_id,
                )
                return False
            self._conn = await self.pool.acquire()
            await self._conn.add_listener(self._channel, self._on_notify)
            self._started = True
            logger.info(
                "Worker %s is listening for belief updates on %s",
                self.worker_id,
                self._channel,
            )
            return True
        except Exception:
            logger.warning(
                "Worker %s could not attach the belief-update listener; durable replay remains available",
                self.worker_id,
                exc_info=True,
            )
            if self._conn is not None:
                try:
                    await self.pool.release(self._conn)
                except Exception:
                    logger.warning(
                        "Failed to release belief listener connection", exc_info=True
                    )
                self._conn = None
            return False

    async def stop(self) -> None:
        connection = self._conn
        self._conn = None
        self._started = False
        if connection is not None:
            try:
                await connection.remove_listener(self._channel, self._on_notify)
            except Exception:
                logger.warning("Failed to detach belief-update listener", exc_info=True)
            finally:
                await self.pool.release(connection)
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._tasks.clear()


__all__ = ["BeliefHandler", "BeliefUpdateListener", "DEFAULT_NOTIFY_CHANNEL"]
