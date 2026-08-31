from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


RABBITMQ_MANAGEMENT_URL = os.getenv(
    "RABBITMQ_MANAGEMENT_URL",
    f"http://localhost:{os.getenv('RABBITMQ_MANAGEMENT_PORT', '45673')}",
).rstrip("/")
RABBITMQ_USER = os.getenv("RABBITMQ_USER") or os.getenv(
    "RABBITMQ_DEFAULT_USER", "hexis"
)
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD") or os.getenv(
    "RABBITMQ_DEFAULT_PASS", "hexis_password"
)
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_OUTBOX_QUEUE = os.getenv("RABBITMQ_OUTBOX_QUEUE", "hexis.outbox")
RABBITMQ_INBOX_QUEUE = os.getenv("RABBITMQ_INBOX_QUEUE", "hexis.inbox")
RABBITMQ_POLL_INBOX_EVERY = float(os.getenv("RABBITMQ_POLL_INBOX_EVERY", 1.0))


class RabbitMQBridge:
    def __init__(self, pool):
        self.pool = pool
        self._last_inbox_poll = 0.0

    def _vhost_path(self) -> str:
        if RABBITMQ_VHOST == "/":
            return "%2F"
        return requests.utils.quote(RABBITMQ_VHOST, safe="")

    async def _request(
        self, method: str, path: str, payload: dict | None = None
    ) -> requests.Response:
        url = f"{RABBITMQ_MANAGEMENT_URL}{path}"
        auth = (RABBITMQ_USER, RABBITMQ_PASSWORD)

        def _do() -> requests.Response:
            return requests.request(method, url, auth=auth, json=payload, timeout=5)

        return await asyncio.to_thread(_do)

    async def ensure_ready(self) -> None:
        try:
            resp = await self._request("GET", "/api/overview")
            if resp.status_code != 200:
                raise RuntimeError(f"rabbitmq overview HTTP {resp.status_code}")

            vhost = self._vhost_path()
            for q in (RABBITMQ_OUTBOX_QUEUE, RABBITMQ_INBOX_QUEUE):
                r = await self._request(
                    "PUT",
                    f"/api/queues/{vhost}/{requests.utils.quote(q, safe='')}",
                    payload={"durable": True, "auto_delete": False, "arguments": {}},
                )
                if r.status_code not in (200, 201, 204):
                    raise RuntimeError(
                        f"rabbitmq queue declare {q!r} HTTP {r.status_code}: {r.text[:200]}"
                    )
        except Exception as e:
            logger.warning("RabbitMQ ensure_ready failed: %s", e)
            return

    async def publish_outbox_payloads(self, payloads: list[dict[str, Any]]) -> int:
        published = 0
        vhost = self._vhost_path()
        for msg in payloads or []:
            kind = msg.get("kind")
            payload = msg.get("payload")
            msg_id = msg.get("message_id") or msg.get("id")
            body = {"id": msg_id, "kind": kind, "payload": payload}
            if msg.get("delivery") is not None:
                body["delivery"] = msg.get("delivery")
            if msg.get("task_name") is not None:
                body["task_name"] = msg.get("task_name")
            try:
                resp = await self._request(
                    "POST",
                    f"/api/exchanges/{vhost}/amq.default/publish",
                    payload={
                        "properties": {"content_type": "application/json"},
                        "routing_key": RABBITMQ_OUTBOX_QUEUE,
                        "payload": json.dumps(body, default=str),
                        "payload_encoding": "string",
                    },
                )
                ok = resp.status_code == 200 and bool(resp.json().get("routed"))
                if not ok:
                    raise RuntimeError(
                        f"publish not routed: HTTP {resp.status_code} body={resp.text[:200]}"
                    )
                published += 1
            except Exception as e:
                logger.warning("Failed to publish outbox message: %s", e)
                return published

        return published

    async def publish_inbox_payload(self, body: dict[str, Any]) -> bool:
        """Publish one durable user-to-agent message to Hexis's inbox.

        Unlike the best-effort outbox batch publisher, this method lets
        transport failures propagate so an interactive caller can tell the
        user that their message was not queued.
        """
        vhost = self._vhost_path()
        message_id = body.get("id") or body.get("message_id")
        resp = await self._request(
            "POST",
            f"/api/exchanges/{vhost}/amq.default/publish",
            payload={
                "properties": {
                    "content_type": "application/json",
                    "delivery_mode": 2,
                    **({"message_id": str(message_id)} if message_id else {}),
                },
                "routing_key": RABBITMQ_INBOX_QUEUE,
                "payload": json.dumps(body, default=str),
                "payload_encoding": "string",
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"inbox publish HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            routed = bool(resp.json().get("routed"))
        except Exception as exc:
            raise RuntimeError(
                f"inbox publish returned invalid JSON: {resp.text[:200]}"
            ) from exc
        if not routed:
            raise RuntimeError(
                f"inbox message was not routed to {RABBITMQ_INBOX_QUEUE!r}"
            )
        return True

    async def poll_inbox_messages(self, max_messages: int = 10) -> int:
        if not self.pool:
            return 0

        now = time.monotonic()
        if now - self._last_inbox_poll < RABBITMQ_POLL_INBOX_EVERY:
            return 0
        self._last_inbox_poll = now

        vhost = self._vhost_path()
        try:
            resp = await self._request(
                "POST",
                f"/api/queues/{vhost}/{requests.utils.quote(RABBITMQ_INBOX_QUEUE, safe='')}/get",
                payload={
                    "count": max_messages,
                    "ackmode": "ack_requeue_false",
                    "encoding": "auto",
                    "truncate": 50000,
                },
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"inbox get HTTP {resp.status_code}: {resp.text[:200]}"
                )
            msgs = resp.json()
            if not isinstance(msgs, list):
                return 0
        except Exception as e:
            logger.warning("Failed to poll inbox messages: %s", e)
            return 0

        ingested = 0
        for msg in msgs:
            payload = msg.get("payload")
            content: Any = payload
            try:
                parsed = json.loads(payload) if isinstance(payload, str) else payload
                if isinstance(parsed, dict) and "content" in parsed:
                    content = parsed["content"]
                else:
                    content = parsed
            except Exception as e:
                logger.debug("Failed to parse inbox message payload: %s", e)

            try:
                async with self.pool.acquire() as conn:
                    await conn.fetchval(
                        "SELECT add_to_working_memory($1::text, INTERVAL '1 day')",
                        str(content),
                    )
                    await conn.execute("SELECT mark_user_contact()")
                ingested += 1
            except Exception as e:
                logger.warning(
                    "Failed to ingest inbox message to working memory: %s", e
                )
                return ingested

        return ingested
