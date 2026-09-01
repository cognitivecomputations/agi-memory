"""Private, explicit Web Push delivery for installed Hexis clients.

The browser grants each subscription. VAPID key material lives in the Hexis
home directory (or an explicit file), never in Postgres. Delivery is advisory:
the durable web inbox remains authoritative if a push service is unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from core.config import hexis_home
from core.integration_reliability import bounded_text

logger = logging.getLogger(__name__)

_DEFAULT_VAPID_SUBJECT = "https://github.com/QuixiAI/Hexis"


@dataclass(frozen=True)
class PushAttempt:
    subscription_id: str
    delivered: bool
    status_code: int | None = None
    error: str | None = None


def vapid_private_key_path() -> Path:
    configured = str(os.getenv("HEXIS_WEB_PUSH_VAPID_PRIVATE_KEY_FILE") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return hexis_home() / "web-push-vapid-private.pem"


def ensure_vapid_keypair() -> tuple[Path, str]:
    """Return a stable VAPID private-key path and URL-safe public key.

    Call this only in response to the user's explicit notification setup.
    Concurrent first-use calls race safely through O_EXCL and never overwrite
    an existing key.
    """

    path = vapid_private_key_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        private_key = ec.generate_private_key(ec.SECP256R1())
        encoded = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as key_file:
                key_file.write(encoded)
    try:
        os.chmod(path, 0o600)
        private_key = serialization.load_pem_private_key(
            path.read_bytes(), password=None
        )
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Web Push key at {path} could not be read. Fix its permissions or "
            "set HEXIS_WEB_PUSH_VAPID_PRIVATE_KEY_FILE to a writable persistent path."
        ) from exc
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise RuntimeError(f"Web Push key at {path} is not an EC private key.")
    numbers = private_key.public_key().public_numbers()
    raw_public = b"\x04" + numbers.x.to_bytes(32, "big") + numbers.y.to_bytes(32, "big")
    public_key = base64.urlsafe_b64encode(raw_public).rstrip(b"=").decode("ascii")
    return path, public_key


def validate_push_endpoint(endpoint: str) -> str | None:
    """Return an actionable error for non-public push-service endpoints."""

    try:
        parsed = urlparse(endpoint)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            return "push endpoint must be a public HTTPS URL"
        if parsed.username or parsed.password:
            return "push endpoint must not contain URL credentials"
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or 443,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
        for _, _, _, _, address in addresses:
            ip = ipaddress.ip_address(address[0])
            if not ip.is_global:
                return f"push endpoint resolves to non-public address {ip}"
    except (OSError, ValueError) as exc:
        return f"push endpoint hostname could not be validated: {exc}"
    return None


async def deliver_web_push(pool: Any, body: dict[str, Any]) -> int:
    """Deliver one outbox envelope to every active browser subscription."""

    payload = body.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {"message": payload}
    if not isinstance(payload, dict):
        payload = {}
    content = str(payload.get("message") or payload.get("content") or "").strip()
    if not content:
        return 0

    async with pool.acquire() as conn:
        enabled = bool(
            await conn.fetchval(
                "SELECT COALESCE(get_config_bool('pwa.push.enabled'), TRUE)"
            )
        )
        if not enabled:
            return 0
        rows = await conn.fetch(
            """
            SELECT id, endpoint, p256dh, auth
            FROM web_push_subscriptions
            WHERE revoked_at IS NULL
              AND (expiration_time IS NULL OR expiration_time > $1)
            ORDER BY updated_at DESC
            """,
            int(time.time() * 1000),
        )
        if not rows:
            return 0
        subject = str(
            await conn.fetchval(
                "SELECT COALESCE(get_config_text('pwa.push.vapid_subject'), $1)",
                _DEFAULT_VAPID_SUBJECT,
            )
            or _DEFAULT_VAPID_SUBJECT
        ).strip()
        show_previews = bool(
            await conn.fetchval(
                "SELECT COALESCE(get_config_bool('pwa.push.show_message_previews'), FALSE)"
            )
        )
        profile = await conn.fetchval("SELECT get_agent_profile_context()")

    key_path, _ = ensure_vapid_keypair()
    agent_name = _agent_name(profile)
    push_payload = json.dumps(
        {
            "title": agent_name,
            "body": _notification_body(body, payload, content, show_previews),
            "tag": str(body.get("id") or payload.get("suggestion_id") or "hexis"),
            "url": "/chat?inbox=1",
            "kind": str(body.get("kind") or payload.get("intent") or "message"),
        },
        ensure_ascii=False,
    )

    attempts = await asyncio.gather(
        *[
            _deliver_one(
                row,
                data=push_payload,
                key_path=key_path,
                subject=subject,
            )
            for row in rows
        ]
    )
    await _record_attempts(pool, attempts)
    return sum(1 for attempt in attempts if attempt.delivered)


async def _deliver_one(
    row: Any,
    *,
    data: str,
    key_path: Path,
    subject: str,
) -> PushAttempt:
    subscription_id = str(row["id"])
    subscription = {
        "endpoint": str(row["endpoint"]),
        "keys": {"p256dh": str(row["p256dh"]), "auth": str(row["auth"])},
    }
    endpoint_error = await asyncio.to_thread(
        validate_push_endpoint, subscription["endpoint"]
    )
    if endpoint_error:
        return PushAttempt(
            subscription_id=subscription_id,
            delivered=False,
            status_code=410,
            error=endpoint_error,
        )

    def send() -> None:
        from pywebpush import webpush

        webpush(
            subscription_info=subscription,
            data=data,
            vapid_private_key=str(key_path),
            vapid_claims={"sub": subject},
            timeout=10,
        )

    try:
        await asyncio.to_thread(send)
        return PushAttempt(subscription_id=subscription_id, delivered=True)
    except Exception as exc:  # WebPushException is optional until send time.
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        detail = bounded_text(exc, limit=300) or type(exc).__name__
        logger.info("Web Push delivery failed (%s): %s", status or "network", detail)
        return PushAttempt(
            subscription_id=subscription_id,
            delivered=False,
            status_code=int(status) if isinstance(status, int) else None,
            error=detail,
        )


async def _record_attempts(pool: Any, attempts: list[PushAttempt]) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            for attempt in attempts:
                if attempt.delivered:
                    await conn.execute(
                        """
                        UPDATE web_push_subscriptions
                        SET failure_count = 0, last_error = NULL,
                            last_delivered_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1::uuid
                        """,
                        attempt.subscription_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE web_push_subscriptions
                        SET failure_count = failure_count + 1,
                            last_error = $2,
                            revoked_at = CASE WHEN $3::integer IN (404, 410)
                                              THEN CURRENT_TIMESTAMP ELSE revoked_at END,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1::uuid
                        """,
                        attempt.subscription_id,
                        attempt.error,
                        attempt.status_code,
                    )


def _agent_name(profile: Any) -> str:
    if isinstance(profile, str):
        try:
            profile = json.loads(profile)
        except json.JSONDecodeError:
            profile = {}
    if isinstance(profile, dict):
        name = profile.get("name")
        if not name and isinstance(profile.get("persona"), dict):
            name = profile["persona"].get("name")
        if str(name or "").strip():
            return str(name).strip()[:80]
    return "Hexis"


def _notification_body(
    envelope: dict[str, Any],
    payload: dict[str, Any],
    content: str,
    show_previews: bool,
) -> str:
    if show_previews:
        return bounded_text(content, limit=220)
    kind = str(envelope.get("kind") or payload.get("intent") or "").lower()
    if "automation" in kind:
        return "A new automation suggestion is ready for your decision."
    if "approval" in kind:
        return "A protected action is waiting for your decision."
    if "question" in kind or payload.get("requires_response"):
        return "Your agent is waiting for your answer."
    return "Your agent has a new message."
