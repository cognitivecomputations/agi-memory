from __future__ import annotations

import os
import stat
import sys
from types import SimpleNamespace

import pytest

from services import web_push


def test_vapid_key_is_stable_private_and_browser_compatible(tmp_path, monkeypatch):
    path = tmp_path / "push" / "vapid.pem"
    monkeypatch.setenv("HEXIS_WEB_PUSH_VAPID_PRIVATE_KEY_FILE", str(path))
    first_path, first_public = web_push.ensure_vapid_keypair()
    second_path, second_public = web_push.ensure_vapid_keypair()
    assert first_path == second_path == path
    assert first_public == second_public
    assert len(first_public) == 87
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_private_notification_body_names_decision_without_message_content():
    content = "Secret proposal details that must not appear on a lock screen"
    body = web_push._notification_body(
        {"kind": "automation_suggestion"},
        {},
        content,
        False,
    )
    assert body == "A new automation suggestion is ready for your decision."
    assert "Secret" not in body


def test_push_endpoint_rejects_loopback_without_connecting():
    assert "non-public" in str(
        web_push.validate_push_endpoint("https://127.0.0.1/subscription")
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_web_push_send_uses_vapid_and_subscription_keys(tmp_path, monkeypatch):
    path = tmp_path / "vapid.pem"
    monkeypatch.setenv("HEXIS_WEB_PUSH_VAPID_PRIVATE_KEY_FILE", str(path))
    web_push.ensure_vapid_keypair()
    captured = {}

    def fake_send(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "pywebpush", SimpleNamespace(webpush=fake_send))
    monkeypatch.setattr(web_push, "validate_push_endpoint", lambda _endpoint: None)
    attempt = await web_push._deliver_one(
        {
            "id": "11111111-1111-4111-8111-111111111111",
            "endpoint": "https://push.example.test/subscription",
            "p256dh": "public",
            "auth": "auth",
        },
        data='{"title":"Hexis"}',
        key_path=path,
        subject="https://github.com/QuixiAI/Hexis",
    )
    assert attempt.delivered is True
    assert captured["subscription_info"]["keys"] == {
        "p256dh": "public",
        "auth": "auth",
    }
    assert captured["vapid_private_key"] == str(path)
