from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.auth.ui_flow import AuthFlowCoordinator, AuthFlowError
from core.auth.callback_server import run_callback_server

pytestmark = pytest.mark.asyncio


async def test_authorization_code_flow_exchanges_without_exposing_secrets():
    coordinator = AuthFlowCoordinator(enable_callbacks=False)
    credentials = SimpleNamespace(
        access="access-secret",
        refresh="refresh-secret",
        expires_ms=4_102_444_800_000,
        account_id="acct-visible",
    )

    with (
        patch(
            "core.auth.ui_flow.generate_pkce", return_value=("verifier", "challenge")
        ),
        patch("core.auth.ui_flow.create_state", return_value="expected-state"),
        patch(
            "core.auth.openai_codex.build_authorize_url",
            return_value="https://example.test/authorize",
        ),
        patch(
            "core.auth.openai_codex.exchange_authorization_code",
            new=AsyncMock(return_value=credentials),
        ) as exchange,
        patch("core.auth.openai_codex.save_openai_codex_credentials") as save,
        patch("core.auth.ui_flow._load_credentials", return_value=credentials),
    ):
        started = await coordinator.start("openai-codex")
        completed = await coordinator.complete(
            started["session_id"],
            "http://localhost:1455/auth/callback?code=the-code&state=expected-state",
        )

    assert started["status"] == "awaiting_code"
    assert started["authorization_url"] == "https://example.test/authorize"
    assert completed["status"] == "complete"
    assert completed["credential"]["account_id"] == "acct-visible"
    assert "access" not in completed["credential"]
    assert "refresh" not in completed["credential"]
    exchange.assert_awaited_once_with(code="the-code", verifier="verifier")
    save.assert_called_once_with(credentials)


async def test_authorization_code_flow_rejects_state_from_another_attempt():
    coordinator = AuthFlowCoordinator(enable_callbacks=False)
    with (
        patch(
            "core.auth.ui_flow.generate_pkce", return_value=("verifier", "challenge")
        ),
        patch("core.auth.ui_flow.create_state", return_value="expected-state"),
        patch(
            "core.auth.openai_codex.build_authorize_url",
            return_value="https://example.test/authorize",
        ),
        patch(
            "core.auth.openai_codex.exchange_authorization_code",
            new=AsyncMock(),
        ) as exchange,
    ):
        started = await coordinator.start("openai-codex")
        with pytest.raises(AuthFlowError, match="does not match"):
            await coordinator.complete(
                started["session_id"],
                "http://localhost:1455/auth/callback?code=the-code&state=wrong-state",
            )

    exchange.assert_not_awaited()


async def test_device_flow_polls_and_persists_in_background():
    coordinator = AuthFlowCoordinator(enable_callbacks=False)
    device = SimpleNamespace(
        device_code="device-secret",
        user_code="ABCD-EFGH",
        verification_uri="https://example.test/device",
        verification_uri_complete=None,
        interval=1,
        expires_in=300,
    )
    credentials = SimpleNamespace(
        access="access-secret",
        refresh="refresh-secret",
        expires_ms=4_102_444_800_000,
        resource_url="https://portal.example.test",
    )

    with (
        patch(
            "core.auth.qwen_portal.start_device_flow",
            new=AsyncMock(return_value=(device, "verifier")),
        ),
        patch(
            "core.auth.qwen_portal.poll_for_token",
            new=AsyncMock(return_value=credentials),
        ) as poll,
        patch("core.auth.qwen_portal.save_credentials") as save,
        patch("core.auth.ui_flow._load_credentials", return_value=credentials),
    ):
        started = await coordinator.start("qwen-portal")
        await asyncio.sleep(0)
        completed = coordinator.session(started["session_id"])

    assert started["flow"] == "device_code"
    assert started["user_code"] == "ABCD-EFGH"
    assert "device-secret" not in str(started)
    assert completed["status"] == "complete"
    poll.assert_awaited_once_with("device-secret", "verifier", 1, 300)
    save.assert_called_once_with(credentials)


async def test_status_is_redacted():
    coordinator = AuthFlowCoordinator(enable_callbacks=False)
    credentials = SimpleNamespace(
        access="access-secret",
        refresh="refresh-secret",
        client_secret="client-secret",
        expires_ms=4_102_444_800_000,
        email="person@example.test",
        project_id="project-visible",
    )

    with patch("core.auth.ui_flow._load_credentials", return_value=credentials):
        status = coordinator.status("google-gemini-cli")

    assert status["configured"] is True
    assert status["email"] == "person@example.test"
    assert status["project_id"] == "project-visible"
    assert "access-secret" not in str(status)
    assert "refresh-secret" not in str(status)
    assert "client-secret" not in str(status)


async def test_validate_refreshes_only_after_explicit_request():
    coordinator = AuthFlowCoordinator(enable_callbacks=False)
    credentials = SimpleNamespace(
        access="access-secret",
        refresh="refresh-secret",
        expires_ms=4_102_444_800_000,
        account_id="acct-visible",
    )
    ensure = AsyncMock(return_value=credentials)

    with (
        patch("core.auth.ui_flow._load_credentials", return_value=credentials),
        patch("core.auth.ui_flow._ensure_fresh_credentials", ensure),
    ):
        status = coordinator.status("openai-codex")
        validated = await coordinator.validate("openai-codex")

    assert status["configured"] is True
    assert validated["configured"] is True
    ensure.assert_awaited_once_with("openai-codex")


async def test_validate_returns_reauthentication_action_when_refresh_fails():
    coordinator = AuthFlowCoordinator(enable_callbacks=False)
    with (
        patch("core.auth.ui_flow._load_credentials", return_value=SimpleNamespace()),
        patch(
            "core.auth.ui_flow._ensure_fresh_credentials",
            new=AsyncMock(side_effect=RuntimeError("refresh rejected")),
        ),
    ):
        with pytest.raises(AuthFlowError, match="Use Authenticate again"):
            await coordinator.validate("chutes")


async def test_callback_listener_can_be_cancelled_without_waiting_for_timeout():
    cancel = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            run_callback_server,
            port=0,
            timeout_seconds=60,
            cancel_event=cancel,
        )
    )
    await asyncio.sleep(0.05)
    cancel.set()

    assert await asyncio.wait_for(task, timeout=2) is None


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("core.auth.google_gemini_cli", "GeminiCliCredentials"),
        ("core.auth.google_antigravity", "AntigravityCredentials"),
    ],
)
async def test_google_auth_records_retain_ui_client_credentials(
    module_name, class_name
):
    import importlib

    module = importlib.import_module(module_name)
    credentials_type = getattr(module, class_name)
    credentials = credentials_type(
        access="access-secret",
        refresh="refresh-secret",
        expires_ms=4_102_444_800_000,
        project_id="project-visible",
        email="person@example.test",
        client_id="client-id",
        client_secret="client-secret",
    )

    restored = module.credentials_from_value(module.credentials_to_dict(credentials))

    assert restored == credentials
