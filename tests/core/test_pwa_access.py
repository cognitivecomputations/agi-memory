from __future__ import annotations

import json
import subprocess

import pytest

from core import cli_api


def test_dashboard_https_rejects_plain_public_url(monkeypatch):
    monkeypatch.setenv("HEXIS_UI_PUBLIC_URL", "http://hexis.example.test")
    check = cli_api.dashboard_https_check()
    assert check["status"] == "WARN"
    assert "not HTTPS" in check["detail"]
    assert "microphone" in check["detail"]


def test_dashboard_https_probes_explicit_secure_url(monkeypatch):
    monkeypatch.setenv("HEXIS_UI_PUBLIC_URL", "https://hexis.example.test")
    monkeypatch.setattr(
        cli_api,
        "_probe_dashboard_https",
        lambda _url, _timeout: (True, "HTTP 200"),
    )
    check = cli_api.dashboard_https_check()
    assert check == {
        "label": "Dashboard HTTPS",
        "status": "OK",
        "detail": "reachable at https://hexis.example.test (HTTP 200)",
    }


def test_dashboard_https_names_private_runbook_when_tailscale_is_absent(monkeypatch):
    monkeypatch.delenv("HEXIS_UI_PUBLIC_URL", raising=False)
    monkeypatch.delenv("HEXIS_PUBLIC_URL", raising=False)
    monkeypatch.setattr(cli_api.shutil, "which", lambda _name: None)
    check = cli_api.dashboard_https_check()
    assert check["status"] == "WARN"
    assert "local dashboard only" in check["detail"]
    assert "secure-remote-access.md" in check["detail"]


def test_dashboard_https_derives_custom_port_and_exact_tunnel_target(monkeypatch):
    monkeypatch.delenv("HEXIS_UI_PUBLIC_URL", raising=False)
    monkeypatch.delenv("HEXIS_PUBLIC_URL", raising=False)
    monkeypatch.setenv("HEXIS_UI_PORT", "4567")
    monkeypatch.setattr(cli_api.shutil, "which", lambda _name: "/mock/tailscale")
    monkeypatch.setattr(
        cli_api.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["tailscale", "status", "--json"],
            0,
            stdout=json.dumps(
                {
                    "Self": {
                        "DNSName": "hexis-host.example-tail.ts.net.",
                        "Online": True,
                    }
                }
            ),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        cli_api,
        "_probe_dashboard_https",
        lambda _url, _timeout: (False, "not reachable"),
    )

    check = cli_api.dashboard_https_check()

    assert check["status"] == "WARN"
    assert "hexis tunnel start" in check["detail"]
    assert "http://127.0.0.1:4567" in check["detail"]


@pytest.mark.asyncio
async def test_doctor_reports_remote_exposure_even_when_database_is_down(monkeypatch):
    async def unavailable_database(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(cli_api, "_connect_with_retry", unavailable_database)
    monkeypatch.setattr(
        cli_api,
        "dashboard_https_check",
        lambda: {"label": "Dashboard HTTPS", "status": "WARN", "detail": "local"},
    )
    monkeypatch.setattr(
        "core.tunnel.remote_exposure_check",
        lambda **_kwargs: {
            "label": "Remote exposure",
            "status": "FAIL",
            "detail": "public bind",
        },
    )

    checks = await cli_api.doctor_payload("postgresql://unavailable", wait_seconds=0)

    assert [check["label"] for check in checks] == [
        "PostgreSQL",
        "Remote exposure",
        "Dashboard HTTPS",
    ]
    assert checks[1]["status"] == "FAIL"
