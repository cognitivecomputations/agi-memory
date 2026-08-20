"""Unit tests for the WSL-aware URL launcher and OAuth callback binding.

Guards the WSL fixes: gio-based webbrowser fails inside WSL, cmd.exe start
truncates URLs at ampersands, and the WSL2 localhost relay is unreliable
with 127.0.0.1-only binds.
"""

import socket
import threading
import urllib.request
from unittest.mock import patch

import pytest

from core import browser
from core.auth.callback_server import run_callback_server

pytestmark = [pytest.mark.core]


# is_wsl

def test_is_wsl_from_env(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert browser.is_wsl() is True


def test_is_wsl_false_outside_wsl(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    with patch("builtins.open", side_effect=OSError):
        assert browser.is_wsl() is False


# _wsl_launcher: command selection and URL safety

URL = "https://auth.example.com/authorize?a=1&b=2&state=xyz"


def test_wsl_launcher_prefers_wslview():
    with patch("shutil.which", side_effect=lambda name: "/usr/bin/wslview" if name == "wslview" else None):
        assert browser._wsl_launcher(URL) == ["/usr/bin/wslview", URL]


def test_wsl_launcher_powershell_keeps_ampersands():
    with patch(
        "shutil.which",
        side_effect=lambda name: "/mnt/c/ps/powershell.exe" if name == "powershell.exe" else None,
    ):
        cmd = browser._wsl_launcher(URL)
    # Argument list (no shell), full URL intact inside a single-quoted
    # PowerShell string — cmd.exe-style `start` would split at every `&`.
    assert cmd[0].endswith("powershell.exe")
    assert cmd[-1] == f"Start-Process '{URL}'"
    assert not any("cmd.exe" in part for part in cmd)


def test_wsl_launcher_escapes_single_quotes():
    with patch(
        "shutil.which",
        side_effect=lambda name: "/mnt/c/ps/powershell.exe" if name == "powershell.exe" else None,
    ):
        cmd = browser._wsl_launcher("https://x/?q='p'")
    assert cmd[-1] == "Start-Process 'https://x/?q=''p'''"


def test_wsl_launcher_none_when_nothing_available():
    with patch("shutil.which", return_value=None), patch.object(
        browser.Path, "exists", return_value=False
    ):
        assert browser._wsl_launcher(URL) is None


def test_open_url_uses_wsl_launcher():
    calls = []
    with patch.object(browser, "is_wsl", return_value=True), patch.object(
        browser, "_wsl_launcher", return_value=["/usr/bin/wslview", URL]
    ), patch.object(browser.subprocess, "run", side_effect=lambda cmd, **kw: calls.append(cmd)):
        assert browser.open_url(URL) is True
    assert calls == [["/usr/bin/wslview", URL]]


# run_callback_server: bind_host must actually take effect.

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_callback_server_serves_on_wildcard_bind():
    port = _free_port()
    state = "test-state"
    result_holder = {}

    def _serve():
        result_holder["result"] = run_callback_server(
            port=port,
            callback_path="/auth/callback",
            timeout_seconds=10,
            expected_state=state,
            bind_host="0.0.0.0",
        )

    t = threading.Thread(target=_serve)
    t.start()
    try:
        for _ in range(50):
            try:
                resp = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/auth/callback?code=abc123&state={state}",
                    timeout=2,
                )
                assert resp.status == 200
                break
            except OSError:
                import time

                time.sleep(0.1)
        else:
            pytest.fail("callback server never came up")
    finally:
        t.join(timeout=5)
    assert result_holder["result"] == {"code": "abc123", "state": state}
