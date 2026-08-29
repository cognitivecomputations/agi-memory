from __future__ import annotations

import copy
import json
import os
import subprocess

import pytest

from core import tunnel


class FakeTailscaleRunner:
    def __init__(self, *, serve_config=None, serve_status_returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.serve_config = copy.deepcopy(serve_config or {})
        self.serve_status_returncode = serve_status_returncode
        self.status = {
            "BackendState": "Running",
            "Self": {"DNSName": "hexis-host.example-tail.ts.net.", "Online": True},
        }

    def __call__(
        self,
        command,
        *,
        capture_output=True,
        check=False,
        timeout=None,
    ):
        del capture_output, check, timeout
        cmd = [str(part) for part in command]
        self.calls.append(cmd)
        if cmd[1:] == ["status", "--json"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(self.status), stderr=""
            )
        if cmd[1:] == ["serve", "status", "--json"]:
            return subprocess.CompletedProcess(
                cmd,
                self.serve_status_returncode,
                stdout=json.dumps(self.serve_config),
                stderr="serve inspection failed"
                if self.serve_status_returncode
                else "",
            )
        if cmd[1] == "serve" and "--bg" in cmd:
            target = cmd[-1]
            self.serve_config = {
                "TCP": {"443": {"HTTPS": True}},
                "Web": {
                    "hexis-host.example-tail.ts.net:443": {
                        "Handlers": {"/": {"Proxy": target}}
                    }
                },
            }
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Serve started\n", stderr=""
            )
        if cmd[1] in {"serve", "funnel"} and cmd[-1] == "off":
            web = self.serve_config.get("Web", {})
            server = web.get("hexis-host.example-tail.ts.net:443", {})
            handlers = server.get("Handlers", {})
            handlers.pop("/", None)
            self.serve_config.get("AllowFunnel", {}).pop(
                "hexis-host.example-tail.ts.net:443", None
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="off\n", stderr="")
        return subprocess.CompletedProcess(
            cmd, 2, stdout="", stderr="unexpected command"
        )


def _serve_config(target: str, *, funnel: bool = False):
    payload = {
        "TCP": {"443": {"HTTPS": True}},
        "Web": {
            "hexis-host.example-tail.ts.net:443": {"Handlers": {"/": {"Proxy": target}}}
        },
    }
    if funnel:
        payload["AllowFunnel"] = {"hexis-host.example-tail.ts.net:443": True}
    return payload


def _mock_tailscale(monkeypatch):
    monkeypatch.setattr(tunnel, "_tailscale_command", lambda: "/mock/tailscale")


def test_status_derives_active_private_route_from_provider_truth(monkeypatch, tmp_path):
    _mock_tailscale(monkeypatch)
    runner = FakeTailscaleRunner(serve_config=_serve_config("http://127.0.0.1:3477"))

    payload = tunnel.tunnel_status(
        home=tmp_path,
        runner=runner,
        probe_local=False,
    )

    assert payload["status"] == "active"
    assert payload["url"] == "https://hexis-host.example-tail.ts.net"
    assert payload["target_matches"] is True
    assert payload["funnel_enabled"] is False
    assert payload["owned"] is False


def test_start_records_ownership_without_copying_provider_state(monkeypatch, tmp_path):
    _mock_tailscale(monkeypatch)
    runner = FakeTailscaleRunner()

    payload = tunnel.start_tunnel(
        home=tmp_path,
        runner=runner,
        probe_local=False,
    )

    assert payload["status"] == "active"
    assert payload["changed"] is True
    assert payload["owned"] is True
    assert runner.calls[2] == [
        "/mock/tailscale",
        "serve",
        "--bg",
        "--https=443",
        "--set-path=/",
        "http://127.0.0.1:3477",
    ]
    state_path = tunnel.tunnel_state_path(tmp_path)
    assert os.stat(state_path).st_mode & 0o777 == 0o600
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["target"] == "http://127.0.0.1:3477"
    assert "TCP" not in state and "Web" not in state


def test_matching_ambient_route_is_never_silently_adopted(monkeypatch, tmp_path):
    _mock_tailscale(monkeypatch)
    runner = FakeTailscaleRunner(serve_config=_serve_config("http://localhost:3477"))

    payload = tunnel.start_tunnel(
        home=tmp_path,
        runner=runner,
        probe_local=False,
    )

    assert payload["changed"] is False
    assert payload["owned"] is False
    assert "not created by Hexis" in payload["warning"]
    assert not tunnel.tunnel_state_path(tmp_path).exists()
    with pytest.raises(tunnel.TunnelError, match="no ownership record"):
        tunnel.stop_tunnel(home=tmp_path, runner=runner)


def test_start_refuses_unrelated_root_route(monkeypatch, tmp_path):
    _mock_tailscale(monkeypatch)
    runner = FakeTailscaleRunner(serve_config=_serve_config("http://127.0.0.1:9000"))

    with pytest.raises(tunnel.TunnelError, match="will not replace"):
        tunnel.start_tunnel(home=tmp_path, runner=runner, probe_local=False)

    assert all("--bg" not in call for call in runner.calls)


def test_start_refuses_when_provider_state_cannot_be_inspected(monkeypatch, tmp_path):
    _mock_tailscale(monkeypatch)
    runner = FakeTailscaleRunner(serve_status_returncode=1)

    payload = tunnel.tunnel_status(
        home=tmp_path,
        runner=runner,
        probe_local=False,
    )

    assert payload["status"] == "unavailable"
    assert "serve inspection failed" in " ".join(payload["issues"])
    with pytest.raises(tunnel.TunnelError, match="serve inspection failed"):
        tunnel.start_tunnel(home=tmp_path, runner=runner, probe_local=False)
    assert all("--bg" not in call for call in runner.calls)


def test_stale_ownership_never_authorizes_a_route_change(monkeypatch, tmp_path):
    _mock_tailscale(monkeypatch)
    runner = FakeTailscaleRunner()
    state_path = tunnel.tunnel_state_path(tmp_path)
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "version": tunnel.STATE_VERSION,
                "provider": "tailscale-serve",
                "dns_name": "other-host.example-tail.ts.net",
                "target": "http://127.0.0.1:3477",
                "ui_port": 3477,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(tunnel.TunnelError, match="does not match this tailnet"):
        tunnel.start_tunnel(home=tmp_path, runner=runner, probe_local=False)

    assert all("--bg" not in call for call in runner.calls)
    assert state_path.exists()


def test_funnel_and_public_bind_are_fail_closed(monkeypatch, tmp_path):
    _mock_tailscale(monkeypatch)
    runner = FakeTailscaleRunner(
        serve_config=_serve_config("http://127.0.0.1:3477", funnel=True)
    )

    payload = tunnel.tunnel_status(
        home=tmp_path,
        runner=runner,
        bind_address="0.0.0.0",
        probe_local=False,
    )
    check = tunnel.remote_exposure_check(
        home=tmp_path,
        runner=runner,
        bind_address="0.0.0.0",
    )

    assert payload["status"] == "risky"
    assert payload["funnel_enabled"] is True
    assert payload["public_bind"] is True
    assert check["status"] == "FAIL"
    assert "out of bounds" in check["detail"]
    assert "public internet" in check["detail"]


def test_stop_removes_only_owned_root_route(monkeypatch, tmp_path):
    _mock_tailscale(monkeypatch)
    runner = FakeTailscaleRunner()
    started = tunnel.start_tunnel(
        home=tmp_path,
        runner=runner,
        probe_local=False,
    )
    assert started["owned"] is True
    runner.serve_config["Web"]["hexis-host.example-tail.ts.net:443"]["Handlers"][
        "/other"
    ] = {"Proxy": "http://127.0.0.1:8000"}

    stopped = tunnel.stop_tunnel(home=tmp_path, runner=runner)

    assert stopped["changed"] is True
    handlers = runner.serve_config["Web"]["hexis-host.example-tail.ts.net:443"][
        "Handlers"
    ]
    assert "/" not in handlers
    assert handlers["/other"]["Proxy"] == "http://127.0.0.1:8000"
    assert not tunnel.tunnel_state_path(tmp_path).exists()
    assert [
        "/mock/tailscale",
        "serve",
        "--https=443",
        "--set-path=/",
        "off",
    ] in runner.calls


def test_safe_loopback_is_ok_without_tailscale(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tunnel,
        "_tailscale_command",
        lambda: (_ for _ in ()).throw(tunnel.TunnelError("not installed")),
    )

    check = tunnel.remote_exposure_check(home=tmp_path)

    assert check == {
        "label": "Remote exposure",
        "status": "OK",
        "detail": "loopback-only; no Hexis Tailscale Funnel route detected",
    }
