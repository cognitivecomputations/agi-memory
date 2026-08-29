from __future__ import annotations

import json

from apps import hexis_cli
from core import tunnel


def _payload(**overrides):
    payload = {
        "status": "active",
        "url": "https://hexis-host.example-tail.ts.net",
        "ui_port": 3477,
        "local_ready": True,
        "connected": True,
        "owned": True,
        "public_bind": False,
        "funnel_enabled": False,
        "issues": [],
        "detail": "private",
    }
    payload.update(overrides)
    return payload


def test_tunnel_parser_defaults_to_read_only_status():
    args = hexis_cli.build_parser().parse_args(["tunnel"])
    assert args.func == "tunnel_status"
    assert args.port is None
    assert args.json is False


def test_tunnel_status_json(monkeypatch, tmp_path, capsys):
    args = hexis_cli.build_parser().parse_args(["tunnel", "status", "--json"])
    expected = _payload()
    monkeypatch.setattr(tunnel, "tunnel_status", lambda **_kwargs: expected)

    rc = hexis_cli._handle_tunnel_command(args, env_file=None)

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_tunnel_start_brings_up_local_stack_before_network_change(monkeypatch):
    args = hexis_cli.build_parser().parse_args(["tunnel", "start"])
    events: list[object] = []
    readiness = iter([False])
    monkeypatch.setattr(hexis_cli, "_http_ready", lambda _url: next(readiness))
    monkeypatch.setattr(hexis_cli, "_wait_http_ready", lambda _url: True)
    monkeypatch.setattr(
        hexis_cli, "main", lambda argv=None: events.append(list(argv or [])) or 0
    )
    monkeypatch.setattr(
        tunnel,
        "start_tunnel",
        lambda **kwargs: events.append(("tunnel", kwargs)) or _payload(changed=True),
    )

    rc = hexis_cli._handle_tunnel_command(args, env_file=None)

    assert rc == 0
    assert events[0] == ["up"]
    assert events[1][0] == "tunnel"
    assert events[1][1]["ui_port"] == 3477
    assert events[1][1]["bind_address"] == "127.0.0.1"


def test_tunnel_start_never_changes_network_when_stack_start_fails(monkeypatch, capsys):
    args = hexis_cli.build_parser().parse_args(["tunnel", "start"])
    changed: list[bool] = []
    monkeypatch.setattr(hexis_cli, "_http_ready", lambda _url: False)
    monkeypatch.setattr(hexis_cli, "main", lambda argv=None: 1)
    monkeypatch.setattr(
        tunnel, "start_tunnel", lambda **_kwargs: changed.append(True) or _payload()
    )

    rc = hexis_cli._handle_tunnel_command(args, env_file=None)

    assert rc == 1
    assert changed == []
    assert "no Tailscale route was changed" in capsys.readouterr().err


def test_tunnel_start_no_start_stack_is_read_only_when_dashboard_is_down(
    monkeypatch, capsys
):
    args = hexis_cli.build_parser().parse_args(["tunnel", "start", "--no-start-stack"])
    events: list[object] = []
    monkeypatch.setattr(hexis_cli, "_http_ready", lambda _url: False)
    monkeypatch.setattr(
        hexis_cli, "main", lambda argv=None: events.append(list(argv or [])) or 0
    )
    monkeypatch.setattr(
        tunnel,
        "start_tunnel",
        lambda **kwargs: events.append(("tunnel", kwargs)) or _payload(),
    )

    rc = hexis_cli._handle_tunnel_command(args, env_file=None)

    assert rc == 1
    assert events == []
    assert "without changing Tailscale state" in capsys.readouterr().err
