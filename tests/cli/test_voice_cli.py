from __future__ import annotations

import json

from apps import hexis_cli
from core import voice_sidecar


def _payload(**overrides):
    payload = {
        "status": "active",
        "ready": True,
        "owned": True,
        "state_present": True,
        "model": "en_US-lessac-medium",
        "url": "http://127.0.0.1:42667",
        "log_path": "/tmp/voice.log",
    }
    payload.update(overrides)
    return payload


def test_voice_parser_defaults_to_read_only_status():
    args = hexis_cli.build_parser().parse_args(["voice"])

    assert args.func == "voice_status"
    assert args.json is False


def test_voice_status_json_is_read_only(monkeypatch, capsys):
    args = hexis_cli.build_parser().parse_args(["voice", "status", "--json"])
    expected = _payload()
    started: list[object] = []
    monkeypatch.setattr(voice_sidecar, "voice_sidecar_status", lambda: expected)
    monkeypatch.setattr(
        voice_sidecar,
        "start_voice_sidecar",
        lambda **kwargs: started.append(kwargs),
    )

    rc = hexis_cli._handle_voice_command(args)

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert started == []


def test_voice_setup_decline_makes_no_process_change(monkeypatch):
    args = hexis_cli.build_parser().parse_args(["voice", "setup"])
    started: list[object] = []
    monkeypatch.setattr(hexis_cli, "_install_voice_support", lambda **_kwargs: False)
    monkeypatch.setattr(
        voice_sidecar,
        "start_voice_sidecar",
        lambda **kwargs: started.append(kwargs),
    )

    rc = hexis_cli._handle_voice_command(args)

    assert rc == 1
    assert started == []


def test_voice_start_uses_live_configured_model(monkeypatch):
    args = hexis_cli.build_parser().parse_args(["voice", "start"])
    started: list[dict[str, object]] = []

    async def configured_model() -> str:
        return "voice-from-db"

    monkeypatch.setattr(hexis_cli, "_voice_support_installed", lambda: True)
    monkeypatch.setattr(hexis_cli, "_configured_voice_model", configured_model)
    monkeypatch.setattr(
        voice_sidecar,
        "start_voice_sidecar",
        lambda **kwargs: started.append(kwargs) or _payload(model="voice-from-db"),
    )

    rc = hexis_cli._handle_voice_command(args)

    assert rc == 0
    assert started == [{"model": "voice-from-db", "wait_seconds": 300.0}]


def test_stack_start_only_launches_when_live_config_enables_voice(monkeypatch):
    launched: list[dict[str, object]] = []

    async def configured_output() -> tuple[bool, str]:
        return True, "voice-from-db"

    monkeypatch.setattr(hexis_cli, "_configured_voice_output", configured_output)
    monkeypatch.setattr(hexis_cli, "_voice_support_installed", lambda: True)
    monkeypatch.setattr(
        voice_sidecar,
        "voice_sidecar_status",
        lambda: _payload(status="inactive", ready=False, owned=False, state_present=False),
    )
    monkeypatch.setattr(
        voice_sidecar,
        "start_voice_sidecar",
        lambda **kwargs: launched.append(kwargs) or _payload(changed=True),
    )

    changed, note = hexis_cli._start_configured_voice_sidecar()

    assert changed is True
    assert note is None
    assert launched == [{"model": "voice-from-db", "wait_seconds": 2}]


def test_stack_stop_leaves_ambient_voice_provider_alone(monkeypatch):
    stopped: list[bool] = []
    monkeypatch.setattr(
        voice_sidecar,
        "voice_sidecar_status",
        lambda: _payload(owned=False, state_present=False),
    )
    monkeypatch.setattr(
        voice_sidecar,
        "stop_voice_sidecar",
        lambda: stopped.append(True),
    )

    ok, changed, note = hexis_cli._stop_owned_voice_sidecar()

    assert ok is True
    assert changed is False
    assert "ambient" in str(note)
    assert stopped == []
