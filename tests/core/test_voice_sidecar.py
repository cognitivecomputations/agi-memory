from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from apps import voice_sidecar as voice_entrypoint
from core import voice_sidecar


def test_entrypoint_refuses_non_loopback_bind(capsys):
    rc = voice_entrypoint.main(
        ["--host", "0.0.0.0", "--port", "42667", "--model", "voice"]
    )

    assert rc == 2
    assert "refuses a non-loopback bind" in capsys.readouterr().err


def test_start_records_exact_owned_process(monkeypatch, tmp_path):
    info_calls = iter(
        [
            None,
            {"voice": {"name": "live-voice"}},
            {"voice": {"name": "live-voice"}},
        ]
    )
    process = SimpleNamespace(pid=4321, poll=lambda: None, terminate=lambda: None)
    monkeypatch.setattr(voice_sidecar, "_provider_info", lambda *_args, **_kwargs: next(info_calls))
    monkeypatch.setattr(voice_sidecar.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        voice_sidecar,
        "_process_command",
        lambda _pid: (
            "python -m apps.voice_sidecar --host 127.0.0.1 --port 42667 "
            "--model model-a --hexis-owner-token fixed-token"
        ),
    )
    monkeypatch.setattr(voice_sidecar.secrets, "token_urlsafe", lambda _size: "fixed-token")

    result = voice_sidecar.start_voice_sidecar(
        model="model-a",
        home=tmp_path,
        command=["python", "-m", "apps.voice_sidecar"],
        wait_seconds=1,
    )

    assert result["ready"] is True
    assert result["owned"] is True
    state = voice_sidecar.state_path(tmp_path)
    assert os.stat(state).st_mode & 0o777 == 0o600
    assert json.loads(state.read_text(encoding="utf-8"))["model"] == "model-a"


def test_ambient_provider_is_never_adopted(monkeypatch, tmp_path):
    monkeypatch.setattr(
        voice_sidecar,
        "_provider_info",
        lambda *_args, **_kwargs: {"voice": {"name": "ambient"}},
    )

    result = voice_sidecar.start_voice_sidecar(model="model-a", home=tmp_path)

    assert result["changed"] is False
    assert result["owned"] is False
    assert "did not adopt" in result["warning"]
    assert not voice_sidecar.state_path(tmp_path).exists()


def test_stale_ownership_refuses_start_and_stop(monkeypatch, tmp_path):
    state = voice_sidecar.state_path(tmp_path)
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps({"version": 1, "pid": 999, "model": "model-a"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(voice_sidecar, "_provider_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(voice_sidecar, "_process_command", lambda _pid: "other process")

    with pytest.raises(voice_sidecar.VoiceSidecarError, match="no longer matches"):
        voice_sidecar.start_voice_sidecar(model="model-a", home=tmp_path)
    with pytest.raises(voice_sidecar.VoiceSidecarError, match="left alone"):
        voice_sidecar.stop_voice_sidecar(home=tmp_path)
    assert state.exists()


def test_stop_terminates_only_matching_owned_process(monkeypatch, tmp_path):
    state = voice_sidecar.state_path(tmp_path)
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "version": 1,
                "pid": 321,
                "model": "model-a",
                "ownership_token": "fixed-token",
            }
        ),
        encoding="utf-8",
    )
    commands = iter(
        [
            "python -m apps.voice_sidecar --model model-a "
            "--hexis-owner-token fixed-token",
            None,
        ]
    )
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(voice_sidecar, "_process_command", lambda _pid: next(commands))
    monkeypatch.setattr(voice_sidecar.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(voice_sidecar, "_provider_info", lambda *_args, **_kwargs: None)

    result = voice_sidecar.stop_voice_sidecar(home=tmp_path)

    assert result["changed"] is True
    assert killed == [(321, voice_sidecar.signal.SIGTERM)]
    assert not state.exists()
