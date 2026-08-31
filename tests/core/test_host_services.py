from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

import core.host_services as host_services


class FakeSystemdRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.active: set[str] = set()
        self.enabled: set[str] = set()

    def __call__(self, command, *, capture_output=True, check=False):
        del capture_output, check
        cmd = [str(part) for part in command]
        self.calls.append(cmd)
        if cmd[0].endswith("loginctl"):
            return subprocess.CompletedProcess(cmd, 0, stdout="no\n", stderr="")
        operation = next(
            (
                value
                for value in (
                    "daemon-reload",
                    "enable",
                    "start",
                    "stop",
                    "restart",
                    "disable",
                    "is-active",
                    "is-enabled",
                )
                if value in cmd
            ),
            "",
        )
        units = [part for part in cmd if part.endswith(".service")]
        if operation == "enable":
            self.enabled.update(units)
            if "--now" in cmd:
                self.active.update(units)
        elif operation == "disable":
            self.enabled.difference_update(units)
            if "--now" in cmd:
                self.active.difference_update(units)
        elif operation == "start":
            self.active.update(units)
        elif operation == "stop":
            self.active.difference_update(units)
        elif operation == "restart":
            self.active.update(units)
        elif operation == "is-active":
            active = bool(units and units[0] in self.active)
            return subprocess.CompletedProcess(
                cmd,
                0 if active else 3,
                stdout="active\n" if active else "inactive\n",
                stderr="",
            )
        elif operation == "is-enabled":
            enabled = bool(units and units[0] in self.enabled)
            return subprocess.CompletedProcess(
                cmd,
                0 if enabled else 1,
                stdout="enabled\n" if enabled else "disabled\n",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


class FakeLaunchdRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.active: set[str] = set()
        self.disabled: set[str] = set()

    def __call__(self, command, *, capture_output=True, check=False):
        del capture_output, check
        cmd = [str(part) for part in command]
        self.calls.append(cmd)
        action = cmd[1] if len(cmd) > 1 else ""
        if action == "print-disabled":
            body = "\n".join(f'"{label}" => disabled' for label in self.disabled)
            return subprocess.CompletedProcess(cmd, 0, stdout=body, stderr="")
        if action == "print":
            target = cmd[2]
            return subprocess.CompletedProcess(
                cmd,
                0 if target in self.active else 113,
                stdout="state = running\n" if target in self.active else "",
                stderr="" if target in self.active else "Could not find service",
            )
        if action == "bootstrap":
            payload = plistlib.loads(Path(cmd[3]).read_bytes())
            self.active.add(f"{cmd[2]}/{payload['Label']}")
        elif action == "bootout":
            self.active.discard(cmd[2])
        elif action == "kickstart":
            self.active.add(cmd[-1])
        elif action == "enable":
            self.disabled.discard(cmd[2].rsplit("/", 1)[-1])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def _which(name: str) -> str:
    return f"/mock/bin/{name}"


def test_systemd_install_control_and_uninstall_store_references_only(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(host_services.shutil, "which", _which)
    env_file = tmp_path / "runtime" / ".env"
    env_file.parent.mkdir()
    env_file.write_text("OPENAI_API_KEY=secret-value-never-copy\n", encoding="utf-8")
    env_file.chmod(0o600)
    runner = FakeSystemdRunner()

    installed = host_services.install_host_services(
        services=["heartbeat", "maintenance"],
        env_file=env_file,
        instance="primary",
        backend="systemd",
        home=tmp_path,
        python_executable=Path(sys.executable),
        runner=runner,
    )

    assert installed["started"] == ["heartbeat", "maintenance"]
    assert installed["env_file"] == str(env_file)
    assert installed["linger"] == "disabled"
    state_path = host_services.host_service_state_path(tmp_path)
    assert os.stat(state_path).st_mode & 0o777 == 0o600
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["instance"] == "primary"
    assert "secret-value-never-copy" not in state_path.read_text(encoding="utf-8")

    heartbeat_unit = host_services.host_service_unit_path(
        "heartbeat", backend="systemd", home=tmp_path
    )
    unit_text = heartbeat_unit.read_text(encoding="utf-8")
    assert host_services.MANAGED_MARKER in unit_text
    assert (
        f"{sys.executable} -m apps.worker --mode heartbeat --instance primary"
        in unit_text
    )
    assert f"WorkingDirectory={env_file.parent}" in unit_text
    assert f'Environment="HEXIS_ENV_FILE={env_file}"' in unit_text
    assert "secret-value-never-copy" not in unit_text

    status = host_services.host_service_status(
        backend="systemd", home=tmp_path, runner=runner
    )
    core = {item["name"]: item for item in status["services"]}
    assert core["heartbeat"]["active"] is True
    assert core["maintenance"]["enabled"] is True
    assert core["channels"]["installed"] is False

    stopped = host_services.control_host_services(
        "stop", ["heartbeat"], backend="systemd", home=tmp_path, runner=runner
    )
    assert stopped["status"]["services"][0]["active"] is False

    removed = host_services.uninstall_host_services(
        ["heartbeat"], backend="systemd", home=tmp_path, runner=runner
    )
    assert removed["uninstalled"] == ["heartbeat"]
    assert heartbeat_unit.exists() is False
    assert host_services.installed_host_services(backend="systemd", home=tmp_path) == [
        "maintenance"
    ]


def test_launchd_units_use_same_python_and_preserve_logs_on_uninstall(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(host_services.shutil, "which", _which)
    runner = FakeLaunchdRunner()
    env_file = tmp_path / ".env.local"
    env_file.write_text("RABBITMQ_MANAGEMENT_PORT=45673\n", encoding="utf-8")
    result = host_services.install_host_services(
        services=["heartbeat", "channels"],
        env_file=env_file,
        working_directory=tmp_path,
        backend="launchd",
        home=tmp_path,
        python_executable=Path(sys.executable),
        runner=runner,
    )
    assert result["started"] == ["heartbeat", "channels"]

    path = host_services.host_service_unit_path(
        "channels", backend="launchd", home=tmp_path
    )
    plist = plistlib.loads(path.read_bytes())
    assert plist["HexisManaged"] is True
    assert plist["ProgramArguments"] == [
        sys.executable,
        "-m",
        "services.channel_worker",
    ]
    assert plist["WorkingDirectory"] == str(tmp_path)
    assert plist["EnvironmentVariables"] == {
        "PYTHONUNBUFFERED": "1",
        "HEXIS_ENV_FILE": str(env_file),
    }

    status = host_services.host_service_status(
        backend="launchd", home=tmp_path, runner=runner
    )
    assert {item["name"] for item in status["services"] if item["active"]} == {
        "heartbeat",
        "channels",
    }
    runner.disabled.add(host_services.SERVICE_DEFINITIONS["channels"].launchd_label)
    disabled_status = host_services.host_service_status(
        backend="launchd", home=tmp_path, runner=runner
    )
    channels = next(
        item for item in disabled_status["services"] if item["name"] == "channels"
    )
    assert channels["enabled"] is False

    log_dir = host_services.host_service_log_dir(tmp_path)
    log_file = log_dir / "heartbeat.log"
    log_file.write_text("preserve me\n", encoding="utf-8")
    removed = host_services.uninstall_host_services(
        ["heartbeat"], backend="launchd", home=tmp_path, runner=runner
    )
    assert removed["preserved_log_directory"] == str(log_dir)
    assert log_file.read_text(encoding="utf-8") == "preserve me\n"


def test_install_refuses_to_overwrite_an_unmanaged_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(host_services.shutil, "which", _which)
    path = host_services.host_service_unit_path(
        "heartbeat", backend="systemd", home=tmp_path
    )
    path.parent.mkdir(parents=True)
    path.write_text("[Service]\nExecStart=/someone/elses/worker\n", encoding="utf-8")

    with pytest.raises(host_services.HostServiceError, match="Refusing to overwrite"):
        host_services.install_host_services(
            backend="systemd",
            home=tmp_path,
            python_executable=Path(sys.executable),
            runner=FakeSystemdRunner(),
        )

    assert "someone/elses" in path.read_text(encoding="utf-8")


def test_rendered_units_reject_invalid_instances(tmp_path):
    with pytest.raises(ValueError, match="Invalid instance name"):
        host_services.render_systemd_unit(
            "heartbeat",
            python_executable=Path(sys.executable),
            working_directory=tmp_path,
            env_file=None,
            instance="bad instance; touch /tmp/nope",
        )
