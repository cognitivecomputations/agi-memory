from __future__ import annotations

import json
from pathlib import Path

from apps import hexis_cli
from core import host_services


def test_service_parser_exposes_explicit_migration_controls():
    args = hexis_cli.build_parser().parse_args(
        [
            "service",
            "install",
            "--channels",
            "--env-file",
            ".env.local",
            "--enable-linger",
            "--replace-docker-workers",
        ]
    )

    assert args.func == "service_install"
    assert args.channels is True
    assert args.env_file == Path(".env.local")
    assert args.enable_linger is True
    assert args.replace_docker_workers is True


def test_install_refuses_when_docker_worker_truth_is_unavailable(
    monkeypatch, tmp_path, capsys
):
    args = hexis_cli.build_parser().parse_args(["service", "install"])
    monkeypatch.setattr(
        hexis_cli, "_running_container_workers", lambda *_args: (None, None)
    )

    rc = hexis_cli._handle_host_service_command(
        args,
        compose_file=tmp_path / "docker-compose.yml",
        stack_root=tmp_path,
        env_file=None,
    )

    assert rc == 1
    assert "refused to risk installing duplicate workers" in capsys.readouterr().err


def test_explicit_migration_installs_before_stopping_docker(monkeypatch, tmp_path):
    args = hexis_cli.build_parser().parse_args(
        ["service", "install", "--replace-docker-workers"]
    )
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        hexis_cli,
        "_running_container_workers",
        lambda *_args: (
            ["heartbeat_worker", "maintenance_worker"],
            ["docker", "compose"],
        ),
    )
    monkeypatch.setattr(hexis_cli, "resolve_instance", lambda: None)
    monkeypatch.setattr(
        hexis_cli,
        "run_compose",
        lambda _cmd, _file, _root, compose_args, _env: (
            events.append(("compose", list(compose_args))) or 0
        ),
    )

    def install(**kwargs):
        events.append(("install", kwargs["start"]))
        return {
            "installed": ["heartbeat", "maintenance"],
            "env_file": None,
            "instance": None,
            "backend": "systemd",
            "linger": "enabled",
            "warning": None,
        }

    def control(action, services):
        events.append(("control", (action, list(services))))
        return {"services": list(services)}

    monkeypatch.setattr(host_services, "install_host_services", install)
    monkeypatch.setattr(host_services, "control_host_services", control)
    monkeypatch.setattr(
        host_services,
        "host_service_status",
        lambda: {
            "services": [
                {"name": "heartbeat", "active": True},
                {"name": "maintenance", "active": True},
            ]
        },
    )

    rc = hexis_cli._handle_host_service_command(
        args,
        compose_file=tmp_path / "docker-compose.yml",
        stack_root=tmp_path,
        env_file=None,
    )

    assert rc == 0
    assert events == [
        ("install", False),
        ("compose", ["stop", "heartbeat_worker", "maintenance_worker"]),
        ("control", ("start", ["heartbeat", "maintenance"])),
    ]


def test_failed_host_start_safely_restores_docker_workers(
    monkeypatch, tmp_path, capsys
):
    args = hexis_cli.build_parser().parse_args(
        ["service", "install", "--replace-docker-workers"]
    )
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        hexis_cli,
        "_running_container_workers",
        lambda *_args: (["heartbeat_worker"], ["docker", "compose"]),
    )
    monkeypatch.setattr(hexis_cli, "resolve_instance", lambda: None)
    monkeypatch.setattr(
        hexis_cli,
        "run_compose",
        lambda _cmd, _file, _root, compose_args, _env: (
            events.append(("compose", list(compose_args))) or 0
        ),
    )
    monkeypatch.setattr(
        host_services,
        "install_host_services",
        lambda **_kwargs: {
            "installed": ["heartbeat", "maintenance"],
            "env_file": None,
            "instance": None,
            "backend": "systemd",
            "linger": "enabled",
            "warning": None,
        },
    )

    def control(action, services):
        events.append(("control", (action, list(services))))
        if action == "start":
            raise host_services.HostServiceError("provider rejected start")
        return {"services": list(services)}

    monkeypatch.setattr(host_services, "control_host_services", control)

    rc = hexis_cli._handle_host_service_command(
        args,
        compose_file=tmp_path / "docker-compose.yml",
        stack_root=tmp_path,
        env_file=None,
    )

    assert rc == 1
    assert events == [
        ("compose", ["stop", "heartbeat_worker"]),
        ("control", ("start", ["heartbeat", "maintenance"])),
        ("control", ("stop", ["heartbeat", "maintenance"])),
        ("compose", ["up", "-d", "heartbeat_worker"]),
    ]
    assert "provider rejected start" in capsys.readouterr().err


def test_service_status_json_is_machine_readable(monkeypatch, tmp_path, capsys):
    args = hexis_cli.build_parser().parse_args(["service", "status", "--json"])
    payload = {
        "backend": "launchd",
        "services": [{"name": "heartbeat", "installed": False}],
    }
    monkeypatch.setattr(host_services, "host_service_status", lambda: payload)

    rc = hexis_cli._handle_host_service_command(
        args,
        compose_file=None,
        stack_root=tmp_path,
        env_file=None,
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == payload
