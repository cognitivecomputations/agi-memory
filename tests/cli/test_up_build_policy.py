from __future__ import annotations

import pytest

from apps import hexis_cli
from apps.hexis_cli import _up_compose_args, build_parser


def test_source_up_uses_published_image_by_default():
    args = _up_compose_args(["active"], is_source=True, build=False)
    assert args == [
        "--profile",
        "active",
        "up",
        "-d",
        "--no-build",
        "--pull",
        "missing",
    ]


def test_source_up_builds_only_when_explicit():
    assert _up_compose_args([], is_source=True, build=True) == [
        "up",
        "-d",
        "--build",
    ]


def test_packaged_up_has_no_build_path():
    assert _up_compose_args([], is_source=False, build=True) == ["up", "-d"]


def test_build_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["up", "--build", "--no-build"])


def test_source_up_failure_names_both_recovery_paths(monkeypatch, capsys):
    calls: list[list[str]] = []

    monkeypatch.setattr(hexis_cli, "ensure_docker", lambda: "docker")
    monkeypatch.setattr(
        hexis_cli, "ensure_compose", lambda _docker: ["docker", "compose"]
    )
    monkeypatch.setattr(
        hexis_cli,
        "run_compose",
        lambda _compose, _file, _root, args, _env: calls.append(list(args)) or 1,
    )

    assert hexis_cli.main(["up"]) == 1
    assert calls == [["up", "-d", "--no-build", "--pull", "missing"]]
    stderr = capsys.readouterr().err
    assert "no source build was attempted" in stderr
    assert "run `hexis up` again" in stderr
    assert "run `hexis up --build`" in stderr


def test_up_excludes_workers_owned_by_host_services(monkeypatch):
    calls: list[list[str]] = []
    configured = [
        "db",
        "rabbitmq",
        "heartbeat_worker",
        "maintenance_worker",
        "channel_worker",
        "api",
        "ui",
    ]
    monkeypatch.setattr(hexis_cli, "ensure_docker", lambda: "docker")
    monkeypatch.setattr(
        hexis_cli, "ensure_compose", lambda _docker: ["docker", "compose"]
    )
    monkeypatch.setattr(
        hexis_cli,
        "_host_managed_compose_workers",
        lambda: {"heartbeat_worker", "maintenance_worker"},
    )
    monkeypatch.setattr(
        hexis_cli,
        "_configured_compose_services",
        lambda *_args, **_kwargs: configured,
    )
    monkeypatch.setattr(
        hexis_cli,
        "run_compose",
        lambda _compose, _file, _root, args, _env: calls.append(list(args)) or 1,
    )

    assert hexis_cli.main(["up"]) == 1
    assert calls == [
        [
            "up",
            "-d",
            "--no-build",
            "--pull",
            "missing",
            "db",
            "rabbitmq",
            "channel_worker",
            "api",
            "ui",
        ]
    ]


def test_reset_stops_host_workers_and_does_not_recreate_docker_copies(monkeypatch):
    calls: list[list[str]] = []
    lifecycle: list[str] = []
    configured = [
        "db",
        "rabbitmq",
        "heartbeat_worker",
        "maintenance_worker",
        "api",
    ]
    monkeypatch.setattr(hexis_cli, "ensure_docker", lambda: "docker")
    monkeypatch.setattr(
        hexis_cli, "ensure_compose", lambda _docker: ["docker", "compose"]
    )
    monkeypatch.setattr(
        hexis_cli,
        "_host_managed_compose_workers",
        lambda: {"heartbeat_worker", "maintenance_worker"},
    )
    monkeypatch.setattr(
        hexis_cli,
        "_configured_compose_services",
        lambda *_args, **_kwargs: configured,
    )
    monkeypatch.setattr(
        hexis_cli,
        "_stop_installed_host_services",
        lambda: lifecycle.append("stop") or (True, None),
    )
    monkeypatch.setattr(
        hexis_cli,
        "_ensure_installed_host_services_running",
        lambda: lifecycle.append("start") or (True, None),
    )
    monkeypatch.setattr(
        hexis_cli,
        "run_compose",
        lambda _compose, _file, _root, args, _env: calls.append(list(args)) or 0,
    )

    assert hexis_cli.main(["reset", "--yes"]) == 0
    assert lifecycle == ["stop", "start"]
    assert calls == [
        ["down", "-v"],
        ["build", "db"],
        ["up", "-d", "db", "rabbitmq", "api"],
    ]


def test_upgrade_restarts_host_workers_on_new_code(monkeypatch):
    calls: list[list[str]] = []
    lifecycle: list[str] = []
    configured = [
        "db",
        "rabbitmq",
        "heartbeat_worker",
        "maintenance_worker",
        "api",
    ]

    async def migrated(*_args, **_kwargs):
        return 0

    monkeypatch.setattr(hexis_cli, "ensure_docker", lambda: "docker")
    monkeypatch.setattr(
        hexis_cli, "ensure_compose", lambda _docker: ["docker", "compose"]
    )
    monkeypatch.setattr(
        hexis_cli,
        "_host_managed_compose_workers",
        lambda: {"heartbeat_worker", "maintenance_worker"},
    )
    monkeypatch.setattr(
        hexis_cli,
        "_configured_compose_services",
        lambda *_args, **_kwargs: configured,
    )
    monkeypatch.setattr(
        hexis_cli,
        "_restart_installed_host_services",
        lambda: lifecycle.append("restart") or (True, None),
    )
    monkeypatch.setattr(hexis_cli, "_migrate", migrated)
    monkeypatch.setattr(
        hexis_cli,
        "run_compose",
        lambda _compose, _file, _root, args, _env: calls.append(list(args)) or 0,
    )

    assert hexis_cli.main(["upgrade"]) == 0
    assert lifecycle == ["restart"]
    assert calls == [
        ["build", "db", "rabbitmq", "api"],
        ["up", "-d", "db", "rabbitmq", "api"],
    ]
