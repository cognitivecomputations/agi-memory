import webbrowser
from pathlib import Path


def _prepare_ui_tree(tmp_path: Path) -> Path:
    stack_root = tmp_path / "hexis"
    ui_dir = stack_root / "hexis-ui"
    (ui_dir / "node_modules").mkdir(parents=True)
    return stack_root


def test_handle_ui_opens_running_dashboard(monkeypatch, tmp_path):
    from apps import hexis_cli

    stack_root = _prepare_ui_tree(tmp_path)
    opened: list[str] = []

    monkeypatch.setattr(hexis_cli.shutil, "which", lambda name: "/usr/bin/bun" if name == "bun" else None)
    monkeypatch.setattr(hexis_cli, "resolve_instance", lambda: None)
    monkeypatch.setattr(hexis_cli, "db_dsn_from_env", lambda *_args, **_kwargs: "postgresql://hexis")
    monkeypatch.setattr(hexis_cli, "resolve_env_file", lambda _root: None)
    monkeypatch.setattr(hexis_cli, "_uses_local_embedding_sidecar", lambda _env_file: False)
    monkeypatch.setattr(hexis_cli, "_warn_legacy_embedding_sidecar_port", lambda _env_file: None)
    monkeypatch.setattr(hexis_cli, "_http_ready", lambda url: url.endswith(":3477/chat"))
    monkeypatch.setattr(hexis_cli, "_port_listener_summary", lambda _port: "node (pid 123)")
    monkeypatch.setattr(webbrowser, "open", opened.append)
    monkeypatch.setenv("HEXIS_API_URL", "https://hexis.example")
    monkeypatch.setattr(
        hexis_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dev server should not start")),
    )

    assert hexis_cli._handle_ui(stack_root, 3477, no_open=False) == 0
    assert opened == ["http://localhost:3477/chat"]


def test_handle_ui_reports_occupied_non_dashboard_port(monkeypatch, tmp_path):
    from apps import hexis_cli

    stack_root = _prepare_ui_tree(tmp_path)

    monkeypatch.setattr(hexis_cli.shutil, "which", lambda name: "/usr/bin/bun" if name == "bun" else None)
    monkeypatch.setattr(hexis_cli, "resolve_instance", lambda: None)
    monkeypatch.setattr(hexis_cli, "db_dsn_from_env", lambda *_args, **_kwargs: "postgresql://hexis")
    monkeypatch.setattr(hexis_cli, "resolve_env_file", lambda _root: None)
    monkeypatch.setattr(hexis_cli, "_uses_local_embedding_sidecar", lambda _env_file: False)
    monkeypatch.setattr(hexis_cli, "_warn_legacy_embedding_sidecar_port", lambda _env_file: None)
    monkeypatch.setattr(hexis_cli, "_http_ready", lambda _url: False)
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: True)
    monkeypatch.setattr(hexis_cli, "_port_listener_summary", lambda _port: "other-server (pid 456)")
    monkeypatch.setenv("HEXIS_API_URL", "https://hexis.example")
    monkeypatch.setattr(
        hexis_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dev server should not start")),
    )

    assert hexis_cli._handle_ui(stack_root, 3477, no_open=True) == 1


def test_handle_ui_container_runs_foreground_and_stops_owned_services(monkeypatch, tmp_path):
    from apps import hexis_cli

    calls: list[list[str]] = []

    def fake_run_compose(_compose_cmd, _compose_file, _stack_root, args, _env_file):
        calls.append(list(args))
        return 0

    monkeypatch.setattr(hexis_cli, "_uses_local_embedding_sidecar", lambda _env_file: False)
    monkeypatch.setattr(hexis_cli, "_warn_legacy_embedding_sidecar_port", lambda _env_file: None)
    monkeypatch.setattr(hexis_cli, "_http_ready", lambda _url: False)
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: False)
    monkeypatch.setattr(hexis_cli, "run_compose", fake_run_compose)

    rc = hexis_cli._handle_ui_container(
        ["docker", "compose"],
        tmp_path / "docker-compose.yml",
        tmp_path,
        None,
        3477,
        no_open=True,
    )

    assert rc == 0
    # The always-on loops come up first, detached, and are never stopped:
    # closing the dashboard must not stop the agent from thinking.
    assert calls[0] == ["up", "-d", "heartbeat_worker", "maintenance_worker"]
    assert calls[1] == ["up", "api", "ui"]
    assert calls[-1] == ["stop", "ui", "api"]
    assert not any(
        "heartbeat_worker" in call or "maintenance_worker" in call
        for call in calls
        if call and call[0] == "stop"
    )
    # The dashboard itself still runs in the foreground.
    assert "-d" not in calls[1]
