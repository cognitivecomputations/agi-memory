import webbrowser
from pathlib import Path
from types import SimpleNamespace


def _prepare_ui_tree(tmp_path: Path) -> Path:
    from apps import hexis_cli

    stack_root = tmp_path / "hexis"
    ui_dir = stack_root / "hexis-ui"
    next_bin = ui_dir / "node_modules" / ".bin" / "next"
    next_bin.parent.mkdir(parents=True)
    next_bin.write_text("next", encoding="utf-8")
    prisma_client = ui_dir / "node_modules" / ".prisma" / "client" / "index.js"
    prisma_client.parent.mkdir(parents=True)
    prisma_client.write_text("prisma", encoding="utf-8")
    (ui_dir / "package.json").write_text("{}", encoding="utf-8")
    (ui_dir / "package-lock.json").write_text("{}", encoding="utf-8")
    hexis_cli._mark_ui_dependencies_ready(ui_dir)
    return stack_root


def test_ui_dependency_stamp_detects_manifest_changes(tmp_path):
    from apps import hexis_cli

    stack_root = _prepare_ui_tree(tmp_path)
    ui_dir = stack_root / "hexis-ui"

    assert hexis_cli._ui_dependencies_ready(ui_dir) is True

    (ui_dir / "package.json").write_text('{"changed":true}', encoding="utf-8")

    assert hexis_cli._ui_dependencies_ready(ui_dir) is False


def test_handle_ui_uses_npm_ci_and_reports_retry(monkeypatch, tmp_path, capsys):
    from apps import hexis_cli

    stack_root = tmp_path / "hexis"
    ui_dir = stack_root / "hexis-ui"
    ui_dir.mkdir(parents=True)
    (ui_dir / "package.json").write_text("{}", encoding="utf-8")
    (ui_dir / "package-lock.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(
        hexis_cli.shutil,
        "which",
        lambda name: "/usr/bin/npm" if name == "npm" else None,
    )

    def fake_run(command, *, cwd, **_kwargs):
        calls.append((list(command), cwd))
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(hexis_cli.subprocess, "run", fake_run)

    assert hexis_cli._handle_ui(stack_root, 3477, no_open=True) == 1
    assert calls == [(["/usr/bin/npm", "ci"], ui_dir)]
    assert "run `hexis ui` to retry" in capsys.readouterr().err


def test_post_init_handoff_recovers_from_dashboard_failure(monkeypatch, capsys):
    from apps import hexis_cli, hexis_init

    choices = iter([2, 3])
    calls: list[list[str]] = []

    async def fake_prompt_choice(*_args, **_kwargs):
        return next(choices)

    def fake_main(args):
        calls.append(list(args))
        return 1

    monkeypatch.setattr(hexis_init, "_prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(hexis_cli, "main", fake_main)

    assert hexis_init._post_init_handoff() == 0
    assert calls == [["ui"]]
    assert "Retry, open chat, or exit" in capsys.readouterr().out


def test_post_init_handoff_preserves_ctrl_c_exit(monkeypatch, capsys):
    from apps import hexis_cli, hexis_init

    prompts = 0

    async def fake_prompt_choice(*_args, **_kwargs):
        nonlocal prompts
        prompts += 1
        return 2

    monkeypatch.setattr(hexis_init, "_prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(hexis_cli, "main", lambda _args: 130)

    assert hexis_init._post_init_handoff() == 130
    assert prompts == 1
    assert "Retry, open chat, or exit" not in capsys.readouterr().out


def test_handle_ui_runs_locked_local_next_without_telemetry(monkeypatch, tmp_path):
    from apps import hexis_cli

    stack_root = _prepare_ui_tree(tmp_path)
    ui_dir = stack_root / "hexis-ui"
    calls: list[tuple[list[str], dict[str, str]]] = []

    monkeypatch.setattr(
        hexis_cli.shutil,
        "which",
        lambda name: "/usr/bin/npm" if name == "npm" else None,
    )
    monkeypatch.setattr(hexis_cli, "resolve_instance", lambda: None)
    monkeypatch.setattr(
        hexis_cli,
        "db_dsn_from_env",
        lambda *_args, **_kwargs: "postgresql://hexis",
    )
    monkeypatch.setattr(hexis_cli, "resolve_env_file", lambda _root: None)
    monkeypatch.setattr(
        hexis_cli, "_uses_local_embedding_sidecar", lambda _env_file: False
    )
    monkeypatch.setattr(
        hexis_cli, "_warn_legacy_embedding_sidecar_port", lambda _env_file: None
    )
    monkeypatch.setattr(hexis_cli, "_http_ready", lambda _url: False)
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: False)
    monkeypatch.setenv("HEXIS_API_URL", "https://hexis.example")

    def fake_run(command, *, cwd, env, **_kwargs):
        assert cwd == ui_dir
        calls.append((list(command), dict(env)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hexis_cli.subprocess, "run", fake_run)

    assert hexis_cli._handle_ui(stack_root, 3477, no_open=True) == 0
    assert calls[0][0] == [
        str(ui_dir / "node_modules" / ".bin" / "next"),
        "dev",
        "-p",
        "3477",
    ]
    assert calls[0][1]["NEXT_TELEMETRY_DISABLED"] == "1"


def test_handle_ui_opens_running_dashboard(monkeypatch, tmp_path):
    from apps import hexis_cli

    stack_root = _prepare_ui_tree(tmp_path)
    opened: list[str] = []

    monkeypatch.setattr(
        hexis_cli.shutil,
        "which",
        lambda name: "/usr/bin/npm" if name == "npm" else None,
    )
    monkeypatch.setattr(hexis_cli, "resolve_instance", lambda: None)
    monkeypatch.setattr(
        hexis_cli, "db_dsn_from_env", lambda *_args, **_kwargs: "postgresql://hexis"
    )
    monkeypatch.setattr(hexis_cli, "resolve_env_file", lambda _root: None)
    monkeypatch.setattr(
        hexis_cli, "_uses_local_embedding_sidecar", lambda _env_file: False
    )
    monkeypatch.setattr(
        hexis_cli, "_warn_legacy_embedding_sidecar_port", lambda _env_file: None
    )
    monkeypatch.setattr(
        hexis_cli, "_http_ready", lambda url: url.endswith(":3477/chat")
    )
    monkeypatch.setattr(
        hexis_cli, "_port_listener_summary", lambda _port: "node (pid 123)"
    )
    monkeypatch.setattr(webbrowser, "open", opened.append)
    monkeypatch.setenv("HEXIS_API_URL", "https://hexis.example")
    monkeypatch.setattr(
        hexis_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dev server should not start")
        ),
    )

    assert hexis_cli._handle_ui(stack_root, 3477, no_open=False) == 0
    assert opened == ["http://localhost:3477/chat"]


def test_handle_ui_reports_occupied_non_dashboard_port(monkeypatch, tmp_path):
    from apps import hexis_cli

    stack_root = _prepare_ui_tree(tmp_path)

    monkeypatch.setattr(
        hexis_cli.shutil,
        "which",
        lambda name: "/usr/bin/npm" if name == "npm" else None,
    )
    monkeypatch.setattr(hexis_cli, "resolve_instance", lambda: None)
    monkeypatch.setattr(
        hexis_cli, "db_dsn_from_env", lambda *_args, **_kwargs: "postgresql://hexis"
    )
    monkeypatch.setattr(hexis_cli, "resolve_env_file", lambda _root: None)
    monkeypatch.setattr(
        hexis_cli, "_uses_local_embedding_sidecar", lambda _env_file: False
    )
    monkeypatch.setattr(
        hexis_cli, "_warn_legacy_embedding_sidecar_port", lambda _env_file: None
    )
    monkeypatch.setattr(hexis_cli, "_http_ready", lambda _url: False)
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: True)
    monkeypatch.setattr(
        hexis_cli, "_port_listener_summary", lambda _port: "other-server (pid 456)"
    )
    monkeypatch.setenv("HEXIS_API_URL", "https://hexis.example")
    monkeypatch.setattr(
        hexis_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dev server should not start")
        ),
    )

    assert hexis_cli._handle_ui(stack_root, 3477, no_open=True) == 1


def test_handle_ui_container_runs_foreground_and_stops_owned_services(
    monkeypatch, tmp_path
):
    from apps import hexis_cli

    calls: list[list[str]] = []

    def fake_run_compose(_compose_cmd, _compose_file, _stack_root, args, _env_file):
        calls.append(list(args))
        return 0

    monkeypatch.setattr(
        hexis_cli, "_uses_local_embedding_sidecar", lambda _env_file: False
    )
    monkeypatch.setattr(
        hexis_cli, "_warn_legacy_embedding_sidecar_port", lambda _env_file: None
    )
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
