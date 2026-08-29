import pytest

from apps import hexis_cli


class _Result:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode


def test_uninstall_parser_preserves_data_by_default():
    args = hexis_cli.build_parser().parse_args(["uninstall"])

    assert args.func == "uninstall"
    assert args.purge is False
    assert args.cli_only is False
    assert args.yes is False


def test_uninstall_parser_keeps_destructive_and_cli_only_modes_exclusive():
    parser = hexis_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["uninstall", "--purge", "--cli-only"])


def test_package_uninstaller_is_derived_from_uv_tool_root(monkeypatch, tmp_path):
    tool_root = tmp_path / "uv-tools"
    prefix = tool_root / "hexis"
    monkeypatch.setattr(hexis_cli.sys, "prefix", str(prefix))
    monkeypatch.setattr(
        hexis_cli,
        "_find_uninstall_program",
        lambda name: "/usr/local/bin/uv" if name == "uv" else None,
    )
    monkeypatch.setattr(hexis_cli, "_capture_path", lambda _command: tool_root)

    command, manager = hexis_cli._package_uninstall_command()

    assert command == ["/usr/local/bin/uv", "tool", "uninstall", "hexis"]
    assert manager == "uv"


def test_package_uninstaller_falls_back_to_current_interpreter_pip(monkeypatch):
    monkeypatch.setattr(hexis_cli, "_find_uninstall_program", lambda _name: None)

    command, manager = hexis_cli._package_uninstall_command()

    assert command == [
        hexis_cli.sys.executable,
        "-m",
        "pip",
        "uninstall",
        "--yes",
        "hexis",
    ]
    assert manager == "pip"


def _patch_successful_uninstall(monkeypatch, tmp_path):
    from core import host_services

    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    data_dir = tmp_path / ".hexis"
    data_dir.mkdir()
    (data_dir / "config.json").write_text("{}\n", encoding="utf-8")
    compose_calls: list[list[str]] = []
    package_calls: list[list[str]] = []

    monkeypatch.setattr(hexis_cli, "_hexis_data_dir", lambda: data_dir)
    monkeypatch.setattr(
        hexis_cli,
        "_package_uninstall_command",
        lambda: (["uv", "tool", "uninstall", "hexis"], "uv"),
    )
    monkeypatch.setattr(hexis_cli, "ensure_docker", lambda: "docker")
    monkeypatch.setattr(
        hexis_cli, "ensure_compose", lambda _docker: ["docker", "compose"]
    )
    monkeypatch.setattr(
        hexis_cli,
        "run_compose",
        lambda _cmd, _file, _root, args, _env: compose_calls.append(list(args)) or 0,
    )
    monkeypatch.setattr(
        hexis_cli,
        "_stop_owned_local_embedding_service",
        lambda: (True, None),
    )
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: False)
    monkeypatch.setattr(
        hexis_cli, "_purge_owned_local_embedding_assets", lambda: ([], [])
    )
    monkeypatch.setattr(hexis_cli, "_local_embedding_binary", lambda: None)
    monkeypatch.setattr(host_services, "installed_host_services", lambda: [])
    monkeypatch.setattr(
        hexis_cli.subprocess,
        "run",
        lambda command, **_kwargs: package_calls.append(list(command)) or _Result(),
    )
    return compose_file, data_dir, compose_calls, package_calls


def test_uninstall_removes_runtime_but_preserves_data_by_default(
    monkeypatch, tmp_path, capsys
):
    compose_file, data_dir, compose_calls, package_calls = _patch_successful_uninstall(
        monkeypatch, tmp_path
    )

    rc = hexis_cli._uninstall(
        compose_file=compose_file,
        stack_root=tmp_path,
        env_file=None,
        is_source=False,
        purge=False,
        cli_only=False,
        yes=True,
    )

    assert rc == 0
    assert compose_calls == [["down", "--remove-orphans", "--rmi", "all"]]
    assert package_calls == [["uv", "tool", "uninstall", "hexis"]]
    assert data_dir.exists()
    output = capsys.readouterr().out
    assert "brain database volumes" in output
    assert "preserved" in output


def test_uninstall_purge_requires_explicit_mode_and_deletes_data(monkeypatch, tmp_path):
    compose_file, data_dir, compose_calls, package_calls = _patch_successful_uninstall(
        monkeypatch, tmp_path
    )

    rc = hexis_cli._uninstall(
        compose_file=compose_file,
        stack_root=tmp_path,
        env_file=None,
        is_source=False,
        purge=True,
        cli_only=False,
        yes=True,
    )

    assert rc == 0
    assert compose_calls == [["down", "--remove-orphans", "--rmi", "all", "--volumes"]]
    assert package_calls == [["uv", "tool", "uninstall", "hexis"]]
    assert not data_dir.exists()


def test_cli_only_uninstall_removes_host_units_before_package(
    monkeypatch, tmp_path, capsys
):
    from core import host_services

    compose_file, data_dir, compose_calls, package_calls = _patch_successful_uninstall(
        monkeypatch, tmp_path
    )
    service_calls: list[list[str]] = []
    monkeypatch.setattr(
        host_services,
        "installed_host_services",
        lambda: ["heartbeat", "maintenance"],
    )
    monkeypatch.setattr(
        host_services,
        "uninstall_host_services",
        lambda services: (
            service_calls.append(list(services)) or {"uninstalled": list(services)}
        ),
    )

    rc = hexis_cli._uninstall(
        compose_file=compose_file,
        stack_root=tmp_path,
        env_file=None,
        is_source=False,
        purge=False,
        cli_only=True,
        yes=True,
    )

    assert rc == 0
    assert compose_calls == []
    assert service_calls == [["heartbeat", "maintenance"]]
    assert package_calls == [["uv", "tool", "uninstall", "hexis"]]
    assert data_dir.exists()
    assert "Host-service logs were preserved" in capsys.readouterr().out


def test_purge_confirmation_names_permanent_data_loss(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: "uninstall and delete data")

    assert hexis_cli._confirm_uninstall(
        purge=True, cli_only=False, data_dir=tmp_path / ".hexis"
    )
    output = capsys.readouterr().out
    assert "PERMANENTLY DELETE the brain database volumes" in output
    assert "hexis backup" in output


def test_uninstall_keeps_cli_when_docker_is_unavailable(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    package_calls: list[list[str]] = []
    monkeypatch.setattr(hexis_cli, "_hexis_data_dir", lambda: tmp_path / ".hexis")
    monkeypatch.setattr(
        hexis_cli,
        "_package_uninstall_command",
        lambda: (["uv", "tool", "uninstall", "hexis"], "uv"),
    )

    def no_docker():
        raise SystemExit(1)

    monkeypatch.setattr(hexis_cli, "ensure_docker", no_docker)
    monkeypatch.setattr(
        hexis_cli.subprocess,
        "run",
        lambda command, **_kwargs: package_calls.append(list(command)) or _Result(),
    )

    rc = hexis_cli._uninstall(
        compose_file=compose_file,
        stack_root=tmp_path,
        env_file=None,
        is_source=False,
        purge=False,
        cli_only=False,
        yes=True,
    )

    assert rc == 1
    assert package_calls == []


def test_purge_refuses_symlinked_data_directory(tmp_path):
    target = tmp_path / "actual-data"
    target.mkdir()
    link = tmp_path / ".hexis"
    link.symlink_to(target, target_is_directory=True)

    removed, error = hexis_cli._purge_hexis_data_dir(link)

    assert removed is False
    assert "symlinked" in (error or "")
    assert target.exists()


def test_owned_embedding_service_is_stopped_by_pid(monkeypatch, tmp_path):
    log_path = tmp_path / "embeddinggemma.log"
    pid_path = tmp_path / "embeddinggemma.pid"
    pid_path.write_text("1234\n", encoding="utf-8")
    commands = iter(["/tmp/embeddinggemma", None])
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setattr(hexis_cli, "_process_command", lambda _pid: next(commands))
    monkeypatch.setattr(
        hexis_cli.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )

    stopped, note = hexis_cli._stop_owned_local_embedding_service()

    assert stopped is True
    assert note is None
    assert signals and signals[0][0] == 1234
    assert not pid_path.exists()


def test_unowned_embedding_service_is_left_running(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hexis_cli, "_LOCAL_EMBEDDING_LOG", tmp_path / "embeddinggemma.log"
    )
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: True)
    monkeypatch.setattr(
        hexis_cli,
        "_port_listener_summary",
        lambda _port: "embeddinggemma (pid 1234)",
    )
    monkeypatch.setattr(hexis_cli, "_legacy_owned_local_embedding_pid", lambda: None)

    stopped, note = hexis_cli._stop_owned_local_embedding_service()

    assert stopped is False
    assert "left running" in (note or "")


def test_legacy_embedding_service_is_verified_by_hexis_log(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hexis_cli, "_LOCAL_EMBEDDING_LOG", tmp_path / "embeddinggemma.log"
    )
    monkeypatch.setattr(hexis_cli, "_port_listener_pids", lambda _port: [1234, 5678])
    monkeypatch.setattr(
        hexis_cli,
        "_process_command",
        lambda pid: "/tmp/embeddinggemma" if pid == 1234 else "/tmp/other",
    )
    monkeypatch.setattr(
        hexis_cli, "_process_uses_hexis_embedding_log", lambda pid: pid == 1234
    )

    assert hexis_cli._legacy_owned_local_embedding_pid() == 1234


def test_legacy_embedding_log_verification_requires_both_output_streams(
    monkeypatch, tmp_path
):
    log_path = tmp_path / "embeddinggemma.log"

    class LsofResult:
        stdout = f"p1234\nf1\nn{log_path} (deleted)\nf2\nn{log_path} (deleted)\n"

    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setattr(hexis_cli.shutil, "which", lambda _name: "/usr/bin/lsof")
    monkeypatch.setattr(
        hexis_cli.subprocess, "run", lambda _command, **_kwargs: LsofResult()
    )

    assert hexis_cli._process_uses_hexis_embedding_log(1234) is True


def test_legacy_verified_embedding_service_is_stopped(monkeypatch, tmp_path):
    log_path = tmp_path / "embeddinggemma.log"
    commands = iter(["/tmp/embeddinggemma", None])
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setattr(hexis_cli, "_legacy_owned_local_embedding_pid", lambda: 1234)
    monkeypatch.setattr(hexis_cli, "_process_command", lambda _pid: next(commands))
    monkeypatch.setattr(
        hexis_cli.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )

    stopped, note = hexis_cli._stop_owned_local_embedding_service()

    assert stopped is True
    assert note is None
    assert signals and signals[0][0] == 1234


def test_owned_embedding_binary_and_cache_are_purged(monkeypatch, tmp_path):
    log_path = tmp_path / ".hexis" / "embeddinggemma.log"
    binary = tmp_path / "bin" / "embeddinggemma"
    binary.parent.mkdir()
    binary.write_bytes(b"hexis-installed-binary")
    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "embeddinggemma.c"
    cache_dir.mkdir(parents=True)
    (cache_dir / "model.gguf").write_bytes(b"model")
    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))

    assert hexis_cli._record_owned_local_embedding_binary(binary) is True
    assert (
        hexis_cli._mark_local_embedding_cache_if_created(existed_before_start=False)
        is True
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "different-cache"))
    removed, notes = hexis_cli._purge_owned_local_embedding_assets()

    assert notes == []
    assert binary in removed
    assert cache_dir.resolve() in removed
    assert not binary.exists()
    assert not cache_dir.exists()


def test_changed_owned_embedding_binary_is_preserved(monkeypatch, tmp_path):
    log_path = tmp_path / ".hexis" / "embeddinggemma.log"
    binary = tmp_path / "bin" / "embeddinggemma"
    binary.parent.mkdir()
    binary.write_bytes(b"original")
    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert hexis_cli._record_owned_local_embedding_binary(binary) is True
    binary.write_bytes(b"replaced-by-someone-else")

    removed, notes = hexis_cli._purge_owned_local_embedding_assets()

    assert removed == []
    assert binary.exists()
    assert any("changed" in note for note in notes)


def test_embedding_cache_is_marked_only_when_created_after_start(monkeypatch, tmp_path):
    log_path = tmp_path / ".hexis" / "embeddinggemma.log"
    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "embeddinggemma.c"
    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))

    assert (
        hexis_cli._mark_local_embedding_cache_if_created(existed_before_start=False)
        is None
    )
    assert not cache_dir.exists()

    cache_dir.mkdir(parents=True)
    assert (
        hexis_cli._mark_local_embedding_cache_if_created(existed_before_start=False)
        is True
    )
    assert (cache_dir / hexis_cli._LOCAL_EMBEDDING_CACHE_MARKER).is_file()


def test_embedding_cache_with_changed_ownership_marker_is_preserved(
    monkeypatch, tmp_path
):
    log_path = tmp_path / ".hexis" / "embeddinggemma.log"
    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "embeddinggemma.c"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
    assert (
        hexis_cli._mark_local_embedding_cache_if_created(existed_before_start=False)
        is True
    )
    (cache_dir / hexis_cli._LOCAL_EMBEDDING_CACHE_MARKER).write_text(
        '{"owner":"someone-else","token":"changed"}\n', encoding="utf-8"
    )

    removed, notes = hexis_cli._purge_owned_local_embedding_assets()

    assert removed == []
    assert cache_dir.exists()
    assert any("ownership marker" in note for note in notes)
