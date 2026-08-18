from core.cli_api import embedding_service_diagnosis


def test_embedding_service_diagnosis_prefers_published_default():
    name, steps = embedding_service_diagnosis("http://host.docker.internal:42666/api/embed")

    assert name == "embeddinggemma local sidecar"
    assert any("embeddinggemma" in step for step in steps)
    assert not any("embeddinggemma-metal" in step for step in steps)


def test_embedding_service_diagnosis_flags_legacy_port():
    name, steps = embedding_service_diagnosis("http://host.docker.internal:11434/api/embed")

    assert name == "legacy embeddinggemma sidecar configuration"
    assert any("42666" in step for step in steps)


def test_local_embedding_binary_uses_published_command(monkeypatch, tmp_path):
    from apps import hexis_cli

    binary = tmp_path / "embeddinggemma"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)

    monkeypatch.setattr(
        hexis_cli.shutil,
        "which",
        lambda name: str(binary) if name == "embeddinggemma" else None,
    )

    assert hexis_cli._local_embedding_binary() == binary


def test_local_embedding_binary_falls_back_to_installer_default(monkeypatch, tmp_path):
    from apps import hexis_cli

    home = tmp_path
    binary = home / ".local" / "bin" / "embeddinggemma"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(hexis_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(hexis_cli.Path, "home", staticmethod(lambda: home))

    assert hexis_cli._local_embedding_binary() == binary


def test_local_embedding_sidecar_detects_published_port(monkeypatch, tmp_path):
    from apps import hexis_cli

    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDING_SERVICE_URL=http://host.docker.internal:42666/api/embed\n", encoding="utf-8")
    monkeypatch.delenv("EMBEDDING_SERVICE_URL", raising=False)

    assert hexis_cli._uses_local_embedding_sidecar(env_file) is True


def test_local_embedding_sidecar_rejects_legacy_port(monkeypatch, tmp_path):
    from apps import hexis_cli

    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDING_SERVICE_URL=http://host.docker.internal:11434/api/embed\n", encoding="utf-8")
    monkeypatch.delenv("EMBEDDING_SERVICE_URL", raising=False)

    assert hexis_cli._uses_local_embedding_sidecar(env_file) is False
    assert hexis_cli._uses_legacy_embedding_sidecar_port(env_file) is True


def test_start_local_embedding_launches_installed_binary_directly(monkeypatch, tmp_path):
    from apps import hexis_cli

    binary = tmp_path / "embeddinggemma"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    log_path = tmp_path / "embeddinggemma.log"
    launched: list[list[str]] = []

    class FakeProc:
        returncode = 0

        def poll(self):
            return 0

    def fake_which(name: str):
        if name == "embeddinggemma":
            return str(binary)
        return None

    def fake_popen(args, **_kwargs):
        launched.append(list(args))
        return FakeProc()

    port_checks = iter([False, True])

    monkeypatch.setattr(hexis_cli.sys, "platform", "darwin")
    monkeypatch.setattr(hexis_cli.shutil, "which", fake_which)
    monkeypatch.setattr(hexis_cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: next(port_checks))

    assert hexis_cli._start_local_embedding_service(wait_seconds=1) is True
    assert launched
    assert launched[0] == [str(binary)]


def test_start_local_embedding_installs_missing_binary(monkeypatch, tmp_path):
    from apps import hexis_cli

    home = tmp_path
    binary = home / ".local" / "bin" / "embeddinggemma"
    log_path = tmp_path / "embeddinggemma.log"
    installs: list[list[str]] = []
    launched: list[list[str]] = []

    class FakeResult:
        returncode = 0

    class FakeProc:
        returncode = 0

        def poll(self):
            return None

    def fake_run(args, **_kwargs):
        installs.append(list(args))
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        return FakeResult()

    def fake_popen(args, **_kwargs):
        launched.append(list(args))
        return FakeProc()

    port_checks = iter([False, True])

    monkeypatch.setattr(hexis_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(hexis_cli.Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(hexis_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(hexis_cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: next(port_checks))

    assert hexis_cli._start_local_embedding_service(wait_seconds=1) is True
    assert installs == [["sh", "-c", hexis_cli._LOCAL_EMBEDDING_INSTALLER]]
    assert launched == [[str(binary)]]


def test_start_local_embedding_reports_failed_install(monkeypatch, tmp_path):
    from apps import hexis_cli

    home = tmp_path
    launched: list[list[str]] = []

    class FakeResult:
        returncode = 1

    monkeypatch.setattr(hexis_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(hexis_cli.Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(hexis_cli.subprocess, "run", lambda *_args, **_kwargs: FakeResult())
    monkeypatch.setattr(
        hexis_cli.subprocess,
        "Popen",
        lambda *args, **kwargs: launched.append(args) or None,
    )
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: False)

    assert hexis_cli._start_local_embedding_service(wait_seconds=1) is False
    assert launched == []


def test_start_local_embedding_reports_early_exit(monkeypatch, tmp_path):
    from apps import hexis_cli

    binary = tmp_path / "embeddinggemma"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    log_path = tmp_path / "embeddinggemma.log"
    log_path.write_text("loading model\nfatal: no model\n", encoding="utf-8")

    class FakeProc:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(hexis_cli.shutil, "which", lambda name: str(binary) if name == "embeddinggemma" else None)
    monkeypatch.setattr(hexis_cli.subprocess, "Popen", lambda *_args, **_kwargs: FakeProc())
    monkeypatch.setattr(hexis_cli, "_LOCAL_EMBEDDING_LOG", log_path)
    monkeypatch.setattr(hexis_cli, "_port_ready", lambda _port: False)

    assert hexis_cli._start_local_embedding_service(wait_seconds=1) is False
