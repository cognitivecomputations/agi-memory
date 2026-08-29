from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.execution_backends import (
    BackendRunResult,
    DockerRemoteExecutionBackend,
    ExecutionBackendError,
    ExecutionProfile,
    ExecutionSettings,
    LocalExecutionBackend,
    SSHExecutionBackend,
    SSH_PROCESS_SUPERVISOR,
    load_execution_settings,
)


def _ssh_profile(
    tmp_path: Path,
    *,
    name: str = "build",
    workspace: str = "/srv/project",
) -> ExecutionProfile:
    identity = tmp_path / "id_ed25519"
    identity.write_text("not-a-real-key")
    identity.chmod(0o600)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("build.example ssh-ed25519 AAAA\n")
    known_hosts.chmod(0o644)
    return ExecutionProfile.from_dict(
        name,
        {
            "type": "ssh",
            "host": "build.example",
            "user": "runner",
            "port": 2222,
            "workspace": workspace,
            "identity_file": str(identity),
            "known_hosts_file": str(known_hosts),
        },
    )


def _docker_profile(tmp_path: Path) -> ExecutionProfile:
    ssh = _ssh_profile(tmp_path, name="temporary")
    return ExecutionProfile.from_dict(
        "container-build",
        {
            "type": "docker_remote",
            "docker_host": "ssh://runner@build.example:2222",
            "image": "registry.example/project/build@sha256:" + "a" * 64,
            "workspace": "/srv/project",
            "identity_file": ssh.identity_file,
            "known_hosts_file": ssh.known_hosts_file,
            "network": "none",
        },
    )


def _settings(*profiles: ExecutionProfile, active: str = "local") -> ExecutionSettings:
    values = {"local": ExecutionProfile(name="local", kind="local")}
    values.update({profile.name: profile for profile in profiles})
    return ExecutionSettings(active=active, profiles=values)


def test_profiles_reject_implicit_or_risky_remote_configuration(tmp_path: Path):
    ssh = _ssh_profile(tmp_path)
    assert ssh.to_dict()["identity_file"] == str(tmp_path / "id_ed25519")
    with pytest.raises(ExecutionBackendError, match="absolute POSIX"):
        ExecutionProfile.from_dict(
            "bad",
            ssh.to_dict() | {"workspace": "~/project"},
        )
    with pytest.raises(ExecutionBackendError, match="without credentials"):
        ExecutionProfile.from_dict(
            "bad",
            {
                "type": "docker_remote",
                "docker_host": "ssh://runner:password@build.example",
                "image": "python:3.13",
                "workspace": "/srv/project",
                "identity_file": ssh.identity_file,
                "known_hosts_file": ssh.known_hosts_file,
            },
        )
    with pytest.raises(ExecutionBackendError, match="network"):
        ExecutionProfile.from_dict(
            "bad",
            {
                "type": "docker_remote",
                "docker_host": "ssh://runner@build.example",
                "image": "python:3.13",
                "workspace": "/srv/project",
                "identity_file": ssh.identity_file,
                "known_hosts_file": ssh.known_hosts_file,
                "network": "host",
            },
        )


def test_settings_require_real_active_profile_and_builtin_local():
    with pytest.raises(ExecutionBackendError, match="does not exist"):
        ExecutionSettings.from_values(
            {"active": "missing", "profiles": {"local": {"type": "local"}}}
        )
    with pytest.raises(ExecutionBackendError, match="include built-in"):
        ExecutionSettings.from_values({"active": "remote", "profiles": {}})


@pytest.mark.asyncio
async def test_local_backend_runs_shell_argv_and_portable_python(tmp_path: Path):
    backend = LocalExecutionBackend(
        ExecutionProfile(name="local", kind="local"),
        _settings(),
        str(tmp_path),
    )
    shell = await backend.run_shell(
        "printf '%s' \"$HEXIS_EXEC_TEST\"", timeout=5, env={"HEXIS_EXEC_TEST": "ok"}
    )
    assert shell.returncode == 0
    assert shell.stdout == b"ok"
    argv = await backend.run_argv(["python3", "-c", "print('argv-ok')"], timeout=5)
    assert argv.returncode == 0
    assert argv.stdout.strip() == b"argv-ok"
    remote_protocol = await backend.run_python(
        "value = 6 * 7\nprint(FINAL_VAR('value'))", session_id="portable", timeout=5
    )
    payload = json.loads(remote_protocol.stdout)
    assert payload["stdout"] == "42"
    assert payload["variables"]["value"] == "int"


@pytest.mark.asyncio
async def test_local_timeout_is_bounded_and_kills_process(tmp_path: Path):
    backend = LocalExecutionBackend(
        ExecutionProfile(name="local", kind="local"),
        _settings(),
        str(tmp_path),
    )
    result = await backend.run_argv(
        ["python3", "-c", "import time; time.sleep(10)"], timeout=1
    )
    assert result.timed_out is True
    assert result.returncode != 0


@pytest.mark.asyncio
async def test_ssh_supervisor_terminates_exact_remote_process_group(tmp_path: Path):
    backend = LocalExecutionBackend(
        ExecutionProfile(name="local", kind="local"),
        _settings(),
        str(tmp_path),
    )
    request = json.dumps(
        {
            "command": "python3 -c 'import time; time.sleep(10)'",
            "argv": None,
            "env": {},
            "input": "",
            "timeout": 1,
            "max_output_bytes": 10_000,
        }
    ).encode()
    result = await backend.run_argv(
        ["python3", "-c", SSH_PROCESS_SUPERVISOR],
        timeout=6,
        input_bytes=request,
    )
    assert result.returncode == 0
    response = json.loads(result.stdout)
    assert response["timed_out"] is True
    assert response["returncode"] != 0


class _BrokenAcquire:
    async def __aenter__(self):
        raise RuntimeError("database unavailable")

    async def __aexit__(self, *_args):
        return False


class _BrokenPool:
    def acquire(self):
        return _BrokenAcquire()


@pytest.mark.asyncio
async def test_database_failure_refuses_silent_local_fallback():
    with pytest.raises(ExecutionBackendError, match="refused to run locally"):
        await load_execution_settings(pool=_BrokenPool())


def test_ssh_command_uses_only_exact_identity_and_known_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    profile = _ssh_profile(tmp_path)
    backend = SSHExecutionBackend(profile, _settings(profile), str(tmp_path))
    monkeypatch.setattr(
        "core.execution_backends.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    argv = backend._ssh_argv("true", 30)
    assert argv[:3] == ["/usr/bin/ssh", "-F", "/dev/null"]
    assert "BatchMode=yes" in argv
    assert "IdentitiesOnly=yes" in argv
    assert "StrictHostKeyChecking=yes" in argv
    assert f"UserKnownHostsFile={profile.known_hosts_file}" in argv
    assert argv[-2:] == ["runner@build.example", "true"]
    command = backend._remote_command(
        "printf '%s' \"$VALUE\"", {"VALUE": "safe; touch /tmp/not-run"}
    )
    assert "cd -- /srv/project" in command
    assert "VALUE='safe; touch /tmp/not-run'" in command


@pytest.mark.asyncio
async def test_ssh_backend_protocol_and_target_timeout_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n"
        "payload = sys.stdin.buffer.read()\n"
        "result = subprocess.run(['/bin/sh', '-lc', sys.argv[-1]], input=payload, capture_output=True)\n"
        "sys.stdout.buffer.write(result.stdout)\n"
        "sys.stderr.buffer.write(result.stderr)\n"
        "raise SystemExit(result.returncode)\n"
    )
    fake_ssh.chmod(0o755)
    workspace = tmp_path / "remote-workspace"
    workspace.mkdir()
    profile = _ssh_profile(tmp_path, workspace=str(workspace))
    backend = SSHExecutionBackend(profile, _settings(profile), str(tmp_path))
    monkeypatch.setattr(
        "core.execution_backends.shutil.which",
        lambda name: str(fake_ssh) if name == "ssh" else f"/usr/bin/{name}",
    )
    success = await backend.run_shell(
        "printf '%s' \"$VALUE\"", timeout=5, env={"VALUE": "remote-ok"}
    )
    assert success.returncode == 0
    assert success.stdout == b"remote-ok"
    timeout = await backend.run_shell(
        "python3 -c 'import time; time.sleep(10)'", timeout=1
    )
    assert timeout.timed_out is True
    assert "exact command process group" in (timeout.timeout_detail or "")


def test_identity_permissions_fail_loudly(tmp_path: Path):
    profile = _ssh_profile(tmp_path)
    Path(profile.identity_file or "").chmod(0o644)
    backend = SSHExecutionBackend(profile, _settings(profile), str(tmp_path))
    with pytest.raises(ExecutionBackendError, match="chmod 600"):
        backend._ssh_argv("true", 10)


def test_remote_docker_is_ephemeral_owned_and_never_pulls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    profile = _docker_profile(tmp_path)
    backend = DockerRemoteExecutionBackend(profile, _settings(profile), str(tmp_path))
    monkeypatch.setattr(
        "core.execution_backends.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    argv, env = backend._run_argv(
        ["/bin/sh", "-lc", "pwd"], {"VALUE": "hello world"}, "hexis-exec-exact"
    )
    assert argv[:4] == [
        "/usr/bin/docker",
        "--host",
        "ssh://runner@build.example:2222",
        "run",
    ]
    assert "--rm" in argv
    assert argv[argv.index("--name") + 1] == "hexis-exec-exact"
    assert argv[argv.index("--pull") + 1] == "never"
    assert argv[argv.index("--network") + 1] == "none"
    assert "io.hexis.execution=true" in argv
    assert "VALUE=hello world" in argv
    assert profile.state_volume in " ".join(argv)
    assert "-F /dev/null" in env["DOCKER_SSH_COMMAND"]
    assert "StrictHostKeyChecking=yes" in env["DOCKER_SSH_COMMAND"]


@pytest.mark.asyncio
async def test_remote_docker_timeout_removes_only_the_owned_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_docker = tmp_path / "docker"
    log_path = tmp_path / "docker.log"
    fake_docker.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys, time\n"
        "args = sys.argv[1:]\n"
        "log = pathlib.Path(os.environ['FAKE_DOCKER_LOG'])\n"
        "if 'run' in args:\n"
        "    name = args[args.index('--name') + 1]\n"
        "    with log.open('a') as handle: handle.write('run ' + name + '\\n')\n"
        "    if 'sleep-long' in args: time.sleep(10)\n"
        "    else: print('docker-ok')\n"
        "elif 'rm' in args:\n"
        "    with log.open('a') as handle: handle.write('rm ' + args[-1] + '\\n')\n"
        "    print(args[-1])\n"
    )
    fake_docker.chmod(0o755)
    profile = _docker_profile(tmp_path)
    backend = DockerRemoteExecutionBackend(profile, _settings(profile), str(tmp_path))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log_path))
    monkeypatch.setattr(
        "core.execution_backends.shutil.which",
        lambda name: str(fake_docker) if name == "docker" else f"/usr/bin/{name}",
    )
    success = await backend.run_argv(["echo", "ok"], timeout=5)
    assert success.returncode == 0
    assert success.stdout.strip() == b"docker-ok"
    timed_out = await backend.run_argv(["sleep-long"], timeout=1)
    assert timed_out.timed_out is True
    assert "exact Hexis-owned remote container was removed" in (
        timed_out.timeout_detail or ""
    )
    lines = log_path.read_text().splitlines()
    run_names = [line.removeprefix("run ") for line in lines if line.startswith("run ")]
    removed = [line.removeprefix("rm ") for line in lines if line.startswith("rm ")]
    assert len(run_names) == 2
    assert removed == [run_names[-1]]


def test_profile_status_does_not_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from apps.cli_execution import _profile_status

    profile = _ssh_profile(tmp_path)
    monkeypatch.setattr(
        "apps.cli_execution.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    status = _profile_status(profile, active=True)
    assert status["active"] is True
    assert status["locally_ready"] is True
    assert status["config"]["host"] == "build.example"


def test_cli_mutations_never_overwrite_or_remove_active_profile(tmp_path: Path):
    from apps.cli_execution import _add_profile_mutation

    profile = _ssh_profile(tmp_path)
    settings = _settings(profile, active=profile.name)
    with pytest.raises(ExecutionBackendError, match="already exists"):
        _add_profile_mutation(profile, replace=False)(settings)


class _FakeRemoteBackend:
    name = "build"
    kind = "ssh"
    settings = ExecutionSettings()

    def cap_timeout(self, timeout: int) -> int:
        return timeout

    async def run_python(self, code: str, *, session_id: str, timeout: int):
        assert code == "print(42)"
        assert session_id == "remote-session"
        return BackendRunResult(
            returncode=0,
            stdout=json.dumps(
                {
                    "stdout": "42",
                    "stderr": "",
                    "variables": {"answer": "int"},
                    "not_persisted": [],
                    "execution_time": 0.01,
                }
            ).encode(),
            stderr=b"",
        )


@pytest.mark.asyncio
async def test_execute_code_keeps_schema_and_routes_remote(
    monkeypatch: pytest.MonkeyPatch,
):
    from core.tools.base import ToolContext, ToolExecutionContext
    from core.tools.code_execution import CodeExecutionHandler

    async def resolve(*_args, **_kwargs):
        return _FakeRemoteBackend()

    monkeypatch.setattr("core.tools.code_execution.resolve_execution_backend", resolve)
    handler = CodeExecutionHandler()
    assert set(handler.spec.parameters["properties"]) == {"code", "timeout"}
    result = await handler.execute(
        {"code": "print(42)"},
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="call",
            session_id="remote-session",
        ),
    )
    assert result.success is True
    assert result.output["backend"] == "build"
    assert result.output["stdout"] == "42"


@pytest.mark.asyncio
async def test_shell_keeps_schema_and_routes_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    from core.tools.base import ToolContext, ToolExecutionContext
    from core.tools.shell import ShellHandler

    class FakeBackend:
        name = "build"
        kind = "ssh"
        settings = ExecutionSettings(max_output_chars=5)

        def cap_timeout(self, timeout: int) -> int:
            return timeout

        async def run_shell(self, command: str, *, timeout: int, env: Any):
            assert command == "printf hello"
            return BackendRunResult(0, b"1234567", b"")

    async def resolve(*_args, **_kwargs):
        return FakeBackend()

    monkeypatch.setattr("core.tools.shell.resolve_execution_backend", resolve)
    handler = ShellHandler()
    assert set(handler.spec.parameters["properties"]) == {"command", "timeout", "env"}
    result = await handler.execute(
        {"command": "printf hello"},
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="call",
            allow_shell=True,
        ),
    )
    assert result.success is True
    assert result.output["backend"] == "build"
    assert result.output["truncated"] is True
    assert result.output["stdout"].endswith("...[truncated]")


@pytest.mark.asyncio
async def test_safe_shell_uses_direct_argv_so_shell_operators_are_inert(tmp_path: Path):
    from core.tools.base import ToolContext, ToolExecutionContext
    from core.tools.shell import SafeShellHandler

    marker = tmp_path / "must-not-exist"
    result = await SafeShellHandler().execute(
        {"command": f"echo harmless > {marker}"},
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="safe-direct-argv",
            workspace_path=str(tmp_path),
            allow_shell=True,
        ),
    )
    assert result.success is True
    assert marker.exists() is False
    assert ">" in result.output["stdout"]


@pytest.mark.parametrize(
    "command, expected",
    [
        ("find . -delete", "Mutating find"),
        ("date --set=tomorrow", "system time"),
        ("sort -o output.txt input.txt", "output files"),
        ("git branch new-branch", "branch creation"),
        ("git tag release", "tag creation"),
        ("git remote remove origin", "read-only git remote"),
        ("env rm file", "not in safe commands"),
    ],
)
def test_safe_shell_rejects_mutating_program_modes(command: str, expected: str):
    from core.tools.shell import SafeShellHandler

    allowed, reason = SafeShellHandler()._is_command_allowed(command)
    assert allowed is False
    assert expected in (reason or "")


def test_safe_shell_keeps_useful_read_only_git_forms():
    from core.tools.shell import SafeShellHandler

    handler = SafeShellHandler()
    for command in (
        "git status --short",
        "git branch --list 'feature/*'",
        "git tag --list 'v*'",
        "git remote -v",
        "git remote get-url origin",
    ):
        assert handler._is_command_allowed(command) == (True, None)


@pytest.mark.asyncio
async def test_run_script_maps_only_the_workspace_relative_path_remotely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from core.tools.base import ToolContext, ToolExecutionContext
    from core.tools.shell import ScriptRunnerHandler

    script = tmp_path / "jobs" / "build.py"
    script.parent.mkdir()
    script.write_text("print('remote')\n")
    captured: dict[str, Any] = {}

    class FakeBackend:
        name = "build"
        kind = "ssh"
        settings = ExecutionSettings()

        def cap_timeout(self, timeout: int) -> int:
            return timeout

        def path_for(self, relative: str) -> str:
            captured["relative"] = relative
            return "/srv/project/" + relative

        async def run_argv(self, argv: list[str], *, timeout: int):
            captured["argv"] = argv
            return BackendRunResult(0, b"remote\n", b"")

    async def resolve(*_args, **_kwargs):
        return FakeBackend()

    monkeypatch.setattr("core.tools.shell.resolve_execution_backend", resolve)
    result = await ScriptRunnerHandler().execute(
        {"path": "jobs/build.py", "args": ["value;still-one-arg"]},
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="remote-script",
            workspace_path=str(tmp_path),
            allow_shell=True,
        ),
    )
    assert result.success is True
    assert captured["relative"] == "jobs/build.py"
    assert captured["argv"] == [
        "python3",
        "/srv/project/jobs/build.py",
        "value;still-one-arg",
    ]
