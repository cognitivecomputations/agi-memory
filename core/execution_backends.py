"""Explicit execution backends for shell, script, and Python tools.

The active profile is database-owned.  A configured remote profile must never
silently degrade to local execution: doing so would run an approved command on
the wrong machine, which is a materially different action.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import posixpath
import re
import shlex
import shutil
import stat
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from core.tools.base import ToolExecutionContext


DEFAULT_MAX_OUTPUT_CHARS = 50_000
DEFAULT_MAX_TIMEOUT_SECONDS = 300
DEFAULT_STATE_TTL_HOURS = 168

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.:-]{0,252}$")
_USER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}(?:@sha256:[a-fA-F0-9]{64})?$"
)
_VOLUME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ExecutionBackendError(RuntimeError):
    """A backend could not be resolved or safely invoked."""


@dataclass(frozen=True)
class ExecutionProfile:
    """Validated, non-secret backend configuration."""

    name: str
    kind: str
    workspace: str | None = None
    host: str | None = None
    user: str | None = None
    port: int = 22
    identity_file: str | None = None
    known_hosts_file: str | None = None
    docker_host: str | None = None
    image: str | None = None
    container_workspace: str = "/workspace"
    network: str = "none"
    state_volume: str | None = None
    python_command: str = "python3"

    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, Any]) -> "ExecutionProfile":
        if not _PROFILE_NAME_RE.fullmatch(name):
            raise ExecutionBackendError(
                f"invalid execution profile name {name!r}; use letters, numbers, '.', '_', or '-'"
            )
        kind = str(raw.get("type") or raw.get("kind") or "").strip()
        if kind not in {"local", "ssh", "docker_remote"}:
            raise ExecutionBackendError(
                f"execution profile {name!r} has unsupported type {kind!r}"
            )
        if kind == "local":
            if name != "local":
                raise ExecutionBackendError(
                    "the built-in local profile must be named 'local'"
                )
            return cls(name="local", kind="local")

        workspace = _absolute_posix_path(raw.get("workspace"), "workspace", name)
        identity_file = _absolute_local_path(
            raw.get("identity_file"), "identity_file", name
        )
        known_hosts_file = _absolute_local_path(
            raw.get("known_hosts_file"), "known_hosts_file", name
        )
        python_command = str(raw.get("python_command") or "python3").strip()
        if not re.fullmatch(r"[A-Za-z0-9_./+-]+", python_command):
            raise ExecutionBackendError(
                f"execution profile {name!r} has an invalid python_command"
            )

        if kind == "ssh":
            host = str(raw.get("host") or "").strip()
            user = str(raw.get("user") or "").strip()
            port = _port(raw.get("port", 22), name)
            if not _HOST_RE.fullmatch(host) or host.startswith("-"):
                raise ExecutionBackendError(
                    f"execution profile {name!r} has an invalid SSH host"
                )
            if not _USER_RE.fullmatch(user) or user.startswith("-"):
                raise ExecutionBackendError(
                    f"execution profile {name!r} has an invalid SSH user"
                )
            return cls(
                name=name,
                kind=kind,
                workspace=workspace,
                host=host,
                user=user,
                port=port,
                identity_file=identity_file,
                known_hosts_file=known_hosts_file,
                python_command=python_command,
            )

        docker_host = str(raw.get("docker_host") or "").strip()
        try:
            parsed = urlsplit(docker_host)
            parsed_port = parsed.port
        except ValueError as exc:
            raise ExecutionBackendError(
                f"execution profile {name!r} has an invalid Docker SSH endpoint"
            ) from exc
        if (
            parsed.scheme != "ssh"
            or not parsed.hostname
            or not parsed.username
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ExecutionBackendError(
                f"execution profile {name!r} docker_host must be ssh://USER@HOST[:PORT] without credentials"
            )
        if not _HOST_RE.fullmatch(parsed.hostname) or not _USER_RE.fullmatch(
            parsed.username
        ):
            raise ExecutionBackendError(
                f"execution profile {name!r} has an invalid Docker SSH identity"
            )
        if parsed_port is not None:
            _port(parsed_port, name)
        image = str(raw.get("image") or "").strip()
        if not _IMAGE_RE.fullmatch(image) or image.startswith("-"):
            raise ExecutionBackendError(
                f"execution profile {name!r} has an invalid container image reference"
            )
        container_workspace = _absolute_posix_path(
            raw.get("container_workspace", "/workspace"),
            "container_workspace",
            name,
        )
        network = str(raw.get("network") or "none").strip()
        if network not in {"none", "bridge"}:
            raise ExecutionBackendError(
                f"execution profile {name!r} network must be 'none' or 'bridge'"
            )
        state_volume = str(
            raw.get("state_volume") or _default_state_volume(name)
        ).strip()
        if not _VOLUME_RE.fullmatch(state_volume):
            raise ExecutionBackendError(
                f"execution profile {name!r} has an invalid state_volume"
            )
        return cls(
            name=name,
            kind=kind,
            workspace=workspace,
            identity_file=identity_file,
            known_hosts_file=known_hosts_file,
            docker_host=docker_host,
            image=image,
            container_workspace=container_workspace,
            network=network,
            state_volume=state_volume,
            python_command=python_command,
        )

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "local":
            return {"type": "local"}
        common = {
            "type": self.kind,
            "workspace": self.workspace,
            "identity_file": self.identity_file,
            "known_hosts_file": self.known_hosts_file,
            "python_command": self.python_command,
        }
        if self.kind == "ssh":
            return common | {
                "host": self.host,
                "user": self.user,
                "port": self.port,
            }
        return common | {
            "docker_host": self.docker_host,
            "image": self.image,
            "container_workspace": self.container_workspace,
            "network": self.network,
            "state_volume": self.state_volume,
        }


@dataclass(frozen=True)
class ExecutionSettings:
    active: str = "local"
    profiles: dict[str, ExecutionProfile] = field(
        default_factory=lambda: {"local": ExecutionProfile(name="local", kind="local")}
    )
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    max_timeout_seconds: int = DEFAULT_MAX_TIMEOUT_SECONDS
    state_ttl_hours: int = DEFAULT_STATE_TTL_HOURS

    @classmethod
    def from_values(
        cls,
        raw: Any,
        *,
        max_output_chars: Any = DEFAULT_MAX_OUTPUT_CHARS,
        max_timeout_seconds: Any = DEFAULT_MAX_TIMEOUT_SECONDS,
        state_ttl_hours: Any = DEFAULT_STATE_TTL_HOURS,
    ) -> "ExecutionSettings":
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ExecutionBackendError(
                    "execution.backends is not valid JSON"
                ) from exc
        if raw is None:
            raw = {"active": "local", "profiles": {"local": {"type": "local"}}}
        if not isinstance(raw, Mapping):
            raise ExecutionBackendError("execution.backends must be a JSON object")
        profiles_raw = raw.get("profiles")
        if not isinstance(profiles_raw, Mapping):
            raise ExecutionBackendError("execution.backends.profiles must be an object")
        profiles = {
            str(name): ExecutionProfile.from_dict(str(name), value)
            for name, value in profiles_raw.items()
            if isinstance(value, Mapping)
        }
        if len(profiles) != len(profiles_raw):
            raise ExecutionBackendError("every execution profile must be an object")
        if "local" not in profiles or profiles["local"].kind != "local":
            raise ExecutionBackendError(
                "execution profiles must include built-in 'local'"
            )
        active = str(raw.get("active") or "local")
        if active not in profiles:
            raise ExecutionBackendError(
                f"active execution profile {active!r} does not exist"
            )
        return cls(
            active=active,
            profiles=profiles,
            max_output_chars=_bounded_int(
                max_output_chars, 1_000, 1_000_000, "execution.max_output_chars"
            ),
            max_timeout_seconds=_bounded_int(
                max_timeout_seconds, 1, 3_600, "execution.max_timeout_seconds"
            ),
            state_ttl_hours=_bounded_int(
                state_ttl_hours, 1, 8_760, "execution.repl_state_ttl_hours"
            ),
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "profiles": {
                name: profile.to_dict() for name, profile in self.profiles.items()
            },
        }


@dataclass
class BackendRunResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    timeout_detail: str | None = None


def _bounded_int(value: Any, low: int, high: int, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionBackendError(f"{label} must be an integer") from exc
    if not low <= parsed <= high:
        raise ExecutionBackendError(f"{label} must be between {low} and {high}")
    return parsed


def _port(value: Any, name: str) -> int:
    return _bounded_int(value, 1, 65_535, f"execution profile {name!r} port")


def _absolute_posix_path(value: Any, field_name: str, profile: str) -> str:
    path = str(value or "").strip()
    if (
        not path.startswith("/")
        or "\x00" in path
        or "\n" in path
        or "\r" in path
        or "," in path
    ):
        raise ExecutionBackendError(
            f"execution profile {profile!r} {field_name} must be an absolute POSIX path"
        )
    normalized = posixpath.normpath(path)
    if normalized == "/":
        raise ExecutionBackendError(
            f"execution profile {profile!r} {field_name} cannot be filesystem root"
        )
    return normalized


def _absolute_local_path(value: Any, field_name: str, profile: str) -> str:
    path = str(value or "").strip()
    if not os.path.isabs(path) or "\x00" in path or "\n" in path or "\r" in path:
        raise ExecutionBackendError(
            f"execution profile {profile!r} {field_name} must be an absolute local path"
        )
    return os.path.normpath(path)


def _default_state_volume(name: str) -> str:
    digest = hashlib.sha256(name.encode()).hexdigest()[:12]
    return f"hexis-exec-state-{digest}"


def validate_ssh_material(profile: ExecutionProfile) -> None:
    """Validate exact SSH material at the point of use, without reading secrets."""
    if profile.kind == "local":
        return
    assert profile.identity_file is not None
    assert profile.known_hosts_file is not None
    identity = Path(profile.identity_file)
    known_hosts = Path(profile.known_hosts_file)
    if not identity.is_file():
        raise ExecutionBackendError(
            f"SSH identity file is unavailable at {identity}. Make it visible to the Hexis worker, then retry."
        )
    mode = stat.S_IMODE(identity.stat().st_mode)
    if mode & 0o077:
        raise ExecutionBackendError(
            f"SSH identity file {identity} has mode {mode:04o}; run `chmod 600 {shlex.quote(str(identity))}` and retry."
        )
    if not known_hosts.is_file():
        raise ExecutionBackendError(
            f"SSH known-hosts file is unavailable at {known_hosts}. Connect once with the selected identity to record the host key, then retry."
        )
    known_hosts_mode = stat.S_IMODE(known_hosts.stat().st_mode)
    if known_hosts_mode & 0o022:
        raise ExecutionBackendError(
            f"SSH known-hosts file {known_hosts} is writable by other users (mode {known_hosts_mode:04o}); run `chmod 644 {shlex.quote(str(known_hosts))}` and retry."
        )


async def load_execution_settings(
    context: ToolExecutionContext | None = None,
    *,
    pool: Any | None = None,
) -> ExecutionSettings:
    """Load live settings; never reinterpret a database failure as local policy."""
    if pool is None and context is not None and context.registry is not None:
        pool = context.registry.pool
    if pool is None:
        return ExecutionSettings.from_values(None)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT get_config('execution.backends') AS backends,
                       COALESCE(get_config_int('execution.max_output_chars'), $1) AS max_output,
                       COALESCE(get_config_int('execution.max_timeout_seconds'), $2) AS max_timeout,
                       COALESCE(get_config_int('execution.repl_state_ttl_hours'), $3) AS state_ttl
                """,
                DEFAULT_MAX_OUTPUT_CHARS,
                DEFAULT_MAX_TIMEOUT_SECONDS,
                DEFAULT_STATE_TTL_HOURS,
            )
    except Exception as exc:
        raise ExecutionBackendError(
            "Could not read the selected execution backend; refused to run locally. Restore database access and retry."
        ) from exc
    return ExecutionSettings.from_values(
        row["backends"],
        max_output_chars=row["max_output"],
        max_timeout_seconds=row["max_timeout"],
        state_ttl_hours=row["state_ttl"],
    )


class ExecutionBackend:
    """Stable process interface used by all OS execution tools."""

    def __init__(
        self,
        profile: ExecutionProfile,
        settings: ExecutionSettings,
        local_workspace: str | None,
    ) -> None:
        self.profile = profile
        self.settings = settings
        self.local_workspace = local_workspace or os.getcwd()

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def kind(self) -> str:
        return self.profile.kind

    def cap_timeout(self, requested: int) -> int:
        return max(1, min(int(requested), self.settings.max_timeout_seconds))

    def path_for(self, relative_path: str) -> str:
        raise NotImplementedError

    async def run_shell(
        self,
        command: str,
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> BackendRunResult:
        raise NotImplementedError

    async def run_argv(
        self,
        argv: list[str],
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> BackendRunResult:
        raise NotImplementedError

    async def run_python(
        self,
        code: str,
        *,
        session_id: str,
        timeout: int,
    ) -> BackendRunResult:
        state_key = hashlib.sha256(
            f"{self.name}:{session_id}".encode("utf-8")
        ).hexdigest()
        payload = json.dumps(
            {
                "code": code,
                "state_key": state_key,
                "state_ttl_hours": self.settings.state_ttl_hours,
            }
        ).encode("utf-8")
        return await self.run_argv(
            [self.profile.python_command, "-c", REMOTE_REPL_RUNNER],
            timeout=timeout,
            env={"HEXIS_EXEC_STATE_DIR": self._state_directory()},
            input_bytes=payload,
        )

    def _state_directory(self) -> str:
        return ""


class LocalExecutionBackend(ExecutionBackend):
    def path_for(self, relative_path: str) -> str:
        return os.path.join(self.local_workspace, relative_path)

    async def run_shell(
        self,
        command: str,
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> BackendRunResult:
        process_env = os.environ.copy()
        process_env.update(_validated_env(env))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.local_workspace,
                env=process_env,
            )
        except Exception as exc:
            raise ExecutionBackendError(
                f"could not start local command: {exc}"
            ) from exc
        return await _communicate(
            proc,
            input_bytes,
            self.cap_timeout(timeout),
            max_output_bytes=min(self.settings.max_output_chars * 4, 4_000_000),
        )

    async def run_argv(
        self,
        argv: list[str],
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> BackendRunResult:
        process_env = os.environ.copy()
        process_env.update(_validated_env(env))
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.local_workspace,
                env=process_env,
            )
        except Exception as exc:
            raise ExecutionBackendError(
                f"could not start local command: {exc}"
            ) from exc
        return await _communicate(
            proc,
            input_bytes,
            self.cap_timeout(timeout),
            max_output_bytes=min(self.settings.max_output_chars * 4, 4_000_000),
        )

    def _state_directory(self) -> str:
        cache = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
        return str(cache / "hexis" / "execution-repl")


class SSHExecutionBackend(ExecutionBackend):
    def path_for(self, relative_path: str) -> str:
        assert self.profile.workspace is not None
        return posixpath.join(self.profile.workspace, relative_path)

    def _ssh_argv(self, remote_command: str, timeout: int) -> list[str]:
        validate_ssh_material(self.profile)
        ssh = shutil.which("ssh")
        if not ssh:
            raise ExecutionBackendError(
                "The ssh client is not installed or not on PATH. Install OpenSSH, then retry."
            )
        assert self.profile.identity_file is not None
        assert self.profile.known_hosts_file is not None
        assert self.profile.host is not None
        assert self.profile.user is not None
        return [
            ssh,
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.profile.known_hosts_file}",
            "-o",
            f"ConnectTimeout={min(max(timeout, 1), 15)}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            "-i",
            self.profile.identity_file,
            "-p",
            str(self.profile.port),
            f"{self.profile.user}@{self.profile.host}",
            remote_command,
        ]

    def _remote_command(self, command: str, env: Mapping[str, str] | None) -> str:
        assert self.profile.workspace is not None
        environment = _validated_env(env)
        env_part = ""
        if environment:
            env_part = (
                "env "
                + " ".join(
                    f"{name}={shlex.quote(value)}"
                    for name, value in environment.items()
                )
                + " "
            )
        return (
            f"cd -- {shlex.quote(self.profile.workspace)} && "
            f"exec {env_part}/bin/sh -lc {shlex.quote(command)}"
        )

    async def run_shell(
        self,
        command: str,
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> BackendRunResult:
        return await self._run_supervised(
            command=command,
            argv=None,
            timeout=timeout,
            env=env,
            input_bytes=input_bytes,
        )

    async def _run_supervised(
        self,
        *,
        command: str | None,
        argv: list[str] | None,
        timeout: int,
        env: Mapping[str, str] | None,
        input_bytes: bytes | None,
    ) -> BackendRunResult:
        capped = self.cap_timeout(timeout)
        assert self.profile.workspace is not None
        supervisor_command = (
            f"cd -- {shlex.quote(self.profile.workspace)} && exec "
            f"{shlex.quote(self.profile.python_command)} -c "
            f"{shlex.quote(SSH_PROCESS_SUPERVISOR)}"
        )
        ssh_argv = self._ssh_argv(supervisor_command, capped)
        payload = json.dumps(
            {
                "command": command,
                "argv": argv,
                "env": _validated_env(env),
                "input": base64.b64encode(input_bytes or b"").decode("ascii"),
                "timeout": capped,
                "max_output_bytes": min(
                    self.settings.max_output_chars * 4,
                    4_000_000,
                ),
            }
        ).encode("utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            raise ExecutionBackendError(f"could not start SSH command: {exc}") from exc
        # The target owns the exact command timeout.  Connection setup and its
        # TERM/KILL grace receive a separate bounded allowance.
        wire = await _communicate(
            proc,
            payload,
            capped + 20,
            max_output_bytes=4_000_000,
        )
        if wire.timed_out:
            wire.timeout_detail = (
                "The SSH control connection exceeded its cleanup allowance. "
                "The remote supervisor was instructed to terminate the exact process group; "
                "verify the selected host before retrying."
            )
            return wire
        if wire.returncode != 0:
            return wire
        try:
            response = json.loads(wire.stdout.decode("utf-8"))
            return BackendRunResult(
                returncode=int(response["returncode"]),
                stdout=base64.b64decode(response["stdout"], validate=True),
                stderr=base64.b64decode(response["stderr"], validate=True),
                timed_out=bool(response.get("timed_out")),
                timeout_detail=(
                    "The remote supervisor terminated the exact command process group."
                    if response.get("timed_out")
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            detail = wire.stderr.decode("utf-8", errors="replace").strip()
            raise ExecutionBackendError(
                "SSH execution protocol failed" + (f": {detail}" if detail else "")
            ) from exc

    async def run_argv(
        self,
        argv: list[str],
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> BackendRunResult:
        return await self._run_supervised(
            command=None,
            argv=argv,
            timeout=timeout,
            env=env,
            input_bytes=input_bytes,
        )

    def _state_directory(self) -> str:
        return "~/.cache/hexis/execution-repl"


class DockerRemoteExecutionBackend(ExecutionBackend):
    def path_for(self, relative_path: str) -> str:
        return posixpath.join(self.profile.container_workspace, relative_path)

    def _docker_base(self) -> tuple[list[str], dict[str, str]]:
        validate_ssh_material(self.profile)
        docker = shutil.which("docker")
        if not docker:
            raise ExecutionBackendError(
                "The Docker CLI is not installed or not on PATH. Install the CLI, then retry."
            )
        assert self.profile.identity_file is not None
        assert self.profile.known_hosts_file is not None
        assert self.profile.docker_host is not None
        ssh_command = shlex.join(
            [
                shutil.which("ssh") or "ssh",
                "-F",
                "/dev/null",
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={self.profile.known_hosts_file}",
                "-i",
                self.profile.identity_file,
            ]
        )
        process_env = os.environ.copy()
        process_env["DOCKER_SSH_COMMAND"] = ssh_command
        return [docker, "--host", self.profile.docker_host], process_env

    def _run_argv(
        self,
        executable: list[str],
        env: Mapping[str, str] | None,
        run_id: str,
    ) -> tuple[list[str], dict[str, str]]:
        base, process_env = self._docker_base()
        assert self.profile.workspace is not None
        assert self.profile.image is not None
        assert self.profile.state_volume is not None
        command = base + [
            "run",
            "--rm",
            "--name",
            run_id,
            "--label",
            "io.hexis.execution=true",
            "--label",
            f"io.hexis.execution.profile={self.profile.name}",
            "--pull",
            "never",
            "--network",
            self.profile.network,
            "--mount",
            (
                f"type=bind,source={self.profile.workspace},"
                f"target={self.profile.container_workspace}"
            ),
            "--mount",
            f"type=volume,source={self.profile.state_volume},target=/var/lib/hexis-exec",
            "--workdir",
            self.profile.container_workspace,
        ]
        for name, value in _validated_env(env).items():
            command.extend(["--env", f"{name}={value}"])
        command.extend([self.profile.image, *executable])
        return command, process_env

    async def _run(
        self,
        executable: list[str],
        *,
        timeout: int,
        env: Mapping[str, str] | None,
        input_bytes: bytes | None,
    ) -> BackendRunResult:
        capped = self.cap_timeout(timeout)
        run_id = f"hexis-exec-{uuid.uuid4().hex}"
        argv, process_env = self._run_argv(executable, env, run_id)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
            )
        except Exception as exc:
            raise ExecutionBackendError(
                f"could not start remote Docker command: {exc}"
            ) from exc
        try:
            result = await _communicate(
                proc,
                input_bytes,
                capped,
                max_output_bytes=min(
                    self.settings.max_output_chars * 4,
                    4_000_000,
                ),
            )
        except asyncio.CancelledError:
            await self._remove_owned_container(run_id)
            raise
        if result.timed_out:
            cleanup = await self._remove_owned_container(run_id)
            result.timeout_detail = (
                "The exact Hexis-owned remote container was removed after timeout."
                if cleanup
                else (
                    "Timeout cleanup could not confirm removal of the exact remote "
                    f"container {run_id}; run `docker --host {self.profile.docker_host} "
                    f"rm -f {run_id}` after checking the target."
                )
            )
        return result

    async def _remove_owned_container(self, run_id: str) -> bool:
        try:
            base, process_env = self._docker_base()
            proc = await asyncio.create_subprocess_exec(
                *base,
                "rm",
                "-f",
                run_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            return proc.returncode == 0
        except Exception:
            return False

    async def run_shell(
        self,
        command: str,
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> BackendRunResult:
        return await self._run(
            ["/bin/sh", "-lc", command],
            timeout=timeout,
            env=env,
            input_bytes=input_bytes,
        )

    async def run_argv(
        self,
        argv: list[str],
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> BackendRunResult:
        return await self._run(argv, timeout=timeout, env=env, input_bytes=input_bytes)

    def _state_directory(self) -> str:
        return "/var/lib/hexis-exec/repl"


async def resolve_execution_backend(
    context: ToolExecutionContext | None = None,
    *,
    pool: Any | None = None,
    profile_name: str | None = None,
    local_workspace: str | None = None,
) -> ExecutionBackend:
    settings = await load_execution_settings(context, pool=pool)
    selected = profile_name or settings.active
    if selected not in settings.profiles:
        raise ExecutionBackendError(f"execution profile {selected!r} does not exist")
    profile = settings.profiles[selected]
    workspace = local_workspace
    if workspace is None and context is not None:
        workspace = context.workspace_path
    if profile.kind == "local":
        return LocalExecutionBackend(profile, settings, workspace)
    if profile.kind == "ssh":
        return SSHExecutionBackend(profile, settings, workspace)
    return DockerRemoteExecutionBackend(profile, settings, workspace)


def _validated_env(env: Mapping[str, str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_name, raw_value in (env or {}).items():
        name = str(raw_name)
        if not _ENV_NAME_RE.fullmatch(name):
            raise ExecutionBackendError(f"invalid environment variable name {name!r}")
        value = str(raw_value)
        if "\x00" in value:
            raise ExecutionBackendError(
                f"environment variable {name!r} contains a NUL byte"
            )
        result[name] = value
    return result


async def _communicate(
    proc: asyncio.subprocess.Process,
    input_bytes: bytes | None,
    timeout: int,
    *,
    max_output_bytes: int,
) -> BackendRunResult:
    async def feed_stdin() -> None:
        if proc.stdin is None:
            return
        try:
            if input_bytes:
                proc.stdin.write(input_bytes)
                await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.stdin.close()

    async def drain_stream(
        stream: asyncio.StreamReader | None,
    ) -> bytes:
        if stream is None:
            return b""
        kept = bytearray()
        truncated = False
        while True:
            chunk = await stream.read(65_536)
            if not chunk:
                break
            remaining = max_output_bytes - len(kept)
            if remaining > 0:
                kept.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        if truncated:
            kept.extend(b"\n...[truncated by execution backend]")
        return bytes(kept)

    stdout_task = asyncio.create_task(drain_stream(proc.stdout))
    stderr_task = asyncio.create_task(drain_stream(proc.stderr))
    stdin_task = asyncio.create_task(feed_stdin())
    wait_task = asyncio.create_task(proc.wait())
    all_tasks = asyncio.gather(stdout_task, stderr_task, stdin_task, wait_task)
    timed_out = False
    try:
        await asyncio.wait_for(asyncio.shield(all_tasks), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        await all_tasks
    except asyncio.CancelledError:
        proc.kill()
        await all_tasks
        raise
    stdout = stdout_task.result()
    stderr = stderr_task.result()
    if timed_out:
        return BackendRunResult(
            returncode=proc.returncode if proc.returncode is not None else -9,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
    return BackendRunResult(
        returncode=proc.returncode or 0,
        stdout=stdout,
        stderr=stderr,
    )


# A standard-library-only worker is sent over the selected transport.  It
# preserves picklable variables by opaque session key and makes unsupported
# bridge behavior explicit.  Nothing here contains credentials or user code.
REMOTE_REPL_RUNNER = textwrap.dedent(
    r"""
    import builtins
    import contextlib
    import fcntl
    import io
    import json
    import os
    import pickle
    import sys
    import time
    from pathlib import Path

    request = json.loads(sys.stdin.read())
    state_root = Path(os.environ["HEXIS_EXEC_STATE_DIR"]).expanduser()
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_root, 0o700)
    ttl = max(1, int(request.get("state_ttl_hours", 168))) * 3600
    now = time.time()
    for candidate in state_root.glob("*.pickle"):
        try:
            if now - candidate.stat().st_mtime > ttl:
                candidate.unlink()
        except OSError:
            pass
    state_path = state_root / (request["state_key"] + ".pickle")
    lock_path = state_root / (request["state_key"] + ".lock")
    lock_handle = lock_path.open("a+b")
    os.chmod(lock_path, 0o600)
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    local_vars = {}
    if state_path.is_file():
        try:
            with state_path.open("rb") as handle:
                loaded = pickle.load(handle)
            if isinstance(loaded, dict):
                local_vars = loaded
        except Exception:
            local_vars = {}

    blocked = {"eval", "exec", "compile", "input", "globals", "locals"}
    safe_builtins = {
        name: value for name, value in vars(builtins).items()
        if name not in blocked
    }
    def final_var(name):
        key = str(name).strip().strip("\"'")
        if key in namespace:
            value = namespace[key]
            return json.dumps(value, default=str) if isinstance(value, dict) else str(value)
        available = [key for key in namespace if not key.startswith("_") and key not in reserved]
        return f"Error: Variable {key!r} not found. Available variables: {available}."
    def show_vars():
        visible = {key: type(value).__name__ for key, value in namespace.items() if not key.startswith("_") and key not in reserved}
        return f"Available variables: {visible}" if visible else "No variables created yet."
    def unavailable_bridge(*args, **kwargs):
        raise RuntimeError("tool_use is unavailable on remote execution backends; call the tool directly in a separate turn")

    reserved = {"__builtins__", "__name__", "FINAL_VAR", "SHOW_VARS", "tool_use"}
    namespace = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        "FINAL_VAR": final_var,
        "SHOW_VARS": show_vars,
        "tool_use": unavailable_bridge,
        **local_vars,
    }
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    error = ""
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        try:
            exec(request["code"], namespace, namespace)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    persisted = {}
    skipped = []
    for key, value in namespace.items():
        if key in reserved or key.startswith("_"):
            continue
        try:
            pickle.dumps(value)
            persisted[key] = value
        except Exception:
            skipped.append(key)
    temporary = state_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(persisted, handle)
    os.chmod(temporary, 0o600)
    temporary.replace(state_path)
    stderr = stderr_buffer.getvalue()
    if error:
        stderr = stderr + ("\n" if stderr else "") + error
    response = {
        "stdout": stdout_buffer.getvalue().strip(),
        "stderr": stderr.strip(),
        "variables": {key: type(value).__name__ for key, value in persisted.items()},
        "not_persisted": skipped,
        "execution_time": round(time.perf_counter() - started, 4),
    }
    sys.stdout.write(json.dumps(response))
    """
).strip()


# SSH cannot prove remote cleanup by killing only the local client. This fixed
# standard-library supervisor owns one process group on the target, bounds its
# output on disk, and terminates that exact group on timeout.
SSH_PROCESS_SUPERVISOR = textwrap.dedent(
    r"""
    import base64
    import json
    import os
    import signal
    import subprocess
    import sys
    import tempfile

    request = json.loads(sys.stdin.read())
    command = request.get("command")
    argv = request.get("argv")
    if (command is None) == (argv is None):
        raise ValueError("exactly one of command or argv is required")
    executable = ["/bin/sh", "-lc", command] if command is not None else [str(item) for item in argv]
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in request.get("env", {}).items()})
    stdin_bytes = base64.b64decode(request.get("input", ""), validate=True)
    limit = max(1024, min(int(request.get("max_output_bytes", 200000)), 4000000))
    timed_out = False
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            executable,
            stdin=subprocess.PIPE if stdin_bytes else subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            env=environment,
            start_new_session=True,
        )
        try:
            process.communicate(stdin_bytes if stdin_bytes else None, timeout=max(1, int(request["timeout"])))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(limit + 1)
        stderr = stderr_file.read(limit + 1)
    marker = b"\n...[truncated by remote supervisor]"
    if len(stdout) > limit:
        stdout = stdout[:limit] + marker
    if len(stderr) > limit:
        stderr = stderr[:limit] + marker
    sys.stdout.write(json.dumps({
        "returncode": process.returncode if process.returncode is not None else -9,
        "stdout": base64.b64encode(stdout).decode("ascii"),
        "stderr": base64.b64encode(stderr).decode("ascii"),
        "timed_out": timed_out,
    }))
    """
).strip()
