"""Install and control Hexis workers as per-user host services.

The database remains container-owned.  These units run only the stateless Python
workers installed with the current Hexis CLI.  Unit files contain paths and an
optional instance name, never copied environment values or provider secrets.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


MANAGED_MARKER = "Managed by Hexis host services"
STATE_VERSION = 1
CORE_SERVICE_NAMES = ("heartbeat", "maintenance")
ALL_SERVICE_NAMES = (*CORE_SERVICE_NAMES, "channels")


class HostServiceError(RuntimeError):
    """A host-service problem with a user-actionable message."""


@dataclass(frozen=True)
class HostServiceDefinition:
    name: str
    description: str
    module: str
    arguments: tuple[str, ...]
    launchd_label: str
    systemd_unit: str


SERVICE_DEFINITIONS = {
    "heartbeat": HostServiceDefinition(
        name="heartbeat",
        description="Hexis heartbeat worker",
        module="apps.worker",
        arguments=("--mode", "heartbeat"),
        launchd_label="ai.hexis.worker.heartbeat",
        systemd_unit="hexis-heartbeat.service",
    ),
    "maintenance": HostServiceDefinition(
        name="maintenance",
        description="Hexis maintenance worker",
        module="apps.worker",
        arguments=("--mode", "maintenance"),
        launchd_label="ai.hexis.worker.maintenance",
        systemd_unit="hexis-maintenance.service",
    ),
    "channels": HostServiceDefinition(
        name="channels",
        description="Hexis channel worker",
        module="services.channel_worker",
        arguments=(),
        launchd_label="ai.hexis.worker.channels",
        systemd_unit="hexis-channels.service",
    ),
}

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _run_command(
    command: Sequence[str],
    *,
    capture_output: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=capture_output,
        text=True,
        check=check,
    )


def _hexis_dir(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".hexis"


def host_service_state_path(home: Path | None = None) -> Path:
    return _hexis_dir(home) / "host-services.json"


def host_service_log_dir(home: Path | None = None) -> Path:
    return _hexis_dir(home) / "logs" / "host-services"


def detect_host_service_backend(system_name: str | None = None) -> str:
    name = (system_name or platform.system()).strip().lower()
    if name == "darwin":
        return "launchd"
    if name == "linux":
        return "systemd"
    raise HostServiceError(
        "Hexis host services support macOS launchd and Linux systemd user services. "
        "Use `hexis worker -- --mode both` on this platform."
    )


def _unit_dir(backend: str, home: Path | None = None) -> Path:
    root = home or Path.home()
    if backend == "launchd":
        return root / "Library" / "LaunchAgents"
    if backend == "systemd":
        return root / ".config" / "systemd" / "user"
    raise HostServiceError(f"Unsupported host-service backend: {backend}")


def host_service_unit_path(
    service_name: str,
    *,
    backend: str,
    home: Path | None = None,
) -> Path:
    definition = SERVICE_DEFINITIONS[_validate_service_name(service_name)]
    filename = (
        f"{definition.launchd_label}.plist"
        if backend == "launchd"
        else definition.systemd_unit
    )
    return _unit_dir(backend, home) / filename


def _validate_service_name(value: str) -> str:
    name = str(value or "").strip().lower()
    if name not in SERVICE_DEFINITIONS:
        raise HostServiceError(
            f"Unknown host service {value!r}; choose {', '.join(ALL_SERVICE_NAMES)}."
        )
    return name


def normalize_service_names(
    values: Iterable[str] | None,
    *,
    default: Iterable[str] = CORE_SERVICE_NAMES,
) -> list[str]:
    raw = list(values or default)
    if "all" in raw:
        raw = list(ALL_SERVICE_NAMES)
    names: list[str] = []
    for value in raw:
        name = _validate_service_name(value)
        if name not in names:
            names.append(name)
    if not names:
        raise HostServiceError("Choose at least one host service.")
    return names


def _validate_instance(instance: str | None) -> str | None:
    if instance is None:
        return None
    value = str(instance).strip()
    if not value:
        return None
    from core.instance import validate_instance_name

    validate_instance_name(value)
    return value


def _validate_unit_text(label: str, value: str | Path) -> None:
    if any(character in str(value) for character in ("\x00", "\r", "\n")):
        raise HostServiceError(
            f"{label} contains a control character that cannot be represented safely "
            "in a service definition. Choose a different path."
        )


def _command_for(
    definition: HostServiceDefinition,
    *,
    python_executable: Path,
    instance: str | None,
) -> list[str]:
    command = [
        str(python_executable),
        "-m",
        definition.module,
        *definition.arguments,
    ]
    if instance:
        command += ["--instance", instance]
    return command


def render_launchd_plist(
    service_name: str,
    *,
    python_executable: Path,
    working_directory: Path,
    env_file: Path | None,
    instance: str | None,
    log_file: Path,
) -> bytes:
    definition = SERVICE_DEFINITIONS[_validate_service_name(service_name)]
    environment = {"PYTHONUNBUFFERED": "1"}
    if env_file:
        environment["HEXIS_ENV_FILE"] = str(env_file)
    if instance:
        environment["HEXIS_INSTANCE"] = _validate_instance(instance) or ""
    payload = {
        "Label": definition.launchd_label,
        "ProgramArguments": _command_for(
            definition,
            python_executable=python_executable,
            instance=instance,
        ),
        "WorkingDirectory": str(working_directory),
        "EnvironmentVariables": environment,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Background",
        "ThrottleInterval": 5,
        "StandardOutPath": str(log_file),
        "StandardErrorPath": str(log_file),
        "HexisManaged": True,
        "HexisManagedMarker": MANAGED_MARKER,
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=False)


def _systemd_quote(value: str | Path) -> str:
    # Percent is a systemd specifier prefix even inside quoted values.
    return shlex.quote(str(value).replace("%", "%%"))


def _systemd_environment(name: str, value: str | Path) -> str:
    """Render one systemd Environment= assignment without shell expansion."""

    raw = f"{name}={value}".replace("%", "%%")
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    return f'Environment="{escaped}"'


def render_systemd_unit(
    service_name: str,
    *,
    python_executable: Path,
    working_directory: Path,
    env_file: Path | None,
    instance: str | None,
) -> str:
    definition = SERVICE_DEFINITIONS[_validate_service_name(service_name)]
    command = _command_for(
        definition,
        python_executable=python_executable,
        instance=instance,
    )
    environment_lines = ['Environment="PYTHONUNBUFFERED=1"']
    if env_file:
        environment_lines.append(_systemd_environment("HEXIS_ENV_FILE", env_file))
    if instance:
        clean_instance = _validate_instance(instance)
        environment_lines.append(
            _systemd_environment("HEXIS_INSTANCE", clean_instance or "")
        )
    return "\n".join(
        [
            f"# {MANAGED_MARKER}",
            "[Unit]",
            f"Description={definition.description}",
            "Wants=network-online.target",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={_systemd_quote(working_directory)}",
            *environment_lines,
            "ExecStart=" + " ".join(_systemd_quote(part) for part in command),
            "Restart=on-failure",
            "RestartSec=5",
            "TimeoutStopSec=30",
            "KillSignal=SIGTERM",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def _is_managed_unit(path: Path, backend: str) -> bool:
    if not path.is_file():
        return False
    try:
        if backend == "launchd":
            payload = plistlib.loads(path.read_bytes())
            return payload.get("HexisManaged") is True
        return MANAGED_MARKER in path.read_text(encoding="utf-8")
    except (OSError, ValueError, plistlib.InvalidFileException):
        return False


def installed_host_services(
    *,
    backend: str | None = None,
    home: Path | None = None,
) -> list[str]:
    selected_backend = backend or detect_host_service_backend()
    return [
        name
        for name in ALL_SERVICE_NAMES
        if _is_managed_unit(
            host_service_unit_path(name, backend=selected_backend, home=home),
            selected_backend,
        )
    ]


def has_core_host_services_installed(
    *,
    backend: str | None = None,
    home: Path | None = None,
) -> bool:
    installed = set(installed_host_services(backend=backend, home=home))
    return set(CORE_SERVICE_NAMES) <= installed


def load_host_service_state(home: Path | None = None) -> dict[str, Any]:
    path = host_service_state_path(home)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HostServiceError(
            f"Host-service state is unreadable at {path}. Move it aside, then run "
            "`hexis service status` again."
        ) from exc
    if not isinstance(payload, dict) or payload.get("version") != STATE_VERSION:
        raise HostServiceError(
            f"Host-service state at {path} has an unsupported format. Upgrade Hexis "
            "or move that file aside before reinstalling services."
        )
    return payload


def _write_state(payload: dict[str, Any], home: Path | None = None) -> None:
    path = host_service_state_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _write_unit(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if isinstance(content, bytes):
        temporary.write_bytes(content)
    else:
        temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _command_error(
    action: str, result: subprocess.CompletedProcess[str]
) -> HostServiceError:
    detail = " ".join((result.stderr or result.stdout or "").strip().split())
    if len(detail) > 800:
        detail = detail[:797] + "..."
    suffix = f" Provider output: {detail}" if detail else ""
    return HostServiceError(
        f"Could not {action} Hexis host services (exit {result.returncode}).{suffix} "
        "Run `hexis service status` for the current state and `hexis service logs` "
        "for worker errors."
    )


def _run_checked(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    action: str,
) -> subprocess.CompletedProcess[str]:
    result = runner(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise _command_error(action, result)
    return result


def _systemd_command() -> str:
    command = shutil.which("systemctl")
    if not command:
        raise HostServiceError(
            "systemctl is not installed. Use `hexis worker -- --mode both`, or install "
            "systemd user services on this Linux host."
        )
    return command


def _launchctl_command() -> str:
    discovered = shutil.which("launchctl")
    if discovered:
        return discovered
    command = "/bin/launchctl"
    if not Path(command).exists():
        raise HostServiceError(
            "launchctl is unavailable. Use `hexis worker -- --mode both` on this macOS host."
        )
    return command


def _systemd_linger_status(runner: CommandRunner) -> str:
    loginctl = shutil.which("loginctl")
    if not loginctl:
        return "unknown"
    result = runner(
        [loginctl, "show-user", getpass.getuser(), "--property=Linger", "--value"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    value = (result.stdout or "").strip().lower()
    return "enabled" if value == "yes" else "disabled"


def install_host_services(
    *,
    services: Iterable[str] | None = None,
    env_file: Path | None = None,
    working_directory: Path | None = None,
    instance: str | None = None,
    start: bool = True,
    enable_linger: bool = False,
    backend: str | None = None,
    home: Path | None = None,
    python_executable: Path | None = None,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    selected_backend = backend or detect_host_service_backend()
    names = normalize_service_names(services)
    clean_instance = _validate_instance(instance)
    selected_env = env_file.expanduser().absolute() if env_file else None
    if selected_env and not selected_env.is_file():
        raise HostServiceError(
            f"Environment file not found: {selected_env}. Choose an existing file with "
            "`--env-file`, or omit that option to use process defaults."
        )
    cwd = (
        working_directory.expanduser().absolute()
        if working_directory
        else (selected_env.parent if selected_env else Path.cwd().absolute())
    )
    if not cwd.is_dir():
        raise HostServiceError(f"Host-service working directory does not exist: {cwd}")
    executable = (python_executable or Path(sys.executable)).expanduser().absolute()
    if not executable.is_file():
        raise HostServiceError(
            f"Hexis Python executable not found: {executable}. Reinstall Hexis with uv, "
            "then run `hexis service install` again."
        )
    _validate_unit_text("Python executable", executable)
    _validate_unit_text("Working directory", cwd)
    if selected_env:
        _validate_unit_text("Environment-file path", selected_env)

    unit_paths = {
        name: host_service_unit_path(name, backend=selected_backend, home=home)
        for name in names
    }
    for name, path in unit_paths.items():
        if path.exists() and not _is_managed_unit(path, selected_backend):
            raise HostServiceError(
                f"Refusing to overwrite non-Hexis service file {path}. Move or rename it, "
                "then run `hexis service install` again."
            )

    previous = load_host_service_state(home)
    previous_backend = previous.get("backend")
    if previous_backend and previous_backend != selected_backend:
        raise HostServiceError(
            f"Host-service state belongs to {previous_backend}, not {selected_backend}. "
            f"Move {host_service_state_path(home)} aside after reviewing it, then retry."
        )

    logs = host_service_log_dir(home)
    logs.mkdir(parents=True, exist_ok=True)
    for name, path in unit_paths.items():
        if selected_backend == "launchd":
            log_file = logs / f"{name}.log"
            if not log_file.exists():
                log_file.touch(mode=0o600)
            content: bytes | str = render_launchd_plist(
                name,
                python_executable=executable,
                working_directory=cwd,
                env_file=selected_env,
                instance=clean_instance,
                log_file=log_file,
            )
        else:
            content = render_systemd_unit(
                name,
                python_executable=executable,
                working_directory=cwd,
                env_file=selected_env,
                instance=clean_instance,
            )
        _write_unit(path, content)

    prior_services = [
        name for name in previous.get("services", []) if name in SERVICE_DEFINITIONS
    ]
    saved_services = list(dict.fromkeys([*prior_services, *names]))
    state = {
        "version": STATE_VERSION,
        "backend": selected_backend,
        "services": saved_services,
        "python_executable": str(executable),
        "working_directory": str(cwd),
        "env_file": str(selected_env) if selected_env else None,
        "instance": clean_instance,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(state, home)

    linger = "not_applicable"
    if selected_backend == "systemd":
        systemctl = _systemd_command()
        _run_checked(
            runner, [systemctl, "--user", "daemon-reload"], action="reload systemd"
        )
        units = [SERVICE_DEFINITIONS[name].systemd_unit for name in names]
        action = "enable and start" if start else "enable"
        command = [systemctl, "--user", "enable"]
        if start:
            command.append("--now")
        _run_checked(runner, [*command, *units], action=action)
        linger = _systemd_linger_status(runner)
        if enable_linger and linger != "enabled":
            loginctl = shutil.which("loginctl")
            if not loginctl:
                raise HostServiceError(
                    "loginctl is unavailable, so Hexis could not enable user lingering. "
                    "The units are installed but may stop after logout."
                )
            _run_checked(
                runner,
                [loginctl, "enable-linger", getpass.getuser()],
                action="enable user lingering",
            )
            linger = _systemd_linger_status(runner)
    else:
        launchctl = _launchctl_command()
        domain = f"gui/{os.getuid()}"
        for name in names:
            definition = SERVICE_DEFINITIONS[name]
            target = f"{domain}/{definition.launchd_label}"
            runner([launchctl, "bootout", target], capture_output=True, check=False)
            _run_checked(
                runner,
                [launchctl, "enable", target],
                action=f"enable {name}",
            )
            if start:
                _run_checked(
                    runner,
                    [launchctl, "bootstrap", domain, str(unit_paths[name])],
                    action=f"start {name}",
                )

    mode_warning = None
    if selected_env:
        try:
            if selected_env.stat().st_mode & 0o077:
                mode_warning = (
                    f"{selected_env} is readable beyond your user account. Hexis did not "
                    "change it; review its permissions if it contains secrets."
                )
        except OSError:
            pass
    return {
        **state,
        "installed": names,
        "started": names if start else [],
        "unit_paths": {name: str(path) for name, path in unit_paths.items()},
        "linger": linger,
        "warning": mode_warning,
    }


def _status_one(
    service_name: str,
    *,
    backend: str,
    home: Path | None,
    runner: CommandRunner,
) -> dict[str, Any]:
    definition = SERVICE_DEFINITIONS[service_name]
    path = host_service_unit_path(service_name, backend=backend, home=home)
    managed = _is_managed_unit(path, backend)
    active = False
    enabled: bool | None = False
    detail = ""
    if managed and backend == "systemd":
        systemctl = _systemd_command()
        active_result = runner(
            [systemctl, "--user", "is-active", definition.systemd_unit],
            capture_output=True,
            check=False,
        )
        enabled_result = runner(
            [systemctl, "--user", "is-enabled", definition.systemd_unit],
            capture_output=True,
            check=False,
        )
        active = (
            active_result.returncode == 0
            and (active_result.stdout or "").strip() == "active"
        )
        enabled = (
            enabled_result.returncode == 0
            and (enabled_result.stdout or "").strip() == "enabled"
        )
        detail = (active_result.stderr or active_result.stdout or "").strip()
    elif managed:
        launchctl = _launchctl_command()
        enabled = None
        domain = f"gui/{os.getuid()}"
        target = f"{domain}/{definition.launchd_label}"
        result = runner(
            [launchctl, "print", target],
            capture_output=True,
            check=False,
        )
        active = result.returncode == 0
        disabled_result = runner(
            [launchctl, "print-disabled", domain],
            capture_output=True,
            check=False,
        )
        if disabled_result.returncode == 0:
            disabled_pattern = re.compile(
                rf'"?{re.escape(definition.launchd_label)}"?\s*=>\s*disabled\b'
            )
            enabled = not bool(disabled_pattern.search(disabled_result.stdout or ""))
        detail = (result.stderr or "").strip()
    return {
        "name": service_name,
        "installed": managed,
        "active": active,
        "enabled": enabled,
        "unit_path": str(path),
        "detail": " ".join(detail.split())[:800],
    }


def host_service_status(
    *,
    backend: str | None = None,
    home: Path | None = None,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    selected_backend = backend or detect_host_service_backend()
    state = load_host_service_state(home)
    services = [
        _status_one(name, backend=selected_backend, home=home, runner=runner)
        for name in ALL_SERVICE_NAMES
    ]
    return {
        "backend": selected_backend,
        "state_path": str(host_service_state_path(home)),
        "instance": state.get("instance"),
        "env_file": state.get("env_file"),
        "working_directory": state.get("working_directory"),
        "python_executable": state.get("python_executable"),
        "linger": (
            _systemd_linger_status(runner)
            if selected_backend == "systemd"
            else "not_applicable"
        ),
        "services": services,
    }


def control_host_services(
    action: str,
    services: Iterable[str] | None = None,
    *,
    backend: str | None = None,
    home: Path | None = None,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"start", "stop", "restart"}:
        raise HostServiceError("Host-service action must be start, stop, or restart.")
    selected_backend = backend or detect_host_service_backend()
    installed = installed_host_services(backend=selected_backend, home=home)
    names = normalize_service_names(services, default=installed or CORE_SERVICE_NAMES)
    missing = [name for name in names if name not in installed]
    if missing:
        raise HostServiceError(
            f"Host services are not installed for: {', '.join(missing)}. Run "
            "`hexis service install` first."
        )

    if selected_backend == "systemd":
        systemctl = _systemd_command()
        units = [SERVICE_DEFINITIONS[name].systemd_unit for name in names]
        _run_checked(
            runner,
            [systemctl, "--user", normalized_action, *units],
            action=normalized_action,
        )
    else:
        launchctl = _launchctl_command()
        domain = f"gui/{os.getuid()}"
        for name in names:
            definition = SERVICE_DEFINITIONS[name]
            target = f"{domain}/{definition.launchd_label}"
            path = host_service_unit_path(name, backend=selected_backend, home=home)
            if normalized_action == "stop":
                print_result = runner(
                    [launchctl, "print", target],
                    capture_output=True,
                    check=False,
                )
                # A service already stopped is the desired state. Checking first
                # avoids depending on launchctl's localized error text.
                if print_result.returncode == 0:
                    _run_checked(
                        runner,
                        [launchctl, "bootout", target],
                        action=f"stop {name}",
                    )
            elif normalized_action == "start":
                print_result = runner(
                    [launchctl, "print", target],
                    capture_output=True,
                    check=False,
                )
                command = (
                    [launchctl, "kickstart", "-k", target]
                    if print_result.returncode == 0
                    else [launchctl, "bootstrap", domain, str(path)]
                )
                _run_checked(runner, command, action=f"start {name}")
            else:
                print_result = runner(
                    [launchctl, "print", target],
                    capture_output=True,
                    check=False,
                )
                if print_result.returncode == 0:
                    _run_checked(
                        runner,
                        [launchctl, "kickstart", "-k", target],
                        action=f"restart {name}",
                    )
                else:
                    _run_checked(
                        runner,
                        [launchctl, "bootstrap", domain, str(path)],
                        action=f"restart {name}",
                    )
    return {
        "action": normalized_action,
        "services": names,
        "status": host_service_status(
            backend=selected_backend,
            home=home,
            runner=runner,
        ),
    }


def uninstall_host_services(
    services: Iterable[str] | None = None,
    *,
    backend: str | None = None,
    home: Path | None = None,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    selected_backend = backend or detect_host_service_backend()
    installed = installed_host_services(backend=selected_backend, home=home)
    names = normalize_service_names(services, default=installed or ALL_SERVICE_NAMES)
    missing = [name for name in names if name not in installed]
    if missing:
        raise HostServiceError(
            f"No Hexis-managed host service is installed for: {', '.join(missing)}."
        )

    if selected_backend == "systemd":
        systemctl = _systemd_command()
        units = [SERVICE_DEFINITIONS[name].systemd_unit for name in names]
        _run_checked(
            runner,
            [systemctl, "--user", "disable", "--now", *units],
            action="stop and disable",
        )
    else:
        launchctl = _launchctl_command()
        domain = f"gui/{os.getuid()}"
        for name in names:
            target = f"{domain}/{SERVICE_DEFINITIONS[name].launchd_label}"
            runner([launchctl, "bootout", target], capture_output=True, check=False)

    removed: list[str] = []
    for name in names:
        path = host_service_unit_path(name, backend=selected_backend, home=home)
        if not _is_managed_unit(path, selected_backend):
            raise HostServiceError(f"Refusing to remove non-Hexis service file {path}.")
        path.unlink()
        removed.append(str(path))

    if selected_backend == "systemd":
        _run_checked(
            runner,
            [_systemd_command(), "--user", "daemon-reload"],
            action="reload systemd",
        )

    previous = load_host_service_state(home)
    remaining = [
        name
        for name in previous.get("services", [])
        if name in SERVICE_DEFINITIONS and name not in names
    ]
    if remaining:
        _write_state({**previous, "services": remaining}, home)
    else:
        state_path = host_service_state_path(home)
        if state_path.exists():
            state_path.unlink()
    return {
        "uninstalled": names,
        "removed_unit_paths": removed,
        "preserved_log_directory": str(host_service_log_dir(home)),
    }


def stream_host_service_logs(
    services: Iterable[str] | None = None,
    *,
    lines: int = 100,
    follow: bool = False,
    backend: str | None = None,
    home: Path | None = None,
) -> int:
    selected_backend = backend or detect_host_service_backend()
    installed = installed_host_services(backend=selected_backend, home=home)
    names = normalize_service_names(services, default=installed or CORE_SERVICE_NAMES)
    missing = [name for name in names if name not in installed]
    if missing:
        raise HostServiceError(
            f"Host services are not installed for: {', '.join(missing)}."
        )
    bounded_lines = max(1, min(int(lines), 10_000))
    if selected_backend == "systemd":
        journalctl = shutil.which("journalctl")
        if not journalctl:
            raise HostServiceError(
                "journalctl is unavailable. Use `hexis service status` and inspect your "
                "systemd user journal directly."
            )
        command = [journalctl, "--user", "--lines", str(bounded_lines)]
        for name in names:
            command += ["--unit", SERVICE_DEFINITIONS[name].systemd_unit]
        if follow:
            command.append("--follow")
    else:
        tail = shutil.which("tail") or "/usr/bin/tail"
        command = [tail, "-n", str(bounded_lines)]
        if follow:
            command.append("-f")
        command += [str(host_service_log_dir(home) / f"{name}.log") for name in names]
    try:
        return subprocess.run(command).returncode
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError as exc:
        raise HostServiceError(f"Log reader is unavailable: {command[0]}") from exc
