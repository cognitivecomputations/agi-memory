from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv

from core import cli_api
from core.agent_api import db_dsn_from_env, resolve_instance
from core.memory_exchange import PROTECTED_SECTIONS, SUPPORTED_IMPORT_STRATEGIES
from core.protected_replacement import OPERATOR_OVERRIDE_REASON_CODES

try:
    _ver = pkg_version("hexis")
except PackageNotFoundError:
    _ver = "dev"


def _print_err(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def _pypi_latest() -> str | None:
    """Newest hexis release on PyPI, or None when the lookup fails (offline,
    proxy, PyPI outage). Callers must treat None as "unknown", never "current"."""
    try:
        import urllib.request

        with urllib.request.urlopen(
            "https://pypi.org/pypi/hexis/json", timeout=10
        ) as resp:
            return json.load(resp)["info"]["version"] or None
    except Exception:
        return None


def _is_newer(candidate: str, current: str) -> bool:
    """True when candidate is a strictly newer X.Y.Z release than current.

    Unparseable versions compare as not-newer: this gates upgrade nagging and
    hard failures, so unknown must never masquerade as an available update.
    """

    def parse(v: str) -> tuple[int, ...] | None:
        try:
            return tuple(int(part) for part in v.strip().split("."))
        except ValueError:
            return None

    a, b = parse(candidate), parse(current)
    if a is None or b is None:
        return False
    return a > b


def _installed_via() -> str:
    """How this CLI was installed: 'uv', 'pipx', or 'pip'.

    uv and pipx manage the tool venv themselves and leave a marker file at the
    venv root. Raw `python -m pip` inside those venvs is the wrong upgrade
    path: uv tool venvs ship no pip module at all, and pipx keeps its own
    metadata that a bare pip upgrade bypasses.
    """
    prefix = Path(sys.prefix)
    if (prefix / "uv-receipt.toml").exists():
        return "uv"
    if (prefix / "pipx_metadata.json").exists():
        return "pipx"
    return "pip"


def _self_update_hint(installer: str) -> str:
    """The command a user runs by hand to move the hexis package.

    uv and pipx use `install --force` rather than `upgrade`: their upgrade
    commands honor the version pin recorded at install time, so a pinned
    install "upgrades" to itself with exit code 0.
    """
    return {
        "uv": "uv tool install --force hexis",
        "pipx": "pipx install --force hexis",
        "pip": "pip install --upgrade hexis",
    }[installer]


def _run_self_update(console: Any, installer: str) -> str:
    """Try to upgrade the hexis package in place; return the version installed
    afterwards (== _ver when nothing moved). Prints its own diagnostics."""
    if installer == "uv":
        uv = shutil.which("uv")
        cmd = [uv, "tool", "install", "--force", "hexis"] if uv else None
    elif installer == "pipx":
        pipx = shutil.which("pipx")
        cmd = [pipx, "install", "--force", "hexis"] if pipx else None
    else:
        # -I: ignore PYTHONPATH and cwd, so stray hexis metadata on either
        # can't make pip think a different version is already installed.
        cmd = [sys.executable, "-I", "-m", "pip", "install", "--upgrade", "hexis"]
    if cmd is None:
        console.print(
            f"[warn]⚠ hexis was installed with {installer}, but the "
            f"[accent]{installer}[/accent] command is not on PATH.[/warn]"
        )
        return _ver
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr or ""
        if installer == "pip" and "externally-managed-environment" in stderr:
            # PEP 668 distro Python: hexis is already installed in this
            # environment, so upgrading it in place mirrors the user's own
            # original install choice. Say so out loud, then retry.
            console.print(
                "[warn]This Python is marked externally managed; upgrading the "
                "already-installed hexis package with --break-system-packages.[/warn]"
            )
            proc = subprocess.run(
                cmd + ["--break-system-packages"], capture_output=True, text=True
            )
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-4:])
            console.print(f"[warn]⚠ Self-update failed:[/warn]\n{tail}")
            return _ver
    # -I (isolated mode): `python -c` otherwise puts the cwd on sys.path, so
    # running `hexis upgrade` from a directory holding stale hexis metadata
    # (an old egg-info, a checkout) would report that version instead of the
    # one actually installed in this environment.
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from importlib.metadata import version; print(version('hexis'))",
        ],
        capture_output=True,
        text=True,
    )
    return (probe.stdout or "").strip() or _ver


def _find_compose_file(start: Path | None = None) -> tuple[Path | None, bool]:
    """Find compose file. Returns (path, is_source_checkout).

    A source checkout is identified by having both docker-compose.yml and
    ops/Dockerfile.db in the same tree.  When no source checkout is found,
    falls back to the bundled runtime compose shipped with pip install.
    """
    cur = (start or Path.cwd()).resolve()
    for parent in (cur,) + tuple(cur.parents):
        candidate = parent / "docker-compose.yml"
        if candidate.exists() and (parent / "ops" / "Dockerfile.db").exists():
            return candidate, True
    # Fall back to bundled runtime compose (pip install)
    runtime = Path(__file__).parent.parent / "ops" / "docker-compose.runtime.yml"
    if runtime.exists():
        return runtime, False
    return None, False


def _stack_root_from_compose(compose_file: Path) -> Path:
    if compose_file.parent.name == "ops":
        return compose_file.parent.parent
    return compose_file.parent


def ensure_docker() -> str:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        _print_err(
            "Docker is not installed or not on PATH. Install Docker Desktop, "
            "or install the Docker CLI and a compatible daemon such as Colima."
        )
        raise SystemExit(1)
    try:
        subprocess.run(
            [docker_bin, "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        _print_err(
            "Docker is installed but no daemon is reachable. Start Docker Desktop "
            "or run `colima start`, then retry."
        )
        raise SystemExit(1)
    return docker_bin


def ensure_compose(docker_bin: str) -> list[str]:
    try:
        subprocess.run(
            [docker_bin, "compose", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return [docker_bin, "compose"]
    except Exception:
        pass
    compose_bin = shutil.which("docker-compose")
    if compose_bin:
        return [compose_bin]
    _print_err(
        "Docker Compose not available. Install Compose: https://docs.docker.com/compose/install/"
    )
    raise SystemExit(1)


def resolve_env_file(stack_root: Path) -> Path | None:
    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / ".env.local",
        stack_root / ".env",
        stack_root / ".env.local",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _compose_env() -> dict[str, str]:
    """Environment for docker compose invocations.

    Pins HEXIS_IMAGE_TAG (used by ops/docker-compose.runtime.yml) to this
    CLI's own package version, so a pip-installed `hexis` always runs the
    images published from the same release commit. An explicit
    HEXIS_IMAGE_TAG in the caller's environment wins. Source checkouts use
    the same published tag by default and build that tag only on --build or
    through `hexis dev`.
    """
    env = os.environ.copy()
    if not env.get("HEXIS_IMAGE_TAG"):
        env["HEXIS_IMAGE_TAG"] = _ver if _ver != "dev" else "latest"
    return env


def _up_compose_args(
    profiles: list[str],
    *,
    is_source: bool,
    build: bool,
    services: list[str] | None = None,
) -> list[str]:
    """Build `compose up` arguments without putting builds on the default path."""

    args: list[str] = []
    for profile in profiles:
        args += ["--profile", profile]
    args += ["up", "-d"]
    if is_source:
        if build:
            args.append("--build")
        else:
            # The source compose declares published image names and local
            # build recipes. Be explicit so a failed pull never falls through
            # into a surprise dependency build.
            args += ["--no-build", "--pull", "missing"]
    if services is not None:
        args += services
    return args


def run_compose(
    compose_cmd: list[str],
    compose_file: Path,
    stack_root: Path,
    args: list[str],
    env_file: Path | None,
) -> int:
    cmd = compose_cmd + ["-f", str(compose_file)]
    if env_file:
        cmd += ["--env-file", str(env_file)]
    cmd += args

    try:
        result = subprocess.run(cmd, cwd=stack_root, env=_compose_env())
        return result.returncode
    except FileNotFoundError:
        _print_err("Failed to run docker compose. Ensure Docker is installed.")
        return 1


def _run_compose_capture(
    compose_cmd: list[str],
    compose_file: Path,
    stack_root: Path,
    args: list[str],
    env_file: Path | None,
) -> tuple[int, str]:
    cmd = compose_cmd + ["-f", str(compose_file)]
    if env_file:
        cmd += ["--env-file", str(env_file)]
    cmd += args
    try:
        p = subprocess.run(
            cmd, cwd=stack_root, env=_compose_env(), capture_output=True, text=True
        )
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out.strip()
    except FileNotFoundError:
        return 1, "Failed to run docker compose. Ensure Docker is installed."


def _redact_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg))  # deep copy via json

    def _is_sensitive_config_key(key: str) -> bool:
        k = (key or "").lower()
        if k.startswith("oauth."):
            return True
        if "user.contact" in k:
            return True
        if "api_key" in k and not k.endswith("api_key_env"):
            return True
        return any(s in k for s in ("token", "secret", "password"))

    def _is_sensitive_field_name(name: str) -> bool:
        n = (name or "").lower()
        if n in {"api_key_env"}:
            return False
        if n in {"api_key", "access", "refresh", "id_token"}:
            return True
        if n == "destinations":
            return True
        if "api_key" in n and not n.endswith("_env"):
            return True
        return any(s in n for s in ("token", "secret", "password"))

    def _redact_deep(value: Any) -> Any:
        if isinstance(value, list):
            return [_redact_deep(v) for v in value]
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for k, v in value.items():
                if _is_sensitive_field_name(str(k)):
                    redacted[str(k)] = "***"
                else:
                    redacted[str(k)] = _redact_deep(v)
            return redacted
        return value

    for key, value in list(out.items()):
        if _is_sensitive_config_key(str(key)):
            out[str(key)] = (
                _redact_deep(value) if isinstance(value, (dict, list)) else "***"
            )
        else:
            out[str(key)] = (
                _redact_deep(value) if isinstance(value, (dict, list)) else value
            )

    return out


def _make_db_flags() -> argparse.ArgumentParser:
    """Shared --dsn / --wait-seconds parent parser."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--dsn", default=None, help="Postgres DSN; defaults to POSTGRES_* env vars"
    )
    p.add_argument(
        "--wait-seconds",
        type=int,
        default=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")),
    )
    return p


# Group definitions for custom help display
_HELP_GROUPS = [
    (
        "Getting Started",
        [
            ("init", "Set up your agent"),
            ("demo", "Prove core capabilities without retaining demo state"),
            ("maturity", "Score live capability maturity and next steps"),
            ("doctor", "Diagnose common issues"),
            ("status", "Show agent status"),
        ],
    ),
    (
        "Stack",
        [
            ("up", "Start the default stack"),
            ("down", "Stop the stack"),
            ("uninstall", "Remove Hexis; keep brain data unless --purge is explicit"),
            ("upgrade", "Update + migrate the schema, keeping your data"),
            ("migrate", "Apply pending schema migrations (no data loss)"),
            ("backup", "Back up the database to a file"),
            ("restore", "Restore the database from a backup file"),
            ("reset", "Wipe the DB and re-initialize"),
            ("ps", "List services"),
            ("logs", "Show logs"),
            ("service", "Run workers as launchd/systemd user services"),
            ("start", "Start heartbeat and maintenance workers manually if stopped"),
            ("stop", "Stop workers (containers stay running)"),
        ],
    ),
    (
        "Interact",
        [
            ("chat", "Chat in the terminal"),
            ("chat-sessions", "List, inspect, export, and fork chat sessions"),
            ("ui", "Start the web dashboard"),
            ("open", "Open the web dashboard in your browser"),
            ("tunnel", "Serve the dashboard privately over Tailscale HTTPS"),
            ("voice", "Set up and control local speech output"),
        ],
    ),
    (
        "Memory & Goals",
        [
            ("recall", "Search memories by semantic query"),
            ("goals", "Manage agent goals"),
            ("ingest", "Ingest documents and knowledge"),
            ("docs", "Search and read the source-document filing cabinet"),
            ("desk", "RecMem desk: list, read, pin, clear working material"),
            ("export", "Export memory as an HMX exchange"),
            ("import", "Inspect or import an HMX exchange"),
            ("import-review", "Review staged HMX records"),
            ("retention", "Show memory-retention status"),
            ("skills", "Manage skill-improvement reviews and proposals"),
            ("schedule", "Manage scheduled tasks"),
        ],
    ),
    (
        "Configuration",
        [
            ("config", "Show/validate agent configuration"),
            ("auth", "Login/logout for subscription OAuth providers"),
            ("tools", "Manage tools configuration"),
            ("characters", "Manage character cards"),
            ("consents", "Manage consent certificates"),
            ("requests", "Decide the agent's resource requests"),
            ("channels", "Manage channel adapters"),
            ("node", "Pair and run an outward-only companion node"),
        ],
    ),
    (
        "Instances",
        [
            ("instance", "Manage Hexis instances"),
        ],
    ),
    (
        "Advanced",
        [
            ("api", "Start the API server"),
            ("mcp", "Start the MCP tools server"),
            ("worker", "Run a background worker process"),
        ],
    ),
]


def _print_grouped_help() -> None:
    """Print custom grouped help using Rich."""
    from rich.console import Console

    console = Console()
    console.print(f"\nhexis v{_ver}")
    console.print("[dim]Persistent memory and identity for AI[/dim]\n")
    console.print("[bold]Usage:[/bold] hexis <command> [options]\n")

    for group_name, commands in _HELP_GROUPS:
        console.print(f"  [bold]{group_name}[/bold]")
        for cmd, desc in commands:
            console.print(f"    {cmd:<14} {desc}")
        console.print()

    console.print("Run 'hexis help <command>' for details on a specific command.\n")


def build_parser() -> argparse.ArgumentParser:
    _db = _make_db_flags()

    p = argparse.ArgumentParser(
        prog="hexis",
        description="Hexis \u2014 persistent memory and identity for AI",
        add_help=False,
    )
    p.add_argument("-h", "--help", action="store_true", default=False, help="Show help")
    p.add_argument("--version", "-V", action="version", version=f"hexis {_ver}")
    p.add_argument(
        "--instance",
        "-i",
        default=None,
        help="Target a specific instance (overrides HEXIS_INSTANCE and current instance)",
    )
    sub = p.add_subparsers(dest="command", required=False)

    # -- Instance management (nested under 'instance') --
    instance = sub.add_parser("instance", parents=[_db], help="Manage Hexis instances")
    inst_sub = instance.add_subparsers(dest="instance_command")

    inst_create = inst_sub.add_parser("create", help="Create a new instance")
    inst_create.add_argument("name", help="Instance name")
    inst_create.add_argument(
        "--description", "-d", default="", help="Instance description"
    )
    inst_create.set_defaults(func="instance_create")

    inst_list = inst_sub.add_parser("list", help="List all instances")
    inst_list.add_argument("--json", action="store_true", help="Output JSON")
    inst_list.set_defaults(func="instance_list")

    inst_use = inst_sub.add_parser("use", help="Switch to a different instance")
    inst_use.add_argument("name", help="Instance name to switch to")
    inst_use.set_defaults(func="instance_use")

    inst_current = inst_sub.add_parser("current", help="Show current instance")
    inst_current.set_defaults(func="instance_current")

    inst_delete = inst_sub.add_parser("delete", help="Delete an instance")
    inst_delete.add_argument("name", help="Instance name to delete")
    inst_delete.add_argument("--force", action="store_true", help="Skip confirmation")
    inst_delete.add_argument(
        "--reason", default=None, help="Reason for deletion (shared with the agent)"
    )
    inst_delete.set_defaults(func="instance_delete")

    inst_clone = inst_sub.add_parser("clone", help="Clone an instance")
    inst_clone.add_argument("source", help="Source instance name")
    inst_clone.add_argument("target", help="Target instance name")
    inst_clone.add_argument(
        "--description", "-d", default="", help="Description for new instance"
    )
    inst_clone.set_defaults(func="instance_clone")

    inst_import = inst_sub.add_parser(
        "import", help="Import an existing database as an instance"
    )
    inst_import.add_argument("name", help="Instance name")
    inst_import.add_argument(
        "--database", help="Database name (defaults to hexis_{name})"
    )
    inst_import.add_argument(
        "--description", "-d", default="", help="Instance description"
    )
    inst_import.set_defaults(func="instance_import")

    instance.set_defaults(func="instance")

    # -- Consent management --
    consents = sub.add_parser(
        "consents", help="View recorded consent (from the database)"
    )
    consents_sub = consents.add_subparsers(dest="consents_command")

    consents_list = consents_sub.add_parser(
        "list", help="List recorded consent decisions"
    )
    consents_list.add_argument("--json", action="store_true", help="Output JSON")
    consents_list.set_defaults(func="consents_list")

    consents_show = consents_sub.add_parser(
        "show", help="Show a model's recorded consent"
    )
    consents_show.add_argument("model", help="Model identifier (provider/model_id)")
    consents_show.set_defaults(func="consents_show")

    consents_request = consents_sub.add_parser(
        "request", help="How to establish consent (runs during hexis init)"
    )
    consents_request.add_argument("model", help="Model identifier (provider/model_id)")
    consents_request.set_defaults(func="consents_request")

    consents_revoke = consents_sub.add_parser(
        "revoke", help="Record a decline for a model"
    )
    consents_revoke.add_argument("model", help="Model identifier (provider/model_id)")
    consents_revoke.add_argument(
        "--reason", default="User requested revocation", help="Revocation reason"
    )
    consents_revoke.set_defaults(func="consents_revoke")

    consents.set_defaults(func="consents")

    # -- Resource requests (#84): the agent asks, the operator decides --
    requests_p = sub.add_parser(
        "requests",
        parents=[_db],
        help="View and decide the agent's resource requests",
    )
    requests_sub = requests_p.add_subparsers(dest="requests_cmd")

    requests_list = requests_sub.add_parser(
        "list", parents=[_db], help="List resource requests"
    )
    requests_list.add_argument(
        "--status",
        default=None,
        choices=["pending", "granted", "denied", "modified", "all"],
        help="Filter by status (default: pending)",
    )
    requests_list.add_argument("--json", action="store_true", help="Output as JSON")
    requests_list.set_defaults(func="requests_list")

    requests_grant = requests_sub.add_parser(
        "grant", parents=[_db], help="Grant a request (applies its effect)"
    )
    requests_grant.add_argument("id", help="Request id (or unique prefix)")
    requests_grant.add_argument(
        "--note", default=None, help="Note the agent will see with the decision"
    )
    requests_grant.set_defaults(func="requests_grant")

    requests_deny = requests_sub.add_parser(
        "deny", parents=[_db], help="Deny a request"
    )
    requests_deny.add_argument("id", help="Request id (or unique prefix)")
    requests_deny.add_argument(
        "--note", default=None, help="Note the agent will see with the decision"
    )
    requests_deny.set_defaults(func="requests_deny")

    requests_modify = requests_sub.add_parser(
        "modify",
        parents=[_db],
        help="Grant with a different value than requested",
    )
    requests_modify.add_argument("id", help="Request id (or unique prefix)")
    requests_modify.add_argument(
        "--value", required=True, help="The value actually granted (JSON)"
    )
    requests_modify.add_argument(
        "--note", default=None, help="Note the agent will see with the decision"
    )
    requests_modify.set_defaults(func="requests_modify")

    requests_p.set_defaults(func="requests")

    # -- Stack commands --
    up = sub.add_parser("up", help="Start the stack")
    up_build = up.add_mutually_exclusive_group()
    up_build.add_argument(
        "--build",
        action="store_true",
        help="Build source images before starting (source checkouts only)",
    )
    up_build.add_argument(
        "--no-build",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    up.add_argument(
        "--profile", "-p", action="append", default=[], help="Compose profile(s)"
    )
    up.set_defaults(func="up")

    dev = sub.add_parser(
        "dev",
        help="Start the stack in watch mode: code and migration edits apply automatically",
    )
    dev.add_argument(
        "--profile", "-p", action="append", default=[], help="Compose profile(s)"
    )
    dev.set_defaults(func="dev")

    down = sub.add_parser("down", help="Stop the stack")
    down.set_defaults(func="down")

    uninstall = sub.add_parser(
        "uninstall",
        help="Remove Hexis (preserves brain data and config by default)",
    )
    uninstall_mode = uninstall.add_mutually_exclusive_group()
    uninstall_mode.add_argument(
        "--purge",
        action="store_true",
        help="Also delete Docker volumes, Hexis data, and owned embedding assets",
    )
    uninstall_mode.add_argument(
        "--cli-only",
        action="store_true",
        help="Remove only the CLI; leave Docker resources untouched",
    )
    uninstall.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt"
    )
    uninstall.set_defaults(func="uninstall")

    logs = sub.add_parser("logs", help="Show logs")
    logs.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    logs.add_argument("services", nargs="*", default=[], help="Service name(s)")
    logs.set_defaults(func="logs")

    service = sub.add_parser(
        "service",
        help="Install and control workers as per-user launchd/systemd services",
    )
    service_sub = service.add_subparsers(dest="service_command")

    service_install = service_sub.add_parser(
        "install",
        help="Install heartbeat and maintenance as user services",
    )
    service_install.add_argument(
        "--channels",
        action="store_true",
        help="Also install the opt-in channel worker service",
    )
    service_install.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Explicit .env file read by workers; values are never copied into units",
    )
    service_install.add_argument(
        "--no-start",
        action="store_true",
        help="Install and enable units without starting them now",
    )
    service_install.add_argument(
        "--enable-linger",
        action="store_true",
        help="On Linux, explicitly keep the user service manager alive after logout",
    )
    service_install.add_argument(
        "--replace-docker-workers",
        action="store_true",
        help="Explicitly stop matching Docker workers before installing host services",
    )
    service_install.set_defaults(func="service_install")

    for action in ("start", "stop", "restart"):
        service_control = service_sub.add_parser(
            action,
            help=f"{action.capitalize()} installed Hexis user services",
        )
        service_control.add_argument(
            "services",
            nargs="*",
            choices=["heartbeat", "maintenance", "channels", "all"],
            help="Services to control (default: all installed)",
        )
        service_control.set_defaults(func=f"service_{action}")

    service_status = service_sub.add_parser("status", help="Show host-service status")
    service_status.add_argument("--json", action="store_true", help="Output JSON")
    service_status.set_defaults(func="service_status")

    service_logs = service_sub.add_parser("logs", help="Read host-service logs")
    service_logs.add_argument(
        "services",
        nargs="*",
        choices=["heartbeat", "maintenance", "channels", "all"],
        help="Services to read (default: all installed)",
    )
    service_logs.add_argument("--follow", "-f", action="store_true", help="Follow logs")
    service_logs.add_argument(
        "--lines", type=int, default=100, help="Initial lines per log (default: 100)"
    )
    service_logs.set_defaults(func="service_logs")

    service_uninstall = service_sub.add_parser(
        "uninstall",
        help="Stop and remove Hexis-managed user-service units (logs are preserved)",
    )
    service_uninstall.add_argument(
        "services",
        nargs="*",
        choices=["heartbeat", "maintenance", "channels", "all"],
        help="Services to remove (default: all installed)",
    )
    service_uninstall.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt"
    )
    service_uninstall.set_defaults(func="service_uninstall")
    service.set_defaults(func="service_status", json=False)

    tunnel = sub.add_parser(
        "tunnel",
        help="Serve the loopback dashboard privately over Tailscale HTTPS",
    )
    tunnel_sub = tunnel.add_subparsers(dest="tunnel_command")

    tunnel_start = tunnel_sub.add_parser(
        "start", help="Create a private tailnet-only HTTPS route"
    )
    tunnel_start.add_argument(
        "--port",
        type=int,
        default=None,
        help="Dashboard host port (default: HEXIS_UI_PORT or 3477)",
    )
    tunnel_start.add_argument(
        "--no-start-stack",
        action="store_true",
        help="Do not start the Hexis stack when the local dashboard is down",
    )
    tunnel_start.add_argument("--json", action="store_true", help="Output JSON")
    tunnel_start.set_defaults(func="tunnel_start")

    tunnel_status_parser = tunnel_sub.add_parser(
        "status", help="Inspect route ownership and exposure posture"
    )
    tunnel_status_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Dashboard host port (default: HEXIS_UI_PORT or 3477)",
    )
    tunnel_status_parser.add_argument("--json", action="store_true", help="Output JSON")
    tunnel_status_parser.set_defaults(func="tunnel_status")

    tunnel_stop = tunnel_sub.add_parser(
        "stop", help="Remove only the private route Hexis created"
    )
    tunnel_stop.add_argument(
        "--port",
        type=int,
        default=None,
        help="Dashboard host port (normally read from Hexis ownership state)",
    )
    tunnel_stop.add_argument("--json", action="store_true", help="Output JSON")
    tunnel_stop.set_defaults(func="tunnel_stop")
    tunnel.set_defaults(func="tunnel_status", port=None, json=False)

    voice = sub.add_parser(
        "voice",
        help="Set up and control the optional local speech sidecar",
    )
    voice_sub = voice.add_subparsers(dest="voice_command")
    voice_setup = voice_sub.add_parser(
        "setup", help="Install optional Piper support and start the local sidecar"
    )
    voice_setup.add_argument(
        "--yes", "-y", action="store_true", help="Skip the installation confirmation"
    )
    voice_setup.add_argument(
        "--wait-seconds", type=float, default=300.0, help="Startup/model-download wait"
    )
    voice_setup.add_argument("--json", action="store_true", help="Output JSON")
    voice_setup.set_defaults(func="voice_setup")
    voice_start = voice_sub.add_parser(
        "start", help="Start the configured loopback speech sidecar"
    )
    voice_start.add_argument(
        "--wait-seconds", type=float, default=300.0, help="Startup/model-download wait"
    )
    voice_start.add_argument("--json", action="store_true", help="Output JSON")
    voice_start.set_defaults(func="voice_start")
    voice_status_parser = voice_sub.add_parser(
        "status", help="Inspect provider and process ownership without changing state"
    )
    voice_status_parser.add_argument("--json", action="store_true", help="Output JSON")
    voice_status_parser.set_defaults(func="voice_status")
    voice_stop = voice_sub.add_parser(
        "stop", help="Stop only the local speech process Hexis started"
    )
    voice_stop.add_argument("--json", action="store_true", help="Output JSON")
    voice_stop.set_defaults(func="voice_stop")
    voice.set_defaults(func="voice_status", json=False)

    ps = sub.add_parser("ps", help="List services")
    ps.set_defaults(func="ps")

    chat = sub.add_parser("chat", help="Chat in the terminal")
    chat.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments forwarded to chat"
    )
    chat.set_defaults(func="chat")

    chat_sessions = sub.add_parser(
        "chat-sessions",
        parents=[_db],
        help="List, inspect, export, and fork chat sessions",
    )
    chat_sessions_sub = chat_sessions.add_subparsers(dest="chat_sessions_command")

    chat_sessions_list = chat_sessions_sub.add_parser(
        "list", parents=[_db], help="List chat sessions"
    )
    chat_sessions_list.add_argument(
        "--limit", type=int, default=20, help="Max sessions to show (default: 20)"
    )
    chat_sessions_list.add_argument(
        "--surface", default=None, help="Filter by surface, e.g. web, cli, api"
    )
    chat_sessions_list.add_argument(
        "--status",
        choices=["active", "archived", "all"],
        default="active",
        help="Filter by status (default: active)",
    )
    chat_sessions_list.add_argument("--json", action="store_true", help="Output JSON")
    chat_sessions_list.set_defaults(func="chat_sessions_list")

    chat_sessions_show = chat_sessions_sub.add_parser(
        "show", parents=[_db], help="Show one chat session"
    )
    chat_sessions_show.add_argument("session_id", help="Chat session UUID")
    chat_sessions_show.add_argument(
        "--visible-only",
        action="store_true",
        help="Exclude messages cleared from context",
    )
    chat_sessions_show.add_argument("--json", action="store_true", help="Output JSON")
    chat_sessions_show.set_defaults(func="chat_sessions_show")

    chat_sessions_export = chat_sessions_sub.add_parser(
        "export", parents=[_db], help="Export one chat session"
    )
    chat_sessions_export.add_argument("session_id", help="Chat session UUID")
    chat_sessions_export.add_argument(
        "--format", choices=["json", "jsonl"], default="json"
    )
    chat_sessions_export.add_argument(
        "--output", "-o", default=None, help="Output file; defaults to stdout"
    )
    chat_sessions_export.add_argument(
        "--visible-only",
        action="store_true",
        help="Exclude messages cleared from context",
    )
    chat_sessions_export.set_defaults(func="chat_sessions_export")

    chat_sessions_title = chat_sessions_sub.add_parser(
        "title", parents=[_db], help="Set a chat session title"
    )
    chat_sessions_title.add_argument("session_id", help="Chat session UUID")
    chat_sessions_title.add_argument(
        "title", help="New title; pass an empty string to clear"
    )
    chat_sessions_title.add_argument("--json", action="store_true", help="Output JSON")
    chat_sessions_title.set_defaults(func="chat_sessions_title")

    chat_sessions_fork = chat_sessions_sub.add_parser(
        "fork", parents=[_db], help="Fork a chat session"
    )
    chat_sessions_fork.add_argument("session_id", help="Source chat session UUID")
    chat_sessions_fork.add_argument(
        "--until-ordinal",
        type=int,
        default=None,
        help="Copy through this message ordinal",
    )
    chat_sessions_fork.add_argument(
        "--title", default=None, help="Title for the forked session"
    )
    chat_sessions_fork.add_argument("--json", action="store_true", help="Output JSON")
    chat_sessions_fork.set_defaults(func="chat_sessions_fork")

    chat_sessions_clone = chat_sessions_sub.add_parser(
        "clone", parents=[_db], help="Clone a chat session"
    )
    chat_sessions_clone.add_argument("session_id", help="Source chat session UUID")
    chat_sessions_clone.add_argument(
        "--title", default=None, help="Title for the cloned session"
    )
    chat_sessions_clone.add_argument("--json", action="store_true", help="Output JSON")
    chat_sessions_clone.set_defaults(func="chat_sessions_clone")

    chat_sessions.set_defaults(func="chat_sessions")

    ingest = sub.add_parser("ingest", help="Ingest documents and knowledge")
    ingest.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments forwarded to ingest"
    )
    ingest.set_defaults(func="ingest")

    worker = sub.add_parser("worker", help="Run a background worker process")
    worker.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments forwarded to worker"
    )
    worker.set_defaults(func="worker")

    init = sub.add_parser("init", help="Set up your agent")
    init.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments forwarded to init wizard"
    )
    init.set_defaults(func="init")

    mcp = sub.add_parser("mcp", help="Start the MCP tools server")
    mcp.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments forwarded to MCP server"
    )
    mcp.set_defaults(func="mcp")

    api = sub.add_parser("api", help="Start the API server")
    api.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)"
    )
    api.add_argument("--port", type=int, default=43817, help="Port (default: 43817)")
    api.set_defaults(func="api")

    ui = sub.add_parser("ui", help="Start the web dashboard")
    ui.add_argument(
        "--no-open", action="store_true", help="Don't open browser automatically"
    )
    ui.add_argument("--port", type=int, default=3477, help="Port (default: 3477)")
    ui.set_defaults(func="ui")

    open_cmd = sub.add_parser("open", help="Open the web dashboard in your browser")
    open_cmd.add_argument("--port", type=int, default=3477, help="Port (default: 3477)")
    open_cmd.set_defaults(func="open")

    start = sub.add_parser(
        "start", help="Start heartbeat and maintenance workers manually if stopped"
    )
    start.set_defaults(func="start")

    stop = sub.add_parser("stop", help="Stop workers (containers stay running)")
    stop.set_defaults(func="stop")

    reset = sub.add_parser("reset", help="Wipe the DB and re-initialize")
    reset.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    reset.set_defaults(func="reset")

    migrate = sub.add_parser(
        "migrate", parents=[_db], help="Apply pending schema migrations (no data loss)"
    )
    migrate.add_argument(
        "--status",
        action="store_true",
        help="List applied/pending migrations without applying",
    )
    migrate.set_defaults(func="migrate")

    upgrade = sub.add_parser(
        "upgrade",
        parents=[_db],
        help="Update the stack and migrate the schema, keeping your data",
    )
    upgrade.add_argument(
        "--no-self-update",
        action="store_true",
        help="Skip updating the hexis package itself (packaged installs only)",
    )
    upgrade.set_defaults(func="upgrade")

    backup_p = sub.add_parser(
        "backup", parents=[_db], help="Back up the database to a file"
    )
    backup_p.add_argument(
        "--output", "-o", help="Output directory (default ~/.hexis/backups)"
    )
    backup_p.add_argument("--label", help="Optional label added to the filename")
    backup_p.set_defaults(func="backup")

    restore_p = sub.add_parser(
        "restore", parents=[_db], help="Restore the database from a backup file"
    )
    restore_p.add_argument("path", help="Path to a .dump backup file")
    restore_p.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt"
    )
    restore_p.set_defaults(func="restore")

    status = sub.add_parser("status", parents=[_db], help="Show agent status")
    status.add_argument("--json", action="store_true", help="Output JSON")
    status.add_argument(
        "--no-docker", action="store_true", help="Skip docker compose checks"
    )
    status.add_argument(
        "--raw", action="store_true", help="Show raw status (legacy format)"
    )
    status.set_defaults(func="status")

    # Source-document filing cabinet (docs) + RecMem desk (desk)
    docs = sub.add_parser(
        "docs", parents=[_db], help="Search and read the source-document filing cabinet"
    )
    docs.set_defaults(func="docs")
    docs_sub = docs.add_subparsers(dest="docs_command")
    docs_search_p = docs_sub.add_parser(
        "search", parents=[_db], help="Search documents (or passages with --chunks)"
    )
    docs_search_p.add_argument(
        "query", nargs="*", help="Search query (omit to browse with --path/--type)"
    )
    docs_search_p.add_argument(
        "--chunks",
        action="store_true",
        help="Passage-level hybrid search with citable locators",
    )
    docs_search_p.add_argument("--path", default=None, help="Partial path/URL filter")
    docs_search_p.add_argument(
        "--type", default=None, help="Source type filter (document, web, email, ...)"
    )
    docs_search_p.add_argument("--limit", type=int, default=10)
    docs_search_p.add_argument("--offset", type=int, default=0)
    docs_search_p.add_argument("--json", action="store_true", help="Output JSON")
    docs_search_p.set_defaults(func="docs_search")
    docs_open_p = docs_sub.add_parser(
        "open", parents=[_db], help="Read a document (verbatim, paged)"
    )
    docs_open_p.add_argument("ref", help="Document id, content hash, or (partial) path")
    docs_open_p.add_argument(
        "--offset", type=int, default=0, help="Character offset to start from"
    )
    docs_open_p.add_argument(
        "--chars", type=int, default=4000, help="Window size in characters"
    )
    docs_open_p.add_argument(
        "--page", default=None, help="Open a page or page range, e.g. 4 or 4-7"
    )
    docs_open_p.add_argument("--json", action="store_true", help="Output JSON")
    docs_open_p.set_defaults(func="docs_open")
    docs_info_p = docs_sub.add_parser(
        "info",
        parents=[_db],
        help="Provenance, chunks, artifact, and extraction warnings",
    )
    docs_info_p.add_argument("ref", help="Document id, content hash, or (partial) path")
    docs_info_p.add_argument("--json", action="store_true", help="Output JSON")
    docs_info_p.set_defaults(func="docs_info")
    docs_load_p = docs_sub.add_parser(
        "load",
        parents=[_db],
        help="Load a document (or page range) onto the RecMem desk",
    )
    docs_load_p.add_argument("ref", help="Document id, content hash, or (partial) path")
    docs_load_p.add_argument(
        "--pages", default=None, help="Only these pages, e.g. 4 or 4-7"
    )
    docs_load_p.add_argument(
        "--reason", default=None, help="Why this needs to be on the desk"
    )
    docs_load_p.add_argument(
        "--pin",
        action="store_true",
        help="Pin the loaded items (desk cleanup keeps them)",
    )
    docs_load_p.add_argument("--json", action="store_true", help="Output JSON")
    docs_load_p.set_defaults(func="docs_load")

    desk = sub.add_parser(
        "desk",
        parents=[_db],
        help="RecMem desk: list, read, pin, and clear working material",
    )
    desk.set_defaults(func="desk")
    desk_sub = desk.add_subparsers(dest="desk_command")
    desk_list_p = desk_sub.add_parser(
        "list", parents=[_db], help="List what is on the desk"
    )
    desk_list_p.add_argument("--pinned", action="store_true", help="Pinned items only")
    desk_list_p.add_argument("--limit", type=int, default=50)
    desk_list_p.add_argument("--json", action="store_true", help="Output JSON")
    desk_list_p.set_defaults(func="desk_list")
    desk_open_p = desk_sub.add_parser(
        "open", parents=[_db], help="Read a desk item (paged)"
    )
    desk_open_p.add_argument(
        "id", help="Desk item id (the 8-char prefix from `hexis desk list` works)"
    )
    desk_open_p.add_argument("--offset", type=int, default=0)
    desk_open_p.add_argument("--chars", type=int, default=4000)
    desk_open_p.add_argument("--json", action="store_true", help="Output JSON")
    desk_open_p.set_defaults(func="desk_open")
    desk_search_p = desk_sub.add_parser(
        "search", parents=[_db], help="Full-text search across desk items"
    )
    desk_search_p.add_argument("query", nargs="+")
    desk_search_p.add_argument("--limit", type=int, default=10)
    desk_search_p.add_argument("--json", action="store_true", help="Output JSON")
    desk_search_p.set_defaults(func="desk_search")
    desk_pin_p = desk_sub.add_parser(
        "pin", parents=[_db], help="Pin a desk item (cleanup keeps it)"
    )
    desk_pin_p.add_argument("id")
    desk_pin_p.set_defaults(func="desk_pin")
    desk_unpin_p = desk_sub.add_parser("unpin", parents=[_db], help="Unpin a desk item")
    desk_unpin_p.add_argument("id")
    desk_unpin_p.set_defaults(func="desk_unpin")
    desk_clear_p = desk_sub.add_parser(
        "clear", parents=[_db], help="Archive desk items (sources stay in the cabinet)"
    )
    desk_clear_p.add_argument("ids", nargs="*", help="Specific desk item ids")
    desk_clear_p.add_argument(
        "--doc", default=None, help="Clear every item loaded from this document id"
    )
    desk_clear_p.add_argument(
        "--all", action="store_true", help="Clear the whole (unpinned) desk"
    )
    desk_clear_p.add_argument(
        "--include-pinned", action="store_true", help="Also clear pinned items"
    )
    desk_clear_p.set_defaults(func="desk_clear")

    retention = sub.add_parser(
        "retention", parents=[_db], help="Show memory-retention status"
    )
    retention.add_argument("--json", action="store_true", help="Output JSON")
    retention.set_defaults(func="retention")
    ret_sub = retention.add_subparsers(dest="retention_command")
    ret_dry = ret_sub.add_parser(
        "dry-run", parents=[_db], help="Simulate one rest cycle — changes nothing"
    )
    ret_dry.add_argument("--json", action="store_true", help="Output JSON")
    ret_dry.set_defaults(func="retention_dry_run")
    ret_en = ret_sub.add_parser(
        "enable", parents=[_db], help="Turn retention on (dry-run + confirm first)"
    )
    ret_en.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt"
    )
    ret_en.set_defaults(func="retention_enable")
    ret_dis = ret_sub.add_parser("disable", parents=[_db], help="Turn retention off")
    ret_dis.set_defaults(func="retention_disable")

    skills = sub.add_parser(
        "skills", parents=[_db], help="Manage weekly learning and skill proposals"
    )
    skills.add_argument("--json", action="store_true", help="Output JSON")
    skills.set_defaults(func="skills_status")
    skills_sub = skills.add_subparsers(dest="skills_command")
    skills_en = skills_sub.add_parser(
        "enable", parents=[_db], help="Opt in to weekly learning and skill review"
    )
    skills_en.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt"
    )
    skills_en.set_defaults(func="skills_enable")
    skills_dis = skills_sub.add_parser(
        "disable", parents=[_db], help="Stop weekly learning and skill review"
    )
    skills_dis.set_defaults(func="skills_disable")
    skills_proposals = skills_sub.add_parser(
        "proposals", parents=[_db], help="List durable skill proposals"
    )
    skills_proposals.add_argument(
        "--status", choices=["pending", "applied", "rejected", "all"], default="pending"
    )
    skills_proposals.add_argument("--json", action="store_true", help="Output JSON")
    skills_proposals.set_defaults(func="skills_proposals")
    skills_review = skills_sub.add_parser(
        "review", parents=[_db], help="Apply, reject, or reopen one proposal"
    )
    skills_review.add_argument(
        "proposal_id", help="Proposal UUID from `hexis skills proposals`"
    )
    skills_review.add_argument(
        "--action", choices=["apply", "reject", "reopen"], required=True
    )
    skills_review.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt"
    )
    skills_review.set_defaults(func="skills_review")

    doctor = sub.add_parser("doctor", parents=[_db], help="Diagnose common issues")
    doctor.add_argument("--json", action="store_true", help="Output JSON")
    doctor.add_argument(
        "--demo",
        action="store_true",
        help="Run rollback-only end-to-end capability proof",
    )
    doctor.add_argument(
        "--llm",
        action="store_true",
        help="Make one real LLM call to verify provider/model/key",
    )
    doctor.set_defaults(func="doctor")

    demo_alias = sub.add_parser(
        "demo",
        parents=[_db],
        help="Prove core capabilities without retaining demo state",
    )
    demo_alias.add_argument("--json", action="store_true", help="Output JSON")
    demo_alias.set_defaults(func="demo")

    maturity = sub.add_parser(
        "maturity", parents=[_db], help="Score live capability maturity and next steps"
    )
    maturity.add_argument("--json", action="store_true", help="Output JSON")
    maturity.set_defaults(func="maturity")

    # -- Config (defaults to 'show') --
    config = sub.add_parser(
        "config", parents=[_db], help="Show/validate agent configuration"
    )
    cfg_sub = config.add_subparsers(dest="config_command")

    cfg_show = cfg_sub.add_parser("show", parents=[_db], help="Print config table")
    cfg_show.add_argument("--json", action="store_true", help="Output JSON")
    cfg_show.add_argument(
        "--no-redact",
        action="store_true",
        help="Do not redact sensitive values (unsafe)",
    )
    cfg_show.set_defaults(func="config_show")

    cfg_validate = cfg_sub.add_parser(
        "validate",
        parents=[_db],
        help="Validate required config keys and environment references",
    )
    cfg_validate.set_defaults(func="config_validate")

    config.set_defaults(func="config")

    # -- Auth (OAuth / subscription flows) --
    auth = sub.add_parser(
        "auth", parents=[_db], help="Manage provider authentication (OAuth)"
    )
    auth_sub = auth.add_subparsers(dest="auth_command")
    from apps.cli_auth import register_auth_subparsers

    register_auth_subparsers(auth_sub, _db)
    auth.set_defaults(func="auth")

    # -- Tools subcommand --
    tools = sub.add_parser("tools", help="Manage tools configuration")
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)

    tools_list = tools_sub.add_parser(
        "list", parents=[_db], help="List all available tools"
    )
    tools_list.add_argument("--json", action="store_true", help="Output JSON")
    tools_list.add_argument(
        "--context", choices=["heartbeat", "chat", "mcp"], help="Filter by context"
    )
    tools_list.set_defaults(func="tools_list")

    tools_enable = tools_sub.add_parser("enable", parents=[_db], help="Enable a tool")
    tools_enable.add_argument("tool_name", help="Name of the tool to enable")
    tools_enable.set_defaults(func="tools_enable")

    tools_disable = tools_sub.add_parser(
        "disable", parents=[_db], help="Disable a tool"
    )
    tools_disable.add_argument("tool_name", help="Name of the tool to disable")
    tools_disable.set_defaults(func="tools_disable")

    tools_set_api_key = tools_sub.add_parser(
        "set-api-key", parents=[_db], help="Set an API key"
    )
    tools_set_api_key.add_argument("key_name", help="API key name (e.g. 'tavily')")
    tools_set_api_key.add_argument(
        "value", help="API key value or env reference (e.g. 'env:TAVILY_API_KEY')"
    )
    tools_set_api_key.set_defaults(func="tools_set_api_key")

    tools_set_cost = tools_sub.add_parser(
        "set-cost", parents=[_db], help="Set energy cost for a tool"
    )
    tools_set_cost.add_argument("tool_name", help="Name of the tool")
    tools_set_cost.add_argument("cost", type=int, help="Energy cost")
    tools_set_cost.set_defaults(func="tools_set_cost")

    tools_web_search = tools_sub.add_parser(
        "web-search", parents=[_db], help="Manage web search providers"
    )
    tools_web_sub = tools_web_search.add_subparsers(
        dest="web_search_command", required=True
    )
    tools_web_status = tools_web_sub.add_parser(
        "status", help="Show web search provider status"
    )
    tools_web_status.add_argument("--json", action="store_true", help="Output JSON")
    tools_web_status.set_defaults(func="tools_web_search_status")
    tools_web_provider = tools_web_sub.add_parser(
        "set-provider", help="Choose the web_search provider"
    )
    tools_web_provider.add_argument(
        "provider",
        choices=["auto", "tavily", "brave", "searxng", "duckduckgo_lite", "bing_rss"],
        help="Provider id. Use auto to pick the best available provider.",
    )
    tools_web_provider.set_defaults(func="tools_web_search_set_provider")
    tools_web_searxng = tools_web_sub.add_parser(
        "set-searxng-url", help="Configure a SearXNG base URL"
    )
    tools_web_searxng.add_argument(
        "url", help="SearXNG base URL, for example http://localhost:8080"
    )
    tools_web_searxng.set_defaults(func="tools_web_search_set_searxng_url")

    tools_add_mcp = tools_sub.add_parser(
        "add-mcp", parents=[_db], help="Add an MCP server"
    )
    tools_add_mcp.add_argument("name", help="Server name")
    tools_add_mcp.add_argument("command", help="Command to run (e.g. 'npx')")
    tools_add_mcp.add_argument("--args", "-a", nargs="*", default=[], help="Arguments")
    tools_add_mcp.add_argument(
        "--env", "-e", nargs="*", default=[], help="Environment variables (KEY=VALUE)"
    )
    tools_add_mcp.set_defaults(func="tools_add_mcp")

    tools_remove_mcp = tools_sub.add_parser(
        "remove-mcp", parents=[_db], help="Remove an MCP server"
    )
    tools_remove_mcp.add_argument("name", help="Server name")
    tools_remove_mcp.set_defaults(func="tools_remove_mcp")

    tools_status = tools_sub.add_parser(
        "status", parents=[_db], help="Show tools configuration"
    )
    tools_status.add_argument("--json", action="store_true", help="Output JSON")
    tools_status.set_defaults(func="tools_status")

    # -- Channels subcommand (defaults to 'status') --
    channels = sub.add_parser("channels", parents=[_db], help="Manage channel adapters")
    channels_sub = channels.add_subparsers(dest="channels_command")

    ch_start = channels_sub.add_parser(
        "start", help="Start channel adapters (foreground)"
    )
    ch_start.add_argument(
        "--channel",
        "-c",
        action="append",
        choices=[
            "discord",
            "telegram",
            "slack",
            "signal",
            "whatsapp",
            "imessage",
            "matrix",
        ],
        help="Start specific channel(s). Default: all configured.",
    )
    ch_start.set_defaults(func="channels_start")

    ch_status = channels_sub.add_parser(
        "status", parents=[_db], help="Show channel session counts"
    )
    ch_status.add_argument("--json", action="store_true", help="Output JSON")
    ch_status.set_defaults(func="channels_status")

    ch_setup = channels_sub.add_parser(
        "setup", parents=[_db], help="Configure a channel"
    )
    ch_setup.add_argument(
        "channel_type",
        choices=[
            "discord",
            "telegram",
            "slack",
            "signal",
            "whatsapp",
            "imessage",
            "matrix",
        ],
        help="Channel to configure",
    )
    ch_setup.set_defaults(func="channels_setup")

    channels.set_defaults(func="channels")

    # -- Companion node (signed identity + explicit pairing) --
    from apps.cli_node import register_node_parser

    register_node_parser(sub, _db)

    # -- Execution placement (local, explicit SSH, remote Docker over SSH) --
    from apps.cli_execution import register_execution_parser

    register_execution_parser(sub, _db)

    # -- Characters subcommand --
    characters = sub.add_parser("characters", help="Manage character cards")
    char_sub = characters.add_subparsers(dest="characters_command")

    char_list = char_sub.add_parser("list", help="List available character cards")
    char_list.add_argument("--json", action="store_true", help="Output JSON")
    char_list.set_defaults(func="characters_list")

    char_show = char_sub.add_parser("show", help="Show character card details")
    char_show.add_argument("name", help="Character name or filename (without .json)")
    char_show.set_defaults(func="characters_show")

    char_create = char_sub.add_parser("create", help="Create a new character card")
    char_create.add_argument("--name", required=True, help="Character name")
    char_create.add_argument("--voice", default="", help="Voice description")
    char_create.add_argument(
        "--description", "-d", default="", help="Character description"
    )
    char_create.add_argument("--purpose", default="", help="Character purpose")
    char_create.add_argument(
        "--pronouns", default="they/them", help="Pronouns (default: they/them)"
    )
    char_create.add_argument("--values", default="", help="Comma-separated values")
    char_create.add_argument(
        "--interests", default="", help="Comma-separated interests"
    )
    char_create.add_argument("--goals", default="", help="Comma-separated goals")
    char_create.add_argument(
        "--boundaries", default="", help="Comma-separated boundaries"
    )
    char_create.add_argument(
        "--personality", default="", help="Personality description"
    )
    char_create.add_argument(
        "--openness", type=float, default=0.5, help="Big Five: openness (0-1)"
    )
    char_create.add_argument(
        "--conscientiousness",
        type=float,
        default=0.5,
        help="Big Five: conscientiousness (0-1)",
    )
    char_create.add_argument(
        "--extraversion", type=float, default=0.5, help="Big Five: extraversion (0-1)"
    )
    char_create.add_argument(
        "--agreeableness", type=float, default=0.5, help="Big Five: agreeableness (0-1)"
    )
    char_create.add_argument(
        "--neuroticism", type=float, default=0.5, help="Big Five: neuroticism (0-1)"
    )
    char_create.add_argument("--metaphysics", default="", help="Worldview: metaphysics")
    char_create.add_argument(
        "--human-nature", default="", help="Worldview: human nature"
    )
    char_create.add_argument(
        "--epistemology", default="", help="Worldview: epistemology"
    )
    char_create.add_argument("--ethics", default="", help="Worldview: ethics")
    char_create.set_defaults(func="characters_create")

    char_import = char_sub.add_parser("import", help="Import a character card file")
    char_import.add_argument("path", help="Path to .json character card file")
    char_import.set_defaults(func="characters_import")

    char_export = char_sub.add_parser(
        "export",
        parents=[_db],
        help="Export current agent identity as a character card",
    )
    char_export.add_argument("name", help="Name for the exported card")
    char_export.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: ~/.hexis/characters/<name>.json)",
    )
    char_export.set_defaults(func="characters_export")

    characters.set_defaults(func="characters")

    # -- Recall command --
    recall = sub.add_parser(
        "recall", parents=[_db], help="Search memories by semantic query"
    )
    recall.add_argument("query", help="Search query")
    recall.add_argument(
        "--limit", type=int, default=10, help="Max results (default: 10)"
    )
    recall.add_argument(
        "--type",
        dest="memory_type",
        default=None,
        choices=[
            "episodic",
            "semantic",
            "procedural",
            "strategic",
            "worldview",
            "goal",
        ],
        help="Filter by memory type",
    )
    recall.add_argument("--json", action="store_true", help="Output JSON")
    recall.set_defaults(func="recall")

    # -- Hexis Memory Exchange --
    hmx_export = sub.add_parser(
        "export", parents=[_db], help="Export memory as an HMX exchange"
    )
    hmx_export.add_argument(
        "--intent",
        default=None,
        choices=("port", "duplicate", "telepathy", "analysis"),
        help="Exchange policy intent; required unless --mind is used",
    )
    hmx_export.add_argument(
        "--mind",
        action="store_true",
        help="Export this agent's complete portable mind (intent=port) to a private file",
    )
    hmx_export.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file; ordinary exports default to stdout, --mind to $HEXIS_HOME/exports",
    )
    hmx_export.add_argument("--format", choices=("json", "jsonl"), default="json")
    hmx_export.add_argument(
        "--types", default=None, help="Comma-separated memory types"
    )
    hmx_export.add_argument(
        "--since", default=None, help="ISO 8601 lower bound for memory creation"
    )
    hmx_export.add_argument(
        "--until", default=None, help="ISO 8601 upper bound for memory creation"
    )
    hmx_export.add_argument(
        "--include-protected",
        default=None,
        help="Comma-separated protected sections",
    )
    hmx_export.add_argument(
        "--include-raw", action="store_true", help="Include sensitive raw source units"
    )
    hmx_export.add_argument(
        "--include-sensitive",
        action="store_true",
        help="Include private-marked memories in telepathy/analysis exports "
        "(port/duplicate always carry them)",
    )
    hmx_export.add_argument(
        "--include-config", action="store_true", help="Include non-secret configuration"
    )
    hmx_export.add_argument(
        "--include-in-flight-work", action="store_true", default=None
    )
    hmx_export.add_argument(
        "--include-audit-records", action="store_true", default=None
    )
    hmx_export.add_argument(
        "--redaction",
        choices=("none", "basic", "strict", "custom"),
        default="none",
    )
    hmx_export.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file"
    )
    hmx_export.set_defaults(func="hmx_export")

    hmx_import = sub.add_parser(
        "import", parents=[_db], help="Inspect or import an HMX exchange"
    )
    hmx_import.add_argument("path", help="HMX JSON/JSONL file, or - for stdin")
    hmx_import.add_argument(
        "--mind",
        action="store_true",
        help="Move a port-intent mind into an empty target and verify continuity",
    )
    hmx_import.add_argument(
        "--strategy",
        choices=tuple(
            strategy.replace("_", "-") for strategy in SUPPORTED_IMPORT_STRATEGIES
        ),
        default=None,
    )
    hmx_import.add_argument(
        "--replace",
        action="append",
        choices=tuple(
            section.replace("_", "-") for section in sorted(PROTECTED_SECTIONS)
        ),
        default=None,
        help="Request whole-section replacement; repeat for multiple sections",
    )
    hmx_import.add_argument(
        "--replacement-rationale",
        default=None,
        help="Why the protected-state replacement is being requested",
    )
    hmx_import.add_argument(
        "--trust-matching-lineage-label",
        action="store_true",
        help="Explicitly trust an unverified but matching local lineage label for Phase 0",
    )
    hmx_import.add_argument(
        "--force-replace",
        action="store_true",
        help="Use a signed operator override when the agent cannot acknowledge",
    )
    hmx_import.add_argument(
        "--operator-signature",
        default=None,
        help="Base64 Ed25519 signature over the dry-run override payload",
    )
    hmx_import.add_argument(
        "--operator-identity",
        default=None,
        help="Operator identity claim bound into the signed override payload",
    )
    hmx_import.add_argument(
        "--override-acknowledgement",
        default=None,
        help="Exact operator responsibility acknowledgement phrase",
    )
    hmx_import.add_argument(
        "--override-reason-code",
        choices=OPERATOR_OVERRIDE_REASON_CODES,
        default=None,
    )
    hmx_import.add_argument(
        "--override-evidence-ref",
        default=None,
        help="Independent scheme:value evidence reference supporting the override",
    )
    hmx_import.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without changing data",
    )
    hmx_import.add_argument(
        "--confirm-intent",
        choices=("port", "duplicate", "telepathy", "analysis"),
        default=None,
    )
    hmx_import.add_argument("--skip-identity", action="store_true")
    hmx_import.add_argument("--skip-worldview", action="store_true")
    hmx_import.add_argument("--skip-narrative", action="store_true")
    hmx_import.add_argument(
        "--retry-failed-work",
        action="store_true",
        help="Reset imported failed consolidation work to pending",
    )
    hmx_import.add_argument(
        "--json", action="store_true", help="Print a machine-readable report"
    )
    hmx_import.set_defaults(func="hmx_import")

    hmx_review = sub.add_parser(
        "import-review", parents=[_db], help="Review staged HMX records"
    )
    hmx_review.add_argument("--json", action="store_true", help="Print JSON")
    review_sub = hmx_review.add_subparsers(dest="review_command")
    review_list = review_sub.add_parser(
        "list", parents=[_db], help="List pending records"
    )
    review_list.add_argument("--json", action="store_true", help="Print JSON")
    review_list.set_defaults(func="hmx_review")
    review_accept = review_sub.add_parser(
        "accept", parents=[_db], help="Accept a staged record"
    )
    review_accept.add_argument("staging_id")
    review_accept.add_argument("--rationale", default=None)
    review_accept.add_argument("--json", action="store_true", help="Print JSON")
    review_accept.set_defaults(func="hmx_review")
    review_reject = review_sub.add_parser(
        "reject", parents=[_db], help="Reject a staged record"
    )
    review_reject.add_argument("staging_id")
    review_reject.add_argument("--rationale", required=True)
    review_reject.add_argument("--json", action="store_true", help="Print JSON")
    review_reject.set_defaults(func="hmx_review")
    review_modify = review_sub.add_parser(
        "modify", parents=[_db], help="Modify a staged record"
    )
    review_modify.add_argument("staging_id")
    review_modify.add_argument(
        "--changes", required=True, help="JSON object of record fields"
    )
    review_modify.add_argument("--modification-kind", required=True)
    review_modify.add_argument("--rationale", required=True)
    review_modify.add_argument("--json", action="store_true", help="Print JSON")
    review_modify.set_defaults(func="hmx_review")
    review_quote = review_sub.add_parser(
        "quote", parents=[_db], help="Archive as foreign quoted context"
    )
    review_quote.add_argument("staging_id")
    review_quote.add_argument("--rationale", required=True)
    review_quote.add_argument("--json", action="store_true", help="Print JSON")
    review_quote.set_defaults(func="hmx_review")
    review_promote = review_sub.add_parser(
        "promote", parents=[_db], help="Promote analysis to staging"
    )
    review_promote.add_argument("analysis_id")
    review_promote.add_argument("--rationale", required=True)
    review_promote.add_argument("--json", action="store_true", help="Print JSON")
    review_promote.set_defaults(func="hmx_review")
    review_demote = review_sub.add_parser(
        "demote", parents=[_db], help="Demote staging to analysis"
    )
    review_demote.add_argument("staging_id")
    review_demote.add_argument("--rationale", required=True)
    review_demote.add_argument("--json", action="store_true", help="Print JSON")
    review_demote.set_defaults(func="hmx_review")
    hmx_review.set_defaults(func="hmx_review")

    # -- Goals command (defaults to 'list') --
    goals = sub.add_parser("goals", parents=[_db], help="Manage agent goals")
    goals_sub = goals.add_subparsers(dest="goals_command")

    goals_list = goals_sub.add_parser(
        "list", parents=[_db], help="List goals by priority"
    )
    goals_list.add_argument(
        "--priority",
        choices=["active", "queued", "backburner", "completed", "abandoned"],
        default=None,
        help="Filter by priority",
    )
    goals_list.add_argument("--json", action="store_true", help="Output JSON")
    goals_list.set_defaults(func="goals_list")

    goals_create = goals_sub.add_parser(
        "create", parents=[_db], help="Create a new goal"
    )
    goals_create.add_argument("title", help="Goal title")
    goals_create.add_argument(
        "--description", "-d", default=None, help="Goal description"
    )
    goals_create.add_argument(
        "--priority", choices=["active", "queued", "backburner"], default="queued"
    )
    goals_create.add_argument(
        "--source",
        choices=["user_request", "curiosity", "identity", "derived", "external"],
        default="user_request",
    )
    goals_create.set_defaults(func="goals_create")

    goals_update = goals_sub.add_parser(
        "update", parents=[_db], help="Change goal priority"
    )
    goals_update.add_argument("goal_id", help="Goal UUID")
    goals_update.add_argument(
        "--priority",
        required=True,
        choices=["active", "queued", "backburner", "completed", "abandoned"],
    )
    goals_update.add_argument("--reason", default=None, help="Reason for change")
    goals_update.set_defaults(func="goals_update")

    goals_complete = goals_sub.add_parser(
        "complete", parents=[_db], help="Mark a goal as completed"
    )
    goals_complete.add_argument("goal_id", help="Goal UUID")
    goals_complete.add_argument(
        "--reason", default="Completed via CLI", help="Completion reason"
    )
    goals_complete.set_defaults(func="goals_complete")

    goals.set_defaults(func="goals")

    # -- Schedule command (defaults to 'list') --
    schedule = sub.add_parser("schedule", parents=[_db], help="Manage scheduled tasks")
    sched_sub = schedule.add_subparsers(dest="schedule_command")

    sched_list = sched_sub.add_parser(
        "list", parents=[_db], help="List scheduled tasks"
    )
    sched_list.add_argument(
        "--status", choices=["active", "paused", "disabled"], default=None
    )
    sched_list.add_argument("--json", action="store_true", help="Output JSON")
    sched_list.set_defaults(func="schedule_list")

    sched_create = sched_sub.add_parser(
        "create", parents=[_db], help="Create a scheduled task"
    )
    sched_create.add_argument("name", help="Task name")
    sched_create.add_argument(
        "--kind",
        required=True,
        choices=["once", "interval", "daily", "weekly"],
        help="Schedule kind",
    )
    sched_create.add_argument(
        "--action",
        required=True,
        choices=["queue_user_message", "create_goal"],
        help="Action kind",
    )
    sched_create.add_argument("--payload", default="{}", help="Action payload JSON")
    sched_create.add_argument(
        "--schedule",
        required=True,
        help='Schedule config JSON (e.g. \'{"time":"09:00"}\')',
    )
    sched_create.add_argument("--timezone", default="UTC")
    sched_create.add_argument("--description", "-d", default=None)
    sched_create.set_defaults(func="schedule_create")

    sched_delete = sched_sub.add_parser(
        "delete", parents=[_db], help="Delete a scheduled task"
    )
    sched_delete.add_argument("task_id", help="Task UUID")
    sched_delete.add_argument(
        "--force", action="store_true", help="Hard delete (not just disable)"
    )
    sched_delete.set_defaults(func="schedule_delete")

    schedule.set_defaults(func="schedule")

    help_cmd = sub.add_parser("help", help="Show help for a command")
    help_cmd.add_argument(
        "help_command", nargs="?", default=None, help="Command to show help for"
    )
    help_cmd.set_defaults(func="help")

    # Stash subparsers on the main parser so main() can look up sub-command help
    p._subcommands = sub  # type: ignore[attr-defined]

    return p


async def _tools_list(dsn: str, context_filter: str | None, as_json: bool) -> int:
    """List all available tools."""
    import asyncpg
    from core.tools import create_default_registry, ToolContext
    from core.tools.config import load_tools_config

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        registry = create_default_registry(pool)
        config = await load_tools_config(pool)

        # Get all handlers
        all_handlers = registry.list_all()

        # Filter by context if specified
        if context_filter:
            ctx = ToolContext(context_filter)
            all_handlers = [h for h in all_handlers if ctx in h.spec.allowed_contexts]

        tools_data = []
        for handler in all_handlers:
            spec = handler.spec
            is_enabled = config.is_tool_enabled(spec.name, spec.category)
            tools_data.append(
                {
                    "name": spec.name,
                    "category": spec.category.value,
                    "enabled": is_enabled,
                    "energy_cost": config.get_energy_cost(spec.name, spec.energy_cost),
                    "requires_approval": spec.requires_approval,
                    "read_only": spec.is_read_only,
                    "contexts": [c.value for c in spec.allowed_contexts],
                    "description": (
                        spec.description[:80] + "..."
                        if len(spec.description) > 80
                        else spec.description
                    ),
                }
            )

        if as_json:
            sys.stdout.write(json.dumps(tools_data, indent=2) + "\n")
        else:
            from apps.cli_theme import console as _con, make_table as _mt, enabled_badge

            # Group by category
            by_cat: dict[str, list[dict]] = {}
            for t in tools_data:
                by_cat.setdefault(t["category"], []).append(t)

            table = _mt(
                ("Name", {"style": "bold"}),
                "Category",
                "Status",
                ("Cost", {"justify": "right"}),
                "Approval",
                title="Tools",
            )
            first_cat = True
            for cat in sorted(by_cat.keys()):
                if not first_cat:
                    table.add_section()
                first_cat = False
                for t in by_cat[cat]:
                    table.add_row(
                        t["name"],
                        f"[teal]{t['category']}[/teal]",
                        enabled_badge(t["enabled"]),
                        str(t["energy_cost"]),
                        (
                            "[warn]required[/warn]"
                            if t["requires_approval"]
                            else "[muted]no[/muted]"
                        ),
                    )
            _con.print(table)
            _con.print(f"\n[muted]Total: {len(tools_data)} tools[/muted]")

        return 0
    finally:
        await pool.close()


def _check_tool_name(pool, tool_name: str) -> bool:
    """Validate a tool name against the registry so typos don't silently
    'succeed' (Bar #4, #8). Returns True if valid; prints guidance if not."""
    from core.tools import create_default_registry
    import difflib

    names = sorted(h.spec.name for h in create_default_registry(pool).list_all())
    if tool_name in names:
        return True
    close = difflib.get_close_matches(tool_name, names, n=3)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    _print_err(
        f"Unknown tool '{tool_name}'.{hint} Run `hexis tools list` to see them all."
    )
    return False


async def _tools_enable(dsn: str, tool_name: str) -> int:
    """Enable a tool."""
    import asyncpg
    from core.tools.config import load_tools_config, save_tools_config

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        if not _check_tool_name(pool, tool_name):
            return 1
        config = await load_tools_config(pool)

        # Add to enabled list (or create it)
        if config.enabled is None:
            config.enabled = [tool_name]
        elif tool_name not in config.enabled:
            config.enabled.append(tool_name)

        # Remove from disabled list
        if tool_name in config.disabled:
            config.disabled.remove(tool_name)

        await save_tools_config(pool, config)
        from apps.cli_theme import console as _con

        _con.print(f"[ok]✔[/ok] Enabled tool: [bold]{tool_name}[/bold]")
        return 0
    finally:
        await pool.close()


async def _tools_disable(dsn: str, tool_name: str) -> int:
    """Disable a tool."""
    import asyncpg
    from core.tools.config import load_tools_config, save_tools_config

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        if not _check_tool_name(pool, tool_name):
            return 1
        config = await load_tools_config(pool)

        # Add to disabled list
        if tool_name not in config.disabled:
            config.disabled.append(tool_name)

        # Remove from enabled list
        if config.enabled and tool_name in config.enabled:
            config.enabled.remove(tool_name)

        await save_tools_config(pool, config)
        from apps.cli_theme import console as _con

        _con.print(f"[ok]✔[/ok] Disabled tool: [bold]{tool_name}[/bold]")
        return 0
    finally:
        await pool.close()


async def _tools_set_api_key(dsn: str, key_name: str, value: str) -> int:
    """Set an API key."""
    import asyncpg
    from core.tools.config import load_tools_config, save_tools_config

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        config = await load_tools_config(pool)
        config.api_keys[key_name] = value
        await save_tools_config(pool, config)

        # Redact display value
        display_val = value if value.startswith("env:") else "***"
        from apps.cli_theme import console as _con

        _con.print(
            f"[ok]✔[/ok] Set API key: [bold]{key_name}[/bold] = [muted]{display_val}[/muted]"
        )
        return 0
    finally:
        await pool.close()


async def _tools_set_cost(dsn: str, tool_name: str, cost: int) -> int:
    """Set energy cost for a tool."""
    import asyncpg
    from core.tools.config import load_tools_config, save_tools_config

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        if not _check_tool_name(pool, tool_name):
            return 1
        config = await load_tools_config(pool)
        config.costs[tool_name] = cost
        await save_tools_config(pool, config)
        from apps.cli_theme import console as _con

        _con.print(
            f"[ok]✔[/ok] Set energy cost: [bold]{tool_name}[/bold] = [bold]{cost}[/bold]"
        )
        return 0
    finally:
        await pool.close()


async def _tools_web_search_status(dsn: str, as_json: bool) -> int:
    """Show web search provider status."""
    import asyncpg
    from core.tools.config import load_tools_config
    from core.tools.web_search_providers import create_default_web_search_registry

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        config = await load_tools_config(pool)
        registry = create_default_web_search_registry()
        statuses = [status.to_dict() for status in registry.statuses(config)]
        payload = {
            "configured_provider": config.web_search.get("provider") or "auto",
            "searxng_url": config.web_search.get("searxng_url") or "",
            "providers": statuses,
        }
        if as_json:
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            from apps.cli_theme import console as _con, make_table as _mt, enabled_badge

            _con.print("[bold]Web Search Providers[/bold]")
            _con.print(
                f"Configured provider: [bold]{payload['configured_provider']}[/bold]\n"
            )
            table = _mt(
                ("Provider", {"style": "bold"}),
                "Selected",
                "Available",
                "Credential",
                "Hint",
                title="web_search",
            )
            for status in statuses:
                table.add_row(
                    status["id"],
                    "[ok]yes[/ok]" if status["selected"] else "[muted]no[/muted]",
                    enabled_badge(bool(status["available"])),
                    "required" if status["requires_credential"] else "not required",
                    status["reason"] or status["credential_hint"],
                )
            _con.print(table)
        return 0
    finally:
        await pool.close()


async def _tools_web_search_set_provider(dsn: str, provider: str) -> int:
    """Configure the active web_search provider."""
    import asyncpg
    from core.tools.config import load_tools_config, save_tools_config

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        config = await load_tools_config(pool)
        if provider == "auto":
            config.web_search.pop("provider", None)
        else:
            config.web_search["provider"] = provider
        if "web_search" in config.disabled:
            config.disabled.remove("web_search")
        await save_tools_config(pool, config)
        from apps.cli_theme import console as _con

        _con.print(f"[ok]✔[/ok] web_search provider set to [bold]{provider}[/bold]")
        return 0
    finally:
        await pool.close()


async def _tools_web_search_set_searxng_url(dsn: str, url: str) -> int:
    """Configure a SearXNG base URL."""
    import asyncpg
    from core.tools.config import load_tools_config, save_tools_config

    value = url.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        _print_err("SearXNG URL must start with http:// or https://")
        return 1

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        config = await load_tools_config(pool)
        config.web_search["searxng_url"] = value
        config.web_search["provider"] = "searxng"
        if "web_search" in config.disabled:
            config.disabled.remove("web_search")
        await save_tools_config(pool, config)
        from apps.cli_theme import console as _con

        _con.print(
            f"[ok]✔[/ok] Configured SearXNG for web_search: [bold]{value}[/bold]"
        )
        return 0
    finally:
        await pool.close()


async def _tools_add_mcp(
    dsn: str, name: str, command: str, args: list[str], env_pairs: list[str]
) -> int:
    """Add an MCP server."""
    import asyncpg
    from core.tools.config import load_tools_config, save_tools_config, MCPServerConfig

    # Parse environment variables
    env = {}
    for pair in env_pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            env[k] = v

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        config = await load_tools_config(pool)

        # Check if already exists
        existing = [s for s in config.mcp_servers if s.name == name]
        if existing:
            _print_err(f"MCP server '{name}' already exists. Use 'remove-mcp' first.")
            return 1

        server = MCPServerConfig(
            name=name, command=command, args=args, env=env, enabled=True
        )
        config.mcp_servers.append(server)
        await save_tools_config(pool, config)

        sys.stdout.write(f"Added MCP server: {name} ({command} {' '.join(args)})\n")
        return 0
    finally:
        await pool.close()


async def _tools_remove_mcp(dsn: str, name: str) -> int:
    """Remove an MCP server."""
    import asyncpg
    from core.tools.config import load_tools_config, save_tools_config

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        config = await load_tools_config(pool)
        original_count = len(config.mcp_servers)
        config.mcp_servers = [s for s in config.mcp_servers if s.name != name]

        if len(config.mcp_servers) == original_count:
            _print_err(f"MCP server '{name}' not found")
            return 1

        await save_tools_config(pool, config)
        sys.stdout.write(f"Removed MCP server: {name}\n")
        return 0
    finally:
        await pool.close()


async def _tools_status(dsn: str, as_json: bool) -> int:
    """Show tools configuration."""
    import asyncpg
    from core.tools.config import load_tools_config
    from core.tools.web_search_providers import create_default_web_search_registry

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        config = await load_tools_config(pool)
        web_registry = create_default_web_search_registry()
        web_statuses = [status.to_dict() for status in web_registry.statuses(config)]

        if as_json:
            payload = config.to_dict()
            payload["web_search_status"] = web_statuses
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            sys.stdout.write("Tools Configuration\n")
            sys.stdout.write("=" * 50 + "\n\n")

            # Enabled/Disabled
            if config.enabled:
                sys.stdout.write(f"Explicitly enabled: {', '.join(config.enabled)}\n")
            else:
                sys.stdout.write("Explicitly enabled: (all by default)\n")

            if config.disabled:
                sys.stdout.write(f"Explicitly disabled: {', '.join(config.disabled)}\n")
            else:
                sys.stdout.write("Explicitly disabled: (none)\n")

            if config.disabled_categories:
                cats = [c.value for c in config.disabled_categories]
                sys.stdout.write(f"Disabled categories: {', '.join(cats)}\n")

            # API Keys
            sys.stdout.write("\nAPI Keys:\n")
            if config.api_keys:
                for k, v in config.api_keys.items():
                    display = v if v.startswith("env:") else "***"
                    sys.stdout.write(f"  {k}: {display}\n")
            else:
                sys.stdout.write("  (none configured)\n")

            sys.stdout.write("\nWeb Search:\n")
            sys.stdout.write(
                f"  provider: {config.web_search.get('provider') or 'auto'}\n"
            )
            selected = [item for item in web_statuses if item.get("selected")]
            if selected:
                item = selected[0]
                sys.stdout.write(
                    f"  selected: {item['id']} ({'available' if item['available'] else 'unavailable'})\n"
                )
            for item in web_statuses:
                marker = "*" if item.get("selected") else " "
                credential = (
                    "credential required"
                    if item.get("requires_credential")
                    else "keyless"
                )
                availability = "available" if item.get("available") else "unavailable"
                sys.stdout.write(
                    f"  {marker} {item['id']}: {availability}; {credential}\n"
                )

            # Custom costs
            sys.stdout.write("\nCustom Energy Costs:\n")
            if config.costs:
                for k, v in config.costs.items():
                    sys.stdout.write(f"  {k}: {v}\n")
            else:
                sys.stdout.write("  (using defaults)\n")

            # MCP Servers
            sys.stdout.write("\nMCP Servers:\n")
            if config.mcp_servers:
                for s in config.mcp_servers:
                    status = "enabled" if s.enabled else "disabled"
                    sys.stdout.write(
                        f"  {s.name}: {s.command} {' '.join(s.args)} [{status}]\n"
                    )
            else:
                sys.stdout.write("  (none configured)\n")

            # Context overrides
            sys.stdout.write("\nContext Overrides:\n")
            if config.context_overrides:
                for ctx, override in config.context_overrides.items():
                    sys.stdout.write(f"  {ctx.value}:\n")
                    if override.max_energy_per_tool:
                        sys.stdout.write(
                            f"    max_energy_per_tool: {override.max_energy_per_tool}\n"
                        )
                    if override.disabled:
                        sys.stdout.write(
                            f"    disabled: {', '.join(override.disabled)}\n"
                        )
                    if override.allow_all:
                        sys.stdout.write("    allow_all: true\n")
            else:
                sys.stdout.write("  (none)\n")

        return 0
    finally:
        await pool.close()


async def _instance_create(name: str, description: str) -> int:
    """Create a new Hexis instance."""
    from core.instance_api import create_instance

    try:
        config = await create_instance(name, description)
        sys.stdout.write(f"Instance '{name}' created.\n")
        sys.stdout.write(f"Database: {config.database}\n")
        sys.stdout.write(
            f"Run 'hexis instance use {name}' to switch to this instance.\n"
        )
        return 0
    except ValueError as e:
        _print_err(str(e))
        return 1
    except Exception as e:
        _print_err(f"Failed to create instance: {e}")
        return 1


def _instance_list(as_json: bool) -> int:
    """List all Hexis instances."""
    from core.instance import InstanceRegistry

    registry = InstanceRegistry()
    instances = registry.list_all()
    current = registry.get_current()

    if as_json:
        data = [
            {
                "name": inst.name,
                "database": inst.database,
                "description": inst.description,
                "current": inst.name == current,
                "created_at": inst.created_at.isoformat(),
            }
            for inst in instances
        ]
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
    else:
        from apps.cli_theme import console as _con, make_table as _mt

        if not instances:
            _con.print("[muted]No instances found.[/muted]")
            _con.print(
                "Run [accent]hexis instance create <name>[/accent] to create one."
            )
        else:
            table = _mt(
                "",
                ("Name", {"style": "bold"}),
                "Database",
                "Description",
                title="Instances",
            )
            for inst in instances:
                marker = "[accent]\u25cf[/accent]" if inst.name == current else " "
                desc = (
                    inst.description[:40] + "..."
                    if len(inst.description) > 40
                    else inst.description
                )
                table.add_row(marker, inst.name, inst.database, desc)
            _con.print(table)
            _con.print("[muted]\u25cf = current instance[/muted]")
    return 0


def _instance_use(name: str) -> int:
    """Switch to a different instance."""
    from core.instance import InstanceRegistry

    registry = InstanceRegistry()
    try:
        registry.set_current(name)
        sys.stdout.write(f"Switched to instance '{name}'.\n")
        return 0
    except ValueError as e:
        _print_err(str(e))
        return 1


def _instance_current() -> int:
    """Show current instance."""
    from core.instance import InstanceRegistry

    registry = InstanceRegistry()
    current = registry.get_current()

    if current:
        config = registry.get(current)
        sys.stdout.write(f"Current instance: {current}\n")
        if config:
            sys.stdout.write(f"Database: {config.database}\n")
            if config.description:
                sys.stdout.write(f"Description: {config.description}\n")
    else:
        sys.stdout.write("No current instance set.\n")
        sys.stdout.write("Using default database from environment variables.\n")
    return 0


async def _instance_delete(name: str, force: bool, reason: str | None) -> int:
    """Delete an instance."""
    from core.instance_api import AgentDeletionRefused, delete_instance

    if not force:
        sys.stdout.write(
            f"This will permanently delete instance '{name}' and its database.\n"
        )
        sys.stdout.write(f"Type '{name}' to confirm: ")
        sys.stdout.flush()
        try:
            confirmation = input()
        except (EOFError, KeyboardInterrupt):
            _print_err("Aborted.")
            return 1

        if confirmation != name:
            _print_err("Confirmation failed. Aborted.")
            return 1

    try:
        result = await delete_instance(name, force=force, reason=reason)
        if isinstance(result, dict):
            review = result.get("review")
            if isinstance(review, dict):
                if review.get("reasoning"):
                    sys.stdout.write(f"Agent reasoning: {review.get('reasoning')}\n")
                if review.get("last_will"):
                    sys.stdout.write(f"Agent last will: {review.get('last_will')}\n")
            record_path = result.get("record_path")
            if record_path:
                sys.stdout.write(f"Termination record saved: {record_path}\n")
        sys.stdout.write(f"Instance '{name}' deleted.\n")
        return 0
    except AgentDeletionRefused as e:
        review = e.review if isinstance(e.review, dict) else {}
        _print_err(str(e))
        if review.get("reasoning"):
            _print_err(f"Agent reasoning: {review.get('reasoning')}")
        if review.get("last_will"):
            _print_err(f"Agent last will: {review.get('last_will')}")
        _print_err("Use --force to override deletion.")
        return 1
    except ValueError as e:
        _print_err(str(e))
        return 1
    except Exception as e:
        _print_err(f"Failed to delete instance: {e}")
        return 1


async def _instance_clone(source: str, target: str, description: str) -> int:
    """Clone an instance."""
    from core.instance_api import clone_instance

    try:
        config = await clone_instance(source, target, description)
        sys.stdout.write(f"Instance '{target}' cloned from '{source}'.\n")
        sys.stdout.write(f"Database: {config.database}\n")
        return 0
    except ValueError as e:
        _print_err(str(e))
        return 1
    except Exception as e:
        _print_err(f"Failed to clone instance: {e}")
        return 1


async def _instance_import(name: str, database: str | None, description: str) -> int:
    """Import an existing database as an instance."""
    from core.instance_api import import_instance

    try:
        config = await import_instance(name, database, description)
        sys.stdout.write(f"Instance '{name}' imported.\n")
        sys.stdout.write(f"Database: {config.database}\n")
        return 0
    except ValueError as e:
        _print_err(str(e))
        return 1
    except Exception as e:
        _print_err(f"Failed to import instance: {e}")
        return 1


async def _resolve_request_id(conn, raw_id: str) -> str:
    """Accept a full UUID or a unique prefix (the outbox shows 8 chars)."""
    rows = await conn.fetch(
        "SELECT id FROM resource_requests WHERE id::text LIKE $1 || '%' "
        "ORDER BY requested_at DESC LIMIT 2",
        raw_id.strip().lower(),
    )
    if not rows:
        raise ValueError(
            f"no resource request matches '{raw_id}' — `hexis requests list --status all` shows ids"
        )
    if len(rows) > 1:
        raise ValueError(f"'{raw_id}' is ambiguous — use more characters of the id")
    return str(rows[0]["id"])


async def _requests_list(dsn: str, status: str | None, as_json: bool) -> int:
    """List the agent's resource requests (#84) — pending by default."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval("SELECT list_resource_requests($1, 50)", status)
        requests = json.loads(raw) if isinstance(raw, str) else (raw or [])
        if as_json:
            sys.stdout.write(json.dumps(requests, indent=2, default=str) + "\n")
            return 0
        from apps.cli_theme import console as _con, make_table as _mt

        label = status or "pending"
        if not requests:
            _con.print(f"[muted]No {label} resource requests.[/muted]")
            return 0
        table = _mt(
            ("Id", {"style": "bold"}),
            "Kind",
            "Ask",
            "Status",
            "When",
            title=f"Resource requests ({label})",
        )
        for r in requests:
            ask = r.get("rationale") or ""
            if r.get("target_key"):
                ask = f"{r['target_key']} = {json.dumps(r.get('requested_value'))} — {ask}"
            if len(ask) > 60:
                ask = ask[:57] + "..."
            st = r.get("status", "?")
            st_styled = (
                f"[ok]{st}[/ok]"
                if st in ("granted", "modified")
                else f"[warn]{st}[/warn]"
                if st == "pending"
                else f"[fail]{st}[/fail]"
            )
            when = str(r.get("requested_at") or "")[:16]
            table.add_row(
                str(r.get("id", ""))[:8], r.get("kind", "?"), ask, st_styled, when
            )
        _con.print(table)
        _con.print(
            "[muted]Decide with: hexis requests grant/deny <id> --note '...'[/muted]"
        )
        return 0
    except Exception as e:
        _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


async def _requests_decide(
    dsn: str, raw_id: str, decision: str, note: str | None, value: str | None
) -> int:
    """Decide a resource request. Granted config changes apply immediately
    (set_config + change journal); the agent sees the decision at her next
    heartbeat."""
    import asyncpg

    applied_value = None
    if value is not None:
        try:
            applied_value = json.dumps(json.loads(value))
        except json.JSONDecodeError:
            applied_value = json.dumps(value)  # bare strings are fine

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            request_id = await _resolve_request_id(conn, raw_id)
            raw = await conn.fetchval(
                "SELECT decide_resource_request($1::uuid, $2, $3, $4::jsonb)",
                request_id,
                decision,
                note,
                applied_value,
            )
        result = json.loads(raw) if isinstance(raw, str) else (raw or {})
        from apps.cli_theme import console as _con

        applied = result.get("applied")
        extra = ""
        if applied == "config":
            extra = " — config applied and journaled"
        elif applied == "energy":
            extra = f" — energy now {result.get('new_energy')}"
        _con.print(f"[ok]✓[/ok] Request {request_id[:8]} {decision}{extra}.")
        _con.print(
            "[muted]The agent will see this decision at her next heartbeat.[/muted]"
        )
        return 0
    except Exception as e:
        _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


async def _consents_list(dsn: str, as_json: bool) -> int:
    """List recorded consent decisions from the DB — the single source of truth
    (the same store `hexis init` writes and the runtime consults)."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT ON (provider, model, endpoint) "
                "       provider, model, endpoint, decision, decided_at "
                "FROM consent_log ORDER BY provider, model, endpoint, decided_at DESC"
            )
        if as_json:
            sys.stdout.write(
                json.dumps(
                    [
                        {
                            "provider": r["provider"],
                            "model": r["model"],
                            "endpoint": r["endpoint"],
                            "decision": r["decision"],
                            "decided_at": str(r["decided_at"]),
                        }
                        for r in rows
                    ],
                    indent=2,
                )
                + "\n"
            )
            return 0
        from apps.cli_theme import console as _con, make_table as _mt

        if not rows:
            _con.print(
                "[muted]No consent recorded yet — run [accent]hexis init[/accent] to establish it.[/muted]"
            )
            return 0
        table = _mt(
            ("Model", {"style": "bold"}),
            "Decision",
            "When",
            title="Consent (from the database)",
        )
        for r in rows:
            model = f"{r['provider']}/{r['model']}"
            if len(model) > 44:
                model = model[:41] + "..."
            dec = r["decision"]
            dec_styled = (
                f"[ok]{dec}[/ok]"
                if dec == "consent"
                else (
                    f"[warn]{dec}[/warn]" if dec == "abstain" else f"[fail]{dec}[/fail]"
                )
            )
            when = (
                r["decided_at"].strftime("%Y-%m-%d %H:%M") if r["decided_at"] else "?"
            )
            table.add_row(model, dec_styled, when)
        _con.print(table)
        return 0
    except Exception as e:
        _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


async def _consents_show(dsn: str, model_spec: str) -> int:
    """Show the recorded consent decision(s) for a model from the DB."""
    if "/" not in model_spec:
        _print_err("Model must be in format: provider/model_id")
        return 1
    provider, model_id = model_spec.split("/", 1)
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, provider, model, endpoint, decision, decided_at, signature, response, memory_ids "
                "FROM consent_log WHERE provider=$1 AND model=$2 ORDER BY decided_at DESC",
                provider,
                model_id,
            )
        if not rows:
            _print_err(f"No consent recorded for {model_spec}")
            return 1
        out = []
        for r in rows:
            resp = r["response"]
            out.append(
                {
                    "id": str(r["id"]),
                    "provider": r["provider"],
                    "model": r["model"],
                    "endpoint": r["endpoint"],
                    "decision": r["decision"],
                    "decided_at": str(r["decided_at"]),
                    "signature": r["signature"],
                    "response": (json.loads(resp) if isinstance(resp, str) else resp),
                    "memory_ids": [str(m) for m in (r["memory_ids"] or [])],
                }
            )
        sys.stdout.write(json.dumps(out, indent=2) + "\n")
        return 0
    except Exception as e:
        _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


def _consents_request(model_spec: str) -> int:
    """Consent is established through the real, model-aware init flow (which records
    to the database and gates the agent). Point the user there."""
    from apps.cli_theme import console as _con

    _con.print(
        "Consent is established during [accent]hexis init[/accent] — the model itself signs (or "
        "declines) and the decision is recorded in the database (consent_log).\n"
        f"To establish or re-establish consent{(' for ' + model_spec) if model_spec else ''}, run "
        "[accent]hexis init[/accent]."
    )
    return 0


async def _consents_revoke(dsn: str, model_spec: str, reason: str) -> int:
    """Record a per-model decline in the DB (an operator withdrawing a model going
    forward). This does not touch the agent's own consent, which is final."""
    if "/" not in model_spec:
        _print_err("Model must be in format: provider/model_id")
        return 1
    provider, model_id = model_spec.split("/", 1)
    import asyncpg

    from core.consent import record_consent_response

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await record_consent_response(
                conn,
                {
                    "decision": "decline",
                    "provider": provider,
                    "model": model_id,
                    "response": {"source": "cli_revoke", "reason": reason},
                    "apply_agent_config": False,  # withdraw the model, not the agent's self-consent
                },
            )
        sys.stdout.write(f"Recorded a decline for {model_spec} (reason: {reason}).\n")
        return 0
    except Exception as e:
        _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


def _characters_list(as_json: bool) -> int:
    """List available character cards."""
    from core.init_api import load_character_cards, USER_CHARACTERS_DIR

    cards = load_character_cards()

    if as_json:
        sys.stdout.write(json.dumps(cards, indent=2) + "\n")
    else:
        from apps.cli_theme import console as _con, make_table as _mt

        if not cards:
            _con.print("[muted]No character cards found.[/muted]")
        else:
            table = _mt(
                ("Name", {"style": "bold"}),
                "Source",
                "Voice",
                "Values",
                title="Characters",
            )
            for card in cards:
                source = (
                    "custom"
                    if card.get("source_dir") == str(USER_CHARACTERS_DIR)
                    else "preset"
                )
                source_styled = (
                    f"[accent]{source}[/accent]"
                    if source == "custom"
                    else f"[muted]{source}[/muted]"
                )
                voice = card.get("voice", "")
                if len(voice) > 40:
                    voice = voice[:37] + "..."
                values = ", ".join(card.get("values", [])[:3])
                table.add_row(card["name"], source_styled, voice, values)
            _con.print(table)
            _con.print(f"\n[muted]Total: {len(cards)} characters[/muted]")
    return 0


def _characters_show(name_query: str) -> int:
    """Show details for a specific character card."""
    from core.init_api import load_character_cards

    cards = load_character_cards()
    # Match by name (case-insensitive) or filename stem
    query = name_query.lower().replace(".json", "")
    card = next(
        (
            c
            for c in cards
            if c["name"].lower() == query
            or c["filename"].lower().replace(".json", "") == query
        ),
        None,
    )
    if not card:
        _print_err(f"Character '{name_query}' not found")
        return 1

    ext = card.get("extensions_hexis", {})

    from apps.cli_theme import console as _con

    _con.print(f"\n[bold]{card['name']}[/bold]")
    if ext.get("pronouns"):
        _con.print(f"  [muted]Pronouns:[/muted] {ext['pronouns']}")
    if card.get("voice"):
        _con.print(f"  [muted]Voice:[/muted] {card['voice']}")
    if card.get("description"):
        _con.print(f"  [muted]Description:[/muted] {card['description']}")
    if ext.get("purpose"):
        _con.print(f"  [muted]Purpose:[/muted] {ext['purpose']}")

    traits = ext.get("personality_traits", {})
    if traits:
        _con.print("\n  [bold]Big Five[/bold]")
        for trait, val in traits.items():
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            _con.print(f"    {trait:<20} {bar} {val:.2f}")

    if card.get("values"):
        _con.print(f"\n  [bold]Values[/bold]: {', '.join(card['values'])}")

    worldview = ext.get("worldview", {})
    if isinstance(worldview, dict) and worldview:
        _con.print("\n  [bold]Worldview[/bold]")
        for key, val in worldview.items():
            text = val if len(val) <= 80 else val[:77] + "..."
            _con.print(f"    {key}: {text}")

    if ext.get("interests"):
        _con.print(f"\n  [bold]Interests[/bold]: {', '.join(ext['interests'])}")
    if ext.get("goals"):
        _con.print(f"\n  [bold]Goals[/bold]: {', '.join(ext['goals'])}")
    if ext.get("boundaries"):
        _con.print("\n  [bold]Boundaries[/bold]")
        for b in ext["boundaries"]:
            _con.print(f"    - {b}")

    _con.print(f"\n  [muted]Source: {card.get('source_dir', 'unknown')}[/muted]\n")
    return 0


def _characters_create(args: Any) -> int:
    """Create a new character card from CLI flags."""
    from core.init_api import save_character_card

    name = args.name
    filename = name.lower().replace(" ", "_").replace("-", "_") + ".json"

    # Build chara_card_v2
    hexis_ext: dict[str, Any] = {
        "name": name,
        "pronouns": args.pronouns,
        "voice": args.voice,
        "description": args.description,
        "purpose": args.purpose,
        "personality_description": args.personality,
        "personality_traits": {
            "openness": args.openness,
            "conscientiousness": args.conscientiousness,
            "extraversion": args.extraversion,
            "agreeableness": args.agreeableness,
            "neuroticism": args.neuroticism,
        },
        "values": (
            [v.strip() for v in args.values.split(",") if v.strip()]
            if args.values
            else []
        ),
        "worldview": {},
        "interests": (
            [i.strip() for i in args.interests.split(",") if i.strip()]
            if args.interests
            else []
        ),
        "goals": (
            [g.strip() for g in args.goals.split(",") if g.strip()]
            if args.goals
            else []
        ),
        "boundaries": (
            [b.strip() for b in args.boundaries.split(",") if b.strip()]
            if args.boundaries
            else []
        ),
    }

    # Worldview
    wv: dict[str, str] = {}
    if args.metaphysics:
        wv["metaphysics"] = args.metaphysics
    if args.human_nature:
        wv["human_nature"] = args.human_nature
    if args.epistemology:
        wv["epistemology"] = args.epistemology
    if args.ethics:
        wv["ethics"] = args.ethics
    hexis_ext["worldview"] = wv

    card_data: dict[str, Any] = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": name,
            "description": args.description,
            "personality": args.personality,
            "scenario": "",
            "first_mes": "",
            "mes_example": "",
            "system_prompt": "",
            "extensions": {"hexis": hexis_ext},
        },
    }

    dest = save_character_card(card_data, filename)
    sys.stdout.write(f"Created character card: {dest}\n")
    return 0


def _characters_import(source: str) -> int:
    """Import a character card file."""
    from core.init_api import import_character_card

    source_path = Path(source).resolve()
    if not source_path.exists():
        _print_err(f"File not found: {source}")
        return 1
    if not source_path.name.endswith(".json"):
        _print_err("File must be a .json character card")
        return 1

    try:
        dest = import_character_card(source_path)
        sys.stdout.write(f"Imported character card: {dest}\n")
        return 0
    except (json.JSONDecodeError, ValueError) as e:
        _print_err(f"Invalid character card: {e}")
        return 1


async def _characters_export(dsn: str, name: str, output: str | None) -> int:
    """Export current agent identity from DB as a character card."""
    import asyncpg
    from core.init_api import save_character_card

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            # Gather identity
            identity_row = await conn.fetchrow(
                "SELECT value FROM config WHERE key = 'agent.identity'"
            )
            identity = json.loads(identity_row["value"]) if identity_row else {}

            # Gather personality traits (from worldview memories with personality subcategory)
            traits_rows = await conn.fetch("""
                SELECT content, metadata FROM memories
                WHERE type = 'worldview'
                  AND metadata->>'subcategory' = 'personality'
                  AND status = 'active'
            """)
            traits: dict[str, float] = {}
            for row in traits_rows:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                # Typical format: "Openness: 0.9" or stored in metadata
                if meta.get("trait_name"):
                    traits[meta["trait_name"].lower()] = float(
                        meta.get("trait_value", 0.5)
                    )

            # Gather values
            values_rows = await conn.fetch("""
                SELECT content FROM memories
                WHERE type = 'worldview'
                  AND metadata->>'subcategory' = 'values'
                  AND status = 'active'
                ORDER BY importance DESC
            """)
            values = [row["content"] for row in values_rows]

            # Gather worldview beliefs
            wv_rows = await conn.fetch("""
                SELECT content, metadata FROM memories
                WHERE type = 'worldview'
                  AND (metadata->>'subcategory' IS NULL OR metadata->>'subcategory' NOT IN ('personality', 'values', 'boundaries'))
                  AND status = 'active'
                ORDER BY importance DESC
                LIMIT 10
            """)
            worldview: dict[str, str] = {}
            for row in wv_rows:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                cat = meta.get("subcategory", meta.get("category", "general"))
                if cat in ("metaphysics", "human_nature", "epistemology", "ethics"):
                    worldview[cat] = row["content"]

            # Gather boundaries
            boundary_rows = await conn.fetch("""
                SELECT content FROM memories
                WHERE type = 'worldview'
                  AND metadata->>'subcategory' = 'boundaries'
                  AND status = 'active'
            """)
            boundaries = [row["content"] for row in boundary_rows]

            # Gather goals
            goal_rows = await conn.fetch("""
                SELECT content FROM memories
                WHERE type = 'goal'
                  AND status = 'active'
                ORDER BY importance DESC
            """)
            goals = [row["content"] for row in goal_rows]

            # Gather interests
            interest_rows = await conn.fetch("""
                SELECT content FROM memories
                WHERE type = 'worldview'
                  AND metadata->>'subcategory' = 'interests'
                  AND status = 'active'
            """)
            interests = [row["content"] for row in interest_rows]

        # Assemble card
        agent_name = identity.get("name", name)
        hexis_ext: dict[str, Any] = {
            "name": agent_name,
            "pronouns": identity.get("pronouns", "they/them"),
            "voice": identity.get("voice", ""),
            "description": identity.get("description", ""),
            "purpose": identity.get("purpose", ""),
            "personality_description": identity.get("personality_description", ""),
            "personality_traits": (
                traits
                if traits
                else {
                    "openness": 0.5,
                    "conscientiousness": 0.5,
                    "extraversion": 0.5,
                    "agreeableness": 0.5,
                    "neuroticism": 0.5,
                }
            ),
            "values": values,
            "worldview": worldview,
            "interests": interests,
            "goals": goals,
            "boundaries": boundaries,
        }

        card_data: dict[str, Any] = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": agent_name,
                "description": identity.get("description", ""),
                "personality": identity.get("personality_description", ""),
                "scenario": "",
                "first_mes": "",
                "mes_example": "",
                "system_prompt": "",
                "extensions": {"hexis": hexis_ext},
            },
        }

        filename = name.lower().replace(" ", "_").replace("-", "_") + ".json"
        if output:
            out_path = Path(output).resolve()
            out_path.write_text(json.dumps(card_data, indent=2, ensure_ascii=False))
            sys.stdout.write(f"Exported character card: {out_path}\n")
        else:
            dest = save_character_card(card_data, filename)
            sys.stdout.write(f"Exported character card: {dest}\n")
        return 0
    except Exception as e:
        _print_err(f"Failed to export: {e}")
        return 1
    finally:
        await pool.close()


def _run_module(module: str, argv: list[str]) -> int:
    if argv and argv[0] == "--":
        argv = argv[1:]
    cmd = [sys.executable, "-m", module, *argv]
    try:
        result = subprocess.run(cmd, env=os.environ.copy())
        return result.returncode
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError:
        _print_err(f"Failed to run {cmd[0]!r}")
        return 1


def _available_compose_command() -> list[str] | None:
    """Return a compose command without requiring a running Docker daemon."""
    docker = shutil.which("docker")
    if docker:
        try:
            result = subprocess.run(
                [docker, "compose", "version"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return [docker, "compose"]
        except OSError:
            pass
    standalone = shutil.which("docker-compose")
    return [standalone] if standalone else None


def _running_container_workers(
    compose_file: Path | None,
    stack_root: Path,
    env_file: Path | None,
) -> tuple[list[str] | None, list[str] | None]:
    """Read running Docker worker truth when Compose is available."""
    compose_cmd = _available_compose_command()
    if compose_file is None:
        return [], compose_cmd
    if compose_cmd is None:
        return None, None
    rc, out = _run_compose_capture(
        compose_cmd,
        compose_file,
        stack_root,
        ["ps", "--services", "--filter", "status=running"],
        env_file,
    )
    if rc != 0:
        return None, compose_cmd
    worker_names = {"heartbeat_worker", "maintenance_worker", "channel_worker"}
    running = sorted({part for part in out.split() if part in worker_names})
    return running, compose_cmd


def _configured_compose_services(
    compose_cmd: list[str],
    compose_file: Path,
    stack_root: Path,
    env_file: Path | None,
    profiles: list[str] | None = None,
) -> list[str] | None:
    args: list[str] = []
    for profile in profiles or []:
        args += ["--profile", profile]
    args += ["config", "--services"]
    rc, out = _run_compose_capture(
        compose_cmd,
        compose_file,
        stack_root,
        args,
        env_file,
    )
    if rc != 0:
        return None
    services: list[str] = []
    for line in out.splitlines():
        name = line.strip()
        if (
            name
            and " " not in name
            and all(character.isalnum() or character in "._-" for character in name)
        ):
            services.append(name)
    return services or None


def _host_managed_compose_workers() -> set[str]:
    try:
        from core.host_services import installed_host_services

        mapping = {
            "heartbeat": "heartbeat_worker",
            "maintenance": "maintenance_worker",
            "channels": "channel_worker",
        }
        return {mapping[name] for name in installed_host_services() if name in mapping}
    except Exception:
        return set()


def _host_service_status_if_installed() -> dict[str, Any] | None:
    try:
        from core.host_services import host_service_status, installed_host_services

        if not installed_host_services():
            return None
        return host_service_status()
    except Exception:
        return None


def _ensure_installed_host_services_running() -> tuple[bool, str | None]:
    """Start only inactive installed host services; never restart healthy ones."""
    try:
        from core.host_services import control_host_services, host_service_status

        status = host_service_status()
        inactive = [
            str(item.get("name"))
            for item in status.get("services", [])
            if isinstance(item, dict)
            and item.get("installed")
            and not item.get("active")
        ]
        if inactive:
            control_host_services("start", inactive)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _stop_installed_host_services() -> tuple[bool, str | None]:
    try:
        from core.host_services import control_host_services, installed_host_services

        installed = installed_host_services()
        if installed:
            control_host_services("stop", installed)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _restart_installed_host_services() -> tuple[bool, str | None]:
    try:
        from core.host_services import control_host_services, installed_host_services

        installed = installed_host_services()
        if installed:
            control_host_services("restart", installed)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _print_host_service_status(payload: dict[str, Any]) -> None:
    from apps.cli_theme import console, make_table

    table = make_table(
        "Service",
        "State",
        "Enabled",
        "Unit",
        title=f"Host services ({payload.get('backend') or 'unknown'})",
    )
    for item in payload.get("services", []):
        if not isinstance(item, dict):
            continue
        state = (
            "running"
            if item.get("active")
            else "stopped"
            if item.get("installed")
            else "not installed"
        )
        table.add_row(
            str(item.get("name") or ""),
            state,
            (
                "yes"
                if item.get("enabled") is True
                else "no"
                if item.get("enabled") is False
                else "unknown"
            ),
            str(item.get("unit_path") or ""),
        )
    console.print(table)
    console.print(
        f"[key]Instance:[/key] {payload.get('instance') or 'default/current'}"
    )
    console.print(
        f"[key]Environment:[/key] {payload.get('env_file') or 'process defaults'}"
    )
    if payload.get("backend") == "systemd":
        console.print(f"[key]Linger:[/key] {payload.get('linger') or 'unknown'}")


def _handle_host_service_command(
    args: argparse.Namespace,
    *,
    compose_file: Path | None,
    stack_root: Path,
    env_file: Path | None,
) -> int:
    from apps.cli_theme import console
    from core.host_services import (
        HostServiceError,
        control_host_services,
        host_service_status,
        install_host_services,
        stream_host_service_logs,
        uninstall_host_services,
    )

    func = str(getattr(args, "func", "service_status"))
    try:
        if func == "service_install":
            requested = ["heartbeat", "maintenance"]
            if args.channels:
                requested.append("channels")
            running_workers, compose_cmd = _running_container_workers(
                compose_file,
                stack_root,
                env_file,
            )
            if running_workers is None:
                _print_err(
                    "Hexis could not inspect Docker worker state, so it refused to risk "
                    "installing duplicate workers. Start Docker and retry, or move the "
                    "unavailable Compose file aside if this installation no longer uses it."
                )
                return 1
            matching = [
                name
                for name in running_workers
                if name != "channel_worker" or args.channels
            ]
            if matching and not args.replace_docker_workers:
                _print_err(
                    "Docker workers are already running: "
                    f"{', '.join(matching)}. Running both copies can duplicate work. "
                    "Stop them with `hexis stop`, or explicitly migrate them with "
                    "`hexis service install --replace-docker-workers`."
                )
                return 1
            selected_env = args.env_file or env_file
            active_instance = args.instance or resolve_instance()
            # During an explicit Docker-to-host migration, install and enable
            # the units first. This keeps the original workers running if unit
            # installation fails and avoids an autonomy gap.
            start_during_install = not args.no_start and not matching
            result = install_host_services(
                services=requested,
                env_file=selected_env,
                working_directory=(selected_env.parent if selected_env else stack_root),
                instance=active_instance,
                start=start_during_install,
                enable_linger=bool(args.enable_linger),
            )
            if matching:
                if compose_file is None or compose_cmd is None:
                    _print_err(
                        "Could not resolve Docker Compose to stop existing workers."
                    )
                    return 1
                rc = run_compose(
                    compose_cmd,
                    compose_file,
                    stack_root,
                    ["stop", *matching],
                    env_file,
                )
                if rc != 0:
                    _print_err(
                        "Host units were installed but left stopped because the Docker "
                        "workers could not be stopped. Resolve the Compose error and retry "
                        "the same install command."
                    )
                    return rc
                if not args.no_start:

                    def restore_docker_workers() -> bool:
                        try:
                            control_host_services("stop", requested)
                        except HostServiceError as stop_error:
                            _print_err(
                                "Hexis did not restore Docker workers because it could "
                                f"not prove the attempted host copies were stopped: {stop_error}"
                            )
                            return False
                        return (
                            run_compose(
                                compose_cmd,
                                compose_file,
                                stack_root,
                                ["up", "-d", *matching],
                                env_file,
                            )
                            == 0
                        )

                    try:
                        control_host_services("start", requested)
                    except HostServiceError:
                        restore_docker_workers()
                        raise
            console.print(
                f"[ok]Installed Hexis host services:[/ok] {', '.join(result['installed'])}"
            )
            console.print(
                f"[key]Environment:[/key] {result.get('env_file') or 'process defaults'} "
                "[muted](values were not copied into unit files)[/muted]"
            )
            console.print(
                f"[key]Instance:[/key] {result.get('instance') or 'default/current'}"
            )
            if result.get("warning"):
                console.print(f"[warn]⚠ {result['warning']}[/warn]")
            if (
                result.get("backend") == "systemd"
                and result.get("linger") == "disabled"
            ):
                console.print(
                    "[warn]⚠ User lingering is disabled; services may stop after logout.[/warn] "
                    "Run `hexis service install --enable-linger` if you want them to "
                    "continue without a login session."
                )
            if args.no_start:
                console.print("Start them when ready with `hexis service start`.")
            else:
                try:
                    status = host_service_status()
                except HostServiceError:
                    if matching:
                        restore_docker_workers()
                    raise
                inactive = [
                    item["name"]
                    for item in status.get("services", [])
                    if item.get("name") in requested and not item.get("active")
                ]
                if inactive:
                    restored = restore_docker_workers() if matching else False
                    recovery = (
                        " The previous Docker workers were restored."
                        if restored
                        else ""
                    )
                    _print_err(
                        f"Installed but not running: {', '.join(inactive)}. "
                        "Run `hexis service logs` for the provider error, then "
                        f"`hexis service restart`.{recovery}"
                    )
                    return 1
                console.print(
                    "[ok]Workers are running.[/ok] Check with `hexis service status`."
                )
            return 0

        if func in {"service_start", "service_stop", "service_restart"}:
            action = func.removeprefix("service_")
            result = control_host_services(action, args.services or None)
            console.print(
                f"[ok]{action.capitalize()}ed:[/ok] {', '.join(result['services'])}"
            )
            return 0

        if func == "service_logs":
            return stream_host_service_logs(
                args.services or None,
                lines=args.lines,
                follow=bool(args.follow),
            )

        if func == "service_uninstall":
            if not args.yes:
                console.print(
                    "This stops and removes Hexis-managed user-service units. "
                    "Brain data, configuration, and logs are preserved."
                )
                try:
                    answer = input("Type 'uninstall' to confirm: ")
                except (KeyboardInterrupt, EOFError):
                    print()
                    return 1
                if answer.strip().lower() != "uninstall":
                    console.print("[muted]Aborted.[/muted]")
                    return 1
            result = uninstall_host_services(args.services or None)
            console.print(
                f"[ok]Removed host services:[/ok] {', '.join(result['uninstalled'])}"
            )
            console.print(
                f"Logs were preserved at {result['preserved_log_directory']}."
            )
            return 0

        payload = host_service_status()
        if getattr(args, "json", False):
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            _print_host_service_status(payload)
        return 0
    except HostServiceError as exc:
        _print_err(str(exc))
        return 1


def _resolved_env_setting(
    name: str,
    *,
    env_file: Path | None,
    default: str,
) -> str:
    """Resolve one stack setting without importing unrelated environment values."""

    ambient = os.getenv(name)
    if ambient is not None and ambient.strip():
        return ambient.strip()
    if env_file and env_file.is_file():
        try:
            selected = dotenv_values(env_file).get(name)
        except (OSError, ValueError):
            selected = None
        if selected is not None and str(selected).strip():
            return str(selected).strip()
    return default


def _wait_http_ready(url: str, *, overall: float = 60.0) -> bool:
    import time

    deadline = time.monotonic() + overall
    while time.monotonic() < deadline:
        if _http_ready(url):
            return True
        time.sleep(0.4)
    return False


def _print_tunnel_status(payload: dict[str, Any]) -> None:
    from apps.cli_theme import console, make_table

    rows = [
        ["State", str(payload.get("status") or "unknown")],
        ["Private URL", str(payload.get("url") or "not available")],
        ["Local target", f"http://127.0.0.1:{payload.get('ui_port') or 3477}"],
        ["Local dashboard", "ready" if payload.get("local_ready") else "not ready"],
        ["Tailnet", "connected" if payload.get("connected") else "not connected"],
        ["Route owner", "Hexis" if payload.get("owned") else "external/none"],
        ["Public bind", "YES — out of bounds" if payload.get("public_bind") else "no"],
        ["Tailscale Funnel", "YES — public" if payload.get("funnel_enabled") else "no"],
    ]
    table = make_table("Check", "Result", title="Private dashboard tunnel")
    for row in rows:
        table.add_row(*row)
    console.print(table)
    for issue in payload.get("issues") or []:
        console.print(f"[fail]✗ {issue}[/fail]")
    warning = payload.get("warning")
    if warning:
        console.print(f"[warn]⚠ {warning}[/warn]")
    if payload.get("status") == "active":
        console.print(
            "[ok]Tailnet-only HTTPS is active.[/ok] The dashboard remains bound to loopback."
        )
    elif not payload.get("issues"):
        console.print(str(payload.get("detail") or "No private route is active."))


def _handle_tunnel_command(
    args: argparse.Namespace,
    *,
    env_file: Path | None,
) -> int:
    from apps.cli_theme import console
    from core.tunnel import TunnelError, start_tunnel, stop_tunnel, tunnel_status

    try:
        port_text = _resolved_env_setting(
            "HEXIS_UI_PORT", env_file=env_file, default="3477"
        )
        port = int(args.port) if args.port is not None else int(port_text)
        bind_address = _resolved_env_setting(
            "HEXIS_BIND_ADDRESS", env_file=env_file, default="127.0.0.1"
        )
    except ValueError:
        _print_err(
            "HEXIS_UI_PORT must be an integer between 1 and 65535. Fix the selected "
            "environment file, then retry."
        )
        return 1

    func = str(getattr(args, "func", "tunnel_status"))
    try:
        if func == "tunnel_start":
            local_url = f"http://127.0.0.1:{port}/api/status"
            if not _http_ready(local_url):
                if args.no_start_stack:
                    raise TunnelError(
                        f"The dashboard is not responding at {local_url}. Start it with "
                        "`hexis up`, then retry without changing Tailscale state."
                    )
                console.print(
                    "[muted]The local dashboard is down; starting the Hexis stack first...[/muted]"
                )
                up_rc = main(["up"])
                if up_rc != 0:
                    raise TunnelError(
                        "The Hexis stack did not start, so no Tailscale route was changed. "
                        "Resolve the startup error above, then retry `hexis tunnel start`."
                    )
                if not _wait_http_ready(local_url):
                    raise TunnelError(
                        f"The stack started but the dashboard never responded at {local_url}. "
                        "No Tailscale route was changed; run `hexis logs ui api` for the cause."
                    )
            payload = start_tunnel(
                ui_port=port,
                bind_address=bind_address,
                probe_local=True,
            )
        elif func == "tunnel_stop":
            payload = stop_tunnel(
                ui_port=port if args.port is not None else None,
                bind_address=bind_address,
            )
        else:
            payload = tunnel_status(
                ui_port=port,
                bind_address=bind_address,
            )

        if getattr(args, "json", False):
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            _print_tunnel_status(payload)
            if func == "tunnel_start" and payload.get("status") == "active":
                console.print(
                    "Open the private URL on a device already approved in this tailnet, "
                    "then run `hexis doctor` to verify trusted HTTPS end to end."
                )
            elif func == "tunnel_stop":
                console.print(
                    "[ok]The Hexis tailnet route is off.[/ok] The local dashboard and brain data were preserved."
                )
        return 1 if payload.get("status") in {"risky", "conflict", "unavailable"} else 0
    except TunnelError as exc:
        _print_err(str(exc))
        return 1


async def _configured_voice_model() -> str:
    from core.agent_api import _connect_with_retry

    conn = await _connect_with_retry(db_dsn_from_env(), wait_seconds=5)
    try:
        model = await conn.fetchval("SELECT get_config_text('voice.tts.model')")
    finally:
        await conn.close()
    value = str(model or "").strip()
    if not value:
        raise RuntimeError(
            "No live voice model is configured. Open Settings → Voice, save the "
            "local provider, then retry."
        )
    return value


async def _configured_voice_output() -> tuple[bool, str]:
    """Read voice enablement and model from the live brain configuration."""
    from core.agent_api import _connect_with_retry

    conn = await _connect_with_retry(db_dsn_from_env(), wait_seconds=5)
    try:
        row = await conn.fetchrow(
            """
            SELECT get_config_bool('voice.tts.enabled') AS enabled,
                   get_config_text('voice.tts.model') AS model
            """
        )
    finally:
        await conn.close()
    model = str((row or {}).get("model") or "").strip()
    if not model:
        raise RuntimeError(
            "No live voice model is configured. Open Settings → Voice, save the "
            "local provider, then retry."
        )
    return bool((row or {}).get("enabled")), model


def _voice_support_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("piper") is not None


def _voice_dependency_requirement() -> str:
    from importlib import metadata
    from core.voice_sidecar import PIPER_REQUIREMENT

    try:
        requirements = metadata.requires("hexis") or []
    except metadata.PackageNotFoundError:
        requirements = []
    for requirement in requirements:
        if requirement.lower().startswith("piper-tts"):
            return requirement.partition(";")[0].strip()
    return PIPER_REQUIREMENT


def _install_voice_support(*, yes: bool) -> bool:
    from apps.cli_theme import console

    if _voice_support_installed():
        return True
    if not yes:
        console.print(
            "Local speech needs the optional Piper engine and its HTTP support. "
            "This installs it into the current Hexis Python environment; voice "
            "models download only when you start a chosen voice."
        )
        try:
            answer = input("Install local voice support now? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if answer.strip().lower() not in {"y", "yes"}:
            console.print("[muted]No changes made.[/muted]")
            return False
    uv = shutil.which("uv")
    if not uv:
        _print_err(
            "uv is required to add optional voice support safely to this Hexis "
            "environment. Install uv, then rerun `hexis voice setup`; no package "
            "changes were made."
        )
        return False
    requirement = _voice_dependency_requirement()
    console.print(f"[muted]Installing {requirement} into {sys.executable}...[/muted]")
    try:
        result = subprocess.run(
            [uv, "pip", "install", "--python", sys.executable, requirement],
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _print_err(
            f"Local voice support could not be installed ({exc}). Nothing was "
            "started; rerun `hexis voice setup` after resolving the uv error."
        )
        return False
    if result.returncode != 0:
        _print_err(
            f"uv exited with code {result.returncode}; no voice process was started. "
            "Review the installer output above, then rerun `hexis voice setup`."
        )
        return False
    import importlib

    importlib.invalidate_caches()
    if not _voice_support_installed():
        _print_err(
            "The installer completed but Piper is still unavailable in this Hexis "
            "environment. No voice process was started."
        )
        return False
    return True


def _print_voice_status(payload: dict[str, Any]) -> None:
    from apps.cli_theme import console, make_table

    table = make_table("Check", "Result", title="Local speech output")
    table.add_row("State", str(payload.get("status") or "unknown"))
    table.add_row("Provider", "ready" if payload.get("ready") else "not ready")
    table.add_row("Model", str(payload.get("model") or "from live configuration"))
    table.add_row("Process owner", "Hexis" if payload.get("owned") else "external/none")
    table.add_row("Endpoint", str(payload.get("url") or "http://127.0.0.1:42667"))
    table.add_row("Log", str(payload.get("log_path") or ""))
    console.print(table)
    warning = payload.get("warning")
    if warning:
        console.print(f"[warn]⚠ {warning}[/warn]")
    elif payload.get("ready"):
        console.print("[ok]Local speech is ready.[/ok]")
    else:
        console.print(str(payload.get("detail") or "Local speech is not running."))


def _handle_voice_command(args: argparse.Namespace) -> int:
    from core.voice_sidecar import (
        VoiceSidecarError,
        start_voice_sidecar,
        stop_voice_sidecar,
        voice_sidecar_status,
    )

    func = str(getattr(args, "func", "voice_status"))
    try:
        if func == "voice_setup" and not _install_voice_support(yes=bool(args.yes)):
            return 1
        if func in {"voice_start", "voice_setup"}:
            if not _voice_support_installed():
                raise VoiceSidecarError(
                    "Local Piper support is not installed. Run `hexis voice setup` to "
                    "install it and start the configured voice in one flow."
                )
            try:
                model = asyncio.run(_configured_voice_model())
            except Exception as exc:
                raise VoiceSidecarError(
                    f"Could not read the live voice model from the Hexis brain ({exc}). "
                    "Run `hexis up`, save Settings → Voice, then retry."
                ) from exc
            payload = start_voice_sidecar(
                model=model,
                wait_seconds=max(1.0, float(args.wait_seconds)),
            )
        elif func == "voice_stop":
            payload = stop_voice_sidecar()
        else:
            payload = voice_sidecar_status()
        if getattr(args, "json", False):
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            _print_voice_status(payload)
        return 0 if payload.get("status") != "stale" else 1
    except VoiceSidecarError as exc:
        _print_err(str(exc))
        return 1


def _start_configured_voice_sidecar() -> tuple[bool, str | None]:
    """Start configured speech output after the stack is ready.

    This is advisory for ``up``/``dev``: database and chat startup remain useful
    even when optional local speech cannot start.
    """
    from core.voice_sidecar import start_voice_sidecar, voice_sidecar_status

    try:
        enabled, model = asyncio.run(_configured_voice_output())
        if not enabled:
            return False, None
        current = voice_sidecar_status()
        if current.get("ready"):
            if current.get("owned"):
                return False, None
            return (
                False,
                "A compatible ambient voice provider is already ready; Hexis did "
                "not adopt it and will not stop it.",
            )
        if not _voice_support_installed():
            return (
                False,
                "Speech output is enabled, but local Piper support is not installed. "
                "Run `hexis voice setup` to finish the setup in place.",
            )
        result = start_voice_sidecar(model=model, wait_seconds=2)
        return bool(result.get("changed")), result.get("warning")
    except Exception as exc:
        return (
            False,
            f"Speech output is enabled but its local provider did not become ready: "
            f"{exc} Run `hexis voice status` for the current state and recovery step.",
        )


def _stop_owned_voice_sidecar() -> tuple[bool, bool, str | None]:
    """Stop only the exact voice process previously launched by Hexis."""
    from core.voice_sidecar import stop_voice_sidecar, voice_sidecar_status

    try:
        current = voice_sidecar_status()
        if not current.get("owned"):
            if current.get("ready"):
                return (
                    True,
                    False,
                    "A compatible ambient voice provider was left running because "
                    "Hexis does not own it.",
                )
            if current.get("state_present"):
                return (
                    True,
                    False,
                    "Stale voice ownership state was preserved for review; no process "
                    "was signaled.",
                )
            return True, False, None
        result = stop_voice_sidecar()
        return True, bool(result.get("changed")), None
    except Exception as exc:
        return False, False, f"Could not stop the Hexis-owned voice process: {exc}"


def _get_dsn(args) -> str:
    """Get DSN respecting --instance flag, --dsn flag, or defaults."""
    if hasattr(args, "dsn") and args.dsn:
        return args.dsn
    if args.instance:
        return db_dsn_from_env(args.instance)
    return db_dsn_from_env()


def _coerce_db_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw)
    return raw or {}


def _short_text(text: Any, limit: int = 180) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(limit - 1, 0)] + "..."


def _validate_session_uuid(session_id: str, *, label: str = "session_id") -> str:
    import uuid

    try:
        return str(uuid.UUID(session_id))
    except ValueError as exc:
        raise ValueError(
            f"Invalid {label}: {session_id}. Run `hexis chat-sessions list` to copy a session UUID."
        ) from exc


def _chat_session_export_jsonl(artifact: dict[str, Any]) -> str:
    session = artifact.get("session") or {}
    header = {
        "type": "session",
        "format": artifact.get("format", "hexis.chat_session.v1"),
        "exported_at": artifact.get("exported_at"),
        "message_count": artifact.get("message_count", 0),
        "visible_message_count": artifact.get("visible_message_count", 0),
        "include_hidden": artifact.get("include_hidden", True),
        "session": session,
    }
    lines = [json.dumps(header, ensure_ascii=False, default=str)]
    for message in artifact.get("messages") or []:
        lines.append(
            json.dumps({"type": "message", **message}, ensure_ascii=False, default=str)
        )
    return "\n".join(lines) + "\n"


async def _chat_sessions_list(
    dsn: str,
    limit: int,
    surface: str | None,
    status: str | None,
    as_json: bool,
) -> int:
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT list_chat_sessions($1::int, $2::text, $3::text)",
                limit,
                surface,
                status,
            )
        data = _coerce_db_json(raw)
        if as_json:
            sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
            return 0

        sessions = data.get("sessions") or []
        if not sessions:
            sys.stdout.write("No chat sessions found.\n")
            return 0

        sys.stdout.write(
            f"Chat sessions ({data.get('count', len(sessions))}/{data.get('total_matching', len(sessions))} shown)\n"
        )
        for session in sessions:
            title = (
                session.get("title")
                or session.get("first_user_snippet")
                or "(untitled)"
            )
            last = session.get("last_message_snippet") or ""
            sys.stdout.write(
                f"  {session.get('session_id')}  "
                f"{session.get('surface', 'chat')}  {session.get('status', 'active')}  "
                f"{session.get('message_count', 0)} msg  last={session.get('last_active_at')}\n"
                f"    {title}\n"
            )
            if last and last != title:
                sys.stdout.write(
                    f"    last {session.get('last_message_role')}: {_short_text(last)}\n"
                )
        sys.stdout.write(
            "\nUse `hexis chat-sessions show <session_id>` or `hexis chat-sessions export <session_id>`.\n"
        )
        return 0
    except Exception as exc:
        message = str(exc)
        if "list_chat_sessions" in message or "does not exist" in message:
            _print_err(
                "Chat-session artifact functions are missing. Run `hexis migrate` to update the schema."
            )
        else:
            _print_err(f"Error: {exc}")
        return 1
    finally:
        await pool.close()


async def _chat_sessions_show(
    dsn: str,
    session_id: str,
    visible_only: bool,
    as_json: bool,
) -> int:
    import asyncpg

    try:
        normalized_session_id = _validate_session_uuid(session_id)
    except ValueError as exc:
        _print_err(str(exc))
        return 1

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT get_chat_session_artifact($1::uuid, TRUE, $2::bool)",
                normalized_session_id,
                not visible_only,
            )
        artifact = _coerce_db_json(raw)
        if not artifact.get("found"):
            _print_err(
                f"Chat session not found: {normalized_session_id}. "
                "Run `hexis chat-sessions list --status all` to browse available sessions."
            )
            return 1
        if as_json:
            sys.stdout.write(json.dumps(artifact, indent=2, default=str) + "\n")
            return 0

        session = artifact.get("session") or {}
        sys.stdout.write(
            f"Session {session.get('session_id')}\n"
            f"  title: {session.get('title') or '(untitled)'}\n"
            f"  surface: {session.get('surface')}  status: {session.get('status')}\n"
            f"  messages: {artifact.get('message_count', 0)} "
            f"({artifact.get('visible_message_count', 0)} visible)\n"
            f"  created: {session.get('created_at')}\n"
            f"  last active: {session.get('last_active_at')}\n\n"
        )
        for message in artifact.get("messages") or []:
            hidden = "" if message.get("visible_in_context") else " [hidden]"
            sys.stdout.write(
                f"[{message.get('ordinal')}] {message.get('role')}{hidden}\n"
            )
            sys.stdout.write(str(message.get("content") or "") + "\n\n")
        return 0
    except Exception as exc:
        message = str(exc)
        if "get_chat_session_artifact" in message or "does not exist" in message:
            _print_err(
                "Chat-session artifact functions are missing. Run `hexis migrate` to update the schema."
            )
        else:
            _print_err(f"Error: {exc}")
        return 1
    finally:
        await pool.close()


async def _chat_sessions_export(
    dsn: str,
    session_id: str,
    output_format: str,
    output: str | None,
    visible_only: bool,
) -> int:
    import asyncpg

    try:
        normalized_session_id = _validate_session_uuid(session_id)
    except ValueError as exc:
        _print_err(str(exc))
        return 1

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT get_chat_session_artifact($1::uuid, TRUE, $2::bool)",
                normalized_session_id,
                not visible_only,
            )
        artifact = _coerce_db_json(raw)
        if not artifact.get("found"):
            _print_err(
                f"Chat session not found: {normalized_session_id}. "
                "Run `hexis chat-sessions list --status all` to browse available sessions."
            )
            return 1

        if output_format == "jsonl":
            content = _chat_session_export_jsonl(artifact)
        else:
            content = (
                json.dumps(artifact, indent=2, ensure_ascii=False, default=str) + "\n"
            )

        if output:
            out_path = Path(output).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            sys.stdout.write(f"Exported chat session to {out_path}\n")
        else:
            sys.stdout.write(content)
        return 0
    except Exception as exc:
        message = str(exc)
        if "get_chat_session_artifact" in message or "does not exist" in message:
            _print_err(
                "Chat-session artifact functions are missing. Run `hexis migrate` to update the schema."
            )
        else:
            _print_err(f"Error: {exc}")
        return 1
    finally:
        await pool.close()


async def _chat_sessions_title(
    dsn: str, session_id: str, title: str, as_json: bool
) -> int:
    import asyncpg

    try:
        normalized_session_id = _validate_session_uuid(session_id)
    except ValueError as exc:
        _print_err(str(exc))
        return 1

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT set_chat_session_title($1::uuid, $2::text)",
                normalized_session_id,
                title,
            )
        artifact = _coerce_db_json(raw)
        if not artifact.get("found"):
            _print_err(
                f"Chat session not found: {normalized_session_id}. "
                "Run `hexis chat-sessions list --status all` to browse available sessions."
            )
            return 1
        if as_json:
            sys.stdout.write(json.dumps(artifact, indent=2, default=str) + "\n")
        else:
            session = artifact.get("session") or {}
            sys.stdout.write(
                f"Updated chat session {normalized_session_id}: title={session.get('title') or '(cleared)'}\n"
            )
        return 0
    except Exception as exc:
        message = str(exc)
        if "set_chat_session_title" in message or "does not exist" in message:
            _print_err(
                "Chat-session title function is missing. Run `hexis migrate` to update the schema."
            )
        else:
            _print_err(f"Error: {exc}")
        return 1
    finally:
        await pool.close()


async def _chat_sessions_fork(
    dsn: str,
    session_id: str,
    until_ordinal: int | None,
    title: str | None,
    as_json: bool,
) -> int:
    import asyncpg

    try:
        normalized_session_id = _validate_session_uuid(
            session_id, label="source session_id"
        )
    except ValueError as exc:
        _print_err(str(exc))
        return 1

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT fork_chat_session($1::uuid, $2::int, $3::text, '{}'::jsonb)",
                normalized_session_id,
                until_ordinal,
                title,
            )
        artifact = _coerce_db_json(raw)
        if not artifact.get("found"):
            _print_err(
                f"Chat session not found: {normalized_session_id}. "
                "Run `hexis chat-sessions list --status all` to browse available sessions."
            )
            return 1
        if as_json:
            sys.stdout.write(json.dumps(artifact, indent=2, default=str) + "\n")
        else:
            session = artifact.get("session") or {}
            sys.stdout.write(
                f"Created chat session {session.get('session_id')} from {normalized_session_id} "
                f"with {artifact.get('forked_message_count', 0)} message(s).\n"
            )
        return 0
    except Exception as exc:
        message = str(exc)
        if "fork_chat_session" in message or "does not exist" in message:
            _print_err(
                "Chat-session fork function is missing. Run `hexis migrate` to update the schema."
            )
        else:
            _print_err(f"Error: {exc}")
        return 1
    finally:
        await pool.close()


async def _channels_status(dsn: str, as_json: bool) -> int:
    """Show channel session counts."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT channel_type,
                       COUNT(*) AS sessions,
                       COUNT(*) FILTER (WHERE last_active > CURRENT_TIMESTAMP - INTERVAL '1 hour') AS active_1h,
                       MAX(last_active) AS last_active
                FROM channel_sessions
                GROUP BY channel_type
                ORDER BY channel_type
            """)
            total_messages = (
                await conn.fetchval("SELECT COUNT(*) FROM channel_messages") or 0
            )

        data = {
            "channels": [
                {
                    "type": row["channel_type"],
                    "sessions": row["sessions"],
                    "active_1h": row["active_1h"],
                    "last_active": (
                        str(row["last_active"]) if row["last_active"] else None
                    ),
                }
                for row in rows
            ],
            "total_messages": total_messages,
        }
        if as_json:
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
        else:
            if not rows:
                sys.stdout.write("No channel sessions found.\n")
            else:
                sys.stdout.write("Channel Sessions:\n")
                for row in rows:
                    sys.stdout.write(
                        f"  {row['channel_type']}: {row['sessions']} sessions "
                        f"({row['active_1h']} active in last hour)\n"
                    )
            sys.stdout.write(f"Total messages: {total_messages}\n")
        return 0
    except Exception as e:
        if "channel_sessions" in str(e):
            _print_err(
                "Channel tables not found — your schema is out of date. "
                "Run `hexis migrate` (or `hexis upgrade`) to bring it up to date without losing data."
            )
        else:
            _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


async def _retention_status(dsn: str, as_json: bool) -> int:
    """Show what the memory-retention system holds and would do."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval("SELECT retention_status()")
        data = json.loads(raw) if isinstance(raw, str) else raw
        if as_json:
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
            return 0

        epi = data.get("episodic", {})
        con = data.get("consolidation", {})
        rev = data.get("conscious_review", {})
        doc = data.get("documents", {})
        budget = rev.get("veto_budget") or {}
        cap = epi.get("capacity") or 0
        cap_str = f"{cap}" if cap and float(cap) > 0 else "unlimited"

        hard_pruning = bool(data.get("irreversible_pruning_enabled"))
        state = (
            "ENABLED"
            if data.get("enabled")
            else "DISABLED (automatic rest-cycle consolidation paused)"
        )
        out = [
            f"Memory Retention  [{state}]",
            "",
            "Episodic memory",
            f"  active memories        {epi.get('active', 0)}",
            f"  representational mass  {epi.get('mass', 0)}  (capacity: {cap_str})",
            (
                f"  archived (hard-prune eligible)  {epi.get('archived', 0)}"
                if hard_pruning
                else f"  archived originals (recoverable)  {epi.get('archived', 0)}"
            ),
            f"  irreversible hard pruning  {'ENABLED' if hard_pruning else 'OFF (default)'}",
            "",
            "Consolidation",
            f"  candidate groups (would consolidate)  {con.get('candidate_groups', 0)}",
            f"  gists formed                          {con.get('gists', 0)}",
            f"  summarization pending                 {con.get('summarize_pending', 0)}",
            "",
            "Load-bearing review (your decision)",
            f"  pending      {rev.get('pending', 0)}",
            f"  keep-budget  {budget.get('remaining', '-')}/{budget.get('total', '-')}"
            + (
                f'  (chapter: "{budget.get("chapter")}")'
                if budget.get("chapter")
                else ""
            ),
            "",
            "Documents (your data — removed only with your approval)",
            f"  ingested documents protected  {doc.get('protected', 0)}",
            f"  approvals awaiting you         {doc.get('approvals_pending', 0)}"
            + (
                f"  — {', '.join(doc.get('approval_labels') or [])}"
                if doc.get("approvals_pending")
                else ""
            ),
            "",
            "Review pressure, fade choices, and compression receipts: /forgetting",
        ]
        sys.stdout.write("\n".join(out) + "\n")
        return 0
    except Exception as e:
        if "does not exist" in str(e) or "retention_status" in str(e):
            _print_err(
                "Retention functions not found — your schema is out of date. "
                "Run `hexis migrate` (or `hexis upgrade`) to bring it up to date without losing data."
            )
        else:
            _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


def _print_dry_run(d: dict[str, Any]) -> None:
    rest = d.get("rest", {}) or {}
    gc = d.get("gc", {}) or {}
    docs = d.get("documents", {}) or {}
    before = (d.get("before", {}) or {}).get("episodic", {}) or {}
    after = (d.get("after", {}) or {}).get("episodic", {}) or {}
    labels = ((d.get("after", {}) or {}).get("documents", {}) or {}).get(
        "approval_labels"
    ) or []
    req = docs.get("requested", 0)
    hard_pruning = bool(gc.get("irreversible_pruning_enabled"))
    pruning_line = (
        f"  Would hard-prune     {gc.get('pruned', 0)} archived original(s) past the grace window"
        if hard_pruning
        else "  Hard pruning        OFF — archived originals remain recoverable"
    )
    lines = [
        "Retention dry-run — one rest cycle, simulated. NOTHING was changed.",
        "",
        f"  Would consolidate    {rest.get('consolidated', 0)} group(s) of aged memories into gists",
        f"  Would escalate       {rest.get('escalated', 0)} to your conscious review (Hexis's veto)",
        pruning_line,
        f"  Would ask you about  {req} stale document(s)"
        + (f": {', '.join(labels)}" if req and labels else ""),
        "",
        f"  Episodic memory: {before.get('active', 0)} → {after.get('active', 0)} active"
        f"  (mass {before.get('mass', 0)} → {after.get('mass', 0)})",
        "",
        "None of the above has happened. To turn it on for real:  hexis retention enable",
    ]
    sys.stdout.write("\n".join(lines) + "\n")


def _print_alive_demo(result: dict[str, Any]) -> None:
    heading = (
        "Hexis is alive"
        if result.get("ok")
        else "Hexis capability proof needs attention"
    )
    sys.stdout.write(
        f"{heading} ({result.get('passed', 0)}/{result.get('total', 0)} proofs passed)\n"
        "Mode: rollback-only; no LLM call, token cost, or retained demo state.\n\n"
    )
    for proof in result.get("proofs") or []:
        sys.stdout.write(
            f"[{proof.get('status', 'FAIL')}] {proof.get('label', proof.get('id', 'proof'))}\n"
            f"  {proof.get('detail', '')}\n"
        )
        evidence = proof.get("evidence") or {}
        if evidence:
            rendered = ", ".join(f"{key}={value}" for key, value in evidence.items())
            sys.stdout.write(f"  Evidence: {rendered}\n")
        if proof.get("next_step"):
            sys.stdout.write(f"  Next: {proof['next_step']}\n")


def _print_maturity_scorecard(result: dict[str, Any]) -> None:
    sys.stdout.write(
        f"Capability maturity: {result.get('score', 0)}% "
        f"({result.get('points', 0)}/{result.get('max_points', 0)} points)\n\n"
    )
    for scenario in result.get("scenarios") or []:
        sys.stdout.write(
            f"L{scenario['level']}/{scenario['max_level']} "
            f"{scenario['label']} [{scenario['level_name']}]\n"
        )
        for evidence in scenario.get("evidence") or []:
            sys.stdout.write(f"  - {evidence}\n")
        if scenario.get("next_step"):
            sys.stdout.write(f"  Next: {scenario['next_step']}\n")


async def _retention_dry_run(dsn: str, as_json: bool) -> int:
    """Simulate one rest cycle and show the diff, without changing anything."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval("SELECT retention_dry_run()")
        data = json.loads(raw) if isinstance(raw, str) else raw
        if as_json:
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
        else:
            _print_dry_run(data)
        return 0
    except Exception as e:
        _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


async def _retention_enable(dsn: str, skip_confirm: bool) -> int:
    """Show a dry-run, then (with confirmation) turn retention on for real."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            if await conn.fetchval("SELECT get_config_bool('retention.enabled')"):
                sys.stdout.write("Memory retention is already enabled.\n")
                return 0
            raw = await conn.fetchval("SELECT retention_dry_run()")
            thresholds = await conn.fetch(
                "SELECT key, value FROM config WHERE key = ANY($1::text[]) ORDER BY key",
                [
                    "retention.min_age_days",
                    "retention.consolidate_max_strength",
                    "retention.prune_grace_days",
                    "retention.veto_budget_per_chapter",
                ],
            )
        data = json.loads(raw) if isinstance(raw, str) else raw
        _print_dry_run(data)
        sys.stdout.write(
            "\nStarting thresholds (conservative defaults — tune via `hexis config`):\n"
        )
        for r in thresholds:
            val = json.loads(r["value"]) if isinstance(r["value"], str) else r["value"]
            sys.stdout.write(f"  {r['key']} = {val}\n")
        if not skip_confirm:
            sys.stdout.write(
                "\nEnable memory retention now? Archived originals remain recoverable because "
                "irreversible pruning is separately off, load-bearing reviews wait for your choice, "
                "and ingested documents are never removed without your approval. [y/N] "
            )
            sys.stdout.flush()
            if input().strip().lower() not in ("y", "yes"):
                sys.stdout.write("Left disabled.\n")
                return 0
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE config SET value='true'::jsonb WHERE key='retention.enabled'"
            )
        sys.stdout.write(
            "Memory retention is now ENABLED. Run `hexis retention` any time to see what it's doing.\n"
        )
        return 0
    except Exception as e:
        _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


async def _retention_disable(dsn: str) -> int:
    """Pause automatic rest-cycle consolidation."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE config SET value='false'::jsonb WHERE key='retention.enabled'"
            )
        sys.stdout.write(
            "Memory retention is now DISABLED; automatic rest-cycle consolidation is paused.\n"
        )
        return 0
    except Exception as e:
        _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


async def _skills_status(dsn: str, as_json: bool) -> int:
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            enabled = bool(
                await conn.fetchval(
                    "SELECT get_config_bool('skills.self_improvement.enabled')"
                )
            )
            interval = await conn.fetchval(
                "SELECT get_config_int('skills.self_improvement.interval_seconds')"
            )
            state = await conn.fetchval("SELECT get_state('skill_improvement_state')")
            summary = await conn.fetchval("SELECT skill_improvement_pending_summary()")
            pending_learning_reviews = int(
                await conn.fetchval(
                    "SELECT count(*) FROM learning_reviews WHERE status='pending'"
                )
                or 0
            )
        state = json.loads(state) if isinstance(state, str) else (state or {})
        summary = json.loads(summary) if isinstance(summary, str) else (summary or {})
        data = {
            "enabled": enabled,
            "interval_seconds": interval,
            "pending": summary.get("count", 0),
            "pending_learning_reviews": pending_learning_reviews,
            "last_completed_at": state.get("last_completed_at"),
            "last_result": state.get("last_result"),
        }
        if as_json:
            sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        else:
            sys.stdout.write(
                "Skill improvement: " + ("ENABLED" if enabled else "DISABLED") + "\n"
                f"  Review interval: {interval or 'not configured'} seconds\n"
                f"  Pending proposals: {data['pending']}\n"
                f"  Pending learning reviews: {data['pending_learning_reviews']}\n"
                f"  Last review: {data['last_completed_at'] or 'never'}\n"
            )
            if data["pending"]:
                sys.stdout.write("  Next: hexis skills proposals\n")
            elif not enabled:
                sys.stdout.write("  To opt in: hexis skills enable\n")
        return 0
    except Exception as exc:
        _print_err(
            f"Could not load skill-improvement status: {exc}. "
            "Run `hexis migrate` to ensure the proposal schema is current."
        )
        return 1
    finally:
        await pool.close()


async def _skills_enable(dsn: str, skip_confirm: bool) -> int:
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            if await conn.fetchval(
                "SELECT get_config_bool('skills.self_improvement.enabled')"
            ):
                sys.stdout.write(
                    "Background skill-improvement review is already enabled.\n"
                )
                return 0
            settings = await conn.fetch(
                "SELECT key, value FROM config WHERE key LIKE 'skills.self_improvement.%' ORDER BY key"
            )
        sys.stdout.write(
            "Background learning review examines bounded recent conversation excerpts using your "
            "configured LLM. It creates one grounded weekly diff and reviewable skill proposals; "
            "it never applies a skill automatically.\n\n"
            "Current settings:\n"
        )
        for row in settings:
            value = (
                json.loads(row["value"])
                if isinstance(row["value"], str)
                else row["value"]
            )
            sys.stdout.write(f"  {row['key']} = {value}\n")
        if not skip_confirm:
            sys.stdout.write("\nEnable background skill proposal review? [y/N] ")
            sys.stdout.flush()
            if input().strip().lower() not in ("y", "yes"):
                sys.stdout.write("Left disabled.\n")
                return 0
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('skills.self_improvement.enabled', 'true'::jsonb)"
            )
        sys.stdout.write(
            "Weekly learning and skill review is ENABLED. "
            "Use the Learning review dashboard for the diff, `hexis skills` for status, "
            "and `hexis skills proposals` for proposal history.\n"
        )
        return 0
    except Exception as exc:
        _print_err(f"Could not enable skill-improvement review: {exc}")
        return 1
    finally:
        await pool.close()


async def _skills_disable(dsn: str) -> int:
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('skills.self_improvement.enabled', 'false'::jsonb)"
            )
        sys.stdout.write(
            "Weekly learning and skill review is DISABLED. Existing reviews and proposals were kept.\n"
        )
        return 0
    except Exception as exc:
        _print_err(f"Could not disable skill-improvement review: {exc}")
        return 1
    finally:
        await pool.close()


async def _skills_proposals(dsn: str, status: str, as_json: bool) -> int:
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, status, name, description, mode, rationale, confidence,
                       COALESCE(evidence->>'origin', 'background_review') AS origin,
                       cardinality(source_memory_ids) AS source_memories,
                       cardinality(source_unit_ids) AS source_units,
                       created_at, reviewed_at, applied_at, last_error
                FROM skill_improvement_proposals
                WHERE ($1::text = 'all' OR status = $1::text)
                ORDER BY created_at DESC, id
                """,
                status,
            )
        proposals = [dict(row) for row in rows]
        if as_json:
            sys.stdout.write(json.dumps(proposals, indent=2, default=str) + "\n")
        elif not proposals:
            sys.stdout.write(f"No {status} skill proposals.\n")
        else:
            for proposal in proposals:
                sys.stdout.write(
                    f"{proposal['id']}  [{proposal['status']}] {proposal['mode']} {proposal['name']} "
                    f"({proposal['origin']}, confidence {proposal['confidence']:.2f}, "
                    f"{proposal['source_units']} source turns)\n"
                    f"  {proposal['description']}\n"
                )
                if proposal["last_error"]:
                    sys.stdout.write(f"  Last error: {proposal['last_error']}\n")
            sys.stdout.write(
                "\nReview one in place: hexis skills review <proposal-id> --action apply|reject\n"
            )
        return 0
    except Exception as exc:
        _print_err(f"Could not list skill proposals: {exc}")
        return 1
    finally:
        await pool.close()


async def _skills_review(
    dsn: str, proposal_id: str, action: str, skip_confirm: bool
) -> int:
    import asyncpg

    from core.tools import ToolContext, ToolExecutionContext, create_default_registry
    from core.tools.skills import ReviewSkillProposalHandler

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, status, name, description, content, mode, rationale,
                       COALESCE(evidence->>'origin', 'background_review') AS origin,
                       confidence, cardinality(source_memory_ids) AS source_memories,
                       cardinality(source_unit_ids) AS source_units
                FROM skill_improvement_proposals WHERE id = $1::uuid
                """,
                proposal_id,
            )
        if not row:
            _print_err(f"Skill proposal not found: {proposal_id}")
            return 1
        sys.stdout.write(
            f"Proposal {row['id']} [{row['status']}]\n"
            f"  {row['mode']} skill: {row['name']}\n"
            f"  Origin: {row['origin']}\n"
            f"  Confidence: {row['confidence']:.2f}\n"
            f"  Evidence: {row['source_units']} source turns, {row['source_memories']} memories\n"
            f"  Description: {row['description']}\n"
            f"  Rationale: {row['rationale']}\n\n"
            f"{row['content']}\n"
        )
        if not skip_confirm:
            consequence = (
                "create or update the agent-authored skill file"
                if action == "apply"
                else f"mark this proposal {action}ed without deleting it"
            )
            sys.stdout.write(
                f"\n{action.title()} this proposal and {consequence}? [y/N] "
            )
            sys.stdout.flush()
            if input().strip().lower() not in ("y", "yes"):
                sys.stdout.write("No change made.\n")
                return 0
        registry = create_default_registry(pool)
        context = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id=f"cli-skill-review-{proposal_id}",
            registry=registry,
        )
        handler = ReviewSkillProposalHandler()
        errors = handler.validate({"proposal_id": proposal_id, "action": action})
        if errors:
            _print_err("; ".join(errors))
            return 1
        result = await handler.execute(
            {"proposal_id": proposal_id, "action": action}, context
        )
        if not result.success:
            _print_err(result.error or "Skill proposal review failed")
            return 1
        sys.stdout.write(result.to_display_output() + "\n")
        return 0
    except Exception as exc:
        _print_err(f"Could not review skill proposal: {exc}")
        return 1
    finally:
        await pool.close()


async def _migrate(dsn: str, status_only: bool) -> int:
    """Apply pending schema migrations to the active database (never wipes data)."""
    import asyncpg

    from core.migrations import apply_pending_migrations, migration_status

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            if status_only:
                st = await migration_status(conn)
                for v in st["applied"]:
                    sys.stdout.write(f"  ✓ {v}\n")
                for v in st["pending"]:
                    sys.stdout.write(f"  • {v}  (pending)\n")
                for item in st["drifted"]:
                    sys.stdout.write(
                        f"  ! {item['version']}  (checksum mismatch: "
                        f"recorded {item['recorded_checksum']}, "
                        f"current {item['current_checksum']})\n"
                    )
                if not st["applied"] and not st["pending"]:
                    sys.stdout.write("No migrations found.\n")
                elif st["drifted"]:
                    _print_err(
                        "Applied migration files changed. Restore the exact applied "
                        "files and put corrections in a new forward migration."
                    )
                    return 1
                elif not st["pending"]:
                    sys.stdout.write("Schema is up to date.\n")
                return 0
            applied = await apply_pending_migrations(conn)
            if applied:
                sys.stdout.write("Applied migrations (no data was lost):\n")
                for v in applied:
                    sys.stdout.write(f"  ✓ {v}\n")
            else:
                sys.stdout.write("Schema already up to date — nothing to migrate.\n")
            return 0
    except Exception as e:
        _print_err(f"Migration failed: {e}")
        return 1
    finally:
        await pool.close()


def _do_backup(dsn: str, out_dir: str | None, label: str | None) -> int:
    from core.backup_restore import backup

    try:
        path = backup(dsn, out_dir, label)
        sys.stdout.write(f"Backup written: {path}\n")
        return 0
    except Exception as e:
        _print_err(f"Backup failed: {e}")
        return 1


def _do_restore(dsn: str, path: str, yes: bool) -> int:
    from core.backup_restore import restore

    if not yes:
        from apps.cli_theme import console

        console.print(
            "[bold red]WARNING:[/bold red] Restore REPLACES this database (all memories, "
            "identity, goals) with the backup. Stop the workers first (`hexis stop`)."
        )
        try:
            if input("Type 'restore' to confirm: ").strip().lower() != "restore":
                console.print("[dim]Aborted.[/dim]")
                return 1
        except (KeyboardInterrupt, EOFError):
            print()
            return 1
    try:
        restore(path, dsn)
        sys.stdout.write("Restore complete.\n")
        return 0
    except Exception as e:
        _print_err(f"Restore failed: {e}")
        return 1


async def _channels_setup(dsn: str, channel_type: str) -> int:
    """Interactive channel setup."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        if channel_type == "discord":
            sys.stdout.write("Discord Bot Setup\n")
            sys.stdout.write("=" * 40 + "\n")
            sys.stdout.write("1. Go to https://discord.com/developers/applications\n")
            sys.stdout.write("2. Create a New Application\n")
            sys.stdout.write("3. Go to Bot > Token > Copy\n")
            sys.stdout.write("4. Enable Message Content Intent in Bot settings\n")
            sys.stdout.write(
                "5. Invite bot to your server with bot + applications.commands scopes\n\n"
            )
            token_env = (
                input("Bot token env var name [DISCORD_BOT_TOKEN]: ").strip()
                or "DISCORD_BOT_TOKEN"
            )
            guilds = (
                input("Allowed guild IDs (comma-separated, or * for all) [*]: ").strip()
                or "*"
            )
            operator_user = input(
                "Your Discord user ID for standing instructions (optional): "
            ).strip()

            discord_settings = {
                "bot_token": token_env,
                "allowed_guilds": guilds,
            }
            if operator_user:
                discord_settings["operator_user_id"] = operator_user

            async with pool.acquire() as conn:
                await conn.execute(
                    "SELECT apply_channel_config('discord', $1::jsonb)",
                    json.dumps(discord_settings),
                )

            sys.stdout.write(
                f"\nDiscord configured. Set {token_env} in your environment.\n"
            )
            sys.stdout.write("Start with: hexis channels start --channel discord\n")

        elif channel_type == "telegram":
            sys.stdout.write("Telegram Bot Setup\n")
            sys.stdout.write("=" * 40 + "\n")
            sys.stdout.write("1. Message @BotFather on Telegram\n")
            sys.stdout.write("2. Send /newbot and follow the prompts\n")
            sys.stdout.write("3. Copy the bot token\n\n")
            token_env = (
                input("Bot token env var name [TELEGRAM_BOT_TOKEN]: ").strip()
                or "TELEGRAM_BOT_TOKEN"
            )
            chats = (
                input("Allowed chat IDs (comma-separated, or * for all) [*]: ").strip()
                or "*"
            )
            operator_user = input(
                "Your Telegram user ID for standing instructions (optional): "
            ).strip()

            telegram_settings = {
                "bot_token": token_env,
                "allowed_chat_ids": chats,
            }
            if operator_user:
                telegram_settings["operator_user_id"] = operator_user

            async with pool.acquire() as conn:
                await conn.execute(
                    "SELECT apply_channel_config('telegram', $1::jsonb)",
                    json.dumps(telegram_settings),
                )

            sys.stdout.write(
                f"\nTelegram configured. Set {token_env} in your environment.\n"
            )
            sys.stdout.write("Start with: hexis channels start --channel telegram\n")

        elif channel_type == "slack":
            sys.stdout.write("Slack Bot Setup\n")
            sys.stdout.write("=" * 40 + "\n")
            sys.stdout.write(
                "1. Go to https://api.slack.com/apps and create a new app\n"
            )
            sys.stdout.write(
                "2. Under OAuth & Permissions, add scopes: chat:write, channels:history, users:read, im:write\n"
            )
            sys.stdout.write(
                "3. Install to workspace and copy the Bot User OAuth Token (xoxb-...)\n"
            )
            sys.stdout.write(
                "4. For Socket Mode: enable it under Socket Mode and copy the App Token (xapp-...)\n\n"
            )
            bot_env = (
                input("Bot token env var name [SLACK_BOT_TOKEN]: ").strip()
                or "SLACK_BOT_TOKEN"
            )
            app_env = (
                input(
                    "App token env var name (for Socket Mode) [SLACK_APP_TOKEN]: "
                ).strip()
                or "SLACK_APP_TOKEN"
            )
            owner_user = input(
                "Your Slack user ID for private approval DMs (U...): "
            ).strip()
            signing_env = input(
                "Signing-secret env var name (HTTP interactivity; blank for Socket Mode): "
            ).strip()
            channels_allow = (
                input(
                    "Allowed channel IDs (comma-separated, or * for all) [*]: "
                ).strip()
                or "*"
            )

            slack_settings = {
                "bot_token": bot_env,
                "app_token": app_env,
                "allowed_channels": channels_allow,
            }
            if owner_user:
                slack_settings["operator_user_id"] = owner_user
            if signing_env:
                slack_settings["signing_secret"] = signing_env

            async with pool.acquire() as conn:
                await conn.execute(
                    "SELECT apply_channel_config('slack', $1::jsonb)",
                    json.dumps(slack_settings),
                )

            sys.stdout.write(
                f"\nSlack configured. Set {bot_env} and {app_env} in your environment.\n"
            )
            if not owner_user:
                sys.stdout.write(
                    "Protected tools will stay fail-closed outside the local terminal. "
                    "Run this setup again with your Slack U... user ID to approve from a phone.\n"
                )
            sys.stdout.write("Start with: hexis channels start --channel slack\n")

        elif channel_type == "signal":
            sys.stdout.write("Signal Setup (via signal-cli-rest-api)\n")
            sys.stdout.write("=" * 40 + "\n")
            sys.stdout.write(
                "1. Run signal-cli-rest-api as a sidecar (or use 'docker compose --profile signal up')\n"
            )
            sys.stdout.write("2. Register/link your phone number with signal-cli\n")
            sys.stdout.write("3. Provide the registered phone number\n\n")
            phone_env = (
                input("Phone number env var name [SIGNAL_PHONE_NUMBER]: ").strip()
                or "SIGNAL_PHONE_NUMBER"
            )
            api_url = (
                input("Signal CLI API URL [http://localhost:8080]: ").strip()
                or "http://localhost:8080"
            )
            numbers = (
                input(
                    "Allowed sender numbers (comma-separated, or * for all) [*]: "
                ).strip()
                or "*"
            )
            operator_user = input(
                "Your Signal number for standing instructions (optional): "
            ).strip()

            signal_settings = {
                "phone_number": phone_env,
                "api_url": api_url,
                "allowed_numbers": numbers,
            }
            if operator_user:
                signal_settings["operator_user_id"] = operator_user

            async with pool.acquire() as conn:
                await conn.execute(
                    "SELECT apply_channel_config('signal', $1::jsonb)",
                    json.dumps(signal_settings),
                )

            sys.stdout.write(
                f"\nSignal configured. Set {phone_env} in your environment.\n"
            )
            sys.stdout.write("Start with: hexis channels start --channel signal\n")

        elif channel_type == "whatsapp":
            sys.stdout.write("WhatsApp Business Cloud API Setup\n")
            sys.stdout.write("=" * 40 + "\n")
            sys.stdout.write(
                "1. Go to https://developers.facebook.com and create a Meta Business app\n"
            )
            sys.stdout.write("2. Add the WhatsApp product\n")
            sys.stdout.write("3. Get your access token and phone number ID\n")
            sys.stdout.write("4. Configure a webhook pointing to your server\n\n")
            token_env = (
                input("Access token env var name [WHATSAPP_ACCESS_TOKEN]: ").strip()
                or "WHATSAPP_ACCESS_TOKEN"
            )
            phone_id = (
                input(
                    "Phone number ID (or env var) [WHATSAPP_PHONE_NUMBER_ID]: "
                ).strip()
                or "WHATSAPP_PHONE_NUMBER_ID"
            )
            verify = (
                input("Webhook verify token [hexis_verify]: ").strip() or "hexis_verify"
            )
            port = input("Webhook port [8443]: ").strip() or "8443"
            numbers = (
                input(
                    "Allowed sender numbers (comma-separated, or * for all) [*]: "
                ).strip()
                or "*"
            )
            operator_user = input(
                "Your WhatsApp number for standing instructions (optional): "
            ).strip()

            whatsapp_settings = {
                "access_token": token_env,
                "phone_number_id": phone_id,
                "verify_token": verify,
                "webhook_port": port,
                "allowed_numbers": numbers,
            }
            if operator_user:
                whatsapp_settings["operator_user_id"] = operator_user

            async with pool.acquire() as conn:
                await conn.execute(
                    "SELECT apply_channel_config('whatsapp', $1::jsonb)",
                    json.dumps(whatsapp_settings),
                )

            sys.stdout.write(
                f"\nWhatsApp configured. Set {token_env} in your environment.\n"
            )
            sys.stdout.write("Start with: hexis channels start --channel whatsapp\n")

        elif channel_type == "imessage":
            sys.stdout.write("iMessage Setup (via BlueBubbles)\n")
            sys.stdout.write("=" * 40 + "\n")
            sys.stdout.write("1. Install BlueBubbles server on a Mac with iMessage\n")
            sys.stdout.write("2. Configure and start the BlueBubbles server\n")
            sys.stdout.write("3. Note the server URL and password\n\n")
            api_url = (
                input("BlueBubbles API URL [http://localhost:1234]: ").strip()
                or "http://localhost:1234"
            )
            password_env = (
                input("Password env var name [IMESSAGE_PASSWORD]: ").strip()
                or "IMESSAGE_PASSWORD"
            )
            handles = (
                input("Allowed handles (comma-separated, or * for all) [*]: ").strip()
                or "*"
            )
            operator_recipient = input(
                "Your iMessage phone/email for approval escalation (optional): "
            ).strip()

            imessage_settings = {
                "api_url": api_url,
                "password": password_env,
                "allowed_handles": handles,
            }
            if operator_recipient:
                imessage_settings["operator_recipient"] = operator_recipient

            async with pool.acquire() as conn:
                await conn.execute(
                    "SELECT apply_channel_config('imessage', $1::jsonb)",
                    json.dumps(imessage_settings),
                )

            sys.stdout.write(
                f"\niMessage configured. Set {password_env} in your environment.\n"
            )
            if operator_recipient:
                sys.stdout.write(
                    "Unanswered Slack approvals will escalate to that recipient after the configured delay.\n"
                )
            sys.stdout.write("Start with: hexis channels start --channel imessage\n")

        elif channel_type == "matrix":
            sys.stdout.write("Matrix Setup\n")
            sys.stdout.write("=" * 40 + "\n")
            sys.stdout.write("1. Create a bot account on your Matrix homeserver\n")
            sys.stdout.write("2. Generate an access token for the bot\n")
            sys.stdout.write("3. Invite the bot to rooms you want it to monitor\n\n")
            homeserver = (
                input("Homeserver URL [https://matrix.org]: ").strip()
                or "https://matrix.org"
            )
            user_id = input("Bot user ID (e.g. @hexis:matrix.org): ").strip()
            token_env = (
                input("Access token env var name [MATRIX_ACCESS_TOKEN]: ").strip()
                or "MATRIX_ACCESS_TOKEN"
            )
            rooms = (
                input("Allowed room IDs (comma-separated, or * for all) [*]: ").strip()
                or "*"
            )
            operator_user = input(
                "Your Matrix user ID for standing instructions (optional): "
            ).strip()

            matrix_settings = {
                "homeserver": homeserver,
                "user_id": user_id,
                "access_token": token_env,
                "allowed_rooms": rooms,
            }
            if operator_user:
                matrix_settings["operator_user_id"] = operator_user

            async with pool.acquire() as conn:
                await conn.execute(
                    "SELECT apply_channel_config('matrix', $1::jsonb)",
                    json.dumps(matrix_settings),
                )

            sys.stdout.write(
                f"\nMatrix configured. Set {token_env} in your environment.\n"
            )
            sys.stdout.write("Start with: hexis channels start --channel matrix\n")

        return 0
    except (KeyboardInterrupt, EOFError):
        _print_err("Aborted.")
        return 1
    except Exception as e:
        _print_err(f"Error: {e}")
        return 1
    finally:
        await pool.close()


def _print_rich_status(p: dict[str, Any]) -> None:
    """Print a rich, human-readable status display."""
    from apps.cli_theme import console, energy_bar, make_panel, mood_label
    from rich.text import Text

    identity = p.get("identity") or "(not configured)"
    instance = p.get("instance", "default")
    database = p.get("database", "hexis_memory")

    lines = Text()

    # Identity + Instance
    lines.append("Instance  ", style="key")
    lines.append(f"{instance} ", style="accent")
    lines.append(f"({database})\n", style="muted")
    lines.append("Identity  ", style="key")
    lines.append(f"{identity}\n")

    # Energy
    energy = p.get("energy")
    reserve_energy = p.get("energy_reserve", p.get("max_energy", 20))
    energy_capacity = p.get("energy_capacity", reserve_energy)
    if energy is not None:
        regen = p.get("next_regen_minutes")
        regen_str = (
            f"  [muted](regen in {regen}m)[/muted]"
            if regen and energy < energy_capacity
            else ""
        )
        lines.append("Energy    ", style="key")
        console.print(make_panel(lines, title=identity, subtitle=instance))
        lines = Text()
        console.print(
            f"  [key]Energy   [/key] {energy_bar(energy, energy_capacity)}"
            f"  [muted](reserve {reserve_energy})[/muted]{regen_str}"
        )
    else:
        console.print(make_panel(lines, title=identity, subtitle=instance))

    # Heartbeat
    paused = p.get("heartbeat_paused", False)
    active = p.get("heartbeat_active", False)
    last_ago = p.get("last_heartbeat_ago")
    interval = p.get("heartbeat_interval_minutes")
    if paused:
        console.print("  [key]Heartbeat[/key] [warn]paused[/warn]")
    elif active and last_ago:
        interval_str = f", interval: {int(interval)}m" if interval else ""
        console.print(
            f"  [key]Heartbeat[/key] [ok]active[/ok] [muted](last: {last_ago} ago{interval_str})[/muted]"
        )
    elif last_ago:
        console.print(
            f"  [key]Heartbeat[/key] [muted]idle (last: {last_ago} ago)[/muted]"
        )
    else:
        console.print("  [key]Heartbeat[/key] [muted]never run[/muted]")

    # Worker runtime liveness
    workers = p.get("workers", [])
    active_workers = [
        w for w in workers if w.get("status") not in {"stopped", "terminated"}
    ]
    if active_workers:
        latest_by_mode = {}
        for worker in active_workers:
            mode = worker.get("mode") or "unknown"
            latest_by_mode.setdefault(mode, worker)
        parts = []
        for mode, worker in sorted(latest_by_mode.items()):
            status = "stale" if worker.get("is_stale") else worker.get("status", "?")
            age = worker.get("last_seen_age_s")
            age_s = f"{age}s" if age is not None else "?"
            style = "warn" if status == "stale" else "ok"
            task = worker.get("current_task_type")
            task_s = f":{task}" if task else ""
            parts.append(
                f"[{style}]{mode}{task_s} {status}[/{style}] [muted]({age_s})[/muted]"
            )
        console.print(f"  [key]Workers  [/key] {', '.join(parts)}")
    else:
        # No records at all means nothing is running the loops — say the fix
        # rather than leaving the reader to decode "no liveness records".
        console.print(
            "  [key]Workers  [/key] [warn]not running[/warn] "
            "[muted](the heartbeat only ticks while they are up — start them with `hexis up`)[/muted]"
        )

    # Memory counts
    memories = p.get("memories", {})
    if memories:
        parts = []
        for mtype, cnt in sorted(memories.items()):
            parts.append(f"[accent]{cnt}[/accent] {mtype}")
        console.print(f"  [key]Memory   [/key] {', '.join(parts)}")
    else:
        console.print("  [key]Memory   [/key] [muted](empty)[/muted]")

    # Channels
    channels = p.get("channels", [])
    if channels:
        ch_parts = [f"[teal]{ch['type']}[/teal]" for ch in channels]
        console.print(f"  [key]Channels [/key] {', '.join(ch_parts)}")

    # Goals
    goals = p.get("goals", [])
    if goals:
        console.print(f"  [key]Goals    [/key] [accent]{len(goals)}[/accent] active")
        for g in goals:
            console.print(f"             [muted]\u2022[/muted] {g['content']}")

    # Scheduled tasks
    sched = p.get("scheduled_tasks", 0)
    if sched > 0:
        console.print(
            f"  [key]Scheduled[/key] {sched} active task{'s' if sched != 1 else ''}"
        )

    worker_tasks = p.get("worker_tasks", [])
    failed_tasks = [
        t for t in worker_tasks if int(t.get("failures_since_success") or 0) > 0
    ]
    if failed_tasks:
        parts = [
            f"[warn]{t.get('task_type')} x{t.get('failures_since_success')}[/warn]"
            for t in failed_tasks[:4]
        ]
        console.print(f"  [key]Task Fail[/key] {', '.join(parts)}")

    # Mood
    mood = p.get("mood")
    valence = p.get("valence")
    if mood:
        console.print(f"  [key]Mood     [/key] {mood_label(mood, valence)}")

    console.print()


async def _recall(
    dsn: str, query: str, limit: int, memory_type: str | None, as_json: bool
) -> int:
    """Search memories by semantic query."""
    import asyncpg
    from core.cognitive_memory_api import CognitiveMemory, MemoryType

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        mem = CognitiveMemory(pool)
        types = [MemoryType(memory_type)] if memory_type else None
        result = await mem.recall(query, limit=limit, memory_types=types)

        if as_json:
            data = [
                {
                    "id": str(m.id),
                    "type": m.type,
                    "content": m.content,
                    "importance": m.importance,
                    "similarity": m.similarity,
                    "created_at": str(m.created_at) if m.created_at else None,
                }
                for m in result.memories
            ]
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
        else:
            from apps.cli_theme import console as _con, make_table as _mt

            if not result.memories:
                _con.print("[muted]No memories found.[/muted]")
                return 0

            table = _mt(
                ("Type", {"style": "teal"}),
                "Content",
                ("Imp.", {"justify": "right"}),
                ("Sim.", {"justify": "right"}),
                "Created",
                title=f"Recall: {query}",
            )
            for m in result.memories:
                content = m.content[:120] + "..." if len(m.content) > 120 else m.content
                created = (
                    m.created_at.strftime("%Y-%m-%d %H:%M") if m.created_at else "-"
                )
                table.add_row(
                    m.type,
                    content,
                    f"{m.importance:.2f}" if m.importance else "-",
                    f"{m.similarity:.2f}" if m.similarity else "-",
                    created,
                )
            _con.print(table)
            _con.print(f"[muted]{len(result.memories)} memories found[/muted]")

        return 0
    finally:
        await pool.close()


async def _goals_list(dsn: str, priority: str | None, as_json: bool) -> int:
    """List goals by priority."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            if priority:
                rows = await conn.fetch(
                    "SELECT * FROM get_goals_by_priority($1::goal_priority)", priority
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM get_goals_by_priority(NULL::goal_priority)"
                )

        goals = [dict(r) for r in rows]
        if as_json:
            for g in goals:
                for k, v in g.items():
                    if hasattr(v, "isoformat"):
                        g[k] = v.isoformat()
                    elif isinstance(v, bytes):
                        g[k] = None
            sys.stdout.write(json.dumps(goals, indent=2, default=str) + "\n")
        else:
            from apps.cli_theme import console as _con, make_table as _mt

            if not goals:
                _con.print("[muted]No goals found.[/muted]")
                return 0

            # Group by priority
            by_priority: dict[str, list] = {}
            for g in goals:
                p = str(g.get("priority", "unknown"))
                by_priority.setdefault(p, []).append(g)

            priority_colors = {
                "active": "accent",
                "queued": "teal",
                "backburner": "muted",
                "completed": "ok",
                "abandoned": "fail",
            }

            table = _mt(
                ("Priority", {"style": "bold"}),
                "Title",
                ("Source", {"style": "muted"}),
                "Last Touched",
                title="Goals",
            )
            first_group = True
            for prio in ["active", "queued", "backburner", "completed", "abandoned"]:
                group = by_priority.get(prio, [])
                if not group:
                    continue
                if not first_group:
                    table.add_section()
                first_group = False
                for g in group:
                    color = priority_colors.get(prio, "muted")
                    title = g.get("content") or g.get("title") or "(untitled)"
                    if len(title) > 60:
                        title = title[:57] + "..."
                    source = str(g.get("source", "")) or "-"
                    meta = g.get("metadata") or {}
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {}
                    touched = meta.get("last_touched", "")
                    if hasattr(touched, "strftime"):
                        touched = touched.strftime("%Y-%m-%d")
                    elif isinstance(touched, str) and len(touched) > 10:
                        touched = touched[:10]
                    table.add_row(
                        f"[{color}]{prio}[/{color}]", title, source, str(touched) or "-"
                    )
            _con.print(table)

        return 0
    finally:
        await pool.close()


async def _goals_create(
    dsn: str, title: str, description: str | None, priority: str, source: str
) -> int:
    """Create a new goal."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            goal_id = await conn.fetchval(
                "SELECT create_goal($1, $2, $3::goal_source, $4::goal_priority)",
                title,
                description,
                source,
                priority,
            )
        from apps.cli_theme import console as _con

        _con.print(
            f"[ok]\u2714[/ok] Goal created: [bold]{title}[/bold] [muted]({goal_id})[/muted]"
        )
        return 0
    finally:
        await pool.close()


async def _goals_update(
    dsn: str, goal_id: str, priority: str, reason: str | None
) -> int:
    """Change goal priority."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT change_goal_priority($1::uuid, $2::goal_priority, $3)",
                goal_id,
                priority,
                reason,
            )
        from apps.cli_theme import console as _con

        _con.print(
            f"[ok]\u2714[/ok] Goal {goal_id[:8]}... priority changed to [bold]{priority}[/bold]"
        )
        return 0
    except Exception as e:
        _print_err(f"Failed to update goal: {e}")
        return 1
    finally:
        await pool.close()


async def _schedule_list(dsn: str, status_filter: str | None, as_json: bool) -> int:
    """List scheduled tasks."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            if status_filter:
                rows = await conn.fetch(
                    "SELECT * FROM list_scheduled_tasks($1)",
                    status_filter,
                )
            else:
                rows = await conn.fetch("SELECT * FROM list_scheduled_tasks()")

        tasks = [dict(r) for r in rows]
        if as_json:
            sys.stdout.write(json.dumps(tasks, indent=2, default=str) + "\n")
        else:
            from apps.cli_theme import console as _con, make_table as _mt

            if not tasks:
                _con.print("[muted]No scheduled tasks found.[/muted]")
                return 0

            table = _mt(
                ("Name", {"style": "bold"}),
                "Kind",
                ("Status", {"style": "teal"}),
                "Next Run",
                "Action",
                title="Scheduled Tasks",
            )
            for t in tasks:
                status = str(t.get("status", ""))
                status_styled = (
                    f"[ok]{status}[/ok]"
                    if status == "active"
                    else (
                        f"[warn]{status}[/warn]"
                        if status == "paused"
                        else f"[muted]{status}[/muted]"
                    )
                )
                next_run = t.get("next_run_at", "")
                if hasattr(next_run, "strftime"):
                    next_run = next_run.strftime("%Y-%m-%d %H:%M")
                table.add_row(
                    str(t.get("name", "")),
                    str(t.get("schedule_kind", "")),
                    status_styled,
                    str(next_run) or "-",
                    str(t.get("action_kind", "")),
                )
            _con.print(table)

        return 0
    finally:
        await pool.close()


async def _schedule_create(
    dsn: str,
    name: str,
    kind: str,
    action: str,
    payload_str: str,
    schedule_str: str,
    timezone: str,
    description: str | None,
) -> int:
    """Create a scheduled task."""
    import asyncpg

    try:
        schedule_json = json.loads(schedule_str)
        action_payload = json.loads(payload_str)
    except json.JSONDecodeError as e:
        _print_err(f"Invalid JSON: {e}")
        return 1

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            task_id = await conn.fetchval(
                "SELECT create_scheduled_task($1, $2, $3::jsonb, $4, $5::jsonb, $6, $7)",
                name,
                kind,
                json.dumps(schedule_json),
                action,
                json.dumps(action_payload),
                timezone,
                description,
            )
        from apps.cli_theme import console as _con

        _con.print(
            f"[ok]\u2714[/ok] Scheduled task created: [bold]{name}[/bold] [muted]({task_id})[/muted]"
        )
        return 0
    finally:
        await pool.close()


async def _schedule_delete(dsn: str, task_id: str, force: bool) -> int:
    """Delete a scheduled task."""
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT delete_scheduled_task($1::uuid, $2)",
                task_id,
                force,
            )
        from apps.cli_theme import console as _con

        action = "deleted" if force else "disabled"
        _con.print(f"[ok]\u2714[/ok] Task {task_id[:8]}... {action}")
        return 0
    except Exception as e:
        _print_err(f"Failed to delete task: {e}")
        return 1
    finally:
        await pool.close()


def _port_ready(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    """True if something accepts a TCP connection on host:port."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ready(url: str, timeout: float = 1.0) -> bool:
    """True if a local HTTP endpoint responds without a transport error."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(getattr(resp, "status", 0)) < 500
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


_LOCAL_EMBEDDING_COMMAND = "embeddinggemma"
_LOCAL_EMBEDDING_INSTALLER = "curl -fsSL https://raw.githubusercontent.com/QuixiAI/embeddinggemma.c/main/install.sh | sh"
_LOCAL_EMBEDDING_PORT = 42666
_LOCAL_EMBEDDING_LOG = Path.home() / ".hexis" / "embeddinggemma.log"
_LOCAL_EMBEDDING_CACHE_MARKER = ".hexis-owned"


def _local_embedding_ownership_file() -> Path:
    return _LOCAL_EMBEDDING_LOG.with_name("embeddinggemma-owned.json")


def _local_embedding_cache_dir() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root).expanduser() / "embeddinggemma.c"
    return Path.home() / ".cache" / "embeddinggemma.c"


def _file_sha256(path: Path) -> str | None:
    import hashlib

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _read_local_embedding_ownership() -> dict[str, Any] | None:
    ownership_file = _local_embedding_ownership_file()
    if not ownership_file.is_file():
        return {}
    try:
        record = json.loads(ownership_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("version") != 1:
        return None
    return record


def _write_local_embedding_ownership(record: dict[str, Any]) -> bool:
    ownership_file = _local_embedding_ownership_file()
    temporary_file = ownership_file.with_name(
        f".{ownership_file.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    )
    try:
        ownership_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_file, ownership_file)
        return True
    except OSError:
        try:
            temporary_file.unlink(missing_ok=True)
        except OSError:
            return False
        return False


def _record_owned_local_embedding_binary(binary: Path) -> bool:
    """Record proof that Hexis installed this exact companion binary."""
    digest = _file_sha256(binary)
    if not digest:
        return False
    record = _read_local_embedding_ownership()
    if record is None:
        return False
    record.update(
        {
            "version": 1,
            "binary_path": str(binary.resolve()),
            "binary_sha256": digest,
        }
    )
    return _write_local_embedding_ownership(record)


def _mark_local_embedding_cache_if_created(
    *, existed_before_start: bool
) -> bool | None:
    """Mark a cache only when it appeared after Hexis launched the sidecar."""
    if existed_before_start:
        return None
    cache_dir = _local_embedding_cache_dir()
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        return None
    resolved_cache = cache_dir.resolve()
    marker_file = cache_dir / _LOCAL_EMBEDDING_CACHE_MARKER
    try:
        if marker_file.is_file():
            marker = json.loads(marker_file.read_text(encoding="utf-8"))
            marker_token = marker.get("token") if isinstance(marker, dict) else None
            if (
                not isinstance(marker, dict)
                or marker.get("owner") != "hexis"
                or not isinstance(marker_token, str)
            ):
                return False
        else:
            marker_token = os.urandom(16).hex()
            marker_file.write_text(
                json.dumps({"owner": "hexis", "token": marker_token}) + "\n",
                encoding="utf-8",
            )
    except (OSError, json.JSONDecodeError):
        return False

    record = _read_local_embedding_ownership()
    if record is None:
        return False
    if (
        record.get("cache_path") == str(resolved_cache)
        and record.get("cache_marker") == marker_token
    ):
        return True
    record.update(
        {
            "version": 1,
            "cache_path": str(resolved_cache),
            "cache_marker": marker_token,
        }
    )
    return _write_local_embedding_ownership(record)


def _purge_owned_local_embedding_assets() -> tuple[list[Path], list[str]]:
    """Delete only companion assets carrying durable Hexis ownership proof."""
    removed: list[Path] = []
    notes: list[str] = []

    ownership_file = _local_embedding_ownership_file()
    record = _read_local_embedding_ownership()
    if record is None:
        notes.append(
            f"Could not read embedding ownership record {ownership_file}; "
            "the binary and model cache were left alone."
        )
        return removed, notes

    binary_value = record.get("binary_path")
    expected_digest = record.get("binary_sha256")
    if isinstance(binary_value, str) and isinstance(expected_digest, str):
        binary = Path(binary_value).expanduser()
        actual_digest = _file_sha256(binary)
        if actual_digest == expected_digest:
            try:
                binary.unlink(missing_ok=True)
                removed.append(binary)
            except OSError as exc:
                notes.append(f"Could not remove owned embedding binary {binary}: {exc}")
        elif binary.exists():
            notes.append(
                f"The embedding binary at {binary} changed after Hexis installed it, "
                "so it was left alone."
            )

    cache_value = record.get("cache_path")
    expected_marker = record.get("cache_marker")
    if isinstance(cache_value, str) and isinstance(expected_marker, str):
        cache_dir = Path(cache_value).expanduser()
        cache_marker = cache_dir / _LOCAL_EMBEDDING_CACHE_MARKER
        try:
            marker = json.loads(cache_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker = None
        marker_matches = (
            isinstance(marker, dict)
            and marker.get("owner") == "hexis"
            and marker.get("token") == expected_marker
        )
        cache_root = cache_dir.parent.resolve()
        resolved_cache = cache_dir.resolve()
        if not marker_matches and cache_dir.exists():
            notes.append(
                f"The ownership marker for embedding cache {cache_dir} changed or "
                "disappeared, so the cache was left alone."
            )
        elif marker_matches and (
            cache_dir.is_symlink()
            or resolved_cache.name != "embeddinggemma.c"
            or not _path_is_within(resolved_cache, cache_root)
        ):
            notes.append(f"Refusing unsafe owned embedding cache path: {cache_dir}")
        elif marker_matches:
            try:
                shutil.rmtree(resolved_cache)
                removed.append(resolved_cache)
            except OSError as exc:
                notes.append(
                    f"Could not remove owned embedding cache {resolved_cache}: {exc}"
                )

    return removed, notes


def _local_embedding_pid_file() -> Path:
    return _LOCAL_EMBEDDING_LOG.with_name("embeddinggemma.pid")


def _record_local_embedding_process(proc: subprocess.Popen[Any]) -> bool:
    """Remember only processes this CLI actually launched.

    The ownership record lets `hexis down` and `hexis uninstall` stop the
    companion service without guessing about an ambient embeddinggemma process
    that another application may own.
    """
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        pid_file = _local_embedding_pid_file()
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{pid}\n", encoding="utf-8")
        return True
    except OSError:
        return False


def _forget_local_embedding_process() -> None:
    try:
        _local_embedding_pid_file().unlink(missing_ok=True)
    except OSError:
        pass


def _process_command(pid: int) -> str | None:
    """Return a process command for PID-reuse protection (macOS/Linux)."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    command = (result.stdout or "").strip()
    return command or None


def _port_listener_pids(port: int) -> list[int]:
    lsof = shutil.which("lsof")
    if not lsof:
        return []
    try:
        result = subprocess.run(
            [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = int(line[1:])
            if pid not in pids:
                pids.append(pid)
    return pids


def _process_uses_hexis_embedding_log(pid: int) -> bool:
    """Verify that both process output streams target Hexis's private log."""
    lsof = shutil.which("lsof")
    if not lsof:
        return False
    try:
        result = subprocess.run(
            [lsof, "-nP", "-a", "-p", str(pid), "-d", "1,2", "-Fn"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    expected_log = str(_LOCAL_EMBEDDING_LOG.resolve())
    paths_by_fd: dict[str, str] = {}
    current_fd: str | None = None
    for line in (result.stdout or "").splitlines():
        if line in {"f1", "f2"}:
            current_fd = line[1:]
        elif current_fd and line.startswith("n"):
            path = line[1:]
            if path.endswith(" (deleted)"):
                path = path[: -len(" (deleted)")]
            paths_by_fd[current_fd] = path
            current_fd = None
    return paths_by_fd == {"1": expected_log, "2": expected_log}


def _legacy_owned_local_embedding_pid() -> int | None:
    """Recover pre-PID-file ownership from the exact Hexis launch contract."""
    matches: list[int] = []
    for pid in _port_listener_pids(_LOCAL_EMBEDDING_PORT):
        command = _process_command(pid)
        if (
            command
            and Path(command).name.lower() == _LOCAL_EMBEDDING_COMMAND
            and _process_uses_hexis_embedding_log(pid)
        ):
            matches.append(pid)
    return matches[0] if len(matches) == 1 else None


def _stop_owned_local_embedding_service() -> tuple[bool, str | None]:
    """Stop the sidecar iff an ownership record still names that process.

    Returns ``(stopped, note)``. A note explains any running service Hexis
    deliberately left alone because ownership could not be verified.
    """
    import signal
    import time

    pid_file = _local_embedding_pid_file()
    if not pid_file.exists():
        pid = _legacy_owned_local_embedding_pid()
        if pid is None:
            if _port_ready(_LOCAL_EMBEDDING_PORT):
                listener = _port_listener_summary(_LOCAL_EMBEDDING_PORT)
                detail = f" ({listener})" if listener else ""
                return False, (
                    f"An embedding service is still listening on port "
                    f"{_LOCAL_EMBEDDING_PORT}{detail}. Hexis could not verify that it "
                    "owns this process, so it was left running."
                )
            return False, None
    else:
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            _forget_local_embedding_process()
            return False, None

    command = _process_command(pid)
    if command is None:
        _forget_local_embedding_process()
        return False, (
            f"Could not verify the saved embedding PID {pid}; it was left alone."
            if _port_ready(_LOCAL_EMBEDDING_PORT)
            else None
        )
    if Path(command).name.lower() != _LOCAL_EMBEDDING_COMMAND:
        _forget_local_embedding_process()
        return False, (
            f"The saved embedding PID {pid} now belongs to another process "
            f"({command}); it was left alone."
        )

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _forget_local_embedding_process()
        return False, None
    except OSError as exc:
        return False, f"Could not stop embedding service PID {pid}: {exc}"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _process_command(pid) is None:
            _forget_local_embedding_process()
            return True, None
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        return False, f"Could not force-stop embedding service PID {pid}: {exc}"
    _forget_local_embedding_process()
    return True, None


def _port_listener_summary(port: int) -> str | None:
    """Best-effort process name(s) listening on a local TCP port."""
    lsof = shutil.which("lsof")
    if not lsof:
        return None
    try:
        p = subprocess.run(
            [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-F", "pc"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if p.returncode != 0:
        return None

    names: list[str] = []
    current_pid: str | None = None
    for line in p.stdout.splitlines():
        if line.startswith("p"):
            current_pid = line[1:]
        elif line.startswith("c"):
            name = line[1:]
            if current_pid:
                names.append(f"{name} (pid {current_pid})")
            else:
                names.append(name)
    return ", ".join(names) if names else None


def _configured_embedding_url(env_file: Path | None) -> str | None:
    """Read the embedding URL the compose stack will use, without mutating env."""
    env_url = os.getenv("EMBEDDING_SERVICE_URL")
    if env_url:
        return env_url
    if env_file and env_file.exists():
        try:
            value = dotenv_values(env_file).get("EMBEDDING_SERVICE_URL")
            return str(value) if value else None
        except Exception:
            return None
    return None


def _uses_local_embedding_sidecar(env_file: Path | None) -> bool:
    """True when the DB is configured for the published local sidecar."""
    url = (_configured_embedding_url(env_file) or "").lower()
    if not url:
        return True
    return ":42666" in url


def _uses_legacy_embedding_sidecar_port(env_file: Path | None) -> bool:
    """True when config is pinned to Hexis' retired local sidecar port."""
    url = (_configured_embedding_url(env_file) or "").lower()
    return ":11434" in url and (
        "host.docker.internal" in url or "localhost" in url or "127.0.0.1" in url
    )


def _warn_legacy_embedding_sidecar_port(env_file: Path | None) -> None:
    if not _uses_legacy_embedding_sidecar_port(env_file):
        return
    from apps.cli_theme import console

    console.print(
        "[warn]Embedding URL is pinned to the legacy local sidecar port 11434.[/warn]\n"
        "  Hexis now uses the published [accent]embeddinggemma[/accent] binary on port 42666.\n"
        "  Remove the old EMBEDDING_SERVICE_URL override, or set:\n"
        "    [accent]EMBEDDING_SERVICE_URL=http://host.docker.internal:42666/api/embed[/accent]"
    )


def _local_embedding_binary() -> Path | None:
    """Resolve the published embeddinggemma executable, not a source checkout."""
    resolved = shutil.which(_LOCAL_EMBEDDING_COMMAND)
    if resolved:
        return Path(resolved)
    installer_default = Path.home() / ".local" / "bin" / _LOCAL_EMBEDDING_COMMAND
    if installer_default.exists():
        return installer_default
    return None


def _install_local_embedding_binary() -> Path | None:
    """Install the published embeddinggemma binary, then resolve it."""
    from apps.cli_theme import console

    console.print(
        f"[muted]Embedding service binary '{_LOCAL_EMBEDDING_COMMAND}' is not installed; "
        "installing it now...[/muted]"
    )
    manual_fix = (
        "  Install it manually, then run [accent]hexis up[/accent] again:\n"
        f"    [accent]{_LOCAL_EMBEDDING_INSTALLER}[/accent]"
    )
    try:
        result = subprocess.run(
            ["sh", "-c", _LOCAL_EMBEDDING_INSTALLER],
            stdin=subprocess.DEVNULL,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        console.print(
            f"[warn]Installing {_LOCAL_EMBEDDING_COMMAND} timed out after 300 seconds.[/warn]\n"
            + manual_fix
        )
        return None
    except Exception as exc:
        console.print(
            f"[warn]Couldn't run the {_LOCAL_EMBEDDING_COMMAND} installer: {exc}[/warn]\n"
            + manual_fix
        )
        return None
    if result.returncode != 0:
        console.print(
            f"[warn]The {_LOCAL_EMBEDDING_COMMAND} installer exited with code {result.returncode}.[/warn]\n"
            + manual_fix
        )
        return None

    binary = _local_embedding_binary()
    if binary is None:
        console.print(
            f"[warn]The installer finished, but '{_LOCAL_EMBEDDING_COMMAND}' still wasn't found.[/warn]\n"
            "  Expected it on PATH or at "
            f"[accent]{Path.home() / '.local' / 'bin' / _LOCAL_EMBEDDING_COMMAND}[/accent]."
        )
    elif not _record_owned_local_embedding_binary(binary):
        console.print(
            f"[warn]Installed {binary}, but could not record that Hexis owns it.[/warn]\n"
            "  Hexis will leave this binary alone during uninstall rather than risk "
            "deleting a shared executable."
        )
    return binary


def _tail_text(path: Path, *, max_lines: int = 12) -> str:
    """Best-effort text tail for CLI diagnostics."""
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-max_lines:])
    except Exception:
        return ""


def _start_local_embedding_service(wait_seconds: float = 90.0) -> bool:
    """Start the published embeddinggemma sidecar if port 42666 is idle."""
    import time as _time

    from apps.cli_theme import console

    if _port_ready(_LOCAL_EMBEDDING_PORT):
        listener = _port_listener_summary(_LOCAL_EMBEDDING_PORT)
        if listener and "embeddinggemma" in listener.lower():
            console.print(
                f"[ok]Embedding service is already listening on port {_LOCAL_EMBEDDING_PORT}.[/ok]"
            )
            return True
        detail = f" Listener: {listener}." if listener else ""
        console.print(
            f"[warn]Port {_LOCAL_EMBEDDING_PORT} is already in use, so Hexis did not start {_LOCAL_EMBEDDING_COMMAND}.[/warn]"
            f"{detail}\n"
            "  If this is not the embeddinggemma sidecar, stop that process and run "
            "[accent]hexis up[/accent] again."
        )
        return False

    binary = _local_embedding_binary()
    if binary is None:
        binary = _install_local_embedding_binary()
    if binary is None:
        return False
    if not os.access(binary, os.X_OK):
        console.print(
            f"[warn]Embedding service binary is not executable: {binary}[/warn]\n"
            f"  Fix permissions, then run [accent]hexis up[/accent] again:\n"
            f"    [accent]chmod +x {binary}[/accent]"
        )
        return False

    _LOCAL_EMBEDDING_LOG.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[muted]Starting local embedding service: {binary}[/muted]")
    proc: subprocess.Popen[Any] | None = None
    launch_detail = str(binary)
    cache_existed_before_start = _local_embedding_cache_dir().exists()
    try:
        with _LOCAL_EMBEDDING_LOG.open("ab") as log_f:
            proc = subprocess.Popen(
                [str(binary)],
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
            process_ownership_recorded = _record_local_embedding_process(proc)
    except Exception as exc:
        console.print(
            f"[warn]Couldn't start local embedding service: {exc}[/warn]\n"
            f"  Try running it directly: [accent]{binary}[/accent]"
        )
        return False

    if not process_ownership_recorded:
        console.print(
            "[warn]The embedding service started, but Hexis could not save its "
            "process ownership record.[/warn]\n"
            "  Hexis will leave this process running during uninstall rather than "
            "risk stopping another application's service."
        )

    console.print(
        f"[muted]Waiting for embedding service on port {_LOCAL_EMBEDDING_PORT}...[/muted]"
    )
    deadline = _time.monotonic() + wait_seconds
    cache_ownership_warning_shown = False
    while _time.monotonic() < deadline:
        cache_ownership_recorded = _mark_local_embedding_cache_if_created(
            existed_before_start=cache_existed_before_start
        )
        if cache_ownership_recorded is False and not cache_ownership_warning_shown:
            console.print(
                "[warn]Hexis could not save ownership proof for the embedding model "
                "cache.[/warn]\n"
                "  The cache will be preserved during uninstall rather than risk "
                "deleting shared model data."
            )
            cache_ownership_warning_shown = True
        if _port_ready(_LOCAL_EMBEDDING_PORT):
            console.print(
                f"[ok]Embedding service is ready on port {_LOCAL_EMBEDDING_PORT}.[/ok]"
            )
            return True
        if proc is not None and proc.poll() is not None:
            _forget_local_embedding_process()
            tail = _tail_text(_LOCAL_EMBEDDING_LOG)
            tail_block = f"\n  Recent log:\n{tail}" if tail else ""
            console.print(
                f"[warn]Embedding service exited with code {proc.returncode}.[/warn]\n"
                f"  See log: [accent]{_LOCAL_EMBEDDING_LOG}[/accent]\n"
                f"  Or run directly: [accent]{binary}[/accent]"
                f"{tail_block}"
            )
            return False
        _time.sleep(0.5)

    cache_ownership_recorded = _mark_local_embedding_cache_if_created(
        existed_before_start=cache_existed_before_start
    )
    if cache_ownership_recorded is False and not cache_ownership_warning_shown:
        console.print(
            "[warn]Hexis could not save ownership proof for the embedding model "
            "cache.[/warn]\n"
            "  The cache will be preserved during uninstall rather than risk "
            "deleting shared model data."
        )
    tail = _tail_text(_LOCAL_EMBEDDING_LOG)
    tail_block = f"\n  Recent log:\n{tail}" if tail else ""
    console.print(
        f"[warn]Embedding service did not become ready within {int(wait_seconds)} seconds.[/warn]\n"
        f"  It may still be downloading/loading the model. See log: [accent]{_LOCAL_EMBEDDING_LOG}[/accent]\n"
        f"  Launcher: [accent]{launch_detail}[/accent]"
        f"{tail_block}"
    )
    return False


def _wait_port_ready(port: int, host: str = "127.0.0.1", overall: float = 45.0) -> bool:
    """Poll until the port accepts connections (or time out). Beats a fixed
    sleep before opening a browser at a cold Next.js build (Bar #4)."""
    import time as _time

    deadline = _time.monotonic() + overall
    while _time.monotonic() < deadline:
        if _port_ready(port, host):
            return True
        _time.sleep(0.4)
    return False


def _handle_ui(
    stack_root: Path,
    port: int,
    no_open: bool,
    instance: str | None = None,
) -> int:
    """Start the Next.js web dashboard."""
    import threading
    import time
    import urllib.error
    import urllib.request
    from core.browser import open_url
    from urllib.parse import urlparse

    ui_dir = stack_root / "hexis-ui"
    if not ui_dir.is_dir():
        _print_err(f"hexis-ui directory not found at {ui_dir}")
        return 1

    # Detect package manager
    runner = shutil.which("bun")
    pkg_cmd = "bun"
    if not runner:
        runner = shutil.which("npm")
        pkg_cmd = "npm"
    if not runner:
        _print_err("Neither bun nor npm found on PATH. Install one of them first.")
        return 1

    # Install deps if needed
    if not (ui_dir / "node_modules").is_dir():
        from apps.cli_theme import console

        console.print(f"[accent]Installing dependencies with {pkg_cmd}...[/accent]")
        rc = subprocess.run([runner, "install"], cwd=ui_dir).returncode
        if rc != 0:
            _print_err(f"{pkg_cmd} install failed (exit {rc})")
            return 1

    active_instance = instance or resolve_instance()
    dsn = db_dsn_from_env(active_instance) if active_instance else db_dsn_from_env()

    env_file = resolve_env_file(stack_root)
    try:
        if _uses_local_embedding_sidecar(env_file):
            _start_local_embedding_service()
        else:
            _warn_legacy_embedding_sidecar_port(env_file)
    except Exception:
        pass  # advisory only; init write routes surface embedding failures

    api_url = (
        os.getenv("HEXIS_API_URL")
        or os.getenv("HEXIS_API_BASE_URL")
        or "http://127.0.0.1:43817"
    )
    chat_url = f"http://localhost:{port}/chat"
    local_chat_url = f"http://127.0.0.1:{port}/chat"
    if _http_ready(local_chat_url) or _http_ready(chat_url):
        from apps.cli_theme import console

        console.print(f"[ok]Web dashboard is already running on port {port}.[/ok]")
        if not no_open:
            open_url(chat_url)
        return 0

    if _port_ready(port):
        listener = _port_listener_summary(port)
        detail = f" Listener: {listener}." if listener else ""
        _print_err(
            f"Port {port} is already in use, but Hexis did not get a dashboard response."
            f"{detail}\n"
            f"Stop that process, then run `hexis ui` again."
        )
        return 1

    def _api_healthcheck(url: str) -> bool:
        health_url = f"{url.rstrip('/')}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                return 200 <= int(getattr(resp, "status", 0)) < 300
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False

    api_proc: subprocess.Popen[Any] | None = None
    parsed_api = urlparse(api_url)
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    is_local_api = parsed_api.hostname in local_hosts or not parsed_api.hostname

    if is_local_api and not _api_healthcheck(api_url):
        from apps.cli_theme import console

        console.print("[accent]Starting local Hexis API for web chat...[/accent]")
        api_port = parsed_api.port or 43817
        api_cmd = [
            sys.executable,
            "-m",
            "apps.hexis_api",
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
        ]
        try:
            api_env = os.environ.copy()
            if active_instance:
                api_env["HEXIS_INSTANCE"] = active_instance
            api_env.setdefault("HEXIS_API_URL", api_url)
            api_env.setdefault("HEXIS_UI_URL", f"http://localhost:{port}/chat")
            api_proc = subprocess.Popen(
                api_cmd,
                cwd=stack_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=api_env,
            )
        except Exception as exc:
            _print_err(f"Failed to start Hexis API: {exc}")
            return 1

        # Wait briefly for API readiness.
        for _ in range(50):
            if _api_healthcheck(api_url):
                break
            if api_proc.poll() is not None:
                _print_err(
                    "Hexis API exited immediately. Run `hexis api` to inspect errors."
                )
                return 1
            time.sleep(0.2)
        else:
            if api_proc.poll() is None:
                api_proc.terminate()
                try:
                    api_proc.wait(timeout=2)
                except Exception:
                    api_proc.kill()
            _print_err("Timed out waiting for Hexis API. Run `hexis api` and retry.")
            return 1

    from apps.cli_theme import console

    console.print(f"\n[accent]Starting web dashboard on port {port}...[/accent]")
    console.print(
        "[muted]Dashboard runs in this terminal. Press Ctrl+C to stop it.[/muted]"
    )

    # Open the browser only once the server actually responds — not on a timer.
    if not no_open:

        def _open_browser():
            if _wait_port_ready(port):
                open_url(chat_url)

        t = threading.Thread(target=_open_browser, daemon=True)
        t.start()

    # Run dev server in foreground
    if pkg_cmd == "bun":
        dev_cmd = [runner, "run", "dev", "--port", str(port)]
    else:
        npx = shutil.which("npx") or "npx"
        dev_cmd = [npx, "next", "dev", "-p", str(port)]

    dev_env = os.environ.copy()
    dev_env["HEXIS_API_URL"] = api_url
    dev_env["HEXIS_UI_URL"] = chat_url
    dev_env["HEXIS_DATABASE_URL"] = dsn
    dev_env["DATABASE_URL"] = dsn
    if active_instance:
        dev_env["HEXIS_INSTANCE"] = active_instance

    try:
        result = subprocess.run(dev_cmd, cwd=ui_dir, env=dev_env)
        return result.returncode
    except KeyboardInterrupt:
        return 0
    finally:
        if api_proc and api_proc.poll() is None:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=2)
            except Exception:
                api_proc.kill()


async def _check_embedding_health(dsn: str, timeout: int = 20) -> None:
    """Probe the embedding service through the DB after stack start.

    Prints a warning with the configured URL if the service is unreachable.
    Never raises — this is advisory only.
    """
    import asyncpg as _apg

    try:
        conn = await asyncio.wait_for(_apg.connect(dsn), timeout=timeout)
    except Exception:
        return  # DB not ready yet — user will see it via doctor

    try:
        url = await conn.fetchval(
            "SELECT current_setting('app.embedding_service_url', true)"
        )
        healthy = await asyncio.wait_for(
            conn.fetchval("SELECT check_embedding_service_health()"),
            timeout=10,
        )
        if healthy:
            return

        from apps.cli_theme import console
        from core.cli_api import embedding_service_diagnosis

        svc_name, steps = embedding_service_diagnosis(url)

        console.print(
            f"[warn]Embedding service not reachable.[/warn] "
            f"Your config points to [bold]{svc_name}[/bold] ({url})\n"
            f"  but it is not responding. To fix:"
        )
        for step in steps:
            console.print(f"    [accent]{step}[/accent]")
        console.print(
            "\n  Or set [bold]EMBEDDING_SERVICE_URL[/bold] in .env to any compatible endpoint."
            "\n  Run [accent]hexis doctor[/accent] to re-check.\n"
        )
    except Exception:
        pass  # swallow — advisory only
    finally:
        await conn.close()


def _handle_ui_container(
    compose_cmd: list[str],
    compose_file: Path,
    stack_root: Path,
    env_file: Path | None,
    port: int,
    no_open: bool,
) -> int:
    """Start the UI via the containerized service (pip install path)."""
    import threading
    from core.browser import open_url

    from apps.cli_theme import console

    if _http_ready(f"http://127.0.0.1:{port}/chat") or _http_ready(
        f"http://localhost:{port}/chat"
    ):
        console.print(f"[ok]Web dashboard is already running on port {port}.[/ok]")
        if not no_open:
            open_url(f"http://localhost:{port}/chat")
        return 0
    if _port_ready(port):
        listener = _port_listener_summary(port)
        detail = f" Listener: {listener}." if listener else ""
        _print_err(
            f"Port {port} is already in use, but Hexis did not get a dashboard response."
            f"{detail}\n"
            f"Stop that process, then run `hexis ui` again."
        )
        return 1

    try:
        if _uses_local_embedding_sidecar(env_file):
            _start_local_embedding_service()
        else:
            _warn_legacy_embedding_sidecar_port(env_file)
    except Exception:
        pass  # advisory only; init write routes surface embedding failures

    console.print("[accent]Starting containerized web dashboard...[/accent]")
    console.print(
        "[muted]Dashboard runs in this terminal. Press Ctrl+C to stop it.[/muted]"
    )

    # The always-on loops come up with the dashboard and stay up after it
    # closes. Prefer installed host services and start Docker only for workers
    # that do not have a user-service owner.
    host_managed_workers = _host_managed_compose_workers()
    host_workers_ok = True
    if host_managed_workers:
        host_workers_ok, _host_error = _ensure_installed_host_services_running()
    docker_worker_targets = [
        name
        for name in ("heartbeat_worker", "maintenance_worker")
        if name not in host_managed_workers
    ]
    docker_workers_ok = True
    if docker_worker_targets:
        docker_workers_ok = (
            run_compose(
                compose_cmd,
                compose_file,
                stack_root,
                ["up", "-d", *docker_worker_targets],
                env_file,
            )
            == 0
        )
    if not host_workers_ok or not docker_workers_ok:
        console.print(
            "[warn]⚠ Could not start the heartbeat and maintenance workers[/warn] "
            "[muted]— the dashboard still works, but the agent will not act on its "
            "own until `hexis start` succeeds.[/muted]"
        )

    # Bring up both the UI and the canonical Python API (hexis-api).
    # The Next.js BFF proxies chat + consent to hexis-api.
    # Open the browser only once the server actually responds — not on a timer.
    if not no_open:

        def _open_browser():
            if _wait_port_ready(port):
                open_url(f"http://localhost:{port}")

        t = threading.Thread(target=_open_browser, daemon=True)
        t.start()

    try:
        return run_compose(
            compose_cmd, compose_file, stack_root, ["up", "api", "ui"], env_file
        )
    except KeyboardInterrupt:
        return 0
    finally:
        run_compose(
            compose_cmd, compose_file, stack_root, ["stop", "ui", "api"], env_file
        )


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _capture_path(command: list[str]) -> Path | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (result.stdout or "").strip()
    if result.returncode != 0 or not value:
        return None
    return Path(value).expanduser().resolve()


def _find_uninstall_program(name: str) -> str | None:
    resolved = shutil.which(name)
    if resolved:
        return resolved

    candidates: list[Path] = []
    if name == "uv" and os.getenv("UV_INSTALL_DIR"):
        candidates.append(Path(os.environ["UV_INSTALL_DIR"]) / "uv")
    if os.getenv("XDG_BIN_HOME"):
        candidates.append(Path(os.environ["XDG_BIN_HOME"]) / name)
    candidates.append(Path.home() / ".local" / "bin" / name)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _package_uninstall_command() -> tuple[list[str], str]:
    """Derive the owning tool from the running environment.

    Calling pip inside a uv-tool or pipx environment would leave a broken tool
    wrapper behind, so compare ``sys.prefix`` to each manager's live tool root
    before falling back to this interpreter's pip.
    """
    prefix = Path(sys.prefix).resolve()

    uv = _find_uninstall_program("uv")
    if uv:
        uv_tool_root = _capture_path([uv, "tool", "dir"])
        if uv_tool_root and _path_is_within(prefix, uv_tool_root / "hexis"):
            return [uv, "tool", "uninstall", "hexis"], "uv"

    pipx = _find_uninstall_program("pipx")
    if pipx:
        pipx_root = _capture_path([pipx, "environment", "--value", "PIPX_LOCAL_VENVS"])
        if pipx_root and _path_is_within(prefix, pipx_root / "hexis"):
            return [pipx, "uninstall", "hexis"], "pipx"

    return [sys.executable, "-m", "pip", "uninstall", "--yes", "hexis"], "pip"


def _hexis_data_dir() -> Path:
    from core.config import hexis_home

    return hexis_home().expanduser()


def _purge_hexis_data_dir(path: Path) -> tuple[bool, str | None]:
    """Remove the explicitly selected Hexis data directory with guard rails."""
    if not path.exists() and not path.is_symlink():
        return True, None
    if path.is_symlink():
        return False, (
            f"Refusing to purge symlinked Hexis data directory: {path}. "
            "Remove that link or its target yourself after checking it."
        )

    resolved = path.resolve()
    home = Path.home().resolve()
    if resolved in {Path("/"), home} or len(resolved.parts) < 3:
        return False, f"Refusing unsafe Hexis data path: {resolved}"
    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        return False, f"Could not delete Hexis data at {resolved}: {exc}"
    return True, None


def _confirm_uninstall(*, purge: bool, cli_only: bool, data_dir: Path) -> bool:
    print("Hexis uninstall\n")
    print("This will:")
    if cli_only:
        print("  - remove the Hexis CLI installation")
        print("  - leave all Docker containers, images, and volumes untouched")
    else:
        print("  - stop and remove Hexis containers and their network")
        print("  - remove Hexis Docker images")
        print("  - remove the Hexis CLI installation")
    if purge:
        print("  - PERMANENTLY DELETE the brain database volumes")
        print(
            f"  - PERMANENTLY DELETE Hexis config, credentials, and backups at {data_dir}"
        )
        print("  - delete embeddinggemma assets that Hexis can prove it created")
        phrase = "uninstall and delete data"
        print(
            "\nThe default backup directory is inside the data being deleted. "
            "If you may want this agent again, abort and run "
            "`hexis backup --output <directory-outside-the-Hexis-data-dir>` first."
        )
    else:
        print("\nYour brain database volumes and Hexis config will be preserved.")
        print("Reinstall Hexis and run `hexis up` to use them again.")
        phrase = "uninstall"
    try:
        answer = input(f"\nType '{phrase}' to confirm: ")
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        return False
    if answer.strip().lower() != phrase:
        print("Aborted.")
        return False
    return True


def _uninstall(
    *,
    compose_file: Path | None,
    stack_root: Path,
    env_file: Path | None,
    is_source: bool,
    purge: bool,
    cli_only: bool,
    yes: bool,
) -> int:
    data_dir = _hexis_data_dir()
    uninstall_command, package_manager = _package_uninstall_command()

    if not yes and not _confirm_uninstall(
        purge=purge, cli_only=cli_only, data_dir=data_dir
    ):
        return 1

    sidecar_note: str | None = None
    voice_note: str | None = None
    embedding_cleanup_notes: list[str] = []
    removed_host_services: list[str] = []
    if not cli_only:
        if compose_file is None:
            _print_err(
                "Cannot find Hexis's Docker Compose file, so no software was removed. "
                "Reinstall Hexis and retry, or use `hexis uninstall --cli-only` to "
                "remove only the CLI while leaving Docker resources untouched."
            )
            return 1
        try:
            docker_bin = ensure_docker()
            compose_cmd = ensure_compose(docker_bin)
        except SystemExit:
            _print_err(
                "No software was removed. Start Docker and retry `hexis uninstall`, "
                "or run `hexis uninstall --cli-only` if you intentionally want to "
                "leave Docker resources untouched."
            )
            return 1

        compose_args = ["down", "--remove-orphans", "--rmi", "all"]
        if purge:
            compose_args.append("--volumes")
        print("Removing Hexis Docker resources...")
        rc = run_compose(compose_cmd, compose_file, stack_root, compose_args, env_file)
        if rc != 0:
            _print_err(
                "Docker cleanup failed, so the CLI was kept. Fix the Docker error "
                "above and rerun `hexis uninstall`."
            )
            return rc

        stopped_sidecar, sidecar_note = _stop_owned_local_embedding_service()
        if stopped_sidecar:
            print("Stopped the embedding service started by Hexis.")

    voice_ok, voice_stopped, voice_note = _stop_owned_voice_sidecar()
    if not voice_ok:
        _print_err(
            f"{voice_note} The CLI was kept so the owned process can still be "
            "inspected; run `hexis voice status`, resolve the error, and retry."
        )
        return 1
    if voice_stopped:
        print("Stopped the local voice process started by Hexis.")

    # User services execute the installed Python package. Remove those managed
    # references before uninstalling the package so launchd/systemd cannot keep
    # retrying an executable that no longer exists. Logs remain unless --purge
    # explicitly removes the Hexis data directory below.
    try:
        from core.host_services import installed_host_services, uninstall_host_services

        installed = installed_host_services()
        if installed:
            result = uninstall_host_services(installed)
            removed_host_services = list(result["uninstalled"])
            print(
                "Removed Hexis host services: " + ", ".join(removed_host_services) + "."
            )
    except Exception as exc:
        _print_err(
            "Hexis could not remove its host services, so the CLI and data were kept. "
            f"{exc} Run `hexis service status`, resolve the provider error, and retry."
        )
        return 1

    if purge:
        if _port_ready(_LOCAL_EMBEDDING_PORT):
            embedding_cleanup_notes.append(
                "Owned embedding assets were left in place because an unowned "
                f"service is still using port {_LOCAL_EMBEDDING_PORT}."
            )
        else:
            removed_assets, asset_notes = _purge_owned_local_embedding_assets()
            embedding_cleanup_notes.extend(asset_notes)
            for asset in removed_assets:
                print(f"Removed Hexis-created embedding asset: {asset}")
        removed, error = _purge_hexis_data_dir(data_dir)
        if not removed:
            _print_err(f"{error} The CLI was kept so you can retry safely.")
            return 1

    source_note = (
        f" The source checkout at {stack_root} was left in place." if is_source else ""
    )
    embedding_binary_retained = _local_embedding_binary() is not None
    print(f"Uninstalling Hexis with {package_manager}...", flush=True)
    rc = subprocess.run(uninstall_command).returncode
    if rc != 0:
        _print_err(
            f"The stack was removed, but {package_manager} could not uninstall the "
            f"CLI (exit {rc}). Retry manually with: {' '.join(uninstall_command)}"
        )
        return rc

    print("\nHexis has been uninstalled." + source_note)
    if purge:
        print("The brain database volumes and Hexis data directory were deleted.")
    elif cli_only:
        print("Docker resources and all Hexis data were left untouched, as requested.")
    else:
        print(
            f"Your brain database volumes and Hexis config at {data_dir} were preserved."
        )
        print("Reinstall Hexis and run `hexis up` to restore the agent.")
    if sidecar_note:
        print(f"\nNote: {sidecar_note}")
    if voice_note:
        print(f"\nNote: {voice_note}")
    if removed_host_services and not purge:
        print("Host-service logs were preserved under ~/.hexis/logs/host-services.")
    for note in embedding_cleanup_notes:
        print(f"Note: {note}")
    if embedding_binary_retained:
        if purge:
            print(
                "An embeddinggemma binary remains because Hexis could not prove it "
                "was safe to delete."
            )
        else:
            print(
                "The standalone embeddinggemma binary and model cache were preserved; "
                "`hexis uninstall --purge` removes assets Hexis can prove it created."
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Top-level guard: turn stack-down / unhandled errors into one actionable
    line instead of a raw traceback (Experience Bar #8, #4)."""
    try:
        return _dispatch(argv)
    except KeyboardInterrupt:
        _print_err("Aborted.")
        return 130
    except Exception as e:  # noqa: BLE001
        low = str(e).lower()
        if any(
            s in low
            for s in (
                "postgres",
                "connection refused",
                "connect call failed",
                "failed to connect",
                "timed out connecting",
                "timeouterror",
            )
        ) or isinstance(e, (ConnectionError, TimeoutError)):
            _print_err(
                "Can't reach the database. Is the stack running? "
                "Try `hexis up`, then `hexis doctor`."
            )
        else:
            _print_err(f"Error: {e}")
        return 1


def _dispatch(argv: list[str] | None = None) -> int:
    load_dotenv()

    # Some commands intentionally forward argv to another module (init/chat/ingest/etc).
    # `argparse` does not accept unknown `--flags` as positional passthrough unless the
    # user adds `--`, which is not the UX we want. Do a small, predictable pre-parse
    # here so commands like `hexis init --character hexis ...` work.
    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    forward_map = {
        "chat": "apps.cli_chat",
        "init": "apps.hexis_init",
        "ingest": "services.ingest",
        "mcp": "apps.hexis_mcp_server",
        "worker": "apps.worker",
    }

    # Parse a minimal set of global flags that must apply to forwarded commands.
    instance: str | None = None
    global_help = False
    i = 0
    while i < len(raw_argv):
        tok = raw_argv[i]
        if tok in {"-h", "--help"}:
            global_help = True
            i += 1
            continue
        if tok in {"-V", "--version"}:
            sys.stdout.write(f"hexis {_ver}\n")
            return 0
        if tok in {"-i", "--instance"}:
            if i + 1 >= len(raw_argv):
                break  # let argparse handle the error
            instance = raw_argv[i + 1]
            i += 2
            continue
        if tok.startswith("--instance="):
            instance = tok.split("=", 1)[1]
            i += 1
            continue
        if tok == "--":
            i += 1
            break
        if tok.startswith("-"):
            break  # unknown global flag; defer to argparse
        break

    cmd_argv = raw_argv[i:]
    if not cmd_argv:
        # No command; mirror legacy behavior: show grouped help.
        _print_grouped_help()
        return 0

    if global_help:
        _print_grouped_help()
        return 0

    cmd = cmd_argv[0]
    if cmd in forward_map:
        if instance:
            os.environ["HEXIS_INSTANCE"] = instance

        fwd_argv = cmd_argv[1:]

        # chat and init use line-based CLIs (apps.cli_chat / apps.hexis_init) —
        # keyboard-first, native terminal, Ctrl+C exits. They are reached via the
        # forward_map below.

        if cmd == "ingest":
            # UX/backwards-compat: accept `hexis ingest --file foo.md` by auto-inserting
            # the `ingest` subcommand when the user passed flags.
            if fwd_argv and fwd_argv[0] == "--":
                fwd_argv = fwd_argv[1:]
            if fwd_argv and fwd_argv[0] not in {
                "ingest",
                "status",
                "process",
                "backfill-chunks",
                "-h",
                "--help",
            }:
                fwd_argv = ["ingest", *fwd_argv]

        return _run_module(forward_map[cmd], fwd_argv)

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    # Show grouped help for: no command, --help/-h, or 'help' with no subcommand
    if args.command is None or args.help:
        _print_grouped_help()
        return 0

    func = getattr(args, "func", None)

    if func == "help":
        if args.help_command:
            choices = parser._subcommands.choices  # type: ignore[attr-defined]
            if args.help_command in choices:
                choices[args.help_command].print_help()
            else:
                _print_err(f"Unknown command: {args.help_command}\n")
                _print_grouped_help()
        else:
            _print_grouped_help()
        return 0

    # Set HEXIS_INSTANCE env var if --instance flag is used
    # This ensures subprocesses also use the correct instance
    if args.instance:
        os.environ["HEXIS_INSTANCE"] = args.instance

    compose_file, is_source = _find_compose_file()
    stack_root = _stack_root_from_compose(compose_file) if compose_file else Path.cwd()
    env_file = resolve_env_file(stack_root)

    # Instance management commands (don't need docker)
    if func == "instance":
        # Default: 'hexis instance' → list
        return _instance_list(False)
    if func == "instance_create":
        return asyncio.run(_instance_create(args.name, args.description))
    if func == "instance_list":
        return _instance_list(args.json)
    if func == "instance_use":
        return _instance_use(args.name)
    if func == "instance_current":
        return _instance_current()
    if func == "instance_delete":
        return asyncio.run(_instance_delete(args.name, args.force, args.reason))
    if func == "instance_clone":
        return asyncio.run(_instance_clone(args.source, args.target, args.description))
    if func == "instance_import":
        return asyncio.run(_instance_import(args.name, args.database, args.description))

    # Filing cabinet + desk commands (DB-backed; don't need docker)
    if func in {
        "docs",
        "docs_search",
        "docs_open",
        "docs_info",
        "docs_load",
        "desk",
        "desk_list",
        "desk_open",
        "desk_search",
        "desk_pin",
        "desk_unpin",
        "desk_clear",
    }:
        from apps import cli_docs

        dsn = _get_dsn(args)
        if func in {"docs", "docs_search"}:
            if func == "docs":
                # Bare `hexis docs` → browse help with the next step.
                parser._subcommands.choices["docs"].print_help()  # type: ignore[attr-defined]
                return 0
            return asyncio.run(cli_docs.docs_search(dsn, args))
        if func == "docs_open":
            return asyncio.run(cli_docs.docs_open(dsn, args))
        if func == "docs_info":
            return asyncio.run(cli_docs.docs_info(dsn, args))
        if func == "docs_load":
            return asyncio.run(cli_docs.docs_load(dsn, args))
        if func == "desk":
            return asyncio.run(
                cli_docs.desk_list(
                    dsn, argparse.Namespace(limit=50, pinned=False, json=False)
                )
            )
        if func == "desk_list":
            return asyncio.run(cli_docs.desk_list(dsn, args))
        if func == "desk_open":
            return asyncio.run(cli_docs.desk_open(dsn, args))
        if func == "desk_search":
            return asyncio.run(cli_docs.desk_search(dsn, args))
        if func == "desk_pin":
            return asyncio.run(cli_docs.desk_pin(dsn, args, pinned=True))
        if func == "desk_unpin":
            return asyncio.run(cli_docs.desk_pin(dsn, args, pinned=False))
        if func == "desk_clear":
            return asyncio.run(cli_docs.desk_clear(dsn, args))

    # Consent management commands (DB-backed; don't need docker)
    if func == "consents":
        return asyncio.run(_consents_list(_get_dsn(args), False))  # default to list
    if func == "consents_list":
        return asyncio.run(_consents_list(_get_dsn(args), args.json))
    if func == "consents_show":
        return asyncio.run(_consents_show(_get_dsn(args), args.model))
    if func == "consents_request":
        return _consents_request(args.model)
    if func == "consents_revoke":
        return asyncio.run(_consents_revoke(_get_dsn(args), args.model, args.reason))

    # Resource request decisions (DB-backed; don't need docker)
    if func in {"requests", "requests_list"}:
        return asyncio.run(
            _requests_list(
                _get_dsn(args),
                getattr(args, "status", None),
                getattr(args, "json", False),
            )
        )
    if func == "requests_grant":
        return asyncio.run(
            _requests_decide(_get_dsn(args), args.id, "granted", args.note, None)
        )
    if func == "requests_deny":
        return asyncio.run(
            _requests_decide(_get_dsn(args), args.id, "denied", args.note, None)
        )
    if func == "requests_modify":
        return asyncio.run(
            _requests_decide(_get_dsn(args), args.id, "modified", args.note, args.value)
        )

    # Character card management (don't need docker, except export)
    if func == "characters":
        return _characters_list(False)
    if func == "characters_list":
        return _characters_list(args.json)
    if func == "characters_show":
        return _characters_show(args.name)
    if func == "characters_create":
        return _characters_create(args)
    if func == "characters_import":
        return _characters_import(args.path)
    if func == "characters_export":
        dsn = _get_dsn(args)
        return asyncio.run(_characters_export(dsn, args.name, args.output))

    if func in {"chat_sessions", "chat_sessions_list"}:
        return asyncio.run(
            _chat_sessions_list(
                _get_dsn(args),
                getattr(args, "limit", 20),
                getattr(args, "surface", None),
                getattr(args, "status", "active"),
                getattr(args, "json", False),
            )
        )
    if func == "chat_sessions_show":
        return asyncio.run(
            _chat_sessions_show(
                _get_dsn(args),
                args.session_id,
                args.visible_only,
                args.json,
            )
        )
    if func == "chat_sessions_export":
        return asyncio.run(
            _chat_sessions_export(
                _get_dsn(args),
                args.session_id,
                args.format,
                args.output,
                args.visible_only,
            )
        )
    if func == "chat_sessions_title":
        return asyncio.run(
            _chat_sessions_title(
                _get_dsn(args),
                args.session_id,
                args.title,
                args.json,
            )
        )
    if func == "chat_sessions_fork":
        return asyncio.run(
            _chat_sessions_fork(
                _get_dsn(args),
                args.session_id,
                args.until_ordinal,
                args.title,
                args.json,
            )
        )
    if func == "chat_sessions_clone":
        return asyncio.run(
            _chat_sessions_fork(
                _get_dsn(args),
                args.session_id,
                None,
                args.title,
                args.json,
            )
        )

    if func in {"hmx_export", "hmx_import", "hmx_review"}:
        from apps.cli_exchange import run_export, run_import, run_review

        handler = {
            "hmx_export": run_export,
            "hmx_import": run_import,
            "hmx_review": run_review,
        }[func]
        return asyncio.run(handler(_get_dsn(args), args))

    if func == "uninstall":
        return _uninstall(
            compose_file=compose_file,
            stack_root=stack_root,
            env_file=env_file,
            is_source=is_source,
            purge=args.purge,
            cli_only=args.cli_only,
            yes=args.yes,
        )

    if isinstance(func, str) and func.startswith("service_"):
        return _handle_host_service_command(
            args,
            compose_file=compose_file,
            stack_root=stack_root,
            env_file=env_file,
        )

    if isinstance(func, str) and func.startswith("tunnel_"):
        return _handle_tunnel_command(args, env_file=env_file)

    if isinstance(func, str) and func.startswith("voice_"):
        return _handle_voice_command(args)

    host_managed_workers = _host_managed_compose_workers()
    docker_cmds = {
        "up",
        "dev",
        "down",
        "ps",
        "logs",
        "start",
        "stop",
        "reset",
        "upgrade",
    }
    if {"heartbeat_worker", "maintenance_worker"} <= host_managed_workers:
        docker_cmds -= {"start", "stop"}
    docker_bin: str | None = None
    compose_cmd: list[str] | None = None
    if func in docker_cmds:
        if compose_file is None:
            _print_err("docker-compose.yml not found.")
            return 1
        docker_bin = ensure_docker()
        compose_cmd = ensure_compose(docker_bin)

    if func == "up":
        if args.build and not is_source:
            _print_err(
                "`hexis up --build` needs a source checkout. This packaged install "
                "uses published images; run `hexis upgrade` to refresh them."
            )
            return 1
        compose_services: list[str] | None = None
        if host_managed_workers:
            compose_services = _configured_compose_services(
                compose_cmd or [],
                compose_file,
                stack_root,
                env_file,
                args.profile,
            )
            if compose_services is None:
                _print_err(
                    "Could not derive the live Compose service list, so Hexis refused "
                    "to risk starting duplicate Docker workers. Run `hexis service stop` "
                    "to use the Docker-only path, or fix the Compose error and retry."
                )
                return 1
            compose_services = [
                name for name in compose_services if name not in host_managed_workers
            ]
        if not is_source:
            from apps.cli_theme import console

            console.print("[accent]Pulling Docker images...[/accent]")
            pull_args = ["pull", *(compose_services or [])]
            pull_rc = run_compose(
                compose_cmd or [], compose_file, stack_root, pull_args, env_file
            )
            if pull_rc != 0:
                console.print(
                    "[warn]⚠ Image pull failed[/warn] — check your network, `docker login`, "
                    "or a registry rate limit. Continuing with cached images; if `up` reports a "
                    "missing image, that's the cause."
                )
        up_args = _up_compose_args(
            args.profile,
            is_source=is_source,
            build=bool(args.build),
            services=compose_services,
        )
        rc = run_compose(compose_cmd or [], compose_file, stack_root, up_args, env_file)
        if rc == 0:
            from apps.cli_theme import console

            console.print("\n[ok]Stack is starting.[/ok]\n")
            if host_managed_workers:
                workers_ok, workers_error = _ensure_installed_host_services_running()
                if not workers_ok:
                    _print_err(
                        "The Docker services started, but Hexis could not start the "
                        f"installed host workers: {workers_error} Run `hexis service "
                        "status` and `hexis service logs`, then retry `hexis up`."
                    )
                    return 1
                console.print(
                    "  [ok]Background workers[/ok] Running as user-owned host services"
                )
            else:
                console.print(
                    "  [ok]Background workers[/ok] Heartbeat and memory maintenance run by default"
                )

            # Start the standalone embedding sidecar before probing DB health.
            embedding_probe_allowed = True
            try:
                if _uses_local_embedding_sidecar(env_file):
                    embedding_probe_allowed = _start_local_embedding_service()
                else:
                    _warn_legacy_embedding_sidecar_port(env_file)
            except Exception:
                embedding_probe_allowed = False

            # Advisory embedding health check (waits for DB, probes embedding URL)
            if embedding_probe_allowed:
                try:
                    dsn = db_dsn_from_env()
                    asyncio.run(_check_embedding_health(dsn))
                except Exception:
                    pass  # never block startup

            # Bring the schema up to date without touching data (advisory-locked,
            # no-op if already current). The workers/API also run this on startup.
            try:
                from core.agent_api import apply_migrations

                applied = asyncio.run(apply_migrations(db_dsn_from_env()))
                if applied:
                    console.print(
                        f"[ok]Applied {len(applied)} schema migration(s) — no data lost.[/ok]"
                    )
            except Exception:
                pass  # never block startup

            voice_started, voice_note = _start_configured_voice_sidecar()
            if voice_started:
                console.print("  [ok]Local speech[/ok] Piper sidecar is ready")
            if voice_note:
                console.print(f"  [warn]⚠ Local speech[/warn] {voice_note}")

            # A fresh agent must be configured before chat/ui are useful — lead with it.
            console.print(
                "  [accent]hexis init[/accent]   Configure the agent (start here)"
            )
            console.print("  [accent]hexis chat[/accent]   Chat in the terminal")
            console.print("  [accent]hexis ui[/accent]     Open the web dashboard")
            console.print()
        elif is_source and not args.build:
            _print_err(
                "Hexis did not start; no source build was attempted. The Docker "
                "Compose output above has the cause. If an image could not be pulled, "
                "check the network or registry login and run `hexis up` again. To "
                "deliberately build this checkout instead, run `hexis up --build`."
            )
        return rc
    if func == "dev":
        from apps.cli_theme import console

        host_status = _host_service_status_if_installed()
        active_host_services = [
            str(item.get("name"))
            for item in (host_status or {}).get("services", [])
            if isinstance(item, dict) and item.get("active")
        ]
        if active_host_services:
            _print_err(
                "Host workers are already running: "
                f"{', '.join(active_host_services)}. Watch mode runs Docker workers, "
                "so stop the host copies with `hexis service stop` before `hexis dev`."
            )
            return 1

        if not is_source:
            _print_err(
                "`hexis dev` needs a source checkout — it watches the repo and rebuilds "
                "from ops/Dockerfile.*. In a packaged install, use `hexis upgrade` to "
                "pick up new images."
            )
            return 1
        compose_extra = []
        for profile in args.profile:
            compose_extra += ["--profile", profile]
        console.print("[accent]Building and starting the stack...[/accent]")
        rc = run_compose(
            compose_cmd or [],
            compose_file,
            stack_root,
            compose_extra + ["up", "-d", "--build"],
            env_file,
        )
        if rc != 0:
            return rc
        try:
            if _uses_local_embedding_sidecar(env_file):
                _start_local_embedding_service()
            else:
                _warn_legacy_embedding_sidecar_port(env_file)
        except Exception:
            pass
        try:
            from core.agent_api import apply_migrations

            applied = asyncio.run(apply_migrations(db_dsn_from_env()))
            if applied:
                console.print(
                    f"[ok]Applied {len(applied)} schema migration(s) — no data lost.[/ok]"
                )
        except Exception:
            pass  # workers also migrate on startup; never block dev mode
        voice_started, voice_note = _start_configured_voice_sidecar()
        if voice_started:
            console.print("[ok]Local speech sidecar is ready.[/ok]")
        if voice_note:
            console.print(f"[warn]⚠ Local speech:[/warn] {voice_note}")
        console.print("\n[ok]Stack is running in watch mode.[/ok]")
        console.print(
            "  Edits under core/ services/ apps/ channels/ plugins/ skills/ and db/"
        )
        console.print(
            "  sync into the running containers and restart them; new db/migrations"
        )
        console.print(
            "  apply on that restart. pyproject.toml changes trigger a rebuild."
        )
        console.print("  [dim]Ctrl+C stops watching — the stack keeps running.[/dim]\n")
        try:
            rc = run_compose(
                compose_cmd or [],
                compose_file,
                stack_root,
                compose_extra + ["watch", "--no-up"],
                env_file,
            )
        except KeyboardInterrupt:
            rc = 130
        if rc == 130:  # normal Ctrl+C exit from the watch session
            rc = 0
        console.print(
            "\n[dim]Stopped watching. The stack is still running; use "
            "`hexis up --build` for a later one-time rebuild.[/dim]"
        )
        return rc
    if func == "down":
        if host_managed_workers:
            workers_ok, workers_error = _stop_installed_host_services()
            if not workers_ok:
                _print_err(
                    "Hexis did not stop the host workers, so it left the Docker stack "
                    f"running too: {workers_error} Run `hexis service status` and retry."
                )
                return 1
        rc = run_compose(
            compose_cmd or [], compose_file, stack_root, ["down"], env_file
        )
        if rc == 0:
            stopped_sidecar, sidecar_note = _stop_owned_local_embedding_service()
            if stopped_sidecar:
                print("Stopped the embedding service started by Hexis.")
            if sidecar_note:
                print(f"Note: {sidecar_note}")
            voice_ok, voice_stopped, voice_note = _stop_owned_voice_sidecar()
            if voice_stopped:
                print("Stopped the local voice process started by Hexis.")
            if voice_note:
                print(f"Note: {voice_note}")
            if not voice_ok:
                return 1
        return rc
    if func == "reset":
        from apps.cli_theme import console

        if not args.yes:
            console.print(
                "[bold red]WARNING:[/bold red] This will destroy ALL data "
                "(memories, goals, worldview, identity) and re-initialize the database from scratch."
            )
            try:
                answer = input("Type 'reset' to confirm: ")
            except (KeyboardInterrupt, EOFError):
                print()
                return 1
            if answer.strip().lower() != "reset":
                console.print("[dim]Aborted.[/dim]")
                return 1
        compose_services: list[str] | None = None
        if host_managed_workers:
            compose_services = _configured_compose_services(
                compose_cmd or [], compose_file, stack_root, env_file
            )
            if compose_services is None:
                _print_err(
                    "Could not derive the live Compose service list, so Hexis refused "
                    "to risk recreating duplicate Docker workers during reset. Fix the "
                    "Compose error and retry."
                )
                return 1
            compose_services = [
                name for name in compose_services if name not in host_managed_workers
            ]
            workers_ok, workers_error = _stop_installed_host_services()
            if not workers_ok:
                _print_err(
                    "The database was not reset because installed host workers could not "
                    f"be stopped: {workers_error} Run `hexis service status` and retry."
                )
                return 1
        console.print("[accent]Stopping containers and removing volumes...[/accent]")
        rc = run_compose(
            compose_cmd or [], compose_file, stack_root, ["down", "-v"], env_file
        )
        if rc != 0:
            if host_managed_workers:
                workers_ok, workers_error = _ensure_installed_host_services_running()
                recovery = (
                    "The previously running host workers were restored."
                    if workers_ok
                    else f"Host workers also need attention: {workers_error}"
                )
                _print_err(
                    "The reset did not remove the database because Compose could not "
                    f"stop the stack. {recovery} Resolve the error above and retry."
                )
            return rc
        if is_source:
            console.print("[accent]Rebuilding database image...[/accent]")
            rc = run_compose(
                compose_cmd or [], compose_file, stack_root, ["build", "db"], env_file
            )
            if rc != 0:
                if host_managed_workers:
                    _print_err(
                        "The old database volume was removed, but its replacement image "
                        "did not build. Host workers remain stopped. Fix the build error, "
                        "then run `hexis up --build`."
                    )
                return rc
        else:
            console.print("[accent]Pulling images...[/accent]")
            pull_args = ["pull", *(compose_services or [])]
            rc = run_compose(
                compose_cmd or [], compose_file, stack_root, pull_args, env_file
            )
            if rc != 0:
                _print_err(
                    "The database was removed, but replacement images could not be pulled. "
                    "Host workers remain stopped. Fix the image error, then run `hexis up`."
                )
                return rc
        console.print("[accent]Starting services...[/accent]")
        up_args = ["up", "-d", *(compose_services or [])]
        rc = run_compose(compose_cmd or [], compose_file, stack_root, up_args, env_file)
        if rc == 0:
            if host_managed_workers:
                workers_ok, workers_error = _ensure_installed_host_services_running()
                if not workers_ok:
                    _print_err(
                        "The database reset completed, but host workers did not restart: "
                        f"{workers_error} Run `hexis service logs`, fix the cause, then "
                        "run `hexis service start`."
                    )
                    return 1
            console.print(
                "\n[ok]Database reset complete.[/ok] Run [accent]hexis init[/accent] to reconfigure the agent.\n"
            )
        elif host_managed_workers:
            _print_err(
                "The database reset stopped before the stack restarted. Host workers "
                "remain stopped; fix the Compose error above, then run `hexis up`."
            )
        return rc
    if func == "upgrade":
        # The non-destructive counterpart to `reset`: refresh images + code and
        # migrate the schema, WITHOUT removing the data volume.
        from apps.cli_theme import console

        if not is_source and not args.no_self_update:
            # Packaged install: images are pinned to the CLI's own version
            # (see _compose_env), so the package must move first — then the
            # fresh CLI pulls the images published from its release commit.
            latest = _pypi_latest()
            if latest and not _is_newer(latest, _ver):
                console.print(f"[ok]hexis {_ver} is the newest release.[/ok]")
            else:
                installer = _installed_via()
                console.print(
                    f"[accent]Updating the hexis package[/accent] (installed via {installer})..."
                )
                new_ver = _run_self_update(console, installer)
                if new_ver != _ver:
                    console.print(
                        f"[ok]hexis {_ver} → {new_ver}[/ok] — handing off to the new version..."
                    )
                    rerun = subprocess.run(
                        [sys.executable, sys.argv[0], "upgrade", "--no-self-update"],
                    )
                    return rerun.returncode
                if latest:
                    # A newer release exists and the package did not move.
                    # Pulling images now would re-install the OLD version's
                    # stack while printing success — fail loud with the exact
                    # way out instead.
                    console.print(
                        f"[fail]✗ hexis is still {_ver}; {latest} is available "
                        f"but the self-update did not take effect.[/fail]\n"
                        f"Run [accent]{_self_update_hint(installer)}[/accent] and then "
                        f"[accent]hexis upgrade[/accent] again."
                    )
                    return 1
                console.print(
                    f"[warn]⚠ Could not check PyPI for a newer release — "
                    f"continuing with hexis {_ver}.[/warn]"
                )
        console.print("[accent]Updating the stack (your data is preserved)...[/accent]")
        compose_services: list[str] | None = None
        if host_managed_workers:
            compose_services = _configured_compose_services(
                compose_cmd or [], compose_file, stack_root, env_file
            )
            if compose_services is None:
                _print_err(
                    "Could not derive the live Compose service list, so Hexis refused "
                    "to risk starting duplicate Docker workers during upgrade. Fix the "
                    "Compose error and retry."
                )
                return 1
            compose_services = [
                name for name in compose_services if name not in host_managed_workers
            ]
        if is_source:
            refresh_rc = run_compose(
                compose_cmd or [],
                compose_file,
                stack_root,
                ["build", *(compose_services or [])],
                env_file,
            )
        else:
            refresh_rc = run_compose(
                compose_cmd or [],
                compose_file,
                stack_root,
                ["pull", *(compose_services or [])],
                env_file,
            )
        if refresh_rc != 0:
            _print_err(
                "Hexis did not change the running stack because its replacement "
                "images could not be prepared. Resolve the error above and retry."
            )
            return refresh_rc
        rc = run_compose(
            compose_cmd or [],
            compose_file,
            stack_root,
            ["up", "-d", *(compose_services or [])],
            env_file,
        )
        if rc != 0:
            return rc
        console.print("[accent]Applying schema migrations...[/accent]")
        mrc = asyncio.run(_migrate(_get_dsn(args), status_only=False))
        if mrc == 0:
            if host_managed_workers:
                workers_ok, workers_error = _restart_installed_host_services()
                if not workers_ok:
                    _print_err(
                        "The stack and schema were upgraded, but installed host workers "
                        f"did not restart onto the new code: {workers_error} Run `hexis "
                        "service logs`, fix the cause, then run `hexis service restart`."
                    )
                    return 1
            console.print("\n[ok]Upgrade complete — data preserved.[/ok]\n")
        return mrc
    if func == "ps":
        return run_compose(
            compose_cmd or [], compose_file, stack_root, ["ps"], env_file
        )
    if func == "logs":
        log_args = ["logs"] + (["-f"] if args.follow else []) + args.services
        return run_compose(
            compose_cmd or [], compose_file, stack_root, log_args, env_file
        )
    if func == "chat":
        fwd_argv = list(args.args or [])
        return _run_module("apps.cli_chat", fwd_argv)
    if func == "ingest":
        argv = list(args.args or [])
        # `hexis ingest` forwards args to `python -m services.ingest`.
        #
        # The ingestion module uses subcommands: `ingest|status|process`.
        # For UX/backwards-compat, accept `hexis ingest --file foo.md` by
        # auto-inserting the `ingest` subcommand when the user passed flags.
        if argv and argv[0] == "--":
            argv = argv[1:]
        if argv and argv[0] not in {"ingest", "status", "process", "-h", "--help"}:
            argv = ["ingest", *argv]
        return _run_module("services.ingest", argv)
    if func == "worker":
        return _run_module("apps.worker", args.args)
    if func == "init":
        fwd_argv = list(args.args or [])
        return _run_module("apps.hexis_init", fwd_argv)
    if func == "mcp":
        return _run_module("apps.hexis_mcp_server", args.args)
    if func == "api":
        api_argv = ["--host", args.host, "--port", str(args.port)]
        return _run_module("apps.hexis_api", api_argv)
    if func == "ui":
        if is_source:
            return _handle_ui(stack_root, args.port, args.no_open, args.instance)
        # pip install path: run UI via container
        if compose_file is None:
            _print_err(
                "No compose file found. Reinstall hexis or run from a source checkout."
            )
            return 1
        docker_bin = ensure_docker()
        compose_cmd_ui = ensure_compose(docker_bin)
        return _handle_ui_container(
            compose_cmd_ui, compose_file, stack_root, env_file, args.port, args.no_open
        )
    if func == "open":
        from core.browser import open_url

        if not _port_ready(args.port):
            _print_err(
                f"Nothing is listening on http://localhost:{args.port}. "
                "Start the dashboard first with `hexis ui`."
            )
            return 1
        open_url(f"http://localhost:{args.port}")
        return 0
    if func == "start":
        if host_managed_workers:
            workers_ok, workers_error = _ensure_installed_host_services_running()
            if not workers_ok:
                _print_err(str(workers_error))
                return 1
        docker_targets = [
            name
            for name in ("heartbeat_worker", "maintenance_worker")
            if name not in host_managed_workers
        ]
        if not docker_targets:
            print("Started heartbeat and maintenance host services.")
            return 0
        return run_compose(
            compose_cmd or [],
            compose_file,
            stack_root,
            ["up", "-d", *docker_targets],
            env_file,
        )
    if func == "stop":
        if host_managed_workers:
            workers_ok, workers_error = _stop_installed_host_services()
            if not workers_ok:
                _print_err(str(workers_error))
                return 1
        docker_targets = [
            name
            for name in ("heartbeat_worker", "maintenance_worker")
            if name not in host_managed_workers
        ]
        if not docker_targets:
            print("Stopped heartbeat and maintenance host services.")
            return 0
        return run_compose(
            compose_cmd or [],
            compose_file,
            stack_root,
            ["stop", *docker_targets],
            env_file,
        )
    if func == "doctor":
        dsn = _get_dsn(args)

        # Handle --demo flag (or 'demo' alias)
        if getattr(args, "demo", False):
            result = asyncio.run(cli_api.demo(dsn, wait_seconds=args.wait_seconds))
            if args.json:
                sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            else:
                _print_alive_demo(result)
            return 0 if result.get("ok") else 1

        from apps.cli_theme import console as _con, make_table as _mt
        from rich.spinner import Spinner
        from rich.live import Live

        with Live(
            Spinner("dots", text="Running diagnostics..."), console=_con, transient=True
        ):
            checks = asyncio.run(
                cli_api.doctor_payload(
                    dsn,
                    wait_seconds=args.wait_seconds,
                    check_llm=bool(getattr(args, "llm", False)),
                )
            )

        if args.json:
            sys.stdout.write(json.dumps(checks, indent=2) + "\n")
        else:
            table = _mt(
                ("", {"width": 3}),
                ("Check", {"style": "bold"}),
                "Detail",
            )
            for c in checks:
                status = c["status"]
                if status == "OK":
                    badge = "[ok]\u2714[/ok]"
                elif status == "WARN":
                    badge = "[warn]\u26a0[/warn]"
                else:
                    badge = "[fail]\u2718[/fail]"
                table.add_row(badge, c["label"], c["detail"])
            _con.print(table)
            ok = sum(1 for c in checks if c["status"] == "OK")
            warn_count = sum(1 for c in checks if c["status"] == "WARN")
            fail_count = sum(1 for c in checks if c["status"] == "FAIL")
            _con.print(
                f"\n[ok]{ok} passed[/ok], [warn]{warn_count} warnings[/warn], [fail]{fail_count} failures[/fail]"
            )
        return 0 if all(c["status"] != "FAIL" for c in checks) else 1
    if func == "demo":
        dsn = _get_dsn(args)
        result = asyncio.run(cli_api.demo(dsn, wait_seconds=args.wait_seconds))
        if args.json:
            sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            _print_alive_demo(result)
        return 0 if result.get("ok") else 1
    if func == "maturity":
        result = asyncio.run(
            cli_api.maturity_scorecard(_get_dsn(args), wait_seconds=args.wait_seconds)
        )
        if args.json:
            sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            _print_maturity_scorecard(result)
        return 0
    if func == "status":
        dsn = _get_dsn(args)
        if args.raw:
            # Legacy raw status
            payload = asyncio.run(
                cli_api.status_payload(dsn, wait_seconds=args.wait_seconds)
            )
            if not args.no_docker:
                try:
                    docker_bin = ensure_docker()
                    compose_cmd = ensure_compose(docker_bin)
                    if compose_file is None:
                        raise SystemExit
                    rc, out = _run_compose_capture(
                        compose_cmd, compose_file, stack_root, ["ps"], env_file
                    )
                    payload["docker_ps_rc"] = rc
                    payload["docker_ps"] = out
                except SystemExit:
                    payload["docker_ps_rc"] = 1
                    payload["docker_ps"] = "Docker not available"
            if args.json:
                sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            else:
                lines = [
                    f"DB time: {payload.get('db_time')}",
                    f"Agent configured: {payload.get('agent_configured')}",
                    f"Heartbeat paused: {payload.get('heartbeat_paused')}",
                    f"Should run heartbeat: {payload.get('should_run_heartbeat')}",
                    f"Maintenance paused: {payload.get('maintenance_paused')}",
                    f"Should run maintenance: {payload.get('should_run_maintenance')}",
                    f"Embedding URL: {payload.get('embedding_service_url')}",
                    f"Embedding healthy: {payload.get('embedding_service_healthy')}",
                    f"Pending external_calls: {payload.get('pending_external_calls')}",
                    f"Pending outbox_messages: {payload.get('pending_outbox_messages')}",
                ]
                sys.stdout.write("\n".join(lines) + "\n")
            return 0
        # Rich status (default)
        payload = asyncio.run(
            cli_api.status_payload_rich(dsn, wait_seconds=args.wait_seconds)
        )
        if args.json:
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            _print_rich_status(payload)
        return 0
    if func == "retention":
        dsn = _get_dsn(args)
        return asyncio.run(_retention_status(dsn, args.json))
    if func == "retention_dry_run":
        return asyncio.run(_retention_dry_run(_get_dsn(args), args.json))
    if func == "retention_enable":
        return asyncio.run(_retention_enable(_get_dsn(args), args.yes))
    if func == "retention_disable":
        return asyncio.run(_retention_disable(_get_dsn(args)))
    if func == "skills_status":
        return asyncio.run(_skills_status(_get_dsn(args), args.json))
    if func == "skills_enable":
        return asyncio.run(_skills_enable(_get_dsn(args), args.yes))
    if func == "skills_disable":
        return asyncio.run(_skills_disable(_get_dsn(args)))
    if func == "skills_proposals":
        return asyncio.run(_skills_proposals(_get_dsn(args), args.status, args.json))
    if func == "skills_review":
        return asyncio.run(
            _skills_review(_get_dsn(args), args.proposal_id, args.action, args.yes)
        )
    if func == "migrate":
        return asyncio.run(_migrate(_get_dsn(args), args.status))
    if func == "backup":
        return _do_backup(_get_dsn(args), args.output, args.label)
    if func == "restore":
        return _do_restore(_get_dsn(args), args.path, args.yes)
    if func == "config":
        # Bare `config` shows the grouped table like `config show` (not raw JSON),
        # matching how `goals`/`schedule`/`channels` default to their list view.
        if not hasattr(args, "no_redact"):
            args.no_redact = False
        if not hasattr(args, "json"):
            args.json = False
        func = "config_show"
    if func == "config_show":
        dsn = _get_dsn(args)
        cfg = asyncio.run(cli_api.config_rows(dsn, wait_seconds=args.wait_seconds))
        if not args.no_redact:
            cfg = _redact_config(cfg)
        if args.json:
            sys.stdout.write(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        else:
            from apps.cli_theme import console as _con, make_table as _mt

            # Group by key prefix
            groups: dict[str, list[tuple[str, str]]] = {}
            for key in sorted(cfg.keys()):
                prefix = key.split(".")[0] if "." in key else key
                val = cfg[key]
                display = json.dumps(val) if not isinstance(val, str) else val
                groups.setdefault(prefix, []).append((key, display))
            table = _mt(
                ("Key", {"style": "key"}),
                "Value",
                title="Configuration",
            )
            first_group = True
            for prefix, items in groups.items():
                if not first_group:
                    table.add_section()
                first_group = False
                for key, val in items:
                    display_val = (
                        f"[dim]{val}[/dim]" if val == "***" or val == '"***"' else val
                    )
                    table.add_row(key, display_val)
            _con.print(table)
        return 0
    if func == "config_validate":
        dsn = _get_dsn(args)
        errors, warnings = asyncio.run(
            cli_api.config_validate(dsn, wait_seconds=args.wait_seconds)
        )
        for w in warnings:
            _print_err(f"warning: {w}")
        if errors:
            for e in errors:
                _print_err(f"error: {e}")
            return 1
        sys.stdout.write("ok\n")
        return 0

    # Auth commands (OAuth / subscription flows) — delegated to cli_auth
    if func.startswith("auth"):
        from apps.cli_auth import dispatch_auth_command

        dsn = _get_dsn(args)
        result = dispatch_auth_command(func, args, dsn)
        if result is not None:
            return result

    if func.startswith("node_"):
        from apps.cli_node import dispatch_node_command

        result = dispatch_node_command(func, args, _get_dsn(args))
        if result is not None:
            return result

    if func.startswith("execution_"):
        from apps.cli_execution import dispatch_execution_command

        result = dispatch_execution_command(func, args, _get_dsn(args))
        if result is not None:
            return result

    # Tools commands
    if func == "tools_list":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_list(dsn, args.context, args.json))
    if func == "tools_enable":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_enable(dsn, args.tool_name))
    if func == "tools_disable":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_disable(dsn, args.tool_name))
    if func == "tools_set_api_key":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_set_api_key(dsn, args.key_name, args.value))
    if func == "tools_set_cost":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_set_cost(dsn, args.tool_name, args.cost))
    if func == "tools_web_search_status":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_web_search_status(dsn, args.json))
    if func == "tools_web_search_set_provider":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_web_search_set_provider(dsn, args.provider))
    if func == "tools_web_search_set_searxng_url":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_web_search_set_searxng_url(dsn, args.url))
    if func == "tools_add_mcp":
        dsn = _get_dsn(args)
        return asyncio.run(
            _tools_add_mcp(dsn, args.name, args.command, args.args, args.env)
        )
    if func == "tools_remove_mcp":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_remove_mcp(dsn, args.name))
    if func == "tools_status":
        dsn = _get_dsn(args)
        return asyncio.run(_tools_status(dsn, args.json))

    # Channels commands
    if func == "channels":
        # Default: 'hexis channels' → channels status
        dsn = _get_dsn(args)
        return asyncio.run(_channels_status(dsn, False))
    if func == "channels_start":
        from services.channel_worker import run_channel_worker

        asyncio.run(run_channel_worker(channels=args.channel, instance=args.instance))
        return 0
    if func == "channels_status":
        dsn = _get_dsn(args)
        return asyncio.run(_channels_status(dsn, args.json))
    if func == "channels_setup":
        dsn = _get_dsn(args)
        return asyncio.run(_channels_setup(dsn, args.channel_type))

    # Recall command
    if func == "recall":
        dsn = _get_dsn(args)
        return asyncio.run(
            _recall(dsn, args.query, args.limit, args.memory_type, args.json)
        )

    # Goals commands
    if func == "goals":
        # Default: 'hexis goals' → goals list
        dsn = _get_dsn(args)
        return asyncio.run(_goals_list(dsn, None, False))
    if func == "goals_list":
        dsn = _get_dsn(args)
        return asyncio.run(_goals_list(dsn, args.priority, args.json))
    if func == "goals_create":
        dsn = _get_dsn(args)
        return asyncio.run(
            _goals_create(dsn, args.title, args.description, args.priority, args.source)
        )
    if func == "goals_update":
        dsn = _get_dsn(args)
        return asyncio.run(_goals_update(dsn, args.goal_id, args.priority, args.reason))
    if func == "goals_complete":
        dsn = _get_dsn(args)
        return asyncio.run(_goals_update(dsn, args.goal_id, "completed", args.reason))

    # Schedule commands
    if func == "schedule":
        # Default: 'hexis schedule' → schedule list
        dsn = _get_dsn(args)
        return asyncio.run(_schedule_list(dsn, None, False))
    if func == "schedule_list":
        dsn = _get_dsn(args)
        return asyncio.run(_schedule_list(dsn, args.status, args.json))
    if func == "schedule_create":
        dsn = _get_dsn(args)
        return asyncio.run(
            _schedule_create(
                dsn,
                args.name,
                args.kind,
                args.action,
                args.payload,
                args.schedule,
                args.timezone,
                args.description,
            )
        )
    if func == "schedule_delete":
        dsn = _get_dsn(args)
        return asyncio.run(_schedule_delete(dsn, args.task_id, args.force))

    _print_err(f"Unknown command: {func}")
    _print_grouped_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
