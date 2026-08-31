"""CLI management for explicit local, SSH, and remote-Docker execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from core.execution_backends import (
    ExecutionBackendError,
    ExecutionProfile,
    ExecutionSettings,
    resolve_execution_backend,
    validate_ssh_material,
)


def register_execution_parser(
    subparsers: Any,
    db_parent: argparse.ArgumentParser,
) -> None:
    execution = subparsers.add_parser(
        "execution",
        help="Choose where shell, script, and code tools run",
        description=(
            "Manage explicit execution profiles. Remote profiles never consume "
            "ambient SSH configuration and never silently fall back to local."
        ),
    )
    execution.set_defaults(func="execution_status", json=False)
    commands = execution.add_subparsers(dest="execution_command")

    status = commands.add_parser(
        "status", parents=[db_parent], help="Show profiles without connecting remotely"
    )
    status.add_argument("--json", action="store_true", help="Output JSON")
    status.set_defaults(func="execution_status")

    add_ssh = commands.add_parser(
        "add-ssh", parents=[db_parent], help="Add an exact SSH execution profile"
    )
    add_ssh.add_argument("name")
    add_ssh.add_argument("--host", required=True)
    add_ssh.add_argument("--user", required=True)
    add_ssh.add_argument("--port", type=int, default=22)
    add_ssh.add_argument(
        "--workspace", required=True, help="Absolute workspace path on the remote host"
    )
    _add_ssh_material_arguments(add_ssh)
    add_ssh.add_argument("--python", default="python3", dest="python_command")
    add_ssh.add_argument(
        "--replace", action="store_true", help="Explicitly replace an existing profile"
    )
    add_ssh.set_defaults(func="execution_add_ssh")

    add_docker = commands.add_parser(
        "add-docker",
        parents=[db_parent],
        help="Add an ephemeral remote-Docker execution profile over SSH",
    )
    add_docker.add_argument("name")
    add_docker.add_argument(
        "--docker-host",
        required=True,
        help="Exact ssh://USER@HOST[:PORT] Docker endpoint",
    )
    add_docker.add_argument("--image", required=True)
    add_docker.add_argument(
        "--workspace",
        required=True,
        help="Absolute workspace path on the remote Docker host",
    )
    add_docker.add_argument(
        "--container-workspace", default="/workspace", help="Container mount target"
    )
    add_docker.add_argument(
        "--network",
        choices=["none", "bridge"],
        default="none",
        help="Container network policy (default: none)",
    )
    add_docker.add_argument(
        "--state-volume",
        default=None,
        help="Named volume for remote execute_code session state",
    )
    _add_ssh_material_arguments(add_docker)
    add_docker.add_argument("--python", default="python3", dest="python_command")
    add_docker.add_argument(
        "--replace", action="store_true", help="Explicitly replace an existing profile"
    )
    add_docker.set_defaults(func="execution_add_docker")

    use = commands.add_parser(
        "use", parents=[db_parent], help="Select one configured profile"
    )
    use.add_argument("name")
    use.set_defaults(func="execution_use")

    test = commands.add_parser(
        "test",
        parents=[db_parent],
        help="Run a read-only shell and Python availability check",
    )
    test.add_argument(
        "name", nargs="?", default=None, help="Profile; active if omitted"
    )
    test.add_argument("--json", action="store_true", help="Output JSON")
    test.add_argument("--timeout", type=int, default=30)
    test.set_defaults(func="execution_test")

    remove = commands.add_parser(
        "remove", parents=[db_parent], help="Remove an inactive remote profile"
    )
    remove.add_argument("name")
    remove.set_defaults(func="execution_remove")


def _add_ssh_material_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--identity-file",
        required=True,
        help="Exact private-key path visible to the Hexis worker",
    )
    parser.add_argument(
        "--known-hosts-file",
        required=True,
        help="Exact known_hosts path; strict host checking is always enabled",
    )


def _print_err(message: str) -> None:
    sys.stderr.write(message.rstrip() + "\n")


def _local_path(raw: str) -> str:
    return str(Path(raw).expanduser().resolve())


async def _open_pool(dsn: str) -> Any:
    import asyncpg

    return await asyncpg.create_pool(dsn, min_size=1, max_size=2)


async def _read_settings_from_conn(conn: Any) -> ExecutionSettings:
    row = await conn.fetchrow(
        """
        SELECT get_config('execution.backends') AS backends,
               COALESCE(get_config_int('execution.max_output_chars'), 50000) AS max_output,
               COALESCE(get_config_int('execution.max_timeout_seconds'), 300) AS max_timeout,
               COALESCE(get_config_int('execution.repl_state_ttl_hours'), 168) AS state_ttl
        """
    )
    return ExecutionSettings.from_values(
        row["backends"],
        max_output_chars=row["max_output"],
        max_timeout_seconds=row["max_timeout"],
        state_ttl_hours=row["state_ttl"],
    )


async def _mutate_settings(
    dsn: str,
    mutation: Callable[[ExecutionSettings], ExecutionSettings],
    *,
    journal_action: str,
    journal_profile: str,
) -> ExecutionSettings:
    pool = await _open_pool(dsn)
    try:
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext('execution.backends'))"
            )
            settings = await _read_settings_from_conn(conn)
            updated = mutation(settings)
            # Re-parse the emitted document before it reaches the source of truth.
            checked = ExecutionSettings.from_values(
                updated.to_config(),
                max_output_chars=updated.max_output_chars,
                max_timeout_seconds=updated.max_timeout_seconds,
                state_ttl_hours=updated.state_ttl_hours,
            )
            await conn.execute(
                "SELECT set_config('execution.backends', $1::jsonb)",
                json.dumps(checked.to_config()),
            )
            profile = checked.profiles.get(journal_profile) or settings.profiles.get(
                journal_profile
            )
            await conn.execute(
                "SELECT record_change('config_flip', $1, $2::jsonb)",
                f"Execution profile {journal_profile!r} {journal_action}",
                json.dumps(
                    {
                        "subsystem": "execution_backend",
                        "action": journal_action,
                        "profile": journal_profile,
                        "backend_type": profile.kind if profile else None,
                        "active_before": settings.active,
                        "active_after": checked.active,
                    }
                ),
            )
            return checked
    finally:
        await pool.close()


def _with_profiles(
    settings: ExecutionSettings,
    profiles: dict[str, ExecutionProfile],
    *,
    active: str | None = None,
) -> ExecutionSettings:
    return ExecutionSettings(
        active=active or settings.active,
        profiles=profiles,
        max_output_chars=settings.max_output_chars,
        max_timeout_seconds=settings.max_timeout_seconds,
        state_ttl_hours=settings.state_ttl_hours,
    )


def _add_profile_mutation(
    profile: ExecutionProfile, *, replace: bool
) -> Callable[[ExecutionSettings], ExecutionSettings]:
    def mutate(settings: ExecutionSettings) -> ExecutionSettings:
        if profile.name == "local":
            raise ExecutionBackendError("the built-in local profile cannot be replaced")
        if profile.name in settings.profiles and not replace:
            raise ExecutionBackendError(
                f"execution profile {profile.name!r} already exists; pass --replace to change it explicitly"
            )
        profiles = dict(settings.profiles)
        profiles[profile.name] = profile
        return _with_profiles(settings, profiles)

    return mutate


async def _add_profile(
    dsn: str,
    profile: ExecutionProfile,
    *,
    replace: bool,
) -> int:
    try:
        validate_ssh_material(profile)
        await _mutate_settings(
            dsn,
            _add_profile_mutation(profile, replace=replace),
            journal_action="replaced" if replace else "added",
            journal_profile=profile.name,
        )
    except Exception as exc:
        _print_err(f"Could not save execution profile: {exc}")
        return 1
    sys.stdout.write(
        f"Saved execution profile '{profile.name}' ({profile.kind}); it is not active yet.\n"
        f"Verify it with `hexis execution test {profile.name}`, then select it with "
        f"`hexis execution use {profile.name}`.\n"
    )
    if profile.kind == "docker_remote":
        sys.stdout.write(
            "Images are never pulled implicitly; pull the selected image on the remote daemon before testing.\n"
        )
    return 0


async def _status(dsn: str, *, as_json: bool) -> int:
    pool: Any | None = None
    try:
        pool = await _open_pool(dsn)
        settings = await _read_settings_from_pool(pool)
    except Exception as exc:
        _print_err(f"Could not read execution profiles: {exc}")
        return 1
    finally:
        if pool is not None:
            await pool.close()

    payload = {
        "active": settings.active,
        "limits": {
            "max_output_chars": settings.max_output_chars,
            "max_timeout_seconds": settings.max_timeout_seconds,
            "repl_state_ttl_hours": settings.state_ttl_hours,
        },
        "profiles": [
            _profile_status(profile, active=name == settings.active)
            for name, profile in sorted(settings.profiles.items())
        ],
        "note": "Status is read-only and makes no remote connection.",
    }
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    sys.stdout.write(f"Active execution profile: {settings.active}\n")
    for item in payload["profiles"]:
        marker = "*" if item["active"] else " "
        readiness = "ready locally" if item["locally_ready"] else "needs attention"
        sys.stdout.write(f"{marker} {item['name']} ({item['type']}) — {readiness}\n")
        for issue in item["issues"]:
            sys.stdout.write(f"    {issue}\n")
    sys.stdout.write("No remote connections were opened.\n")
    return 0


def _profile_status(profile: ExecutionProfile, *, active: bool) -> dict[str, Any]:
    issues: list[str] = []
    if profile.kind != "local":
        try:
            validate_ssh_material(profile)
        except ExecutionBackendError as exc:
            issues.append(str(exc))
        if shutil.which("ssh") is None:
            issues.append("ssh is not on PATH")
        if profile.kind == "docker_remote" and shutil.which("docker") is None:
            issues.append("docker is not on PATH")
    public = profile.to_dict()
    return {
        "name": profile.name,
        "type": profile.kind,
        "active": active,
        "locally_ready": not issues,
        "issues": issues,
        "config": public,
    }


async def _read_settings_from_pool(pool: Any) -> ExecutionSettings:
    async with pool.acquire() as conn:
        return await _read_settings_from_conn(conn)


async def _use(dsn: str, name: str) -> int:
    def mutate(settings: ExecutionSettings) -> ExecutionSettings:
        if name not in settings.profiles:
            raise ExecutionBackendError(
                f"execution profile {name!r} does not exist; run `hexis execution status`"
            )
        return _with_profiles(settings, dict(settings.profiles), active=name)

    try:
        await _mutate_settings(
            dsn,
            mutate,
            journal_action="selected",
            journal_profile=name,
        )
    except Exception as exc:
        _print_err(f"Could not select execution profile: {exc}")
        return 1
    sys.stdout.write(
        f"Execution profile '{name}' is now active for shell, run_script, and execute_code.\n"
        "New tool calls use this choice immediately; running calls are unchanged.\n"
    )
    return 0


async def _remove(dsn: str, name: str) -> int:
    preserved_volume: str | None = None

    def mutate(settings: ExecutionSettings) -> ExecutionSettings:
        nonlocal preserved_volume
        if name == "local":
            raise ExecutionBackendError("the built-in local profile cannot be removed")
        if name not in settings.profiles:
            raise ExecutionBackendError(f"execution profile {name!r} does not exist")
        if settings.active == name:
            raise ExecutionBackendError(
                f"execution profile {name!r} is active; select another profile before removing it"
            )
        profile = settings.profiles[name]
        preserved_volume = profile.state_volume
        profiles = dict(settings.profiles)
        del profiles[name]
        return _with_profiles(settings, profiles)

    try:
        await _mutate_settings(
            dsn,
            mutate,
            journal_action="removed",
            journal_profile=name,
        )
    except Exception as exc:
        _print_err(f"Could not remove execution profile: {exc}")
        return 1
    sys.stdout.write(f"Removed execution profile '{name}'.\n")
    if preserved_volume:
        sys.stdout.write(
            f"Remote state volume '{preserved_volume}' was preserved. Hexis never deletes it implicitly.\n"
        )
    else:
        sys.stdout.write("Remote files and cached REPL state were preserved.\n")
    return 0


async def _test(dsn: str, name: str | None, *, timeout: int, as_json: bool) -> int:
    pool: Any | None = None
    try:
        pool = await _open_pool(dsn)
        backend = await resolve_execution_backend(
            pool=pool,
            profile_name=name,
            local_workspace=os.getcwd(),
        )
        quoted_python = shlex.quote(backend.profile.python_command)
        run = await backend.run_shell(
            "printf 'hexis-execution-ok\\n'; pwd; command -v " + quoted_python,
            timeout=timeout,
        )
    except Exception as exc:
        _print_err(f"Execution profile test failed: {exc}")
        return 1
    finally:
        if pool is not None:
            await pool.close()
    payload = {
        "profile": backend.name,
        "type": backend.kind,
        "success": not run.timed_out and run.returncode == 0,
        "timed_out": run.timed_out,
        "exit_code": run.returncode,
        "stdout": run.stdout.decode("utf-8", errors="replace"),
        "stderr": run.stderr.decode("utf-8", errors="replace"),
        "timeout_detail": run.timeout_detail,
    }
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    elif payload["success"]:
        sys.stdout.write(
            f"Execution profile '{backend.name}' is reachable.\n{payload['stdout']}"
        )
    else:
        _print_err(
            f"Execution profile '{backend.name}' failed with exit {run.returncode}."
        )
        if payload["stderr"]:
            _print_err(payload["stderr"])
        if run.timeout_detail:
            _print_err(run.timeout_detail)
    return 0 if payload["success"] else 1


def _ssh_profile(args: Any) -> ExecutionProfile:
    return ExecutionProfile.from_dict(
        args.name,
        {
            "type": "ssh",
            "host": args.host,
            "user": args.user,
            "port": args.port,
            "workspace": args.workspace,
            "identity_file": _local_path(args.identity_file),
            "known_hosts_file": _local_path(args.known_hosts_file),
            "python_command": args.python_command,
        },
    )


def _docker_profile(args: Any) -> ExecutionProfile:
    raw = {
        "type": "docker_remote",
        "docker_host": args.docker_host,
        "image": args.image,
        "workspace": args.workspace,
        "container_workspace": args.container_workspace,
        "network": args.network,
        "identity_file": _local_path(args.identity_file),
        "known_hosts_file": _local_path(args.known_hosts_file),
        "python_command": args.python_command,
    }
    if args.state_volume:
        raw["state_volume"] = args.state_volume
    return ExecutionProfile.from_dict(args.name, raw)


def dispatch_execution_command(func: str, args: Any, dsn: str) -> int | None:
    if not func.startswith("execution_"):
        return None
    if func == "execution_status":
        return asyncio.run(_status(dsn, as_json=getattr(args, "json", False)))
    if func == "execution_add_ssh":
        try:
            profile = _ssh_profile(args)
        except ExecutionBackendError as exc:
            _print_err(str(exc))
            return 1
        return asyncio.run(_add_profile(dsn, profile, replace=args.replace))
    if func == "execution_add_docker":
        try:
            profile = _docker_profile(args)
        except (ExecutionBackendError, ValueError) as exc:
            _print_err(str(exc))
            return 1
        return asyncio.run(_add_profile(dsn, profile, replace=args.replace))
    if func == "execution_use":
        return asyncio.run(_use(dsn, args.name))
    if func == "execution_test":
        return asyncio.run(
            _test(
                dsn,
                args.name,
                timeout=args.timeout,
                as_json=args.json,
            )
        )
    if func == "execution_remove":
        return asyncio.run(_remove(dsn, args.name))
    return None
