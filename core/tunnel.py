"""Private Tailscale Serve lifecycle and OSS exposure posture.

Hexis OSS has no application authentication layer. This module therefore owns
only a loopback-to-tailnet HTTPS route and refuses public binds, Tailscale
Funnel, or replacement of an unrelated root Serve handler.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


STATE_VERSION = 1
SAFE_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class TunnelError(RuntimeError):
    """A tunnel problem with a concrete recovery step."""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _run_command(
    command: Sequence[str],
    *,
    capture_output: bool = True,
    check: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=capture_output,
        text=True,
        check=check,
        timeout=timeout,
    )


def tunnel_state_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".hexis" / "tunnel.json"


def _tailscale_command() -> str:
    command = shutil.which("tailscale")
    if command:
        return command
    mac_app_command = Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale")
    if mac_app_command.is_file() and os.access(mac_app_command, os.X_OK):
        return str(mac_app_command)
    raise TunnelError(
        "Tailscale is not installed or its CLI is not on PATH. Install it from "
        "https://tailscale.com/download, join this host to your tailnet, then run "
        "`hexis tunnel start` again."
    )


def _compact_output(result: subprocess.CompletedProcess[str]) -> str:
    detail = " ".join((result.stderr or result.stdout or "").strip().split())
    return detail[:997] + "..." if len(detail) > 1_000 else detail


def _command_failure(
    action: str, result: subprocess.CompletedProcess[str]
) -> TunnelError:
    detail = _compact_output(result)
    suffix = f" Tailscale said: {detail}" if detail else ""
    return TunnelError(
        f"Could not {action} (exit {result.returncode}).{suffix} Run `hexis tunnel "
        "status` to inspect the unchanged route, or follow "
        "docs/operations/secure-remote-access.md for Tailscale setup."
    )


def _read_json_command(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    action: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        result = runner(
            command,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TunnelError(f"Could not {action}: {exc}") from exc
    if result.returncode != 0:
        raise _command_failure(action, result)
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise TunnelError(
            f"Could not read {action}: Tailscale returned invalid JSON. Upgrade "
            "Tailscale, then retry."
        ) from exc
    if not isinstance(payload, dict):
        raise TunnelError(
            f"Could not read {action}: Tailscale returned an unexpected JSON shape. "
            "Upgrade Tailscale, then retry."
        )
    return payload


def _load_state(home: Path | None = None) -> dict[str, Any]:
    path = tunnel_state_path(home)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TunnelError(
            f"Hexis tunnel ownership state is unreadable at {path}. Review and move "
            "that file aside before changing the route."
        ) from exc
    if not isinstance(payload, dict) or payload.get("version") != STATE_VERSION:
        raise TunnelError(
            f"Hexis tunnel ownership state at {path} has an unsupported format. "
            "Upgrade Hexis or move the file aside after reviewing it."
        )
    return payload


def _write_state(payload: dict[str, Any], home: Path | None = None) -> None:
    path = tunnel_state_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _iter_serve_configs(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield config
    foreground = config.get("Foreground")
    if isinstance(foreground, dict):
        for nested in foreground.values():
            if isinstance(nested, dict):
                yield from _iter_serve_configs(nested)


def _serve_routes(config: dict[str, Any]) -> list[dict[str, str]]:
    routes: list[dict[str, str]] = []
    for candidate in _iter_serve_configs(config):
        web = candidate.get("Web")
        if not isinstance(web, dict):
            continue
        for host_port, server in web.items():
            if not isinstance(server, dict):
                continue
            handlers = server.get("Handlers")
            if not isinstance(handlers, dict):
                continue
            for path, handler in handlers.items():
                if not isinstance(handler, dict):
                    continue
                proxy = handler.get("Proxy")
                if isinstance(proxy, str) and proxy.strip():
                    routes.append(
                        {
                            "host_port": str(host_port),
                            "path": str(path),
                            "proxy": proxy.strip(),
                        }
                    )
    return routes


def _funnel_host_ports(config: dict[str, Any]) -> set[str]:
    host_ports: set[str] = set()
    for candidate in _iter_serve_configs(config):
        allow_funnel = candidate.get("AllowFunnel")
        if not isinstance(allow_funnel, dict):
            continue
        host_ports.update(
            str(host_port)
            for host_port, enabled in allow_funnel.items()
            if enabled is True
        )
    return host_ports


def _proxy_matches_loopback_port(proxy: str, port: int) -> bool:
    try:
        parsed = urllib.parse.urlparse(proxy)
        return (
            parsed.scheme.lower() == "http"
            and (parsed.hostname or "").lower() in SAFE_LOOPBACK_HOSTS
            and parsed.port == port
        )
    except ValueError:
        return False


def _local_dashboard_ready(port: int, *, timeout_seconds: float) -> bool:
    target = f"http://127.0.0.1:{port}/api/status"
    request = urllib.request.Request(target, headers={"User-Agent": "Hexis tunnel"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 500
    except urllib.error.HTTPError:
        return True
    except (OSError, ValueError, urllib.error.URLError, TimeoutError):
        return False


def _validate_port(port: int) -> int:
    value = int(port)
    if value < 1 or value > 65_535:
        raise TunnelError("Dashboard port must be between 1 and 65535.")
    return value


def _is_safe_bind(bind_address: str | None) -> bool:
    value = str(bind_address or "127.0.0.1").strip().lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value in SAFE_LOOPBACK_HOSTS


def tunnel_status(
    *,
    ui_port: int = 3477,
    bind_address: str | None = None,
    home: Path | None = None,
    runner: CommandRunner = _run_command,
    timeout_seconds: float = 3.0,
    probe_local: bool = True,
) -> dict[str, Any]:
    """Return Tailscale route truth without changing local or network state."""

    port = _validate_port(ui_port)
    bind = str(bind_address or "127.0.0.1").strip()
    public_bind = not _is_safe_bind(bind)
    state = _load_state(home)
    state_path = tunnel_state_path(home)
    local_ready = (
        _local_dashboard_ready(port, timeout_seconds=timeout_seconds)
        if probe_local
        else None
    )
    try:
        tailscale = _tailscale_command()
    except TunnelError as exc:
        issues = []
        if public_bind:
            issues.append(
                f"HEXIS_BIND_ADDRESS={bind} exposes the unauthenticated OSS dashboard "
                "beyond loopback. Set it to 127.0.0.1; public binding is out of bounds."
            )
        return {
            "status": "risky" if issues else "unavailable",
            "available": False,
            "connected": False,
            "dns_name": None,
            "url": None,
            "ui_port": port,
            "bind_address": bind,
            "public_bind": public_bind,
            "local_ready": local_ready,
            "serve_configured": False,
            "target_matches": False,
            "root_conflict": None,
            "funnel_enabled": False,
            "owned": bool(state),
            "state_present": bool(state),
            "state_path": str(state_path),
            "issues": issues,
            "detail": str(exc),
        }

    try:
        tailscale_status = _read_json_command(
            runner,
            [tailscale, "status", "--json"],
            action="read Tailscale status",
            timeout_seconds=timeout_seconds,
        )
    except TunnelError as exc:
        issues = [str(exc)]
        if public_bind:
            issues.insert(
                0,
                f"HEXIS_BIND_ADDRESS={bind} exposes the unauthenticated OSS dashboard "
                "beyond loopback. Set it to 127.0.0.1; public binding is out of bounds.",
            )
        return {
            "status": "risky" if public_bind else "unavailable",
            "available": True,
            "connected": False,
            "dns_name": None,
            "url": None,
            "ui_port": port,
            "bind_address": bind,
            "public_bind": public_bind,
            "local_ready": local_ready,
            "serve_configured": False,
            "target_matches": False,
            "root_conflict": None,
            "funnel_enabled": False,
            "owned": bool(state),
            "state_present": bool(state),
            "state_path": str(state_path),
            "issues": issues,
            "detail": str(exc),
        }

    self_status = tailscale_status.get("Self")
    dns_name = (
        str(self_status.get("DNSName") or "").strip().rstrip(".")
        if isinstance(self_status, dict)
        else ""
    )
    backend_state = str(tailscale_status.get("BackendState") or "").strip().lower()
    self_online = self_status.get("Online") if isinstance(self_status, dict) else None
    connected = (
        bool(dns_name) and backend_state in {"", "running"} and self_online is not False
    )

    try:
        serve_config = _read_json_command(
            runner,
            [tailscale, "serve", "status", "--json"],
            action="read Tailscale Serve status",
            timeout_seconds=timeout_seconds,
        )
        serve_error = None
    except TunnelError as exc:
        serve_config = {}
        serve_error = str(exc)

    routes = _serve_routes(serve_config)
    funnel_host_ports = _funnel_host_ports(serve_config)
    expected_host_port = f"{dns_name}:443" if dns_name else None
    root_routes = [
        route
        for route in routes
        if route["path"] == "/"
        and (expected_host_port is None or route["host_port"] == expected_host_port)
    ]
    matching_routes = [
        route
        for route in root_routes
        if _proxy_matches_loopback_port(route["proxy"], port)
    ]
    target_matches = bool(matching_routes)
    root_conflict = next(
        (route["proxy"] for route in root_routes if route not in matching_routes),
        None,
    )
    hexis_funnel = any(
        route["host_port"] in funnel_host_ports
        and _proxy_matches_loopback_port(route["proxy"], port)
        for route in routes
    )
    funnel_on_https_port = bool(
        expected_host_port and expected_host_port in funnel_host_ports
    )
    expected_target = f"http://127.0.0.1:{port}"
    owned = bool(
        state
        and state.get("dns_name") == dns_name
        and state.get("target") == expected_target
        and int(state.get("ui_port") or 0) == port
    )

    issues: list[str] = []
    if public_bind:
        issues.append(
            f"HEXIS_BIND_ADDRESS={bind} exposes the unauthenticated OSS dashboard "
            "beyond loopback. Set it to 127.0.0.1; public binding is out of bounds."
        )
    if hexis_funnel:
        issues.append(
            "Tailscale Funnel exposes the Hexis dashboard to the public internet. "
            "Run `hexis tunnel stop` if Hexis owns this route, or disable the exact "
            "Funnel route with Tailscale before continuing."
        )
    if serve_error:
        issues.append(serve_error)
    if root_conflict:
        issues.append(
            f"The tailnet HTTPS root already proxies to {root_conflict}; Hexis will not "
            "replace that unrelated route."
        )

    if public_bind or hexis_funnel:
        overall = "risky"
    elif serve_error:
        overall = "unavailable"
    elif root_conflict:
        overall = "conflict"
    elif connected and target_matches:
        overall = "active"
    elif not connected:
        overall = "disconnected"
    else:
        overall = "inactive"
    detail = (
        f"private tailnet HTTPS routes to {expected_target}"
        if overall == "active"
        else "Tailscale is not connected; run `tailscale up`, complete sign-in, then retry"
        if overall == "disconnected"
        else "no Hexis Tailscale Serve route is active"
    )
    return {
        "status": overall,
        "available": True,
        "connected": connected,
        "dns_name": dns_name or None,
        "url": f"https://{dns_name}" if dns_name else None,
        "ui_port": port,
        "bind_address": bind,
        "public_bind": public_bind,
        "local_ready": local_ready,
        "serve_configured": bool(routes),
        "target_matches": target_matches,
        "root_conflict": root_conflict,
        "funnel_enabled": hexis_funnel,
        "funnel_on_https_port": funnel_on_https_port,
        "owned": owned,
        "state_present": bool(state),
        "state_path": str(state_path),
        "route_count": len(routes),
        "issues": issues,
        "detail": detail,
    }


def start_tunnel(
    *,
    ui_port: int = 3477,
    bind_address: str | None = None,
    home: Path | None = None,
    runner: CommandRunner = _run_command,
    timeout_seconds: float = 10.0,
    probe_local: bool = True,
) -> dict[str, Any]:
    port = _validate_port(ui_port)
    bind = str(bind_address or "127.0.0.1").strip()
    if not _is_safe_bind(bind):
        raise TunnelError(
            f"HEXIS_BIND_ADDRESS={bind} exposes the unauthenticated OSS dashboard beyond "
            "loopback. Set it to 127.0.0.1 before creating a private tunnel."
        )
    if probe_local and not _local_dashboard_ready(port, timeout_seconds=2.0):
        raise TunnelError(
            f"The dashboard is not responding at http://127.0.0.1:{port}. Run `hexis "
            "up`, resolve any startup error, then retry `hexis tunnel start`."
        )
    before = tunnel_status(
        ui_port=port,
        bind_address=bind,
        home=home,
        runner=runner,
        timeout_seconds=timeout_seconds,
        probe_local=False,
    )
    if not before["available"]:
        raise TunnelError(str(before["detail"]))
    if not before["connected"]:
        raise TunnelError(str(before["detail"]))
    if before["status"] == "unavailable":
        raise TunnelError(" ".join(before["issues"]) or str(before["detail"]))
    if before["state_present"] and not before["owned"]:
        raise TunnelError(
            f"Hexis already has tunnel ownership state at {before['state_path']}, but "
            "it does not match this tailnet or dashboard port. Review that state and "
            "the current Serve configuration before changing either one."
        )
    if before["public_bind"] or before["funnel_enabled"]:
        raise TunnelError(" ".join(before["issues"]))
    if before["root_conflict"]:
        raise TunnelError(" ".join(before["issues"]))
    if before["target_matches"]:
        return {
            **before,
            "changed": False,
            "warning": None
            if before["owned"]
            else (
                "This matching route already existed and was not created by Hexis. Hexis "
                "left it unchanged and will not remove it with `hexis tunnel stop`."
            ),
        }
    if before.get("funnel_on_https_port"):
        raise TunnelError(
            "Port 443 already has a Tailscale Funnel route. Adding Serve would change "
            "that public route, so Hexis left it untouched. Disable Funnel explicitly, "
            "then retry."
        )

    tailscale = _tailscale_command()
    target = f"http://127.0.0.1:{port}"
    command = [
        tailscale,
        "serve",
        "--bg",
        "--https=443",
        "--set-path=/",
        target,
    ]
    try:
        result = runner(
            command,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TunnelError(
            f"Could not create the private Tailscale Serve route: {exc}"
        ) from exc
    if result.returncode != 0:
        raise _command_failure("create the private Tailscale Serve route", result)

    after = tunnel_status(
        ui_port=port,
        bind_address=bind,
        home=home,
        runner=runner,
        timeout_seconds=timeout_seconds,
        probe_local=False,
    )
    if not after["target_matches"] or after["funnel_enabled"]:
        rollback_provider = "funnel" if after["funnel_enabled"] else "serve"
        try:
            rollback = runner(
                [
                    tailscale,
                    rollback_provider,
                    "--https=443",
                    "--set-path=/",
                    "off",
                ],
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
            recovered = rollback.returncode == 0
        except (OSError, subprocess.SubprocessError):
            recovered = False
        recovery = (
            "The attempted route was turned back off."
            if recovered
            else "The route may still be active; inspect `tailscale serve status` now."
        )
        raise TunnelError(
            "Tailscale returned success but the resulting route did not match the private "
            f"Hexis target {target}. Hexis did not claim ownership. Run `tailscale serve "
            f"status --json` and review the configuration before retrying. {recovery}"
        )
    if probe_local:
        after["local_ready"] = True
    state = {
        "version": STATE_VERSION,
        "provider": "tailscale-serve",
        "dns_name": after["dns_name"],
        "url": after["url"],
        "target": target,
        "ui_port": port,
        "https_port": 443,
        "path": "/",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _write_state(state, home)
    except OSError as exc:
        # Do not leave behind a route that `hexis tunnel stop` would later
        # refuse to remove because ownership was never recorded.
        try:
            rollback = runner(
                [tailscale, "serve", "--https=443", "--set-path=/", "off"],
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
            rolled_back = rollback.returncode == 0
        except (OSError, subprocess.SubprocessError):
            rolled_back = False
        recovery = (
            "Hexis turned the route back off."
            if rolled_back
            else "The route may still be active; inspect `tailscale serve status` now."
        )
        raise TunnelError(
            f"The private route was created but ownership state could not be saved at "
            f"{tunnel_state_path(home)} ({exc}). {recovery}"
        ) from exc
    return {**after, "owned": True, "changed": True, "warning": None}


def stop_tunnel(
    *,
    ui_port: int | None = None,
    bind_address: str | None = None,
    home: Path | None = None,
    runner: CommandRunner = _run_command,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    state = _load_state(home)
    if not state:
        raise TunnelError(
            "Hexis has no ownership record for the current Serve route, so it refused "
            "to remove ambient Tailscale configuration. Use `tailscale serve status` to "
            "review it and disable the exact route manually."
        )
    port = _validate_port(ui_port or int(state.get("ui_port") or 3477))
    expected_target = f"http://127.0.0.1:{port}"
    if state.get("target") != expected_target:
        raise TunnelError(
            f"Tunnel state expects {state.get('target')}, not {expected_target}. Hexis "
            "left the route untouched; review the state and Tailscale status first."
        )
    before = tunnel_status(
        ui_port=port,
        bind_address=bind_address,
        home=home,
        runner=runner,
        timeout_seconds=timeout_seconds,
        probe_local=False,
    )
    if before["status"] == "unavailable":
        raise TunnelError(" ".join(before["issues"]) or str(before["detail"]))
    if before.get("dns_name") != state.get("dns_name"):
        raise TunnelError(
            "The active Tailscale identity no longer matches the one Hexis configured. "
            "Hexis left the route untouched; sign into the original tailnet or remove "
            "the stale route manually after review."
        )
    if not before["target_matches"]:
        path = tunnel_state_path(home)
        path.unlink(missing_ok=True)
        return {**before, "changed": False, "already_stopped": True, "owned": False}

    tailscale = _tailscale_command()
    provider = "funnel" if before["funnel_enabled"] else "serve"
    command = [tailscale, provider, "--https=443", "--set-path=/", "off"]
    try:
        result = runner(
            command,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TunnelError(f"Could not remove the Hexis tailnet route: {exc}") from exc
    if result.returncode != 0:
        raise _command_failure("remove the Hexis tailnet route", result)

    after = tunnel_status(
        ui_port=port,
        bind_address=bind_address,
        home=home,
        runner=runner,
        timeout_seconds=timeout_seconds,
        probe_local=False,
    )
    if after["target_matches"]:
        raise TunnelError(
            "Tailscale returned success but the Hexis route is still active. Ownership "
            "state was preserved; inspect `tailscale serve status --json` and retry."
        )
    tunnel_state_path(home).unlink(missing_ok=True)
    return {**after, "changed": True, "already_stopped": False, "owned": False}


def remote_exposure_check(
    *,
    ui_port: int = 3477,
    bind_address: str | None = None,
    home: Path | None = None,
    runner: CommandRunner = _run_command,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    """Doctor check for deterministic OSS exposure hazards."""

    try:
        payload = tunnel_status(
            ui_port=ui_port,
            bind_address=bind_address,
            home=home,
            runner=runner,
            timeout_seconds=timeout_seconds,
            probe_local=False,
        )
    except TunnelError as exc:
        return {
            "label": "Remote exposure",
            "status": "WARN",
            "detail": str(exc),
        }
    if payload["public_bind"] or payload["funnel_enabled"]:
        return {
            "label": "Remote exposure",
            "status": "FAIL",
            "detail": " ".join(payload["issues"]),
        }
    if payload["available"] and payload["status"] == "unavailable":
        return {
            "label": "Remote exposure",
            "status": "WARN",
            "detail": payload["detail"],
        }
    detail = (
        f"private tailnet Serve only at {payload['url']}; loopback target retained"
        if payload["target_matches"]
        else "loopback-only; no Hexis Tailscale Funnel route detected"
    )
    return {"label": "Remote exposure", "status": "OK", "detail": detail}
