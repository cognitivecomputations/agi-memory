"""Owned lifecycle for the optional loopback speech sidecar."""

from __future__ import annotations

import json
import os
import secrets
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

VOICE_PORT = 42667
VOICE_HOST = "127.0.0.1"
PIPER_REQUIREMENT = "piper-tts[http]>=1.4.2,<2"
STATE_VERSION = 1


class VoiceSidecarError(RuntimeError):
    """A local voice lifecycle failure with a concrete recovery step."""


def state_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".hexis" / "voice-sidecar.json"


def log_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".hexis" / "voice-sidecar.log"


def _read_state(home: Path | None = None) -> dict[str, Any]:
    path = state_path(home)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VoiceSidecarError(
            f"Voice sidecar ownership state is unreadable at {path}. Review and move "
            "that file aside before starting or stopping a process."
        ) from exc
    if not isinstance(payload, dict) or payload.get("version") != STATE_VERSION:
        raise VoiceSidecarError(
            f"Voice sidecar ownership state at {path} has an unsupported format. "
            "Upgrade Hexis or move it aside after review."
        )
    return payload


def _write_state(payload: dict[str, Any], home: Path | None = None) -> None:
    path = state_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _process_command(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    command = str(result.stdout or "").strip()
    return command or None


def _is_owned_command(command: str | None, state: dict[str, Any]) -> bool:
    token = str(state.get("ownership_token") or "")
    if not command or not token:
        return False
    try:
        arguments = shlex.split(command)
    except ValueError:
        return False
    try:
        module_index = arguments.index("-m")
        token_index = arguments.index("--hexis-owner-token")
    except ValueError:
        return False
    return (
        module_index + 1 < len(arguments)
        and arguments[module_index + 1] == "apps.voice_sidecar"
        and token_index + 1 < len(arguments)
        and secrets.compare_digest(arguments[token_index + 1], token)
    )


def _provider_info(timeout_seconds: float = 1.0) -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"http://{VOICE_HOST}:{VOICE_PORT}/info",
        headers={"User-Agent": "Hexis voice lifecycle"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read(64 * 1024).decode("utf-8"))
    except (
        OSError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("voice"), dict):
        return None
    return payload


def voice_sidecar_status(*, home: Path | None = None) -> dict[str, Any]:
    state = _read_state(home)
    info = _provider_info()
    pid = int(state.get("pid") or 0) if state else 0
    command = _process_command(pid) if pid > 0 else None
    owned = bool(state and _is_owned_command(command, state))
    stale = bool(state and not owned)
    provider_voice = None
    if info:
        provider_voice = str(info.get("voice", {}).get("name") or "").strip() or None
    return {
        "status": "active" if info else "stale" if stale else "inactive",
        "ready": bool(info),
        "owned": owned,
        "state_present": bool(state),
        "stale": stale,
        "pid": pid or None,
        "model": provider_voice or (state.get("model") if state else None),
        "url": f"http://{VOICE_HOST}:{VOICE_PORT}",
        "state_path": str(state_path(home)),
        "log_path": str(log_path(home)),
        "detail": (
            "local voice sidecar is ready"
            if info
            else "saved ownership no longer matches a running process"
            if stale
            else "local voice sidecar is not running"
        ),
    }


def start_voice_sidecar(
    *,
    model: str,
    home: Path | None = None,
    command: Sequence[str] | None = None,
    wait_seconds: float = 300.0,
) -> dict[str, Any]:
    selected_model = str(model or "").strip()
    if not selected_model:
        raise VoiceSidecarError(
            "No live voice model is configured. Open Settings → Voice, save a local "
            "provider, then retry `hexis voice start`."
        )
    before = voice_sidecar_status(home=home)
    if before["ready"]:
        return {
            **before,
            "changed": False,
            "warning": None
            if before["owned"]
            else (
                "A compatible local voice sidecar already existed. Hexis did not "
                "adopt it and will not stop it."
            ),
        }
    if before["state_present"]:
        raise VoiceSidecarError(
            f"Voice ownership state at {before['state_path']} no longer matches a "
            "running Hexis voice process. Review the saved PID and log, then move the "
            "stale state aside before retrying."
        )

    ownership_token = secrets.token_urlsafe(24)
    launch = list(command or [sys.executable, "-m", "apps.voice_sidecar"])
    launch += [
        "--host",
        VOICE_HOST,
        "--port",
        str(VOICE_PORT),
        "--model",
        selected_model,
        "--hexis-owner-token",
        ownership_token,
    ]
    log = log_path(home)
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("ab") as output:
            process = subprocess.Popen(
                launch,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
    except OSError as exc:
        raise VoiceSidecarError(f"Could not start the local voice sidecar: {exc}") from exc

    state = {
        "version": STATE_VERSION,
        "pid": process.pid,
        "model": selected_model,
        "ownership_token": ownership_token,
        "host": VOICE_HOST,
        "port": VOICE_PORT,
        "log_path": str(log),
        "started_at": time.time(),
    }
    try:
        _write_state(state, home)
    except OSError as exc:
        process.terminate()
        raise VoiceSidecarError(
            f"The voice process started but ownership could not be recorded at "
            f"{state_path(home)} ({exc}); Hexis terminated that process."
        ) from exc

    deadline = time.monotonic() + max(1.0, wait_seconds)
    while time.monotonic() < deadline:
        if _provider_info():
            return {
                **voice_sidecar_status(home=home),
                "changed": True,
                "warning": None,
            }
        return_code = process.poll()
        if return_code is not None:
            state_path(home).unlink(missing_ok=True)
            detail = _tail(log)
            suffix = f" Last log lines: {detail}" if detail else ""
            raise VoiceSidecarError(
                f"The local voice sidecar exited with code {return_code}.{suffix} "
                "Run `hexis voice setup` if Piper is not installed."
            )
        time.sleep(0.25)

    # Preserve the process and ownership state: a first model download may
    # still be progressing, and killing it on a timer would discard user work.
    raise VoiceSidecarError(
        f"The voice sidecar is still starting after {wait_seconds:g} seconds. It was "
        f"left running; follow {log} or run `hexis voice status` until it is ready."
    )


def stop_voice_sidecar(*, home: Path | None = None) -> dict[str, Any]:
    state = _read_state(home)
    if not state:
        raise VoiceSidecarError(
            "Hexis has no ownership record for the local voice process, so it refused "
            "to stop an ambient service. Run `hexis voice status` to inspect it."
        )
    pid = int(state.get("pid") or 0)
    command = _process_command(pid) if pid > 0 else None
    if not _is_owned_command(command, state):
        raise VoiceSidecarError(
            f"Saved voice PID {pid or 'unknown'} no longer matches the Hexis launch "
            "contract. The process was left alone; review the state and log first."
        )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        state_path(home).unlink(missing_ok=True)
        return {**voice_sidecar_status(home=home), "changed": False}
    except OSError as exc:
        raise VoiceSidecarError(f"Could not stop voice process {pid}: {exc}") from exc
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _process_command(pid) is None:
            state_path(home).unlink(missing_ok=True)
            return {**voice_sidecar_status(home=home), "changed": True}
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise VoiceSidecarError(
            f"Voice process {pid} did not stop and could not be terminated: {exc}"
        ) from exc
    state_path(home).unlink(missing_ok=True)
    return {**voice_sidecar_status(home=home), "changed": True}


def _tail(path: Path, *, max_lines: int = 12) -> str:
    try:
        return " | ".join(path.read_text(errors="replace").splitlines()[-max_lines:])
    except OSError:
        return ""
