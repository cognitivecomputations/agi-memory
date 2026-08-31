"""Outward-only companion node daemon and its locally enforced capabilities."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from core.node_actions import APPLE_NODE_ACTIONS, ONEPASSWORD_NODE_ACTIONS
from core.node_identity import NodeIdentity, load_node_identity
from core.node_life import detect_life_capabilities, execute_life_action
from core.node_wake import NodeWakeError

_MAX_OUTPUT_CHARS = 50_000
_MAX_CAPTURE_BYTES = 8 * 1024 * 1024


class NodeAccessDeniedError(RuntimeError):
    """A signed identity was explicitly rejected and should not retry."""


def node_gateway_url(value: str | None) -> str:
    raw = str(
        value
        or os.getenv("HEXIS_NODE_GATEWAY_URL")
        or os.getenv("HEXIS_API_URL")
        or "http://127.0.0.1:43817"
    ).strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        raise ValueError(
            "Node gateway must be an http(s) or ws(s) URL, for example "
            "https://your-host.tailnet.ts.net."
        )
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    path = parsed.path.rstrip("/")
    if not path.endswith("/api/nodes/connect"):
        path = f"{path}/api/nodes/connect" if path else "/api/nodes/connect"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _screen_capture_command(target: Path) -> list[str] | None:
    if platform.system() == "Darwin":
        executable = shutil.which("screencapture") or "/usr/sbin/screencapture"
        if Path(executable).exists():
            return [executable, "-x", str(target)]
    grim = shutil.which("grim")
    if grim:
        return [grim, str(target)]
    gnome = shutil.which("gnome-screenshot")
    if gnome:
        return [gnome, "-f", str(target)]
    return None


def advertised_capabilities(identity: NodeIdentity) -> list[str]:
    capabilities: list[str] = []
    if identity.commands:
        capabilities.append("system.run")
    with tempfile.TemporaryDirectory() as temporary:
        if _screen_capture_command(Path(temporary) / "probe.png"):
            capabilities.append("screen.capture")
    capabilities.extend(detect_life_capabilities())
    if identity.wake.get("enabled"):
        capabilities.append("audio.wake")
    return sorted(set(capabilities))


async def _run_process(
    argv: list[str],
    *,
    timeout: int,
) -> tuple[int, str, str]:
    # Spool to anonymous files so a noisy but allowlisted process cannot grow
    # daemon memory without bound. Only the capped prefix enters the result.
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            env=os.environ.copy(),
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"Host command timed out after {timeout} seconds.")
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise

        def read_capped(handle: Any) -> str:
            handle.seek(0)
            raw = handle.read(_MAX_OUTPUT_CHARS * 4 + 1)
            text = raw.decode("utf-8", errors="replace")
            if len(text) > _MAX_OUTPUT_CHARS:
                return text[:_MAX_OUTPUT_CHARS] + "\n[output truncated by node]"
            return text

        return (
            int(proc.returncode or 0),
            read_capped(stdout_file),
            read_capped(stderr_file),
        )


async def execute_node_action(
    identity: NodeIdentity,
    action: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if action == "system.run":
        alias = str(arguments.get("command") or "").strip()
        configured = identity.commands.get(alias)
        if not isinstance(configured, dict):
            return {
                "success": False,
                "error": (
                    f"Host command {alias!r} is not allowlisted on this node. "
                    f"Run `hexis node allow {alias or '<alias>'} <executable> ...` "
                    "on the node, then restart the node so it advertises the update."
                ),
            }
        fixed_argv = configured.get("argv")
        if not isinstance(fixed_argv, list) or not all(
            isinstance(item, str) and item for item in fixed_argv
        ):
            return {"success": False, "error": f"Allowlist entry {alias!r} is invalid."}
        executable = Path(fixed_argv[0])
        if (
            not executable.is_absolute()
            or not executable.is_file()
            or not os.access(executable, os.X_OK)
        ):
            return {
                "success": False,
                "error": (
                    f"Allowlist entry {alias!r} no longer points to an absolute, "
                    "runnable executable. Recreate it with `hexis node allow "
                    f"{alias} <executable> ...`."
                ),
            }
        supplied_args = arguments.get("args") or []
        if not isinstance(supplied_args, list) or not all(
            isinstance(item, str) and len(item) <= 1000 for item in supplied_args
        ):
            return {
                "success": False,
                "error": "Node command args must be short strings.",
            }
        if supplied_args and not bool(configured.get("allow_args")):
            return {
                "success": False,
                "error": f"Host command {alias!r} permits no invocation-time arguments.",
            }
        if len(supplied_args) > 40:
            return {
                "success": False,
                "error": "At most 40 host-command arguments are allowed.",
            }
        try:
            timeout = int(arguments.get("timeout") or 30)
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "Node command timeout must be a whole number of seconds.",
            }
        timeout = min(max(timeout, 1), 120)
        try:
            returncode, stdout, stderr = await _run_process(
                [*fixed_argv, *supplied_args], timeout=timeout
            )
        except (OSError, TimeoutError) as exc:
            return {"success": False, "error": str(exc)}
        return {
            "success": returncode == 0,
            "result": {
                "command": alias,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
            "error": None if returncode == 0 else f"Host command exited {returncode}.",
        }

    if action == "screen.capture":
        cache_root = (
            Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache"))
            / "hexis"
            / "node"
            / "captures"
        )
        cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            os.chmod(cache_root, 0o700)
        fd, raw_path = tempfile.mkstemp(suffix=".png", dir=cache_root)
        os.close(fd)
        target = Path(raw_path)
        command = _screen_capture_command(target)
        if not command:
            target.unlink(missing_ok=True)
            return {
                "success": False,
                "error": (
                    "Screen capture is unavailable on this host. On macOS grant Screen "
                    "Recording permission to the terminal/service; on Linux install grim "
                    "or gnome-screenshot, then restart the node."
                ),
            }
        try:
            returncode, _stdout, stderr = await _run_process(command, timeout=30)
            if returncode != 0 or not target.exists():
                return {
                    "success": False,
                    "error": stderr or f"Screen capture exited {returncode}.",
                }
            payload = target.read_bytes()
            if not payload:
                return {
                    "success": False,
                    "error": "Screen capture completed without producing an image.",
                }
            if len(payload) > _MAX_CAPTURE_BYTES:
                return {
                    "success": False,
                    "error": (
                        f"Screen capture was {len(payload)} bytes; the safe transfer "
                        f"limit is {_MAX_CAPTURE_BYTES} bytes. Reduce display resolution and retry."
                    ),
                }
            return {
                "success": True,
                "result": {
                    "mime_type": "image/png",
                    "bytes": len(payload),
                    "data_base64": base64.b64encode(payload).decode("ascii"),
                },
                "error": None,
            }
        finally:
            target.unlink(missing_ok=True)

    if action in APPLE_NODE_ACTIONS | ONEPASSWORD_NODE_ACTIONS:
        available = set(advertised_capabilities(identity))
        if action not in available:
            return {
                "success": False,
                "error": (
                    f"Capability {action} is no longer available on this node. "
                    "Install or restore the required local app/CLI, restart "
                    "`hexis node run`, and approve any changed capability set."
                ),
            }
        try:
            timeout = int(arguments.get("timeout") or 30)
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "Node action timeout must be a whole number of seconds.",
            }
        timeout = min(max(timeout, 5), 120)
        return await execute_life_action(
            action,
            arguments,
            timeout=timeout,
            runner=_run_process,
        )

    return {"success": False, "error": f"Unsupported node action: {action}"}


async def _send_json(
    websocket: Any, payload: dict[str, Any], lock: asyncio.Lock
) -> None:
    async with lock:
        await websocket.send(json.dumps(payload))


async def _heartbeat(websocket: Any, lock: asyncio.Lock) -> None:
    while True:
        await asyncio.sleep(10)
        await _send_json(websocket, {"type": "heartbeat"}, lock)


async def run_node(
    *,
    gateway_url: str | None = None,
    reconnect: bool = True,
    status_callback: Any = print,
) -> None:
    try:
        import websockets
        from websockets.exceptions import ConnectionClosed
    except ImportError as exc:
        raise RuntimeError(
            "The node transport is not installed. Reinstall Hexis so its `websockets` "
            "dependency is present."
        ) from exc

    identity = load_node_identity()
    url = node_gateway_url(gateway_url)
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                url,
                max_size=12 * 1024 * 1024,
                open_timeout=15,
                ping_interval=20,
                ping_timeout=20,
            ) as websocket:
                challenge_message = json.loads(await websocket.recv())
                if challenge_message.get("type") != "challenge":
                    raise RuntimeError(
                        "Gateway did not begin with a signed-node challenge."
                    )
                challenge = str(challenge_message.get("challenge") or "")
                proof = {"challenge": challenge, "node_id": identity.node_id}
                await websocket.send(
                    json.dumps(
                        {
                            "type": "hello",
                            "node_id": identity.node_id,
                            "name": identity.name,
                            "public_key": identity.public_key,
                            "signature": identity.sign(proof),
                            "capabilities": advertised_capabilities(identity),
                            "metadata": {
                                "platform": platform.system(),
                                "release": platform.release(),
                                "command_aliases": sorted(identity.commands),
                            },
                        }
                    )
                )
                status = json.loads(await websocket.recv())
                if status.get("status") == "pairing_required":
                    status_callback(
                        f"Pairing approval required for {identity.name} "
                        f"(code {status.get('code')}). Open the Hexis inbox to approve or deny it."
                    )
                    status = json.loads(await websocket.recv())
                if status.get("status") != "paired":
                    raise NodeAccessDeniedError(
                        str(
                            status.get("reason")
                            or f"Node connection ended: {status.get('status')}"
                        )
                    )
                status_callback(
                    f"Node {identity.name} is paired and connected. Ctrl+C exits."
                )
                backoff = 1.0
                send_lock = asyncio.Lock()
                heartbeat_task = asyncio.create_task(_heartbeat(websocket, send_lock))
                wake_stop = threading.Event()
                wake_pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
                wake_task: asyncio.Task[Any] | None = None
                wake_session_id: str | None = None
                wake_session_last_at = 0.0

                async def submit_wake_utterance(
                    audio: bytes, detection: dict[str, Any]
                ) -> dict[str, Any]:
                    nonlocal wake_session_id, wake_session_last_at
                    if len(audio) > 4 * 1024 * 1024:
                        return {
                            "error": (
                                "The captured utterance exceeded the node's 4 MiB "
                                "transfer ceiling. Shorten it and wake Hexis again."
                            )
                        }
                    now = time.monotonic()
                    idle_minutes = min(
                        max(int(identity.wake.get("session_idle_minutes") or 15), 1),
                        120,
                    )
                    if (
                        wake_session_id is None
                        or now - wake_session_last_at > idle_minutes * 60
                    ):
                        wake_session_id = str(uuid.uuid4())
                    wake_session_last_at = now
                    request_id = str(uuid.uuid4())
                    encoded = base64.b64encode(audio).decode("ascii")
                    signed = {
                        "request_id": request_id,
                        "session_id": wake_session_id,
                        "mime_type": "audio/wav",
                        "audio_bytes": len(audio),
                        "audio_sha256": hashlib.sha256(audio).hexdigest(),
                        "audio_base64": encoded,
                        "detector_model": str(detection.get("model") or "")[:200],
                        "detector_label": str(detection.get("label") or "")[:200],
                        "detector_score": float(detection.get("score") or 0),
                    }
                    response_future = asyncio.get_running_loop().create_future()
                    wake_pending[request_id] = response_future
                    await _send_json(
                        websocket,
                        {
                            "type": "wake_utterance",
                            **signed,
                            "signature": identity.sign(signed),
                        },
                        send_lock,
                    )
                    try:
                        message = await asyncio.wait_for(response_future, timeout=300)
                    finally:
                        wake_pending.pop(request_id, None)
                    reply: dict[str, Any] = {
                        "assistant": str(message.get("assistant") or ""),
                        "transcript": str(message.get("transcript") or ""),
                    }
                    if message.get("status") != "succeeded":
                        reply["error"] = str(
                            message.get("error")
                            or "Wake turn failed without a recovery detail."
                        )
                        return reply
                    encoded_audio = str(message.get("audio_base64") or "")
                    if encoded_audio:
                        try:
                            response_audio = base64.b64decode(
                                encoded_audio, validate=True
                            )
                        except ValueError:
                            reply["error"] = (
                                "Hexis returned invalid response audio; the written "
                                "response is printed above."
                            )
                        else:
                            if len(response_audio) > 8 * 1024 * 1024:
                                reply["error"] = (
                                    "Hexis returned response audio above the node's 8 "
                                    "MiB ceiling; the written response is printed above."
                                )
                            else:
                                reply["audio"] = response_audio
                    return reply

                if identity.wake.get("enabled"):
                    from core.node_wake import WakeListener

                    listener = WakeListener(identity.wake)
                    loop = asyncio.get_running_loop()

                    def on_utterance(
                        audio: bytes, detection: dict[str, Any]
                    ) -> dict[str, Any]:
                        future = asyncio.run_coroutine_threadsafe(
                            submit_wake_utterance(audio, detection), loop
                        )
                        try:
                            return future.result(timeout=305)
                        except concurrent.futures.TimeoutError:
                            future.cancel()
                            return {
                                "error": (
                                    "Hexis did not answer the signed wake request within "
                                    "five minutes. Wake listening resumed."
                                )
                            }
                        except Exception as exc:  # connection failure is shown in place
                            return {"error": f"Wake request failed: {exc}"}

                    wake_task = asyncio.create_task(
                        asyncio.to_thread(
                            listener.run,
                            stop_event=wake_stop,
                            on_utterance=on_utterance,
                            status_callback=status_callback,
                        )
                    )
                try:
                    while True:
                        receive_task = asyncio.create_task(websocket.recv())
                        waiting = {receive_task}
                        if wake_task is not None:
                            waiting.add(wake_task)
                        done, _pending = await asyncio.wait(
                            waiting, return_when=asyncio.FIRST_COMPLETED
                        )
                        if wake_task is not None and wake_task in done:
                            receive_task.cancel()
                            await asyncio.gather(receive_task, return_exceptions=True)
                            error = wake_task.exception()
                            if error:
                                raise error
                            raise NodeWakeError(
                                "Wake listening stopped unexpectedly. Run `hexis node "
                                "wake status`, then restart the node."
                            )
                        raw = receive_task.result()
                        message = json.loads(raw)
                        if message.get("type") == "wake_response":
                            request_id = str(message.get("request_id") or "")
                            response_future = wake_pending.get(request_id)
                            if (
                                response_future is not None
                                and not response_future.done()
                            ):
                                response_future.set_result(message)
                            continue
                        if message.get("type") != "invoke":
                            continue
                        invocation_id = str(message.get("invocation_id") or "")
                        action = str(message.get("action") or "")
                        arguments = message.get("arguments")
                        if not isinstance(arguments, dict):
                            arguments = {}
                        outcome = await execute_node_action(identity, action, arguments)
                        signed = {
                            "invocation_id": invocation_id,
                            "success": bool(outcome.get("success")),
                            "result": outcome.get("result"),
                            "error": outcome.get("error"),
                        }
                        await _send_json(
                            websocket,
                            {
                                "type": "result",
                                **signed,
                                "signature": identity.sign(signed),
                            },
                            send_lock,
                        )
                finally:
                    wake_stop.set()
                    for future in wake_pending.values():
                        if not future.done():
                            future.set_exception(
                                RuntimeError(
                                    "The node connection closed before the reply."
                                )
                            )
                    if wake_task is not None:
                        try:
                            await asyncio.wait_for(wake_task, timeout=2)
                        except (asyncio.TimeoutError, Exception):
                            wake_task.cancel()
                    heartbeat_task.cancel()
                    await asyncio.gather(heartbeat_task, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            return
        except NodeAccessDeniedError:
            raise
        except NodeWakeError:
            raise
        except ConnectionClosed as exc:
            error = (
                f"Gateway connection closed ({exc.code}): {exc.reason or 'no reason'}"
            )
        except Exception as exc:  # noqa: BLE001 - daemon retries with the cause visible
            error = str(exc)
        if not reconnect:
            raise RuntimeError(error)
        status_callback(
            f"Node disconnected: {error}. Reconnecting in {backoff:g}s; Ctrl+C exits."
        )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)
