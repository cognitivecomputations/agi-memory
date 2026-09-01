"""CLI lifecycle for signed, outward-only companion nodes."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def register_node_parser(
    subparsers: Any,
    db_parent: argparse.ArgumentParser,
) -> None:
    node = subparsers.add_parser(
        "node",
        help="Pair and run an outward-only companion node",
        description=(
            "Give Hexis explicitly approved access to allowlisted host commands "
            "and screen capture without opening an inbound port."
        ),
    )
    node.set_defaults(func="node_status", local_only=False, json=False)
    commands = node.add_subparsers(dest="node_command")

    init = commands.add_parser("init", help="Create this device's signed identity")
    init.add_argument("--name", required=True, help="Human-readable device name")
    init.set_defaults(func="node_init")

    status = commands.add_parser(
        "status", parents=[db_parent], help="Show local identity and paired nodes"
    )
    status.add_argument("--local-only", action="store_true", help="Do not query Hexis")
    status.add_argument("--json", action="store_true", help="Output JSON")
    status.set_defaults(func="node_status")

    allow = commands.add_parser(
        "allow", help="Allow one fixed host command under a local alias"
    )
    allow.add_argument("alias", help="Alias exposed to Hexis (not a shell string)")
    allow.add_argument(
        "argv",
        nargs="+",
        help="Executable and fixed arguments; use -- before arguments beginning with -",
    )
    allow.add_argument(
        "--allow-args",
        action="store_true",
        help="Permit invocation-time arguments in addition to the fixed argv",
    )
    allow.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing alias",
    )
    allow.set_defaults(func="node_allow")

    disallow = commands.add_parser("disallow", help="Remove one local command alias")
    disallow.add_argument("alias")
    disallow.set_defaults(func="node_disallow")

    run = commands.add_parser("run", help="Connect outward and serve approved work")
    run.add_argument(
        "--gateway",
        default=None,
        help=(
            "Hexis HTTP(S) or WS(S) base URL; otherwise HEXIS_NODE_GATEWAY_URL, "
            "HEXIS_API_URL, then localhost"
        ),
    )
    run.add_argument(
        "--once",
        action="store_true",
        help="Exit after the first disconnect instead of reconnecting",
    )
    run.set_defaults(func="node_run")

    wake = commands.add_parser(
        "wake", help="Configure explicit local wake-word listening"
    )
    wake.set_defaults(func="node_wake_status", json=False)
    wake_commands = wake.add_subparsers(dest="node_wake_command")
    wake_status = wake_commands.add_parser(
        "status", help="Show local wake configuration without opening the microphone"
    )
    wake_status.add_argument("--json", action="store_true", help="Output JSON")
    wake_status.set_defaults(func="node_wake_status")
    wake_setup = wake_commands.add_parser(
        "setup", help="Install wake support and explicitly select a model"
    )
    wake_setup.add_argument(
        "--model",
        default=None,
        help="Pretrained catalog name or absolute .onnx/.tflite model path",
    )
    wake_setup.add_argument("--threshold", type=float, default=0.5)
    wake_setup.add_argument(
        "--device", default=None, help="sounddevice input device id or name"
    )
    wake_setup.add_argument("--max-utterance-seconds", type=int, default=30)
    wake_setup.add_argument("--silence-ms", type=int, default=1200)
    wake_setup.add_argument("--session-idle-minutes", type=int, default=15)
    wake_setup.add_argument(
        "--accept-model-license",
        action="store_true",
        help="Accept the upstream pretrained-model CC BY-NC-SA 4.0 license",
    )
    wake_setup.add_argument(
        "-y", "--yes", action="store_true", help="Confirm optional package install"
    )
    wake_setup.set_defaults(func="node_wake_setup")
    wake_disable = wake_commands.add_parser(
        "disable", help="Disable wake listening for future node runs"
    )
    wake_disable.set_defaults(func="node_wake_disable")

    pairing = commands.add_parser(
        "pairing", parents=[db_parent], help="Review pending signed identities"
    )
    pairing.set_defaults(func="node_pairing_list", json=False)
    pairing_commands = pairing.add_subparsers(dest="node_pairing_command")
    pairing_list = pairing_commands.add_parser("list", help="List pairing requests")
    pairing_list.add_argument("--json", action="store_true", help="Output JSON")
    pairing_list.set_defaults(func="node_pairing_list")
    for decision in ("approve", "deny"):
        decide = pairing_commands.add_parser(
            decision, help=f"{decision.title()} one pairing request"
        )
        decide.add_argument("request", help="Pairing UUID or exact short code")
        decide.add_argument("--note", default=None)
        decide.set_defaults(func=f"node_pairing_{decision}")

    revoke = commands.add_parser(
        "revoke", parents=[db_parent], help="Revoke a paired node identity"
    )
    revoke.add_argument("node_id", help="Complete node id")
    revoke.add_argument("--reason", default=None)
    revoke.add_argument("--yes", action="store_true", help="Skip typed confirmation")
    revoke.set_defaults(func="node_revoke")

    invoke = commands.add_parser(
        "invoke", parents=[db_parent], help="Explicitly invoke a paired node"
    )
    invoke.add_argument("node_id", help="Complete node id")
    invoke.add_argument("action", choices=["system.run", "screen.capture"])
    invoke.add_argument(
        "--command",
        dest="command_alias",
        help="Local allowlist alias for system.run",
    )
    invoke.add_argument(
        "--arg",
        dest="invoke_args",
        action="append",
        default=[],
        help="One invocation-time argument; repeat as needed",
    )
    invoke.add_argument("--timeout", type=int, default=30)
    invoke.add_argument("--output", help="PNG destination for screen.capture")
    invoke.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing screen.capture output file",
    )
    invoke.add_argument("--yes", action="store_true", help="Skip typed confirmation")
    invoke.set_defaults(func="node_invoke")


def _local_payload() -> dict[str, Any]:
    from core.node_daemon import advertised_capabilities
    from core.node_identity import load_node_identity, node_config_path

    path = node_config_path()
    try:
        identity = load_node_identity(path)
    except FileNotFoundError:
        return {
            "configured": False,
            "config_path": str(path),
            "next_step": "Run `hexis node init --name <device-name>` on this device.",
        }
    return {
        "configured": True,
        "config_path": str(path),
        "node_id": identity.node_id,
        "name": identity.name,
        "public_key_fingerprint": identity.node_id,
        "capabilities": advertised_capabilities(identity),
        "commands": {
            alias: {
                "argv": entry.get("argv", []),
                "allow_args": bool(entry.get("allow_args")),
            }
            for alias, entry in sorted(identity.commands.items())
            if isinstance(entry, dict)
        },
        "wake": {
            "enabled": bool(identity.wake.get("enabled")),
            "model_name": identity.wake.get("model_name"),
            "model_path": identity.wake.get("model_path"),
            "model_source": identity.wake.get("model_source"),
            "threshold": identity.wake.get("threshold"),
            "input_device": identity.wake.get("input_device"),
            "max_utterance_seconds": identity.wake.get("max_utterance_seconds"),
            "silence_ms": identity.wake.get("silence_ms"),
            "session_idle_minutes": identity.wake.get("session_idle_minutes"),
        },
    }


def _print_local(local: dict[str, Any]) -> None:
    if not local.get("configured"):
        sys.stdout.write(f"Local node: not initialized\n  {local.get('next_step')}\n")
        return
    sys.stdout.write(
        f"Local node: {local.get('name')}\n"
        f"  id: {local.get('node_id')}\n"
        f"  identity: {local.get('config_path')}\n"
        f"  available capabilities: "
        f"{', '.join(local.get('capabilities') or []) or 'none'}\n"
    )
    commands = local.get("commands") or {}
    if not commands:
        sys.stdout.write("  host commands: none allowlisted\n")
    else:
        sys.stdout.write("  host commands:\n")
        for alias, entry in commands.items():
            suffix = (
                " + invocation args" if entry.get("allow_args") else " fixed argv only"
            )
            sys.stdout.write(f"    {alias}: {entry.get('argv')} ({suffix.strip()})\n")
    wake = local.get("wake") or {}
    if wake.get("enabled"):
        sys.stdout.write(
            "  wake word: enabled for future `hexis node run` sessions\n"
            f"    model: {wake.get('model_name')} ({wake.get('model_path')})\n"
            f"    threshold: {wake.get('threshold')}\n"
        )
    else:
        sys.stdout.write(
            "  wake word: off (the microphone will not open for wake listening)\n"
        )


async def _status(dsn: str, *, local_only: bool, as_json: bool) -> int:
    local = _local_payload()
    payload: dict[str, Any] = {"local": local, "nodes": [], "pending_pairings": []}
    server_error: str | None = None
    if not local_only:
        try:
            import asyncpg

            pool = await asyncpg.create_pool(
                dsn, min_size=1, max_size=1, timeout=5, command_timeout=10
            )
            try:
                async with pool.acquire() as conn:
                    raw_nodes = await conn.fetchval("SELECT list_hexis_nodes()")
                    raw_pending = await conn.fetchval(
                        "SELECT list_node_pairing_requests('pending', 50)"
                    )
                payload["nodes"] = _json_value(raw_nodes, [])
                payload["pending_pairings"] = _json_value(raw_pending, [])
            finally:
                await pool.close()
        except Exception as exc:  # local status remains useful on a satellite device
            server_error = str(exc)
            payload["server_error"] = server_error
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
        return 0
    _print_local(local)
    if local_only:
        return 0
    if server_error:
        sys.stdout.write(
            "Server registry: unavailable from this device. Local node policy is shown "
            "above; run this command on the Hexis host or pass --dsn to inspect pairing.\n"
        )
        return 0
    nodes = payload["nodes"]
    sys.stdout.write(f"Paired nodes: {len(nodes)}\n")
    for item in nodes:
        sys.stdout.write(
            f"  {item.get('name')}: {item.get('status')}\n"
            f"    id: {item.get('node_id')}\n"
            f"    capabilities: {', '.join(item.get('capabilities') or []) or 'none'}\n"
        )
    pending = payload["pending_pairings"]
    sys.stdout.write(f"Pending pairings: {len(pending)}\n")
    for item in pending:
        sys.stdout.write(
            f"  {item.get('name')} code={item.get('code')} "
            f"capabilities={','.join(item.get('capabilities') or []) or 'none'}\n"
        )
    return 0


def _json_value(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return fallback
    return fallback if raw is None else raw


def _init(name: str) -> int:
    from core.node_identity import initialize_node_identity, node_config_path

    identity = initialize_node_identity(name=name)
    sys.stdout.write(
        f"Created signed node identity for {identity.name}.\n"
        f"  id: {identity.node_id}\n"
        f"  stored at: {node_config_path()} (private key mode 0600)\n"
        "Next: allow only the host commands you need, then run `hexis node run "
        "--gateway <your-hexis-url>`. Pairing completes in place after approval.\n"
    )
    return 0


def _allow(alias: str, argv: list[str], *, allow_args: bool, replace: bool) -> int:
    from core.node_identity import set_node_command

    updated = set_node_command(
        alias,
        argv,
        allow_args=allow_args,
        replace=replace,
    )
    stored_argv = updated.commands[alias]["argv"]
    sys.stdout.write(
        f"Allowlisted {alias!r} on {updated.name}: {stored_argv!r}. "
        f"Invocation-time args are {'allowed' if allow_args else 'blocked'}.\n"
        "Restart `hexis node run` if it is already connected so the updated "
        "capability metadata is advertised.\n"
    )
    return 0


def _disallow(alias: str) -> int:
    from core.node_identity import remove_node_command

    updated = remove_node_command(alias)
    sys.stdout.write(
        f"Removed host-command alias {alias!r} from {updated.name}. "
        "Restart a running node to advertise the change.\n"
    )
    return 0


async def _run(gateway: str | None, *, reconnect: bool) -> int:
    from core.node_daemon import node_gateway_url, run_node
    from core.node_identity import load_node_identity

    identity = load_node_identity()
    url = node_gateway_url(gateway)
    sys.stdout.write(
        f"Connecting outward to {url} as {identity.name} ({identity.node_id[:12]}…).\n"
    )
    if identity.wake.get("enabled"):
        sys.stdout.write(
            "Wake listening is explicitly enabled in this node's local policy. "
            "The microphone opens only after signed pairing succeeds; Ctrl+C stops it.\n"
        )
    await run_node(gateway_url=gateway, reconnect=reconnect)
    return 0


def _wake_support_installed() -> bool:
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None
        for name in ("openwakeword", "sounddevice")
    )


def _wake_requirements() -> list[str]:
    from importlib import metadata

    from core.node_wake import WAKE_REQUIREMENTS

    try:
        installed = metadata.requires("hexis") or []
    except metadata.PackageNotFoundError:
        installed = []
    derived: list[str] = []
    for requirement in installed:
        lowered = requirement.lower()
        if 'extra == "wake"' not in lowered and "extra == 'wake'" not in lowered:
            continue
        derived.append(requirement.partition(";")[0].strip())
    return derived or list(WAKE_REQUIREMENTS)


def _install_wake_support(*, yes: bool) -> bool:
    if _wake_support_installed():
        return True
    if not yes:
        sys.stdout.write(
            "Wake listening needs the optional openWakeWord detector and sounddevice "
            "audio layer. This changes only the current Hexis Python environment; "
            "it does not open the microphone or download a wake model.\n"
        )
        try:
            answer = input("Install local wake support now? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            return False
        if answer.strip().lower() not in {"y", "yes"}:
            sys.stdout.write("No changes made.\n")
            return False
    uv = shutil.which("uv")
    if not uv:
        sys.stderr.write(
            "uv is required to add wake support to the exact Hexis environment. "
            "Install uv, then rerun `hexis node wake setup`; no microphone was opened.\n"
        )
        return False
    requirements = _wake_requirements()
    sys.stdout.write("Installing local wake support: " + ", ".join(requirements) + "\n")
    try:
        result = subprocess.run(
            [uv, "pip", "install", "--python", sys.executable, *requirements],
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write(
            f"Wake support could not be installed ({exc}). Nothing was enabled and "
            "the microphone was not opened.\n"
        )
        return False
    if result.returncode != 0:
        system_step = (
            " On Linux, install the PortAudio runtime (often libportaudio2), then retry."
            if sys.platform.startswith("linux")
            else ""
        )
        sys.stderr.write(
            f"uv exited with code {result.returncode}; wake listening remains off."
            f"{system_step}\n"
        )
        return False
    import importlib

    importlib.invalidate_caches()
    if not _wake_support_installed():
        sys.stderr.write(
            "The installer completed but wake dependencies are still unavailable in "
            "this Hexis environment. Wake listening remains off.\n"
        )
        return False
    return True


def _select_wake_model(raw_model: str | None) -> str | None:
    if raw_model:
        return str(raw_model).strip()
    from core.node_wake import pretrained_model_catalog

    names = sorted(pretrained_model_catalog())
    if not sys.stdin.isatty():
        sys.stderr.write(
            "Choose a wake model explicitly in non-interactive setup. Available "
            f"pretrained names: {', '.join(names)}. Retry with `hexis node wake "
            "setup --model <name> --accept-model-license -y`, or pass an absolute "
            "custom .onnx/.tflite path.\n"
        )
        return None
    sys.stdout.write("Available pretrained wake models from the installed package:\n")
    for index, name in enumerate(names, start=1):
        sys.stdout.write(f"  {index}. {name}\n")
    try:
        answer = input("Choose a model number (blank cancels): ").strip()
        selected_index = int(answer)
    except (EOFError, KeyboardInterrupt, ValueError):
        sys.stdout.write("No wake model selected; wake listening remains off.\n")
        return None
    if not 1 <= selected_index <= len(names):
        sys.stderr.write("That model number is not in the displayed catalog.\n")
        return None
    return names[selected_index - 1]


def _accept_pretrained_model_license(*, accepted: bool) -> bool:
    if accepted:
        return True
    notice = (
        "openWakeWord's bundled pretrained models are licensed CC BY-NC-SA 4.0 "
        "and are English-only. This may not permit commercial use. The detector "
        "code has a separate Apache 2.0 license."
    )
    sys.stdout.write(notice + "\n")
    if not sys.stdin.isatty():
        sys.stderr.write(
            "No model was downloaded. Review the upstream terms, then add "
            "--accept-model-license, or pass your own licensed model path.\n"
        )
        return False
    try:
        answer = input("Download this pretrained model under those terms? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n")
        return False
    return answer.strip().lower() in {"y", "yes"}


def _wake_setup(args: Any) -> int:
    from core.node_identity import load_node_identity, set_node_wake

    # Prove an identity exists before changing this environment.
    load_node_identity()
    if not _install_wake_support(yes=bool(args.yes)):
        return 1
    selected = _select_wake_model(args.model)
    if not selected:
        return 1
    candidate = Path(selected).expanduser()
    if candidate.is_file() or candidate.is_absolute():
        model_path = candidate.resolve()
        model_name = model_path.stem
        model_source = "custom"
    else:
        if not _accept_pretrained_model_license(
            accepted=bool(args.accept_model_license)
        ):
            return 1
        from core.node_wake import download_pretrained_model

        model_path = download_pretrained_model(selected)
        model_name = selected
        model_source = "openwakeword_pretrained_cc_by_nc_sa_4"
    updated = set_node_wake(
        model_path=str(model_path),
        model_name=model_name,
        threshold=args.threshold,
        input_device=args.device,
        max_utterance_seconds=args.max_utterance_seconds,
        silence_ms=args.silence_ms,
        session_idle_minutes=args.session_idle_minutes,
        model_source=model_source,
    )
    sys.stdout.write(
        f"Wake listening is enabled in local policy for {updated.name} using "
        f"{model_name!r}. The microphone is still off.\n"
        "Next: enable paired-node wake turns in Settings → Voice, ensure voice-note "
        "transcription and local speech are enabled, then run `hexis node run "
        "--gateway <your-hexis-url>`. Pairing will advertise audio.wake explicitly.\n"
    )
    return 0


def _wake_status(*, as_json: bool) -> int:
    local = _local_payload()
    wake = local.get("wake") or {}
    payload = {
        **wake,
        "enabled": bool(wake.get("enabled")),
        "configured": bool(local.get("configured")),
        "microphone_active": False,
        "note": (
            "Status is read-only. The microphone opens only inside a paired `hexis node run` process."
        ),
    }
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
    else:
        state = "enabled for future node runs" if wake.get("enabled") else "off"
        sys.stdout.write(f"Local wake listening: {state}\n")
        if wake.get("model_name"):
            sys.stdout.write(
                f"  model: {wake.get('model_name')}\n"
                f"  path: {wake.get('model_path')}\n"
                f"  threshold: {wake.get('threshold')}\n"
            )
        sys.stdout.write(
            "  microphone: not opened by this status command\n"
            "  control: Ctrl+C stops a running node; `hexis node wake disable` "
            "prevents listening on its next start\n"
        )
    return 0


def _wake_disable() -> int:
    from core.node_identity import disable_node_wake

    identity = disable_node_wake()
    sys.stdout.write(
        f"Wake listening is disabled in local policy for {identity.name}. No model "
        "or identity was deleted. If `hexis node run` is active, press Ctrl+C in "
        "that foreground process to close its microphone now.\n"
    )
    return 0


async def _pairing_list(dsn: str, *, as_json: bool) -> int:
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT list_node_pairing_requests('pending', 50)"
            )
        pending = _json_value(raw, [])
    finally:
        await pool.close()
    if as_json:
        sys.stdout.write(json.dumps(pending, indent=2, default=str) + "\n")
    elif not pending:
        sys.stdout.write("No pending node pairing requests.\n")
    else:
        for item in pending:
            sys.stdout.write(
                f"{item.get('code')}  {item.get('name')}  {item.get('node_id')}\n"
                f"  capabilities: {', '.join(item.get('capabilities') or []) or 'none'}\n"
                f"  expires: {item.get('expires_at')}\n"
            )
        sys.stdout.write(
            "Approve with `hexis node pairing approve <code>` or deny with "
            "`hexis node pairing deny <code>`.\n"
        )
    return 0


async def _pairing_decide(
    dsn: str, request: str, decision: str, note: str | None
) -> int:
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT decide_node_pairing($1, $2, 'cli', $3)",
                request,
                decision,
                note,
            )
        result = _json_value(raw, {})
    finally:
        await pool.close()
    if result.get("status") not in {"approved", "denied"}:
        sys.stderr.write(str(result.get("reason") or result) + "\n")
        return 1
    sys.stdout.write(
        f"Pairing {result.get('status')} for {result.get('name') or result.get('node_id')}.\n"
        f"{result.get('next_step') or ''}\n"
    )
    return 0


def _typed_confirmation(prompt: str, phrase: str) -> bool:
    if not sys.stdin.isatty():
        sys.stderr.write(
            "Refusing without an interactive confirmation. Retry with --yes.\n"
        )
        return False
    answer = input(f"{prompt}\nType '{phrase}' to confirm: ")
    return answer.strip() == phrase


async def _revoke(
    dsn: str,
    node_id: str,
    reason: str | None,
    *,
    yes: bool,
) -> int:
    if not yes and not _typed_confirmation(
        "Revocation stops queued work and this identity cannot reconnect.",
        f"revoke {node_id[:12]}",
    ):
        return 1
    import asyncpg

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT revoke_hexis_node($1, 'cli', $2)", node_id, reason
            )
        result = _json_value(raw, {})
    finally:
        await pool.close()
    if not result.get("revoked"):
        sys.stderr.write(
            "Node was not found or was already revoked. Run `hexis node status` to "
            "copy the complete active node id.\n"
        )
        return 1
    sys.stdout.write(f"Revoked node {node_id}. Queued work was cancelled.\n")
    return 0


def _capture_output_path(raw: str | None, invocation_id: str) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    cache = Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return cache / "hexis" / "node" / "captures" / f"{invocation_id}.png"


async def _invoke(args: Any, dsn: str) -> int:
    action = args.action
    capture_target: Path | None = None
    if action == "system.run" and not args.command_alias:
        sys.stderr.write("system.run requires --command <local-alias>.\n")
        return 1
    if action == "screen.capture" and (args.command_alias or args.invoke_args):
        sys.stderr.write("screen.capture does not accept --command or --arg.\n")
        return 1
    if action != "screen.capture" and args.overwrite:
        sys.stderr.write("--overwrite applies only to screen.capture.\n")
        return 1
    if action == "screen.capture" and args.output:
        capture_target = _capture_output_path(args.output, "")
        if capture_target.exists() and not args.overwrite:
            sys.stderr.write(
                f"Refusing to overwrite existing screen capture: {capture_target}\n"
                "Choose another --output path or retry with --overwrite.\n"
            )
            return 1
    if not 5 <= args.timeout <= 120:
        sys.stderr.write("--timeout must be from 5 through 120 seconds.\n")
        return 1
    summary = f"Invoke {action} on node {args.node_id}" + (
        f" using alias {args.command_alias!r} and args {args.invoke_args!r}"
        if args.command_alias
        else ""
    )
    if not args.yes and not _typed_confirmation(summary, "invoke"):
        return 1

    import asyncpg
    from services.node_gateway import request_node_invocation

    node_arguments: dict[str, Any] = {"timeout": args.timeout}
    if action == "system.run":
        node_arguments.update({"command": args.command_alias, "args": args.invoke_args})
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        result = await request_node_invocation(
            pool,
            node_id=args.node_id,
            action=action,
            arguments=node_arguments,
            requested_by="operator:cli",
            timeout_seconds=args.timeout,
            metadata={"explicit_cli_invocation": True},
        )
    finally:
        await pool.close()
    if result.get("status") != "succeeded":
        sys.stderr.write(
            str(result.get("error") or result.get("reason") or result) + "\n"
        )
        return 1
    output = result.get("result") if isinstance(result.get("result"), dict) else {}
    if action == "screen.capture":
        try:
            content = base64.b64decode(
                str(output.get("data_base64") or ""), validate=True
            )
        except Exception:
            sys.stderr.write("The node returned an invalid screen capture.\n")
            return 1
        target = capture_target or _capture_output_path(
            None, str(result.get("invocation_id"))
        )
        if target.exists() and not args.overwrite:
            sys.stderr.write(
                f"Refusing to overwrite existing screen capture: {target}\n"
                "Choose another --output path or retry with --overwrite.\n"
            )
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        os.chmod(target, 0o600)
        sys.stdout.write(f"Screen capture saved to {target}\n")
    else:
        sys.stdout.write(json.dumps(output, indent=2, default=str) + "\n")
    return 0


def dispatch_node_command(func: str, args: Any, dsn: str) -> int | None:
    if not func.startswith("node_"):
        return None
    if func == "node_init":
        return _init(args.name)
    if func == "node_status":
        return asyncio.run(
            _status(
                dsn,
                local_only=getattr(args, "local_only", False),
                as_json=getattr(args, "json", False),
            )
        )
    if func == "node_allow":
        return _allow(
            args.alias,
            args.argv,
            allow_args=args.allow_args,
            replace=args.replace,
        )
    if func == "node_disallow":
        return _disallow(args.alias)
    if func == "node_run":
        return asyncio.run(_run(args.gateway, reconnect=not args.once))
    if func == "node_wake_status":
        return _wake_status(as_json=getattr(args, "json", False))
    if func == "node_wake_setup":
        return _wake_setup(args)
    if func == "node_wake_disable":
        return _wake_disable()
    if func == "node_pairing_list":
        return asyncio.run(_pairing_list(dsn, as_json=getattr(args, "json", False)))
    if func in {"node_pairing_approve", "node_pairing_deny"}:
        return asyncio.run(
            _pairing_decide(
                dsn,
                args.request,
                "approve" if func.endswith("approve") else "deny",
                args.note,
            )
        )
    if func == "node_revoke":
        return asyncio.run(_revoke(dsn, args.node_id, args.reason, yes=args.yes))
    if func == "node_invoke":
        return asyncio.run(_invoke(args, dsn))
    return None
