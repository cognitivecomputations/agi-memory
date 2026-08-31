"""Ed25519 identity and local policy for a headless Hexis companion node."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

_ALIAS_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,63}$")


def node_config_path() -> Path:
    root = Path(os.getenv("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return root / "hexis" / "node.json"


def canonical_payload(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def node_id_for_public_key(public_key: str) -> str:
    return hashlib.sha256(_decode(public_key)).hexdigest()


@dataclass(frozen=True)
class NodeIdentity:
    node_id: str
    name: str
    public_key: str
    private_key: str
    commands: dict[str, dict[str, Any]]
    wake: dict[str, Any]

    def sign(self, payload: dict[str, Any]) -> str:
        key = Ed25519PrivateKey.from_private_bytes(_decode(self.private_key))
        return _b64(key.sign(canonical_payload(payload)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "node_id": self.node_id,
            "name": self.name,
            "public_key": self.public_key,
            "private_key": self.private_key,
            "commands": self.commands,
            "wake": self.wake,
        }


def verify_signature(
    public_key: str,
    payload: dict[str, Any],
    signature: str,
) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(_decode(public_key)).verify(
            _decode(signature), canonical_payload(payload)
        )
        return True
    except (ValueError, TypeError, InvalidSignature):
        return False


def initialize_node_identity(
    *,
    name: str,
    path: Path | None = None,
) -> NodeIdentity:
    target = path or node_config_path()
    if target.exists():
        raise FileExistsError(
            f"Node identity already exists at {target}. Use `hexis node status`; "
            "delete it only after revoking the paired node."
        )
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Node name is required.")
    if len(clean_name) > 100:
        raise ValueError("Node name must be 100 characters or fewer.")
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_raw = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    identity = NodeIdentity(
        node_id=hashlib.sha256(public_raw).hexdigest(),
        name=clean_name,
        public_key=_b64(public_raw),
        private_key=_b64(private_raw),
        commands={},
        wake={},
    )
    save_node_identity(identity, target)
    return identity


def load_node_identity(path: Path | None = None) -> NodeIdentity:
    target = path or node_config_path()
    if not target.exists():
        raise FileNotFoundError(
            f"No node identity exists at {target}. Create one with "
            "`hexis node init --name <device-name>`."
        )
    if os.name != "nt" and stat.S_IMODE(target.stat().st_mode) & 0o077:
        raise PermissionError(
            f"Node identity at {target} is readable by other users. "
            f"Run `chmod 600 {target}` before using it."
        )
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Node config at {target} is not a JSON object.")
    public_key = str(raw.get("public_key") or "")
    private_key = str(raw.get("private_key") or "")
    expected_id = node_id_for_public_key(public_key)
    node_id = str(raw.get("node_id") or "")
    if node_id != expected_id:
        raise ValueError(
            f"Node identity at {target} failed its public-key fingerprint check. "
            "Do not run it; restore the original file or revoke and re-pair."
        )
    # Prove the private key matches rather than discovering it at handshake.
    probe = {"node_identity_probe": node_id}
    identity = NodeIdentity(
        node_id=node_id,
        name=str(raw.get("name") or "Unnamed node"),
        public_key=public_key,
        private_key=private_key,
        commands=(raw.get("commands") if isinstance(raw.get("commands"), dict) else {}),
        wake=(raw.get("wake") if isinstance(raw.get("wake"), dict) else {}),
    )
    if not verify_signature(public_key, probe, identity.sign(probe)):
        raise ValueError(f"Node private key at {target} does not match its public key.")
    return identity


def save_node_identity(identity: NodeIdentity, path: Path | None = None) -> None:
    target = path or node_config_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    fd, temporary_name = tempfile.mkstemp(prefix=".node-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(identity.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def set_node_command(
    alias: str,
    argv: list[str],
    *,
    allow_args: bool,
    replace: bool = False,
    path: Path | None = None,
) -> NodeIdentity:
    if not _ALIAS_RE.fullmatch(alias):
        raise ValueError(
            "Command alias must be 2–64 letters, digits, dots, underscores, or hyphens."
        )
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError(
            "The allowlisted command needs an executable and fixed arguments."
        )
    identity = load_node_identity(path)
    commands = dict(identity.commands)
    if alias in commands and not replace:
        raise FileExistsError(
            f"Command alias {alias!r} already exists. Use --replace to overwrite it explicitly."
        )
    executable = Path(argv[0]).expanduser()
    if not executable.is_absolute():
        resolved = shutil.which(argv[0])
        if not resolved:
            raise FileNotFoundError(
                f"Executable {argv[0]!r} was not found. Provide an absolute path or "
                "a command available on this device now."
            )
        executable = Path(resolved)
    executable = executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PermissionError(f"Allowlisted executable is not runnable: {executable}")
    commands[alias] = {
        "argv": [str(executable), *argv[1:]],
        "allow_args": bool(allow_args),
    }
    updated = NodeIdentity(
        node_id=identity.node_id,
        name=identity.name,
        public_key=identity.public_key,
        private_key=identity.private_key,
        commands=commands,
        wake=identity.wake,
    )
    save_node_identity(updated, path)
    return updated


def remove_node_command(alias: str, path: Path | None = None) -> NodeIdentity:
    identity = load_node_identity(path)
    if alias not in identity.commands:
        raise KeyError(f"Command alias {alias!r} is not allowlisted.")
    commands = dict(identity.commands)
    del commands[alias]
    updated = NodeIdentity(
        node_id=identity.node_id,
        name=identity.name,
        public_key=identity.public_key,
        private_key=identity.private_key,
        commands=commands,
        wake=identity.wake,
    )
    save_node_identity(updated, path)
    return updated


def set_node_wake(
    *,
    model_path: str,
    model_name: str,
    threshold: float = 0.5,
    input_device: str | None = None,
    max_utterance_seconds: int = 30,
    silence_ms: int = 1200,
    session_idle_minutes: int = 15,
    model_source: str = "custom",
    path: Path | None = None,
) -> NodeIdentity:
    """Explicitly enable local wake capture for future node runs."""
    identity = load_node_identity(path)
    model = Path(model_path).expanduser().resolve()
    if not model.is_file() or model.suffix.lower() not in {".onnx", ".tflite"}:
        raise FileNotFoundError(
            "Wake model must be an existing .onnx or .tflite file. Run `hexis "
            "node wake setup` to choose and download a supported model."
        )
    clean_name = str(model_name or model.stem).strip()
    if not clean_name or len(clean_name) > 200:
        raise ValueError("Wake model name must be from 1 through 200 characters.")
    score = float(threshold)
    if not 0.1 <= score <= 0.99:
        raise ValueError("Wake threshold must be from 0.10 through 0.99.")
    utterance_seconds = int(max_utterance_seconds)
    if not 5 <= utterance_seconds <= 60:
        raise ValueError("Wake utterances must be limited to 5 through 60 seconds.")
    trailing_silence = int(silence_ms)
    if not 500 <= trailing_silence <= 3000:
        raise ValueError("Wake silence cutoff must be from 500 through 3000 ms.")
    idle_minutes = int(session_idle_minutes)
    if not 1 <= idle_minutes <= 120:
        raise ValueError("Wake session idle time must be from 1 through 120 minutes.")
    wake = {
        "enabled": True,
        "model_path": str(model),
        "model_name": clean_name,
        "model_source": str(model_source or "custom")[:100],
        "threshold": score,
        "input_device": str(input_device).strip() if input_device else None,
        "max_utterance_seconds": utterance_seconds,
        "silence_ms": trailing_silence,
        "session_idle_minutes": idle_minutes,
    }
    updated = NodeIdentity(
        node_id=identity.node_id,
        name=identity.name,
        public_key=identity.public_key,
        private_key=identity.private_key,
        commands=identity.commands,
        wake=wake,
    )
    save_node_identity(updated, path)
    return updated


def disable_node_wake(path: Path | None = None) -> NodeIdentity:
    """Disable microphone activation without deleting identity or model assets."""
    identity = load_node_identity(path)
    wake = dict(identity.wake)
    wake["enabled"] = False
    updated = NodeIdentity(
        node_id=identity.node_id,
        name=identity.name,
        public_key=identity.public_key,
        private_key=identity.private_key,
        commands=identity.commands,
        wake=wake,
    )
    save_node_identity(updated, path)
    return updated
