"""Anthropic setup-token (Claude Code CLI subscription) auth module."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Validation constants (from OpenClaw src/commands/auth-token.ts)
ANTHROPIC_SETUP_TOKEN_PREFIX = "sk-ant-oat01-"
ANTHROPIC_SETUP_TOKEN_MIN_LENGTH = 80

ANTHROPIC_SETUP_TOKEN_CONFIG_KEY = "token.anthropic_setup_token"


@dataclass(frozen=True)
class AnthropicSetupTokenCredentials:
    token: str


def validate_setup_token(token: str) -> str | None:
    """Return an error message if the token is invalid, else None."""
    if not token:
        return "Token is empty."
    if not token.startswith(ANTHROPIC_SETUP_TOKEN_PREFIX):
        return f"Token must start with '{ANTHROPIC_SETUP_TOKEN_PREFIX}'."
    if len(token) < ANTHROPIC_SETUP_TOKEN_MIN_LENGTH:
        return f"Token is too short (min {ANTHROPIC_SETUP_TOKEN_MIN_LENGTH} chars)."
    return None


def credentials_to_dict(creds: AnthropicSetupTokenCredentials) -> dict[str, Any]:
    return {"token": creds.token}


def credentials_from_value(value: Any) -> AnthropicSetupTokenCredentials | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, dict):
        return None
    token = value.get("token")
    if not isinstance(token, str) or not token:
        return None
    return AnthropicSetupTokenCredentials(token=token)


def load_credentials() -> AnthropicSetupTokenCredentials | None:
    from core.auth.store import load_auth
    return credentials_from_value(load_auth(ANTHROPIC_SETUP_TOKEN_CONFIG_KEY))


def save_credentials(creds: AnthropicSetupTokenCredentials) -> None:
    from core.auth.store import save_auth
    save_auth(ANTHROPIC_SETUP_TOKEN_CONFIG_KEY, credentials_to_dict(creds))


def delete_credentials() -> None:
    from core.auth.store import delete_auth
    delete_auth(ANTHROPIC_SETUP_TOKEN_CONFIG_KEY)


async def resolve_anthropic_token() -> tuple[str | None, str]:
    """Resolve an Anthropic subscription token from Hexis's OWN store only.

    Hexis manages its own Anthropic auth and deliberately does **not** read
    Claude Code's credentials (``~/.claude/.credentials.json`` / the macOS
    Keychain) or environment tokens. Populate the store with:
        hexis auth anthropic setup-token     # paste a setup token

    Returns (token, auth_mode): auth_mode is "setup-token" for a stored
    subscription token, or "" if nothing is stored.

    Note: plain API-key auth for ``provider=anthropic`` is handled separately by
    ``normalize_llm_config`` (via ``api_key_env`` / the ``ANTHROPIC_API_KEY``
    env var) and does not go through this function.
    """
    creds = load_credentials()
    if creds:
        return creds.token, "setup-token"
    return None, ""


# ---------------------------------------------------------------------------
# Claude Code credential auto-detection (status display only — never consumed)
# ---------------------------------------------------------------------------

def _read_claude_code_keychain() -> dict[str, Any] | None:
    """Read Claude Code OAuth from macOS Keychain (Claude Code >= 2.1.114)."""
    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None

    oauth_data = data.get("claudeAiOauth")
    if isinstance(oauth_data, dict) and oauth_data.get("accessToken"):
        return {
            "accessToken": oauth_data["accessToken"],
            "refreshToken": oauth_data.get("refreshToken", ""),
            "expiresAt": oauth_data.get("expiresAt", 0),
            "source": "macos_keychain",
        }
    return None


def _read_claude_code_file() -> dict[str, Any] | None:
    """Read Claude Code OAuth from ~/.claude/.credentials.json."""
    cred_path = Path.home() / ".claude" / ".credentials.json"
    try:
        data = json.loads(cred_path.read_text(encoding="utf-8"))
        oauth_data = data.get("claudeAiOauth")
        if isinstance(oauth_data, dict) and oauth_data.get("accessToken"):
            return {
                "accessToken": oauth_data["accessToken"],
                "refreshToken": oauth_data.get("refreshToken", ""),
                "expiresAt": oauth_data.get("expiresAt", 0),
                "source": "claude_code_file",
            }
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return None


def read_claude_code_credentials() -> dict[str, Any] | None:
    """Read Claude Code OAuth credentials (Keychain first, then file)."""
    return _read_claude_code_keychain() or _read_claude_code_file()
