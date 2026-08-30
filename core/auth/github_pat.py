"""GitHub personal access token auth module (#102).

Interim credential flow while a real first-class OAuth/device-code GitHub
connector isn't built yet -- the issue's own Non-Goals explicitly allow this:
"Do not require GitHub OAuth on day one if a fine-grained PAT bootstrap is
faster, but make the PAT path productized and resumable." The token is
collected via a terminal prompt (never chat) and verified against GitHub's
own API before being stored, mirroring core/auth/anthropic_setup_token.py.

Hexis manages this credential in its own store ONLY. It deliberately never
reads `gh auth token`, a `GITHUB_TOKEN`/`GITHUB_PERSONAL_ACCESS_TOKEN`
environment variable, or any other ambient credential -- the issue's other
explicit Non-Goal ("do not silently consume gh auth ... just because it
exists").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

GITHUB_PAT_CONFIG_KEY = "token.github_pat"
GITHUB_FINE_GRAINED_PAT_PREFIX = "github_pat_"
GITHUB_CLASSIC_PAT_PREFIX = "ghp_"


@dataclass(frozen=True)
class GithubPatCredentials:
    token: str
    login: str | None = None


def validate_pat_format(token: str) -> str | None:
    """Return an error message if the token is obviously malformed, else None."""
    if not token:
        return "Token is empty."
    if not (
        token.startswith(GITHUB_FINE_GRAINED_PAT_PREFIX)
        or token.startswith(GITHUB_CLASSIC_PAT_PREFIX)
    ):
        return (
            f"Token must start with '{GITHUB_FINE_GRAINED_PAT_PREFIX}' (fine-grained, "
            f"preferred) or '{GITHUB_CLASSIC_PAT_PREFIX}' (classic)."
        )
    return None


async def verify_pat(token: str) -> tuple[str | None, str | None]:
    """Call GitHub's own API to confirm the token actually works.

    Returns (login, error) -- exactly one is set.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        "https://api.github.com/user",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "Hexis",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return None, "GitHub rejected the token (401 Unauthorized) — check it wasn't mistyped or revoked."
        return None, f"GitHub API error: HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 -- network/DNS/timeout, report verbatim
        return None, f"Could not reach GitHub API: {exc}"

    login = data.get("login") if isinstance(data, dict) else None
    if not login:
        return None, "GitHub API response did not include a login — unexpected response shape."
    return str(login), None


def credentials_to_dict(creds: GithubPatCredentials) -> dict[str, Any]:
    return {"token": creds.token, "login": creds.login}


def credentials_from_value(value: Any) -> GithubPatCredentials | None:
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
    login = value.get("login")
    return GithubPatCredentials(token=token, login=login if isinstance(login, str) else None)


def load_credentials() -> GithubPatCredentials | None:
    from core.auth.store import load_auth
    return credentials_from_value(load_auth(GITHUB_PAT_CONFIG_KEY))


def save_credentials(creds: GithubPatCredentials) -> None:
    from core.auth.store import save_auth
    save_auth(GITHUB_PAT_CONFIG_KEY, credentials_to_dict(creds))


def delete_credentials() -> None:
    from core.auth.store import delete_auth
    delete_auth(GITHUB_PAT_CONFIG_KEY)


def resolve_github_token() -> tuple[str | None, str]:
    """Resolve a GitHub token from Hexis's OWN store only.

    Populate the store with:
        hexis auth github setup-token     # paste a fine-grained or classic PAT

    Returns (token, login): login is the verified GitHub username the token
    was last confirmed against, or "" if nothing is stored. Never falls back
    to `gh auth token`, `GITHUB_TOKEN`, or `GITHUB_PERSONAL_ACCESS_TOKEN` from
    the process environment -- see the module docstring.
    """
    creds = load_credentials()
    if creds:
        return creds.token, (creds.login or "")
    return None, ""
