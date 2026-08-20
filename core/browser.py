"""Open URLs in the user's browser, including from inside WSL.

Plain ``webbrowser.open`` inside WSL shells out to gio/xdg-open, which fails
with "Operation not supported": there is no Linux desktop session — the
browser lives on the Windows side of the boundary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser
from pathlib import Path

_POWERSHELL_FALLBACK = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def is_wsl() -> bool:
    """True when running inside Windows Subsystem for Linux."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _wsl_launcher(url: str) -> list[str] | None:
    """The command that opens a URL in the Windows browser from WSL.

    Prefers wslview (from wslu, present on most WSL distros), then
    PowerShell's Start-Process. cmd.exe's ``start`` is never safe here:
    it splits the URL at every ``&``, silently truncating query strings.
    """
    wslview = shutil.which("wslview")
    if wslview:
        return [wslview, url]
    powershell = shutil.which("powershell.exe")
    if not powershell and Path(_POWERSHELL_FALLBACK).exists():
        powershell = _POWERSHELL_FALLBACK
    if powershell:
        # Single-quoted PowerShell string; ' escapes as ''.
        quoted = url.replace("'", "''")
        return [powershell, "-NoProfile", "-Command", f"Start-Process '{quoted}'"]
    return None


def open_url(url: str) -> bool:
    """Open ``url`` in the user's browser; True when a launcher accepted it.

    No launcher is guaranteed to exist (headless boxes, SSH sessions), so
    callers must always print the URL as well.
    """
    if is_wsl():
        cmd = _wsl_launcher(url)
        if cmd:
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    timeout=15,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                pass
        # Fall through: $BROWSER or a real X session may still work.
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False
