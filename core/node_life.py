"""Fixed, structured everyday-life capabilities executed on a companion node.

No model-authored AppleScript or shell text enters these paths. The node selects
one source-controlled script for one typed action, validates all arguments, and
uses direct argv execution. 1Password secret values can be copied on the node but
never cross the gateway.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from core.node_actions import APPLE_NODE_ACTIONS, ONEPASSWORD_NODE_ACTIONS


_MAX_TEXT = 20_000
_MAX_SECRET_BYTES = 65_536
_OP_REFERENCE_RE = re.compile(r"^op://[^/\s]+/[^/\s]+/[^/\s]+$")
_CLIPBOARD_TASKS: set[asyncio.Task[None]] = set()

ProcessRunner = Callable[..., Awaitable[tuple[int, str, str]]]


_REMINDERS_LIST_JXA = r"""
function iso(value) { try { return value ? new Date(value).toISOString() : null; } catch (_) { return null; } }
function run(argv) {
  const wanted = argv[0] || "";
  const includeCompleted = argv[1] === "true";
  const limit = Math.max(1, Math.min(100, Number(argv[2] || 25)));
  const app = Application("Reminders");
  const out = [];
  for (const list of app.lists()) {
    const listName = String(list.name());
    if (wanted && listName !== wanted) continue;
    for (const reminder of list.reminders()) {
      const completed = Boolean(reminder.completed());
      if (!includeCompleted && completed) continue;
      out.push({
        id: String(reminder.id()), title: String(reminder.name()), list: listName,
        completed: completed, due_at: iso(reminder.dueDate()),
        notes_preview: String(reminder.body() || "").slice(0, 300)
      });
      if (out.length >= limit) return JSON.stringify({reminders: out, truncated: true});
    }
  }
  return JSON.stringify({reminders: out, truncated: false});
}
"""

_REMINDERS_CREATE_JXA = r"""
function run(argv) {
  const title = argv[0], listName = argv[1] || "", notes = argv[2] || "", due = argv[3] || "";
  const app = Application("Reminders");
  let target = app.defaultList();
  if (listName) {
    const matches = app.lists.whose({name: listName})();
    if (!matches.length) throw new Error("Reminder list not found: " + listName);
    target = matches[0];
  }
  const properties = {name: title};
  if (notes) properties.body = notes;
  if (due) properties.dueDate = new Date(due);
  const reminder = app.Reminder(properties);
  target.reminders.push(reminder);
  return JSON.stringify({created: true, id: String(reminder.id()), title: title, list: String(target.name())});
}
"""

_NOTES_SEARCH_JXA = r"""
function run(argv) {
  const query = String(argv[0] || "").toLowerCase(), wantedFolder = argv[1] || "";
  const limit = Math.max(1, Math.min(100, Number(argv[2] || 25)));
  const app = Application("Notes"), out = [];
  for (const account of app.accounts()) for (const folder of account.folders()) {
    const folderName = String(folder.name());
    if (wantedFolder && folderName !== wantedFolder) continue;
    for (const note of folder.notes()) {
      const title = String(note.name() || ""), body = String(note.plaintext() || "");
      if (query && !(title + "\n" + body).toLowerCase().includes(query)) continue;
      out.push({id: String(note.id()), title: title, folder: folderName, preview: body.slice(0, 500)});
      if (out.length >= limit) return JSON.stringify({notes: out, truncated: true});
    }
  }
  return JSON.stringify({notes: out, truncated: false});
}
"""

_NOTES_CREATE_JXA = r"""
function escapeHtml(value) { return String(value).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\n/g,"<br>"); }
function run(argv) {
  const title = argv[0], body = argv[1], wantedFolder = argv[2] || "";
  const app = Application("Notes");
  let target = app.defaultAccount().defaultFolder();
  if (wantedFolder) {
    let found = null;
    for (const account of app.accounts()) {
      const matches = account.folders.whose({name: wantedFolder})();
      if (matches.length) { found = matches[0]; break; }
    }
    if (!found) throw new Error("Notes folder not found: " + wantedFolder);
    target = found;
  }
  const note = app.Note({name: title, body: "<h1>" + escapeHtml(title) + "</h1><br>" + escapeHtml(body)});
  target.notes.push(note);
  return JSON.stringify({created: true, id: String(note.id()), title: title, folder: String(target.name())});
}
"""

_CALENDAR_LIST_JXA = r"""
function run(argv) {
  const start = new Date(argv[0]), end = new Date(argv[1]), wanted = argv[2] || "";
  const limit = Math.max(1, Math.min(100, Number(argv[3] || 25)));
  const app = Application("Calendar"), out = [];
  for (const calendar of app.calendars()) {
    const calendarName = String(calendar.name());
    if (wanted && calendarName !== wanted) continue;
    const events = calendar.events.whose({startDate: {_greaterThanEquals: start, _lessThan: end}})();
    for (const event of events) {
      out.push({
        id: String(event.uid()), title: String(event.summary()), calendar: calendarName,
        start_at: new Date(event.startDate()).toISOString(), end_at: new Date(event.endDate()).toISOString(),
        location: String(event.location() || ""), all_day: Boolean(event.alldayEvent())
      });
      if (out.length >= limit) return JSON.stringify({events: out, truncated: true});
    }
  }
  out.sort((a,b) => a.start_at.localeCompare(b.start_at));
  return JSON.stringify({events: out, truncated: false});
}
"""

_CALENDAR_CREATE_JXA = r"""
function run(argv) {
  const title = argv[0], start = new Date(argv[1]), end = new Date(argv[2]);
  const wanted = argv[3] || "", location = argv[4] || "", notes = argv[5] || "";
  const app = Application("Calendar");
  let target = null;
  if (wanted) {
    const matches = app.calendars.whose({name: wanted})();
    if (matches.length) target = matches[0];
  } else {
    const writable = app.calendars().filter(c => Boolean(c.writable()));
    if (writable.length) target = writable[0];
  }
  if (!target) throw new Error(wanted ? "Writable calendar not found: " + wanted : "No writable calendar is available.");
  const properties = {summary: title, startDate: start, endDate: end};
  if (location) properties.location = location;
  if (notes) properties.description = notes;
  const event = app.Event(properties);
  target.events.push(event);
  return JSON.stringify({created: true, id: String(event.uid()), title: title, calendar: String(target.name()), start_at: start.toISOString(), end_at: end.toISOString()});
}
"""


def _text(value: Any, *, label: str, maximum: int, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{label} is required.")
    if len(text) > maximum:
        raise ValueError(f"{label} must be {maximum} characters or fewer.")
    return text


def _limit(value: Any, default: int = 25) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("limit must be a whole number from 1 through 100.")
    result = int(value)
    if not 1 <= result <= 100:
        raise ValueError("limit must be a whole number from 1 through 100.")
    return result


def _iso(value: Any, *, label: str) -> str:
    text = _text(value, label=label, maximum=80, required=True)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{label} must be an ISO 8601 date-time with timezone."
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone offset.")
    return parsed.isoformat()


def detect_life_capabilities() -> list[str]:
    """Derive capabilities from the host that will actually execute them."""
    capabilities: list[str] = []
    if platform.system() == "Darwin" and shutil.which("osascript"):
        capabilities.extend(
            sorted(APPLE_NODE_ACTIONS - {"apple.shortcuts.list", "apple.shortcuts.run"})
        )
    if platform.system() == "Darwin" and shutil.which("shortcuts"):
        capabilities.extend(["apple.shortcuts.list", "apple.shortcuts.run"])
    if shutil.which("op"):
        capabilities.append("onepassword.items")
        if platform.system() == "Darwin" and (
            shutil.which("pbcopy") or Path("/usr/bin/pbcopy").exists()
        ):
            capabilities.append("onepassword.copy")
    return sorted(capabilities)


async def _jxa(
    runner: ProcessRunner,
    script: str,
    args: list[str],
    *,
    timeout: int,
) -> dict[str, Any]:
    executable = shutil.which("osascript") or "/usr/bin/osascript"
    returncode, stdout, stderr = await runner(
        [executable, "-l", "JavaScript", "-e", script, "--", *args],
        timeout=timeout,
    )
    if returncode != 0:
        detail = stderr.strip()[:1000] or f"osascript exited {returncode}."
        return {"success": False, "error": detail}
    try:
        payload = json.loads(stdout.strip())
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": "The Apple app returned an unreadable result. Nothing further was changed.",
        }
    return {"success": True, "result": payload, "error": None}


async def _shortcuts(
    runner: ProcessRunner,
    action: str,
    arguments: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    executable = shutil.which("shortcuts")
    if not executable:
        return {
            "success": False,
            "error": "Apple Shortcuts is unavailable on this node. Update macOS, then restart `hexis node run`.",
        }
    if action == "apple.shortcuts.list":
        argv = [executable, "list"]
    else:
        name = _text(arguments.get("name"), label="name", maximum=200, required=True)
        input_text = _text(arguments.get("input"), label="input", maximum=_MAX_TEXT)
        argv = [executable, "run", name]
        if input_text:
            argv.extend(["--input-path", "-"])
            return {
                "success": False,
                "error": (
                    "This node does not pass text through a shell or temporary input file. "
                    "Run the Shortcut without input or create a locally allowlisted fixed command for that workflow."
                ),
            }
    returncode, stdout, stderr = await runner(argv, timeout=timeout)
    if returncode != 0:
        return {
            "success": False,
            "error": stderr.strip()[:1000] or f"Shortcuts exited {returncode}.",
        }
    if action == "apple.shortcuts.list":
        names = [line.strip() for line in stdout.splitlines() if line.strip()][:100]
        return {"success": True, "result": {"shortcuts": names}, "error": None}
    return {
        "success": True,
        "result": {
            "ran": True,
            "name": arguments.get("name"),
            "output": stdout[:_MAX_TEXT],
        },
        "error": None,
    }


async def _op_items(
    runner: ProcessRunner,
    arguments: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    executable = shutil.which("op")
    if not executable:
        return {
            "success": False,
            "error": "1Password CLI is not installed on this node. Install `op`, sign in locally, then restart `hexis node run`.",
        }
    vault = _text(arguments.get("vault"), label="vault", maximum=200)
    query = _text(arguments.get("query"), label="query", maximum=200).casefold()
    limit = _limit(arguments.get("limit"))
    argv = [executable, "item", "list", "--format", "json"]
    if vault:
        argv.extend(["--vault", vault])
    returncode, stdout, stderr = await runner(argv, timeout=timeout)
    if returncode != 0:
        return {
            "success": False,
            "error": stderr.strip()[:1000]
            or "1Password CLI could not list items. Run `op signin` on the node and retry.",
        }
    try:
        items = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": "1Password CLI returned unreadable item metadata.",
        }
    safe: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        if query and query not in title.casefold():
            continue
        vault_value = item.get("vault") if isinstance(item.get("vault"), dict) else {}
        safe.append(
            {
                "id": str(item.get("id") or ""),
                "title": title,
                "category": str(item.get("category") or ""),
                "vault": str(vault_value.get("name") or vault_value.get("id") or ""),
                "updated_at": item.get("updated_at"),
            }
        )
        if len(safe) >= limit:
            break
    return {
        "success": True,
        "result": {"items": safe, "secrets_included": False},
        "error": None,
    }


async def _clipboard_clear_after(
    secret_digest: str,
    seconds: int,
    pbcopy: str,
    pbpaste: str | None,
) -> None:
    await asyncio.sleep(seconds)
    if not pbpaste:
        return
    proc = await asyncio.create_subprocess_exec(
        pbpaste,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    current, _ = await proc.communicate()
    if proc.returncode != 0 or hashlib.sha256(current).hexdigest() != secret_digest:
        return
    clear = await asyncio.create_subprocess_exec(
        pbcopy,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await clear.communicate(b"")


async def _op_copy(arguments: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    executable = shutil.which("op")
    pbcopy = shutil.which("pbcopy") or (
        "/usr/bin/pbcopy" if Path("/usr/bin/pbcopy").exists() else None
    )
    if not executable or not pbcopy:
        return {
            "success": False,
            "error": "1Password local copy needs both `op` and macOS `pbcopy` on this node. Install/sign in to 1Password CLI, then restart `hexis node run`.",
        }
    reference = _text(
        arguments.get("secret_ref"), label="secret_ref", maximum=500, required=True
    )
    if not _OP_REFERENCE_RE.fullmatch(reference):
        return {
            "success": False,
            "error": "secret_ref must be an exact op://vault/item/field reference; item values are never accepted.",
        }
    seconds = int(arguments.get("clipboard_seconds") or 60)
    if not 10 <= seconds <= 300:
        return {
            "success": False,
            "error": "clipboard_seconds must be from 10 through 300.",
        }
    proc = await asyncio.create_subprocess_exec(
        executable,
        "read",
        reference,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "success": False,
            "error": f"1Password CLI timed out after {timeout} seconds. Sign in locally and retry.",
        }
    if proc.returncode != 0:
        return {
            "success": False,
            "error": stderr.decode("utf-8", errors="replace")[:1000]
            or "1Password CLI could not read that field. Sign in locally and verify the reference.",
        }
    if not stdout or len(stdout) > _MAX_SECRET_BYTES:
        return {
            "success": False,
            "error": "1Password returned an empty or oversized field; the clipboard was not changed.",
        }
    copy = await asyncio.create_subprocess_exec(
        pbcopy,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _ignored, copy_error = await copy.communicate(stdout)
    if copy.returncode != 0:
        return {
            "success": False,
            "error": copy_error.decode("utf-8", errors="replace")[:1000]
            or "The node could not write to the local clipboard.",
        }
    digest = hashlib.sha256(stdout).hexdigest()
    task = asyncio.create_task(
        _clipboard_clear_after(
            digest, seconds, pbcopy, shutil.which("pbpaste") or "/usr/bin/pbpaste"
        ),
        name="hexis-onepassword-clipboard-clear",
    )
    _CLIPBOARD_TASKS.add(task)
    task.add_done_callback(_CLIPBOARD_TASKS.discard)
    return {
        "success": True,
        "result": {
            "copied": True,
            "secret_transmitted_to_hexis": False,
            "clipboard_clears_after_seconds": seconds,
        },
        "error": None,
    }


async def execute_life_action(
    action: str,
    arguments: dict[str, Any],
    *,
    timeout: int,
    runner: ProcessRunner,
) -> dict[str, Any]:
    """Validate and execute one advertised Wave C action."""
    try:
        if action == "apple.reminders.list":
            args = [
                _text(arguments.get("list_name"), label="list_name", maximum=200),
                "true" if bool(arguments.get("include_completed")) else "false",
                str(_limit(arguments.get("limit"))),
            ]
            return await _jxa(runner, _REMINDERS_LIST_JXA, args, timeout=timeout)
        if action == "apple.reminders.create":
            args = [
                _text(
                    arguments.get("title"), label="title", maximum=500, required=True
                ),
                _text(arguments.get("list_name"), label="list_name", maximum=200),
                _text(arguments.get("notes"), label="notes", maximum=5000),
                _iso(arguments.get("due_at"), label="due_at")
                if arguments.get("due_at")
                else "",
            ]
            return await _jxa(runner, _REMINDERS_CREATE_JXA, args, timeout=timeout)
        if action == "apple.notes.search":
            args = [
                _text(
                    arguments.get("query"), label="query", maximum=500, required=True
                ),
                _text(arguments.get("folder"), label="folder", maximum=200),
                str(_limit(arguments.get("limit"))),
            ]
            return await _jxa(runner, _NOTES_SEARCH_JXA, args, timeout=timeout)
        if action == "apple.notes.create":
            args = [
                _text(
                    arguments.get("title"), label="title", maximum=500, required=True
                ),
                _text(
                    arguments.get("body"),
                    label="body",
                    maximum=_MAX_TEXT,
                    required=True,
                ),
                _text(arguments.get("folder"), label="folder", maximum=200),
            ]
            return await _jxa(runner, _NOTES_CREATE_JXA, args, timeout=timeout)
        if action == "apple.calendar.list":
            start = _iso(arguments.get("start_at"), label="start_at")
            end = _iso(arguments.get("end_at"), label="end_at")
            if datetime.fromisoformat(end) <= datetime.fromisoformat(start):
                raise ValueError("end_at must be after start_at.")
            args = [
                start,
                end,
                _text(arguments.get("calendar"), label="calendar", maximum=200),
                str(_limit(arguments.get("limit"))),
            ]
            return await _jxa(runner, _CALENDAR_LIST_JXA, args, timeout=timeout)
        if action == "apple.calendar.create":
            start = _iso(arguments.get("start_at"), label="start_at")
            end = _iso(arguments.get("end_at"), label="end_at")
            if datetime.fromisoformat(end) <= datetime.fromisoformat(start):
                raise ValueError("end_at must be after start_at.")
            args = [
                _text(
                    arguments.get("title"), label="title", maximum=500, required=True
                ),
                start,
                end,
                _text(arguments.get("calendar"), label="calendar", maximum=200),
                _text(arguments.get("location"), label="location", maximum=500),
                _text(arguments.get("notes"), label="notes", maximum=5000),
            ]
            return await _jxa(runner, _CALENDAR_CREATE_JXA, args, timeout=timeout)
        if action in {"apple.shortcuts.list", "apple.shortcuts.run"}:
            return await _shortcuts(runner, action, arguments, timeout=timeout)
        if action == "onepassword.items":
            return await _op_items(runner, arguments, timeout=timeout)
        if action == "onepassword.copy":
            return await _op_copy(arguments, timeout=timeout)
        if action in APPLE_NODE_ACTIONS | ONEPASSWORD_NODE_ACTIONS:
            return {
                "success": False,
                "error": f"Unsupported structured node action: {action}",
            }
    except (TypeError, ValueError) as exc:
        return {"success": False, "error": str(exc)}
    return {"success": False, "error": f"Unsupported node action: {action}"}
