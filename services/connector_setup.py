"""Transport-neutral connector setup intent and UI handoff.

This is the Hexis-side equivalent of OpenClaw's setup wizard bridge: direct
setup requests are product commands first and model prompts second. Chat/UI/CLI
surfaces consume the same typed UI artifact instead of hoping the assistant
freehands OAuth instructions.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from core.tools import ToolContext, ToolExecutionContext, ToolRegistry


ConnectorSetupAction = Literal["choose_scope", "choose_memory", "choose_autonomy", "start", "complete"]

_GMAIL_READ_CAPABILITIES = ["read", "search"]
_GMAIL_WRITE_CAPABILITIES = [*_GMAIL_READ_CAPABILITIES, "send", "reply"]
_GMAIL_MANAGE_CAPABILITIES = [
    *_GMAIL_WRITE_CAPABILITIES,
    "label",
    "spam_triage",
    "delete",
]
_CONNECTOR_SETUP_WORDS = r"(?:connect|link|set\s*up|setup|authorize|auth|hook\s*up)"
_GMAIL_WORDS = r"(?:gmail|google\s+mail|mailbox|email|inbox|mail)"
_GMAIL_SETUP_RE = re.compile(
    rf"\b{_CONNECTOR_SETUP_WORDS}\b[\s\S]{{0,80}}\b{_GMAIL_WORDS}\b"
    rf"|\b{_GMAIL_WORDS}\b[\s\S]{{0,80}}\b{_CONNECTOR_SETUP_WORDS}\b",
    re.IGNORECASE,
)
_OAUTH_REDIRECT_RE = re.compile(
    r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?/[^\s\"'<>]*[?&]code=",
    re.IGNORECASE,
)
_JSON_PATH_RE = re.compile(r"(?P<path>(?:~|/)[^\s\"'<>]+?\.json)\b", re.IGNORECASE)
_CLIENT_SECRET_HINT_RE = re.compile(
    r"\b(?:gmail|google|oauth|client[_ -]?secret|desktop client|json file|credentials?)\b",
    re.IGNORECASE,
)
_READ_ONLY_RE = re.compile(r"\b(?:read\s*only|just\s+read|only\s+read|read|browse|search)\b", re.IGNORECASE)
_WRITE_RE = re.compile(r"\b(?:write|send|reply|respond|email\s+on\s+my\s+behalf)\b", re.IGNORECASE)
_MANAGE_RE = re.compile(r"\b(?:delete|trash|manage|label|labels|spam|archive|filter)\b", re.IGNORECASE)
_REMEMBER_RE = re.compile(r"\b(?:remember|learn|ingest|store|keep|retain)\b", re.IGNORECASE)
_FORGET_RE = re.compile(
    r"\b(?:forget|do\s+not\s+remember|don't\s+remember|dont\s+remember|do\s+not\s+ingest|don't\s+ingest|dont\s+ingest|ephemeral|temporary)\b",
    re.IGNORECASE,
)
_AUTONOMOUS_EMAIL_RE = re.compile(
    r"\b(?:heartbeat|background|autonomous|automatically|auto|proactive|hourly|digest|while\s+i(?:'|’)?m\s+away|without\s+me\s+asking|only\s+(?:check\s+)?when\s+i\s+ask|when\s+i\s+ask)\b",
    re.IGNORECASE,
)
_AUTONOMOUS_ENABLE_RE = re.compile(
    r"\b(?:yes|sure|ok|okay|enable|allow|authorize|proactive|automatic|automatically|hourly|heartbeat|background|digest)\b",
    re.IGNORECASE,
)
_AUTONOMOUS_DISABLE_RE = re.compile(
    r"\b(?:no|not|never|disable|only\s+when\s+i\s+ask|when\s+i\s+ask|do\s+not|don't|dont|manual)\b",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(r"^\s*(?:cancel|stop|never mind|nevermind)\s*$", re.IGNORECASE)

_PENDING_SETUP_BY_SESSION: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class ConnectorSetupIntent:
    connector_id: str
    action: ConnectorSetupAction
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectorSetupRun:
    connector_id: str
    action: ConnectorSetupAction
    assistant_message: str
    ui: dict[str, Any] | None
    tool_name: str
    tool_result: dict[str, Any]


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _ui_from_tool_output(output: Any) -> dict[str, Any] | None:
    payload = _json(output)
    if not isinstance(payload, dict):
        return None
    ui = payload.get("ui")
    if isinstance(ui, dict) and ui.get("kind") == "connector_setup":
        return ui
    return None


def _tool_result_payload(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "output": result.output,
        "display_output": result.display_output,
        "error": result.error,
        "error_type": result.error_type.value if result.error_type else None,
        "energy_spent": result.energy_spent,
        "duration_seconds": result.duration_seconds,
        "metadata": result.metadata,
    }


def _extract_client_secret_path(message: str, *, pending_setup: bool = False) -> str | None:
    match = _JSON_PATH_RE.search(message)
    if not match:
        return None
    if (
        not pending_setup
        and not _CLIENT_SECRET_HINT_RE.search(message)
        and "client_secret" not in match.group("path").lower()
    ):
        return None
    return match.group("path").rstrip(".,;)")


def _capability_options() -> list[dict[str, Any]]:
    return [
        {
            "id": "read_only",
            "label": "Read and search only",
            "description": "Samantha can read/search email when you ask, without sending or changing mailbox state.",
            "capabilities": list(_GMAIL_READ_CAPABILITIES),
            "risk": "read",
        },
        {
            "id": "write",
            "label": "Read, send, and reply",
            "description": "Adds the ability to send new emails and reply in threads when separately authorized.",
            "capabilities": list(_GMAIL_WRITE_CAPABILITIES),
            "risk": "external_message",
        },
        {
            "id": "manage",
            "label": "Read, write, manage, and delete",
            "description": "Adds labels, spam/archive triage, and delete/trash powers. Destructive actions still require explicit authorization.",
            "capabilities": list(_GMAIL_MANAGE_CAPABILITIES),
            "risk": "destructive",
        },
    ]


def _memory_options() -> list[dict[str, Any]]:
    return [
        {
            "id": "remember",
            "label": "Remember and learn",
            "description": "Save a Hexis memory policy allowing email contents to feed ingestion and evidence-backed memories.",
            "memory_policy": "remember",
        },
        {
            "id": "forget",
            "label": "Forget after reading",
            "description": "Save a Hexis memory policy that keeps email reads task-scoped and avoids email-derived memories by default.",
            "memory_policy": "forget",
        },
    ]


def _autonomy_options() -> list[dict[str, Any]]:
    return [
        {
            "id": "ask_only",
            "label": "Only when I ask",
            "description": "Samantha can read/search Gmail during a live request, but heartbeats will not check email on their own.",
            "heartbeat_digest_enabled": False,
        },
        {
            "id": "heartbeat_digest",
            "label": "Allow heartbeat checks",
            "description": "Samantha may check connected Gmail during hourly heartbeats for important messages and digests.",
            "heartbeat_digest_enabled": True,
        },
    ]


def _gmail_scope_choice_ui() -> dict[str, Any]:
    return {
        "kind": "connector_setup",
        "version": 1,
        "id": "connector_setup:gmail:capability_choice",
        "connector_id": "gmail",
        "display_name": "Gmail",
        "title": "Connect Gmail",
        "status": "needs_capability_choice",
        "summary": "Choose what email powers Samantha should ask Google for.",
        "question": (
            "Do you want me to just be able to read them, write emails on your behalf, "
            "or also manage and delete emails on your behalf?"
        ),
        "capabilities": [],
        "capability_options": _capability_options(),
        "docs_url": "https://console.cloud.google.com/apis/credentials",
        "safety_note": (
            "Connecting a capability is not blanket permission to use it. Sends, replies, mailbox changes, "
            "and deletes still go through connector action authorization."
        ),
    }


def _gmail_memory_choice_ui(base_capabilities: list[str], tier: str | None = None) -> dict[str, Any]:
    return {
        "kind": "connector_setup",
        "version": 1,
        "id": f"connector_setup:gmail:memory_choice:{tier or 'custom'}",
        "connector_id": "gmail",
        "display_name": "Gmail",
        "title": "Connect Gmail",
        "status": "needs_memory_choice",
        "summary": "Choose whether email contents should become memory material.",
        "question": (
            "Do you want me to remember what I read in your emails so I can learn about you, "
            "or should I forget what they say after the task?"
        ),
        "capabilities": list(base_capabilities),
        "memory_options": _memory_options(),
        "memory_config_key": "integrations.gmail.memory_policy",
        "docs_url": "https://console.cloud.google.com/apis/credentials",
        "safety_note": (
            "Google controls provider permissions. Remembering or forgetting is a Hexis-side memory setting, "
            "not a Google permission."
        ),
    }


def _gmail_autonomy_choice_ui(
    base_capabilities: list[str],
    memory_policy: str,
    tier: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "connector_setup",
        "version": 1,
        "id": f"connector_setup:gmail:autonomy_choice:{tier or 'custom'}",
        "connector_id": "gmail",
        "display_name": "Gmail",
        "title": "Connect Gmail",
        "status": "needs_autonomy_choice",
        "summary": "Choose whether Samantha may check Gmail while you are away.",
        "question": (
            "Do you want me to check Gmail during heartbeats on my own, or only read it "
            "when you ask while you are here?"
        ),
        "capabilities": list(base_capabilities),
        "memory_policy": memory_policy,
        "memory_config_key": "integrations.gmail.memory_policy",
        "autonomy_options": _autonomy_options(),
        "heartbeat_digest_config_key": "integrations.gmail.heartbeat_digest_enabled",
        "docs_url": "https://console.cloud.google.com/apis/credentials",
        "safety_note": (
            "Google sign-in grants provider access. Background heartbeat reading is a separate Hexis "
            "autonomy setting and is off until you choose it."
        ),
    }


def _capabilities_for_tier(tier: str) -> list[str]:
    if tier == "manage":
        return list(_GMAIL_MANAGE_CAPABILITIES)
    if tier == "write":
        return list(_GMAIL_WRITE_CAPABILITIES)
    return list(_GMAIL_READ_CAPABILITIES)


def _tier_from_text(text: str) -> str | None:
    if _MANAGE_RE.search(text):
        return "manage"
    if _WRITE_RE.search(text):
        return "write"
    if _READ_ONLY_RE.search(text):
        return "read_only"
    return None


def _memory_choice_from_text(text: str) -> str | None:
    if _FORGET_RE.search(text):
        return "forget"
    if _REMEMBER_RE.search(text):
        return "remember"
    return None


def _autonomy_choice_from_text(text: str) -> bool | None:
    if _AUTONOMOUS_DISABLE_RE.search(text):
        return False
    if _AUTONOMOUS_ENABLE_RE.search(text):
        return True
    return None


def _dedupe_capabilities(capabilities: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in capabilities if str(item).strip()))


def _pop_pending(session_id: str | None) -> dict[str, Any] | None:
    if not session_id:
        return None
    return _PENDING_SETUP_BY_SESSION.pop(session_id, None)


def _set_pending(session_id: str | None, payload: dict[str, Any]) -> None:
    if session_id:
        _PENDING_SETUP_BY_SESSION[session_id] = payload


async def _has_pending_gmail_attempt(pool: Any) -> bool:
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval("SELECT integration_status('gmail')")
        payload = _json(raw) or {}
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    for item in payload.get("recent_attempts", []):
        if not isinstance(item, dict):
            continue
        if item.get("connector_id") == "gmail" and item.get("status") in {
            "pending_user",
            "awaiting_input",
            "pending",
            "in_progress",
            "error",
        }:
            return True
    return False


async def detect_connector_setup_intent(
    pool: Any,
    message: str,
    session_id: str | None = None,
) -> ConnectorSetupIntent | None:
    """Detect user-initiated connector setup before the LLM sees the turn."""
    text = str(message or "").strip()
    if not text:
        return None

    pending = _PENDING_SETUP_BY_SESSION.get(session_id or "")
    if pending and _CANCEL_RE.search(text):
        _pop_pending(session_id)
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="choose_scope",
            arguments={"cancelled": True},
        )

    if pending and pending.get("stage") == "capability_choice":
        tier = _tier_from_text(text)
        if tier:
            base = _capabilities_for_tier(tier)
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="choose_memory",
                arguments={
                    "base_capabilities": base,
                    "tier": tier,
                    "client_secret_path": pending.get("client_secret_path"),
                },
            )

    if pending and pending.get("stage") == "memory_choice":
        choice = _memory_choice_from_text(text)
        if choice:
            base = [str(item) for item in pending.get("base_capabilities") or _GMAIL_READ_CAPABILITIES]
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="choose_autonomy",
                arguments={
                    "base_capabilities": _dedupe_capabilities(base),
                    "tier": pending.get("tier"),
                    "memory_policy": choice,
                    "client_secret_path": pending.get("client_secret_path"),
                },
            )

    if pending and pending.get("stage") == "autonomy_choice":
        choice = _autonomy_choice_from_text(text)
        if choice is not None:
            base = [str(item) for item in pending.get("base_capabilities") or _GMAIL_READ_CAPABILITIES]
            arguments: dict[str, Any] = {
                "capabilities": _dedupe_capabilities(base),
                "memory_policy": str(pending.get("memory_policy") or "forget"),
                "heartbeat_digest_enabled": choice,
            }
            if pending.get("client_secret_path"):
                arguments["client_secret_path"] = pending["client_secret_path"]
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="start",
                arguments=arguments,
            )

    if _OAUTH_REDIRECT_RE.search(text) and await _has_pending_gmail_attempt(pool):
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="complete",
            arguments={"authorization_response": text},
        )

    client_secret_path = _extract_client_secret_path(text, pending_setup=bool(pending))
    if client_secret_path:
        pending = _PENDING_SETUP_BY_SESSION.get(session_id or "")
        tier = _tier_from_text(text)
        memory_choice = _memory_choice_from_text(text)
        if pending and pending.get("stage") == "memory_choice":
            base = [str(item) for item in pending.get("base_capabilities") or _GMAIL_READ_CAPABILITIES]
            if memory_choice:
                return ConnectorSetupIntent(
                    connector_id="gmail",
                    action="choose_autonomy",
                    arguments={
                        "base_capabilities": _dedupe_capabilities(base),
                        "tier": pending.get("tier"),
                        "client_secret_path": client_secret_path,
                        "memory_policy": memory_choice,
                    },
                )
            _set_pending(session_id, {**pending, "client_secret_path": client_secret_path})
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="choose_memory",
                arguments={
                    "base_capabilities": base,
                    "tier": pending.get("tier"),
                    "client_secret_path": client_secret_path,
                },
            )
        if pending and pending.get("stage") == "capability_choice":
            _set_pending(session_id, {**pending, "client_secret_path": client_secret_path})
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="choose_scope",
                arguments={"client_secret_path": client_secret_path},
            )
        if pending and pending.get("stage") == "autonomy_choice":
            _set_pending(session_id, {**pending, "client_secret_path": client_secret_path})
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="choose_autonomy",
                arguments={
                    "base_capabilities": pending.get("base_capabilities") or _GMAIL_READ_CAPABILITIES,
                    "tier": pending.get("tier"),
                    "client_secret_path": client_secret_path,
                    "memory_policy": pending.get("memory_policy") or "forget",
                },
            )
        if tier and memory_choice:
            heartbeat_digest = (
                _autonomy_choice_from_text(text)
                if _AUTONOMOUS_EMAIL_RE.search(text)
                else None
            )
            if heartbeat_digest is None:
                return ConnectorSetupIntent(
                    connector_id="gmail",
                    action="choose_autonomy",
                    arguments={
                        "base_capabilities": _dedupe_capabilities(_capabilities_for_tier(tier)),
                        "tier": tier,
                        "client_secret_path": client_secret_path,
                        "memory_policy": memory_choice,
                    },
                )
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="start",
                arguments={
                    "capabilities": _dedupe_capabilities(_capabilities_for_tier(tier)),
                    "client_secret_path": client_secret_path,
                    "memory_policy": memory_choice,
                    "heartbeat_digest_enabled": heartbeat_digest,
                },
            )
        if tier:
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="choose_memory",
                arguments={
                    "base_capabilities": _capabilities_for_tier(tier),
                    "tier": tier,
                    "client_secret_path": client_secret_path,
                },
            )
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="choose_scope",
            arguments={"client_secret_path": client_secret_path},
        )

    if _GMAIL_SETUP_RE.search(text):
        tier = _tier_from_text(text)
        memory_choice = _memory_choice_from_text(text)
        if tier and memory_choice:
            heartbeat_digest = (
                _autonomy_choice_from_text(text)
                if _AUTONOMOUS_EMAIL_RE.search(text)
                else None
            )
            if heartbeat_digest is None:
                return ConnectorSetupIntent(
                    connector_id="gmail",
                    action="choose_autonomy",
                    arguments={
                        "base_capabilities": _dedupe_capabilities(_capabilities_for_tier(tier)),
                        "tier": tier,
                        "memory_policy": memory_choice,
                    },
                )
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="start",
                arguments={
                    "capabilities": _dedupe_capabilities(_capabilities_for_tier(tier)),
                    "memory_policy": memory_choice,
                    "heartbeat_digest_enabled": heartbeat_digest,
                },
            )
        if tier:
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="choose_memory",
                arguments={"base_capabilities": _capabilities_for_tier(tier), "tier": tier},
            )
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="choose_scope",
        )

    return None


def _assistant_message_for(intent: ConnectorSetupIntent, result_payload: dict[str, Any], ui: dict[str, Any] | None) -> str:
    if intent.action == "choose_scope":
        if intent.arguments.get("cancelled"):
            return "Okay. I stopped Gmail setup."
        return (
            "Do you want me to just be able to read them, write emails on your behalf, "
            "or also manage and delete emails on your behalf?"
        )
    if intent.action == "choose_memory":
        return (
            "Do you want me to remember what I read in your emails so I can learn about you, "
            "or should I forget what they say after the task?"
        )
    if intent.action == "choose_autonomy":
        return (
            "Do you want me to check Gmail during heartbeats on my own, or only read it "
            "when you ask while you are here?"
        )

    if not result_payload.get("success"):
        error = str(result_payload.get("error") or "setup could not start")
        return f"I opened Gmail setup, but it needs attention: {error}"

    status = str((ui or {}).get("status") or "")
    if intent.action == "complete":
        if status == "connected":
            return "Gmail is connected now. I will stay within the email powers and memory policy you approved."
        return "I checked the Gmail authorization step and opened the setup panel with the next action."

    if status == "connected":
        return "Gmail is already connected. I opened the connection status."
    if status == "pending_authorization":
        return "I started Google sign-in for Gmail and opened the setup panel. Approve it in Google, then paste the localhost redirect back into the panel."
    if status in {"needs_client_secret", "setup"}:
        if _ui_has_built_in_gmail_sign_in(ui):
            return "Gmail sign-in is ready. I opened the setup panel so you can approve access with Google."
        return (
            "I opened Gmail setup. The panel walks you through the one-time Google setup this local build needs before sign-in can start."
        )
    if status == "client_secret_saved":
        return "I opened Gmail setup. The Google setup file is saved; start Google sign-in from the panel."
    return "I opened Gmail setup."


def _ui_has_built_in_gmail_sign_in(ui: dict[str, Any] | None) -> bool:
    """Return whether the UI payload can start Google sign-in without local setup."""
    if not isinstance(ui, dict):
        return False
    if ui.get("hexis_oauth_client_available") is True:
        return True
    step = ui.get("credential_step")
    if not isinstance(step, dict):
        return False
    modes = step.get("modes")
    if not isinstance(modes, list):
        return False
    return any(
        isinstance(mode, dict)
        and mode.get("id") == "hosted_oauth"
        and mode.get("available") is True
        for mode in modes
    )


async def run_connector_setup_intent(
    pool: Any,
    registry: ToolRegistry,
    intent: ConnectorSetupIntent,
    *,
    session_id: str | None,
    source_channel: str,
) -> ConnectorSetupRun:
    """Execute the deterministic setup intent and return UI-ready state."""
    if intent.connector_id != "gmail":
        raise ValueError(f"unsupported connector setup: {intent.connector_id}")

    if intent.action == "choose_scope":
        if intent.arguments.get("cancelled"):
            _pop_pending(session_id)
        else:
            pending_payload = {"connector_id": "gmail", "stage": "capability_choice"}
            if intent.arguments.get("client_secret_path"):
                pending_payload["client_secret_path"] = intent.arguments["client_secret_path"]
            _set_pending(session_id, pending_payload)
        ui = None if intent.arguments.get("cancelled") else _gmail_scope_choice_ui()
        return ConnectorSetupRun(
            connector_id=intent.connector_id,
            action=intent.action,
            assistant_message=_assistant_message_for(intent, {"success": True}, ui),
            ui=ui,
            tool_name="connector_setup_scope",
            tool_result={"success": True, "output": {"ui": ui}, "display_output": None},
        )

    if intent.action == "choose_memory":
        base = [str(item) for item in intent.arguments.get("base_capabilities") or _GMAIL_READ_CAPABILITIES]
        tier = str(intent.arguments.get("tier") or "")
        _set_pending(
            session_id,
            {
                "connector_id": "gmail",
                "stage": "memory_choice",
                "tier": tier,
                "base_capabilities": base,
                "client_secret_path": intent.arguments.get("client_secret_path"),
            },
        )
        ui = _gmail_memory_choice_ui(base, tier)
        return ConnectorSetupRun(
            connector_id=intent.connector_id,
            action=intent.action,
            assistant_message=_assistant_message_for(intent, {"success": True}, ui),
            ui=ui,
            tool_name="connector_setup_memory",
            tool_result={"success": True, "output": {"ui": ui}, "display_output": None},
        )

    if intent.action == "choose_autonomy":
        base = [str(item) for item in intent.arguments.get("base_capabilities") or _GMAIL_READ_CAPABILITIES]
        tier = str(intent.arguments.get("tier") or "")
        memory_policy = str(intent.arguments.get("memory_policy") or "forget")
        _set_pending(
            session_id,
            {
                "connector_id": "gmail",
                "stage": "autonomy_choice",
                "tier": tier,
                "base_capabilities": base,
                "memory_policy": memory_policy,
                "client_secret_path": intent.arguments.get("client_secret_path"),
            },
        )
        ui = _gmail_autonomy_choice_ui(base, memory_policy, tier)
        return ConnectorSetupRun(
            connector_id=intent.connector_id,
            action=intent.action,
            assistant_message=_assistant_message_for(intent, {"success": True}, ui),
            ui=ui,
            tool_name="connector_setup_autonomy",
            tool_result={"success": True, "output": {"ui": ui}, "display_output": None},
        )

    tool_name = "complete_gmail_connection" if intent.action == "complete" else "connect_gmail"
    args = dict(intent.arguments)
    if intent.action == "start":
        args.setdefault("capabilities", list(_GMAIL_READ_CAPABILITIES))
        args.setdefault("source_channel", source_channel)
        if session_id:
            args.setdefault("source_session_id", session_id)
        _pop_pending(session_id)

    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id=f"connector-setup:{uuid.uuid4()}",
        session_id=session_id,
        allow_network=True,
        allow_shell=False,
        allow_file_write=False,
        allow_file_read=True,
    )
    result = await registry.execute(tool_name, args, context)
    result_payload = _tool_result_payload(result)
    ui = _ui_from_tool_output(result.output)

    # If the mutating setup call fails before returning UI, fall back to the
    # read-only status tool so the client still has an actionable setup panel.
    if ui is None:
        status_context = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id=f"connector-setup-status:{uuid.uuid4()}",
            session_id=session_id,
            allow_network=False,
        )
        status_result = await registry.execute("gmail_setup_status", {}, status_context)
        status_payload = _tool_result_payload(status_result)
        ui = _ui_from_tool_output(status_result.output)
        if ui and result_payload.get("error"):
            ui = {
                **ui,
                "status": ui.get("status") or "needs_attention",
                "next_step": str(result_payload["error"]),
            }
        result_payload.setdefault("status_probe", status_payload)

    return ConnectorSetupRun(
        connector_id=intent.connector_id,
        action=intent.action,
        assistant_message=_assistant_message_for(intent, result_payload, ui),
        ui=ui,
        tool_name=tool_name,
        tool_result=result_payload,
    )
