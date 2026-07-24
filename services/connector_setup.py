"""Transport-neutral connector setup intent and UI handoff.

This is the Hexis-side equivalent of OpenClaw's setup wizard bridge: direct
setup requests are product commands first and model prompts second. Chat/UI/CLI
surfaces consume the same typed UI artifact instead of hoping the assistant
freehands OAuth instructions.
"""

from __future__ import annotations

import json
import shlex
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from core.tools import ToolContext, ToolExecutionContext, ToolRegistry
from core.llm_config import load_llm_config
from core.llm_json import chat_json


ConnectorSetupAction = Literal["choose_scope", "choose_memory", "choose_autonomy", "start", "complete"]

_GMAIL_READ_CAPABILITIES = ["read", "search"]
_GMAIL_WRITE_CAPABILITIES = [*_GMAIL_READ_CAPABILITIES, "send", "reply"]
_GMAIL_MANAGE_CAPABILITIES = [
    *_GMAIL_WRITE_CAPABILITIES,
    "label",
    "spam_triage",
    "delete",
]
_CANCEL_MESSAGES = {"cancel", "stop", "never mind", "nevermind"}
_CLIENT_SECRET_HINTS = (
    "gmail",
    "google",
    "oauth",
    "client_secret",
    "client secret",
    "desktop client",
    "json file",
    "credentials",
)

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


def _message_tokens(message: str) -> list[str]:
    try:
        tokens = shlex.split(message)
    except ValueError:
        tokens = message.split()
    return [token.strip().strip("<>'\".,;)(") for token in tokens if token.strip()]


def _extract_client_secret_path(message: str, *, pending_setup: bool = False) -> str | None:
    normalized = " ".join(message.lower().replace("_", " ").replace("-", " ").split())
    for token in _message_tokens(message):
        lowered = token.lower()
        if not (token.startswith("/") or token.startswith("~")):
            continue
        if not lowered.endswith(".json"):
            continue
        if (
            not pending_setup
            and "client_secret" not in lowered
            and not any(hint in normalized for hint in _CLIENT_SECRET_HINTS)
        ):
            continue
        return token.rstrip(".,;)")
    return None


def _extract_oauth_redirect(message: str) -> str | None:
    for token in _message_tokens(message):
        parsed = urlparse(token)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            continue
        params = parse_qs(parsed.query)
        if params.get("code") or params.get("error"):
            return token
    return None


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


def _dedupe_capabilities(capabilities: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in capabilities if str(item).strip()))


def _normalized_choice_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _gmail_word_candidate(text: str) -> bool:
    """Cheap gate before spending an LLM call; the model still decides intent."""
    lowered = f" {text.lower()} "
    return any(
        marker in lowered
        for marker in (
            " gmail",
            " email",
            " e-mail",
            " inbox",
            " mailbox",
            " mail ",
            " emails",
            " messages",
        )
    )


def _tier_from_classification(classification: dict[str, Any]) -> str | None:
    tier = str(classification.get("capability_tier") or "").strip().lower()
    if tier in {"read_only", "write", "manage"}:
        return tier

    capabilities = classification.get("capabilities")
    if not isinstance(capabilities, list):
        return None
    normalized = {str(item).strip().lower() for item in capabilities}
    if normalized & {"delete", "trash", "label", "archive", "spam_triage", "spam", "manage"}:
        return "manage"
    if normalized & {"send", "reply", "write", "compose"}:
        return "write"
    if normalized & {"read", "search", "list", "check"}:
        return "read_only"
    return None


def _memory_choice_from_classification(classification: dict[str, Any]) -> str | None:
    memory_policy = str(classification.get("memory_policy") or "").strip().lower()
    if memory_policy in {"remember", "forget"}:
        return memory_policy
    return None


def _autonomy_choice_from_classification(classification: dict[str, Any]) -> bool | None:
    value = classification.get("heartbeat_digest_enabled")
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "allow", "allowed", "enable", "enabled"}:
        return True
    if normalized in {"false", "no", "deny", "denied", "disable", "disabled"}:
        return False
    return None


def _pop_pending(session_id: str | None) -> dict[str, Any] | None:
    if not session_id:
        return None
    return _PENDING_SETUP_BY_SESSION.pop(session_id, None)


def _set_pending(session_id: str | None, payload: dict[str, Any]) -> None:
    if session_id:
        _PENDING_SETUP_BY_SESSION[session_id] = payload


async def _gmail_setup_status(pool: Any) -> dict[str, Any]:
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval("SELECT integration_status('gmail')")
    except Exception:
        return {}
    payload = _json(raw) or {}
    return payload if isinstance(payload, dict) else {}


def _payload_has_connected_gmail(payload: dict[str, Any]) -> bool:
    for item in payload.get("connections", []):
        if not isinstance(item, dict):
            continue
        if item.get("connector_id") == "gmail" and item.get("status") == "connected":
            return True
    return False


def _payload_has_pending_gmail_attempt(payload: dict[str, Any]) -> bool:
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


async def _has_pending_gmail_attempt(pool: Any) -> bool:
    return _payload_has_pending_gmail_attempt(await _gmail_setup_status(pool))


async def _classify_connector_setup_intent(
    pool: Any,
    text: str,
    *,
    pending: dict[str, Any] | None,
    gmail_connected: bool,
) -> dict[str, Any]:
    """Use the chat LLM to separate connector setup from ordinary email work."""
    pending_stage = str((pending or {}).get("stage") or "")
    if not pending and not _gmail_word_candidate(text):
        return {"route": "normal_chat", "reason": "no mail connector candidate"}

    system = (
        "You classify whether a user message should enter the Gmail connector setup wizard. "
        "Return JSON only with keys: route, action, capability_tier, memory_policy, "
        "heartbeat_digest_enabled, reason. route is connector_setup or normal_chat. "
        "Only classify as connector_setup when the user is explicitly connecting, reconnecting, "
        "changing Gmail permissions/configuration, answering an active setup step, or trying to use "
        "Gmail while Gmail is not connected. When Gmail is already connected, requests to read, "
        "search, check, summarize, triage, send, reply, label, delete, or batch-process actual email "
        "are normal_chat so the operational email tools can run. For capability_tier use read_only, "
        "write, manage, or null. write means send/reply; manage means label/spam/archive/delete. "
        "memory_policy is remember, forget, or null. heartbeat_digest_enabled is true, false, or null."
    )
    user = {
        "message": text,
        "gmail_connected": gmail_connected,
        "pending_setup_stage": pending_stage or None,
        "pending_setup_payload": pending or None,
    }
    try:
        async with pool.acquire() as conn:
            llm_config = await load_llm_config(conn, "llm.intent", fallback_key="llm.chat")
        doc, _raw = await chat_json(
            llm_config=llm_config,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, sort_keys=True)},
            ],
            max_tokens=240,
            temperature=0,
            response_format={"type": "json_object"},
            fallback={"route": "normal_chat", "reason": "classifier fallback"},
        )
    except Exception:
        return {"route": "normal_chat", "reason": "classifier unavailable"}
    return doc if isinstance(doc, dict) else {"route": "normal_chat"}


def _intent_from_classification(
    classification: dict[str, Any],
    *,
    pending: dict[str, Any] | None,
    client_secret_path: str | None,
    session_id: str | None,
) -> ConnectorSetupIntent | None:
    if str(classification.get("route") or "").strip().lower() != "connector_setup":
        return None

    action = str(classification.get("action") or "").strip().lower()
    if action == "cancel":
        _pop_pending(session_id)
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="choose_scope",
            arguments={"cancelled": True},
        )

    tier = _tier_from_classification(classification)
    memory_choice = _memory_choice_from_classification(classification)
    heartbeat_digest = _autonomy_choice_from_classification(classification)

    if pending and pending.get("stage") == "capability_choice":
        if not tier:
            return None
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="choose_memory",
            arguments={
                "base_capabilities": _capabilities_for_tier(tier),
                "tier": tier,
                "client_secret_path": client_secret_path or pending.get("client_secret_path"),
            },
        )

    if pending and pending.get("stage") == "memory_choice":
        base = [str(item) for item in pending.get("base_capabilities") or _GMAIL_READ_CAPABILITIES]
        if not memory_choice:
            return None
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="choose_autonomy",
            arguments={
                "base_capabilities": _dedupe_capabilities(base),
                "tier": pending.get("tier"),
                "memory_policy": memory_choice,
                "client_secret_path": client_secret_path or pending.get("client_secret_path"),
            },
        )

    if pending and pending.get("stage") == "autonomy_choice":
        if heartbeat_digest is None:
            return None
        base = [str(item) for item in pending.get("base_capabilities") or _GMAIL_READ_CAPABILITIES]
        arguments: dict[str, Any] = {
            "capabilities": _dedupe_capabilities(base),
            "memory_policy": str(pending.get("memory_policy") or "forget"),
            "heartbeat_digest_enabled": heartbeat_digest,
        }
        if client_secret_path or pending.get("client_secret_path"):
            arguments["client_secret_path"] = client_secret_path or pending.get("client_secret_path")
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="start",
            arguments=arguments,
        )

    if client_secret_path and not tier:
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="choose_scope",
            arguments={"client_secret_path": client_secret_path},
        )

    if tier and memory_choice:
        base = _dedupe_capabilities(_capabilities_for_tier(tier))
        if heartbeat_digest is None:
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="choose_autonomy",
                arguments={
                    "base_capabilities": base,
                    "tier": tier,
                    **({"client_secret_path": client_secret_path} if client_secret_path else {}),
                    "memory_policy": memory_choice,
                },
            )
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="start",
            arguments={
                "capabilities": base,
                **({"client_secret_path": client_secret_path} if client_secret_path else {}),
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
                **({"client_secret_path": client_secret_path} if client_secret_path else {}),
            },
        )

    return ConnectorSetupIntent(
        connector_id="gmail",
        action="choose_scope",
        arguments=({"client_secret_path": client_secret_path} if client_secret_path else {}),
    )


async def detect_connector_setup_intent(
    pool: Any,
    message: str,
    session_id: str | None = None,
) -> ConnectorSetupIntent | None:
    """Detect user-initiated connector setup before the LLM sees the turn."""
    text = str(message or "").strip()
    if not text:
        return None

    status_payload = await _gmail_setup_status(pool)
    gmail_connected = _payload_has_connected_gmail(status_payload)
    if gmail_connected:
        _pop_pending(session_id)

    pending = None if gmail_connected else _PENDING_SETUP_BY_SESSION.get(session_id or "")
    normalized = _normalized_choice_text(text).strip(".,!?:;")
    if pending and normalized in _CANCEL_MESSAGES:
        _pop_pending(session_id)
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="choose_scope",
            arguments={"cancelled": True},
        )

    oauth_redirect = _extract_oauth_redirect(text)
    if oauth_redirect and not gmail_connected and _payload_has_pending_gmail_attempt(status_payload):
        return ConnectorSetupIntent(
            connector_id="gmail",
            action="complete",
            arguments={"authorization_response": oauth_redirect},
        )

    client_secret_path = _extract_client_secret_path(text, pending_setup=bool(pending))
    if client_secret_path:
        if not pending:
            return ConnectorSetupIntent(
                connector_id="gmail",
                action="choose_scope",
                arguments={"client_secret_path": client_secret_path},
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

    if not pending:
        return None

    classification = await _classify_connector_setup_intent(
        pool,
        text,
        pending=pending,
        gmail_connected=gmail_connected,
    )
    return _intent_from_classification(
        classification,
        pending=pending,
        client_secret_path=client_secret_path,
        session_id=session_id,
    )


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
