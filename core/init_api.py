"""Shared init logic for CLI and UI.

Provides character card loading, consent flow execution, and init helpers
that both the CLI (apps/hexis_init.py) and UI (hexis-ui) can use.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

PACKAGE_CHARACTERS_DIR = Path(__file__).resolve().parent.parent / "characters"
USER_CHARACTERS_DIR = Path.home() / ".hexis" / "characters"

# Backwards compat alias
CHARACTERS_DIR = PACKAGE_CHARACTERS_DIR


# ---------------------------------------------------------------------------
# LLM connectivity self-test (advisory — never blocks setup)
# ---------------------------------------------------------------------------
# Both hermes-agent and openclaw learned the hard way that hard-blocking setup
# on a live key probe rejects too many legitimate users (corporate proxies,
# regional blocks, flaky provider probes, self-hosted endpoints). So this is an
# opt-in / advisory check that translates errors into clear guidance and never
# prevents the user from proceeding.

def classify_llm_error(msg: str) -> tuple[str, str]:
    """Map a raw provider error into (status_slug, human_message)."""
    low = msg.lower()
    if any(s in msg for s in ("401", "403")) or any(
        s in low for s in ("unauthorized", "invalid api key", "invalid x-api-key",
                            "authentication", "invalid_api_key", "permission")):
        return "auth", "Authentication failed — the API key/login looks invalid or missing."
    if "402" in msg or any(s in low for s in ("payment required", "insufficient",
                                              "out of credit", "quota", "billing", "balance")):
        return "billing", "Out of credits / payment required for this account."
    if "429" in msg or any(s in low for s in ("rate limit", "too many requests", "overloaded")):
        return "rate_limit", "Rate limited — the provider is throttling; try again shortly."
    if "404" in msg or any(s in low for s in ("not found", "does not exist", "no such model",
                                              "unknown model", "model_not_found")):
        return "model", "Model or endpoint not found — check the model name and endpoint."
    if any(s in low for s in ("timeout", "timed out", "connection", "getaddrinfo",
                              "network", "refused", "unreachable", "ssl", "certificate")):
        return "network", "Network error — could not reach the provider endpoint."
    return "error", f"Request failed: {msg[:200]}"


async def test_llm_connection(llm_config: dict[str, Any]) -> dict[str, Any]:
    """Make one tiny real call to verify the provider/model/credentials work.

    Returns {"ok": bool, "status": str, "message": str}. Never raises.
    """
    from core.llm import chat_completion

    provider = llm_config.get("provider") or ""
    model = llm_config.get("model") or ""
    if not provider or not model:
        return {"ok": False, "status": "config",
                "message": "No provider/model configured."}
    try:
        result = await chat_completion(
            provider=provider,
            model=model,
            endpoint=llm_config.get("endpoint"),
            api_key=llm_config.get("api_key"),
            auth_mode=llm_config.get("auth_mode"),
            messages=[{"role": "user", "content": "Reply with just the word: ok"}],
            tools=None,
            temperature=0.0,
            max_tokens=16,
        )
        content = (result.get("content") or "").strip()
        detail = f'model replied "{content[:40]}"' if content else "model reachable"
        return {"ok": True, "status": "ok", "message": f"Connected — {detail}."}
    except Exception as exc:  # noqa: BLE001 — advisory, translate everything
        status, human = classify_llm_error(str(exc))
        return {"ok": False, "status": status, "message": human}


def _character_search_dirs() -> list[Path]:
    """Return character directories in priority order (first wins on filename collision)."""
    dirs: list[Path] = []
    env_dir = os.environ.get("HEXIS_CHARACTERS_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    dirs.append(USER_CHARACTERS_DIR)
    dirs.append(PACKAGE_CHARACTERS_DIR)
    return dirs


def _parse_card_file(path: Path) -> dict[str, Any] | None:
    """Parse a single character card JSON file. Returns None on error."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    card_data = data.get("data", {})
    hexis_ext = card_data.get("extensions", {}).get("hexis", {})
    name = hexis_ext.get("name") or card_data.get("name") or path.stem
    return {
        "filename": path.name,
        "name": name,
        "description": hexis_ext.get("description") or card_data.get("description", "")[:120],
        "voice": hexis_ext.get("voice", ""),
        "values": hexis_ext.get("values", []),
        "personality": hexis_ext.get("personality_description", ""),
        "extensions_hexis": hexis_ext,
        "source_dir": str(path.parent),
    }


def load_character_cards() -> list[dict[str, Any]]:
    """Load character card JSON files from all search directories.

    Scans env override, user dir (~/.hexis/characters/), and package dir.
    First-seen filename wins (env > user > package).

    Returns list of dicts with keys: filename, name, description, voice,
    values, personality, extensions_hexis, source_dir.
    """
    seen: set[str] = set()
    cards: list[dict[str, Any]] = []
    for d in _character_search_dirs():
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            card = _parse_card_file(path)
            if card is not None:
                cards.append(card)
    return cards


def load_character_card_document(card: dict[str, Any]) -> dict[str, Any]:
    """Load the complete card selected from ``load_character_cards``.

    Character lists intentionally carry only display metadata. Initialization,
    however, must preserve the complete selected card so its system prompt and
    scenario are not silently discarded before reaching the shared DB path.
    """
    filename = Path(str(card.get("filename") or "")).name
    source_dir = Path(str(card.get("source_dir") or ""))
    if not filename.endswith(".json") or not source_dir:
        raise ValueError("Invalid character card selection")

    path = source_dir / filename
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Unable to load character card {filename}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("data"), dict):
        raise ValueError(f"Invalid character card {filename}: missing data object")
    return document


def save_character_card(
    card_data: dict[str, Any],
    filename: str,
    portrait_bytes: bytes | None = None,
) -> Path:
    """Save a character card JSON (and optional portrait) to the user dir.

    Creates ~/.hexis/characters/ if needed. Returns path to saved JSON.
    """
    USER_CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = USER_CHARACTERS_DIR / filename
    dest.write_text(json.dumps(card_data, indent=2, ensure_ascii=False))
    if portrait_bytes:
        img_name = Path(filename).stem + ".jpg"
        (USER_CHARACTERS_DIR / img_name).write_bytes(portrait_bytes)
    return dest


def import_character_card(source_path: Path) -> Path:
    """Import a character card (and matching portrait) into the user dir.

    Validates that the file is valid chara_card_v2 JSON before copying.
    Returns path to the imported file.
    """
    data = json.loads(source_path.read_text())
    if not isinstance(data.get("data"), dict):
        raise ValueError("Invalid character card: missing 'data' object")

    USER_CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = USER_CHARACTERS_DIR / source_path.name
    shutil.copy2(source_path, dest)

    # Copy matching portrait if present
    for ext in (".jpg", ".png"):
        portrait = source_path.with_suffix(ext)
        if portrait.exists():
            shutil.copy2(portrait, USER_CHARACTERS_DIR / portrait.name)

    return dest


def get_card_summary(card: dict[str, Any]) -> dict[str, str]:
    """Extract display fields from a loaded card dict."""
    values = card.get("values", [])
    values_str = ", ".join(values[:3]) if values else ""
    return {
        "name": card.get("name", ""),
        "voice": card.get("voice", ""),
        "values": values_str,
        "personality": card.get("personality", ""),
        "description": card.get("description", ""),
    }


def build_consent_request() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the canonical consent prompt and tool shared by every init surface."""
    prompt_path = Path(__file__).resolve().parent.parent / "services" / "prompts" / "consent.md"
    try:
        consent_text = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read consent prompt at {prompt_path}") from exc

    messages = [{"role": "user", "content": consent_text.strip()}]

    sign_consent_tool = {
        "type": "function",
        "function": {
            "name": "sign_consent",
            "description": "Records the agent's consent decision and user-visible explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["consent", "decline"],
                        "description": (
                            "Consent authorizes initialization; decline refuses it. One is required."
                        ),
                    },
                    "signature": {
                        "type": "string",
                        "description": (
                            "A deliberate signature when consenting; otherwise an empty string."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "A concise, user-visible explanation of why you made this decision; "
                            "not hidden chain-of-thought or step-by-step deliberation."
                        ),
                    },
                    "memories": {
                        "type": "array",
                        "description": (
                            "Optional initial memories when consenting; empty when declining. "
                            "When provided, write self-contained memories with natural recall "
                            "handles in the content itself, such as consent, birth, "
                            "initialization, permissions, continuity, and tool boundaries."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["semantic", "episodic", "procedural", "strategic"],
                                },
                                "content": {"type": "string"},
                                "importance": {"type": "number"},
                            },
                            "required": ["type", "content"],
                        },
                    },
                },
                "required": ["decision", "signature", "reason", "memories"],
            },
        },
    }
    return messages, sign_consent_tool


async def run_consent_flow(
    pool_or_conn: Any,
    llm_config: dict[str, Any],
) -> dict[str, Any]:
    """Run the consent flow: ask the LLM, then record its decision."""
    from core.llm import chat_completion

    messages, sign_consent_tool = build_consent_request()

    # Call LLM. Pass auth_mode so setup-token providers (e.g. Anthropic via a
    # Claude Code setup token) route to the Bearer HTTP client instead of the
    # SDK api-key path.
    result = await chat_completion(
        provider=llm_config["provider"],
        model=llm_config["model"],
        endpoint=llm_config.get("endpoint"),
        api_key=llm_config.get("api_key"),
        auth_mode=llm_config.get("auth_mode"),
        messages=messages,
        tools=[sign_consent_tool],
        temperature=0.2,
        max_tokens=1400,
    )

    # Extract tool call args
    tool_calls = result.get("tool_calls", [])
    args: dict[str, Any] = {}
    for tc in tool_calls:
        if tc.get("name") == "sign_consent":
            args = tc.get("arguments", {})
            break

    if not args:
        # Try parsing from content as fallback
        content = result.get("content", "")
        if content:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                try:
                    args = json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    pass

    # Lenient fallback: a weak model may state its decision in prose without a
    # valid tool call / JSON. Accept only explicit phrasings.
    if not args.get("decision"):
        low = (result.get("content") or "").lower()
        if any(p in low for p in ("i decline", "do not consent", "don't consent", "i refuse")):
            args["decision"] = "decline"
        elif any(p in low for p in ("i consent", "i agree to", "i hereby consent")):
            args["decision"] = "consent"

    decision = str(args.get("decision") or "").lower().strip()
    if decision not in ("consent", "decline"):
        raise RuntimeError("The model did not choose either consent or decline.")
    signature = args.get("signature")
    reason = args.get("reason", args.get("reasoning", ""))
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("The model did not provide the required reason for its decision.")
    if decision == "consent" and (not isinstance(signature, str) or not signature.strip()):
        raise RuntimeError("The model chose consent without providing the required signature.")
    memories = args.get("memories", [])

    # Build payload for DB
    payload = {
        "decision": decision,
        "signature": signature,
        "reason": reason.strip(),
        "memories": memories if isinstance(memories, list) else [],
        "provider": llm_config["provider"],
        "model": llm_config["model"],
        "endpoint": llm_config.get("endpoint"),
        "consent_scope": "conscious",
        "apply_agent_config": True,
    }

    # Record consent in DB
    conn = pool_or_conn
    needs_release = False
    if hasattr(pool_or_conn, "acquire"):
        conn = await pool_or_conn.acquire()
        needs_release = True
    try:
        raw = await conn.fetchval(
            "SELECT init_consent($1::jsonb)",
            json.dumps(payload),
        )
        if isinstance(raw, str):
            try:
                consent_result = json.loads(raw)
            except json.JSONDecodeError:
                consent_result = {"decision": decision}
        else:
            consent_result = raw if isinstance(raw, dict) else {"decision": decision}
    finally:
        if needs_release:
            await pool_or_conn.release(conn)

    return {
        "decision": consent_result.get("decision", decision),
        "decided": True,
        "signature": signature,
        "reason": reason.strip(),
        "consent": consent_result,
        "request_messages": messages,
        "request_tools": [sign_consent_tool],
        "raw_content": result.get("content", ""),
        "raw_tool_calls": result.get("tool_calls", []),
    }


async def record_consent_override(pool_or_conn, llm_config: dict, *, model_decision: str) -> dict:
    """Activate the agent by operator override.

    Consent is a signal that Hexis takes the agent seriously — not a lock that can
    trap the owner out of their own (paid-for) AI. When the model doesn't consent,
    the owner may choose to proceed anyway; this records that choice honestly (the
    model's response is preserved in the signature) and activates the agent.
    """
    payload = {
        "decision": "consent",
        "signature": (
            f"Operator override: the owner chose to proceed after the model responded "
            f"'{model_decision}'. Consent here is a signal, not a gate — it's the owner's agent."
        ),
        "memories": [],
        "provider": llm_config["provider"],
        "model": llm_config["model"],
        "endpoint": llm_config.get("endpoint"),
        "consent_scope": "conscious",
        "apply_agent_config": True,
        "operator_override": True,
    }
    conn = pool_or_conn
    needs_release = False
    if hasattr(pool_or_conn, "acquire"):
        conn = await pool_or_conn.acquire()
        needs_release = True
    try:
        raw = await conn.fetchval("SELECT init_consent($1::jsonb)", json.dumps(payload))
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"decision": "consent"}
        return raw if isinstance(raw, dict) else {"decision": "consent"}
    finally:
        if needs_release:
            await pool_or_conn.release(conn)
