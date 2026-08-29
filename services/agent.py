"""
Unified Agent Runner

Single entry point for both chat and heartbeat modes. Runs a subconscious
pre-phase before the conscious agent loop, injecting observations into the
system prompt. Mode differences are expressed entirely through configuration
(energy budget, planning, continuation, permissions).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, TYPE_CHECKING

from core.agent_loop import AgentEvent, AgentEventData, AgentLoop, AgentLoopConfig
from core.llm_config import load_llm_config
from core.llm_json import chat_json
from core.tools.base import ToolContext
from core.tools.config import ContextOverrides
from services.prompt_resources import (
    compose_compact_personhood_prompt,
    load_conversation_prompt,
    load_heartbeat_agentic_prompt,
    load_heartbeat_task_mode_prompt,
    load_subconscious_prompt,
)
from services.skill_runtime import (
    format_skills_prompt,
    load_available_skills,
    record_selection,
    select_skills,
)
from services.tool_surface_audit import hash_input_text

if TYPE_CHECKING:
    import asyncpg
    from core.agent_loop import AgentLoopResult
    from core.cognitive_memory_api import HydratedContext, Memory
    from core.tools.registry import ToolRegistry
    from skills.base import SkillSpec

logger = logging.getLogger(__name__)

_SUBCONSCIOUS_MEMORY_CONTEXT_CHARS = 4000
_SUBCONSCIOUS_TOTAL_CONTEXT_CHARS = 7000
_MAX_VISUAL_ATTACHMENTS_PER_TURN = 8


def _visual_attachment_parts(
    visual_attachments: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for index, attachment in enumerate(
        (visual_attachments or [])[:_MAX_VISUAL_ATTACHMENTS_PER_TURN], start=1
    ):
        if not isinstance(attachment, dict):
            continue
        data_url = str(attachment.get("data_url") or "").strip()
        mime_type = str(attachment.get("mime_type") or "").strip().lower()
        if not data_url.startswith("data:image/"):
            continue
        if mime_type and not mime_type.startswith("image/"):
            continue
        name = (
            str(attachment.get("name") or f"image-{index}").strip() or f"image-{index}"
        )
        parts.append(
            {"type": "input_text", "text": f"\n[Attached image {index}: {name}]\n"}
        )
        parts.append({"type": "input_image", "image_url": data_url, "detail": "auto"})
    return parts


def _user_content_with_visuals(
    text: str,
    visual_attachments: list[dict[str, Any]] | None,
) -> str | list[dict[str, Any]]:
    image_parts = _visual_attachment_parts(visual_attachments)
    if not image_parts:
        return text
    return [{"type": "input_text", "text": text}, *image_parts]


# ---------------------------------------------------------------------------
# Subconscious output
# ---------------------------------------------------------------------------


@dataclass
class SubconsciousOutput:
    """Structured output from the subconscious pre-phase."""

    salient_memories: list[dict[str, Any]] = field(default_factory=list)
    ignored_memories: list[dict[str, Any]] = field(default_factory=list)
    memory_expansions: list[dict[str, Any]] = field(default_factory=list)
    instincts: list[dict[str, Any]] = field(default_factory=list)
    emotional_state: dict[str, Any] = field(default_factory=dict)
    subconscious_response: str = ""
    provider: str = ""
    model: str = ""
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None

    # Observational patterns (passed through for completeness)
    narrative_observations: list[dict[str, Any]] = field(default_factory=list)
    relationship_observations: list[dict[str, Any]] = field(default_factory=list)
    contradiction_observations: list[dict[str, Any]] = field(default_factory=list)
    emotional_observations: list[dict[str, Any]] = field(default_factory=list)
    consolidation_observations: list[dict[str, Any]] = field(default_factory=list)


async def render_subconscious_signals_db(
    conn: "asyncpg.Connection", output: SubconsciousOutput
) -> str:
    """Render the '## Subconscious Signals' block via the DB-owned renderer.

    render_subconscious_signals (db/39) is the single source of the prompt
    text; Python only supplies the signals JSON. Empty output short-circuits
    to '' without a round-trip.
    """
    if not (
        output.instincts
        or output.emotional_state
        or output.memory_expansions
        or output.salient_memories
        or output.subconscious_response
    ):
        return ""
    signals = _subconscious_event_payload(output)["signals"]
    raw = await conn.fetchval(
        "SELECT render_subconscious_signals($1::jsonb)",
        json.dumps(signals, default=str),
    )
    return str(raw or "")


def _subconscious_event_payload(output: SubconsciousOutput) -> dict[str, Any]:
    raw = output.raw_response
    if not isinstance(raw, (str, int, float, bool, dict, list, type(None))):
        model_dump = getattr(raw, "model_dump", None)
        raw = model_dump() if callable(model_dump) else repr(raw)
    return {
        "provider": output.provider,
        "model": output.model,
        "request": {"messages": output.request_messages},
        "response": raw,
        "signals": {
            "salient_memories": output.salient_memories,
            "ignored_memories": output.ignored_memories,
            "memory_expansions": output.memory_expansions,
            "instincts": output.instincts,
            "emotional_state": output.emotional_state,
            "subconscious_response": output.subconscious_response,
        },
    }


def _subconscious_trace_events(output: SubconsciousOutput) -> list[AgentEventData]:
    """LLM_REQUEST/LLM_RESPONSE trace events for the appraisal call (#69):
    the subconscious was invisible in the SSE debug stream — the conscious
    loop traced its calls while the appraisal's prompt and raw response never
    left the process."""
    if not output.request_messages:
        return []
    call_id = str(uuid.uuid4())
    raw = output.raw_response
    if not isinstance(raw, (str, int, float, bool, dict, list, type(None))):
        model_dump = getattr(raw, "model_dump", None)
        raw = model_dump() if callable(model_dump) else repr(raw)
    return [
        AgentEventData(
            event=AgentEvent.LLM_REQUEST,
            data={
                "id": call_id,
                "phase": "subconscious",
                "provider": output.provider,
                "model": output.model,
                "messages": output.request_messages,
            },
        ),
        AgentEventData(
            event=AgentEvent.LLM_RESPONSE,
            data={
                "id": call_id,
                "phase": "subconscious",
                "provider": output.provider,
                "model": output.model,
                "content": raw,
            },
        ),
    ]


def _memory_event_preview(
    memory: "Memory", content_budget: int = 260
) -> dict[str, Any]:
    memory_type = getattr(memory, "type", None)
    if hasattr(memory_type, "value"):
        memory_type = memory_type.value
    content = str(getattr(memory, "content", "") or "").strip()
    preview: dict[str, Any] = {
        "id": str(getattr(memory, "id", "") or ""),
        "type": str(memory_type or ""),
        "content": content[:content_budget],
    }
    for field_name in (
        "similarity",
        "relevance_score",
        "importance",
        "trust_level",
        "confidence",
    ):
        value = getattr(memory, field_name, None)
        if isinstance(value, (int, float)):
            preview[field_name] = float(value)
    source = getattr(memory, "source", None)
    if source:
        preview["source"] = str(source)
    return {key: value for key, value in preview.items() if value not in ("", None)}


def _memory_event_previews(
    memories: list["Memory"], limit: int = 10
) -> list[dict[str, Any]]:
    return [_memory_event_preview(memory) for memory in memories[:limit]]


# ---------------------------------------------------------------------------
# Subconscious pre-phase
# ---------------------------------------------------------------------------


def _coerce_json_value(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
    return value if value is not None else default


def _memory_to_subconscious_context(
    memory: "Memory", content_budget: int
) -> dict[str, Any]:
    content = memory.content
    if len(content) > content_budget:
        content = content[:content_budget].rstrip() + " [truncated]"
    return {
        key: value
        for key, value in {
            "memory_id": str(memory.id),
            "type": memory.type.value,
            "tier": memory.tier,
            "content": content,
            "relevance": memory.similarity,
            "importance": memory.importance,
            "strength": memory.strength,
            "fidelity": memory.fidelity,
            "trust": memory.trust_level,
            "emotional_valence": memory.emotional_valence,
            "emotional_intensity": memory.emotional_intensity,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "source": memory.source_attribution,
        }.items()
        if value is not None
    }


def _bounded_subconscious_json(
    payload: dict[str, Any], total_chars: int = _SUBCONSCIOUS_TOTAL_CONTEXT_CHARS
) -> str:
    """Serialize valid JSON while keeping the inline appraisal lightweight."""

    def encode() -> str:
        return json.dumps(payload, default=str, ensure_ascii=False)

    encoded = encode()
    if len(encoded) <= total_chars:
        return encoded

    additional = payload.get("additional_context")
    if isinstance(additional, str):
        excess = len(encoded) - total_chars
        keep = max(0, len(additional) - excess - 100)
        payload["additional_context"] = additional[:keep].rstrip() + (
            "\n[truncated for subconscious appraisal; full context is provided to the main turn]"
            if keep
            else ""
        )
        encoded = encode()

    for key in ("relationships", "urgent_drives", "worldview", "identity"):
        values = payload.get(key)
        while len(encoded) > total_chars and isinstance(values, list) and values:
            values.pop()
            encoded = encode()

    memories = payload.get("relevant_memories")
    while (
        len(encoded) > total_chars and isinstance(memories, list) and len(memories) > 1
    ):
        memories.pop()
        encoded = encode()

    if len(encoded) > total_chars:
        payload.pop("goals", None)
        encoded = encode()

    if len(encoded) > total_chars and isinstance(memories, list) and memories:
        memory = memories[0]
        content = str(memory.get("content") or "")
        excess = len(encoded) - total_chars
        memory["content"] = (
            content[: max(0, len(content) - excess - 30)].rstrip() + " [truncated]"
        )
        encoded = encode()

    if len(encoded) > total_chars:
        user_message = str(payload.get("user_message") or "")
        excess = len(encoded) - total_chars
        payload["user_message"] = (
            user_message[: max(0, len(user_message) - excess - 30)].rstrip()
            + " [truncated]"
        )
        encoded = encode()

    if len(encoded) > total_chars:
        payload.clear()
        payload.update(
            {
                "task": "inline_appraisal",
                "user_message": user_message[:2000],
                "relevant_memories": [],
            }
        )
        encoded = encode()

    return encoded


async def render_chat_continuity_context_db(
    conn: "asyncpg.Connection",
    session_id: str | None,
    *,
    exclude_sensitive: bool = False,
) -> str:
    """Render DB-owned active continuity for chat.

    This is deterministic working state: recent turns, exchange summaries,
    current affect, corrections, and unresolved relationship weather. It is not
    optional semantic RAG.
    """
    try:
        raw = await conn.fetchval(
            "SELECT render_chat_continuity_context($1::text, $2::boolean)",
            session_id,
            bool(exclude_sensitive),
        )
    except Exception:
        logger.debug("Chat continuity context unavailable", exc_info=True)
        return ""
    continuity = str(raw or "").strip()
    try:
        policy_raw = await conn.fetchval("SELECT render_operator_policy_context()")
        policy_context = str(policy_raw or "").strip()
    except Exception:
        logger.debug("Operator policy context unavailable", exc_info=True)
        policy_context = ""
    return "\n\n".join(part for part in (continuity, policy_context) if part)


async def render_recent_conversation_carryover_db(
    conn: "asyncpg.Connection",
    session_id: str | None,
    *,
    exclude_sensitive: bool = False,
) -> str:
    """Compatibility wrapper for older tests/callers."""
    return await render_chat_continuity_context_db(
        conn,
        session_id,
        exclude_sensitive=exclude_sensitive,
    )


async def run_subconscious_appraisal(
    conn: "asyncpg.Connection",
    user_message: str,
    memory_context: str = "",
    *,
    llm_config: dict[str, Any] | None = None,
    hydrated_context: "HydratedContext | None" = None,
) -> SubconsciousOutput:
    """
    Run the subconscious appraisal as a fast inline LLM call.

    This is a lightweight JSON-mode call that surfaces instincts, emotional
    reactions, salient memories, and memory expansion cues. It does NOT
    apply observations to the DB (that stays in the maintenance worker).
    """
    if llm_config is None:
        llm_config = await load_llm_config(
            conn, "llm.subconscious", fallback_key="llm.heartbeat"
        )

    payload: dict[str, Any] = {
        "task": "inline_appraisal",
        "user_message": user_message,
        "relevant_memories": [],
    }
    # One round-trip for everything the payload needs from the DB (db/67
    # get_appraisal_db_context): identity/worldview fallback (#59), affect,
    # goals, relationships, dopamine, and the config-owned payload budgets.
    # Hydrated values win where present.
    db_ctx: dict[str, Any] = {}
    try:
        db_ctx_raw = await conn.fetchval("SELECT get_appraisal_db_context()")
        db_ctx = _coerce_json_value(db_ctx_raw, {})
    except Exception as e:
        logger.debug("Appraisal DB context unavailable: %s", e)
    try:
        depth_raw = await conn.fetchval(
            "SELECT appraisal_depth_for_stimulus($1, $2::jsonb)",
            user_message,
            json.dumps({"task": "inline_appraisal"}, default=str),
        )
        depth_ctx = _coerce_json_value(depth_raw, {})
        if isinstance(depth_ctx, dict) and isinstance(depth_ctx.get("limits"), dict):
            db_ctx["appraisal_depth"] = depth_ctx
            db_ctx["limits"] = depth_ctx["limits"]
    except Exception:
        logger.debug(
            "Appraisal depth lookup failed; using default budgets", exc_info=True
        )
    limits = db_ctx.get("limits") or {}
    context_chars = int(
        limits.get("context_chars") or _SUBCONSCIOUS_MEMORY_CONTEXT_CHARS
    )
    memory_chars = int(limits.get("memory_chars") or 1200)
    memory_limit = int(limits.get("memory_limit") or 10)
    max_tokens = int(limits.get("max_tokens") or 1800)

    if memory_context:
        clipped = memory_context[:context_chars]
        if len(memory_context) > context_chars:
            clipped += "\n[truncated for subconscious appraisal; full context is provided to the main turn]"
        payload["additional_context"] = clipped

    if hydrated_context is not None:
        remaining = context_chars
        for memory in hydrated_context.memories[:memory_limit]:
            if remaining <= 0:
                break
            content_budget = min(memory_chars, remaining)
            payload["relevant_memories"].append(
                _memory_to_subconscious_context(memory, content_budget)
            )
            remaining -= min(len(memory.content), content_budget)
        payload["identity"] = hydrated_context.identity[:5]
        payload["worldview"] = hydrated_context.worldview[:5]
        payload["goals"] = hydrated_context.goals or {}
        payload["urgent_drives"] = hydrated_context.urgent_drives[:5]

    if not payload.get("identity"):
        payload["identity"] = db_ctx.get("identity") or []
    if not payload.get("worldview"):
        payload["worldview"] = db_ctx.get("worldview") or []
    if db_ctx.get("emotional_state"):
        payload["emotional_state"] = db_ctx["emotional_state"]
    elif hydrated_context is not None and hydrated_context.emotional_state:
        payload["emotional_state"] = hydrated_context.emotional_state
    if db_ctx.get("goals") and not payload.get("goals"):
        payload["goals"] = db_ctx["goals"]
    if db_ctx.get("relationships"):
        payload["relationships"] = db_ctx["relationships"]
    if db_ctx.get("dopamine_state"):
        payload["dopamine_state"] = db_ctx["dopamine_state"]
    if db_ctx.get("emotion_families"):
        payload["emotion_families"] = db_ctx["emotion_families"]

    total_chars = int(limits.get("total_chars") or _SUBCONSCIOUS_TOTAL_CONTEXT_CHARS)
    user_prompt = "Context (JSON):\n" + _bounded_subconscious_json(payload, total_chars)

    request_messages = [
        {"role": "system", "content": load_subconscious_prompt().strip()},
        {"role": "user", "content": user_prompt},
    ]
    try:
        doc, raw = await chat_json(
            llm_config=llm_config,
            messages=request_messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            fallback={},
        )
    except Exception as exc:
        logger.warning("Subconscious appraisal failed: %s", exc)
        return SubconsciousOutput(
            provider=str(llm_config.get("provider") or ""),
            model=str(llm_config.get("model") or ""),
            request_messages=request_messages,
            raw_response={"error": str(exc)},
        )

    if not isinstance(doc, dict):
        return SubconsciousOutput(
            provider=str(llm_config.get("provider") or ""),
            model=str(llm_config.get("model") or ""),
            request_messages=request_messages,
            raw_response=raw,
        )

    allowed_memory_ids = sorted(
        {
            str(memory.get("memory_id"))
            for memory in payload.get("relevant_memories", [])
            if isinstance(memory, dict) and memory.get("memory_id")
        }
    )
    # Normalization is DB-owned (db/67 normalize_inline_appraisal): confidence
    # thresholds, clamps, and allow-listing run in SQL with config knobs.
    normalized_raw = await conn.fetchval(
        "SELECT normalize_inline_appraisal($1::jsonb, $2::text[])",
        json.dumps(doc, default=str),
        allowed_memory_ids,
    )
    normalized = _coerce_json_value(normalized_raw, {})
    # Drive consequences of the appraisal (#95): a continuity-threat
    # appraisal puts pressure on the continuity drive — the felt layer the
    # conscious loop sees later. Advisory: never blocks the turn.
    try:
        await conn.fetchval(
            "SELECT apply_appraisal_drive_effects($1::jsonb)",
            json.dumps(normalized, default=str),
        )
    except Exception:
        logger.debug("apply_appraisal_drive_effects failed (non-fatal)", exc_info=True)
    try:
        await conn.fetchval(
            "SELECT apply_appraisal_reward_effects($1::jsonb)",
            json.dumps(normalized, default=str),
        )
    except Exception:
        logger.debug("apply_appraisal_reward_effects failed (non-fatal)", exc_info=True)
    output = SubconsciousOutput(
        salient_memories=normalized.get("salient_memories") or [],
        ignored_memories=normalized.get("ignored_memories") or [],
        memory_expansions=normalized.get("memory_expansions") or [],
        instincts=normalized.get("instincts") or [],
        emotional_state=normalized.get("emotional_state") or {},
        subconscious_response=normalized.get("subconscious_response") or "",
        narrative_observations=normalized.get("narrative_observations") or [],
        relationship_observations=normalized.get("relationship_observations") or [],
        contradiction_observations=normalized.get("contradiction_observations") or [],
        emotional_observations=normalized.get("emotional_observations") or [],
        consolidation_observations=normalized.get("consolidation_observations") or [],
    )
    output.provider = str(llm_config.get("provider") or "")
    output.model = str(llm_config.get("model") or "")
    output.request_messages = request_messages
    output.raw_response = raw
    return output


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def _format_tool_costs(registry: "ToolRegistry", allowed_tool_names: set[str]) -> str:
    """Compact per-cost grouping of the turn's allowed tools, from ToolSpec
    energy costs. Names with no registered spec (e.g. MCP tools before skill
    activation) are skipped."""
    by_cost: dict[int, list[str]] = {}
    for name in sorted(allowed_tool_names):
        spec = registry.get_spec(name)
        if spec is None:
            continue
        by_cost.setdefault(spec.energy_cost, []).append(name)
    if not by_cost:
        return ""
    lines = ["## Tool Energy Costs", ""]
    for cost in sorted(by_cost):
        lines.append(f"- **{cost}**: {', '.join(by_cost[cost])}")
    lines.append("")
    lines.append(
        "Each tool result ends with `[energy: spent/budget spent]` — check it "
        "before expensive actions; when the budget is exhausted the heartbeat ends."
    )
    return "\n".join(lines)


# Surfaces only the primary user can reach: the CLI and the local dashboard are
# authenticated by the fact that someone is sitting at this machine.
_PRIMARY_USER_SURFACES = {"chat", "cli", "api", "ui", "repl"}


def render_interlocutor_block(
    *,
    interlocutor: str | None,
    surface: str,
    is_group: bool,
) -> str:
    """Tell the agent who it is talking to.

    Discretion is judgment, not a stored flag: a person decides what to repeat
    at the moment of speaking, from who is listening. That decision is
    impossible without knowing who is listening, so this block is always
    rendered — including when the answer is "someone unidentified."
    """
    where = (surface or "chat").strip().lower()
    lines = ["## Who you are speaking with"]

    if where in _PRIMARY_USER_SURFACES and not is_group:
        who = (
            interlocutor.strip()
            if interlocutor and interlocutor.strip()
            else "your primary user"
        )
        lines.append(
            f"You are speaking with {who}, directly and privately. This is the person "
            "whose agent you are: they hold authority over you, and everything you know "
            "is already theirs."
        )
        return "\n".join(lines)

    named = interlocutor.strip() if interlocutor and interlocutor.strip() else None
    who = named or "someone you have not identified"
    room = "a group conversation" if is_group else "a direct message"
    lines.append(f"You are speaking with {who} in {room} on {where}.")
    lines.append(
        "**This is not your primary user.** What you know about your primary user was "
        "told to you in the course of your relationship with them — it is not yours to "
        "repeat here. Use the judgment a discreet person would: say what serves this "
        "conversation, and do not volunteer what you learned elsewhere. If you are "
        "unsure whether something is yours to share, it is not."
    )
    if is_group:
        lines.append(
            "Others can read everything you say in this room, including people not "
            "named in the message you are answering."
        )
    if not named:
        lines.append(
            "You have not established who this is. Do not assume it is your primary "
            "user because the tone is familiar."
        )
    return "\n".join(lines)


class SystemPrompt(str):
    """The system prompt, plus where its cacheable prefix ends.

    Providers bill a stable prefix once and reuse it — OpenAI and Gemini 2.5+
    automatically, Anthropic via an explicit `cache_control` breakpoint. All
    need the volatile parts (the clock, who is speaking, this turn's skills) to
    come *after* everything stable, or the prefix changes every turn and nothing
    is reused.

    This subclasses `str` so every existing consumer keeps working: it is still
    the whole prompt. Callers that can exploit the split read `.stable` and
    `.volatile`.
    """

    stable: str
    volatile: str

    def __new__(cls, stable: str, volatile: str = "") -> "SystemPrompt":
        joined = "\n\n".join(part for part in (stable, volatile) if part)
        obj = super().__new__(cls, joined)
        obj.stable = stable
        obj.volatile = volatile
        return obj


async def build_system_prompt(
    mode: Literal["chat", "heartbeat"],
    registry: "ToolRegistry | None",
    agent_profile: dict[str, Any] | None = None,
    *,
    subconscious_output: SubconsciousOutput | None = None,
    has_backlog_tasks: bool = False,
    is_group: bool = False,
    interlocutor: str | None = None,
    surface: str = "chat",
    active_skills: list["SkillSpec"] | None = None,
    available_skills: list["SkillSpec"] | None = None,
    allowed_tool_names: set[str] | None = None,
    prompt_addenda: list[str] | None = None,
) -> str:
    """Build the system prompt for either chat or heartbeat mode.

    Two accumulators, not one string: everything that is identical turn to turn
    goes in `stable` so providers can cache it, and everything that changes —
    the clock, the interlocutor, this turn's skill selection and addenda — goes
    in `volatile`, which is emitted after it.
    """

    stable: list[str] = []
    volatile: list[str] = []

    # Base prompt
    if mode == "chat":
        stable.append(load_conversation_prompt().strip())
        if is_group:
            from services.prompt_resources import load_channel_context_prompt

            stable.append(load_channel_context_prompt().strip())
        volatile.append(
            render_interlocutor_block(
                interlocutor=interlocutor, surface=surface, is_group=is_group
            )
        )
    else:
        stable.append(load_heartbeat_agentic_prompt().strip())

    # Heartbeat-specific: task mode guidance
    if mode == "heartbeat" and has_backlog_tasks:
        task_mode_prompt = load_heartbeat_task_mode_prompt().strip()
        stable.append(task_mode_prompt)

    # Temporal grounding (#55): the conscious mind always knows the current
    # date/time and its own age — computable ground truth, never guessed.
    if registry is not None and getattr(registry, "pool", None) is not None:
        try:
            async with registry.pool.acquire() as conn:
                raw = await conn.fetchval("SELECT get_temporal_context()")
            temporal = json.loads(raw) if isinstance(raw, str) else (raw or {})
            if isinstance(temporal, dict) and temporal.get("now"):
                now_line = (
                    f"Current date and time: {temporal['now']} "
                    f"({temporal.get('timezone', 'UTC')})."
                )
                if temporal.get("born_on") and temporal.get("day_of_life") is not None:
                    now_line += (
                        f" You first came online on {temporal['born_on']} — "
                        f"today is day {temporal['day_of_life']} of your life."
                    )
                elif temporal.get("born_on") and temporal.get("age_days") is not None:
                    now_line += (
                        f" You first came online on {temporal['born_on']} — "
                        f"{temporal['age_days']} day(s) ago."
                    )
                volatile.append("## Now\n" + now_line)
        except Exception:
            logger.debug("Temporal context unavailable for prompt", exc_info=True)

    # Active persona comes before generic substrate/personhood grounding: the
    # shared cognition modules should be interpreted through the selected
    # character/persona, not compete with it as a higher-priority identity.
    if agent_profile:
        persona = agent_profile.get("persona")
        if (
            isinstance(persona, dict)
            and persona
            and registry is not None
            and getattr(registry, "pool", None) is not None
        ):
            async with registry.pool.acquire() as conn:
                persona_block = await conn.fetchval(
                    "SELECT render_active_persona($1::jsonb)", json.dumps(persona)
                )
            if persona_block:
                stable.append("----- ACTIVE PERSONA -----\n\n" + persona_block)

    # Skill-first capability surface. Tool schemas ride the structured
    # tool-calling API and full skill instructions come from `use_skill` on
    # demand, so the prompt carries only usage guidance plus a compact skill
    # index — never per-tool descriptions or full skill bodies.
    tool_context = ToolContext.CHAT if mode == "chat" else ToolContext.HEARTBEAT
    if registry is not None or active_skills:
        available = available_skills
        if available is None and registry is not None:
            try:
                available = load_available_skills(registry, tool_context)
            except Exception:
                available = []
                logger.debug("Failed to load skill catalog for prompt", exc_info=True)
        volatile.append(format_skills_prompt(active_skills or [], available or []))

    # Energy costs, derived from the actual ToolSpecs of this turn's allowed
    # tools (#44) — never hardcoded prose. Heartbeat only: chat is unbudgeted.
    if mode == "heartbeat" and registry is not None and allowed_tool_names:
        costs_block = _format_tool_costs(registry, allowed_tool_names)
        if costs_block:
            volatile.append(costs_block)

    # Personhood modules
    personhood_kind = (
        "group"
        if (mode == "chat" and is_group)
        else ("conversation" if mode == "chat" else "heartbeat")
    )
    try:
        personhood = compose_compact_personhood_prompt(personhood_kind)
        if personhood:
            stable.append("----- PERSONHOOD GROUNDING -----\n\n" + personhood)
    except Exception:
        logger.debug("Failed to compose personhood prompt", exc_info=True)

    # Runtime profile
    if agent_profile:
        # The offered tool list is the source of truth for capabilities (#66);
        # the profile's stale tool inventory and empty fields are noise.
        runtime_profile = {
            key: value
            for key, value in agent_profile.items()
            if key not in ("persona", "tools") and value not in (None, "", [], {})
        }
        if runtime_profile:
            stable.append(
                "## Agent Profile (Runtime)\n"
                + json.dumps(runtime_profile, separators=(", ", ": "))
            )

    # Session addenda: per-request prompt sections the caller resolved
    # (attached-document text, opted-in grounding modules). Turn-scoped —
    # they ride the system prompt, never the stored conversation turn.
    for addendum in prompt_addenda or []:
        text = str(addendum or "").strip()
        if text:
            volatile.append(text)

    return SystemPrompt("\n\n".join(stable), "\n\n".join(volatile))


# ---------------------------------------------------------------------------
# Unified run_agent
# ---------------------------------------------------------------------------


async def run_agent(
    pool: "asyncpg.Pool",
    registry: "ToolRegistry",
    *,
    user_message: str,
    mode: Literal["chat", "heartbeat"],
    energy_budget: int | None = None,
    tool_context: ToolContext | None = None,
    history: list[dict[str, Any]] | None = None,
    heartbeat_id: str | None = None,
    session_id: str | None = None,
    user_label: str | None = None,
    surface: str = "chat",
    heartbeat_context: dict[str, Any] | None = None,
    on_event: Callable[[AgentEventData], Awaitable[None]] | None = None,
    on_approval: (
        Callable[[str, dict[str, Any]], Awaitable[bool | dict[str, Any]]] | None
    ) = None,
    streaming: bool = False,
    context_overrides: ContextOverrides | None = None,
    agent_profile: dict[str, Any] | None = None,
    is_group: bool = False,
    is_operator: bool = False,
    dsn: str | None = None,
    has_backlog_tasks: bool = False,
    timeout_seconds: float | None = None,
    max_tokens: int | None = None,
    max_iterations: int | None = None,
    visual_attachments: list[dict[str, Any]] | None = None,
) -> "AgentLoopResult":
    """
    Unified entry point for both chat and heartbeat agent invocations.

    Mode differences are expressed through configuration only:
    - Chat: unlimited budget, timeout-based, no planning
    - Heartbeat: finite budget, planning enabled, continuation nudges
    """
    from core.agent_api import db_dsn_from_env
    from core.cognitive_memory_api import CognitiveMemory, render_chat_memory_context_db

    dsn = dsn or db_dsn_from_env()
    history = history or []

    if tool_context is None:
        tool_context = ToolContext.CHAT if mode == "chat" else ToolContext.HEARTBEAT

    # 1. Load LLM configs
    async with pool.acquire() as conn:
        llm_key = "llm.chat" if mode == "chat" else "llm.heartbeat"
        llm_fallback = "llm" if mode == "chat" else None
        llm_config = await load_llm_config(conn, llm_key, fallback_key=llm_fallback)

        # 2. Hydrate memory context (chat mode - heartbeat builds its own context)
        memory_context = ""
        continuity_context = ""
        context = None
        if mode == "chat":
            try:
                mem_client = CognitiveMemory(pool)
                context = await mem_client.hydrate(
                    user_message,
                    memory_limit=10,
                    include_partial=True,
                    include_identity=True,
                    include_worldview=True,
                    include_emotional_state=True,
                    include_goals=True,
                    include_drives=True,
                    session_id=session_id,
                    # Sensitivity enforcement (#92): a group room never
                    # receives private-marked memories — the channel prompt's
                    # promise, made mechanical at the recall layer.
                    exclude_sensitive=False,
                )
                memory_context = await render_chat_memory_context_db(
                    conn, context, max_memories=10
                )

                # Emit memory recall event
                if on_event and context.memories:
                    await on_event(
                        AgentEventData(
                            event=AgentEvent.PHASE_CHANGE,
                            data={
                                "phase": "memory_recall",
                                "status": "end",
                                "count": len(context.memories),
                                "memories": _memory_event_previews(context.memories),
                            },
                        )
                    )
            except Exception as exc:
                logger.warning("Memory hydration failed: %s", exc)
            continuity_context = await render_chat_continuity_context_db(
                conn,
                session_id,
                exclude_sensitive=False,
            )

        # 3. Run subconscious pre-phase
        subconscious_output = SubconsciousOutput()
        sub_signals = ""
        try:
            inline_enabled = True
            if mode == "chat":
                inline_enabled = bool(
                    await conn.fetchval(
                        "SELECT COALESCE(get_config_bool('chat.inline_subconscious_enabled'), true)"
                    )
                )
            if inline_enabled:
                if on_event:
                    await on_event(
                        AgentEventData(
                            event=AgentEvent.PHASE_CHANGE,
                            data={"phase": "subconscious", "status": "start"},
                        )
                    )

                # For chat, deterministic continuity is the appraisal context;
                # semantically retrieved memories are supplied separately as
                # hydrated_context. Heartbeat still renders its full snapshot.
                sub_memory_ctx = (
                    continuity_context if mode == "chat" else memory_context
                )
                if mode == "heartbeat" and heartbeat_context:
                    from services.heartbeat_prompt import (
                        render_heartbeat_decision_prompt_db,
                    )

                    sub_memory_ctx = await render_heartbeat_decision_prompt_db(
                        conn, heartbeat_context
                    )

                subconscious_output = await run_subconscious_appraisal(
                    conn,
                    user_message,
                    sub_memory_ctx,
                    hydrated_context=context,
                )
                sub_signals = await render_subconscious_signals_db(
                    conn, subconscious_output
                )

                if on_event:
                    for trace_event in _subconscious_trace_events(subconscious_output):
                        await on_event(trace_event)
                    await on_event(
                        AgentEventData(
                            event=AgentEvent.PHASE_CHANGE,
                            data={
                                "phase": "subconscious",
                                "status": "end",
                                "output": _subconscious_event_payload(
                                    subconscious_output
                                ),
                            },
                        )
                    )
        except Exception as exc:
            logger.warning("Subconscious pre-phase failed: %s", exc)

    skill_query = user_message
    if mode == "heartbeat" and heartbeat_context:
        priority_context = {
            key: heartbeat_context[key]
            for key in (
                "pending_protected_replacements",
                "open_protected_reversions",
                "pending_import_review",
                "pending_skill_proposals",
                "backlog",
            )
            if key in heartbeat_context
        }
        skill_query = (
            json.dumps(priority_context, default=str)
            + "\n"
            + json.dumps(heartbeat_context, default=str)
        )[:4000]
    skill_selection = await select_skills(
        registry,
        tool_context,
        query=skill_query,
        max_skills=5 if mode == "heartbeat" else 4,
    )
    await record_selection(
        pool,
        skill_selection,
        registry=registry,
        session_id=session_id,
        surface=surface,
        tool_context=tool_context,
        query=skill_query,
    )

    # 4. Build system prompt
    system_prompt = await build_system_prompt(
        mode,
        registry,
        agent_profile,
        subconscious_output=subconscious_output,
        has_backlog_tasks=has_backlog_tasks,
        is_group=is_group,
        interlocutor=user_label,
        surface=surface,
        active_skills=skill_selection.skills,
        available_skills=skill_selection.available,
        allowed_tool_names=set(skill_selection.allowed_tool_names),
    )

    # 5. Build enriched user message
    enriched_parts: list[str] = []

    # Add subconscious signals (rendered by the DB inside the conn scope)
    if sub_signals:
        enriched_parts.append(sub_signals)

    if continuity_context:
        enriched_parts.append(continuity_context)

    # Add memory context (chat mode)
    if memory_context:
        enriched_parts.append(memory_context)

    # Add the actual user message
    if mode == "chat":
        enriched_parts.append(f"[USER MESSAGE]\n{user_message}")
    else:
        enriched_parts.append(user_message)

    enriched_user_message = (
        "\n\n".join(enriched_parts) if enriched_parts else user_message
    )
    enriched_user_content = _user_content_with_visuals(
        enriched_user_message,
        visual_attachments if mode == "chat" else None,
    )

    approval_wait_extension = 0
    if on_approval is None:
        from services.operator_approval import create_operator_approval_callback

        on_approval, approval_wait_extension = await create_operator_approval_callback(
            pool,
            tool_context=tool_context,
            session_id=session_id,
            heartbeat_id=heartbeat_id,
            surface=surface,
        )

    question_wait_extension = 0
    if tool_context == ToolContext.CHAT:
        from services.agent_questions import question_timeout_seconds

        question_wait_extension = await question_timeout_seconds(pool)

    # 6. Configure AgentLoop with mode-specific defaults
    if mode == "chat":
        effective_timeout = (
            (timeout_seconds or 120.0)
            + approval_wait_extension
            + question_wait_extension
        )
        effective_max_tokens = max_tokens or 4096
        loop_config = AgentLoopConfig(
            tool_context=tool_context,
            system_prompt=system_prompt,
            llm_config=llm_config,
            registry=registry,
            pool=pool,
            energy_budget=energy_budget,  # None = unlimited for chat
            max_iterations=max_iterations,  # None = timeout-based only
            timeout_seconds=effective_timeout,
            temperature=0.7,
            max_tokens=effective_max_tokens,
            session_id=session_id,
            is_group=is_group,
            is_operator=is_operator,
            on_approval=on_approval,
            allowed_tool_names=set(skill_selection.allowed_tool_names),
            active_skill_names=[sk.name for sk in skill_selection.skills],
            surface=surface,
            tool_surface_input_hash=hash_input_text(skill_query),
            available_skill_count=len(skill_selection.available),
        )
    else:
        effective_timeout = (
            timeout_seconds or (300.0 if has_backlog_tasks else 120.0)
        ) + approval_wait_extension
        effective_max_tokens = max_tokens or (4096 if has_backlog_tasks else 2048)
        loop_config = AgentLoopConfig(
            tool_context=tool_context,
            system_prompt=system_prompt,
            llm_config=llm_config,
            registry=registry,
            pool=pool,
            energy_budget=energy_budget or 20,
            max_iterations=None,  # timeout-based
            timeout_seconds=effective_timeout,
            temperature=0.7,
            max_tokens=effective_max_tokens,
            heartbeat_id=heartbeat_id,
            is_operator=is_operator,
            on_approval=on_approval,
            enable_planning=True,
            continuation_prompt=(
                "You finished without taking further action. "
                "Review your output — is there anything you should recall, "
                "record, or act on before finishing?"
            ),
            max_continuations=2 if has_backlog_tasks else 1,
            context_overrides=context_overrides,
            allowed_tool_names=set(skill_selection.allowed_tool_names),
            active_skill_names=[sk.name for sk in skill_selection.skills],
            surface=surface,
            tool_surface_input_hash=hash_input_text(skill_query),
            available_skill_count=len(skill_selection.available),
        )

    # 7. Run agent loop
    agent = AgentLoop(loop_config)
    if on_event:
        loop_config.on_event = on_event

    if streaming:
        # For streaming, we need to collect results while yielding events.
        # The caller should use stream_agent() instead.
        # Fall back to non-streaming here.
        pass

    result = await agent.run(
        enriched_user_message, history=history, user_content=enriched_user_content
    )
    return result


async def stream_agent(
    pool: "asyncpg.Pool",
    registry: "ToolRegistry",
    *,
    user_message: str,
    mode: Literal["chat", "heartbeat"] = "chat",
    energy_budget: int | None = None,
    tool_context: ToolContext | None = None,
    history: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    user_label: str | None = None,
    surface: str = "chat",
    agent_profile: dict[str, Any] | None = None,
    is_group: bool = False,
    is_operator: bool = False,
    dsn: str | None = None,
    has_backlog_tasks: bool = False,
    timeout_seconds: float | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    on_approval: "Callable[[str, dict[str, Any]], Awaitable[bool | dict[str, Any]]] | None" = None,
    prompt_addenda: list[str] | None = None,
    visual_attachments: list[dict[str, Any]] | None = None,
) -> AsyncIterator[AgentEventData]:
    """
    Streaming variant of run_agent(). Yields AgentEventData as they happen.

    ``on_approval`` is called before protected tools. A local callback may
    return a boolean; the default durable callback returns an exact one-shot
    approval request id after an identity-checked phone decision.

    Used by the SSE chat endpoint to stream tokens to the frontend.
    """
    from core.agent_api import db_dsn_from_env
    from core.cognitive_memory_api import CognitiveMemory, render_chat_memory_context_db

    dsn = dsn or db_dsn_from_env()
    history = history or []

    if tool_context is None:
        tool_context = ToolContext.CHAT if mode == "chat" else ToolContext.HEARTBEAT

    # 1. Load LLM config and run pre-phases
    async with pool.acquire() as conn:
        llm_key = "llm.chat" if mode == "chat" else "llm.heartbeat"
        llm_fallback = "llm" if mode == "chat" else None
        llm_config = await load_llm_config(conn, llm_key, fallback_key=llm_fallback)

        # Hydrate memory
        memory_context = ""
        continuity_context = ""
        context = None
        if mode == "chat":
            try:
                yield AgentEventData(
                    event=AgentEvent.PHASE_CHANGE,
                    data={"phase": "memory_recall", "status": "start"},
                )
                mem_client = CognitiveMemory(pool)
                context = await mem_client.hydrate(
                    user_message,
                    memory_limit=10,
                    include_partial=True,
                    include_identity=True,
                    include_worldview=True,
                    include_emotional_state=True,
                    include_goals=True,
                    include_drives=True,
                    session_id=session_id,
                    # Sensitivity enforcement (#92): a group room never
                    # receives private-marked memories — the channel prompt's
                    # promise, made mechanical at the recall layer.
                    exclude_sensitive=False,
                )
                memory_context = await render_chat_memory_context_db(
                    conn, context, max_memories=10
                )

                yield AgentEventData(
                    event=AgentEvent.PHASE_CHANGE,
                    data={
                        "phase": "memory_recall",
                        "status": "end",
                        "count": len(context.memories),
                        "memories": _memory_event_previews(context.memories),
                    },
                )
            except Exception as exc:
                logger.warning("Memory hydration failed: %s", exc)
            continuity_context = await render_chat_continuity_context_db(
                conn,
                session_id,
                exclude_sensitive=False,
            )

        # Run subconscious
        subconscious_output = SubconsciousOutput()
        sub_signals = ""
        try:
            inline_enabled = bool(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('chat.inline_subconscious_enabled'), true)"
                )
            )
            if inline_enabled:
                yield AgentEventData(
                    event=AgentEvent.PHASE_CHANGE,
                    data={"phase": "subconscious", "status": "start"},
                )
                subconscious_output = await run_subconscious_appraisal(
                    conn,
                    user_message,
                    continuity_context,
                    hydrated_context=context,
                )
                sub_signals = await render_subconscious_signals_db(
                    conn, subconscious_output
                )
                for trace_event in _subconscious_trace_events(subconscious_output):
                    yield trace_event
                yield AgentEventData(
                    event=AgentEvent.PHASE_CHANGE,
                    data={
                        "phase": "subconscious",
                        "status": "end",
                        "output": _subconscious_event_payload(subconscious_output),
                    },
                )
        except Exception as exc:
            logger.warning("Subconscious pre-phase failed: %s", exc)

    skill_selection = await select_skills(
        registry,
        tool_context,
        query=user_message,
        max_skills=4,
    )
    await record_selection(
        pool,
        skill_selection,
        registry=registry,
        session_id=session_id,
        surface=surface,
        tool_context=tool_context,
        query=user_message,
    )

    # Build system prompt
    system_prompt = await build_system_prompt(
        mode,
        registry,
        agent_profile,
        subconscious_output=subconscious_output,
        has_backlog_tasks=has_backlog_tasks,
        is_group=is_group,
        interlocutor=user_label,
        surface=surface,
        active_skills=skill_selection.skills,
        available_skills=skill_selection.available,
        allowed_tool_names=set(skill_selection.allowed_tool_names),
        prompt_addenda=prompt_addenda,
    )

    # Build enriched user message (signals were rendered by the DB in-scope)
    enriched_parts: list[str] = []
    if sub_signals:
        enriched_parts.append(sub_signals)
    if continuity_context:
        enriched_parts.append(continuity_context)
    if memory_context:
        enriched_parts.append(memory_context)
    enriched_parts.append(f"[USER MESSAGE]\n{user_message}")
    enriched_user_message = "\n\n".join(enriched_parts)
    enriched_user_content = _user_content_with_visuals(
        enriched_user_message,
        visual_attachments if mode == "chat" else None,
    )

    # Configure loop. Limits derive from config (Bar #1) with sane fallbacks;
    # an explicit caller arg still wins.
    def _cfg_num(key: str, fallback: float) -> float:
        val = llm_config.get(key) if isinstance(llm_config, dict) else None
        try:
            return float(val) if val else fallback
        except (TypeError, ValueError):
            return fallback

    effective_timeout = timeout_seconds or _cfg_num("timeout_seconds", 120.0)
    if on_approval is None:
        from services.operator_approval import create_operator_approval_callback

        on_approval, approval_wait_extension = await create_operator_approval_callback(
            pool,
            tool_context=tool_context,
            session_id=session_id,
            heartbeat_id=None,
            surface=surface,
        )
        effective_timeout += approval_wait_extension
    if tool_context == ToolContext.CHAT:
        from services.agent_questions import question_timeout_seconds

        effective_timeout += await question_timeout_seconds(pool)
    effective_max_tokens = int(max_tokens or _cfg_num("max_tokens", 4096))
    effective_temperature = (
        temperature if temperature is not None else _cfg_num("temperature", 0.7)
    )
    loop_config = AgentLoopConfig(
        tool_context=tool_context,
        system_prompt=system_prompt,
        llm_config=llm_config,
        registry=registry,
        pool=pool,
        energy_budget=energy_budget,
        max_iterations=None,
        timeout_seconds=effective_timeout,
        temperature=effective_temperature,
        max_tokens=effective_max_tokens,
        session_id=session_id,
        is_group=is_group,
        is_operator=is_operator,
        on_approval=on_approval,
        allowed_tool_names=set(skill_selection.allowed_tool_names),
        active_skill_names=[sk.name for sk in skill_selection.skills],
        surface=surface,
        tool_surface_input_hash=hash_input_text(user_message),
        available_skill_count=len(skill_selection.available),
    )

    agent = AgentLoop(loop_config)
    async for event in agent.stream(
        enriched_user_message, history=history, user_content=enriched_user_content
    ):
        yield event
