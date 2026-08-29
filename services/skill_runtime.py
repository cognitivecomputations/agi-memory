"""Skill-first runtime helpers.

Skills are the model-facing capability layer. Tools are implementation details:
the LLM sees only skill discovery tools plus tools bound by selected/activated
skills. This keeps prompts smaller and makes capability use intentional.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.tools.base import ToolContext
from skills.base import MCPBinding, SkillCategory, SkillContext, SkillSpec
from skills.loader import load_skills

if False:  # pragma: no cover - typing only
    from core.tools.config import MCPServerConfig
    from core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


DISCOVERY_TOOL_NAMES = {
    "list_skills",
    "use_skill",
    "propose_skill",
    "queue_user_message",
    "ask_user",
}

# The read-only floor every turn gets, regardless of which skills activate.
#
# Skill-gating is right for tools that act — sending, writing, spending. It is
# wrong for the ones an assistant reaches for constantly just to answer, because
# a selector that fails to guess the topic leaves the agent unable to look
# anything up. Measured: seven of ten ordinary requests activated `core-memory`
# alone, so a question about email or the calendar had no way to reach either.
#
# Everything here is read-only and cheap. The gate still earns its keep on
# email_send, shell, and the rest.
#
# CHAT ONLY, deliberately. A live turn means a person asked; an autonomous
# heartbeat means nobody did, and reading someone's mail unprompted is exactly
# what `integrations.gmail.heartbeat_digest_enabled` exists to authorize. A
# convenience floor must not become a way around a consent gate.
ALWAYS_AVAILABLE_TOOL_NAMES = {
    "web_search",
    "web_fetch",
    "calendar_events",
    "email_list",
    "email_search",
    "search_contacts",
    "get_contact",
}
DEFAULT_SKILL_NAMES = {"core-memory"}
HEARTBEAT_DEFAULT_SKILL_NAMES = {"core-memory", "self-reflection"}
GMAIL_HEARTBEAT_DIGEST_CONFIG_KEY = "integrations.gmail.heartbeat_digest_enabled"
AUTO_ACTIVATE_SCORE_THRESHOLD = 5
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "before",
    "could",
    "did",
    "does",
    "for",
    "from",
    "give",
    "have",
    "help",
    "into",
    "last",
    "more",
    "next",
    "please",
    "show",
    "that",
    "the",
    "then",
    "this",
    "time",
    "want",
    "what",
    "when",
    "where",
    "with",
    "would",
    "you",
}


@dataclass(frozen=True)
class SkillSelection:
    skills: list[SkillSpec]
    allowed_tool_names: set[str]
    # Every candidate with its score and whether a specialized gate excluded it.
    # Kept so the selection decision can be recorded (#0.6): a selector that
    # quietly fails to activate the right skill is otherwise invisible.
    considered: list[dict[str, Any]] = field(default_factory=list)
    # Full catalog for this context — used for the compact skill index in the
    # system prompt, so the model can discover skills without a list_skills call.
    available: list[SkillSpec] = field(default_factory=list)


def tool_context_to_skill_context(context: ToolContext) -> SkillContext:
    if context == ToolContext.HEARTBEAT:
        return SkillContext.HEARTBEAT
    if context == ToolContext.MCP:
        return SkillContext.MCP
    return SkillContext.CHAT


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9_]+", text.lower())
        if len(t) >= 3 and t not in STOPWORDS
    }


def _score_skill(skill: SkillSpec, query_tokens: set[str]) -> int:
    if not query_tokens:
        return 0
    haystack = " ".join([skill.name, skill.description, skill.content[:1500]]).lower()
    score = 0
    # Aliases score as name tokens: they are the words a person would use, and
    # the skill's own name usually is not one of them.
    name_tokens = _tokens(skill.name.replace("-", " "))
    for alias in getattr(skill, "aliases", []) or []:
        name_tokens |= _tokens(alias.replace("-", " "))
    desc_tokens = _tokens(skill.description)
    for tok in query_tokens:
        if tok in name_tokens:
            score += 5
        elif tok in desc_tokens:
            score += 3
        elif tok in haystack:
            score += 1
    return score


def _passes_specialized_gate(skill: SkillSpec, query_tokens: set[str]) -> bool:
    """Avoid auto-activating narrow integrations from generic overlap. They stay
    discoverable through `list_skills`/`use_skill`."""
    gates = {
        "twitter-research": {"twitter", "tweet", "tweets", "x", "social", "sentiment"},
        "youtube-analytics": {
            "youtube",
            "video",
            "channel",
            "subscriber",
            "subscribers",
        },
        "image-gen": {"image", "picture", "draw", "illustration", "generate", "visual"},
        "cost-report": {
            "cost",
            "costs",
            "spend",
            "spent",
            "usage",
            "tokens",
            "budget",
            "bill",
        },
        "humanizer": {"humanize", "natural", "voice", "rewrite", "prose", "ai"},
        "skill-authoring": {
            "author",
            "write",
            "create",
            "update",
            "revise",
            "skill",
            "skills",
            "procedure",
        },
    }
    # Only consulted on the lexical fallback path (embedding service down).
    # Semantic selection needs no gates: a skill that does not match the request
    # simply does not rank.
    required = gates.get(skill.name)
    return True if required is None else bool(query_tokens & required)


async def _gmail_heartbeat_digest_authorized(registry: "ToolRegistry") -> bool:
    """Autonomous email checks require both a connected account and a DB grant.

    Google OAuth authorizes provider access. This config authorizes Hexis to use
    that access from heartbeat without a live user turn.
    """
    pool = getattr(registry, "pool", None)
    if pool is None:
        return False
    try:
        async with pool.acquire() as conn:
            enabled = await conn.fetchval(
                "SELECT COALESCE(get_config_bool($1), FALSE)",
                GMAIL_HEARTBEAT_DIGEST_CONFIG_KEY,
            )
            if not enabled:
                return False
            raw = await conn.fetchval("SELECT integration_status('gmail')")
    except Exception:
        return False
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        return False
    return any(
        isinstance(item, dict)
        and item.get("connector_id") == "gmail"
        and item.get("status") == "connected"
        for item in payload.get("connections", [])
    )


def skill_bound_tools(skill: SkillSpec) -> list[str]:
    """Tools a skill may use. `bound_tools` is preferred; `requires.tools` is a
    fallback for older skill files."""
    return list(dict.fromkeys([*(skill.bound_tools or []), *skill.requires_tools]))


def _plugin_skill_dirs(registry: "ToolRegistry") -> list[Path]:
    """Skill directories contributed by plugins, carried on the tool registry."""
    dirs = getattr(registry, "extra_skill_dirs", None) or []
    return [Path(d) for d in dirs]


def synthesize_implicit_mcp_skills(
    configs: list["MCPServerConfig"],
    bound_servers: set[str],
) -> list[SkillSpec]:
    """Back-compat (#41): every configured MCP server not bound by a manifest
    gets an implicit `mcp-<server>` skill, so existing integrations stay
    reachable through the skills front door with zero manifest work."""
    implicit: list[SkillSpec] = []
    for config in configs:
        if not config.enabled or config.name in bound_servers:
            continue
        implicit.append(
            SkillSpec(
                name=f"mcp-{config.name}",
                description=(
                    f"Tools from the configured MCP server '{config.name}'. "
                    "Write a proper skill manifest to customize instructions and "
                    "bound tools."
                ),
                content=(
                    f"This skill exposes every tool of the configured MCP server "
                    f"'{config.name}'. Activating it connects the server (if not "
                    "already running) and unlocks its tools for this turn."
                ),
                category=SkillCategory.SYSTEM,
                bound_tools=[f"mcp_{config.name}_*"],
                mcp_binding=MCPBinding(server=config.name),
                provenance={"generated": "mcp_server_config"},
            )
        )
    return implicit


def load_available_skills(
    registry: "ToolRegistry",
    context: ToolContext,
    *,
    include_unmet: bool = False,
    mcp_configs: list["MCPServerConfig"] | None = None,
) -> list[SkillSpec]:
    """All loadable skills for a context, including plugin-provided ones and
    (when mcp_configs is supplied) implicit skills for configured MCP servers
    no manifest binds."""
    skills = load_skills(
        tool_context_to_skill_context(context),
        available_tools=set(registry.list_names()),
        available_config=None,  # tool handlers validate credentials at execution time
        extra_dirs=_plugin_skill_dirs(registry),
        include_unmet=include_unmet,
    )
    if mcp_configs:
        bound_servers = {
            s.mcp_binding.server for s in skills if s.mcp_binding is not None
        }
        skills = [*skills, *synthesize_implicit_mcp_skills(mcp_configs, bound_servers)]
    return skills


async def _mcp_configs(registry: "ToolRegistry") -> list["MCPServerConfig"]:
    try:
        config = await registry.get_config()
        return list(config.mcp_servers or [])
    except Exception:
        return []


async def _semantic_selection_enabled(pool: Any) -> bool:
    if pool is None:
        return False
    try:
        async with pool.acquire() as conn:
            return bool(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('skills.semantic_selection_enabled'), TRUE)"
                )
            )
    except Exception:
        return False


async def _semantic_thresholds(pool: Any) -> tuple[float, float]:
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COALESCE(get_config_float('skills.semantic_z_threshold'), 2.0) AS z, "
                "COALESCE(get_config_float('skills.semantic_threshold'), 0.40) AS floor"
            )
        return float(row["z"]), float(row["floor"])
    except Exception:
        return 2.0, 0.40


def _explicitly_named(
    skill: SkillSpec, lowered_query: str, query_tokens: set[str]
) -> bool:
    """Did the request name this skill, or one of its tools, outright?

    Not a keyword list: these are identifiers the system itself defines, so a
    match is exact rather than a guess about phrasing. This is the backstop for
    domain jargon — a query about a "protected replacement decision" names
    `protected_replacement_review` almost verbatim while scoring flat against
    every skill description.
    """
    if skill.name in query_tokens or skill.name.replace("-", " ") in lowered_query:
        return True
    for phrase in getattr(skill, "activation_phrases", None) or []:
        normalized = " ".join(str(phrase).lower().split())
        if normalized and re.search(
            rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])",
            lowered_query,
        ):
            return True
    for tool in skill_bound_tools(skill):
        words = tool.split("_")
        if len(words) < 2:
            # Single-word tool names ("shell", "browser") are ordinary English
            # and would match constantly.
            continue
        if " ".join(words) in lowered_query:
            return True
        # The leading pair, when both halves are distinctive enough to be
        # jargon rather than English. "protected replacement" identifies the
        # tool family even when the request says "decision" instead of
        # "review"; "promote to" is skipped because "to" is not a signal.
        head, second = words[0], words[1]
        if len(head) >= 4 and len(second) >= 4 and f"{head} {second}" in lowered_query:
            return True
    return False


def skill_embedding_text(skill: SkillSpec) -> str:
    """What a skill *means*, as text to embed.

    Aliases survive here as example phrasings rather than match tokens: they
    enrich the vector instead of acting as a lookup table, so nobody has to
    guess in advance that "book time" implies the calendar.
    """
    parts = [skill.name.replace("-", " "), (skill.description or "").strip()]
    aliases = [a for a in (getattr(skill, "aliases", None) or []) if a]
    if aliases:
        parts.append("Also called: " + ", ".join(aliases))
    triggers = [
        phrase
        for phrase in (getattr(skill, "activation_phrases", None) or [])
        if phrase
    ]
    if triggers:
        parts.append("Explicit requests: " + ", ".join(triggers))
    return ". ".join(part for part in parts if part)


async def _semantic_scores(
    pool: Any, skills: list[SkillSpec], query: str
) -> dict[str, float]:
    """Cosine similarity per skill, one round trip.

    Every skill text plus the query goes to `get_embedding` in a single call and
    the cosine happens in Postgres. In the steady state this costs no model
    calls at all: `get_embedding` caches by content hash, and skill descriptions
    do not change between turns.
    """
    if pool is None or not skills or not query.strip():
        return {}
    names = [s.name for s in skills]
    texts = [skill_embedding_text(s) for s in skills]
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT skill_name, similarity FROM rank_skills_by_similarity($1::text[], $2::text[], $3)",
                names,
                texts,
                query[:2000],
            )
        return {r["skill_name"]: float(r["similarity"]) for r in rows}
    except Exception:
        logger.debug("Semantic skill ranking unavailable; falling back", exc_info=True)
        return {}


def _select_by_distribution(
    scores: dict[str, float], *, z_threshold: float, floor: float
) -> list[tuple[float, str]]:
    """Skills that stand out from the run, ranked.

    The gate is the *shape* of the distribution, not an absolute cutoff.
    Absolute similarity from the embedding model is compressed and
    query-dependent — measured, signal spans 0.46–0.73 and noise 0.40–0.54, so
    any fixed line cuts through both. A peaked distribution means the request is
    about something in particular; a flat one means it is not, which is the
    correct read of "hello".
    """
    values = list(scores.values())
    if len(values) < 3:
        return []
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    sd = variance**0.5
    if sd <= 0:
        return []
    picked = [
        (score, name)
        for name, score in scores.items()
        if score >= floor and (score - mean) / sd >= z_threshold
    ]
    return sorted(picked, key=lambda item: (-item[0], item[1]))


async def select_skills(
    registry: "ToolRegistry",
    tool_context: ToolContext,
    *,
    query: str = "",
    max_skills: int = 4,
) -> SkillSelection:
    """Select active skills for this turn and derive the exposed tool set."""
    available_tools = set(registry.list_names())
    skills = load_available_skills(
        registry, tool_context, mcp_configs=await _mcp_configs(registry)
    )

    default_names = (
        set(HEARTBEAT_DEFAULT_SKILL_NAMES)
        if tool_context == ToolContext.HEARTBEAT
        else set(DEFAULT_SKILL_NAMES)
    )
    if (
        tool_context == ToolContext.HEARTBEAT
        and await _gmail_heartbeat_digest_authorized(registry)
    ):
        default_names.add("email-digest")
    selected: list[SkillSpec] = [s for s in skills if s.name in default_names]

    selected_names = {s.name for s in selected}
    q_tokens = _tokens(query)
    considered: list[dict[str, Any]] = []
    candidates = [s for s in skills if s.name not in selected_names]
    by_name = {s.name: s for s in candidates}

    ranked: list[tuple[float, str]] = []
    semantic_used = False
    pool = getattr(registry, "pool", None)
    if await _semantic_selection_enabled(pool):
        scores = await _semantic_scores(pool, candidates, query)
        if scores:
            semantic_used = True
            z_threshold, floor = await _semantic_thresholds(pool)
            ranked = _select_by_distribution(
                scores, z_threshold=z_threshold, floor=floor
            )
            considered = [
                {"name": name, "score": round(score, 4), "gated": False}
                for score, name in sorted(
                    ((v, k) for k, v in scores.items()), key=lambda i: -i[0]
                )
            ]

    if not semantic_used:
        # Lexical fallback: only when the embedding service is unavailable.
        for candidate in candidates:
            gated = not _passes_specialized_gate(candidate, q_tokens)
            score = _score_skill(candidate, q_tokens)
            considered.append({"name": candidate.name, "score": score, "gated": gated})
            if not gated and score >= AUTO_ACTIVATE_SCORE_THRESHOLD:
                ranked.append((float(score), candidate.name))
        ranked.sort(key=lambda item: (-item[0], item[1]))

    # Explicit mention is not a similarity question, and it covers what
    # embeddings are worst at: internal jargon. "protected replacement" and
    # "HMX" have no useful representation in a general embedding model — the
    # whole distribution goes flat — but they name a tool exactly, and an exact
    # name is a precise signal rather than a guessed keyword.
    lowered = query.lower()
    already = {name for _score, name in ranked}
    for candidate in candidates:
        if candidate.name in already:
            continue
        if _explicitly_named(candidate, lowered, q_tokens):
            ranked.insert(0, (1.0, candidate.name))
            already.add(candidate.name)

    for _score, name in ranked:
        skill = by_name.get(name)
        if skill is None or name in selected_names:
            continue
        selected.append(skill)
        selected_names.add(name)
        if len(selected) >= max_skills:
            break

    allowed = set(DISCOVERY_TOOL_NAMES)
    if tool_context != ToolContext.HEARTBEAT:
        allowed.update(t for t in ALWAYS_AVAILABLE_TOOL_NAMES if t in available_tools)
    for skill in selected:
        allowed.update(t for t in skill_bound_tools(skill) if t in available_tools)

    return SkillSelection(
        skills=selected,
        allowed_tool_names=allowed,
        available=skills,
        considered=sorted(considered, key=lambda item: -item["score"]),
    )


async def record_selection(
    pool: Any,
    selection: "SkillSelection",
    *,
    registry: "ToolRegistry | None" = None,
    session_id: str | None,
    surface: str,
    tool_context: ToolContext,
    query: str,
) -> None:
    """Record what the selector decided. Advisory — never breaks a turn.

    Which skills activate decides which tools exist, and a tool outside the
    active set is hard-refused. Both decisions were previously unrecorded, so a
    selector that failed to activate the right skill left no trace.
    """
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT record_skill_selection($1::uuid, $2, $3, $4, $5::text[], $6::jsonb, $7)",
                _uuid_or_none(session_id),
                surface,
                getattr(tool_context, "value", str(tool_context)),
                query[:400],
                [s.name for s in selection.skills],
                json.dumps(selection.considered[:25]),
                len(selection.allowed_tool_names),
            )
    except Exception:
        logger.debug("Skill-selection telemetry failed (non-fatal)", exc_info=True)

    if registry is not None:
        # The selection row records why skills won. The immutable surface audit
        # records whether their tools actually survived registry + config
        # resolution. Keep these separate so existing telemetry remains stable.
        from services.tool_surface_audit import record_tool_surface_decision

        await record_tool_surface_decision(
            pool,
            registry=registry,
            selection=selection,
            session_id=session_id,
            surface=surface,
            tool_context=tool_context,
            query=query,
        )


async def record_gate_refusal(
    pool: Any,
    *,
    session_id: str | None,
    tool_name: str,
    reason: str,
    active_skills: list[str],
) -> None:
    """Record a tool the gate turned away. Advisory — never breaks a turn."""
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT record_tool_gate_refusal($1::uuid, $2, $3, $4::text[])",
                _uuid_or_none(session_id),
                tool_name,
                reason,
                active_skills,
            )
    except Exception:
        logger.debug("Tool-gate telemetry failed (non-fatal)", exc_info=True)


def _uuid_or_none(value: str | None) -> str | None:
    import uuid as _uuid

    try:
        return str(_uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


def format_skills_prompt(
    active: list[SkillSpec],
    available: list[SkillSpec] | None = None,
) -> str:
    """Compact skill section for the system prompt.

    One index line per skill — never full skill bodies. Full instructions are
    fetched on demand via `use_skill`, and tool schemas ride the structured
    tool-calling API, so this block stays flat regardless of skill size.
    """
    lines = [
        "## Skills",
        "Use skills first: capabilities are packaged as skills, and a skill's "
        "tools are exposed through the tool API only while that skill is active. "
        "If the task needs a capability that is not active, call `use_skill` with "
        "the skill's name — it returns the skill's full instructions and unlocks "
        "its tools for this turn. `list_skills` shows the catalog with bound tools. "
        "If no existing skill fits a reusable need, call `propose_skill`; it creates "
        "a reviewable proposal only, not a live skill file.",
    ]
    if active:
        lines.append("Active now:\n" + "\n".join(s.to_index_line() for s in active))
    active_names = {s.name for s in active}
    inactive = [s for s in (available or []) if s.name not in active_names]
    if inactive:
        lines.append(
            "Available (activate with `use_skill`):\n"
            + "\n".join(s.to_index_line() for s in inactive)
        )
    return "\n\n".join(lines)


async def skill_catalog(
    registry: "ToolRegistry", context: ToolContext
) -> list[dict[str, Any]]:
    """The honest answer to "what can I do?" (#39): every skill for this
    context — including ones whose requirements currently fail — with a
    tri-state status and the exact next step. No silent drops, no dead-ends."""
    available_tools = set(registry.list_names())
    mcp_configs = await _mcp_configs(registry)
    server_names = {c.name for c in mcp_configs if c.enabled}
    skills = load_available_skills(
        registry, context, include_unmet=True, mcp_configs=mcp_configs
    )
    catalog: list[dict[str, Any]] = []
    for s in skills:
        status, missing, next_step = s.usability(available_tools, server_names)
        if s.mcp_binding is not None:
            # MCP tools exist only after activation: list them from the
            # manifest, not from the live registry.
            bound = skill_bound_tools(s)
            transport = f"mcp:{s.mcp_binding.server}"
        else:
            bound = [t for t in skill_bound_tools(s) if t in available_tools]
            transport = None
        entry: dict[str, Any] = {
            "name": s.name,
            "description": s.description,
            "category": s.category.value,
            "bound_tools": bound,
            "status": status,
        }
        if missing:
            entry["missing"] = missing
        if next_step:
            entry["next_step"] = next_step
        if transport:
            entry["transport"] = transport
            entry["note"] = (
                "MCP tools become callable after use_skill activates this skill."
            )
        catalog.append(entry)
    return catalog


async def get_skill_by_name(
    registry: "ToolRegistry", context: ToolContext, name: str
) -> SkillSpec | None:
    wanted = name.strip().lower()
    skills = load_available_skills(
        registry, context, include_unmet=True, mcp_configs=await _mcp_configs(registry)
    )
    for skill in skills:
        if skill.name.lower() == wanted:
            return skill
    return None
