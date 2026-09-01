"""Refresh and resolve consent-first automation suggestions.

Postgres owns proposal state and the accept/dismiss transition. This module is
the thin filesystem edge needed to discover ``blueprint:`` blocks in installed
skills, plus the channel wrapper used for explicit numbered replies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, TYPE_CHECKING

from skills.loader import discover_skill_dirs, load_skills_from_dir

if TYPE_CHECKING:
    import asyncpg
    from skills.base import SkillSpec

logger = logging.getLogger("automation_suggestions")

_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:api[_ -]?key|password|secret|access[_ -]?token|refresh[_ -]?token)"
    r"\s*[:=]\s*[`'\"]?[A-Za-z0-9_./+\-=]{8,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{16,}"
)
_SAFE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9:._/-]{1,299}$")


def _coerce_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _installed_skills() -> list["SkillSpec"]:
    skills: list[SkillSpec] = []
    seen: set[str] = set()
    for directory in discover_skill_dirs():
        for skill in load_skills_from_dir(directory):
            if skill.name in seen:
                continue
            seen.add(skill.name)
            skills.append(skill)
    return skills


def normalize_skill_blueprint(skill: "SkillSpec") -> dict[str, Any] | None:
    """Turn one skill frontmatter block into exact manage_schedule args."""
    raw = skill.blueprint
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"skill {skill.name} blueprint must be an object")

    title = str(raw.get("title") or "").strip()
    rationale = str(raw.get("rationale") or "").strip()
    if not title or not rationale:
        raise ValueError(f"skill {skill.name} blueprint requires title and rationale")

    nested = raw.get("task_spec")
    if nested is not None and not isinstance(nested, dict):
        raise ValueError(f"skill {skill.name} blueprint.task_spec must be an object")
    if isinstance(nested, dict):
        task_spec = dict(nested)
    else:
        # The flat form is intentionally supported by Hexis's dependency-free
        # frontmatter parser as well as PyYAML.
        task_spec = {
            key: raw[key]
            for key in (
                "name",
                "description",
                "schedule",
                "schedule_kind",
                "timezone",
                "action_kind",
                "message",
                "goal_title",
                "max_runs",
                "delivery_mode",
                "delivery_channel",
                "delivery_target_id",
                "delivery_topic",
                "delivery_webhook_url",
            )
            if key in raw
        }
    task_spec.setdefault("name", title)
    task_spec.setdefault("action", "create")
    task_spec.setdefault("action_kind", "queue_user_message")
    task_spec.setdefault("delivery_mode", "outbox")
    if not str(task_spec.get("schedule") or task_spec.get("schedule_kind") or "").strip():
        raise ValueError(f"skill {skill.name} blueprint requires a schedule")
    if task_spec.get("action_kind") == "queue_user_message" and not str(
        task_spec.get("message") or ""
    ).strip():
        raise ValueError(f"skill {skill.name} blueprint requires a reminder message")

    secret_surface = json.dumps(
        {"title": title, "rationale": rationale, "task_spec": task_spec},
        sort_keys=True,
        default=str,
    )
    if _SECRET_VALUE_RE.search(secret_surface):
        raise ValueError(f"skill {skill.name} blueprint appears to contain secret material")

    raw_key = str(raw.get("dedup_key") or f"blueprint:{skill.name}").strip().lower()
    if not _SAFE_KEY_RE.fullmatch(raw_key):
        raise ValueError(
            f"skill {skill.name} blueprint dedup_key must be 2-300 lowercase safe characters"
        )
    return {
        "dedup_key": raw_key,
        "title": title,
        "rationale": rationale,
        "task_spec": task_spec,
        "metadata": {
            "skill": skill.name,
            "skill_source": skill.source,
            "blueprint_digest": hashlib.sha256(secret_surface.encode("utf-8")).hexdigest(),
        },
    }


async def refresh_automation_suggestions(
    pool: "asyncpg.Pool",
    *,
    skills: list["SkillSpec"] | None = None,
) -> dict[str, Any]:
    """Run one due catalog + installed-skill blueprint refresh."""
    async with pool.acquire() as conn:
        claimed = bool(await conn.fetchval("SELECT claim_automation_suggestion_refresh()"))
    if not claimed:
        return {"skipped": True, "reason": "disabled_not_due_or_agent_not_ready"}

    result: dict[str, Any]
    try:
        async with pool.acquire() as conn:
            catalog = _coerce_object(
                await conn.fetchval("SELECT refresh_automation_suggestion_catalog()")
            )
            blueprint_enabled = bool(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('automation.suggestions.blueprint_enabled'), TRUE)"
                )
            )

        registered = 0
        existing = 0
        invalid: list[dict[str, str]] = []
        if blueprint_enabled:
            for skill in skills if skills is not None else _installed_skills():
                if skill.blueprint is None:
                    continue
                try:
                    blueprint = normalize_skill_blueprint(skill)
                    if blueprint is None:
                        continue
                    async with pool.acquire() as conn:
                        proposed = _coerce_object(
                            await conn.fetchval(
                                """
                                SELECT propose_automation(
                                    'blueprint', $1, $2, $3, $4::jsonb, $5::jsonb
                                )
                                """,
                                blueprint["dedup_key"],
                                blueprint["title"],
                                blueprint["rationale"],
                                json.dumps(blueprint["task_spec"]),
                                json.dumps(blueprint["metadata"]),
                            )
                        )
                    if proposed.get("created"):
                        registered += 1
                    else:
                        existing += 1
                except Exception as exc:
                    logger.warning("Invalid automation blueprint in %s: %s", skill.source, exc)
                    invalid.append({"skill": skill.name, "error": str(exc)[:500]})

        result = {
            "catalog": catalog,
            "blueprints_registered": registered,
            "blueprints_existing": existing,
            "blueprints_invalid": invalid,
        }
    except Exception as exc:
        logger.error("Automation suggestion refresh failed: %s", exc, exc_info=True)
        result = {"error": str(exc), "status": "error"}

    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT mark_automation_suggestion_refresh($1::jsonb)",
                json.dumps(result, default=str),
            )
    except Exception:
        logger.warning("Could not record automation suggestion refresh outcome", exc_info=True)
    return result


async def resolve_automation_suggestion_from_inbound(
    pool: "asyncpg.Pool",
    *,
    channel: str,
    actor: str,
    text: str,
) -> dict[str, Any]:
    """Resolve a private-channel numbered response without treating it as chat."""
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT try_resolve_automation_suggestion_from_inbound($1, $2, $3)",
                channel,
                actor,
                text,
            )
        return _coerce_object(raw)
    except Exception:
        logger.warning("Inbound automation suggestion resolution failed", exc_info=True)
        return {"recognized": False, "matched": False, "reason": "resolution_error"}
