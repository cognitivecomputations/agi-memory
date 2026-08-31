"""Opt-in background review that creates durable skill proposals.

This service never writes skill files. The approved proposal tool owns that
separate transition so background work cannot silently change future behavior.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from core.llm_config import load_llm_config
from core.llm_json import chat_json, extract_json_object
from services.prompt_resources import load_skill_improvement_prompt
from services.skill_runtime import load_available_skills
from skills.base import SkillCategory, SkillContext
from skills.loader import discover_skill_dirs, load_skills_from_dir

logger = logging.getLogger("skill_improvement")

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:api[_ -]?key|password|secret|access[_ -]?token|refresh[_ -]?token)"
    r"\s*[:=]\s*[`'\"]?[A-Za-z0-9_./+\-=]{8,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{16,}"
)


def _coerce_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _catalog(registry: Any | None) -> tuple[dict[str, dict[str, Any]], set[str]]:
    if registry is not None:
        tools = set(registry.list_names())
        skills = load_available_skills(registry, registry_context())
    else:
        tools = set()
        skills = []
        seen: set[str] = set()
        for directory in discover_skill_dirs():
            for skill in load_skills_from_dir(directory):
                if skill.name not in seen:
                    skills.append(skill)
                    seen.add(skill.name)
    return {
        skill.name: {
            "name": skill.name,
            "description": skill.description,
            "managed_by": skill.provenance.get("managed_by"),
            "authored_by": skill.provenance.get("authored_by"),
        }
        for skill in skills
    }, tools


def registry_context():
    # Local import avoids making the service layer initialize the tool package.
    from core.tools.base import ToolContext

    return ToolContext.CHAT


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"proposal {field} must be an array of strings")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


def _normalize_proposal(
    doc: dict[str, Any],
    *,
    existing_skills: dict[str, dict[str, Any]],
    available_tools: set[str],
    min_confidence: float,
) -> dict[str, Any] | None:
    if "proposal" not in doc:
        raise ValueError("review response is missing the proposal field")
    raw = doc["proposal"]
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("review proposal must be an object or null")

    name = str(raw.get("name") or "").strip()
    description = str(raw.get("description") or "").strip()
    content = str(raw.get("content") or "").strip()
    rationale = str(raw.get("rationale") or "").strip()
    mode = str(raw.get("mode") or "create").strip()
    category = str(raw.get("category") or SkillCategory.OTHER.value).strip()
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal confidence must be numeric") from exc

    if not _NAME_RE.fullmatch(name):
        raise ValueError("proposal name must be lowercase kebab-case/underscore, 2-64 chars")
    if not description:
        raise ValueError("proposal description is required")
    if len(content) < 120:
        raise ValueError("proposal content must be substantive (at least 120 characters)")
    if not rationale:
        raise ValueError("proposal rationale is required")
    if category not in {item.value for item in SkillCategory}:
        raise ValueError(f"unknown proposal category: {category}")
    if mode not in {"create", "update"}:
        raise ValueError("proposal mode must be create or update")
    if not 0 <= confidence <= 1:
        raise ValueError("proposal confidence must be between 0 and 1")
    if confidence < min_confidence:
        return None

    contexts = _string_list(raw.get("contexts") or ["chat", "heartbeat"], "contexts")
    unknown_contexts = sorted(set(contexts) - {item.value for item in SkillContext})
    if unknown_contexts:
        raise ValueError("unknown proposal context(s): " + ", ".join(unknown_contexts))
    bound_tools = _string_list(raw.get("bound_tools"), "bound_tools")
    requires_tools = _string_list(raw.get("requires_tools"), "requires_tools") or bound_tools[:]
    unknown_tools = sorted((set(bound_tools) | set(requires_tools)) - available_tools)
    if unknown_tools:
        raise ValueError("unknown proposal tool(s): " + ", ".join(unknown_tools))

    existing = existing_skills.get(name)
    if mode == "create" and existing is not None:
        raise ValueError(f"proposal cannot create existing skill: {name}")
    if mode == "update" and (
        existing is None
        or existing.get("authored_by") != "hexis"
        or existing.get("managed_by") != "author_skill"
    ):
        raise ValueError(f"proposal cannot update skill without Hexis ownership: {name}")
    if _SECRET_VALUE_RE.search("\n".join((description, content, rationale))):
        raise ValueError("proposal appears to contain credential or secret material")

    return {
        "name": name,
        "description": description,
        "content": content,
        "category": category,
        "contexts": contexts,
        "bound_tools": bound_tools,
        "requires_tools": requires_tools,
        "mode": mode,
        "rationale": rationale,
        "confidence": confidence,
    }


def _normalize_automation_suggestion(
    doc: dict[str, Any],
    *,
    evidence: dict[str, Any],
    min_confidence: float,
) -> dict[str, Any] | None:
    raw = doc.get("automation_suggestion")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("automation_suggestion must be an object or null")

    title = str(raw.get("title") or "").strip()
    rationale = str(raw.get("rationale") or "").strip()
    pattern = " ".join(str(raw.get("pattern") or "").strip().lower().split())
    task_spec = raw.get("task_spec")
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("automation suggestion confidence must be numeric") from exc
    if confidence < min_confidence:
        return None
    if not 0 <= confidence <= 1:
        raise ValueError("automation suggestion confidence must be between 0 and 1")
    if not title or not rationale or not pattern:
        raise ValueError("automation suggestion title, rationale, and pattern are required")
    if not isinstance(task_spec, dict):
        raise ValueError("automation suggestion task_spec must be an object")

    source_ids = _string_list(raw.get("evidence_unit_ids"), "evidence_unit_ids")
    source_ids = list(dict.fromkeys(source_ids))
    available_ids = {str(value) for value in evidence.get("source_unit_ids") or []}
    if len(source_ids) < 3:
        raise ValueError("automation suggestion requires at least three matching asks")
    if not set(source_ids).issubset(available_ids):
        raise ValueError("automation suggestion cites evidence outside the supplied review window")

    normalized_spec = dict(task_spec)
    normalized_spec.setdefault("action", "create")
    normalized_spec.setdefault("action_kind", "queue_user_message")
    normalized_spec.setdefault("delivery_mode", "outbox")
    if normalized_spec.get("action") != "create":
        raise ValueError("automation suggestion task_spec.action must be create")
    if normalized_spec.get("action_kind") not in {"queue_user_message", "create_goal"}:
        raise ValueError("automation suggestion task_spec has an unsupported action_kind")
    if not str(normalized_spec.get("name") or "").strip():
        raise ValueError("automation suggestion task_spec.name is required")
    if not str(
        normalized_spec.get("schedule") or normalized_spec.get("schedule_kind") or ""
    ).strip():
        raise ValueError("automation suggestion task_spec.schedule is required")
    if normalized_spec.get("action_kind") == "queue_user_message" and not str(
        normalized_spec.get("message") or ""
    ).strip():
        raise ValueError("automation suggestion task_spec.message is required")

    secret_surface = json.dumps(
        {
            "title": title,
            "rationale": rationale,
            "pattern": pattern,
            "task_spec": normalized_spec,
        },
        sort_keys=True,
        default=str,
    )
    if _SECRET_VALUE_RE.search(secret_surface):
        raise ValueError("automation suggestion appears to contain credential or secret material")
    dedup_key = "usage:" + hashlib.sha256(pattern.encode("utf-8")).hexdigest()
    return {
        "dedup_key": dedup_key,
        "title": title,
        "rationale": rationale,
        "task_spec": normalized_spec,
        "metadata": {
            "pattern": pattern,
            "confidence": confidence,
            "evidence_unit_ids": source_ids,
            "evidence_count": len(source_ids),
        },
    }


def _normalize_learning_review(
    doc: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    max_items: int,
) -> dict[str, Any] | None:
    """Validate the model's choice while deriving every item from DB truth."""
    raw = doc.get("learning_review")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("learning_review must be an object or null")
    if raw.get("should_review") is not True:
        return None
    summary = str(raw.get("summary") or "").strip()
    if not summary:
        raise ValueError("learning_review summary is required when should_review is true")
    selected = _string_list(raw.get("memory_ids"), "learning_review.memory_ids")
    available = {str(item["id"]) for item in candidates}
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(
            "learning_review cites memories outside the supplied change window: "
            + ", ".join(unknown)
        )
    return {
        "summary": summary,
        "memory_ids": list(dict.fromkeys(selected))[:max_items],
    }


async def _learning_review_context(
    conn: Any,
    *,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    state = _coerce_json(await conn.fetchval("SELECT get_state('skill_improvement_state')"))
    lookback_days = int(
        await conn.fetchval(
            "SELECT COALESCE(get_config_int('skills.self_improvement.lookback_days'), 30)"
        )
        or 30
    )
    now = datetime.now(timezone.utc)
    period_start: datetime
    raw_start = state.get("last_completed_at")
    try:
        period_start = datetime.fromisoformat(str(raw_start).replace("Z", "+00:00"))
        if period_start.tzinfo is None:
            period_start = period_start.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        period_start = now - timedelta(days=max(1, min(lookback_days, 365)))
    period_start = max(period_start.astimezone(timezone.utc), now - timedelta(days=365))

    evidence_memory_ids = [str(value) for value in evidence.get("source_memory_ids") or []]
    rows = await conn.fetch(
        """
        SELECT m.id, m.type::text AS type, m.content, m.importance,
               m.trust_level, m.source_attribution, m.created_at, m.updated_at,
               CASE
                   WHEN m.created_at >= $1::timestamptz THEN 'new'
                   ELSE 'revised'
               END AS change_kind
        FROM memories m
        WHERE m.status = 'active'
          AND m.type IN ('semantic', 'worldview', 'procedural', 'strategic')
          AND (
              m.created_at >= $1::timestamptz
              OR EXISTS (
                  SELECT 1 FROM belief_revision_audit b
                  WHERE b.memory_id = m.id AND b.created_at >= $1::timestamptz
              )
          )
          AND (
              cardinality($2::uuid[]) = 0
              OR m.id = ANY($2::uuid[])
              OR m.type IN ('procedural', 'strategic')
          )
        ORDER BY m.updated_at DESC, m.created_at DESC, m.id
        LIMIT 60
        """,
        period_start,
        evidence_memory_ids,
    )
    candidates = []
    for row in rows:
        item = dict(row)
        for key in ("id", "created_at", "updated_at"):
            item[key] = str(item[key])
        candidates.append(item)
    return {
        "period_start": period_start.isoformat(),
        "period_end": now.isoformat(),
        "candidates": candidates,
    }


async def run_skill_improvement_review_step(conn, *, registry: Any | None = None) -> dict[str, Any]:
    """Run one due review and persist at most one proposal."""
    claimed = bool(await conn.fetchval("SELECT claim_skill_improvement_review()"))
    if not claimed:
        return {"skipped": True, "reason": "disabled_not_due_or_claimed"}

    result: dict[str, Any]
    try:
        evidence = _coerce_json(await conn.fetchval("SELECT load_skill_improvement_evidence()"))
        if not evidence.get("eligible"):
            result = {
                "status": "no_evidence",
                "reason": evidence.get("reason") or "insufficient_evidence",
                "unit_count": int(evidence.get("unit_count") or 0),
                "session_count": int(evidence.get("session_count") or 0),
            }
        else:
            existing, available_tools = _catalog(registry)
            if not available_tools:
                rows = await conn.fetch("SELECT name FROM tool_definitions ORDER BY name")
                available_tools = {str(row["name"]) for row in rows}
            min_confidence = float(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_float('skills.self_improvement.min_confidence'), 0.8)"
                )
                or 0.8
            )
            llm_config = await load_llm_config(
                conn, "llm.skill_improvement", fallback_key="llm.subconscious"
            )
            learning_context = await _learning_review_context(conn, evidence=evidence)
            learning_max_items = int(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_int('learning.review.max_items'), 20)"
                )
                or 20
            )
            review_context = {
                "constraints": {
                    "minimum_confidence": min_confidence,
                    "available_categories": [item.value for item in SkillCategory],
                    "available_contexts": [item.value for item in SkillContext],
                    "available_tools": sorted(available_tools),
                },
                "existing_skills": sorted(existing.values(), key=lambda item: item["name"]),
                "evidence": evidence,
                "learning_changes": learning_context,
            }
            doc, raw = await chat_json(
                llm_config=llm_config,
                messages=[
                    {"role": "system", "content": load_skill_improvement_prompt().strip()},
                    {"role": "user", "content": json.dumps(review_context, default=str)[:30000]},
                ],
                max_tokens=2400,
                temperature=0.1,
                response_format={"type": "json_object"},
                fallback={},
            )
            parsed_raw = extract_json_object(raw)
            if not parsed_raw or (
                "proposal" not in parsed_raw
                and "automation_suggestion" not in parsed_raw
                and "learning_review" not in parsed_raw
            ):
                raise ValueError("skill-improvement model returned invalid JSON")
            proposal = _normalize_proposal(
                doc,
                existing_skills=existing,
                available_tools=available_tools,
                min_confidence=min_confidence,
            )
            usage_enabled = bool(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('automation.suggestions.usage_enabled'), TRUE)"
                )
            )
            usage_min_confidence = float(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_float('automation.suggestions.usage_min_confidence'), 0.85)"
                )
                or 0.85
            )
            automation = (
                _normalize_automation_suggestion(
                    doc,
                    evidence=evidence,
                    min_confidence=usage_min_confidence,
                )
                if usage_enabled
                else None
            )
            learning_review = _normalize_learning_review(
                doc,
                candidates=learning_context["candidates"],
                max_items=max(1, min(learning_max_items, 50)),
            )

            created_skill: dict[str, Any] | None = None
            created_automation: dict[str, Any] | None = None
            created_learning_review: dict[str, Any] | None = None
            if proposal is not None:
                source_ids = sorted(str(value) for value in evidence.get("source_unit_ids") or [])
                digest = hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest()
                created_skill = _coerce_json(
                    await conn.fetchval(
                        "SELECT create_skill_improvement_proposal($1::jsonb, $2::jsonb, $3::text)",
                        json.dumps(proposal),
                        json.dumps(evidence),
                        digest,
                    )
                )
            if automation is not None:
                created_automation = _coerce_json(
                    await conn.fetchval(
                        """
                        SELECT propose_automation(
                            'usage', $1, $2, $3, $4::jsonb, $5::jsonb
                        )
                        """,
                        automation["dedup_key"],
                        automation["title"],
                        automation["rationale"],
                        json.dumps(automation["task_spec"]),
                        json.dumps(automation["metadata"]),
                    )
                )

            skill_proposal_ids = []
            if created_skill and created_skill.get("proposal_id"):
                skill_proposal_ids.append(str(created_skill["proposal_id"]))
            if learning_review is not None or skill_proposal_ids:
                summary = (
                    learning_review["summary"]
                    if learning_review is not None
                    else "I found a repeated workflow worth proposing as a reusable skill."
                )
                created_learning_review = _coerce_json(
                    await conn.fetchval(
                        """
                        SELECT create_learning_review(
                            $1::timestamptz, $2::timestamptz, $3,
                            $4::uuid[], $5::uuid[], $6::jsonb
                        )
                        """,
                        datetime.fromisoformat(learning_context["period_start"]),
                        datetime.fromisoformat(learning_context["period_end"]),
                        summary,
                        learning_review["memory_ids"] if learning_review else [],
                        skill_proposal_ids,
                        json.dumps(
                            {
                                "source": "skill_improvement_review",
                                "evidence_unit_count": evidence.get("unit_count", 0),
                                "evidence_session_count": evidence.get("session_count", 0),
                            }
                        ),
                    )
                )

            if created_skill is not None:
                result = {
                    **created_skill,
                    "status": "proposed",
                    "skill": proposal["name"] if proposal else None,
                }
                if created_automation is not None:
                    result["automation_suggestion"] = created_automation
                if created_learning_review is not None:
                    result["learning_review"] = created_learning_review
            elif created_automation is not None:
                result = {
                    **created_automation,
                    "status": "automation_proposed",
                    "automation": automation["title"] if automation else None,
                }
                if created_learning_review is not None:
                    result["learning_review"] = created_learning_review
            elif created_learning_review and created_learning_review.get("created"):
                result = {
                    "status": "learning_review_proposed",
                    "learning_review": created_learning_review,
                }
            else:
                result = {"status": "no_proposal", "reason": "insufficient_recurrence_or_confidence"}
    except Exception as exc:
        logger.error("skill-improvement review failed: %s", exc)
        result = {"status": "error", "error": str(exc)}

    await conn.fetchval("SELECT mark_skill_improvement_review($1::jsonb)", json.dumps(result))
    return result


async def apply_approved_learning_skill_step(
    pool: Any,
    *,
    registry: Any,
) -> dict[str, Any]:
    """Apply one explicitly approved learning-review skill proposal."""
    async with pool.acquire() as conn:
        claim = _coerce_json(
            await conn.fetchval("SELECT claim_approved_learning_skill_application()")
        )
    if not claim.get("claimed"):
        return {"skipped": True, "reason": claim.get("reason") or "none_pending"}

    from core.tools.base import ToolContext, ToolExecutionContext
    from core.tools.skills import ReviewSkillProposalHandler

    item_id = str(claim["item_id"])
    proposal_id = str(claim["proposal_id"])
    context = ToolExecutionContext(
        tool_context=ToolContext.HEARTBEAT,
        call_id=f"learning-review-skill-{item_id}",
        registry=registry,
    )
    result = await ReviewSkillProposalHandler().execute(
        {"proposal_id": proposal_id, "action": "apply"},
        context,
    )
    async with pool.acquire() as conn:
        await conn.fetchval(
            "SELECT finish_learning_skill_application($1::uuid, $2, $3)",
            item_id,
            "applied" if result.success else "failed",
            result.error,
        )
    if result.success:
        return {"applied": True, "item_id": item_id, "proposal_id": proposal_id}
    return {
        "status": "error",
        "item_id": item_id,
        "proposal_id": proposal_id,
        "error": result.error or "skill application failed",
        "next_step": (
            "Inspect the proposal's last error; the worker will retry without "
            "losing the explicit approval."
        ),
    }


async def resolve_learning_review_from_inbound(
    pool: Any,
    *,
    channel: str,
    actor: str,
    text: str,
) -> dict[str, Any]:
    """Consume an exact verified-operator reply to a weekly review."""
    try:
        async with pool.acquire() as conn:
            return _coerce_json(
                await conn.fetchval(
                    "SELECT try_resolve_learning_review_from_inbound($1, $2, $3)",
                    channel,
                    actor,
                    text,
                )
            )
    except Exception:
        logger.warning("Inbound learning-review resolution failed", exc_info=True)
        return {"recognized": False, "matched": False, "reason": "resolution_error"}
