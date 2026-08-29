"""Durable, advisory adversarial deliberation.

The council remains a thinking aid: it records perspectives, challenges, dissent,
and review conditions, but it never authorizes, blocks, or executes an action.
PostgreSQL owns the durable lifecycle; this module only orchestrates bounded model
calls and writes their inspectable artifacts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from core.llm_json import parse_json_response

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).resolve().parent / "prompts"
_PERSPECTIVE_PROMPT = _PROMPTS / "deliberation_perspective.md"
_CHALLENGE_PROMPT = _PROMPTS / "deliberation_challenge.md"
_SYNTHESIS_PROMPT = _PROMPTS / "deliberation_synthesis.md"


class DeliberationUnavailable(RuntimeError):
    """The durable deliberation substrate is missing or unusable."""


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def _prompt(path: Path, fallback: str) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return text or fallback


def _safe_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return (text or type(exc).__name__)[:500]


def _string_list(value: Any, *, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = " ".join(str(item).split())
        if text:
            result.append(text[:1000])
        if len(result) >= limit:
            break
    return result


async def load_deliberation_config(pool: Any) -> dict[str, Any]:
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval("SELECT get_deliberation_config()")
    except Exception as exc:
        raise DeliberationUnavailable(
            "Deliberation storage is unavailable. Run `hexis migrate` and retry."
        ) from exc
    config = _json(raw)
    if not config:
        raise DeliberationUnavailable(
            "Deliberation configuration is unavailable. Run `hexis migrate` and retry."
        )
    return config


async def begin_deliberation(
    pool: Any,
    *,
    topic: str,
    stakes: str,
    source_context: str,
    source_session_id: str | None,
    heartbeat_id: str | None,
    call_id: str | None,
    persona_keys: list[str],
    signals: list[str],
    extra_context: str,
    model_identity: dict[str, str] | None,
    collection_warnings: list[str] | None = None,
) -> dict[str, Any]:
    input_context = {
        "additional_context": extra_context,
        "signals": signals,
        "collection_warnings": collection_warnings or [],
        "model": model_identity or {},
    }
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT begin_deliberation(
                $1, $2, $3, $4, $5::uuid, $6, $7::jsonb, $8, $9::jsonb
            )
            """,
            topic,
            stakes,
            source_context,
            source_session_id,
            heartbeat_id,
            call_id,
            json.dumps(persona_keys),
            len(signals),
            json.dumps(input_context),
        )
    result = _json(raw)
    if not result.get("id"):
        raise DeliberationUnavailable(
            "The database did not create a deliberation session."
        )
    return result


async def record_deliberation_move(
    pool: Any,
    *,
    deliberation_id: str,
    move_key: str,
    role: str,
    content: str,
    round_number: int,
    ordinal: int,
    persona_key: str | None = None,
    target_move_id: str | None = None,
    evidence_memory_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    async with pool.acquire() as conn:
        move_id = await conn.fetchval(
            """
            SELECT record_deliberation_move(
                $1::uuid, $2, $3, $4, $5, $6, $7, $8::uuid,
                $9::uuid[], $10::jsonb
            )
            """,
            deliberation_id,
            move_key,
            role,
            content,
            round_number,
            ordinal,
            persona_key,
            target_move_id,
            evidence_memory_ids or [],
            json.dumps(metadata or {}),
        )
    return str(move_id)


async def complete_deliberation(
    pool: Any,
    *,
    deliberation_id: str,
    recommendation: str,
    report: str,
    agreements: list[str],
    disagreements: list[str],
    risks: list[str],
    missing_evidence: list[str],
    dissent: list[str],
    invalidation_conditions: list[str],
    evidence_memory_ids: list[str],
    create_summary_memory: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT complete_deliberation(
                $1::uuid, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb,
                $7::jsonb, $8::jsonb, $9::jsonb, $10::uuid[], $11,
                $12::jsonb
            )
            """,
            deliberation_id,
            recommendation,
            report,
            json.dumps(agreements),
            json.dumps(disagreements),
            json.dumps(risks),
            json.dumps(missing_evidence),
            json.dumps(dissent),
            json.dumps(invalidation_conditions),
            evidence_memory_ids,
            create_summary_memory,
            json.dumps(metadata),
        )
    return _json(raw)


async def fail_deliberation(
    pool: Any,
    deliberation_id: str,
    error: BaseException | str,
) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT fail_deliberation($1::uuid, $2)",
                deliberation_id,
                _safe_error(
                    error if isinstance(error, BaseException) else RuntimeError(error)
                ),
            )
    except Exception:
        logger.warning(
            "Could not mark deliberation %s failed", deliberation_id, exc_info=True
        )


async def list_deliberations(
    pool: Any,
    *,
    limit: int = 20,
    status: str | None = None,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        raw = await conn.fetchval("SELECT list_deliberations($1, $2)", limit, status)
    return _json(raw)


async def inspect_deliberation(pool: Any, deliberation_id: str) -> dict[str, Any]:
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT inspect_deliberation($1::uuid)", deliberation_id
        )
    return _json(raw)


async def _resolve_llm(pool: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from core.llm_config import resolve_llm_config

        config = await resolve_llm_config(pool, "llm.chat", fallback_key="llm")
        return config, None
    except Exception as exc:
        return None, _safe_error(exc)


async def _chat(
    config: dict[str, Any],
    *,
    system: str,
    user: str,
    max_tokens: int,
    json_result: bool = False,
) -> tuple[str, dict[str, Any]]:
    from core.llm import chat_completion

    response = await chat_completion(
        provider=config["provider"],
        model=config["model"],
        endpoint=config.get("endpoint"),
        api_key=config.get("api_key"),
        auth_mode=config.get("auth_mode"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        response_format={"type": "json_object"} if json_result else None,
        tools=None,
    )
    raw = str((response or {}).get("content") or "").strip()
    return raw, parse_json_response(raw, {}) if json_result else {}


async def run_adversarial_deliberation(
    pool: Any,
    *,
    topic: str,
    personas: dict[str, dict[str, str]],
    selected_keys: list[str],
    extra_context: str,
    signals: list[str],
    stakes: str,
    source_context: str,
    source_session_id: str | None,
    heartbeat_id: str | None,
    call_id: str | None,
    evidence_memory_ids: list[str] | None = None,
    collection_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Run one bounded perspective → challenge → synthesis cycle."""

    policy = await load_deliberation_config(pool)
    max_personas = int(policy["max_personas"])
    topic = topic.strip()
    extra_context = extra_context.strip()
    if not selected_keys:
        raise ValueError("Deliberation requires at least one council persona.")
    if len(selected_keys) > max_personas:
        raise ValueError(
            f"Deliberation exceeds the live maximum of {max_personas} personas."
        )
    if len(set(selected_keys)) != len(selected_keys):
        raise ValueError("Deliberation persona keys must be unique.")
    unknown_keys = [key for key in selected_keys if key not in personas]
    if unknown_keys:
        raise ValueError(
            "Unknown deliberation persona(s): " + ", ".join(sorted(unknown_keys))
        )
    signal_limit = int(policy["signal_limit"])
    if len(signals) > signal_limit:
        raise ValueError(
            f"Deliberation exceeds the live maximum of {signal_limit} evidence signals."
        )

    max_topic_chars = int(policy["max_topic_chars"])
    max_context_chars = int(policy["max_context_chars"])
    if len(topic) > max_topic_chars:
        raise ValueError(
            f"Deliberation topic exceeds the live limit of {max_topic_chars} characters."
        )
    if len(extra_context) > max_context_chars:
        raise ValueError(
            "Deliberation context exceeds the live limit of "
            f"{max_context_chars} characters."
        )

    config, config_error = await _resolve_llm(pool)
    model_identity = (
        {
            "provider": str(config.get("provider") or ""),
            "model": str(config.get("model") or ""),
        }
        if config
        else None
    )
    session = await begin_deliberation(
        pool,
        topic=topic,
        stakes=stakes,
        source_context=source_context,
        source_session_id=source_session_id,
        heartbeat_id=heartbeat_id,
        call_id=call_id,
        persona_keys=selected_keys,
        signals=signals,
        extra_context=extra_context,
        model_identity=model_identity,
        collection_warnings=collection_warnings,
    )
    deliberation_id = str(session["id"])

    perspective_instruction = _prompt(
        _PERSPECTIVE_PROMPT,
        "Give a concise advisory position, its evidence, its strongest challenge, "
        "and what would change your view.",
    )
    challenge_instruction = _prompt(
        _CHALLENGE_PROMPT,
        "Return JSON challenges, unresolved_disagreements, and missing_evidence.",
    )
    synthesis_instruction = _prompt(
        _SYNTHESIS_PROMPT,
        "Return JSON recommendation, report, agreements, disagreements, risks, "
        "dissent, and invalidation_conditions.",
    )
    signal_block = "\n".join(f"- {item}" for item in signals) or "- none supplied"
    context_block = extra_context or "No additional context was supplied."

    async def one_perspective(key: str) -> dict[str, Any]:
        persona = personas[key]
        if config is None:
            return {
                "persona_key": key,
                "persona_name": persona["name"],
                "analysis": (
                    "Analysis unavailable because the configured chat model could not "
                    f"be loaded: {config_error or 'unknown configuration error'}."
                ),
                "available": False,
            }
        user = (
            f"Topic: {topic}\nStakes: {stakes}\n\n"
            f"Compacted evidence signals:\n{signal_block}\n\n"
            f"Additional context:\n{context_block}"
        )
        try:
            raw, _ = await _chat(
                config,
                system=persona["system_prompt"] + "\n\n" + perspective_instruction,
                user=user,
                max_tokens=int(policy["perspective_max_tokens"]),
            )
            if not raw:
                raise RuntimeError("the model returned no analysis")
            return {
                "persona_key": key,
                "persona_name": persona["name"],
                "analysis": raw,
                "available": True,
            }
        except Exception as exc:
            return {
                "persona_key": key,
                "persona_name": persona["name"],
                "analysis": f"Analysis unavailable: {_safe_error(exc)}.",
                "available": False,
            }

    try:
        perspectives = await asyncio.gather(
            *(one_perspective(key) for key in selected_keys)
        )
        move_ids: dict[str, str] = {}
        for ordinal, item in enumerate(perspectives):
            move_ids[item["persona_key"]] = await record_deliberation_move(
                pool,
                deliberation_id=deliberation_id,
                move_key=f"perspective:{item['persona_key']}",
                role="perspective",
                content=item["analysis"],
                round_number=1,
                ordinal=ordinal,
                persona_key=item["persona_key"],
                evidence_memory_ids=evidence_memory_ids,
                metadata={"available": item["available"]},
            )

        available = [item for item in perspectives if item["available"]]
        challenge_doc: dict[str, Any] = {
            "challenges": [],
            "unresolved_disagreements": [],
            "missing_evidence": [],
        }
        challenge_error: str | None = None
        if config is not None and available:
            try:
                _, parsed = await _chat(
                    config,
                    system=challenge_instruction,
                    user=json.dumps(
                        {
                            "topic": topic,
                            "stakes": stakes,
                            "evidence_signals": signals,
                            "additional_context": extra_context,
                            "perspectives": available,
                        },
                        ensure_ascii=True,
                    ),
                    max_tokens=int(policy["challenge_max_tokens"]),
                    json_result=True,
                )
                required_challenge_fields = (
                    "challenges",
                    "unresolved_disagreements",
                    "missing_evidence",
                )
                if not all(
                    isinstance(parsed.get(field), list)
                    for field in required_challenge_fields
                ):
                    raise RuntimeError(
                        "the adversarial reviewer returned incomplete structured output"
                    )
                challenge_doc.update(parsed)
            except Exception as exc:
                challenge_error = _safe_error(exc)
        elif config_error:
            challenge_error = config_error

        normalized_challenges: list[dict[str, str]] = []
        raw_challenges = challenge_doc.get("challenges")
        if isinstance(raw_challenges, list):
            for ordinal, item in enumerate(raw_challenges[: max_personas * 2]):
                if not isinstance(item, dict):
                    continue
                target = str(item.get("target_persona") or "").strip()
                challenge = " ".join(str(item.get("challenge") or "").split())
                severity = str(item.get("severity") or "serious").strip().lower()
                if target not in selected_keys or not challenge:
                    continue
                if severity not in {"fatal", "serious", "minor"}:
                    severity = "serious"
                normalized = {
                    "target_persona": target,
                    "challenge": challenge[:2000],
                    "severity": severity,
                }
                normalized_challenges.append(normalized)
                await record_deliberation_move(
                    pool,
                    deliberation_id=deliberation_id,
                    move_key=f"challenge:{ordinal}:{target}",
                    role="challenge",
                    content=normalized["challenge"],
                    round_number=2,
                    ordinal=ordinal,
                    persona_key="adversarial_reviewer",
                    target_move_id=move_ids.get(target),
                    evidence_memory_ids=evidence_memory_ids,
                    metadata={"severity": severity, "target_persona": target},
                )

        unresolved = _string_list(challenge_doc.get("unresolved_disagreements"))
        missing_evidence = _string_list(challenge_doc.get("missing_evidence"))
        synthesis_doc: dict[str, Any] = {}
        synthesis_error: str | None = None
        if config is not None and available:
            try:
                raw_synthesis, synthesis_doc = await _chat(
                    config,
                    system=synthesis_instruction,
                    user=json.dumps(
                        {
                            "topic": topic,
                            "stakes": stakes,
                            "evidence_signals": signals,
                            "additional_context": extra_context,
                            "perspectives": available,
                            "challenges": normalized_challenges,
                            "unresolved_disagreements": unresolved,
                            "missing_evidence": missing_evidence,
                        },
                        ensure_ascii=True,
                    ),
                    max_tokens=int(policy["synthesis_max_tokens"]),
                    json_result=True,
                )
                if not synthesis_doc:
                    synthesis_error = "the moderator did not return valid JSON"
                    synthesis_doc = {"report": raw_synthesis}
            except Exception as exc:
                synthesis_error = _safe_error(exc)
        else:
            synthesis_error = config_error or "no council perspective completed"

        required_synthesis_lists = (
            "agreements",
            "disagreements",
            "risks",
            "dissent",
            "invalidation_conditions",
        )
        synthesis_ok = bool(
            str(synthesis_doc.get("recommendation") or "").strip()
            and str(synthesis_doc.get("report") or "").strip()
            and all(
                isinstance(synthesis_doc.get(field), list)
                for field in required_synthesis_lists
            )
        )
        if synthesis_doc and not synthesis_ok and synthesis_error is None:
            synthesis_error = "the moderator returned incomplete structured output"
        recommendation = str(
            synthesis_doc.get("recommendation")
            or "No grounded recommendation was produced; review the recorded perspectives and retry."
        ).strip()
        report = str(
            synthesis_doc.get("report")
            or (
                "Deliberation could not complete its synthesis. "
                f"Cause: {synthesis_error or 'unknown synthesis failure'}. "
                "The available perspective records remain inspectable."
            )
        ).strip()
        agreements = _string_list(synthesis_doc.get("agreements"))
        disagreements = _string_list(synthesis_doc.get("disagreements"))
        for item in unresolved:
            if item not in disagreements:
                disagreements.append(item)
        risks = _string_list(synthesis_doc.get("risks"))
        dissent = _string_list(synthesis_doc.get("dissent"))
        invalidation_conditions = _string_list(
            synthesis_doc.get("invalidation_conditions")
        )
        degraded_reasons = [
            item
            for item in (
                config_error,
                challenge_error,
                synthesis_error,
                (
                    f"{len(perspectives) - len(available)} perspective(s) unavailable"
                    if len(available) != len(perspectives)
                    else None
                ),
                *(collection_warnings or []),
            )
            if item
        ]
        completion = await complete_deliberation(
            pool,
            deliberation_id=deliberation_id,
            recommendation=recommendation,
            report=report,
            agreements=agreements,
            disagreements=disagreements,
            risks=risks,
            missing_evidence=missing_evidence,
            dissent=dissent,
            invalidation_conditions=invalidation_conditions,
            evidence_memory_ids=evidence_memory_ids or [],
            create_summary_memory=(
                bool(policy.get("create_summary_memory"))
                and synthesis_ok
                and not degraded_reasons
            ),
            metadata={
                "degraded": bool(degraded_reasons),
                "degraded_reasons": degraded_reasons,
                "model": model_identity or {},
            },
        )

        return {
            "deliberation_id": deliberation_id,
            "topic": topic,
            "stakes": stakes,
            "persona_count": len(perspectives),
            "personas_included": selected_keys,
            "signals": signals,
            "council": perspectives,
            "challenges": normalized_challenges,
            "missing_evidence": missing_evidence,
            "collection_warnings": collection_warnings or [],
            "recommendation": recommendation,
            "moderator_report": report,
            "agreements": agreements,
            "disagreements": disagreements,
            "risks": risks,
            "dissent": dissent,
            "invalidation_conditions": invalidation_conditions,
            "degraded": bool(degraded_reasons),
            "degraded_reasons": degraded_reasons,
            "memory_id": completion.get("memory_id"),
            "instructions": (
                "This deliberation is advisory. Preserve material dissent and review "
                "the invalidation conditions before acting."
            ),
        }
    except Exception as exc:
        await fail_deliberation(pool, deliberation_id, exc)
        raise


__all__ = [
    "DeliberationUnavailable",
    "begin_deliberation",
    "complete_deliberation",
    "fail_deliberation",
    "inspect_deliberation",
    "list_deliberations",
    "load_deliberation_config",
    "record_deliberation_move",
    "run_adversarial_deliberation",
]
