"""Connector cognition workers.

Connector source items preserve exact history. This module performs the
stateless pass that turns those source items into DB-owned user-model claims
and importance records.
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
from typing import Any

from core.llm_config import load_llm_config
from core.llm_batch import batch_classify

logger = logging.getLogger(__name__)

LLM_DETECTOR_VERSION = "connector_cognition_llm_v3"
CACHE_DETECTOR_VERSION = "connector_cognition_llm_cache_v3"
FALLBACK_DETECTOR_VERSION = "connector_cognition_rules_fallback_v3"
CLAIMS_CACHE_VERSION = "user_model_claims_v3"
IMPORTANCE_CACHE_VERSION = "item_importance_v3"

_PREFERENCE_PATTERNS = (
    re.compile(
        r"\bI\s+(?:really\s+)?(?:prefer|like|love|enjoy)\s+([^.!?\n]{2,120})", re.I
    ),
    re.compile(
        r"\bI\s+(?:really\s+)?(?:hate|dislike|can't stand|cannot stand)\s+([^.!?\n]{2,120})",
        re.I,
    ),
)
_ROUTINE_PATTERNS = (
    re.compile(r"\bI\s+(?:usually|always|often|normally)\s+([^.!?\n]{2,120})", re.I),
    re.compile(r"\bmy\s+(?:routine|schedule)\s+is\s+([^.!?\n]{2,120})", re.I),
)
_IDENTITY_PATTERNS = (
    re.compile(
        r"\bI\s+(?:am|work as|work at|live in|live near)\s+([^.!?\n]{2,120})", re.I
    ),
    re.compile(r"\bmy\s+(?:name|job|role|company|city)\s+is\s+([^.!?\n]{2,120})", re.I),
)
_RELATIONSHIP_PATTERNS = (
    re.compile(
        r"\b(?:my|our)\s+(?:friend|partner|spouse|wife|husband|boss|manager|coworker|colleague|client)\s+([^.!?\n]{2,120})",
        re.I,
    ),
    re.compile(
        r"\b([A-Z][a-z][^.!?\n]{0,60})\s+is\s+(?:my|our)\s+(friend|partner|spouse|wife|husband|boss|manager|coworker|colleague|client)\b",
        re.I,
    ),
)
_COMMITMENT_PATTERNS = (
    re.compile(
        r"\bI\s+(?:promised|committed|agreed|need|have)\s+to\s+([^.!?\n]{2,140})", re.I
    ),
    re.compile(
        r"\b(?:please remind me|remind me|I should)\s+to\s+([^.!?\n]{2,140})", re.I
    ),
)
_JUDGMENT_PATTERNS = (
    re.compile(r"\bI\s+(?:decide|judge|evaluate|prioritize)\s+([^.!?\n]{2,140})", re.I),
    re.compile(
        r"\b(?:what matters to me is|the important thing is)\s+([^.!?\n]{2,140})", re.I
    ),
)
_EPHEMERAL_USER_MODEL_PATTERNS = (
    re.compile(r"\bthis is (?:just )?(?:a )?test\b", re.I),
    re.compile(r"\bjust testing\b", re.I),
    re.compile(r"\bignore this\b", re.I),
    re.compile(r"\bpretend (?:that )?I\b", re.I),
    re.compile(r"\bsample (?:message|data|conversation)\b", re.I),
)
_THIRD_PARTY_QUOTE_LINE = re.compile(
    r"^\s*(?:>|-{2,}\s*forwarded|begin forwarded|from:|to:|sent:|subject:|"
    r"[A-Z][\w .'-]{1,80}\s+(?:wrote|said|asked|emailed|texted):)",
    re.I,
)


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _content_hash(item: dict[str, Any]) -> str:
    supplied = str(item.get("content_hash") or "").strip().lower()
    if supplied:
        return supplied
    content = str(item.get("content") or "")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _detector_version(provenance: str) -> str:
    if provenance == "cache":
        return CACHE_DETECTOR_VERSION
    if provenance == "llm":
        return LLM_DETECTOR_VERSION
    return FALLBACK_DETECTOR_VERSION


async def _load_llm_cache(
    conn: Any,
    *,
    task: str,
    detector_version: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    hashes = sorted({_content_hash(item) for item in items})
    if not hashes:
        return {}
    try:
        rows = await conn.fetch(
            """
            UPDATE connector_cognition_cache
            SET hit_count = hit_count + 1,
                last_used_at = CURRENT_TIMESTAMP
            WHERE task = $1
              AND detector_version = $2
              AND content_hash = ANY($3::text[])
            RETURNING content_hash, result
            """,
            task,
            detector_version,
            hashes,
        )
    except Exception:
        logger.debug("Connector cognition cache lookup unavailable", exc_info=True)
        return {}
    return {str(row["content_hash"]): _json(row["result"]) for row in rows}


async def _store_llm_cache(
    conn: Any,
    *,
    task: str,
    detector_version: str,
    results_by_hash: dict[str, Any],
    llm_config: dict[str, Any],
) -> None:
    if not results_by_hash:
        return
    payload = [
        {"content_hash": content_hash, "result": result}
        for content_hash, result in results_by_hash.items()
    ]
    try:
        await conn.execute(
            """
            INSERT INTO connector_cognition_cache (
                task, content_hash, detector_version, result,
                provenance, provider, model, last_used_at
            )
            SELECT
                $1, x.content_hash, $2, x.result,
                'llm', $3, $4, CURRENT_TIMESTAMP
            FROM jsonb_to_recordset($5::jsonb) AS x(content_hash text, result jsonb)
            ON CONFLICT (task, content_hash, detector_version) DO UPDATE SET
                result = EXCLUDED.result,
                provenance = 'llm',
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                last_used_at = CURRENT_TIMESTAMP
            """,
            task,
            detector_version,
            str(llm_config.get("provider") or "") or None,
            str(llm_config.get("model") or "") or None,
            _json_dumps(payload),
        )
    except Exception:
        logger.debug("Connector cognition cache write unavailable", exc_info=True)


def _clean_fragment(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" \t\r\n\"'`.,;:")
    return cleaned[:160]


def _claim_key(category: str, fragment: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", fragment.lower()).strip("_")
    return f"{category}:{slug[:120]}"


def _bounded_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _dedupe_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for claim in claims:
        key = str(claim.get("claim_key") or "").strip().lower()
        text = str(claim.get("claim") or "").strip()
        category = str(claim.get("category") or "preference").strip().lower()
        if not key and text:
            key = _claim_key(category, text)
        if not key or not text:
            continue
        normalized = dict(claim)
        normalized["claim_key"] = key
        normalized["claim"] = text[:600]
        normalized["category"] = category
        normalized["confidence"] = _bounded_float(normalized.get("confidence"), 0.5)
        normalized["importance"] = _bounded_float(normalized.get("importance"), 0.5)
        if normalized.get("supersedes_claim_key"):
            normalized["supersedes_claim_key"] = (
                str(normalized["supersedes_claim_key"]).strip().lower()
            )
        contradictions = normalized.get("contradicts_claim_keys")
        if not isinstance(contradictions, list):
            contradictions = []
        normalized["contradicts_claim_keys"] = [
            str(item).strip().lower() for item in contradictions if str(item).strip()
        ][:5]
        metadata = normalized.get("metadata")
        normalized["metadata"] = metadata if isinstance(metadata, dict) else {}
        deduped.setdefault(key, normalized)
    return list(deduped.values())[:12]


def _message_body(content: str) -> str:
    marker = "\nMessage:"
    if marker in content:
        return content.split(marker, 1)[1]
    marker = "\nBody:"
    if marker in content:
        return content.split(marker, 1)[1]
    return content


def _rule_extraction_text(content: str) -> str:
    """Return first-person text that is plausible user-authored material.

    Source artifacts often contain forwards, quoted replies, and copied third-
    party messages. Regex rules do not understand attribution, so strip obvious
    quoted/forwarded lines before looking for "I prefer..." style claims.
    """
    body = _message_body(content)
    kept: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if _THIRD_PARTY_QUOTE_LINE.search(stripped):
            continue
        if stripped[:1] in {'"', "'", "“", "‘"} and re.search(
            r"\bI\s+(?:prefer|like|love|hate|dislike|usually|always|often|normally|am|work|live)\b",
            stripped,
            re.I,
        ):
            continue
        kept.append(line)
    return "\n".join(kept)


def _looks_ephemeral_user_model_text(text: str) -> bool:
    lowered = text.lower()
    if len(lowered.strip()) < 12:
        return True
    return any(pattern.search(text) for pattern in _EPHEMERAL_USER_MODEL_PATTERNS)


def extract_user_model_claims(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract conservative user-model claims from one connector source item.

    This is intentionally narrow. The DB stores evidence/provenance; richer LLM
    synthesis can replace this detector without changing storage.
    """
    text = _rule_extraction_text(str(item.get("content") or ""))
    if _looks_ephemeral_user_model_text(text):
        return []
    claims: list[dict[str, Any]] = []

    for pattern in _PREFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            fragment = _clean_fragment(match.group(1))
            if not fragment:
                continue
            verb = match.group(0).split()[1].lower()
            negative = verb in {"hate", "dislike"} or "stand" in match.group(0).lower()
            category = "preference"
            claim = f"User {'dislikes' if negative else 'prefers'} {fragment}."
            claims.append(
                {
                    "claim_key": _claim_key(
                        category, f"{'dislikes' if negative else 'prefers'} {fragment}"
                    ),
                    "category": category,
                    "claim": claim,
                    "confidence": 0.62,
                    "importance": 0.55,
                }
            )

    for pattern in _ROUTINE_PATTERNS:
        for match in pattern.finditer(text):
            fragment = _clean_fragment(match.group(1))
            if not fragment:
                continue
            claims.append(
                {
                    "claim_key": _claim_key("routine", fragment),
                    "category": "routine",
                    "claim": f"User has a routine or recurring pattern: {fragment}.",
                    "confidence": 0.58,
                    "importance": 0.5,
                }
            )

    for pattern in _IDENTITY_PATTERNS:
        for match in pattern.finditer(text):
            fragment = _clean_fragment(match.group(1))
            if not fragment or fragment.lower() in {"fine", "okay", "good", "here"}:
                continue
            claims.append(
                {
                    "claim_key": _claim_key("identity", fragment),
                    "category": "identity",
                    "claim": f"User stated an identity/context fact: {fragment}.",
                    "confidence": 0.56,
                    "importance": 0.52,
                }
            )

    for pattern in _RELATIONSHIP_PATTERNS:
        for match in pattern.finditer(text):
            fragment = _clean_fragment(
                " ".join(part for part in match.groups() if part)
            )
            if not fragment:
                continue
            claims.append(
                {
                    "claim_key": _claim_key("relationship", fragment),
                    "category": "relationship",
                    "claim": f"User described a relationship context: {fragment}.",
                    "confidence": 0.54,
                    "importance": 0.58,
                }
            )

    for pattern in _COMMITMENT_PATTERNS:
        for match in pattern.finditer(text):
            fragment = _clean_fragment(match.group(1))
            if not fragment:
                continue
            claims.append(
                {
                    "claim_key": _claim_key("commitment", fragment),
                    "category": "commitment",
                    "claim": f"User has a commitment or intended action: {fragment}.",
                    "confidence": 0.57,
                    "importance": 0.62,
                }
            )

    for pattern in _JUDGMENT_PATTERNS:
        for match in pattern.finditer(text):
            fragment = _clean_fragment(match.group(1))
            if not fragment:
                continue
            claims.append(
                {
                    "claim_key": _claim_key("judgment_pattern", fragment),
                    "category": "judgment_pattern",
                    "claim": f"User expressed a judgment pattern or decision heuristic: {fragment}.",
                    "confidence": 0.55,
                    "importance": 0.6,
                }
            )

    return _dedupe_claims(claims)


_CLAIMS_SYSTEM = (
    "Extract durable user-model claims from communication history. "
    "Return JSON only. Claims must be evidence-backed, not generic summary. "
    "Allowed categories: preference, relationship, commitment, routine, "
    "judgment_pattern, identity. Include contradictions or supersession only "
    "when the new evidence directly conflicts with an existing claim. "
    "Do not create claims for one-off test instructions, jokes, or ephemeral chat filler."
)

_CLAIMS_OUTPUT_SCHEMA = {
    "claims": [
        {
            "claim_key": "stable lowercase key, e.g. routine:morning_planning",
            "category": "preference|relationship|commitment|routine|judgment_pattern|identity",
            "claim": "one sentence phrased as a belief about the user",
            "confidence": 0.0,
            "importance": 0.0,
            "supersedes_claim_key": "optional existing claim_key",
            "contradicts_claim_keys": ["optional existing claim_key"],
            "metadata": {"reason": "brief rationale"},
        }
    ]
}


def _claims_item_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "connector_id": item.get("connector_id"),
        "account_key": item.get("account_key"),
        "provider_item_id": item.get("provider_item_id"),
        "title": item.get("title"),
        "timestamp": str(item.get("item_timestamp") or ""),
        "content": _message_body(str(item.get("content") or ""))[:6000],
    }


def _parse_claims(raw: Any) -> list[dict[str, Any]]:
    claims = raw.get("claims") if isinstance(raw, dict) else None
    if not isinstance(claims, list):
        return []
    return _dedupe_claims([c for c in claims if isinstance(c, dict)])


async def _active_user_model_claims(conn: Any) -> list[dict[str, Any]]:
    """The existing-claims context. Identical for every item in a run, so it is
    fetched once per run rather than once per item."""
    rows = await conn.fetch("""
        SELECT claim_key, category, claim, confidence, importance
        FROM user_model_claims
        WHERE status = 'active'
        ORDER BY updated_at DESC
        LIMIT 80
        """)
    return [
        {
            "claim_key": row["claim_key"],
            "category": row["category"],
            "claim": row["claim"],
            "confidence": float(row["confidence"] or 0),
            "importance": float(row["importance"] or 0),
        }
        for row in rows
    ]


async def extract_user_model_claims_llm_batch(
    conn: Any,
    items: list[dict[str, Any]],
    *,
    provenance: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Claims for a whole batch in as few model calls as the budget allows.

    Keyed by ``source_item_id``. Every item is present: anything the model did
    not answer for comes back empty rather than missing.
    """
    if not items:
        return {}
    llm_config = await load_llm_config(
        conn, "llm.connector_cognition", fallback_key="llm.subconscious"
    )
    cached = await _load_llm_cache(
        conn,
        task="user_model_claims",
        detector_version=CLAIMS_CACHE_VERSION,
        items=items,
    )
    results: dict[str, list[dict[str, Any]]] = {}
    representatives: dict[str, dict[str, Any]] = {}
    keys_by_hash: dict[str, list[str]] = {}
    for item in items:
        item_key = str(item.get("source_item_id") or "")
        content_hash = _content_hash(item)
        keys_by_hash.setdefault(content_hash, []).append(item_key)
        if content_hash in cached:
            results[item_key] = _parse_claims(cached[content_hash])
            if provenance is not None:
                provenance[item_key] = "cache"
        else:
            representatives.setdefault(content_hash, item)

    uncached = list(representatives.values())
    if not uncached:
        return results

    existing = await _active_user_model_claims(conn)
    batch_provenance: dict[str, str] = {}
    classified = await batch_classify(
        uncached,
        llm_config=llm_config,
        system=_CLAIMS_SYSTEM,
        key=lambda it: str(it.get("source_item_id") or ""),
        item_payload=_claims_item_payload,
        parse=_parse_claims,
        fallback=extract_user_model_claims,
        shared={"existing_claims": existing},
        output_hint=_CLAIMS_OUTPUT_SCHEMA,
        max_tokens=8000,
        provenance=batch_provenance,
    )
    cache_writes: dict[str, Any] = {}
    for content_hash, representative in representatives.items():
        representative_key = str(representative.get("source_item_id") or "")
        claims = classified.get(representative_key, [])
        source = batch_provenance.get(representative_key, "fallback")
        for item_key in keys_by_hash.get(content_hash, []):
            results[item_key] = claims
            if provenance is not None:
                provenance[item_key] = source
        if source == "llm":
            cache_writes[content_hash] = {"claims": claims}

    await _store_llm_cache(
        conn,
        task="user_model_claims",
        detector_version=CLAIMS_CACHE_VERSION,
        results_by_hash=cache_writes,
        llm_config=llm_config,
    )
    return results


async def extract_user_model_claims_llm(
    conn: Any, item: dict[str, Any]
) -> list[dict[str, Any]]:
    """Single-item claims extraction. Retained for callers with one item;
    the batch path is what the synthesis run uses."""
    key = str(item.get("source_item_id") or "")
    out = await extract_user_model_claims_llm_batch(conn, [item])
    return out.get(key, [])


def estimate_connector_item_importance(item: dict[str, Any]) -> dict[str, Any]:
    """Non-semantic fallback used only when the model is unavailable.

    Provider-supplied priority is structured evidence and can be honored. Raw
    message prose is deliberately not searched for words that merely sound
    important; without a provider signal the fallback reports uncertainty.
    """
    metadata = item.get("raw_metadata")
    if not isinstance(metadata, dict):
        metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    priority = metadata.get("priority", metadata.get("importance"))
    score = 0.5
    reasons = ["LLM unavailable; no structured provider priority was supplied"]
    actions: list[dict[str, Any]] = []
    if isinstance(priority, (int, float)) and not isinstance(priority, bool):
        score = _bounded_float(priority, 0.5)
        reasons = ["structured provider importance score"]
    elif isinstance(priority, str):
        normalized = priority.strip().lower()
        provider_scores = {
            "urgent": 0.96,
            "critical": 0.96,
            "high": 0.86,
            "important": 0.86,
            "normal": 0.5,
            "low": 0.2,
        }
        if normalized in provider_scores:
            score = provider_scores[normalized]
            reasons = [f"structured provider priority: {normalized}"]
    label = (
        "urgent"
        if score >= 0.95
        else "important" if score >= 0.85 else "low" if score < 0.35 else "normal"
    )
    return {
        "score": score,
        "label": label,
        "reasons": reasons,
        "recommended_actions": actions,
    }


_IMPORTANCE_SYSTEM = (
    "Score a connector item for user-visible importance and route it to suggested actions. "
    "Return JSON only with score 0..1, label low|normal|important|urgent, reasons array, "
    "and recommended_actions array. High-stakes safety, finance, legal, health, security, "
    "deadline, relationship, or explicit user requests should score higher. "
    "Actions are suggestions only; sending/responding/modifying external state requires authorization."
)


async def estimate_connector_item_importance_llm_batch(
    conn: Any,
    items: list[dict[str, Any]],
    *,
    provenance: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Importance for a whole batch in as few model calls as the budget allows.

    Keyed by ``source_item_id``. A valid model verdict is authoritative; only an
    unavailable, missing, or invalid verdict receives the structural fallback.
    """
    if not items:
        return {}
    llm_config = await load_llm_config(
        conn, "llm.connector_importance", fallback_key="llm.subconscious"
    )
    cached = await _load_llm_cache(
        conn,
        task="item_importance",
        detector_version=IMPORTANCE_CACHE_VERSION,
        items=items,
    )
    results: dict[str, dict[str, Any]] = {}
    representatives: dict[str, dict[str, Any]] = {}
    keys_by_hash: dict[str, list[str]] = {}
    for item in items:
        item_key = str(item.get("source_item_id") or "")
        content_hash = _content_hash(item)
        keys_by_hash.setdefault(content_hash, []).append(item_key)
        if content_hash in cached:
            results[item_key] = _normalize_importance(cached[content_hash])
            if provenance is not None:
                provenance[item_key] = "cache"
        else:
            representatives.setdefault(content_hash, item)

    uncached = list(representatives.values())
    if not uncached:
        return results

    def _payload(it: dict[str, Any]) -> dict[str, Any]:
        return {
            "item": {
                "title": it.get("title"),
                "content": str(it.get("content") or "")[:6000],
                "connector_id": it.get("connector_id"),
                "timestamp": str(it.get("item_timestamp") or ""),
            }
        }

    batch_provenance: dict[str, str] = {}
    classified = await batch_classify(
        uncached,
        llm_config=llm_config,
        system=_IMPORTANCE_SYSTEM,
        key=lambda it: str(it.get("source_item_id") or ""),
        item_payload=_payload,
        parse=_normalize_importance,
        fallback=estimate_connector_item_importance,
        max_tokens=8000,
        provenance=batch_provenance,
    )
    cache_writes: dict[str, Any] = {}
    for content_hash, representative in representatives.items():
        representative_key = str(representative.get("source_item_id") or "")
        estimate = classified.get(
            representative_key, estimate_connector_item_importance(representative)
        )
        source = batch_provenance.get(representative_key, "fallback")
        for item_key in keys_by_hash.get(content_hash, []):
            results[item_key] = estimate
            if provenance is not None:
                provenance[item_key] = source
        if source == "llm":
            cache_writes[content_hash] = estimate

    await _store_llm_cache(
        conn,
        task="item_importance",
        detector_version=IMPORTANCE_CACHE_VERSION,
        results_by_hash=cache_writes,
        llm_config=llm_config,
    )
    return results


async def estimate_connector_item_importance_llm(
    conn: Any, item: dict[str, Any]
) -> dict[str, Any]:
    """Single-item importance. Retained for callers with one item; the batch
    path is what the importance run uses."""
    key = str(item.get("source_item_id") or "")
    out = await estimate_connector_item_importance_llm_batch(conn, [item])
    return out.get(key) or estimate_connector_item_importance(item)


def _normalize_importance(doc: Any) -> dict[str, Any]:
    """Validate one authoritative model verdict without a keyword floor."""
    if not isinstance(doc, dict):
        raise ValueError("importance verdict must be an object")
    try:
        score = max(0.0, min(1.0, float(doc.get("score"))))
    except (TypeError, ValueError):
        raise ValueError("importance verdict score must be numeric") from None
    label = str(doc.get("label") or "").lower()
    if label not in {"low", "normal", "important", "urgent"}:
        label = (
            "urgent"
            if score >= 0.95
            else "important" if score >= 0.85 else "low" if score < 0.35 else "normal"
        )
    if score >= 0.95:
        label = "urgent"
    elif score >= 0.85 and label not in {"urgent", "important"}:
        label = "important"
    reasons = (
        [str(reason) for reason in doc["reasons"]]
        if isinstance(doc.get("reasons"), list)
        else []
    )
    actions = (
        [action for action in doc["recommended_actions"] if isinstance(action, dict)]
        if isinstance(doc.get("recommended_actions"), list)
        else []
    )
    return {
        "score": score,
        "label": label,
        "reasons": list(
            dict.fromkeys(str(reason) for reason in reasons if str(reason).strip())
        )[:8],
        "recommended_actions": actions[:8],
    }


async def run_user_model_synthesis_step(
    conn: Any, *, limit: int | None = None
) -> dict[str, Any]:
    if not bool(
        await conn.fetchval(
            "SELECT COALESCE(get_config_bool('connector.user_model_synthesis_enabled'), TRUE)"
        )
    ):
        return {"skipped": True, "reason": "disabled"}
    raw = await conn.fetchval("SELECT claim_user_model_source_items($1::int)", limit)
    items = _json(raw) or []
    if not isinstance(items, list) or not items:
        return {"skipped": True, "reason": "no_connector_sources"}

    completed = 0
    failed = 0
    claims_created = 0
    llm_used = 0
    cache_used = 0
    fallback_used = 0

    # Config is read once per run, not once per item: these cannot change
    # mid-run, and eighty items previously cost a hundred and sixty round trips
    # to Postgres for two values.
    mode = str(
        await conn.fetchval(
            "SELECT COALESCE(get_config_text('connector.user_model_synthesis_mode'), 'llm')"
        )
        or "llm"
    ).lower()
    llm_enabled = bool(
        await conn.fetchval(
            "SELECT COALESCE(get_config_bool('connector.user_model_llm_enabled'), TRUE)"
        )
    )

    # One model call for the batch rather than one per item.
    llm_by_item: dict[str, list[dict[str, Any]]] = {}
    provenance_by_item: dict[str, str] = {}
    if mode != "rules" and llm_enabled:
        candidates = [it for it in items if isinstance(it, dict)]
        try:
            llm_by_item = await extract_user_model_claims_llm_batch(
                conn, candidates, provenance=provenance_by_item
            )
        except Exception as exc:
            logger.warning(
                "connector user-model LLM synthesis fell back to rules: %s", exc
            )
            llm_by_item = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        source_item_id = str(item.get("source_item_id") or "")
        try:
            source = provenance_by_item.get(source_item_id, "fallback")
            if source in {"llm", "cache"}:
                claims = llm_by_item.get(source_item_id, [])
            else:
                claims = llm_by_item.get(source_item_id)
                if claims is None:
                    claims = extract_user_model_claims(item)
            if source == "llm":
                llm_used += 1
            elif source == "cache":
                cache_used += 1
            else:
                fallback_used += 1
            result = (
                _json(
                    await conn.fetchval(
                        "SELECT record_user_model_synthesis($1::uuid, $2::jsonb, $3)",
                        source_item_id,
                        _json_dumps(claims),
                        _detector_version(source),
                    )
                )
                or {}
            )
            claims_created += int(result.get("claim_count") or 0)
            completed += 1
        except Exception as exc:
            failed += 1
            await conn.fetchval(
                "SELECT fail_user_model_source_item($1::uuid, $2)",
                source_item_id,
                str(exc),
            )

    return {
        "claimed": len(items),
        "completed": completed,
        "failed": failed,
        "claims": claims_created,
        "llm_used": llm_used,
        "cache_used": cache_used,
        "fallback_used": fallback_used,
    }


async def run_connector_importance_step(
    conn: Any, *, limit: int | None = None
) -> dict[str, Any]:
    if not bool(
        await conn.fetchval(
            "SELECT COALESCE(get_config_bool('connector.importance_detection_enabled'), TRUE)"
        )
    ):
        return {"skipped": True, "reason": "disabled"}
    raw = await conn.fetchval("SELECT claim_connector_importance_items($1::int)", limit)
    items = _json(raw) or []
    if not isinstance(items, list) or not items:
        return {"skipped": True, "reason": "no_connector_sources"}

    completed = 0
    failed = 0
    notified = 0
    llm_used = 0
    cache_used = 0
    fallback_used = 0

    # Read once per run: this cannot change mid-run, and it previously cost one
    # round trip per item.
    llm_enabled = bool(
        await conn.fetchval(
            "SELECT COALESCE(get_config_bool('connector.importance_llm_enabled'), TRUE)"
        )
    )

    # One model call for the batch rather than one per item.
    estimates: dict[str, dict[str, Any]] = {}
    provenance_by_item: dict[str, str] = {}
    if llm_enabled:
        try:
            estimates = await estimate_connector_item_importance_llm_batch(
                conn,
                [it for it in items if isinstance(it, dict)],
                provenance=provenance_by_item,
            )
        except Exception as exc:
            logger.warning(
                "connector importance LLM detector fell back to rules: %s", exc
            )
            estimates = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        source_item_id = str(item.get("source_item_id") or "")
        try:
            estimate = estimates.get(
                source_item_id
            ) or estimate_connector_item_importance(item)
            source = provenance_by_item.get(source_item_id, "fallback")
            if source == "llm":
                llm_used += 1
            elif source == "cache":
                cache_used += 1
            else:
                fallback_used += 1
            result = (
                _json(
                    await conn.fetchval(
                        """
                        SELECT record_connector_item_importance(
                            $1::uuid,
                            $2::float,
                            $3,
                            $4::jsonb,
                            $5::jsonb,
                            $6,
                            TRUE
                        )
                        """,
                        source_item_id,
                        float(estimate["score"]),
                        estimate["label"],
                        _json_dumps(estimate["reasons"]),
                        _json_dumps(estimate["recommended_actions"]),
                        _detector_version(source),
                    )
                )
                or {}
            )
            if result.get("notification_queued"):
                notified += 1
            completed += 1
        except Exception as exc:
            failed += 1
            await conn.fetchval(
                "SELECT fail_connector_item_importance($1::uuid, $2)",
                source_item_id,
                str(exc),
            )

    return {
        "claimed": len(items),
        "completed": completed,
        "failed": failed,
        "notified": notified,
        "llm_used": llm_used,
        "cache_used": cache_used,
        "fallback_used": fallback_used,
    }
