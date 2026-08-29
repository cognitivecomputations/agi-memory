"""Ambient responsibility worker.

Postgres owns responsibility state, due claiming, run audit, and delivery
envelopes. This module is the provider I/O shim for condition checks that need
external data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from core.auth.google_gmail import (
    GmailOAuthError,
    load_default_credentials,
    refresh_default_credentials_if_needed,
)
from services.connector_cognition import (
    estimate_connector_item_importance,
    estimate_connector_item_importance_llm,
)
from services.gmail_backfill import (
    GmailBackfillError,
    _connected_account_email,
    _gmail_get,
    _upsert_source_item,
    gmail_message_to_source_item,
)

logger = logging.getLogger(__name__)


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _as_dict(value: Any) -> dict[str, Any]:
    parsed = _json(value)
    return parsed if isinstance(parsed, dict) else {}


def _as_list(value: Any) -> list[Any]:
    parsed = _json(value)
    return parsed if isinstance(parsed, list) else []


def _first_dict(items: Any) -> dict[str, Any]:
    for item in _as_list(items):
        if isinstance(item, dict):
            return item
    return {}


def _response(claim: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(claim.get("responsibility"))


def _first_notify_action(responsibility: dict[str, Any]) -> dict[str, Any]:
    for action in _as_list(responsibility.get("actions")):
        if isinstance(action, dict) and str(action.get("type") or "notify_user") in {
            "notify_user",
            "notify",
            "",
        }:
            return action
    return {}


def _notify_message(
    responsibility: dict[str, Any], *, observation: dict[str, Any] | None = None
) -> str:
    action = _first_notify_action(responsibility)
    message = str(action.get("message") or "").strip()
    if message:
        if observation:
            try:
                return message.format(**observation)
            except Exception:
                return message
        return message
    title = str(responsibility.get("title") or "Ambient responsibility").strip()
    if observation and observation.get("title"):
        return f"{title}: {observation['title']}"
    return title


async def _record_observation(pool: Any, payload: dict[str, Any]) -> dict[str, Any]:
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT record_ambient_observation($1::jsonb)",
            _json_dumps(payload),
        )
    return _as_dict(raw)


async def _complete(
    pool: Any,
    run_id: str,
    status: str,
    decision: dict[str, Any],
    observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT complete_ambient_responsibility_run($1::uuid, $2, $3::jsonb, $4::jsonb)",
            run_id,
            status,
            _json_dumps(decision),
            _json_dumps(observations or []),
        )
    return _as_dict(raw)


async def _checkin_count(
    pool: Any, responsibility_id: str, *, lookback_minutes: int
) -> int:
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM ambient_checkins
            WHERE responsibility_id = $1::uuid
              AND occurred_at >= CURRENT_TIMESTAMP - make_interval(mins => $2::int)
            """,
            responsibility_id,
            int(max(lookback_minutes, 1)),
        )
    return int(raw or 0)


async def _evaluate_checkin(
    pool: Any, claim: dict[str, Any]
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    responsibility = _response(claim)
    evaluator = _as_dict(responsibility.get("evaluator"))
    lookback_minutes = int(
        evaluator.get("lookback_minutes") or evaluator.get("grace_minutes") or 720
    )
    count = await _checkin_count(
        pool, str(responsibility["id"]), lookback_minutes=lookback_minutes
    )
    if count > 0:
        return (
            "silent",
            {
                "reason": "recent_checkin_present",
                "checkins": count,
                "lookback_minutes": lookback_minutes,
            },
            [],
        )
    return (
        "fired",
        {
            "reason": "missing_checkin",
            "checkins": 0,
            "lookback_minutes": lookback_minutes,
            "notify_message": _notify_message(responsibility),
        },
        [],
    )


def _gmail_source(responsibility: dict[str, Any]) -> dict[str, Any] | None:
    for source in _as_list(responsibility.get("sources")):
        if not isinstance(source, dict):
            continue
        connector = str(
            source.get("connector_id") or source.get("connector") or ""
        ).lower()
        if connector == "gmail":
            return source
    evaluator = _as_dict(responsibility.get("evaluator"))
    if (
        str(evaluator.get("connector_id") or evaluator.get("connector") or "").lower()
        == "gmail"
    ):
        return {"connector_id": "gmail", **evaluator}
    return None


def _connector_id(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _source_connector(source: dict[str, Any]) -> str:
    return _connector_id(source.get("connector_id") or source.get("connector"))


def _generic_connector_sources(responsibility: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in _as_list(responsibility.get("sources")):
        if not isinstance(source, dict):
            continue
        connector = _source_connector(source)
        if connector and connector not in {
            "gmail",
            "health",
            "fitness",
            "wearable",
            "steps",
        }:
            result.append(source)
    return result


def _metric_source(responsibility: dict[str, Any]) -> dict[str, Any] | None:
    metric_connectors = {"health", "fitness", "wearable", "steps"}
    for source in _as_list(responsibility.get("sources")):
        if isinstance(source, dict) and _source_connector(source) in metric_connectors:
            return source
    evaluator = _as_dict(responsibility.get("evaluator"))
    if (
        _connector_id(evaluator.get("connector_id") or evaluator.get("connector"))
        in metric_connectors
    ):
        return {"connector_id": "health", **evaluator}
    if evaluator.get("metric"):
        return {"connector_id": "health", **evaluator}
    return None


def _gmail_query(responsibility: dict[str, Any], source: dict[str, Any]) -> str:
    evaluator = _as_dict(responsibility.get("evaluator"))
    query = str(source.get("query") or evaluator.get("query") or "").strip()
    if source.get("from"):
        query = f"{query} from:{source['from']}".strip()
    if evaluator.get("from"):
        query = f"{query} from:{evaluator['from']}".strip()
    if not query:
        query = "in:inbox"
    if "after:" not in query.lower():
        last_checked = responsibility.get("last_checked_at") or responsibility.get(
            "created_at"
        )
        if isinstance(last_checked, str) and last_checked.strip():
            try:
                parsed = datetime.fromisoformat(last_checked.replace("Z", "+00:00"))
                query = f"{query} after:{parsed.astimezone(timezone.utc).strftime('%Y/%m/%d')}"
            except ValueError:
                pass
    return query


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return (
            value.astimezone(timezone.utc)
            if value.tzinfo
            else value.replace(tzinfo=timezone.utc)
        )
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return (
                parsed.astimezone(timezone.utc)
                if parsed.tzinfo
                else parsed.replace(tzinfo=timezone.utc)
            )
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _message_summary(source: dict[str, Any]) -> dict[str, Any]:
    participants = (
        source.get("participants")
        if isinstance(source.get("participants"), list)
        else []
    )
    sender = ""
    for participant in participants:
        if isinstance(participant, dict) and participant.get("role") == "from":
            sender = str(participant.get("value") or "")
            break
    return {
        "provider_item_id": source.get("provider_item_id"),
        "provider_thread_id": source.get("provider_thread_id"),
        "title": source.get("title") or "(No subject)",
        "from": sender,
        "snippet": str((_as_dict(source.get("metadata")).get("gmail_snippet") or ""))[
            :500
        ],
        "observed_at": source.get("item_timestamp"),
    }


async def _importance_estimate(pool: Any, item: dict[str, Any]) -> dict[str, Any]:
    baseline = estimate_connector_item_importance(item)
    try:
        async with pool.acquire() as conn:
            llm_enabled = bool(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('ambient.importance_llm_enabled'), TRUE)"
                )
            )
            if llm_enabled:
                return await estimate_connector_item_importance_llm(conn, item)
    except Exception as exc:
        logger.debug("Ambient importance LLM detector fell back to rules: %s", exc)
    return baseline


async def _select_fire_items(
    pool: Any, responsibility: dict[str, Any], items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    evaluator = _as_dict(responsibility.get("evaluator"))
    evaluator_type = str(evaluator.get("type") or "").lower()
    if evaluator_type in {"importance", "urgent", "llm_importance"}:
        threshold = 0.95 if evaluator_type == "urgent" else 0.85
        try:
            threshold = float(evaluator.get("threshold") or threshold)
        except (TypeError, ValueError):
            pass
        selected = []
        for item in items:
            content = " ".join(
                str(item.get(key) or "")
                for key in ("title", "from", "snippet", "content")
            )
            estimate = await _importance_estimate(pool, {**item, "content": content})
            item["importance"] = estimate
            if float(estimate.get("score") or 0.0) >= threshold:
                selected.append(item)
        return selected
    return items


def _source_query_terms(
    source: dict[str, Any], evaluator: dict[str, Any]
) -> tuple[list[str], list[str]]:
    query = str(source.get("query") or evaluator.get("query") or "").strip()
    from_terms: list[str] = []
    text_terms: list[str] = []
    for raw in query.replace("(", " ").replace(")", " ").split():
        token = raw.strip().strip("\"'")
        if not token:
            continue
        lowered = token.lower()
        if lowered.startswith("from:") and len(token) > 5:
            from_terms.append(token[5:].lower())
        elif ":" not in token:
            text_terms.append(lowered)
    for key in ("from", "sender"):
        if source.get(key):
            from_terms.append(str(source[key]).lower())
        if evaluator.get(key):
            from_terms.append(str(evaluator[key]).lower())
    return from_terms, text_terms


def _generic_item_matches(
    source: dict[str, Any], evaluator: dict[str, Any], item: dict[str, Any]
) -> bool:
    from_terms, text_terms = _source_query_terms(source, evaluator)
    participants = json.dumps(item.get("participants") or [], default=str).lower()
    text = " ".join(
        str(item.get(key) or "")
        for key in ("title", "content", "provider_item_id", "provider_thread_id")
    ).lower()
    labels = {str(label).lower() for label in _as_list(item.get("labels"))}
    requested_labels = source.get("labels") or evaluator.get("labels")
    if isinstance(requested_labels, str):
        requested_labels = [requested_labels]
    if isinstance(requested_labels, list):
        wanted = {
            str(label).lower() for label in requested_labels if str(label).strip()
        }
        if wanted and labels.isdisjoint(wanted):
            return False
    if from_terms and not all(term in participants for term in from_terms):
        return False
    if text_terms and not all(term in text for term in text_terms):
        return False
    return True


async def _fetch_generic_source_items(
    pool: Any,
    responsibility: dict[str, Any],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    connector = _source_connector(source)
    if not connector:
        return []
    account_key = str(source.get("account_key") or "").strip() or None
    item_kind = str(source.get("item_kind") or "").strip() or None
    since = _parse_timestamp(
        responsibility.get("last_checked_at") or responsibility.get("created_at")
    )
    page_size = int(source.get("page_size") or source.get("max_results") or 0)
    if page_size <= 0:
        async with pool.acquire() as conn:
            page_size = int(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_int('ambient.generic_source_page_size'), 25)"
                )
                or 25
            )
    page_size = min(max(page_size, 1), 100)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                csi.id::text AS source_item_id,
                csi.connector_id,
                csi.account_key,
                csi.provider_item_id,
                csi.provider_thread_id,
                csi.item_kind,
                csi.source_document_id::text AS source_document_id,
                csi.item_timestamp,
                csi.labels,
                csi.participants,
                csi.raw_metadata,
                csi.first_seen_at,
                csi.last_seen_at,
                COALESCE(sd.title, csi.provider_item_id) AS title,
                COALESCE(sd.content, '') AS content
            FROM connector_source_items csi
            LEFT JOIN source_documents sd ON sd.id = csi.source_document_id
            WHERE csi.status = 'active'
              AND csi.connector_id = $1
              AND ($2::text IS NULL OR csi.account_key = $2)
              AND ($3::text IS NULL OR csi.item_kind = $3)
              AND COALESCE(csi.item_timestamp, csi.first_seen_at, csi.created_at) > $4
            ORDER BY COALESCE(csi.item_timestamp, csi.first_seen_at, csi.created_at) ASC
            LIMIT $5
            """,
            connector,
            account_key,
            item_kind,
            since,
            page_size,
        )
    evaluator = _as_dict(responsibility.get("evaluator"))
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["labels"] = list(item.get("labels") or [])
        item["participants"] = _json(item.get("participants")) or []
        item["raw_metadata"] = _json(item.get("raw_metadata")) or {}
        if _generic_item_matches(source, evaluator, item):
            items.append(item)
    return items


async def _evaluate_gmail(
    pool: Any, claim: dict[str, Any]
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    responsibility = _response(claim)
    source = _gmail_source(responsibility)
    if source is None:
        return "silent", {"reason": "no_gmail_source"}, []

    if load_default_credentials() is None:
        return (
            "blocked",
            {
                "reason": "gmail_not_connected",
                "missing_connectors": [
                    {"connector_id": "gmail", "status": "not_connected"}
                ],
            },
            [],
        )

    credentials = await refresh_default_credentials_if_needed()
    account_key = str(source.get("account_key") or "").strip().lower()
    if not account_key:
        account_key = await _connected_account_email(credentials) or "default"

    page_size = int(source.get("page_size") or source.get("max_results") or 10)
    page_size = min(max(page_size, 1), 25)
    query = _gmail_query(responsibility, source)
    listed = await _gmail_get(
        credentials,
        "/users/me/messages",
        params={
            "q": query,
            "maxResults": page_size,
            "includeSpamTrash": bool(source.get("include_spam_trash", False)),
        },
    )
    stubs = listed.get("messages") if isinstance(listed.get("messages"), list) else []
    observations: list[dict[str, Any]] = []
    new_items: list[dict[str, Any]] = []

    for stub in stubs:
        if not isinstance(stub, dict) or not isinstance(stub.get("id"), str):
            continue
        message = await _gmail_get(
            credentials,
            f"/users/me/messages/{stub['id']}",
            params={"format": "full"},
        )
        source_item = gmail_message_to_source_item(message)
        stored = await _upsert_source_item(
            pool,
            {
                "connector_id": "gmail",
                "account_key": account_key,
            },
            source_item,
        )
        summary = _message_summary(source_item)
        observation = await _record_observation(
            pool,
            {
                "responsibility_id": responsibility["id"],
                "connector_id": "gmail",
                "account_key": account_key,
                "item_kind": "message",
                "provider_item_id": source_item["provider_item_id"],
                "provider_thread_id": source_item.get("provider_thread_id"),
                "observed_at": source_item.get("item_timestamp"),
                "title": source_item.get("title"),
                "content": source_item.get("content"),
                "participants": source_item.get("participants"),
                "labels": source_item.get("labels"),
                "source_item_id": stored.get("source_item_id"),
                "source_document_id": stored.get("document_id"),
                "raw": {
                    "query": query,
                    "metadata": source_item.get("metadata") or {},
                },
            },
        )
        observation = {**summary, **observation}
        observations.append(observation)
        if observation.get("created"):
            new_items.append(
                {**summary, "content": source_item.get("content", ""), **observation}
            )

    selected = await _select_fire_items(pool, responsibility, new_items)
    if not selected:
        return (
            "silent",
            {
                "reason": "no_new_matching_messages",
                "query": query,
                "new_observations": len(new_items),
                "observations_seen": len(observations),
            },
            observations,
        )

    if len(selected) == 1:
        first = selected[0]
        notify = _notify_message(responsibility, observation=first)
        if notify == str(responsibility.get("title") or "").strip():
            notify = f"{responsibility['title']}: {first.get('from') or 'Gmail'} - {first.get('title') or '(No subject)'}"
    else:
        notify = (
            f"{responsibility['title']}: {len(selected)} new matching Gmail messages."
        )

    return (
        "fired",
        {
            "reason": "gmail_messages_matched",
            "query": query,
            "matched_count": len(selected),
            "new_observations": len(new_items),
            "notify_message": notify,
            "messages": [
                {
                    k: item.get(k)
                    for k in (
                        "provider_item_id",
                        "title",
                        "from",
                        "snippet",
                        "observation_id",
                    )
                }
                for item in selected[:10]
            ],
        },
        observations,
    )


async def _evaluate_generic_connector_sources(
    pool: Any,
    claim: dict[str, Any],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    responsibility = _response(claim)
    sources = _generic_connector_sources(responsibility)
    if not sources:
        return "silent", {"reason": "no_generic_sources"}, []

    observations: list[dict[str, Any]] = []
    new_items: list[dict[str, Any]] = []
    for source in sources:
        connector = _source_connector(source)
        for item in await _fetch_generic_source_items(pool, responsibility, source):
            observation = await _record_observation(
                pool,
                {
                    "responsibility_id": responsibility["id"],
                    "connector_id": connector,
                    "account_key": item.get("account_key"),
                    "item_kind": item.get("item_kind") or "message",
                    "provider_item_id": item.get("provider_item_id"),
                    "provider_thread_id": item.get("provider_thread_id"),
                    "observed_at": item.get("item_timestamp")
                    or item.get("first_seen_at"),
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "participants": item.get("participants"),
                    "labels": item.get("labels"),
                    "source_item_id": item.get("source_item_id"),
                    "source_document_id": item.get("source_document_id"),
                    "raw": {
                        "query": source.get("query"),
                        "metadata": item.get("raw_metadata") or {},
                    },
                },
            )
            summary = {
                "provider_item_id": item.get("provider_item_id"),
                "provider_thread_id": item.get("provider_thread_id"),
                "title": item.get("title") or "(Untitled)",
                "from": json.dumps(item.get("participants") or [], default=str)[:240],
                "snippet": str(item.get("content") or "")[:500],
                "observed_at": item.get("item_timestamp") or item.get("first_seen_at"),
                "source_item_id": item.get("source_item_id"),
                "source_document_id": item.get("source_document_id"),
                "connector_id": connector,
            }
            observation = {**summary, **observation}
            observations.append(observation)
            if observation.get("created"):
                new_items.append(
                    {**summary, "content": item.get("content") or "", **observation}
                )

    selected = await _select_fire_items(pool, responsibility, new_items)
    if not selected:
        return (
            "silent",
            {
                "reason": "no_new_matching_source_items",
                "new_observations": len(new_items),
                "observations_seen": len(observations),
                "source_count": len(sources),
            },
            observations,
        )

    if len(selected) == 1:
        first = selected[0]
        notify = _notify_message(responsibility, observation=first)
        if notify == str(responsibility.get("title") or "").strip():
            notify = f"{responsibility['title']}: {first.get('title') or '(Untitled)'}"
    else:
        notify = f"{responsibility['title']}: {len(selected)} new matching items."

    return (
        "fired",
        {
            "reason": "connector_source_items_matched",
            "matched_count": len(selected),
            "new_observations": len(new_items),
            "notify_message": notify,
            "items": [
                {
                    k: item.get(k)
                    for k in (
                        "connector_id",
                        "source_item_id",
                        "provider_item_id",
                        "title",
                        "snippet",
                        "observation_id",
                        "importance",
                    )
                }
                for item in selected[:10]
            ],
        },
        observations,
    )


def _compare_metric(value: float, operator: str, threshold: float) -> bool:
    if operator in {"<", "lt", "below", "less_than"}:
        return value < threshold
    if operator in {"<=", "lte", "at_or_below"}:
        return value <= threshold
    if operator in {">", "gt", "above", "greater_than"}:
        return value > threshold
    if operator in {">=", "gte", "at_or_above"}:
        return value >= threshold
    if operator in {"=", "==", "eq"}:
        return value == threshold
    return False


async def _evaluate_metric_threshold(
    _pool: Any,
    claim: dict[str, Any],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    responsibility = _response(claim)
    source = _metric_source(responsibility)
    evaluator = _as_dict(responsibility.get("evaluator"))
    if source is None:
        return "silent", {"reason": "no_metric_source"}, []

    raw_value = source.get("current_value", evaluator.get("current_value"))
    raw_threshold = source.get(
        "value", evaluator.get("value", evaluator.get("threshold"))
    )
    metric = str(source.get("metric") or evaluator.get("metric") or "metric")
    operator = str(source.get("operator") or evaluator.get("operator") or "<").lower()
    if raw_value is None:
        return (
            "blocked",
            {
                "reason": "metric_provider_not_configured",
                "metric": metric,
                "missing_connectors": [
                    {"connector_id": "health", "status": "not_connected"}
                ],
                "next_step": "Connect a health or wearable provider that can write metric observations.",
            },
            [],
        )
    try:
        value = float(raw_value)
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        return "failed", {"reason": "invalid_metric_threshold", "metric": metric}, []

    if not _compare_metric(value, operator, threshold):
        return (
            "silent",
            {
                "reason": "threshold_not_crossed",
                "metric": metric,
                "value": value,
                "operator": operator,
                "threshold": threshold,
            },
            [],
        )
    return (
        "fired",
        {
            "reason": "threshold_crossed",
            "metric": metric,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "notify_message": _notify_message(responsibility),
        },
        [],
    )


async def _evaluate_reminder(
    _pool: Any,
    claim: dict[str, Any],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    responsibility = _response(claim)
    return (
        "fired",
        {
            "reason": "scheduled_reminder",
            "due_at": claim.get("due_at"),
            "notify_message": _notify_message(responsibility),
        },
        [],
    )


async def evaluate_ambient_responsibility(
    pool: Any,
    claim: dict[str, Any],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    responsibility = _response(claim)
    kind = str(responsibility.get("kind") or "").lower()
    evaluator = _as_dict(responsibility.get("evaluator"))
    evaluator_type = str(evaluator.get("type") or "").lower()

    if kind == "checkin" or evaluator_type in {"missing_checkin", "checkin_missing"}:
        return await _evaluate_checkin(pool, claim)
    if _metric_source(responsibility) is not None:
        return await _evaluate_metric_threshold(pool, claim)
    if _gmail_source(responsibility) is not None:
        return await _evaluate_gmail(pool, claim)
    if _generic_connector_sources(responsibility):
        return await _evaluate_generic_connector_sources(pool, claim)
    if kind == "reminder":
        return await _evaluate_reminder(pool, claim)
    return "silent", {"reason": "no_supported_evaluator"}, []


async def run_ambient_responsibility_step(
    pool: Any, *, limit: int | None = None
) -> dict[str, Any]:
    """Claim and evaluate due ambient responsibilities."""
    async with pool.acquire() as conn:
        enabled = await conn.fetchval(
            "SELECT COALESCE(get_config_bool('ambient.enabled'), TRUE)"
        )
        if not bool(enabled):
            return {"skipped": True, "reason": "disabled"}
        raw_limit = limit or await conn.fetchval(
            "SELECT COALESCE(get_config_int('ambient.batch_size'), 20)"
        )
        raw = await conn.fetchval(
            "SELECT claim_due_ambient_responsibilities($1::int)", int(raw_limit or 20)
        )
    claims = _as_list(raw)
    if not claims:
        return {"skipped": True, "reason": "no_due_ambient_responsibilities"}

    result: dict[str, Any] = {
        "claimed": len(claims),
        "completed": 0,
        "fired": 0,
        "silent": 0,
        "blocked": 0,
        "failed": 0,
        "outbox_messages": [],
        "runs": [],
    }
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        run_id = str(claim.get("run_id") or "")
        if not run_id:
            continue
        try:
            status, decision, observations = await evaluate_ambient_responsibility(
                pool, claim
            )
        except (GmailBackfillError, GmailOAuthError) as exc:
            status, decision, observations = (
                "blocked",
                {"reason": "provider_unavailable", "error": str(exc)},
                [],
            )
        except Exception as exc:
            logger.exception("Ambient responsibility evaluation failed")
            status, decision, observations = (
                "failed",
                {"reason": "evaluation_error", "error": str(exc)},
                [],
            )

        completed = await _complete(pool, run_id, status, decision, observations)
        if completed.get("success"):
            result["completed"] += 1
        normalized = str(completed.get("status") or status)
        if normalized in {"fired", "silent", "blocked", "failed"}:
            result[normalized] += 1
        outbox = completed.get("outbox_messages")
        if isinstance(outbox, list) and outbox:
            result["outbox_messages"].extend(outbox)
        result["runs"].append(
            {
                "run_id": run_id,
                "responsibility_id": completed.get("responsibility_id")
                or _response(claim).get("id"),
                "status": normalized,
                "decision": completed.get("decision") or decision,
            }
        )

    return result
