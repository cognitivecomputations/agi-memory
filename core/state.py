from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _coerce_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


async def run_heartbeat(conn) -> dict[str, Any] | None:
    raw = await conn.fetchval("SELECT run_heartbeat()")
    if raw is None:
        return None
    return _coerce_json(raw)


async def has_pending_inbound_disposition_wake(conn) -> bool:
    """Whether a fresh operator correction should bypass active-hour cadence."""
    try:
        raw = await conn.fetchval("SELECT has_pending_inbound_disposition_wake()")
        return raw is True or (isinstance(raw, str) and raw.strip().lower() == "true")
    except Exception:
        # Migration skew retains the established heartbeat cadence.
        return False


async def apply_heartbeat_decision(
    conn,
    *,
    heartbeat_id: str,
    decision: dict[str, Any],
    start_index: int,
    pre_executed_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw = await conn.fetchval(
        "SELECT apply_heartbeat_decision($1::uuid, $2::jsonb, $3::int, $4::jsonb)",
        heartbeat_id,
        json.dumps(decision),
        start_index,
        json.dumps(pre_executed_actions or []),
    )
    return _coerce_json(raw)


async def run_maintenance_if_due(
    conn, stats_hint: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    raw = await conn.fetchval(
        "SELECT run_maintenance_if_due($1::jsonb)",
        json.dumps(stats_hint or {}),
    )
    if raw is None:
        return None
    return _coerce_json(raw)


async def run_scheduled_tasks(conn, limit: int = 25) -> dict[str, Any] | None:
    raw = await conn.fetchval("SELECT run_scheduled_tasks($1::int)", int(limit))
    if raw is None:
        return None
    return _coerce_json(raw)


async def ambient_responsibility_status(conn) -> dict[str, Any] | None:
    raw = await conn.fetchval("SELECT ambient_responsibility_status()")
    if raw is None:
        return None
    return _coerce_json(raw)


async def recompute_cron_next_runs(conn, task_ids: list[str]) -> int:
    """Ask Postgres to recompute cron next-run placeholders.

    recompute_cron_next_runs (db/36) owns the cron math; the former Python
    croniter fallback was deleted (migrations guarantee the function exists).
    """
    if not task_ids:
        return 0
    raw = await conn.fetchval("SELECT recompute_cron_next_runs($1::uuid[])", task_ids)
    return int(raw or 0)


async def apply_external_call_result(
    conn,
    *,
    call: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    raw = await conn.fetchval(
        "SELECT apply_external_call_result($1::jsonb, $2::jsonb)",
        json.dumps(call),
        json.dumps(output),
    )
    return _coerce_json(raw)


async def should_run_subconscious_decider(conn) -> bool:
    return bool(await conn.fetchval("SELECT should_run_subconscious_decider()"))


async def mark_subconscious_decider_run(conn) -> None:
    await conn.execute("SELECT mark_subconscious_decider_run()")


async def is_agent_terminated(conn) -> bool:
    return bool(await conn.fetchval("SELECT is_agent_terminated()"))
