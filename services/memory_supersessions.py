"""Typed access to database-owned memory supersession lineage.

Postgres performs the atomic validity-window and lineage updates.  This module
does not duplicate that policy or hide failures: callers either receive the
durable row identifier or an actionable exception from the authoritative write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

_ALLOWED_STATUSES = frozenset({"active", "pending", "reverted"})


@dataclass(frozen=True)
class Supersession:
    superseded_memory_id: UUID
    replacement_memory_id: UUID | None = None
    reason: str = ""
    actor: str = ""
    status: str = "active"
    superseded_at: datetime | None = None
    resolved_at: datetime | None = None
    replacement_planned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID | None = None


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (str, bytes)):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _from_row(row: Any) -> Supersession:
    return Supersession(
        id=row["id"],
        superseded_memory_id=row["superseded_memory_id"],
        replacement_memory_id=row["replacement_memory_id"],
        reason=row["reason"],
        actor=row["actor"],
        status=row["status"],
        superseded_at=row["superseded_at"],
        resolved_at=row["resolved_at"],
        replacement_planned=row["replacement_planned"],
        metadata=_metadata(row["metadata"]),
    )


def _validate(event: Supersession) -> None:
    if event.superseded_memory_id is None:
        raise ValueError("superseded_memory_id is required")
    if event.replacement_memory_id == event.superseded_memory_id:
        raise ValueError("a memory cannot supersede itself")
    if not event.reason.strip():
        raise ValueError("a supersession reason is required")
    if not event.actor.strip():
        raise ValueError("a supersession actor is required")
    if event.status not in _ALLOWED_STATUSES:
        raise ValueError(f"invalid supersession status: {event.status}")
    if event.status == "active" and event.resolved_at is not None:
        raise ValueError("an active supersession cannot already be resolved")


async def record_supersession(pool: Any, event: Supersession) -> UUID:
    """Atomically record lineage and close the old memory's validity window."""

    if pool is None:
        raise ValueError("a database pool is required to record supersession lineage")
    _validate(event)
    async with pool.acquire() as conn:
        row_id = await conn.fetchval(
            """
            SELECT record_supersession(
                $1::uuid, $2::uuid, $3, $4, $5, $6::timestamptz,
                $7::timestamptz, $8, $9::jsonb
            )
            """,
            event.superseded_memory_id,
            event.replacement_memory_id,
            event.reason,
            event.actor,
            event.status,
            event.superseded_at,
            event.resolved_at,
            event.replacement_planned,
            json.dumps(event.metadata, default=str),
        )
    if row_id is None:
        raise RuntimeError("the database did not return a supersession row id")
    return row_id


async def active_supersession_for(pool: Any, memory_id: UUID) -> Supersession | None:
    """Return the current explicit revision of ``memory_id``, if one exists."""

    if pool is None:
        raise ValueError("a database pool is required to read supersession lineage")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM memory_supersessions_active
            WHERE superseded_memory_id = $1::uuid
            """,
            memory_id,
        )
    return None if row is None else _from_row(row)


async def supersession_history_for(pool: Any, memory_id: UUID) -> list[Supersession]:
    """Return every revision event involving a memory, newest first."""

    if pool is None:
        raise ValueError("a database pool is required to read supersession lineage")
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *
            FROM memory_supersessions
            WHERE superseded_memory_id = $1::uuid
               OR replacement_memory_id = $1::uuid
            ORDER BY superseded_at DESC, id DESC
            """,
            memory_id,
        )
    return [_from_row(row) for row in rows]


async def revert_supersession(
    pool: Any,
    supersession_id: UUID,
    *,
    reason: str,
    actor: str,
) -> bool:
    """Explicitly undo an active revision and restore the old validity window."""

    if pool is None:
        raise ValueError("a database pool is required to revert supersession lineage")
    if not reason.strip() or not actor.strip():
        raise ValueError("a revert reason and actor are required")
    async with pool.acquire() as conn:
        return bool(
            await conn.fetchval(
                "SELECT revert_supersession($1::uuid, $2, $3)",
                supersession_id,
                reason,
                actor,
            )
        )


__all__ = [
    "Supersession",
    "active_supersession_for",
    "record_supersession",
    "revert_supersession",
    "supersession_history_for",
]
