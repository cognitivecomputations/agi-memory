"""Tests for maybe_write_daily_journal_fallback() (db/46, migration 0239).

Issue #114: journal_entries had zero rows ever despite a fully working write
path, because journaling stayed 100% deliberate/opt-in. This is the one
DB-owned exception: subconscious maintenance may write a single minimal entry
for a local day with real activity and nothing journaled deliberately.

Every test resets the shared `daily_journal_fallback` state key and pins
agent.timezone to UTC inside its own rolled-back transaction, so real
(possibly live) data in this database can't make the assertions flaky.
"""
from __future__ import annotations

import json

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


async def _reset_for_test(conn, *, hour: int, min_episodes: int = 1):
    """Pin timezone/config and clear the once-per-day gate, all transaction-local."""
    await conn.execute("SELECT set_state('daily_journal_fallback', '{}'::jsonb)")
    await conn.execute(
        "INSERT INTO config (key, value) VALUES ('agent.timezone', to_jsonb('UTC'::text)) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
    )
    await conn.execute(
        "INSERT INTO config (key, value) VALUES ('maintenance.daily_journal_fallback_hour', $1::text::jsonb) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        str(hour),
    )
    await conn.execute(
        "INSERT INTO config (key, value) VALUES "
        "('maintenance.daily_journal_fallback_min_episodes', $1::text::jsonb) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        str(min_episodes),
    )
    await conn.execute(
        "INSERT INTO config (key, value) VALUES "
        "('maintenance.daily_journal_fallback_enabled', 'true'::jsonb) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
    )
    # A clean slate for "today" -- transaction-scoped, rolled back at test end.
    await conn.execute("DELETE FROM journal_entries WHERE written_at::date = CURRENT_DATE")


async def _seed_episodic(conn, content: str, importance: float = 0.7):
    await conn.fetchval(
        """
        INSERT INTO memories (type, content, embedding, importance, trust_level, status)
        VALUES ('episodic', $1, array_fill(0.1, ARRAY[embedding_dimension()])::vector, $2, 0.9, 'active')
        RETURNING id
        """,
        content,
        importance,
    )


async def test_disabled_is_skipped(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _reset_for_test(conn, hour=0, min_episodes=1)
            await conn.execute(
                "UPDATE config SET value = 'false'::jsonb "
                "WHERE key = 'maintenance.daily_journal_fallback_enabled'"
            )
            result = _j(await conn.fetchval("SELECT maybe_write_daily_journal_fallback()"))
            assert result == {"skipped": True, "reason": "disabled"}
        finally:
            await tr.rollback()


async def test_too_early_in_local_day_is_skipped(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # Threshold of 24 can never be reached by an hour-of-day (0-23).
            await _reset_for_test(conn, hour=24, min_episodes=1)
            result = _j(await conn.fetchval("SELECT maybe_write_daily_journal_fallback()"))
            assert result["skipped"] is True
            assert result["reason"] == "too_early"
        finally:
            await tr.rollback()


async def test_not_enough_activity_is_skipped(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # No real day has a million notable episodes -- deterministic regardless of live data.
            await _reset_for_test(conn, hour=0, min_episodes=1_000_000)
            result = _j(await conn.fetchval("SELECT maybe_write_daily_journal_fallback()"))
            assert result["skipped"] is True
            assert result["reason"] == "not_enough_activity"
            assert result["threshold"] == 1_000_000
        finally:
            await tr.rollback()


async def test_already_journaled_today_is_skipped(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _reset_for_test(conn, hour=0, min_episodes=1)
            await conn.fetchval(
                "SELECT write_journal_entry($1, $2)",
                "I chose to write about it myself today.",
                get_test_identifier("deliberate_entry"),
            )
            result = _j(await conn.fetchval("SELECT maybe_write_daily_journal_fallback()"))
            assert result["skipped"] is True
            assert result["reason"] == "already_journaled"
        finally:
            await tr.rollback()


async def test_meaningful_activity_and_no_entry_writes_one_fallback_entry(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _reset_for_test(conn, hour=0, min_episodes=1)
            await _seed_episodic(conn, get_test_identifier("a_notable_moment"), importance=0.8)

            result = _j(await conn.fetchval("SELECT maybe_write_daily_journal_fallback()"))
            assert result["journaled"] is True
            entry_id = result["entry_id"]
            assert entry_id is not None

            row = await conn.fetchrow(
                "SELECT title, tags, metadata FROM journal_entries WHERE id = $1", entry_id
            )
            assert row is not None
            assert "auto-generated" in (row["tags"] or [])
            assert _j(row["metadata"])["source"] == "subconscious_daily_fallback"

            # Never fires twice for the same local date, even with activity to spare.
            second = _j(await conn.fetchval("SELECT maybe_write_daily_journal_fallback()"))
            assert second == {"skipped": True, "reason": "already_checked_today"}
        finally:
            await tr.rollback()
