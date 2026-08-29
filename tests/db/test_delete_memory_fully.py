"""Tests for true deletion + GC (db/47) -- the irreversible part of the fade
ladder. delete_memory_fully must clean every store with no FK violation and no
orphans; run_retention_gc must respect the grace window and protection."""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_DUMMY = "array_fill(0.1, ARRAY[embedding_dimension()])::vector"


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


async def _mk(conn, content, *, importance=0.3, age_days=0):
    return await conn.fetchval(
        f"INSERT INTO memories (type, content, embedding, importance, trust_level, status, created_at) "
        f"VALUES ('episodic', $1, {_DUMMY}, $2, 0.9, 'active', now() - ($3 || ' days')::interval) RETURNING id",
        content,
        importance,
        str(age_days),
    )


async def test_delete_memory_fully_cross_store_no_orphans(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        tr = conn.transaction()
        await tr.start()
        try:
            a = await _mk(conn, "node A")
            b = await _mk(conn, "node B")
            # edges in both directions (AGE + memory_edges via create_memory_relationship)
            await conn.execute(
                "SELECT create_memory_relationship($1, $2, 'CAUSES', '{}'::jsonb)", a, b
            )
            await conn.execute(
                "SELECT create_memory_relationship($1, $2, 'SUPPORTS', '{}'::jsonb)",
                b,
                a,
            )
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM memory_edges WHERE src_id=$1 OR dst_id=$1",
                    str(a),
                )
                >= 1
            )

            assert await conn.fetchval("SELECT delete_memory_fully($1)", a) is True
            # gone everywhere; no orphan edges; b survives
            assert await conn.fetchval(
                "SELECT NOT EXISTS(SELECT 1 FROM memories WHERE id=$1)", a
            )
            assert await conn.fetchval(
                "SELECT NOT EXISTS(SELECT 1 FROM memory_edges WHERE src_id=$1 OR dst_id=$1)",
                str(a),
            )
            assert await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM memories WHERE id=$1)", b
            )
        finally:
            await tr.rollback()


async def test_delete_memory_fully_handles_reconsolidation_fk(db_pool):
    """reconsolidation_tasks has NO-ACTION FKs -> a raw delete would raise; the
    function must clear them first."""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        tr = conn.transaction()
        await tr.start()
        try:
            belief = await _mk(conn, "a belief memory")
            await conn.execute(
                "INSERT INTO reconsolidation_tasks (belief_id, old_content, new_content, transformation_type, status) "
                "VALUES ($1, 'old belief text', 'new belief text', 'shift', 'pending')",
                belief,
            )
            assert await conn.fetchval("SELECT delete_memory_fully($1)", belief) is True
            assert await conn.fetchval(
                "SELECT NOT EXISTS(SELECT 1 FROM memories WHERE id=$1)", belief
            )
        finally:
            await tr.rollback()


async def test_gc_respects_grace_and_protection(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SELECT set_config('retention.enabled', 'true'::jsonb)")
            await conn.execute(
                "SELECT set_config('retention.irreversible_pruning_enabled', 'true'::jsonb)"
            )

            gist = await _mk(conn, "retention gist")

            async def archive(mid, archived_days_ago):
                await conn.execute(
                    "UPDATE memories SET status='archived', superseded_by=$2, "
                    "metadata=jsonb_build_object('consolidation', jsonb_build_object('archived_at', (now() - ($3||' days')::interval)::text)) "
                    "WHERE id=$1",
                    mid,
                    gist,
                    str(archived_days_ago),
                )

            past_grace = await _mk(conn, "old archived original", age_days=90)
            await archive(past_grace, 30)  # past 14-day grace
            within_grace = await _mk(conn, "recent archived original")
            await archive(within_grace, 1)  # within grace
            protected = await _mk(conn, "archived but precious", importance=0.95)
            await archive(protected, 30)  # past grace but protected

            _j(await conn.fetchval("SELECT run_retention_gc()"))
            assert await conn.fetchval(
                "SELECT NOT EXISTS(SELECT 1 FROM memories WHERE id=$1)", past_grace
            )  # pruned
            assert await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM memories WHERE id=$1)", within_grace
            )  # grace
            assert await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM memories WHERE id=$1)", protected
            )  # protected
        finally:
            await tr.rollback()
