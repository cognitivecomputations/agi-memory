"""Tests for subconscious triage -> conscious veto (db/47, db/17) -- Phase 4 of
docs/memory_retention_design.md §5. Borderline consolidations escalate to the
conscious heartbeat, which may spend a finite per-chapter point to KEEP them."""
from __future__ import annotations

import json
from uuid import uuid4

import asyncpg
import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_DUMMY = "array_fill(0.1, ARRAY[embedding_dimension()])::vector"


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


def _onehot(k):
    return (f"(array_fill(0.0::float, ARRAY[{k}]) || ARRAY[1.0::float] "
            f"|| array_fill(0.0::float, ARRAY[embedding_dimension() - {k} - 1]))::vector")


async def test_relational_evidence_is_borderline(db_pool):
    """A mundane memory cited as evidence for a relationship is borderline -- memories
    about the people we care about get a conscious look before fading."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mid = await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic','an ordinary afternoon with a friend', {_DUMMY}, 0.3, 0.9, 'active') RETURNING id")
            assert await conn.fetchval("SELECT is_consolidation_borderline(ARRAY[$1]::uuid[])", mid) is False
            await conn.execute(
                "INSERT INTO memory_edges (src_type, src_id, rel_type, dst_type, dst_id, properties) "
                "VALUES ('self','self','ASSOCIATED','concept','Alex', "
                "        jsonb_build_object('kind','relationship','evidence_memory_id',$1::text))", str(mid))
            assert await conn.fetchval("SELECT is_consolidation_borderline(ARRAY[$1]::uuid[])", mid) is True
        finally:
            await tr.rollback()


async def test_schema_fit_signal_is_opt_in(db_pool):
    """Poor schema fit (nothing in the schema is close) escalates a novel memory --
    but only when retention.borderline_schema_fit is turned on."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('semantic','a known fact', {_onehot(0)}, 0.5, 0.9, 'active')")
            novel = await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic','a wholly novel unrelated moment', {_onehot(10)}, 0.3, 0.9, 'active') RETURNING id")
            # off by default -> mundane novel memory is not borderline
            assert await conn.fetchval("SELECT is_consolidation_borderline(ARRAY[$1]::uuid[])", novel) is False
            # enable it -> nothing in the schema is close, so it escalates
            await conn.execute("SELECT set_config('retention.borderline_schema_fit', '0.5'::jsonb)")
            assert await conn.fetchval("SELECT is_consolidation_borderline(ARRAY[$1]::uuid[])", novel) is True
        finally:
            await tr.rollback()


async def _enable(conn):
    await conn.execute("SELECT set_config('retention.enabled', 'true'::jsonb)")


async def _aged_group(conn, ep_key, importance):
    """3 aged/idle/low-strength episodic memories grouped under one episode."""
    ids = []
    for k in range(3):
        i = await conn.fetchval(
            f"INSERT INTO memories (type, content, embedding, importance, trust_level, status, created_at, last_reinforced) "
            f"VALUES ('episodic', $1, {_DUMMY}, $2, 0.9, 'active', now() - interval '90 days', now() - interval '90 days') "
            f"RETURNING id",
            f"{ep_key} memory {k}", importance)
        await conn.execute(
            "INSERT INTO memory_edges (src_type, src_id, rel_type, dst_type, dst_id) "
            "VALUES ('memory', $1, 'IN_EPISODE', 'episode', $2) ON CONFLICT DO NOTHING", str(i), ep_key)
        ids.append(i)
    return ids


# ---------------------------------------------------------------- borderline fn
async def test_is_consolidation_borderline(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            mundane = await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic','plain', {_DUMMY}, 0.3, 0.9, 'active') RETURNING id")
            near = await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic','nearly precious', {_DUMMY}, 0.78, 0.9, 'active') RETURNING id")  # 0.78 >= 0.85-0.15
            assert await conn.fetchval("SELECT is_consolidation_borderline(ARRAY[$1]::uuid[])", mundane) is False
            assert await conn.fetchval("SELECT is_consolidation_borderline(ARRAY[$1]::uuid[])", near) is True
            # a mixed group is borderline if ANY member is near a threshold
            assert await conn.fetchval("SELECT is_consolidation_borderline(ARRAY[$1,$2]::uuid[])", mundane, near) is True
        finally:
            await tr.rollback()


# ---------------------------------------------------------------- triage gate
async def test_borderline_group_is_escalated_not_consolidated(db_pool):
    # escalate path touches no LLM/graph, so this needs neither AGE nor embeddings
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _enable(conn)
            await conn.execute("SELECT set_state('retention_veto_budget', $1::jsonb)",
                               json.dumps({"chapter": "unknown", "remaining": 5, "total": 5}))
            ids = await _aged_group(conn, "ep-borderline", importance=0.78)
            _j(await conn.fetchval("SELECT run_memory_rest()"))
            # escalated: a pending review over these ids exists, and none were archived
            assert await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM memory_review_queue WHERE status='pending' AND memory_ids && $1::uuid[])", ids)
            assert await conn.fetchval(
                "SELECT count(*) FROM memories WHERE id = ANY($1::uuid[]) AND status='active'", ids) == 3
        finally:
            await tr.rollback()


async def test_mundane_group_consolidates(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        tr = conn.transaction()
        await tr.start()
        try:
            await _enable(conn)
            await conn.execute("SELECT set_state('retention_veto_budget', $1::jsonb)",
                               json.dumps({"chapter": "unknown", "remaining": 5, "total": 5}))
            ids = await _aged_group(conn, "ep-mundane", importance=0.3)
            _j(await conn.fetchval("SELECT run_memory_rest()"))
            # consolidated: not escalated, and the originals were archived
            assert not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM memory_review_queue WHERE status='pending' AND memory_ids && $1::uuid[])", ids)
            assert await conn.fetchval(
                "SELECT count(*) FROM memories WHERE id = ANY($1::uuid[]) AND status='archived'", ids) == 3
        finally:
            await tr.rollback()


async def test_no_budget_lets_borderline_consolidate(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        tr = conn.transaction()
        await tr.start()
        try:
            await _enable(conn)
            # 0 budget, same chapter so the rest pass won't refill it
            await conn.execute("SELECT set_state('retention_veto_budget', $1::jsonb)",
                               json.dumps({"chapter": "unknown", "remaining": 0, "total": 5}))
            ids = await _aged_group(conn, "ep-nobudget", importance=0.78)
            _j(await conn.fetchval("SELECT run_memory_rest()"))
            assert not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM memory_review_queue WHERE status='pending' AND memory_ids && $1::uuid[])", ids)
            assert await conn.fetchval(
                "SELECT count(*) FROM memories WHERE id = ANY($1::uuid[]) AND status='archived'", ids) == 3
        finally:
            await tr.rollback()


# ---------------------------------------------------------------- budget
async def test_budget_refills_on_new_chapter(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SELECT set_state('retention_veto_budget', $1::jsonb)",
                               json.dumps({"chapter": "Old Era", "remaining": 0, "total": 5}))
            # move to a new chapter (relational life_chapter edge)
            await conn.execute(
                "INSERT INTO memory_edges (src_type, src_id, rel_type, dst_type, dst_id, properties) "
                "VALUES ('self','self','ASSOCIATED','life_chapter','current', jsonb_build_object('name','New Era')) "
                "ON CONFLICT (src_type, src_id, rel_type, dst_type, dst_id) DO UPDATE SET properties = EXCLUDED.properties")
            await conn.execute("SELECT reset_retention_budget_if_new_chapter()")
            assert await conn.fetchval("SELECT retention_budget_remaining()") == 5
        finally:
            await tr.rollback()


# ---------------------------------------------------------------- conscious actions
async def _pending_review(conn, mem_ids):
    return await conn.fetchval(
        "INSERT INTO memory_review_queue (memory_ids, reason, preview) VALUES ($1::uuid[], 'near', 'a preview') RETURNING id",
        mem_ids)


async def test_keep_memory_protects_and_spends(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("UPDATE heartbeat_state SET current_energy = 20 WHERE id = 1")
            await conn.execute("SELECT set_state('retention_veto_budget', $1::jsonb)",
                               json.dumps({"chapter": "unknown", "remaining": 2, "total": 5}))
            mid = await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic','worth keeping', {_DUMMY}, 0.5, 0.9, 'active') RETURNING id")
            rid = await _pending_review(conn, [mid])
            res = _j(await conn.fetchval("SELECT execute_heartbeat_action($1, 'keep_memory', $2::jsonb)",
                                         uuid4(), json.dumps({"review_id": str(rid)})))
            assert res["success"] is True
            assert await conn.fetchval("SELECT status FROM memory_review_queue WHERE id=$1", rid) == "kept"
            assert await conn.fetchval("SELECT is_memory_protected($1)", mid) is True
            assert await conn.fetchval("SELECT retention_budget_remaining()") == 1  # spent one
        finally:
            await tr.rollback()


async def test_keep_memory_refused_when_no_budget(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("UPDATE heartbeat_state SET current_energy = 20 WHERE id = 1")
            await conn.execute("SELECT set_state('retention_veto_budget', $1::jsonb)",
                               json.dumps({"chapter": "unknown", "remaining": 0, "total": 5}))
            mid = await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic','cannot keep', {_DUMMY}, 0.5, 0.9, 'active') RETURNING id")
            rid = await _pending_review(conn, [mid])
            res = _j(await conn.fetchval("SELECT execute_heartbeat_action($1, 'keep_memory', $2::jsonb)",
                                         uuid4(), json.dumps({"review_id": str(rid)})))
            assert res["result"]["kept"] is False
            assert res["result"]["reason"] == "no_budget"
            assert await conn.fetchval("SELECT status FROM memory_review_queue WHERE id=$1", rid) == "pending"
            assert await conn.fetchval("SELECT is_memory_protected($1)", mid) is False
        finally:
            await tr.rollback()


async def test_legacy_heartbeat_cannot_release_user_review(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("UPDATE heartbeat_state SET current_energy = 20 WHERE id = 1")
            ids = [await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic', $1, {_DUMMY}, 0.3, 0.9, 'active') RETURNING id", f"let go {k}") for k in range(3)]
            rid = await _pending_review(conn, ids)
            savepoint = conn.transaction()
            await savepoint.start()
            with pytest.raises(asyncpg.InsufficientPrivilegeError, match="explicit user decision"):
                await conn.fetchval(
                    "SELECT execute_heartbeat_action($1, 'release_memory', $2::jsonb)",
                    uuid4(),
                    json.dumps({"review_id": str(rid)}),
                )
            await savepoint.rollback()
            assert await conn.fetchval("SELECT status FROM memory_review_queue WHERE id=$1", rid) == "pending"
            assert await conn.fetchval(
                "SELECT count(*) FROM memories WHERE id = ANY($1::uuid[]) AND status='active'", ids) == 3
        finally:
            await tr.rollback()


async def test_legacy_heartbeat_cannot_journal_and_fade_user_review(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("UPDATE heartbeat_state SET current_energy = 20 WHERE id = 1")
            ids = [await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic', $1, {_DUMMY}, 0.3, 0.9, 'active') RETURNING id", f"journal {k}") for k in range(3)]
            rid = await _pending_review(conn, ids)
            before = await conn.fetchval("SELECT count(*) FROM journal_entries")
            savepoint = conn.transaction()
            await savepoint.start()
            with pytest.raises(asyncpg.InsufficientPrivilegeError, match="explicit user decision"):
                await conn.fetchval(
                    "SELECT execute_heartbeat_action($1, 'journal_memory', $2::jsonb)",
                    uuid4(),
                    json.dumps({"review_id": str(rid), "content": "I want to remember this."}),
                )
            await savepoint.rollback()
            assert await conn.fetchval("SELECT count(*) FROM journal_entries") == before
            assert await conn.fetchval("SELECT status FROM memory_review_queue WHERE id=$1", rid) == "pending"
            assert await conn.fetchval(
                "SELECT count(*) FROM memories WHERE id = ANY($1::uuid[]) AND status='active'", ids) == 3
        finally:
            await tr.rollback()


# ---------------------------------------------------------------- safe expiry
async def test_expired_review_waits_for_an_explicit_decision(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        tr = conn.transaction()
        await tr.start()
        try:
            await _enable(conn)
            ids = [await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic', $1, {_DUMMY}, 0.3, 0.9, 'active') RETURNING id", f"expire {k}") for k in range(3)]
            rid = await conn.fetchval(
                "INSERT INTO memory_review_queue (memory_ids, reason, preview, expires_at) "
                "VALUES ($1::uuid[], 'near', 'p', now() - interval '1 day') RETURNING id", ids)
            result = _j(await conn.fetchval("SELECT run_retention_gc()"))
            assert result["reviews_awaiting_decision"] >= 1
            assert await conn.fetchval("SELECT status FROM memory_review_queue WHERE id=$1", rid) == "pending"
            assert await conn.fetchval(
                "SELECT count(*) FROM memories WHERE id = ANY($1::uuid[]) AND status='active'", ids) == 3
        finally:
            await tr.rollback()


# ---------------------------------------------------------------- context slice
async def test_context_slice_surfaces_reviews(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SELECT set_state('retention_veto_budget', $1::jsonb)",
                               json.dumps({"chapter": "unknown", "remaining": 4, "total": 5}))
            mid = await conn.fetchval(
                f"INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                f"VALUES ('episodic','surfaced', {_DUMMY}, 0.5, 0.9, 'active') RETURNING id")
            await _pending_review(conn, [mid])
            slice_ = _j(await conn.fetchval("SELECT get_memories_at_threshold_context(5)"))
            assert slice_["budget_remaining"] == 4
            assert len(slice_["reviews"]) == 1
            assert slice_["reviews"][0]["preview"] == "a preview"
        finally:
            await tr.rollback()
