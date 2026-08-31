import json
import uuid

import pytest

from tests.utils import _coerce_json, get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


async def test_apply_brainstormed_goals_creates_goals(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            goals = [
                {
                    "title": f"Goal A {get_test_identifier('brainstorm')}",
                    "priority": "queued",
                    "source": "user_request",
                },
                {
                    "title": f"Goal B {get_test_identifier('brainstorm')}",
                    "priority": "queued",
                    "source": "curiosity",
                },
            ]
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT apply_brainstormed_goals($1::uuid, $2::jsonb)",
                    str(uuid.uuid4()),
                    json.dumps(goals),
                )
            )
            created_ids = result["created_goal_ids"]
            assert len(created_ids) == 2

            titles = [goal["title"] for goal in goals]
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM memories
                WHERE type = 'goal'
                  AND metadata->>'title' = ANY($1::text[])
                """,
                titles,
            )
            assert int(count) == 2
            origins = await conn.fetch(
                """
                SELECT goal_origin::text AS origin
                FROM memories
                WHERE type = 'goal'
                  AND metadata->>'title' = ANY($1::text[])
                """,
                titles,
            )
            assert {row["origin"] for row in origins} == {"derived"}
        finally:
            await tr.rollback()


async def test_apply_inquiry_result_creates_semantic_memory(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            summary = f"Inquiry summary {get_test_identifier('inquiry')}"
            payload = {
                "summary": summary,
                "confidence": 0.7,
                "sources": [],
                "depth": "inquire_shallow",
                "query": "test query",
            }
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT apply_inquiry_result($1::uuid, $2::jsonb)",
                    str(uuid.uuid4()),
                    json.dumps(payload),
                )
            )
            mem_id = result.get("memory_id")
            assert mem_id is not None

            row = await conn.fetchrow(
                "SELECT content, type FROM memories WHERE id = $1::uuid",
                mem_id,
            )
            assert row["type"] == "semantic"
            assert row["content"] == summary
        finally:
            await tr.rollback()


async def test_apply_goal_changes_updates_priority(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            goal_id = await conn.fetchval(
                "SELECT create_goal($1, $2, $3, $4, NULL, NULL)",
                f"Complete {get_test_identifier('goal_change')}",
                "desc",
                "curiosity",
                "queued",
            )
            changes = [
                {"goal_id": str(goal_id), "change": "completed", "reason": "done"}
            ]
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT apply_goal_changes($1::jsonb)",
                    json.dumps(changes),
                )
            )
            assert result["applied"] == 1

            row = await conn.fetchrow(
                "SELECT status, metadata->>'priority' as priority FROM memories WHERE id = $1::uuid",
                goal_id,
            )
            assert row["priority"] == "completed"
            assert row["status"] == "archived"
        finally:
            await tr.rollback()


async def test_apply_goal_changes_normalizes_new_priority_alias(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            goal_id = await conn.fetchval(
                "SELECT create_goal($1, $2, $3, $4, NULL, NULL)",
                f"Prioritize {get_test_identifier('goal_alias')}",
                "desc",
                "curiosity",
                "queued",
            )
            changes = [
                {
                    "goal_id": str(goal_id),
                    "new_priority": "high",
                    "reason": "urgent audit",
                }
            ]
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT apply_goal_changes($1::jsonb)",
                    json.dumps(changes),
                )
            )

            assert result["applied"] == 1
            assert result["skipped"] == 0

            row = await conn.fetchrow(
                "SELECT status, metadata->>'priority' as priority FROM memories WHERE id = $1::uuid",
                goal_id,
            )
            assert row["priority"] == "active"
            assert row["status"] == "active"
        finally:
            await tr.rollback()


async def test_apply_goal_changes_skips_invalid_priority_without_crashing(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            goal_id = await conn.fetchval(
                "SELECT create_goal($1, $2, $3, $4, NULL, NULL)",
                f"Invalid priority {get_test_identifier('goal_invalid')}",
                "desc",
                "curiosity",
                "queued",
            )
            changes = [
                {
                    "goal_id": str(goal_id),
                    "new_priority": "impossibly-important",
                    "reason": "bad label",
                }
            ]
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT apply_goal_changes($1::jsonb)",
                    json.dumps(changes),
                )
            )

            assert result["applied"] == 0
            assert result["skipped"] == 1

            priority = await conn.fetchval(
                "SELECT metadata->>'priority' FROM memories WHERE id = $1::uuid",
                goal_id,
            )
            assert priority == "queued"
        finally:
            await tr.rollback()


async def test_execute_heartbeat_actions_batch_halts_on_external_call(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "UPDATE heartbeat_state SET current_energy = 20, is_paused = FALSE WHERE id = 1"
            )
            await conn.execute(
                "SELECT set_config('heartbeat.allowed_actions', $1::jsonb)",
                json.dumps(["brainstorm_goals", "rest"]),
            )

            hb_payload = _coerce_json(await conn.fetchval("SELECT start_heartbeat()"))
            hb_id = hb_payload.get("heartbeat_id")
            assert hb_id is not None

            actions = [{"action": "brainstorm_goals", "params": {}}]
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT execute_heartbeat_actions_batch($1::uuid, $2::jsonb, 0)",
                    hb_id,
                    json.dumps(actions),
                )
            )
            assert result["halt_reason"] == "external_call"
            assert result.get("pending_external_call")
        finally:
            await tr.rollback()


async def test_apply_heartbeat_decision_completes(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "UPDATE heartbeat_state SET current_energy = 20, is_paused = FALSE WHERE id = 1"
            )
            await conn.execute(
                "SELECT set_config('heartbeat.allowed_actions', $1::jsonb)",
                json.dumps(["rest"]),
            )

            hb_payload = _coerce_json(await conn.fetchval("SELECT start_heartbeat()"))
            hb_id = hb_payload.get("heartbeat_id")
            assert hb_id is not None

            decision = {
                "actions": [{"action": "rest", "params": {}}],
                "reasoning": "cooldown",
                "goal_changes": [],
            }
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT apply_heartbeat_decision($1::uuid, $2::jsonb, 0)",
                    hb_id,
                    json.dumps(decision),
                )
            )
            assert result["completed"] is True
            assert result.get("memory_id")

            actions_taken = result.get("actions_taken", [])
            assert any(action.get("action") == "rest" for action in actions_taken)
        finally:
            await tr.rollback()
