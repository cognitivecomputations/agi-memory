from __future__ import annotations

import json

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


async def _stub_get_embedding(conn):
    await conn.execute("""
        CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
        RETURNS vector[] AS $$
            SELECT COALESCE(
                array_agg((
                    ARRAY[1.0::float] ||
                    array_fill(0.0::float, ARRAY[embedding_dimension() - 1])
                )::vector),
                ARRAY[]::vector[]
            )
            FROM unnest(text_contents)
        $$ LANGUAGE sql;
        """)


async def test_user_model_claim_review_and_supersession(db_pool):
    marker = get_test_identifier("usermodelv2")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            first = _j(
                await conn.fetchval(
                    """
                SELECT upsert_user_model_claim(
                    $1, $2, 'preference', 0.7, 0.6,
                    NULL, NULL, '{}'::jsonb, '{"test": true}'::jsonb,
                    'pending_review'
                )
                """,
                    f"preference:{marker}:coffee",
                    f"User prefers coffee for focused work {marker}.",
                )
            )
            second = _j(
                await conn.fetchval(
                    """
                SELECT upsert_user_model_claim(
                    $1, $2, 'preference', 0.8, 0.7,
                    NULL, NULL, '{}'::jsonb, '{"test": true}'::jsonb,
                    'pending_review', $3, $4::jsonb
                )
                """,
                    f"preference:{marker}:tea",
                    f"User now prefers tea for focused work {marker}.",
                    f"preference:{marker}:coffee",
                    json.dumps([f"preference:{marker}:coffee"]),
                )
            )
            reviewed = _j(
                await conn.fetchval(
                    "SELECT review_user_model_claim($1::uuid, 'approve', 'looks right', 'test')",
                    second["claim_id"],
                )
            )
            listed = _j(
                await conn.fetchval(
                    "SELECT list_user_model_claims(NULL, 'approved', 'preference', 20, 0)"
                )
            )
            old_status = await conn.fetchrow(
                "SELECT status, review_status, superseded_by::text FROM user_model_claims WHERE id = $1::uuid",
                first["claim_id"],
            )
        finally:
            await tr.rollback()

    assert reviewed["review_status"] == "approved"
    assert any(item["id"] == second["claim_id"] for item in listed["claims"])
    assert old_status["status"] == "superseded"
    assert old_status["review_status"] == "superseded"
    assert old_status["superseded_by"] == second["claim_id"]


async def test_approved_user_model_claims_render_into_prompt_context(db_pool):
    marker = get_test_identifier("usermodelcontext")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            approved = _j(
                await conn.fetchval(
                    """
                SELECT upsert_user_model_claim(
                    $1, $2, 'routine', 0.84, 0.75,
                    NULL, NULL, '[{"ref": "test"}]'::jsonb, '{"test": true}'::jsonb,
                    'pending_review'
                )
                """,
                    f"routine:{marker}:planning",
                    f"User does careful Monday planning {marker}.",
                )
            )
            pending = _j(
                await conn.fetchval(
                    """
                SELECT upsert_user_model_claim(
                    $1, $2, 'preference', 0.7, 0.5,
                    NULL, NULL, '[{"ref": "test"}]'::jsonb, '{"test": true}'::jsonb,
                    'pending_review'
                )
                """,
                    f"preference:{marker}:unapproved",
                    f"User supposedly likes unreviewed filler {marker}.",
                )
            )
            await conn.fetchval(
                "SELECT review_user_model_claim($1::uuid, 'approve', 'context test', 'test')",
                approved["claim_id"],
            )
            context = _j(
                await conn.fetchval("SELECT get_approved_user_model_context(10)")
            )
            rendered = await conn.fetchval(
                "SELECT render_user_model_context($1::jsonb)", json.dumps(context)
            )
        finally:
            await tr.rollback()

    assert any(item["id"] == approved["claim_id"] for item in context)
    assert all(item["id"] != pending["claim_id"] for item in context)
    assert "## User Model" in rendered
    assert f"careful Monday planning {marker}" in rendered
    assert "unreviewed filler" not in rendered


async def test_graph_reconcile_paths_and_reward_rpe(db_pool):
    marker = get_test_identifier("graphreward")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            a = await conn.fetchval(
                "SELECT create_semantic_memory($1, 0.8, ARRAY['test'], NULL, '[]'::jsonb, 0.7)",
                f"Cause memory {marker}",
            )
            b = await conn.fetchval(
                "SELECT create_semantic_memory($1, 0.8, ARRAY['test'], NULL, '[]'::jsonb, 0.7)",
                f"Effect memory {marker}",
            )
            await conn.execute(
                "SELECT upsert_memory_edge($1::uuid, $2::uuid, 'CAUSES'::graph_edge_type, '{\"strength\": 0.9}'::jsonb)",
                a,
                b,
            )
            paths = _j(
                await conn.fetchval(
                    "SELECT memory_graph_paths($1::uuid, ARRAY['CAUSES'], 2, 10)",
                    a,
                )
            )
            subgraph = _j(
                await conn.fetchval(
                    "SELECT build_context_subgraph(ARRAY[$1::uuid], 2, ARRAY['CAUSES'], 40)",
                    a,
                )
            )
            context_paths = _j(
                await conn.fetchval(
                    "SELECT memory_context_paths(ARRAY[$1::uuid], 2)",
                    a,
                )
            )
            rendered = await conn.fetchval(
                "SELECT render_chat_memory_context($1::jsonb)",
                json.dumps({"subgraph": subgraph, "context_paths": context_paths}),
            )
            reconcile = _j(await conn.fetchval("SELECT reconcile_graph(false)"))
            rpe = _j(
                await conn.fetchval(
                    "SELECT record_prediction_error(0.1, 0.8, 'test_rpe', 'test', $1::jsonb)",
                    json.dumps({"summary": marker}),
                )
            )
            reward = _j(
                await conn.fetchval("SELECT reward_state_summary(INTERVAL '1 hour')")
            )
        finally:
            await tr.rollback()

    assert paths["paths"]
    assert paths["paths"][0]["edges"][0]["rel"] == "CAUSES"
    assert "## Causal/Contradiction Paths" in rendered
    assert f"Cause memory {marker}" in rendered
    assert "causes" in rendered
    assert reconcile["status"] in {"ok", "needs_repair"}
    assert rpe["dopamine_triggered"] is True
    assert reward["events"] >= 1


async def test_reward_hooks_record_drive_resource_backup_and_social_events(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "UPDATE drives SET current_level = 0.9, baseline = 0.3, last_satisfied = NULL WHERE name = 'curiosity'"
            )
            await conn.execute("SELECT satisfy_drive('curiosity', 0.4)")
            drive_count = await conn.fetchval(
                "SELECT COUNT(*) FROM reward_events WHERE kind = 'drive_satisfied:curiosity'"
            )

            filed = _j(
                await conn.fetchval(
                    "SELECT file_resource_request('energy_boost', 'need focus for a long task', NULL, '2'::jsonb, NULL)"
                )
            )
            decided = _j(
                await conn.fetchval(
                    "SELECT decide_resource_request($1::uuid, 'granted', 'reasonable', NULL)",
                    filed["request_id"],
                )
            )
            resource_count = await conn.fetchval(
                "SELECT COUNT(*) FROM reward_events WHERE kind = 'resource_request_granted:energy_boost'"
            )

            await conn.execute(
                "UPDATE drives SET current_level = 0.8, baseline = 0.3, last_satisfied = NULL WHERE name = 'continuity'"
            )
            backup = _j(
                await conn.fetchval(
                    "SELECT record_backup_completed('hook-test', '/tmp/hook.dump')"
                )
            )
            backup_count = await conn.fetchval(
                "SELECT COUNT(*) FROM reward_events WHERE kind = 'backup_completed'"
            )

            social = _j(
                await conn.fetchval(
                    """
                SELECT apply_appraisal_reward_effects($1::jsonb)
                """,
                    json.dumps(
                        {
                            "emotional_state": {
                                "primary_emotion": "appreciative glow",
                                "family": "reward",
                                "valence": 0.7,
                                "arousal": 0.45,
                                "intensity": 0.6,
                                "confidence": 0.8,
                            },
                            "subconscious_response": "That felt warmly appreciated.",
                        }
                    ),
                )
            )
            social_count = await conn.fetchval(
                "SELECT COUNT(*) FROM reward_events WHERE kind = 'social:appreciative glow'"
            )
            rejected = _j(
                await conn.fetchval(
                    "SELECT apply_appraisal_reward_effects($1::jsonb)",
                    json.dumps(
                        {
                            "emotional_state": {
                                "primary_emotion": "gratitude",
                                "family": "loss",
                                "valence": 0.7,
                                "intensity": 0.6,
                                "confidence": 0.8,
                            }
                        }
                    ),
                )
            )
        finally:
            await tr.rollback()

    assert drive_count >= 1
    assert decided["applied"] == "energy"
    assert resource_count >= 1
    assert backup["recorded"] is True
    assert backup_count >= 1
    assert social["recorded"] is True
    assert social_count >= 1
    assert rejected["recorded"] is False
