from __future__ import annotations

import json
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_heartbeat_action_failures_create_diagnosable_defect_reports(db_pool):
    heartbeat_id = str(uuid4())
    actions = [
        {
            "action": "get_strategies",
            "params": {"query": "recover from heartbeat failure"},
            "result": {
                "success": False,
                "error": "Validation errors: Missing required field: situation",
                "energy_spent": 1,
            },
        }
    ]

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            first_ids = _j(await conn.fetchval(
                "SELECT record_heartbeat_action_defects($1::uuid, $2::jsonb, $3)",
                heartbeat_id,
                json.dumps(actions),
                "I tried a malformed get_strategies call.",
            ))
            second_ids = _j(await conn.fetchval(
                "SELECT record_heartbeat_action_defects($1::uuid, $2::jsonb, $3)",
                heartbeat_id,
                json.dumps(actions),
                "I tried a malformed get_strategies call again.",
            ))
            assert first_ids == second_ids
            defect_id = first_ids[0]

            row = await conn.fetchrow(
                """
                SELECT category, severity, occurrence_count, tool_names, last_error
                FROM defect_reports
                WHERE id = $1::uuid
                """,
                defect_id,
            )
            diagnosis = _j(await conn.fetchval(
                "SELECT diagnose_defect_report($1::uuid)",
                defect_id,
            ))
            context = await conn.fetchval(
                "SELECT render_defect_reports_context(5)"
            )
        finally:
            await tr.rollback()

    assert row["category"] == "tool_contract"
    assert row["severity"] == "medium"
    assert row["occurrence_count"] == 2
    assert "get_strategies" in row["tool_names"]
    assert "Missing required field" in row["last_error"]
    assert diagnosis["success"] is True
    assert diagnosis["proposed_repair"]["mode"] == "proposal_only"
    assert "core/tools/memory.py" in diagnosis["diagnosis"]["likely_files"]
    assert "Tool/action contract failure" in context


async def test_chat_continuity_surfaces_unresolved_defects(db_pool):
    marker = uuid4().hex
    heartbeat_id = str(uuid4())

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.fetchval(
                """
                SELECT record_defect_event(
                    'heartbeat',
                    'embedding',
                    $1,
                    $2::jsonb
                )
                """,
                f"Embedding service not reachable for marker {marker}",
                json.dumps({"heartbeat_id": heartbeat_id, "tool_name": "embedding"}),
            )
            continuity = await conn.fetchval(
                "SELECT render_chat_continuity_context($1::text, false)",
                str(uuid4()),
            )
            excluded = await conn.fetchval(
                "SELECT render_chat_continuity_context($1::text, true)",
                str(uuid4()),
            )
        finally:
            await tr.rollback()

    assert "### Unresolved Software Defects" in continuity
    assert marker in continuity
    assert "operational responsibilities" in continuity
    assert excluded == ""
