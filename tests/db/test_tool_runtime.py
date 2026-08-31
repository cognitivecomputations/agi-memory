from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _coerce_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


async def test_agent_tools_default_names_are_all_registered(db_pool):
    """#108: agent.tools (rendered into the heartbeat's advertised "Tools:"
    block) named 9 tool names that were never registered -- recall_recent,
    recall_episode, explore_cluster, list_recent_episodes, create_goal,
    schedule_task, list_scheduled_tasks, update_scheduled_task,
    delete_scheduled_task -- causing live "Unknown tool" heartbeat failures.
    Every name in the default must resolve to a real handler."""
    from core.tools.registry import create_default_registry

    async with db_pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT value FROM config_defaults WHERE key = 'agent.tools'"
        )
    names = _coerce_json(raw)
    assert isinstance(names, list) and names

    registry = create_default_registry(pool=None)
    registered = set(registry.list_names())
    unregistered = [n for n in names if n not in registered]
    assert not unregistered, (
        f"agent.tools default advertises unregistered tool(s): {unregistered}"
    )


async def test_tool_catalog_specs_and_policy_are_db_owned(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SELECT set_config('tools', '{}'::jsonb)")
            synced = await conn.fetchval(
                "SELECT sync_tool_definitions($1::jsonb)",
                json.dumps([
                    {
                        "name": "db_echo",
                        "description": "Echo from DB catalog",
                        "schema": {"type": "object", "properties": {"message": {"type": "string"}}},
                        "category": "external",
                        "energy_cost": 2,
                        "allowed_contexts": ["chat", "heartbeat"],
                        "supports_parallel": True,
                    }
                ]),
            )
            specs = _coerce_json(await conn.fetchval("SELECT get_tool_specs_for_context('chat')"))
            decision = _coerce_json(
                await conn.fetchval(
                    "SELECT evaluate_tool_call('db_echo', '{}'::jsonb, $1::jsonb)",
                    json.dumps({"tool_context": "heartbeat", "energy_available": 3}),
                )
            )
            denied = _coerce_json(
                await conn.fetchval(
                    "SELECT evaluate_tool_call('db_echo', '{}'::jsonb, $1::jsonb)",
                    json.dumps({"tool_context": "heartbeat", "energy_available": 1}),
                )
            )

            assert _coerce_json(synced)["synced"] == 1
            assert any(item["function"]["name"] == "db_echo" for item in specs)
            assert decision["allowed"] is True
            assert decision["energy_cost"] == 2
            assert denied["allowed"] is False
            assert denied["error_type"] == "insufficient_energy"
        finally:
            await tr.rollback()


async def test_stale_tool_definitions_flags_old_unsynced_rows(db_pool):
    """#115: sync_tool_definitions only upserts, so a renamed/removed tool
    would advertise forever with nothing noticing. stale_tool_definitions()
    is the read-only, never-auto-deleting advisory hexis doctor surfaces
    instead."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "SELECT upsert_tool_definition('stale_check_fresh', 'external')"
            )
            await conn.execute(
                "SELECT upsert_tool_definition('stale_check_old', 'external')"
            )
            await conn.execute(
                "UPDATE tool_definitions SET updated_at = CURRENT_TIMESTAMP - INTERVAL '30 days' "
                "WHERE name = 'stale_check_old'"
            )

            stale = _coerce_json(await conn.fetchval("SELECT stale_tool_definitions()"))
            names = {item["name"]: item for item in stale}

            assert "stale_check_fresh" not in names
            assert "stale_check_old" in names
            assert names["stale_check_old"]["age_days"] >= 29
        finally:
            await tr.rollback()


async def test_stale_tool_definitions_respects_config_threshold(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "SELECT upsert_tool_definition('stale_check_recent', 'external')"
            )
            await conn.execute(
                "UPDATE tool_definitions SET updated_at = CURRENT_TIMESTAMP - INTERVAL '2 days' "
                "WHERE name = 'stale_check_recent'"
            )

            # Under the default 14-day threshold, 2 days old is not stale.
            default_stale = _coerce_json(await conn.fetchval("SELECT stale_tool_definitions()"))
            assert "stale_check_recent" not in {i["name"] for i in default_stale}

            await conn.execute(
                "INSERT INTO config (key, value) VALUES ('tools.definition_stale_days', '1'::jsonb) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
            )
            tightened = _coerce_json(await conn.fetchval("SELECT stale_tool_definitions()"))
            assert "stale_check_recent" in {i["name"] for i in tightened}
        finally:
            await tr.rollback()


async def test_db_schedule_parser_and_manage_schedule_tool(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            parsed = _coerce_json(
                await conn.fetchval(
                    "SELECT parse_schedule_input($1::jsonb)",
                    json.dumps({"schedule": "every:5m", "timezone": "UTC"}),
                )
            )
            created = _coerce_json(
                await conn.fetchval(
                    "SELECT manage_schedule_tool($1::jsonb)",
                    json.dumps({
                        "action": "create",
                        "name": "db schedule",
                        "schedule": "once:+2h",
                        "message": "wake up",
                    }),
                )
            )
            listed = _coerce_json(await conn.fetchval("SELECT manage_schedule_tool('{\"action\":\"list\"}'::jsonb)"))

            assert parsed["schedule_kind"] == "interval"
            assert parsed["schedule"]["every_minutes"] == 5
            assert created["success"] is True
            assert created["output"]["schedule_kind"] == "once"
            assert listed["success"] is True
            assert listed["output"]["count"] >= 1
        finally:
            await tr.rollback()


async def test_workflow_runtime_functions_create_claim_finalize(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            created = _coerce_json(
                await conn.fetchval(
                    "SELECT create_workflow_execution($1::jsonb, $2::jsonb)",
                    json.dumps({
                        "name": "db workflow",
                        "steps": [
                            {"name": "a", "tool": "db_echo", "arguments": {"x": 1}},
                            {"name": "b", "tool": "db_echo", "arguments": {"y": 2}, "depends_on": ["a"]},
                        ],
                    }),
                    json.dumps({"session_id": "session-test"}),
                )
            )
            claimed = _coerce_json(
                await conn.fetchval("SELECT claim_workflow_steps($1::uuid)", created["workflow_id"])
            )
            assert len(claimed) == 1
            assert claimed[0]["step_name"] == "a"

            applied = _coerce_json(
                await conn.fetchval(
                    "SELECT apply_workflow_step_result($1::uuid, $2::jsonb)",
                    claimed[0]["id"],
                    json.dumps({"success": True, "output": {"energy_spent": 1}}),
                )
            )
            assert applied["status"] == "completed"

            claimed_next = _coerce_json(
                await conn.fetchval("SELECT claim_workflow_steps($1::uuid)", created["workflow_id"])
            )
            assert len(claimed_next) == 1
            assert claimed_next[0]["step_name"] == "b"
        finally:
            await tr.rollback()
