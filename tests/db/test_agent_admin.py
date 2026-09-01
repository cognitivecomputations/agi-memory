"""Tranche 2 pushdown (plans/db_pushdown.md): agent-admin functions own what
were Python sagas — atomic config apply, composite status, conditional energy
debit, mood ladder, set-based contact upserts, and the tool-audit insert.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


async def test_apply_agent_config_is_atomic_and_complete(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            config = {
                "heartbeat_interval_minutes": 45,
                "maintenance_interval_seconds": 90,
                "subconscious_interval_seconds": None,
                "max_energy": 25.0,
                "base_regeneration": 12.0,
                "max_active_goals": 4,
                "objectives": ["stay curious"],
                "guardrails": ["be kind"],
                "initial_message": "hello",
                "tools": ["recall", "remember"],
                "llm_heartbeat": {"provider": "test", "model": "hb"},
                "llm_chat": {"provider": "test", "model": "chat"},
                "llm_subconscious": None,
                "contact_channels": ["email"],
                "contact_destinations": {"email": "e@example.com"},
                "enable_autonomy": True,
                "enable_maintenance": False,
                "enable_subconscious": None,
                "mark_configured": True,
            }
            await conn.execute(
                "SELECT apply_agent_config($1::jsonb)", json.dumps(config)
            )

            assert (
                await conn.fetchval("SELECT get_config_float('heartbeat.max_energy')")
                == 25.0
            )
            budget = _json(await conn.fetchval("SELECT get_config('agent.budget')"))
            assert budget["heartbeat_interval_minutes"] == 45
            tools = _json(await conn.fetchval("SELECT get_config('agent.tools')"))
            assert tools == [
                {"name": "recall", "enabled": True},
                {"name": "remember", "enabled": True},
            ]
            # llm_subconscious None falls back to llm_heartbeat.
            sub = _json(await conn.fetchval("SELECT get_config('llm.subconscious')"))
            assert sub["model"] == "hb"
            assert (
                await conn.fetchval("SELECT is_paused FROM heartbeat_state WHERE id=1")
                is False
            )
            assert (
                await conn.fetchval("SELECT get_config_bool('agent.is_configured')")
                is True
            )
        finally:
            await tr.rollback()


async def test_apply_agent_config_uses_config_defaults_for_missing_budget(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            for key in (
                "heartbeat.heartbeat_interval_minutes",
                "heartbeat.max_energy",
                "heartbeat.base_regeneration",
                "heartbeat.max_active_goals",
                "maintenance.maintenance_interval_seconds",
            ):
                await conn.execute("SELECT delete_config_key($1)", key)

            await conn.execute(
                "SELECT apply_agent_config($1::jsonb)",
                json.dumps(
                    {
                        "objectives": [],
                        "guardrails": [],
                        "tools": [],
                        "enable_autonomy": False,
                        "enable_maintenance": False,
                    }
                ),
            )

            budget = _json(await conn.fetchval("SELECT get_config('agent.budget')"))
            assert budget == {
                "max_energy": 20,
                "base_regeneration": 10,
                "heartbeat_interval_minutes": 60,
                "max_active_goals": 3,
            }
            assert (
                await conn.fetchval(
                    "SELECT get_config_float('maintenance.maintenance_interval_seconds')"
                )
                == 60.0
            )
        finally:
            await tr.rollback()


async def test_get_agent_status_and_policy(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # Control preconditions: flag set, but no consent contract.
            await conn.execute(
                "SELECT set_config('agent.is_configured', 'true'::jsonb)"
            )
            await conn.execute("SELECT delete_config_key('agent.consent_log_id')")
            await conn.execute("SELECT delete_config_key('agent.consent_status')")
            await conn.execute("SELECT set_config('llm.heartbeat', 'null'::jsonb)")
            status = _json(await conn.fetchval("SELECT get_agent_status()"))
            # Flag alone is not enough: configured needs consent contract + decision.
            assert status["configured"] is False
            assert status["terminated"] is False

            await conn.execute(
                "SELECT set_config('agent.consent_log_id', '\"cl-1\"'::jsonb)"
            )
            await conn.execute(
                "SELECT set_config('agent.consent_status', '\"consent\"'::jsonb)"
            )
            status = _json(await conn.fetchval("SELECT get_agent_status()"))
            assert status["configured"] is True
        finally:
            await tr.rollback()


async def test_spend_energy_is_conditional(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "UPDATE heartbeat_state SET current_energy = 5 WHERE id = 1"
            )
            assert await conn.fetchval("SELECT spend_energy(3.0)") is True
            assert (
                await conn.fetchval(
                    "SELECT current_energy FROM heartbeat_state WHERE id=1"
                )
                == 2
            )
            assert await conn.fetchval("SELECT spend_energy(3.0)") is False
            assert (
                await conn.fetchval(
                    "SELECT current_energy FROM heartbeat_state WHERE id=1"
                )
                == 2
            )
        finally:
            await tr.rollback()


async def test_mood_label_ladder(db_pool):
    async with db_pool.acquire() as conn:
        cases = [
            (0.7, 0.7, "enthusiastic"),
            (0.7, 0.2, "content"),
            (0.3, 0.5, "curious"),
            (0.3, 0.1, "calm"),
            (0.0, 0.5, "focused"),
            (0.0, 0.1, "neutral"),
            (-0.3, 0.5, "concerned"),
            (-0.3, 0.1, "subdued"),
            (-0.8, 0.7, "distressed"),
            (-0.8, 0.2, "withdrawn"),
        ]
        for valence, arousal, expected in cases:
            got = await conn.fetchval("SELECT mood_label($1, $2)", valence, arousal)
            assert got == expected, (valence, arousal)


async def test_contact_upserts(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            first = _json(
                await conn.fetchval(
                    "SELECT upsert_contact('Ada', 'ada@example.com', 'email')"
                )
            )
            assert first["created"] is True
            again = _json(
                await conn.fetchval(
                    "SELECT upsert_contact('Ada L', 'ada@example.com', 'email')"
                )
            )
            assert again["created"] is False and again["id"] == first["id"]

            bad = _json(
                await conn.fetchval("SELECT upsert_contact('X', 'not-an-email')")
            )
            assert bad.get("skipped") is True

            batch = _json(await conn.fetchval("""SELECT upsert_contacts_from_attendees(
                    '[{"email": "bob@example.com", "displayName": "Bob"},
                      {"email": "ada@example.com"},
                      "carol@example.com",
                      {"email": "nope"}]'::jsonb, 'calendar')"""))
            assert batch == {"created": 2, "updated": 1}

            by_name = _json(
                await conn.fetchval("SELECT upsert_contact_by_name('Bob', 'fathom')")
            )
            assert by_name["created"] is False  # matched the existing Bob
        finally:
            await tr.rollback()


async def test_record_tool_execution(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            rec_id = await conn.fetchval("""
                SELECT record_tool_execution('{
                    "tool_name": "probe", "arguments": {"a": 1},
                    "tool_context": "chat", "call_id": "c-1",
                    "success": true, "energy_spent": 2.0,
                    "duration_seconds": 0.5
                }'::jsonb)
                """)
            row = await conn.fetchrow(
                "SELECT tool_name, success, energy_spent FROM tool_executions WHERE id = $1",
                rec_id,
            )
            assert row["tool_name"] == "probe"
            assert row["success"] is True
            assert row["energy_spent"] == 2
        finally:
            await tr.rollback()


async def test_apply_channel_config_writes_catalog_keys(db_pool):
    """The channel setting catalog is DB policy: known settings write
    channel.<type>.<name> config keys atomically; unknown ones fail loud."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            result = await conn.fetchval(
                "SELECT apply_channel_config('signal', $1::jsonb)",
                '{"phone_number": "SIGNAL_PHONE_NUMBER", '
                '"api_url": "http://localhost:8080", '
                '"operator_user_id": "+15551234567", "allowed_numbers": "*"}',
            )
            payload = json.loads(result)
            assert payload["channel"] == "signal"
            assert sorted(payload["applied"]) == [
                "allowed_numbers",
                "api_url",
                "operator_user_id",
                "phone_number",
            ]
            assert (
                await conn.fetchval("SELECT get_config_text('channel.signal.api_url')")
                == "http://localhost:8080"
            )
            assert (
                await conn.fetchval(
                    "SELECT get_config_text('channel.signal.operator_user_id')"
                )
                == "+15551234567"
            )
        finally:
            await tr.rollback()


async def test_apply_channel_config_rejects_unknowns(db_pool):
    async with db_pool.acquire() as conn:
        with pytest.raises(Exception, match="Unknown channel type: carrier_pigeon"):
            await conn.fetchval(
                "SELECT apply_channel_config('carrier_pigeon', '{\"a\": \"b\"}'::jsonb)"
            )
        with pytest.raises(Exception, match="Unknown discord setting"):
            await conn.fetchval(
                "SELECT apply_channel_config('discord', '{\"webhook\": \"x\"}'::jsonb)"
            )


async def test_worldview_trust_derives_from_source(db_pool):
    """#83 sources-are-authority: a worldview row may never assert more trust
    than its provenance carries; sync_memory_trust converges it."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            row_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding, importance, trust_level, status, source_attribution)
                VALUES ('worldview', 'Trust-authority pin belief',
                        array_fill(0.1, ARRAY[embedding_dimension()])::vector,
                        0.8, 0.97, 'active',
                        '{"kind": "internal", "trust": 0.5}'::jsonb)
                RETURNING id
                """)
            await conn.execute("SELECT sync_memory_trust($1)", row_id)
            converged = await conn.fetchval(
                "SELECT trust_level FROM memories WHERE id = $1", row_id
            )
        finally:
            await tr.rollback()

    assert abs(converged - 0.5) < 1e-9
