from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _configure_economy(conn) -> None:
    for key, value in (
        ("heartbeat.max_energy", 20),
        ("heartbeat.base_regeneration", 10),
        ("heartbeat.energy_bank_multiplier", 3),
        ("heartbeat.energy_surplus_half_life_hours", 12),
        ("heartbeat.outcome_regen_floor_multiplier", 0.75),
        ("heartbeat.outcome_regen_score_scale", 0.5),
        ("heartbeat.outcome_regen_ceiling_multiplier", 1.5),
        ("heartbeat.heartbeat_interval_minutes", 60),
        ("heartbeat.cadence_min_minutes", 15),
        ("heartbeat.cadence_max_minutes", 120),
        ("heartbeat.cadence_idle_multiplier", 1.5),
        ("heartbeat.cadence_urgency_slope", 0.75),
    ):
        await conn.execute(
            "SELECT set_config($1, $2::jsonb)", key, json.dumps(value)
        )
    await conn.execute(
        """
        UPDATE heartbeat_state
        SET current_energy = 20,
            active_heartbeat_id = NULL,
            active_heartbeat_number = NULL,
            active_actions = '[]'::jsonb,
            active_reasoning = NULL,
            next_heartbeat_at = NULL,
            is_paused = FALSE
        WHERE id = 1;
        UPDATE heartbeat_economy_state
        SET last_regenerated_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = 1;
        DELETE FROM heartbeat_outcomes;
        """
    )


async def test_energy_banks_above_reserve_and_surplus_decays(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _configure_economy(conn)
            assert await conn.fetchval("SELECT heartbeat_bank_capacity()") == 60

            await conn.execute(
                """
                SELECT set_config('heartbeat.base_regeneration', '0'::jsonb);
                UPDATE heartbeat_state SET current_energy = 40 WHERE id = 1;
                UPDATE heartbeat_economy_state
                SET last_regenerated_at = CURRENT_TIMESTAMP - INTERVAL '12 hours'
                WHERE id = 1;
                """
            )
            decayed = _json(
                await conn.fetchval(
                    "SELECT regenerate_heartbeat_energy(CURRENT_TIMESTAMP)"
                )
            )
            assert decayed["after_energy"] == pytest.approx(30.0)
            assert decayed["surplus_decayed"] == pytest.approx(10.0)

            await conn.execute(
                """
                SELECT set_config('heartbeat.base_regeneration', '10'::jsonb);
                UPDATE heartbeat_state SET current_energy = 20 WHERE id = 1;
                UPDATE heartbeat_economy_state
                SET last_regenerated_at = CURRENT_TIMESTAMP - INTERVAL '4 hours'
                WHERE id = 1;
                """
            )
            banked = _json(
                await conn.fetchval(
                    "SELECT regenerate_heartbeat_energy(CURRENT_TIMESTAMP)"
                )
            )
            assert banked["after_energy"] == pytest.approx(60.0)
            assert banked["after_energy"] > banked["reserve_energy"]
        finally:
            await tr.rollback()


async def test_useful_outcomes_raise_regeneration_and_spend_is_once(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _configure_economy(conn)
            started = _json(await conn.fetchval("SELECT start_heartbeat()"))
            heartbeat_id = started["heartbeat_id"]
            energy_after_regen = await conn.fetchval(
                "SELECT energy_after_regen FROM heartbeat_outcomes WHERE heartbeat_id=$1",
                heartbeat_id,
            )
            assert energy_after_regen is not None, started["external_calls"][0][
                "input"
            ]["context"].get("heartbeat_economy")
            await conn.fetchval(
                "SELECT record_heartbeat_outcome_signal($1, 'durable_memory', 'memory:1')",
                heartbeat_id,
            )
            await conn.fetchval(
                "SELECT record_heartbeat_outcome_signal($1, 'contradiction_resolved', 'contradiction:1')",
                heartbeat_id,
            )
            await conn.fetchval(
                "SELECT record_heartbeat_outcome_signal($1, 'goal_advanced', 'goal:1')",
                heartbeat_id,
            )
            finalized = _json(
                await conn.fetchval(
                    "SELECT finalize_heartbeat_economy($1, 3.5, 'completed')",
                    heartbeat_id,
                )
            )
            assert finalized["applied"] is True, finalized
            assert finalized["applied"] is True
            assert finalized["outcome_tier"] == "high_value"
            assert finalized["outcome_score"] == pytest.approx(1.15)
            assert await conn.fetchval(
                "SELECT current_energy FROM heartbeat_state WHERE id=1"
            ) == pytest.approx(float(energy_after_regen) - 3.5)
            assert await conn.fetchval(
                "SELECT heartbeat_outcome_regen_multiplier()"
            ) == pytest.approx(1.325)

            repeated = _json(
                await conn.fetchval(
                    "SELECT finalize_heartbeat_economy($1, 3.5, 'completed')",
                    heartbeat_id,
                )
            )
            assert repeated["applied"] is False
            assert repeated["reason"] == "already_finalized"
            assert await conn.fetchval(
                "SELECT current_energy FROM heartbeat_state WHERE id=1"
            ) == pytest.approx(float(energy_after_regen) - 3.5)

            await conn.execute(
                """
                INSERT INTO heartbeat_outcomes (
                    heartbeat_id, heartbeat_number, status, completed_at,
                    outcome_score
                ) VALUES ($1, 999, 'error', CURRENT_TIMESTAMP + INTERVAL '1 second', 0)
                """,
                uuid4(),
            )
            assert await conn.fetchval(
                "SELECT heartbeat_outcome_regen_multiplier()"
            ) == pytest.approx(0.75)
        finally:
            await tr.rollback()


async def test_plan_budget_can_draw_bank_only_for_actionable_backlog(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _configure_economy(conn)
            low = _json(
                await conn.fetchval(
                    "SELECT enforce_heartbeat_plan_energy($1::jsonb)",
                    json.dumps(
                        {
                            "context": {"energy": {"current": 7, "max": 20}},
                            "has_backlog_tasks": True,
                        }
                    ),
                )
            )
            ordinary = _json(
                await conn.fetchval(
                    "SELECT enforce_heartbeat_plan_energy($1::jsonb)",
                    json.dumps(
                        {
                            "context": {"energy": {"current": 50, "max": 20}},
                            "has_backlog_tasks": False,
                        }
                    ),
                )
            )
            backlog = _json(
                await conn.fetchval(
                    "SELECT enforce_heartbeat_plan_energy($1::jsonb)",
                    json.dumps(
                        {
                            "context": {"energy": {"current": 50, "max": 20}},
                            "has_backlog_tasks": True,
                        }
                    ),
                )
            )
            assert low["energy_budget"] == 7
            assert ordinary["energy_budget"] == 20
            assert backlog["energy_budget"] == 40
            assert backlog["energy_bank_capacity"] == 60
        finally:
            await tr.rollback()


async def test_cadence_uses_drive_urgency_and_controls_due_time(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _configure_economy(conn)
            await conn.execute("UPDATE drives SET current_level=0")
            quiet = _json(await conn.fetchval("SELECT heartbeat_adaptive_cadence()"))
            assert quiet["max_urgency_ratio"] == 0
            assert quiet["cadence_minutes"] == pytest.approx(90.0)
            assert quiet["urgent_drives"] == []

            await conn.execute(
                "UPDATE drives SET current_level=1, urgency_threshold=0.5 WHERE name='curiosity'"
            )
            urgent = _json(await conn.fetchval("SELECT heartbeat_adaptive_cadence()"))
            assert urgent["max_urgency_ratio"] == pytest.approx(2.0)
            assert urgent["cadence_minutes"] == pytest.approx(15.0)
            assert urgent["urgent_drives"][0]["name"] == "curiosity"

            await conn.execute(
                """
                UPDATE heartbeat_state
                SET last_heartbeat_at=CURRENT_TIMESTAMP - INTERVAL '1 day',
                    next_heartbeat_at=CURRENT_TIMESTAMP + INTERVAL '5 minutes'
                WHERE id=1
                """
            )
            schedule = await conn.fetchrow(
                "SELECT last_heartbeat_at,next_heartbeat_at,active_heartbeat_id FROM heartbeat_state WHERE id=1"
            )
            assert await conn.fetchval("SELECT should_run_heartbeat()") is False, dict(schedule)
            await conn.execute(
                "UPDATE heartbeat_state SET next_heartbeat_at=CURRENT_TIMESTAMP - INTERVAL '1 second' WHERE id=1"
            )
            assert await conn.fetchval("SELECT should_run_heartbeat()") is True
        finally:
            await tr.rollback()


async def test_only_verified_operator_thanks_credit_proactive_work(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _configure_economy(conn)
            started = _json(await conn.fetchval("SELECT start_heartbeat()"))
            heartbeat_id = started["heartbeat_id"]
            assert await conn.fetchval(
                "SELECT count(*) FROM heartbeat_outcomes WHERE heartbeat_id=$1",
                heartbeat_id,
            ) == 1, started
            await conn.fetchval(
                "SELECT record_heartbeat_outcome_signal($1,'proactive_contact','contact:1')",
                heartbeat_id,
            )
            await conn.fetchval(
                "SELECT finalize_heartbeat_economy($1,0,'completed')", heartbeat_id
            )

            untrusted = _json(
                await conn.fetchval(
                    "SELECT credit_heartbeat_user_feedback('Thanks for that',false,'{}'::jsonb)"
                )
            )
            credited = _json(
                await conn.fetchval(
                    "SELECT credit_heartbeat_user_feedback('Thanks for that',true,$1::jsonb)",
                    json.dumps({"surface": "chat", "session_id": str(uuid4())}),
                )
            )
            repeated = _json(
                await conn.fetchval(
                    "SELECT credit_heartbeat_user_feedback('I appreciate it',true,'{}'::jsonb)"
                )
            )
            assert untrusted["reason"] == "not_verified_operator"
            assert credited["credited"] is True
            assert credited["heartbeat_id"] == str(heartbeat_id)
            assert repeated["reason"] == "duplicate_feedback"
            row = await conn.fetchrow(
                "SELECT user_feedback_score,outcome_score FROM heartbeat_outcomes WHERE heartbeat_id=$1",
                heartbeat_id,
            )
            assert row["user_feedback_score"] == 1
            assert row["outcome_score"] == pytest.approx(0.5)
        finally:
            await tr.rollback()


async def test_agent_tool_receipts_drive_outcomes(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _configure_economy(conn)
            started = _json(await conn.fetchval("SELECT start_heartbeat()"))
            heartbeat_id = started["heartbeat_id"]
            assert await conn.fetchval(
                "SELECT count(*) FROM heartbeat_outcomes WHERE heartbeat_id=$1",
                heartbeat_id,
            ) == 1, started
            turn_id = await conn.fetchval(
                """
                INSERT INTO agent_turns (mode,heartbeat_id,status)
                VALUES ('heartbeat',$1,'completed') RETURNING id
                """,
                heartbeat_id,
            )
            memory_id = uuid4()
            await conn.execute(
                """
                INSERT INTO agent_turn_events (turn_id,event_type,payload)
                VALUES ($1,'tool_result',$2::jsonb)
                """,
                turn_id,
                json.dumps(
                    {
                        "tool_name": "remember",
                        "success": True,
                        "arguments": {"content": "durable result"},
                        "output": {"memory_id": str(memory_id), "reused": False},
                    }
                ),
            )
            finalized = _json(
                await conn.fetchval(
                    "SELECT finalize_heartbeat_economy($1,2,'completed')",
                    heartbeat_id,
                )
            )
            assert finalized["applied"] is True, finalized
            assert finalized["durable_memories_created"] == 1
            kinds = await conn.fetch(
                "SELECT signal_kind FROM heartbeat_outcome_signals WHERE heartbeat_id=$1 ORDER BY signal_kind",
                heartbeat_id,
            )
            assert [row["signal_kind"] for row in kinds] == [
                "durable_memory",
                "tool_success",
            ]
        finally:
            await tr.rollback()


async def test_legacy_release_finalizes_outcome_and_adaptive_schedule(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _configure_economy(conn)
            started = _json(await conn.fetchval("SELECT start_heartbeat()"))
            heartbeat_id = started["heartbeat_id"]
            assert await conn.fetchval(
                "SELECT count(*) FROM heartbeat_outcomes WHERE heartbeat_id=$1",
                heartbeat_id,
            ) == 1, started
            await conn.execute(
                """
                UPDATE heartbeat_state
                SET active_actions=$2::jsonb
                WHERE id=1 AND active_heartbeat_id=$1
                """,
                heartbeat_id,
                json.dumps(
                    [
                        {
                            "action": "resolve_contradiction",
                            "result": {"success": True},
                        }
                    ]
                ),
            )
            assert await conn.fetchval(
                "SELECT release_active_heartbeat($1)", heartbeat_id
            ) is True
            outcome = await conn.fetchrow(
                """
                SELECT status,durable_memories_created,contradictions_resolved,
                       cadence_minutes,next_heartbeat_at
                FROM heartbeat_outcomes WHERE heartbeat_id=$1
                """,
                heartbeat_id,
            )
            assert outcome is not None, {
                "started": started,
                "active": await conn.fetchval(
                    "SELECT active_heartbeat_id FROM heartbeat_state WHERE id=1"
                ),
            }
            assert outcome["status"] == "completed"
            assert outcome["durable_memories_created"] == 1
            assert outcome["contradictions_resolved"] == 1
            assert outcome["cadence_minutes"] is not None
            assert outcome["next_heartbeat_at"] is not None
            assert await conn.fetchval(
                "SELECT next_heartbeat_at FROM heartbeat_state WHERE id=1"
            ) == outcome["next_heartbeat_at"]
        finally:
            await tr.rollback()


async def test_z_0225_migration_is_self_contained(db_pool):
    migration = (
        Path(__file__).resolve().parents[2]
        / "db"
        / "migrations"
        / "0225_heartbeat_economy.sql"
    ).read_text(encoding="utf-8")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                """
                DROP TRIGGER IF EXISTS trg_heartbeat_economy_state_transition ON state;
                DROP FUNCTION IF EXISTS heartbeat_economy_status();
                DROP FUNCTION IF EXISTS credit_heartbeat_user_feedback(TEXT,BOOLEAN,JSONB);
                DROP FUNCTION IF EXISTS heartbeat_economy_state_transition_trigger();
                DROP FUNCTION IF EXISTS finalize_heartbeat_economy(UUID,FLOAT,TEXT,JSONB,JSONB,BOOLEAN);
                DROP FUNCTION IF EXISTS refresh_heartbeat_outcome(UUID);
                DROP FUNCTION IF EXISTS enforce_heartbeat_plan_energy(JSONB);
                DROP FUNCTION IF EXISTS record_heartbeat_tool_outcome(UUID,UUID,TEXT,JSONB);
                DROP FUNCTION IF EXISTS record_heartbeat_outcome_signal(UUID,TEXT,TEXT,FLOAT,JSONB);
                DROP FUNCTION IF EXISTS begin_heartbeat_outcome(UUID,INTEGER,JSONB);
                DROP FUNCTION IF EXISTS heartbeat_adaptive_cadence(TIMESTAMPTZ);
                DROP FUNCTION IF EXISTS heartbeat_urgency_snapshot();
                DROP FUNCTION IF EXISTS regenerate_heartbeat_energy(TIMESTAMPTZ);
                DROP FUNCTION IF EXISTS heartbeat_outcome_regen_multiplier();
                DROP FUNCTION IF EXISTS heartbeat_bank_capacity();
                DROP TABLE heartbeat_outcome_signals, heartbeat_outcomes,
                           heartbeat_economy_state;
                CREATE OR REPLACE FUNCTION start_heartbeat()
                RETURNS JSONB LANGUAGE sql AS
                $$ SELECT '{"legacy": true}'::jsonb $$;
                CREATE OR REPLACE FUNCTION should_run_heartbeat()
                RETURNS BOOLEAN LANGUAGE sql AS
                $$ SELECT FALSE $$;
                CREATE OR REPLACE FUNCTION update_energy(p_delta FLOAT)
                RETURNS FLOAT LANGUAGE sql AS
                $$ SELECT 0::float $$;
                UPDATE heartbeat_state
                SET last_heartbeat_at = CURRENT_TIMESTAMP - INTERVAL '3 hours'
                WHERE id = 1;
                """
            )
            await conn.execute(migration)
            assert await conn.fetchval(
                "SELECT to_regclass('public.heartbeat_outcomes')"
            ) == "heartbeat_outcomes"
            assert await conn.fetchval(
                "SELECT to_regprocedure('public.finalize_heartbeat_economy(uuid,double precision,text,jsonb,jsonb,boolean)') IS NOT NULL"
            ) is True
            assert "begin_heartbeat_outcome" in await conn.fetchval(
                "SELECT pg_get_functiondef('start_heartbeat()'::regprocedure)"
            )
            assert "next_heartbeat_at" in await conn.fetchval(
                "SELECT pg_get_functiondef('should_run_heartbeat()'::regprocedure)"
            )
            assert "heartbeat_bank_capacity" in await conn.fetchval(
                "SELECT pg_get_functiondef('update_energy(double precision)'::regprocedure)"
            )
            assert await conn.fetchval("SELECT heartbeat_bank_capacity()") > 0
            assert await conn.fetchval(
                """
                SELECT e.last_regenerated_at = h.last_heartbeat_at
                FROM heartbeat_economy_state e
                CROSS JOIN heartbeat_state h
                WHERE e.id=1 AND h.id=1
                """
            ) is True
            status = _json(await conn.fetchval("SELECT heartbeat_economy_status()"))
            assert status["bank_capacity"] >= status["reserve_energy"]
        finally:
            await tr.rollback()
