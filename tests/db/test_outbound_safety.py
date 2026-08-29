from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


async def _contact(conn):
    marker = uuid4().hex
    email = f"{marker}@example.com"
    phone = f"+1555{marker[:7]}"
    contact_id = await conn.fetchval(
        "INSERT INTO contacts(name, email, phone) VALUES ($1, $2, $3) RETURNING id",
        f"Outbound {marker}",
        email,
        phone,
    )
    return marker, email, phone, contact_id


async def _goal(conn, marker: str, *, origin: str = "user_request"):
    return await conn.fetchval(
        """
        INSERT INTO memories (
            type, goal_origin, content, importance, trust_level, status, metadata
        ) VALUES (
            'goal', $1::goal_source, $2, 0.7, 0.8, 'active',
            '{"priority":"active"}'::jsonb
        )
        RETURNING id
        """,
        origin,
        f"Outbound goal {marker}",
    )


async def test_cross_channel_stop_is_immediate_acknowledged_once_and_reversible(
    db_pool,
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker, email, phone, contact_id = await _contact(conn)
            email_entity = _j(
                await conn.fetchval(
                    "SELECT resolve_outbound_entity('email', $1)", email
                )
            )
            phone_entity = _j(
                await conn.fetchval(
                    "SELECT resolve_outbound_entity('signal', $1)", phone
                )
            )
            assert (
                email_entity["entity"]
                == phone_entity["entity"]
                == f"contact:{contact_id}"
            )
            slack_identity = f"U-{marker[:12]}"
            await conn.execute(
                "UPDATE contacts SET metadata=jsonb_build_object('channels', jsonb_build_object('slack', $2::text)) WHERE id=$1",
                contact_id,
                slack_identity,
            )
            room_delivery = _j(
                await conn.fetchval(
                    "SELECT resolve_outbound_entity('slack', 'C-ROOM', $1, false, false)",
                    slack_identity,
                )
            )
            assert room_delivery["entity"] == f"contact:{contact_id}"
            assert room_delivery["address"] == "c-room"

            _, comparable_email, _, comparable_id = await _contact(conn)
            primary_name = await conn.fetchval(
                "SELECT name FROM contacts WHERE id=$1", contact_id
            )
            comparable_name = await conn.fetchval(
                "SELECT name FROM contacts WHERE id=$1", comparable_id
            )
            await conn.execute(
                "SELECT upsert_self_concept_edge('relationship', $1, 0.7, NULL)",
                primary_name,
            )
            await conn.execute(
                "SELECT upsert_self_concept_edge('relationship', $1, 0.72, NULL)",
                comparable_name,
            )
            comparable_entity = _j(
                await conn.fetchval(
                    "SELECT resolve_outbound_entity('email', $1)", comparable_email
                )
            )
            await conn.execute(
                "SELECT _outbound_ensure_contact_budget($1, 'email')",
                email_entity["entity"],
            )
            await conn.execute(
                "SELECT _outbound_ensure_contact_budget($1, 'email')",
                comparable_entity["entity"],
            )
            regen_before = await conn.fetchval(
                "SELECT regen_per_day FROM contact_budgets WHERE entity=$1 AND channel='email'",
                comparable_entity["entity"],
            )

            stopped = _j(
                await conn.fetchval(
                    "SELECT handle_inbound_contact_control('signal', $1, 'STOP.', false, '{}'::jsonb)",
                    phone,
                )
            )
            repeated = _j(
                await conn.fetchval(
                    "SELECT handle_inbound_contact_control('signal', $1, 'unsubscribe!', false, '{}'::jsonb)",
                    phone,
                )
            )
            assert stopped["action"] == "stop"
            assert stopped["acknowledge"] is True
            assert repeated["acknowledge"] is False
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM outbox_messages WHERE source='contact_opt_out'"
                )
                == 1
            )
            regen_after = await conn.fetchval(
                "SELECT regen_per_day FROM contact_budgets WHERE entity=$1 AND channel='email'",
                comparable_entity["entity"],
            )
            assert float(regen_after) == pytest.approx(float(regen_before) * 0.9)

            blocked = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'user_request', 'current_turn', NULL, 'normal',
                        '{"tool_context":"chat"}'::jsonb, 'hello', false, false
                    )
                    """,
                    f"stop:{marker}",
                    email,
                )
            )
            assert blocked["allowed"] is False
            assert "opted out" in blocked["reason"]

            restarted = _j(
                await conn.fetchval(
                    "SELECT handle_inbound_contact_control('email', $1, 'OPT-OUT', false, '{}'::jsonb)",
                    email,
                )
            )
            assert restarted["action"] == "stop"
            unstop = _j(
                await conn.fetchval(
                    "SELECT handle_inbound_contact_control('email', $1, 'UNSTOP', false, '{}'::jsonb)",
                    email,
                )
            )
            assert unstop["action"] == "start"
            assert unstop["acknowledge"] is True
        finally:
            await tr.rollback()


async def test_purpose_budget_disclosure_and_failure_refund(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker, email, _, _ = await _contact(conn)
            goal_id = await _goal(conn, marker)

            missing = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        NULL, NULL, NULL, 'normal', '{"tool_context":"chat"}'::jsonb,
                        'hello', false, false
                    )
                    """,
                    f"missing:{marker}",
                    email,
                )
            )
            assert missing["allowed"] is False
            assert missing["error_type"] == "purpose_required"

            authorized = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'goal', $3, NULL, 'normal', '{"tool_context":"heartbeat"}'::jsonb,
                        'hello', false, false
                    )
                    """,
                    f"goal:{marker}",
                    email,
                    str(goal_id),
                )
            )
            assert authorized["allowed"] is True
            assert authorized["assigned_goal"] is True
            assert authorized["charged_cost"] > 0
            assert authorized["disclosure_mode"] == "full"
            assert "Reply STOP" in authorized["disclosure"]
            assert "Why you received this" in authorized["disclosure"]

            points_reserved = await conn.fetchval(
                "SELECT points FROM contact_budgets WHERE entity=$1 AND channel='email'",
                authorized["entity"],
            )
            duplicate = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'goal', $3, NULL, 'normal', '{"tool_context":"heartbeat"}'::jsonb,
                        'hello', false, false
                    )
                    """,
                    f"goal:{marker}",
                    email,
                    str(goal_id),
                )
            )
            assert duplicate["event_id"] == authorized["event_id"]
            assert duplicate["reused_reservation"] is True
            assert await conn.fetchval(
                "SELECT points FROM contact_budgets WHERE entity=$1 AND channel='email'",
                authorized["entity"],
            ) == pytest.approx(points_reserved)

            await conn.fetchval(
                "SELECT finalize_outbound($1::uuid, false, NULL, 'provider failed', '{}'::jsonb)",
                authorized["event_id"],
            )
            unassigned = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'user_request', 'current_turn', NULL, 'normal',
                        '{"tool_context":"chat"}'::jsonb, 'hello', false, false
                    )
                    """,
                    f"unassigned:{marker}",
                    email,
                )
            )
            assert unassigned["allowed"] is True
            assert authorized["charged_cost"] == pytest.approx(
                unassigned["charged_cost"]
                * float(
                    await conn.fetchval(
                        "SELECT get_config_float('outbound.assigned_goal_contact_discount')"
                    )
                )
            )
            assert authorized["charged_cost"] > 0
            await conn.fetchval(
                "SELECT finalize_outbound($1::uuid, false, NULL, 'comparison only', '{}'::jsonb)",
                unassigned["event_id"],
            )
            resolved = _j(
                await conn.fetchval(
                    "SELECT resolve_outbound_entity('email', $1)", email
                )
            )
            points = await conn.fetchval(
                "SELECT points FROM contact_budgets WHERE entity=$1 AND channel='email'",
                resolved["entity"],
            )
            assert float(points) == pytest.approx(float(authorized["points_before"]))

            # A successful full disclosure makes the next same-channel contact
            # use the permanent AI marker without repeating the STOP paragraph.
            second = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'goal', $3, NULL, 'normal', '{"tool_context":"heartbeat"}'::jsonb,
                        'hello', false, false
                    )
                    """,
                    f"full:{marker}",
                    email,
                    str(goal_id),
                )
            )
            await conn.fetchval(
                "SELECT finalize_outbound($1::uuid, true, 'provider-1', NULL, '{}'::jsonb)",
                second["event_id"],
            )
            await conn.execute(
                "UPDATE contact_budgets SET points=max_points WHERE entity=$1 AND channel='email'",
                resolved["entity"],
            )
            third = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'goal', $3, NULL, 'normal', '{"tool_context":"heartbeat"}'::jsonb,
                        'hello again', false, false
                    )
                    """,
                    f"marker:{marker}",
                    email,
                    str(goal_id),
                )
            )
            assert third["disclosure_mode"] == "marker"
            assert "Reply STOP" not in third["disclosure"]
            await conn.fetchval(
                "SELECT finalize_outbound($1::uuid, true, 'provider-2', NULL, '{}'::jsonb)",
                third["event_id"],
            )
            await conn.execute(
                "UPDATE contact_budgets SET points=max_points WHERE entity=$1 AND channel='email'",
                resolved["entity"],
            )
            new_thread = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'goal', $3, 'new-thread', 'normal',
                        '{"tool_context":"heartbeat"}'::jsonb,
                        'threaded update', false, false
                    )
                    """,
                    f"new-thread:{marker}",
                    email,
                    str(goal_id),
                )
            )
            assert new_thread["disclosure_mode"] == "full"
            await conn.fetchval(
                "SELECT finalize_outbound($1::uuid, false, NULL, 'thread test', '{}'::jsonb)",
                new_thread["event_id"],
            )

            await conn.execute(
                """
                UPDATE outbound_events
                SET created_at = CURRENT_TIMESTAMP - INTERVAL '31 days'
                WHERE entity=$1 AND disclosure_mode='full' AND status='delivered'
                """,
                resolved["entity"],
            )
            await conn.execute(
                "UPDATE contact_budgets SET points=max_points WHERE entity=$1 AND channel='email'",
                resolved["entity"],
            )
            long_gap = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'goal', $3, NULL, 'normal', '{"tool_context":"heartbeat"}'::jsonb,
                        'after a long gap', false, false
                    )
                    """,
                    f"long-gap:{marker}",
                    email,
                    str(goal_id),
                )
            )
            assert long_gap["disclosure_mode"] == "full"
        finally:
            await tr.rollback()


async def test_replies_are_free_urgent_overdraft_is_backed_and_inbound_repays(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker, email, _, _ = await _contact(conn)
            session_id = await conn.fetchval(
                """
                INSERT INTO channel_sessions(channel_type, channel_id, sender_id)
                VALUES ('email', $1, $2) RETURNING id
                """,
                f"thread:{marker}",
                email,
            )
            inbound_id = f"inbound:{marker}"
            await conn.execute(
                """
                INSERT INTO channel_messages(session_id, direction, content, platform_message_id)
                VALUES ($1, 'inbound', 'hello', $2)
                """,
                session_id,
                inbound_id,
            )
            reply = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'gmail_reply', 'email', $2, NULL,
                        'reply', $3, $3, 'normal', '{}'::jsonb, 'reply', false, false
                    )
                    """,
                    f"reply:{marker}",
                    email,
                    inbound_id,
                )
            )
            assert reply["allowed"] is True
            assert reply["charged_cost"] == 0

            resolved = _j(
                await conn.fetchval(
                    "SELECT resolve_outbound_entity('email', $1)", email
                )
            )
            await conn.fetchval(
                "SELECT _outbound_ensure_contact_budget($1, 'email')",
                resolved["entity"],
            )
            await conn.fetchval(
                "SELECT set_config('outbound.quiet_hours', $1::jsonb)",
                json.dumps({"start": 0, "end": 24, "multiplier": 2}),
            )
            await conn.execute(
                "UPDATE contact_budgets SET points=max_points, strain=0 WHERE entity=$1 AND channel='email'",
                resolved["entity"],
            )
            quiet_normal = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'user_request', 'current_turn', NULL, 'normal',
                        '{"tool_context":"chat"}'::jsonb, 'quiet normal', false, false
                    )
                    """,
                    f"quiet-normal:{marker}",
                    email,
                )
            )
            await conn.fetchval(
                "SELECT finalize_outbound($1::uuid, false, NULL, 'cost comparison', '{}'::jsonb)",
                quiet_normal["event_id"],
            )
            quiet_high = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'user_request', 'current_turn', NULL, 'high',
                        '{"tool_context":"chat"}'::jsonb, 'quiet high', false, false
                    )
                    """,
                    f"quiet-high:{marker}",
                    email,
                )
            )
            assert quiet_normal["allowed"] is True
            assert quiet_high["allowed"] is True
            assert quiet_normal["charged_cost"] == pytest.approx(
                quiet_high["charged_cost"] * 2
            )
            await conn.fetchval(
                "SELECT finalize_outbound($1::uuid, false, NULL, 'cost comparison', '{}'::jsonb)",
                quiet_high["event_id"],
            )
            await conn.execute(
                """
                UPDATE contact_budgets
                SET points=max_points, strain=0, consecutive_silent=4
                WHERE entity=$1 AND channel='email'
                """,
                resolved["entity"],
            )
            silent_gate = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'user_request', 'current_turn', NULL, 'normal',
                        '{"tool_context":"chat"}'::jsonb, 'another nudge', false, false
                    )
                    """,
                    f"silent-gate:{marker}",
                    email,
                )
            )
            assert silent_gate["allowed"] is False
            assert "has not replied" in silent_gate["reason"]
            await conn.execute(
                """
                UPDATE contact_budgets
                SET points=0, strain=0, consecutive_silent=0
                WHERE entity=$1 AND channel='email'
                """,
                resolved["entity"],
            )
            urgent = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'email_send', 'email', $2, NULL,
                        'user_request', 'current_turn', NULL, 'urgent',
                        '{"tool_context":"chat"}'::jsonb, 'urgent', false, false
                    )
                    """,
                    f"urgent:{marker}",
                    email,
                )
            )
            assert urgent["allowed"] is True
            assert urgent["points_after"] < 0
            await conn.fetchval(
                "SELECT finalize_outbound($1::uuid, true, 'provider-urgent', NULL, '{}'::jsonb)",
                urgent["event_id"],
            )
            before = await conn.fetchrow(
                "SELECT points, strain, consecutive_silent FROM contact_budgets WHERE entity=$1 AND channel='email'",
                resolved["entity"],
            )
            credited = _j(
                await conn.fetchval(
                    "SELECT record_outbound_contact_inbound('email', $1, 'thanks', false, '{}'::jsonb)",
                    email,
                )
            )
            after = await conn.fetchrow(
                "SELECT points, strain, consecutive_silent FROM contact_budgets WHERE entity=$1 AND channel='email'",
                resolved["entity"],
            )
            assert credited["credited"] > 0
            assert after["points"] > before["points"]
            assert after["strain"] < before["strain"]
            assert after["consecutive_silent"] == 0
        finally:
            await tr.rollback()


async def test_primary_user_and_kill_switch_controls(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = uuid4().hex
            primary = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'queue_user_message', 'outbox', 'primary_user', NULL,
                        'connection', 'current_turn', NULL, 'normal',
                        '{"tool_context":"chat"}'::jsonb, 'home', true, false
                    )
                    """,
                    f"primary:{marker}",
                )
            )
            assert primary["allowed"] is True
            assert primary["charged_cost"] == 0
            assert primary["disclosure_mode"] == "none"
            primary_stop = _j(
                await conn.fetchval(
                    "SELECT handle_inbound_contact_control('outbox', 'primary_user', 'STOP', true, '{}'::jsonb)"
                )
            )
            assert primary_stop["recognized"] is False

            await conn.fetchval("SELECT set_outbound_global_suspension(true)")
            paused = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'queue_user_message', 'outbox', 'primary_user', NULL,
                        'connection', 'current_turn', NULL, 'normal',
                        '{"tool_context":"chat"}'::jsonb, 'home', true, false
                    )
                    """,
                    f"paused:{marker}",
                )
            )
            assert paused["allowed"] is False
            assert "globally suspended" in paused["reason"]
            ledger = _j(await conn.fetchval("SELECT get_outbound_ledger(10, NULL)"))
            assert ledger["suspended"] is True
            assert any(item["id"] == paused["event_id"] for item in ledger["events"])

            await conn.fetchval("SELECT set_outbound_global_suspension(false)")
            await conn.fetchval(
                "SELECT set_outbound_entity_suspension('primary:user', true, 'test')"
            )
            person_paused = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        $1, 'test', 'queue_user_message', 'outbox', 'primary_user', NULL,
                        'connection', 'current_turn', NULL, 'normal',
                        '{"tool_context":"chat"}'::jsonb, 'home', true, false
                    )
                    """,
                    f"person-paused:{marker}",
                )
            )
            assert person_paused["allowed"] is False
            assert "suspended" in person_paused["reason"]
        finally:
            await tr.rollback()


async def test_budget_bootstraps_from_observed_history_then_relationship_strength(
    db_pool,
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker, _, _, contact_id = await _contact(conn)
            slack_id = f"U-{marker[:12]}"
            await conn.execute(
                "UPDATE contacts SET metadata=jsonb_build_object('channels', jsonb_build_object('slack', $2::text)) WHERE id=$1",
                contact_id,
                slack_id,
            )
            resolved = _j(
                await conn.fetchval(
                    "SELECT resolve_outbound_entity('slack', $1)", slack_id
                )
            )
            session_id = await conn.fetchval(
                """
                INSERT INTO channel_sessions(channel_type, channel_id, sender_id)
                VALUES ('slack', $1, $1) RETURNING id
                """,
                slack_id,
            )
            for index in range(7):
                await conn.execute(
                    """
                    INSERT INTO channel_messages(
                        session_id, direction, content, platform_message_id, created_at
                    ) VALUES (
                        $1, 'outbound', 'historical message', $2,
                        CURRENT_TIMESTAMP - make_interval(days => $3)
                    )
                    """,
                    session_id,
                    f"history-{marker}-{index}",
                    index,
                )
            await conn.execute(
                """
                UPDATE channel_source_items source
                SET created_at = message.created_at
                FROM channel_messages message
                WHERE source.channel_message_id = message.id
                  AND source.session_id = $1
                """,
                session_id,
            )
            await conn.execute(
                "SELECT _outbound_ensure_contact_budget($1, 'slack')",
                resolved["entity"],
            )
            observed = await conn.fetchrow(
                "SELECT observed_per_week, regen_per_day FROM contact_budgets WHERE entity=$1 AND channel='slack'",
                resolved["entity"],
            )
            assert float(observed["observed_per_week"]) == pytest.approx(7.0)
            assert float(observed["regen_per_day"]) == pytest.approx(1.0)

            strong_marker, _, _, strong_id = await _contact(conn)
            strong_slack_id = f"U-{strong_marker[:12]}"
            strong_name = await conn.fetchval(
                "SELECT name FROM contacts WHERE id=$1", strong_id
            )
            await conn.execute(
                "UPDATE contacts SET metadata=jsonb_build_object('channels', jsonb_build_object('slack', $2::text)) WHERE id=$1",
                strong_id,
                strong_slack_id,
            )
            await conn.execute(
                "SELECT upsert_self_concept_edge('relationship', $1, 0.95, NULL)",
                strong_name,
            )
            strong = _j(
                await conn.fetchval(
                    "SELECT resolve_outbound_entity('slack', $1)", strong_slack_id
                )
            )
            await conn.execute(
                "SELECT _outbound_ensure_contact_budget($1, 'slack')",
                strong["entity"],
            )
            relationship_budget = await conn.fetchrow(
                "SELECT observed_per_week, regen_per_day FROM contact_budgets WHERE entity=$1 AND channel='slack'",
                strong["entity"],
            )
            assert float(relationship_budget["observed_per_week"]) == 0
            assert float(relationship_budget["regen_per_day"]) == pytest.approx(3.0)
        finally:
            await tr.rollback()


async def test_legacy_heartbeat_outreach_requires_backing_and_exact_public_target(
    db_pool,
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            heartbeat_id = uuid4()
            missing_user_purpose = _j(
                await conn.fetchval(
                    "SELECT execute_heartbeat_action($1, 'reach_out_user', $2::jsonb)",
                    heartbeat_id,
                    json.dumps({"message": "hello"}),
                )
            )
            assert missing_user_purpose["success"] is False
            assert "backed purpose" in missing_user_purpose["error"]

            unsafe_public_fallback = _j(
                await conn.fetchval(
                    "SELECT execute_heartbeat_action($1, 'reach_out_public', $2::jsonb)",
                    heartbeat_id,
                    json.dumps(
                        {
                            "content": "announcement",
                            "platform": "slack",
                            "purpose_kind": "goal",
                            "purpose_reference": str(uuid4()),
                        }
                    ),
                )
            )
            assert unsafe_public_fallback["success"] is False
            assert "target_id" in unsafe_public_fallback["error"]
            assert "private channel" in unsafe_public_fallback["error"]

            marker = uuid4().hex
            goal_id = await _goal(conn, marker)
            public = _j(
                await conn.fetchval(
                    "SELECT execute_heartbeat_action($1, 'reach_out_public', $2::jsonb)",
                    heartbeat_id,
                    json.dumps(
                        {
                            "content": "A verified announcement.",
                            "platform": "slack",
                            "target_id": "C-PUBLIC",
                            "purpose_kind": "goal",
                            "purpose_reference": str(goal_id),
                            "urgency": "normal",
                        }
                    ),
                )
            )
            assert public["success"] is True
            assert public["cost"] == pytest.approx(
                float(await conn.fetchval("SELECT get_action_cost('reach_out_public')"))
                * float(
                    await conn.fetchval(
                        "SELECT get_config_float('outbound.assigned_goal_energy_multiplier')"
                    )
                )
            )
            envelope = public["outbox_messages"][0]
            assert envelope["kind"] == "public"
            assert envelope["payload"]["delivery_mode"] == "direct"
            assert envelope["payload"]["target_channel"] == "slack"
            assert envelope["payload"]["target_id"] == "C-PUBLIC"
            assert envelope["payload"]["purpose_reference"] == str(goal_id)
        finally:
            await tr.rollback()


async def test_z_0223_migration_is_self_contained_for_existing_installations(db_pool):
    migration = (
        Path(__file__).resolve().parents[2]
        / "db"
        / "migrations"
        / "0223_outbound_safety.sql"
    ).read_text(encoding="utf-8")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                """
                DO $drop$
                DECLARE fn RECORD;
                BEGIN
                    FOR fn IN
                        SELECT p.oid::regprocedure AS signature
                        FROM pg_proc p
                        JOIN pg_namespace n ON n.oid=p.pronamespace
                        WHERE n.nspname='public' AND p.proname IN (
                            '_outbound_normalize_address',
                            '_outbound_channel_float',
                            'resolve_outbound_entity',
                            'check_outbound_controls',
                            '_outbound_observed_per_week',
                            '_outbound_relationship_strength',
                            '_outbound_ensure_contact_budget',
                            'verify_outbound_purpose',
                            '_outbound_disclosure_text',
                            'authorize_outbound',
                            'finalize_outbound',
                            'record_outbound_contact_inbound',
                            'handle_inbound_contact_control',
                            'set_outbound_global_suspension',
                            'set_outbound_entity_suspension',
                            'get_outbound_ledger'
                        )
                    LOOP
                        EXECUTE format('DROP FUNCTION %s CASCADE', fn.signature);
                    END LOOP;
                END;
                $drop$;
                """,
            )
            await conn.execute(
                """
                DROP TABLE outbound_events,
                           outbound_contact_control_events,
                           outbound_contact_controls,
                           contact_budgets,
                           outbound_contact_endpoints
                CASCADE
                """
            )
            assert (
                await conn.fetchval("SELECT to_regclass('public.outbound_events')")
                is None
            )

            await conn.execute(migration)

            assert (
                await conn.fetchval(
                    "SELECT to_regclass('public.outbound_events') IS NOT NULL"
                )
                is True
            )
            assert (
                await conn.fetchval(
                    "SELECT to_regprocedure('public.authorize_outbound(text,text,text,text,text,text,text,text,text,text,jsonb,text,boolean,boolean)') IS NOT NULL"
                )
                is True
            )
            assert (
                await conn.fetchval(
                    "SELECT to_regprocedure('public.check_outbound_controls(text,text,text,boolean,boolean)') IS NOT NULL"
                )
                is True
            )
            assert "assigned_goal_energy_multiplier" in await conn.fetchval(
                "SELECT pg_get_functiondef('evaluate_tool_call(text,jsonb,jsonb)'::regprocedure)"
            )
            assert "exact platform and public target_id" in await conn.fetchval(
                "SELECT pg_get_functiondef('execute_heartbeat_action(uuid,text,jsonb)'::regprocedure)"
            )
            primary = _j(
                await conn.fetchval(
                    """
                    SELECT authorize_outbound(
                        'migration-probe', 'test', 'queue_user_message', 'outbox',
                        'primary_user', NULL, 'connection', 'migration-probe', NULL,
                        'normal', '{"tool_context":"chat"}'::jsonb,
                        'migration probe', true, false
                    )
                    """
                )
            )
            assert primary["allowed"] is True
        finally:
            await tr.rollback()
