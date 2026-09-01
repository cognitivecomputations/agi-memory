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


async def test_dns_failure_classifies_as_network_or_provider(db_pool):
    """#111: a real-world provider outage manifested as a Python DNS error
    ("[Errno -2] Name or service not known"), which previously had no
    matching pattern and fell into the generic execution_failure bucket."""
    async with db_pool.acquire() as conn:
        classification = _j(
            await conn.fetchval(
                "SELECT classify_defect_event('llm.heartbeat', $1)",
                "[Errno -2] Name or service not known",
            )
        )
    assert classification["category"] == "network_or_provider"


async def test_heartbeat_llm_total_failure_creates_a_defect_report(db_pool):
    """#111: a total LLM failure during heartbeat decision-making (both the
    main loop and the repair call failed) must be recorded, not silently
    swallowed -- this is the exact call self-repair otherwise never sees,
    since record_heartbeat_action_defects is gated on a non-empty actions
    array and a total failure produces zero actions."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            defect_id = await conn.fetchval(
                "SELECT record_defect_event($1, $2, $3, $4::jsonb)",
                "heartbeat",
                "llm.heartbeat",
                "[Errno -2] Name or service not known",
                json.dumps({"heartbeat_id": str(uuid4()), "fallback": "legacy"}),
            )
            row = await conn.fetchrow(
                "SELECT category, severity, source, component, last_error "
                "FROM defect_reports WHERE id = $1::uuid",
                defect_id,
            )
        finally:
            await tr.rollback()

    assert row["category"] == "network_or_provider"
    assert row["source"] == "heartbeat"
    assert row["component"] == "llm.heartbeat"
    assert "Name or service not known" in row["last_error"]


async def test_rlm_heartbeat_total_failure_records_defect_then_falls_back(
    db_pool, monkeypatch
):
    """#111 end to end at the dispatcher level: an RLM heartbeat decision
    that fails completely (a) records a real defect (the exact case
    record_heartbeat_action_defects can never see, since a total failure
    produces zero actions) and (b) still falls back to the legacy path
    rather than fabricating a decision or losing the heartbeat's turn."""
    import services.external_calls as external_calls_mod
    from services.external_calls import ExternalCallProcessor
    from services.hexis_rlm import RlmLoopFailure

    heartbeat_id = str(uuid4())

    async def fake_run_heartbeat_decision(**_kwargs):
        raise RlmLoopFailure("[Errno -2] Name or service not known")

    legacy_decision = {"reasoning": "legacy fallback", "actions": [], "goal_changes": []}

    async def fake_chat_json(**_kwargs):
        return legacy_decision, json.dumps(legacy_decision)

    monkeypatch.setattr(
        "services.hexis_rlm.run_heartbeat_decision", fake_run_heartbeat_decision
    )
    monkeypatch.setattr(external_calls_mod, "chat_json", fake_chat_json)

    processor = ExternalCallProcessor()
    async with db_pool.acquire() as conn:
        before = await conn.fetchval(
            "SELECT count(*) FROM defect_reports WHERE component = 'llm.heartbeat'"
        )
        result = await processor.process_call_payload(
            conn,
            "think",
            {"kind": "heartbeat_decision_rlm", "heartbeat_id": heartbeat_id, "context": {}},
        )
        after = await conn.fetchval(
            "SELECT count(*) FROM defect_reports WHERE component = 'llm.heartbeat'"
        )
        row = await conn.fetchrow(
            "SELECT category, last_error FROM defect_reports "
            "WHERE component = 'llm.heartbeat' ORDER BY created_at DESC LIMIT 1"
        )

    assert after == before + 1
    assert row["category"] == "network_or_provider"
    assert "Name or service not known" in row["last_error"]
    # The legacy fallback actually ran and produced the real decision.
    assert result["kind"] == "heartbeat_decision"
    assert result["decision"] == legacy_decision


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
