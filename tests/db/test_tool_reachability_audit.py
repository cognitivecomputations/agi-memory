from __future__ import annotations

import json

import pytest


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def test_capability_bulk_writer_and_health(db_pool):
    payload = [
        {
            "tool_name": "reachable_test_tool",
            "tool_context": "chat",
            "available": True,
        },
        {
            "tool_name": "missing_test_tool",
            "tool_context": "chat",
            "available": False,
            "reason_code": "skill_unbound",
            "reason_if_missing": "test gap",
        },
    ]
    async with db_pool.acquire() as conn:
        assert await conn.fetchval(
            "SELECT record_worker_capabilities('test-worker', NULL, 'full', $1::jsonb)",
            json.dumps(payload),
        ) == 2
        raw = await conn.fetchval("SELECT capability_reachability_health()")
        health = json.loads(raw) if isinstance(raw, str) else raw
        assert health["unexpected_gaps"] >= 1
        assert any(
            item["tool"] == "missing_test_tool" for item in health["gap_examples"]
        )


async def test_tool_surface_ledger_is_append_only(db_pool):
    async with db_pool.acquire() as conn:
        event_id = await conn.fetchval(
            """
            SELECT record_tool_surface_decision(
                NULL, 'test', 'chat', 'selection', repeat('a', 64), ARRAY['core-memory'],
                '[]'::jsonb, ARRAY['recall'], ARRAY['recall'], ARRAY[]::text[],
                1, 'full'
            )
            """
        )
        assert event_id is not None
        with pytest.raises(Exception, match="append-only"):
            await conn.execute(
                "DELETE FROM tool_surface_decision_events WHERE id = $1", event_id
            )


async def test_doctor_surfaces_reachability_and_audit_health(db_pool):  # noqa: ARG001
    from core.cli_api import doctor_payload
    from tests.utils import _db_dsn

    checks = await doctor_payload(_db_dsn(), wait_seconds=2)
    by_label = {check["label"]: check for check in checks}
    assert by_label["Tool reachability"]["status"] in {"OK", "WARN"}
    assert by_label["Tool reachability"]["detail"]
    assert by_label["Tool surface audit"]["status"] == "OK"
