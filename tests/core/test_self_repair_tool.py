from __future__ import annotations

import json
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.registry import create_default_registry
from core.tools.self_repair import SelfRepairHandler, create_self_repair_tools

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _ctx(db_pool) -> ToolExecutionContext:
    registry = MagicMock()
    registry.pool = db_pool
    return ToolExecutionContext(
        tool_context=ToolContext.HEARTBEAT,
        call_id="self-repair-test",
        registry=registry,
    )


async def test_self_repair_tool_lists_and_diagnoses_defects(db_pool):
    marker = uuid4().hex
    try:
        async with db_pool.acquire() as conn:
            defect_id = await conn.fetchval(
                """
                SELECT record_defect_event(
                    'heartbeat',
                    'reflect',
                    $1,
                    $2::jsonb
                )
                """,
                f"Unknown action: reflect marker {marker}",
                json.dumps({"heartbeat_id": str(uuid4()), "action": "reflect"}),
            )

        listed = await SelfRepairHandler().execute(
            {"action": "list", "status": "open", "limit": 50},
            _ctx(db_pool),
        )
        diagnosed = await SelfRepairHandler().execute(
            {"action": "diagnose", "defect_id": str(defect_id)},
            _ctx(db_pool),
        )
    finally:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id FROM defect_reports WHERE last_error LIKE $1",
                f"%{marker}%",
            )
            for row in rows:
                await conn.execute("DELETE FROM defect_reports WHERE id = $1::uuid", row["id"])

    assert listed.success, listed.error
    assert listed.output["count"] >= 1
    assert any(marker in (report.get("last_error") or "") for report in listed.output["reports"])
    assert "defect report" in (listed.display_output or "")

    assert diagnosed.success, diagnosed.error
    assert diagnosed.output["success"] is True
    assert diagnosed.output["diagnosis"]["category"] == "tool_contract"
    assert "services/prompts/rlm_heartbeat_system.md" in diagnosed.output["diagnosis"]["likely_files"]
    assert "Diagnosis:" in (diagnosed.display_output or "")


async def test_self_repair_tool_registered_in_default_registry(db_pool):
    names = [handler.spec.name for handler in create_self_repair_tools()]
    assert names == ["self_repair"]

    registry = create_default_registry(db_pool)
    assert registry.get("self_repair") is not None
