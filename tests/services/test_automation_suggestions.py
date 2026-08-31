from __future__ import annotations

import json
import uuid

import pytest

from skills.base import SkillSpec


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_installed_skill_blueprint_registers_inert_suggestion(db_pool):
    from services.automation_suggestions import refresh_automation_suggestions

    name = f"blueprint-{uuid.uuid4().hex[:10]}"
    skill = SkillSpec(
        name=name,
        description="Test blueprint registration",
        content="# Test\n\nA test-only skill.",
        source=f"/tmp/{name}/SKILL.md",
        blueprint={
            "title": "Friday planning prompt",
            "rationale": "A weekly prompt can make planning easier to remember.",
            "schedule": "weekly:friday:16:00",
            "message": "Friday planning time — open Hexis and review next week.",
        },
    )

    async with db_pool.acquire() as conn:
        await conn.execute("SELECT set_config('agent.is_configured', 'true'::jsonb)")
        await conn.execute("UPDATE heartbeat_state SET init_stage = 'complete' WHERE id = 1")
        await conn.execute("DELETE FROM state WHERE key = 'automation_suggestions_state'")
        before = await conn.fetchval("SELECT count(*) FROM scheduled_tasks")

    result = await refresh_automation_suggestions(db_pool, skills=[skill])

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT source, status, task_spec, metadata
            FROM automation_suggestions
            WHERE dedup_key = $1
            """,
            f"blueprint:{name}",
        )
        after = await conn.fetchval("SELECT count(*) FROM scheduled_tasks")

    assert result["blueprints_registered"] == 1
    assert row["source"] == "blueprint"
    assert row["status"] == "pending"
    assert _json(row["task_spec"])["schedule"] == "weekly:friday:16:00"
    assert _json(row["metadata"])["skill"] == name
    assert after == before


async def test_blueprint_with_secret_material_is_rejected():
    from services.automation_suggestions import normalize_skill_blueprint

    skill = SkillSpec(
        name="unsafe-blueprint",
        description="Unsafe test",
        content="test",
        blueprint={
            "title": "Unsafe",
            "rationale": "Use api_key=sk-this-should-never-be-persisted in the routine.",
            "schedule": "daily:09:00",
            "message": "Run it.",
        },
    )

    with pytest.raises(ValueError, match="secret material"):
        normalize_skill_blueprint(skill)
