from __future__ import annotations

import json
import uuid

import pytest


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def _insert_turns(conn, *, sessions: int = 2, turns_per_session: int = 3) -> list[str]:
    ids: list[str] = []
    for session_index in range(sessions):
        session_id = uuid.uuid4()
        for turn_index in range(turns_per_session):
            unit_id = await conn.fetchval(
                """
                INSERT INTO subconscious_units (
                    session_id, content, user_text, assistant_text, idempotency_key
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                session_id,
                "Repeated release review: inspect changes, run focused tests, then verify the hosted check.",
                f"Please continue release review {session_index}-{turn_index}",
                "Inspected changes, ran focused tests, and verified the result before continuing.",
                f"skill-improvement-{uuid.uuid4()}",
            )
            ids.append(str(unit_id))
    return ids


async def test_review_ships_disabled_and_requires_cross_session_evidence(db_pool):
    from services.skill_improvement import run_skill_improvement_review_step

    async with db_pool.acquire() as conn:
        assert await conn.fetchval("SELECT get_config_bool('skills.self_improvement.enabled')") is False
        disabled = await run_skill_improvement_review_step(conn)
        assert disabled == {"skipped": True, "reason": "disabled_not_due_or_claimed"}

        await conn.execute("SELECT set_config('skills.self_improvement.enabled', 'true'::jsonb)")
        await _insert_turns(conn, sessions=1, turns_per_session=6)
        result = await run_skill_improvement_review_step(conn)
        assert result["status"] == "no_evidence"
        assert result["reason"] == "insufficient_sessions"
        assert await conn.fetchval("SELECT count(*) FROM skill_improvement_proposals") == 0


async def test_review_to_approved_skill_preserves_evidence_lineage(
    db_pool, monkeypatch, tmp_path
):
    from core.tools import ToolContext, ToolExecutionContext, create_default_registry
    from core.tools.skills import (
        ListSkillProposalsHandler,
        ReviewSkillProposalHandler,
    )
    from services.skill_improvement import run_skill_improvement_review_step
    from skills.loader import load_skills_from_dir

    proposal = {
        "name": "hosted-release-review",
        "description": "Review a release through focused and hosted verification",
        "content": (
            "# Hosted Release Review\n\nUse this after a meaningful code change. Inspect the exact diff, "
            "run the narrow tests that exercise the changed contract, then run the broader suite. "
            "Publish only when local checks pass, inspect the hosted result, and record any failure "
            "with its cause and concrete recovery step before attempting another release."
        ),
        "category": "system",
        "contexts": ["chat", "heartbeat"],
        "bound_tools": [],
        "requires_tools": [],
        "mode": "create",
        "rationale": "The same verified release sequence succeeded across multiple sessions.",
        "confidence": 0.93,
    }
    raw = json.dumps({"proposal": proposal})

    async def fake_chat_json(**_kwargs):
        return {"proposal": proposal}, raw

    async def fake_load_llm_config(*_args, **_kwargs):
        return {"provider": "test", "model": "test"}

    monkeypatch.setattr("services.skill_improvement.chat_json", fake_chat_json)
    monkeypatch.setattr("services.skill_improvement.load_llm_config", fake_load_llm_config)
    agent_root = tmp_path / "agent-authored"
    monkeypatch.setattr("core.tools.skills.AGENT_AUTHORED_SKILLS_DIR", agent_root)

    registry = create_default_registry(db_pool)
    async with db_pool.acquire() as conn:
        await conn.execute("SELECT set_config('skills.self_improvement.enabled', 'true'::jsonb)")
        await conn.execute("SELECT set_config('skills.self_improvement.interval_seconds', '1'::jsonb)")
        await conn.execute("DELETE FROM state WHERE key = 'skill_improvement_state'")
        source_ids = await _insert_turns(conn)
        result = await run_skill_improvement_review_step(conn, registry=registry)

    assert result["status"] == "proposed"
    assert result["created"] is True
    proposal_id = str(result["proposal_id"])
    assert not (agent_root / proposal["name"] / "SKILL.md").exists()
    async with db_pool.acquire() as conn:
        summary = await conn.fetchval("SELECT skill_improvement_pending_summary()")
        if isinstance(summary, str):
            summary = json.loads(summary)
    assert summary["count"] >= 1
    assert any(item["id"] == proposal_id for item in summary["proposals"])

    ctx = ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="skill-improvement-review",
        registry=registry,
    )
    listed = await ListSkillProposalsHandler().execute({}, ctx)
    assert listed.success is True
    listed_proposal = next(
        item for item in listed.output["proposals"] if str(item["id"]) == proposal_id
    )
    assert listed_proposal["status"] == "pending"
    assert listed_proposal["confidence"] == pytest.approx(0.93)
    assert set(map(str, listed_proposal["source_unit_ids"])) >= set(source_ids)

    review_handler = ReviewSkillProposalHandler()
    assert review_handler.spec.requires_approval is True
    applied = await review_handler.execute(
        {"proposal_id": proposal_id, "action": "apply"}, ctx
    )
    assert applied.success is True

    parsed = load_skills_from_dir(agent_root)
    assert len(parsed) == 1
    provenance = parsed[0].provenance
    assert provenance["proposal_id"] == proposal_id
    assert float(provenance["confidence"]) == pytest.approx(0.93)
    assert set(map(str, provenance["source_unit_ids"])) >= set(source_ids)
    async with db_pool.acquire() as conn:
        assert await conn.fetchval(
            "SELECT status FROM skill_improvement_proposals WHERE id = $1::uuid",
            proposal_id,
        ) == "applied"


async def test_reject_is_recoverable_and_invalid_model_output_is_visible(
    db_pool, monkeypatch
):
    from core.tools import ToolContext, ToolExecutionContext, create_default_registry
    from core.tools.skills import ReviewSkillProposalHandler
    from services.skill_improvement import run_skill_improvement_review_step

    async def malformed_chat_json(**_kwargs):
        return {}, "not json"

    async def fake_load_llm_config(*_args, **_kwargs):
        return {"provider": "test", "model": "test"}

    monkeypatch.setattr("services.skill_improvement.chat_json", malformed_chat_json)
    monkeypatch.setattr("services.skill_improvement.load_llm_config", fake_load_llm_config)
    registry = create_default_registry(db_pool)

    async with db_pool.acquire() as conn:
        await conn.execute("SELECT set_config('skills.self_improvement.enabled', 'true'::jsonb)")
        await conn.execute("DELETE FROM state WHERE key = 'skill_improvement_state'")
        await _insert_turns(conn)
        result = await run_skill_improvement_review_step(conn, registry=registry)
        state = await conn.fetchval("SELECT get_state('skill_improvement_state')")
        if isinstance(state, str):
            state = json.loads(state)
        proposal_id = await conn.fetchval(
            """
            INSERT INTO skill_improvement_proposals (
                name, description, content, rationale, confidence,
                source_unit_ids, evidence_digest
            ) VALUES (
                'recoverable-review', 'A recoverable review proposal',
                $1, 'Evidence supports review without destructive deletion.', 0.9,
                ARRAY[$2::uuid], $3
            ) RETURNING id
            """,
            "# Recoverable Review\n\nKeep rejected proposals available for later reconsideration. "
            "Review the original evidence, explain the decision, and reopen only when the workflow "
            "is still useful and its assumptions remain valid across the relevant contexts.",
            str(uuid.uuid4()),
            uuid.uuid4().hex,
        )

    assert result["status"] == "error"
    assert "invalid JSON" in result["error"]
    assert state["last_result"]["status"] == "error"
    assert state["in_progress"] is False

    ctx = ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="skill-proposal-recovery",
        registry=registry,
    )
    handler = ReviewSkillProposalHandler()
    rejected = await handler.execute(
        {"proposal_id": str(proposal_id), "action": "reject"}, ctx
    )
    reopened = await handler.execute(
        {"proposal_id": str(proposal_id), "action": "reopen"}, ctx
    )
    assert rejected.success is True
    assert reopened.success is True
    async with db_pool.acquire() as conn:
        assert await conn.fetchval(
            "SELECT status FROM skill_improvement_proposals WHERE id = $1",
            proposal_id,
        ) == "pending"


async def test_opted_in_review_proposes_automation_only_after_three_matching_asks(
    db_pool, monkeypatch
):
    from core.tools import create_default_registry
    from services.skill_improvement import run_skill_improvement_review_step

    source_ids: list[str] = []

    async def fake_chat_json(**_kwargs):
        doc = {
            "proposal": None,
            "automation_suggestion": {
                "title": "Monday release check",
                "rationale": "The user asked for the same release check on three Mondays.",
                "pattern": "review release status every monday",
                "confidence": 0.94,
                "evidence_unit_ids": source_ids[:3],
                "task_spec": {
                    "action": "create",
                    "name": "Monday release check",
                    "schedule": "weekly:monday:09:00",
                    "action_kind": "queue_user_message",
                    "message": "Release check — open Hexis and ask for the current release status.",
                    "delivery_mode": "outbox",
                },
            },
        }
        return doc, json.dumps(doc)

    async def fake_load_llm_config(*_args, **_kwargs):
        return {"provider": "test", "model": "test"}

    monkeypatch.setattr("services.skill_improvement.chat_json", fake_chat_json)
    monkeypatch.setattr("services.skill_improvement.load_llm_config", fake_load_llm_config)
    registry = create_default_registry(db_pool)

    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute("SELECT set_config('skills.self_improvement.enabled', 'true'::jsonb)")
            await conn.execute("SELECT set_config('skills.self_improvement.interval_seconds', '1'::jsonb)")
            await conn.execute("DELETE FROM state WHERE key = 'skill_improvement_state'")
            source_ids.extend(await _insert_turns(conn))
            before = await conn.fetchval("SELECT count(*) FROM scheduled_tasks")
            result = await run_skill_improvement_review_step(conn, registry=registry)
            row = await conn.fetchrow(
                """
                SELECT source, status, metadata
                FROM automation_suggestions
                WHERE id = $1::uuid
                """,
                result["suggestion_id"],
            )
            after = await conn.fetchval("SELECT count(*) FROM scheduled_tasks")
        finally:
            await transaction.rollback()

    assert result["status"] == "automation_proposed"
    assert result["created"] is True
    assert row["source"] == "usage"
    assert row["status"] == "pending"
    metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    assert metadata["evidence_count"] == 3
    assert after == before


async def test_weekly_review_lets_model_choose_grounded_changes_but_not_rewrite_them(
    db_pool, monkeypatch
):
    from core.tools import create_default_registry
    from services.skill_improvement import run_skill_improvement_review_step

    selected_memory_id = ""
    captured_context: dict = {}

    async def fake_chat_json(**kwargs):
        nonlocal captured_context
        captured_context = json.loads(kwargs["messages"][1]["content"])
        doc = {
            "proposal": None,
            "automation_suggestion": None,
            "learning_review": {
                "should_review": True,
                "summary": "I learned one stable release preference worth confirming.",
                "memory_ids": [selected_memory_id],
            },
        }
        return doc, json.dumps(doc)

    async def fake_load_llm_config(*_args, **_kwargs):
        return {"provider": "test", "model": "test"}

    monkeypatch.setattr("services.skill_improvement.chat_json", fake_chat_json)
    monkeypatch.setattr("services.skill_improvement.load_llm_config", fake_load_llm_config)
    registry = create_default_registry(db_pool)

    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(
                "SELECT set_config('skills.self_improvement.enabled', 'true'::jsonb)"
            )
            await conn.execute(
                "SELECT set_config('skills.self_improvement.interval_seconds', '1'::jsonb)"
            )
            await conn.execute("DELETE FROM state WHERE key = 'skill_improvement_state'")
            source_ids = await _insert_turns(conn)
            selected_memory_id = str(
                await conn.fetchval(
                    """
                    SELECT create_memory(
                        'semantic',
                        'The operator prefers focused verification before the full suite.',
                        0.7,
                        '{"kind":"user_testimony","label":"Release conversation","trust":1.0}'::jsonb,
                        1.0,
                        '{"confidence":0.9}'::jsonb
                    )
                    """
                )
            )
            await conn.execute(
                """
                INSERT INTO memory_source_units (memory_id, subconscious_unit_id, role)
                VALUES ($1::uuid, $2::uuid, 'source')
                """,
                selected_memory_id,
                source_ids[0],
            )

            result = await run_skill_improvement_review_step(conn, registry=registry)

            assert result["status"] == "learning_review_proposed"
            assert result["learning_review"]["created"] is True
            review_id = result["learning_review"]["review_id"]
            item = await conn.fetchrow(
                """
                SELECT content, kind, source_memory_id
                FROM learning_review_items
                WHERE review_id=$1::uuid
                """,
                review_id,
            )
            assert item["kind"] == "semantic_belief"
            assert str(item["source_memory_id"]) == selected_memory_id
            assert item["content"] == (
                "The operator prefers focused verification before the full suite."
            )
            supplied = {
                candidate["id"]
                for candidate in captured_context["learning_changes"]["candidates"]
            }
            assert selected_memory_id in supplied
        finally:
            await transaction.rollback()
