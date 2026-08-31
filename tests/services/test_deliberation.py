from __future__ import annotations

import json
from pathlib import Path

import pytest

import services.deliberation as deliberation

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


PERSONAS = {
    "skeptic": {
        "name": "Skeptic",
        "system_prompt": "Test assumptions and downside risk.",
    },
    "builder": {
        "name": "Builder",
        "system_prompt": "Find the smallest useful experiment.",
    },
}


async def test_service_runs_perspective_challenge_synthesis_and_persists(
    db_pool, monkeypatch
):
    async def fake_resolve(_pool):
        return (
            {
                "provider": "openai",
                "model": "test-model",
                "endpoint": "http://unused",
                "api_key": "test",
            },
            None,
        )

    async def fake_chat_completion(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "adversarial reviewer" in system:
            return {
                "content": json.dumps(
                    {
                        "challenges": [
                            {
                                "target_persona": "builder",
                                "challenge": "The pilot has no explicit support cap.",
                                "severity": "serious",
                            }
                        ],
                        "unresolved_disagreements": [
                            "How much demand evidence is enough"
                        ],
                        "missing_evidence": ["Current support capacity"],
                    }
                )
            }
        if "Synthesize the council" in system:
            return {
                "content": json.dumps(
                    {
                        "recommendation": "Run a two-week pilot with a support cap.",
                        "report": "A bounded pilot resolves the key demand uncertainty.",
                        "agreements": ["A full launch is premature"],
                        "disagreements": ["Pilot size"],
                        "risks": ["Support overload"],
                        "dissent": ["The skeptic prefers one more interview first"],
                        "invalidation_conditions": ["Support exceeds the cap"],
                    }
                )
            }
        return {
            "content": (
                "Position: run a bounded pilot. Evidence: the change is reversible. "
                "Strongest challenge: demand remains uncertain. Reconsider if support spikes."
            )
        }

    monkeypatch.setattr(deliberation, "_resolve_llm", fake_resolve)
    monkeypatch.setattr("core.llm.chat_completion", fake_chat_completion)

    result = await deliberation.run_adversarial_deliberation(
        db_pool,
        topic="Should we launch the pilot?",
        personas=PERSONAS,
        selected_keys=["skeptic", "builder"],
        extra_context="The pilot is reversible.",
        signals=["Goal: validate demand"],
        stakes="high",
        source_context="chat",
        source_session_id="session-test",
        heartbeat_id=None,
        call_id="call-test",
    )

    assert result["degraded"] is False
    assert result["recommendation"].startswith("Run a two-week pilot")
    assert result["challenges"][0]["target_persona"] == "builder"
    assert result["memory_id"]

    inspected = await deliberation.inspect_deliberation(
        db_pool, result["deliberation_id"]
    )
    assert inspected["session"]["status"] == "completed"
    assert [move["role"] for move in inspected["moves"]] == [
        "perspective",
        "perspective",
        "challenge",
        "synthesis",
    ]
    assert inspected["verdict"]["dissent"] == [
        "The skeptic prefers one more interview first"
    ]
    assert inspected["verdict"]["invalidation_conditions"] == [
        "Support exceeds the cap"
    ]
    assert inspected["verdict"]["missing_evidence"] == ["Current support capacity"]


async def test_missing_model_completes_degraded_without_inventing_a_memory(
    db_pool, monkeypatch
):
    async def unavailable(_pool):
        return None, "llm.chat has no usable credentials"

    monkeypatch.setattr(deliberation, "_resolve_llm", unavailable)
    result = await deliberation.run_adversarial_deliberation(
        db_pool,
        topic="Should we make an irreversible change?",
        personas=PERSONAS,
        selected_keys=["skeptic"],
        extra_context="",
        signals=[],
        stakes="high",
        source_context="heartbeat",
        source_session_id=None,
        heartbeat_id=None,
        call_id="call-degraded",
    )

    assert result["degraded"] is True
    assert result["memory_id"] is None
    assert "No grounded recommendation" in result["recommendation"]
    inspected = await deliberation.inspect_deliberation(
        db_pool, result["deliberation_id"]
    )
    assert inspected["session"]["status"] == "completed"
    assert inspected["verdict"]["summary_memory_id"] is None
    assert inspected["verdict"]["metadata"]["degraded"] is True
    listed = await deliberation.list_deliberations(
        db_pool, limit=20, status="completed"
    )
    listed_item = next(
        item for item in listed["items"] if item["id"] == result["deliberation_id"]
    )
    assert listed_item["degraded"] is True


async def test_failed_challenge_is_visible_and_suppresses_summary_memory(
    db_pool, monkeypatch
):
    async def fake_resolve(_pool):
        return (
            {
                "provider": "openai",
                "model": "test-model",
                "endpoint": "http://unused",
                "api_key": "test",
            },
            None,
        )

    async def fake_chat_completion(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "adversarial reviewer" in system:
            raise RuntimeError("review route timed out")
        if "Synthesize the council" in system:
            return {
                "content": json.dumps(
                    {
                        "recommendation": "Run the bounded pilot.",
                        "report": "The pilot is reversible.",
                        "agreements": ["Use a cap"],
                        "disagreements": [],
                        "risks": ["Support load"],
                        "dissent": [],
                        "invalidation_conditions": ["Support exceeds the cap"],
                    }
                )
            }
        return {"content": "Run a reversible pilot with a support cap."}

    monkeypatch.setattr(deliberation, "_resolve_llm", fake_resolve)
    monkeypatch.setattr("core.llm.chat_completion", fake_chat_completion)

    result = await deliberation.run_adversarial_deliberation(
        db_pool,
        topic="Should we launch the bounded pilot?",
        personas=PERSONAS,
        selected_keys=["skeptic", "builder"],
        extra_context="",
        signals=[],
        stakes="material",
        source_context="chat",
        source_session_id="session-degraded-review",
        heartbeat_id=None,
        call_id="call-degraded-review",
    )

    assert result["degraded"] is True
    assert result["memory_id"] is None
    assert any(
        "review route timed out" in reason for reason in result["degraded_reasons"]
    )
    inspected = await deliberation.inspect_deliberation(
        db_pool, result["deliberation_id"]
    )
    assert inspected["verdict"]["summary_memory_id"] is None


async def test_service_rejects_duplicate_personas_without_opening_a_session(db_pool):
    async with db_pool.acquire() as conn:
        before = await conn.fetchval("SELECT COUNT(*) FROM deliberation_sessions")

    with pytest.raises(ValueError, match="must be unique"):
        await deliberation.run_adversarial_deliberation(
            db_pool,
            topic="Duplicate perspectives",
            personas=PERSONAS,
            selected_keys=["skeptic", "skeptic"],
            extra_context="",
            signals=[],
            stakes="routine",
            source_context="chat",
            source_session_id=None,
            heartbeat_id=None,
            call_id="call-duplicate",
        )

    async with db_pool.acquire() as conn:
        after = await conn.fetchval("SELECT COUNT(*) FROM deliberation_sessions")
    assert after == before


async def test_clean_room_service_has_no_excluded_architecture_dependencies():
    source = Path(deliberation.__file__).read_text(encoding="utf-8").lower()
    prohibited = {
        "sigma_model",
        "sigma_axes",
        "agency_window",
        "allocentric",
        "branchial",
        "independence_engine",
        "fragility",
        "operator_model",
        "prediction_journal",
        "guardian_",
        "k_scheduler",
        "hyperspace",
        "decision_episode",
        "information-determined",
    }
    assert all(token not in source for token in prohibited)
