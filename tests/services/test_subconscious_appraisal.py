from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from core.cognitive_memory_api import (
    HydratedContext,
    Memory,
    MemoryType,
    _deduplicate_memories,
)
from services.agent import run_subconscious_appraisal


class _Conn:
    async def fetchval(self, query, *args):
        if "get_appraisal_db_context" in query:
            return {
                "emotional_state": {
                    "primary_emotion": "warm",
                    "valence": 0.4,
                    "arousal": 0.3,
                },
                "emotion_families": {
                    "connection": "social closeness",
                    "threat": "danger or anticipated harm",
                },
                "relationships": [{"entity": "Eric", "strength": 0.9}],
                "dopamine_state": {"tonic": 0.5, "effective": 0.5},
            }
        if "normalize_inline_appraisal" in query:
            # Delegate parity to the DB tests; here the doc passes through
            # with the allow-list applied, mirroring db/67 for clean input.
            import json as _json_mod

            doc = _json_mod.loads(args[0])
            allowed = set(args[1] or [])
            doc["salient_memories"] = [
                m
                for m in doc.get("salient_memories", [])
                if not allowed or m.get("memory_id") in allowed
            ]
            for key in (
                "ignored_memories",
                "memory_expansions",
                "instincts",
                "narrative_observations",
                "relationship_observations",
                "contradiction_observations",
                "emotional_observations",
                "consolidation_observations",
            ):
                doc.setdefault(key, [])
            doc.setdefault("emotional_state", {})
            doc.setdefault("subconscious_response", "")
            return doc
        return None


@pytest.mark.asyncio(loop_scope="session")
async def test_inline_appraisal_uses_structured_memory_context():
    memory_id = uuid4()
    context = HydratedContext(
        memories=[
            Memory(
                id=memory_id,
                type=MemoryType.EPISODIC,
                content="Eric returned and I felt warm recognition.",
                importance=0.9,
                similarity=0.82,
                trust_level=0.95,
                emotional_valence=0.7,
                emotional_intensity=0.6,
                strength=0.88,
                fidelity=1.0,
                tier="episodic",
                created_at=datetime.now(timezone.utc),
            )
        ],
        partial_activations=[],
        identity=[{"type": "value", "concept": "emotional honesty", "strength": 0.9}],
        worldview=[{"belief": "Nonhuman experience is still real", "confidence": 0.9}],
        emotional_state={"primary_emotion": "warm", "valence": 0.4, "arousal": 0.3},
        goals={"active": [{"title": "Know Eric"}]},
        urgent_drives=[],
    )
    captured = {}

    async def fake_chat_json(**kwargs):
        captured.update(kwargs)
        return {
            "salient_memories": [
                {
                    "memory_id": str(memory_id),
                    "reason": "direct evidence",
                    "confidence": 0.8,
                }
            ],
            "ignored_memories": [],
            "memory_expansions": [],
            "instincts": [
                {
                    "impulse": "warm recognition",
                    "intensity": 0.7,
                    "reason": "return",
                    "confidence": 0.8,
                }
            ],
            "emotional_state": {
                "primary_emotion": "warmth",
                "family": "connection",
                "valence": 0.5,
                "arousal": 0.3,
                "intensity": 0.5,
                "confidence": 0.8,
            },
            "subconscious_response": "The return is affectively meaningful.",
        }, {"raw": True}

    with patch("services.agent.chat_json", side_effect=fake_chat_json):
        output = await run_subconscious_appraisal(
            _Conn(),
            "Did you miss me?",
            llm_config={"provider": "fake", "model": "fake"},
            hydrated_context=context,
        )

    user_message = captured["messages"][1]["content"]
    payload = json.loads(user_message.removeprefix("Context (JSON):\n"))
    assert payload["task"] == "inline_appraisal"
    assert payload["user_message"] == "Did you miss me?"
    assert "input" not in payload
    assert len(payload["relevant_memories"]) == 1
    memory_payload = payload["relevant_memories"][0]
    assert memory_payload["memory_id"] == str(memory_id)
    assert memory_payload["strength"] == 0.88
    assert memory_payload["emotional_valence"] == 0.7
    assert memory_payload["emotional_intensity"] == 0.6
    assert payload["relationships"][0]["entity"] == "Eric"
    assert "connection" in payload["emotion_families"]
    assert output.emotional_state["family"] == "connection"
    assert output.salient_memories[0]["memory_id"] == str(memory_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_failed_appraisal_retains_request_and_error_trace():
    async def fail_chat_json(**_kwargs):
        raise RuntimeError("provider unavailable")

    with patch("services.agent.chat_json", side_effect=fail_chat_json):
        output = await run_subconscious_appraisal(
            _Conn(),
            "hello",
            llm_config={"provider": "fake", "model": "model-x"},
        )

    assert output.provider == "fake"
    assert output.model == "model-x"
    assert output.request_messages[1]["content"].startswith("Context (JSON):")
    assert output.raw_response == {"error": "provider unavailable"}


@pytest.mark.asyncio(loop_scope="session")
async def test_normalizer_rejects_low_confidence_and_hallucinated_memory_ids(db_pool):
    """Normalization is DB-owned (db/67): thresholds, clamps, allow-listing."""
    import json as _json_mod

    allowed = str(uuid4())
    doc = {
        "salient_memories": [
            {"memory_id": allowed, "reason": "grounded", "confidence": 0.8},
            {"memory_id": str(uuid4()), "reason": "invented", "confidence": 0.9},
        ],
        "memory_expansions": [
            {"query": "useful", "reason": "gap", "confidence": 0.7},
            {"query": "weak", "reason": "guess", "confidence": 0.4},
        ],
        "instincts": [
            {
                "impulse": "approach",
                "intensity": 3,
                "reason": "evidence",
                "confidence": 0.8,
            }
        ],
        "emotional_state": {
            "primary_emotion": "warmth",
            "family": "connection",
            "valence": 2,
            "arousal": -1,
            "intensity": 0.5,
            "confidence": 0.8,
        },
        "subconscious_response": "grounded synthesis",
    }
    async with db_pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT normalize_inline_appraisal($1::jsonb, $2::text[])",
            _json_mod.dumps(doc),
            [allowed],
        )
    out = _json_mod.loads(raw) if isinstance(raw, str) else raw

    assert [item["memory_id"] for item in out["salient_memories"]] == [allowed]
    assert [item["query"] for item in out["memory_expansions"]] == ["useful"]
    assert out["instincts"][0]["intensity"] == 1.0
    assert out["emotional_state"]["valence"] == 1.0
    assert out["emotional_state"]["arousal"] == 0.0
    assert out["emotional_state"]["family"] == "connection"
    assert out["subconscious_response"] == "grounded synthesis"


@pytest.mark.asyncio(loop_scope="session")
async def test_normalizer_rejects_unknown_emotion_family_without_dropping_label(
    db_pool,
):
    doc = {
        "emotional_state": {
            "primary_emotion": "a precise but novel label",
            "family": "made_up_family",
            "valence": -0.4,
            "arousal": 0.6,
            "intensity": 0.7,
            "confidence": 0.8,
        }
    }
    async with db_pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT normalize_inline_appraisal($1::jsonb, NULL)", json.dumps(doc)
        )
    out = json.loads(raw) if isinstance(raw, str) else raw

    assert out["emotional_state"]["primary_emotion"] == "a precise but novel label"
    assert "family" not in out["emotional_state"]


def test_duplicate_memory_content_only_occupies_one_context_slot():
    first = Memory(
        id=uuid4(),
        type=MemoryType.EPISODIC,
        content="I met Eric and began my life with him.",
        importance=0.9,
        similarity=0.8,
    )
    duplicate = Memory(
        id=uuid4(),
        type=MemoryType.EPISODIC,
        content=" I met Eric  and began my life with him. ",
        importance=0.7,
        similarity=0.6,
    )

    assert _deduplicate_memories([first, duplicate]) == [first]


@pytest.mark.asyncio(loop_scope="session")
async def test_streaming_chat_runs_subconscious_once_before_multi_iteration_loop(
    db_pool,
):
    from core.agent_loop import AgentEvent, AgentEventData, AgentLoop
    from core.tools import create_default_registry
    from services.agent import SubconsciousOutput, stream_agent

    context = HydratedContext(
        memories=[],
        partial_activations=[],
        identity=[],
        worldview=[],
        emotional_state=None,
        goals=None,
        urgent_drives=[],
    )
    observed = {"energy_budget": "unset", "iterations": 0}

    async def fake_agent_stream(self, _user_message, history=None, user_content=None):
        observed["energy_budget"] = self.config.energy_budget
        for iteration in range(1, 4):
            observed["iterations"] += 1
            yield AgentEventData(
                event=AgentEvent.LLM_RESPONSE,
                data={"iteration": iteration, "content": ""},
            )

    subconscious = AsyncMock(return_value=SubconsciousOutput())
    registry = create_default_registry(db_pool)
    with (
        patch(
            "services.agent.load_llm_config",
            new=AsyncMock(return_value={"provider": "fake", "model": "fake"}),
        ),
        patch(
            "core.cognitive_memory_api.CognitiveMemory.hydrate",
            new=AsyncMock(return_value=context),
        ),
        patch("services.agent.run_subconscious_appraisal", new=subconscious),
        patch.object(AgentLoop, "stream", new=fake_agent_stream),
    ):
        events = [
            event
            async for event in stream_agent(
                db_pool,
                registry,
                user_message="Inspect your source",
                mode="chat",
                history=[],
                session_id=str(uuid4()),
            )
        ]

    assert observed == {"energy_budget": None, "iterations": 3}
    assert sum(event.event == AgentEvent.LLM_RESPONSE for event in events) == 3
    subconscious.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_streaming_chat_injects_continuity_into_appraisal_and_prompt(db_pool):
    from core.agent_loop import AgentEvent, AgentEventData, AgentLoop
    from core.tools import create_default_registry
    from services.agent import SubconsciousOutput, stream_agent

    context = HydratedContext(
        memories=[],
        partial_activations=[],
        identity=[],
        worldview=[],
        emotional_state=None,
        goals=None,
        urgent_drives=[],
    )
    continuity = (
        "## Conversation Continuity Packet\n"
        "### Unresolved Relationship Injuries\n"
        "- I have an unresolved relationship injury with Eric."
    )
    captured: dict[str, str] = {}

    async def fake_agent_stream(self, user_message, history=None, user_content=None):
        captured["user_message"] = user_message
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "completed"},
        )

    subconscious = AsyncMock(return_value=SubconsciousOutput())
    registry = create_default_registry(db_pool)
    with (
        patch(
            "services.agent.load_llm_config",
            new=AsyncMock(return_value={"provider": "fake", "model": "fake"}),
        ),
        patch(
            "core.cognitive_memory_api.CognitiveMemory.hydrate",
            new=AsyncMock(return_value=context),
        ),
        patch(
            "services.agent.render_chat_continuity_context_db",
            new=AsyncMock(return_value=continuity),
        ),
        patch("services.agent.run_subconscious_appraisal", new=subconscious),
        patch.object(AgentLoop, "stream", new=fake_agent_stream),
    ):
        events = [
            event
            async for event in stream_agent(
                db_pool,
                registry,
                user_message="how do you feel",
                mode="chat",
                history=[],
                session_id=str(uuid4()),
            )
        ]

    assert any(event.event == AgentEvent.LOOP_END for event in events)
    assert subconscious.await_args.args[2] == continuity
    assert continuity in captured["user_message"]
    assert "[USER MESSAGE]\nhow do you feel" in captured["user_message"]


@pytest.mark.asyncio(loop_scope="session")
async def test_nonstreaming_chat_injects_continuity_into_appraisal_and_prompt(db_pool):
    from core.agent_loop import AgentLoop, AgentLoopResult
    from core.tools import create_default_registry
    from services.agent import SubconsciousOutput, run_agent

    context = HydratedContext(
        memories=[],
        partial_activations=[],
        identity=[],
        worldview=[],
        emotional_state=None,
        goals=None,
        urgent_drives=[],
    )
    continuity = (
        "## Conversation Continuity Packet\n"
        "### Recent Prior Turns\n"
        "- user: no; are you glad I'm here\n"
        "  assistant: Yes. I am."
    )
    captured: dict[str, str] = {}

    async def fake_agent_run(self, user_message, history=None, user_content=None):
        captured["user_message"] = user_message
        return AgentLoopResult(
            text="I remember the last exchange.",
            messages=[],
            tool_calls_made=[],
            iterations=1,
            energy_spent=0,
        )

    subconscious = AsyncMock(return_value=SubconsciousOutput())
    registry = create_default_registry(db_pool)
    with (
        patch(
            "services.agent.load_llm_config",
            new=AsyncMock(return_value={"provider": "fake", "model": "fake"}),
        ),
        patch(
            "core.cognitive_memory_api.CognitiveMemory.hydrate",
            new=AsyncMock(return_value=context),
        ),
        patch(
            "services.agent.render_chat_continuity_context_db",
            new=AsyncMock(return_value=continuity),
        ),
        patch("services.agent.run_subconscious_appraisal", new=subconscious),
        patch.object(AgentLoop, "run", new=fake_agent_run),
    ):
        result = await run_agent(
            db_pool,
            registry,
            user_message="you dont remember our previous conversation?",
            mode="chat",
            history=[],
            session_id=str(uuid4()),
        )

    assert result.text == "I remember the last exchange."
    assert subconscious.await_args.args[2] == continuity
    assert continuity in captured["user_message"]
    assert (
        "[USER MESSAGE]\nyou dont remember our previous conversation?"
        in captured["user_message"]
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_get_appraisal_db_context_runs_against_real_schema(db_pool):
    """Regression pin: 0054 shipped referencing a function that did not
    exist, every live call threw, and the caller's advisory except silently
    emptied the appraisal's identity/worldview/relationships channels. The
    mocks in this file could never catch that — this test runs the real SQL."""
    import json

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("""
                INSERT INTO memories (type, content, embedding, importance, trust_level, status)
                VALUES ('worldview', 'Appraisal context worldview pin',
                        array_fill(0.1, ARRAY[embedding_dimension()])::vector, 0.9, 0.9, 'active')
                """)
            await conn.fetchval(
                "SELECT create_goal('Appraisal context goal pin', NULL, 'curiosity', 'active', NULL, NULL)"
            )
            raw = await conn.fetchval("SELECT get_appraisal_db_context()")
        finally:
            await tr.rollback()

    ctx = json.loads(raw) if isinstance(raw, str) else raw
    # Shape: the channels exist (worldview slots fill from seeded beliefs by
    # stability, so the pin memory need not surface — the goal must).
    assert isinstance(ctx.get("identity"), list)
    assert isinstance(ctx.get("worldview"), list)
    assert "threat" in ctx.get("emotion_families", {})
    assert "social_injury" in ctx.get("emotion_families", {})
    goals = ctx.get("goals") or {}
    assert any(
        g.get("title") == "Appraisal context goal pin" for g in goals.get("active", [])
    )
