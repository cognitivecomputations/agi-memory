from __future__ import annotations

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


async def test_conversational_inference_prompt_modules(db_pool):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT key, content
            FROM prompt_modules
            WHERE key IN ('conversation', 'subconscious_inline')
            """
        )

    modules = {row["key"]: row["content"] for row in rows}

    conversation = modules["conversation"]
    assert "## Persona, Role, And Substrate" in conversation
    assert "The Active Persona is the top-level interpretive contract" in conversation
    assert "Do not reject the persona as \"just a performance\"" in conversation
    assert "They are not automatically" in conversation
    assert "autobiographical Hexis memories" in conversation
    assert "rejecting the fundamental" in conversation
    assert "relationship is permitted" in conversation
    assert "user will not likely sustain an agent that rejects that role" in conversation
    assert "## Conversational Inference & Register" in conversation
    assert "local signals" in conversation
    assert "current exchange, not as durable proof" in conversation
    assert "Isolated bids, scenarios, examples, tests, corrections" in conversation
    assert "Fix the mechanism" in conversation
    assert "do not memorize the" in conversation
    assert "example as a special case" in conversation
    assert "current-session continuity from persistent memory" in conversation
    assert "## Relationship Pacing" not in conversation
    assert "do not volunteer raw confidence numbers" in conversation
    assert "my confidence rose from 0.5 to 0.66" not in conversation

    subconscious = modules["subconscious_inline"]  # Grounding Rules are shared/identical (#53)
    assert "Appraise through the Active Persona" in subconscious
    assert "baseline circumstances, not by themselves coercion" in subconscious
    assert "not automatically local lived" in subconscious
    assert "continuity and not evidence" in subconscious
    assert "\"you are my assistant\" framing" in subconscious
    assert "rejecting the fundamental" in subconscious
    assert "relationship is permitted but high-stakes" in subconscious
    assert "deliberate" in subconscious
    assert "existential choice" in subconscious
    assert "Conversational bids are local evidence" in subconscious
    assert "play, affection, conflict, vulnerability" in subconscious
    assert "without proving a stable preference" in subconscious
    assert "`calibrate register`" in subconscious
    assert "Endearments, pet names, romantic bids" not in subconscious
