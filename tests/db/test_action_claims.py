"""Action-claim guardrail (#38): detect_unsupported_action_claims must flag
prose claims of actions with no matching successful tool call in the turn,
and stay quiet when the claim is supported, negated, or merely future tense.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _coerce_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


async def _start_turn(conn, messages: list[dict] | None = None) -> str:
    started = _coerce_json(
        await conn.fetchval(
            "SELECT start_agent_turn('chat', 'test', NULL, $1::jsonb)",
            json.dumps({"messages": messages or []}),
        )
    )
    return started["turn_id"]


async def _apply_call(
    conn, turn_id: str, name: str, arguments: dict, success: bool = True
):
    await conn.fetchval(
        "SELECT apply_agent_tool_result($1::uuid, $2::text, $3::jsonb)",
        turn_id,
        f"call-{name}",
        json.dumps(
            {
                "tool_name": name,
                "arguments": arguments,
                "success": success,
                "energy_spent": 1,
                "model_output": "ok" if success else None,
                "error": None if success else "boom",
            }
        ),
    )


async def _detect(conn, turn_id: str, text: str) -> dict:
    return _coerce_json(
        await conn.fetchval(
            "SELECT detect_unsupported_action_claims($1::uuid, $2::text)",
            turn_id,
            text,
        )
    )


def _kinds(report: dict) -> list[str]:
    return [f["kind"] for f in report.get("flagged", [])]


async def test_memory_claim_without_call_is_flagged(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            report = await _detect(
                conn, turn_id, "I've stored that as a memory for next time."
            )
            assert _kinds(report) == ["memory_write"]
        finally:
            await tr.rollback()


async def test_memory_claim_with_successful_remember_is_clean(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            await _apply_call(conn, turn_id, "remember", {"content": "a fact"})
            report = await _detect(
                conn, turn_id, "I've stored that as a memory for next time."
            )
            assert report["flagged"] == []
            assert report["successful_tool_calls"] == 1
        finally:
            await tr.rollback()


async def test_failed_tool_call_does_not_satisfy_claim(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            await _apply_call(
                conn, turn_id, "remember", {"content": "a fact"}, success=False
            )
            report = await _detect(
                conn, turn_id, "I've stored that as a memory for next time."
            )
            assert _kinds(report) == ["memory_write"]
        finally:
            await tr.rollback()


async def test_futurity_and_negation_are_not_flagged(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            report = await _detect(
                conn,
                turn_id,
                "I will store this as a memory. Let me save that for you. "
                "I have not stored anything yet. Should I remember this?",
            )
            assert report["flagged"] == []
        finally:
            await tr.rollback()


async def test_source_claim_requires_matching_path(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            await _apply_call(
                conn,
                turn_id,
                "inspect_source",
                {"action": "read", "path": "services/prompts/philosophy.md"},
            )
            # Claim about a file that was actually read: clean.
            clean = await _detect(
                conn, turn_id, "I inspected services/prompts/philosophy.md just now."
            )
            assert clean["flagged"] == []
            # Claim about a different file: flagged despite the successful call.
            flagged = await _detect(
                conn, turn_id, "I read core/agent_loop.py at line 42."
            )
            assert _kinds(flagged) == ["source_inspection"]
        finally:
            await tr.rollback()


async def test_mcp_wildcard_satisfies_external_send(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            await _apply_call(
                conn, turn_id, "mcp_github_create_issue", {"title": "bug"}
            )
            report = await _detect(conn, turn_id, "I've filed the issue on GitHub.")
            assert report["flagged"] == []
        finally:
            await tr.rollback()


async def test_fabricated_uuid_is_flagged_unless_grounded(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            fabricated = "900e379e-3019-4dca-928f-5f5429510b5c"
            report = await _detect(conn, turn_id, f"Created backlog item {fabricated}.")
            assert "fabricated_artifact" in _kinds(report)

            # Same UUID present in a tool message is grounded: clean.
            grounded = "11111111-2222-4333-8444-555555555555"
            await _apply_call(conn, turn_id, "remember", {"content": "x"})
            await conn.fetchval(
                "SELECT apply_agent_tool_result($1::uuid, $2::text, $3::jsonb)",
                turn_id,
                "call-grounded",
                json.dumps(
                    {
                        "tool_name": "remember",
                        "arguments": {"content": "y"},
                        "success": True,
                        "energy_spent": 1,
                        "model_output": f"stored with id {grounded}",
                    }
                ),
            )
            report2 = await _detect(
                conn, turn_id, f"I've stored it as memory {grounded}."
            )
            assert report2["flagged"] == []
        finally:
            await tr.rollback()


async def test_unknown_turn_fails_soft(db_pool):
    async with db_pool.acquire() as conn:
        report = await _detect(
            conn, "00000000-0000-0000-0000-000000000000", "I've stored that."
        )
        assert report["flagged"] == []
        assert report.get("error") == "turn_not_found"


async def test_tool_call_arguments_recorded_in_runtime_state(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            await _apply_call(conn, turn_id, "inspect_source", {"path": "core/llm.py"})
            calls = _coerce_json(
                await conn.fetchval(
                    "SELECT runtime_state->'tool_calls_made' FROM agent_turns WHERE id = $1::uuid",
                    turn_id,
                )
            )
            assert calls[0]["name"] == "inspect_source"
            assert calls[0]["arguments"] == {"path": "core/llm.py"}
        finally:
            await tr.rollback()


async def test_markdown_negation_is_not_flagged(db_pool):
    """The live false positive (#48): truthful markdown-bold negation stays clean."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            await _apply_call(conn, turn_id, "recall", {"query": "philosophy"})
            report = await _detect(
                conn,
                turn_id,
                "No. I inspected `services/prompts/philosophy.md`, but I did **not** "
                "create a normal persistent memory from it—neither an explicit "
                "`remember` entry nor a document-ingested semantic memory.",
            )
            assert report["flagged"] == []

        finally:
            await tr.rollback()


async def test_historical_action_claim_requires_durable_receipt(db_pool):
    memory_id = "11111111-2222-4333-8444-555555555555"
    receipt = {
        "name": "remember",
        "success": True,
        "arguments": {"content": "the lighthouse agreement"},
        "result": {"memory_id": memory_id},
    }

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            unsupported_turn = await _start_turn(conn)
            unsupported = await _detect(
                conn,
                unsupported_turn,
                "I already stored that as a memory for next time.",
            )
            assert _kinds(unsupported) == ["memory_write"]

            supported_turn = await _start_turn(
                conn,
                [
                    {
                        "role": "assistant",
                        "content": "Stored it.",
                        "metadata": {"action_receipts": [receipt]},
                    }
                ],
            )
            supported = await _detect(
                conn,
                supported_turn,
                f"I already stored that as memory {memory_id}.",
            )
            assert supported["flagged"] == []
            assert supported["prior_action_receipts"] == 1
        finally:
            await tr.rollback()


async def test_negative_search_claims_require_a_search(db_pool):
    """#50: 'I searched and found nothing' is a claim like any other."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            turn_id = await _start_turn(conn)
            unbacked = await _detect(
                conn,
                turn_id,
                "I searched the repository and it returns no matching path or text.",
            )
            assert _kinds(unbacked) == ["search_negative"]

            await _apply_call(
                conn,
                turn_id,
                "inspect_source",
                {"action": "search", "query": "philosophy"},
            )
            backed = await _detect(
                conn,
                turn_id,
                "I searched the repository and it returns no matching path or text.",
            )
            assert backed["flagged"] == []
        finally:
            await tr.rollback()


async def test_correction_claim_requires_revision_not_just_any_write(db_pool):
    """#67: 'I've corrected that in my memory' passed because an unrelated
    remember succeeded — correction claims are only satisfied by add_evidence."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            claim = "I've corrected that attribution in my memory."

            turn_id = await _start_turn(conn)
            await _apply_call(
                conn, turn_id, "remember", {"content": "a new unrelated note"}
            )
            report = await _detect(conn, turn_id, claim)
            assert "memory_correction" in _kinds(report)

            turn_id = await _start_turn(conn)
            await _apply_call(
                conn,
                turn_id,
                "add_evidence",
                {"memory_id": "x", "stance": "contradicts"},
            )
            report = await _detect(conn, turn_id, claim)
            assert "memory_correction" not in _kinds(report)
        finally:
            await tr.rollback()
