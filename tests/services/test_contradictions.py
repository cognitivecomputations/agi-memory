from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from services import contradictions


pytestmark = pytest.mark.asyncio


def _claimed(*, candidates: list[dict] | None = None):
    memory_id = str(uuid4())
    queue_id = str(uuid4())
    return {
        "memory_id": memory_id,
        "queue_id": queue_id,
        "document": {
            "skipped": False,
            "minimum_confidence": 0.83,
            "items": [
                {
                    "queue_id": queue_id,
                    "memory": {
                        "memory_id": memory_id,
                        "content": "The retainer is quarterly.",
                        "type": "semantic",
                    },
                    "candidates": candidates or [],
                }
            ],
        },
    }


async def test_no_candidates_finishes_without_spending_a_model_call(monkeypatch):
    fixture = _claimed()
    conn = AsyncMock()
    conn.fetchval.side_effect = [fixture["document"], {"completed": 1}]
    chat = AsyncMock()
    monkeypatch.setattr(contradictions, "chat_json", chat)

    result = await contradictions.run_contradiction_detection_step(conn, force=True)

    assert result == {"checked": 1, "filed": 0, "reason": "no_candidates"}
    chat.assert_not_awaited()
    assert "finish_contradiction_detection_batch" in conn.fetchval.await_args_list[1].args[0]
    assert conn.fetchval.await_args_list[1].args[1] == [fixture["queue_id"]]


async def test_only_db_supplied_pairs_can_be_filed(monkeypatch):
    candidate_id = str(uuid4())
    unknown_id = str(uuid4())
    fixture = _claimed(
        candidates=[
            {
                "memory_id": candidate_id,
                "content": "The retainer is monthly.",
                "type": "semantic",
            }
        ]
    )
    conn = AsyncMock()
    conn.fetchval.side_effect = [
        fixture["document"],
        {"created": True, "case_id": str(uuid4()), "code": "ABC12345"},
        {"completed": 1},
    ]
    monkeypatch.setattr(contradictions, "load_llm_config", AsyncMock(return_value={}))
    monkeypatch.setattr(
        contradictions,
        "chat_json",
        AsyncMock(
            return_value=(
                {
                    "contradictions": [
                        {
                            "memory_a": fixture["memory_id"],
                            "memory_b": candidate_id,
                            "tension": "Monthly and quarterly cannot both be the current cadence.",
                            "confidence": 0.94,
                        },
                        {
                            "memory_a": fixture["memory_id"],
                            "memory_b": unknown_id,
                            "tension": "This pair was never supplied by the database.",
                            "confidence": 0.99,
                        },
                        {
                            "memory_a": candidate_id,
                            "memory_b": fixture["memory_id"],
                            "tension": "Duplicate of the already filed pair.",
                            "confidence": 0.96,
                        },
                    ]
                },
                '{"contradictions":[]}',
            )
        ),
    )

    result = await contradictions.run_contradiction_detection_step(conn, force=True)

    assert result["checked"] == 1
    assert result["candidate_sets"] == 1
    assert result["filed"] == 1
    assert result["rejected"] == 2
    model_payload = json.loads(
        contradictions.chat_json.await_args.kwargs["messages"][1]["content"].split("\n", 1)[1]
    )
    assert model_payload["minimum_confidence"] == 0.83
    filing = conn.fetchval.await_args_list[1]
    assert "file_contradiction_case" in filing.args[0]
    assert {filing.args[1], filing.args[2]} == {fixture["memory_id"], candidate_id}
    assert filing.args[3] == fixture["memory_id"]
    assert json.loads(filing.args[6])["raw_response_type"] == "str"


async def test_model_failure_returns_queue_to_durable_retry(monkeypatch):
    candidate_id = str(uuid4())
    fixture = _claimed(candidates=[{"memory_id": candidate_id, "content": "Old"}])
    conn = AsyncMock()
    conn.fetchval.side_effect = [fixture["document"], {"retried": 1}]
    monkeypatch.setattr(contradictions, "load_llm_config", AsyncMock(return_value={}))
    monkeypatch.setattr(
        contradictions,
        "chat_json",
        AsyncMock(side_effect=RuntimeError("provider unavailable")),
    )

    result = await contradictions.run_contradiction_detection_step(conn)

    assert result == {
        "failed": True,
        "error": "provider unavailable",
        "checked": 1,
    }
    retry = conn.fetchval.await_args_list[1]
    assert "finish_contradiction_detection_batch" in retry.args[0]
    assert retry.args[1] == [fixture["queue_id"]]
    assert retry.args[2] == "provider unavailable"


async def test_private_reply_resolution_fails_open_for_normal_conversation():
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock()
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value.__aenter__.return_value.fetchval = AsyncMock(
        side_effect=RuntimeError(
        "database unavailable"
        )
    )

    result = await contradictions.resolve_contradiction_from_inbound(
        pool,
        channel="signal",
        actor="operator",
        text="3 ABC12345",
    )

    assert result == {
        "recognized": False,
        "matched": False,
        "reason": "resolution_error",
    }
