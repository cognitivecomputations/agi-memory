"""Policy-level retrieval trajectories through the real RLM chat loop.

The LLM responses are scripted for CI stability; everything after the model
decision is real: the RLM loop, REPL, source-document syscalls, RecMem desk, and
metrics. These tests catch wiring/policy regressions that primitive tool tests
cannot see, especially around when to search the cabinet, load the desk, and
reuse already-loaded material.
"""

from __future__ import annotations

import os

import pytest

from evals.retrieval.harness import EvalHarness
from evals.retrieval.trajectory import ScriptedRLM, final_var, repl_block, retrieval_calls
from tests.utils import _db_dsn

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_LLM_CONFIG = {"provider": "openai", "model": "scripted", "api_key": "scripted"}


async def _clear_eval_desk(db_pool) -> None:
    async with db_pool.acquire() as conn:
        await conn.fetchval("SELECT clear_recmem_desk(NULL, NULL, NULL, NULL, TRUE, TRUE)")


async def _run_scripted_chat(monkeypatch, *, user_message: str, responses: list[str], session_id: str | None = None):
    import services.hexis_rlm as rlm

    scripted = ScriptedRLM(responses)
    monkeypatch.setattr(rlm, "_llm_completion", scripted)
    result = await rlm.run_chat_turn(
        user_message=user_message,
        llm_config=_LLM_CONFIG,
        dsn=_db_dsn(os.environ.get("POSTGRES_DB")),
        session_id=session_id,
        max_iterations=5,
        timeout_seconds=60,
    )
    return result, scripted


async def _cleanup_chat_session(session_id: str) -> None:
    import services.hexis_rlm as rlm

    async with rlm._session_lock:
        repl = rlm._chat_sessions.pop(session_id, None)
        rlm._session_last_used.pop(session_id, None)
    if repl is not None:
        repl.cleanup()


async def test_chat_trajectory_exact_source_opens_chunk(monkeypatch, db_pool, corpus, report):
    """Exact source questions should climb to chunk search/fetch and cite."""
    await _clear_eval_desk(db_pool)
    h = EvalHarness(db_pool, "trajectory_exact_source")
    result, _scripted = await _run_scripted_chat(
        monkeypatch,
        user_message="What exactly is the verdigris retention window? Please cite the source.",
        responses=[
            repl_block(
                """
chunks = document_chunk_search("verdigris retention window", limit=5)
target = None
for chunk in chunks:
    if "verdigris retention window" in chunk.get("snippet", "").lower():
        target = chunk
        break
if target is None:
    target = chunks[0]
opened = document_chunk_fetch([target["chunk_id"]])
source = opened["chunks"][0]
heading = " > ".join(source.get("heading_path") or [])
final_answer = source["content"] + "\\n\\nSource: " + source.get("path", "") + " " + heading
"""
            ) + "\n" + final_var(),
        ],
    )

    metrics = result["metrics"]
    h.record.tool_calls = retrieval_calls(metrics)
    h.record.output_chars = len(result["response"])
    passed = (
        corpus["gold"]["exact_phrase"] in result["response"]
        and metrics["document_chunk_search_count"] == 1
        and metrics["document_chunk_fetch_count"] == 1
        and metrics["document_fetch_count"] == 0
        and metrics["document_load_count"] == 0
    )
    report.add(h, passed=passed, metrics=metrics)
    assert passed, result


async def test_chat_trajectory_large_source_loads_desk(monkeypatch, db_pool, corpus, report):
    """Multi-step source work should load selected passages onto the desk."""
    await _clear_eval_desk(db_pool)
    h = EvalHarness(db_pool, "trajectory_large_source_loads_desk")
    result, _scripted = await _run_scripted_chat(
        monkeypatch,
        user_message=(
            "Compare the Northwind escalation threshold with the archive cadence. "
            "Use the source spec rather than relying on memory."
        ),
        responses=[
            repl_block(
                """
docs = document_search("Northwind Operations Specification", limit=3)
spec = docs[0]
escalation = document_chunk_search(
    "northwind escalation threshold",
    document_id=spec["document_id"],
    limit=3,
)
archive = document_chunk_search(
    "northwind archive cadence",
    document_id=spec["document_id"],
    limit=3,
)
chunk_ids = [escalation[0]["chunk_id"], archive[0]["chunk_id"]]
loaded = document_chunk_load_to_desk(
    chunk_ids,
    reason="compare two distant source-spec sections",
    pin=True,
)
desk = desk_list(document_id=spec["document_id"])
opened = []
for item in desk[:2]:
    opened.append(desk_fetch(item["desk_unit_id"], max_chars=1200))
body = "\\n\\n".join(item.get("content", "") for item in opened)
final_answer = body + "\\n\\nSource: " + spec.get("path", "")
"""
            ) + "\n" + final_var(),
        ],
    )

    metrics = result["metrics"]
    h.record.tool_calls = retrieval_calls(metrics)
    h.record.output_chars = len(result["response"])
    passed = (
        "12 incidents" in result["response"]
        and "quarterly" in result["response"]
        and metrics["document_search_count"] == 1
        and metrics["document_chunk_search_count"] == 2
        and metrics["document_chunk_load_count"] == 1
        and metrics["desk_list_count"] >= 1
        and metrics["desk_fetch_count"] >= 1
        and metrics["document_fetch_count"] == 0
    )
    report.add(h, passed=passed, metrics=metrics)
    assert passed, result


async def test_chat_trajectory_followup_reuses_existing_desk(monkeypatch, db_pool, corpus, report):
    """Follow-up turns should inspect the existing desk before reloading."""
    await _clear_eval_desk(db_pool)
    session_id = "eval-retrieval-desk-followup"
    h = EvalHarness(db_pool, "trajectory_followup_reuses_desk")

    try:
        first, _scripted_first = await _run_scripted_chat(
            monkeypatch,
            user_message="Put the Northwind spec on your desk for a few follow-up questions.",
            session_id=session_id,
            responses=[
                repl_block(
                    """
docs = document_search("Northwind Operations Specification", limit=1)
spec = docs[0]
loaded = document_chunk_load_to_desk(
    document_id=spec["document_id"],
    limit=4,
    reason="keep source spec available for follow-up questions",
)
final_answer = "I loaded the source spec onto the desk for follow-up work."
"""
                ) + "\n" + final_var(),
            ],
        )

        second, _scripted_second = await _run_scripted_chat(
            monkeypatch,
            user_message="Now answer from the material already on your desk: what is the retention window?",
            session_id=session_id,
            responses=[
                repl_block(
                    """
desk = desk_list(document_id=spec["document_id"])
picked = None
for item in desk:
    if "verdigris" in item.get("snippet", "").lower() or "retention" in item.get("snippet", "").lower():
        picked = item
        break
if picked is None:
    picked = desk[0]
opened = desk_fetch(picked["desk_unit_id"], max_chars=1200)
final_answer = opened.get("content", "") + "\\n\\nSource: existing RecMem desk item"
"""
                ) + "\n" + final_var(),
            ],
        )
    finally:
        await _cleanup_chat_session(session_id)
        await _clear_eval_desk(db_pool)

    first_metrics = first["metrics"]
    second_metrics = second["metrics"]
    h.record.tool_calls = retrieval_calls(first_metrics) + retrieval_calls(second_metrics)
    h.record.output_chars = len(first["response"]) + len(second["response"])
    passed = (
        first_metrics["document_search_count"] == 1
        and first_metrics["document_chunk_load_count"] == 1
        and second_metrics["desk_list_count"] == 1
        and second_metrics["desk_fetch_count"] == 1
        and second_metrics["document_search_count"] == 0
        and second_metrics["document_fetch_count"] == 0
        and second_metrics["document_chunk_load_count"] == 0
        and corpus["gold"]["exact_phrase"] in second["response"]
    )
    report.add(h, passed=passed, first_metrics=first_metrics, second_metrics=second_metrics)
    assert passed, {"first": first, "second": second}
