"""N things to ask about is one call, not N calls.

`services/connector_cognition.py` called the model once per item over a query with
`LIMIT 80` — ~160 calls across its two passes where 2–4 would do. These tests pin
the properties that make batching safe to substitute for that loop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.llm_batch import batch_classify, chunk_by_size

CFG = {"provider": "openai", "model": "gpt-4o"}


def _items(n: int, size: int = 10):
    return [{"id": f"i{k}", "text": "x" * size} for k in range(n)]


def _args(**over):
    base = dict(
        llm_config=CFG,
        system="classify",
        key=lambda it: it["id"],
        item_payload=lambda it: {"text": it["text"]},
        parse=lambda raw: raw["verdict"],
        fallback=lambda it: "FALLBACK",
    )
    base.update(over)
    return base


class TestChunking:
    def test_small_items_share_one_chunk(self):
        chunks = chunk_by_size(list(range(80)), size_of=lambda _: 10, budget=1000)
        assert len(chunks) == 1

    def test_large_items_split(self):
        chunks = chunk_by_size(list(range(80)), size_of=lambda _: 500, budget=1000)
        assert len(chunks) == 40

    def test_an_oversized_item_is_not_silently_dropped(self):
        """Better one oversized request the model can refuse than a lost item."""
        chunks = chunk_by_size(
            [1, 2], size_of=lambda i: 10_000 if i == 1 else 5, budget=100
        )
        assert [i for c in chunks for i in c] == [1, 2]


class TestBatching:
    @pytest.mark.asyncio
    async def test_eighty_items_take_one_call(self):
        reply = {"results": {f"i{k}": {"verdict": k} for k in range(80)}}
        with patch(
            "core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))
        ) as m:
            out = await batch_classify(_items(80), **_args())
        assert m.await_count == 1, f"{m.await_count} calls for 80 items"
        assert len(out) == 80
        assert out["i7"] == 7

    @pytest.mark.asyncio
    async def test_an_out_of_order_response_still_maps_correctly(self):
        """The property that makes id-keying non-negotiable."""
        reply = {
            "results": {
                "i2": {"verdict": "C"},
                "i0": {"verdict": "A"},
                "i1": {"verdict": "B"},
            }
        }
        with patch("core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))):
            out = await batch_classify(_items(3), **_args())
        assert out == {"i0": "A", "i1": "B", "i2": "C"}

    @pytest.mark.asyncio
    async def test_a_partial_response_falls_back_only_for_what_is_missing(self):
        reply = {"results": {"i0": {"verdict": "A"}}}
        provenance = {}
        with patch("core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))):
            out = await batch_classify(_items(3), **_args(), provenance=provenance)
        assert out == {"i0": "A", "i1": "FALLBACK", "i2": "FALLBACK"}
        assert provenance == {"i0": "llm", "i1": "fallback", "i2": "fallback"}

    @pytest.mark.asyncio
    async def test_an_explicit_empty_verdict_is_still_an_llm_answer(self):
        reply = {"results": {"i0": {"claims": []}}}
        provenance = {}
        args = _args(parse=lambda doc: doc["claims"], fallback=lambda _item: ["rule"])
        with patch("core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))):
            out = await batch_classify(_items(1), **args, provenance=provenance)
        assert out == {"i0": []}
        assert provenance == {"i0": "llm"}

    @pytest.mark.asyncio
    async def test_a_list_shaped_response_is_accepted(self):
        reply = {
            "results": [{"id": "i1", "verdict": "B"}, {"id": "i0", "verdict": "A"}]
        }
        with patch("core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))):
            out = await batch_classify(_items(2), **_args())
        assert out == {"i0": "A", "i1": "B"}

    @pytest.mark.asyncio
    async def test_unknown_ids_in_the_response_are_ignored(self):
        reply = {"results": {"i0": {"verdict": "A"}, "hallucinated": {"verdict": "X"}}}
        with patch("core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))):
            out = await batch_classify(_items(1), **_args())
        assert out == {"i0": "A"}

    @pytest.mark.asyncio
    async def test_every_item_is_present_even_when_the_call_dies(self):
        with patch(
            "core.llm_batch.chat_json", AsyncMock(side_effect=RuntimeError("boom"))
        ):
            out = await batch_classify(_items(3), **_args())
        assert out == {"i0": "FALLBACK", "i1": "FALLBACK", "i2": "FALLBACK"}

    @pytest.mark.asyncio
    async def test_a_failed_chunk_does_not_take_the_others_with_it(self):
        big = [{"id": f"i{k}", "text": "x" * 20_000} for k in range(2)]
        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:  # first chunk fails both attempts
                raise RuntimeError("boom")
            return ({"results": {"i1": {"verdict": "OK"}}}, "")

        with patch("core.llm_batch.chat_json", AsyncMock(side_effect=flaky)):
            out = await batch_classify(big, **_args())
        assert out["i0"] == "FALLBACK"
        assert out["i1"] == "OK", "a healthy chunk must survive a sibling's failure"

    @pytest.mark.asyncio
    async def test_a_transient_failure_is_retried_once(self):
        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return ({"results": {"i0": {"verdict": "A"}}}, "")

        with patch("core.llm_batch.chat_json", AsyncMock(side_effect=flaky)):
            out = await batch_classify(_items(1), **_args())
        assert out == {"i0": "A"}
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_shared_context_is_sent_once_per_chunk_not_once_per_item(self):
        seen = {}

        async def capture(**kwargs):
            seen["body"] = kwargs["messages"][1]["content"]
            return ({"results": {f"i{k}": {"verdict": k} for k in range(5)}}, "")

        with patch("core.llm_batch.chat_json", AsyncMock(side_effect=capture)):
            await batch_classify(_items(5), **_args(shared={"existing": ["a", "b"]}))
        assert seen["body"].count('"shared_context"') == 1

    @pytest.mark.asyncio
    async def test_no_items_makes_no_call(self):
        with patch("core.llm_batch.chat_json", AsyncMock()) as m:
            out = await batch_classify([], **_args())
        assert out == {}
        assert m.await_count == 0


class TestConnectorCognitionUsesOneCallPerRun:
    """The measured win: ~160 model calls per pass become 2–4.

    `run_connector_user_model_step` and `run_connector_importance_step` each
    called the model once per item over a query with `LIMIT 80`, and re-read
    their config inside the loop for another 240 Postgres round trips.
    """

    @pytest.mark.asyncio
    async def test_claims_extraction_makes_one_call_for_eighty_items(self):
        from services.connector_cognition import extract_user_model_claims_llm_batch

        items = [
            {"source_item_id": f"s{k}", "content": f"message {k}", "title": "t"}
            for k in range(80)
        ]
        reply = {"results": {f"s{k}": {"claims": []} for k in range(80)}}

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        with (
            patch("core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))) as m,
            patch(
                "services.connector_cognition.load_llm_config",
                AsyncMock(return_value=CFG),
            ),
        ):
            out = await extract_user_model_claims_llm_batch(conn, items)

        assert m.await_count == 1, f"{m.await_count} model calls for 80 items"
        assert len(out) == 80
        # One cache read plus one existing-claims read; neither scales with items.
        assert conn.fetch.await_count == 2

    @pytest.mark.asyncio
    async def test_importance_model_is_authoritative_over_fallback_metadata(self):
        """A successful model verdict is not merged with fallback scoring."""
        from services.connector_cognition import (
            estimate_connector_item_importance_llm_batch,
        )

        item = {
            "source_item_id": "s0",
            "content": "ordinary text",
            "raw_metadata": {"priority": "urgent"},
        }
        reply = {
            "results": {
                "s0": {"score": 0.1, "label": "low", "reasons": ["model judgment"]}
            }
        }

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        with (
            patch("core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))),
            patch(
                "services.connector_cognition.load_llm_config",
                AsyncMock(return_value=CFG),
            ),
        ):
            out = await estimate_connector_item_importance_llm_batch(conn, [item])

        assert out["s0"]["score"] == 0.1
        assert out["s0"]["label"] == "low"
        assert out["s0"]["reasons"] == ["model judgment"]
