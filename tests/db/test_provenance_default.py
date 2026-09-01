from __future__ import annotations

import json
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_source_defaults_preserve_locators_and_vary_by_kind(db_pool):
    async with db_pool.acquire() as conn:
        empty = _json(
            await conn.fetchval("SELECT normalize_source_reference('{}'::jsonb)")
        )
        document = _json(
            await conn.fetchval(
                "SELECT normalize_source_reference($1::jsonb)",
                json.dumps(
                    {
                        "kind": "document",
                        "ref": "contract.pdf",
                        "path": "/cabinet/contracts/contract.pdf",
                        "page_start": 4,
                    }
                ),
            )
        )
        web = _json(
            await conn.fetchval(
                "SELECT normalize_source_reference($1::jsonb)",
                json.dumps({"kind": "web", "ref": "https://example.test"}),
            )
        )

    assert empty == {}
    assert document["path"] == "/cabinet/contracts/contract.pdf"
    assert document["page_start"] == 4
    assert document["trust_origin"] == "default"
    assert document["trust"] != web["trust"]


async def test_semantic_trust_varies_with_confidence_instead_of_plateauing(db_pool):
    marker = uuid4().hex
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            low_id = await conn.fetchval(
                "SELECT create_semantic_memory($1, 0.60, NULL, NULL, $2::jsonb)",
                f"lower confidence {marker}",
                json.dumps([{"kind": "document", "ref": f"doc-low-{marker}"}]),
            )
            high_id = await conn.fetchval(
                "SELECT create_semantic_memory($1, 0.95, NULL, NULL, $2::jsonb)",
                f"higher confidence {marker}",
                json.dumps([{"kind": "document", "ref": f"doc-high-{marker}"}]),
            )
            low, high = await conn.fetchrow(
                """
                SELECT
                    (SELECT trust_level FROM memories WHERE id = $1),
                    (SELECT trust_level FROM memories WHERE id = $2)
                """,
                low_id,
                high_id,
            )
            assert float(low) < float(high)
            assert float(low) != pytest.approx(0.4302279608697066)
            assert float(high) != pytest.approx(0.4302279608697066)
        finally:
            await tr.rollback()


async def test_recall_enrichment_returns_full_provenance_and_stable_citation(db_pool):
    marker = uuid4().hex
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            memory_id = await conn.fetchval(
                "SELECT create_semantic_memory($1, 0.85, NULL, NULL, $2::jsonb)",
                f"citation contract {marker}",
                json.dumps(
                    [
                        {
                            "kind": "document",
                            "ref": f"contract-{marker}",
                            "path": f"/contracts/{marker}.pdf",
                            "page_start": 7,
                        }
                    ]
                ),
            )
            base = _json(
                    await conn.fetchval(
                        "SELECT tool_success(jsonb_build_object('memories', jsonb_build_array(jsonb_build_object('memory_id', $1::text))))",
                        str(memory_id),
                )
            )
            enriched = _json(
                await conn.fetchval(
                    "SELECT enrich_memory_tool_result('recall', $1::jsonb)",
                    json.dumps(base),
                )
            )
            item = enriched["output"]["memories"][0]
            citation = item["citation"]

            assert item["source_attribution"]["path"] == f"/contracts/{marker}.pdf"
            assert item["provenance"] == citation
            assert citation["citation_id"] == f"mem-{memory_id}"
            assert citation["href"] == f"/memories?memory={memory_id}"
            assert citation["trust_level"] == item["provenance"]["trust_level"]
        finally:
            await tr.rollback()


async def test_trust_distribution_is_an_observable_contract(db_pool):
    async with db_pool.acquire() as conn:
        distribution = _json(await conn.fetchval("SELECT memory_trust_distribution()"))

    assert distribution["active_memories"] >= 0
    assert distribution["distinct_trust_levels"] >= 0
    assert isinstance(distribution["by_source_kind"], dict)
