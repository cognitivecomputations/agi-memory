from __future__ import annotations

import json
from typing import Any

import pytest

from services import connector_cognition as cognition
from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


def _j(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


async def _stub_get_embedding(conn) -> None:
    await conn.execute("""
        CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
        RETURNS vector[] AS $$
            SELECT COALESCE(
                array_agg((
                    ARRAY[1.0::float] ||
                    array_fill(0.0::float, ARRAY[embedding_dimension() - 1])
                )::vector),
                ARRAY[]::vector[]
            )
            FROM unnest(text_contents)
        $$ LANGUAGE sql;
        """)


async def _set_json_config(conn, key: str, value: Any) -> None:
    await conn.execute("SELECT set_config($1, $2::jsonb)", key, json.dumps(value))


async def _connect_slack(conn, marker: str, account_key: str) -> None:
    attempt = _j(
        await conn.fetchval(
            """
        SELECT start_connection_attempt(
            'slack',
            '["live_chat", "send", "ingest_live"]'::jsonb,
            ARRAY[]::text[],
            '{}'::jsonb,
            NULL,
            NULL,
            'test',
            $1,
            CURRENT_TIMESTAMP + INTERVAL '10 minutes'
        )
        """,
            marker,
        )
    )
    await conn.fetchval(
        """
        SELECT complete_connection_attempt(
            $1::uuid,
            $2,
            'Slack trajectory eval',
            'config:channel.slack',
            ARRAY[]::text[],
            '["live_chat", "send", "ingest_live"]'::jsonb,
            '{"test": true}'::jsonb
        )
        """,
        attempt["attempt_id"],
        account_key,
    )


async def _source_item(
    conn,
    *,
    account_key: str,
    provider_item_id: str,
    content: str,
    offset_seconds: int,
    sensitivity: str = "private",
) -> dict[str, Any]:
    raw = await conn.fetchval(
        """
        SELECT upsert_connector_source_item(
            'slack',
            $1,
            $2,
            $3,
            $4,
            'message',
            'eval-thread',
            CURRENT_TIMESTAMP + make_interval(secs => $5),
            ARRAY['trajectory-eval']::text[],
            '[{"role": "sender", "id": "Eric"}]'::jsonb,
            '[]'::jsonb,
            $6::jsonb,
            $7,
            FALSE
        )
        """,
        account_key,
        provider_item_id,
        f"Trajectory source {provider_item_id}",
        content,
        offset_seconds,
        json.dumps({"trajectory_eval": True}),
        sensitivity,
    )
    return _j(raw)


async def _run_synthesis_until_idle(conn, *, limit: int = 10) -> dict[str, int]:
    totals = {"claimed": 0, "completed": 0, "failed": 0, "claims": 0, "llm_used": 0}
    for _ in range(10):
        result = await cognition.run_user_model_synthesis_step(conn, limit=limit)
        if result.get("skipped"):
            return totals
        for key in totals:
            totals[key] += int(result.get(key) or 0)
    raise AssertionError("user-model synthesis did not drain within 10 passes")


async def _claims(conn, marker: str) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT c.id::text AS id,
               c.claim_key,
               c.category,
               c.claim,
               c.status,
               c.review_status,
               c.evidence_count,
               c.evidence_refs,
               c.contradiction_refs,
               c.superseded_by::text AS superseded_by,
               c.memory_id::text AS memory_id,
               m.status::text AS memory_status
        FROM user_model_claims c
        JOIN memories m ON m.id = c.memory_id
        WHERE c.claim LIKE $1 OR c.claim_key LIKE $1
        ORDER BY c.created_at, c.claim_key
        """,
        f"%{marker}%",
    )
    return [dict(row) for row in rows]


async def _approve_all(conn, claims: list[dict[str, Any]]) -> None:
    for claim in claims:
        if claim["status"] == "active" and claim["review_status"] == "pending_review":
            await conn.fetchval(
                "SELECT review_user_model_claim($1::uuid, 'approve', 'trajectory eval', 'test')",
                claim["id"],
            )


async def test_rules_trajectory_preserves_evidence_and_review_gate(db_pool):
    marker = get_test_identifier("user_model_rules_traj")
    account = f"channel:slack:{marker}"
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await _set_json_config(conn, "connector.user_model_synthesis_mode", "rules")
            await _set_json_config(conn, "connector.user_model_llm_enabled", False)
            await _connect_slack(conn, marker, account)
            first = await _source_item(
                conn,
                account_key=account,
                provider_item_id=f"{marker}-brief-1",
                offset_seconds=1,
                content=f"Message:\nI prefer written project briefs before calls {marker}.",
            )
            second = await _source_item(
                conn,
                account_key=account,
                provider_item_id=f"{marker}-brief-2",
                offset_seconds=2,
                content=f"Message:\nI prefer written project briefs before calls {marker}.",
            )
            await _source_item(
                conn,
                account_key=account,
                provider_item_id=f"{marker}-routine",
                offset_seconds=3,
                content=f"Message:\nI usually block quiet Monday planning time {marker}.",
            )

            totals = await _run_synthesis_until_idle(conn)
            before_context = _j(
                await conn.fetchval("SELECT get_approved_user_model_context(20)")
            )
            before_rendered = await conn.fetchval(
                "SELECT render_user_model_context($1::jsonb)",
                json.dumps(before_context),
            )
            claims = await _claims(conn, marker)
            await _approve_all(conn, claims)
            after_context = _j(
                await conn.fetchval("SELECT get_approved_user_model_context(20)")
            )
            after_rendered = await conn.fetchval(
                "SELECT render_user_model_context($1::jsonb)",
                json.dumps(after_context),
            )
        finally:
            await tr.rollback()

    by_category = {claim["category"]: claim for claim in claims}
    preference = by_category["preference"]
    routine = by_category["routine"]

    assert totals["claimed"] == 3
    assert totals["completed"] == 3
    assert totals["failed"] == 0
    assert totals["claims"] == 3
    assert marker not in before_rendered
    assert f"written project briefs before calls {marker}" in after_rendered
    assert f"quiet Monday planning time {marker}" in after_rendered

    assert preference["evidence_count"] == 2
    evidence = _j(preference["evidence_refs"])
    assert first["source_item_id"] in json.dumps(evidence)
    assert second["source_item_id"] in json.dumps(evidence)
    assert all(ref.get("source_document_id") for ref in evidence)
    assert routine["evidence_count"] == 1


async def test_rules_trajectory_rejects_ephemeral_quoted_and_third_party_false_positives(
    db_pool,
):
    marker = get_test_identifier("user_model_false_positive")
    account = f"channel:slack:{marker}"
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await _set_json_config(conn, "connector.user_model_synthesis_mode", "rules")
            await _set_json_config(conn, "connector.user_model_llm_enabled", False)
            await _connect_slack(conn, marker, account)
            messages = [
                f"Message:\nThis is just a test. I like emerald buttons {marker}.",
                f'Message:\nDana wrote: "I like orange dashboards {marker}."',
                f"Message:\nPlease review this forwarded note.\n> I prefer vendor X {marker}\nI need your summary.",
                f"Message:\nPretend I like midnight standups {marker} for this sample.",
            ]
            for index, content in enumerate(messages):
                await _source_item(
                    conn,
                    account_key=account,
                    provider_item_id=f"{marker}-noise-{index}",
                    offset_seconds=index,
                    content=content,
                )

            totals = await _run_synthesis_until_idle(conn)
            claims = await _claims(conn, marker)
            progress_count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM user_model_source_progress p
                JOIN connector_source_items csi ON csi.id = p.source_item_id
                WHERE csi.account_key = $1 AND p.status = 'completed'
                """,
                account,
            )
        finally:
            await tr.rollback()

    assert totals["claimed"] == 4
    assert totals["completed"] == 4
    assert totals["failed"] == 0
    assert totals["claims"] == 0
    assert progress_count == 4
    assert claims == []


async def test_llm_stub_trajectory_supersedes_stale_claim_and_renders_only_current_context(
    db_pool,
    monkeypatch,
):
    marker = get_test_identifier("user_model_supersede")
    account = f"channel:slack:{marker}"
    old_key = f"preference:{marker}:focus_coffee"
    new_key = f"preference:{marker}:focus_tea"

    async def fake_llm_claims(_conn, item):
        provider_id = str(item.get("provider_item_id") or "")
        if provider_id.endswith("coffee"):
            return [
                {
                    "claim_key": old_key,
                    "category": "preference",
                    "claim": f"User prefers coffee for focused work {marker}.",
                    "confidence": 0.74,
                    "importance": 0.64,
                    "metadata": {"eval": "initial"},
                }
            ]
        if provider_id.endswith("tea"):
            return [
                {
                    "claim_key": new_key,
                    "category": "preference",
                    "claim": f"User now prefers tea for focused work {marker}.",
                    "confidence": 0.86,
                    "importance": 0.78,
                    "supersedes_claim_key": old_key,
                    "contradicts_claim_keys": [old_key],
                    "metadata": {"eval": "supersession"},
                }
            ]
        return []

    async def fake_llm_claims_batch(conn, items, *, provenance=None):
        results = {}
        for item in items:
            source_id = str(item.get("source_item_id") or "")
            results[source_id] = await fake_llm_claims(conn, item)
            if provenance is not None:
                provenance[source_id] = "llm"
        return results

    monkeypatch.setattr(
        cognition, "extract_user_model_claims_llm_batch", fake_llm_claims_batch
    )

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await _set_json_config(conn, "connector.user_model_synthesis_mode", "llm")
            await _set_json_config(conn, "connector.user_model_llm_enabled", True)
            await _connect_slack(conn, marker, account)
            await _source_item(
                conn,
                account_key=account,
                provider_item_id=f"{marker}-coffee",
                offset_seconds=1,
                content=f"Message:\nFor focus sessions, coffee is my default {marker}.",
            )
            await _source_item(
                conn,
                account_key=account,
                provider_item_id=f"{marker}-tea",
                offset_seconds=2,
                content=(
                    "Message:\nActually coffee wrecks my sleep now. "
                    f"For focused work, use tea as the current signal {marker}."
                ),
            )

            totals = await _run_synthesis_until_idle(conn, limit=1)
            claims = {
                claim["claim_key"]: claim for claim in await _claims(conn, marker)
            }
            await conn.fetchval(
                "SELECT review_user_model_claim($1::uuid, 'approve', 'current preference', 'test')",
                claims[new_key]["id"],
            )
            context = _j(
                await conn.fetchval("SELECT get_approved_user_model_context(20)")
            )
            rendered = await conn.fetchval(
                "SELECT render_user_model_context($1::jsonb)",
                json.dumps(context),
            )
            refreshed = {
                claim["claim_key"]: claim for claim in await _claims(conn, marker)
            }
        finally:
            await tr.rollback()

    old_claim = refreshed[old_key]
    new_claim = refreshed[new_key]
    contradictions = _j(new_claim["contradiction_refs"])

    assert totals["claimed"] == 2
    assert totals["completed"] == 2
    assert totals["failed"] == 0
    assert totals["claims"] == 2
    assert totals["llm_used"] == 2

    assert old_claim["status"] == "superseded"
    assert old_claim["review_status"] == "superseded"
    assert old_claim["memory_status"] == "archived"
    assert old_claim["superseded_by"] == new_claim["id"]
    assert new_claim["status"] == "active"
    assert new_claim["review_status"] == "approved"
    assert new_claim["memory_status"] == "active"
    assert any(item["claim_key"] == old_key for item in contradictions)

    assert f"now prefers tea for focused work {marker}" in rendered
    assert f"prefers coffee for focused work {marker}" not in rendered
