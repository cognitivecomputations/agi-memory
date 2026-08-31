from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from services.connector_cognition import (
    LLM_DETECTOR_VERSION,
    estimate_connector_item_importance,
    estimate_connector_item_importance_llm_batch,
    extract_user_model_claims,
    extract_user_model_claims_llm_batch,
    run_connector_importance_step,
    run_user_model_synthesis_step,
)
from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_user_model_rules_ignore_explicit_test_filler():
    claims = extract_user_model_claims(
        {
            "content": "Message:\nThis is just a test. I like green buttons in this sample conversation.",
        }
    )
    assert claims == []


async def test_importance_fallback_does_not_score_free_text_keywords():
    estimate = estimate_connector_item_importance(
        {"content": "URGENT hospital deadline invoice action required"}
    )
    assert estimate["score"] == 0.5
    assert estimate["label"] == "normal"
    assert "LLM unavailable" in estimate["reasons"][0]


async def test_importance_fallback_honors_structured_provider_priority():
    estimate = estimate_connector_item_importance(
        {"content": "ordinary prose", "raw_metadata": {"priority": "urgent"}}
    )
    assert estimate["score"] == 0.96
    assert estimate["label"] == "urgent"


async def test_claim_verdict_cache_reuses_content_hash(db_pool):
    marker = get_test_identifier("connector-claim-cache")
    first = {
        "source_item_id": f"first-{marker}",
        "content_hash": marker,
        "content": "I prefer careful written plans.",
    }
    second = {**first, "source_item_id": f"second-{marker}"}
    reply = {
        "results": {
            first["source_item_id"]: {
                "claims": [
                    {
                        "claim_key": f"preference:{marker}",
                        "category": "preference",
                        "claim": f"User prefers careful written plans {marker}.",
                        "confidence": 0.8,
                        "importance": 0.7,
                    }
                ]
            }
        }
    }

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            first_provenance = {}
            second_provenance = {}
            with (
                patch(
                    "core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))
                ) as model,
                patch(
                    "services.connector_cognition.load_llm_config",
                    AsyncMock(return_value={"provider": "fake", "model": "fake"}),
                ),
            ):
                first_result = await extract_user_model_claims_llm_batch(
                    conn, [first], provenance=first_provenance
                )
                second_result = await extract_user_model_claims_llm_batch(
                    conn, [second], provenance=second_provenance
                )
            hit_count = await conn.fetchval(
                """
                SELECT hit_count FROM connector_cognition_cache
                WHERE task = 'user_model_claims' AND content_hash = $1
                """,
                marker,
            )
        finally:
            await tr.rollback()

    assert model.await_count == 1
    assert first_provenance[first["source_item_id"]] == "llm"
    assert second_provenance[second["source_item_id"]] == "cache"
    assert (
        first_result[first["source_item_id"]] == second_result[second["source_item_id"]]
    )
    assert hit_count == 1


async def test_importance_verdict_cache_reuses_content_hash(db_pool):
    marker = get_test_identifier("connector-importance-cache")
    first = {
        "source_item_id": f"first-{marker}",
        "content_hash": marker,
        "content": "A message whose meaning needs judgment.",
    }
    second = {**first, "source_item_id": f"second-{marker}"}
    reply = {
        "results": {
            first["source_item_id"]: {
                "score": 0.88,
                "label": "important",
                "reasons": ["material decision"],
                "recommended_actions": [],
            }
        }
    }

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            first_provenance = {}
            second_provenance = {}
            with (
                patch(
                    "core.llm_batch.chat_json", AsyncMock(return_value=(reply, ""))
                ) as model,
                patch(
                    "services.connector_cognition.load_llm_config",
                    AsyncMock(return_value={"provider": "fake", "model": "fake"}),
                ),
            ):
                first_result = await estimate_connector_item_importance_llm_batch(
                    conn, [first], provenance=first_provenance
                )
                second_result = await estimate_connector_item_importance_llm_batch(
                    conn, [second], provenance=second_provenance
                )
        finally:
            await tr.rollback()

    assert model.await_count == 1
    assert first_provenance[first["source_item_id"]] == "llm"
    assert second_provenance[second["source_item_id"]] == "cache"
    assert (
        first_result[first["source_item_id"]] == second_result[second["source_item_id"]]
    )


async def _stub_get_embedding(conn):
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


async def _connected_channel(
    conn, connector_id: str, marker: str, account_key: str
) -> None:
    attempt = _j(
        await conn.fetchval(
            """
        SELECT start_connection_attempt(
            $1,
            '["live_chat", "send", "ingest_live"]'::jsonb,
            ARRAY[]::text[],
            '{}'::jsonb,
            NULL,
            NULL,
            'test',
            $2,
            CURRENT_TIMESTAMP + INTERVAL '10 minutes'
        )
        """,
            connector_id,
            marker,
        )
    )
    await conn.fetchval(
        """
        SELECT complete_connection_attempt(
            $1::uuid,
            $2,
            $3,
            $4,
            ARRAY[]::text[],
            '["live_chat", "send", "ingest_live"]'::jsonb,
            '{"test": true}'::jsonb
        )
        """,
        attempt["attempt_id"],
        account_key,
        connector_id,
        f"config:channel.{connector_id}",
    )


async def test_connector_source_items_become_claims_and_importance_notifications(
    db_pool, monkeypatch
):
    marker = get_test_identifier("connector-cognition")
    account = f"channel:slack:{marker}"
    provider_item_id = f"msg-{marker}"
    content = (
        f"Slack channel: CCOGNITION\n"
        f"Slack timestamp: 1710000000.000100\n"
        f"Sender: U1\n\n"
        f"Message:\n"
        f"I prefer quiet morning planning {marker}. "
        "The deadline is due today. Can you please flag this?"
    )

    async def fake_claims(_conn, items, *, provenance=None):
        result = {}
        for item in items:
            source_id = str(item["source_item_id"])
            if provenance is not None:
                provenance[source_id] = "llm"
            result[source_id] = [
                {
                    "claim_key": f"preference:quiet_morning_planning_{marker}",
                    "category": "preference",
                    "claim": f"User prefers quiet morning planning {marker}.",
                    "confidence": 0.82,
                    "importance": 0.7,
                }
            ]
        return result

    async def fake_importance(_conn, items, *, provenance=None):
        result = {}
        for item in items:
            source_id = str(item["source_item_id"])
            if provenance is not None:
                provenance[source_id] = "llm"
            result[source_id] = {
                "score": 0.9,
                "label": "important",
                "reasons": ["time-sensitive direct request"],
                "recommended_actions": [
                    {"kind": "notify_user", "urgency": "important"}
                ],
            }
        return result

    monkeypatch.setattr(
        "services.connector_cognition.extract_user_model_claims_llm_batch", fake_claims
    )
    monkeypatch.setattr(
        "services.connector_cognition.estimate_connector_item_importance_llm_batch",
        fake_importance,
    )

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await _connected_channel(conn, "slack", marker, account)
            source_item = _j(
                await conn.fetchval(
                    """
                SELECT upsert_connector_source_item(
                    'slack',
                    $1,
                    $2,
                    'Cognition message',
                    $3,
                    'message',
                    NULL,
                    CURRENT_TIMESTAMP,
                    ARRAY['slack', 'CCOGNITION']::text[],
                    '[{"role": "sender", "id": "U1"}]'::jsonb,
                    '[]'::jsonb,
                    '{"test": true}'::jsonb,
                    'private',
                    TRUE
                )
                """,
                    account,
                    provider_item_id,
                    content,
                )
            )
            await conn.execute(
                "UPDATE connector_source_items SET status = 'archived' WHERE id <> $1::uuid",
                source_item["source_item_id"],
            )

            synthesis = await run_user_model_synthesis_step(conn, limit=5)
            importance = await run_connector_importance_step(conn, limit=5)

            claim = await conn.fetchrow(
                """
                SELECT c.claim_key, c.claim, c.memory_id, m.type::text AS memory_type,
                       c.evidence_refs
                FROM user_model_claims c
                JOIN memories m ON m.id = c.memory_id
                WHERE c.claim LIKE $1
                """,
                f"%{marker}%",
            )
            progress = await conn.fetchrow(
                """
                SELECT status, detector_version, result
                FROM user_model_source_progress
                WHERE source_item_id = $1::uuid
                """,
                source_item["source_item_id"],
            )
            item_importance = await conn.fetchrow(
                """
                SELECT score, label, status, detector_version,
                       notification_queued_at, metadata
                FROM connector_item_importance
                WHERE source_item_id = $1::uuid
                """,
                source_item["source_item_id"],
            )
            outbox = await conn.fetchrow("""
                SELECT source, envelope
                FROM outbox_messages
                WHERE source = 'connector_importance'
                  AND envelope->'payload'->>'intent' = 'connector_importance'
                ORDER BY created_at DESC
                LIMIT 1
                """)
        finally:
            await tr.rollback()

    assert synthesis["claimed"] == 1
    assert synthesis["completed"] == 1
    assert synthesis["claims"] >= 1
    assert synthesis["llm_used"] == 1
    assert synthesis["fallback_used"] == 0
    assert importance["claimed"] == 1
    assert importance["completed"] == 1
    assert importance["notified"] == 1
    assert importance["llm_used"] == 1
    assert importance["fallback_used"] == 0

    assert claim is not None
    assert claim["claim_key"].startswith("preference:")
    assert claim["memory_type"] == "semantic"
    assert source_item["source_item_id"] in json.dumps(_j(claim["evidence_refs"]))
    assert progress["status"] == "completed"
    assert progress["detector_version"] == LLM_DETECTOR_VERSION
    assert _j(progress["result"])["claim_count"] >= 1

    assert item_importance["status"] == "completed"
    assert item_importance["detector_version"] == LLM_DETECTOR_VERSION
    assert item_importance["label"] == "important"
    assert float(item_importance["score"]) >= 0.85
    assert item_importance["notification_queued_at"] is not None
    assert "outbox_message_id" in _j(item_importance["metadata"])

    assert outbox is not None
    envelope = _j(outbox["envelope"])
    assert envelope["payload"]["delivery"]["mode"] == "web_inbox"
    assert (
        envelope["payload"]["delivery"]["source_item_id"]
        == source_item["source_item_id"]
    )
