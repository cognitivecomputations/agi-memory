import json
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


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


async def test_record_chat_turn_memory_always_uses_recmem(db_pool):
    async with db_pool.acquire() as conn:
        await _stub_get_embedding(conn)

        # Direct promotion is a 0.95 safety valve now (#73) — signal phrases
        # (0.8) route through scene consolidation, so promotion needs explicit
        # exceptional importance.
        raw = await conn.fetchval(
            "SELECT record_chat_turn_memory($1, $2, $3, $4, '{\"importance\": 0.96}'::jsonb)",
            "remember this important preference",
            "noted",
            str(uuid4()),
            f"test:chat:{uuid4()}",
        )
        result = _json(raw)

        assert result["direct_promoted"] is True
        assert result["raw"]["status"] == "stored"
        assert result["promoted_memory_id"] is not None
        linked = await conn.fetchval(
            "SELECT COUNT(*) FROM memory_source_units WHERE memory_id = $1::uuid AND subconscious_unit_id = $2::uuid",
            result["promoted_memory_id"],
            result["raw_unit_id"],
        )
        assert int(linked) == 1


async def test_record_chat_turn_memory_derives_source_identity(db_pool):
    """With no caller identity, the DB self-derives chat:<session>:<ordinal>:
    <digest> — the ordinal from its own unit count, the digest from content."""
    session_id = str(uuid4())
    async with db_pool.acquire() as conn:
        await _stub_get_embedding(conn)

        first = _json(
            await conn.fetchval(
                "SELECT record_chat_turn_memory($1, $2, $3, NULL, '{}'::jsonb)",
                "identity derivation turn one",
                "first reply",
                session_id,
            )
        )
        second = _json(
            await conn.fetchval(
                "SELECT record_chat_turn_memory($1, $2, $3, NULL, '{}'::jsonb)",
                "identity derivation turn two",
                "second reply",
                session_id,
            )
        )
        identities = [
            await conn.fetchval(
                "SELECT source_identity FROM subconscious_units WHERE id = $1::uuid",
                result["raw_unit_id"],
            )
            for result in (first, second)
        ]

    assert identities[0].startswith(f"chat:{session_id}:0:")
    assert identities[1].startswith(f"chat:{session_id}:1:")
    digests = [identity.rsplit(":", 1)[1] for identity in identities]
    assert all(len(digest) == 16 for digest in digests)
    assert digests[0] != digests[1]


async def test_record_chat_turn_memory_keeps_caller_identity(db_pool):
    async with db_pool.acquire() as conn:
        await _stub_get_embedding(conn)
        explicit = f"channel:telegram:{uuid4()}"
        result = _json(
            await conn.fetchval(
                "SELECT record_chat_turn_memory($1, $2, $3, $4, '{}'::jsonb)",
                "caller identity wins",
                "kept verbatim",
                str(uuid4()),
                explicit,
            )
        )
        stored = await conn.fetchval(
            "SELECT source_identity FROM subconscious_units WHERE id = $1::uuid",
            result["raw_unit_id"],
        )

    assert stored == explicit


async def test_record_chat_turn_memory_low_importance_no_promotion(db_pool):
    async with db_pool.acquire() as conn:
        await _stub_get_embedding(conn)

        raw = await conn.fetchval(
            "SELECT record_chat_turn_memory($1, $2, $3, $4, '{}'::jsonb)",
            "hi",
            "hello",
            str(uuid4()),
            f"test:chat:{uuid4()}",
        )
        result = _json(raw)

        assert result["direct_promoted"] is False
        assert result["promoted_memory_id"] is None
        assert result["raw"]["status"] == "stored"


async def test_prepare_and_finalize_channel_turn_db_lifecycle(db_pool):
    async with db_pool.acquire() as conn:
        channel_id = f"chan-{uuid4()}"
        sender_id = f"sender-{uuid4()}"
        prepared_raw = await conn.fetchval(
            "SELECT prepare_channel_turn($1::jsonb)",
            json.dumps(
                {
                    "channel_type": "unit",
                    "channel_id": channel_id,
                    "sender_id": sender_id,
                    "sender_name": "Tester",
                    "content": "hello",
                    "message_id": "m1",
                }
            ),
        )
        prepared = _json(prepared_raw)
        assert prepared["allowed"] is True
        assert prepared["history"] == []

        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        finalized_raw = await conn.fetchval(
            "SELECT finalize_channel_turn($1::uuid, $2, $3, $4::jsonb)",
            prepared["session_id"],
            "hello",
            "hi",
            json.dumps({"history": history, "metadata": {"channel_type": "unit"}}),
        )
        finalized = _json(finalized_raw)

        assert finalized["history_count"] == 2
        outbound = await conn.fetchval(
            "SELECT COUNT(*) FROM channel_messages WHERE session_id = $1::uuid AND direction = 'outbound'",
            prepared["session_id"],
        )
        assert int(outbound) == 1


async def test_turn_write_stamps_appraisal_affect(db_pool):
    """#81: the stored unit carries THIS turn's appraisal affect; without an
    appraisal it snapshots the current state instead of storing zeros."""
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            await conn.fetchval(
                """SELECT record_chat_turn_memory('kiss turn', 'warm reply', $1, NULL,
                    '{"emotional_state": {"valence": 0.66, "arousal": 0.53, "intensity": 0.62, "primary_emotion": "affection", "family": "connection"}}'::jsonb)""",
                str(uuid4()),
            )
            stamped = await conn.fetchrow(
                """SELECT metadata #>> '{emotional_context,primary_emotion}' AS emotion,
                          metadata #>> '{emotional_context,family}' AS family,
                          (metadata #>> '{emotional_context,valence}')::float AS valence,
                          metadata #>> '{emotional_context,source}' AS source
                   FROM subconscious_units WHERE user_text = 'kiss turn'"""
            )
            await conn.fetchval(
                "SELECT record_chat_turn_memory('plain turn', 'reply', $1, NULL, '{}'::jsonb)",
                str(uuid4()),
            )
            fallback = await conn.fetchval(
                "SELECT metadata #>> '{emotional_context,source}' FROM subconscious_units WHERE user_text = 'plain turn'"
            )
        finally:
            await tr.rollback()

    assert stamped["emotion"] == "affection"
    assert stamped["family"] == "connection"
    assert abs(stamped["valence"] - 0.66) < 1e-9
    assert stamped["source"] == "appraisal"
    assert fallback == "state_snapshot"


async def test_extraction_carries_turn_affect_not_sweep_mood(db_pool):
    """#81: a memory minted by the extraction sweep carries the turn's stored
    affect, overriding the creation trigger's current-state stamp."""
    import json

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await _stub_get_embedding(conn)
            unit_id = await conn.fetchval("""
                INSERT INTO subconscious_units (
                    content, user_text, assistant_text, embedding, embedding_status,
                    route_status, importance, idempotency_key, metadata
                ) VALUES (
                    'Eric: I adore this. / Samantha: warm reply', 'I adore this', 'warm reply',
                    array_fill(0.1, ARRAY[embedding_dimension()])::vector, 'embedded',
                    'raw_only', 0.7, gen_random_uuid()::text,
                    '{"emotional_context": {"valence": 0.66, "arousal": 0.53, "intensity": 0.62, "primary_emotion": "affection", "source": "appraisal"}}'::jsonb
                ) RETURNING id
                """)
            result = json.loads(
                await conn.fetchval(
                    "SELECT apply_conscious_extraction(ARRAY[$1::uuid], $2::jsonb)",
                    unit_id,
                    json.dumps(
                        [
                            {
                                "content": "Affect-carry check: Eric adores this project.",
                                "kind": "user_testimony",
                                "confidence": 0.8,
                                "unit_id": str(unit_id),
                            }
                        ]
                    ),
                )
            )
            assert result["created"] == 1
            row = await conn.fetchrow(
                """SELECT (metadata #>> '{emotional_context,valence}')::float AS valence,
                          metadata #>> '{emotional_context,primary_emotion}' AS emotion
                   FROM memories WHERE content = 'Affect-carry check: Eric adores this project.'"""
            )
        finally:
            await tr.rollback()

    assert row["emotion"] == "affection"
    assert abs(row["valence"] - 0.66) < 1e-9
