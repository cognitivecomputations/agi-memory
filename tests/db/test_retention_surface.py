"""User-facing memory pressure, fade choices, and compression reports."""

from __future__ import annotations

import json

import pytest


pytestmark = [pytest.mark.asyncio(loop_scope="session")]
_DUMMY = "array_fill(0.1, ARRAY[embedding_dimension()])::vector"


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _episodic(conn, content: str, *, importance: float = 0.4, fidelity: float = 1.0):
    return await conn.fetchval(
        f"""
        INSERT INTO memories (
            type, content, embedding, importance, trust_level, status, fidelity
        ) VALUES ('episodic', $1, {_DUMMY}, $2, 0.9, 'active', $3)
        RETURNING id
        """,
        content,
        importance,
        fidelity,
    )


async def _pending(conn, ids, *, preview="A meaningful sequence nearing compression"):
    return await conn.fetchval(
        """
        INSERT INTO memory_review_queue (memory_ids, reason, preview)
        VALUES ($1::uuid[], 'near_protection_threshold', $2)
        RETURNING id
        """,
        ids,
        preview,
    )


async def test_observe_packet_surfaces_pressure_and_low_fidelity(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            low = await _episodic(conn, "A hazy but accountable recollection", fidelity=0.42)
            await conn.execute("SELECT set_config('retention.capacity', '10'::jsonb)")

            packet = _json(await conn.fetchval("SELECT retention_observe_packet(5)"))
            context = _json(
                await conn.fetchval("SELECT get_memories_at_threshold_context(5)")
            )

            assert packet["pressure"]["episodic_mass"] is not None
            assert packet["pressure"]["capacity"] == 10
            assert packet["low_fidelity_count"] >= 1
            assert any(item["memory_id"] == str(low) for item in packet["low_fidelity"])
            assert context["pressure"] == packet["pressure"]
            assert context["irreversible_pruning_enabled"] is False
        finally:
            await transaction.rollback()


async def test_fade_proposal_is_one_outbox_ask_and_keep_is_explicit(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute("SELECT set_config('retention.enabled', 'true'::jsonb)")
            await conn.execute(
                "SELECT set_state('retention_veto_budget', $1::jsonb)",
                json.dumps({"chapter": "test", "remaining": 2, "total": 2}),
            )
            ids = [await _episodic(conn, f"Load-bearing moment {index}") for index in range(3)]
            review_id = await _pending(conn, ids)

            digest = _json(
                await conn.fetchval("SELECT publish_memory_fade_review_digest()")
            )
            duplicate = _json(
                await conn.fetchval("SELECT publish_memory_fade_review_digest()")
            )
            assert digest["count"] == 1
            assert duplicate["skipped"] is True
            envelope = _json(
                await conn.fetchval(
                    "SELECT envelope FROM outbox_messages WHERE id=$1::uuid",
                    digest["outbox_message_id"],
                )
            )
            message = envelope["payload"]["message"]
            assert "I have not compressed them" in message
            assert all(choice in message for choice in ("keep", "release", "journal"))

            decision = _json(
                await conn.fetchval(
                    "SELECT decide_memory_fade_review($1, 'keep', NULL, 'web', 'operator')",
                    review_id,
                )
            )
            assert decision["ok"] is True
            assert decision["status"] == "kept"
            assert decision["budget_remaining"] == 1
            assert await conn.fetchval(
                "SELECT bool_and(is_memory_protected(id)) FROM memories WHERE id=ANY($1::uuid[])",
                ids,
            ) is True
        finally:
            await transaction.rollback()


async def test_release_reports_exact_compression_and_originals_stay_recoverable(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute("SELECT set_config('retention.enabled', 'true'::jsonb)")
            await conn.execute(
                "SELECT set_config('retention.irreversible_pruning_enabled', 'false'::jsonb)"
            )
            ids = [await _episodic(conn, f"Compressible detail {index}", importance=0.3) for index in range(3)]
            review_id = await _pending(conn, ids)

            decision = _json(
                await conn.fetchval(
                    "SELECT decide_memory_fade_review($1, 'release', NULL, 'web', 'operator')",
                    review_id,
                )
            )
            gist = decision["compression"]["gist_memory_id"]
            assert decision["compression"]["source_count"] == 3
            assert decision["compression"]["originals_recoverable"] is True
            assert await conn.fetchval(
                "SELECT count(*) FROM memories WHERE id=ANY($1::uuid[]) AND status='archived'",
                ids,
            ) == 3

            await conn.execute(
                """
                UPDATE memories
                SET content='One honest compressed recollection',
                    fidelity=0.7,
                    metadata=jsonb_set(metadata, '{consolidation,summarized}', 'true'::jsonb, true)
                WHERE id=$1::uuid
                """,
                gist,
            )
            report = await conn.fetchrow(
                "SELECT source_count, fidelity, summary_preview FROM memory_compression_reports WHERE gist_memory_id=$1::uuid",
                gist,
            )
            assert report["source_count"] == 3
            assert report["fidelity"] == pytest.approx(0.7)
            published = _json(
                await conn.fetchval(
                    "SELECT publish_retention_compression_report_if_due(TRUE)"
                )
            )
            assert published["compression_count"] == 1
            assert published["source_count"] == 3
            envelope = _json(
                await conn.fetchval(
                    "SELECT envelope FROM outbox_messages WHERE id=$1::uuid",
                    published["outbox_message_id"],
                )
            )
            assert "70% fidelity" in envelope["payload"]["message"]
            assert "One honest compressed recollection" in envelope["payload"]["message"]
        finally:
            await transaction.rollback()


async def test_exact_operator_reply_can_journal_before_release(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        transaction = conn.transaction()
        await transaction.start()
        try:
            ids = [await _episodic(conn, f"Journal detail {index}", importance=0.3) for index in range(3)]
            review_id = await _pending(conn, ids)
            code = await conn.fetchval(
                "SELECT memory_fade_review_code($1::uuid)", review_id
            )
            before = await conn.fetchval("SELECT count(*) FROM journal_entries")

            result = _json(
                await conn.fetchval(
                    "SELECT try_resolve_memory_fade_review_from_inbound('slack', 'operator', $1)",
                    f"journal {code}: Keep the lesson, not every detail.",
                )
            )
            assert result["recognized"] is True
            assert result["matched"] is True
            assert result["decision"] == "journal"
            assert await conn.fetchval("SELECT count(*) FROM journal_entries") == before + 1
        finally:
            await transaction.rollback()
