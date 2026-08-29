"""Weekly learning diffs and explicit user decisions."""

from __future__ import annotations

import json
import uuid

import pytest


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _memory(conn, memory_type: str, content: str, *, importance: float = 0.5):
    return await conn.fetchval(
        """
        SELECT create_memory(
            $1::memory_type, $2, $3,
            jsonb_build_object(
                'kind', 'user_testimony',
                'ref', $4::text,
                'label', 'Learning review test',
                'trust', 1.0
            ),
            1.0,
            '{}'::jsonb
        )
        """,
        memory_type,
        content,
        importance,
        f"test:{uuid.uuid4()}",
    )


async def _review(conn, memory_ids, skill_ids=()):
    return _json(
        await conn.fetchval(
            """
            SELECT create_learning_review(
                CURRENT_TIMESTAMP - INTERVAL '7 days',
                CURRENT_TIMESTAMP + INTERVAL '1 second',
                'The week produced a grounded set of changes worth reviewing.',
                $1::uuid[], $2::uuid[],
                '{"test": true}'::jsonb
            )
            """,
            memory_ids,
            list(skill_ids),
        )
    )


async def test_learning_review_is_one_outbox_diff_derived_from_durable_truth(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            belief = await _memory(conn, "semantic", "The release review happens on Friday.")
            procedure = await _memory(conn, "procedural", "Verify the focused test before the full suite.")
            strategy = await _memory(conn, "strategic", "Prefer a bounded rollout when uncertainty is high.")
            proposal = await conn.fetchval(
                """
                INSERT INTO skill_improvement_proposals (
                    name, description, content, rationale, confidence,
                    source_unit_ids, evidence_digest
                ) VALUES (
                    'weekly-release-diff',
                    'Review release changes as one bounded weekly diff',
                    $1,
                    'Repeated review evidence supports a reusable workflow.',
                    0.93,
                    ARRAY[$2::uuid],
                    $3
                ) RETURNING id
                """,
                "# Weekly release diff\n\nInspect the grounded changes and present them together. "
                "Require an explicit decision before changing future behavior, preserve the "
                "source evidence, and report the exact recovery step when application fails.",
                uuid.uuid4(),
                uuid.uuid4().hex,
            )

            created = await _review(conn, [belief, procedure, strategy], [proposal])

            assert created["created"] is True
            assert created["item_count"] == 4
            reviews = _json(await conn.fetchval("SELECT list_learning_reviews('pending', 10)"))
            review = next(item for item in reviews if item["id"] == created["review_id"])
            assert {item["kind"] for item in review["items"]} == {
                "semantic_belief",
                "new_procedure",
                "revised_strategy",
                "proposed_skill",
            }
            belief_item = next(
                item
                for item in review["items"]
                if item.get("source_memory_id") == str(belief)
            )
            assert belief_item["content"] == "The release review happens on Friday."
            assert belief_item["evidence"]["source_attribution"]["label"] == "Learning review test"
            outbox = await conn.fetchrow(
                "SELECT envelope FROM outbox_messages WHERE id=$1::uuid",
                created["outbox_message_id"],
            )
            envelope = _json(outbox["envelope"])
            assert envelope["payload"]["intent"] == "learning_review"
            assert "approve" in envelope["payload"]["message"]
            assert "correct" in envelope["payload"]["message"]
            assert "forget" in envelope["payload"]["message"]
            assert envelope["payload"]["delivery"]["review_url"] == "/learning-review"
        finally:
            await transaction.rollback()


async def test_correction_uses_contradiction_resolution_and_preserves_history(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            belief = await _memory(conn, "semantic", "The planning call is on Thursday.")
            created = await _review(conn, [belief])
            item_id = await conn.fetchval(
                "SELECT id FROM learning_review_items WHERE review_id=$1::uuid",
                created["review_id"],
            )

            result = _json(
                await conn.fetchval(
                    "SELECT decide_learning_review_item($1, 'correct', $2, 'web', 'operator', FALSE)",
                    item_id,
                    "The planning call is on Friday.",
                )
            )

            assert result["ok"] is True
            assert result["status"] == "corrected"
            assert result["contradiction_case_id"]
            old = await conn.fetchrow(
                "SELECT status, valid_until, superseded_by FROM memories WHERE id=$1",
                belief,
            )
            assert old["status"] == "active"
            assert old["valid_until"] is not None
            assert str(old["superseded_by"]) == result["correction_memory_id"]
            case = await conn.fetchrow(
                "SELECT status, outcome, winner_memory_id, loser_memory_id FROM contradiction_cases WHERE id=$1::uuid",
                result["contradiction_case_id"],
            )
            assert case["status"] == "resolved"
            assert case["outcome"] == "new_right"
            assert str(case["loser_memory_id"]) == str(belief)
            review_status = await conn.fetchval(
                "SELECT status FROM learning_reviews WHERE id=$1::uuid",
                created["review_id"],
            )
            assert review_status == "completed"
        finally:
            await transaction.rollback()


async def test_forgetting_load_bearing_learning_requires_second_confirmation(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            protected = await _memory(
                conn,
                "semantic",
                "This relationship commitment is load-bearing.",
                importance=0.95,
            )
            created = await _review(conn, [protected])
            item_id = await conn.fetchval(
                "SELECT id FROM learning_review_items WHERE review_id=$1::uuid",
                created["review_id"],
            )

            refused = _json(
                await conn.fetchval(
                    "SELECT decide_learning_review_item($1, 'forget', NULL, 'web', 'operator', FALSE)",
                    item_id,
                )
            )
            assert refused["confirmation_required"] is True
            assert await conn.fetchval("SELECT status FROM memories WHERE id=$1", protected) == "active"

            confirmed = _json(
                await conn.fetchval(
                    "SELECT decide_learning_review_item($1, 'forget', NULL, 'web', 'operator', TRUE)",
                    item_id,
                )
            )
            assert confirmed["ok"] is True
            assert confirmed["status"] == "forgotten"
            memory = await conn.fetchrow(
                "SELECT status, valid_until FROM memories WHERE id=$1", protected
            )
            assert memory["status"] == "archived"
            assert memory["valid_until"] is not None
        finally:
            await transaction.rollback()


async def test_skill_approval_queues_application_and_exact_inbound_reply_works(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            proposal = await conn.fetchval(
                """
                INSERT INTO skill_improvement_proposals (
                    name, description, content, rationale, confidence,
                    source_unit_ids, evidence_digest
                ) VALUES ('review-queue-test', 'A queued application test', $1,
                    'The operator should remain in control of application.', 0.9,
                    ARRAY[$2::uuid], $3) RETURNING id
                """,
                "# Review queue test\n\nApply this only after an explicit decision. Preserve "
                "the proposal if writing fails, expose the failure, and retry through the "
                "same idempotent ownership-checked authoring path without another decision.",
                uuid.uuid4(),
                uuid.uuid4().hex,
            )
            created = await _review(conn, [], [proposal])
            item = await conn.fetchrow(
                "SELECT id, learning_review_item_code(id) AS code FROM learning_review_items WHERE review_id=$1::uuid",
                created["review_id"],
            )
            inbound = _json(
                await conn.fetchval(
                    "SELECT try_resolve_learning_review_from_inbound('slack', 'operator', $1)",
                    f"approve {item['code']}",
                )
            )
            assert inbound["recognized"] is True
            assert inbound["matched"] is True
            assert "queued for application" in inbound["message"]

            claim = _json(
                await conn.fetchval("SELECT claim_approved_learning_skill_application()")
            )
            assert claim["claimed"] is True
            assert claim["proposal_id"] == str(proposal)
            finished = _json(
                await conn.fetchval(
                    "SELECT finish_learning_skill_application($1::uuid, 'applied', NULL)",
                    item["id"],
                )
            )
            assert finished["application_status"] == "applied"
        finally:
            await transaction.rollback()
