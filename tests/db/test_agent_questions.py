from __future__ import annotations

import json
import uuid
from pathlib import Path

import asyncpg
import pytest


pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "db"
    / "migrations"
    / "0208_agent_questions.sql"
)


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _create(
    conn,
    *,
    session_id: str | None = None,
    heartbeat_id: str | None = None,
    surface: str = "api",
    prompt: str = "Which contract?",
    choices: list[str] | None = None,
    free_text: bool = True,
    wait: bool = True,
    timeout: int = 300,
):
    return _json(
        await conn.fetchval(
            "SELECT create_agent_question($1::uuid, $2::uuid, $3, $4, $5::jsonb, $6, $7, $8)",
            session_id,
            heartbeat_id,
            surface,
            prompt,
            json.dumps(choices or []),
            free_text,
            wait,
            timeout,
        )
    )


async def test_question_validation_and_same_session_supersession(db_pool):
    session_id = str(uuid.uuid4())
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            first = await _create(conn, session_id=session_id)
            second = await _create(
                conn,
                session_id=session_id,
                prompt="Which section?",
                choices=["Payment", "Termination"],
            )
            statuses = {
                row["id"]: row["status"]
                for row in await conn.fetch(
                    "SELECT id::text AS id, status FROM agent_questions WHERE id = ANY($1::uuid[])",
                    [first["id"], second["id"]],
                )
            }
            with pytest.raises(asyncpg.PostgresError, match="at most four"):
                await _create(
                    conn,
                    session_id=str(uuid.uuid4()),
                    choices=["1", "2", "3", "4", "5"],
                )
        finally:
            await transaction.rollback()

    assert statuses[first["id"]] == "superseded"
    assert statuses[second["id"]] == "pending"
    assert second["choices"] == ["Payment", "Termination"]


async def test_answer_choice_is_exact_idempotent_and_claimed_once(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            question = await _create(
                conn,
                session_id=str(uuid.uuid4()),
                choices=["The Manning one", "The Hartford one"],
                free_text=False,
            )
            rejected = _json(
                await conn.fetchval(
                    "SELECT answer_agent_question($1::uuid, 'something else', NULL, 'web', 'test')",
                    question["id"],
                )
            )
            answered = _json(
                await conn.fetchval(
                    "SELECT answer_agent_question($1::uuid, NULL, 2, 'web', 'test')",
                    question["id"],
                )
            )
            repeated = _json(
                await conn.fetchval(
                    "SELECT answer_agent_question($1::uuid, NULL, 1, 'web', 'test')",
                    question["id"],
                )
            )
            claimed = _json(
                await conn.fetchval(
                    "SELECT claim_agent_question_answer($1::uuid)", question["id"]
                )
            )
            free_text_question = await _create(
                conn,
                session_id=str(uuid.uuid4()),
                choices=["The Manning one", "The Hartford one"],
                free_text=True,
            )
            other_required = _json(
                await conn.fetchval(
                    "SELECT answer_agent_question($1::uuid, NULL, 3, 'web_inbox', 'test')",
                    free_text_question["id"],
                )
            )
        finally:
            await transaction.rollback()

    assert rejected["ok"] is False
    assert rejected["error"] == "free_text_disabled"
    assert answered["answer"] == "The Hartford one"
    assert answered["answer_choice_index"] == 2
    assert repeated["already_answered"] is True
    assert repeated["answer"] == "The Hartford one"
    assert claimed["resumed_at"] is not None
    assert other_required["error"] == "free_text_required"
    assert "Type your answer" in other_required["message"]


async def test_timeout_is_graceful_and_late_answer_is_rejected(db_pool):
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            question = await _create(conn, session_id=str(uuid.uuid4()), timeout=1)
            timed_out = _json(
                await conn.fetchval(
                    "SELECT timeout_agent_question($1::uuid)", question["id"]
                )
            )
            late = _json(
                await conn.fetchval(
                    "SELECT answer_agent_question($1::uuid, 'late', NULL, 'web', 'test')",
                    question["id"],
                )
            )
        finally:
            await transaction.rollback()

    assert timed_out["ok"] is True
    assert timed_out["status"] == "timed_out"
    assert late["ok"] is False
    assert late["error"] == "question_timed_out"


async def test_heartbeat_question_routes_outbox_and_resumes_exactly_once(db_pool):
    channel_type = f"test-{uuid.uuid4().hex[:8]}"
    channel_id = f"room-{uuid.uuid4().hex[:8]}"
    actor = f"user-{uuid.uuid4().hex[:8]}"
    heartbeat_id = str(uuid.uuid4())
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(
                "INSERT INTO channel_sessions(channel_type, channel_id, sender_id, sender_name) VALUES ($1, $2, $3, 'Test')",
                channel_type,
                channel_id,
                actor,
            )
            question = await _create(
                conn,
                heartbeat_id=heartbeat_id,
                surface="heartbeat",
                choices=["Manning", "Hartford"],
                wait=False,
            )
            envelope = _json(
                await conn.fetchval(
                    "SELECT envelope FROM outbox_messages WHERE id = $1::uuid",
                    question["outbox_message_id"],
                )
            )
            resolved = _json(
                await conn.fetchval(
                    "SELECT try_resolve_agent_question_from_inbound($1, $2, $3, '2')",
                    channel_type,
                    channel_id,
                    actor,
                )
            )
            first_context = _json(
                await conn.fetchval(
                    "SELECT attach_answered_agent_questions('{}'::jsonb)"
                )
            )
            second_context = _json(
                await conn.fetchval(
                    "SELECT attach_answered_agent_questions('{}'::jsonb)"
                )
            )
            rendered = await conn.fetchval(
                "SELECT render_answered_agent_questions($1::jsonb)",
                json.dumps(first_context["answered_questions"]),
            )
        finally:
            await transaction.rollback()

    assert question["status"] == "pending"
    assert question["expires_at"] is None
    assert question["session_id"] is not None
    assert envelope["payload"]["delivery"]["question_id"] == question["id"]
    assert "1. Manning" in envelope["payload"]["message"]
    assert resolved["ok"] is True
    assert resolved["answer"] == "Hartford"
    assert len(first_context["answered_questions"]) == 1
    assert second_context["answered_questions"] == []
    assert "Answer: Hartford" in rendered


async def test_agent_question_migration_is_idempotent_and_preserves_rows(db_pool):
    migration = _MIGRATION.read_text(encoding="utf-8")
    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            question = await _create(
                conn,
                session_id=str(uuid.uuid4()),
                prompt="Migration sentinel?",
            )
            await conn.fetchval(
                "SELECT answer_agent_question($1::uuid, 'keep me', NULL, 'test', 'test')",
                question["id"],
            )
            await conn.execute(migration)
            await conn.execute(migration)
            row = await conn.fetchrow(
                "SELECT status, answer FROM agent_questions WHERE id = $1::uuid",
                question["id"],
            )
        finally:
            await transaction.rollback()

    assert row["status"] == "answered"
    assert row["answer"] == "keep me"
