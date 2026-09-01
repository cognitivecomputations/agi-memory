"""Durability and control tests for operator standing instructions."""

from __future__ import annotations

import json

import asyncpg
import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


def _object(value) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


@pytest.fixture
async def conn(db_pool):
    async with db_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            yield connection
        finally:
            await transaction.rollback()


async def _capture(
    conn,
    *,
    text: str,
    message_id: str,
    is_operator: bool = True,
    disposition: str = "engage",
) -> dict:
    raw = await conn.fetchval(
        """
        SELECT capture_operator_policy_correction(
            'slack', $1, 'U-operator', 'Operator', $2, $3, $4,
            'test_operator_turn', $5::jsonb
        )
        """,
        f"C-{get_test_identifier('operator-policy')}",
        text,
        is_operator,
        disposition,
        json.dumps({"message_id": message_id}),
    )
    return _object(raw)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Always cite the source when you summarize a document.", True),
        ("When you summarize a document, cite the source.", True),
        ("Can you always cite the source?", True),
        ("When should I use citations?", False),
        ("Can you cite this source once?", False),
        ("This must be fixed.", False),
    ],
)
async def test_classifier_requires_an_explicit_standing_instruction(
    conn, text, expected
):
    result = _object(
        await conn.fetchval(
            "SELECT classify_operator_policy_correction('slack', $1)", text
        )
    )
    assert result["is_policy_correction"] is expected


async def test_non_operator_cannot_create_policy_state(conn):
    before = await conn.fetchval("SELECT count(*) FROM operator_policy_corrections")
    result = await _capture(
        conn,
        text="Always cite the source when you summarize a document.",
        message_id=get_test_identifier("not-operator"),
        is_operator=False,
    )
    after = await conn.fetchval("SELECT count(*) FROM operator_policy_corrections")

    assert result == {"captured": False, "reason": "not_operator"}
    assert after == before


async def test_capture_is_idempotent_and_restatements_reinforce_one_policy(conn):
    suffix = get_test_identifier("policy-reinforcement")
    directive = f"Always cite the source when you summarize document {suffix}."
    first = await _capture(conn, text=directive, message_id=f"first-{suffix}")
    retry = await _capture(conn, text=directive, message_id=f"first-{suffix}")
    repeated = await _capture(conn, text=directive, message_id=f"second-{suffix}")

    assert first["captured"] is True
    assert first["outcome"] == "created"
    assert retry["outcome"] == "already_captured"
    assert retry["correction_id"] == first["correction_id"]
    assert repeated["outcome"] == "reinforced"
    assert repeated["procedural_memory_id"] == first["procedural_memory_id"]
    assert repeated["improvement_backlog_id"] == first["improvement_backlog_id"]

    policy_key = first["policy_key"]
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM operator_policy_corrections WHERE policy_key = $1",
            policy_key,
        )
        == 2
    )
    memory = await conn.fetchrow(
        """
        SELECT type::text AS type, status::text AS status, content,
               source_attribution, metadata, reinforcement_count
        FROM memories WHERE id = $1::uuid
        """,
        first["procedural_memory_id"],
    )
    assert memory["type"] == "procedural"
    assert memory["status"] == "active"
    assert memory["content"] == directive
    assert (
        _object(memory["source_attribution"])["ref"] == f"operator_policy:{policy_key}"
    )
    assert _object(memory["metadata"])["operator_policy"]["observation_count"] == 2
    assert memory["reinforcement_count"] == 1

    backlog = await conn.fetchrow(
        "SELECT status, tags, checkpoint FROM backlog WHERE id = $1::uuid",
        first["improvement_backlog_id"],
    )
    assert backlog["status"] == "todo"
    assert "operator_policy_review" in backlog["tags"]
    improvement = _object(backlog["checkpoint"])["improvement"]
    assert improvement["skill_synthesis"] == {
        "auto_authorized": False,
        "requires_review": True,
    }

    context = await conn.fetchval("SELECT render_operator_policy_context()")
    assert policy_key in context
    assert directive in context


async def test_ledger_is_append_only(conn):
    suffix = get_test_identifier("policy-immutable")
    result = await _capture(
        conn,
        text=f"Going forward, always verify source {suffix} before citing it.",
        message_id=suffix,
    )

    with pytest.raises(asyncpg.RaiseError, match="append-only"):
        async with conn.transaction():
            await conn.execute(
                "UPDATE operator_policy_corrections SET reason = 'changed' WHERE id = $1",
                result["correction_id"],
            )
    with pytest.raises(asyncpg.RaiseError, match="append-only"):
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM operator_policy_corrections WHERE id = $1",
                result["correction_id"],
            )


async def test_explicit_revoke_archives_policy_and_cancels_review_item(conn):
    suffix = get_test_identifier("policy-revoke")
    result = await _capture(
        conn,
        text=f"Never send report {suffix} without asking me first.",
        message_id=f"set-{suffix}",
    )
    revoked = _object(
        await conn.fetchval(
            "SELECT revoke_operator_policy($1, 'operator:test', 'changed preference', $2::jsonb)",
            result["policy_key"],
            json.dumps({"event_id": f"revoke-{suffix}"}),
        )
    )
    retry = _object(
        await conn.fetchval(
            "SELECT revoke_operator_policy($1, 'operator:test', 'changed preference', $2::jsonb)",
            result["policy_key"],
            json.dumps({"event_id": f"revoke-{suffix}"}),
        )
    )

    assert revoked["revoked"] is True
    assert revoked["outcome"] == "revoked"
    assert retry["outcome"] == "already_recorded"
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM active_operator_policies WHERE policy_key = $1",
            result["policy_key"],
        )
        == 0
    )
    assert (
        await conn.fetchval(
            "SELECT status::text FROM memories WHERE id = $1::uuid",
            result["procedural_memory_id"],
        )
        == "archived"
    )
    assert (
        await conn.fetchval(
            "SELECT status FROM backlog WHERE id = $1::uuid",
            result["improvement_backlog_id"],
        )
        == "cancelled"
    )
    assert result["policy_key"] not in await conn.fetchval(
        "SELECT render_operator_policy_context()"
    )


async def test_channel_operator_identity_is_explicit_and_fails_closed(conn):
    await conn.fetchval(
        "SELECT set_config('channel.slack.operator_user_id', $1::jsonb)",
        json.dumps("U-owner"),
    )
    await conn.fetchval(
        "SELECT set_config('channel.telegram.allowed_users', $1::jsonb)",
        json.dumps(["T-owner"]),
    )

    assert (
        await conn.fetchval("SELECT channel_sender_is_operator('slack', 'U-owner')")
        is True
    )
    assert (
        await conn.fetchval("SELECT channel_sender_is_operator('slack', 'U-other')")
        is False
    )
    # A conversation allowlist is intentionally not operator authority.
    assert (
        await conn.fetchval("SELECT channel_sender_is_operator('telegram', 'T-owner')")
        is False
    )

    await conn.fetchval(
        "SELECT set_config('channel.telegram.operator_user_id', $1::jsonb)",
        json.dumps("T-owner"),
    )
    assert (
        await conn.fetchval("SELECT channel_sender_is_operator('telegram', 'T-owner')")
        is True
    )
