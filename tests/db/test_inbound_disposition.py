"""DB-brain contracts for inbound engagement disposition."""

from __future__ import annotations

import json

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


async def _set(conn, key: str, value) -> None:
    await conn.execute("SELECT set_config($1::text, $2::jsonb)", key, json.dumps(value))


async def _resolve(
    conn,
    *,
    channel: str = "imessage",
    sender: str = "allowed-sender",
    room: str = "test-room",
    text: str = "hello",
    metadata: dict | None = None,
    dry_run: bool = True,
) -> dict:
    raw = await conn.fetchval(
        """
        SELECT resolve_inbound_disposition(
            $1::text, $2::text, $3::text, $4::text, $5::jsonb, $6::boolean
        )
        """,
        channel,
        sender,
        room,
        text,
        json.dumps({"channel_id": room, **(metadata or {})}),
        dry_run,
    )
    return _object(raw)


async def test_defaults_are_dark_and_continuation_is_safe_noop(conn):
    assert (
        await conn.fetchval("SELECT get_config_bool('channel.disposition.enabled')")
        is False
    )
    assert (
        await conn.fetchval(
            "SELECT get_config_int('channel.imessage.disposition.continuation_window_seconds')"
        )
        == 0
    )
    assert await conn.fetchval("SELECT get_config('llm.inbound_disposition')") in (
        None,
        "null",
    )


async def test_empty_direct_and_allowlist_ceiling(conn):
    await _set(conn, "channel.imessage.allowed_handles", ["allowed-sender"])

    empty = await _resolve(conn, text=" \n\t ")
    direct = await _resolve(conn, text="hello")
    blocked = await _resolve(conn, sender="blocked-sender", text="hello")

    assert (empty["disposition"], empty["reason"]) == ("drop", "empty")
    assert (direct["disposition"], direct["reason"]) == (
        "engage",
        "allowed_conversation",
    )
    assert direct["reply_allowed"] is True
    assert (blocked["disposition"], blocked["reason"]) == (
        "observe",
        "allowlist_ceiling",
    )
    assert blocked["reply_allowed"] is False


@pytest.mark.parametrize(
    ("channel", "config_key", "candidate_kind"),
    [
        ("imessage", "channel.imessage.allowed_handles", "sender"),
        ("signal", "channel.signal.allowed_numbers", "sender"),
        ("whatsapp", "channel.whatsapp.allowed_numbers", "sender"),
        ("slack", "channel.slack.allowed_channels", "room"),
        ("telegram", "channel.telegram.allowed_chat_ids", "room"),
        ("matrix", "channel.matrix.allowed_rooms", "room"),
    ],
)
async def test_live_channel_allowlists_are_canonical(
    conn, channel, config_key, candidate_kind
):
    allowed = "allowed-sender" if candidate_kind == "sender" else "allowed-room"
    await _set(conn, config_key, [allowed])
    metadata = {"is_group": candidate_kind == "room", "is_mention": False}

    accepted = await _resolve(
        conn,
        channel=channel,
        sender="allowed-sender",
        room="allowed-room",
        metadata=metadata,
    )
    rejected = await _resolve(
        conn,
        channel=channel,
        sender="other-sender",
        room="other-room",
        metadata=metadata,
    )

    assert accepted["reply_allowed"] is True
    assert accepted["disposition"] == "engage"
    assert rejected["reply_allowed"] is False
    assert rejected["disposition"] == "observe"


async def test_native_mention_exception_does_not_bypass_discord_guild_ceiling(conn):
    await _set(conn, "channel.discord.allowed_guilds", ["guild-a"])
    await _set(conn, "channel.discord.allowed_channels", ["room-a"])

    wrong_room_mention = await _resolve(
        conn,
        channel="discord",
        room="room-b",
        metadata={"is_group": True, "guild_id": "guild-a", "is_mention": True},
    )
    wrong_guild_mention = await _resolve(
        conn,
        channel="discord",
        room="room-b",
        metadata={"is_group": True, "guild_id": "guild-b", "is_mention": True},
    )

    assert wrong_room_mention["reply_allowed"] is True
    assert wrong_guild_mention["reply_allowed"] is False


async def test_trigger_prefix_mention_anywhere_and_ambient_observation(conn):
    await _set(conn, "channel.imessage.allowed_handles", ["allowed-sender"])
    await _set(conn, "channel.imessage.disposition.trigger_word", "hexis")

    prefix = await _resolve(conn, text="Hexis: summarize this")
    mention = await _resolve(conn, text="No, @hexis, that date is wrong")
    false_prefix = await _resolve(conn, text="hexisology is a word")
    ambient = await _resolve(conn, text="talking to another person")

    assert prefix["reason"] == "trigger_match"
    assert prefix["trigger_stripped_text"] == "summarize this"
    assert mention["reason"] == "mention_match"
    assert false_prefix["disposition"] == "observe"
    assert ambient["disposition"] == "observe"
    assert ambient["reason"] == "default_observe"
    assert ambient["ambiguous"] is False


async def test_text_session_key_drives_continuation_window(conn):
    suffix = get_test_identifier("disposition-continuation")
    room = f"room-{suffix}"
    sender = f"sender-{suffix}"
    await _set(conn, "channel.imessage.allowed_handles", [sender])
    await _set(conn, "channel.imessage.disposition.trigger_word", "hexis")
    await _set(conn, "channel.imessage.disposition.continuation_window_seconds", 300)
    session_id = await conn.fetchval(
        """
        INSERT INTO channel_sessions(channel_type, channel_id, sender_id, sender_name)
        VALUES ('imessage', $1, $2, 'Test') RETURNING id
        """,
        room,
        sender,
    )
    await conn.execute(
        """
        INSERT INTO channel_messages(session_id, direction, content)
        VALUES ($1, 'outbound', 'prior answer')
        """,
        session_id,
    )

    result = await _resolve(conn, sender=sender, room=room, text="follow up")

    assert result["reason"] == "continuation_window"
    assert result["session_id"] == str(session_id)


async def test_operator_correction_wake_is_durable_and_stale_bounded(conn):
    suffix = get_test_identifier("disposition-wake")
    room = f"room-{suffix}"
    operator = f"operator-{suffix}"
    await _set(conn, "channel.disposition.enabled", True)
    await _set(conn, "channel.imessage.operator_recipient", operator)
    await _set(conn, "channel.imessage.disposition.trigger_word", "hexis")
    session_id = await conn.fetchval(
        """
        INSERT INTO channel_sessions(channel_type, channel_id, sender_id, sender_name)
        VALUES ('imessage', $1, $2, 'Operator') RETURNING id
        """,
        room,
        operator,
    )
    await conn.execute(
        """
        INSERT INTO channel_messages(session_id, direction, content)
        VALUES ($1, 'outbound', 'prior answer')
        """,
        session_id,
    )
    wake = await _resolve(
        conn,
        sender=operator,
        room=room,
        text="No, that's wrong",
        dry_run=False,
    )

    assert wake["disposition"] == "wake"
    assert wake["is_operator"] is True
    assert await conn.fetchval("SELECT has_pending_inbound_disposition_wake()")

    await conn.execute(
        """
        UPDATE inbound_disposition_events
        SET ts = CURRENT_TIMESTAMP - INTERVAL '2 hours'
        WHERE id = $1
        """,
        wake["audit_id"],
    )
    await conn.fetchval("SELECT run_heartbeat()")
    row = await conn.fetchrow(
        """
        SELECT wake_outcome, wake_heartbeat_id
        FROM inbound_disposition_events WHERE id = $1
        """,
        wake["audit_id"],
    )
    assert row["wake_outcome"] == "stale"
    assert row["wake_heartbeat_id"] is None


async def test_classifier_finalizer_cannot_exceed_allowlist_or_operator_authority(conn):
    await _set(conn, "channel.imessage.allowed_handles", ["someone-else"])
    blocked = await _resolve(conn, sender="blocked", dry_run=False)

    with pytest.raises(Exception, match="allowlist ceiling"):
        async with conn.transaction():
            await conn.fetchval(
                "SELECT finalize_inbound_disposition($1, 'engage', 'test')",
                blocked["audit_id"],
            )

    allowed = await _resolve(
        conn,
        sender="someone-else",
        dry_run=False,
    )
    with pytest.raises(Exception, match="identity-verified operator"):
        async with conn.transaction():
            await conn.fetchval(
                "SELECT finalize_inbound_disposition($1, 'wake', 'test')",
                allowed["audit_id"],
            )


async def test_audit_and_passive_ledger_are_idempotent_by_platform_message(conn):
    suffix = get_test_identifier("disposition-passive")
    sender = f"sender-{suffix}"
    room = f"room-{suffix}"
    await _set(conn, "channel.imessage.allowed_handles", [])
    result = await _resolve(
        conn,
        sender=sender,
        room=room,
        text="ambient message",
        metadata={"platform_message_id": f"message-{suffix}"},
        dry_run=False,
    )
    before = await conn.fetchval(
        "SELECT count(*) FROM inbound_disposition_events WHERE id = $1",
        result["audit_id"],
    )
    first = _object(
        await conn.fetchval(
            """
            SELECT record_inbound_disposition_observation(
                $1, 'imessage', $2, $3, 'Sender', 'ambient message', $4, '{}'::jsonb
            )
            """,
            result["audit_id"],
            room,
            sender,
            f"message-{suffix}",
        )
    )
    second = _object(
        await conn.fetchval(
            """
            SELECT record_inbound_disposition_observation(
                $1, 'imessage', $2, $3, 'Sender', 'ambient message', $4, '{}'::jsonb
            )
            """,
            result["audit_id"],
            room,
            sender,
            f"message-{suffix}",
        )
    )

    assert before == 1
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["message_id"] == first["message_id"]
    metadata = _object(
        await conn.fetchval(
            "SELECT metadata FROM channel_messages WHERE id = $1::uuid",
            first["message_id"],
        )
    )
    assert metadata["passive"] is True
    assert metadata["inbound_disposition_event_id"] == result["audit_id"]
