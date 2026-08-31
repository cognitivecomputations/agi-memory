"""Tests for get_agent_transcript() (db/37, migration 0240) -- the DB-owned
assembly behind `hexis transcript` (#54): render a conversation -- turns,
tool calls, subconscious/signal events, energy, corrections -- from
agent_turns + agent_turn_events without copy-pasting out of the UI.
"""
from __future__ import annotations

import json
import uuid

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


async def _start_turn(conn, mode: str, user_message: str, *, session_id=None, heartbeat_id=None):
    context = {"messages": []}
    if heartbeat_id:
        context["heartbeat_id"] = heartbeat_id
    started = _j(
        await conn.fetchval(
            "SELECT start_agent_turn($1::text, $2::text, $3::uuid, $4::jsonb)",
            mode, user_message, session_id, json.dumps(context),
        )
    )
    return started["turn_id"]


async def test_transcript_by_session_id_includes_events_and_reply(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = get_test_identifier("transcript")
            session_id = str(uuid.uuid4())
            turn_id = await _start_turn(conn, "chat", f"hello {marker}", session_id=session_id)

            await conn.fetchval(
                "SELECT record_agent_turn_event($1::uuid, 'tool_start', $2::jsonb)",
                turn_id, json.dumps({"tool_name": "recall", "arguments": {"query": marker}}),
            )
            await conn.fetchval(
                "SELECT record_agent_turn_event($1::uuid, 'tool_result', $2::jsonb)",
                turn_id, json.dumps({
                    "tool_name": "recall", "success": True,
                    "energy_spent": 1, "duration": 0.05,
                }),
            )
            # Noise event types (raw model I/O) must not leak into the transcript.
            await conn.fetchval(
                "SELECT record_agent_turn_event($1::uuid, 'llm_request', '{}'::jsonb)",
                turn_id,
            )

            await conn.fetchval(
                "SELECT finish_agent_turn($1::uuid, $2::jsonb)",
                turn_id, json.dumps({
                    "text": f"reply to {marker}",
                    "visible_text": f"reply to {marker}",
                    "iterations": 1, "energy_spent": 1,
                }),
            )

            payload = _j(
                await conn.fetchval("SELECT get_agent_transcript(NULL, $1::uuid)", session_id)
            )
            assert len(payload) == 1
            turn = payload[0]
            assert turn["session_id"] == session_id
            assert turn["mode"] == "chat"
            assert turn["user_message"] == f"hello {marker}"
            assert turn["reply_text"] == f"reply to {marker}"
            assert turn["energy_spent"] == "1" or turn["energy_spent"] == 1

            event_types = [e["event_type"] for e in turn["events"]]
            assert event_types == ["tool_start", "tool_result"]
            assert turn["events"][1]["payload"]["success"] is True
        finally:
            await tr.rollback()


async def test_transcript_last_n_orders_ascending_across_sessions(db_pool):
    # This DB is a live instance whose own heartbeat/maintenance workers keep
    # running concurrently, so an unscoped "last N" query can legitimately
    # come back with real turns interleaved with these two. Ask for a huge N
    # (guaranteed to include both regardless of concurrent activity) and
    # check relative order only among the turns this test itself created.
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = get_test_identifier("transcript_last")
            first = await _start_turn(conn, "chat", f"first {marker}", session_id=str(uuid.uuid4()))
            await conn.fetchval(
                "SELECT finish_agent_turn($1::uuid, $2::jsonb)",
                first, json.dumps({"text": "a", "visible_text": "a"}),
            )
            second = await _start_turn(conn, "chat", f"second {marker}", session_id=str(uuid.uuid4()))
            await conn.fetchval(
                "SELECT finish_agent_turn($1::uuid, $2::jsonb)",
                second, json.dumps({"text": "b", "visible_text": "b"}),
            )

            payload = _j(await conn.fetchval("SELECT get_agent_transcript(1000, NULL)"))
            mine = [t["id"] for t in payload if t["id"] in (first, second)]
            assert mine == [first, second]  # ascending (chronological) order
        finally:
            await tr.rollback()


async def test_transcript_by_heartbeat_id(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            marker = get_test_identifier("transcript_hb")
            heartbeat_id = str(uuid.uuid4())
            turn_id = await _start_turn(conn, "heartbeat", f"go {marker}", heartbeat_id=heartbeat_id)
            await conn.fetchval(
                "SELECT finish_agent_turn($1::uuid, $2::jsonb)",
                turn_id, json.dumps({"text": f"did something {marker}"}),
            )

            payload = _j(
                await conn.fetchval("SELECT get_agent_transcript(NULL, $1::uuid)", heartbeat_id)
            )
            assert len(payload) == 1
            assert payload[0]["heartbeat_id"] == heartbeat_id
            assert payload[0]["mode"] == "heartbeat"
        finally:
            await tr.rollback()


async def test_transcript_no_matches_returns_empty_array(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            payload = _j(
                await conn.fetchval(
                    "SELECT get_agent_transcript(NULL, $1::uuid)", str(uuid.uuid4())
                )
            )
            assert payload == []
        finally:
            await tr.rollback()
