"""`hexis transcript` (#54): render a conversation -- turns, tool calls,
subconscious signals, energy, and corrections -- from the DB, end to end."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.cli]

_ROOT = str(Path(__file__).resolve().parents[2])


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "apps.hexis_cli", *argv],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=_ROOT,
    )


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


async def _seed_turn(db_pool, marker: str, session_id: str) -> str:
    async with db_pool.acquire() as conn:
        # A realistic chat user_message: internal context ahead of the stable
        # '[USER MESSAGE]' marker (see services/agent.py), then the literal text.
        user_message = (
            "## Subconscious Signals\n"
            f"- Instinct: curiosity (0.7) — test signal {marker}\n"
            "\n"
            f"[USER MESSAGE]\nwhat is {marker}?"
        )
        started = _j(
            await conn.fetchval(
                "SELECT start_agent_turn('chat', $1::text, $2::uuid, $3::jsonb)",
                user_message, session_id, json.dumps({"messages": []}),
            )
        )
        turn_id = started["turn_id"]
        await conn.fetchval(
            "SELECT record_agent_turn_event($1::uuid, 'tool_start', $2::jsonb)",
            turn_id, json.dumps({"tool_name": "recall", "arguments": {"query": marker}}),
        )
        await conn.fetchval(
            "SELECT record_agent_turn_event($1::uuid, 'tool_result', $2::jsonb)",
            turn_id, json.dumps({"tool_name": "recall", "success": True, "energy_spent": 1, "duration": 0.02}),
        )
        await conn.fetchval(
            "SELECT finish_agent_turn($1::uuid, $2::jsonb)",
            turn_id, json.dumps({
                "text": f"{marker} is a test fixture.",
                "visible_text": f"{marker} is a test fixture.",
                "iterations": 1, "energy_spent": 1,
            }),
        )
        return turn_id


async def test_transcript_session_json_and_markdown(db_pool):
    marker = get_test_identifier("clitranscript")
    session_id = str(uuid.uuid4())
    await _seed_turn(db_pool, marker, session_id)

    p = _run("transcript", "--session", session_id, "--json")
    assert p.returncode == 0, p.stderr
    turns = json.loads(p.stdout)
    assert len(turns) == 1
    assert turns[0]["session_id"] == session_id
    # --json is the raw DB record: the internal prefix is still present.
    assert "[USER MESSAGE]" in turns[0]["user_message"]

    p = _run("transcript", "--session", session_id)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    # Markdown strips the internal prefix down to what was actually typed...
    assert f"**You:** what is {marker}?" in out
    # ...but still surfaces the subconscious-signals block the issue asks for.
    assert "## Subconscious Signals" in out
    assert f"test signal {marker}" in out
    assert f"**Agent:** {marker} is a test fixture." in out
    assert "[tool] calling `recall`" in out
    assert "[tool] `recall` -> ok" in out
    assert "1 energy" in out


async def test_transcript_last_json_smoke(db_pool):
    marker = get_test_identifier("clitranscript_last")
    await _seed_turn(db_pool, marker, str(uuid.uuid4()))

    p = _run("transcript", "--last", "1", "--json")
    assert p.returncode == 0, p.stderr
    turns = json.loads(p.stdout)
    assert len(turns) == 1


async def test_transcript_no_matches_gives_clear_message(db_pool):
    p = _run("transcript", "--session", str(uuid.uuid4()))
    assert p.returncode == 0, p.stderr
    assert "No turns found" in p.stdout


async def test_transcript_last_and_session_are_mutually_exclusive(db_pool):
    p = _run("transcript", "--last", "1", "--session", str(uuid.uuid4()))
    assert p.returncode != 0
    assert "not allowed with argument" in p.stderr
