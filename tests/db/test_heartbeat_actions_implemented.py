"""Every action the heartbeat is offered can actually be performed.

`heartbeat.allowed_actions` is the menu the decision call sees, and each entry
has a configured energy cost — so an unimplemented one looks exactly like a real
capability right up until it is chosen, at which point the beat's whole model
call has been spent picking something impossible.

Three were dead: fast_ingest, hybrid_ingest and slow_ingest returned "Unknown
action". They were also redundant, since tools of the same name are bound to the
`knowledge-ingest` skill, which loads in heartbeat context.

This test executes every offered action rather than reading the source, because
reading the source is how the count was got wrong in the first place: the
handler shares multi-literal WHEN branches (`WHEN 'contemplate', 'meditate',
'study', 'debate_internally'`), and a regex over `WHEN 'x'` sees only the first
literal — which made four implemented actions look dead.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

# Actions with side effects too large to exercise here, or that end the agent.
_UNSAFE_TO_EXECUTE = {"terminate", "pause_heartbeat", "reach_out_user", "reach_out_public"}


async def test_every_allowed_action_has_a_handler(db_pool):
    async with db_pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT COALESCE(
                (SELECT value FROM config WHERE key = 'heartbeat.allowed_actions'),
                (SELECT value FROM config_defaults WHERE key = 'heartbeat.allowed_actions')
            )
            """
        )
        allowed = json.loads(raw) if isinstance(raw, str) else raw
        assert allowed, "heartbeat.allowed_actions must not be empty"

        unknown: list[str] = []
        params = json.dumps(
            {"query": "probe", "topic": "probe", "content": "probe",
             "path": "/tmp/probe.txt", "entity": "probe"}
        )
        # Executing actions writes, so the whole probe is rolled back.
        tx = conn.transaction()
        await tx.start()
        try:
            for action in allowed:
                if action in _UNSAFE_TO_EXECUTE:
                    continue
                try:
                    result = await conn.fetchval(
                        "SELECT execute_heartbeat_action(NULL::uuid, $1::text, $2::jsonb)",
                        action, params,
                    )
                except Exception:
                    # A handler that dislikes these probe parameters is still a
                    # handler. This test asks whether the action has a body, not
                    # whether it works on junk input.
                    await tx.rollback()
                    tx = conn.transaction()
                    await tx.start()
                    continue
                doc = json.loads(result) if isinstance(result, str) else (result or {})
                if str(doc.get("error", "")).startswith("Unknown action"):
                    unknown.append(action)
        finally:
            await tx.rollback()

    assert not unknown, (
        f"offered as heartbeat actions but unimplemented: {sorted(unknown)}. "
        "Either give them a handler or remove them from heartbeat.allowed_actions "
        "— an action with a price and no body spends a whole decision call."
    )


async def test_the_retired_ingest_actions_are_no_longer_offered(db_pool):
    """The tools of the same name remain; only the duplicate actions are gone."""
    async with db_pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT COALESCE(
                (SELECT value FROM config WHERE key = 'heartbeat.allowed_actions'),
                (SELECT value FROM config_defaults WHERE key = 'heartbeat.allowed_actions')
            )
            """
        )
        allowed = set(json.loads(raw) if isinstance(raw, str) else raw)

    for retired in ("fast_ingest", "hybrid_ingest", "slow_ingest"):
        assert retired not in allowed, (
            f"{retired} is offered as a heartbeat action but has no handler; "
            "the knowledge-ingest skill provides the tool instead"
        )

    # The cognitive actions that only *looked* dead are still offered.
    for live in ("contemplate", "meditate", "study", "debate_internally", "inquire_deep"):
        assert live in allowed
