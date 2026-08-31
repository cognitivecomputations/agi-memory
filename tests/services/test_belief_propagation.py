from __future__ import annotations

import asyncio
import json

import pytest

from services.belief_propagation import BeliefUpdateListener
from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


async def _memory(conn, content: str):
    return await conn.fetchval(
        """
        INSERT INTO memories (
            type, content, embedding, embedding_status, status,
            metadata, source_attribution
        ) VALUES (
            'semantic', $1,
            array_fill(0.1::float, ARRAY[embedding_dimension()])::vector,
            'embedded', 'active', '{"confidence":0.7}'::jsonb,
            '{"kind":"test","worker_id":"listener-source"}'::jsonb
        )
        RETURNING id
        """,
        content,
    )


async def test_listener_dispatches_and_records_delivery(db_pool):
    worker_id = f"heartbeat-{get_test_identifier('belief-listener')}"
    received: list[dict] = []
    delivered = asyncio.Event()

    async def handler(payload: dict) -> None:
        received.append(payload)
        delivered.set()

    async with db_pool.acquire() as conn:
        await conn.execute(
            "SELECT set_config('belief.propagation_subscribers', $1::jsonb)",
            json.dumps([worker_id]),
        )
    listener = BeliefUpdateListener(db_pool, worker_id=worker_id)
    listener.add_handler(handler)
    memory_id = None
    try:
        assert await listener.start()
        assert listener.started
        async with db_pool.acquire() as conn:
            memory_id = await _memory(conn, f"Listener belief {worker_id}")
        await asyncio.wait_for(delivered.wait(), timeout=3)
        ours = [item for item in received if item.get("memory_id") == str(memory_id)]
        assert ours
        log_id = int(ours[0]["log_id"])
        async with db_pool.acquire() as conn:
            deadline = asyncio.get_running_loop().time() + 3
            while not await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM belief_update_deliveries
                    WHERE log_id=$1 AND subscriber=$2
                )
                """,
                log_id,
                worker_id,
            ):
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail("belief update delivery receipt was not recorded")
                await asyncio.sleep(0.02)
    finally:
        await listener.stop()
        async with db_pool.acquire() as conn:
            if memory_id is not None:
                await conn.execute("DELETE FROM memories WHERE id=$1", memory_id)
                await conn.execute(
                    "DELETE FROM belief_update_log WHERE memory_id=$1", memory_id
                )
            await conn.execute(
                "DELETE FROM config WHERE key='belief.propagation_subscribers'"
            )


async def test_listener_honors_subscriber_filter(db_pool):
    worker_id = f"filtered-{get_test_identifier('belief-listener')}"
    async with db_pool.acquire() as conn:
        await conn.execute(
            "SELECT set_config('belief.propagation_subscribers', $1::jsonb)",
            json.dumps(["heartbeat"]),
        )
    listener = BeliefUpdateListener(db_pool, worker_id=worker_id)
    try:
        assert not await listener.start()
        assert not listener.started
    finally:
        await listener.stop()
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM config WHERE key='belief.propagation_subscribers'"
            )
