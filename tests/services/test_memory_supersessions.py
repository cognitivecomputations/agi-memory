from __future__ import annotations

from uuid import uuid4

import pytest

from services.memory_supersessions import (
    Supersession,
    active_supersession_for,
    record_supersession,
    revert_supersession,
    supersession_history_for,
)
from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


async def _memory(conn, content: str):
    return await conn.fetchval(
        """
        INSERT INTO memories (type, content, embedding, embedding_status, status)
        VALUES (
            'semantic', $1,
            array_fill(0.1::float, ARRAY[embedding_dimension()])::vector,
            'embedded', 'active'
        )
        RETURNING id
        """,
        content,
    )


async def test_service_requires_database_and_complete_provenance():
    with pytest.raises(ValueError, match="database pool"):
        await record_supersession(
            None,
            Supersession(superseded_memory_id=uuid4(), reason="reason", actor="test"),
        )
    with pytest.raises(ValueError, match="reason"):
        await record_supersession(
            object(), Supersession(superseded_memory_id=uuid4(), actor="test")
        )


async def test_service_records_reads_history_and_reverts(db_pool):
    marker = get_test_identifier("memory-supersession-service")
    async with db_pool.acquire() as conn:
        old_id = await _memory(conn, f"old {marker}")
        replacement_id = await _memory(conn, f"replacement {marker}")
    try:
        event_id = await record_supersession(
            db_pool,
            Supersession(
                superseded_memory_id=old_id,
                replacement_memory_id=replacement_id,
                reason="service correction",
                actor="test",
                replacement_planned=True,
                metadata={"marker": marker},
            ),
        )
        current = await active_supersession_for(db_pool, old_id)
        assert current is not None
        assert current.id == event_id
        assert current.replacement_memory_id == replacement_id
        assert current.reason == "service correction"
        assert current.metadata["marker"] == marker

        history = await supersession_history_for(db_pool, replacement_id)
        assert [event.id for event in history] == [event_id]

        assert await revert_supersession(
            db_pool,
            event_id,
            reason="operator restored original",
            actor="operator",
        )
        assert await active_supersession_for(db_pool, old_id) is None
        history = await supersession_history_for(db_pool, old_id)
        assert history[0].status == "reverted"
        assert history[0].metadata["reverted_by"] == "operator"
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM memories WHERE id = ANY($1::uuid[])",
                [old_id, replacement_id],
            )
