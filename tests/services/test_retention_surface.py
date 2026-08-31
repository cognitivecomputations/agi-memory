from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from services.retention_surface import (
    publish_retention_surface,
    resolve_memory_fade_review_from_inbound,
)


pytestmark = [pytest.mark.asyncio]


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_args):
        return None


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


async def test_publish_combines_user_asks_and_factual_receipts():
    conn = AsyncMock()
    conn.fetchval.side_effect = [
        {"skipped": False, "count": 2},
        {"skipped": False, "compression_count": 1, "source_count": 3},
    ]

    result = await publish_retention_surface(conn)

    assert result["reviews"]["count"] == 2
    assert result["compressions"]["source_count"] == 3
    assert conn.fetchval.await_count == 2


async def test_publish_reports_an_idle_tick_without_hiding_evidence():
    conn = AsyncMock()
    conn.fetchval.side_effect = [
        '{"skipped":true,"reason":"no_unpublished_reviews"}',
        '{"skipped":true,"reason":"no_unreported_compressions"}',
    ]

    result = await publish_retention_surface(conn)

    assert result == {
        "skipped": True,
        "reason": "no_retention_surface_work",
        "reviews": {"skipped": True, "reason": "no_unpublished_reviews"},
        "compressions": {
            "skipped": True,
            "reason": "no_unreported_compressions",
        },
    }


async def test_verified_inbound_resolution_uses_the_database_contract():
    conn = AsyncMock()
    conn.fetchval.return_value = '{"recognized":true,"matched":true,"decision":"keep"}'

    result = await resolve_memory_fade_review_from_inbound(
        _Pool(conn),
        channel="signal",
        actor="operator-1",
        text="keep ABC12345",
    )

    assert result["decision"] == "keep"
    conn.fetchval.assert_awaited_once_with(
        "SELECT try_resolve_memory_fade_review_from_inbound($1, $2, $3)",
        "signal",
        "operator-1",
        "keep ABC12345",
    )
