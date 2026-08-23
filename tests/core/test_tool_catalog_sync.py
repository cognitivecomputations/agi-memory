"""The DB tool catalog is written when it changes, not on every tool call.

`sync_tool_catalog()` upserts ~150 rows. It was awaited from
`_evaluate_tool_policy`, which runs once per tool execution, so a turn making six
tool calls performed six full catalog rewrites of values that cannot have changed
since process start.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.tools.registry import ToolRegistry


class _FakeConn:
    def __init__(self, counter: list[int]):
        self._counter = counter

    async def fetchval(self, *args, **kwargs):
        self._counter.append(1)
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self.sync_calls: list[int] = []

    def acquire(self):
        return _FakeConn(self.sync_calls)


def _registry() -> tuple[ToolRegistry, _FakePool]:
    pool = _FakePool()
    reg = ToolRegistry(pool)  # type: ignore[arg-type]
    reg._tool_catalog_payload = lambda: []  # type: ignore[method-assign]
    return reg, pool


@pytest.mark.asyncio
async def test_repeated_syncs_write_once():
    reg, pool = _registry()
    for _ in range(6):
        await reg.sync_tool_catalog()
    assert len(pool.sync_calls) == 1, (
        f"catalog written {len(pool.sync_calls)} times; a turn with six tool calls "
        "must not rewrite ~150 rows six times"
    )


@pytest.mark.asyncio
async def test_registering_a_tool_makes_the_catalog_dirty_again():
    reg, pool = _registry()
    await reg.sync_tool_catalog()
    assert len(pool.sync_calls) == 1

    handler = MagicMock()
    handler.spec.name = "new_tool"
    reg.register(handler)

    await reg.sync_tool_catalog()
    assert len(pool.sync_calls) == 2, "a newly registered tool must reach the catalog"


@pytest.mark.asyncio
async def test_mcp_attachment_and_unregistration_also_invalidate():
    reg, pool = _registry()
    await reg.sync_tool_catalog()

    mcp = MagicMock()
    mcp.spec.name = "mcp_tool"
    reg.register_mcp(mcp)
    await reg.sync_tool_catalog()
    assert len(pool.sync_calls) == 2

    reg.unregister("mcp_tool")
    await reg.sync_tool_catalog()
    assert len(pool.sync_calls) == 3


@pytest.mark.asyncio
async def test_a_failed_sync_is_retried_rather_than_marked_clean():
    """The flag must only clear on success, or a transient DB blip loses the write."""
    reg, _pool = _registry()
    boom = MagicMock()
    boom.acquire.side_effect = RuntimeError("db down")
    reg.pool = boom  # type: ignore[assignment]

    await reg.sync_tool_catalog()  # swallowed, stays dirty
    assert reg._catalog_dirty is True

    ok_pool = _FakePool()
    reg.pool = ok_pool  # type: ignore[assignment]
    await reg.sync_tool_catalog()
    assert len(ok_pool.sync_calls) == 1
    assert reg._catalog_dirty is False


@pytest.mark.asyncio
async def test_force_bypasses_the_flag():
    reg, pool = _registry()
    await reg.sync_tool_catalog()
    await reg.sync_tool_catalog(force=True)
    assert len(pool.sync_calls) == 2
