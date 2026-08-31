from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.tools.base import ToolCategory, ToolContext, ToolSpec
from core.tools.config import ToolsConfig
from services import capability_probe


class _Handler:
    def __init__(self, spec: ToolSpec):
        self.spec = spec


class _Registry:
    registry_kind = "full"

    def __init__(self, handlers: list[_Handler], config: ToolsConfig | None = None):
        self.handlers = {handler.spec.name: handler for handler in handlers}
        self.config = config or ToolsConfig()

    def list_names(self):
        return list(self.handlers)

    def get(self, name):
        return self.handlers.get(name)

    async def get_config(self, force_refresh=False):  # noqa: ARG002
        return self.config


def _spec(name: str, **changes) -> ToolSpec:
    spec = ToolSpec(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        category=ToolCategory.MEMORY,
    )
    return replace(spec, **changes)


@pytest.mark.asyncio
async def test_probe_classifies_callable_and_unbound_tools(monkeypatch):
    registry = _Registry([_Handler(_spec("callable")), _Handler(_spec("orphan"))])
    monkeypatch.setattr(
        capability_probe,
        "_reachable_tool_names",
        lambda _registry, _context: {"callable"},
    )

    results = await capability_probe.probe_and_record(
        None,
        registry,
        worker_name="heartbeat",
        contexts=(ToolContext.CHAT,),
    )

    by_name = {result.tool_name: result for result in results}
    assert by_name["callable"].available is True
    assert by_name["orphan"].available is False
    assert by_name["orphan"].reason_code == "skill_unbound"
    assert all(result.registry_kind == "full" for result in results)


@pytest.mark.asyncio
async def test_probe_distinguishes_intentional_exclusions(monkeypatch):
    registry = _Registry(
        [
            _Handler(_spec("disabled")),
            _Handler(_spec("internal", internal=True)),
            _Handler(_spec("chat_only", allowed_contexts={ToolContext.CHAT})),
        ],
        ToolsConfig(disabled=["disabled"]),
    )
    monkeypatch.setattr(
        capability_probe,
        "_reachable_tool_names",
        lambda _registry, _context: set(registry.list_names()),
    )

    results = await capability_probe.probe_and_record(
        None,
        registry,
        worker_name="heartbeat",
        contexts=(ToolContext.HEARTBEAT,),
    )
    by_name = {result.tool_name: result for result in results}
    assert by_name["disabled"].reason_code == "config_disabled"
    assert by_name["internal"].reason_code == "internal_only"
    assert by_name["chat_only"].reason_code == "context_denied"


@pytest.mark.asyncio
async def test_probe_surfaces_catalog_entry_without_worker_handler(monkeypatch):
    registry = _Registry([_Handler(_spec("registered"))])
    monkeypatch.setattr(
        capability_probe,
        "_catalog_tool_names",
        lambda _pool: _async_value({"registered", "stale_catalog_tool"}),
    )
    monkeypatch.setattr(
        capability_probe,
        "_reachable_tool_names",
        lambda _registry, _context: {"registered"},
    )

    results = await capability_probe.probe_and_record(
        None,
        registry,
        worker_name="maintenance",
        contexts=(ToolContext.CHAT,),
    )
    by_name = {result.tool_name: result for result in results}
    assert by_name["stale_catalog_tool"].reason_code == "handler_not_registered"


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_due_gate_is_independent_per_worker(monkeypatch):
    capability_probe._reset_state_for_tests()
    calls: list[str] = []

    async def fake_probe(pool, registry, *, worker_name, worker_id=None):  # noqa: ARG001
        calls.append(worker_name)
        return []

    monkeypatch.setattr(capability_probe, "probe_and_record", fake_probe)
    monkeypatch.setattr(capability_probe, "_read_interval_minutes", lambda _pool: _async_value(15))
    registry = _Registry([])

    assert await capability_probe.run_probe_if_due(
        None, registry, worker_name="heartbeat"
    ) == []
    assert await capability_probe.run_probe_if_due(
        None, registry, worker_name="heartbeat"
    ) is None
    assert await capability_probe.run_probe_if_due(
        None, registry, worker_name="maintenance"
    ) == []
    assert calls == ["heartbeat", "maintenance"]


@pytest.mark.asyncio
async def test_failed_probe_is_due_again_on_next_tick(monkeypatch):
    capability_probe._reset_state_for_tests()
    attempts = 0

    async def flaky_probe(pool, registry, *, worker_name, worker_id=None):  # noqa: ARG001
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient failure")
        return []

    monkeypatch.setattr(capability_probe, "probe_and_record", flaky_probe)
    monkeypatch.setattr(capability_probe, "_read_interval_minutes", lambda _pool: _async_value(15))
    registry = _Registry([])

    with pytest.raises(RuntimeError, match="transient failure"):
        await capability_probe.run_probe_if_due(None, registry, worker_name="heartbeat")
    assert await capability_probe.run_probe_if_due(
        None, registry, worker_name="heartbeat"
    ) == []


def test_agent_runtimes_build_the_full_registry():
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "services/chat.py",
        "services/worker_service.py",
        "apps/cli_chat.py",
        "apps/tui/chat_app.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "create_full_registry" in source, relative
