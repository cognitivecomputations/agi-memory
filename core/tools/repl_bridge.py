"""Sync-to-async tool bridge for RLM REPL environments.

Provides synchronous `tool_use()` and `list_tools()` functions that can be
called from exec() inside the REPL sandbox. Tool calls are bridged back to
the async ToolRegistry via `asyncio.run_coroutine_threadsafe()`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from core.tools.base import ToolContext, ToolExecutionContext, ToolResult

if TYPE_CHECKING:
    from core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

TOOL_CALL_TIMEOUT = 60  # seconds
TOOL_DISCOVERY_TIMEOUT = 10  # seconds


@dataclass
class ToolCallRecord:
    """Record of a single tool call made from the REPL."""

    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult | None = None
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    error: str | None = None

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def energy_spent(self) -> int:
        if self.result:
            return self.result.energy_spent
        return 0

    def to_action_taken(self) -> dict[str, Any]:
        """Convert to the actions_taken format used by apply_heartbeat_decision."""
        output = None
        if self.result:
            output = self.result.to_model_output() if self.result.success else self.result.error

        return {
            "action": self.tool_name,
            "params": self.arguments,
            "source": "rlm_repl",
            "result": {
                "success": self.result.success if self.result else False,
                "output_preview": str(output)[:500] if output else None,
                "energy_spent": self.energy_spent,
                "duration_seconds": round(self.duration, 3),
            },
        }


class ReplToolBridge:
    """
    Synchronous bridge for calling async ToolRegistry from REPL threads.

    The REPL runs in a thread pool executor. This bridge uses
    `asyncio.run_coroutine_threadsafe()` to dispatch tool calls back
    to the main async event loop.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        loop: asyncio.AbstractEventLoop,
        *,
        heartbeat_id: str | None = None,
        initial_energy: float = 20.0,
        tool_context: ToolContext = ToolContext.HEARTBEAT,
        allow_network: bool = True,
        allow_shell: bool = False,
        allow_file_write: bool = False,
    ):
        self._registry = registry
        self._loop = loop
        self._heartbeat_id = heartbeat_id
        self._remaining_energy = initial_energy
        self._tool_context = tool_context
        self._allow_network = allow_network
        self._allow_shell = allow_shell
        self._allow_file_write = allow_file_write
        self._call_records: list[ToolCallRecord] = []

    def tool_use(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute a tool synchronously from the REPL.

        Args:
            name: Tool name (e.g. "recall", "reflect")
            args: Tool arguments dict

        Returns:
            Dict with keys: success, output, energy_spent, error
        """
        args = args or {}
        record = ToolCallRecord(tool_name=name, arguments=args)

        try:
            ctx = ToolExecutionContext(
                tool_context=self._tool_context,
                call_id=record.call_id,
                heartbeat_id=self._heartbeat_id,
                energy_available=int(self._remaining_energy),
                allow_network=self._allow_network,
                allow_shell=self._allow_shell,
                allow_file_write=self._allow_file_write,
            )

            # Bridge async call from sync thread
            future = asyncio.run_coroutine_threadsafe(
                self._registry.execute(name, args, ctx),
                self._loop,
            )
            result: ToolResult = future.result(timeout=TOOL_CALL_TIMEOUT)

            record.result = result
            record.end_time = time.time()

            # Track energy
            self._remaining_energy -= result.energy_spent

            self._call_records.append(record)

            logger.info(
                "REPL tool_use: %s -> success=%s energy=%d duration=%.2fs",
                name,
                result.success,
                result.energy_spent,
                record.duration,
            )

            return {
                "success": result.success,
                "output": result.output if result.success else None,
                "error": result.error if not result.success else None,
                "energy_spent": result.energy_spent,
            }

        except TimeoutError:
            record.error = f"Tool call timed out after {TOOL_CALL_TIMEOUT}s"
            record.end_time = time.time()
            self._call_records.append(record)
            logger.warning("REPL tool_use: %s timed out", name)
            return {"success": False, "output": None, "error": record.error, "energy_spent": 0}

        except Exception as e:
            record.error = str(e)
            record.end_time = time.time()
            self._call_records.append(record)
            logger.exception("REPL tool_use: %s failed", name)
            return {"success": False, "output": None, "error": str(e), "energy_spent": 0}

    def list_tools(self) -> list[dict[str, Any]]:
        """Return available tools with descriptions, schemas, and energy costs."""
        if self._loop.is_closed():
            return self._registered_tool_entries("event loop is closed")

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            return self._registered_tool_entries("called from owning event loop thread")

        future = asyncio.run_coroutine_threadsafe(
            self._registry.get_enabled_tools(self._tool_context),
            self._loop,
        )
        try:
            handlers = future.result(timeout=TOOL_DISCOVERY_TIMEOUT)
        except TimeoutError:
            future.cancel()
            return self._registered_tool_entries(
                f"registry discovery timed out after {TOOL_DISCOVERY_TIMEOUT}s"
            )
        except Exception as exc:
            return self._registered_tool_entries(f"registry discovery failed: {exc}")

        return self._tool_entries(handlers)

    def _registered_tool_entries(self, reason: str) -> list[dict[str, Any]]:
        """Fallback to context-filtered in-process registry entries."""
        logger.warning("REPL list_tools falling back to registered tools: %s", reason)
        try:
            handlers = [
                handler
                for handler in self._registry.list_all()
                if self._tool_context in handler.spec.allowed_contexts
            ]
        except Exception as exc:
            logger.exception("REPL list_tools fallback failed")
            return [{
                "error": f"Tool inventory unavailable: {exc}",
                "name": "__tool_inventory_error__",
                "description": "The tool registry could not be inspected.",
                "energy_cost": 0,
                "category": "internal",
                "is_read_only": True,
                "parameters": {},
                "allowed_contexts": [],
            }]
        return self._tool_entries(handlers)

    @staticmethod
    def _tool_entries(handlers: list[Any]) -> list[dict[str, Any]]:
        entries = []
        for handler in handlers:
            spec = handler.spec
            entries.append({
                "name": spec.name,
                "description": spec.description,
                "energy_cost": spec.energy_cost,
                "category": spec.category.value,
                "is_read_only": spec.is_read_only,
                "requires_approval": spec.requires_approval,
                "supports_parallel": spec.supports_parallel,
                "optional": spec.optional,
                "parameters": spec.parameters,
                "allowed_contexts": sorted(context.value for context in spec.allowed_contexts),
            })
        return sorted(entries, key=lambda item: item["name"])

    def energy_remaining(self) -> float:
        """Return remaining energy budget."""
        return max(0.0, self._remaining_energy)

    def get_call_records(self) -> list[ToolCallRecord]:
        """Return all tool call records."""
        return list(self._call_records)

    def get_total_energy_spent(self) -> int:
        """Total energy spent across all REPL tool calls."""
        return sum(r.energy_spent for r in self._call_records)


def call_records_to_actions_taken(records: list[ToolCallRecord]) -> list[dict[str, Any]]:
    """Convert REPL tool call records to actions_taken format for apply_heartbeat_decision."""
    return [r.to_action_taken() for r in records if r.result is not None]
