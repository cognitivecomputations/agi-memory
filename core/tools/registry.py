"""
Hexis Tools System - Tool Registry

Central registry for all tools with:
- Registration and discovery
- Policy enforcement
- Execution with context
- MCP server management
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolInvocation,
    ToolResult,
    ToolSpec,
)
from .config import ToolsConfig, load_tools_config
from .hooks import HookContext, HookEvent, HookRegistry
from .policy import ToolPolicy

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


# These handlers perform a person-facing network effect in the OSS runtime.
# Registration fails if one loses its descriptor: a new/renamed provider send
# must update this inventory and declare its shape before it can start.
_NETWORK_MESSAGING_TOOL_NAMES = frozenset(
    {
        "discord_send",
        "slack_send",
        "telegram_send",
        "signal_send",
        "email_send",
        "email_send_sendgrid",
        "gmail_send",
        "gmail_reply",
        "twitter_x_post",
        "twitter_x_reply",
        "twitter_x_dm_send",
        "queue_user_message",
    }
)


def _requires_outbound_descriptor(spec: ToolSpec) -> bool:
    """Recognize provider sends, including future category additions.

    The explicit inventory catches special names such as queue_user_message.
    The shape check makes adding another send/reply/post handler in either
    person-facing category fail at startup until its transport is declared.
    """
    if spec.name in _NETWORK_MESSAGING_TOOL_NAMES:
        return True
    return (
        not spec.is_read_only
        and spec.category in {ToolCategory.EMAIL, ToolCategory.MESSAGING}
        and spec.name.endswith(("_send", "_reply", "_post"))
    )


@dataclass
class ExecutionStats:
    """Statistics for tool execution."""

    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_duration: float = 0.0
    total_energy_spent: int = 0
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    errors_by_type: dict[str, int] = field(default_factory=dict)

    def record(self, tool_name: str, result: ToolResult) -> None:
        self.total_calls += 1
        self.total_duration += result.duration_seconds
        self.total_energy_spent += result.energy_spent
        self.calls_by_tool[tool_name] = self.calls_by_tool.get(tool_name, 0) + 1

        if result.success:
            self.total_successes += 1
        else:
            self.total_failures += 1
            if result.error_type:
                key = result.error_type.value
                self.errors_by_type[key] = self.errors_by_type.get(key, 0) + 1


class ToolRegistry:
    """
    Central registry for all tools.

    Manages tool registration, discovery, and execution with policy enforcement.
    """

    def __init__(self, pool: "asyncpg.Pool"):
        self.pool = pool
        # Surfaced in reachability/audit telemetry. Builders update this when
        # dynamic tools and plugins have been loaded.
        self.registry_kind = "default"
        self._handlers: dict[str, ToolHandler] = {}
        self._mcp_handlers: dict[str, ToolHandler] = {}
        self._policy = ToolPolicy(pool)
        self._hooks = HookRegistry()
        self._stats = ExecutionStats()
        self._config_cache: ToolsConfig | None = None
        self._config_cache_time: float = 0
        self._config_cache_ttl: float = 60.0  # Refresh config every 60s
        # Skill directories contributed by plugins; the skill runtime scans
        # these in addition to the bundled/user skill dirs.
        self.extra_skill_dirs: list[Path] = []
        # The DB catalog mirrors the registered handlers, so it only needs
        # rewriting when that set changes. It was previously re-synced from
        # _evaluate_tool_policy — i.e. ~150 upserted rows on *every tool call*.
        # A flag rather than a TTL: the invalidating events are all explicit.
        self._catalog_dirty: bool = True

    @property
    def hooks(self) -> HookRegistry:
        """Access the hook registry."""
        return self._hooks

    # =========================================================================
    # Registration
    # =========================================================================

    def register(self, handler: ToolHandler) -> None:
        """Register a tool handler."""
        spec = handler.spec
        name = spec.name
        if _requires_outbound_descriptor(spec) and spec.outbound is None:
            raise ValueError(
                f"Network messaging tool {name!r} must declare ToolSpec.outbound"
            )
        if name in self._handlers:
            logger.warning(f"Overwriting existing handler for tool: {name}")
        self._handlers[name] = handler
        self._catalog_dirty = True
        logger.debug(f"Registered tool: {name}")

    def register_all(self, handlers: list[ToolHandler]) -> None:
        """Register multiple tool handlers."""
        for handler in handlers:
            self.register(handler)

    def unregister(self, name: str) -> bool:
        """Unregister a tool handler."""
        if name in self._handlers:
            del self._handlers[name]
            self._catalog_dirty = True
            return True
        if name in self._mcp_handlers:
            del self._mcp_handlers[name]
            self._catalog_dirty = True
            return True
        return False

    def register_mcp(self, handler: ToolHandler) -> None:
        """Register an MCP tool handler."""
        spec = handler.spec
        name = spec.name
        if _requires_outbound_descriptor(spec) and spec.outbound is None:
            raise ValueError(
                f"Network messaging tool {name!r} must declare ToolSpec.outbound"
            )
        if name in self._mcp_handlers:
            logger.warning(f"Overwriting existing MCP handler: {name}")
        self._mcp_handlers[name] = handler
        self._catalog_dirty = True
        logger.debug(f"Registered MCP tool: {name}")

    # =========================================================================
    # Discovery
    # =========================================================================

    def get(self, name: str) -> ToolHandler | None:
        """Get a tool handler by name."""
        return self._handlers.get(name) or self._mcp_handlers.get(name)

    def get_spec(self, name: str) -> ToolSpec | None:
        """Get a tool spec by name."""
        handler = self.get(name)
        return handler.spec if handler else None

    def list_all(self) -> list[ToolHandler]:
        """List all registered handlers."""
        return list(self._handlers.values()) + list(self._mcp_handlers.values())

    def list_by_category(self, category: ToolCategory) -> list[ToolHandler]:
        """List handlers by category."""
        return [h for h in self.list_all() if h.spec.category == category]

    def list_names(self) -> list[str]:
        """List all tool names."""
        return list(self._handlers.keys()) + list(self._mcp_handlers.keys())

    def _tool_catalog_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        mcp_names = set(self._mcp_handlers.keys())
        for handler in self.list_all():
            spec = handler.spec
            entry: dict[str, Any] = {
                "name": spec.name,
                "description": spec.description,
                "schema": spec.parameters,
                "category": spec.category.value,
                "energy_cost": spec.energy_cost,
                "requires_approval": spec.requires_approval,
                "is_read_only": spec.is_read_only,
                "supports_parallel": spec.supports_parallel,
                "optional": spec.optional,
                "allowed_contexts": [ctx.value for ctx in spec.allowed_contexts],
                "execution_kind": "python_driver",
                "metadata": (
                    {"outbound": spec.outbound.to_dict()}
                    if spec.outbound is not None
                    else {}
                ),
            }
            if spec.name in mcp_names:
                # Transport truth in the DB catalog (#41).
                entry["metadata"] = entry["metadata"] | {
                    "transport": "mcp",
                    "server": getattr(handler, "_server_name", None),
                }
            payload.append(entry)
        return payload

    async def sync_tool_catalog(self, *, force: bool = False) -> None:
        """Mirror registered Python drivers into the DB-owned tool catalog.

        No-op unless the handler set changed since the last successful sync.
        Callers on the hot path (policy evaluation runs per tool call) can call
        this freely; only registration, unregistration, and MCP attachment mark
        the catalog dirty. On failure the flag stays set, so the next call
        retries rather than leaving the DB stale.
        """
        if not (force or self._catalog_dirty):
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT sync_tool_definitions($1::jsonb)",
                    json.dumps(self._tool_catalog_payload()),
                )
            self._catalog_dirty = False
        except Exception:
            logger.debug(
                "Failed to sync DB tool catalog; using in-process registry",
                exc_info=True,
            )

    async def get_config(self, force_refresh: bool = False) -> ToolsConfig:
        """Get cached or fresh configuration."""
        now = time.time()
        if (
            force_refresh
            or self._config_cache is None
            or (now - self._config_cache_time) > self._config_cache_ttl
        ):
            self._config_cache = await load_tools_config(self.pool)
            self._config_cache_time = now
        return self._config_cache

    async def get_enabled_tools(
        self,
        context: ToolContext,
        config: ToolsConfig | None = None,
    ) -> list[ToolHandler]:
        """Get tools enabled for a specific context."""
        if config is None:
            config = await self.get_config()

        await self.sync_tool_catalog()
        enabled = []
        for handler in self.list_all():
            spec = handler.spec
            # Skip optional tools unless explicitly allowlisted
            if spec.optional and not config.is_optional_allowed(
                spec.name, spec.category
            ):
                continue
            if config.is_tool_enabled_for_context(spec.name, spec.category, context):
                if context in spec.allowed_contexts:
                    enabled.append(handler)

        return enabled

    async def get_specs(
        self,
        context: ToolContext,
        config: ToolsConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Get OpenAI function specs for enabled tools."""
        await self.sync_tool_catalog()
        try:
            async with self.pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT get_tool_specs_for_context($1::text)", context.value
                )
            specs = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(specs, list):
                return specs
        except Exception:
            logger.debug(
                "DB tool spec lookup failed; falling back to in-process specs",
                exc_info=True,
            )
        handlers = await self.get_enabled_tools(context, config)
        return [h.spec.to_openai_function() for h in handlers]

    async def get_mcp_tools(
        self,
        context: ToolContext,
        config: ToolsConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Get MCP tool specs for enabled tools."""
        await self.sync_tool_catalog()
        try:
            async with self.pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT get_tool_specs_for_context($1::text)", context.value
                )
            specs = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(specs, list):
                return [
                    {
                        "name": item.get("function", {}).get("name"),
                        "description": item.get("function", {}).get("description", ""),
                        "inputSchema": item.get("function", {}).get("parameters", {}),
                    }
                    for item in specs
                    if isinstance(item, dict) and item.get("function", {}).get("name")
                ]
        except Exception:
            logger.debug(
                "DB MCP spec lookup failed; falling back to in-process specs",
                exc_info=True,
            )
        handlers = await self.get_enabled_tools(context, config)
        return [h.spec.to_mcp_tool() for h in handlers]

    async def _evaluate_tool_policy(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
        config: ToolsConfig,
    ) -> tuple[bool, ToolResult | None, int]:
        await self.sync_tool_catalog()
        payload = {
            "tool_context": context.tool_context.value,
            "call_id": context.call_id,
            "heartbeat_id": context.heartbeat_id,
            "session_id": context.session_id,
            "energy_available": context.energy_available,
            "action_approved": context.action_approved,
            "approval_request_id": context.approval_request_id,
            "approval_channel": context.approval_channel,
        }
        try:
            async with self.pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT evaluate_tool_call($1::text, $2::jsonb, $3::jsonb)",
                    spec.name,
                    json.dumps(arguments),
                    json.dumps(payload),
                )
            decision = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(decision, dict):
                if decision.get("allowed") is True:
                    return (
                        True,
                        None,
                        int(decision.get("energy_cost", spec.energy_cost) or 0),
                    )
                error_type = ToolErrorType(
                    decision.get("error_type", ToolErrorType.EXECUTION_FAILED.value)
                )
                return (
                    False,
                    ToolResult.error_result(
                        decision.get("reason") or "Policy denied", error_type
                    ),
                    0,
                )
        except Exception:
            logger.debug(
                "DB tool policy check failed; falling back to in-process policy",
                exc_info=True,
            )

        policy_result = await self._policy.check_all(
            spec=spec,
            context=context.tool_context,
            config=config,
            energy_available=context.energy_available,
        )
        if not policy_result.allowed:
            return False, policy_result.to_result(), 0
        return True, None, config.get_energy_cost(spec.name, spec.energy_cost)

    async def preflight_outbound_controls(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
    ) -> ToolResult | None:
        """Return a refusal when STOP or a kill switch blocks a provider call.

        AgentLoop invokes this before its approval gate. The complete outbound
        authorization still runs again in ``execute`` immediately before the
        handler, so control changes cannot race the preflight.
        """
        if spec.outbound is None:
            return None
        from services.outbound_safety import preflight_tool_outbound_controls

        preparation = await preflight_tool_outbound_controls(
            self.pool,
            spec=spec.outbound,
            arguments=arguments,
        )
        return None if preparation.allowed else preparation.error_result()

    # =========================================================================
    # Execution
    # =========================================================================

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """
        Execute a tool with full policy enforcement.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            context: Execution context

        Returns:
            ToolResult with success/error status and output
        """
        # Create invocation for tracking
        invocation = ToolInvocation(
            tool_name=tool_name,
            arguments=arguments,
            context=context,
            call_id=context.call_id,
        )

        # Get handler
        handler = self.get(tool_name)
        if not handler:
            result = ToolResult.error_result(
                f"Unknown tool: {tool_name}",
                ToolErrorType.UNKNOWN_TOOL,
            )
            invocation.complete(result)
            self._stats.record(tool_name, result)
            return result

        spec = handler.spec

        # Get config
        config = await self.get_config()

        # Policy checks are DB-owned; Python falls back only if the DB policy
        # layer is unavailable in tests or non-DB contexts.
        allowed, denied_result, energy_cost = await self._evaluate_tool_policy(
            spec, arguments, context, config
        )

        if not allowed:
            result = denied_result or ToolResult.error_result("Policy denied")
            invocation.complete(result)
            self._stats.record(tool_name, result)
            logger.info(f"Tool {tool_name} denied: {result.error}")
            return result

        # Validate arguments
        validation_errors = handler.validate(arguments)
        if validation_errors:
            result = ToolResult.error_result(
                f"Validation errors: {', '.join(validation_errors)}",
                ToolErrorType.INVALID_PARAMS,
            )
            invocation.complete(result)
            self._stats.record(tool_name, result)
            return result

        # Run before-tool-call hooks
        hook_outcome = await self._hooks.run(
            HookEvent.BEFORE_TOOL_CALL,
            HookContext(
                event=HookEvent.BEFORE_TOOL_CALL,
                tool_name=tool_name,
                arguments=arguments,
            ),
        )
        if hook_outcome.block:
            result = ToolResult.error_result(
                hook_outcome.block_reason or "Blocked by hook",
                ToolErrorType.DISABLED,
            )
            invocation.complete(result)
            self._stats.record(tool_name, result)
            return result
        if hook_outcome.mutated_arguments is not None:
            arguments = hook_outcome.mutated_arguments

        outbound_preparation = None
        if spec.outbound is not None:
            from services.outbound_safety import prepare_tool_outbound

            outbound_preparation = await prepare_tool_outbound(
                self.pool,
                tool_name=tool_name,
                spec=spec.outbound,
                arguments=arguments,
                context=context,
            )
            if not outbound_preparation.allowed:
                result = outbound_preparation.error_result()
                invocation.complete(result)
                self._stats.record(tool_name, result)
                logger.info("Outbound tool %s denied: %s", tool_name, result.error)
                return result
            arguments = outbound_preparation.arguments

        # Execute with timeout
        try:
            # Set registry reference in context for nested calls
            context.registry = self

            if spec.execution_timeout_seconds is None:
                result = await handler.execute(arguments, context)
            else:
                result = await asyncio.wait_for(
                    handler.execute(arguments, context),
                    timeout=spec.execution_timeout_seconds,
                )

            # Set energy spent from config (may override default)
            result.energy_spent = energy_cost

        except asyncio.TimeoutError:
            result = ToolResult.error_result(
                f"Tool execution timed out after {spec.execution_timeout_seconds:g} seconds",
                ToolErrorType.TIMEOUT,
            )
        except asyncio.CancelledError:
            # Cancellation belongs to the containing turn (including Ctrl+C
            # in the CLI). Swallowing it here makes the agent keep running
            # after its caller has explicitly stopped waiting.
            raise
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            result = ToolResult.error_result(
                str(e),
                ToolErrorType.EXECUTION_FAILED,
            )

        if outbound_preparation is not None:
            try:
                from services.outbound_safety import finalize_tool_outbound

                await finalize_tool_outbound(self.pool, outbound_preparation, result)
            except Exception as exc:
                # The provider effect may already have happened.  Never turn a
                # successful send into an automatic retry; surface the audit
                # failure on the result and in logs instead.
                logger.exception("Failed to finalize outbound ledger for %s", tool_name)
                result.metadata["outbound_ledger_error"] = str(exc)

        invocation.complete(result)
        self._stats.record(tool_name, result)

        # Run after-tool-call hooks
        await self._hooks.run(
            HookEvent.AFTER_TOOL_CALL,
            HookContext(
                event=HookEvent.AFTER_TOOL_CALL,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                metadata={
                    "tool_context": context.tool_context.value,
                    "call_id": context.call_id,
                    "session_id": context.session_id,
                },
            ),
        )

        logger.debug(
            f"Tool {tool_name} completed: success={result.success}, "
            f"duration={result.duration_seconds:.3f}s, energy={result.energy_spent}"
        )

        return result

    async def execute_batch(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        context: ToolExecutionContext,
        parallel: bool = True,
    ) -> list[ToolResult]:
        """
        Execute multiple tools.

        Args:
            calls: List of (tool_name, arguments) tuples
            context: Shared execution context
            parallel: If True, execute parallel-safe tools concurrently

        Returns:
            List of results in same order as calls
        """
        if not parallel:
            # Sequential execution
            results = []
            for tool_name, arguments in calls:
                # Create unique call_id for each
                call_context = ToolExecutionContext(
                    tool_context=context.tool_context,
                    call_id=str(uuid.uuid4()),
                    heartbeat_id=context.heartbeat_id,
                    session_id=context.session_id,
                    is_group=context.is_group,
                    is_operator=context.is_operator,
                    surface=context.surface,
                    event_callback=context.event_callback,
                    energy_available=context.energy_available,
                    workspace_path=context.workspace_path,
                    allow_network=context.allow_network,
                    allow_shell=context.allow_shell,
                    allow_file_write=context.allow_file_write,
                    allow_file_read=context.allow_file_read,
                )
                result = await self.execute(tool_name, arguments, call_context)
                results.append(result)

                # Update energy for next call
                if context.energy_available is not None:
                    context.energy_available -= result.energy_spent

            return results

        config = await self.get_config()
        budgeting = (
            context.tool_context == ToolContext.HEARTBEAT
            and context.energy_available is not None
        )
        denied_reasons: dict[int, str] = {}
        energy_budget: dict[int, int] = {}

        if budgeting:
            remaining = int(context.energy_available or 0)
            max_per_tool = config.get_context_overrides(
                ToolContext.HEARTBEAT
            ).max_energy_per_tool

            for idx, (tool_name, _) in enumerate(calls):
                handler = self.get(tool_name)
                cost = 0
                if handler:
                    spec = handler.spec
                    if (
                        config.is_tool_enabled_for_context(
                            spec.name, spec.category, context.tool_context
                        )
                        and context.tool_context in spec.allowed_contexts
                    ):
                        cost = config.get_energy_cost(tool_name, spec.energy_cost)

                energy_budget[idx] = cost
                if max_per_tool is not None and cost > max_per_tool:
                    denied_reasons[idx] = (
                        f"Tool '{tool_name}' cost ({cost}) exceeds max per tool ({max_per_tool})"
                    )
                    continue
                if cost > remaining:
                    denied_reasons[idx] = (
                        f"Insufficient energy: need {cost}, have {remaining}"
                    )
                    continue
                remaining -= cost

            context.energy_available = remaining

        # Parallel execution: separate parallel-safe from sequential
        parallel_calls = []
        sequential_calls = []

        for i, (tool_name, arguments) in enumerate(calls):
            if i in denied_reasons:
                continue
            handler = self.get(tool_name)
            if handler and handler.spec.supports_parallel:
                parallel_calls.append((i, tool_name, arguments))
            else:
                sequential_calls.append((i, tool_name, arguments))

        results: list[tuple[int, ToolResult]] = []

        # Add pre-denied results (energy budget)
        for i in sorted(denied_reasons.keys()):
            result = ToolResult.error_result(
                denied_reasons[i],
                ToolErrorType.INSUFFICIENT_ENERGY,
            )
            results.append((i, result))

        # Run parallel calls concurrently
        if parallel_calls:

            async def run_one(
                idx: int, name: str, args: dict
            ) -> tuple[int, ToolResult]:
                energy_available = context.energy_available
                if budgeting:
                    energy_available = energy_budget.get(idx, 0)
                call_context = ToolExecutionContext(
                    tool_context=context.tool_context,
                    call_id=str(uuid.uuid4()),
                    heartbeat_id=context.heartbeat_id,
                    session_id=context.session_id,
                    is_group=context.is_group,
                    is_operator=context.is_operator,
                    surface=context.surface,
                    event_callback=context.event_callback,
                    energy_available=energy_available,
                    workspace_path=context.workspace_path,
                    allow_network=context.allow_network,
                    allow_shell=context.allow_shell,
                    allow_file_write=context.allow_file_write,
                    allow_file_read=context.allow_file_read,
                )
                result = await self.execute(name, args, call_context)
                return (idx, result)

            parallel_results = await asyncio.gather(
                *[run_one(i, n, a) for i, n, a in parallel_calls]
            )
            results.extend(parallel_results)

        # Run sequential calls in order
        for idx, tool_name, arguments in sequential_calls:
            energy_available = context.energy_available
            if budgeting:
                energy_available = energy_budget.get(idx, 0)
            call_context = ToolExecutionContext(
                tool_context=context.tool_context,
                call_id=str(uuid.uuid4()),
                heartbeat_id=context.heartbeat_id,
                session_id=context.session_id,
                is_group=context.is_group,
                is_operator=context.is_operator,
                surface=context.surface,
                event_callback=context.event_callback,
                energy_available=energy_available,
                workspace_path=context.workspace_path,
                allow_network=context.allow_network,
                allow_shell=context.allow_shell,
                allow_file_write=context.allow_file_write,
                allow_file_read=context.allow_file_read,
            )
            result = await self.execute(tool_name, arguments, call_context)
            results.append((idx, result))

        # Sort by original index and return results
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> ExecutionStats:
        """Get execution statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._stats = ExecutionStats()


class ToolRegistryBuilder:
    """Fluent builder for constructing a ToolRegistry."""

    def __init__(self, pool: "asyncpg.Pool"):
        self._pool = pool
        self._handlers: list[ToolHandler] = []
        self._exclude: set[str] = set()
        self._include_only: set[str] | None = None

    def add(self, handler: ToolHandler) -> "ToolRegistryBuilder":
        """Add a single handler."""
        self._handlers.append(handler)
        return self

    def add_all(self, handlers: list[ToolHandler]) -> "ToolRegistryBuilder":
        """Add multiple handlers."""
        self._handlers.extend(handlers)
        return self

    def exclude(self, *names: str) -> "ToolRegistryBuilder":
        """Exclude tools by name."""
        self._exclude.update(names)
        return self

    def include_only(self, *names: str) -> "ToolRegistryBuilder":
        """Only include specified tools."""
        self._include_only = set(names)
        return self

    def build(self) -> ToolRegistry:
        """Build the registry."""
        registry = ToolRegistry(self._pool)

        for handler in self._handlers:
            name = handler.spec.name

            # Check exclusions
            if name in self._exclude:
                continue

            # Check inclusion list
            if self._include_only is not None and name not in self._include_only:
                continue

            registry.register(handler)

        return registry


def create_default_registry(pool: "asyncpg.Pool") -> ToolRegistry:
    """Create a registry with all default tools (no plugins)."""
    from .memory import create_memory_tools
    from .memory_exchange import create_memory_exchange_tools
    from .journal import create_journal_tools
    from .documents import create_document_tools
    from .desk import create_desk_tools
    from .web import create_web_tools
    from .filesystem import create_filesystem_tools
    from .shell import create_shell_tools
    from .code_execution import create_code_execution_tools
    from .browser import create_browser_tools
    from .calendar import create_calendar_tools
    from .email import create_email_tools
    from .gmail_actions import create_gmail_action_tools
    from .twitter_x_actions import create_twitter_x_action_tools
    from .messaging import create_messaging_tools
    from .ingest import create_ingest_tools
    from .workflow import create_workflow_tools
    from .dynamic import create_dynamic_tools
    from .goals import create_goal_tools
    from .backlog import create_backlog_tools
    from .cron import create_cron_tools
    from .responsibilities import create_responsibility_tools
    from .sessions import create_session_tools
    from .contacts import create_contact_tools
    from .image_gen import create_image_gen_tools
    from .usage_query import create_usage_tools
    from .integrations import create_integration_tools
    from .life_integrations import create_life_integration_tools
    from .brave_search import create_brave_search_tools
    from .firecrawl import create_firecrawl_tools
    from .council import create_council_tools
    from .backup import create_backup_tools
    from .humanizer import create_humanizer_tools
    from .self_inspection import create_self_inspection_tools
    from .self_repair import create_self_repair_tools
    from .resources import create_resource_tools
    from .questions import create_question_tools
    from .local_audio_analysis import create_local_audio_analysis_tools
    from .nodes import create_node_tools
    from .speech import create_speech_tools
    from .skills import create_skill_tools
    from .hooks import AuditTrailHook

    def _env_resolver(*names: str):
        def _resolve() -> str | None:
            for name in names:
                value = os.getenv(name)
                if value:
                    return value
            return None

        return _resolve

    def _json_env_resolver(*names: str):
        def _resolve() -> dict[str, Any] | None:
            raw = _env_resolver(*names)()
            if not raw:
                return None
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

        return _resolve

    def _gmail_credentials_resolver() -> dict[str, Any] | None:
        env_credentials = _json_env_resolver(
            "GOOGLE_GMAIL_CREDENTIALS", "GOOGLE_CALENDAR_CREDENTIALS"
        )()
        if env_credentials:
            return env_credentials
        try:
            from core.auth.google_gmail import load_default_credentials

            return load_default_credentials()
        except Exception:
            return None

    builder = ToolRegistryBuilder(pool)
    builder.add_all(create_memory_tools())
    builder.add_all(create_memory_exchange_tools())
    builder.add_all(create_skill_tools())
    builder.add_all(create_journal_tools())
    builder.add_all(create_document_tools())
    builder.add_all(create_desk_tools())
    builder.add_all(create_web_tools())
    builder.add_all(create_filesystem_tools())
    builder.add_all(create_shell_tools())
    builder.add_all(create_code_execution_tools())
    builder.add_all(create_browser_tools())
    builder.add_all(
        create_calendar_tools(
            credentials_resolver=_json_env_resolver(
                "GOOGLE_CALENDAR_CREDENTIALS", "GOOGLE_GMAIL_CREDENTIALS"
            ),
        )
    )
    builder.add_all(
        create_email_tools(
            smtp_config_resolver=_json_env_resolver("EMAIL_CONFIG"),
            sendgrid_api_key_resolver=_env_resolver("SENDGRID_API_KEY"),
            sendgrid_from_email=os.getenv("SENDGRID_FROM_EMAIL"),
            gmail_credentials_resolver=_gmail_credentials_resolver,
        )
    )
    builder.add_all(create_gmail_action_tools())
    builder.add_all(create_twitter_x_action_tools())
    builder.add_all(create_integration_tools())
    builder.add_all(create_life_integration_tools())
    builder.add_all(create_messaging_tools())
    builder.add_all(create_ingest_tools())
    builder.add_all(create_workflow_tools())
    builder.add_all(create_dynamic_tools())
    builder.add_all(create_goal_tools())
    builder.add_all(create_backlog_tools())
    builder.add_all(create_cron_tools())
    builder.add_all(create_responsibility_tools())
    builder.add_all(create_session_tools())
    builder.add_all(create_contact_tools())
    builder.add_all(create_image_gen_tools())
    builder.add_all(create_usage_tools())
    builder.add_all(
        create_brave_search_tools(
            api_key_resolver=_env_resolver("BRAVE_SEARCH_API_KEY"),
        )
    )
    builder.add_all(
        create_firecrawl_tools(
            api_key_resolver=_env_resolver("FIRECRAWL_API_KEY"),
        )
    )
    builder.add_all(create_council_tools())
    builder.add_all(create_backup_tools())
    builder.add_all(create_humanizer_tools())
    builder.add_all(create_self_inspection_tools())
    builder.add_all(create_self_repair_tools())
    builder.add_all(create_resource_tools())
    builder.add_all(create_question_tools())
    builder.add_all(create_local_audio_analysis_tools())
    builder.add_all(create_speech_tools())
    builder.add_all(create_node_tools())
    from .operator_policies import create_operator_policy_tools

    builder.add_all(create_operator_policy_tools())

    registry = builder.build()

    # Register built-in audit trail hook
    registry.hooks.register(
        HookEvent.AFTER_TOOL_CALL,
        AuditTrailHook(pool),
        source="core.audit",
    )

    return registry


async def create_full_registry(pool: "asyncpg.Pool") -> ToolRegistry:
    """Create a registry with default tools + plugins + dynamic tools."""
    registry = create_default_registry(pool)

    # Load persisted dynamic tools
    try:
        from .dynamic import load_dynamic_tools

        dynamic_handlers = await load_dynamic_tools(pool)
        existing_names = set(registry.list_names())
        for handler in dynamic_handlers:
            name = handler.spec.name
            if name in existing_names:
                logger.warning(
                    "Dynamic tool '%s' conflicts with existing tool, skipping", name
                )
                continue
            registry.register(handler)
            existing_names.add(name)
    except Exception:
        logger.debug("Failed to load dynamic tools", exc_info=True)

    # Load plugins
    try:
        from plugins.loader import load_plugins

        plugin_registry = await load_plugins(pool)

        # Register plugin tools
        existing_names = set(registry.list_names())
        for handler in plugin_registry.get_tool_handlers():
            name = handler.spec.name
            if name in existing_names:
                logger.warning(
                    "Plugin tool '%s' conflicts with core tool, skipping", name
                )
                continue
            registry.register(handler)
            existing_names.add(name)

        # Register plugin hooks
        for event, handler, plugin_id in plugin_registry.get_hooks():
            registry.hooks.register(event, handler, source=plugin_id)

        # Expose plugin skill directories to the skill runtime so plugin
        # skills are discoverable and activatable like bundled ones.
        registry.extra_skill_dirs = list(plugin_registry.get_skill_dirs())

    except ImportError:
        logger.debug("Plugin system not available")
    except Exception:
        logger.exception("Failed to load plugins")

    registry.registry_kind = "full"
    return registry
