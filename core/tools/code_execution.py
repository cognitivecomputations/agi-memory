"""
Hexis Tools System - Code Execution

Exposes HexisLocalREPL as a chat/heartbeat-callable tool.
Per-session REPL instances with persistent state across calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from .base import (
    ToolCategory,
    ToolContext,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

if TYPE_CHECKING:
    from services.rlm_repl import HexisLocalREPL

logger = logging.getLogger(__name__)

# Per-session REPL instances
_session_repls: dict[str, "HexisLocalREPL"] = {}

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120


def _get_or_create_repl(session_id: str | None) -> "HexisLocalREPL":
    """Get or create a REPL instance for the session."""
    from services.rlm_repl import HexisLocalREPL

    key = session_id or "__default__"
    if key not in _session_repls:
        repl = HexisLocalREPL()
        repl.setup(context_payload=None)
        _session_repls[key] = repl
    return _session_repls[key]


def cleanup_session_repl(session_id: str) -> None:
    """Clean up a REPL instance for a session."""
    key = session_id or "__default__"
    if key in _session_repls:
        _session_repls[key].cleanup()
        del _session_repls[key]


class CodeExecutionHandler(ToolHandler):
    """Execute Python code in a sandboxed REPL."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="execute_code",
            description=(
                "Execute Python code in a sandboxed REPL environment. "
                "Variables persist across calls within the same session. "
                "Has access to standard Python builtins (except eval/exec/compile). "
                "Use tool_use(name, args) to call other tools from within code. "
                "Use FINAL_VAR('name') to return a variable as the result."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": f"Execution timeout in seconds (default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT})",
                    },
                },
                "required": ["code"],
            },
            category=ToolCategory.CODE,
            energy_cost=3,
            is_read_only=False,
            # Arbitrary Python in this process, with persistent globals and a
            # bridge to the whole tool registry — there is no sandbox. Until
            # there is one, a person says yes or it does not run. The gate in
            # core/agent_loop.py refuses when nobody is available to ask.
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        code = arguments.get("code", "")
        if not code.strip():
            return ToolResult.error_result("No code provided")

        timeout = min(
            int(arguments.get("timeout", DEFAULT_TIMEOUT)),
            MAX_TIMEOUT,
        )

        repl = _get_or_create_repl(context.session_id)

        # Wire up the tool bridge if registry is available
        if context.registry is not None and not hasattr(repl, "_bridge_installed"):
            try:
                from core.tools.repl_bridge import ReplToolBridge

                loop = asyncio.get_running_loop()
                bridge = ReplToolBridge(
                    context.registry,
                    loop,
                    tool_context=context.tool_context,
                    allow_network=context.allow_network,
                    allow_shell=context.allow_shell,
                    allow_file_write=context.allow_file_write,
                )
                repl.globals["tool_use"] = bridge.tool_use
                repl.globals["list_tools"] = bridge.list_tools
                repl._bridge_installed = True  # noqa: SLF001
            except Exception:
                logger.debug("Could not install tool bridge in REPL", exc_info=True)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(repl.execute_code, code),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult.error_result(
                f"Code execution timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult.error_result(f"Execution error: {e}")

        output = {
            "stdout": result.stdout.strip() if result.stdout else "",
            "stderr": result.stderr.strip() if result.stderr else "",
            "variables": result.local_vars,
            "execution_time": round(result.execution_time, 4),
        }

        # Check for errors in stderr
        has_error = bool(result.stderr and result.stderr.strip())

        if has_error:
            # Still return output (stdout may have useful info)
            display = result.stderr.strip()
            if result.stdout.strip():
                display = f"Output:\n{result.stdout.strip()}\n\nError:\n{result.stderr.strip()}"
            return ToolResult(
                success=False,
                output=output,
                display_output=display,
                error=result.stderr.strip()[:500],
            )

        display = result.stdout.strip() if result.stdout.strip() else "(no output)"
        if result.local_vars:
            var_lines = [f"  {k}: {v}" for k, v in result.local_vars.items()]
            display += f"\n\nVariables:\n" + "\n".join(var_lines)

        return ToolResult(
            success=True,
            output=output,
            display_output=display,
        )


def create_code_execution_tools() -> list[ToolHandler]:
    """Create code execution tool handlers."""
    return [CodeExecutionHandler()]
