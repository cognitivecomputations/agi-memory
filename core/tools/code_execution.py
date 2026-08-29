"""
Hexis Tools System - Code Execution

Exposes HexisLocalREPL as a chat/heartbeat-callable tool.
Per-session REPL instances with persistent state across calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

from core.execution_backends import (
    ExecutionBackendError,
    resolve_execution_backend,
)

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
                "Execute Python code in a restricted REPL environment. "
                "Variables persist across calls within the same session. "
                "Has access to standard Python builtins (except eval/exec/compile). "
                "Uses the operator-selected execution profile. On local execution, "
                "tool_use(name, args) can call other tools from within code; on remote "
                "execution, call those tools directly in a separate turn. "
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

        try:
            backend = await resolve_execution_backend(
                context,
                local_workspace=context.workspace_path,
            )
        except ExecutionBackendError as exc:
            return ToolResult.error_result(str(exc))

        if backend.kind != "local":
            return await self._execute_remote(code, timeout, context, backend)

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
            "backend": backend.name,
            "backend_type": backend.kind,
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
            display += "\n\nVariables:\n" + "\n".join(var_lines)

        return ToolResult(
            success=True,
            output=output,
            display_output=display,
            metadata={"execution_backend": backend.name, "backend_type": backend.kind},
        )

    async def _execute_remote(
        self,
        code: str,
        timeout: int,
        context: ToolExecutionContext,
        backend: Any,
    ) -> ToolResult:
        """Run the same code contract through a selected remote process."""
        try:
            run = await backend.run_python(
                code,
                session_id=context.session_id or "__default__",
                timeout=timeout,
            )
        except ExecutionBackendError as exc:
            return ToolResult.error_result(str(exc))
        except Exception as exc:
            logger.exception("Remote code execution failed")
            return ToolResult.error_result(f"Execution error: {exc}")

        if run.timed_out:
            detail = (
                f"Code execution timed out after {backend.cap_timeout(timeout)} seconds"
            )
            if run.timeout_detail:
                detail += f". {run.timeout_detail}"
            return ToolResult.error_result(detail)
        stderr_transport = run.stderr.decode("utf-8", errors="replace").strip()
        if run.returncode != 0:
            return ToolResult.error_result(
                stderr_transport or f"Remote Python exited with status {run.returncode}"
            )
        try:
            payload = json.loads(run.stdout.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = stderr_transport or "remote Python returned a non-JSON response"
            return ToolResult.error_result(
                f"Remote execution protocol failed: {detail}"
            )
        if not isinstance(payload, dict):
            return ToolResult.error_result(
                "Remote execution protocol failed: response was not an object"
            )

        output = {
            "stdout": str(payload.get("stdout") or ""),
            "stderr": str(payload.get("stderr") or ""),
            "variables": payload.get("variables") or {},
            "execution_time": payload.get("execution_time") or 0,
            "backend": backend.name,
            "backend_type": backend.kind,
        }
        skipped = payload.get("not_persisted") or []
        if skipped:
            output["not_persisted"] = skipped
        has_error = bool(output["stderr"].strip())
        if has_error:
            display = output["stderr"].strip()
            if output["stdout"].strip():
                display = (
                    f"Output:\n{output['stdout'].strip()}\n\n"
                    f"Error:\n{output['stderr'].strip()}"
                )
            return ToolResult(
                success=False,
                output=output,
                display_output=display,
                error=output["stderr"].strip()[:500],
                metadata={
                    "execution_backend": backend.name,
                    "backend_type": backend.kind,
                },
            )

        display = output["stdout"].strip() or "(no output)"
        if output["variables"]:
            var_lines = [
                f"  {name}: {kind}" for name, kind in output["variables"].items()
            ]
            display += "\n\nVariables:\n" + "\n".join(var_lines)
        if skipped:
            display += "\n\nNot persisted remotely: " + ", ".join(
                str(name) for name in skipped
            )
        return ToolResult(
            success=True,
            output=output,
            display_output=display,
            metadata={
                "execution_backend": backend.name,
                "backend_type": backend.kind,
            },
        )


def create_code_execution_tools() -> list[ToolHandler]:
    """Create code execution tool handlers."""
    return [CodeExecutionHandler()]
