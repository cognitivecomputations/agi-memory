"""
Hexis Tools System - Shell Tools

Tools for shell command execution with sandboxing and safety.
"""

from __future__ import annotations

import logging
import os
import shlex
from typing import Any

from core.execution_backends import (
    ExecutionBackendError,
    resolve_execution_backend,
)

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

logger = logging.getLogger(__name__)


# Commands that are generally safe for read-only operations
SAFE_COMMANDS = {
    "ls",
    "pwd",
    "cat",
    "head",
    "tail",
    "grep",
    "find",
    "wc",
    "date",
    "echo",
    "whoami",
    "hostname",
    "uname",
    "which",
    "file",
    "stat",
    "du",
    "df",
    "tree",
    "sort",
    "uniq",
    "cut",
    "tr",
    "diff",
    "comm",
    "join",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    # Git read-only
    "git status",
    "git log",
    "git show",
    "git diff",
    "git branch",
    "git remote",
    "git tag",
    "git describe",
    "git rev-parse",
    # Python/Node
    "python --version",
    "python3 --version",
    "node --version",
    "npm --version",
}

# Commands that should never be allowed
BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf ~/*",
    "dd",
    "mkfs",
    "fdisk",
    "parted",
    "mount",
    "umount",
    "sudo",
    "su",
    "doas",
    "chmod -R 777",
    "chown -R",
    ":(){ :|:& };:",  # Fork bomb
}


class ShellHandler(ToolHandler):
    """
    Execute shell commands with sandboxing.

    Provides controlled shell access with:
    - Command allow/block lists
    - Working directory restriction
    - Timeout enforcement
    - Output capture and truncation
    """

    def __init__(
        self,
        safe_commands_only: bool = False,
        additional_blocked: set[str] | None = None,
    ):
        """
        Args:
            safe_commands_only: Only allow commands in SAFE_COMMANDS list.
            additional_blocked: Additional command patterns to block.
        """
        self.safe_commands_only = safe_commands_only
        self.additional_blocked = additional_blocked or set()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell",
            description=(
                "Execute shell commands. Use for automation, file operations, "
                "running scripts, and system tasks. Commands run in the workspace "
                "through the operator-selected local, SSH, or remote-Docker profile."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30, max: 120).",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 120,
                    },
                    "env": {
                        "type": "object",
                        "description": "Additional environment variables.",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["command"],
            },
            category=ToolCategory.SHELL,
            energy_cost=3,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,  # Shell commands should be sequential
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors = []
        command = arguments.get("command", "")

        if not command or not command.strip():
            errors.append("command is required")

        return errors

    def _is_command_allowed(self, command: str) -> tuple[bool, str | None]:
        """
        Check if a command is allowed to execute.

        Returns (allowed, reason) tuple.
        """
        command_lower = command.lower().strip()

        # Check blocked commands
        for blocked in BLOCKED_COMMANDS | self.additional_blocked:
            if blocked in command_lower:
                return False, f"Command contains blocked pattern: {blocked}"

        # Normalize the command for evasion-resistant checks:
        # remove backslash escapes, collapse whitespace, strip env/command prefixes
        import re

        normalized = re.sub(r"\\(.)", r"\1", command_lower)  # remove backslash escapes
        normalized = re.sub(r"\s+", " ", normalized).strip()  # collapse whitespace

        # Block common evasion patterns
        dangerous_patterns = [
            (
                r"rm\s+.*-\s*r\s*.*-\s*f|rm\s+.*-\s*f\s*.*-\s*r|rm\s+-rf",
                "rm -rf variant is blocked",
            ),
            (r">\s*/dev/", "Cannot write to /dev/"),
            (r"curl\s.*\|\s*(sh|bash)", "Piping curl to shell is blocked"),
            (r"wget\s.*\|\s*(sh|bash)", "Piping wget to shell is blocked"),
            (r"\|\s*bash", "Piping to bash is discouraged"),
            (r"\$\(.*rm\b", "Command substitution with rm is blocked"),
            (r"`.*rm\b", "Backtick substitution with rm is blocked"),
            (r"eval\s", "eval is blocked"),
            (r"base64\s.*-d.*\|\s*(sh|bash)", "Encoded shell execution is blocked"),
        ]

        for pattern, reason in dangerous_patterns:
            if re.search(pattern, normalized):
                return False, reason

        # If safe_commands_only, check whitelist
        if self.safe_commands_only:
            try:
                tokens = shlex.split(command)
            except ValueError as exc:
                return False, f"Could not parse safe command: {exc}"
            first_word = tokens[0] if tokens else ""
            if first_word not in SAFE_COMMANDS:
                # Check for compound commands like "git status"
                first_two = " ".join(tokens[:2]) if len(tokens) >= 2 else first_word
                if first_two not in SAFE_COMMANDS:
                    return False, f"Command '{first_word}' not in safe commands list"
            safe, safe_reason = self._is_safe_argv(tokens)
            if not safe:
                return False, safe_reason

        return True, None

    @staticmethod
    def _is_safe_argv(tokens: list[str]) -> tuple[bool, str | None]:
        """Reject mutating modes of commands that are otherwise read-oriented.

        Safe-shell invocations are executed as direct argv, so shell operators
        are inert arguments.  This second layer handles programs such as
        ``find`` and ``git branch`` that can mutate without a shell.
        """
        if not tokens:
            return False, "Command is empty"
        command = tokens[0]
        args = tokens[1:]
        if command == "find" and any(
            arg
            in {
                "-delete",
                "-exec",
                "-execdir",
                "-ok",
                "-okdir",
                "-fprint",
                "-fprintf",
                "-fls",
            }
            for arg in args
        ):
            return False, "Mutating find actions are not allowed in safe_shell"
        if command == "date" and any(
            arg == "-s" or arg.startswith("--set") for arg in args
        ):
            return False, "Changing system time is not allowed in safe_shell"
        if command in {"sort", "tree"} and any(
            arg == "-o" or arg.startswith("--output") for arg in args
        ):
            return (
                False,
                f"Writing output files with {command} is not allowed in safe_shell",
            )
        if command == "uniq":
            positional = [arg for arg in args if not arg.startswith("-")]
            if len(positional) > 1:
                return False, "uniq output files are not allowed in safe_shell"
        if command == "file" and any(arg in {"-C", "--compile"} for arg in args):
            return False, "Compiling magic databases is not allowed in safe_shell"
        if command != "git":
            return True, None

        if not args:
            return False, "A read-only git subcommand is required"
        subcommand = args[0]
        sub_args = args[1:]
        if subcommand in {"status", "log", "show", "diff", "describe", "rev-parse"}:
            dangerous = {
                "--ext-diff",
                "--textconv",
                "--output",
                "--exec-path",
            }
            if any(
                arg in dangerous
                or any(arg.startswith(item + "=") for item in dangerous)
                for arg in sub_args
            ):
                return (
                    False,
                    f"Mutating or executable git {subcommand} options are not allowed",
                )
            return True, None
        if subcommand == "branch":
            dangerous = {
                "-d",
                "-D",
                "-m",
                "-M",
                "-c",
                "-C",
                "--delete",
                "--move",
                "--copy",
                "--edit-description",
                "--set-upstream-to",
                "--unset-upstream",
            }
            if any(
                arg in dangerous
                or any(arg.startswith(item + "=") for item in dangerous)
                for arg in sub_args
            ):
                return (
                    False,
                    "Mutating git branch options are not allowed in safe_shell",
                )
            if sub_args and not any(
                arg
                in {"--list", "-l", "-a", "--all", "-r", "--remotes", "--show-current"}
                for arg in sub_args
            ):
                return (
                    False,
                    "git branch creation is not allowed in safe_shell; use --list",
                )
            return True, None
        if subcommand == "tag":
            if not sub_args:
                return True, None
            if not any(arg in {"--list", "-l"} for arg in sub_args):
                return (
                    False,
                    "git tag creation is not allowed in safe_shell; use --list",
                )
            if any(arg in {"-d", "--delete", "-f", "--force"} for arg in sub_args):
                return False, "Mutating git tag options are not allowed in safe_shell"
            return True, None
        if subcommand == "remote":
            if not sub_args or sub_args == ["-v"]:
                return True, None
            if sub_args[0] == "get-url":
                return True, None
            if sub_args[0] == "show" and "-n" in sub_args:
                return True, None
            return False, "Only read-only git remote forms are allowed in safe_shell"
        return False, f"git {subcommand} is not in the safe_shell read-only allowlist"

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.allow_shell:
            return ToolResult.error_result(
                "Shell access not allowed in this context",
                ToolErrorType.SHELL_DISABLED,
            )

        command = arguments["command"]
        timeout = min(arguments.get("timeout", 30), 120)
        extra_env = arguments.get("env", {})

        # Validate command
        allowed, reason = self._is_command_allowed(command)
        if not allowed:
            return ToolResult.error_result(
                reason or "Command not allowed",
                ToolErrorType.PERMISSION_DENIED,
            )

        try:
            backend = await resolve_execution_backend(
                context,
                local_workspace=context.workspace_path or os.getcwd(),
            )
            if self.safe_commands_only:
                safe_env = dict(extra_env)
                safe_env.update(
                    {
                        "GIT_OPTIONAL_LOCKS": "0",
                        "GIT_PAGER": "cat",
                        "PAGER": "cat",
                    }
                )
                run = await backend.run_argv(
                    shlex.split(command),
                    timeout=timeout,
                    env=safe_env,
                )
            else:
                run = await backend.run_shell(
                    command,
                    timeout=timeout,
                    env=extra_env,
                )
            if run.timed_out:
                detail = (
                    f"Command timed out after {backend.cap_timeout(timeout)} seconds"
                )
                if run.timeout_detail:
                    detail += f". {run.timeout_detail}"
                return ToolResult.error_result(
                    detail,
                    ToolErrorType.SHELL_TIMEOUT,
                )

            # Decode output
            stdout_str = run.stdout.decode("utf-8", errors="replace")
            stderr_str = run.stderr.decode("utf-8", errors="replace")

            # Truncate if too long
            max_output = backend.settings.max_output_chars
            stdout_truncated = False
            stderr_truncated = False

            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + "\n...[truncated]"
                stdout_truncated = True

            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + "\n...[truncated]"
                stderr_truncated = True

            success = run.returncode == 0

            return ToolResult(
                success=success,
                output={
                    "command": command,
                    "backend": backend.name,
                    "backend_type": backend.kind,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": run.returncode,
                    "truncated": stdout_truncated or stderr_truncated,
                },
                display_output=stdout_str[:500]
                if success
                else f"Error: {stderr_str[:500]}",
                error=stderr_str if not success else None,
                error_type=ToolErrorType.SHELL_EXIT_ERROR if not success else None,
                metadata={
                    "execution_backend": backend.name,
                    "backend_type": backend.kind,
                },
            )

        except ExecutionBackendError as e:
            return ToolResult.error_result(str(e), ToolErrorType.MISSING_CONFIG)
        except Exception as e:
            logger.exception(f"Shell execution failed: {command[:50]}")
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


class SafeShellHandler(ShellHandler):
    """
    Shell handler that only allows safe read-only commands.

    Good for heartbeat context where more restrictive access is desired.
    """

    def __init__(self):
        super().__init__(safe_commands_only=True)

    @property
    def spec(self) -> ToolSpec:
        base_spec = super().spec
        return ToolSpec(
            name="safe_shell",
            description=(
                "Execute safe read-only shell commands. Limited to common utilities "
                "like ls, cat, grep, git status, etc. Use for inspecting files and "
                "gathering system information without making changes."
            ),
            parameters=base_spec.parameters,
            category=ToolCategory.SHELL,
            energy_cost=2,  # Lower cost for safe commands
            is_read_only=True,
            requires_approval=False,  # Safe commands don't need approval
            supports_parallel=False,
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )


class ScriptRunnerHandler(ToolHandler):
    """
    Execute a script file with controlled permissions.

    Supports Python, bash, and node scripts.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="run_script",
            description=(
                "Execute a script file. Supports Python (.py), Bash (.sh), and "
                "Node.js (.js) scripts. Uses the operator-selected execution profile "
                "with a controlled timeout and captured output."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the script file.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments to pass to the script.",
                        "default": [],
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60, max: 300).",
                        "default": 60,
                        "minimum": 1,
                        "maximum": 300,
                    },
                },
                "required": ["path"],
            },
            category=ToolCategory.SHELL,
            energy_cost=3,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.HEARTBEAT, ToolContext.CHAT},
        )

    # Map file extensions to interpreters
    INTERPRETERS = {
        ".py": ["python3"],
        ".sh": ["bash"],
        ".bash": ["bash"],
        ".js": ["node"],
        ".mjs": ["node"],
        ".ts": ["npx", "ts-node"],
    }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.allow_shell:
            return ToolResult.error_result(
                "Shell access not allowed",
                ToolErrorType.SHELL_DISABLED,
            )

        raw_path = arguments["path"]
        args = arguments.get("args", [])
        timeout = min(arguments.get("timeout", 60), 300)

        # Resolve path
        resolved_path = context.resolve_path(raw_path)

        if not context.is_path_allowed(resolved_path):
            return ToolResult.error_result(
                f"Script path not allowed: {raw_path}",
                ToolErrorType.PATH_NOT_ALLOWED,
            )

        from pathlib import Path

        script_path = Path(resolved_path)

        if not script_path.exists():
            return ToolResult.error_result(
                f"Script not found: {raw_path}",
                ToolErrorType.FILE_NOT_FOUND,
            )

        # Determine interpreter
        suffix = script_path.suffix.lower()
        interpreter = self.INTERPRETERS.get(suffix)

        if not interpreter:
            return ToolResult.error_result(
                f"Unsupported script type: {suffix}",
                ToolErrorType.INVALID_PARAMS,
            )

        try:
            backend = await resolve_execution_backend(
                context,
                local_workspace=context.workspace_path or str(script_path.parent),
            )
            if backend.kind == "local":
                backend_script = str(script_path)
            else:
                if not context.workspace_path:
                    return ToolResult.error_result(
                        "Remote script execution requires a workspace path so the local script can be mapped to the configured remote workspace.",
                        ToolErrorType.PATH_NOT_ALLOWED,
                    )
                try:
                    relative = script_path.resolve().relative_to(
                        Path(context.workspace_path).resolve()
                    )
                except ValueError:
                    return ToolResult.error_result(
                        f"Script path not inside the mapped workspace: {raw_path}",
                        ToolErrorType.PATH_NOT_ALLOWED,
                    )
                backend_script = backend.path_for(relative.as_posix())
            cmd = interpreter + [backend_script] + [str(arg) for arg in args]
            run = await backend.run_argv(cmd, timeout=timeout)
            if run.timed_out:
                detail = (
                    f"Script timed out after {backend.cap_timeout(timeout)} seconds"
                )
                if run.timeout_detail:
                    detail += f". {run.timeout_detail}"
                return ToolResult.error_result(
                    detail,
                    ToolErrorType.SHELL_TIMEOUT,
                )

            stdout_str = run.stdout.decode("utf-8", errors="replace")
            stderr_str = run.stderr.decode("utf-8", errors="replace")

            # Truncate
            max_output = backend.settings.max_output_chars
            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + "\n...[truncated]"
            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + "\n...[truncated]"

            success = run.returncode == 0

            return ToolResult(
                success=success,
                output={
                    "script": str(script_path),
                    "args": args,
                    "backend": backend.name,
                    "backend_type": backend.kind,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": run.returncode,
                },
                display_output=stdout_str[:500]
                if success
                else f"Error: {stderr_str[:500]}",
                error=stderr_str if not success else None,
                error_type=ToolErrorType.SHELL_EXIT_ERROR if not success else None,
                metadata={
                    "execution_backend": backend.name,
                    "backend_type": backend.kind,
                },
            )

        except ExecutionBackendError as e:
            return ToolResult.error_result(str(e), ToolErrorType.MISSING_CONFIG)
        except Exception as e:
            logger.exception(f"Script execution failed: {raw_path}")
            return ToolResult.error_result(str(e), ToolErrorType.EXECUTION_FAILED)


def create_shell_tools(safe_only: bool = False) -> list[ToolHandler]:
    """
    Create shell tool handlers.

    Args:
        safe_only: If True, only include SafeShellHandler (no full shell access).

    Returns:
        List of shell tool handlers.
    """
    if safe_only:
        return [SafeShellHandler()]

    return [
        ShellHandler(),
        SafeShellHandler(),
        ScriptRunnerHandler(),
    ]
