"""
Hexis Tools System - Base Classes

Provides the foundational abstractions for the tools system:
- ToolSpec: Tool definition exposed to LLMs
- ToolResult: Structured result from tool execution
- ToolHandler: Abstract base class for tool implementations
- ToolExecutionContext: Context passed to tool execution
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolRegistry


class ToolCategory(str, Enum):
    """Categories of tools for organization and policy."""

    MEMORY = "memory"  # Memory operations (recall, remember, etc.)
    WEB = "web"  # Web search, fetch
    FILESYSTEM = "filesystem"  # File read, write, glob, grep
    SHELL = "shell"  # Command execution
    CODE = "code"  # Code execution (sandboxed REPL)
    BROWSER = "browser"  # Browser automation (Playwright/CDP)
    CALENDAR = "calendar"  # Calendar integrations
    EMAIL = "email"  # Email sending
    MESSAGING = "messaging"  # Discord, Slack, Telegram
    INGEST = "ingest"  # Content ingestion (fast, slow, hybrid)
    EXTERNAL = "external"  # MCP and custom tools


class ToolContext(str, Enum):
    """Contexts in which tools can be executed."""

    HEARTBEAT = "heartbeat"  # Autonomous heartbeat loop
    CHAT = "chat"  # Interactive conversation
    MCP = "mcp"  # External MCP client


class ToolErrorType(str, Enum):
    """Typed error categories for tool execution."""

    # General
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_PARAMS = "invalid_params"
    EXECUTION_FAILED = "execution_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

    # Policy
    CONTEXT_DENIED = "context_denied"
    INSUFFICIENT_ENERGY = "insufficient_energy"
    BOUNDARY_VIOLATION = "boundary_violation"
    APPROVAL_REQUIRED = "approval_required"
    DISABLED = "disabled"
    OUTBOUND_BLOCKED = "outbound_blocked"
    PURPOSE_REQUIRED = "purpose_required"
    CONTACT_BUDGET_EXHAUSTED = "contact_budget_exhausted"

    # Filesystem
    FILE_NOT_FOUND = "file_not_found"
    DIRECTORY_NOT_FOUND = "directory_not_found"
    PERMISSION_DENIED = "permission_denied"
    FILE_TOO_LARGE = "file_too_large"
    PATH_NOT_ALLOWED = "path_not_allowed"

    # Shell
    SHELL_DISABLED = "shell_disabled"
    SHELL_TIMEOUT = "shell_timeout"
    SHELL_EXIT_ERROR = "shell_exit_error"

    # Web
    NETWORK_ERROR = "network_error"
    HTTP_ERROR = "http_error"
    FETCH_TIMEOUT = "fetch_timeout"

    # Config
    MISSING_CONFIG = "missing_config"
    MISSING_API_KEY = "missing_api_key"
    MISSING_DEPENDENCY = "missing_dependency"

    # Auth/API
    AUTH_FAILED = "auth_failed"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True)
class OutboundSpec:
    """Declares the recipient and content carried by a messaging tool.

    The dispatcher consumes this descriptor before provider code runs.  Keeping
    transport shape here makes STOP, purpose, cadence, and disclosure policy
    impossible to bypass by adding one more provider handler.
    """

    recipient_arg: str | None
    body_arg: str
    channel: str
    additional_recipient_args: tuple[str, ...] = ()
    thread_arg: str | None = None
    html_body_arg: str | None = None
    fixed_recipient: str | None = None
    primary_recipient: bool = False
    public_recipient: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipient_arg": self.recipient_arg,
            "additional_recipient_args": list(self.additional_recipient_args),
            "body_arg": self.body_arg,
            "html_body_arg": self.html_body_arg,
            "thread_arg": self.thread_arg,
            "channel": self.channel,
            "fixed_recipient": self.fixed_recipient,
            "primary_recipient": self.primary_recipient,
            "public_recipient": self.public_recipient,
        }


@dataclass
class ToolSpec:
    """Tool definition exposed to LLMs."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    category: ToolCategory
    energy_cost: int = 1
    requires_approval: bool = False
    is_read_only: bool = True
    supports_parallel: bool = True
    optional: bool = False  # Requires explicit allowlist inclusion
    # Operator/system machinery, deliberately unbound by skills (#99): the
    # coverage test requires every non-internal tool to be reachable.
    internal: bool = False
    # Any provider action that communicates with a person or public audience
    # must declare its transport shape here.  ToolRegistry enforces the known
    # network-messaging surface at registration time.
    outbound: OutboundSpec | None = None
    # Most tools are short-lived. A tool that owns its own bounded wait (such
    # as ask_user) can opt out of the registry's generic two-minute wrapper.
    execution_timeout_seconds: float | None = 120.0
    allowed_contexts: set[ToolContext] = field(
        default_factory=lambda: {
            ToolContext.HEARTBEAT,
            ToolContext.CHAT,
            ToolContext.MCP,
        }
    )

    def __post_init__(self) -> None:
        """Expose the mandatory purpose contract on every outbound tool."""
        if self.outbound is None:
            return
        properties = self.parameters.setdefault("properties", {})
        properties.setdefault(
            "purpose_kind",
            {
                "type": "string",
                "enum": [
                    "goal",
                    "responsibility",
                    "reply",
                    "user_request",
                    "connection",
                ],
                "description": "Why this communication is legitimate.",
            },
        )
        properties.setdefault(
            "purpose_reference",
            {
                "type": "string",
                "description": (
                    "Durable goal/responsibility id, inbound thread/message id, "
                    "or trusted user-turn/session reference supporting the purpose."
                ),
            },
        )
        properties.setdefault(
            "urgency",
            {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
                "default": "normal",
                "description": "Urgency used by the contact-attention budget.",
            },
        )
        required = self.parameters.setdefault("required", [])
        for name in ("purpose_kind", "purpose_reference"):
            if name not in required:
                required.append(name)

    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_mcp_tool(self) -> dict[str, Any]:
        """Convert to MCP tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }


@dataclass
class ToolResult:
    """Structured result from tool execution."""

    success: bool
    output: Any  # For LLM consumption (JSON-serializable)
    display_output: str | None = None  # For UI display (human-readable)
    error: str | None = None
    error_type: ToolErrorType | None = None
    duration_seconds: float = 0.0
    energy_spent: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_model_output(self) -> str:
        """Format for LLM consumption."""
        import json

        if self.success:
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output, indent=2, default=str)
        return f"Error: {self.error}"

    def to_display_output(self) -> str:
        """Format for UI display."""
        if self.display_output:
            return self.display_output
        return self.to_model_output()

    def log_preview(self, max_len: int = 100) -> str:
        """Short preview for logging."""
        output = self.to_display_output()
        if len(output) > max_len:
            return output[:max_len] + "..."
        return output

    @classmethod
    def error_result(
        cls,
        error: str,
        error_type: ToolErrorType = ToolErrorType.EXECUTION_FAILED,
    ) -> "ToolResult":
        """Create an error result."""
        return cls(
            success=False,
            output=None,
            error=error,
            error_type=error_type,
        )

    @classmethod
    def success_result(
        cls,
        output: Any,
        display_output: str | None = None,
    ) -> "ToolResult":
        """Create a success result."""
        return cls(
            success=True,
            output=output,
            display_output=display_output,
        )


@dataclass
class ToolExecutionContext:
    """Context passed to tool execution."""

    tool_context: ToolContext
    call_id: str
    heartbeat_id: str | None = None
    session_id: str | None = None
    energy_available: int | None = None
    workspace_path: str | None = None
    # Group-context turn (#92/#96): recall-class tools exclude private
    # memories when the audience is a shared room.
    is_group: bool = False
    # Set only by a transport that proved this turn belongs to the configured
    # operator. Conversation allowlists are not sufficient authority.
    is_operator: bool = False
    surface: str = "chat"
    # Transport-neutral event bridge. Tool handlers emit semantic events; the
    # agent loop records and projects them to SSE, CLI, or channels.
    event_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None

    # Policy flags (can be overridden per-context)
    allow_network: bool = True
    allow_shell: bool = False
    allow_file_write: bool = False
    allow_file_read: bool = True

    # Exact one-shot proof returned by an operator approval callback. The DB
    # policy consumes this request id only when tool, arguments, and context all
    # match the approved request.
    action_approved: bool = False
    approval_request_id: str | None = None
    approval_channel: str | None = None

    # Registry reference (set by registry during execution)
    registry: "ToolRegistry | None" = None

    @property
    def trusted_goal_origin(self) -> str:
        """Map the authenticated execution surface to durable goal provenance.

        Model-authored arguments are intentionally excluded: a model cannot grant
        itself the energy and outbound permissions attached to a user-assigned goal.
        """
        if self.tool_context == ToolContext.CHAT:
            return "user_request"
        if self.tool_context == ToolContext.MCP:
            return "external"
        return "derived"

    async def emit_event(self, event: str, data: dict[str, Any]) -> None:
        if self.event_callback is not None:
            await self.event_callback(event, data)

    def resolve_path(self, path: str) -> str:
        """Resolve a path relative to workspace."""
        import os

        if self.workspace_path:
            if not os.path.isabs(path):
                return os.path.normpath(os.path.join(self.workspace_path, path))
        return os.path.normpath(path)

    def is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed workspace.

        When workspace_path is set, restricts access to that directory tree.
        When workspace_path is not set, restricts to the user's home directory
        and /tmp as a safety baseline.
        """
        import os

        if not self.workspace_path:
            # Restrict to home directory and /tmp when no workspace is configured
            target = os.path.realpath(os.path.abspath(path))
            home = os.path.expanduser("~")
            allowed_roots = [os.path.realpath(home), "/tmp"]
            return any(
                os.path.commonpath([target, root]) == root
                for root in allowed_roots
                if os.path.isdir(root)
            )

        resolved = self.resolve_path(path)
        workspace = os.path.realpath(os.path.abspath(self.workspace_path))
        target = os.path.realpath(os.path.abspath(resolved))
        try:
            return os.path.commonpath([target, workspace]) == workspace
        except ValueError:
            return False


class ToolHandler(ABC):
    """
    Base class for all tool handlers.

    Subclasses must implement:
    - spec: Property returning ToolSpec
    - execute: Async method performing the tool action
    """

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        ...

    @abstractmethod
    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """
        Execute the tool with given arguments.

        Args:
            arguments: Tool arguments (validated against spec.parameters)
            context: Execution context with policy flags and metadata

        Returns:
            ToolResult with success/error and output
        """
        ...

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        """
        Validate arguments against schema.

        Returns list of validation errors (empty if valid).
        Override for custom validation beyond JSON schema.
        """
        errors = []
        schema = self.spec.parameters

        # Check required fields
        required = schema.get("required", [])
        for schema_field in required:
            if schema_field not in arguments:
                errors.append(f"Missing required field: {schema_field}")

        # Check types for provided fields
        properties = schema.get("properties", {})
        for key, value in arguments.items():
            if key not in properties:
                continue  # Skip unknown fields (additionalProperties handling)

            prop_schema = properties[key]
            prop_type = prop_schema.get("type")

            if prop_type == "string" and not isinstance(value, str):
                errors.append(f"Field '{key}' must be a string")
            elif prop_type == "integer" and not isinstance(value, int):
                errors.append(f"Field '{key}' must be an integer")
            elif prop_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Field '{key}' must be a number")
            elif prop_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Field '{key}' must be a boolean")
            elif prop_type == "array" and not isinstance(value, list):
                errors.append(f"Field '{key}' must be an array")
            elif prop_type == "object" and not isinstance(value, dict):
                errors.append(f"Field '{key}' must be an object")

        return errors


class SyncToolHandler(ToolHandler):
    """
    Wrapper for synchronous tool implementations.

    Subclasses implement execute_sync() instead of execute().
    The wrapper handles running sync code in an executor.
    """

    @abstractmethod
    def execute_sync(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Synchronous execution method."""
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Run sync method in executor."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.execute_sync,
            arguments,
            context,
        )


@dataclass
class ToolInvocation:
    """Represents a single tool call for logging/tracking."""

    tool_name: str
    arguments: dict[str, Any]
    context: ToolExecutionContext
    call_id: str
    start_time: float = field(default_factory=time.time)
    result: ToolResult | None = None
    end_time: float | None = None

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def complete(self, result: ToolResult) -> None:
        self.result = result
        self.end_time = time.time()
        result.duration_seconds = self.duration
