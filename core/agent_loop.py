"""
Hexis Unified Agent Loop

A single agentic loop shared by both chat and heartbeat contexts.
The LLM calls tools via the standard tool_use API, with results fed
back into the conversation for self-correction.

Differences between contexts are confined to:
- System prompt (chat vs heartbeat)
- Energy budget (None = unlimited for chat; int for heartbeat)
- Approval mechanism (callback for interactive; DB-based for autonomous)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, TYPE_CHECKING

from core.llm import chat_completion, stream_chat_completion
from core.tools.base import ToolContext, ToolExecutionContext
from core.usage import record_llm_usage

if TYPE_CHECKING:
    import asyncpg
    from core.tools.config import ContextOverrides
    from core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _trace_safe_content(content: Any) -> Any:
    if isinstance(content, str):
        if content.startswith("data:image/"):
            return "[image data redacted]"
        return content
    if not isinstance(content, list):
        return content

    safe_parts: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            safe_parts.append(part)
            continue
        ptype = part.get("type")
        if ptype in {"image_url", "input_image"}:
            redacted = dict(part)
            if isinstance(redacted.get("image_url"), dict):
                image_url = dict(redacted["image_url"])
                image_url["url"] = "[image data redacted]"
                redacted["image_url"] = image_url
            elif "image_url" in redacted:
                redacted["image_url"] = "[image data redacted]"
            if "url" in redacted:
                redacted["url"] = "[image data redacted]"
            safe_parts.append(redacted)
            continue
        safe_parts.append(part)
    return safe_parts


def _trace_safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        next_msg = dict(msg)
        next_msg["content"] = _trace_safe_content(next_msg.get("content"))
        safe.append(next_msg)
    return safe


def _model_safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove Hexis-only message metadata before calling a provider.

    Chat history carries durable action receipts in ``metadata`` so the DB can
    audit historical claims. Provider message schemas do not understand that
    field; the human-readable receipt is supplied as a normal system message.
    """
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    return [
        {key: value for key, value in message.items() if key in allowed}
        for message in messages
        if isinstance(message, dict)
    ]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class AgentEvent(str, Enum):
    """Events emitted during the agent loop."""

    LOOP_START = "loop_start"
    TEXT_DELTA = "text_delta"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUEST = "approval_request"
    ENERGY_EXHAUSTED = "energy_exhausted"
    LOOP_END = "loop_end"
    ERROR = "error"
    PHASE_CHANGE = "phase_change"
    CONTINUATION = "continuation"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    CLAIM_FLAGGED = "claim_flagged"
    UI_ARTIFACT = "ui_artifact"
    QUESTION = "question"


@dataclass
class AgentEventData:
    """Payload for an agent loop event."""

    event: AgentEvent
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentLoopConfig:
    """Configuration for an agent loop run."""

    tool_context: ToolContext
    system_prompt: str
    llm_config: dict[str, Any]  # {provider, model, endpoint, api_key}
    registry: "ToolRegistry"
    pool: "asyncpg.Pool"

    # Energy budget — None means unlimited (chat mode)
    energy_budget: int | None = None

    # Limits
    max_iterations: int | None = None  # None = timeout-based only
    timeout_seconds: float = 300.0

    # LLM params
    temperature: float = 0.7
    max_tokens: int = 4096

    # Session
    session_id: str | None = None
    is_group: bool = False
    is_operator: bool = False
    heartbeat_id: str | None = None

    # Callbacks
    on_event: Callable[[AgentEventData], Awaitable[None]] | None = None
    on_approval: (
        Callable[[str, dict[str, Any]], Awaitable[bool | dict[str, Any]]] | None
    ) = None

    # Planning phases (Gap 1)
    enable_planning: bool = False
    planning_prompt: str | None = None
    verify_prompt: str | None = None

    # Runtime permission overrides (Gap 4)
    context_overrides: "ContextOverrides | None" = None

    # Skill-first routing: when set, only these tool schemas are exposed to the
    # model. `use_skill` can expand the set mid-turn.
    allowed_tool_names: set[str] | None = None
    # Names only, for telemetry: which skills were active when a tool was refused.
    active_skill_names: list[str] = field(default_factory=list)
    # Immutable surface-audit context. The prompt itself is never stored there.
    surface: str = "chat"
    tool_surface_input_hash: str = ""
    available_skill_count: int = 0

    # Continuation nudge (Gap 5)
    continuation_prompt: str | None = None
    max_continuations: int = 0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class AgentLoopResult:
    """Result of a completed agent loop run."""

    text: str
    messages: list[dict[str, Any]]
    tool_calls_made: list[dict[str, Any]]
    iterations: int
    energy_spent: int
    turn_id: str | None = None
    visible_text: str = ""
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    timed_out: bool = False
    stopped_reason: str = "completed"
    plan_text: str = ""
    phases_completed: list[str] = field(default_factory=list)
    continuations_used: int = 0


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """
    Unified agentic loop for Hexis.

    Chat and heartbeat share the same loop. The only differences are the
    system prompt and energy budget, configured via AgentLoopConfig.

    Usage::

        config = AgentLoopConfig(
            tool_context=ToolContext.CHAT,
            system_prompt="...",
            llm_config=normalized,
            registry=registry,
            pool=pool,
        )
        agent = AgentLoop(config)
        result = await agent.run("Hello!")
    """

    def __init__(self, config: AgentLoopConfig) -> None:
        self.config = config
        self._energy_spent: int = 0
        self._iteration_count: int = 0
        self._tool_calls_made: list[dict[str, Any]] = []
        self._tool_results: list[dict[str, Any]] = []
        self._last_text: str = ""
        self._visible_text_parts: list[str] = []
        self._streaming: bool = False
        self._continuations_used: int = 0
        self._plan_text: str = ""
        self._phases_completed: list[str] = []
        self._turn_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
        user_content: Any | None = None,
    ) -> AgentLoopResult:
        """Run the agent loop to completion.

        The DB (``agent_turns``) owns the authoritative message log, energy
        accounting and stop decisions. Python builds only the *initial*
        messages, then reads the conversation back from the DB each step.
        """
        # Two system messages when the prompt knows where its cacheable prefix
        # ends (services.agent.SystemPrompt): providers reuse the stable one and
        # re-read only the volatile one. A plain string still works as before.
        sp = self.config.system_prompt
        stable = getattr(sp, "stable", None)
        volatile = getattr(sp, "volatile", "")
        if stable:
            messages: list[dict[str, Any]] = [{"role": "system", "content": stable}]
            if volatile:
                messages.append({"role": "system", "content": volatile})
        else:
            messages = [{"role": "system", "content": sp}]
        messages.extend(history or [])
        messages.append(
            {
                "role": "user",
                "content": user_content if user_content is not None else user_message,
            }
        )

        tools = await self._load_tools_for_turn()
        # Fail loud: turn state is authoritative, so a failed start must surface
        # rather than silently degrading to a Python-only loop.
        await self._start_turn(user_message, messages)

        await self._emit(
            AgentEvent.LOOP_START,
            {
                "tool_context": self.config.tool_context.value,
                "energy_budget": self.config.energy_budget,
                "tool_count": len(tools),
            },
        )

        try:
            result = await asyncio.wait_for(
                self._loop(tools),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = self._make_result(await self._get_messages(), "timeout")
            result.timed_out = True

        await self._emit(
            AgentEvent.LOOP_END,
            {
                "stopped_reason": result.stopped_reason,
                "iterations": result.iterations,
                "energy_spent": result.energy_spent,
                "timed_out": result.timed_out,
            },
        )
        # The DB message log is authoritative for the final transcript.
        result.messages = await self._get_messages()
        await self._enforce_action_claims(result)
        await self._finish_turn(result)

        return result

    async def stream(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
        user_content: Any | None = None,
    ) -> AsyncIterator[AgentEventData]:
        """
        Streaming variant of run().

        Yields AgentEventData as they happen. Callers can filter by
        event type (e.g. TEXT_DELTA for text streaming).
        """
        queue: asyncio.Queue[AgentEventData | None] = asyncio.Queue()
        original_on_event = self.config.on_event

        async def _enqueue(event: AgentEventData) -> None:
            await queue.put(event)
            if original_on_event:
                await original_on_event(event)

        self.config.on_event = _enqueue
        self._streaming = True

        # Run loop in background task
        task = asyncio.create_task(
            self.run(user_message, history, user_content=user_content)
        )

        # Signal completion via sentinel
        def _on_done(_: asyncio.Task) -> None:  # type: ignore[type-arg]
            queue.put_nowait(None)

        task.add_done_callback(_on_done)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            # Restore original callback
            self.config.on_event = original_on_event
            # Ensure task exceptions propagate
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            elif task.exception():
                raise task.exception()  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(
        self,
        tools: list[dict[str, Any]],
    ) -> AgentLoopResult:
        """Dispatcher: routes to planned or direct execution loop."""
        if not self.config.enable_planning:
            return await self._execute_loop(tools)
        return await self._planned_loop(tools)

    async def _load_tools_for_turn(self) -> list[dict[str, Any]]:
        tools = await self.config.registry.get_specs(self.config.tool_context)
        allowed = self.config.allowed_tool_names
        if allowed is None:
            # Sole-front-door safety net (#41): callers that skip skill routing
            # never see MCP tool schemas — those are reachable only through a
            # skill's bound_tools (or the mcp.expose_unbound escape hatch).
            if any(
                spec.get("function", {}).get("name", "").startswith("mcp_")
                for spec in tools
            ):
                expose_unbound = False
                try:
                    async with self.config.pool.acquire() as conn:
                        expose_unbound = bool(
                            await conn.fetchval(
                                "SELECT COALESCE(get_config_bool('mcp.expose_unbound'), FALSE)"
                            )
                        )
                except Exception:
                    logger.debug(
                        "mcp.expose_unbound lookup failed; hiding unbound MCP tools",
                        exc_info=True,
                    )
                if not expose_unbound:
                    tools = [
                        spec
                        for spec in tools
                        if not spec.get("function", {})
                        .get("name", "")
                        .startswith("mcp_")
                    ]
            return tools
        return [
            spec for spec in tools if spec.get("function", {}).get("name") in allowed
        ]

    async def _llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Dispatch a single LLM call (streaming or non-streaming).

        Returns the raw response dict with 'content' and 'tool_calls'.
        Raises on LLM failure (caller is responsible for error handling).
        """
        cfg = self.config
        llm = cfg.llm_config

        model_messages = _model_safe_messages(messages)
        await self._emit(
            AgentEvent.LLM_REQUEST,
            {
                "iteration": self._iteration_count,
                "provider": llm.get("provider"),
                "model": llm.get("model"),
                "messages": _trace_safe_messages(model_messages),
                "tools": [
                    spec.get("function", {}).get("name", "unknown")
                    for spec in (tools or [])
                ],
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
            },
        )

        if self._streaming:

            async def _on_text_delta(token: str) -> None:
                await self._emit(
                    AgentEvent.TEXT_DELTA,
                    {
                        "text": token,
                        "iteration": self._iteration_count,
                    },
                )

            result = await stream_chat_completion(
                provider=llm["provider"],
                model=llm["model"],
                endpoint=llm.get("endpoint"),
                api_key=llm.get("api_key"),
                messages=model_messages,
                tools=tools if tools else None,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                on_text_delta=_on_text_delta,
                auth_mode=llm.get("auth_mode"),
            )
        else:
            result = await chat_completion(
                provider=llm["provider"],
                model=llm["model"],
                endpoint=llm.get("endpoint"),
                api_key=llm.get("api_key"),
                messages=model_messages,
                tools=tools if tools else None,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                auth_mode=llm.get("auth_mode"),
            )

        # Record API usage (fire-and-forget)
        source = "heartbeat" if cfg.heartbeat_id else "chat"
        session_key = cfg.session_id or cfg.heartbeat_id
        asyncio.ensure_future(
            record_llm_usage(
                provider=llm["provider"],
                model=llm["model"],
                raw_response=result.get("raw"),
                operation="stream" if self._streaming else "chat",
                session_key=session_key,
                source=source,
                pool=cfg.pool,
            )
        )

        await self._emit(
            AgentEvent.LLM_RESPONSE,
            {
                "iteration": self._iteration_count,
                "provider": llm.get("provider"),
                "model": llm.get("model"),
                "content": result.get("content") or "",
                "tool_calls": result.get("tool_calls") or [],
            },
        )

        return result

    async def _execute_loop(
        self,
        tools: list[dict[str, Any]],
    ) -> AgentLoopResult:
        """Core agentic loop: LLM -> tool calls -> results -> LLM.

        The DB drives the loop: ``next_agent_step`` decides stop-vs-continue
        (energy/iteration budgets) and hands back the authoritative message
        log; each LLM/tool result is applied back to ``agent_turns`` so the
        next step sees the updated conversation. Python owns only the model
        call, tool execution and event emission.
        """
        cfg = self.config

        while True:
            db_step = await self._next_agent_step()
            if db_step.get("action") == "stop":
                reason = db_step.get("reason") or "completed"
                if reason == "energy":
                    await self._emit(
                        AgentEvent.ENERGY_EXHAUSTED,
                        {
                            "budget": cfg.energy_budget,
                            "spent": self._energy_spent,
                        },
                    )
                return self._make_result(await self._get_messages(), reason)

            # DB owns iteration/energy budgets; trust its decision above and use
            # the message log it hands back for this LLM call (no parallel list).
            self._iteration_count = int(
                db_step.get("iteration", self._iteration_count + 1)
            )
            messages = db_step.get("messages")
            if messages is None:
                messages = await self._get_messages()

            # LLM call
            try:
                response = await self._llm_call(messages, tools)
            except Exception as e:
                logger.error(
                    "LLM call failed at iteration %d: %s", self._iteration_count, e
                )
                await self._emit(
                    AgentEvent.ERROR,
                    {"error": str(e), "iteration": self._iteration_count},
                )
                return self._make_result(await self._get_messages(), "error")

            text = response.get("content", "") or ""
            tool_calls = response.get("tool_calls") or []
            # Record the assistant message in the DB (OpenAI-format tool_calls),
            # which appends to agent_turns.messages and bumps the iteration count.
            await self._apply_llm_result(response)

            if text:
                self._last_text = text
                self._visible_text_parts.append(text)
                # Only emit per-iteration TEXT_DELTA in non-streaming mode
                # (streaming mode emits per-token via the callback)
                if not self._streaming:
                    await self._emit(
                        AgentEvent.TEXT_DELTA,
                        {"text": text, "iteration": self._iteration_count},
                    )

            if not tool_calls:
                if (
                    cfg.continuation_prompt is not None
                    and self._continuations_used < cfg.max_continuations
                ):
                    self._continuations_used += 1
                    await self._emit(
                        AgentEvent.CONTINUATION,
                        {
                            "continuation_number": self._continuations_used,
                            "max_continuations": cfg.max_continuations,
                        },
                    )
                    await self._append_user_message(cfg.continuation_prompt)
                    continue
                return self._make_result(await self._get_messages(), "completed")

            # Process tool calls. Energy budgets are enforced at the step
            # boundary by next_agent_step, so every tool the model requested in
            # one turn runs before the next budget check.
            for call in tool_calls:
                tool_name = call.get("name", "")
                arguments = call.get("arguments", {})
                call_id = call.get("id") or str(uuid.uuid4())

                if (
                    cfg.allowed_tool_names is not None
                    and tool_name not in cfg.allowed_tool_names
                ):
                    await self._record_tool_result(
                        call_id,
                        {
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "success": False,
                            "error": "tool not available in the active skill set",
                            "energy_spent": 0,
                            "model_output": (
                                f"Tool '{tool_name}' is not available. Use list_skills/use_skill "
                                "to activate the relevant skill first."
                            ),
                        },
                    )
                    self._tool_calls_made.append(
                        {
                            "id": call_id,
                            "name": tool_name,
                            "arguments": arguments,
                            "success": False,
                            "error": "not_available_in_active_skills",
                            "energy_spent": 0,
                        }
                    )
                    await self._record_gate_refusal(
                        tool_name, "not_available_in_active_skills"
                    )
                    continue

                spec = cfg.registry.get_spec(tool_name)

                # STOP is above every other communication gate, including the
                # act of asking the operator to approve a provider send. The
                # registry repeats the full authorization just before delivery,
                # so this is a no-side-effect ordering preflight rather than the
                # final policy decision.
                if spec and spec.outbound is not None:
                    outbound_refusal = await cfg.registry.preflight_outbound_controls(
                        spec, arguments
                    )
                    if outbound_refusal is not None:
                        await self._record_tool_result(
                            call_id,
                            {
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "success": False,
                                "error": outbound_refusal.error,
                                "error_type": (
                                    outbound_refusal.error_type.value
                                    if outbound_refusal.error_type
                                    else None
                                ),
                                "energy_spent": 0,
                                "model_output": outbound_refusal.to_model_output(),
                            },
                        )
                        self._tool_calls_made.append(
                            {
                                "id": call_id,
                                "name": tool_name,
                                "arguments": arguments,
                                "success": False,
                                "error": "outbound_control_blocked",
                                "energy_spent": 0,
                            }
                        )
                        await self._record_gate_refusal(
                            tool_name, "outbound_control_blocked"
                        )
                        continue

                # Approval gate. A tool marked `requires_approval` needs a person
                # to say yes. When no approver is wired — the heartbeat, the API
                # chat path — that is not permission, it is the absence of anyone
                # to ask, so the call is refused rather than waved through.
                approval_evidence: dict[str, Any] = {}
                if spec and spec.requires_approval:
                    await self._emit(
                        AgentEvent.APPROVAL_REQUEST,
                        {
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "approver": (
                                "none" if cfg.on_approval is None else "callback"
                            ),
                        },
                    )

                    if cfg.on_approval is None:
                        await self._record_tool_result(
                            call_id,
                            {
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "success": False,
                                "error": "no_approver_available",
                                "energy_spent": 0,
                                "model_output": (
                                    f"'{tool_name}' needs a person's approval and nobody is "
                                    "available to give it right now. It was not run. Either "
                                    "do this another way, or use queue_user_message to ask "
                                    "for it and pick it up once they answer."
                                ),
                            },
                        )
                        self._tool_calls_made.append(
                            {
                                "id": call_id,
                                "name": tool_name,
                                "arguments": arguments,
                                "success": False,
                                "error": "no_approver_available",
                                "energy_spent": 0,
                            }
                        )
                        await self._record_gate_refusal(
                            tool_name, "no_approver_available"
                        )
                        continue

                    try:
                        decision = await cfg.on_approval(tool_name, arguments)
                    except Exception:
                        logger.warning(
                            "Approval callback raised for %s; treating as denied",
                            tool_name,
                            exc_info=True,
                        )
                        decision = False

                    if isinstance(decision, dict):
                        approved = bool(
                            decision.get("approved") and decision.get("request_id")
                        )
                        approval_evidence = {
                            "request_id": decision.get("request_id"),
                            "channel": decision.get("approval_channel"),
                        }
                        denial_reason = str(
                            decision.get("reason")
                            or (
                                "Approval returned no exact request proof."
                                if decision.get("approved")
                                else f"Approval ended with status {decision.get('status') or 'denied'}."
                            )
                        )
                        if decision.get("next_step"):
                            denial_reason += f" Next: {decision['next_step']}"
                    else:
                        approved = bool(decision)
                        denial_reason = "The user denied this tool call."

                    if not approved:
                        await self._record_tool_result(
                            call_id,
                            {
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "success": False,
                                "error": "denied",
                                "energy_spent": 0,
                                "model_output": (
                                    f"Tool call '{tool_name}' was not approved. "
                                    f"{denial_reason}"
                                ),
                            },
                        )
                        self._tool_calls_made.append(
                            {
                                "id": call_id,
                                "name": tool_name,
                                "arguments": arguments,
                                "success": False,
                                "denied": True,
                                "energy_spent": 0,
                            }
                        )
                        await self._record_gate_refusal(tool_name, "denied")
                        continue

                # Build execution context
                exec_ctx = await self._build_exec_context(
                    call_id, approval_evidence=approval_evidence
                )

                await self._emit(
                    AgentEvent.TOOL_START,
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "iteration": self._iteration_count,
                    },
                )

                # Execute tool via registry (policy + hooks + audit)
                result = await cfg.registry.execute(tool_name, arguments, exec_ctx)
                # DB appends the tool message and sums energy into runtime_state;
                # read the authoritative running total back from it.
                applied = await self._record_tool_result(
                    call_id,
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "success": result.success,
                        "output": result.output,
                        "display_output": result.display_output,
                        "model_output": result.to_model_output(),
                        "error": result.error,
                        "error_type": (
                            result.error_type.value if result.error_type else None
                        ),
                        "energy_spent": result.energy_spent,
                        "duration_seconds": result.duration_seconds,
                    },
                )
                self._energy_spent = int(
                    applied.get(
                        "energy_spent", self._energy_spent + result.energy_spent
                    )
                )
                await self._append_tool_visual_context(result.metadata)

                await self._emit(
                    AgentEvent.TOOL_RESULT,
                    {
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "success": result.success,
                        "energy_spent": result.energy_spent,
                        "total_energy_spent": self._energy_spent,
                        "duration": result.duration_seconds,
                        "error": result.error,
                        "output": result.output,
                        "display_output": result.display_output,
                    },
                )

                self._tool_calls_made.append(
                    {
                        "id": call_id,
                        "name": tool_name,
                        "arguments": arguments,
                        "success": result.success,
                        "energy_spent": result.energy_spent,
                        "error": result.error,
                    }
                )
                self._tool_results.append(
                    {
                        "id": call_id,
                        "name": tool_name,
                        "arguments": arguments,
                        "success": result.success,
                        "output": result.output,
                    }
                )

                if (
                    result.success
                    and tool_name == "use_skill"
                    and cfg.allowed_tool_names is not None
                ):
                    output = result.output if isinstance(result.output, dict) else {}
                    newly_bound = {
                        str(name)
                        for name in output.get("bound_tools", [])
                        if cfg.registry.get_spec(str(name)) is not None
                    }
                    if newly_bound:
                        cfg.allowed_tool_names.update(newly_bound)
                        activated_name = str(output.get("name") or "").strip()
                        if (
                            activated_name
                            and activated_name not in cfg.active_skill_names
                        ):
                            cfg.active_skill_names.append(activated_name)
                        tools = await self._load_tools_for_turn()
                        await self._record_tool_surface_change()

        # Should not reach here, but safety net
        return self._make_result(
            await self._get_messages(), "completed"
        )  # pragma: no cover

    # ------------------------------------------------------------------
    # Planned loop (Gap 1: plan → execute → verify)
    # ------------------------------------------------------------------

    _DEFAULT_PLANNING_PROMPT = (
        "Before acting, think through your approach. What are the steps needed? "
        "What could go wrong? How will you verify success? Produce a brief plan."
    )
    _DEFAULT_VERIFY_PROMPT = (
        "Review what you just did. Did you achieve the goal? If something needs "
        "fixing, take action now. If everything looks good, summarize what was accomplished."
    )

    async def _planned_loop(
        self,
        tools: list[dict[str, Any]],
    ) -> AgentLoopResult:
        """
        Three-phase agentic loop: Plan → Execute → Verify.

        - Plan: LLM thinks without tools, producing a plan
        - Execute: Normal tool-use loop (_execute_loop)
        - Verify: LLM reviews results, may call tools for corrections

        Like _execute_loop, the message log lives in the DB — plan/verify
        prompts and the plan response are appended there, not to a Python list.
        """
        # Phase 1: Plan
        await self._emit(AgentEvent.PHASE_CHANGE, {"phase": "plan"})
        self._phases_completed.append("plan")

        planning_prompt = self.config.planning_prompt or self._DEFAULT_PLANNING_PROMPT
        await self._append_user_message(planning_prompt)
        messages = await self._get_messages()

        try:
            response = await self._llm_call(messages, tools=None)
        except Exception as e:
            logger.error("Plan phase LLM call failed: %s", e)
            await self._emit(AgentEvent.ERROR, {"error": str(e), "phase": "plan"})
            return self._make_result(await self._get_messages(), "error")

        plan_text = response.get("content", "") or ""
        if plan_text:
            self._last_text = plan_text
            self._visible_text_parts.append(plan_text)
            self._plan_text = plan_text
            if not self._streaming:
                await self._emit(
                    AgentEvent.TEXT_DELTA,
                    {"text": plan_text, "iteration": self._iteration_count},
                )

        # Record the plan as an assistant message (no tool calls) in the DB.
        await self._apply_llm_result(response)

        # Phase 2: Execute
        await self._emit(AgentEvent.PHASE_CHANGE, {"phase": "execute"})
        self._phases_completed.append("execute")

        exec_result = await self._execute_loop(tools)

        # If execute didn't complete normally, skip verify
        if exec_result.stopped_reason != "completed":
            return exec_result

        # Phase 3: Verify
        await self._emit(AgentEvent.PHASE_CHANGE, {"phase": "verify"})
        self._phases_completed.append("verify")

        verify_prompt = self.config.verify_prompt or self._DEFAULT_VERIFY_PROMPT
        await self._append_user_message(verify_prompt)

        # Reset continuation counter for verify phase
        self._continuations_used = 0

        return await self._execute_loop(await self._load_tools_for_turn())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _record_gate_refusal(self, tool_name: str, reason: str) -> None:
        """A refused tool leaves a trace. Advisory — never breaks the turn."""
        cfg = self.config
        try:
            from services.skill_runtime import record_gate_refusal

            await record_gate_refusal(
                getattr(cfg.registry, "pool", None),
                session_id=getattr(cfg, "session_id", None),
                tool_name=tool_name,
                reason=reason,
                active_skills=list(getattr(cfg, "active_skill_names", []) or []),
            )
        except Exception:
            logger.debug("Gate-refusal telemetry failed (non-fatal)", exc_info=True)

    async def _record_tool_surface_change(self) -> None:
        """Audit a successful use_skill expansion without retaining user text."""
        cfg = self.config
        if cfg.allowed_tool_names is None:
            return
        try:
            from services.tool_surface_audit import record_tool_surface_snapshot

            await record_tool_surface_snapshot(
                cfg.pool,
                registry=cfg.registry,
                allowed_tool_names=set(cfg.allowed_tool_names),
                active_skill_names=list(cfg.active_skill_names),
                considered=[],
                available_skill_count=cfg.available_skill_count,
                session_id=cfg.session_id,
                surface=cfg.surface,
                tool_context=cfg.tool_context,
                input_text_hash=cfg.tool_surface_input_hash,
                decision_kind="skill_activation",
            )
        except Exception:
            logger.debug("Tool-surface telemetry failed (non-fatal)", exc_info=True)

    async def _build_exec_context(
        self,
        call_id: str,
        *,
        approval_evidence: dict[str, Any] | None = None,
    ) -> ToolExecutionContext:
        """Build ToolExecutionContext with config overrides and remaining energy."""
        cfg = self.config
        remaining_energy: int | None = None
        if cfg.energy_budget is not None:
            remaining_energy = max(0, cfg.energy_budget - self._energy_spent)

        async def emit_tool_event(event_name: str, data: dict[str, Any]) -> None:
            try:
                event = AgentEvent(event_name)
            except ValueError:
                logger.warning("Tool emitted unknown agent event %s", event_name)
                return
            await self._emit(event, data)

        ctx = ToolExecutionContext(
            tool_context=cfg.tool_context,
            call_id=call_id,
            session_id=cfg.session_id,
            is_group=cfg.is_group,
            is_operator=cfg.is_operator,
            surface=cfg.surface,
            event_callback=emit_tool_event,
            heartbeat_id=cfg.heartbeat_id,
            energy_available=remaining_energy,
            allow_network=True,
            allow_shell=False,
            allow_file_read=True,
            allow_file_write=False,
            action_approved=bool(
                approval_evidence and approval_evidence.get("request_id")
            ),
            approval_request_id=(
                str(approval_evidence.get("request_id"))
                if approval_evidence and approval_evidence.get("request_id")
                else None
            ),
            approval_channel=(
                str(approval_evidence.get("channel"))
                if approval_evidence and approval_evidence.get("channel")
                else None
            ),
        )

        # Apply overrides from ToolsConfig
        try:
            tc = await cfg.registry.get_config()
            overrides = tc.get_context_overrides(cfg.tool_context)
            ctx.allow_shell = overrides.allow_shell
            ctx.allow_file_write = overrides.allow_file_write
            if tc.workspace_path:
                ctx.workspace_path = tc.workspace_path
        except Exception as e:
            logger.debug("Failed to apply config overrides: %s", e)

        # Apply runtime overrides from AgentLoopConfig (additive only — can
        # grant permissions but never revoke what the DB config granted)
        if cfg.context_overrides is not None:
            rt = cfg.context_overrides
            if rt.allow_shell:
                ctx.allow_shell = True
            if rt.allow_file_write:
                ctx.allow_file_write = True
            if rt.allow_all:
                ctx.allow_shell = True
                ctx.allow_file_write = True

        return ctx

    def _session_uuid_or_none(self) -> str | None:
        if not self.config.session_id:
            return None
        try:
            return str(uuid.UUID(str(self.config.session_id)))
        except Exception:
            return None

    async def _start_turn(
        self, user_message: str, messages: list[dict[str, Any]]
    ) -> None:
        """Open a DB-owned turn. Fails loud — the turn state is authoritative,
        so a failed start must surface instead of degrading to a Python loop."""
        async with self.config.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT start_agent_turn($1::text, $2::text, $3::uuid, $4::jsonb)",
                self.config.tool_context.value,
                user_message,
                self._session_uuid_or_none(),
                json.dumps(
                    {
                        "messages": messages,
                        "energy_budget": self.config.energy_budget,
                        "max_iterations": self.config.max_iterations,
                        "max_continuations": self.config.max_continuations,
                        "heartbeat_id": self.config.heartbeat_id,
                    }
                ),
            )
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not (isinstance(payload, dict) and payload.get("turn_id")):
            raise RuntimeError(f"start_agent_turn returned no turn_id: {payload!r}")
        self._turn_id = str(payload["turn_id"])

    async def _next_agent_step(self) -> dict[str, Any]:
        """Ask the DB what to do next; the DB owns the loop (stop) decision."""
        async with self.config.pool.acquire() as conn:
            raw = await conn.fetchval("SELECT next_agent_step($1::uuid)", self._turn_id)
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict):
            raise RuntimeError(f"next_agent_step returned non-dict: {payload!r}")
        return payload

    async def _get_messages(self) -> list[dict[str, Any]]:
        """Read the DB-authoritative message log for this turn."""
        async with self.config.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT messages FROM agent_turns WHERE id = $1::uuid", self._turn_id
            )
        if raw is None:
            return []
        return json.loads(raw) if isinstance(raw, str) else raw

    async def _append_user_message(self, content: str) -> None:
        """Append a user message (continuation / plan / verify) to the DB log."""
        async with self.config.pool.acquire() as conn:
            await conn.fetchval(
                "SELECT append_agent_message($1::uuid, $2::text, $3::text)",
                self._turn_id,
                "user",
                content,
            )

    async def _apply_llm_result(self, response: dict[str, Any]) -> None:
        """Record the assistant message in the DB. tool_calls are stored in
        OpenAI format so the log is directly replayable to the model."""
        tool_calls = response.get("tool_calls") or []
        openai_tool_calls = [_to_openai_tool_call(tc) for tc in tool_calls]
        async with self.config.pool.acquire() as conn:
            await conn.fetchval(
                "SELECT apply_agent_llm_result($1::uuid, $2::jsonb)",
                self._turn_id,
                json.dumps(
                    {
                        "content": response.get("content", "") or "",
                        "tool_calls": openai_tool_calls,
                    }
                ),
            )

    async def _record_tool_result(
        self, call_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Append a tool result to the DB log; return the DB's running state
        (incl. the authoritative total energy_spent)."""
        async with self.config.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT apply_agent_tool_result($1::uuid, $2::text, $3::jsonb)",
                self._turn_id,
                call_id,
                json.dumps(payload),
            )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else {}

    async def _append_tool_visual_context(self, metadata: dict[str, Any]) -> None:
        """Put trusted tool images into the DB-owned model conversation."""
        attachments = metadata.get("model_visual_attachments")
        if not isinstance(attachments, list):
            return
        for attachment in attachments[:4]:
            if not isinstance(attachment, dict):
                continue
            data_url = str(attachment.get("data_url") or "")
            mime_type = str(attachment.get("mime_type") or "").lower()
            if not data_url.startswith("data:image/") or not mime_type.startswith(
                "image/"
            ):
                continue
            async with self.config.pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT append_agent_visual_message($1::uuid, $2, $3)",
                    self._turn_id,
                    str(attachment.get("name") or "tool image"),
                    data_url,
                )

    async def _enforce_action_claims(self, result: AgentLoopResult) -> None:
        """Correct action claims unsupported by a call or durable prior receipt.

        The reply has already streamed, so enforcement is detect + correct +
        record, never block. Advisory: any failure leaves the reply untouched.
        """
        text = result.visible_text or result.text or ""
        if not text.strip() or not self._turn_id:
            return
        try:
            async with self.config.pool.acquire() as conn:
                enabled = await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('guardrails.action_claims.enabled'), TRUE)"
                )
                if not enabled:
                    return
                raw = await conn.fetchval(
                    "SELECT detect_unsupported_action_claims($1::uuid, $2::text)",
                    self._turn_id,
                    text,
                )
            report = json.loads(raw) if isinstance(raw, str) else (raw or {})
            findings = report.get("flagged") or []
            if not findings:
                return

            verifier_used = False
            verified = await self._verify_claims_with_llm(findings, text)
            if verified is not None:
                findings = verified
                verifier_used = True
            if not findings:
                return

            summary = "; ".join(
                f'{f.get("kind", "action")}: "{(f.get("sentence") or "")[:160]}"'
                for f in findings[:5]
            )
            correction = (
                "\n\n[Correction] I described actions I could not verify — "
                f"{summary} — no matching successful tool call or durable "
                "prior-action receipt. Treat those "
                "statements as unverified."
            )
            result.text += correction
            result.visible_text = (result.visible_text or text) + correction
            self._last_text = result.text
            await self._emit(
                AgentEvent.TEXT_DELTA, {"text": correction, "correction": True}
            )
            await self._emit(
                AgentEvent.CLAIM_FLAGGED,
                {
                    "findings": findings,
                    "verifier_used": verifier_used,
                },
            )
        except Exception:
            logger.warning(
                "action-claim guardrail failed for turn %s; reply left unmodified",
                self._turn_id,
                exc_info=True,
            )

    async def _verify_claims_with_llm(
        self, findings: list[dict[str, Any]], text: str
    ) -> list[dict[str, Any]] | None:
        """Config-gated LLM pass over heuristic findings. Returns the confirmed
        (possibly extended) findings, or None when the verifier is disabled.
        On LLM failure the heuristic findings stand (fail-open)."""
        async with self.config.pool.acquire() as conn:
            verifier_on = await conn.fetchval(
                "SELECT COALESCE(get_config_bool('guardrails.action_claims.llm_verifier_enabled'), FALSE)"
            )
            if not verifier_on:
                return None
            try:
                from core.llm_config import load_llm_config
                from core.llm_json import chat_json

                llm_config = await load_llm_config(
                    conn, "llm.guardrails", fallback_key="llm.subconscious"
                )
                system = await conn.fetchval(
                    "SELECT content FROM prompt_modules WHERE key = 'action_claim_verify'"
                )
                prior_raw = await conn.fetchval(
                    """
                    SELECT COALESCE(jsonb_agg(receipt), '[]'::jsonb)
                    FROM agent_turns turn_state
                    CROSS JOIN LATERAL jsonb_array_elements(
                        COALESCE(turn_state.messages, '[]'::jsonb)
                    ) message
                    CROSS JOIN LATERAL jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(
                                message#>'{metadata,action_receipts}'
                            ) = 'array'
                                THEN message#>'{metadata,action_receipts}'
                            ELSE '[]'::jsonb
                        END
                    ) receipt
                    WHERE turn_state.id = $1::uuid
                      AND message->>'role' = 'assistant'
                      AND COALESCE((receipt->>'success')::boolean, FALSE)
                    """,
                    self._turn_id,
                )
            except Exception:
                logger.warning(
                    "action-claim verifier setup failed; keeping heuristic findings",
                    exc_info=True,
                )
                return findings
        prior_receipts = (
            json.loads(prior_raw) if isinstance(prior_raw, str) else prior_raw
        )
        if not isinstance(prior_receipts, list):
            prior_receipts = []
        payload = {
            "final_text": text[:12000],
            "flagged": findings,
            "successful_tool_calls": [
                {"name": c.get("name"), "arguments": c.get("arguments")}
                for c in self._tool_calls_made
                if c.get("success")
            ],
            "prior_action_receipts": prior_receipts,
        }
        try:
            doc, _raw = await chat_json(
                llm_config=llm_config,
                messages=[
                    {"role": "system", "content": (system or "").strip()},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            )
        except Exception:
            logger.warning(
                "action-claim LLM verifier failed; keeping heuristic findings",
                exc_info=True,
            )
            return findings
        confirmed = doc.get("confirmed") if isinstance(doc, dict) else None
        if isinstance(confirmed, list):
            kept = [
                findings[i]
                for i in confirmed
                if isinstance(i, int) and 0 <= i < len(findings)
            ]
        else:
            kept = list(findings)
        for extra in (doc.get("additional") or []) if isinstance(doc, dict) else []:
            if isinstance(extra, dict) and extra.get("sentence"):
                kept.append(
                    {
                        "kind": extra.get("kind", "action"),
                        "sentence": str(extra["sentence"])[:300],
                        "source": "llm_verifier",
                    }
                )
        return kept

    async def _finish_turn(self, result: AgentLoopResult) -> None:
        """Mark the turn complete. Best-effort with a loud warning: the reply is
        already in hand, so a finalize failure must not drop it."""
        try:
            async with self.config.pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT finish_agent_turn($1::uuid, $2::jsonb)",
                    self._turn_id,
                    json.dumps(
                        {
                            "status": "completed",
                            "stopped_reason": result.stopped_reason,
                            "text": result.text,
                            "visible_text": result.visible_text,
                            "iterations": result.iterations,
                            "energy_spent": result.energy_spent,
                            "timed_out": result.timed_out,
                        }
                    ),
                )
        except Exception:
            logger.warning(
                "finish_agent_turn failed for turn %s", self._turn_id, exc_info=True
            )

    async def _emit(
        self, event: AgentEvent, data: dict[str, Any] | None = None
    ) -> None:
        """Emit an event via the configured callback."""
        event_data = dict(data or {})
        if self._turn_id:
            event_data.setdefault("turn_id", self._turn_id)
        if self._turn_id:
            try:
                async with self.config.pool.acquire() as conn:
                    await conn.fetchval(
                        "SELECT record_agent_turn_event($1::uuid, $2::text, $3::jsonb)",
                        self._turn_id,
                        event.value,
                        json.dumps(event_data),
                    )
            except Exception:
                logger.debug("DB agent event record failed", exc_info=True)
        if self.config.on_event:
            try:
                await self.config.on_event(
                    AgentEventData(
                        event=event,
                        data=event_data,
                    )
                )
            except Exception:
                logger.debug("Event callback failed for %s", event, exc_info=True)

    def _make_result(
        self, messages: list[dict[str, Any]] | None, stopped_reason: str
    ) -> AgentLoopResult:
        """Build an AgentLoopResult from current state."""
        return AgentLoopResult(
            text=self._last_text,
            messages=messages or [],
            tool_calls_made=self._tool_calls_made,
            iterations=self._iteration_count,
            energy_spent=self._energy_spent,
            turn_id=self._turn_id,
            visible_text="\n\n".join(self._visible_text_parts),
            tool_results=self._tool_results,
            timed_out=False,
            stopped_reason=stopped_reason,
            plan_text=self._plan_text,
            phases_completed=list(self._phases_completed),
            continuations_used=self._continuations_used,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _to_openai_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """Convert simplified tool call dict to OpenAI assistant message format."""
    arguments = call.get("arguments", {})
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {
        "id": call.get("id") or str(uuid.uuid4()),
        "type": "function",
        "function": {
            "name": call.get("name", ""),
            "arguments": arguments,
        },
    }
