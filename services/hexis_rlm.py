"""Vendored RLM engine for Hexis.

Implements the Recursive Language Model loop (Algorithm 1 from paper 2512.24601v2)
adapted for Hexis. Does NOT import the upstream `rlm` package.

Provides:
- run_heartbeat_decision(): RLM loop for heartbeat decisions
- run_chat_turn(): RLM loop for chat conversations
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from core.llm import chat_completion, normalize_llm_config
from core.memory_repo import MemoryRepo
from core.tools.repl_bridge import ReplToolBridge, call_records_to_actions_taken
from services.prompt_resources import (
    compose_compact_personhood_prompt,
    load_rlm_chat_prompt,
    load_rlm_heartbeat_prompt,
)
from services.rlm_memory_env import RLMMemoryEnv, RLMWorkspace, WorkspaceBudgets
from services.rlm_repl import HexisLocalREPL, REPLResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsing utilities (vendored from docs/reference/rlm-main/rlm/utils/parsing.py)
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)
_FINAL_VAR_RE = re.compile(r"^\s*FINAL_VAR\((.*?)\)", re.MULTILINE | re.DOTALL)
_FINAL_RE = re.compile(r"^\s*FINAL\((.*)\)\s*$", re.MULTILINE | re.DOTALL)

MAX_OUTPUT_CHARS = 20_000


@dataclass(frozen=True)
class FinalAnswerResolution:
    """Parsed FINAL/FINAL_VAR result plus any internal repair diagnostic."""

    answer: str | None
    error: str | None = None


class RlmLoopFailure(RuntimeError):
    """The RLM loop got no usable LLM response at all (#111).

    Raised instead of fabricating a well-formed-looking decision string, so
    the caller's existing legacy-path fallback (services/external_calls.py)
    actually runs, and the failure can be recorded as a real defect rather
    than persisted as a heartbeat that "chose" to do nothing.
    """


def _workspace_metrics(workspace: RLMWorkspace) -> dict[str, Any]:
    """Return every retrieval/workspace counter the RLM can affect."""
    return {
        "search_count": workspace.metrics.search_count,
        "fetch_count": workspace.metrics.fetch_count,
        "fetched_chars_total": workspace.metrics.fetched_chars_total,
        "document_search_count": workspace.metrics.document_search_count,
        "document_fetch_count": workspace.metrics.document_fetch_count,
        "document_load_count": workspace.metrics.document_load_count,
        "document_chunk_search_count": workspace.metrics.document_chunk_search_count,
        "document_chunk_fetch_count": workspace.metrics.document_chunk_fetch_count,
        "document_chunk_load_count": workspace.metrics.document_chunk_load_count,
        "desk_list_count": workspace.metrics.desk_list_count,
        "desk_fetch_count": workspace.metrics.desk_fetch_count,
        "desk_pin_count": workspace.metrics.desk_pin_count,
        "summarize_events": workspace.metrics.summarize_events,
    }


def find_code_blocks(text: str) -> list[str]:
    """Find REPL code blocks wrapped in ```repl ... ```."""
    return [m.group(1).strip() for m in _CODE_BLOCK_RE.finditer(text)]


def _find_final_var_name(text: str) -> str | None:
    match = _FINAL_VAR_RE.search(text)
    if not match:
        return None
    return match.group(1).strip().strip('"').strip("'")


def _assigned_names_from_code(code: str) -> set[str]:
    """Return variable names assigned by a REPL block."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _is_final_var_error(text: str) -> bool:
    """Return true when FINAL_VAR returned its own missing-variable diagnostic."""
    stripped = text.strip()
    return (
        stripped.startswith("Error: Variable ")
        and (
            "BEFORE calling FINAL_VAR" in stripped
            or "No variables have been created yet" in stripped
        )
    )


def resolve_final_answer(
    text: str,
    repl: HexisLocalREPL | None = None,
    *,
    allow_final_var: bool = True,
    allowed_final_var_names: set[str] | None = None,
) -> FinalAnswerResolution:
    """Find FINAL(...) or FINAL_VAR(...), separating answers from repairable errors."""
    # Check FINAL_VAR first
    match = _FINAL_VAR_RE.search(text)
    if match:
        variable_name = _find_final_var_name(text) or ""
        if not allow_final_var:
            return FinalAnswerResolution(
                answer=None,
                error=(
                    "FINAL_VAR is not valid for chat responses. Use FINAL(...) "
                    "directly with the user-visible reply."
                ),
            )
        if (
            allowed_final_var_names is not None
            and variable_name not in allowed_final_var_names
        ):
            return FinalAnswerResolution(
                answer=None,
                error=(
                    f"FINAL_VAR({variable_name}) was not assigned in this "
                    "iteration. Use FINAL(...) directly, or assign that exact "
                    "variable in the same ```repl``` block before FINAL_VAR."
                ),
            )
        if repl is not None:
            result = repl.execute_code(f"print(FINAL_VAR({variable_name!r}))")
            answer = result.stdout.strip()
            stderr = result.stderr.strip()
            if answer and not _is_final_var_error(answer) and not stderr:
                return FinalAnswerResolution(answer=answer)
            diagnostic = stderr or answer or f"FINAL_VAR({variable_name}) did not produce output."
            return FinalAnswerResolution(answer=None, error=diagnostic)
        return FinalAnswerResolution(
            answer=None,
            error="FINAL_VAR requires an initialized REPL.",
        )

    # Check FINAL
    match = _FINAL_RE.search(text)
    if match:
        return FinalAnswerResolution(answer=match.group(1).strip())

    return FinalAnswerResolution(answer=None)


def find_final_answer(text: str, repl: HexisLocalREPL | None = None) -> str | None:
    """Find FINAL(...) or a valid FINAL_VAR(...) in response text."""
    return resolve_final_answer(text, repl).answer


def _format_final_error(error: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Your attempted final answer could not be used:\n"
            f"{error}\n\n"
            "Repair this internally. Do not expose that diagnostic to the user. "
            "Either assign the variable first in a ```repl``` block before using "
            "FINAL_VAR, or use FINAL(...) directly with the user-visible answer."
        ),
    }


def format_execution_result(result: REPLResult) -> str:
    """Format a REPLResult for inclusion in message history."""
    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(result.stderr)
    if result.local_vars:
        parts.append(f"REPL variables: {list(result.local_vars.keys())}")
    return "\n\n".join(parts) if parts else "No output"


def format_iteration(response: str, code_blocks: list[str], results: list[REPLResult]) -> list[dict[str, str]]:
    """Format an RLM iteration for the message history."""
    messages = [{"role": "assistant", "content": response}]

    for code, result in zip(code_blocks, results):
        formatted = format_execution_result(result)
        if len(formatted) > MAX_OUTPUT_CHARS:
            formatted = (
                formatted[:MAX_OUTPUT_CHARS]
                + f"... + [{len(formatted) - MAX_OUTPUT_CHARS} chars truncated]"
            )
        messages.append({
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{formatted}",
        })

    return messages


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------

async def _llm_completion(
    messages: list[dict[str, str]],
    llm_config: dict[str, Any],
    max_tokens: int = 4096,
) -> str:
    """Call Hexis LLM and return the text response."""
    result = await chat_completion(
        provider=llm_config["provider"],
        model=llm_config["model"],
        endpoint=llm_config.get("endpoint"),
        api_key=llm_config.get("api_key"),
        messages=messages,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    import asyncio
    from core.usage import record_llm_usage
    asyncio.ensure_future(record_llm_usage(
        provider=llm_config["provider"],
        model=llm_config["model"],
        raw_response=result.get("raw"),
        source="rlm",
    ))
    return result.get("content", "")


def _make_sync_llm_query(
    llm_config: dict[str, Any],
    loop: asyncio.AbstractEventLoop,
) -> Callable[[str], str]:
    """Create a synchronous llm_query function for use in the REPL."""

    def llm_query(prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": prompt},
        ]
        try:
            future = asyncio.run_coroutine_threadsafe(
                _llm_completion(messages, llm_config, max_tokens=2048),
                loop,
            )
            return future.result(timeout=120)
        except Exception as e:
            return f"Error: LLM query failed - {e}"

    return llm_query


# ---------------------------------------------------------------------------
# User prompt (iteration suffix)
# ---------------------------------------------------------------------------

_USER_PROMPT_FIRST = (
    "You have not interacted with the REPL environment or seen your context yet. "
    "Your next action should be to examine the context variable and figure out how "
    "to approach the task. Don't provide a final answer yet.\n\n"
    "Think step-by-step on what to do using the REPL environment. "
    "Continue writing ```repl``` code blocks and determine your answer. Your next action:"
)

_CHAT_USER_PROMPT_FIRST = (
    "You are answering a chat message. If this is ordinary conversation and you "
    "do not need memory, tools, or document inspection, answer now with FINAL(...). "
    "If the user's message depends on prior conversations, preferences, documents, "
    "tool use, or exact context, inspect the `context` variable and use the REPL "
    "environment first. Your next action:"
)

_USER_PROMPT_CONTINUE = (
    "The history above shows your previous interactions with the REPL environment. "
    "Continue using the REPL environment, which has the `context` variable and memory syscalls, "
    "and determine your answer. Your next action:"
)


def _build_user_prompt(iteration: int, *, allow_direct_chat_final: bool = False) -> dict[str, str]:
    if iteration == 0:
        if allow_direct_chat_final:
            return {"role": "user", "content": _CHAT_USER_PROMPT_FIRST}
        return {"role": "user", "content": _USER_PROMPT_FIRST}
    return {"role": "user", "content": _USER_PROMPT_CONTINUE}


# ---------------------------------------------------------------------------
# Core RLM loop (synchronous, runs in thread pool)
# ---------------------------------------------------------------------------

def _run_loop(
    repl: HexisLocalREPL,
    llm_config: dict[str, Any],
    loop: asyncio.AbstractEventLoop,
    system_prompt: str,
    max_iterations: int,
    allow_final_var: bool = True,
    allow_direct_chat_final: bool = False,
) -> dict[str, Any]:
    """
    Synchronous RLM iteration loop (Algorithm 1).

    Runs in a thread pool executor to avoid blocking the async event loop.
    LLM calls bridge back to the async loop via run_coroutine_threadsafe.
    """
    message_history: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    final_answer: str | None = None
    iteration_count = 0
    # The real cause of an early exit (#111): distinct from genuinely
    # exhausting max_iterations with real (if unresolved) responses, so the
    # exhaustion log below reports what actually happened instead of always
    # blaming the configured limit, and so a total failure can raise instead
    # of silently fabricating a decision.
    loop_error: str | None = None

    for i in range(max_iterations):
        iteration_count = i + 1

        # Build current prompt
        current_prompt = message_history + [
            _build_user_prompt(i, allow_direct_chat_final=allow_direct_chat_final)
        ]

        # Call LLM
        future = asyncio.run_coroutine_threadsafe(
            _llm_completion(current_prompt, llm_config, max_tokens=4096),
            loop,
        )
        try:
            response = future.result(timeout=120)
        except Exception as e:
            logger.error("LLM call failed at iteration %d: %s", i, e)
            loop_error = str(e)
            break

        if not response:
            logger.warning("Empty LLM response at iteration %d", i)
            break

        # Check for final answer BEFORE executing code. A bad FINAL_VAR is not
        # final; it becomes a repair message inside the RLM loop.
        final_resolution = resolve_final_answer(
            response,
            repl,
            allow_final_var=allow_final_var,
        )
        if final_resolution.answer is not None:
            final_answer = final_resolution.answer
            logger.info("RLM loop completed at iteration %d with FINAL", i + 1)
            break

        # Extract and execute code blocks
        code_blocks = find_code_blocks(response)
        assigned_names: set[str] = set()
        results: list[REPLResult] = []

        for code in code_blocks:
            assigned_names.update(_assigned_names_from_code(code))
            result = repl.execute_code(code)
            results.append(result)

        # Re-check FINAL_VAR after executing code blocks. This supports the
        # common repair case where the model emits a variable assignment and
        # FINAL_VAR in the same assistant message.
        if final_resolution.error and code_blocks:
            final_resolution = resolve_final_answer(
                response,
                repl,
                allow_final_var=True,
                allowed_final_var_names=assigned_names if not allow_final_var else None,
            )
            if final_resolution.answer is not None:
                final_answer = final_resolution.answer

        if final_answer is not None:
            logger.info("RLM loop completed at iteration %d with FINAL_VAR", i + 1)
            break

        # Format iteration and append to history
        new_messages = format_iteration(response, code_blocks, results)
        message_history.extend(new_messages)
        if final_resolution.error:
            message_history.append(_format_final_error(final_resolution.error))

    # If we ran out of iterations without a FINAL, use the last response
    if final_answer is None:
        if loop_error:
            # An LLM call itself failed (#111) -- this is not "used up all
            # its iterations thinking," it's a real infrastructure failure
            # after iteration_count response(s). Report the true cause.
            logger.warning(
                "RLM loop broke early after %d iteration(s) (%s); attempting one repair call",
                iteration_count,
                loop_error,
            )
        else:
            logger.warning(
                "RLM loop exhausted %d iterations without FINAL, using last response",
                max_iterations,
            )
        # Make one more call asking for a final answer
        message_history.append({
            "role": "user",
            "content": (
                "You have used all your iterations. You MUST provide your final answer NOW "
                "using FINAL(...). Based on everything you've learned, provide your answer."
            ),
        })
        future = asyncio.run_coroutine_threadsafe(
            _llm_completion(message_history, llm_config, max_tokens=4096),
            loop,
        )
        try:
            response = future.result(timeout=120)
            final_resolution = resolve_final_answer(
                response,
                repl,
                allow_final_var=allow_final_var,
            )
            final_answer = final_resolution.answer
            if final_answer is None:
                if final_resolution.error:
                    logger.warning("RLM final-answer repair failed: %s", final_resolution.error)
                    final_answer = (
                        "I hit an internal scratchpad formatting issue while answering. "
                        "Please send that again."
                    )
                else:
                    final_answer = response
        except Exception as repair_exc:
            # Both the main loop AND the repair call failed to get any real
            # LLM response -- a total failure, not a decision. Raising here
            # (rather than fabricating a well-formed-looking decision, #111)
            # lets the caller's existing legacy-path fallback actually run,
            # and the real error surface instead of a fictitious "timed out."
            reason = loop_error or str(repair_exc)
            logger.error("RLM loop got no usable LLM response at all: %s", reason)
            raise RlmLoopFailure(reason) from repair_exc

    return {
        "final_answer": final_answer,
        "iterations": iteration_count,
        "message_count": len(message_history),
    }


# ---------------------------------------------------------------------------
# Heartbeat decision entry point
# ---------------------------------------------------------------------------

async def run_heartbeat_decision(
    *,
    heartbeat_id: str,
    turn_snapshot: dict[str, Any],
    llm_config: dict[str, Any],
    dsn: str,
    tool_registry: Any,
    loop: asyncio.AbstractEventLoop,
    max_iterations: int = 10,
    timeout_seconds: int = 300,
    workspace_budgets: WorkspaceBudgets | None = None,
) -> dict[str, Any]:
    """
    Run the RLM loop for a heartbeat decision.

    Returns a dict with keys: kind, decision, heartbeat_id, rlm_repl_actions,
    raw_response, metrics.
    """
    time_start = time.perf_counter()

    # Normalize LLM config
    llm_cfg = normalize_llm_config(llm_config)

    # Create memory repo (psycopg2 sync)
    repo = MemoryRepo(dsn)

    # Create workspace
    budgets = workspace_budgets or WorkspaceBudgets()
    workspace = RLMWorkspace(
        task="heartbeat_decision",
        turn_snapshot=turn_snapshot,
        budgets=budgets,
    )

    # Create sync LLM query function for sub-calls
    llm_query_fn = _make_sync_llm_query(llm_cfg, loop)

    # Create memory env
    memory_env = RLMMemoryEnv(repo, workspace, llm_query_fn=llm_query_fn)

    # Create tool bridge
    initial_energy = turn_snapshot.get("energy", {}).get("current", 20.0)
    bridge = ReplToolBridge(
        registry=tool_registry,
        loop=loop,
        heartbeat_id=heartbeat_id,
        initial_energy=initial_energy,
    )

    # Create REPL
    repl = HexisLocalREPL()
    repl.setup(
        context_payload=turn_snapshot,
        memory_env=memory_env,
        tool_bridge=bridge,
        llm_query_fn=llm_query_fn,
    )

    # Build system prompt
    system_prompt = load_rlm_heartbeat_prompt()
    personhood_addendum = compose_compact_personhood_prompt("heartbeat")
    if personhood_addendum:
        system_prompt = system_prompt + "\n\n---\n\n" + personhood_addendum

    # Run RLM loop in thread pool
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                _run_loop,
                repl,
                llm_cfg,
                loop,
                system_prompt,
                max_iterations,
                True,
                False,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error("RLM heartbeat timed out after %ds", timeout_seconds)
        result = {
            "final_answer": '{"reasoning": "RLM heartbeat timed out", "actions": [{"action": "rest", "params": {}}], "goal_changes": []}',
            "iterations": 0,
            "message_count": 0,
        }
    finally:
        repl.cleanup()
        repo.close()

    # Parse decision from final answer
    raw_answer = result["final_answer"]
    try:
        decision = json.loads(raw_answer)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse RLM final answer as JSON, wrapping as reasoning")
        decision = {
            "reasoning": str(raw_answer)[:2000],
            "actions": [],
            "goal_changes": [],
        }

    # Collect REPL tool call records
    repl_actions = call_records_to_actions_taken(bridge.get_call_records())

    duration = time.perf_counter() - time_start

    logger.info(
        "rlm_heartbeat_complete heartbeat_id=%s iterations=%d "
        "search_count=%d fetch_count=%d tool_calls=%d "
        "tool_energy=%d duration=%.1fs",
        heartbeat_id,
        result["iterations"],
        workspace.metrics.search_count,
        workspace.metrics.fetch_count,
        len(bridge.get_call_records()),
        bridge.get_total_energy_spent(),
        duration,
    )

    return {
        "kind": "heartbeat_decision",
        "decision": decision,
        "heartbeat_id": heartbeat_id,
        "rlm_repl_actions": repl_actions,
        "raw_response": raw_answer[:5000],
        "metrics": {
            "iterations": result["iterations"],
            "message_count": result["message_count"],
            **_workspace_metrics(workspace),
            "tool_calls": len(bridge.get_call_records()),
            "tool_energy_spent": bridge.get_total_energy_spent(),
            "total_duration_seconds": round(duration, 2),
        },
    }


# ---------------------------------------------------------------------------
# Chat entry point
# ---------------------------------------------------------------------------

# Session management for persistent multi-turn chat
_chat_sessions: dict[str, HexisLocalREPL] = {}
_session_last_used: dict[str, float] = {}
_session_lock = asyncio.Lock()
_SESSION_TTL = 300  # 5 minutes


async def _cleanup_stale_sessions() -> None:
    """Remove chat sessions idle longer than TTL."""
    now = time.time()
    async with _session_lock:
        stale = [
            sid for sid, ts in _session_last_used.items()
            if now - ts > _SESSION_TTL
        ]
        for sid in stale:
            session = _chat_sessions.pop(sid, None)
            if session:
                session.cleanup()
            _session_last_used.pop(sid, None)


async def run_chat_turn(
    *,
    user_message: str,
    history: list[dict[str, str]] | None = None,
    llm_config: dict[str, Any],
    dsn: str,
    session_id: str | None = None,
    max_iterations: int = 15,
    timeout_seconds: int = 120,
    workspace_budgets: WorkspaceBudgets | None = None,
) -> dict[str, Any]:
    """
    Run the RLM loop for a chat turn.

    Returns a dict with keys: response, history, metrics.
    """
    await _cleanup_stale_sessions()

    time_start = time.perf_counter()
    llm_cfg = normalize_llm_config(llm_config)
    loop = asyncio.get_running_loop()

    # Create memory repo
    repo = MemoryRepo(dsn)
    budgets = workspace_budgets or WorkspaceBudgets()
    workspace = RLMWorkspace(task="chat", budgets=budgets)
    llm_query_fn = _make_sync_llm_query(llm_cfg, loop)
    memory_env = RLMMemoryEnv(repo, workspace, llm_query_fn=llm_query_fn)

    # Get or create REPL session
    repl: HexisLocalREPL
    async with _session_lock:
        if session_id and session_id in _chat_sessions:
            repl = _chat_sessions[session_id]
            repl.bind_memory_env(memory_env)
            repl.bind_llm_query(llm_query_fn)
            # Add new user message as context
            repl.load_context(user_message, index=repl._context_count)
        else:
            repl = HexisLocalREPL()
            repl.setup(
                context_payload=user_message,
                memory_env=memory_env,
                llm_query_fn=llm_query_fn,
            )
            if session_id:
                _chat_sessions[session_id] = repl

        if session_id:
            _session_last_used[session_id] = time.time()

    # Build system prompt
    system_prompt = load_rlm_chat_prompt()
    personhood_addendum = compose_compact_personhood_prompt("conversation")
    if personhood_addendum:
        system_prompt = system_prompt + "\n\n---\n\n" + personhood_addendum

    # Run RLM loop
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                _run_loop,
                repl,
                llm_cfg,
                loop,
                system_prompt,
                max_iterations,
                False,
                True,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error("RLM chat timed out after %ds", timeout_seconds)
        result = {
            "final_answer": "I apologize, but I need more time to think about that. Could you try again?",
            "iterations": 0,
            "message_count": 0,
        }
    except RlmLoopFailure as exc:
        # #111: a total LLM failure -- unlike the heartbeat path (which lets
        # this propagate so the legacy-path fallback can actually run), a
        # chat turn has no fallback provider to hand off to and must still
        # answer the user, honestly, rather than crash the request.
        logger.error("RLM chat got no usable LLM response at all: %s", exc)
        result = {
            "final_answer": "I'm having trouble reaching my model provider right now. Please try again in a moment.",
            "iterations": 0,
            "message_count": 0,
        }
    finally:
        # Don't cleanup persistent sessions; do cleanup repo
        if not session_id:
            repl.cleanup()
        repo.close()

    duration = time.perf_counter() - time_start

    logger.info(
        "rlm_chat_complete session=%s iterations=%d "
        "search_count=%d fetch_count=%d duration=%.1fs",
        session_id or "ephemeral",
        result["iterations"],
        workspace.metrics.search_count,
        workspace.metrics.fetch_count,
        duration,
    )

    return {
        "response": result["final_answer"],
        "metrics": {
            "iterations": result["iterations"],
            "message_count": result["message_count"],
            **_workspace_metrics(workspace),
            "total_duration_seconds": round(duration, 2),
        },
    }
