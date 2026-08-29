"""Hexis CLI chat — streaming conversation via AgentLoop.

A plain, line-based streaming REPL: your terminal owns the mouse/scrollback/copy,
Ctrl+C interrupts a reply. Features:
  - Token-by-token streaming; leaked <think>/tool-call scaffolding is stripped
    from the visible output (not just from what's persisted)
  - Tool call visibility as inline dim text
  - Slash commands (type /help for the full list)
  - --verbose shows hydrated context and tool I/O
  - --debug also dumps the system prompt and LLM request
  - --greet seeds a first "wake up" turn (used right after init)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from typing import Any

from dotenv import load_dotenv

from apps.cli_theme import console, err_console

logger = logging.getLogger(__name__)

# Single source of truth for slash commands (drives /help + the greeting).
COMMANDS: list[tuple[str, str]] = [
    ("/help", "show this list"),
    ("/recall <q>", "search long-term memory"),
    ("/status", "agent status — energy, mood, consent"),
    ("/tools", "list available tools"),
    ("/history", "show this session's turns"),
    ("/clear", "reset local context (keeps long-term memory)"),
    ("/verbose", "toggle context + tool I/O"),
    ("/debug", "toggle full debug"),
    ("/prompt", "print the system prompt"),
    ("/quit", "exit (or Ctrl+C)"),
]


def _print_commands() -> None:
    console.print("[muted]Commands:[/muted]")
    for name, desc in COMMANDS:
        console.print(f"  [teal]{name:<12}[/teal] [muted]{desc}[/muted]")
    console.print()


async def _approve_tool(tool_name: str, arguments: dict[str, Any]) -> bool:
    """Human [y/N] gate for side-effecting tools (email, DMs, shell, …).

    A person being at the keyboard is NOT approval of a specific irreversible
    act (Bar #2). Denies on EOF/Ctrl+C — the safe default.
    """
    console.print()
    console.print(f"[warn]⚠ The agent wants to run:[/warn] [bold]{tool_name}[/bold]")
    args_preview = _fmt_json(arguments, 600)
    if args_preview and args_preview not in ("{}", "null"):
        console.print(f"[muted]{args_preview}[/muted]")
    try:
        console.print("[accent]Allow this? [y/N][/accent] ", end="")
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[muted]Denied.[/muted]")
        return False
    approved = answer in ("y", "yes")
    console.print("[ok]Allowed.[/ok]" if approved else "[muted]Denied.[/muted]")
    return approved


async def _answer_cli_question(pool: Any, payload: dict[str, Any]) -> None:
    """Render ask_user with the native arrow-key picker and persist the answer."""
    from apps.cli_prompts import select_index, text
    from services.agent_questions import answer_agent_question

    question_id = str(payload.get("id") or "")
    prompt = str(payload.get("prompt") or "I need your input.").strip()
    choices = [
        str(item).strip()
        for item in payload.get("choices", [])
        if str(item).strip()
    ][:4]
    allow_free_text = payload.get("allow_free_text") is not False
    if not question_id:
        err_console.print("[fail]The clarification question has no durable id.[/fail]")
        return

    while True:
        choice_index: int | None = None
        free_text: str | None = None
        if choices:
            options = list(choices)
            if allow_free_text:
                options.append("Other (type your answer)")
            selected = await select_index(prompt, options)
            if selected <= len(choices):
                choice_index = selected
            else:
                free_text = (await text("Your answer")).strip()
                if not free_text:
                    err_console.print("[warn]An answer is required.[/warn]")
                    continue
        else:
            free_text = (await text(prompt)).strip()
            if not free_text:
                err_console.print("[warn]An answer is required.[/warn]")
                continue

        result = await answer_agent_question(
            pool,
            question_id,
            answer=free_text,
            choice_index=choice_index,
            channel="cli",
            actor="local-user",
        )
        if result.get("ok"):
            return
        err_console.print(
            f"[warn]{result.get('message') or 'That answer was not accepted.'}[/warn]"
        )
        if result.get("status") != "pending":
            return


def _fmt_json(obj: Any, max_len: int = 400) -> str:
    """Format an object as compact JSON, truncating if needed."""
    try:
        s = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + "…"
    return s


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _as_record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _connector_setup_ui(output: Any) -> dict[str, Any] | None:
    ui = _as_record(_as_record(output).get("ui"))
    if ui.get("kind") != "connector_setup":
        return None
    return ui


def _connector_setup_requires_action(ui: dict[str, Any]) -> bool:
    status = str(ui.get("status") or "").strip().lower()
    return status not in {"connected", "complete", "verified"}


def _looks_like_json_path(value: str) -> bool:
    text = value.strip()
    return (text.startswith("/") or text.startswith("~/")) and text.lower().endswith(".json")


def _ui_has_built_in_gmail_sign_in(ui: dict[str, Any]) -> bool:
    if ui.get("hexis_oauth_client_available") is True:
        return True
    credential_step = ui.get("credential_step")
    if not isinstance(credential_step, dict):
        return False
    modes = credential_step.get("modes")
    if not isinstance(modes, list):
        return False
    return any(
        isinstance(mode, dict)
        and mode.get("id") == "hosted_oauth"
        and mode.get("available") is True
        for mode in modes
    )


def _print_connector_setup_ui(ui: dict[str, Any]) -> None:
    from rich.panel import Panel

    connector = str(ui.get("display_name") or ui.get("connector_id") or "Connector")
    status = str(ui.get("status") or "setup")
    caps = _string_list(ui.get("capabilities"))
    memory_policy = str(ui.get("memory_policy") or "")
    memory_config_key = str(ui.get("memory_config_key") or "")
    heartbeat_digest_enabled = bool(ui.get("heartbeat_digest_enabled"))
    heartbeat_digest_config_key = str(ui.get("heartbeat_digest_config_key") or "")
    docs_url = str(ui.get("docs_url") or "")
    next_step = str(ui.get("next_step") or "")
    authorization_url = str(ui.get("authorization_url") or "")
    attempt_id = str(ui.get("attempt_id") or "")
    client_secret_saved = bool(ui.get("client_secret_saved"))
    built_in_sign_in_available = _ui_has_built_in_gmail_sign_in(ui)
    credential_step_label = str(ui.get("credential_step_label") or "").strip()
    setup_steps = _string_list(ui.get("setup_steps"))

    lines = [
        f"[bold]{connector} setup[/bold]",
        f"[muted]Status:[/muted] {status.replace('_', ' ')}",
    ]
    if caps:
        lines.append(f"[muted]Capabilities:[/muted] {', '.join(caps)}")
    if memory_policy:
        lines.append(f"[muted]Memory policy:[/muted] {memory_policy}")
    if memory_config_key:
        lines.append(f"[muted]Memory config:[/muted] {memory_config_key}")
    if heartbeat_digest_config_key:
        heartbeat_state = "enabled" if heartbeat_digest_enabled else "off until you authorize it"
        lines.append(f"[muted]Heartbeat email checks:[/muted] {heartbeat_state}")
    if credential_step_label:
        lines.append(f"[muted]Google setup:[/muted] {credential_step_label}")
    if not client_secret_saved and not built_in_sign_in_available and status in {"needs_client_secret", "setup"}:
        lines.extend(["", "Gmail sign-in is not enabled for this local Hexis build yet."])
        if setup_steps:
            lines.append("Walkthrough:")
            for index, step in enumerate(setup_steps, start=1):
                lines.append(f"{index}. {step}")
        else:
            lines.extend([
                "1. Open Google Cloud Console.",
                "2. Create or choose a project for Hexis.",
                "3. Enable the Gmail API.",
                "4. Configure the consent screen, adding your Gmail address as a test user if Google asks.",
                "5. Create a Desktop app sign-in credential and download the setup file.",
            ])
        lines.extend([
            "",
            "When you have the downloaded setup file, paste its path here:",
            "[accent]/Users/eric/Downloads/client_secret.json[/accent]",
        ])
    elif not client_secret_saved and built_in_sign_in_available and status in {"needs_client_secret", "setup"}:
        lines.extend(["", "Built-in Google sign-in is ready. Start Gmail setup to open Google's consent page."])
    if authorization_url:
        lines.extend([
            "",
            "[muted]Open this authorization URL:[/muted]",
            authorization_url,
        ])
    if attempt_id:
        lines.extend([
            "",
            "After Google redirects to localhost, paste the full redirected URL here to complete setup.",
        ])
    if next_step and next_step not in lines:
        lines.extend(["", next_step])
    if docs_url:
        lines.extend(["", f"[muted]Google setup page:[/muted] {docs_url}"])

    console.print()
    console.print(Panel("\n".join(lines), border_style="teal", title="Connector setup"))


def _append_visible_turn(
    history: list[dict[str, Any]],
    *,
    user_input: str,
    assistant_text: str,
    was_greet: bool,
) -> None:
    """Append only real user turns to CLI-local history.

    ``--greet`` is a synthetic wake-up nudge. Keeping only its assistant reply
    creates an invalid assistant-first history, which can make the next user
    turn appear to include Samantha's prior words.
    """
    if was_greet:
        return
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": assistant_text})


def _print_debug_panel(title: str, content: str, *, style: str = "blue") -> None:
    from rich.panel import Panel
    from rich.syntax import Syntax

    # Try to render as syntax-highlighted if it looks like JSON
    try:
        parsed = json.loads(content)
        content = json.dumps(parsed, indent=2, ensure_ascii=False)
        renderable = Syntax(content, "json", theme="monokai", word_wrap=True)
    except (json.JSONDecodeError, TypeError):
        renderable = content  # type: ignore[assignment]

    console.print(Panel(
        renderable,
        title=f"[bold]{title}[/bold]",
        border_style=style,
        expand=True,
    ))


async def _run_chat(dsn: str, *, verbose: bool = False, debug: bool = False,
                    greet: bool = False) -> int:
    import asyncpg
    from apps.tui.textkit import strip_scaffolding
    from apps.tui.tips import mark_seen, random_tip, seen
    from core.agent_api import get_agent_profile_context
    from core.agent_loop import AgentEvent
    from core.cognitive_memory_api import CognitiveMemory
    from core.llm_config import load_llm_config
    from core.tools import ToolContext, create_full_registry
    from services.chat import _build_system_prompt, _hydrate_chat_history, stream_chat_events
    from rich.table import Table

    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)
    history: list[dict[str, Any]] = []
    # One stable id for the whole REPL session; the DB owns active context for it.
    chat_session_id = str(uuid.uuid4())

    try:
        # First-run detection — an unconfigured agent can't answer.
        async with pool.acquire() as conn:
            try:
                if not bool(await conn.fetchval("SELECT is_agent_configured()")):
                    return 2
            except Exception:
                pass  # never block chat on this probe

        # Load config once
        async with pool.acquire() as conn:
            llm_config = await load_llm_config(conn, "llm.chat", fallback_key="llm")

        registry = await create_full_registry(pool)
        agent_profile = await get_agent_profile_context(dsn)
        system_prompt = await _build_system_prompt(agent_profile, registry)

        agent_name = "Hexis"
        if isinstance(agent_profile, dict):
            agent_name = agent_profile.get("name", "Hexis")

        console.print(f"\n[accent]Hexis Chat[/accent] [muted]— streaming conversation with {agent_name}[/muted]")
        mode_flags = []
        if verbose:
            mode_flags.append("verbose")
        if debug:
            mode_flags.append("debug")
        if mode_flags:
            console.print(f"[muted]Mode: {', '.join(mode_flags)} — showing {'full prompt + ' if debug else ''}hydrated context and tool I/O[/muted]")
        console.print("[muted]Type /help for commands. Ctrl+C interrupts a reply; again (or /quit) to exit.[/muted]")
        if not seen("chat.opened"):
            mark_seen("chat.opened")
            console.print("[muted]This is a normal terminal — select text to copy, scroll with your terminal.[/muted]")
        console.print(f"[muted]{random_tip()}[/muted]\n")

        # Debug: show system prompt and LLM config on startup
        if debug:
            _print_debug_panel(
                f"LLM Config ({llm_config.get('provider', '?')}/{llm_config.get('model', '?')})",
                json.dumps({k: v for k, v in llm_config.items() if k != "api_key"}, indent=2),
                style="cyan",
            )
            _print_debug_panel(
                f"System Prompt ({len(system_prompt)} chars)",
                system_prompt,
                style="magenta",
            )

        while True:
            was_greet = False
            if greet:
                # Seed the first turn so the agent wakes and (with consent, no
                # external lookups) offers to learn who you are.
                greet = False
                was_greet = True
                user_input = (
                    "Hi — this is the first time we're talking. Please introduce "
                    "yourself briefly, and if you'd like, feel free to ask a little "
                    "about me — only what I choose to share, and without looking "
                    "anything up unless I say it's okay."
                )
                console.print(f"[accent]you:[/accent] [dim]{user_input}[/dim]")
            else:
                try:
                    console.print("[accent]you:[/accent] ", end="")
                    user_input = input().strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[muted]Goodbye.[/muted]")
                    break

            if not user_input:
                continue

            # Slash commands — one failing command must never crash the session.
            if user_input.startswith("/") and not _looks_like_json_path(user_input):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                try:
                    if cmd in ("/quit", "/exit"):
                        console.print("[muted]Goodbye.[/muted]")
                        break

                    elif cmd == "/help":
                        _print_commands()

                    elif cmd == "/clear":
                        try:
                            await CognitiveMemory(pool).clear_chat_session_context(
                                chat_session_id,
                                reason="cli_clear_command",
                            )
                        except Exception:
                            logger.warning("chat session context clear failed", exc_info=True)
                        history.clear()
                        console.print("[muted]Active context reset (long-term memories are kept).[/muted]\n")

                    elif cmd == "/recall":
                        query = cmd_parts[1] if len(cmd_parts) > 1 else ""
                        if not query:
                            err_console.print("[fail]Usage: /recall <query>[/fail]")
                        else:
                            async with CognitiveMemory.connect(dsn) as mem:
                                result = await mem.recall(query, limit=5)
                            if not result.memories:
                                console.print("[muted]No memories found.[/muted]\n")
                            else:
                                for m in result.memories:
                                    content = m.content[:100] + "..." if len(m.content) > 100 else m.content
                                    console.print(f"  [teal]{m.type}[/teal] {content} [muted](sim: {m.similarity:.2f})[/muted]")
                                console.print()

                    elif cmd == "/status":
                        from core.cli_api import status_payload_rich
                        from apps.hexis_cli import _print_rich_status
                        payload = await status_payload_rich(dsn)
                        _print_rich_status(payload)

                    elif cmd == "/verbose":
                        verbose = not verbose
                        console.print(f"[muted]Verbose mode: {'on' if verbose else 'off'}[/muted]\n")

                    elif cmd == "/debug":
                        debug = not debug
                        verbose = verbose or debug  # debug implies verbose
                        console.print(f"[muted]Debug mode: {'on' if debug else 'off'} (verbose: {'on' if verbose else 'off'})[/muted]\n")
                        if debug:
                            _print_debug_panel(
                                f"System Prompt ({len(system_prompt)} chars)",
                                system_prompt, style="magenta",
                            )

                    elif cmd == "/prompt":
                        _print_debug_panel(
                            f"System Prompt ({len(system_prompt)} chars)",
                            system_prompt, style="magenta",
                        )

                    elif cmd == "/tools":
                        specs = await registry.get_specs(ToolContext.CHAT)
                        table = Table(title="Available Tools", show_lines=False, border_style="dim")
                        table.add_column("Name", style="teal")
                        table.add_column("Description", style="dim", max_width=80)
                        for spec in specs:
                            func = spec.get("function", {})
                            table.add_row(func.get("name", "?"), _truncate(func.get("description", ""), 80))
                        console.print(table)
                        console.print()

                    elif cmd == "/history":
                        history = await _hydrate_chat_history(pool, chat_session_id, history)
                        if not history:
                            console.print("[muted]No conversation history yet.[/muted]\n")
                        else:
                            for i, msg in enumerate(history):
                                console.print(f"  [dim]{i}[/dim] [teal]{msg['role']}[/teal]: {_truncate(msg['content'], 120)}")
                            console.print()

                    else:
                        err_console.print(f"[fail]Unknown command: {cmd}[/fail]")
                        _print_commands()
                except Exception as e:
                    err_console.print(f"[fail]{cmd} failed: {e}[/fail]")
                    err_console.print("[muted]Your conversation is intact — try again or /help.[/muted]")
                continue

            # Normal message — stream response via unified agent runner
            # (subconscious pre-phase → memory hydration → conscious loop)
            session_id = chat_session_id

            try:
                history = await _hydrate_chat_history(pool, session_id, history)
                raw_buf = ""   # full raw model output
                shown = ""     # what we've actually printed (scaffolding-stripped)
                turn_timed_out = False
                tool_calls_log: list[dict[str, Any]] = []

                # Debug: show conversation history being sent
                if debug and history:
                    hist_summary = "\n".join(
                        f"[{m['role']}] {_truncate(m['content'], 150)}"
                        for m in history
                    )
                    _print_debug_panel(
                        f"Conversation History ({len(history)} messages)",
                        hist_summary,
                        style="dim",
                    )

                console.print(f"[teal]{agent_name}:[/teal] ", end="")
                async for event in stream_chat_events(
                    user_message=user_input,
                    history=history,
                    session_id=session_id,
                    dsn=dsn,
                    pool=pool,
                    on_approval=_approve_tool,
                    surface="cli",
                ):
                    if event.event == AgentEvent.PHASE_CHANGE:
                        phase = event.data.get("phase", "")
                        status = event.data.get("status", "")
                        if phase == "memory_recall":
                            count = event.data.get("count", 0)
                            if verbose:
                                console.print(f"\n  [muted]Recalled {count} memories[/muted]", end="")
                        elif phase == "subconscious":
                            if verbose:
                                if status == "start":
                                    console.print("\n  [muted]Subconscious appraisal...[/muted]", end="")
                                elif status == "end":
                                    console.print(" [ok]done[/ok]", end="")

                    elif event.event == AgentEvent.TEXT_DELTA:
                        text = event.data.get("text", "")
                        if text:
                            raw_buf += text
                            # Strip leaked <think>/tool-call scaffolding from what the
                            # user SEES, not just from what we persist. Streaming-safe:
                            # only emit newly-revealed prose (partial tags are held back).
                            visible = strip_scaffolding(raw_buf)[0]
                            if visible.startswith(shown):
                                new = visible[len(shown):]
                                if new:
                                    sys.stdout.write(new)
                                    sys.stdout.flush()
                                    shown = visible
                            else:
                                shown = visible

                    elif event.event == AgentEvent.TOOL_START:
                        tool_name = event.data.get("tool_name", "tool")
                        arguments = event.data.get("arguments", {})
                        tool_calls_log.append({"tool": tool_name, "args": arguments})
                        if verbose:
                            console.print(f"\n  [dim]{tool_name}({_fmt_json(arguments, 300)})[/dim]", end="")
                        else:
                            console.print(f"\n  [dim]{tool_name}...[/dim]", end="")

                    elif event.event == AgentEvent.TOOL_RESULT:
                        tool_name = event.data.get("tool_name", "tool")
                        success = event.data.get("success", False)
                        duration = event.data.get("duration")
                        dur_str = f" [{duration:.1f}s]" if isinstance(duration, (int, float)) else ""
                        ui = _connector_setup_ui(event.data.get("output"))
                        if success:
                            console.print(f" [ok]done[/ok][dim]{dur_str}[/dim]")
                            if ui and _connector_setup_requires_action(ui):
                                _print_connector_setup_ui(ui)
                            if verbose:
                                display = event.data.get("display_output") or event.data.get("output")
                                if display:
                                    console.print(f"    [dim]{_fmt_json(display, 500)}[/dim]")
                        else:
                            error_msg = event.data.get("error", "")
                            console.print(f" [fail]failed[/fail][dim]{dur_str}[/dim] [muted]{error_msg[:120]}[/muted]")
                            if ui and _connector_setup_requires_action(ui):
                                _print_connector_setup_ui(ui)

                    elif event.event == AgentEvent.UI_ARTIFACT:
                        ui = _connector_setup_ui(event.data)
                        if ui and _connector_setup_requires_action(ui):
                            _print_connector_setup_ui(ui)

                    elif event.event == AgentEvent.QUESTION:
                        console.print()
                        await _answer_cli_question(pool, dict(event.data))

                    elif event.event == AgentEvent.LOOP_START:
                        if debug:
                            tool_count = event.data.get("tool_count", 0)
                            energy = event.data.get("energy_budget", "unlimited")
                            console.print(f"\n  [dim]Loop started: {tool_count} tools, energy={energy}[/dim]")

                    elif event.event == AgentEvent.LOOP_END:
                        turn_timed_out = bool(event.data.get("timed_out", False))
                        if debug:
                            reason = event.data.get("stopped_reason", "?")
                            iters = event.data.get("iterations", 0)
                            energy_spent = event.data.get("energy_spent", 0)
                            console.print(
                                f"  [dim]Loop ended: reason={reason}, iterations={iters}, "
                                f"energy_spent={energy_spent}{', TIMED OUT' if turn_timed_out else ''}[/dim]"
                            )

                    elif event.event == AgentEvent.ERROR:
                        error_msg = event.data.get("error", "Unknown error")
                        console.print(f"\n[fail]Error: {error_msg}[/fail]")

                # End the streaming line
                sys.stdout.write("\n")

                # Debug: post-turn summary
                if debug and tool_calls_log:
                    summary = "\n".join(
                        f"  {tc['tool']}({json.dumps(tc['args'], default=str)[:100]})"
                        for tc in tool_calls_log
                    )
                    console.print(f"[dim]Tool calls this turn:\n{summary}[/dim]")

                clean_text = strip_scaffolding(raw_buf)[0].strip() if raw_buf else ""

                if turn_timed_out:
                    console.print("[warn](response timed out — the reply above may be incomplete)[/warn]")

                # Empty response — say so, don't poison history with a blank turn.
                if not clean_text:
                    console.print(
                        "[muted](the model returned an empty response — try rephrasing, "
                        "or check the connection with `hexis doctor --llm`)[/muted]\n"
                    )
                    continue

                console.print()

                if was_greet:
                    continue

                fallback_history = [
                    *history,
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": clean_text},
                ]
                history = await _hydrate_chat_history(pool, chat_session_id, fallback_history)

            except KeyboardInterrupt:
                sys.stdout.write("\n")
                console.print("[muted](interrupted)[/muted]\n")
                continue
            except Exception as e:
                err_console.print(f"\n[fail]Something went wrong: {e}[/fail]")
                err_console.print(
                    "[muted]Your conversation is intact. Try again, or run "
                    "`hexis doctor --llm` to check the model connection.[/muted]\n"
                )

    finally:
        await pool.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hexis chat",
        description="Interactive streaming chat with your Hexis agent.",
    )
    p.add_argument("--dsn", default=None, help="Postgres DSN; defaults to POSTGRES_* env vars")
    p.add_argument("-v", "--verbose", action="store_true", help="Show hydrated context and tool I/O")
    p.add_argument("-d", "--debug", action="store_true", help="Full debug: system prompt, enriched messages, LLM config, tool specs")
    p.add_argument("--greet", action="store_true", help="Seed a first 'wake up' turn (used right after init)")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    from core.agent_api import db_dsn_from_env

    args = build_parser().parse_args(argv)
    dsn = args.dsn or db_dsn_from_env()
    verbose = args.verbose or args.debug
    debug = args.debug

    try:
        rc = asyncio.run(_run_chat(dsn, verbose=verbose, debug=debug, greet=args.greet))
    except KeyboardInterrupt:
        console.print("\n[muted]Goodbye.[/muted]")
        return 0
    except Exception as e:
        err_console.print(f"[fail]Chat failed: {e}[/fail]")
        return 1

    # First-run: agent not configured → offer setup (sync layer, no nested loop).
    if rc == 2:
        console.print("\n[warn]No agent is configured yet.[/warn]")
        try:
            console.print("[accent]Run setup (hexis init) now? [Y/n][/accent] ", end="")
            answer = input().strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"
        if answer in ("", "y", "yes"):
            from apps import hexis_init
            return hexis_init.main([])
        console.print("[muted]Run `hexis init` to set up your agent.[/muted]")
        return 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
