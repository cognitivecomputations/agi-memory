"""Hexis Chat TUI — Textual app for streaming conversations.

Streams through ``services.agent.stream_agent`` (unchanged backend contract),
rendering grouped turns with markup-safe text, batched updates, a collapsible
reasoning block, an inline tool tree, and a progressive-disclosure status bar.
"""
from __future__ import annotations

import uuid
from typing import Any

from dotenv import load_dotenv
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Input, Static
from textual.worker import Worker, WorkerState

from apps.tui.chat_widgets import COMMANDS, Composer, SlashMenu, Transcript
from apps.tui.design import COLORS, GLYPHS
from apps.tui.status_bar import StatusBar
from apps.tui.theme import hexis_theme
from apps.tui.tips import mark_seen, random_tip, seen


class ChatScreen(Screen):
    """Main chat screen: header · transcript · status bar · composer."""

    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", "Interrupt / Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+t", "toggle_thinking", "Thinking"),
        # Keyboard scrolling (mouse wheel is off so the terminal owns the mouse
        # for text selection / copy).
        Binding("pageup", "scroll_transcript('page_up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_transcript('page_down')", "Scroll down", show=False),
        Binding("ctrl+home", "scroll_transcript('home')", "Top", show=False),
        Binding("ctrl+end", "scroll_transcript('end')", "Bottom", show=False),
    ]

    def action_scroll_transcript(self, how: str) -> None:
        tr = self.query_one(Transcript)
        {
            "page_up": tr.scroll_page_up,
            "page_down": tr.scroll_page_down,
            "home": tr.scroll_home,
            "end": tr.scroll_end,
        }.get(how, lambda: None)()

    def __init__(self) -> None:
        super().__init__()
        self._history: list[dict[str, Any]] = []
        self._verbose = False
        self._debug = False
        self._show_reasoning = True
        self._agent_name = "Hexis"
        self._mood = ""
        self._streaming = False
        self._current_turn: Any = None
        self._pending_user = ""
        self._queued: list[str] = []
        self._tool_count = 0
        self._flush_timer: Any = None
        self._greet = False
        self._chat_session_id = str(uuid.uuid4())

    def compose(self) -> ComposeResult:
        yield Static("", id="chat-header")
        yield Transcript(id="transcript")
        yield StatusBar(id="status-bar")
        yield Composer()
        yield SlashMenu(id="slash-menu")

    async def on_mount(self) -> None:
        app: HexisChatApp = self.app  # type: ignore[assignment]
        self._agent_name = app.agent_name
        self._mood = app.mood
        self._render_header()
        self.query_one(SlashMenu).display = False
        status = self.query_one(StatusBar)
        status.update_state(model=app.model_name, mood=self._mood)
        self.query_one(Transcript).write_info(
            "Keyboard-only — type to chat, / for commands, PageUp/PageDown to "
            "scroll. Drag with the mouse to select & copy text."
        )
        # First-ever open: a one-time contextual hint (show-once).
        if not seen("chat.opened"):
            mark_seen("chat.opened")
            self.query_one(Transcript).write_info(
                "First time here — Ctrl+C interrupts a reply, Ctrl+T toggles the "
                "agent's thinking, /help lists everything."
            )
        self.query_one(Transcript).write_info(random_tip())
        await self._refresh_energy()
        # First-run: seed a wake-up turn so the agent greets and (with consent,
        # no external lookups) offers to learn who you are.
        if self._greet:
            self.run_worker(self._seed_greet(), name="greet", exit_on_error=False)

    async def _seed_greet(self) -> None:
        await self._send_message(
            "Hi — this is the first time we're talking. Please introduce yourself "
            "briefly, and if you'd like, feel free to ask a little about me — only "
            "what I choose to share, and without looking anything up unless I say "
            "it's okay."
        )
        self.query_one(Composer).focus()

    def _render_header(self) -> None:
        header = self.query_one("#chat-header", Static)
        t = Text()
        t.append(f"{GLYPHS['logo']} ", style=Style(color=COLORS["accent"]))
        t.append(self._agent_name, style=Style(color=COLORS["accent"], bold=True))
        if self._mood:
            t.append(f"  {self._mood}", style=Style(color=COLORS["dim"]))
        header.update(t)

    async def _refresh_energy(self) -> None:
        """Populate the status bar's energy meter from heartbeat_state."""
        app: HexisChatApp = self.app  # type: ignore[assignment]
        try:
            async with app.pool.acquire() as conn:
                energy = await conn.fetchval(
                    "SELECT current_energy FROM heartbeat_state WHERE id = 1"
                )
                energy_capacity = await conn.fetchval(
                    "SELECT heartbeat_bank_capacity()"
                )
            if energy is not None:
                self.query_one(StatusBar).update_state(
                    energy=int(energy), max_energy=int(energy_capacity or 20)
                )
        except Exception:
            pass  # energy meter is ambient; never block chat on it

    # ── input ────────────────────────────────────────────────────────────────
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "composer":
            return
        menu = self.query_one(SlashMenu)
        value = event.value
        if value.startswith("/") and " " not in value:
            menu.show(value)
        else:
            menu.hide()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "composer":
            return
        text = event.value.strip()
        composer = self.query_one(Composer)
        self.query_one(SlashMenu).hide()
        if not text:
            return

        if text.startswith("/"):
            composer.push_history(text)
            composer.value = ""
            await self._handle_slash(text)
            return

        composer.push_history(text)
        composer.value = ""
        if self._streaming:
            self._queued.append(text)
            self.query_one(Transcript).write_info(f"queued ({len(self._queued)})")
            return
        await self._send_message(text)

    # ── slash commands ───────────────────────────────────────────────────────
    async def _handle_slash(self, text: str) -> None:
        tr = self.query_one(Transcript)
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        app: HexisChatApp = self.app  # type: ignore[assignment]

        if cmd in ("/quit", "/exit"):
            self.app.exit(0)
        elif cmd == "/clear":
            await self._clear_active_context("tui_clear_command")
            self._history.clear()
            await tr.remove_children()
            tr.write_info("Conversation cleared.")
        elif cmd == "/help":
            tr.write_info("Commands:")
            for c, desc in COMMANDS:
                tr.write_info(f"  {c:<11} {desc}")
            tr.write_info("Keys: ctrl+c interrupt · ctrl+l clear · ctrl+t thinking · tab complete")
        elif cmd == "/thinking":
            self._show_reasoning = not self._show_reasoning
            tr.write_info(f"Reasoning display: {'on' if self._show_reasoning else 'off'}")
        elif cmd == "/debug":
            self._debug = not self._debug
            self._verbose = self._debug
            tr.write_info(f"Debug: {'on' if self._debug else 'off'}")
        elif cmd == "/recall":
            if not arg:
                tr.write_error("Usage: /recall <query>")
                return
            try:
                from core.cognitive_memory_api import CognitiveMemory
                async with CognitiveMemory.connect(app.dsn) as mem:
                    result = await mem.recall(arg, limit=5)
                if not result.memories:
                    tr.write_info("No memories found.")
                else:
                    tr.write_recall(result.memories)
            except Exception as e:
                tr.write_error(str(e))
        elif cmd == "/status":
            try:
                from core.cli_api import status_payload_rich
                payload = await status_payload_rich(app.dsn)
                agent = payload.get("agent", {}) if isinstance(payload.get("agent"), dict) else {}
                tr.write_info(f"Agent: {agent.get('name', self._agent_name)}")
                if payload.get("mood"):
                    tr.write_info(f"Mood: {payload.get('mood')}")
                energy = payload.get("energy")
                max_e = payload.get("energy_capacity", payload.get("max_energy"))
                reserve = payload.get("energy_reserve", payload.get("max_energy"))
                if energy is not None:
                    tr.write_info(f"Energy: {energy}/{max_e} (reserve {reserve})")
                consent = payload.get("consent", {})
                if isinstance(consent, dict) and consent.get("status"):
                    tr.write_info(f"Consent: {consent.get('status')}")
            except Exception as e:
                tr.write_error(str(e))
        elif cmd == "/tools":
            try:
                from core.tools import ToolContext
                specs = await app.registry.get_specs(ToolContext.CHAT)
                tr.write_info(f"Available tools ({len(specs)}):")
                for spec in specs:
                    func = spec.get("function", {})
                    tr.write_info(f"  {func.get('name', '?')} — "
                                  f"{(func.get('description') or '')[:60]}")
            except Exception as e:
                tr.write_error(str(e))
        elif cmd == "/history":
            try:
                from services.chat import _hydrate_chat_history
                self._history = await _hydrate_chat_history(app.pool, self._chat_session_id, self._history)
            except Exception:
                pass
            if not self._history:
                tr.write_info("No conversation history yet.")
            else:
                for i, msg in enumerate(self._history):
                    tr.write_info(f"  {i} [{msg['role']}]: {msg['content'][:100]}")
        else:
            tr.write_error(f"Unknown command: {cmd}")
            tr.write_info("Type /help for available commands.")

    # ── streaming ──────────────────────────────────────────────────────────────
    async def _send_message(self, text: str) -> None:
        tr = self.query_one(Transcript)
        await tr.add_user(text)
        self._pending_user = text
        self._tool_count = 0
        self._streaming = True
        self._current_turn = await tr.add_assistant(self._agent_name)
        self._current_turn.show_reasoning(self._show_reasoning)
        self.query_one(StatusBar).set_busy("thinking")
        self._flush_timer = self.set_interval(0.1, self._flush)
        # exit_on_error=False → an LLM/stream failure shows as an error line
        # instead of crashing the chat app (on_worker_state_changed handles it).
        self.run_worker(self._stream_response(text), name="chat-stream",
                        exclusive=True, exit_on_error=False)

    async def _flush(self) -> None:
        if self._current_turn is not None:
            await self._current_turn.flush()
            self.query_one(Transcript).scroll_end(animate=False)

    async def _stream_response(self, user_input: str) -> None:
        from core.agent_loop import AgentEvent
        from services.agent import stream_agent
        from services.chat import _hydrate_chat_history

        app: HexisChatApp = self.app  # type: ignore[assignment]
        tr = self.query_one(Transcript)
        status = self.query_one(StatusBar)
        turn = self._current_turn
        session_id = self._chat_session_id
        saw_token = False
        self._history = await _hydrate_chat_history(app.pool, session_id, self._history)

        async for event in stream_agent(
            app.pool, app.registry,
            user_message=user_input, mode="chat",
            history=self._history, session_id=session_id, dsn=app.dsn,
        ):
            ev = event.event
            data = event.data
            if ev == AgentEvent.PHASE_CHANGE:
                phase = data.get("phase", "")
                st = data.get("status", "")
                if phase == "memory_recall":
                    status.set_busy("recalling memories" if st == "start" else "thinking")
                    if st == "end" and self._verbose:
                        tr.write_info(f"recalled {data.get('count', 0)} memories")
                elif phase == "subconscious":
                    status.set_busy("reflecting" if st == "start" else "thinking")
                elif phase in ("plan", "execute", "verify"):
                    status.set_busy(phase + "ing" if phase != "plan" else "planning")
            elif ev == AgentEvent.TEXT_DELTA:
                text = data.get("text", "")
                if text and turn is not None:
                    if not saw_token:
                        saw_token = True
                        status.set_busy("")  # switch to rotating think-verbs
                    turn.append_delta(text)
            elif ev == AgentEvent.TOOL_START:
                name = data.get("tool_name", "tool")
                self._tool_count += 1
                status.update_state(tools=self._tool_count)
                status.set_busy(f"running {name}")
                if turn is not None:
                    await turn.tool_start(name)
            elif ev == AgentEvent.TOOL_RESULT:
                if turn is not None:
                    await turn.tool_result(
                        data.get("tool_name", "tool"), data.get("success", False),
                        data.get("duration"), data.get("error", "") or "",
                    )
                status.set_busy("")
            elif ev == AgentEvent.CONTINUATION:
                tr.write_info("continuing…")
            elif ev == AgentEvent.ENERGY_EXHAUSTED:
                tr.write_info("energy exhausted")
            elif ev == AgentEvent.ERROR:
                tr.write_error(data.get("error", "Unknown error"))

    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "chat-stream":
            return
        if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            return

        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None

        turn = self._current_turn
        if turn is not None:
            await turn.finalize()
        self._streaming = False
        self._current_turn = None
        status = self.query_one(StatusBar)
        status.set_idle()

        tr = self.query_one(Transcript)
        if event.state == WorkerState.ERROR:
            tr.write_error(str(event.worker.error))
        elif event.state == WorkerState.CANCELLED:
            tr.write_info("interrupted")

        if event.state == WorkerState.SUCCESS and turn is not None:
            final_text = getattr(turn, "_visible", "") or ""
            self._history.append({"role": "user", "content": self._pending_user})
            self._history.append({"role": "assistant", "content": final_text})
            if final_text:
                self.run_worker(
                    self._remember(self._pending_user, final_text),
                    name="chat-remember", exit_on_error=False,
                )

        await self._refresh_energy()
        self.query_one(Composer).focus()

        # Drain one queued message.
        if self._queued and not self._streaming:
            await self._send_message(self._queued.pop(0))

    async def _remember(self, user_input: str, assistant_text: str) -> None:
        app: HexisChatApp = self.app  # type: ignore[assignment]
        try:
            from core.cognitive_memory_api import CognitiveMemory
            from services.chat import _hydrate_chat_history, _remember_conversation
            await _remember_conversation(
                CognitiveMemory(app.pool),
                user_message=user_input,
                assistant_message=assistant_text,
                session_id=self._chat_session_id,
                surface="tui",
            )
            self._history = await _hydrate_chat_history(app.pool, self._chat_session_id, self._history)
        except Exception as e:
            if self._debug:
                self.query_one(Transcript).write_info(f"[memory] not stored: {e}")

    # ── actions ────────────────────────────────────────────────────────────────
    def action_interrupt_or_quit(self) -> None:
        if self._streaming:
            for worker in list(self.workers):
                if worker.name == "chat-stream":
                    worker.cancel()
        else:
            self.app.exit(0)

    async def action_clear_chat(self) -> None:
        await self._clear_active_context("tui_clear_action")
        self._history.clear()
        await self.query_one(Transcript).remove_children()
        self.query_one(Transcript).write_info("Conversation cleared.")

    async def _clear_active_context(self, reason: str) -> None:
        app: HexisChatApp = self.app  # type: ignore[assignment]
        try:
            from core.cognitive_memory_api import CognitiveMemory
            await CognitiveMemory(app.pool).clear_chat_session_context(
                self._chat_session_id,
                reason=reason,
            )
        except Exception as e:
            if self._debug:
                self.query_one(Transcript).write_info(f"[memory] context clear failed: {e}")

    def action_toggle_thinking(self) -> None:
        self._show_reasoning = not self._show_reasoning
        if self._current_turn is not None:
            self._current_turn.show_reasoning(self._show_reasoning)
        self.query_one(Transcript).write_info(
            f"Reasoning display: {'on' if self._show_reasoning else 'off'}"
        )


class HexisChatApp(App):
    """Textual TUI for `hexis chat`."""

    TITLE = "Hexis Chat"
    CSS_PATH = "hexis.tcss"

    def __init__(self, argv: list[str] | None = None) -> None:
        super().__init__()
        self.register_theme(hexis_theme)
        self.theme = "hexis"
        self._argv = argv or []
        self.pool: Any = None
        self.registry: Any = None
        self.agent_name: str = "Hexis"
        self.model_name: str = ""
        self.mood: str = ""
        self.dsn: str = ""
        self._verbose = False
        self._debug = False
        self._greet = False  # --greet: seed a first "wake up" turn on first run
        # Set to "init" when a first-run chat should hand off to setup.
        self.next_action: str | None = None

    def get_css_variables(self) -> dict[str, str]:
        from apps.tui.design import CSS_VARS
        variables = super().get_css_variables()
        variables.update(CSS_VARS)
        return variables

    def run(self, **kwargs):
        # Keyboard-only by default: don't capture the mouse, so the terminal's
        # native click-drag text selection / copy keeps working. Set
        # HEXIS_TUI_MOUSE=1 to re-enable mouse interaction.
        import os
        kwargs.setdefault("mouse", os.getenv("HEXIS_TUI_MOUSE", "") == "1")
        return super().run(**kwargs)

    async def on_mount(self) -> None:
        load_dotenv()
        import asyncpg
        from core.agent_api import db_dsn_from_env, get_agent_profile_context
        from core.llm_config import load_llm_config
        from core.tools import create_full_registry

        dsn = None
        i = 0
        while i < len(self._argv):
            a = self._argv[i]
            if a == "--dsn" and i + 1 < len(self._argv):
                dsn = self._argv[i + 1]
                i += 2
            elif a in ("-v", "--verbose"):
                self._verbose = True
                i += 1
            elif a in ("-d", "--debug"):
                self._debug = True
                self._verbose = True
                i += 1
            elif a == "--greet":
                self._greet = True
                i += 1
            else:
                i += 1

        self.dsn = dsn or db_dsn_from_env()
        configured = True
        try:
            self.pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=5)
            async with self.pool.acquire() as conn:
                try:
                    configured = bool(await conn.fetchval("SELECT is_agent_configured()"))
                except Exception:
                    configured = True  # never block chat on this probe
                llm_config = await load_llm_config(conn, "llm.chat", fallback_key="llm")
            self.model_name = llm_config.get("model", "") if isinstance(llm_config, dict) else ""
            self.registry = await create_full_registry(self.pool)
            agent_profile = await get_agent_profile_context(self.dsn)
            if isinstance(agent_profile, dict):
                self.agent_name = agent_profile.get("name", "Hexis")
                self.mood = agent_profile.get("mood", "") or ""
        except Exception as e:
            from apps.tui.dialogs import ErrorDialog
            await self.push_screen(ErrorDialog("Connection Error", str(e)))
            return

        # First-run detection: an empty chat on an unconfigured agent just fails
        # at the first message. Offer setup instead.
        if not configured:
            from apps.tui.dialogs import ConfirmDialog

            def _decide(run_init: bool | None) -> None:
                if run_init:
                    self.next_action = "init"
                self.exit(0)

            await self.push_screen(
                ConfirmDialog(
                    "Not set up yet",
                    "No agent is configured. Run setup (hexis init) now?",
                ),
                _decide,
            )
            return

        screen = ChatScreen()
        screen._verbose = self._verbose
        screen._debug = self._debug
        screen._greet = self._greet
        await self.push_screen(screen)

    async def on_unmount(self) -> None:
        if self.pool:
            try:
                await self.pool.close()
            except Exception:
                pass
