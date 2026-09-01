"""Chat widgets for the Hexis TUI.

Design goals (vs. the old implementation):
  * grouped, role-aligned turns; one label per turn
  * markup-safe rendering — model text goes through Rich ``Text``, never a
    markup f-string, so a stray ``[`` can't corrupt output or raise
  * batched streaming (update on a timer, not per token) + a blinking caret
  * finished answers render as Markdown (headings / code / lists)
  * a dim, collapsible reasoning block fed by the scaffolding strip
  * a compact inline tool tree with status glyphs + duration
"""
from __future__ import annotations

from typing import Any

from rich.style import Style
from rich.text import Text
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Collapsible, Input, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from apps.tui import textkit
from apps.tui.activity import ToolActivity
from apps.tui.design import COLORS, GLYPHS

_TEXT = Style(color=COLORS["text"])
_DIM = Style(color=COLORS["dim"])
_MUTED = Style(color=COLORS["muted"])


# ── Streaming body ───────────────────────────────────────────────────────────

class StreamingBlock(Static):
    """Live assistant text during streaming (plain Rich Text + caret)."""

    def __init__(self) -> None:
        super().__init__("", classes="msg-body")

    def set_text(self, text: str, *, caret: bool = True) -> None:
        t = Text(text, style=_TEXT, no_wrap=False)
        if caret:
            t.append(GLYPHS["caret"], style=Style(color=COLORS["accent"]))
        self.update(t)


# ── Reasoning (downplayed, collapsible) ──────────────────────────────────────

class ReasoningBlock(Collapsible):
    """Dim, dashed, collapsed-by-default block holding captured <think> text."""

    def __init__(self) -> None:
        self._body = Static("")
        super().__init__(self._body, title="thinking", collapsed=True,
                         classes="reasoning")

    def set_text(self, text: str) -> None:
        self._body.update(Text(text, style=_DIM))


# ── Tool tree (compact, inline) ──────────────────────────────────────────────

class ToolTree(Static):
    """Box-drawing list of the turn's tool calls with status + duration."""

    def __init__(self) -> None:
        super().__init__("", classes="tool-line")

    def render_activity(self, activity: ToolActivity) -> None:
        out = Text()
        n = len(activity.entries)
        for i, e in enumerate(activity.entries):
            connector = GLYPHS["leaf"] if i == n - 1 else GLYPHS["branch"]
            out.append(f"{connector} ", style=_MUTED)
            out.append(f"{e.glyph} ", style=Style(color=COLORS[e.token]))
            out.append(e.name, style=_TEXT)
            dur = e.duration_str()
            if dur:
                out.append(f"  {dur}", style=_DIM)
            if e.status == "error" and e.error:
                out.append("\n")
                out.append(f"   {textkit.truncate(e.error, 200)}",
                           style=Style(color=COLORS["danger"]))
            if i != n - 1:
                out.append("\n")
        self.update(out)


# ── Turns ────────────────────────────────────────────────────────────────────

class UserTurn(Vertical):
    def __init__(self, text: str) -> None:
        super().__init__(classes="turn turn-user")
        self._text = text

    def compose(self):
        yield Static(Text("you", style=Style(color=COLORS["accent"], bold=True)),
                     classes="role-label role-user")
        yield Static(Text(self._text, style=_TEXT), classes="msg-body")


class AssistantTurn(Vertical):
    """Streams text, captures reasoning, and tracks tool activity for one turn."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(classes="turn turn-assistant")
        self._agent = agent_name
        self._raw = ""
        self._visible = ""
        self._reasoning = ""
        self._structured_reasoning = ""
        self._show_reasoning = False
        self._activity = ToolActivity()
        self._stream: StreamingBlock | None = None
        self._reasoning_block: ReasoningBlock | None = None
        self._tool_tree: ToolTree | None = None
        self._caret_on = True
        self._done = False

    def compose(self):
        yield Static(Text(self._agent, style=Style(color=COLORS["teal"], bold=True)),
                     classes="role-label role-assistant")
        self._stream = StreamingBlock()
        yield self._stream

    # incoming stream ---------------------------------------------------------
    def append_delta(self, raw: str) -> None:
        self._raw += raw

    def append_reasoning_delta(self, raw: str) -> None:
        self._structured_reasoning += raw

    async def flush(self) -> None:
        """Recompute visible/reasoning from the raw buffer and repaint."""
        if self._done:
            return
        visible, inline_reasoning = textkit.strip_scaffolding(self._raw)
        reasoning = self._structured_reasoning + inline_reasoning
        self._visible = visible
        self._caret_on = not self._caret_on
        if self._stream is not None:
            self._stream.set_text(visible, caret=self._caret_on)
        if reasoning and reasoning != self._reasoning:
            self._reasoning = reasoning
            await self._ensure_reasoning()
            self._reasoning_block.set_text(reasoning)

    async def _ensure_reasoning(self) -> None:
        if self._reasoning_block is None:
            self._reasoning_block = ReasoningBlock()
            self._reasoning_block.collapsed = not self._show_reasoning
            await self.mount(self._reasoning_block)

    # tools -------------------------------------------------------------------
    async def tool_start(self, name: str) -> None:
        self._activity.start(name)
        await self._ensure_tree()
        self._tool_tree.render_activity(self._activity)

    async def tool_result(self, name: str, success: bool,
                          duration: float | None, error: str) -> None:
        self._activity.complete(name, success=success, duration=duration, error=error)
        await self._ensure_tree()
        self._tool_tree.render_activity(self._activity)

    async def _ensure_tree(self) -> None:
        if self._tool_tree is None:
            self._tool_tree = ToolTree()
            await self.mount(self._tool_tree)

    # finalize ----------------------------------------------------------------
    async def finalize(self) -> None:
        self._done = True
        visible, inline_reasoning = textkit.strip_scaffolding(self._raw)
        reasoning = self._structured_reasoning + inline_reasoning
        self._visible = visible
        if reasoning:
            await self._ensure_reasoning()
            self._reasoning_block.set_text(reasoning)
        # Swap the streaming Static for a Markdown render of the final text.
        if self._stream is not None:
            if visible.strip():
                md = Markdown(visible)
                md.add_class("msg-body")
                await self.mount(md, after=self._stream)
            await self._stream.remove()
            self._stream = None

    def show_reasoning(self, show: bool) -> None:
        self._show_reasoning = show
        if self._reasoning_block is not None:
            self._reasoning_block.collapsed = not show


# ── Transcript ───────────────────────────────────────────────────────────────

class Transcript(VerticalScroll):
    """Scrollable list of turns and inline notices."""

    async def add_user(self, text: str) -> None:
        await self.mount(UserTurn(text))
        self.scroll_end(animate=False)

    async def add_assistant(self, agent_name: str) -> AssistantTurn:
        turn = AssistantTurn(agent_name)
        await self.mount(turn)
        self.scroll_end(animate=False)
        return turn

    def write_info(self, text: str) -> None:
        self.mount(Static(Text(text, style=_DIM), classes="msg-info"))
        self.scroll_end(animate=False)

    def write_error(self, text: str) -> None:
        self.mount(Static(Text(f"{GLYPHS['err']} {text}",
                               style=Style(color=COLORS["danger"], bold=True)),
                          classes="msg-error"))
        self.scroll_end(animate=False)

    def write_recall(self, memories: list[Any]) -> None:
        for m in memories:
            line = Text()
            line.append(f"{GLYPHS['bullet']} ", style=_MUTED)
            line.append(f"{m.type} ", style=Style(color=COLORS["teal"]))
            line.append(textkit.truncate(m.content, 110), style=_TEXT)
            sim = getattr(m, "similarity", None)
            if isinstance(sim, (int, float)):
                line.append(f"  ({sim:.2f})", style=_DIM)
            self.mount(Static(line, classes="msg-info"))
        self.scroll_end(animate=False)


# ── Slash-command menu ───────────────────────────────────────────────────────

COMMANDS: list[tuple[str, str]] = [
    ("/help", "show commands"),
    ("/recall", "search memories — /recall <query>"),
    ("/status", "agent status"),
    ("/tools", "list available tools"),
    ("/thinking", "toggle reasoning display"),
    ("/history", "show conversation history"),
    ("/clear", "clear the conversation"),
    ("/debug", "toggle debug logging"),
    ("/quit", "exit chat"),
]


class SlashMenu(OptionList):
    """Autocomplete popup shown while the composer holds a bare '/command'."""

    def show(self, prefix: str) -> None:
        self.clear_options()
        matches = [(c, d) for c, d in COMMANDS if c.startswith(prefix.lower())]
        for cmd, desc in matches:
            label = Text.assemble(
                (cmd, Style(color=COLORS["accent"], bold=True)),
                ("  " + desc, _DIM),
            )
            self.add_option(Option(label, id=cmd))
        self.display = bool(matches)
        if matches:
            self.highlighted = 0

    def hide(self) -> None:
        self.display = False

    def current(self) -> str | None:
        if not self.display or self.highlighted is None:
            return None
        try:
            return self.get_option_at_index(self.highlighted).id
        except Exception:
            return None

    def move(self, delta: int) -> None:
        if not self.option_count:
            return
        cur = self.highlighted or 0
        self.highlighted = (cur + delta) % self.option_count


# ── Composer ─────────────────────────────────────────────────────────────────

class Composer(Input):
    """Message input with slash autocomplete and command history."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(placeholder=textkit.PLACEHOLDERS[0], id="composer", **kwargs)
        self._history: list[str] = []
        self._history_idx: int = -1
        self._saved: str = ""

    def _menu(self) -> SlashMenu | None:
        try:
            return self.screen.query_one(SlashMenu)
        except Exception:
            return None

    def on_key(self, event: Any) -> None:
        menu = self._menu()
        menu_open = bool(menu and menu.display)

        if event.key == "escape" and menu_open:
            menu.hide()
            event.prevent_default()
            return

        if event.key == "tab" and menu_open:
            choice = menu.current()
            if choice:
                self.value = choice + " "
                self.cursor_position = len(self.value)
                menu.hide()
            event.prevent_default()
            return

        if event.key in ("up", "down"):
            if menu_open:
                menu.move(-1 if event.key == "up" else 1)
                event.prevent_default()
                return
            self._walk_history(event.key)
            event.prevent_default()

    def _walk_history(self, key: str) -> None:
        if key == "up":
            if not self._history:
                return
            if self._history_idx == -1:
                self._saved = self.value
                self._history_idx = len(self._history) - 1
            elif self._history_idx > 0:
                self._history_idx -= 1
            self.value = self._history[self._history_idx]
        else:  # down
            if self._history_idx < 0:
                return
            if self._history_idx < len(self._history) - 1:
                self._history_idx += 1
                self.value = self._history[self._history_idx]
            else:
                self._history_idx = -1
                self.value = self._saved
        self.cursor_position = len(self.value)

    def push_history(self, text: str) -> None:
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_idx = -1
        self._saved = ""
