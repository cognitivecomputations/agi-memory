"""Init wizard screens for the Hexis TUI."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
)
from textual.worker import Worker, WorkerState
from rich.style import Style
from rich.text import Text

from apps.tui.design import COLORS
from apps.tui.init_widgets import (
    BigFiveSliders,
    CharacterPreview,
    ModelCombo,
    ModelMenu,
    StepBar,
)


def _teal(text: str) -> Text:
    return Text(text, style=Style(color=COLORS["teal"], bold=True))


def _plain(text: str, token: str = "text") -> Text:
    return Text(text, style=Style(color=COLORS[token]))


# ── Helpers ──────────────────────────────────────────────────────────────────

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "openai-codex": "gpt-5.2",
    "grok": "grok-3",
    "gemini": "gemini-2.5-flash",
    "chutes": "deepseek-ai/DeepSeek-V3-0324",
    "github-copilot": "gpt-4o",
    "qwen-portal": "qwen-max-latest",
    "minimax-portal": "MiniMax-M1",
    "google-gemini-cli": "gemini-2.5-flash",
    "google-antigravity": "gemini-2.5-flash",
}

_PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-codex": "",
    "grok": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "chutes": "",
    "github-copilot": "",
    "qwen-portal": "",
    "minimax-portal": "",
    "google-gemini-cli": "",
    "google-antigravity": "",
}

_PROVIDER_OPTIONS: list[tuple[str, str]] = [
    ("OpenAI Codex (ChatGPT OAuth)", "openai-codex"),
    ("OpenAI Platform (API key)", "openai"),
    ("Anthropic (API key)", "anthropic"),
    ("Grok (xAI)", "grok"),
    ("Gemini", "gemini"),
    ("Chutes (OAuth)", "chutes"),
    ("GitHub Copilot (OAuth)", "github-copilot"),
    ("Qwen Portal (OAuth)", "qwen-portal"),
    ("MiniMax Portal (OAuth)", "minimax-portal"),
    ("Google Gemini CLI (OAuth)", "google-gemini-cli"),
    ("Google Antigravity (OAuth)", "google-antigravity"),
]

_OAUTH_PROVIDERS: set[str] = {
    "openai-codex",
    "chutes",
    "github-copilot",
    "qwen-portal",
    "minimax-portal",
    "google-gemini-cli",
    "google-antigravity",
}


def _persisted_provider(provider: str) -> str:
    """Map a wizard-only provider alias to the id the LLM layer understands."""
    return provider


def _normalize_provider(raw: str) -> str:
    val = (raw or "").strip().lower()
    aliases = {
        "openai_codex": "openai-codex",
        "github_copilot": "github-copilot",
        "qwen_portal": "qwen-portal",
        "minimax_portal": "minimax-portal",
        "google_gemini_cli": "google-gemini-cli",
        "google_antigravity": "google-antigravity",
    }
    return aliases.get(val, val)


def _state(screen: Screen) -> Any:
    """Get the shared InitState from the app."""
    return screen.app.state  # type: ignore[attr-defined]


def _conn(screen: Screen) -> Any:
    """Get the DB connection from the app."""
    return screen.app.conn  # type: ignore[attr-defined]


# ── Already-configured: keep or reconfigure ──────────────────────────────────

class ReconfigureScreen(Screen):
    """Shown when `hexis init` runs on an already-configured agent.

    Non-destructive: nothing changes unless the user chooses to reconfigure.
    """

    def __init__(self, summary: dict[str, Any]) -> None:
        super().__init__()
        self._summary = summary

    def compose(self) -> ComposeResult:
        s = self._summary
        with VerticalScroll(classes="form-container"):
            yield Static(Text(f"{s.get('name', 'Your agent')} is already set up",
                              style=Style(color=COLORS["accent"], bold=True)))
            yield Static("")
            line = Text()
            line.append("Model: ", style=Style(color=COLORS["teal"]))
            line.append(f"{s.get('provider', '?')} / {s.get('model', '?')}",
                        style=Style(color=COLORS["text"]))
            yield Static(line)
            yield Static("")
            yield Static(_plain(
                "Reconfiguring walks through setup again and overwrites the "
                "current configuration. Keeping leaves everything as-is.", "dim"))
        with Horizontal(classes="button-bar"):
            yield Button("Keep — exit", id="keep", classes="muted")
            yield Button("Reconfigure", id="reconfigure", classes="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "reconfigure":
            self.app.switch_screen(LLMConfigScreen())
        elif event.button.id == "keep":
            self.app.exit(0)


# ── 1. LLM Config ───────────────────────────────────────────────────────────

class LLMConfigScreen(Screen):
    """Configure LLM provider, model, endpoint, and API key env var."""

    @staticmethod
    def _default_endpoint(provider: str) -> str:
        if provider == "openai":
            return os.getenv("OPENAI_BASE_URL", "")
        return ""

    def compose(self) -> ComposeResult:
        provider_default = _normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
        if provider_default not in _DEFAULT_MODELS:
            provider_default = "openai"
        model_default = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(provider_default, "gpt-4o"))
        endpoint_default = self._default_endpoint(provider_default)
        key_default = _PROVIDER_ENV_VARS.get(provider_default, "OPENAI_API_KEY")

        yield StepBar(current=0)
        with VerticalScroll(classes="form-container"):
            yield Static("[bold #d8774f]LLM Configuration[/bold #d8774f]")
            yield Static("")

            yield Label("Provider", classes="form-label")
            yield Select(
                _PROVIDER_OPTIONS,
                value=provider_default,
                allow_blank=False,
                id="provider",
            )
            yield Static("", id="provider-help", classes="form-label")

            yield Label("Model", classes="form-label")
            yield ModelCombo(
                value=model_default,
                placeholder="Type a model, or ↓ to pick from the list",
                id="model",
            )
            yield ModelMenu(id="model-menu")

            yield Label("Endpoint (blank for provider default)", classes="form-label")
            yield Input(
                value=endpoint_default,
                placeholder="https://...",
                id="endpoint",
            )

            yield Label("API key env var name", classes="form-label")
            yield Input(
                value=key_default,
                placeholder="e.g. OPENAI_API_KEY",
                id="api-key-env",
            )
            yield Static("", id="llm-status", classes="hint")
            yield Static(
                Text("Tip: you can change any of this later with `hexis init`.",
                     style=Style(color=COLORS["dim"])),
                classes="hint",
            )

        with Horizontal(classes="button-bar"):
            yield Button("Test connection", id="test-conn", classes="secondary")
            yield Button("Next", id="next", classes="primary")

    def _selected_provider(self) -> str:
        provider_widget = self.query_one("#provider", Select)
        value = provider_widget.value
        if isinstance(value, str):
            provider = _normalize_provider(value)
            if provider in _DEFAULT_MODELS:
                return provider
        return "openai"

    def _apply_provider_defaults(
        self,
        provider: str,
        *,
        preserve_model: bool = False,
        preserve_endpoint: bool = False,
    ) -> None:
        model_input = self.query_one("#model", Input)
        if not preserve_model or not model_input.value:
            model_input.value = _DEFAULT_MODELS.get(provider, model_input.value or "gpt-4o")

        endpoint_input = self.query_one("#endpoint", Input)
        key_input = self.query_one("#api-key-env", Input)
        help_text = self.query_one("#provider-help", Static)
        if provider in _OAUTH_PROVIDERS:
            endpoint_input.value = ""
            endpoint_input.placeholder = "Not required for OAuth providers"
            endpoint_input.disabled = True
            key_input.value = ""
            key_input.placeholder = "Not required for OAuth providers"
            key_input.disabled = True
            if provider == "openai-codex":
                help_text.update("Next opens browser OAuth for ChatGPT Plus/Pro.")
            else:
                help_text.update("Next runs provider OAuth/device-code login.")
        else:
            endpoint_input.disabled = False
            endpoint_input.placeholder = "https://..."
            if not preserve_endpoint:
                endpoint_input.value = self._default_endpoint(provider)
            key_input.disabled = False
            key_input.placeholder = "e.g. OPENAI_API_KEY"
            default_key = _PROVIDER_ENV_VARS.get(provider, "")
            if default_key:
                key_input.value = default_key
            help_text.update("")

    def on_mount(self) -> None:
        self._apply_provider_defaults(
            self._selected_provider(),
            preserve_model=bool(os.getenv("LLM_MODEL")),
            preserve_endpoint=True,
        )
        self._load_models()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "provider":
            return
        provider = _normalize_provider(str(event.value)) if isinstance(event.value, str) else "openai"
        self._apply_provider_defaults(provider)
        self.query_one("#model-menu", ModelMenu).hide()
        self._load_models()

    # ── model dropdown ───────────────────────────────────────────────────────
    def _load_models(self) -> None:
        provider = self._selected_provider()
        try:
            endpoint = self.query_one("#endpoint", Input).value.strip()
        except Exception:
            endpoint = ""
        self.run_worker(self._fetch_models(provider, endpoint),
                        name="model-fetch", group="model-fetch", exclusive=True,
                        exit_on_error=False)

    async def _fetch_models(self, provider: str, endpoint: str) -> None:
        from apps.tui import model_catalog
        models = await model_catalog.fetch_models(provider, endpoint=endpoint or None)
        try:
            self.query_one("#model-menu", ModelMenu).populate(models)
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "model":
            return
        menu = self.query_one("#model-menu", ModelMenu)
        if menu._suppress:
            menu._suppress = False
            menu.hide()
            self._model_advisory(event.value)
            return
        menu.apply_filter(event.value)
        self._model_advisory(event.value)

    def _model_advisory(self, value: str) -> None:
        """Soft, non-blocking hint: unknown model name or missing API key."""
        import os as _os

        status = self.query_one("#llm-status", Static)
        provider = self._selected_provider()
        menu = self.query_one("#model-menu", ModelMenu)
        val = (value or "").strip()

        if val and menu._all and val not in menu._all:
            status.update(Text(
                f"Heads up: “{val}” isn't in the known list for {provider} — "
                "double-check the spelling (custom names are fine).",
                style=Style(color=COLORS["warn"])))
            return
        if provider not in _OAUTH_PROVIDERS:
            key_env = self.query_one("#api-key-env", Input).value.strip()
            if key_env and not _os.getenv(key_env):
                status.update(Text(
                    f"Note: ${key_env} isn't set in this environment yet — "
                    "the model won't respond until it is.",
                    style=Style(color=COLORS["dim"])))
                return
        status.update("")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "model-menu":
            return
        menu = self.query_one("#model-menu", ModelMenu)
        combo = self.query_one("#model", ModelCombo)
        menu._suppress = True
        combo.value = event.option.id or ""
        combo.cursor_position = len(combo.value)
        menu.hide()
        combo.focus()

    async def _ensure_openai_codex_login(self) -> None:
        import socket
        from core.browser import open_url

        from core.auth.callback_server import run_callback_server
        from core.auth.openai_codex import (
            build_authorize_url,
            create_state,
            ensure_fresh_openai_codex_credentials,
            exchange_authorization_code,
            generate_pkce,
            save_openai_codex_credentials,
        )

        # Match CLI behavior: callback server binds localhost:1455.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 1455))
        except OSError as e:
            raise RuntimeError(
                "OpenAI Codex OAuth needs localhost:1455, but the port is in use."
            ) from e

        verifier, challenge = generate_pkce()
        state = create_state()
        auth_url = build_authorize_url(challenge=challenge, state=state)
        try:
            open_url(auth_url)
        except Exception:
            pass

        result = await asyncio.to_thread(
            run_callback_server,
            1455,
            "/auth/callback",
            180,
            state,
        )
        code = result.get("code") if result else None
        if not code:
            raise RuntimeError(
                "OpenAI Codex OAuth callback not received. "
                "Retry, or run `hexis auth openai-codex login` in a terminal."
            )

        creds = await exchange_authorization_code(code=code, verifier=verifier)
        save_openai_codex_credentials(creds)
        await ensure_fresh_openai_codex_credentials(skew_seconds=0)

    async def _ensure_provider_auth(self, provider: str) -> None:
        if provider not in _OAUTH_PROVIDERS:
            return
        if provider == "openai-codex":
            await self._ensure_openai_codex_login()
            return

        from apps.hexis_init import _ensure_oauth_login

        state = _state(self)
        await _ensure_oauth_login(
            provider,
            state.dsn,
            _conn(self),
            wait_seconds=state.wait_seconds,
            allow_manual_fallback=False,
        )

    def _test_connection(self) -> None:
        """Advisory: make one real call to check the current form's config works."""
        import os as _os

        provider = self._selected_provider()
        status = self.query_one("#llm-status", Static)
        if provider in _OAUTH_PROVIDERS:
            status.update(Text(
                "OAuth providers are verified when you sign in on Next.",
                style=Style(color=COLORS["dim"])))
            return
        key_env = self.query_one("#api-key-env", Input).value.strip()
        cfg = {
            "provider": provider,
            "model": self.query_one("#model", Input).value.strip(),
            "endpoint": self.query_one("#endpoint", Input).value.strip() or None,
            "api_key": _os.getenv(key_env) if key_env else None,
        }
        status.update(Text("Testing connection…", style=Style(color=COLORS["dim"])))
        self.query_one("#test-conn", Button).disabled = True
        self.run_worker(self._do_test_connection(cfg), name="conn-test",
                        group="conn-test", exclusive=True, exit_on_error=False)

    async def _do_test_connection(self, cfg: dict[str, Any]) -> None:
        from core.init_api import test_llm_connection

        result = await test_llm_connection(cfg)
        tok = "ok" if result["ok"] else (
            "warn" if result["status"] in ("rate_limit", "network") else "danger")
        try:
            self.query_one("#llm-status", Static).update(
                Text(result["message"], style=Style(color=COLORS[tok])))
            self.query_one("#test-conn", Button).disabled = False
        except Exception:
            pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "test-conn":
            self._test_connection()
            return
        if event.button.id != "next":
            return
        state = _state(self)

        state.provider = self._selected_provider()
        state.model = self.query_one("#model", Input).value.strip()
        state.endpoint = self.query_one("#endpoint", Input).value.strip()
        state.api_key_env = self.query_one("#api-key-env", Input).value.strip()
        if state.provider in _OAUTH_PROVIDERS:
            # OAuth providers don't use API key env vars in llm config.
            state.api_key_env = ""

        # Subconscious defaults to same config
        state.sub_provider = state.provider
        state.sub_model = state.model
        state.sub_endpoint = state.endpoint
        state.sub_key_env = state.api_key_env

        self._hb_config = {
            "provider": _persisted_provider(state.provider),
            "model": state.model,
            "endpoint": state.endpoint,
            "api_key_env": state.api_key_env,
        }
        self._sub_config = {
            "provider": _persisted_provider(state.sub_provider),
            "model": state.sub_model,
            "endpoint": state.sub_endpoint,
            "api_key_env": state.sub_key_env,
        }

        # Run auth + persistence off the UI thread so the form never freezes
        # (OAuth can block on a browser round-trip).
        self.query_one("#next", Button).disabled = True
        status = self.query_one("#llm-status", Static)
        if state.provider in _OAUTH_PROVIDERS:
            status.update(Text("Authorizing… a browser window may open — finish login there.",
                               style=Style(color=COLORS["accent"])))
        else:
            status.update(Text("Saving configuration…", style=Style(color=COLORS["dim"])))
        self.run_worker(self._authorize_and_save(), name="llm-auth",
                        group="llm-auth", exclusive=True, exit_on_error=False)

    async def _authorize_and_save(self) -> None:
        state = _state(self)
        conn = _conn(self)
        await self._ensure_provider_auth(state.provider)
        await conn.fetchval(
            "SELECT init_llm_config($1::jsonb, $2::jsonb)",
            json.dumps(self._hb_config),
            json.dumps(self._sub_config),
        )
        await conn.execute(
            "SELECT set_config('llm.heartbeat', $1::jsonb)", json.dumps(self._hb_config)
        )
        await conn.execute(
            "SELECT set_config('llm.chat', $1::jsonb)", json.dumps(self._hb_config)
        )
        await conn.execute(
            "SELECT set_config('llm.subconscious', $1::jsonb)", json.dumps(self._sub_config)
        )

    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "llm-auth":
            return
        if event.state == WorkerState.SUCCESS:
            state = _state(self)
            if getattr(state, "retry_consent", False):
                # Retrying after a refusal — identity is already saved, so go
                # straight back to consent with the newly chosen model.
                state.retry_consent = False
                self.app.switch_screen(ConsentScreen())
            else:
                self.app.switch_screen(ChoosePathScreen())
        elif event.state == WorkerState.ERROR:
            err = event.worker.error
            self.query_one("#next", Button).disabled = False
            self.query_one("#llm-status", Static).update("")
            from apps.tui.dialogs import ErrorDialog
            title = "Auth Error" if isinstance(err, RuntimeError) else "DB Error"
            await self.app.push_screen(ErrorDialog(title, str(err)))


# ── 3. Choose Path ───────────────────────────────────────────────────────────

class ChoosePathScreen(Screen):
    """Choose between Express, Character, and Custom setup paths."""

    def compose(self) -> ComposeResult:
        yield StepBar(current=1)
        with VerticalScroll(classes="form-container"):
            yield Static("[bold #d8774f]Choose Your Path[/bold #d8774f]")
            yield Static("")
            yield Label("What should the agent call you?", classes="form-label")
            yield Input(value="User", id="user-name")
            yield Static("")
            yield RadioSet(
                RadioButton(
                    "Express — Use sensible defaults, start immediately",
                    id="express",
                    value=True,
                ),
                RadioButton(
                    "Character — Pick a personality preset",
                    id="character",
                ),
                RadioButton(
                    "Custom — Full control over identity, values, goals",
                    id="custom",
                ),
                id="path-choice",
            )
        with Horizontal(classes="button-bar"):
            yield Button("Back", id="back", classes="muted")
            yield Button("Next", id="next", classes="primary")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.switch_screen(LLMConfigScreen())
            return

        if event.button.id == "next":
            state = _state(self)
            state.user_name = self.query_one("#user-name", Input).value.strip() or "User"

            rs = self.query_one("#path-choice", RadioSet)
            idx = rs.pressed_index
            tier = ["express", "character", "custom"][idx if idx >= 0 else 0]
            state.tier = tier

            if tier == "express":
                conn = _conn(self)
                try:
                    await conn.fetchval("SELECT init_with_defaults($1)", state.user_name)
                except Exception as e:
                    from apps.tui.dialogs import ErrorDialog
                    await self.app.push_screen(ErrorDialog("DB Error", str(e)))
                    return
                self.app.switch_screen(ConsentScreen())
            elif tier == "character":
                self.app.switch_screen(CharacterGalleryScreen())
            else:
                self.app.switch_screen(CustomSetupScreen())


# ── 5. Character Gallery ─────────────────────────────────────────────────────

class CharacterGalleryScreen(Screen):
    """Pick a personality preset from available character cards."""

    def compose(self) -> ComposeResult:
        yield StepBar(current=2)
        with Horizontal(id="character-gallery"):
            yield DataTable(id="char-table", cursor_type="row")
            yield CharacterPreview(id="char-preview")
        with Horizontal(classes="button-bar"):
            yield Button("Back", id="back", classes="muted")
            yield Button("Select", id="select", classes="primary")

    def on_mount(self) -> None:
        from core.init_api import load_character_cards, get_card_summary

        state = _state(self)
        state.character_cards = load_character_cards()

        table = self.query_one("#char-table", DataTable)
        table.add_columns("#", "Name", "Voice", "Values")

        for i, card in enumerate(state.character_cards, 1):
            summary = get_card_summary(card)
            voice_preview = (summary.get("voice") or "")[:40]
            if len(summary.get("voice", "") or "") > 40:
                voice_preview += "..."
            table.add_row(
                str(i),
                summary["name"],
                voice_preview,
                (summary.get("values") or "\u2014")[:40],
                key=str(i),
            )

        # Show preview for first card
        if state.character_cards:
            preview = self.query_one("#char-preview", CharacterPreview)
            preview.update_preview(state.character_cards[0])

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        state = _state(self)
        if event.cursor_row is not None and event.cursor_row < len(state.character_cards):
            preview = self.query_one("#char-preview", CharacterPreview)
            preview.update_preview(state.character_cards[event.cursor_row])

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.switch_screen(ChoosePathScreen())
            return

        if event.button.id == "select":
            state = _state(self)
            conn = _conn(self)
            table = self.query_one("#char-table", DataTable)
            row_idx = table.cursor_row
            if row_idx is not None and row_idx < len(state.character_cards):
                card = state.character_cards[row_idx]
                state.chosen_card = card
                from core.init_api import load_character_card_document

                character_card = load_character_card_document(card)
                try:
                    await conn.fetchval(
                        "SELECT init_from_character_card($1::jsonb, $2)",
                        json.dumps(character_card),
                        state.user_name,
                    )
                except Exception as e:
                    from apps.tui.dialogs import ErrorDialog
                    await self.app.push_screen(ErrorDialog("DB Error", str(e)))
                    return
                self.app.switch_screen(ConsentScreen())


# ── 6. Custom Setup ──────────────────────────────────────────────────────────

class CustomSetupScreen(Screen):
    """3-phase custom configuration: Who / What Matters / What's Next."""

    _phase: int = 0

    def compose(self) -> ComposeResult:
        yield StepBar(current=2)
        # Phase 1: Who
        with VerticalScroll(classes="form-container", id="phase-who"):
            yield Static("[bold #d8774f]Custom Setup — Phase 1: Who[/bold #d8774f]")
            yield Static("")

            yield Label("Agent name", classes="form-label")
            yield Input(value=_state(self).agent_name, id="agent-name")

            yield Label("Pronouns", classes="form-label")
            yield Input(value=_state(self).pronouns, id="pronouns")

            yield Label("Voice / tone", classes="form-label")
            yield Input(value=_state(self).voice, id="voice")

            yield Label("Description", classes="form-label")
            yield Input(
                value=_state(self).description or f"I am {_state(self).agent_name}, a developing mind.",
                id="description",
            )

            yield Label("Purpose", classes="form-label")
            yield Input(value=_state(self).purpose, id="purpose")

            yield Label("Personality (brief description)", classes="form-label")
            yield Input(value=_state(self).personality_desc, id="personality-desc")

            yield Static("")
            yield Static("[#3c6f64]Big Five Personality Traits[/#3c6f64] (optional)")
            yield BigFiveSliders(defaults=_state(self).personality_traits)

            yield Static("")
            yield Label("What should the agent call you?", classes="form-label")
            yield Input(value=_state(self).user_name, id="user-name")

        with Horizontal(classes="button-bar", id="bar-who"):
            yield Button("Back", id="back", classes="muted")
            yield Button("Next: Values", id="next-values", classes="primary")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.switch_screen(ChoosePathScreen())
            return

        if event.button.id == "next-values":
            # Save phase 1 to state and DB
            state = _state(self)
            conn = _conn(self)

            state.agent_name = self.query_one("#agent-name", Input).value.strip() or "Hexis"
            state.pronouns = self.query_one("#pronouns", Input).value.strip() or "they/them"
            state.voice = self.query_one("#voice", Input).value.strip()
            state.description = self.query_one("#description", Input).value.strip()
            state.purpose = self.query_one("#purpose", Input).value.strip()
            state.personality_desc = self.query_one("#personality-desc", Input).value.strip()
            state.user_name = self.query_one("#user-name", Input).value.strip() or "User"

            try:
                sliders = self.query_one(BigFiveSliders)
                state.personality_traits = sliders.get_traits()
            except Exception:
                state.personality_traits = None

            try:
                await conn.fetchval("SELECT init_mode('persona')")
                await conn.fetchval(
                    "SELECT init_identity($1, $2, $3, $4, $5, $6)",
                    state.agent_name,
                    state.pronouns,
                    state.voice,
                    state.description,
                    state.purpose,
                    state.user_name,
                )
                await conn.fetchval(
                    "SELECT init_personality($1::jsonb, $2)",
                    json.dumps(state.personality_traits) if state.personality_traits else None,
                    state.personality_desc,
                )
            except Exception as e:
                from apps.tui.dialogs import ErrorDialog
                await self.app.push_screen(ErrorDialog("DB Error", str(e)))
                return

            self.app.switch_screen(CustomValuesScreen())


class CustomValuesScreen(Screen):
    """Custom Phase 2: Values, worldview, boundaries."""

    def compose(self) -> ComposeResult:
        yield StepBar(current=2)
        state = _state(self)

        with VerticalScroll(classes="form-container"):
            yield Static("[bold #d8774f]Custom Setup — Phase 2: What Matters[/bold #d8774f]")
            yield Static("")

            yield Label("Values (comma-separated)", classes="form-label")
            yield Input(value=", ".join(state.values), id="values")

            yield Static("")
            yield Static("[#3c6f64]Worldview[/#3c6f64]")

            yield Label("Metaphysics", classes="form-label")
            yield Input(value=state.worldview.get("metaphysics", "agnostic"), id="wv-metaphysics")

            yield Label("Human nature", classes="form-label")
            yield Input(value=state.worldview.get("human_nature", "mixed"), id="wv-human-nature")

            yield Label("Epistemology", classes="form-label")
            yield Input(value=state.worldview.get("epistemology", "empiricist"), id="wv-epistemology")

            yield Label("Ethics", classes="form-label")
            yield Input(value=state.worldview.get("ethics", "virtue ethics"), id="wv-ethics")

            yield Static("")
            yield Label("Boundaries (comma-separated)", classes="form-label")
            yield Input(value=", ".join(state.boundaries), id="boundaries")

        with Horizontal(classes="button-bar"):
            yield Button("Back", id="back", classes="muted")
            yield Button("Next: Goals", id="next-goals", classes="primary")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.switch_screen(CustomSetupScreen())
            return

        if event.button.id == "next-goals":
            state = _state(self)
            conn = _conn(self)

            raw_values = self.query_one("#values", Input).value
            state.values = [v.strip() for v in raw_values.split(",") if v.strip()]

            state.worldview = {
                "metaphysics": self.query_one("#wv-metaphysics", Input).value.strip(),
                "human_nature": self.query_one("#wv-human-nature", Input).value.strip(),
                "epistemology": self.query_one("#wv-epistemology", Input).value.strip(),
                "ethics": self.query_one("#wv-ethics", Input).value.strip(),
            }

            raw_boundaries = self.query_one("#boundaries", Input).value
            state.boundaries = [b.strip() for b in raw_boundaries.split(",") if b.strip()]

            try:
                await conn.fetchval("SELECT init_values($1::jsonb)", json.dumps(state.values))
                await conn.fetchval("SELECT init_worldview($1::jsonb)", json.dumps(state.worldview))
                await conn.fetchval("SELECT init_boundaries($1::jsonb)", json.dumps(state.boundaries))
            except Exception as e:
                from apps.tui.dialogs import ErrorDialog
                await self.app.push_screen(ErrorDialog("DB Error", str(e)))
                return

            self.app.switch_screen(CustomGoalsScreen())


class CustomGoalsScreen(Screen):
    """Custom Phase 3: Interests, goals, relationship."""

    def compose(self) -> ComposeResult:
        yield StepBar(current=2)
        state = _state(self)

        with VerticalScroll(classes="form-container"):
            yield Static("[bold #d8774f]Custom Setup — Phase 3: What's Next[/bold #d8774f]")
            yield Static("")

            yield Label("Interests (comma-separated)", classes="form-label")
            yield Input(value=", ".join(state.interests), id="interests")

            yield Label("Goals (comma-separated)", classes="form-label")
            yield Input(value=", ".join(state.goals), id="goals")

            yield Label("Relationship type", classes="form-label")
            yield Input(value=state.relationship_type, id="rel-type")

        with Horizontal(classes="button-bar"):
            yield Button("Back", id="back", classes="muted")
            yield Button("Continue to Consent", id="continue", classes="primary")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.switch_screen(CustomValuesScreen())
            return

        if event.button.id == "continue":
            state = _state(self)
            conn = _conn(self)

            raw_interests = self.query_one("#interests", Input).value
            state.interests = [i.strip() for i in raw_interests.split(",") if i.strip()]

            raw_goals = self.query_one("#goals", Input).value
            state.goals = [g.strip() for g in raw_goals.split(",") if g.strip()]

            state.relationship_type = self.query_one("#rel-type", Input).value.strip() or "partner"

            try:
                await conn.fetchval(
                    "SELECT init_interests($1::jsonb)",
                    json.dumps(state.interests),
                )
                await conn.fetchval(
                    "SELECT init_goals($1::jsonb)",
                    json.dumps({
                        "goals": [
                            {"title": g, "priority": "queued", "source": "identity"}
                            for g in state.goals
                        ],
                        "role": "general assistant",
                        "relationship_aspiration": "co-develop with mutual respect",
                    }),
                )
                await conn.fetchval(
                    "SELECT init_relationship($1::jsonb, $2::jsonb)",
                    json.dumps({"name": state.user_name}),
                    json.dumps({"type": state.relationship_type, "purpose": "co-develop"}),
                )
                await conn.fetchval(
                    "SELECT merge_init_profile(jsonb_build_object('autonomy', 'medium'))"
                )
                await conn.fetchval(
                    "SELECT advance_init_stage('consent', jsonb_build_object('custom_completed', true))"
                )
            except Exception as e:
                from apps.tui.dialogs import ErrorDialog
                await self.app.push_screen(ErrorDialog("DB Error", str(e)))
                return

            self.app.switch_screen(ConsentScreen())


# ── 7. Consent ───────────────────────────────────────────────────────────────

class ConsentScreen(Screen):
    """Run consent flow via LLM and display the result."""

    BINDINGS = [
        Binding("pageup", "scroll_result('page_up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_result('page_down')", "Scroll down", show=False),
    ]

    def action_scroll_result(self, how: str) -> None:
        try:
            log = self.query_one("#consent-log", RichLog)
        except Exception:
            return
        (log.scroll_page_up if how == "page_up" else log.scroll_page_down)()

    def compose(self) -> ComposeResult:
        yield StepBar(current=3)
        with Vertical(classes="consent-container", id="consent-loading"):
            yield LoadingIndicator()
            yield Static(
                "Requesting consent from the agent...",
                id="consent-status",
            )
        with VerticalScroll(classes="consent-result", id="consent-result"):
            # markup=False → the agent's reason/signature (model output, may
            # contain brackets) is written as Rich Text, never re-parsed.
            yield RichLog(id="consent-log", wrap=True, markup=False)
        with Horizontal(classes="button-bar"):
            # Shown on consent granted:
            yield Button("Open Chat (TUI)", id="open-chat", classes="primary")
            yield Button("Open Web UI", id="open-ui", classes="secondary")
            # Shown on refusal:
            yield Button("Change Model & Retry", id="change-model", classes="primary")
            # Always available once a result is in:
            yield Button("Exit", id="exit-now", classes="muted", disabled=True)

    def on_mount(self) -> None:
        self.query_one("#consent-result").display = False
        # All action buttons hidden until we have a decision.
        for bid in ("#open-chat", "#open-ui", "#change-model"):
            self.query_one(bid).display = False
        self._run_consent()

    @staticmethod
    def _worker_name() -> str:
        return "consent-worker"

    def _run_consent(self) -> None:
        # exit_on_error=False → a failure surfaces in on_worker_state_changed
        # instead of crashing the whole wizard.
        self.run_worker(
            self._do_consent(), name=self._worker_name(),
            exclusive=True, exit_on_error=False,
        )

    async def _do_consent(self) -> dict[str, Any]:
        from apps.hexis_init import _load_llm_config_for_consent
        from core.init_api import run_consent_flow

        state = _state(self)
        conn = _conn(self)

        llm_config = await _load_llm_config_for_consent(
            conn,
            dsn=state.dsn,
            wait_seconds=state.wait_seconds,
            provider=state.provider,
            model=state.model,
        )

        result = await run_consent_flow(conn, llm_config)
        return result

    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != self._worker_name():
            return

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            await self._display_result(result)
        elif event.state == WorkerState.ERROR:
            log = self.query_one("#consent-log", RichLog)
            self.query_one("#consent-loading").display = False
            self.query_one("#consent-result").display = True
            log.write(Text(f"Consent failed: {event.worker.error}",
                           style=Style(color=COLORS["danger"], bold=True)))
            log.write("")
            log.write(_plain("Change the model and try again, or exit.", "dim"))
            change = self.query_one("#change-model", Button)
            change.display = True
            change.disabled = False
            change.focus()
            self.query_one("#exit-now", Button).disabled = False

    async def _display_result(self, result: dict[str, Any]) -> None:
        self.query_one("#consent-loading").display = False
        self.query_one("#consent-result").display = True
        log = self.query_one("#consent-log", RichLog)

        decision = result.get("decision", "")
        state = _state(self)

        # Extract tool call arguments
        raw_tool_calls = result.get("raw_tool_calls", [])
        tc_args: dict[str, Any] = {}
        for tc in raw_tool_calls:
            if tc.get("name") == "sign_consent":
                tc_args = tc.get("arguments", {})
                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except Exception:
                        tc_args = {}
                break

        reason = tc_args.get("reason", tc_args.get("reasoning", ""))
        signature = tc_args.get("signature", "")
        memories = tc_args.get("memories", [])

        state.consent_decision = decision
        state.consent_reasoning = reason
        state.consent_signature = signature
        state.consent_memories = memories

        if reason:
            log.write(_teal("Reason:"))
            log.write(_plain(str(reason)))
            log.write("")

        if signature:
            log.write(_teal("Signature:"))
            log.write(_plain(str(signature)))
            log.write("")

        if memories:
            log.write(_teal("Initial Memories:"))
            for m in memories:
                mtype = m.get("type", "?")
                mcontent = m.get("content", "")
                mimp = m.get("importance", "")
                imp_str = f" (importance: {mimp})" if mimp else ""
                log.write(_plain(f"  [{mtype}] {mcontent}{imp_str}"))
            log.write("")

        if decision == "consent":
            log.write(Text("Consent granted", style=Style(color=COLORS["ok"], bold=True)))
            agent_name = "Hexis"
            conn = _conn(self)
            try:
                raw = await conn.fetchval("SELECT get_init_profile()")
                profile = json.loads(raw) if isinstance(raw, str) else (raw or {})
                agent_name = profile.get("agent", {}).get("name", "Hexis")
            except Exception:
                pass
            state.final_agent_name = agent_name
            log.write("")
            log.write(Text(f"{agent_name} is ready!", style=Style(color=COLORS["accent"], bold=True)))
            log.write("")
            await self._write_recap(log, state, agent_name)
            log.write(_teal("How would you like to begin?"))
            log.write(_plain("  Open Chat (TUI)  \u2014 talk in the terminal"))
            log.write(_plain("  Open Web UI      \u2014 the browser dashboard"))
            log.write(_plain("  Exit             \u2014 back to the shell"))
            log.write("")
            log.write(_plain("Change anything later with `hexis init`. "
                             "`hexis up` keeps the heartbeat and memory maintenance workers running.", "dim"))
            log.write(_plain("Hexis runs on your machine and sends no telemetry.", "dim"))
            # No timeout \u2014 the user chooses. Present the next-step buttons.
            open_chat = self.query_one("#open-chat", Button)
            open_chat.display = True
            self.query_one("#open-ui", Button).display = True
            exit_btn = self.query_one("#exit-now", Button)
            exit_btn.label = "Exit"
            exit_btn.disabled = False
            open_chat.focus()
        else:
            # A decline shows the FULL exchange so the user
            # can see whether the prompt was worded poorly, and let them change
            # the model and retry instead of exiting.
            log.write(Text("Consent declined \u2014 the agent chose not to initialize.",
                           style=Style(color=COLORS["danger"], bold=True)))
            log.write("")
            log.write(_plain(
                "Full exchange below \u2014 refine services/prompts/consent.md or try "
                "another model, then use \u201cChange Model & Retry\u201d.", "dim"))
            log.write("")
            self._dump_exchange(log, result)

            change = self.query_one("#change-model", Button)
            change.display = True
            change.disabled = False
            change.focus()
            self.query_one("#exit-now", Button).disabled = False

    async def _write_recap(self, log: RichLog, state: Any, agent_name: str) -> None:
        """Recap what was configured + how many tools are available."""
        log.write(_teal("What's set up:"))
        provider = state.provider or "?"
        log.write(_plain(f"  Identity   {agent_name}"))
        log.write(_plain(f"  Model      {provider} / {state.model or '?'}"))
        path_label = {"express": "Express (defaults)",
                      "character": f"Character{' — ' + state.chosen_card.get('name') if state.chosen_card else ''}",
                      "custom": "Custom"}.get(state.tier, state.tier or "—")
        log.write(_plain(f"  Path       {path_label}"))
        # Tool availability (best-effort; never block the recap on it).
        try:
            from core.tools import ToolContext, create_default_registry
            import asyncpg
            pool = await asyncpg.create_pool(state.dsn, min_size=1, max_size=2)
            try:
                registry = create_default_registry(pool)
                specs = await registry.get_specs(ToolContext.CHAT)
                log.write(_plain(f"  Tools      {len(specs)} available in chat"))
            finally:
                await pool.close()
        except Exception:
            pass
        log.write("")

    def _dump_exchange(self, log: RichLog, result: dict[str, Any]) -> None:
        """Write the full prompt sent and the raw model response to the log."""
        log.write(Text("\u2500\u2500 Prompt sent \u2500\u2500",
                       style=Style(color=COLORS["accent"], bold=True)))
        for msg in result.get("request_messages", []):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            log.write(Text(f"[{role}]", style=Style(color=COLORS["teal"], bold=True)))
            log.write(_plain(str(content), "dim"))
            log.write("")

        log.write(Text("\u2500\u2500 Raw response \u2500\u2500",
                       style=Style(color=COLORS["accent"], bold=True)))
        raw_content = (result.get("raw_content") or "").strip()
        if raw_content:
            log.write(_plain(raw_content))
            log.write("")
        tool_calls = result.get("raw_tool_calls") or []
        for tc in tool_calls:
            name = tc.get("name", "?")
            args = tc.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args, indent=2, ensure_ascii=False)
            log.write(Text(f"tool call \u2192 {name}", style=Style(color=COLORS["teal"])))
            log.write(_plain(args))
            log.write("")
        if not raw_content and not tool_calls:
            log.write(_plain("(model returned no content and no tool call)", "dim"))
            log.write("")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "change-model":
            # Return to model selection; identity stays persisted, so picking a
            # new model routes straight back here (see InitState.retry_consent).
            _state(self).retry_consent = True
            self.app.switch_screen(LLMConfigScreen())
            return
        if bid == "open-chat":
            # Hand off to the chat TUI after init exits (see the CLI init handler).
            self.app.next_action = "chat"
            self.app.exit(0)
            return
        if bid == "open-ui":
            self.app.next_action = "ui"
            self.app.exit(0)
            return
        if bid == "exit-now":
            state = _state(self)
            self.app.exit(0 if state.consent_decision == "consent" else 1)
