"""Independent first-run onboarding for the CLI TUI."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.chatgpt_oauth import run_login_flow
from kolega_code.cli.config import build_agent_config, key_status
from kolega_code.cli.model_connection import test_model_connection
from kolega_code.cli.provider_registry import (
    UI_DEFAULT_PROVIDER,
    default_model_for_provider,
    default_ui_thinking_effort,
    ui_model_options,
    ui_provider_options,
    ui_thinking_effort_options,
)
from kolega_code.config import ModelProvider

if TYPE_CHECKING:
    from ..app import KolegaCodeApp


ONBOARDING_STEPS = ("welcome", "connection", "model", "ready")


class OnboardingScreen(ModalScreen[None]):
    """Short provider-connection wizard, deliberately separate from Settings."""

    BINDINGS = [Binding("escape", "skip", "Skip for now", show=True, priority=True)]

    def __init__(self, owner: "KolegaCodeApp") -> None:
        super().__init__()
        self.owner = owner
        self.draft = deepcopy(owner.settings)
        self.step_index = 0
        self._test_signature: tuple[str | None, str | None, str | None, str] | None = None

    def compose(self) -> ComposeResult:
        initial_provider = self._initial_provider()
        model_options = ui_model_options(initial_provider)
        valid_models = {value for _, value in model_options}
        initial_model = (
            self.draft.active_model
            if self.draft.active_model in valid_models
            else default_model_for_provider(ModelProvider(initial_provider))
        )
        effort_options = ui_thinking_effort_options(initial_provider, initial_model)
        valid_efforts = {value for _, value in effort_options}
        initial_effort = (
            self.draft.active_thinking_effort
            if self.draft.active_thinking_effort in valid_efforts
            else default_ui_thinking_effort(initial_provider, initial_model)
        )
        with Vertical(id="onboarding_dialog"):
            yield Static("Welcome to Kolega Code", id="onboarding_title")
            yield Static("Step 1 of 4", id="onboarding_progress")
            with Vertical(id="onboarding_pages"):
                with Vertical(id="onboarding_step_welcome", classes="onboarding-step"):
                    yield Static(
                        "Connect a model to start coding. Onboarding only handles the initial "
                        "connection; everything else remains available in Settings.",
                    )
                    yield Static(
                        "API keys and sign-in tokens are stored locally with restrictive file permissions.",
                        classes="onboarding-hint",
                    )
                with Vertical(id="onboarding_step_connection", classes="onboarding-step"):
                    yield Label("How would you like to connect?")
                    yield Select(
                        [
                            ("Sign in with ChatGPT", "chatgpt"),
                            ("Use an API key", "api"),
                        ],
                        id="onboarding_auth_method",
                        allow_blank=False,
                        value=self._initial_auth_method(),
                    )
                    with Vertical(id="onboarding_chatgpt_panel"):
                        yield Static(
                            "Use models available through your ChatGPT subscription.",
                            classes="onboarding-hint",
                        )
                        yield Button("Sign in with ChatGPT", variant="primary", id="onboarding_chatgpt_login")
                        yield Static("", id="onboarding_chatgpt_status")
                    with Vertical(id="onboarding_api_panel"):
                        yield Label("Provider")
                        yield Select(
                            self._api_provider_options(),
                            id="onboarding_provider",
                            allow_blank=False,
                            value=self._initial_api_provider(),
                        )
                        yield Label("API key")
                        yield Input(password=True, id="onboarding_api_key")
                        yield Static("", id="onboarding_key_status", classes="onboarding-hint")
                with Vertical(id="onboarding_step_model", classes="onboarding-step"):
                    yield Static("Choose the model used by default for this session and future sessions.")
                    yield Label("Model")
                    yield Select(
                        model_options,
                        id="onboarding_model",
                        allow_blank=False,
                        value=initial_model,
                    )
                    yield Label("Thinking effort")
                    yield Select(
                        effort_options,
                        id="onboarding_effort",
                        allow_blank=True,
                        value=initial_effort,
                    )
                with Vertical(id="onboarding_step_ready", classes="onboarding-step"):
                    yield Static("Ready to start", id="onboarding_ready_title")
                    yield Static("", id="onboarding_summary")
                    yield Button("Test Connection", id="onboarding_test_connection")
                    yield Static(
                        "Optional: sends only “Reply with OK.” as a tiny, potentially billable request.",
                        classes="onboarding-hint",
                    )
                    yield Static("", id="onboarding_test_status")
            yield Static("", id="onboarding_status")
            with Horizontal(id="onboarding_actions"):
                yield Button("Skip for now", id="onboarding_skip")
                yield Button("Back", id="onboarding_back")
                yield Button("Continue", variant="primary", id="onboarding_next")

    def on_mount(self) -> None:
        self.owner._onboarding_screen = self
        self._show_step(0)
        self._show_auth_method(self._initial_auth_method())
        self._update_api_key_status()
        if self.draft.has_oauth_token(chatgpt_constants.PROVIDER_KEY):
            self.query_one("#onboarding_chatgpt_login", Button).label = "Sign in again"
            self.query_one("#onboarding_chatgpt_status", Static).update(
                f"Credential status: {key_status(chatgpt_constants.PROVIDER_KEY, self.owner.project_path, self.draft)}"
            )
        startup_error = getattr(self.owner, "startup_config_error", None)
        if startup_error:
            self._set_status(str(startup_error))

    def on_unmount(self) -> None:
        if self.owner._onboarding_screen is self:
            self.owner._onboarding_screen = None

    def _api_provider_options(self) -> list[tuple[str, str]]:
        return [(label, value) for label, value in ui_provider_options() if value != chatgpt_constants.PROVIDER_KEY]

    def _initial_auth_method(self) -> str:
        return "chatgpt" if self.draft.active_provider == chatgpt_constants.PROVIDER_KEY else "api"

    def _initial_api_provider(self) -> str:
        supported = {value for _, value in self._api_provider_options()}
        return self.draft.active_provider if self.draft.active_provider in supported else UI_DEFAULT_PROVIDER

    def _initial_provider(self) -> str:
        if self._initial_auth_method() == "chatgpt":
            return chatgpt_constants.PROVIDER_KEY
        return self._initial_api_provider()

    def _show_step(self, index: int) -> None:
        self.step_index = max(0, min(index, len(ONBOARDING_STEPS) - 1))
        current = ONBOARDING_STEPS[self.step_index]
        for step in ONBOARDING_STEPS:
            self.query_one(f"#onboarding_step_{step}").display = step == current
        self.query_one("#onboarding_progress", Static).update(f"Step {self.step_index + 1} of {len(ONBOARDING_STEPS)}")
        back = self.query_one("#onboarding_back", Button)
        back.disabled = self.step_index == 0
        next_button = self.query_one("#onboarding_next", Button)
        next_button.label = "Finish" if current == "ready" else "Continue"
        if current == "ready":
            self._refresh_summary()

    def _show_auth_method(self, method: str) -> None:
        self.query_one("#onboarding_chatgpt_panel").display = method == "chatgpt"
        self.query_one("#onboarding_api_panel").display = method == "api"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "onboarding_auth_method":
            self._show_auth_method(str(event.value))
            self._reset_test_status()
        elif event.select.id == "onboarding_provider":
            self.query_one("#onboarding_api_key", Input).value = ""
            self._update_api_key_status()
            self._reset_test_status()
        elif event.select.id == "onboarding_model":
            self._populate_efforts(str(event.value))
            self._reset_test_status()
        elif event.select.id == "onboarding_effort":
            self._reset_test_status()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "onboarding_api_key":
            self._reset_test_status()

    def _update_api_key_status(self) -> None:
        try:
            provider = str(self.query_one("#onboarding_provider", Select).value)
        except Exception:
            return
        status = key_status(provider, self.owner.project_path, self.draft)
        widget = self.query_one("#onboarding_key_status", Static)
        widget.update(f"Credential status: {status}")
        key_input = self.query_one("#onboarding_api_key", Input)
        key_input.placeholder = "Existing credential will be kept if blank" if status != "missing" else "Paste API key"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "onboarding_skip":
            event.stop()
            self.action_skip()
        elif button_id == "onboarding_back":
            event.stop()
            self._show_step(self.step_index - 1)
        elif button_id == "onboarding_next":
            event.stop()
            await self._continue()
        elif button_id == "onboarding_chatgpt_login":
            event.stop()
            await self._login_chatgpt()
        elif button_id == "onboarding_test_connection":
            event.stop()
            await self._test_connection()

    async def _continue(self) -> None:
        if self.step_index == 0:
            self._set_status("")
            self._show_step(1)
            return
        if self.step_index == 1:
            if not self._collect_connection():
                return
            self._populate_models()
            self._set_status("")
            self._show_step(2)
            return
        if self.step_index == 2:
            if not self._collect_model():
                return
            self._set_status("")
            self._show_step(3)
            return
        await self._finish()

    def _collect_connection(self) -> bool:
        method = str(self.query_one("#onboarding_auth_method", Select).value)
        if method == "chatgpt":
            if not self.draft.has_oauth_token(chatgpt_constants.PROVIDER_KEY):
                self._set_status("Sign in with ChatGPT before continuing.")
                return False
            self.draft.active_provider = chatgpt_constants.PROVIDER_KEY
            return True

        provider = str(self.query_one("#onboarding_provider", Select).value)
        key = self.query_one("#onboarding_api_key", Input).value.strip()
        if key:
            self.draft.set_api_key(provider, key)
        if key_status(provider, self.owner.project_path, self.draft) == "missing":
            self._set_status("Enter an API key or configure the provider's environment variable.")
            return False
        self.draft.active_provider = provider
        return True

    def _populate_models(self) -> None:
        provider = self.draft.active_provider or UI_DEFAULT_PROVIDER
        options = ui_model_options(provider)
        model_select = self.query_one("#onboarding_model", Select)
        model_select.set_options(options)
        valid_models = {value for _, value in options}
        saved = self.draft.active_model if self.draft.active_model in valid_models else None
        default = saved or default_model_for_provider(ModelProvider(provider))
        model_select.value = default
        self._populate_efforts(default)

    def _populate_efforts(self, model: str) -> None:
        provider = self.draft.active_provider or UI_DEFAULT_PROVIDER
        effort_select = self.query_one("#onboarding_effort", Select)
        options = ui_thinking_effort_options(provider, model)
        effort_select.set_options(options)
        valid = {value for _, value in options}
        saved = self.draft.active_thinking_effort if self.draft.active_thinking_effort in valid else None
        selected = saved or default_ui_thinking_effort(provider, model)
        if selected is not None:
            effort_select.value = selected

    def _collect_model(self) -> bool:
        model_value = self.query_one("#onboarding_model", Select).value
        if model_value is Select.NULL:
            self._set_status("Choose a model before continuing.")
            return False
        self.draft.active_model = str(model_value)
        effort_value = self.query_one("#onboarding_effort", Select).value
        self.draft.active_thinking_effort = None if effort_value is Select.NULL else str(effort_value)
        return True

    def _refresh_summary(self) -> None:
        provider = self.draft.active_provider or "not configured"
        model = self.draft.active_model or "not configured"
        effort = self.draft.active_thinking_effort or "model default"
        auth = key_status(provider, self.owner.project_path, self.draft) if self.draft.active_provider else "missing"
        self.query_one("#onboarding_summary", Static).update(
            f"Provider: {provider}\nModel: {model}\nThinking effort: {effort}\nCredential: {auth}"
        )

    async def _login_chatgpt(self) -> None:
        button = self.query_one("#onboarding_chatgpt_login", Button)
        status = self.query_one("#onboarding_chatgpt_status", Static)
        button.disabled = True
        status.update("Opening your browser to sign in…")

        def on_url(url: str) -> None:
            status.update(f"If the browser did not open, visit:\n{url}")

        try:
            tokens = await run_login_flow(on_url=on_url)
        except Exception as exc:
            status.update(f"Sign-in failed: {exc}")
            button.disabled = False
            return
        self.draft.set_oauth_token(chatgpt_constants.PROVIDER_KEY, tokens.model_dump(mode="json"))
        self.draft.active_provider = chatgpt_constants.PROVIDER_KEY
        status.update(f"Signed in as {tokens.email or 'your ChatGPT account'}.")
        button.label = "Sign in again"
        button.disabled = False
        self._reset_test_status()

    def _candidate_config(self):
        return build_agent_config(
            self.owner.project_path,
            self.owner.overrides,
            settings=self.draft,
            settings_store=None,
        )

    async def _test_connection(self) -> None:
        if not self._collect_model():
            return
        status = self.query_one("#onboarding_test_status", Static)
        button = self.query_one("#onboarding_test_connection", Button)
        try:
            config = self._candidate_config()
        except Exception as exc:
            status.update(f"Configuration is incomplete: {exc}")
            return
        button.disabled = True
        status.update("Testing connection…")
        result = await test_model_connection(config)
        status.update(result.message)
        button.disabled = False
        if result.ok:
            self._test_signature = self._current_test_signature()

    async def _finish(self) -> None:
        if not self._collect_model():
            return
        ok, error = await self.owner._apply_settings_candidate(deepcopy(self.draft), rebuild=True)
        if not ok:
            self._set_status(f"Configuration is incomplete: {error}")
            return
        self.owner._onboarding_skipped = False
        self.owner._onboarding_screen = None
        self.dismiss()
        self.owner._schedule_primary_focus_restore()

    def _current_test_signature(self) -> tuple[str | None, str | None, str | None, str]:
        method = str(self.query_one("#onboarding_auth_method", Select).value)
        return (
            self.draft.active_provider,
            self.draft.active_model,
            self.draft.active_thinking_effort,
            method,
        )

    def _reset_test_status(self) -> None:
        if self._test_signature is None:
            return
        if self._test_signature != self._current_test_signature():
            self._test_signature = None
            try:
                self.query_one("#onboarding_test_status", Static).update("")
            except Exception:
                pass

    def _set_status(self, text: str) -> None:
        self.query_one("#onboarding_status", Static).update(text)

    def action_skip(self) -> None:
        self.owner._onboarding_skipped = True
        self.owner._onboarding_screen = None
        self.dismiss()
        self.owner._schedule_primary_focus_restore()
