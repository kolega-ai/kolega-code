"""Full-screen settings UI for the CLI TUI."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Select, Static
from textual.widgets.option_list import Option

from kolega_code.agent.tool_backend.search_backends import (
    DEFAULT_BACKEND as DEFAULT_WEB_SEARCH_BACKEND,
    available_backends,
)

from .. import theme
from ..provider_registry import (
    INHERIT_SENTINEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    agent_role_options,
    agent_role_provider_options,
    default_ui_thinking_effort,
    ui_model_options,
    ui_provider_options,
    ui_thinking_effort_options,
)
from . import settings_panel

if TYPE_CHECKING:
    from ..app import KolegaCodeApp


SETTINGS_CATEGORIES = (
    ("Model & Account", "model"),
    ("Agent Models", "agents"),
    ("Tools", "tools"),
    ("MCP Servers", "mcp"),
    ("Appearance", "appearance"),
)


class ConfirmDiscardSettingsScreen(ModalScreen[bool]):
    """Small confirmation shown before dropping a dirty settings draft."""

    AUTO_FOCUS = "#settings_keep_editing"
    BINDINGS = [Binding("escape", "keep_editing", "Keep editing", show=False, priority=True)]

    def compose(self) -> ComposeResult:
        with Vertical(id="settings_discard_dialog", classes="modal-dialog"):
            yield Static("Discard unsaved settings?", id="settings_discard_title")
            yield Static(
                "Provider, model, tool, credential, and theme edits will be lost. "
                "MCP actions that were already saved are not reverted.",
                id="settings_discard_copy",
            )
            with Horizontal(id="settings_discard_actions"):
                yield Static("esc Keep editing", classes="dialog-hint")
                yield Button(
                    "Discard",
                    id="settings_confirm_discard",
                    classes="quiet danger",
                )
                yield Button(
                    "Keep Editing",
                    id="settings_keep_editing",
                    classes="solid-primary",
                )

    def action_keep_editing(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings_keep_editing":
            event.stop()
            self.dismiss(False)
        elif event.button.id == "settings_confirm_discard":
            event.stop()
            self.dismiss(True)


class ConfirmSettingsActionScreen(ModalScreen[bool]):
    """Confirm an immediate, security-sensitive settings action."""

    AUTO_FOCUS = "#settings_action_cancel"
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False, priority=True)]

    def __init__(self, title: str, copy: str, confirm_label: str, *, danger: bool = False) -> None:
        super().__init__()
        self.action_title = title
        self.action_copy = copy
        self.action_confirm_label = confirm_label
        self.danger = danger

    def compose(self) -> ComposeResult:
        with Vertical(id="settings_action_dialog", classes="modal-dialog"):
            yield Static(self.action_title, id="settings_action_title")
            yield Static(self.action_copy, id="settings_action_copy")
            with Horizontal(id="settings_action_buttons"):
                yield Static("esc Cancel", classes="dialog-hint")
                yield Button(
                    self.action_confirm_label,
                    id="settings_action_confirm",
                    classes="quiet danger" if self.danger else "quiet",
                )
                yield Button(
                    "Cancel",
                    id="settings_action_cancel",
                    classes="solid-primary",
                )

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings_action_cancel":
            event.stop()
            self.dismiss(False)
        elif event.button.id == "settings_action_confirm":
            event.stop()
            self.dismiss(True)


class SettingsScreen(ModalScreen[None]):
    """Categorized settings editor with a fixed action footer."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True, priority=True),
    ]

    def __init__(self, owner: "KolegaCodeApp", category: str = "model") -> None:
        super().__init__()
        self.owner = owner
        self.category = category if category in {value for _, value in SETTINGS_CATEGORIES} else "model"
        self._initializing = True
        self._baseline: tuple[tuple[str, Any], ...] = ()
        self._original_theme = owner.settings.active_theme or theme.DEFAULT_THEME_NAME
        self.pending_oauth_tokens = deepcopy(owner.settings.oauth_tokens)
        self._oauth_baseline = deepcopy(self.pending_oauth_tokens)
        self.pending_api_key_removals: set[str] = set()

    def compose(self) -> ComposeResult:
        with Horizontal(id="settings_screen_header"):
            yield Static("Settings", id="settings_screen_title")
            yield Static("esc Close", id="settings_screen_hint")
        with Horizontal(id="settings_screen_body"):
            yield OptionList(
                *(Option(label, id=f"settings_category_{value}") for label, value in SETTINGS_CATEGORIES),
                id="settings_categories",
            )
            with Vertical(id="settings_screen_detail"):
                yield from self._compose_model_page()
                yield from self._compose_agent_page()
                yield from self._compose_tools_page()
                yield from self._compose_mcp_page()
                yield from self._compose_appearance_page()
        with Horizontal(id="settings_screen_footer"):
            yield Static("", id="settings_status")
            with Horizontal(id="settings_screen_actions"):
                yield Button("Close", id="close_settings", classes="quiet")
                yield Button(
                    "Apply Changes",
                    id="save_settings",
                    classes="solid-primary",
                )

    def _compose_model_page(self) -> ComposeResult:
        with VerticalScroll(id="settings_page_model", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_model") as section:
                section.border_title = "Model & Account"
                yield Label("Provider")
                yield Select(
                    ui_provider_options(),
                    id="provider_select",
                    allow_blank=False,
                    value=UI_DEFAULT_PROVIDER,
                )
                yield Label("Model")
                yield Select(
                    ui_model_options(UI_DEFAULT_PROVIDER),
                    id="model_select",
                    allow_blank=False,
                    value=UI_DEFAULT_MODEL,
                )
                yield Label("Thinking effort")
                yield Select(
                    ui_thinking_effort_options(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL),
                    id="thinking_effort_select",
                    allow_blank=True,
                    value=default_ui_thinking_effort(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL),
                )
                yield Label("API key", id="settings_api_key_label")
                yield Input(password=True, id="api_key_input")
                yield Button(
                    "Remove Stored Key",
                    id="settings_remove_api_key",
                    classes="quiet",
                )
                with Horizontal(classes="settings-button-row"):
                    yield Button(
                        "Sign in with ChatGPT",
                        id="settings_chatgpt_login",
                        classes="quiet",
                    )
                    yield Button(
                        "Sign out",
                        id="settings_chatgpt_logout",
                        classes="quiet",
                    )
                yield Button(
                    "Test Connection",
                    id="settings_test_connection",
                    classes="quiet",
                )
                yield Static(
                    "Connection testing sends a tiny, potentially billable model request.",
                    classes="settings-hint",
                )
                yield Static("", id="settings_connection_status")

    def _compose_agent_page(self) -> ComposeResult:
        with VerticalScroll(id="settings_page_agents", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_agent_models") as section:
                section.border_title = "Agent Models"
                yield Static(
                    "Override one role at a time. Default inherits the active model.",
                    classes="settings-hint",
                )
                yield Label("Agent role")
                yield Select(
                    agent_role_options(),
                    id="agent_role_select",
                    allow_blank=False,
                    value="planning",
                )
                for role_label, role_value in agent_role_options():
                    with Vertical(classes="agent-model-group", id=f"agent_model_group_{role_value}"):
                        yield Static(role_label, classes="agent-model-role")
                        with Horizontal(classes="agent-model-field"):
                            yield Label("Provider", classes="agent-model-field-label")
                            yield Select(
                                agent_role_provider_options(),
                                id=f"am_provider_{role_value}",
                                allow_blank=False,
                                value=INHERIT_SENTINEL,
                            )
                        with Horizontal(classes="agent-model-field"):
                            yield Label("Model", classes="agent-model-field-label")
                            yield Select(
                                [],
                                id=f"am_model_{role_value}",
                                allow_blank=True,
                                prompt="—",
                            )
                        with Horizontal(classes="agent-model-field"):
                            yield Label("Effort", classes="agent-model-field-label")
                            yield Select(
                                [],
                                id=f"am_effort_{role_value}",
                                allow_blank=True,
                                prompt="—",
                            )
                        if role_value == "browser":
                            yield Static("", id="am_status_browser", classes="settings-hint")

    def _compose_tools_page(self) -> ComposeResult:
        with VerticalScroll(id="settings_page_tools", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_web_search") as search_section:
                search_section.border_title = "Web Search"
                yield Static(
                    "Choose the backend used by the web_search tool.",
                    classes="settings-hint",
                )
                yield Label("Backend")
                yield Select(
                    available_backends(),
                    id="web_search_backend_select",
                    allow_blank=False,
                    value=DEFAULT_WEB_SEARCH_BACKEND,
                )
                yield Label("API key", id="web_search_api_key_label")
                yield Input(password=True, id="web_search_api_key_input")
                yield Label("SearXNG base URL", id="web_search_base_url_label")
                yield Input(id="web_search_base_url_input", placeholder="https://searxng.example.com")
            with Vertical(classes="settings-section", id="settings_lsp") as lsp_section:
                lsp_section.border_title = "Language Servers (LSP)"
                yield Static(
                    "Auto-detect project languages and run language servers for diagnostics.",
                    classes="settings-hint",
                )
                yield Static("", id="lsp_status")
                yield Label("LSP")
                yield Select(
                    [("Enabled", "true"), ("Disabled", "false")],
                    id="lsp_enabled_select",
                    allow_blank=False,
                    value="true",
                )

    def _compose_mcp_page(self) -> ComposeResult:
        with VerticalScroll(id="settings_page_mcp", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_mcp") as section:
                section.border_title = "MCP Servers"
                yield Static("MCP actions save immediately.", classes="settings-hint")
                yield Static("", id="mcp_status")
                yield Label("Server")
                yield Select(
                    [("New user server", settings_panel.MCP_NEW_SERVER_VALUE)],
                    id="mcp_server_select",
                    allow_blank=False,
                    value=settings_panel.MCP_NEW_SERVER_VALUE,
                )
                yield Static("", id="mcp_source_hint", classes="settings-hint")
                with Horizontal(classes="settings-button-row"):
                    yield Button("Reload", id="mcp_refresh", classes="quiet")
                    yield Button(
                        "Trust Project MCP",
                        id="mcp_trust_project",
                        classes="quiet",
                    )
                yield Label("Display name")
                yield Input(id="mcp_name_input", placeholder="GitHub MCP")
                yield Label("Transport")
                yield Select(
                    settings_panel.MCP_TRANSPORT_OPTIONS,
                    id="mcp_transport_select",
                    allow_blank=False,
                    value="streamable_http",
                )
                yield Label("Enabled")
                yield Select(
                    settings_panel.MCP_ENABLED_OPTIONS,
                    id="mcp_enabled_select",
                    allow_blank=False,
                    value="true",
                )
                yield Label("HTTP URL", id="mcp_url_label")
                yield Input(id="mcp_url_input", placeholder="https://example.com/mcp")
                yield Label("HTTP headers JSON", id="mcp_headers_label")
                yield Input(id="mcp_headers_input", placeholder='{"Authorization":"Bearer ..."}', password=True)
                yield Label("OAuth", id="mcp_oauth_label")
                yield Select(
                    [("Disabled", "false"), ("Enabled", "true")],
                    id="mcp_oauth_select",
                    allow_blank=False,
                    value="false",
                )
                yield Label("Command", id="mcp_command_label")
                yield Input(id="mcp_command_input", placeholder="npx")
                yield Label("Arguments", id="mcp_args_label")
                yield Input(id="mcp_args_input", placeholder="-y @vendor/mcp-server")
                yield Label("Environment JSON", id="mcp_env_label")
                yield Input(id="mcp_env_input", placeholder='{"TOKEN":"..."}', password=True)
                yield Label("Working directory", id="mcp_cwd_label")
                yield Input(id="mcp_cwd_input", placeholder="optional project-relative path")
                with Horizontal(classes="settings-button-row"):
                    yield Button("Save Server", id="mcp_save_server", classes="quiet")
                    yield Button("Verify", id="mcp_verify_server", classes="quiet")
                with Horizontal(classes="settings-button-row"):
                    yield Button("Delete", id="mcp_delete_server", classes="quiet danger")
                    yield Button("Clear OAuth", id="mcp_clear_tokens", classes="quiet danger")

    def _compose_appearance_page(self) -> ComposeResult:
        with VerticalScroll(id="settings_page_appearance", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_appearance") as section:
                section.border_title = "Appearance"
                yield Static("Theme changes preview immediately and are saved when you Apply.", classes="settings-hint")
                yield Label("Theme")
                yield Select(
                    [(name, name) for name in theme.available_themes()],
                    id="theme_select",
                    allow_blank=False,
                    value=theme.DEFAULT_THEME_NAME,
                )

    def on_mount(self) -> None:
        self.owner._settings_screen = self
        self._show_category(self.category)
        self._show_agent_role("planning")
        self.owner._populate_settings_controls()

        def finish_initializing() -> None:
            self._baseline = self._snapshot()
            self._initializing = False
            self._refresh_apply_label()

        self.call_after_refresh(finish_initializing)

    def on_unmount(self) -> None:
        if self.owner._settings_screen is self:
            self.owner._settings_screen = None

    def _show_category(self, category: str) -> None:
        self.category = category
        for index, (_, value) in enumerate(SETTINGS_CATEGORIES):
            try:
                self.query_one(f"#settings_page_{value}").display = value == category
            except Exception:
                pass
            if value == category:
                try:
                    self.query_one("#settings_categories", OptionList).highlighted = index
                except Exception:
                    pass

    def _show_agent_role(self, role: str) -> None:
        for _, value in agent_role_options():
            try:
                self.query_one(f"#agent_model_group_{value}").display = value == role
            except Exception:
                pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "settings_categories":
            return
        event.stop()
        option_id = event.option_id or ""
        self._show_category(option_id.removeprefix("settings_category_"))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "agent_role_select":
            self._show_agent_role(str(event.value))
        if event.select.id == "provider_select" and not self._initializing:
            self.query_one("#api_key_input", Input).value = ""
        if event.select.id in {"provider_select", "model_select", "thinking_effort_select"}:
            self._reset_connection_status()
        self.call_after_refresh(self._refresh_apply_label)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "api_key_input":
            self._reset_connection_status()
            if event.value.strip():
                try:
                    provider = str(self.query_one("#provider_select", Select).value)
                    self.pending_api_key_removals.discard(provider)
                except Exception:
                    pass
        self.call_after_refresh(self._refresh_apply_label)

    def _reset_connection_status(self) -> None:
        if self._initializing:
            return
        try:
            self.query_one("#settings_connection_status", Static).update("")
        except Exception:
            pass

    def _snapshot(self) -> tuple[tuple[str, Any], ...]:
        values: list[tuple[str, Any]] = []
        for widget in self.query("Select, Input"):
            if not isinstance(widget, (Select, Input)):
                continue
            widget_id = widget.id or ""
            if not widget_id or widget_id.startswith("mcp_") or widget_id == "agent_role_select":
                continue
            value = widget.value
            if value is Select.NULL:
                value = None
            values.append((widget_id, value))
        return tuple(sorted(values))

    @property
    def dirty(self) -> bool:
        return not self._initializing and (
            self._snapshot() != self._baseline
            or self.pending_oauth_tokens != self._oauth_baseline
            or bool(self.pending_api_key_removals)
        )

    def mark_clean(self) -> None:
        self.pending_api_key_removals.clear()
        self._baseline = self._snapshot()
        self._oauth_baseline = deepcopy(self.pending_oauth_tokens)
        self._original_theme = self.owner.settings.active_theme or theme.DEFAULT_THEME_NAME
        self._refresh_apply_label()

    def _refresh_apply_label(self) -> None:
        if self._initializing:
            return
        try:
            button = self.query_one("#save_settings", Button)
            button.label = "Apply Changes" if self.dirty else "Applied"
            button.disabled = not self.dirty
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close_settings":
            event.stop()
            self.action_close()
        elif event.button.id in {"mcp_delete_server", "mcp_clear_tokens", "mcp_trust_project"}:
            if self._confirm_immediate_action(event.button.id):
                event.stop()

    def _confirm_immediate_action(self, button_id: str) -> bool:
        if button_id == "mcp_trust_project":
            screen = ConfirmSettingsActionScreen(
                "Trust project MCP configuration?",
                "Trusted project servers may start local commands or connect to remote services. "
                "Only trust repositories you control or have reviewed.",
                "Trust Project",
            )
        else:
            selected = self.query_one("#mcp_server_select", Select).value
            if selected is Select.NULL or selected == settings_panel.MCP_NEW_SERVER_VALUE:
                return False
            server_id = str(selected)
            if button_id == "mcp_delete_server":
                screen = ConfirmSettingsActionScreen(
                    "Delete MCP server?",
                    f"The user MCP server '{server_id}' and its saved verification and OAuth state will be removed.",
                    "Delete Server",
                    danger=True,
                )
            else:
                screen = ConfirmSettingsActionScreen(
                    "Clear MCP OAuth tokens?",
                    f"Stored OAuth credentials for '{server_id}' will be removed. You may need to verify it again.",
                    "Clear Tokens",
                    danger=True,
                )
        self.app.push_screen(
            screen,
            callback=lambda confirmed: self._on_immediate_action_decision(button_id, confirmed),
        )
        return True

    def _on_immediate_action_decision(self, button_id: str, confirmed: bool | None) -> None:
        if confirmed:
            self.run_worker(
                self.owner._handle_mcp_settings_button(button_id),
                name=f"settings-{button_id}",
                exclusive=True,
            )

    def action_close(self) -> None:
        if not self.dirty:
            self._dismiss_settings()
            return
        self.app.push_screen(ConfirmDiscardSettingsScreen(), callback=self._on_discard_decision)

    def _on_discard_decision(self, discard: bool | None) -> None:
        if discard:
            self.owner._apply_theme(self._original_theme)
            self._dismiss_settings()

    def _dismiss_settings(self) -> None:
        self.owner._settings_screen = None
        self.dismiss()
        self.owner._schedule_primary_focus_restore()
