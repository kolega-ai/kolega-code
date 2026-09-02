"""Full-screen settings UI for the CLI TUI."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

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
from kolega_code.gateway.stt import DEFAULT_STT_PROVIDER, available_stt_providers

from .. import messages, theme
from ..provider_registry import (
    INHERIT_SENTINEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    agent_role_options,
    agent_role_provider_options,
    default_ui_thinking_effort,
    model_slot_options,
    ui_model_options,
    ui_provider_options,
    ui_thinking_effort_options,
)
from . import custom_model, settings_panel

if TYPE_CHECKING:
    from ..app import KolegaCodeApp


SETTINGS_CATEGORIES = (
    # Credentials first, matching the order onboarding already walks users through
    # (connection, then model). The "model" value is load-bearing: /model and
    # action_open_settings(category=...) both key off it, so only the label changed.
    ("Providers", "providers"),
    ("Custom Endpoints", "endpoints"),
    ("Models", "model"),
    ("Agent Models", "agents"),
    ("Tools", "tools"),
    ("MCP Servers", "mcp"),
    ("Memory", "memory"),
    ("Appearance", "appearance"),
    ("Gateway", "gateway"),
)


class ConfirmDiscardSettingsScreen(ModalScreen[bool]):
    """Small confirmation shown before dropping a dirty settings draft."""

    AUTO_FOCUS = "#settings_keep_editing"
    BINDINGS = [Binding("escape", "keep_editing", "Keep editing", show=False, priority=True)]

    def compose(self) -> ComposeResult:
        with Vertical(id="settings_discard_dialog", classes="modal-dialog"):
            yield Static("Discard unsaved settings?", id="settings_discard_title")
            yield Static(
                "Provider, model, tool, credential, memory, and theme edits will be lost. "
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
        self.pending_custom_endpoints = deepcopy(owner.settings.custom_endpoints)
        self._endpoints_baseline = deepcopy(self.pending_custom_endpoints)
        self.pending_api_key_removals: set[str] = set()
        # Keys typed on the Providers page, keyed by provider. The page lets several
        # providers be edited before Apply, so staging cannot live in the Input alone —
        # its value follows the highlighted row.
        self.pending_api_keys: dict[str, str] = {}
        # Whether the Gateway page's "Remove token" was clicked before Apply.
        self.pending_gateway_token_removal = False
        # Which provider the key Input is currently showing. Staging targets this rather
        # than the highlighted row: it is updated before the Input is rewritten, so the
        # Changed event that rewrite posts can never be filed against the wrong provider.
        self.credential_provider: Optional[str] = None

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
                yield from self._compose_providers_page()
                yield from self._compose_endpoints_page()
                yield from self._compose_model_page()
                yield from self._compose_agent_page()
                yield from self._compose_tools_page()
                yield from self._compose_mcp_page()
                yield from self._compose_memory_page()
                yield from self._compose_appearance_page()
                yield from self._compose_gateway_page()
        with Horizontal(id="settings_screen_footer"):
            yield Static("", id="settings_status")
            with Horizontal(id="settings_screen_actions"):
                yield Button("Close", id="close_settings", classes="quiet")
                yield Button(
                    "Apply Changes",
                    id="save_settings",
                    classes="solid-primary",
                )

    def _compose_providers_page(self) -> ComposeResult:
        with VerticalScroll(id="settings_page_providers", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_providers") as section:
                section.border_title = "Providers"
                yield Static(messages.PROVIDERS_HINT, classes="settings-hint")
                yield OptionList(id="provider_list")
            with Vertical(classes="settings-section", id="settings_provider_credential") as credential:
                # Retitled to the highlighted provider by _update_provider_credential_controls.
                credential.border_title = "Credential"
                yield Label("API key", id="provider_api_key_label")
                yield Input(password=True, id="provider_api_key_input")
                yield Button(
                    "Remove Stored Key",
                    id="provider_remove_api_key",
                    classes="quiet",
                )
                with Horizontal(classes="settings-button-row", id="provider_chatgpt_row"):
                    yield Button(
                        "Sign in with ChatGPT",
                        id="provider_chatgpt_login",
                        classes="quiet",
                    )
                    yield Button(
                        "Sign out",
                        id="provider_chatgpt_logout",
                        classes="quiet",
                    )
                yield Button(
                    "Test Connection",
                    id="provider_test_connection",
                    classes="quiet",
                )
                yield Static(
                    "Connection testing sends a tiny, potentially billable model request.",
                    classes="settings-hint",
                )
                endpoint_key_hint = Static(
                    messages.ENDPOINT_KEY_EDITED_ON_PAGE, id="provider_endpoint_key_hint", classes="settings-hint"
                )
                endpoint_key_hint.display = False
                yield endpoint_key_hint
                yield Static("", id="provider_credential_status")

    def _compose_endpoints_page(self) -> ComposeResult:
        with VerticalScroll(id="settings_page_endpoints", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_endpoints") as section:
                section.border_title = "Custom Endpoints"
                yield Static(messages.ENDPOINTS_HINT, classes="settings-hint")
                yield Static("", id="endpoint_status", classes="settings-hint")
                yield Label("Endpoint")
                yield Select(
                    [(settings_panel.ENDPOINT_NEW_VALUE_LABEL, settings_panel.ENDPOINT_NEW_VALUE)],
                    id="endpoint_select",
                    allow_blank=False,
                    value=settings_panel.ENDPOINT_NEW_VALUE,
                )
                yield Label("Endpoint id")
                yield Input(id="ep_id_input", placeholder="lmstudio")
                yield Label("Label")
                yield Input(id="ep_label_input", placeholder="LM Studio")
                yield Label("API style")
                yield Select(
                    [
                        ("OpenAI Chat Completions", "openai_chat"),
                        ("OpenAI Responses", "openai_responses"),
                        ("Anthropic Messages", "anthropic"),
                    ],
                    id="ep_style_select",
                    allow_blank=False,
                    value="openai_chat",
                )
                yield Label("Base URL")
                yield Input(id="ep_base_url_input", placeholder="http://localhost:1234/v1")
                yield Label("API key (optional)")
                yield Input(password=True, id="ep_api_key_input", placeholder="Leave blank for keyless servers")
                yield Label("Default model (optional)")
                yield Input(id="ep_default_model_input", placeholder="qwen2.5-coder-7b-instruct")
                yield Label("Context length")
                yield Input(id="ep_context_input", placeholder="32768")
                yield Label("Max output tokens")
                yield Input(id="ep_max_output_input", placeholder="8192")
                yield Label("Temperature", id="ep_temperature_label")
                yield Input(id="ep_temperature_input", placeholder="1.0 (leave blank for default)")
                yield Label("Vision")
                yield Select(
                    [("No", "false"), ("Yes", "true")],
                    id="ep_vision_select",
                    allow_blank=False,
                    value="false",
                )
                yield Label("Thinking")
                yield Select(
                    [("Off", "")] + [(label, mode) for label, mode in settings_panel.ENDPOINT_THINKING_OPTIONS],
                    id="ep_thinking_mode_select",
                    allow_blank=False,
                    value="",
                )
                budgets_label = Label("Thinking budgets (effort=tokens)", id="ep_thinking_budgets_label")
                budgets_label.display = False
                yield budgets_label
                budgets_input = Input(id="ep_thinking_budgets_input", placeholder="low=2048, medium=8192, high=16384")
                budgets_input.display = False
                yield budgets_input
                reasoning_label = Label("Reasoning replay", id="ep_reasoning_label")
                yield reasoning_label
                yield Select(
                    [
                        ("Auto (detect emitted field)", "auto"),
                        ("reasoning_content", "reasoning_content"),
                        ("reasoning", "reasoning"),
                        ("Off (visible text)", "off"),
                    ],
                    id="ep_reasoning_select",
                    allow_blank=False,
                    value="auto",
                )
                yield Static(messages.ENDPOINT_APPLY_REQUIRED, classes="settings-hint")
                with Horizontal(classes="settings-button-row"):
                    yield Button("Save Endpoint", id="ep_save", classes="quiet")
                    yield Button("Delete", id="ep_delete", classes="quiet danger")

    def _compose_model_page(self) -> ComposeResult:
        with VerticalScroll(id="settings_page_model", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_model") as section:
                section.border_title = "Models"
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
                custom_model_input = Input(id="model_custom_input", placeholder=custom_model.CUSTOM_MODEL_PLACEHOLDER)
                custom_model_input.display = False
                yield custom_model_input
                yield Label("Thinking effort")
                yield Select(
                    ui_thinking_effort_options(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL),
                    id="thinking_effort_select",
                    allow_blank=True,
                    value=default_ui_thinking_effort(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL),
                )
                yield Static(messages.MODEL_CREDENTIAL_POINTER, classes="settings-hint")
            with Vertical(classes="settings-section", id="settings_model_slots") as slots_section:
                slots_section.border_title = "Model Slots"
                yield Static(messages.MODEL_SLOT_HINT, classes="settings-hint")
                for slot_label, slot_value in model_slot_options():
                    with Vertical(classes="agent-model-group", id=f"slot_group_{slot_value}"):
                        yield Static(slot_label, classes="agent-model-role")
                        with Horizontal(classes="agent-model-field"):
                            yield Label("Provider", classes="agent-model-field-label")
                            yield Select(
                                agent_role_provider_options(),
                                id=f"slot_provider_{slot_value}",
                                allow_blank=False,
                                value=INHERIT_SENTINEL,
                            )
                        with Horizontal(classes="agent-model-field"):
                            yield Label("Model", classes="agent-model-field-label")
                            yield Select(
                                [],
                                id=f"slot_model_{slot_value}",
                                allow_blank=True,
                                prompt="—",
                            )
                        row_custom_model = Input(
                            id=f"slot_custom_model_{slot_value}",
                            placeholder=custom_model.CUSTOM_MODEL_PLACEHOLDER,
                        )
                        row_custom_model.display = False
                        yield row_custom_model
                        yield Static("", id=f"slot_hint_{slot_value}", classes="settings-hint")

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
                        row_custom_model = Input(
                            id=f"am_custom_model_{role_value}",
                            placeholder=custom_model.CUSTOM_MODEL_PLACEHOLDER,
                        )
                        row_custom_model.display = False
                        yield row_custom_model
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
                    "Mode picks who searches: hosted runs the provider's server-side web_search "
                    "tool (Responses-API models); client uses the local web_search/web_fetch tools.",
                    classes="settings-hint",
                )
                yield Label("Mode")
                yield Select(
                    [
                        ("Auto (hosted when the model supports it)", "auto"),
                        ("Hosted (server-side)", "hosted"),
                        ("Client tools", "client"),
                        ("Off (no web tools)", "off"),
                    ],
                    id="web_search_mode_select",
                    allow_blank=False,
                    value="auto",
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
            with Vertical(classes="settings-section", id="settings_stt") as stt_section:
                stt_section.border_title = "Voice transcription"
                yield Static(
                    "Transcribe gateway voice notes with a remote provider: Groq's hosted "
                    "whisper-large-v3-turbo, which reuses the Groq API key from the Providers page.",
                    classes="settings-hint",
                )
                yield Label("Speech-to-text")
                yield Select(
                    [("Enabled", "true"), ("Disabled", "false")],
                    id="stt_enabled_select",
                    allow_blank=False,
                    value="false",
                )
                yield Label("Provider")
                yield Select(
                    available_stt_providers(),
                    id="stt_provider_select",
                    allow_blank=False,
                    value=DEFAULT_STT_PROVIDER,
                )
                yield Label("Model")
                yield Input(id="stt_model_input", placeholder="whisper-large-v3-turbo")
                yield Static("", id="stt_key_status", classes="settings-hint")
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
            with Vertical(classes="settings-section", id="settings_subagents") as subagents_section:
                subagents_section.border_title = "Sub-agents"
                yield Static(
                    "Let the agent dispatch focused sub-agents (dispatch_agent) for parallel or isolated work.",
                    classes="settings-hint",
                )
                yield Static("", id="subagents_status")
                yield Label("Sub-agents")
                yield Select(
                    [("Enabled", "true"), ("Disabled", "false")],
                    id="subagents_enabled_select",
                    allow_blank=False,
                    value="true",
                )
            with Vertical(classes="settings-section", id="settings_skills") as skills_section:
                skills_section.border_title = "Agent Skills"
                yield Static(
                    "Discover Agent Skills and expose their catalog and activation tool to the agent.",
                    classes="settings-hint",
                )
                yield Static("", id="skills_status")
                yield Label("Agent Skills")
                yield Select(
                    [("Enabled", "true"), ("Disabled", "false")],
                    id="skills_enabled_select",
                    allow_blank=False,
                    value="true",
                )
            with Vertical(classes="settings-section", id="settings_compression") as compression_section:
                compression_section.border_title = "Context Compression"
                yield Static(
                    "Summarize older conversation history once context-window usage crosses this "
                    "threshold. 100% disables proactive compression; over-limit recovery and explicit "
                    "context caps remain enforced; /compress stays manual.",
                    classes="settings-hint",
                )
                yield Static("", id="compression_status")
                yield Label("Threshold")
                yield Select(
                    settings_panel.compression_threshold_options(),
                    id="compression_threshold_select",
                    allow_blank=False,
                    value=settings_panel.COMPRESSION_THRESHOLD_DEFAULT_VALUE,
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

    def _compose_gateway_page(self) -> ComposeResult:
        with VerticalScroll(id="settings_page_gateway", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_gateway_telegram") as telegram_section:
                telegram_section.border_title = "Telegram"
                yield Static(
                    "The messaging gateway's bot token from @BotFather. Applies to gateway runs.",
                    classes="settings-hint",
                )
                yield Static("", id="gateway_token_status")
                yield Label("Bot token")
                yield Input(password=True, id="gateway_token_input", placeholder="Saved — type to replace")
                yield Button("Remove token", id="gateway_token_remove", classes="quiet")
            with Vertical(classes="settings-section", id="settings_gateway_access") as access_section:
                access_section.border_title = "Access"
                yield Label("Allowed users (comma-separated Telegram ids; empty = anyone)")
                yield Input(id="gateway_allowed_users_input", placeholder="123456789, 987654321")
                yield Label("Pairing for unknown senders")
                yield Select(
                    [("Enabled", "true"), ("Disabled", "false")],
                    id="gateway_pairing_select",
                    allow_blank=False,
                    value="false",
                )
                yield Label("Permission mode")
                yield Select(
                    [("Ask via buttons", "ask"), ("Auto-approve", "auto")],
                    id="gateway_permission_select",
                    allow_blank=False,
                    value="ask",
                )
            with Vertical(classes="settings-section", id="settings_gateway_runtime") as runtime_section:
                runtime_section.border_title = "Runtime"
                yield Label("Adapter")
                yield Select(
                    [("Telegram", "telegram"), ("Echo (test harness)", "echo")],
                    id="gateway_adapter_select",
                    allow_blank=False,
                    value="echo",
                )
                yield Label("Project (workspace directory)")
                yield Input(id="gateway_project_input", placeholder="~/kolega-code-workspace")

    def _compose_memory_page(self) -> ComposeResult:
        backend_options = [(backend_id, backend_id) for backend_id in self.owner.memory_manager.registry.backend_ids]
        with VerticalScroll(id="settings_page_memory", classes="settings-page"):
            with Vertical(classes="settings-section", id="settings_memory") as section:
                section.border_title = "Private Project Memory"
                yield Static(
                    "Memory is private local state shared by linked worktrees. "
                    "Enabled and backend changes take effect with Apply Changes.",
                    classes="settings-hint",
                )
                yield Static("Loading project memory…", id="memory_settings_status")
                yield Label("Enabled")
                yield Select(
                    [("On", "true"), ("Off", "false")],
                    id="memory_enabled_select",
                    allow_blank=False,
                    value="true",
                )
                yield Label("Backend")
                yield Select(
                    backend_options,
                    id="memory_backend_select",
                    allow_blank=False,
                    value=backend_options[0][1] if backend_options else Select.NULL,
                )
                yield Label("Private storage")
                yield Static("", id="memory_settings_path", classes="settings-hint")
                with Horizontal(classes="settings-button-row"):
                    yield Button("Browse Memory", id="memory_settings_browse", classes="quiet")
                    yield Button(
                        "Inspect Disabled Bank",
                        id="memory_settings_inspect",
                        classes="quiet",
                    )

    def on_mount(self) -> None:
        self.owner._settings_screen = self
        self._show_category(self.category)
        self._show_agent_role("planning")
        self.owner._populate_settings_controls()
        self.run_worker(
            self._refresh_memory_controls(),
            name="settings-memory-status",
            group="settings-memory",
            exclusive=True,
        )

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
        if event.option_list.id == "settings_categories":
            event.stop()
            self._show_category((event.option_id or "").removeprefix("settings_category_"))

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "provider_list":
            return
        event.stop()
        self.owner._switch_provider_credential((event.option_id or "").removeprefix("provider_row_"))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "agent_role_select":
            self._show_agent_role(str(event.value))
        self.call_after_refresh(self._refresh_apply_label)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "provider_api_key_input":
            self._reset_connection_status()
            self.owner._commit_visible_api_key()
        self.call_after_refresh(self._refresh_apply_label)

    def _reset_connection_status(self) -> None:
        if self._initializing:
            return
        try:
            self.query_one("#provider_credential_status", Static).update("")
        except Exception:
            pass

    def _snapshot(self) -> tuple[tuple[str, Any], ...]:
        values: list[tuple[str, Any]] = []
        for widget in self.query("Select, Input"):
            if not isinstance(widget, (Select, Input)):
                continue
            widget_id = widget.id or ""
            # provider_api_key_input follows the highlighted provider rather than holding
            # one value, so its content is tracked in pending_api_keys, not the snapshot.
            if (
                not widget_id
                or widget_id.startswith("mcp_")
                or widget_id in {"agent_role_select", "provider_api_key_input"}
            ):
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
            or self.pending_custom_endpoints != self._endpoints_baseline
            or bool(self.pending_api_key_removals)
            or bool(self.pending_api_keys)
        )

    def mark_clean(self, *, preserve_memory_draft: bool = False) -> None:
        self.pending_api_key_removals.clear()
        self.pending_api_keys.clear()
        baseline = dict(self._snapshot())
        if preserve_memory_draft:
            previous = dict(self._baseline)
            for widget_id in ("memory_enabled_select", "memory_backend_select"):
                baseline[widget_id] = previous.get(widget_id)
        self._baseline = tuple(sorted(baseline.items()))
        self._oauth_baseline = deepcopy(self.pending_oauth_tokens)
        self._endpoints_baseline = deepcopy(self.pending_custom_endpoints)
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
        elif event.button.id == "memory_settings_browse":
            event.stop()
            self.owner.action_open_memory()
        elif event.button.id == "memory_settings_inspect":
            event.stop()
            self.owner.action_open_memory(inspect_disabled=True)
        elif event.button.id == "gateway_token_remove":
            event.stop()
            self.pending_gateway_token_removal = True
            self.query_one("#gateway_token_input", Input).value = ""
            self.query_one("#gateway_token_status", Static).update("Token removal staged — apply to save")
        elif event.button.id in {"ep_save", "ep_delete"}:
            event.stop()
            self.owner._handle_endpoint_settings_button(event.button.id)
        elif event.button.id in {"mcp_delete_server", "mcp_clear_tokens", "mcp_trust_project"}:
            if self._confirm_immediate_action(event.button.id):
                event.stop()

    async def _refresh_memory_controls(self, *, update_draft: bool = True) -> None:
        try:
            status = await asyncio.to_thread(self.owner.memory_manager.status)
        except Exception as error:
            self.query_one("#memory_settings_status", Static).update(f"Memory unavailable: {error}")
            return
        if update_draft:
            self.query_one("#memory_enabled_select", Select).value = "true" if status.enabled else "false"
            backend_select = self.query_one("#memory_backend_select", Select)
            if status.backend_id in self.owner.memory_manager.registry.backend_ids:
                backend_select.value = status.backend_id
        backend = status.backend
        detail = (
            f"{backend.entry_count} entries · {backend.total_bytes:,} bytes"
            if backend is not None
            else "Backend unavailable"
        )
        self.query_one("#memory_settings_status", Static).update(
            f"{status.backend_id} · {'on' if status.enabled else 'off'} · {detail}"
        )
        self.query_one("#memory_settings_path", Static).update(
            (backend.private_path if backend is not None else None) or str(self.owner.memory_manager.memory_dir)
        )
        self.query_one("#memory_settings_inspect", Button).disabled = status.enabled
        if update_draft:
            self.call_after_refresh(self._rebaseline_memory_controls)

    def _rebaseline_memory_controls(self) -> None:
        """Committed manifest state is the draft baseline for the memory selects."""
        if self._initializing:
            return
        entries = dict(self._baseline)
        for widget_id in ("memory_enabled_select", "memory_backend_select"):
            value = self.query_one(f"#{widget_id}", Select).value
            entries[widget_id] = None if value is Select.NULL else value
        self._baseline = tuple(sorted(entries.items()))
        self._refresh_apply_label()

    async def apply_memory_draft(self) -> bool:
        """Commit staged memory settings as part of Apply Changes."""
        manager = self.owner.memory_manager
        enabled_value = self.query_one("#memory_enabled_select", Select).value
        backend_value = self.query_one("#memory_backend_select", Select).value
        try:
            status = await asyncio.to_thread(manager.status)
        except Exception as error:
            self.owner.notify(f"Could not update project memory: {error}", severity="error")
            return False
        desired_enabled = enabled_value == "true"
        desired_backend = None if backend_value is Select.NULL else str(backend_value)
        backend_changed = desired_backend is not None and desired_backend != status.backend_id
        enabled_changed = desired_enabled != status.enabled
        if not backend_changed and not enabled_changed:
            return True
        try:
            if desired_backend is not None and backend_changed:
                await asyncio.to_thread(manager.select_backend, desired_backend)
            if enabled_changed:
                await asyncio.to_thread(manager.set_enabled, desired_enabled)
            await self.owner._refresh_agent_memory()
        except Exception as error:
            rollback_errors: list[str] = []
            if enabled_changed:
                try:
                    await asyncio.to_thread(manager.set_enabled, status.enabled)
                except Exception as rollback_error:
                    rollback_errors.append(str(rollback_error))
            if backend_changed:
                try:
                    await asyncio.to_thread(manager.select_backend, status.backend_id)
                except Exception as rollback_error:
                    rollback_errors.append(str(rollback_error))
            try:
                await self.owner._refresh_agent_memory()
            except Exception as rollback_error:
                rollback_errors.append(str(rollback_error))
            detail = f"Could not update project memory: {error}"
            if rollback_errors:
                detail += " Previous memory settings could not be fully restored."
            self.owner.notify(detail, severity="error")
            return False
        await self._refresh_memory_controls()
        return True

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
