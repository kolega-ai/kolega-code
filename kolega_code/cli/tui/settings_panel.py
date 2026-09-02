"""Settings panel behavior for the CLI TUI."""

from __future__ import annotations

import json
import os
import re
import shlex
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, TypeVar, cast, overload
from urllib.parse import urlparse

from textual.css.query import NoMatches
from textual.widget import Widget
from rich.text import Text
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.chatgpt_oauth import run_login_flow
from kolega_code.config import ModelProvider
from kolega_code.agent.tool_backend.search_backends import (
    DEFAULT_BACKEND as DEFAULT_WEB_SEARCH_BACKEND,
    SearchBackendError,
    available_backends,
    get_backend_class,
)
from kolega_code.llm.specs.custom_endpoints import (
    API_STYLES,
    CUSTOM_PROVIDER_PREFIX,
    CUSTOM_THINKING_PRESETS,
    REASONING_REPLAY_VALUES,
    valid_custom_endpoint_id,
)
from kolega_code.mcp.config import (
    MCPConfigError,
    MCPOAuthConfig,
    MCPServerConfig,
    global_mcp_config_path,
    load_mcp_config,
    remove_server_config,
    upsert_server_config,
)
from kolega_code.mcp.service import MCPService, mcp_tool_name_adjustment_note
from kolega_code.mcp.state import MCPStatusStore, MCPOAuthTokenStore

from .. import messages, theme
from ..config import (
    COMPRESSION_THRESHOLD_ENV,
    SKILLS_MODE_ENV,
    SUBAGENTS_MODE_ENV,
    CliConfigError,
    CliConfigOverrides,
    _merged_custom_endpoints,
    active_model_override_message,
    build_agent_config,
    key_status,
    load_cli_env,
    probe_token_manager,
    resolved_api_key,
)
from ..model_connection import test_model_connection
from ..provider_registry import (
    INHERIT_SENTINEL,
    default_model_for_provider,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    get_ui_model,
    agent_role_options,
    default_ui_thinking_effort,
    model_slot_options,
    ui_model_options,
    ui_provider_options,
    ui_thinking_effort_options,
)
from . import app_base as tui_app_base
from .custom_model import (
    CUSTOM_MODEL_SENTINEL,
    resolve_custom_model,
    settings_model_options,
)
from ..settings import WEB_SEARCH_KEY_NAMES, CliSettings
from ..theme import Color, Glyph

MCP_NEW_SERVER_VALUE = "__new_mcp_server__"
ENDPOINT_NEW_VALUE = "__new_endpoint__"
ENDPOINT_NEW_VALUE_LABEL = "New endpoint"
ENDPOINT_THINKING_OPTIONS = [
    ("Reasoning effort (Chat)", "openai_reasoning_effort"),
    ("Responses reasoning", "openai_responses_reasoning"),
    ("Thinking toggle (Qwen3)", "thinking_toggle"),
    ("Anthropic budget tokens", "anthropic_budget"),
]
MCP_TRANSPORT_OPTIONS = [
    ("Streamable HTTP", "streamable_http"),
    ("Server-Sent Events (legacy/deprecated)", "sse"),
    ("stdio command", "stdio"),
]
MCP_ENABLED_OPTIONS = [("Enabled", "true"), ("Disabled", "false")]
MCP_STATUS_MESSAGE_MAX = 96
MCP_STATUS_NAME_MAX = 34
# Compression threshold select: "default" maps to None (the agent's built-in 95%).
COMPRESSION_THRESHOLD_DEFAULT_VALUE = "default"
COMPRESSION_THRESHOLD_PRESET_PERCENTS = (50, 60, 70, 80, 90, 95, 100)
MCP_ATTENTION_STATUSES = {"failed", "stale", "unverified"}
MCP_TRANSPORT_LABELS = {
    "streamable_http": "HTTP",
    "sse": "SSE",
    "stdio": "stdio",
}

SettingsWidget = TypeVar("SettingsWidget", bound=Widget)

# Settings has two kinds of provider→model override row: per-agent-role rows on the
# Agent Models page ("am_" ids) and operational model-slot rows on the Model page
# ("slot_" ids). They share one cascade, so the widget ids are described once here
# instead of being prefix-parsed at each handler.
# Longest first: "custom_model_" must be matched before "model_".
_ROW_FIELDS = ("provider_", "custom_model_", "model_", "effort_")


@dataclass(frozen=True)
class _OverrideRow:
    """The widget ids making up one provider→model override row."""

    key: str  # agent role, or model slot
    provider_id: str
    model_id: str
    custom_id: str
    # Slot rows carry no effort control: --fast-model/KOLEGA_CODE_FAST_MODEL cannot
    # express an effort either, so the UI stays level with the flags.
    effort_id: Optional[str]

    @property
    def is_slot(self) -> bool:
        """True for an operational model-slot row, False for a per-agent-role row."""
        return self.provider_id.startswith("slot_")


def _agent_row(role: str) -> _OverrideRow:
    return _OverrideRow(
        key=role,
        provider_id=f"am_provider_{role}",
        model_id=f"am_model_{role}",
        custom_id=f"am_custom_model_{role}",
        effort_id=f"am_effort_{role}",
    )


def _slot_row(slot: str) -> _OverrideRow:
    return _OverrideRow(
        key=slot,
        provider_id=f"slot_provider_{slot}",
        model_id=f"slot_model_{slot}",
        custom_id=f"slot_custom_model_{slot}",
        effort_id=None,
    )


def _row_for_widget(widget_id: str) -> Optional[_OverrideRow]:
    """Return the override row a widget id belongs to, or None if it is not one."""
    for prefix, build in (("am_", _agent_row), ("slot_", _slot_row)):
        if not widget_id.startswith(prefix):
            continue
        rest = widget_id[len(prefix) :]
        for field in _ROW_FIELDS:
            if rest.startswith(field):
                return build(rest[len(field) :])
    return None


def _compression_threshold_option_value(percent: float) -> str:
    """Select value for a threshold percent: integral values stay bare ("80")."""
    return str(int(percent)) if float(percent).is_integer() else str(percent)


def compression_threshold_options(saved: Optional[float] = None) -> list[tuple[str, str]]:
    """Select options for the compression threshold, presets plus any saved off-list value."""
    options: list[tuple[str, str]] = [("Default (95%)", COMPRESSION_THRESHOLD_DEFAULT_VALUE)]
    options.extend((f"{percent}%", str(percent)) for percent in COMPRESSION_THRESHOLD_PRESET_PERCENTS)
    if saved is not None:
        saved_value = _compression_threshold_option_value(saved)
        if saved_value not in {value for _, value in options}:
            options.append((f"{saved_value}%", saved_value))
    return options


def _mcp_separator() -> str:
    return f" {theme.g(Glyph.BULLET_SEP)} "


def _mcp_transport_label(transport: object) -> str:
    """Human-friendly transport label for the TUI."""
    return MCP_TRANSPORT_LABELS.get(str(transport), str(transport))


def _mcp_plural(count: int, singular: str, plural: Optional[str] = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _mcp_ellipsize(value: str, max_chars: int) -> str:
    value = " ".join(str(value).split())
    if len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return value[:max_chars]
    return f"{value[: max_chars - 1]}{theme.g(Glyph.ELLIPSIS)}"


def _mcp_server_display_label(row: dict[str, object]) -> str:
    server_id = str(row.get("id") or "").strip()
    return str(row.get("name") or server_id).strip() or server_id


def _mcp_status_label(row: dict[str, object]) -> tuple[str, str]:
    """Return the row-level status label and Rich style."""
    if not bool(row.get("enabled")):
        return "disabled", Color.MUTED

    status = str(row.get("status") or "unverified")
    if status == "verified":
        return "verified", Color.SUCCESS
    if status == "stale":
        return "needs re-verify", Color.WARNING
    if status == "failed":
        return "verify failed", Color.ERROR
    if status == "unverified":
        return "not verified", Color.WARNING
    return status.replace("_", " "), Color.MUTED


def _mcp_status_attention_message(row: dict[str, object]) -> str:
    if not bool(row.get("enabled")):
        return ""
    status = str(row.get("status") or "unverified")
    if status not in MCP_ATTENTION_STATUSES:
        return ""
    if status == "stale":
        return "Config changed; verify again."

    message = " ".join(str(row.get("message") or "").split())
    if status == "unverified" and message.rstrip(".") == "Not verified":
        return ""
    if not message and status == "failed":
        message = "Verification failed."
    return _mcp_ellipsize(message, MCP_STATUS_MESSAGE_MAX) if message else ""


def _mcp_status_metadata(row: dict[str, object]) -> list[str]:
    metadata: list[str] = []
    if bool(row.get("enabled")) and str(row.get("status") or "") == "verified":
        try:
            tool_count = int(row.get("tool_count") or 0)  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            tool_count = 0
        metadata.append(_mcp_plural(tool_count, "tool"))
    metadata.append(str(row.get("source") or "unknown"))
    metadata.append(_mcp_transport_label(row.get("transport") or "unknown"))
    if bool(row.get("oauth")):
        metadata.append("oauth")
    return metadata


def _mcp_status_summary(rows: list[dict[str, object]]) -> tuple[str, str]:
    server_count = len(rows)
    if server_count == 0:
        return "No MCP servers configured. Add one below, then Verify.", "info"

    separator = _mcp_separator()
    enabled_rows = [row for row in rows if bool(row.get("enabled"))]
    if not enabled_rows:
        return f"{_mcp_plural(server_count, 'MCP server')} configured{separator}all disabled", "info"

    attention_count = sum(1 for row in enabled_rows if str(row.get("status") or "unverified") != "verified")
    if attention_count:
        verb = "needs" if attention_count == 1 else "need"
        return (
            f"{_mcp_plural(server_count, 'MCP server')} configured{separator}{attention_count} {verb} verification",
            "warning",
        )

    return f"{_mcp_plural(server_count, 'MCP server')} configured{separator}all enabled verified", "ok"


def _render_mcp_status_text(diagnostics: list[str], rows: list[dict[str, object]]) -> tuple[Text, str]:
    """Build the styled MCP status block shown in Settings."""
    summary, tone = _mcp_status_summary(rows)
    content = Text(summary)

    for diagnostic in diagnostics:
        message = " ".join(str(diagnostic).split())
        if not message:
            continue
        content.append("\n  ")
        content.append(message, style=Color.WARNING)

    if not rows:
        return content, tone

    display_labels = [_mcp_ellipsize(_mcp_server_display_label(row), MCP_STATUS_NAME_MAX) for row in rows]
    label_width = max(len(label) for label in display_labels)
    separator = _mcp_separator()

    for row, display_label in zip(rows, display_labels):
        content.append("\n  ")
        content.append(display_label.ljust(label_width))
        content.append("  ")
        status_label, status_style = _mcp_status_label(row)
        content.append(status_label, style=status_style)
        for item in _mcp_status_metadata(row):
            content.append(separator)
            content.append(item)
        message = _mcp_status_attention_message(row)
        if message:
            content.append(" — ")
            content.append(message, style=Color.MUTED)

    return content, tone


def _mcp_server_select_label(server: MCPServerConfig) -> str:
    state = "enabled" if server.enabled else "disabled"
    separator = _mcp_separator()
    return (
        f"{server.display_name} — {server.source}{separator}{_mcp_transport_label(server.transport)}{separator}{state}"
    )


class SettingsPanelMixin(tui_app_base.KolegaAppBase):
    @overload
    def _settings_query_one(self, selector: str) -> Widget: ...

    @overload
    def _settings_query_one(self, selector: str, expect_type: type[SettingsWidget]) -> SettingsWidget: ...

    def _settings_query_one(
        self, selector: str, expect_type: type[SettingsWidget] | None = None
    ) -> SettingsWidget | Widget:
        """Query controls on the open Settings screen, falling back to the app pre-attach."""
        screen = getattr(self, "_settings_screen", None)
        host = screen if screen is not None and getattr(screen, "is_attached", False) else self
        if expect_type is None:
            return host.query_one(selector)
        return host.query_one(selector, expect_type)

    @property
    def _settings_status(self) -> Static:
        return self._settings_query_one("#settings_status", Static)

    def on_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id or ""

        if select_id == "provider_select":
            provider = str(event.value)
            if str(event.select.value) != provider:
                # Stale event: the select already holds a newer value (e.g. a
                # mount-time Changed posted with the compose-time default, delivered
                # after _populate_settings_controls restored the real provider).
                return
            # Purely a model choice now: credentials are edited on the Providers page,
            # so changing the active provider no longer touches any key field.
            self._repopulate_model_select(provider, "model_select", "thinking_effort_select")
            self._update_browser_model_hint()
            self._update_slot_model_hints()
            self._update_settings_status()
            return

        if select_id == "model_select":
            if str(event.select.value) != str(event.value):
                # Stale event: a newer model was set after this event was posted.
                return
            try:
                provider = str(self._settings_query_one("#provider_select", Select).value)
            except NoMatches:
                return
            self._sync_effort_for_model_value(provider, str(event.value), "model_select", "thinking_effort_select")
            self._update_browser_model_hint()
            self._update_slot_model_hints()
            return

        if select_id == "web_search_backend_select":
            self._update_search_backend_fields(str(event.value))
            return

        if select_id == "endpoint_select":
            if str(event.select.value) != str(event.value):
                return
            self._populate_endpoint_form(str(event.value))
            return

        if select_id in {"ep_style_select", "ep_thinking_mode_select"}:
            if str(event.select.value) != str(event.value):
                return
            self._update_endpoint_field_visibility()
            return

        if select_id == "mcp_server_select":
            self._populate_mcp_server_form(str(event.value))
            return

        if select_id == "mcp_transport_select":
            self._update_mcp_transport_fields(str(event.value))
            return

        row = _row_for_widget(select_id)
        if row is not None and select_id == row.provider_id:
            provider = str(event.value)
            if str(event.select.value) != provider:
                return
            if provider == INHERIT_SENTINEL:
                # Don't drop pending here: the selects post an initial inherit-valued
                # Changed on mount, which would clear a restore before its real cascade
                # runs. The populate helpers clear stale pending per row instead.
                self._clear_model_effort_selects(row.model_id, row.effort_id)
            else:
                model_value = self._pending_agent_models.pop(row.model_id, None)
                self._repopulate_model_select(provider, row.model_id, row.effort_id, model_value=model_value)
            self._update_row_hints(row)
            return

        if row is not None and select_id == row.model_id:
            try:
                provider = str(self._settings_query_one(f"#{row.provider_id}", Select).value)
            except NoMatches:
                return
            if provider != INHERIT_SENTINEL and event.value is not Select.NULL:
                # A restored effort waits here for the model that hosts it; a manual
                # model change has none pending and falls back to preserve/default.
                preferred = self._pending_agent_efforts.pop(row.effort_id, None) if row.effort_id is not None else None
                self._sync_effort_for_model_value(
                    provider,
                    str(event.value),
                    row.model_id,
                    row.effort_id,
                    preferred=preferred,
                )
            else:
                self._sync_custom_model_input(row.model_id, row.custom_id)
            self._update_row_hints(row)
            return

        if select_id == "theme_select":
            name = str(event.value)
            if str(event.select.value) != name:
                return
            if name != theme.active_theme().name:
                self._apply_theme(name)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Keep effort selects and the browser hint in sync with "Other…" text.

        The Settings screen has its own ``on_input_changed`` (API key staging and
        dirty tracking); this app-level handler only reacts to the custom-model
        inputs, so the two never conflict.
        """
        input_id = event.input.id or ""
        row: Optional[_OverrideRow] = None
        if input_id == "model_custom_input":
            model_select_id, effort_select_id, provider_id = (
                "model_select",
                "thinking_effort_select",
                "provider_select",
            )
        else:
            row = _row_for_widget(input_id)
            if row is None or input_id != row.custom_id:
                return
            model_select_id, effort_select_id, provider_id = (row.model_id, row.effort_id, row.provider_id)
        try:
            model_select = self._settings_query_one(f"#{model_select_id}", Select)
            provider = str(self._settings_query_one(f"#{provider_id}", Select).value)
        except NoMatches:
            return
        if str(model_select.value) != CUSTOM_MODEL_SENTINEL:
            return
        typed = self._typed_custom_model(provider, input_id)
        if typed is not None and effort_select_id is not None:
            self._set_effort_select_default(provider, typed, effort_select_id)
        if row is None:
            # The active model changed: every inheriting slot hint follows it.
            self._update_browser_model_hint()
            self._update_slot_model_hints()
        else:
            self._update_row_hints(row)

    def _populate_settings_controls(self) -> None:
        screen = getattr(self, "_settings_screen", None)
        if screen is None or not getattr(screen, "is_attached", False):
            # No settings editor mounted: only the sidebar summary needs refreshing.
            self._update_settings_status()
            return
        provider_values = {value for _, value in ui_provider_options()}
        provider = (
            self.settings.active_provider if self.settings.active_provider in provider_values else UI_DEFAULT_PROVIDER
        )
        model_options = settings_model_options(provider)
        valid_models = {value for _, value in model_options}
        model = self.settings.active_model if self.settings.active_model in valid_models else None
        custom_value: Optional[str] = None
        if model is None and self.settings.active_model:
            # A saved catalogued id outside the listed (featured) models maps to
            # the "Other…" entry backed by the custom input.
            if CUSTOM_MODEL_SENTINEL in valid_models and get_ui_model(provider, self.settings.active_model) is not None:
                model = CUSTOM_MODEL_SENTINEL
                custom_value = self.settings.active_model
        if model is None:
            model = model_options[0][1] if model_options else UI_DEFAULT_MODEL
        # The sentinel itself has no spec: effort always comes from the real id.
        effort_model = custom_value or model
        effort_options = {value for _, value in ui_thinking_effort_options(provider, effort_model)}
        effort = (
            self.settings.active_thinking_effort if self.settings.active_thinking_effort in effort_options else None
        )
        if effort is None:
            effort = default_ui_thinking_effort(provider, effort_model)
        provider_select = self._settings_query_one("#provider_select", Select)
        model_select = self._settings_query_one("#model_select", Select)
        effort_select = self._settings_query_one("#thinking_effort_select", Select)

        provider_select.value = provider
        model_select.set_options(model_options)
        model_select.value = model
        try:
            self._settings_query_one("#model_custom_input", Input).value = custom_value or ""
        except NoMatches:
            pass
        self._sync_custom_model_input("model_select", "model_custom_input")
        effort_select.set_options(ui_thinking_effort_options(provider, effort_model))
        if effort is not None:
            effort_select.value = effort
        theme_select = self._settings_query_one("#theme_select", Select)
        theme_select.value = (
            self.settings.active_theme
            if self.settings.active_theme in theme.available_themes()
            else theme.DEFAULT_THEME_NAME
        )
        self._populate_provider_rows()
        self._populate_endpoint_controls()
        self._populate_agent_model_rows()
        self._populate_slot_model_rows()
        self._update_browser_model_hint()
        self._update_slot_model_hints()
        self._populate_web_search_controls()
        self._populate_mcp_controls()
        self._populate_lsp_controls()
        self._populate_subagents_controls()
        self._populate_skills_controls()
        self._populate_compression_controls()
        self._populate_gateway_controls()
        self._update_settings_status()

    def _populate_gateway_controls(self) -> None:
        """Seed the Gateway page from saved settings."""
        try:
            self._settings_query_one("#gateway_token_input", Input)
        except NoMatches:
            return
        gateway = dict(self.settings.gateway or {})
        self._settings_query_one("#gateway_token_input", Input).value = ""
        self._settings_query_one("#gateway_token_status", Static).update(
            "Token saved."
            if self.settings.telegram_bot_token
            else "No token saved yet — paste a @BotFather token here (or run `kolega-code gateway telegram setup`)."
        )
        self._settings_query_one("#gateway_allowed_users_input", Input).value = ", ".join(
            gateway.get("allowed_users") or []
        )
        self._settings_query_one("#gateway_pairing_select", Select).value = (
            "true" if gateway.get("pairing_enabled") else "false"
        )
        self._settings_query_one("#gateway_permission_select", Select).value = str(
            gateway.get("permission_mode") or "ask"
        )
        self._settings_query_one("#gateway_adapter_select", Select).value = str(gateway.get("adapter") or "echo")
        self._settings_query_one("#gateway_project_input", Input).value = str(gateway.get("project") or "")
        self._settings_query_one("#gateway_stt_select", Select).value = (
            "true" if gateway.get("stt_enabled") else "false"
        )
        self._settings_query_one("#gateway_stt_model_input", Input).value = str(gateway.get("stt_model") or "base")

    def _collect_gateway_from_ui(self) -> None:
        """Write the Gateway page into the gateway settings section.

        The token Input only replaces the saved token when something was typed
        (blank = keep); the screen's staged removal wins over a typed value.
        """
        try:
            token_input = self._settings_query_one("#gateway_token_input", Input)
            allowed_input = self._settings_query_one("#gateway_allowed_users_input", Input)
            pairing = str(self._settings_query_one("#gateway_pairing_select", Select).value)
            permission = str(self._settings_query_one("#gateway_permission_select", Select).value)
            adapter = str(self._settings_query_one("#gateway_adapter_select", Select).value)
            project = self._settings_query_one("#gateway_project_input", Input).value.strip()
            stt = str(self._settings_query_one("#gateway_stt_select", Select).value)
            stt_model = self._settings_query_one("#gateway_stt_model_input", Input).value.strip()
        except NoMatches:
            return
        screen = getattr(self, "_settings_screen", None)
        if screen is not None and getattr(screen, "pending_gateway_token_removal", False):
            self.settings.telegram_bot_token = None
        else:
            typed_token = token_input.value.strip()
            if typed_token:
                self.settings.telegram_bot_token = typed_token
        gateway = dict(self.settings.gateway or {})
        allowed = [part.strip() for part in allowed_input.value.split(",") if part.strip()]
        if allowed:
            gateway["allowed_users"] = allowed
        else:
            gateway.pop("allowed_users", None)
        gateway["pairing_enabled"] = pairing == "true"
        gateway["permission_mode"] = permission
        gateway["adapter"] = adapter
        if project:
            gateway["project"] = project
        else:
            gateway.pop("project", None)
        gateway["stt_enabled"] = stt == "true"
        gateway["stt_model"] = stt_model or "base"
        self.settings.gateway = gateway

    def _draft_credential_settings(self) -> CliSettings:
        """Saved settings with the Providers page's unapplied edits layered on.

        Removals are applied before staged keys, so re-typing a key for a provider you
        just cleared wins — matching the order ``_settings_candidate_from_ui`` uses when
        the same edits are actually written.
        """
        draft = deepcopy(self.settings)
        screen = getattr(self, "_settings_screen", None)
        if screen is None:
            return draft
        draft.oauth_tokens = deepcopy(screen.pending_oauth_tokens)
        for removed_provider in screen.pending_api_key_removals:
            draft.api_keys.pop(removed_provider, None)
        for staged_provider, staged_key in screen.pending_api_keys.items():
            draft.set_api_key(staged_provider, staged_key)
        return draft

    def _selected_credential_provider(self) -> Optional[str]:
        """The provider whose credential the Providers page is editing."""
        try:
            provider_list = self._settings_query_one("#provider_list", OptionList)
        except NoMatches:
            return None
        index = provider_list.highlighted
        if index is None:
            return None
        try:
            option = provider_list.get_option_at_index(index)
        except IndexError:
            return None
        return (option.id or "").removeprefix("provider_row_") or None

    def _provider_row_prompt(self, label: str, provider: str, draft: CliSettings) -> Text:
        return Text.assemble(f"{label:<34}", (key_status(provider, self.project_path, draft), Color.MUTED))

    def _populate_provider_rows(self) -> None:
        """Build the provider list and open the active provider's credential editor."""
        try:
            provider_list = self._settings_query_one("#provider_list", OptionList)
        except NoMatches:
            return
        draft = self._draft_credential_settings()
        rows = ui_provider_options()
        provider_list.clear_options()
        provider_list.add_options(
            [
                Option(self._provider_row_prompt(label, value, draft), id=f"provider_row_{value}")
                for label, value in rows
            ]
        )
        values = [value for _, value in rows]
        active = self.settings.active_provider
        provider_list.highlighted = values.index(active) if active in values else 0
        self._switch_provider_credential(self._selected_credential_provider() or values[0])

    def _switch_provider_credential(self, provider: str) -> None:
        """Point the credential editor at ``provider``.

        The outgoing provider's typed key is banked first, then ``credential_provider``
        moves, and only then is the Input rewritten — so the ``Changed`` that rewrite
        posts is already filed against the incoming provider and cannot clobber the key
        just banked for the outgoing one.
        """
        screen = getattr(self, "_settings_screen", None)
        if screen is None or not provider:
            return
        if screen.credential_provider != provider:
            self._commit_visible_api_key()
            screen.credential_provider = provider
            try:
                self._settings_query_one("#provider_api_key_input", Input).value = screen.pending_api_keys.get(
                    provider, ""
                )
            except NoMatches:
                return
        self._update_provider_credential_controls(provider)
        self._refresh_provider_row_labels()

    def _commit_visible_api_key(self) -> None:
        """Bank whatever the key field holds against the provider it is showing."""
        screen = getattr(self, "_settings_screen", None)
        provider = getattr(screen, "credential_provider", None)
        if screen is None or provider is None:
            return
        try:
            typed = self._settings_query_one("#provider_api_key_input", Input).value.strip()
        except NoMatches:
            return
        if typed:
            screen.pending_api_keys[provider] = typed
            # Typing a replacement supersedes a staged removal for the same provider.
            screen.pending_api_key_removals.discard(provider)
        else:
            screen.pending_api_keys.pop(provider, None)
        self._update_provider_credential_controls(provider)
        self._refresh_provider_row_labels()

    def _refresh_provider_row_labels(self) -> None:
        """Restate row statuses in place.

        Relabelling beats rebuilding: ``clear_options`` would drop the highlight and
        post a fresh Highlighted event, which reloads the key Input mid-keystroke.
        """
        try:
            provider_list = self._settings_query_one("#provider_list", OptionList)
        except NoMatches:
            return
        draft = self._draft_credential_settings()
        for label, value in ui_provider_options():
            try:
                provider_list.replace_option_prompt(
                    f"provider_row_{value}", self._provider_row_prompt(label, value, draft)
                )
            except Exception:
                # The list lags the registry after an apply that adds endpoints;
                # _populate_provider_rows rebuilds it in that path.
                continue

    def _update_provider_credential_controls(self, provider: str) -> None:
        """Swap the editor between API-key, ChatGPT sign-in, and custom-endpoint shape."""
        oauth = provider == chatgpt_constants.PROVIDER_KEY
        custom = provider.startswith(CUSTOM_PROVIDER_PREFIX)
        for widget_id in ("provider_chatgpt_login", "provider_chatgpt_logout"):
            try:
                self._settings_query_one(f"#{widget_id}").display = oauth
            except NoMatches:
                pass
        for widget_id in ("provider_api_key_label", "provider_api_key_input"):
            try:
                self._settings_query_one(f"#{widget_id}").display = not oauth and not custom
            except NoMatches:
                pass
        try:
            # The OAuth row is a Horizontal: hiding only its buttons would leave the
            # empty band behind, so the container goes with them.
            self._settings_query_one("#provider_chatgpt_row").display = oauth
        except NoMatches:
            pass
        try:
            self._settings_query_one("#provider_endpoint_key_hint").display = custom
        except NoMatches:
            pass
        try:
            if not custom:
                self._settings_query_one("#provider_api_key_input", Input).placeholder = self._api_key_placeholder(
                    provider
                )
            section = self._settings_query_one("#settings_provider_credential")
            # ui_provider_options() is (label, value); index it the other way round.
            labels = {value: label for label, value in ui_provider_options()}
            section.border_title = labels.get(provider, provider)
        except NoMatches:
            pass
        try:
            remove_key = self._settings_query_one("#provider_remove_api_key", Button)
        except NoMatches:
            return
        remove_key.display = not oauth and not custom
        screen = getattr(self, "_settings_screen", None)
        pending_removals = getattr(screen, "pending_api_key_removals", set())
        remove_key.disabled = not self.settings.has_api_key(provider) or provider in pending_removals
        if screen is None:
            return
        draft = self._draft_credential_settings()
        try:
            status = self._settings_query_one("#provider_credential_status", Static)
            status.update(
                messages.PROVIDER_CREDENTIAL_STATUS.format(status=key_status(provider, self.project_path, draft))
            )
            login = self._settings_query_one("#provider_chatgpt_login", Button)
            logout = self._settings_query_one("#provider_chatgpt_logout", Button)
            signed_in = draft.has_oauth_token(chatgpt_constants.PROVIDER_KEY)
            login.label = "Sign in again" if signed_in else "Sign in with ChatGPT"
            logout.disabled = not signed_in
        except NoMatches:
            pass

    # --- Custom Endpoints page -------------------------------------------------

    def _populate_endpoint_controls(self) -> None:
        """Rebuild the endpoint selector from the pending draft and open its form."""
        screen = getattr(self, "_settings_screen", None)
        if screen is None:
            return
        try:
            endpoint_select = self._settings_query_one("#endpoint_select", Select)
        except NoMatches:
            return
        pending = getattr(screen, "pending_custom_endpoints", {}) or {}
        selected = getattr(self, "_ep_selected", None)
        if selected is None:
            selected = sorted(pending)[0] if pending else ENDPOINT_NEW_VALUE
        options = [(ENDPOINT_NEW_VALUE_LABEL, ENDPOINT_NEW_VALUE)]
        options.extend((endpoint_id, endpoint_id) for endpoint_id in sorted(pending))
        endpoint_select.set_options(options)
        endpoint_select.value = selected if selected in {value for _, value in options} else ENDPOINT_NEW_VALUE
        self._populate_endpoint_form(endpoint_select.value)

    def _populate_endpoint_form(self, endpoint_id: str) -> None:
        self._ep_selected = endpoint_id
        screen = getattr(self, "_settings_screen", None)
        pending = getattr(screen, "pending_custom_endpoints", {}) if screen else {}
        entry = pending.get(endpoint_id) or {} if endpoint_id != ENDPOINT_NEW_VALUE else {}
        thinking = entry.get("thinking") or {}
        try:
            self._settings_query_one("#ep_id_input", Input).value = endpoint_id if entry else ""
            self._settings_query_one("#ep_id_input", Input).disabled = bool(entry)
            self._settings_query_one("#ep_label_input", Input).value = str(entry.get("label") or "")
            self._settings_query_one("#ep_style_select", Select).value = entry.get("api_style", "openai_chat")
            self._settings_query_one("#ep_base_url_input", Input).value = str(entry.get("base_url") or "")
            self._settings_query_one("#ep_api_key_input", Input).value = str(entry.get("api_key") or "")
            self._settings_query_one("#ep_default_model_input", Input).value = str(entry.get("default_model") or "")
            self._settings_query_one("#ep_context_input", Input).value = str(entry.get("context_length") or "")
            self._settings_query_one("#ep_max_output_input", Input).value = str(entry.get("max_output_tokens") or "")
            temperature = entry.get("temperature")
            self._settings_query_one("#ep_temperature_input", Input).value = (
                str(temperature) if temperature is not None else ""
            )
            self._settings_query_one("#ep_vision_select", Select).value = (
                "true" if entry.get("supports_vision") else "false"
            )
            self._settings_query_one("#ep_thinking_mode_select", Select).value = thinking.get("mode", "")
            self._settings_query_one("#ep_thinking_budgets_input", Input).value = self._format_thinking_budgets(
                thinking
            )
            self._settings_query_one("#ep_reasoning_select", Select).value = entry.get("reasoning_replay", "auto")
        except NoMatches:
            return
        self._update_endpoint_field_visibility()
        self._update_endpoint_status()

    @staticmethod
    def _format_thinking_budgets(thinking: Mapping[str, Any]) -> str:
        budgets = thinking.get("budgets") if isinstance(thinking, dict) else None
        if not isinstance(budgets, dict):
            return ""
        return ", ".join(f"{option}={budgets[option]}" for option in sorted(budgets))

    def _update_endpoint_field_visibility(self) -> None:
        try:
            style = str(self._settings_query_one("#ep_style_select", Select).value)
            thinking = str(self._settings_query_one("#ep_thinking_mode_select", Select).value)
        except NoMatches:
            return
        budgets = thinking == "anthropic_budget"
        for widget_id, visible in (
            ("ep_thinking_budgets_label", budgets),
            ("ep_thinking_budgets_input", budgets),
            ("ep_reasoning_label", style == "openai_chat"),
            ("ep_reasoning_select", style == "openai_chat"),
            ("ep_temperature_label", style != "openai_responses"),
            ("ep_temperature_input", style != "openai_responses"),
        ):
            try:
                self._settings_query_one(f"#{widget_id}").display = visible
            except NoMatches:
                pass

    @staticmethod
    def _optional_positive_int(raw: str, label: str) -> Optional[int]:
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value <= 0:
            raise ValueError(f"{label} must be a positive integer.")
        return value

    @staticmethod
    def _parse_thinking_budgets(raw: str, cap: int) -> dict[str, int]:
        budgets: dict[str, int] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError("Budgets must be effort=tokens pairs, e.g. 'low=2048, medium=8192'.")
            option, _, value = part.partition("=")
            option = option.strip()
            try:
                budget = int(value.strip())
            except ValueError:
                budget = 0
            if budget <= 0:
                raise ValueError(f"Budget for '{option}' must be a positive integer.")
            if budget >= cap:
                raise ValueError(f"Budget for '{option}' must stay below max output tokens ({cap}).")
            budgets[option] = budget
        if not budgets:
            raise ValueError("anthropic_budget needs budgets, e.g. 'low=2048, medium=8192'.")
        return budgets

    def _collect_endpoint_from_ui(self) -> tuple[str, dict[str, Any]]:
        def input_value(widget_id: str) -> str:
            return self._settings_query_one(f"#{widget_id}", Input).value.strip()

        endpoint_id = input_value("ep_id_input").lower()
        if not valid_custom_endpoint_id(endpoint_id):
            raise ValueError("Endpoint id must be a lowercase slug (letters, digits, '-', '_').")
        base_url = input_value("ep_base_url_input")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Base URL must be an http(s) URL.")
        style = str(self._settings_query_one("#ep_style_select", Select).value)
        if style not in API_STYLES:
            raise ValueError("API style is required.")
        context = self._optional_positive_int(input_value("ep_context_input"), "Context length")
        max_output = self._optional_positive_int(input_value("ep_max_output_input"), "Max output tokens")

        entry: dict[str, Any] = {"api_style": style, "base_url": base_url.rstrip("/")}
        label = input_value("ep_label_input")
        if label:
            entry["label"] = label
        api_key = self._settings_query_one("#ep_api_key_input", Input).value
        if api_key:
            entry["api_key"] = api_key
        default_model = input_value("ep_default_model_input")
        if default_model:
            entry["default_model"] = default_model
        if context is not None:
            entry["context_length"] = context
        if max_output is not None:
            entry["max_output_tokens"] = max_output
        raw_temperature = input_value("ep_temperature_input")
        if raw_temperature and style != "openai_responses":
            try:
                temperature = float(raw_temperature)
            except ValueError:
                temperature = 0.0
            if not (0 < temperature <= 2):
                raise ValueError("Temperature must be a number between 0 and 2.")
            if style == "anthropic" and temperature > 1:
                raise ValueError("Anthropic-style endpoints accept temperature up to 1.")
            entry["temperature"] = temperature
        if str(self._settings_query_one("#ep_vision_select", Select).value) == "true":
            entry["supports_vision"] = True
        thinking_mode = str(self._settings_query_one("#ep_thinking_mode_select", Select).value)
        if thinking_mode:
            preset = CUSTOM_THINKING_PRESETS[thinking_mode]
            thinking: dict[str, Any] = {
                "mode": thinking_mode,
                "options": list(preset["options"]),
                "default": preset["default"],
            }
            if preset["budgets_required"]:
                cap = max_output if max_output is not None else 8192
                thinking["budgets"] = self._parse_thinking_budgets(input_value("ep_thinking_budgets_input"), cap)
            entry["thinking"] = thinking
        if style == "openai_chat":
            replay = str(self._settings_query_one("#ep_reasoning_select", Select).value)
            if replay in REASONING_REPLAY_VALUES:
                entry["reasoning_replay"] = replay
        screen = getattr(self, "_settings_screen", None)
        existing = (getattr(screen, "pending_custom_endpoints", {}) or {}).get(endpoint_id) or {}
        if existing.get("models"):
            entry["models"] = existing["models"]
        return endpoint_id, entry

    def _save_endpoint_from_ui(self) -> None:
        try:
            endpoint_id, entry = self._collect_endpoint_from_ui()
        except ValueError as exc:
            self._set_endpoint_status(str(exc), "warning")
            return
        screen = getattr(self, "_settings_screen", None)
        if screen is None:
            return
        previous = deepcopy(screen.pending_custom_endpoints)
        screen.pending_custom_endpoints[endpoint_id] = entry
        if screen.pending_custom_endpoints != previous:
            self._ep_selected = endpoint_id
            self._populate_endpoint_controls()
        if self.settings.active_provider == f"{CUSTOM_PROVIDER_PREFIX}{endpoint_id}":
            self._set_endpoint_status(
                f"{messages.ENDPOINT_APPLY_REQUIRED} It is the active provider, so this edit applies to it too.",
                "ok",
            )
        else:
            self._set_endpoint_status(messages.ENDPOINT_APPLY_REQUIRED, "ok")
        screen.call_after_refresh(screen._refresh_apply_label)

    def _delete_endpoint_from_ui(self) -> None:
        screen = getattr(self, "_settings_screen", None)
        if screen is None:
            return
        endpoint_id = getattr(self, "_ep_selected", ENDPOINT_NEW_VALUE)
        if endpoint_id == ENDPOINT_NEW_VALUE:
            self._set_endpoint_status("Select an endpoint to delete.", "warning")
            return
        screen.pending_custom_endpoints.pop(endpoint_id, None)
        self._ep_selected = ENDPOINT_NEW_VALUE
        self._populate_endpoint_controls()
        if self.settings.active_provider == f"{CUSTOM_PROVIDER_PREFIX}{endpoint_id}":
            self._set_endpoint_status(
                f"{messages.ENDPOINT_DELETED} It is the active provider; pick another model before applying.", "warning"
            )
        else:
            self._set_endpoint_status(messages.ENDPOINT_DELETED, "ok")
        screen.call_after_refresh(screen._refresh_apply_label)

    def _handle_endpoint_settings_button(self, button_id: str) -> None:
        if self._turn_active or self.agent_worker is not None:
            self._set_endpoint_status("Stop the active turn before changing endpoints.", "warning")
            return
        if button_id == "ep_save":
            self._save_endpoint_from_ui()
        elif button_id == "ep_delete":
            self._delete_endpoint_from_ui()

    def _update_endpoint_status(self) -> None:
        screen = getattr(self, "_settings_screen", None)
        pending = getattr(screen, "pending_custom_endpoints", {}) if screen else {}
        if not pending:
            self._set_endpoint_status(messages.ENDPOINT_NONE)

    def _set_endpoint_status(self, text: str, tone: str = "info") -> None:
        glyph, style = {
            "ok": (Glyph.CHECK, Color.SUCCESS),
            "error": (Glyph.CROSS, Color.ERROR),
            "warning": (Glyph.STATUS, Color.WARNING),
        }.get(tone, (Glyph.STATUS, Color.MUTED))
        content = Text()
        content.append(theme.g(glyph) + " ", style=style)
        content.append(text)
        try:
            self._settings_query_one("#endpoint_status", Static).update(content)
        except NoMatches:
            pass

    def _populate_agent_model_rows(self) -> None:
        """Seed each per-agent row from saved settings (absent role -> inherit).

        Setting the provider value posts a Changed event that re-runs the cascade,
        but Textual may deliver that event after other awaited startup work. Apply
        the model/effort directly as the deterministic path, while also leaving the
        pending values for the Changed event to consume if it arrives later.
        """
        provider_values = {value for _, value in ui_provider_options()}
        for _, role in agent_role_options():
            try:
                provider_select = self._settings_query_one(f"#am_provider_{role}", Select)
            except NoMatches:
                continue
            entry = self.settings.get_agent_model(role) or {}
            provider = entry.get("provider")
            model_id = f"am_model_{role}"
            effort_id = f"am_effort_{role}"
            self._pending_agent_models.pop(model_id, None)
            self._pending_agent_efforts.pop(effort_id, None)
            if provider not in provider_values:
                provider_select.value = INHERIT_SENTINEL
                self._clear_model_effort_selects(model_id, effort_id)
                continue
            model_value = str(entry["model"]) if entry.get("model") else None
            effort_value = str(entry["thinking_effort"]) if entry.get("thinking_effort") else None
            if model_value:
                self._pending_agent_models[model_id] = model_value
            if effort_value:
                self._pending_agent_efforts[effort_id] = effort_value
            provider_select.value = provider
            self._repopulate_model_select(
                provider, model_id, effort_id, model_value=model_value, effort_value=effort_value
            )

    def _populate_slot_model_rows(self) -> None:
        """Seed each model-slot row from saved settings (absent slot -> inherit).

        Same restore dance as ``_populate_agent_model_rows``: the pending maps are
        keyed by widget id, so a Changed event arriving after this deterministic pass
        consumes the same values instead of clobbering them.
        """
        provider_values = {value for _, value in ui_provider_options()}
        for _, slot in model_slot_options():
            row = _slot_row(slot)
            try:
                provider_select = self._settings_query_one(f"#{row.provider_id}", Select)
            except NoMatches:
                continue
            entry = self.settings.get_model_slot(slot) or {}
            provider = entry.get("provider")
            self._pending_agent_models.pop(row.model_id, None)
            if provider not in provider_values:
                provider_select.value = INHERIT_SENTINEL
                self._clear_model_effort_selects(row.model_id, row.effort_id)
                continue
            model_value = str(entry["model"]) if entry.get("model") else None
            if model_value:
                self._pending_agent_models[row.model_id] = model_value
            provider_select.value = provider
            self._repopulate_model_select(provider, row.model_id, row.effort_id, model_value=model_value)

    def _slot_model_status(self, slot: str) -> str:
        """Return the hint line describing which model a slot currently resolves to."""
        row = _slot_row(slot)
        try:
            slot_provider = str(self._settings_query_one(f"#{row.provider_id}", Select).value)
        except NoMatches:
            return ""

        inherited = slot_provider == INHERIT_SENTINEL
        provider_id, model_id, custom_id = (
            ("provider_select", "model_select", "model_custom_input")
            if inherited
            else (row.provider_id, row.model_id, row.custom_id)
        )
        try:
            provider = str(self._settings_query_one(f"#{provider_id}", Select).value)
            model_value = self._settings_query_one(f"#{model_id}", Select).value
        except NoMatches:
            return ""
        if model_value is Select.NULL:
            return ""
        if str(model_value) == CUSTOM_MODEL_SENTINEL:
            model = self._typed_custom_model(provider, custom_id)
            if model is None:
                return ""
        else:
            model = str(model_value)

        template = messages.MODEL_SLOT_INHERITED if inherited else messages.MODEL_SLOT_PINNED
        return template.format(provider=provider, model=model)

    def _update_slot_model_hints(self) -> None:
        """Keep every slot hint synchronized with its resolved model."""
        for _, slot in model_slot_options():
            try:
                hint = self._settings_query_one(f"#slot_hint_{slot}", Static)
            except NoMatches:
                continue
            hint.update(self._slot_model_status(slot))

    def _update_row_hints(self, row: _OverrideRow) -> None:
        """Refresh whichever hint the changed override row feeds."""
        if row.is_slot:
            self._update_slot_model_hints()
        elif row.key == "browser":
            self._update_browser_model_hint()

    def _browser_model_status(self) -> tuple[str, str, bool]:
        """Return the Browser-role model message, tone, and whether saving must stop."""
        try:
            browser_provider = str(self._settings_query_one("#am_provider_browser", Select).value)
        except NoMatches:
            return "", "info", False

        inherited = browser_provider == INHERIT_SENTINEL
        if inherited:
            try:
                provider = str(self._settings_query_one("#provider_select", Select).value)
                model_value = str(self._settings_query_one("#model_select", Select).value)
            except NoMatches:
                return "", "info", False
            if model_value == CUSTOM_MODEL_SENTINEL:
                model = self._typed_custom_model(provider, "model_custom_input")
                if model is None:
                    return "", "info", False
            else:
                model = model_value
        else:
            provider = browser_provider
            try:
                model_value = self._settings_query_one("#am_model_browser", Select).value
            except NoMatches:
                return "", "info", False
            if model_value is Select.NULL:
                return messages.BROWSER_MODEL_PROVIDER_NO_VISION.format(provider=provider), "error", True
            if model_value == CUSTOM_MODEL_SENTINEL:
                model = self._typed_custom_model(provider, "am_custom_model_browser")
                if model is None:
                    return "", "info", False
            else:
                model = str(model_value)

        option = get_ui_model(provider, model)
        supports_vision = bool(option and option.supports_vision)
        if supports_vision:
            message = messages.BROWSER_MODEL_INHERIT_VISION_READY if inherited else messages.BROWSER_MODEL_VISION_READY
            return message, "ok", False
        if inherited:
            return (
                messages.BROWSER_MODEL_INHERIT_NO_VISION.format(provider=provider, model=model),
                "warning",
                False,
            )
        return (
            messages.BROWSER_MODEL_EXPLICIT_NO_VISION.format(provider=provider, model=model),
            "error",
            True,
        )

    def _update_browser_model_hint(self) -> None:
        """Keep the Browser-role capability hint synchronized with its resolved model."""
        message, tone, _ = self._browser_model_status()
        if not message:
            return
        glyph, style = {
            "ok": (Glyph.CHECK, Color.SUCCESS),
            "error": (Glyph.CROSS, Color.ERROR),
            "warning": (Glyph.STATUS, Color.WARNING),
        }.get(tone, (Glyph.STATUS, Color.MUTED))
        content = Text()
        content.append(theme.g(glyph) + " ", style=style)
        content.append(message)
        try:
            self._settings_query_one("#am_status_browser", Static).update(content)
        except NoMatches:
            return

    def _update_search_backend_fields(self, backend: str) -> None:
        """Show only the inputs the selected web-search backend needs.

        Called from on_select_changed (which can fire its initial Changed on mount,
        before the section is fully populated) and from populate, so every query_one
        is guarded against NoMatches."""
        try:
            backend_cls = get_backend_class(backend)
        except SearchBackendError:
            backend_cls = None
        needs_key = bool(backend_cls and backend_cls.accepts_api_key)
        needs_url = bool(backend_cls and backend_cls.requires_base_url)
        for widget_id, visible in (
            ("web_search_api_key_label", needs_key),
            ("web_search_api_key_input", needs_key),
            ("web_search_base_url_label", needs_url),
            ("web_search_base_url_input", needs_url),
        ):
            try:
                self._settings_query_one(f"#{widget_id}").display = visible
            except NoMatches:
                pass
        if needs_key:
            try:
                key_input = self._settings_query_one("#web_search_api_key_input", Input)
            except NoMatches:
                return
            env_var = (backend_cls.env_var if backend_cls else None) or "API"
            if self.settings.has_api_key(backend):
                key_input.placeholder = "Stored API key will be kept if blank"
            elif backend_cls and backend_cls.requires_api_key:
                key_input.placeholder = f"{env_var} key"
            else:
                key_input.placeholder = f"Optional — {env_var} key for higher rate limits"

    def _populate_web_search_controls(self) -> None:
        """Seed the Web Search controls from saved settings (key field stays blank)."""
        from kolega_code.cli.config import DEFAULT_WEB_SEARCH_MODE, WEB_SEARCH_MODES

        valid = {name for _, name in available_backends()}
        backend = self.settings.web_search_backend
        if backend not in valid:
            backend = DEFAULT_WEB_SEARCH_BACKEND
        mode = self.settings.web_search_mode
        if mode not in WEB_SEARCH_MODES:
            mode = DEFAULT_WEB_SEARCH_MODE
        try:
            self._settings_query_one("#web_search_mode_select", Select).value = mode
            self._settings_query_one("#web_search_backend_select", Select).value = backend
            self._settings_query_one("#web_search_base_url_input", Input).value = (
                self.settings.web_search_base_url or ""
            )
            self._settings_query_one("#web_search_api_key_input", Input).value = ""
        except NoMatches:
            pass
        self._update_search_backend_fields(backend)

    def _collect_web_search_from_ui(self) -> None:
        """Write the Web Search controls into settings (keys only when newly typed)."""
        try:
            mode = str(self._settings_query_one("#web_search_mode_select", Select).value)
            backend = str(self._settings_query_one("#web_search_backend_select", Select).value)
            base_url_input = self._settings_query_one("#web_search_base_url_input", Input)
            key_input = self._settings_query_one("#web_search_api_key_input", Input)
        except NoMatches:
            return
        self.settings.web_search_mode = mode
        self.settings.web_search_backend = backend
        self.settings.web_search_base_url = base_url_input.value.strip() or None
        key = key_input.value.strip()
        if key and backend in WEB_SEARCH_KEY_NAMES:
            self.settings.set_api_key(backend, key)
        self._update_search_backend_fields(backend)

    def _load_mcp_config_for_ui(self):
        """Load MCP config for the settings panel and attach it to the active AgentConfig."""
        trusted = bool(self.settings.is_mcp_project_trusted(self.project_path))
        config = load_mcp_config(self.project_path, self.settings_store.root, project_trusted=trusted)
        if self.config is not None:
            self.config.mcp_config = config
        return config

    def _populate_mcp_controls(self) -> None:
        """Seed the MCP settings controls from global/trusted project config and status."""
        try:
            config = self._load_mcp_config_for_ui()
            server_select = self._settings_query_one("#mcp_server_select", Select)
        except NoMatches:
            return
        except Exception as exc:
            self._set_mcp_status(f"MCP config could not be loaded: {exc}", tone="error")
            return

        options = [("New user server", MCP_NEW_SERVER_VALUE)]
        options.extend((self._mcp_server_option_label(server), server.id) for server in config.servers.values())
        selected = getattr(self, "_mcp_selected_server_id", MCP_NEW_SERVER_VALUE)
        if selected not in {value for _, value in options}:
            selected = MCP_NEW_SERVER_VALUE
        server_select.set_options(options)
        server_select.value = selected
        self._populate_mcp_server_form(selected)
        self._update_mcp_status_text(config)

    def _mcp_server_option_label(self, server: MCPServerConfig) -> str:
        return _mcp_server_select_label(server)

    def _populate_mcp_server_form(self, server_id: str) -> None:
        self._mcp_selected_server_id = server_id
        try:
            config = self._load_mcp_config_for_ui()
        except Exception:
            config = None
        server = None if server_id == MCP_NEW_SERVER_VALUE or config is None else config.servers.get(server_id)

        def set_input(widget_id: str, value: str) -> None:
            try:
                self._settings_query_one(f"#{widget_id}", Input).value = value
            except NoMatches:
                pass

        def set_select(widget_id: str, value: str) -> None:
            try:
                select = self._settings_query_one(f"#{widget_id}", Select)
                if value is not Select.NULL:
                    select.value = value
            except NoMatches:
                pass

        if server is None:
            set_input("mcp_name_input", "")
            set_select("mcp_transport_select", "streamable_http")
            set_select("mcp_enabled_select", "true")
            set_input("mcp_url_input", "")
            set_input("mcp_headers_input", "")
            set_select("mcp_oauth_select", "false")
            set_input("mcp_command_input", "")
            set_input("mcp_args_input", "")
            set_input("mcp_env_input", "")
            set_input("mcp_cwd_input", "")
            self._set_mcp_source_hint("Create or update a user MCP server in the global state config.")
            self._update_mcp_transport_fields("streamable_http")
            return

        set_input("mcp_name_input", server.name or "")
        set_select("mcp_transport_select", server.transport)
        set_select("mcp_enabled_select", "true" if server.enabled else "false")
        set_input("mcp_url_input", server.url or "")
        set_input("mcp_headers_input", json.dumps(server.headers, sort_keys=True) if server.headers else "")
        set_select("mcp_oauth_select", "true" if server.oauth.enabled else "false")
        set_input("mcp_command_input", server.command or "")
        set_input("mcp_args_input", " ".join(shlex.quote(arg) for arg in server.args))
        set_input("mcp_env_input", json.dumps(server.env, sort_keys=True) if server.env else "")
        set_input("mcp_cwd_input", server.cwd or "")
        if server.source == "project":
            self._set_mcp_source_hint(
                "This server comes from the trusted project config and is read-only here; edit .kolega/mcp_servers.json."
            )
        else:
            self._set_mcp_source_hint("This server is stored in your global MCP config.")
        self._update_mcp_transport_fields(server.transport)

    def _update_mcp_transport_fields(self, transport: str) -> None:
        http = transport in {"streamable_http", "sse"}
        for widget_id, visible in (
            ("mcp_url_label", http),
            ("mcp_url_input", http),
            ("mcp_headers_label", http),
            ("mcp_headers_input", http),
            ("mcp_oauth_label", http),
            ("mcp_oauth_select", http),
            ("mcp_command_label", not http),
            ("mcp_command_input", not http),
            ("mcp_args_label", not http),
            ("mcp_args_input", not http),
            ("mcp_env_label", not http),
            ("mcp_env_input", not http),
            ("mcp_cwd_label", not http),
            ("mcp_cwd_input", not http),
        ):
            try:
                self._settings_query_one(f"#{widget_id}").display = visible
            except NoMatches:
                pass
        try:
            url_input = self._settings_query_one("#mcp_url_input", Input)
            if transport == "streamable_http":
                url_input.placeholder = "https://example.com/mcp"
            elif transport == "sse":
                url_input.placeholder = "https://example.com/sse"
        except NoMatches:
            pass

    def _update_mcp_status_text(self, config=None) -> None:
        try:
            config = config or self._load_mcp_config_for_ui()
        except Exception as exc:
            self._set_mcp_status(f"MCP config could not be loaded: {exc}", tone="error")
            return

        rows = MCPService(config, self.settings_store.root, self.project_path).list_status_rows()
        content, tone = _render_mcp_status_text(list(config.diagnostics), rows)
        self._set_mcp_status(content, tone=tone)

    def _set_mcp_status(self, text: str | Text, tone: str = "info") -> None:
        glyph, style = {
            "ok": (Glyph.CHECK, Color.SUCCESS),
            "error": (Glyph.CROSS, Color.ERROR),
            "warning": (Glyph.STATUS, Color.WARNING),
        }.get(tone, (Glyph.STATUS, Color.MUTED))
        content = Text()
        content.append(theme.g(glyph) + " ", style=style)
        if isinstance(text, Text):
            content.append_text(text)
        else:
            content.append(text)
        try:
            self._settings_query_one("#mcp_status", Static).update(content)
        except NoMatches:
            return

    def _set_mcp_source_hint(self, text: str) -> None:
        try:
            self._settings_query_one("#mcp_source_hint", Static).update(text)
        except NoMatches:
            pass

    def _slug_mcp_server_id(self, value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip().lower())
        slug = re.sub(r"-+", "-", slug).strip("-_")
        return slug[:64].strip("-_")

    def _auto_mcp_server_id(
        self,
        *,
        name: Optional[str],
        url: Optional[str],
        command: Optional[str],
        args: list[str],
    ) -> str:
        candidates: list[str] = []
        if name:
            candidates.append(name)
        if url:
            parsed = urlparse(url)
            if parsed.hostname:
                candidates.append(parsed.hostname.removeprefix("www."))
        if args:
            transport_args = {"stdio", "sse", "streamableHttp", "streamable_http"}
            meaningful_args = [arg for arg in args if not arg.startswith("-") and arg not in transport_args]
            candidates.extend(reversed(meaningful_args))
        if command:
            candidates.append(command)
        candidates.append("mcp-server")

        base = next((slug for candidate in candidates if (slug := self._slug_mcp_server_id(candidate))), "mcp-server")
        try:
            config = self._load_mcp_config_for_ui()
            existing_ids = set(config.servers)
        except Exception:
            existing_ids = set()
        selected = self._selected_mcp_server_id()
        if selected != MCP_NEW_SERVER_VALUE:
            existing_ids.discard(selected)
        if base not in existing_ids:
            return base
        suffix = 2
        while f"{base}-{suffix}" in existing_ids:
            suffix += 1
        return f"{base}-{suffix}"

    def _collect_mcp_server_from_ui(self) -> MCPServerConfig:
        name = self._settings_query_one("#mcp_name_input", Input).value.strip() or None
        transport = cast(
            Literal["streamable_http", "sse", "stdio"],
            str(self._settings_query_one("#mcp_transport_select", Select).value),
        )
        enabled = str(self._settings_query_one("#mcp_enabled_select", Select).value) == "true"
        url = self._settings_query_one("#mcp_url_input", Input).value.strip() or None
        headers_text = self._settings_query_one("#mcp_headers_input", Input).value.strip()
        oauth_enabled = str(self._settings_query_one("#mcp_oauth_select", Select).value) == "true"
        command = self._settings_query_one("#mcp_command_input", Input).value.strip() or None
        args_text = self._settings_query_one("#mcp_args_input", Input).value.strip()
        env_text = self._settings_query_one("#mcp_env_input", Input).value.strip()
        cwd = self._settings_query_one("#mcp_cwd_input", Input).value.strip() or None

        headers = self._parse_mcp_json_object(headers_text, "headers")
        env = self._parse_mcp_json_object(env_text, "env")
        try:
            args = shlex.split(args_text) if args_text else []
        except ValueError as exc:
            raise ValueError(f"MCP args must be shell-like tokens: {exc}") from exc
        selected = self._selected_mcp_server_id()
        server_id = (
            self._auto_mcp_server_id(name=name, url=url, command=command, args=args)
            if selected == MCP_NEW_SERVER_VALUE
            else selected
        )

        return MCPServerConfig(
            id=server_id,
            name=name,
            transport=transport,
            enabled=enabled,
            url=url,
            headers=headers,
            oauth=MCPOAuthConfig(enabled=oauth_enabled),
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            source="global",
        )

    def _parse_mcp_json_object(self, value: str, label: str) -> dict[str, str]:
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"MCP {label} must be a JSON object: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"MCP {label} must be a JSON object")
        return {str(key): str(item) for key, item in parsed.items() if item is not None}

    async def _handle_mcp_settings_button(self, button_id: str) -> bool:
        if button_id != "mcp_refresh" and (self._turn_active or self.agent_worker is not None):
            self._set_mcp_status("Stop the active turn before changing MCP servers.", "warning")
            return True
        if button_id == "mcp_refresh":
            self._populate_mcp_controls()
            return True
        if button_id == "mcp_trust_project":
            self.settings.trust_mcp_project(self.project_path)
            self.settings_store.save(self.settings)
            self._populate_mcp_controls()
            await self._ensure_agent_from_settings(rebuild=True)
            self._notify_user("Trusted project MCP config for this project.")
            return True
        if button_id == "mcp_save_server":
            await self._save_mcp_server_from_ui()
            return True
        if button_id == "mcp_delete_server":
            await self._delete_mcp_server_from_ui()
            return True
        if button_id == "mcp_verify_server":
            await self._verify_mcp_server_from_ui()
            return True
        if button_id == "mcp_clear_tokens":
            self._clear_mcp_tokens_from_ui()
            return True
        return False

    def _selected_mcp_server_id(self) -> str:
        try:
            value = self._settings_query_one("#mcp_server_select", Select).value
        except NoMatches:
            return MCP_NEW_SERVER_VALUE
        return MCP_NEW_SERVER_VALUE if value is Select.NULL else str(value)

    async def _save_mcp_server_from_ui(self) -> None:
        selected = self._selected_mcp_server_id()
        try:
            server = self._collect_mcp_server_from_ui()
            config = self._load_mcp_config_for_ui()
            existing = config.servers.get(selected) if selected != MCP_NEW_SERVER_VALUE else None
            target_existing = config.servers.get(server.id)
            if (existing is not None and existing.source == "project") or (
                target_existing is not None and target_existing.source == "project"
            ):
                self._set_mcp_status(
                    "Project MCP servers are read-only in the TUI; edit .kolega/mcp_servers.json.", "warning"
                )
                return
            path = global_mcp_config_path(self.settings_store.root)
            if selected != MCP_NEW_SERVER_VALUE and selected != server.id:
                remove_server_config(path, selected, source="global")
            upsert_server_config(path, server, source="global")
        except (MCPConfigError, ValueError) as exc:
            self._set_mcp_status(str(exc), "error")
            return
        self._mcp_selected_server_id = server.id
        self._populate_mcp_controls()
        await self._ensure_agent_from_settings(rebuild=True)
        self._notify_user(f"Saved MCP server '{server.id}'.")

    async def _delete_mcp_server_from_ui(self) -> None:
        selected = self._selected_mcp_server_id()
        if selected == MCP_NEW_SERVER_VALUE:
            self._set_mcp_status("Select a user MCP server to delete.", "warning")
            return
        try:
            config = self._load_mcp_config_for_ui()
            existing = config.servers.get(selected)
            if existing is not None and existing.source == "project":
                self._set_mcp_status(
                    "Project MCP servers are read-only in the TUI; edit .kolega/mcp_servers.json.", "warning"
                )
                return
            removed = remove_server_config(global_mcp_config_path(self.settings_store.root), selected, source="global")
            MCPStatusStore(self.settings_store.root).clear(selected)
            MCPOAuthTokenStore(self.settings_store.root).clear(selected)
        except MCPConfigError as exc:
            self._set_mcp_status(str(exc), "error")
            return
        if not removed:
            self._set_mcp_status(f"No user MCP server named '{selected}' was found.", "warning")
            return
        self._mcp_selected_server_id = MCP_NEW_SERVER_VALUE
        self._populate_mcp_controls()
        await self._ensure_agent_from_settings(rebuild=True)
        self._notify_user(f"Deleted MCP server '{selected}'.")

    async def _verify_mcp_server_from_ui(self) -> None:
        selected = self._selected_mcp_server_id()
        if selected == MCP_NEW_SERVER_VALUE:
            self._set_mcp_status("Select a configured MCP server to verify.", "warning")
            return
        self._set_mcp_status(
            f"Verifying MCP server '{selected}'... stdio servers execute their configured command.", "warning"
        )
        config = self._load_mcp_config_for_ui()
        result = await MCPService(config, self.settings_store.root, self.project_path).verify_server(
            selected,
            interactive_oauth=True,
            open_browser=True,
            output=self.console,
        )
        self._populate_mcp_controls()
        await self._ensure_agent_from_settings(rebuild=True)
        if result.ok:
            message = f"Verified MCP server '{selected}' ({result.tool_count} tool(s))."
            if result.status is not None:
                note = mcp_tool_name_adjustment_note(selected, result.status.tools)
                if note:
                    message = f"{message} {note}"
            self._notify_user(message)
        else:
            self._notify_user(f"MCP verification failed for '{selected}': {result.message}", severity="warning")

    def _clear_mcp_tokens_from_ui(self) -> None:
        selected = self._selected_mcp_server_id()
        if selected == MCP_NEW_SERVER_VALUE:
            self._set_mcp_status("Select an MCP server before clearing tokens.", "warning")
            return
        MCPOAuthTokenStore(self.settings_store.root).clear(selected)
        self._set_mcp_status(f"Cleared stored MCP OAuth tokens for '{selected}'.", "ok")
        self._notify_user(f"Cleared MCP OAuth tokens for '{selected}'.")

    def _set_effort_select_default(
        self, provider: str, model: str, effort_id: str = "thinking_effort_select", *, preferred: Optional[str] = None
    ) -> None:
        try:
            effort_select = self._settings_query_one(f"#{effort_id}", Select)
        except Exception:
            return
        # Prefer an explicit value (a restored effort), else keep the current one if it
        # is still valid for this model, else fall back to the model's default. This
        # keeps a restore or a provider switch from clobbering the chosen effort.
        current = effort_select.value
        current = None if current is Select.NULL else str(current)
        effort_options = ui_thinking_effort_options(provider, model)
        valid_efforts = {value for _, value in effort_options}
        effort_select.set_options(effort_options)
        if preferred in valid_efforts:
            chosen = preferred
        elif current in valid_efforts:
            chosen = current
        else:
            chosen = default_ui_thinking_effort(provider, model)
        if chosen is not None:
            effort_select.value = chosen

    def _repopulate_model_select(
        self,
        provider: str,
        model_id: str,
        effort_id: Optional[str],
        *,
        model_value: Optional[str] = None,
        effort_value: Optional[str] = None,
    ) -> None:
        """Fill a provider→model→effort trio for a provider.

        Used by the global Model section and each per-agent row. ``model_value`` /
        ``effort_value`` pre-select a model/effort (used while restoring saved
        settings). Otherwise the select's current model is kept when it is still valid
        for ``provider`` (so a restore is not clobbered), falling back to the
        provider's first model. A catalogued model that is not among the listed
        (featured) options maps to the "Other…" entry backed by the row's custom
        input; on a non-gateway Browser row the stale model keeps its explicit
        "(vision required)" option instead."""
        try:
            model_select = self._settings_query_one(f"#{model_id}", Select)
        except NoMatches:
            return
        manual_switch = model_value is None
        if model_value is None:
            current = model_select.value
            model_value = None if current is Select.NULL else str(current)
        browser_role = model_id == "am_model_browser"
        model_options = settings_model_options(provider, vision_only=browser_role)
        custom_input_id = self._custom_input_id(model_id)
        custom_value: Optional[str] = None
        option_values = {value for _, value in model_options}
        if model_value and model_value not in option_values:
            stale_option = get_ui_model(provider, model_value)
            if stale_option is not None:
                if CUSTOM_MODEL_SENTINEL in option_values:
                    # Gateway row: surface any catalogued-but-unlisted saved model
                    # through "Other…" (non-featured ids, and on the Browser row
                    # saved non-vision models).
                    model_value = CUSTOM_MODEL_SENTINEL
                    custom_value = stale_option.model
                    if manual_switch and provider.startswith(CUSTOM_PROVIDER_PREFIX):
                        # A manual provider switch would carry the previous
                        # provider's model id into the free-text input; seed the
                        # endpoint's default model instead.
                        endpoint_id = provider[len(CUSTOM_PROVIDER_PREFIX) :]
                        endpoint_default = self._merged_endpoints_for_probe().get(endpoint_id, {}).get("default_model")
                        custom_value = str(endpoint_default) if endpoint_default else ""
                elif browser_role:
                    # Non-gateway Browser row: keep the explicit stale option so a
                    # saved non-vision model stays visible and selectable.
                    model_options.append((f"{stale_option.model_label} (vision required)", stale_option.model))
        if model_value == CUSTOM_MODEL_SENTINEL and custom_value is None:
            # A cascade re-entered while "Other…" was already selected (e.g. the
            # provider Changed posted during restore): keep what was typed.
            try:
                custom_input = self._settings_query_one(f"#{custom_input_id}", Input)
            except NoMatches:
                custom_input = None
            if custom_input is not None:
                custom_value = custom_input.value or None
        model_select.set_options(model_options)
        valid_models = {value for _, value in model_options}
        model = model_value if (model_value and model_value in valid_models) else None
        if model is None:
            model = model_options[0][1] if model_options else UI_DEFAULT_MODEL
        if model_options:
            model_select.value = model
        try:
            custom_input = self._settings_query_one(f"#{custom_input_id}", Input)
        except NoMatches:
            custom_input = None
        if custom_input is not None:
            custom_input.value = custom_value or ""
        # The sentinel itself has no spec; effort comes from the real id, and is
        # left untouched when no valid id is typed yet (mid-typing text must not
        # blank the effort select).
        effort_model = custom_value or model
        if (
            effort_id is not None
            and effort_model != CUSTOM_MODEL_SENTINEL
            and resolve_custom_model(provider, effort_model) is not None
        ):
            self._set_effort_select_default(provider, effort_model, effort_id, preferred=effort_value)
        self._sync_custom_model_input(model_id, custom_input_id)

    def _clear_model_effort_selects(self, model_id: str, effort_id: Optional[str]) -> None:
        """Blank an override row's model (and effort, when it has one) selects."""
        for select_id in (model_id, effort_id) if effort_id is not None else (model_id,):
            try:
                select = self._settings_query_one(f"#{select_id}", Select)
            except NoMatches:
                continue
            select.set_options([])
            select.value = Select.NULL
        self._sync_custom_model_input(model_id, self._custom_input_id(model_id))

    def _custom_input_id(self, model_select_id: str) -> str:
        """Return the custom-model Input id paired with a model Select id."""
        if model_select_id == "model_select":
            return "model_custom_input"
        row = _row_for_widget(model_select_id)
        # Every other model select in Settings belongs to an override row.
        assert row is not None, f"No override row owns model select '{model_select_id}'"
        return row.custom_id

    def _sync_custom_model_input(self, model_select_id: str, custom_input_id: str) -> None:
        """Show the custom-model input iff its select holds the "Other…" sentinel.

        A hidden input is cleared so it can never fake dirty state. Focus is a
        user-action concern and is handled by the select-change handlers, not here.
        """
        try:
            model_select = self._settings_query_one(f"#{model_select_id}", Select)
            custom_input = self._settings_query_one(f"#{custom_input_id}", Input)
        except NoMatches:
            return
        if str(model_select.value) == CUSTOM_MODEL_SENTINEL:
            custom_input.display = True
        else:
            custom_input.display = False
            custom_input.value = ""

    def _typed_custom_model(self, provider: str, custom_input_id: str) -> Optional[str]:
        """Resolve a custom input's current text to a catalogued model id."""
        try:
            custom_input = self._settings_query_one(f"#{custom_input_id}", Input)
        except NoMatches:
            return None
        return resolve_custom_model(provider, custom_input.value)

    def _custom_model_error(self, provider: str, model_select_id: str, custom_input_id: str) -> Optional[str]:
        """Return a blocking status message for an invalid "Other…" entry, else None."""
        try:
            model_select = self._settings_query_one(f"#{model_select_id}", Select)
            custom_input = self._settings_query_one(f"#{custom_input_id}", Input)
        except NoMatches:
            return None
        if str(model_select.value) != CUSTOM_MODEL_SENTINEL:
            return None
        typed = custom_input.value.strip()
        if not typed:
            return messages.MODEL_CUSTOM_EMPTY
        if resolve_custom_model(provider, typed) is None:
            return messages.MODEL_CUSTOM_UNKNOWN.format(model=typed, provider=provider)
        return None

    def _sync_effort_for_model_value(
        self,
        provider: str,
        value: str,
        model_select_id: str,
        effort_select_id: Optional[str],
        *,
        preferred: Optional[str] = None,
    ) -> None:
        """Apply an "Other…"-aware model selection to its effort select.

        The sentinel has no model spec, so while it is selected the effort
        select is only refreshed when the custom input already resolves to a
        catalogued id; otherwise it is left untouched. The custom input is
        focused only for genuine user selections, never while restoring.
        ``effort_select_id`` is None on rows that carry no effort control.
        """
        custom_input_id = self._custom_input_id(model_select_id)
        self._sync_custom_model_input(model_select_id, custom_input_id)
        if value != CUSTOM_MODEL_SENTINEL:
            if effort_select_id is not None:
                self._set_effort_select_default(provider, value, effort_select_id, preferred=preferred)
            return
        screen = getattr(self, "_settings_screen", None)
        initializing = bool(screen and getattr(screen, "_initializing", False))
        if not initializing:
            try:
                custom_input = self._settings_query_one(f"#{custom_input_id}", Input)
            except NoMatches:
                return
            self.call_after_refresh(custom_input.focus)
        if effort_select_id is None:
            return
        typed = self._typed_custom_model(provider, custom_input_id)
        if typed is not None:
            self._set_effort_select_default(provider, typed, effort_select_id, preferred=preferred)

    def _populate_lsp_controls(self) -> None:
        """Seed the LSP settings toggle from saved settings."""
        try:
            lsp_select = self._settings_query_one("#lsp_enabled_select", Select)
        except NoMatches:
            return
        enabled = self.settings.lsp_enabled
        if enabled is not None:
            lsp_select.value = "true" if enabled else "false"
        # Update the LSP status text
        self._update_lsp_settings_status()

    def _update_lsp_settings_status(self) -> None:
        """Show current LSP status in the settings panel."""
        try:
            status = self._settings_query_one("#lsp_status", Static)
        except NoMatches:
            return
        lsp_mode = getattr(self.overrides, "lsp_mode", None)
        if lsp_mode:
            status.update(
                f"LSP is forced {lsp_mode} for this session by the --lsp launch flag; "
                "the toggle applies from the next launch."
            )
            return
        agent = self.agent
        if agent is None or agent.tool_collection is None:
            status.update("LSP is not active. Enable it above and save settings.")
            return
        manager = agent.tool_collection.lsp_manager
        if manager is None or not manager.enabled:
            status.update("LSP is not active. Enable it above and save settings.")
            return
        lsp_status = manager.status()
        if not lsp_status.get("initialized"):
            status.update("LSP status will appear after the agent starts.")
            return
        detected_names = [d["display_name"] for d in lsp_status.get("detected", [])]
        missing_names = [m["display_name"] for m in lsp_status.get("missing", [])]
        incomplete = not lsp_status.get("scan_complete", True)
        incomplete_text = (
            "Detection incomplete "
            f"({lsp_status.get('scan_stop_reason') or 'scan limit'}, "
            f"{lsp_status.get('scanned_entries', 0)} entries)."
        )
        if detected_names:
            parts = [f"Detected: {', '.join(detected_names)}"]
            if incomplete:
                parts.append(incomplete_text)
            if missing_names:
                parts.append(f"Missing servers: {', '.join(missing_names)}")
            # Show active sessions with live state
            active = []
            for session in lsp_status.get("sessions", []):
                server_name = session["server_name"]
                if session.get("connected"):
                    active.append(server_name)
                elif session.get("status") == "error":
                    active.append(f"{server_name} (error)")
            if active:
                parts.append(f"Active: {', '.join(active)}")
            status.update(" ".join(parts))
        elif incomplete:
            status.update(incomplete_text)
        else:
            status.update("No supported languages detected in this project.")

    def _collect_lsp_from_ui(self) -> None:
        """Read the LSP toggle and save into settings."""
        try:
            value = str(self._settings_query_one("#lsp_enabled_select", Select).value)
        except NoMatches:
            return
        self.settings.lsp_enabled = value == "true"

    def _populate_subagents_controls(self) -> None:
        """Seed the sub-agents toggle from saved settings."""
        try:
            subagents_select = self._settings_query_one("#subagents_enabled_select", Select)
        except NoMatches:
            return
        enabled = self.settings.subagents_enabled
        if enabled is not None:
            subagents_select.value = "true" if enabled else "false"
        self._update_subagents_settings_status()

    def _update_subagents_settings_status(self) -> None:
        """Note when a launch flag or env var pins sub-agent dispatch for this session."""
        try:
            status = self._settings_query_one("#subagents_status", Static)
        except NoMatches:
            return
        flag_value = getattr(self.overrides, "subagents_mode", None)
        env_value = os.environ.get(SUBAGENTS_MODE_ENV)
        forced = flag_value or env_value
        if forced:
            source = "--subagents" if flag_value else SUBAGENTS_MODE_ENV
            status.update(
                f"Sub-agents are forced {forced} for this session by {source}; "
                "the setting applies from the next launch."
            )
            return
        status.update("")

    def _collect_subagents_from_ui(self) -> None:
        """Read the sub-agents toggle and save into settings."""
        try:
            value = str(self._settings_query_one("#subagents_enabled_select", Select).value)
        except NoMatches:
            return
        self.settings.subagents_enabled = value == "true"

    def _populate_skills_controls(self) -> None:
        """Seed the Agent Skills toggle from saved settings."""
        try:
            skills_select = self._settings_query_one("#skills_enabled_select", Select)
        except NoMatches:
            return
        enabled = self.settings.skills_enabled
        if enabled is not None:
            skills_select.value = "true" if enabled else "false"
        self._update_skills_settings_status()

    def _update_skills_settings_status(self) -> None:
        """Note when a launch flag or env var pins Agent Skills for this session."""
        try:
            status = self._settings_query_one("#skills_status", Static)
        except NoMatches:
            return
        flag_value = getattr(self.overrides, "skills_mode", None)
        env_value = os.environ.get(SKILLS_MODE_ENV)
        forced = flag_value or env_value
        if forced:
            source = "--skills" if flag_value else SKILLS_MODE_ENV
            status.update(
                f"Agent Skills are forced {forced} for this session by {source}; "
                "the setting applies from the next launch."
            )
            return
        status.update("")

    def _collect_skills_from_ui(self) -> None:
        """Read the Agent Skills toggle and save it into settings."""
        try:
            value = str(self._settings_query_one("#skills_enabled_select", Select).value)
        except NoMatches:
            return
        self.settings.skills_enabled = value == "true"

    def _populate_compression_controls(self) -> None:
        """Seed the compression-threshold select from saved settings."""
        try:
            select = self._settings_query_one("#compression_threshold_select", Select)
        except NoMatches:
            return
        saved = self.settings.compression_threshold
        select.set_options(compression_threshold_options(saved))
        select.value = (
            COMPRESSION_THRESHOLD_DEFAULT_VALUE if saved is None else _compression_threshold_option_value(saved)
        )
        self._update_compression_settings_status()

    def _update_compression_settings_status(self) -> None:
        """Note when a launch flag or env var pins the threshold for this session."""
        try:
            status = self._settings_query_one("#compression_status", Static)
        except NoMatches:
            return
        flag_value = getattr(self.overrides, "compression_threshold", None)
        env_value = os.environ.get(COMPRESSION_THRESHOLD_ENV)
        forced = flag_value or env_value
        if forced:
            source = "--compression-threshold" if flag_value else COMPRESSION_THRESHOLD_ENV
            status.update(
                f"Compression threshold is forced to {forced}% for this session by {source}; "
                "the setting applies from the next launch."
            )
            return
        status.update("")

    def _collect_compression_from_ui(self) -> None:
        """Read the compression-threshold select and save into settings."""
        try:
            value = str(self._settings_query_one("#compression_threshold_select", Select).value)
        except NoMatches:
            return
        self.settings.compression_threshold = None if value == COMPRESSION_THRESHOLD_DEFAULT_VALUE else float(value)

    def _settings_candidate_from_ui(self) -> tuple[CliSettings, str, str, str]:
        """Collect the mounted form into a detached settings candidate."""
        provider = str(self._settings_query_one("#provider_select", Select).value)
        model_value = str(self._settings_query_one("#model_select", Select).value)
        if model_value == CUSTOM_MODEL_SENTINEL:
            # Save is pre-validated in _save_settings_from_ui, so the typed id
            # resolves to a catalogued model here; the sentinel never persists.
            model = self._typed_custom_model(provider, "model_custom_input") or model_value
        else:
            model = model_value
        effort = str(self._settings_query_one("#thinking_effort_select", Select).value)
        valid_efforts = {value for _, value in ui_thinking_effort_options(provider, model)}
        if effort not in valid_efforts:
            effort = default_ui_thinking_effort(provider, model) or ""
        original = self.settings
        candidate = deepcopy(original)
        screen = getattr(self, "_settings_screen", None)
        if screen is not None:
            candidate.oauth_tokens = deepcopy(screen.pending_oauth_tokens)
            candidate.custom_endpoints = deepcopy(screen.pending_custom_endpoints)
            for removed_provider in screen.pending_api_key_removals:
                candidate.api_keys.pop(removed_provider, None)
            # Removals first, so re-typing a key for a provider you just cleared wins.
            for staged_provider, staged_key in screen.pending_api_keys.items():
                candidate.set_api_key(staged_provider, staged_key)
        self.settings = candidate
        try:
            candidate.active_provider = provider
            candidate.active_model = model
            candidate.active_thinking_effort = effort or default_ui_thinking_effort(provider, model)
            candidate.active_theme = str(self._settings_query_one("#theme_select", Select).value)
            self._collect_agent_models_from_ui()
            self._collect_model_slots_from_ui()
            self._collect_web_search_from_ui()
            self._collect_lsp_from_ui()
            self._collect_subagents_from_ui()
            self._collect_skills_from_ui()
            self._collect_compression_from_ui()
            self._collect_gateway_from_ui()
        finally:
            self.settings = original
        return candidate, provider, model, effort

    async def _save_settings_from_ui(self) -> None:
        if self._turn_active or self.agent_worker is not None:
            self._set_settings_status("Stop the active turn before applying settings.", "warning")
            return
        browser_message, browser_tone, browser_blocks_save = self._browser_model_status()
        if browser_blocks_save:
            self._update_browser_model_hint()
            self._set_settings_status(browser_message, "error")
            self._notify_user(browser_message, severity="error")
            return
        # A gateway token is a BotFather credential; catch obviously wrong
        # values before anything is written.
        try:
            typed_token = self._settings_query_one("#gateway_token_input", Input).value.strip()
        except NoMatches:
            typed_token = ""
        if typed_token:
            from kolega_code.gateway.adapters.telegram.adapter import validate_bot_token

            try:
                validate_bot_token(typed_token)
            except ValueError as exc:
                self._set_settings_status(str(exc), "error")
                self._notify_user(str(exc), severity="error")
                return
        # "Other…" entries are UI-only: every typed id must resolve to a
        # catalogued model before anything is written, so an unknown id can
        # never produce a build-time "Configuration incomplete" lockout.
        error = self._custom_model_error(
            str(self._settings_query_one("#provider_select", Select).value), "model_select", "model_custom_input"
        )
        if error is None:
            rows = [_agent_row(role) for _, role in agent_role_options()]
            rows.extend(_slot_row(slot) for _, slot in model_slot_options())
            for row in rows:
                try:
                    row_provider = str(self._settings_query_one(f"#{row.provider_id}", Select).value)
                except NoMatches:
                    continue
                if row_provider == INHERIT_SENTINEL:
                    continue
                error = self._custom_model_error(row_provider, row.model_id, row.custom_id)
                if error is not None:
                    break
        if error is not None:
            self._set_settings_status(error, "error")
            self._notify_user(error, severity="error")
            return
        candidate, provider, _model, _effort = self._settings_candidate_from_ui()

        ok, error = await self._apply_settings_candidate(candidate, rebuild=True)
        if not ok:
            self._set_settings_status(messages.SETTINGS_INCOMPLETE.format(error=error), "error")
            return
        try:
            self._settings_query_one("#provider_api_key_input", Input).value = ""
        except NoMatches:
            pass
        try:
            self._settings_query_one("#web_search_api_key_input", Input).value = ""
        except NoMatches:
            pass
        screen = getattr(self, "_settings_screen", None)
        if screen is not None:
            memory_applied = await screen.apply_memory_draft()
            screen.mark_clean(preserve_memory_draft=not memory_applied)
            # New/changed endpoints alter the provider set and pickers; rebuild.
            self._populate_provider_rows()
            self._populate_endpoint_controls()
            self._update_provider_credential_controls(self._selected_credential_provider() or "")
            if not memory_applied:
                self._set_settings_status(
                    "Other settings were saved, but project memory changes failed. Review and retry them.",
                    "warning",
                )
                return
        if self.config is not None:
            override_message = active_model_override_message(
                self.config,
                self.project_path,
                self.overrides,
                self.settings,
            )
            if override_message:
                self._notify_user(f"{messages.SETTINGS_SAVED} {override_message}", severity="warning")
            elif browser_tone == "warning":
                self._notify_user(f"{messages.SETTINGS_SAVED} {browser_message}", severity="warning")
            else:
                self._notify_user(messages.SETTINGS_SAVED)

    async def _apply_settings_candidate(self, candidate: CliSettings, *, rebuild: bool = True) -> tuple[bool, str]:
        """Validate, persist, and activate a settings candidate without partial writes."""
        if self._turn_active or self.agent_worker is not None:
            return False, "Stop the active turn before applying settings."
        try:
            build_agent_config(
                self.project_path,
                self.overrides,
                settings=candidate,
                settings_store=self.settings_store,
            )
        except CliConfigError as exc:
            return False, str(exc)
        try:
            self.settings_store.save(candidate)
        except Exception as exc:
            return False, str(exc)
        self.settings = candidate
        self.startup_config_error = None
        await self._ensure_agent_from_settings(rebuild=rebuild)
        self._refresh_settings_summary()
        return (
            self.config is not None,
            "" if self.config is not None else "The model configuration could not be activated.",
        )

    def _settings_remove_api_key(self) -> None:
        screen = getattr(self, "_settings_screen", None)
        if screen is None:
            return
        if self._turn_active or self.agent_worker is not None:
            self._set_settings_status("Stop the active turn before changing credentials.", "warning")
            return
        provider = self._selected_credential_provider()
        status = self._settings_query_one("#provider_credential_status", Static)
        if provider is None:
            return
        if not self.settings.has_api_key(provider):
            status.update(messages.PROVIDER_KEY_NOT_STORED)
            return
        screen.pending_api_key_removals.add(provider)
        screen.pending_api_keys.pop(provider, None)
        self._settings_query_one("#provider_api_key_input", Input).value = ""
        self._update_provider_credential_controls(provider)
        self._refresh_provider_row_labels()
        status.update(messages.PROVIDER_KEY_REMOVAL_STAGED)
        screen._refresh_apply_label()

    async def _settings_login_chatgpt(self) -> None:
        if self._turn_active or self.agent_worker is not None:
            self._set_settings_status("Stop the active turn before signing in.", "warning")
            return
        screen = getattr(self, "_settings_screen", None)
        if screen is None:
            return
        button = self._settings_query_one("#provider_chatgpt_login", Button)
        status = self._settings_query_one("#provider_credential_status", Static)
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
        screen.pending_oauth_tokens[chatgpt_constants.PROVIDER_KEY] = tokens.model_dump(mode="json")
        self._update_provider_credential_controls(chatgpt_constants.PROVIDER_KEY)
        self._refresh_provider_row_labels()
        status.update(f"Signed in as {tokens.email or 'your ChatGPT account'}. Apply to save this sign-in.")
        button.label = "Sign in again"
        button.disabled = False
        screen._refresh_apply_label()

    def _settings_logout_chatgpt(self) -> None:
        screen = getattr(self, "_settings_screen", None)
        if screen is None:
            return
        if self._turn_active or self.agent_worker is not None:
            self._set_settings_status("Stop the active turn before signing out.", "warning")
            return
        removed = screen.pending_oauth_tokens.pop(chatgpt_constants.PROVIDER_KEY, None)
        status = self._settings_query_one("#provider_credential_status", Static)
        self._update_provider_credential_controls(chatgpt_constants.PROVIDER_KEY)
        self._refresh_provider_row_labels()
        status.update("ChatGPT sign-out will be saved when you Apply." if removed else "You are not signed in.")
        screen._refresh_apply_label()

    def _merged_endpoints_for_probe(self) -> dict[str, dict]:
        env = load_cli_env(self.project_path)
        overrides = getattr(self, "overrides", None) or CliConfigOverrides()
        return _merged_custom_endpoints(env, overrides, self.settings)

    def _credential_probe_model(self, provider: str) -> str:
        """Which model to probe a provider's credential with.

        The active model when it belongs to this provider — that is the one that
        matters — otherwise the provider's registry default. This is a probe target,
        not configuration: it pins nothing and is never written anywhere.
        """
        if self.settings.active_provider == provider and self.settings.active_model:
            return self.settings.active_model
        if provider.startswith(CUSTOM_PROVIDER_PREFIX):
            endpoint_id = provider[len(CUSTOM_PROVIDER_PREFIX) :]
            entry = self._merged_endpoints_for_probe().get(endpoint_id) or {}
            if entry.get("default_model"):
                return str(entry["default_model"])
            models = entry.get("models") or {}
            if models:
                return str(next(iter(models)))
            raise ValueError(
                f"Custom endpoint '{endpoint_id}' has no default model; add one on the Custom Endpoints page."
            )
        return default_model_for_provider(ModelProvider(provider))

    async def _test_settings_connection(self) -> None:
        """Probe the highlighted provider's credential, independent of model config."""
        if self._turn_active or self.agent_worker is not None:
            self._set_settings_status("Stop the active turn before testing a connection.", "warning")
            return
        status = self._settings_query_one("#provider_credential_status", Static)
        button = self._settings_query_one("#provider_test_connection", Button)
        provider = self._selected_credential_provider()
        if provider is None:
            return
        draft = self._draft_credential_settings()
        try:
            model = self._credential_probe_model(provider)
        except ValueError as exc:
            status.update(str(exc))
            return
        base_url = None
        api_style = None
        if provider.startswith(CUSTOM_PROVIDER_PREFIX):
            endpoint_id = provider[len(CUSTOM_PROVIDER_PREFIX) :]
            entry = self._merged_endpoints_for_probe().get(endpoint_id) or {}
            base_url = entry.get("base_url")
            api_style = entry.get("api_style")
        button.disabled = True
        status.update(messages.PROVIDER_TEST_RUNNING.format(provider=provider, model=model))
        result = await test_model_connection(
            ModelProvider(provider),
            model,
            api_key=resolved_api_key(provider, self.project_path, draft) or "",
            token_manager=probe_token_manager(draft),
            usage_ledger=getattr(self, "_usage_ledger", None),
            base_url=base_url,
            api_style=api_style,
        )
        status.update(result.message)
        button.disabled = False

    def _collect_agent_models_from_ui(self) -> None:
        """Write each per-agent row into settings.agent_models (inherit rows removed)."""
        for _, role in agent_role_options():
            try:
                provider = str(self._settings_query_one(f"#am_provider_{role}", Select).value)
                model_select = self._settings_query_one(f"#am_model_{role}", Select)
                effort_select = self._settings_query_one(f"#am_effort_{role}", Select)
            except NoMatches:
                continue
            if provider == INHERIT_SENTINEL or model_select.value is Select.NULL:
                self.settings.clear_agent_model(role)
                continue
            model_value = str(model_select.value)
            if model_value == CUSTOM_MODEL_SENTINEL:
                # Save is pre-validated in _save_settings_from_ui; the defensive
                # fallback clears the override rather than persisting the sentinel.
                model = self._typed_custom_model(provider, f"am_custom_model_{role}")
                if model is None:
                    self.settings.clear_agent_model(role)
                    continue
            else:
                model = model_value
            effort = "" if effort_select.value is Select.NULL else str(effort_select.value)
            valid_efforts = {value for _, value in ui_thinking_effort_options(provider, model)}
            if effort not in valid_efforts:
                effort = default_ui_thinking_effort(provider, model) or ""
            self.settings.set_agent_model(role, provider, model, effort or None)

    def _collect_model_slots_from_ui(self) -> None:
        """Write each model-slot row into settings.model_slots (inherit rows removed)."""
        for _, slot in model_slot_options():
            row = _slot_row(slot)
            try:
                provider = str(self._settings_query_one(f"#{row.provider_id}", Select).value)
                model_select = self._settings_query_one(f"#{row.model_id}", Select)
            except NoMatches:
                continue
            if provider == INHERIT_SENTINEL or model_select.value is Select.NULL:
                self.settings.clear_model_slot(slot)
                continue
            model_value = str(model_select.value)
            if model_value == CUSTOM_MODEL_SENTINEL:
                # Save is pre-validated in _save_settings_from_ui; the defensive
                # fallback clears the override rather than persisting the sentinel.
                model = self._typed_custom_model(provider, row.custom_id)
                if model is None:
                    self.settings.clear_model_slot(slot)
                    continue
            else:
                model = model_value
            self.settings.set_model_slot(slot, provider, model)

    def _set_settings_status(self, text: str, tone: str = "info") -> None:
        """Update the settings status with a tone glyph in the semantic palette."""
        glyph, style = {
            "ok": (Glyph.CHECK, Color.SUCCESS),
            "error": (Glyph.CROSS, Color.ERROR),
            "warning": (Glyph.STATUS, Color.WARNING),
        }.get(tone, (Glyph.STATUS, Color.MUTED))
        content = Text()
        content.append(theme.g(glyph) + " ", style=style)
        content.append(text)
        try:
            self._settings_status.update(content)
        except Exception:
            pass
        try:
            summary_status = self.default_screen.query_one("#settings_summary_status", Static)
            summary_status.update(content if self.config is None or tone == "error" else "")
        except Exception:
            pass

    def _refresh_settings_summary(self) -> None:
        """Render the small, read-only Settings sidebar card."""
        try:
            summary = self.default_screen.query_one("#settings_summary", Static)
            launch = self.default_screen.query_one("#open_settings", Button)
        except Exception:
            return

        saved_provider = self.settings.active_provider
        saved_model = self.settings.active_model
        if self.config is not None:
            effective_provider = self.config.long_context_config.provider.value
            effective_model = self.config.long_context_config.model
            model_line = f"Model: {effective_provider}/{effective_model}"
            if (
                (saved_provider, saved_model) != (effective_provider, effective_model)
                and saved_provider
                and saved_model
            ):
                model_line += f"\nSaved default: {saved_provider}/{saved_model}"
            credential = key_status(effective_provider, self.project_path, self.settings)
        elif saved_provider and saved_model:
            model_line = f"Model: {saved_provider}/{saved_model} (not connected)"
            credential = key_status(saved_provider, self.project_path, self.settings)
        else:
            model_line = "Model: not connected"
            credential = "not configured"

        override_count = len(self.settings.agent_models)
        search_backend = self.settings.web_search_backend or DEFAULT_WEB_SEARCH_BACKEND
        lsp_enabled = self.settings.lsp_enabled is not False
        theme_name = self.settings.active_theme or theme.DEFAULT_THEME_NAME
        try:
            mcp_config = self._load_mcp_config_for_ui()
            rows = MCPService(mcp_config, self.settings_store.root, self.project_path).list_status_rows()
            enabled = sum(1 for row in rows if bool(row.get("enabled")))
            mcp_line = f"MCP: {enabled}/{len(rows)} enabled"
        except Exception:
            mcp_line = "MCP: unavailable"
        lines = [
            model_line,
            f"Credential: {credential}",
            f"Agent overrides: {override_count}",
        ]
        # Only shown once a slot is pinned; inheriting slots are the quiet default.
        slots = self.settings.model_slots
        if slots:
            lines.append("Model slots: " + ", ".join(sorted(slots)))
        lines.extend(
            [
                f"Web search: {search_backend}",
                mcp_line,
                f"LSP: {'enabled' if lsp_enabled else 'disabled'}",
                f"Sub-agents: {'enabled' if self.settings.subagents_enabled is not False else 'disabled'}",
                f"Agent Skills: {'enabled' if self.settings.skills_enabled is not False else 'disabled'}",
                f"Theme: {theme_name}",
            ]
        )
        summary.update("\n".join(lines))
        launch.label = "Open Settings →" if self.config is not None else "Continue Setup →"

    def _update_settings_status(self) -> None:
        if not (self.settings.active_provider and self.settings.active_model):
            text = "\n".join(
                [
                    messages.SETTINGS_ACTIVE_MODEL_UNCONFIGURED,
                    messages.SETTINGS_THINKING_EFFORT_LINE.format(effort="not configured"),
                    messages.SETTINGS_API_KEY_LINE.format(status="not checked until a model is configured"),
                ]
            )
            self._set_settings_status(text, "warning")
            self._refresh_status_dashboard()
            self._refresh_settings_summary()
            return

        provider = self.settings.active_provider
        model = self.settings.active_model
        effort = self.settings.active_thinking_effort or default_ui_thinking_effort(provider, model) or "not supported"
        status = key_status(provider, self.project_path, self.settings)
        tone = "warning" if "missing" in status.lower() else "ok"
        lines = [
            messages.SETTINGS_ACTIVE_MODEL.format(provider=provider, model=model),
            messages.SETTINGS_THINKING_EFFORT_LINE.format(effort=effort),
            messages.SETTINGS_API_KEY_LINE.format(status=status),
        ]
        if self.config is not None:
            override_message = active_model_override_message(
                self.config,
                self.project_path,
                self.overrides,
                self.settings,
            )
            if override_message:
                lines.append(override_message)
                tone = "warning"
        browser_message, browser_tone, _ = self._browser_model_status()
        if browser_message and browser_tone in {"warning", "error"}:
            lines.append(browser_message)
            tone = browser_tone
        self._set_settings_status("\n".join(lines), tone)
        self._refresh_status_dashboard()
        self._refresh_settings_summary()

    def _api_key_placeholder(self, provider: str) -> str:
        if provider == chatgpt_constants.PROVIDER_KEY:
            # OAuth provider: no API key — the field is informational only.
            if self.settings.has_oauth_token(provider):
                return "Signed in with ChatGPT — run /login chatgpt to switch accounts"
            return "Run /login chatgpt to sign in with your ChatGPT subscription"
        if self.settings.has_api_key(provider):
            return "Stored API key will be kept if blank"
        model = get_ui_model(provider, (ui_model_options(provider) or [("", "")])[0][1])
        return f"{model.provider_label} API key" if model else "API key"
