"""Settings panel behavior for the CLI TUI."""

from __future__ import annotations

import json
import re
import shlex
from copy import deepcopy
from typing import Literal, Optional, TypeVar, cast, overload
from urllib.parse import urlparse

from textual.css.query import NoMatches
from textual.widget import Widget
from rich.text import Text
from textual.widgets import Button, Input, Select, Static

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.chatgpt_oauth import run_login_flow
from kolega_code.agent.tool_backend.search_backends import (
    DEFAULT_BACKEND as DEFAULT_WEB_SEARCH_BACKEND,
    SearchBackendError,
    available_backends,
    get_backend_class,
)
from kolega_code.mcp.config import (
    MCPConfigError,
    MCPOAuthConfig,
    MCPServerConfig,
    global_mcp_config_path,
    load_mcp_config,
    remove_server_config,
    set_server_enabled,
    upsert_server_config,
)
from kolega_code.mcp.service import MCPService
from kolega_code.mcp.state import MCPStatusStore, MCPOAuthTokenStore

from .. import messages, theme
from ..config import CliConfigError, active_model_override_message, build_agent_config, key_status
from ..model_connection import test_model_connection
from ..provider_registry import (
    INHERIT_SENTINEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    get_ui_model,
    agent_role_options,
    default_ui_thinking_effort,
    ui_model_options,
    ui_provider_options,
    ui_thinking_effort_options,
)
from . import app_base as tui_app_base
from ..settings import WEB_SEARCH_KEY_NAMES, CliSettings
from ..theme import Color, Glyph

MCP_NEW_SERVER_VALUE = "__new_mcp_server__"
MCP_TRANSPORT_OPTIONS = [
    ("Streamable HTTP", "streamable_http"),
    ("Server-Sent Events (legacy/deprecated)", "sse"),
    ("stdio command", "stdio"),
]
MCP_ENABLED_OPTIONS = [("Enabled", "true"), ("Disabled", "false")]
MCP_STATUS_MESSAGE_MAX = 96
MCP_STATUS_NAME_MAX = 34
MCP_ATTENTION_STATUSES = {"failed", "stale", "unverified"}
MCP_TRANSPORT_LABELS = {
    "streamable_http": "HTTP",
    "sse": "SSE",
    "stdio": "stdio",
}

SettingsWidget = TypeVar("SettingsWidget", bound=Widget)


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
        """Query controls on the open Settings screen, falling back for legacy tests."""
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
            self._repopulate_model_select(provider, "model_select", "thinking_effort_select")
            self._update_browser_model_hint()
            try:
                api_key_input = self._settings_query_one("#api_key_input", Input)
                api_key_input.placeholder = self._api_key_placeholder(provider)
                # OAuth providers sign in via /login, so the key field is read-only.
                api_key_input.disabled = provider == chatgpt_constants.PROVIDER_KEY
            except NoMatches:
                pass
            self._update_model_auth_controls(provider)
            return

        if select_id == "model_select":
            try:
                provider = str(self._settings_query_one("#provider_select", Select).value)
            except NoMatches:
                return
            self._set_effort_select_default(provider, str(event.value))
            self._update_browser_model_hint()
            return

        if select_id == "web_search_backend_select":
            self._update_search_backend_fields(str(event.value))
            return

        if select_id == "mcp_server_select":
            self._populate_mcp_server_form(str(event.value))
            return

        if select_id == "mcp_transport_select":
            self._update_mcp_transport_fields(str(event.value))
            return

        if select_id.startswith("am_provider_"):
            role = select_id[len("am_provider_") :]
            provider = str(event.value)
            if str(event.select.value) != provider:
                return
            if provider == INHERIT_SENTINEL:
                # Don't drop pending here: the selects post an initial inherit-valued
                # Changed on mount, which would clear a restore before its real cascade
                # runs. _populate_agent_model_rows clears stale pending per row instead.
                self._clear_model_effort_selects(f"am_model_{role}", f"am_effort_{role}")
            else:
                model_value = self._pending_agent_models.pop(f"am_model_{role}", None)
                self._repopulate_model_select(
                    provider, f"am_model_{role}", f"am_effort_{role}", model_value=model_value
                )
            if role == "browser":
                self._update_browser_model_hint()
            return

        if select_id.startswith("am_model_"):
            role = select_id[len("am_model_") :]
            try:
                provider = str(self._settings_query_one(f"#am_provider_{role}", Select).value)
            except NoMatches:
                return
            if provider != INHERIT_SENTINEL and event.value is not Select.NULL:
                # A restored effort waits here for the model that hosts it; a manual
                # model change has none pending and falls back to preserve/default.
                preferred = self._pending_agent_efforts.pop(f"am_effort_{role}", None)
                self._set_effort_select_default(provider, str(event.value), f"am_effort_{role}", preferred=preferred)
            if role == "browser":
                self._update_browser_model_hint()
            return

        if select_id == "theme_select":
            name = str(event.value)
            if str(event.select.value) != name:
                return
            if name != theme.active_theme().name:
                self._apply_theme(name)

    def _populate_settings_controls(self) -> None:
        provider_values = {value for _, value in ui_provider_options()}
        provider = (
            self.settings.active_provider if self.settings.active_provider in provider_values else UI_DEFAULT_PROVIDER
        )
        model_options = ui_model_options(provider)
        valid_models = {value for _, value in model_options}
        model = self.settings.active_model if self.settings.active_model in valid_models else None
        if model is None:
            model = model_options[0][1] if model_options else UI_DEFAULT_MODEL
        effort_options = {value for _, value in ui_thinking_effort_options(provider, model)}
        effort = (
            self.settings.active_thinking_effort if self.settings.active_thinking_effort in effort_options else None
        )
        if effort is None:
            effort = default_ui_thinking_effort(provider, model)
        provider_select = self._settings_query_one("#provider_select", Select)
        model_select = self._settings_query_one("#model_select", Select)
        effort_select = self._settings_query_one("#thinking_effort_select", Select)
        api_key_input = self._settings_query_one("#api_key_input", Input)

        provider_select.value = provider
        model_select.set_options(model_options)
        model_select.value = model
        effort_select.set_options(ui_thinking_effort_options(provider, model))
        if effort is not None:
            effort_select.value = effort
        theme_select = self._settings_query_one("#theme_select", Select)
        theme_select.value = (
            self.settings.active_theme
            if self.settings.active_theme in theme.available_themes()
            else theme.DEFAULT_THEME_NAME
        )
        api_key_input.placeholder = self._api_key_placeholder(provider)
        api_key_input.disabled = provider == chatgpt_constants.PROVIDER_KEY
        self._update_model_auth_controls(provider)
        self._populate_agent_model_rows()
        self._update_browser_model_hint()
        self._populate_web_search_controls()
        self._populate_mcp_controls()
        self._populate_lsp_controls()
        self._update_settings_status()

    def _update_model_auth_controls(self, provider: str) -> None:
        oauth = provider == chatgpt_constants.PROVIDER_KEY
        for widget_id in ("settings_chatgpt_login", "settings_chatgpt_logout"):
            try:
                self._settings_query_one(f"#{widget_id}").display = oauth
            except NoMatches:
                pass
        for widget_id in ("settings_api_key_label", "api_key_input"):
            try:
                self._settings_query_one(f"#{widget_id}").display = not oauth
            except NoMatches:
                pass
        try:
            remove_key = self._settings_query_one("#settings_remove_api_key", Button)
        except NoMatches:
            return
        remove_key.display = not oauth
        screen = getattr(self, "_settings_screen", None)
        pending_removals = getattr(screen, "pending_api_key_removals", set())
        remove_key.disabled = not self.settings.has_api_key(provider) or provider in pending_removals
        if screen is None:
            return
        draft = deepcopy(self.settings)
        draft.oauth_tokens = deepcopy(screen.pending_oauth_tokens)
        for removed_provider in pending_removals:
            draft.api_keys.pop(removed_provider, None)
        try:
            status = self._settings_query_one("#settings_connection_status", Static)
            status.update(f"Credential status: {key_status(provider, self.project_path, draft)}")
            login = self._settings_query_one("#settings_chatgpt_login", Button)
            logout = self._settings_query_one("#settings_chatgpt_logout", Button)
            signed_in = draft.has_oauth_token(chatgpt_constants.PROVIDER_KEY)
            login.label = "Sign in again" if signed_in else "Sign in with ChatGPT"
            logout.disabled = not signed_in
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
                model = str(self._settings_query_one("#model_select", Select).value)
            except NoMatches:
                return "", "info", False
        else:
            provider = browser_provider
            try:
                model_value = self._settings_query_one("#am_model_browser", Select).value
            except NoMatches:
                return "", "info", False
            if model_value is Select.NULL:
                return messages.BROWSER_MODEL_PROVIDER_NO_VISION.format(provider=provider), "error", True
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
        valid = {name for _, name in available_backends()}
        backend = self.settings.web_search_backend
        if backend not in valid:
            backend = DEFAULT_WEB_SEARCH_BACKEND
        try:
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
            backend = str(self._settings_query_one("#web_search_backend_select", Select).value)
            base_url_input = self._settings_query_one("#web_search_base_url_input", Input)
            key_input = self._settings_query_one("#web_search_api_key_input", Input)
        except NoMatches:
            return
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
        if button_id in {"mcp_enable_server", "mcp_disable_server"}:
            await self._set_mcp_enabled_from_ui(enabled=button_id == "mcp_enable_server")
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

    async def _set_mcp_enabled_from_ui(self, *, enabled: bool) -> None:
        selected = self._selected_mcp_server_id()
        if selected == MCP_NEW_SERVER_VALUE:
            self._set_mcp_status("Select a user MCP server first.", "warning")
            return
        try:
            config = self._load_mcp_config_for_ui()
            existing = config.servers.get(selected)
            if existing is not None and existing.source == "project":
                self._set_mcp_status(
                    "Project MCP servers are read-only in the TUI; edit .kolega/mcp_servers.json.", "warning"
                )
                return
            changed = set_server_enabled(
                global_mcp_config_path(self.settings_store.root), selected, enabled, source="global"
            )
        except MCPConfigError as exc:
            self._set_mcp_status(str(exc), "error")
            return
        if not changed:
            self._set_mcp_status(f"No user MCP server named '{selected}' was found.", "warning")
            return
        self._populate_mcp_controls()
        await self._ensure_agent_from_settings(rebuild=True)
        self._notify_user(f"{'Enabled' if enabled else 'Disabled'} MCP server '{selected}'.")

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
            self._notify_user(f"Verified MCP server '{selected}' ({result.tool_count} tool(s)).")
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
        effort_id: str,
        *,
        model_value: Optional[str] = None,
        effort_value: Optional[str] = None,
    ) -> None:
        """Fill a provider→model→effort trio for a provider.

        Used by the global Model section and each per-agent row. ``model_value`` /
        ``effort_value`` pre-select a model/effort (used while restoring saved
        settings). Otherwise the select's current model is kept when it is still valid
        for ``provider`` (so a restore is not clobbered), falling back to the
        provider's first model."""
        try:
            model_select = self._settings_query_one(f"#{model_id}", Select)
        except NoMatches:
            return
        if model_value is None:
            current = model_select.value
            model_value = None if current is Select.NULL else str(current)
        browser_role = model_id == "am_model_browser"
        model_options = ui_model_options(provider, vision_only=True) if browser_role else ui_model_options(provider)
        valid_models = {value for _, value in model_options}
        if browser_role and model_value and model_value not in valid_models:
            stale_option = get_ui_model(provider, model_value)
            if stale_option is not None:
                model_options.append((f"{stale_option.model_label} (vision required)", model_value))
        model_select.set_options(model_options)
        valid_models = {value for _, value in model_options}
        model = model_value if (model_value and model_value in valid_models) else None
        if model is None:
            model = model_options[0][1] if model_options else UI_DEFAULT_MODEL
        if model_options:
            model_select.value = model
        self._set_effort_select_default(provider, model, effort_id, preferred=effort_value)

    def _clear_model_effort_selects(self, model_id: str, effort_id: str) -> None:
        """Blank a per-agent row's model+effort selects (the role inherits)."""
        for select_id in (model_id, effort_id):
            try:
                select = self._settings_query_one(f"#{select_id}", Select)
            except NoMatches:
                continue
            select.set_options([])
            select.value = Select.NULL

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

    def _settings_candidate_from_ui(self) -> tuple[CliSettings, Input, str, str, str]:
        """Collect the mounted form into a detached settings candidate."""
        provider = str(self._settings_query_one("#provider_select", Select).value)
        model = str(self._settings_query_one("#model_select", Select).value)
        effort = str(self._settings_query_one("#thinking_effort_select", Select).value)
        valid_efforts = {value for _, value in ui_thinking_effort_options(provider, model)}
        if effort not in valid_efforts:
            effort = default_ui_thinking_effort(provider, model) or ""
        api_key_input = self._settings_query_one("#api_key_input", Input)

        original = self.settings
        candidate = deepcopy(original)
        screen = getattr(self, "_settings_screen", None)
        if screen is not None:
            candidate.oauth_tokens = deepcopy(screen.pending_oauth_tokens)
            for removed_provider in screen.pending_api_key_removals:
                candidate.api_keys.pop(removed_provider, None)
        self.settings = candidate
        try:
            candidate.active_provider = provider
            candidate.active_model = model
            candidate.active_thinking_effort = effort or default_ui_thinking_effort(provider, model)
            candidate.active_theme = str(self._settings_query_one("#theme_select", Select).value)
            api_key = api_key_input.value.strip()
            if api_key:
                candidate.set_api_key(provider, api_key)
            self._collect_agent_models_from_ui()
            self._collect_web_search_from_ui()
            self._collect_lsp_from_ui()
        finally:
            self.settings = original
        return candidate, api_key_input, provider, model, effort

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
        candidate, api_key_input, provider, _model, _effort = self._settings_candidate_from_ui()

        ok, error = await self._apply_settings_candidate(candidate, rebuild=True)
        if not ok:
            self._set_settings_status(messages.SETTINGS_INCOMPLETE.format(error=error), "error")
            return
        api_key_input.value = ""
        api_key_input.placeholder = self._api_key_placeholder(provider)
        try:
            self._settings_query_one("#web_search_api_key_input", Input).value = ""
        except NoMatches:
            pass
        screen = getattr(self, "_settings_screen", None)
        if screen is not None:
            screen.mark_clean()
            self._update_model_auth_controls(provider)
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
        provider = str(self._settings_query_one("#provider_select", Select).value)
        if not self.settings.has_api_key(provider):
            self._settings_query_one("#settings_connection_status", Static).update(
                "There is no locally stored key for this provider. Environment credentials are not changed here."
            )
            return
        screen.pending_api_key_removals.add(provider)
        self._settings_query_one("#api_key_input", Input).value = ""
        self._update_model_auth_controls(provider)
        self._settings_query_one("#settings_connection_status", Static).update(
            "The locally stored API key will be removed when you Apply."
        )
        screen._refresh_apply_label()

    async def _settings_login_chatgpt(self) -> None:
        if self._turn_active or self.agent_worker is not None:
            self._set_settings_status("Stop the active turn before signing in.", "warning")
            return
        screen = getattr(self, "_settings_screen", None)
        if screen is None:
            return
        button = self._settings_query_one("#settings_chatgpt_login", Button)
        status = self._settings_query_one("#settings_connection_status", Static)
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
        self._update_model_auth_controls(chatgpt_constants.PROVIDER_KEY)
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
        status = self._settings_query_one("#settings_connection_status", Static)
        self._update_model_auth_controls(chatgpt_constants.PROVIDER_KEY)
        status.update("ChatGPT sign-out will be saved when you Apply." if removed else "You are not signed in.")
        screen._refresh_apply_label()

    async def _test_settings_connection(self) -> None:
        if self._turn_active or self.agent_worker is not None:
            self._set_settings_status("Stop the active turn before testing a connection.", "warning")
            return
        status = self._settings_query_one("#settings_connection_status", Static)
        button = self._settings_query_one("#settings_test_connection", Button)
        try:
            candidate, _api_key_input, _provider, _model, _effort = self._settings_candidate_from_ui()
            config = build_agent_config(
                self.project_path,
                self.overrides,
                settings=candidate,
                settings_store=None,
            )
        except Exception as exc:
            status.update(f"Configuration is incomplete: {exc}")
            return
        button.disabled = True
        status.update("Testing connection…")
        result = await test_model_connection(config)
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
            model = str(model_select.value)
            effort = "" if effort_select.value is Select.NULL else str(effort_select.value)
            valid_efforts = {value for _, value in ui_thinking_effort_options(provider, model)}
            if effort not in valid_efforts:
                effort = default_ui_thinking_effort(provider, model) or ""
            self.settings.set_agent_model(role, provider, model, effort or None)

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
        summary.update(
            "\n".join(
                [
                    model_line,
                    f"Credential: {credential}",
                    f"Agent overrides: {override_count}",
                    f"Web search: {search_backend}",
                    mcp_line,
                    f"LSP: {'enabled' if lsp_enabled else 'disabled'}",
                    f"Theme: {theme_name}",
                ]
            )
        )
        launch.label = "Open Settings" if self.config is not None else "Continue Setup"

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
