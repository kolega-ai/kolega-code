"""Settings panel behavior for the CLI TUI."""

from __future__ import annotations

from typing import Optional

from textual.css.query import NoMatches
from rich.text import Text
from textual.widgets import Input, Select, Static

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.agent.tool_backend.search_backends import (
    DEFAULT_BACKEND as DEFAULT_WEB_SEARCH_BACKEND,
    SearchBackendError,
    available_backends,
    get_backend_class,
)

from .. import messages, theme
from ..config import key_status
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
from ..settings import WEB_SEARCH_KEY_NAMES
from ..theme import Color, Glyph


class SettingsPanelMixin:
    @property
    def _settings_status(self) -> Static:
        return self.query_one("#settings_status", Static)

    def on_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id or ""

        if select_id == "provider_select":
            provider = str(event.value)
            self._repopulate_model_select(provider, "model_select", "thinking_effort_select")
            try:
                api_key_input = self.query_one("#api_key_input", Input)
                api_key_input.placeholder = self._api_key_placeholder(provider)
                # OAuth providers sign in via /login, so the key field is read-only.
                api_key_input.disabled = provider == chatgpt_constants.PROVIDER_KEY
            except NoMatches:
                pass
            return

        if select_id == "model_select":
            try:
                provider = str(self.query_one("#provider_select", Select).value)
            except NoMatches:
                return
            self._set_effort_select_default(provider, str(event.value))
            return

        if select_id == "web_search_backend_select":
            self._update_search_backend_fields(str(event.value))
            return

        if select_id.startswith("am_provider_"):
            role = select_id[len("am_provider_") :]
            provider = str(event.value)
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
            return

        if select_id.startswith("am_model_"):
            role = select_id[len("am_model_") :]
            try:
                provider = str(self.query_one(f"#am_provider_{role}", Select).value)
            except NoMatches:
                return
            if provider != INHERIT_SENTINEL and event.value is not Select.NULL:
                # A restored effort waits here for the model that hosts it; a manual
                # model change has none pending and falls back to preserve/default.
                preferred = self._pending_agent_efforts.pop(f"am_effort_{role}", None)
                self._set_effort_select_default(provider, str(event.value), f"am_effort_{role}", preferred=preferred)
            return

        if select_id == "theme_select":
            name = str(event.value)
            if name != (self.settings.active_theme or theme.DEFAULT_THEME_NAME):
                self.settings.active_theme = name
                self.settings_store.save(self.settings)
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
        provider_select = self.query_one("#provider_select", Select)
        model_select = self.query_one("#model_select", Select)
        effort_select = self.query_one("#thinking_effort_select", Select)
        api_key_input = self.query_one("#api_key_input", Input)

        provider_select.value = provider
        model_select.set_options(model_options)
        model_select.value = model
        effort_select.set_options(ui_thinking_effort_options(provider, model))
        if effort is not None:
            effort_select.value = effort
        theme_select = self.query_one("#theme_select", Select)
        theme_select.value = (
            self.settings.active_theme
            if self.settings.active_theme in theme.available_themes()
            else theme.DEFAULT_THEME_NAME
        )
        api_key_input.placeholder = self._api_key_placeholder(provider)
        api_key_input.disabled = provider == chatgpt_constants.PROVIDER_KEY
        self._populate_agent_model_rows()
        self._populate_web_search_controls()
        self._update_settings_status()

    def _populate_agent_model_rows(self) -> None:
        """Seed each per-agent row from saved settings (absent role -> inherit).

        A model select that was just given its options can't accept a value until the
        next refresh, and setting the provider value posts a Changed that re-runs the
        cascade afterwards. So we stash the saved model/effort and let that cascade —
        the last writer — apply them, rather than assigning here and being clobbered."""
        provider_values = {value for _, value in ui_provider_options()}
        for _, role in agent_role_options():
            try:
                provider_select = self.query_one(f"#am_provider_{role}", Select)
            except NoMatches:
                continue
            entry = self.settings.get_agent_model(role) or {}
            provider = entry.get("provider")
            self._pending_agent_models.pop(f"am_model_{role}", None)
            self._pending_agent_efforts.pop(f"am_effort_{role}", None)
            if provider not in provider_values:
                provider_select.value = INHERIT_SENTINEL
                self._clear_model_effort_selects(f"am_model_{role}", f"am_effort_{role}")
                continue
            if entry.get("model"):
                self._pending_agent_models[f"am_model_{role}"] = str(entry["model"])
            if entry.get("thinking_effort"):
                self._pending_agent_efforts[f"am_effort_{role}"] = str(entry["thinking_effort"])
            # Triggers on_select_changed, which consumes the pending model/effort.
            provider_select.value = provider

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
                self.query_one(f"#{widget_id}").display = visible
            except NoMatches:
                pass
        if needs_key:
            try:
                key_input = self.query_one("#web_search_api_key_input", Input)
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
            self.query_one("#web_search_backend_select", Select).value = backend
            self.query_one("#web_search_base_url_input", Input).value = self.settings.web_search_base_url or ""
            self.query_one("#web_search_api_key_input", Input).value = ""
        except NoMatches:
            pass
        self._update_search_backend_fields(backend)

    def _collect_web_search_from_ui(self) -> None:
        """Write the Web Search controls into settings (keys only when newly typed)."""
        try:
            backend = str(self.query_one("#web_search_backend_select", Select).value)
            base_url_input = self.query_one("#web_search_base_url_input", Input)
            key_input = self.query_one("#web_search_api_key_input", Input)
        except NoMatches:
            return
        self.settings.web_search_backend = backend
        self.settings.web_search_base_url = base_url_input.value.strip() or None
        key = key_input.value.strip()
        if key and backend in WEB_SEARCH_KEY_NAMES:
            self.settings.set_api_key(backend, key)
        key_input.value = ""
        self._update_search_backend_fields(backend)

    def _set_effort_select_default(
        self, provider: str, model: str, effort_id: str = "thinking_effort_select", *, preferred: Optional[str] = None
    ) -> None:
        try:
            effort_select = self.query_one(f"#{effort_id}", Select)
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
            model_select = self.query_one(f"#{model_id}", Select)
        except NoMatches:
            return
        if model_value is None:
            current = model_select.value
            model_value = None if current is Select.NULL else str(current)
        model_options = ui_model_options(provider)
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
                select = self.query_one(f"#{select_id}", Select)
            except NoMatches:
                continue
            select.set_options([])
            select.value = Select.NULL

    async def _save_settings_from_ui(self) -> None:
        provider = str(self.query_one("#provider_select", Select).value)
        model = str(self.query_one("#model_select", Select).value)
        effort = str(self.query_one("#thinking_effort_select", Select).value)
        valid_efforts = {value for _, value in ui_thinking_effort_options(provider, model)}
        if effort not in valid_efforts:
            effort = default_ui_thinking_effort(provider, model) or ""
        api_key_input = self.query_one("#api_key_input", Input)
        api_key = api_key_input.value.strip()

        self.settings.active_provider = provider
        self.settings.active_model = model
        self.settings.active_thinking_effort = effort or default_ui_thinking_effort(provider, model)
        self.settings.active_theme = str(self.query_one("#theme_select", Select).value)
        if api_key:
            self.settings.set_api_key(provider, api_key)
        self._collect_agent_models_from_ui()
        self._collect_web_search_from_ui()
        self.settings_store.save(self.settings)
        api_key_input.value = ""
        api_key_input.placeholder = self._api_key_placeholder(provider)

        await self._ensure_agent_from_settings(rebuild=True)
        if self.config is not None:
            self._notify_user(messages.SETTINGS_SAVED)

    def _collect_agent_models_from_ui(self) -> None:
        """Write each per-agent row into settings.agent_models (inherit rows removed)."""
        for _, role in agent_role_options():
            try:
                provider = str(self.query_one(f"#am_provider_{role}", Select).value)
                model_select = self.query_one(f"#am_model_{role}", Select)
                effort_select = self.query_one(f"#am_effort_{role}", Select)
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
            return

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
            return

        provider = self.settings.active_provider
        model = self.settings.active_model
        effort = self.settings.active_thinking_effort or default_ui_thinking_effort(provider, model) or "not supported"
        status = key_status(provider, self.project_path, self.settings)
        tone = "warning" if "missing" in status.lower() else "ok"
        text = "\n".join(
            [
                messages.SETTINGS_ACTIVE_MODEL.format(provider=provider, model=model),
                messages.SETTINGS_THINKING_EFFORT_LINE.format(effort=effort),
                messages.SETTINGS_API_KEY_LINE.format(status=status),
            ]
        )
        self._set_settings_status(text, tone)
        self._refresh_status_dashboard()

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
