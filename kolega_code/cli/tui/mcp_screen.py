"""The ``/mcp`` panel: review and manage MCP servers without leaving the session."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from kolega_code.mcp.config import LoadedMCPConfig, MCPScope, MCPServerConfig
from kolega_code.mcp.health import MCPServerHealth, health_hint

from . import mcp_ops
from .mcp_ops import plain_text
from .settings_screen import ConfirmSettingsActionScreen

if TYPE_CHECKING:
    from ..app import KolegaCodeApp

MASKED_VALUE = "‹hidden›"
TRANSPORT_LABELS = {"streamable_http": "http", "sse": "sse", "stdio": "stdio"}
_HEALTH_STYLES = {
    "connected": "green",
    "failed": "red",
    "needs_auth": "yellow",
    "stale": "yellow",
    "pending_trust": "yellow",
    "disabled": "dim",
    "unverified": "",
}
KEY_HINT = (
    "enter Tools · v Reconnect · a Sign in · space Enable/Disable · c Clear sign-in · "
    "t Trust project · i Import · s Edit in Settings · r Refresh · esc Close"
)
NO_SERVERS_TEXT = (
    "No MCP servers configured.\n\n"
    "  kolega-code mcp add --transport http notion https://mcp.notion.com/mcp\n"
    "  kolega-code mcp add airtable -- npx -y airtable-mcp-server\n\n"
    "Press i to import servers from another tool, or s to open Settings."
)


class MCPScreen(ModalScreen[None]):
    """List MCP servers with their status and act on the selected one."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True, priority=True),
        Binding("enter", "tools", "Tools", show=False),
        Binding("r", "refresh", "Refresh", show=False),
        Binding("v", "reconnect", "Reconnect", show=False),
        Binding("a", "authenticate", "Sign in", show=False),
        Binding("space", "toggle", "Enable/Disable", show=False),
        Binding("c", "clear_auth", "Clear sign-in", show=False),
        Binding("t", "trust", "Trust project", show=False),
        Binding("i", "start_import", "Import", show=False),
        Binding("s", "edit", "Edit", show=False),
    ]

    def __init__(self, owner: "KolegaCodeApp") -> None:
        super().__init__()
        self.owner = owner
        self._rows: list[dict[str, Any]] = []
        self._servers: dict[tuple[str, MCPScope | None, str | None], MCPServerConfig] = {}
        self._in_flight = False
        self._busy_label = ""
        self._showing_tools = False
        self._tools: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp_panel"):
            yield Static("", id="mcp_panel_title", markup=False)
            yield OptionList(id="mcp_panel_servers")
            yield OptionList(id="mcp_panel_tools")
            yield Static("", id="mcp_panel_detail", markup=False)
            yield Static("", id="mcp_panel_hint", markup=False)

    def on_mount(self) -> None:
        self.owner._mcp_screen = self
        self._refresh()

    def on_unmount(self) -> None:
        if self.owner._mcp_screen is self:
            self.owner._mcp_screen = None

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id == "mcp_panel_servers":
            event.stop()
            self._render_detail()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.action_tools()

    def _selected_row(self) -> Optional[dict[str, Any]]:
        highlighted = self.query_one("#mcp_panel_servers", OptionList).highlighted
        if highlighted is None or highlighted >= len(self._rows):
            return None
        return self._rows[highlighted]

    def _refresh(self) -> None:
        previous_row = self._selected_row()
        previous_key = _row_key(previous_row) if previous_row is not None else None
        try:
            config = mcp_ops.load_config(self.owner)
            rows = mcp_ops.status_rows(self.owner, config)
        except mcp_ops.OPERATION_ERRORS as exc:
            self.owner._notify_user(mcp_ops.operation_error(exc), severity="error")
            self._render_hint()
            return
        self._servers = {
            (server.id, server.scope, str(path) if (path := files.get(server.id)) is not None else None): server
            for servers, files in (
                (config.servers.values(), config.server_files),
                (config.untrusted_project_servers, config.untrusted_project_server_files),
            )
            for server in servers
        }
        self._rows = rows
        listing = self.query_one("#mcp_panel_servers", OptionList)
        previous = listing.highlighted or 0
        selected = next((index for index, row in enumerate(rows) if _row_key(row) == previous_key), None)
        listing.clear_options()
        for index, row in enumerate(self._rows):
            listing.add_option(Option(self._row_text(row), id=f"mcp_row_{index}"))
        if self._rows:
            listing.highlighted = selected if selected is not None else min(previous, len(self._rows) - 1)
        self._render_title(config)
        if self._showing_tools:
            self._load_tools()
        self._render_detail()
        self._render_hint()

    def _row_text(self, row: dict[str, Any]) -> Text:
        health: MCPServerHealth = row["health"]
        hint = _hint(row)
        text = Text()
        text.append(f"{health.glyph} ", style=_HEALTH_STYLES[health.health])
        text.append(f"{plain_text(row['id']):<24}")
        text.append(f" {str(row['scope']):<8}")
        text.append(f" {TRANSPORT_LABELS.get(str(row['transport']), str(row['transport'])):<6}")
        text.append(health.label, style=_HEALTH_STYLES[health.health])
        if hint:
            text.append(f"  {hint}", style="dim")
        return text

    def _render_title(self, config: LoadedMCPConfig) -> None:
        trusted = config.project_trusted
        path = self.owner.active_project_path
        trust_scope = (
            f" (launch checkout: {plain_text(str(self.owner.project_path))})" if path != self.owner.project_path else ""
        )
        title = Text.assemble(
            ("MCP servers  ", "bold"),
            plain_text(str(path)),
            "  ·  ",
            ("project trusted" if trusted else "project not trusted", "green" if trusted else "yellow"),
            trust_scope,
        )
        self.query_one("#mcp_panel_title", Static).update(title)

    def _render_detail(self) -> None:
        if self._showing_tools:
            self.query_one("#mcp_panel_detail", Static).update(
                Text.assemble(("Tools  ", "dim"), str(len(self._tools)), " exposed to the agent")
            )
            return
        row = self._selected_row()
        self.query_one("#mcp_panel_detail", Static).update(self._server_text(row) if row else Text(NO_SERVERS_TEXT))

    def _server_text(self, row: dict[str, Any]) -> Text:
        health: MCPServerHealth = row["health"]
        server = self._definition_for(row)
        text = Text()
        text.append(plain_text(str(row["id"])), style="bold")
        text.append(f"  {_scope_location(row)}", style="dim")
        if server is not None:
            text.append("\n")
            text.append(
                _kv("Type", f"{TRANSPORT_LABELS.get(server.transport, server.transport)} · {_oauth_label(server)}")
            )
            text.append("\n")
            text.append(_kv(*_endpoint(server)))
            if server.headers or server.env:
                text.append("\n")
                text.append(_kv("Secrets", _masked(server)))
            if row.get("missing_env"):
                text.append("\n")
                text.append(_kv("Missing env", ", ".join(row["missing_env"])))
        status = (
            "Pending approval: trust the project to load it"
            if health.health == "pending_trust"
            else f"{health.label} — {_hint(row)}"
            if _hint(row)
            else health.label
        )
        text.append("\n")
        text.append(_kv("Status", status, style=_HEALTH_STYLES[health.health]))
        return text

    def _load_tools(self) -> None:
        row = self._selected_row()
        server = self._definition_for(row) if row and row["health"].health == "connected" else None
        self._tools = [
            (name, (description or "").strip().splitlines()[0] if (description or "").strip() else "")
            for name, _, description in (mcp_ops.exposed_tools(self.owner, server) if server is not None else [])
        ]
        tools = self.query_one("#mcp_panel_tools", OptionList)
        tools.clear_options()
        for name, summary in self._tools:
            option = Text(name, style="bold")
            if summary:
                option.append(f"  {summary}", style="dim")
            tools.add_option(Option(option))
        if self._tools:
            tools.highlighted = 0

    def _show_tools(self, show: bool) -> None:
        self._showing_tools = show
        if show:
            self._load_tools()
        self.query_one("#mcp_panel_servers", OptionList).display = not show
        self.query_one("#mcp_panel_tools", OptionList).display = show
        self.set_focus(self.query_one("#mcp_panel_tools" if show else "#mcp_panel_servers", OptionList))
        self._render_detail()
        self._render_hint()

    def _render_hint(self) -> None:
        hint = self.query_one("#mcp_panel_hint", Static)
        if self._in_flight:
            hint.update(
                Text.assemble(
                    (f"{self._busy_label}… ", "yellow"),
                    "stdio servers run their configured command; OAuth opens a browser",
                )
            )
        elif self._showing_tools:
            hint.update("enter/esc Back to servers")
        else:
            hint.update(KEY_HINT)

    def action_close(self) -> None:
        if self._in_flight:
            return
        if self._showing_tools:
            self._show_tools(False)
            return
        if self.owner._mcp_screen is self:
            self.owner._mcp_screen = None
        self.dismiss()

    def action_refresh(self) -> None:
        if not self._in_flight:
            self._refresh()

    def action_tools(self) -> None:
        if self._in_flight:
            return
        row = self._selected_row()
        if not self._showing_tools and (row is None or row["health"].health != "connected"):
            self.owner._notify_user("Tools are listed once a server is connected. Select Reconnect (v) to verify it.")
            return
        self._show_tools(not self._showing_tools)

    def action_reconnect(self) -> None:
        self._verify_selected(authenticate=False)

    def action_authenticate(self) -> None:
        self._verify_selected(authenticate=True)

    def _verify_selected(self, *, authenticate: bool) -> None:
        row = self._selected_row()
        if row is None or not self._ready():
            return
        server_id = str(row["id"])
        if row["health"].health == "pending_trust":
            self.owner._notify_user("Trust the project first (t) to load its servers.")
            return
        server = self._definition_for(row)
        if authenticate and (server is None or not server.oauth.enabled):
            self.owner._notify_user(f"MCP server '{server_id}' has no OAuth configured. Use Reconnect (v) instead.")
            return

        async def operation() -> None:
            result = await mcp_ops.verify(self.owner, server_id)
            if result is None:
                self.owner._notify_user(mcp_ops.BLOCKED_MESSAGE, severity="warning")
            elif result.ok:
                self.owner._notify_user(f"Verified MCP server '{server_id}' ({result.tool_count} tool(s)).")
            else:
                self.owner._notify_user(
                    f"MCP verification failed for '{server_id}': {result.message}", severity="warning"
                )

        self._run(f"Verifying '{plain_text(server_id)}'", operation)

    def action_toggle(self) -> None:
        row = self._selected_row()
        if row is None or not self._ready():
            return
        server_id = str(row["id"])
        scope = row["scope"]
        enabled = not bool(row["enabled"])

        async def operation() -> None:
            if await mcp_ops.set_enabled(self.owner, server_id, scope, enabled):
                self.owner._notify_user(f"{'Enabled' if enabled else 'Disabled'} MCP server '{server_id}'.")
            else:
                self.owner._notify_user(f"MCP server '{server_id}' is no longer defined.", severity="warning")

        self._run("Updating", operation)

    def action_clear_auth(self) -> None:
        row = self._selected_row()
        if row is None or not self._ready():
            return
        if row["health"].health == "pending_trust":
            self.owner._notify_user("Trust the project first (t) to load its servers.")
            return
        server_id = str(row["id"])
        self._confirm(
            ConfirmSettingsActionScreen(
                "Clear MCP sign-in?",
                f"Stored OAuth credentials and cached tools for '{server_id}' will be removed. "
                "Sign in again before using the server.",
                "Clear Sign-in",
                danger=True,
            ),
            lambda: self._clear_sign_in(server_id),
        )

    async def _clear_sign_in(self, server_id: str) -> None:
        await mcp_ops.clear_authentication(self.owner, server_id)
        self.owner._notify_user(f"Cleared the sign-in for MCP server '{server_id}'.")

    def action_trust(self) -> None:
        if not self._ready():
            return
        if mcp_ops.project_trusted(self.owner):
            self.owner._notify_user("This project is already trusted for MCP.")
            return

        async def operation() -> None:
            if await mcp_ops.trust_project(self.owner):
                self.owner._notify_user("Trusted this project's MCP servers.")

        self._confirm(
            ConfirmSettingsActionScreen(
                "Trust project MCP servers?",
                "Trusted project servers may start local commands or connect to remote services. "
                "Only trust repositories you control or have reviewed.",
                "Trust Project",
            ),
            operation,
        )

    def action_start_import(self) -> None:
        if not self._ready():
            return
        from .mcp_import_screen import MCPImportScreen

        self.app.push_screen(MCPImportScreen(self.owner), callback=lambda _imported: self._refresh())

    def action_edit(self) -> None:
        row = self._selected_row()
        if row is None or self._in_flight or self.app.screen is not self:
            return
        server_id = str(row["id"])
        if row["scope"] == "project":
            self.owner._notify_user(
                "Project servers are read-only in Settings. Edit .mcp.json in the repository.", severity="warning"
            )
            return
        self.owner._mcp_screen = None
        self.dismiss()
        self.app.call_after_refresh(lambda: self.owner._open_settings_for_mcp_server(server_id))

    def _ready(self) -> bool:
        """False when an operation is running or a turn would be interrupted."""
        if self._in_flight:
            return False
        if self.owner._mcp_operation_in_flight():
            self.owner._notify_user(mcp_ops.BUSY_MESSAGE, severity="warning")
            return False
        if mcp_ops.change_blocked(self.owner):
            self.owner._notify_user(mcp_ops.BLOCKED_MESSAGE, severity="warning")
            return False
        return True

    def _confirm(self, dialog: ConfirmSettingsActionScreen, operation: Callable[[], Awaitable[None]]) -> None:
        # The dialog pops before this runs; a covered screen must not start work.
        def start(confirmed: bool | None) -> None:
            if confirmed and self.app.screen is self:
                self._run("Updating", operation)

        self.app.push_screen(dialog, callback=start)

    def _run(self, label: str, operation: Callable[[], Awaitable[None]]) -> None:
        if not self._ready():
            return
        self._in_flight = True
        self._busy_label = label
        self.owner._set_mcp_operation_in_flight(self, True)
        self._render_hint()

        async def run() -> None:
            try:
                await operation()
            except mcp_ops.OPERATION_ERRORS as exc:
                self.owner._notify_user(mcp_ops.operation_error(exc), severity="error")
            finally:
                self._in_flight = False
                self._busy_label = ""
                self.owner._set_mcp_operation_in_flight(self, False)
                if self.is_mounted:
                    self._refresh()
                self.owner._schedule_maybe_start_queued_message()

        # Keep network and OAuth waits off the app's message pump.
        self.run_worker(run, name="mcp-panel-action", group="mcp-panel-actions", exit_on_error=False)

    def _definition_for(self, row: dict[str, Any]) -> MCPServerConfig | None:
        return self._servers.get(_row_key(row))


def _row_key(row: dict[str, Any]) -> tuple[str, MCPScope | None, str | None]:
    return row["id"], row["scope"], row["file"]


def _hint(row: dict[str, Any]) -> str:
    return plain_text(health_hint(row["health"], row["missing_env"]))


def _kv(label: str, value: str, style: str = "") -> Text:
    return Text.assemble((f"{label:<12}", "dim"), (value, style))


def _scope_location(row: dict[str, Any]) -> str:
    if row.get("file"):
        return f"{row['scope']} scope · {plain_text(str(row['file']))}"
    return f"{row['scope']} scope · settings.json (this project only)"


def _endpoint(server: MCPServerConfig) -> tuple[str, str]:
    if server.transport == "stdio":
        return "Command", plain_text(" ".join([server.command or "", *server.args]))
    return "URL", plain_text(server.url or "")


def _masked(server: MCPServerConfig) -> str:
    pairs = [f"{key}={MASKED_VALUE}" for key in server.headers]
    pairs.extend(f"{key}={MASKED_VALUE}" for key in server.env)
    return plain_text(", ".join(pairs))


def _oauth_label(server: MCPServerConfig) -> str:
    if not server.oauth.enabled:
        return "OAuth off"
    if not server.oauth.client_id:
        return "OAuth on (dynamic client registration)"
    if server.oauth.client_secret_env:
        return f"OAuth on (pre-registered client, secret from {server.oauth.client_secret_env})"
    return "OAuth on (pre-registered client)"
