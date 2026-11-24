"""Two-step picker that copies MCP servers from other tools into this project's config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, SelectionList, Static
from textual.widgets.selection_list import Selection

from kolega_code.mcp.config import MCPScope
from kolega_code.mcp.importers import ImportCandidate, ImportSource, discover_import_sources, import_servers
from kolega_code.mcp.scopes import find_definitions

from .mcp_screen import TRANSPORT_LABELS, plain_text

if TYPE_CHECKING:
    from ..app import KolegaCodeApp

SCOPE_CYCLE: tuple[MCPScope, ...] = ("user", "project", "local")
SCOPE_LABELS = {
    "user": "user (all your projects)",
    "project": "project (shared .mcp.json)",
    "local": "local (this project only)",
}
_SEARCHED = "Claude Code, Claude Desktop, Cursor, VS Code/Copilot, Windsurf, Gemini CLI and Codex"


class MCPImportScreen(ModalScreen[bool]):
    """Choose which found sources to read, then which of their servers to add."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True, priority=True)]

    def __init__(self, owner: "KolegaCodeApp") -> None:
        super().__init__()
        self.owner = owner
        self._sources: list[ImportSource] = []
        self._candidates: dict[str, ImportCandidate] = {}
        self._step = "sources"
        self._scope: MCPScope = "user"
        self._in_flight = False

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp_import_panel"):
            yield Static("Import MCP servers", id="mcp_import_title", markup=False)
            yield SelectionList[str](id="mcp_import_sources")
            yield SelectionList[str](id="mcp_import_servers")
            yield Static("", id="mcp_import_detail", markup=False)
            with Horizontal(id="mcp_import_actions"):
                yield Button("Cancel", id="mcp_import_cancel", classes="quiet")
                yield Button("Scope", id="mcp_import_scope_toggle", classes="quiet")
                yield Button("Next", id="mcp_import_next", classes="solid-primary")
            yield Static("", id="mcp_import_hint", markup=False)

    def on_mount(self) -> None:
        self._sources = discover_import_sources(self.owner.project_path)
        self._render_sources()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._in_flight:
            return
        if event.button.id == "mcp_import_cancel":
            self.dismiss(False)
        elif event.button.id == "mcp_import_scope_toggle":
            self._cycle_scope()
        elif event.button.id == "mcp_import_next":
            if self._step == "sources":
                self._show_servers()
            else:
                self._import_selected()

    def action_cancel(self) -> None:
        if not self._in_flight:
            if self._step == "servers":
                self._step = "sources"
                self._render_sources()
            else:
                self.dismiss(False)

    def _sources_list(self) -> SelectionList[str]:
        return self.query_one("#mcp_import_sources", SelectionList)

    def _servers_list(self) -> SelectionList[str]:
        return self.query_one("#mcp_import_servers", SelectionList)

    def _set_step_widgets(self) -> None:
        servers_step = self._step == "servers"
        self._sources_list().display = not servers_step
        self._servers_list().display = servers_step
        scope = self.query_one("#mcp_import_scope_toggle", Button)
        scope.display = servers_step
        scope.label = f"Scope: {SCOPE_LABELS[self._scope]}"
        self.query_one("#mcp_import_next", Button).label = "Next" if not servers_step else "Import"
        self.query_one("#mcp_import_hint", Static).update(
            "space toggles a row · esc goes back" if servers_step else "space toggles a source · Enter moves on"
        )

    def _render_sources(self) -> None:
        listing = self._sources_list()
        listing.clear_options()
        for source in self._sources:
            count = len(source.candidates)
            prompt = Text.assemble(
                (f"{source.label:<18}", "bold"),
                f"{count} server{'s' if count != 1 else ''}  ",
                (source.origins, "dim"),
            )
            listing.add_option(Selection(prompt, source.id, True))
        detail = self.query_one("#mcp_import_detail", Static)
        if self._sources:
            servers = sum(len(source.candidates) for source in self._sources)
            detail.update(
                Text.assemble(
                    f"Found {servers} MCP server{'s' if servers != 1 else ''} in ",
                    (str(len(self._sources)), "bold"),
                    f" tool{'s' if len(self._sources) != 1 else ''}. Only tools that define servers are listed.",
                )
            )
        else:
            detail.update(
                Text.assemble(
                    "No MCP servers found in other tools.\n\nSearched: ",
                    (_SEARCHED, "dim"),
                    "\n\nAdd a server with `kolega-code mcp add`, or import a file with "
                    "`kolega-code mcp import --file PATH`.",
                )
            )
        self.query_one("#mcp_import_next", Button).disabled = not self._sources
        self._set_step_widgets()
        self.set_focus(listing)

    def _show_servers(self) -> None:
        selected = set(self._sources_list().selected)
        self._candidates = {}
        for source in self._sources:
            if source.id not in selected:
                continue
            for index, candidate in enumerate(source.candidates):
                self._candidates[f"{source.id}:{index}"] = candidate
        self._step = "servers"
        self._render_servers()

    def _existing(self, name: str) -> list[str]:
        return [
            definition.scope
            for definition in find_definitions(
                name,
                project_path=self.owner.project_path,
                state_dir=self.owner.settings_store.root,
                settings=self.owner.settings,
            )
        ]

    def _render_servers(self) -> None:
        listing = self._servers_list()
        listing.clear_options()
        importable = 0
        for key, candidate in self._candidates.items():
            defined = self._existing(candidate.name) if candidate.importable else []
            blocked = "" if candidate.importable else candidate.unsupported
            if candidate.importable and self._scope in defined:
                blocked = f"already defined in {self._scope} scope"
            take = candidate.importable and not blocked
            importable += take
            prompt = Text()
            prompt.append(plain_text(candidate.original_name), style="bold" if take else "dim")
            transport = TRANSPORT_LABELS.get(candidate.server.transport, "?") if candidate.server else "?"
            prompt.append(f"  {transport:<6} {candidate.origin:<8} {plain_text(candidate.endpoint)}", style="dim")
            notes = " · ".join(part for part in (candidate.notes, blocked) if part)
            if notes:
                prompt.append(f"  · {plain_text(notes)}", style="yellow" if blocked else "dim")
            listing.add_option(Selection(prompt, key, take, disabled=bool(blocked)))
        self.query_one("#mcp_import_detail", Static).update(
            Text.assemble(
                f"{importable} of {len(self._candidates)} server{'s' if len(self._candidates) != 1 else ''} will be added to the ",
                (SCOPE_LABELS[self._scope], "bold"),
                " scope. Servers are added unverified: reconnect them from /mcp afterwards.",
            )
        )
        self.query_one("#mcp_import_next", Button).disabled = importable == 0
        self._set_step_widgets()
        self.set_focus(listing)

    def _cycle_scope(self) -> None:
        self._scope = SCOPE_CYCLE[(SCOPE_CYCLE.index(self._scope) + 1) % len(SCOPE_CYCLE)]
        self._render_servers()

    def _import_selected(self) -> None:
        if self._in_flight:
            return
        chosen = [self._candidates[str(key)] for key in self._servers_list().selected if str(key) in self._candidates]
        if not chosen:
            return
        self._in_flight = True
        self.owner._set_mcp_operation_in_flight(self, True)
        self.query_one("#mcp_import_next", Button).disabled = True
        self.query_one("#mcp_import_hint", Static).update("Importing…")
        scope = self._scope

        async def run() -> None:
            try:
                outcomes = import_servers(
                    chosen,
                    scope=scope,
                    project_path=self.owner.project_path,
                    state_dir=self.owner.settings_store.root,
                    settings=self.owner.settings,
                )
                self.owner.settings_store.save(self.owner.settings)
            finally:
                self._in_flight = False
                self.owner._set_mcp_operation_in_flight(self, False)
                self.owner._schedule_maybe_start_queued_message()

            added = [outcome for outcome in outcomes if outcome.outcome != "skipped"]
            self.owner._notify_user(
                f"Imported {len(added)} MCP server{'s' if len(added) != 1 else ''} into {scope} scope. "
                "Reconnect them from /mcp.",
                severity="information" if added else "warning",
            )
            # Servers import unverified, so the agent's tool set is unchanged: no rebuild.
            self.dismiss(True)

        self.run_worker(run(), name="mcp-import", group="mcp-import-actions")
