"""Capability-driven private project-memory browser and editor."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from kolega_code.memory import (
    MISSING_REVISION,
    MemoryCapability,
    MemoryEntry,
    MemoryEntrySummary,
    MemoryUnavailableError,
    ProjectMemoryManager,
    ProjectMemoryStatus,
)

from .. import theme
from ..theme import Color, Glyph
from .settings_screen import ConfirmSettingsActionScreen

if TYPE_CHECKING:
    from ..app import KolegaCodeApp


class MemoryScreen(ModalScreen[None]):
    """Browse, inspect, and explicitly author the current project's memory bank."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True, priority=True),
        Binding("ctrl+s", "save", "Save", show=True, priority=True),
        Binding("n", "create", "New", show=True),
        Binding("e", "edit", "Edit", show=True),
        Binding("r", "refresh", "Refresh", show=True),
    ]

    def __init__(
        self,
        owner: "KolegaCodeApp",
        *,
        inspect_disabled: bool = False,
    ) -> None:
        super().__init__()
        self._owner = owner
        self._status: ProjectMemoryStatus | None = None
        self._all_entries: list[MemoryEntrySummary] = []
        self._entries: list[MemoryEntrySummary] = []
        self._selected_reference: str | None = None
        self._loaded_entry: MemoryEntry | None = None
        self._editing = False
        self._creating = False
        self._stale_conflict = False
        self._inspect_disabled = inspect_disabled
        self._agent_view = False
        self._busy = False
        self._work_generation = 0

    @property
    def _manager(self) -> ProjectMemoryManager:
        return self._owner.memory_manager

    def compose(self) -> ComposeResult:
        with Vertical(id="memory_shell"):
            with Horizontal(id="memory_header"):
                yield Static(
                    f"{theme.g(Glyph.TOOL)}  Project Memory",
                    id="memory_title",
                    markup=False,
                )
                yield Static("private local state  ·  esc close  ·  r refresh", id="memory_header_hint")
            with Horizontal(id="memory_toolbar"):
                yield Static("Loading private memory…", id="memory_status", markup=False)
                with Horizontal(id="memory_toolbar_buttons"):
                    yield Button("Agent view", id="memory_agent_view", classes="quiet")
                    yield Button("Inspect bank", id="memory_inspect", classes="quiet")
                    yield Button("Turn off", id="memory_toggle", classes="quiet")
            with Horizontal(id="memory_body"):
                with Vertical(id="memory_roster"):
                    yield Input(placeholder="Filter entries by path…", id="memory_filter")
                    yield OptionList(id="memory_entries")
                with Vertical(id="memory_detail"):
                    yield Static("Select an entry to inspect.", id="memory_metadata", markup=False)
                    yield Input(placeholder="topics/example.md", id="memory_reference")
                    with VerticalScroll(id="memory_preview_scroll"):
                        yield Markdown(
                            "Memory entries are private local state and are not written to the repository.",
                            id="memory_preview",
                        )
                    yield TextArea("", id="memory_editor")
                    yield Static("", id="memory_notice", markup=False)
            with Horizontal(id="memory_footer"):
                yield Button("New", id="memory_new", classes="quiet")
                yield Button("Delete", id="memory_delete", classes="quiet danger")
                yield Button("Clear all", id="memory_clear", classes="quiet danger")
                yield Button("Edit", id="memory_edit", classes="solid-primary")
                yield Button("Cancel", id="memory_cancel", classes="quiet")
                yield Button("Reload latest", id="memory_reload", classes="quiet")
                yield Button("Save", id="memory_save", classes="solid-primary")

    def on_mount(self) -> None:
        self.border_title = "Private Project Memory"
        self._set_edit_mode(False)
        self._start_refresh()

    def on_unmount(self) -> None:
        if self._owner._memory_screen is self:
            self._owner._memory_screen = None

    # ---- loading -------------------------------------------------------------

    def _start_refresh(self, *, select_reference: str | None = None, from_disk: bool = False) -> None:
        if self._editing:
            return
        self._exit_agent_view()
        self._work_generation += 1
        generation = self._work_generation
        self._set_busy(True)
        self.run_worker(
            self._refresh(generation, select_reference, from_disk=from_disk),
            name="Refresh project memory",
            group="memory-screen-refresh",
            exclusive=True,
        )

    async def _refresh(self, generation: int, select_reference: str | None, *, from_disk: bool = False) -> None:
        try:
            if from_disk:
                await asyncio.to_thread(self._manager.refresh)
            status = await asyncio.to_thread(self._manager.status)
            entries: list[MemoryEntrySummary] = []
            if (
                status.available
                and (status.enabled or self._inspect_disabled)
                and self._supports(MemoryCapability.BROWSE)
            ):
                entries = await asyncio.to_thread(
                    self._manager.list_entries,
                    None,
                    allow_disabled=True,
                )
        except Exception as error:
            if generation == self._work_generation:
                self._status = None
                self._all_entries = []
                self._entries = []
                self._render_error(str(error))
            return
        finally:
            if generation == self._work_generation:
                self._set_busy(False)

        if generation != self._work_generation:
            return
        self._status = status
        self._all_entries = entries
        self._entries = self._filtered(entries)
        self._render_status()
        self._render_roster(select_reference=select_reference)

    def _start_load(self, reference: str) -> None:
        if self._editing:
            return
        self._exit_agent_view()
        self._work_generation += 1
        generation = self._work_generation
        self._selected_reference = reference
        self._set_busy(True)
        self.run_worker(
            self._load_entry(generation, reference),
            name="Read project memory entry",
            group="memory-screen-read",
            exclusive=True,
        )

    async def _load_entry(self, generation: int, reference: str) -> None:
        try:
            entry = await asyncio.to_thread(
                self._manager.read_entry,
                reference,
                allow_disabled=True,
            )
        except Exception as error:
            if generation == self._work_generation:
                self._render_error(str(error))
            return
        finally:
            if generation == self._work_generation:
                self._set_busy(False)
        if generation != self._work_generation:
            return
        self._loaded_entry = entry
        self._render_entry(entry)

    # ---- rendering -----------------------------------------------------------

    def _render_status(self) -> None:
        status = self._status
        if status is None:
            return
        backend = status.backend
        state = "on" if status.enabled else "off"
        availability = "available" if status.available else "unavailable"
        count = backend.entry_count if backend else 0
        total = backend.total_bytes if backend else 0
        line = Text()
        line.append("PRIVATE  ", style=f"bold {Color.ACCENT}")
        line.append(f"{status.backend_id} · {state} · {availability}")
        line.append(
            f"\n{count} entries · {total:,} bytes · {status.identity_kind} identity",
            style="dim",
        )
        if backend is not None:
            startup_state = " · startup truncated" if backend.startup_truncated else ""
            line.append(
                f"\nstartup {backend.startup_lines} lines / {backend.startup_bytes:,} bytes{startup_state}",
                style="dim",
            )
            if backend.private_path:
                line.append(f"\n{backend.private_path}", style="dim")
        if status.diagnostic:
            line.append(f" · {status.diagnostic}", style=Color.WARNING)
        elif backend and backend.warnings:
            line.append(f" · {backend.warnings[0]}", style=Color.WARNING)
        self.query_one("#memory_status", Static).update(line)

        toggle = self.query_one("#memory_toggle", Button)
        toggle.label = "Turn off" if status.enabled else "Turn on"
        inspect = self.query_one("#memory_inspect", Button)
        inspect.display = not status.enabled
        inspect.label = "Hide bank" if self._inspect_disabled else "Inspect bank"
        self._refresh_controls()

    def _render_roster(self, *, select_reference: str | None = None) -> None:
        roster = self.query_one("#memory_entries", OptionList)
        roster.clear_options()
        if self._status is not None and not self._status.enabled and not self._inspect_disabled:
            self._selected_reference = None
            self._loaded_entry = None
            self._render_empty(
                "Memory is off. The existing bank is preserved but hidden.\n"
                "Choose “Inspect bank” to browse or edit it without enabling agent access."
            )
            return
        if not self._supports(MemoryCapability.BROWSE):
            self._selected_reference = None
            self._loaded_entry = None
            self._render_empty(
                "The active backend does not provide generic entry browsing. "
                "Its agent-facing recall tools remain available according to its capabilities."
            )
            return
        if not self._entries:
            self._selected_reference = None
            self._loaded_entry = None
            self._render_empty("No matching memory entries.")
            return

        roster.add_options(
            [
                Option(
                    self._entry_label(summary),
                    id=f"memory_entry_{index}",
                )
                for index, summary in enumerate(self._entries)
            ]
        )
        references = [entry.reference for entry in self._entries]
        target = select_reference or self._selected_reference
        index = references.index(target) if target in references else 0
        roster.highlighted = index
        reference = references[index]
        if select_reference is None and self._loaded_entry is not None and self._loaded_entry.reference == reference:
            return
        self._start_load(reference)

    def _filtered(self, entries: list[MemoryEntrySummary]) -> list[MemoryEntrySummary]:
        query = self.query_one("#memory_filter", Input).value.strip().casefold()
        if not query:
            return list(entries)
        return [
            entry
            for entry in entries
            if query in entry.reference.casefold() or query in (entry.display_name or "").casefold()
        ]

    def _entry_label(self, summary: MemoryEntrySummary) -> Text:
        line = Text()
        line.append(summary.display_name or summary.reference)
        line.append(f"\n{summary.byte_count:,} bytes", style="dim")
        return line

    def _render_entry(self, entry: MemoryEntry) -> None:
        self._selected_reference = entry.reference
        modified = self._summary_for(entry.reference)
        metadata = f"{entry.reference}\n{entry.byte_count:,} bytes · sha256 {entry.revision}"
        if modified is not None and modified.modified_ns is not None:
            stamp = datetime.fromtimestamp(modified.modified_ns / 1_000_000_000)
            metadata += f" · modified {stamp:%Y-%m-%d %H:%M}"
        self.query_one("#memory_metadata", Static).update(metadata)
        self.query_one("#memory_preview", Markdown).update(entry.content or "[Empty entry]")
        notice = " · ".join(entry.warnings)
        self.query_one("#memory_notice", Static).update(notice)
        self._refresh_controls()

    def _render_empty(self, message: str) -> None:
        self.query_one("#memory_metadata", Static).update("No entry selected.")
        self.query_one("#memory_preview", Markdown).update(message)
        self.query_one("#memory_notice", Static).update("")
        self._refresh_controls()

    def _render_error(self, message: str) -> None:
        self.query_one("#memory_status", Static).update(f"Memory unavailable · {message}")
        self.query_one("#memory_notice", Static).update(message)
        self._set_busy(False)
        self._refresh_controls()

    def _summary_for(self, reference: str) -> MemoryEntrySummary | None:
        return next((entry for entry in self._entries if entry.reference == reference), None)

    def _supports(self, capability: MemoryCapability) -> bool:
        backend = self._manager.backend
        return bool(backend is not None and capability in backend.metadata.capabilities)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        status = self._status
        available = bool(status is not None and status.available)
        turn_active = self._owner._turn_active or self._owner.agent_worker is not None
        selected = self._loaded_entry is not None and self._loaded_entry.present
        mutating_blocked = self._busy or turn_active or not available

        self._set_button_disabled("memory_agent_view", self._busy or self._editing or not available)
        self._set_button_disabled("memory_inspect", self._busy or self._editing)
        self._set_button_disabled("memory_toggle", self._busy or self._editing or not available)
        self._set_button_disabled(
            "memory_new",
            mutating_blocked
            or self._editing
            or not (self._supports(MemoryCapability.REPLACE) or self._supports(MemoryCapability.APPEND)),
        )
        self._set_button_disabled(
            "memory_edit",
            mutating_blocked or self._editing or not selected or not self._supports(MemoryCapability.REPLACE),
        )
        self._set_button_disabled(
            "memory_delete",
            mutating_blocked or self._editing or not selected or not self._supports(MemoryCapability.DELETE),
        )
        self._set_button_disabled(
            "memory_clear",
            mutating_blocked
            or self._editing
            or not bool(self._all_entries)
            or not self._supports(MemoryCapability.CLEAR),
        )
        self._set_button_disabled("memory_save", mutating_blocked or not self._editing)
        self._set_button_disabled(
            "memory_reload",
            self._busy or not self._editing or not self._stale_conflict,
        )
        for widget_id in ("memory_new", "memory_delete", "memory_clear", "memory_edit"):
            self.query_one(f"#{widget_id}", Button).display = not self._editing
        self.query_one("#memory_cancel", Button).display = self._editing
        self.query_one("#memory_save", Button).display = self._editing
        self.query_one("#memory_reload", Button).display = self._editing and self._stale_conflict

    def _set_button_disabled(self, widget_id: str, disabled: bool) -> None:
        self.query_one(f"#{widget_id}", Button).disabled = disabled

    def _set_edit_mode(self, editing: bool, *, creating: bool = False) -> None:
        self._editing = editing
        self._creating = creating if editing else False
        if not editing:
            self._stale_conflict = False
        self.query_one("#memory_filter", Input).disabled = editing
        self.query_one("#memory_entries", OptionList).disabled = editing
        self.query_one("#memory_preview_scroll").display = not editing
        self.query_one("#memory_editor", TextArea).display = editing
        self.query_one("#memory_reference", Input).display = editing
        self.query_one("#memory_reference", Input).disabled = editing and not creating
        if editing:
            self.call_after_refresh(self.query_one("#memory_editor", TextArea).focus)
        self._refresh_controls()

    # ---- events and actions --------------------------------------------------

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "memory_entries" or event.option_id is None:
            return
        event.stop()
        if self._editing:
            return
        try:
            index = int(event.option_id.removeprefix("memory_entry_"))
            reference = self._entries[index].reference
        except (ValueError, IndexError):
            return
        self._start_load(reference)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "memory_filter" or self._editing:
            return
        self._entries = self._filtered(self._all_entries)
        self._render_roster()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "memory_agent_view": self.action_agent_view,
            "memory_inspect": self.action_inspect_disabled,
            "memory_toggle": self.action_toggle_enabled,
            "memory_new": self.action_create,
            "memory_edit": self.action_edit,
            "memory_delete": self.action_delete,
            "memory_clear": self.action_clear,
            "memory_save": self.action_save,
            "memory_reload": self.action_reload_latest,
            "memory_cancel": self.action_cancel_edit,
        }
        action = actions.get(event.button.id or "")
        if action is not None:
            event.stop()
            action()

    def action_inspect_disabled(self) -> None:
        self._inspect_disabled = not self._inspect_disabled
        self._start_refresh()

    def action_refresh(self) -> None:
        if self._busy or self._editing:
            return
        self._start_refresh(select_reference=self._selected_reference, from_disk=True)

    def action_agent_view(self) -> None:
        if self._busy or self._editing:
            return
        if self._agent_view:
            self._exit_agent_view()
            if self._loaded_entry is not None:
                self._render_entry(self._loaded_entry)
            else:
                self._start_refresh()
            return
        self._work_generation += 1
        generation = self._work_generation
        self._set_busy(True)
        self.run_worker(
            self._load_agent_view(generation),
            name="Preview agent memory context",
            group="memory-screen-read",
            exclusive=True,
        )

    async def _load_agent_view(self, generation: int) -> None:
        try:
            context = await asyncio.to_thread(self._manager.prompt_context)
        except MemoryUnavailableError as error:
            if generation != self._work_generation:
                return
            self._set_busy(False)
            self._enter_agent_view(
                "Agent startup context · none",
                f"The agent receives no memory context: {error}",
                "",
            )
            return
        except Exception as error:
            if generation == self._work_generation:
                self._set_busy(False)
                self.query_one("#memory_notice", Static).update(str(error))
            return
        if generation != self._work_generation:
            return
        self._set_busy(False)
        truncated = " · truncated" if context.truncated else ""
        self._enter_agent_view(
            f"Agent startup context · {context.line_count} lines · {context.byte_count:,} bytes{truncated}",
            context.text,
            " · ".join(context.warnings),
        )

    def _enter_agent_view(self, metadata: str, preview: str, notice: str) -> None:
        self._agent_view = True
        self.query_one("#memory_agent_view", Button).label = "Entries"
        self.query_one("#memory_metadata", Static).update(metadata)
        self.query_one("#memory_preview", Markdown).update(preview)
        self.query_one("#memory_notice", Static).update(notice)

    def _exit_agent_view(self) -> None:
        if not self._agent_view:
            return
        self._agent_view = False
        self.query_one("#memory_agent_view", Button).label = "Agent view"

    def action_toggle_enabled(self) -> None:
        status = self._status
        if status is None or self._owner._memory_mutation_blocked():
            return
        self._run_set_enabled(not status.enabled)

    def _run_set_enabled(self, enabled: bool, *, create_after: bool = False) -> None:
        self._set_busy(True)
        self.run_worker(
            self._set_enabled(enabled, create_after=create_after),
            name="Update project memory",
            group="memory-screen-mutation",
            exclusive=True,
        )

    async def _set_enabled(self, enabled: bool, *, create_after: bool) -> None:
        try:
            await self._owner._apply_memory_enabled(enabled)
        except Exception as error:
            self._render_error(str(error))
            return
        self._inspect_disabled = False
        self._set_busy(False)
        if create_after:
            self._begin_create()
        else:
            self._start_refresh()

    def action_create(self) -> None:
        if self._busy or self._editing or self._owner._memory_mutation_blocked():
            return
        if self._status is not None and not self._status.enabled:

            def enable_then_create(confirmed: bool | None) -> None:
                if confirmed:
                    self._run_set_enabled(True, create_after=True)

            self._owner.push_screen(
                ConfirmSettingsActionScreen(
                    "Enable project memory",
                    "Project memory is off. Enable it and create a new private entry?",
                    "Enable and create",
                ),
                enable_then_create,
            )
            return
        self._begin_create()

    def _begin_create(self) -> None:
        self._loaded_entry = None
        self._stale_conflict = False
        self.query_one("#memory_reference", Input).value = "topics/new.md"
        self.query_one("#memory_editor", TextArea).text = ""
        self.query_one("#memory_metadata", Static).update("Create a private Markdown entry. Use a relative .md path.")
        self.query_one("#memory_notice", Static).update("")
        self._set_edit_mode(True, creating=True)

    def action_edit(self) -> None:
        entry = self._loaded_entry
        if self._busy or self._editing or entry is None or self._owner._memory_mutation_blocked():
            return
        self._exit_agent_view()
        self._stale_conflict = False
        self.query_one("#memory_reference", Input).value = entry.reference
        self.query_one("#memory_editor", TextArea).text = entry.content or ""
        self._set_edit_mode(True)

    def action_cancel_edit(self) -> None:
        self._set_edit_mode(False)
        if self._loaded_entry is not None:
            self._render_entry(self._loaded_entry)

    def action_save(self) -> None:
        if not self._editing or self._owner._memory_mutation_blocked():
            return
        reference = self.query_one("#memory_reference", Input).value.strip()
        content = self.query_one("#memory_editor", TextArea).text
        expected = (
            MISSING_REVISION
            if self._creating
            else (self._loaded_entry.revision if self._loaded_entry is not None else MISSING_REVISION)
        )
        if not reference:
            self.query_one("#memory_notice", Static).update("Enter an entry path before saving.")
            return
        self._set_busy(True)
        self.run_worker(
            self._save(reference, content, expected),
            name="Save project memory entry",
            group="memory-screen-mutation",
            exclusive=True,
        )

    async def _save(self, reference: str, content: str, expected: str) -> None:
        try:
            if self._creating and not self._supports(MemoryCapability.REPLACE):
                result = await asyncio.to_thread(
                    self._manager.append_entry,
                    reference,
                    content,
                    allow_disabled=True,
                )
            else:
                result = await asyncio.to_thread(
                    self._manager.replace_entry,
                    reference,
                    content,
                    expected,
                    allow_disabled=True,
                )
        except Exception as error:
            self._set_busy(False)
            self.query_one("#memory_notice", Static).update(str(error))
            return
        if not result.ok:
            self._set_busy(False)
            self._stale_conflict = result.error == "stale revision"
            if result.current_revision:
                message = (
                    f"{result.error or 'Save failed'}; current revision is "
                    f"{result.current_revision}. Your editor was preserved; "
                    "choose Reload latest to load the current file."
                )
            else:
                message = result.error or "Save failed. Your editor was preserved."
            self.query_one("#memory_notice", Static).update(message)
            self._refresh_controls()
            return
        await self._owner._refresh_agent_memory()
        self._set_edit_mode(False)
        self._owner.notify(f"Saved private memory entry {result.reference}.")
        self._start_refresh(select_reference=result.reference)

    def action_reload_latest(self) -> None:
        if not self._editing or not self._stale_conflict or self._busy:
            return
        reference = self.query_one("#memory_reference", Input).value.strip()

        def reload_after_confirmation(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self._set_busy(True)
            self.run_worker(
                self._reload_latest(reference),
                name="Reload latest project memory entry",
                group="memory-screen-read",
                exclusive=True,
            )

        self._owner.push_screen(
            ConfirmSettingsActionScreen(
                "Reload latest memory entry",
                "Discard the preserved editor text and load the latest private memory revision?",
                "Reload latest",
            ),
            reload_after_confirmation,
        )

    async def _reload_latest(self, reference: str) -> None:
        try:
            entry = await asyncio.to_thread(
                self._manager.read_entry,
                reference,
                allow_disabled=True,
            )
        except Exception as error:
            self._set_busy(False)
            self.query_one("#memory_notice", Static).update(str(error))
            return
        self._set_busy(False)
        if not entry.present:
            self.query_one("#memory_notice", Static).update("The entry is now missing. Your editor was preserved.")
            return
        self._loaded_entry = entry
        self._creating = False
        self._stale_conflict = False
        self.query_one("#memory_reference", Input).value = entry.reference
        self.query_one("#memory_reference", Input).disabled = True
        self.query_one("#memory_editor", TextArea).text = entry.content or ""
        self._render_entry(entry)
        self.query_one("#memory_notice", Static).update("Loaded the latest revision. Review it before saving.")
        self._refresh_controls()

    def action_delete(self) -> None:
        entry = self._loaded_entry
        if entry is None or self._owner._memory_mutation_blocked():
            return
        reference = entry.reference
        revision = entry.revision

        def delete_after_confirmation(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self._set_busy(True)
            self.run_worker(
                self._delete(reference, revision),
                name="Delete project memory entry",
                group="memory-screen-mutation",
                exclusive=True,
            )

        self._owner.push_screen(
            ConfirmSettingsActionScreen(
                "Delete memory entry",
                f"Delete {reference} from private project memory?",
                "Delete entry",
                danger=True,
            ),
            delete_after_confirmation,
        )

    async def _delete(self, reference: str, revision: str) -> None:
        try:
            result = await asyncio.to_thread(
                self._manager.delete_entry,
                reference,
                revision,
                allow_disabled=True,
            )
        except Exception as error:
            self._render_error(str(error))
            return
        if not result.ok:
            self._set_busy(False)
            self.query_one("#memory_notice", Static).update(
                f"{result.error or 'Delete failed'}. Refresh and try again."
            )
            return
        await self._owner._refresh_agent_memory()
        self._loaded_entry = None
        self._owner.notify(f"Deleted private memory entry {reference}.")
        self._start_refresh()

    def action_clear(self) -> None:
        if self._busy or self._editing:
            return
        self._owner.confirm_memory_clear(on_done=self._after_clear)

    def _after_clear(self, _deleted: int) -> None:
        self._loaded_entry = None
        self._start_refresh()

    def action_close(self) -> None:
        if not self._editing:
            self.dismiss()
            return

        def close_after_confirmation(confirmed: bool | None) -> None:
            if confirmed:
                self.dismiss()

        self._owner.push_screen(
            ConfirmSettingsActionScreen(
                "Discard unsaved memory changes",
                "Close Project Memory and discard the current editor contents?",
                "Discard changes",
                danger=True,
            ),
            close_after_confirmation,
        )
