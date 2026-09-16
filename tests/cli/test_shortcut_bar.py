from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.pilot import Pilot
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Static
from textual.widgets._footer import FooterKey
from textual.widgets.option_list import Option

from kolega_code.cli.slash_commands import TUI_COMMAND_ENTRIES
from kolega_code.cli.tui.shortcut_bar import ContextFooter, Shortcut, ShortcutHelpScreen
from kolega_code.cli.tui.widgets import ActionList, ChatComposer, CompletionDropdown, command_completion_item

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


class OtherModal(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        yield Input(id="modal-input")
        yield Footer()


class ShortcutApp(App[None]):
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    ChatComposer { height: 5; }
    CompletionDropdown, ActionList { height: 4; }
    """
    BINDINGS = [
        Binding("f1", "shortcut_help", "Shortcuts", priority=True),
        Binding("shift+tab", "toggle_interaction_mode", "Plan/Build", priority=True),
        Binding("ctrl+o", "toggle_sidebar", "Sidebar", priority=True),
        Binding("ctrl+g", "open_sub_agent", "Agents", priority=True),
        Binding("ctrl+p", "toggle_permission_mode", "Permissions", priority=True),
        Binding("ctrl+r", "open_changes", "Changes", priority=True),
        Binding("ctrl+c", "cancel_generation", "Cancel"),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+q", "record_quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent: object | None = object()
        self.agent_worker: object | None = None
        self._turn_active: bool = False
        self._pending_question: object | None = None
        self.prompt_active: bool = False
        self.calls: list[str] = []
        self.submissions: list[str] = []
        self.selections: list[int] = []

    def compose(self) -> ComposeResult:
        yield CompletionDropdown(id="completion_dropdown")
        yield ActionList("First", "Second", id="question_actions")
        yield ChatComposer(id="composer")
        yield Input(id="unrelated")
        yield ContextFooter()

    def on_mount(self) -> None:
        self.query_one(CompletionDropdown).display = False
        self.query_one(ActionList).display = False
        self.screen.set_focus(self.query_one(ChatComposer))

    def _active_prompt_actions(self) -> ActionList | None:
        actions = self.query_one(ActionList)
        return actions if self.prompt_active and actions.display else None

    def action_shortcut_help(self) -> None:
        if isinstance(self.screen, ShortcutHelpScreen):
            self.screen.action_close()
        elif len(self.screen_stack) == 1:
            self.push_screen(ShortcutHelpScreen())

    def action_toggle_interaction_mode(self) -> None:
        self.calls.append("mode")

    def action_toggle_sidebar(self) -> None:
        self.calls.append("sidebar")

    def action_open_sub_agent(self) -> None:
        self.calls.append("agents")

    def action_toggle_permission_mode(self) -> None:
        self.calls.append("permissions")

    def action_open_changes(self) -> None:
        self.calls.append("changes")

    def action_cancel_generation(self) -> None:
        self.calls.append("cancel")

    def action_record_quit(self) -> None:
        self.calls.append("quit")

    def on_chat_composer_submitted(self, event: ChatComposer.Submitted) -> None:
        self.submissions.append(event.value)

    def on_option_list_option_selected(self, event: ActionList.OptionSelected) -> None:
        if isinstance(event.option_list, ActionList):
            self.selections.append(event.option_index)


async def wait_for(pilot: Pilot, condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        await pilot.pause()
        if condition():
            return
    assert condition()


def labels(footer: ContextFooter) -> list[str]:
    return [key.description for key in footer.query(FooterKey) if key.key != "f1"]


def key_widget(footer: ContextFooter, key: str) -> FooterKey:
    return next(widget for widget in footer.query(FooterKey) if widget.key == key)


async def test_idle_hints_and_real_click_dispatch() -> None:
    app = ShortcutApp()
    async with app.run_test(size=(120, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        composer = app.query_one(ChatComposer)
        await wait_for(pilot, lambda: labels(footer) == ["send", "newline", "commands", "Plan/Build", "sidebar"])
        assert isinstance(footer, Footer)
        composer.load_text("draft")
        composer.move_cursor((0, 5))
        await pilot.click(key_widget(footer, "shift+enter"))
        await wait_for(pilot, lambda: composer.text == "draft\n")
        await pilot.click(key_widget(footer, "enter"))
        await wait_for(pilot, lambda: app.submissions == ["draft\n"])
        await pilot.click(key_widget(footer, "shift+tab"))
        await pilot.click(key_widget(footer, "ctrl+o"))
        await wait_for(pilot, lambda: app.calls == ["mode", "sidebar"])
        composer.load_text("")
        await pilot.click(key_widget(footer, "/"))
        await wait_for(pilot, lambda: composer.text == "/")
        assert app.screen.focused is composer


@pytest.mark.parametrize("busy_source", ["_turn_active", "agent_worker"])
async def test_working_context_accepts_either_busy_signal(busy_source: str) -> None:
    app = ShortcutApp()
    async with app.run_test(size=(120, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        setattr(app, busy_source, True if busy_source == "_turn_active" else object())
        footer.refresh_context()
        await wait_for(pilot, lambda: labels(footer) == ["queue", "newline", "sidebar", "agents"])
        await pilot.click(key_widget(footer, "ctrl+g"))
        await wait_for(pilot, lambda: app.calls == ["agents"])
        setattr(app, busy_source, False if busy_source == "_turn_active" else None)
        footer.refresh_context()
        await wait_for(pilot, lambda: "send" in labels(footer))


async def test_question_list_and_text_hints_follow_focus_not_pending_flags() -> None:
    app = ShortcutApp()
    async with app.run_test(size=(120, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        actions = app.query_one(ActionList)
        composer = app.query_one(ChatComposer)
        app._pending_question = object()
        app.prompt_active = True
        actions.display = True
        actions.highlighted = 0
        app.screen.set_focus(actions)
        await wait_for(pilot, lambda: labels(footer) == ["choose", "select", "sidebar"])
        await pilot.click(key_widget(footer, "down"))
        await wait_for(pilot, lambda: actions.highlighted == 1)
        await pilot.click(key_widget(footer, "enter"))
        await wait_for(pilot, lambda: app.selections == [1])
        app.screen.set_focus(composer)
        await wait_for(pilot, lambda: labels(footer) == ["answer", "newline", "sidebar"])
        composer.load_text("custom answer")
        await pilot.click(key_widget(footer, "enter"))
        await wait_for(pilot, lambda: app.submissions == ["custom answer"])
        app.screen.set_focus(app.query_one(Input))
        await wait_for(pilot, lambda: labels(footer) == ["sidebar"])
        app.screen.set_focus(actions)
        await wait_for(pilot, lambda: "select" in labels(footer))


async def test_completion_precedes_question_and_dismiss_click_is_not_cancel() -> None:
    app = ShortcutApp()
    async with app.run_test(size=(120, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        dropdown = app.query_one(CompletionDropdown)
        app._pending_question = object()
        dropdown.add_options([Option("one"), Option("two")])
        dropdown.highlighted = 0
        dropdown.display = True
        footer.refresh_context()
        await wait_for(pilot, lambda: labels(footer) == ["choose", "select", "dismiss", "sidebar"])
        await pilot.click(key_widget(footer, "down"))
        await wait_for(pilot, lambda: dropdown.highlighted == 1)
        await pilot.click(key_widget(footer, "escape"))
        await wait_for(pilot, lambda: not dropdown.display)
        footer.refresh_context()  # Parent's dropdown change hook.
        await wait_for(pilot, lambda: labels(footer) == ["answer", "newline", "sidebar"])
        assert app.calls == []


async def test_disabled_and_disconnected_composer_do_not_advertise_send() -> None:
    app = ShortcutApp()
    async with app.run_test(size=(120, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        composer = app.query_one(ChatComposer)
        app.agent = None
        footer.refresh_context()
        await wait_for(pilot, lambda: labels(footer) == ["newline", "commands", "Plan/Build", "sidebar"])
        composer.disabled = True
        footer.refresh_context()
        await wait_for(pilot, lambda: labels(footer) == ["sidebar"])


@pytest.mark.parametrize("width", [1, 2, 7, 14, 24, 40, 60, 80, 120])
async def test_resize_fits_whole_keys_without_clipping_and_keeps_help(width: int) -> None:
    app = ShortcutApp()
    async with app.run_test(size=(120, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        await wait_for(pilot, lambda: len(list(footer.query(FooterKey))) == 6)
        await pilot.resize_terminal(width, 30)

        def fits() -> bool:
            children = list(footer.query(FooterKey))
            return (
                footer.size.width == width
                and bool(children)
                and all(
                    child.region.width == cell_len(child.render().plain)
                    and child.region.x >= 0
                    and child.region.right <= width
                    for child in children
                )
                and children[-1].key == "f1"
            )

        await wait_for(pilot, fits)
        assert footer.max_scroll_x == 0
        for child in footer.query(FooterKey):
            assert child.region.height == 1
        await pilot.resize_terminal(120, 30)
        await wait_for(pilot, lambda: len(list(footer.query(FooterKey))) == 6)


async def test_hidden_keys_still_execute_and_f1_opens_help_at_one_column() -> None:
    app = ShortcutApp()
    async with app.run_test(size=(1, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        await wait_for(pilot, lambda: [key.key for key in footer.query(FooterKey)] == ["f1"])
        await pilot.press("ctrl+p", "ctrl+r", "shift+tab", "ctrl+g", "escape", "ctrl+q")
        await wait_for(pilot, lambda: app.calls == ["permissions", "changes", "mode", "agents", "cancel", "quit"])
        await pilot.press("f1")
        await wait_for(pilot, lambda: isinstance(app.screen, ShortcutHelpScreen))
        await pilot.press("escape")
        await wait_for(pilot, lambda: len(app.screen_stack) == 1)


@pytest.mark.parametrize("close_method", ["escape", "f1", "button"])
async def test_help_click_lists_fallbacks_and_restores_draft_focus_and_selection(close_method: str) -> None:
    app = ShortcutApp()
    async with app.run_test(size=(100, 35)) as pilot:
        footer = app.query_one(ContextFooter)
        composer = app.query_one(ChatComposer)
        composer.load_text("unfinished\nanswer")
        composer.move_cursor((1, 3))
        selection = composer.selection
        await wait_for(pilot, lambda: any(key.key == "f1" for key in footer.query(FooterKey)))
        await pilot.click(key_widget(footer, "f1"))
        await wait_for(pilot, lambda: isinstance(app.screen, ShortcutHelpScreen))
        help_screen = app.screen
        assert isinstance(help_screen, ShortcutHelpScreen)
        reference = help_screen.query_one("#shortcut-help-content", Static).render()
        assert isinstance(reference, Content)
        text = reference.plain
        for expected in ("Ctrl+J", "Ctrl+Enter", "Alt+V", "/attach", "/help", "ctrl+p", "ctrl+q", "ctrl+u"):
            assert expected in text
        assert len(list(help_screen.query(ContextFooter))) == 0
        if close_method == "button":
            await pilot.click(help_screen.query_one(Button))
        else:
            await pilot.press(close_method)
        await wait_for(pilot, lambda: len(app.screen_stack) == 1 and app.screen.focused is composer)
        assert composer.text == "unfinished\nanswer"
        assert composer.selection == selection
        assert app.calls == []
        assert app.submissions == []


async def test_modal_footer_untouched_and_resume_recomputes_context() -> None:
    app = ShortcutApp()
    async with app.run_test(size=(120, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        await wait_for(pilot, lambda: "send" in labels(footer))
        await app.push_screen(OtherModal())
        await wait_for(pilot, lambda: isinstance(app.screen, OtherModal))
        assert type(app.screen.query_one(Footer)) is Footer
        app._turn_active = True
        await pilot.press("escape")
        await wait_for(pilot, lambda: len(app.screen_stack) == 1 and "queue" in labels(footer))


async def test_repeated_invalidations_do_not_remount_unchanged_keys() -> None:
    app = ShortcutApp()
    async with app.run_test(size=(120, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        await wait_for(pilot, lambda: "send" in labels(footer))
        original = list(footer.query(FooterKey))
        for _ in range(10):
            footer.refresh_context()
            app.screen.refresh_bindings()
        await pilot.pause()
        await pilot.pause()
        assert list(footer.query(FooterKey)) == original


async def test_width_measurement_uses_terminal_cells() -> None:
    assert Shortcut("x", "界", "e\u0301").cell_width == 6


@pytest.mark.parametrize("selection_key", ["click", "tab"])
async def test_completion_select_uses_real_composer_handler(selection_key: str) -> None:
    app = ShortcutApp()
    async with app.run_test(size=(100, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        composer = app.query_one(ChatComposer)
        dropdown = app.query_one(CompletionDropdown)
        entry = TUI_COMMAND_ENTRIES[0]
        composer.load_text("/")
        composer.move_cursor((0, 1))
        dropdown.open_with([command_completion_item(entry)])
        app._pending_question = object()
        footer.refresh_context()
        await wait_for(pilot, lambda: "select" in labels(footer))
        if selection_key == "click":
            await pilot.click(key_widget(footer, "enter"))
        else:
            await pilot.press("tab")
        await wait_for(pilot, lambda: composer.text == entry.token + " " and not dropdown.display)
        assert app.submissions == []
        assert app.screen.focused is composer


async def test_help_restores_option_focus_and_slash_help_still_submits() -> None:
    app = ShortcutApp()
    async with app.run_test(size=(100, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        actions = app.query_one(ActionList)
        app.prompt_active = True
        actions.display = True
        actions.highlighted = 1
        app.screen.set_focus(actions)
        await wait_for(pilot, lambda: "choose" in labels(footer))
        await pilot.press("f1")
        await wait_for(pilot, lambda: isinstance(app.screen, ShortcutHelpScreen))
        await pilot.press("escape")
        await wait_for(pilot, lambda: len(app.screen_stack) == 1 and app.screen.focused is actions)
        assert actions.highlighted == 1
        composer = app.query_one(ChatComposer)
        app.screen.set_focus(composer)
        composer.load_text("/help")
        await pilot.press("enter")
        await wait_for(pilot, lambda: app.submissions == ["/help"])
        assert len(app.screen_stack) == 1


async def test_unicode_hint_layout_fits_without_partial_items(monkeypatch: pytest.MonkeyPatch) -> None:
    app = ShortcutApp()
    async with app.run_test(size=(10, 30)) as pilot:
        footer = app.query_one(ContextFooter)
        monkeypatch.setattr(footer, "context_shortcuts", lambda: (Shortcut("x", "界", "e\u0301"),))
        footer.refresh_context()
        await wait_for(pilot, lambda: labels(footer) == ["e\u0301"])
        await wait_for(pilot, lambda: key_widget(footer, "x").region.width == 4)
        await pilot.resize_terminal(6, 30)
        await wait_for(pilot, lambda: [key.key for key in footer.query(FooterKey)] == ["f1"])
        await pilot.resize_terminal(7, 30)
        await wait_for(pilot, lambda: labels(footer) == ["e\u0301"] and key_widget(footer, "f1").region.right == 7)
