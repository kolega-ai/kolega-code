"""Task-state styling is display-only, static, selectable, and width-aware."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

import pytest
from rich.console import Console
from rich.segment import Segment
from textual import events
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.pilot import Pilot
from textual.selection import Selection

from kolega_code.cli.tui.task_list import TaskListMarkdown
from kolega_code.cli.tui.widgets import PlanningMarkdown

from ._app_test_utils import FakeCoderAgent, _build_sub_agent_test_app, extension_by_name


async def _wait(pilot: Pilot, predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    assert predicate(), "Task-list layout did not settle."


def _render(source: str, width: int = 60) -> list[Segment]:
    console = Console(width=width)
    return list(console.render(TaskListMarkdown(source, code_theme="monokai")))


def _tagged_text(segments: list[Segment], checked: bool, *, marker: bool = False) -> str:
    return "".join(
        segment.text
        for segment in segments
        if segment.style is not None
        and segment.style.meta.get("task_checked") is checked
        and bool(segment.style.meta.get("task_marker")) is marker
    )


def test_tasks_have_distinct_markers_without_extra_bullets_or_source_mutation() -> None:
    source = "- [ ] **Inspect** `src/app.py`\n- [x] Completed work\n- [X] Also complete"
    rendered = _render(source)
    text = "".join(segment.text for segment in rendered)
    assert "[ ] Inspect src/app.py" in text
    assert "[x] Completed work" in text
    assert "[x] Also complete" in text
    assert "•" not in text
    assert "Inspect" in _tagged_text(rendered, False)
    assert "Completed work" in _tagged_text(rendered, True)
    assert "Also complete" in _tagged_text(rendered, True)
    assert any(segment.text == "Inspect" and segment.style and segment.style.bold for segment in rendered)
    document = TaskListMarkdown(source, code_theme="monokai")
    assert document.markup == source


@pytest.mark.parametrize(
    "source",
    [
        "[x] Not a list",
        "- \\[x] Escaped checkbox",
        "- `[x]` Code span",
        "```\n- [x] Fenced example\n```",
        "    - [x] Indented code",
        "- [x](https://example.invalid) Link label",
        "- [x]attached",
        "- [not] Not a checkbox",
        "- Ordinary list item",
    ],
)
def test_only_real_checkbox_items_receive_task_styles(source: str) -> None:
    segments = _render(source)
    assert not any(segment.style and "task_checked" in segment.style.meta for segment in segments)
    assert not any(segment.style and segment.style.link for segment in segments)


def test_nested_tasks_do_not_inherit_parent_completion_and_numbers_survive() -> None:
    source = (
        "3. [x] Parent finished\n"
        "   - [ ] Child pending\n"
        "   - [X] Child complete\n"
        "4. [ ] Next parent\n\n"
        "   Its continuation paragraph.\n\n"
        "## Notes\n\nOrdinary notes."
    )
    segments = _render(source)
    text = "".join(segment.text for segment in segments)
    assert "3. [x] Parent finished" in text
    assert "4. [ ] Next parent" in text
    assert "Parent finished" in _tagged_text(segments, True)
    assert "Child complete" in _tagged_text(segments, True)
    assert "Child pending" in _tagged_text(segments, False)
    assert "Its continuation paragraph." in _tagged_text(segments, False)
    assert "Child pending" not in _tagged_text(segments, True)
    assert "Notes" in text and "Ordinary notes." in text
    assert "Ordinary notes." not in _tagged_text(segments, False)


@pytest.mark.parametrize("width", [1, 2, 4, 8, 20, 40])
def test_wrapped_unicode_tasks_fit_available_cells(width: int) -> None:
    console = Console(width=width)
    document = TaskListMarkdown("- [x] Verify 測試 e\u0301 and wrapping\n- [ ] Next task", code_theme="monokai")
    lines = console.render_lines(document, console.options)
    assert all(Segment.get_line_length(line) <= width for line in lines)
    if width >= 8:
        assert "Verify" in "".join(_tagged_text(line, True).strip() for line in lines)


def test_empty_tasks_remain_visible_and_links_are_not_activated() -> None:
    segments = _render("- [x]\n- [ ]\n- [ ] Read [guide](https://example.invalid/guide)")
    text = "".join(segment.text for segment in segments)
    assert "[x]" in text and text.count("[ ]") == 2
    assert "guide" in text
    assert not any(segment.style and segment.style.link for segment in segments)


class _TaskApp(App[None]):
    CSS = "PlanningMarkdown { height: auto; width: 100%; }"

    def compose(self) -> ComposeResult:
        yield PlanningMarkdown(
            "- [ ] Pending **important** work\n- [x] Completed `src/app.py` with a longer wrapped description",
            task_list=True,
            id="tasks",
        )


def _segments(widget: PlanningMarkdown) -> list[Segment]:
    return [segment for y in range(widget.content_size.height) for segment in widget.render_line(y)]


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [20, 32, 80])
@pytest.mark.parametrize("theme_name", ["textual-dark", "textual-light"])
async def test_static_styles_wrapping_selection_and_theme_changes(width: int, theme_name: str) -> None:
    app = _TaskApp()
    app.theme = theme_name
    async with app.run_test(size=(width, 30)) as pilot:
        tasks = app.query_one("#tasks", PlanningMarkdown)
        await _wait(pilot, lambda: tasks.content_size.height >= 2 and tasks.content_size.width == width)
        original = tasks.source
        rendered = _segments(tasks)
        pending_marker = [s for s in rendered if s.style and s.style.meta.get("task_marker") and "[ ]" in s.text]
        done_marker = [s for s in rendered if s.style and s.style.meta.get("task_marker") and "[x]" in s.text]
        done_text = [
            s
            for s in rendered
            if s.text.strip() and s.style and s.style.meta.get("task_checked") and not s.style.meta.get("task_marker")
        ]
        assert pending_marker and done_marker and done_text
        assert all(s.style and s.style.bold and not s.style.strike for s in pending_marker)
        assert all(s.style and not s.style.strike for s in done_marker)
        assert all(s.style and s.style.strike and not s.style.bold for s in done_text)
        assert all(
            s.style and s.style.color == tasks.get_component_rich_style("task-list--completed").color for s in done_text
        )
        assert not any(s.style and (s.style.link or s.style.blink) for s in rendered)
        assert "Pending" not in "".join(s.text for s in rendered if s.style and s.style.strike)
        assert all(tasks.render_line(y).cell_length <= width for y in range(tasks.content_size.height))
        selected = tasks.get_selection(Selection(None, None))
        assert selected is not None
        assert "[ ] Pending" in selected[0] and "[x] Completed" in selected[0]
        assert "**" not in selected[0] and "`" not in selected[0]
        assert all(s.style and "offset" in s.style.meta for s in rendered if not s.control)

        app.theme = "textual-light" if theme_name == "textual-dark" else "textual-dark"
        await _wait(
            pilot,
            lambda: all(
                s.style and s.style.color == tasks.get_component_rich_style("task-list--completed").color
                for s in _segments(tasks)
                if s.text.strip() and s.style and s.style.meta.get("task_checked")
            ),
        )
        assert tasks.source == original
        assert len(tasks.children) == 0


@pytest.mark.asyncio
async def test_mouse_drag_copies_task_text_without_toggling() -> None:
    app = _TaskApp()
    async with app.run_test(size=(80, 20)) as pilot:
        tasks = app.query_one("#tasks", PlanningMarkdown)
        await _wait(pilot, lambda: tasks.region.height == 2)
        original = tasks.source
        await pilot.mouse_down(tasks, offset=(4, 0))
        await pilot._post_mouse_events([events.MouseMove], tasks, offset=(10, 0), button=1)
        await pilot.mouse_up(tasks, offset=(10, 0))
        assert app.screen.get_selected_text() == "Pending"
        await pilot.click(tasks, offset=(1, 0))
        assert tasks.source == original


@pytest.mark.asyncio
async def test_no_color_keeps_checkbox_and_strikethrough_states(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    app = _TaskApp()
    async with app.run_test(size=(40, 20)) as pilot:
        tasks = app.query_one("#tasks", PlanningMarkdown)
        await _wait(pilot, lambda: tasks.content_size.height >= 3)
        strips = tasks.render_lines(Region(0, 0, tasks.size.width, tasks.size.height))
        text = "".join(strip.text for strip in strips)
        assert "[ ] Pending" in text and "[x] Completed" in text
        struck = "".join(s.text for strip in strips for s in strip if s.style and s.style.strike)
        assert "Completed" in struck and "description" in struck
        assert "Pending" not in struck
        assert not any(s.style and (s.style.link or s.style.blink) for strip in strips for s in strip)


@pytest.mark.asyncio
async def test_long_task_list_stays_one_widget_and_unchanged_refresh_preserves_scroll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        app._set_sidebar_visible(True)
        app.session.task_list_markdown = "\n".join(f"- [ ] Task {i}: check the sidebar." for i in range(200))
        app._refresh_planning_sidebar()
        view = app.query_one("#status_form", VerticalScroll)
        tasks = app.query_one("#status_task_list_markdown", PlanningMarkdown)
        await _wait(pilot, lambda: tasks.content_size.height >= 200 and view.max_scroll_y > 30)
        view.scroll_to(y=30, animate=False, immediate=True)
        await _wait(pilot, lambda: view.scroll_y == 30)
        renderable = tasks.content
        app._refresh_planning_sidebar()
        await pilot.pause()
        assert view.scroll_y == 30
        assert tasks.content is renderable
        assert len(tasks.children) == 0
        await pilot.resize_terminal(80, 30)
        await _wait(pilot, lambda: view.size.width <= 40)
        assert view.max_scroll_x == 0
        assert tasks.source == app.session.task_list_markdown


@pytest.mark.asyncio
async def test_production_task_updates_preserve_source_caching_and_saved_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 50)) as pilot:
        assert isinstance(app.agent, FakeCoderAgent)
        tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools
        app._set_sidebar_visible(True)
        tasks = app.query_one("#status_task_list_markdown", PlanningMarkdown)
        source = "- [ ] Inspect sidebar\n- [x] Keep saved state"
        await tools["update_task_list"](source)
        await _wait(pilot, lambda: tasks.source == source and tasks.content_size.height == 2)
        assert isinstance(tasks.content, TaskListMarkdown)
        assert "Keep saved state" in "".join(s.text for s in _segments(tasks) if s.style and s.style.strike)
        assert await tools["get_task_list"]() == source
        renderable = tasks.content
        app._refresh_planning_sidebar()
        assert tasks.content is renderable
        assert len(tasks.children) == 0

        for checkbox in ("[x]", "[ ]"):
            source = f"- {checkbox} Inspect sidebar\n- [x] Keep saved state"
            await tools["update_task_list"](source)
            await _wait(pilot, lambda: tasks.source == source)
            completed = "".join(s.text for s in _segments(tasks) if s.style and s.style.strike)
            assert ("Inspect sidebar" in completed) is (checkbox == "[x]")
            assert app.session.task_list_markdown == source
            assert await tools["get_task_list"]() == source
        await app._save_session_async()
        assert app.store.load(app.session.session_id).task_list_markdown == source
        # Styling is opt-in for the task list, not the separate plan document.
        app._latest_plan = "- [x] A plan checkbox"
        app._refresh_planning_sidebar()
        plan = app.query_one("#planning_plan_markdown", PlanningMarkdown)
        assert not isinstance(plan.content, TaskListMarkdown)
        await tools["update_task_list"]("")
        await _wait(pilot, lambda: tasks.has_class("empty-state") and tasks.content_size.height == 1)
        assert tasks.source == "Task List · Not set"
        assert not any(s.style and s.style.strike for s in _segments(tasks))
