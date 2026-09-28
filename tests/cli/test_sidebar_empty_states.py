"""Empty sidebar rows stay compact without hiding meaningful session state."""

from pathlib import Path

import pytest
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from kolega_code.cli import theme
from kolega_code.cli.app import KolegaCodeApp
from kolega_code.cli.tui.widgets import PlanningMarkdown
from kolega_code.events import AgentEvent
from kolega_code.llm.models import Message, TextBlock
from kolega_code.llm.usage import normalize_usage

from ._app_test_utils import FakeCoderAgent, _build_sub_agent_test_app, extension_by_name
from .test_app_status_usage import _settle
from .test_tui_stylesheet import _wait_for_layout

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [80, 120, 180])
@pytest.mark.parametrize("palette", ["kolega-dark", "nord", "textual-light"])
async def test_empty_rows_are_single_line_and_populated_cards_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, palette: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(width, 48)) as pilot:
        app.theme = palette
        app._set_sidebar_visible(True)
        usage = app.query_one("#status_usage_section")
        tasks = app.query_one("#status_task_list_section")
        task_md = app.query_one("#status_task_list_markdown", PlanningMarkdown)
        usage_text = app.query_one("#status_usage", Static)
        await _wait_for_layout(pilot, lambda: usage.size.height == tasks.size.height == 1)

        assert task_md.source == "Task List · Not set"
        assert str(usage_text.render()) == "Usage · None yet"
        assert tasks.region.y == usage.region.bottom
        assert task_md.styles.padding.top == usage_text.styles.padding.top == 0
        assert task_md.styles.padding.bottom == usage_text.styles.padding.bottom == 0
        assert task_md.styles.color == usage_text.styles.color
        assert isinstance(task_md.content, Text)  # No Markdown paragraph padding.
        assert app.query_one("#status_form", VerticalScroll).max_scroll_x == 0
        assert app.query_one("#composer").has_focus

        app.session.task_list_markdown = "- [ ] Inspect sidebar\n- [x] Preserve existing state"
        app._refresh_planning_sidebar()
        _settle(app._usage_ledger, inp=1200, out=100, cache_read=800)
        app._refresh_status_dashboard()
        await _wait_for_layout(pilot, lambda: usage.size.height > 1 and tasks.size.height > 1)

        assert not usage.has_class("empty-state")
        assert not tasks.has_class("empty-state")
        assert str(usage.styles.border) != "Edges()"
        assert str(tasks.styles.border) != "Edges()"
        assert task_md.styles.padding.bottom == 1
        assert usage_text.styles.padding.top == usage_text.styles.padding.bottom == 1
        assert task_md.source == "- [ ] Inspect sidebar\n- [x] Preserve existing state"
        assert "Requests: 1" in str(usage_text.render())
        assert "Cache reads 800" in str(usage_text.render())

        # A whitespace-only task list is empty for display, never rewritten in storage.
        app.session.task_list_markdown = " \n\t "
        app._refresh_planning_sidebar()
        await _wait_for_layout(pilot, lambda: tasks.size.height == 1)
        assert task_md.source == "Task List · Not set"
        assert app.session.task_list_markdown == " \n\t "
        assert usage.size.height > 1


@pytest.mark.asyncio
async def test_task_tool_contract_and_reset_preserve_lifetime_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 48)) as pilot:
        assert isinstance(app.agent, FakeCoderAgent)
        tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools
        assert await tools["get_task_list"]() == "No task list has been set."
        await tools["update_task_list"]("- [x] Finished work")
        assert await tools["get_task_list"]() == "- [x] Finished work"
        _settle(app._usage_ledger, failed=True)
        app._refresh_status_dashboard()
        tasks = app.query_one("#status_task_list_section")
        usage = app.query_one("#status_usage_section")
        await _wait_for_layout(pilot, lambda: tasks.size.height > 1 and usage.size.height > 1)

        await app._reset_current_thread()
        await _wait_for_layout(pilot, lambda: tasks.size.height == 1)
        assert await tools["get_task_list"]() == "No task list has been set."
        assert app.store.load(app.session.session_id).task_list_markdown == ""
        assert not usage.has_class("empty-state")
        assert "1 failed" in str(app.query_one("#status_usage", Static).render())


@pytest.mark.asyncio
async def test_hidden_sidebar_receives_updates_and_survives_resize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        app._set_sidebar_visible(False)
        app.session.task_list_markdown = "- [ ] Added while hidden"
        _settle(app._usage_ledger, inp=100, out=50)
        app._refresh_planning_sidebar()
        app._refresh_status_dashboard()
        await pilot.resize_terminal(80, 32)
        app._set_sidebar_visible(True)
        tasks = app.query_one("#status_task_list_section")
        usage = app.query_one("#status_usage_section")
        await _wait_for_layout(pilot, lambda: tasks.size.height > 1 and usage.size.height > 1)
        assert app.query_one("#status_task_list_markdown", PlanningMarkdown).source == "- [ ] Added while hidden"
        assert "150" in str(app.query_one("#status_usage", Static).render())
        assert app.query_one("#status_form", VerticalScroll).max_scroll_x == 0

        app._set_sidebar_visible(False)
        app.session.task_list_markdown = ""
        app._refresh_planning_sidebar()
        app._set_sidebar_visible(True)
        await _wait_for_layout(pilot, lambda: tasks.size.height == 1)
        assert usage.size.height > 1


@pytest.mark.asyncio
async def test_restored_session_starts_with_full_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.session.task_list_markdown = "- [x] Recorded task"
        request_id = app._usage_ledger.begin("anthropic", "m")
        usage = normalize_usage({"input_tokens": 100, "output_tokens": 25}, "anthropic", "m")
        app._usage_ledger.record_response(
            request_id, usage, message=Message(role="assistant", content=[TextBlock(text="Recorded")], usage=usage)
        )
        await app._save_session_async()
        assert app._usage_sink is not None
        await app._usage_sink.aclose()

    session = app.store.load(app.session.session_id)
    restored = KolegaCodeApp(
        project_path=app.project_path, config=app.config, mode="code", store=app.store, session=session
    )
    async with restored.run_test(size=(100, 48)) as pilot:
        tasks = restored.query_one("#status_task_list_section")
        usage = restored.query_one("#status_usage_section")
        await _wait_for_layout(pilot, lambda: tasks.size.height > 1 and usage.size.height > 1)
        assert not tasks.has_class("empty-state")
        assert not usage.has_class("empty-state")
        assert restored.query_one("#status_task_list_markdown", PlanningMarkdown).source == "- [x] Recorded task"
        rendered = str(restored.query_one("#status_usage", Static).render())
        assert "125" in rendered
        assert "Requests: 1" in rendered


@pytest.mark.asyncio
async def test_context_distinguishes_unmeasured_from_zero_and_keeps_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        dashboard = Text.from_markup(app._format_status_dashboard()).plain
        assert "Context · Not measured" in dashboard
        app._render_event(
            AgentEvent(
                event_type="llm_context_update",
                sender="coder",
                content={
                    "input_tokens": 0,
                    "max_tokens": 10000,
                    "usage_percentage": 0.0,
                    "compression_threshold": 80.0,
                },
            )
        )
        dashboard = Text.from_markup(app._format_status_dashboard()).plain
        assert "Not measured" not in dashboard
        assert "Context\n" in dashboard and "0.0%" in dashboard
        assert "Tokens: 0 / 10,000" in dashboard
        assert "Compresses at 80%" in dashboard

        app._apply_context_status_update(
            {"usage_percentage": None, "alert_level": "critical", "message": "Context warning"}
        )
        app._apply_compaction_status({"phase": "started", "message": "Compacting context"})
        dashboard = app._format_status_dashboard()
        assert "Context · Not measured" in dashboard
        assert f"[{theme.Color.ERROR}]Context warning[/{theme.Color.ERROR}]" in dashboard
        assert "Compacting context" in dashboard
