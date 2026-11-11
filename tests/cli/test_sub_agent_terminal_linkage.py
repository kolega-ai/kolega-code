import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec

import pytest
from textual.pilot import Pilot
from textual.widgets import Collapsible, Static, TabbedContent

from kolega_code.agent.tool_backend.terminal_tool import TerminalTool
from kolega_code.cli.app import KolegaCodeApp
from kolega_code.cli.tui.state import SubAgentTerminalEntry
from kolega_code.cli.tui.sub_agent_screen import SubAgentInspectorScreen
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager, AgentEvent
from kolega_code.services.terminal import LocalTerminalManager

from ._app_test_utils import _build_sub_agent_test_app, wait_for_layout


def _event(
    event_type: str,
    content: dict[str, Any],
    *,
    info: dict[str, Any] | None = None,
) -> AgentEvent:
    return AgentEvent(
        event_type=event_type,
        sender="worker",
        content=content,
        sub_agent_info={"agent_id": "worker"} if info is None else info,
    )


async def _open_inspector(app: KolegaCodeApp, pilot: Pilot) -> SubAgentInspectorScreen:
    await pilot.press("ctrl+g")
    await wait_for_layout(
        pilot,
        lambda: isinstance(app.screen, SubAgentInspectorScreen) and bool(app.screen.query("#inspector_trajectory")),
    )
    assert isinstance(app.screen, SubAgentInspectorScreen)
    return app.screen


def _body(screen: SubAgentInspectorScreen, step: SubAgentTerminalEntry) -> str:
    widget = screen._step_widgets.get(step.entry_id)
    if widget is None:
        return ""
    bodies = widget.query(".tool-body")
    return str(bodies.first(Static).render()) if bodies else ""


@pytest.mark.asyncio
@pytest.mark.parametrize("first_type", ["terminal_command", "terminal_output"])
async def test_terminal_first_creates_activity_and_updates_open_inspector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, first_type: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        root_writes = Mock()
        monkeypatch.setattr(app._terminal, "write_terminal", root_writes)
        diff_dirty = Mock()
        monkeypatch.setattr(app, "_mark_session_diff_dirty", diff_dirty)
        first = {"command": "printf alpha"} if first_type == "terminal_command" else {"output": "alpha"}
        app._render_event(_event(first_type, {"tool_call_id": "shell-1", **first}))
        assert set(app._sub_agent_activities) == {"worker"}
        activity = app._sub_agent_activities["worker"]
        assert activity.parent_agent_id is None
        assert activity.parent_tool_call_id is None
        step = activity.terminal_steps["shell-1"]
        assert step.tool_call_id == "shell-1"
        screen = await _open_inspector(app, pilot)
        await wait_for_layout(pilot, lambda: step.entry_id in screen._step_widgets)
        widget = screen._step_widgets[step.entry_id]
        await wait_for_layout(pilot, lambda: bool(widget.query(Collapsible)))
        widget.query_one(Collapsible).collapsed = False
        progress = Mock()
        monkeypatch.setattr(app, "_update_activity_progress", progress)

        app._render_event(
            _event("terminal_output", {"tool_call_id": "shell-1", "output": "wrong", "display_output": " beta"})
        )
        app._render_event(
            _event("terminal_output", {"tool_call_id": "shell-1", "output": " gamma", "display_output": None})
        )
        expected = "alpha beta gamma" if first_type == "terminal_output" else " beta gamma"
        await wait_for_layout(pilot, lambda: _body(screen, step) == expected)
        assert step.full_content == expected
        if first_type == "terminal_command":
            assert step.tool_display == {"command": "printf alpha"}
            assert "printf alpha" in screen._trajectory_text(activity)
        assert len(activity.terminal_steps) == 1
        assert screen._selected_key == "worker"
        assert app._terminal_output_buffer == []
        assert app._terminal_has_content is False
        assert str(app.query_one("#events", TabbedContent).get_tab("terminal_pane").label) == "Terminal"
        root_writes.assert_not_called()
        progress.assert_not_called()
        assert diff_dirty.call_count == (3 if first_type == "terminal_output" else 2)


@pytest.mark.asyncio
async def test_parallel_workers_and_commands_with_reused_tool_ids_are_isolated_from_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        root_writes: list[object] = []
        monkeypatch.setattr(app._terminal, "write_terminal", root_writes.append)
        app._turn_active = True
        app._render_event(
            AgentEvent(
                event_type="terminal_command", sender="coder", content={"command": "echo root", "tool_call_id": "same"}
            )
        )
        app._render_event(
            AgentEvent(event_type="terminal_output", sender="coder", content={"output": "ROOT", "tool_call_id": "same"})
        )
        root_activity = app._status_state.activity
        for agent_id, tool_id, command in (
            ("a", "same", "echo ALPHA"),
            ("a", "other", "echo SECOND"),
            ("b", "same", "echo BRAVO"),
        ):
            app._render_event(
                _event("terminal_command", {"command": command, "tool_call_id": tool_id}, info={"agent_id": agent_id})
            )
        progress = Mock()
        monkeypatch.setattr(app, "_update_activity_progress", progress)
        for agent_id, tool_id, output in (
            ("a", "same", "AL"),
            ("b", "same", "BRA"),
            ("a", "other", "SECOND"),
            ("a", "same", "PHA"),
            ("b", "same", "VO"),
        ):
            app._render_event(
                _event("terminal_output", {"output": output, "tool_call_id": tool_id}, info={"agent_id": agent_id})
            )
        app._flush_terminal_output()
        assert [getattr(item, "plain", item) for item in root_writes][1:] == ["ROOT"]
        assert root_activity == "Running terminal command…"
        progress.assert_not_called()
        assert set(app._sub_agent_activities["a"].terminal_steps) == {"same", "other"}
        assert set(app._sub_agent_activities["b"].terminal_steps) == {"same"}

        screen = await _open_inspector(app, pilot)
        for agent_id, expected in (("a", {"same": "ALPHA", "other": "SECOND"}), ("b", {"same": "BRAVO"})):
            screen._selected_key = agent_id
            screen._select_changed()
            activity = app._sub_agent_activities[agent_id]
            await wait_for_layout(
                pilot,
                lambda: all(_body(screen, activity.terminal_steps[key]) == text for key, text in expected.items()),
            )
            assert {widget.entry.tool_call_id for widget in screen._step_widgets.values()} == set(expected)
            for key, text in expected.items():
                assert activity.terminal_steps[key].full_content == text
            copied = screen._trajectory_text(activity)
            assert "ROOT" not in copied
            assert ("BRAVO" if agent_id == "a" else "ALPHA") not in copied


@pytest.mark.asyncio
@pytest.mark.parametrize("info", [{"agent_name": "legacy"}, {"parent_tool_call_id": "legacy-dispatch"}])
async def test_partial_legacy_tagged_terminals_remain_worker_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, info: dict[str, str]
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        writes = Mock()
        monkeypatch.setattr(app._terminal, "write_terminal", writes)
        for command, output in (("echo first", "FIRST"), ("echo second", "SECOND")):
            app._render_event(_event("terminal_command", {"command": command}, info=info))
            app._render_event(_event("terminal_output", {"output": output}, info=info))
        activity = next(iter(app._sub_agent_activities.values()))
        screen = await _open_inspector(app, pilot)
        await wait_for_layout(pilot, lambda: len(screen._step_widgets) == 2)
        assert [step.materialize() for step in activity.steps] == ["FIRST", "SECOND"]
        assert all(step.tool_call_id is None for step in activity.steps)
        assert activity.parent_agent_id is None
        assert activity.parent_tool_call_id == info.get("parent_tool_call_id")
        assert "Parent:" not in screen._trajectory_text(activity)
        app._flush_terminal_output()
        writes.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_parser_state_is_per_worker_and_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        for agent_id, tool_id, output in (
            ("a", "same", "safe\x1b]8;;secret"),
            ("b", "same", "BRAVO"),
            ("a", "other", "OTHER"),
            ("a", "same", "\x1b\\link\x1b]8;;\x1b\\\x07"),
        ):
            app._render_event(
                _event("terminal_output", {"output": output, "tool_call_id": tool_id}, info={"agent_id": agent_id})
            )
        screen = await _open_inspector(app, pilot)
        a = app._sub_agent_activities["a"]
        assert a.terminal_steps["same"].materialize() == "safelink"
        assert a.terminal_steps["other"].materialize() == "OTHER"
        assert app._sub_agent_activities["b"].terminal_steps["same"].materialize() == "BRAVO"
        assert "secret" not in screen._trajectory_text(a)
        assert "\x1b" not in screen._trajectory_text(a)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["none", "tool_result", "tool_error", "worker_finished", "turn_finalized"])
@pytest.mark.parametrize(
    ("prefix", "suffix", "expected"),
    [
        ("safe\x1b[3", "1mred\x1b[0m", "safered"),
        ("safe\x1b]8;;secret", "continuation\x1b\\link\x1b]8;;\x1b\\", "safelink"),
        ("one\r", "\ntwo", "one\ntwo"),
    ],
    ids=["csi", "osc", "crlf"],
)
async def test_terminal_parser_survives_invocation_and_worker_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    boundary: str,
    prefix: str,
    suffix: str,
    expected: str,
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        app._render_event(_event("terminal_output", {"output": prefix, "tool_call_id": "shell"}))
        screen = await _open_inspector(app, pilot)
        if boundary in {"tool_result", "tool_error"}:
            app._render_event(
                _event(
                    "chat_message",
                    {
                        "message_type": boundary,
                        "tool_call_id": "shell",
                        "text": "Status: running\nSession: s_1\nOutput:\n",
                    },
                )
            )
        elif boundary == "worker_finished":
            app._render_event(_event("chat_message", {"status": "STOPPED"}))
        elif boundary == "turn_finalized":
            app._finalize_sub_agent_activities()
        app._render_event(_event("terminal_output", {"output": suffix, "tool_call_id": "shell"}))
        activity = app._sub_agent_activities["worker"]
        step = activity.terminal_steps["shell"]
        assert step.materialize() == expected
        await wait_for_layout(pilot, lambda: _body(screen, step) == expected)
        assert app._terminal_output_buffer == []
        assert "secret" not in screen._trajectory_text(activity)


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True], ids=["pty", "detached"])
async def test_real_terminal_control_sequence_spans_running_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, background: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        partial_seen = asyncio.Event()
        connection = create_autospec(AgentConnectionManager, instance=True)

        async def broadcast(event: AgentEvent, _workspace_id: str, _thread_id: str) -> None:
            app._render_event(event)
            if event.content.get("output") == "\x1b[3":
                partial_seen.set()

        connection.broadcast_event.side_effect = broadcast
        manager = LocalTerminalManager("workspace", "thread", connection, default_workdir=tmp_path)
        manager.auto_activate_venv = False
        terminal = TerminalTool(
            tmp_path,
            "workspace",
            "thread",
            connection,
            AgentConfig(anthropic_api_key="fake-test-key", openai_api_key="fake-test-key"),
            SimpleNamespace(
                current_tool_execution_id="shell",
                current_tool_call_id="wire",
                sub_agent_context={"agent_id": "worker"},
                scratchpad_dir=None,
            ),
            terminal_manager=manager,
        )
        try:
            launched = await terminal.exec_command(
                "stty -echo 2>/dev/null; printf '\\033[3'; read -r value; printf '1mred\\033[0m\\n'",
                yield_time_ms=250,
                background=background,
            )
            assert "Status: running" in launched
            await asyncio.wait_for(partial_seen.wait(), timeout=10)
            app._render_event(
                _event(
                    "chat_message",
                    {"message_type": "tool_result", "tool_call_id": "shell", "text": launched},
                )
            )
            sid = next(iter(manager.sessions))
            await terminal.write_stdin(sid, "release\n", yield_time_ms=1000)
            activity = app._sub_agent_activities["worker"]
            step = activity.terminal_steps["shell"]
            assert step.materialize().splitlines()[0] == "red"
            screen = await _open_inspector(app, pilot)
            await wait_for_layout(pilot, lambda: _body(screen, step).splitlines()[:1] == ["red"])
            assert app._terminal_output_buffer == []
        finally:
            for session in list(manager.sessions.values()):
                if session.running:
                    await session.kill()
            await manager.close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort_key", ["thinking_effort", "effort"])
async def test_sparse_route_preserves_known_effort_but_explicit_updates_replace_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, effort_key: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        route = {"provider": "supplied", "model": "reported-model", effort_key: "high"}
        app._render_event(
            _event(
                "llm_request",
                {},
                info={"agent_id": "worker", "effective_routing": route, "requested_routing": route},
            )
        )
        screen = await _open_inspector(app, pilot)
        app._render_event(
            _event(
                "llm_request",
                {},
                info={"agent_id": "worker", "effective_routing": {"provider": "supplied", "model": "reported-model"}},
            )
        )
        activity = app._sub_agent_activities["worker"]
        assert activity.routing is not None
        assert activity.routing.effort == "high"
        assert activity.routing.overridden is True
        await wait_for_layout(
            pilot, lambda: "reported-model (high)" in str(screen.query_one("#inspector_header", Static).render())
        )
        for effort in (None, "low"):
            app._render_event(
                _event(
                    "llm_request",
                    {},
                    info={"agent_id": "worker", "effective_routing": {**route, effort_key: effort}},
                )
            )
            assert activity.routing.effort == effort
            assert activity.routing.overridden is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["STOPPED", "ERROR"])
async def test_terminal_chunks_survive_tool_result_and_agent_lifecycle_with_zero_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, status: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        app._render_event(_event("terminal_output", {"output": "one", "tool_call_id": "shell"}))
        app._render_event(_event("terminal_output", {"output": " two", "tool_call_id": "shell"}))
        screen = await _open_inspector(app, pilot)
        activity = app._sub_agent_activities["worker"]
        step = activity.terminal_steps["shell"]
        app._render_event(
            _event("chat_message", {"message_type": "tool_result", "tool_call_id": "shell", "text": "Process exited"})
        )
        app._render_event(_event("chat_message", {"status": status, "total_tokens": 0}))
        await wait_for_layout(
            pilot,
            lambda: (
                _body(screen, step) == "one two"
                and "0 tok" in str(screen.query_one("#inspector_header", Static).render())
            ),
        )
        assert step.complete is True
        assert activity.tokens == 0
        assert activity.status == ("failed" if status == "ERROR" else "completed")
        assert activity.current_action == ""
        assert activity.finished_at is not None
        assert activity.entry.complete is True
        assert activity.tool_steps["shell"].full_content == "Process exited"
        assert "0 tok" in activity.entry.content
        assert "Tokens: 0" in screen._trajectory_text(activity)


@pytest.mark.asyncio
async def test_terminal_presentation_finalizes_only_foreground_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        for agent_id, background in (("foreground", False), ("background", True)):
            app._render_event(
                _event(
                    "terminal_output",
                    {"output": agent_id, "tool_call_id": "shell"},
                    info={"agent_id": agent_id, "background": background},
                )
            )
        screen = await _open_inspector(app, pilot)
        app._finalize_sub_agent_activities()
        foreground = app._sub_agent_activities["foreground"]
        background_activity = app._sub_agent_activities["background"]
        assert foreground.status == "stopped"
        assert foreground.terminal_steps["shell"].complete is True
        assert background_activity.status == "running"
        assert background_activity.terminal_steps["shell"].complete is False
        app._render_event(
            _event(
                "terminal_output", {"output": " continues", "tool_call_id": "shell"}, info={"agent_id": "background"}
            )
        )
        screen._selected_key = "background"
        screen._select_changed()
        await wait_for_layout(
            pilot, lambda: _body(screen, background_activity.terminal_steps["shell"]) == "background continues"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type",
    [
        "terminal_output",
        "tool_streaming_update",
        "llm_context_update",
        "compaction_status",
        "llm_request",
        "assistant_delta",
    ],
)
async def test_late_sparse_metadata_enriches_activity_on_every_child_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, event_type: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        app._render_event(
            _event("chat_message", {"status": "GENERATING"}, info={"agent_id": "parent", "agent_name": "planner"})
        )
        app._render_event(_event("terminal_output", {"tool_call_id": "shell", "output": "child"}))
        screen = await _open_inspector(app, pilot)
        screen._selected_key = "worker"
        screen._select_changed()
        app._render_event(
            _event(
                event_type,
                {},
                info={
                    "agent_id": "worker",
                    "parent_agent_id": "parent",
                    "parent_tool_call_id": "dispatch",
                    "depth": 2,
                    "background": True,
                    "task": "inspect",
                    "task_full": "inspect everything",
                    "effective_routing": {"provider": "anthropic", "model": "test-model", "effort": "high"},
                },
            )
        )
        activity = app._sub_agent_activities["worker"]
        await wait_for_layout(
            pilot,
            lambda: (
                "Parent: #1 planner" in str(screen.query_one("#inspector_header", Static).render())
                and "test-model (high)" in str(screen.query_one("#inspector_header", Static).render())
            ),
        )
        assert activity.parent_agent_id == "parent"
        assert activity.parent_tool_call_id == "dispatch"
        assert activity.depth == 2
        assert activity.background is True
        assert activity.task_full == "inspect everything"
        assert activity.steps[0].content == "inspect everything"
        assert activity.routing is not None and activity.routing.effort == "high"
        assert app._sub_agent_by_tool_call["dispatch"] == "worker"
        assert "Parent: #1 planner" in screen._trajectory_text(activity)
        assert "Parent: #1 planner" in screen._roster_row(activity, selected=True).plain
        assert "Parent: #1 planner" in activity.entry.content
        app._render_event(_event("terminal_output", {"tool_call_id": "shell", "output": " tail"}))
        assert activity.parent_agent_id == "parent"
        assert activity.parent_tool_call_id == "dispatch"
        assert activity.depth == 2 and activity.background is True
        assert activity.task_full == "inspect everything"
        assert activity.routing is not None and activity.routing.model == "test-model"


@pytest.mark.asyncio
async def test_root_parent_and_legacy_unknown_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        app._render_event(_event("terminal_output", {"output": "legacy"}))
        activity = app._sub_agent_activities["worker"]
        assert activity.parent_agent_id is None
        screen = await _open_inspector(app, pilot)
        assert "Parent:" not in screen._trajectory_text(activity)
        app._render_event(_event("llm_request", {}, info={"agent_id": "worker", "parent_agent_id": "root"}))
        await wait_for_layout(
            pilot, lambda: "Parent: root" in str(screen.query_one("#inspector_header", Static).render())
        )
        assert activity.parent_agent_id == "root"
        assert "Parent: root" in activity.entry.content
        app._render_event(
            _event("llm_request", {}, info={"agent_id": "worker", "parent_agent_id": None, "depth": None})
        )
        assert activity.parent_agent_id == "root"


@pytest.mark.asyncio
async def test_sparse_and_malformed_metadata_preserves_known_values_and_accepts_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        app._render_event(
            _event(
                "terminal_output",
                {"tool_call_id": "shell", "output": "kept", "total_tokens": 15},
                info={
                    "agent_id": "worker",
                    "agent_name": "named worker",
                    "parent_agent_id": "root",
                    "parent_tool_call_id": "dispatch",
                    "depth": 3,
                    "background": True,
                    "effective_routing": {"provider": "anthropic", "model": "first"},
                },
            )
        )
        screen = await _open_inspector(app, pilot)
        app._render_event(
            _event(
                "llm_request",
                {"total_tokens": False},
                info={
                    "agent_id": "worker",
                    "agent_name": "",
                    "parent_agent_id": None,
                    "parent_tool_call_id": None,
                    "depth": False,
                    "background": None,
                    "effective_routing": {"provider": "broken"},
                },
            )
        )
        activity = app._sub_agent_activities["worker"]
        assert activity.agent_name == "named worker"
        assert activity.parent_agent_id == "root"
        assert activity.parent_tool_call_id == "dispatch"
        assert activity.depth == 3
        assert activity.background is True
        assert activity.tokens == 15
        assert activity.routing is not None and activity.routing.model == "first"
        app._render_event(
            _event(
                "llm_request",
                {"total_tokens": None, "tokens": 0},
                info={"agent_id": "worker", "requested_routing": {"model": "second"}},
            )
        )
        assert activity.routing is not None and activity.routing.overridden is True
        app._render_event(
            _event(
                "terminal_output",
                {"tool_call_id": "shell", "display_output": "", "output": "must not fall back"},
                info={
                    "agent_id": "worker",
                    "background": False,
                    "effective_routing": {"provider": "anthropic", "model": "second"},
                },
            )
        )
        await wait_for_layout(pilot, lambda: "second" in str(screen.query_one("#inspector_header", Static).render()))
        assert activity.background is False
        assert activity.tokens == 0
        assert activity.routing is not None and activity.routing.overridden is True
        assert activity.terminal_steps["shell"].materialize() == "kept"
