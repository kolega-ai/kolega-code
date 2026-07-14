# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

from kolega_code.cli.tui import agent_runtime as agent_runtime_module

from kolega_code.config import ModelProvider
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
)
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.events import AgentEvent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    MOONSHOT_K26_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
)
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import (
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    question_payload,
    renderable_text,
)


@pytest.mark.asyncio
async def test_sub_agent_context_update_does_not_stomp_main_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        # Main agent reports its context usage -> the status dashboard reflects it.
        app._render_event(
            AgentEvent(
                event_type="llm_context_update",
                sender="coder",
                content={
                    "input_tokens": 123456,
                    "max_tokens": 200000,
                    "usage_percentage": 61.7,
                    "alert_level": "normal",
                    "message": None,
                    "compression_threshold": 80.0,
                },
            )
        )
        assert "61.7%" in app._format_status_dashboard()

        # A sub-agent reports its much smaller context usage. It must NOT overwrite the
        # main dashboard; it lands on the sub-agent's own card instead.
        app._render_event(_sub_agent_context_event(3.0, input_tokens=6000))

        dashboard = app._format_status_dashboard()
        assert "61.7%" in dashboard  # main agent's value preserved
        assert "3.0%" not in dashboard

        activities = list(app._sub_agent_activities.values())
        assert len(activities) == 1
        assert activities[0].context_percentage == 3.0
        assert activities[0].context_input_tokens == 6000
        # The cumulative-token field is a different metric and stays untouched.
        assert activities[0].tokens is None
        # The per-agent context shows on its own card.
        assert "ctx 3%" in activities[0].entry.content


@pytest.mark.asyncio
async def test_concurrent_sub_agent_context_updates_keep_main_dashboard_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="llm_context_update",
                sender="coder",
                content={
                    "input_tokens": 100000,
                    "max_tokens": 200000,
                    "usage_percentage": 50.0,
                    "alert_level": "normal",
                    "message": None,
                    "compression_threshold": 80.0,
                },
            )
        )

        # Several sub-agents interleave context updates; none may move the main bar.
        for pct, aid in [(2.0, "agent-1"), (4.0, "agent-2"), (1.5, "agent-3")]:
            app._render_event(_sub_agent_context_event(pct, agent_id=aid, agent_name=aid))

        assert "50.0%" in app._format_status_dashboard()
        assert len(app._sub_agent_activities) == 3


@pytest.mark.asyncio
async def test_sub_agent_status_update_routes_to_card_not_main_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        # Create the sub-agent card first; this is what legitimately updates the main
        # activity line (with the "running sub-agent" notice), before the spy is set.
        app._render_event(_sub_agent_event(text="working"))

        calls: list = []
        monkeypatch.setattr(app, "_update_activity_progress", lambda *a, **k: calls.append(a))

        app._render_event(
            AgentEvent(
                event_type="llm_status_update",
                sender="general-agent",
                content={"status": "overloaded", "message": "Provider overloaded, retrying"},
                sub_agent_info={
                    "agent_id": "agent-1",
                    "agent_name": "general-agent",
                    "task": "inspect sessions",
                    "parent_tool_call_id": "tc-1",
                    "conversation_id": None,
                    "depth": 1,
                },
            )
        )

        assert calls == []  # the main activity line is untouched
        activity = next(iter(app._sub_agent_activities.values()))
        assert "Provider overloaded, retrying" in activity.last_activity


@pytest.mark.asyncio
async def test_sub_agent_stream_chunks_group_into_single_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(uuid="u1", text="The session store wri"))
        app._render_event(_sub_agent_event(uuid="u1", text="tes JSON records"))

        entries = _sub_agent_entries(app)
        assert len(entries) == 1
        assert not any(entry.kind == "message" for entry in app.conversation_entries)
        assert "general-agent" in entries[0].content
        assert "#1" in entries[0].content
        assert "The session store writes JSON records" in entries[0].content
        assert "Task: inspect sessions" in entries[0].content


@pytest.mark.asyncio
async def test_parallel_sub_agents_create_separate_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(agent_id="a1", task="task one", uuid="u1", text="alpha"))
        app._render_event(
            _sub_agent_event(agent_id="a2", task="task two", parent_tool_call_id="tc-2", uuid="u2", text="beta")
        )
        app._render_event(_sub_agent_event(agent_id="a1", task="task one", uuid="u1", text=" more"))

        entries = _sub_agent_entries(app)
        assert len(entries) == 2
        assert "#1" in entries[0].content and "alpha more" in entries[0].content
        assert "#2" in entries[1].content and "beta" in entries[1].content
        assert "alpha" not in entries[1].content


@pytest.mark.asyncio
async def test_sub_agent_tool_events_update_counters_not_top_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            _sub_agent_event(
                message_type="tool_call", text="Calling search_codebase", tool_description="search_codebase"
            )
        )
        app._render_event(
            _sub_agent_event(message_type="tool_result", text="found things", tool_description="search_codebase")
        )

        assert not any(entry.kind.startswith("tool") for entry in app.conversation_entries)
        entries = _sub_agent_entries(app)
        assert len(entries) == 1
        assert "1 tool" in entries[0].content
        assert "last: search_codebase done" in entries[0].content
        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.tool_calls == 1


@pytest.mark.asyncio
async def test_sub_agent_status_events_complete_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(status="GENERATING", message="Starting general-agent task"))
        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.status == "running"
        assert activity.entry.complete is False

        app._render_event(_sub_agent_event(status="STOPPED", message="Completed general-agent task"))
        assert activity.status == "completed"
        assert activity.entry.complete is True
        assert "completed in" in activity.entry.content


@pytest.mark.asyncio
async def test_sub_agent_error_status_marks_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(status="GENERATING", message="Starting general-agent task"))
        app._render_event(_sub_agent_event(status="ERROR", message="Error in general-agent: boom"))

        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.status == "failed"
        assert "failed after" in activity.entry.content


@pytest.mark.asyncio
async def test_activity_strip_running_sub_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._turn_active = True
        app._render_event(_sub_agent_event(agent_id="a1", status="GENERATING", message="Starting"))
        assert app._status_state.activity == "Running sub-agent general-agent #1…"

        app._render_event(
            _sub_agent_event(agent_id="a2", parent_tool_call_id="tc-2", status="GENERATING", message="Starting")
        )
        assert app._status_state.activity == "Running 2 sub-agents…"
        assert app._status_state.turn_state == "Running sub-agents"

        app._render_event(_sub_agent_event(agent_id="a1", status="STOPPED", message="Completed"))
        app._render_event(
            _sub_agent_event(agent_id="a2", parent_tool_call_id="tc-2", status="STOPPED", message="Completed")
        )
        assert app._status_state.activity == "Working…"


@pytest.mark.asyncio
async def test_main_agent_tool_events_unaffected_by_sub_agent_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind == "tool_call"]
        assert len(tool_entries) == 1
        assert not _sub_agent_entries(app)


@pytest.mark.asyncio
async def test_sub_agent_event_without_agent_id_uses_fallback_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        event1 = _sub_agent_event(uuid="u1", text="part one ")
        event2 = _sub_agent_event(uuid="u1", text="part two")
        for event in (event1, event2):
            assert event.sub_agent_info is not None
            del event.sub_agent_info["agent_id"]
            app._render_event(event)

        entries = _sub_agent_entries(app)
        assert len(entries) == 1
        assert "part one part two" in entries[0].content
        assert "tc-1" in app._sub_agent_activities


@pytest.mark.asyncio
async def test_sub_agent_tool_streaming_update_routes_to_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        event = _sub_agent_event(text="ignored")
        streaming = AgentEvent(
            event_type="tool_streaming_update",
            sender="general-agent",
            content={"text": "partial", "tool_call_id": "t1", "tool_name": "run_command_tracked", "is_complete": False},
            sub_agent_info=event.sub_agent_info,
        )
        app._render_event(streaming)

        assert not any(entry.kind.startswith("tool") for entry in app.conversation_entries)
        entries = _sub_agent_entries(app)
        assert len(entries) == 1
        assert "run_command_tracked streaming" in entries[0].content


@pytest.mark.asyncio
async def test_cancel_finalizes_running_sub_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(status="GENERATING", message="Starting"))
        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.status == "running"

        app._finalize_sub_agent_activities()

        assert activity.status == "stopped"
        assert activity.entry.complete is True
        assert "stopped after" in activity.entry.content


@pytest.mark.asyncio
async def test_thread_reset_clears_sub_agent_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(uuid="u1", text="some output"))
        assert app._sub_agent_activities

        await app._reset_current_thread()

        assert app._sub_agent_activities == {}
        assert app._sub_agent_by_tool_call == {}
        assert not _sub_agent_entries(app)


@pytest.mark.asyncio
async def test_sub_agent_steps_capture_full_trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        # A tool call + its result pair into one step keyed by tool_call_id.
        app._render_event(
            _sub_agent_event(
                message_type="tool_call", text="Reading app.py", tool_description="read_file", tool_call_id="t1"
            )
        )
        app._render_event(
            _sub_agent_event(
                message_type="tool_result", text="file contents", tool_description="read_file", tool_call_id="t1"
            )
        )
        # Thinking + streamed response accumulate into steps by uuid.
        app._render_event(_sub_agent_event(message_type="thinking", uuid="th1", text="planning the edit"))
        app._render_event(_sub_agent_event(uuid="r1", text="Here is "))
        app._render_event(_sub_agent_event(uuid="r1", text="the answer"))

        activity = next(iter(app._sub_agent_activities.values()))
        # Seeded task step + 1 paired tool step + 1 thinking + 1 response = 4 steps.
        assert len(activity.steps) == 4
        assert activity.steps[0].kind == "sub_agent_task"
        assert activity.steps[0].content == "inspect sessions"
        tool_step = activity.steps[1]
        assert tool_step.kind == "tool_result"
        assert tool_step.tool_call_id == "t1"
        assert tool_step.full_content == "file contents"
        assert activity.steps[2].kind == "thinking"
        assert activity.steps[2].content == "planning the edit"
        assert activity.steps[3].kind == "assistant"
        assert activity.steps[3].content == "Here is the answer"
        # The card advertises the inspect affordance once steps exist.
        from kolega_code.cli import messages as cli_messages

        assert cli_messages.SUB_AGENT_INSPECT_HINT in activity.entry.content


@pytest.mark.asyncio
async def test_sub_agent_completion_event_records_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(status="GENERATING", message="Starting"))
        app._render_event(_sub_agent_event(status="STOPPED", message="Completed", total_tokens=3100))

        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.tokens == 3100
        assert "3.1k tok" in activity.entry.content


@pytest.mark.asyncio
async def test_open_sub_agent_inspector_renders_trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.cli.tui.sub_agent_screen import SubAgentInspectorScreen

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(
            _sub_agent_event(message_type="tool_call", text="Reading", tool_description="read_file", tool_call_id="t1")
        )
        app._render_event(_sub_agent_event(uuid="r1", text="some response"))

        app.action_open_sub_agent()
        await pilot.pause()

        assert isinstance(app._sub_agent_inspector, SubAgentInspectorScreen)
        screen = app._sub_agent_inspector
        # One roster row per agent, and a trajectory widget per captured step.
        assert len(screen._rows) == 1
        # Seeded task step + tool_call + assistant response.
        assert len(screen._step_widgets) == 3
        # The mounted widgets wrap the real step entries (not empty placeholders).
        kinds = {w.entry.kind for w in screen._step_widgets.values()}
        contents = {w.entry.content for w in screen._step_widgets.values()}
        assert kinds == {"sub_agent_task", "tool_call", "assistant"}
        assert "some response" in contents
        assert "inspect sessions" in contents


@pytest.mark.asyncio
async def test_sub_agent_inspector_close_refocuses_idle_composer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(uuid="r1", text="some response"))
        app.action_open_sub_agent()
        await pilot.pause()

        screen = app._sub_agent_inspector
        assert screen is not None
        screen.action_close()
        composer = app.query_one("#composer", ChatComposer)
        for _ in range(5):
            await pilot.pause()
            if app.focused is composer:
                break

        assert composer.disabled is False
        assert app.focused is composer


@pytest.mark.asyncio
async def test_sub_agent_inspector_switches_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(agent_id="a1", task="one", uuid="u1", text="alpha"))
        app._render_event(
            _sub_agent_event(agent_id="a2", task="two", parent_tool_call_id="tc-2", uuid="u2", text="beta")
        )

        app.action_open_sub_agent("a1")
        await pilot.pause()
        screen = app._sub_agent_inspector
        assert screen is not None
        assert screen._selected_key == "a1"

        screen.action_next_agent()
        await pilot.pause()
        assert screen._selected_key == "a2"


@pytest.mark.asyncio
async def test_sub_agent_inspector_escape_closes_without_cancelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        cancelled = False

        def fake_cancel() -> None:
            nonlocal cancelled
            cancelled = True

        monkeypatch.setattr(app, "action_cancel_generation", fake_cancel)

        app._render_event(_sub_agent_event(uuid="u1", text="output"))
        app.action_open_sub_agent()
        await pilot.pause()
        assert app._sub_agent_inspector is not None

        await pilot.press("escape")
        await pilot.pause()

        assert app._sub_agent_inspector is None
        assert cancelled is False


@pytest.mark.asyncio
async def test_open_sub_agent_inspector_empty_notifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        notes: list[str] = []
        monkeypatch.setattr(app, "_notify_user", lambda message, **kw: notes.append(message))

        app.action_open_sub_agent()

        assert app._sub_agent_inspector is None
        from kolega_code.cli import messages as cli_messages

        assert cli_messages.SUB_AGENT_INSPECTOR_EMPTY in notes


@pytest.mark.asyncio
async def test_sub_agent_tool_error_step_captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            _sub_agent_event(
                message_type="tool_call", text="Running", tool_description="run_command", tool_call_id="t1"
            )
        )
        app._render_event(
            _sub_agent_event(
                message_type="tool_error", text="boom: exit 1", tool_description="run_command", tool_call_id="t1"
            )
        )

        activity = next(iter(app._sub_agent_activities.values()))
        assert len(activity.steps) == 2  # seeded task step + the paired tool_error
        step = activity.steps[1]
        assert step.kind == "tool_error"
        assert step.complete is True
        assert "boom" in step.full_content
        assert "last: run_command failed" in activity.entry.content


@pytest.mark.asyncio
async def test_sub_agent_tool_steps_without_id_do_not_collide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Defensive: two separate executions of the same tool, neither carrying a tool_call_id,
    # must produce two distinct paired steps rather than overwriting one shared step.
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(message_type="tool_call", text="grep a", tool_description="grep"))
        app._render_event(_sub_agent_event(message_type="tool_result", text="result a", tool_description="grep"))
        app._render_event(_sub_agent_event(message_type="tool_call", text="grep b", tool_description="grep"))
        app._render_event(_sub_agent_event(message_type="tool_result", text="result b", tool_description="grep"))

        activity = next(iter(app._sub_agent_activities.values()))
        assert len(activity.steps) == 3  # seeded task step + two distinct paired tool steps
        assert [s.kind for s in activity.steps] == ["sub_agent_task", "tool_result", "tool_result"]
        assert activity.steps[1].full_content == "result a"
        assert activity.steps[2].full_content == "result b"


@pytest.mark.asyncio
async def test_sub_agent_stream_without_uuid_merges_by_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        # Empty uuid on the response chunks: they must merge into one step, not fragment.
        app._render_event(_sub_agent_event(uuid="", text="part one "))
        app._render_event(_sub_agent_event(uuid="", text="part two"))
        # A thinking chunk with an empty uuid must stay a separate step (kind-qualified sentinel).
        app._render_event(_sub_agent_event(uuid="", message_type="thinking", text="a thought"))

        activity = next(iter(app._sub_agent_activities.values()))
        kinds = [s.kind for s in activity.steps]
        assert kinds.count("assistant") == 1
        assert kinds.count("thinking") == 1
        assistant = next(s for s in activity.steps if s.kind == "assistant")
        assert assistant.content == "part one part two"


@pytest.mark.asyncio
async def test_sub_agent_empty_final_response_creates_no_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent flushes an empty complete response after every tool-call round; those
    chunks must not become trajectory steps (each rendered as an orphan glyph line)."""
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            _sub_agent_event(
                message_type="tool_call", text="Reading app.py", tool_description="read_file", tool_call_id="t1"
            )
        )
        app._render_event(
            _sub_agent_event(
                message_type="tool_result", text="file contents", tool_description="read_file", tool_call_id="t1"
            )
        )
        # End-of-round flushes: complete, fresh uuids, no text.
        app._render_event(_sub_agent_event(uuid="r1", text=""))
        app._render_event(_sub_agent_event(uuid="th1", message_type="thinking", text=""))

        activity = next(iter(app._sub_agent_activities.values()))
        kinds = [step.kind for step in activity.steps]
        assert "assistant" not in kinds
        assert "thinking" not in kinds
        assert "r1" not in activity.stream_steps
        assert "th1" not in activity.stream_steps

        # An empty complete flush must still finalize a step that streamed text.
        app._render_event(_sub_agent_event(uuid="r2", text="partial", is_streaming=True))
        app._render_event(_sub_agent_event(uuid="r2", text=""))

        assistants = [step for step in activity.steps if step.kind == "assistant"]
        assert len(assistants) == 1
        assert assistants[0].content == "partial"
        assert assistants[0].complete is True


@pytest.mark.asyncio
async def test_sub_agent_inspector_shows_empty_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import Static

    from kolega_code.cli import messages as cli_messages

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        # A task-less lifecycle event: the agent exists but has captured no trajectory steps
        # (and has no task to seed), so the empty-state placeholder still applies.
        app._render_event(_sub_agent_event(task="", status="GENERATING", message="Starting"))

        app.action_open_sub_agent()
        await pilot.pause()
        screen = app._sub_agent_inspector
        assert screen is not None
        assert screen._empty_shown is True
        assert not screen._step_widgets
        view = screen.query_one("#inspector_trajectory")
        placeholder = view.query(Static).first()
        assert cli_messages.SUB_AGENT_INSPECTOR_NO_STEPS in str(placeholder.render())

        # Once a real step arrives, the placeholder is replaced by step widgets.
        app._render_event(_sub_agent_event(uuid="r1", text="now working"))
        screen._flush()
        await pilot.pause()
        assert screen._empty_shown is False
        assert len(screen._step_widgets) == 1


@pytest.mark.asyncio
async def test_workflow_card_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            _workflow_event(
                "workflow_start",
                name="review-changes",
                description="Review then verify",
                phases=[{"title": "Review", "detail": "scan"}, {"title": "Verify"}],
            )
        )
        cards = [e for e in app.conversation_entries if e.kind == "workflow"]
        assert len(cards) == 1
        card = next(iter(app._workflow_activities.values()))
        assert card.name == "review-changes"
        assert [p.title for p in card.phases] == ["Review", "Verify"]
        assert all(p.state == "pending" for p in card.phases)

        # The rich renderable used by the widget builds and prints without markup errors.
        import io

        from rich.console import Console

        buf = io.StringIO()
        Console(file=buf, width=80).print(app._format_workflow_renderable(card))
        rendered = buf.getvalue()
        assert "review-changes" in rendered
        assert "Review" in rendered and "Verify" in rendered

        # A phase event marks it active; a log lands in the footer.
        app._render_event(_workflow_event("workflow_phase", text="Review"))
        review_phase = card.phase_by_title("Review")
        assert review_phase is not None
        assert review_phase.state == "active"
        app._render_event(_workflow_event("workflow_log", text="grepping"))
        assert card.latest_log == "grepping"

        # Moving to the next phase retires the prior one.
        app._render_event(_workflow_event("workflow_phase", text="Verify"))
        assert review_phase.state == "done"
        verify_phase = card.phase_by_title("Verify")
        assert verify_phase is not None
        assert verify_phase.state == "active"

        # End completes the card and any remaining phases.
        app._render_event(_workflow_event("workflow_end", status="completed"))
        assert card.status == "completed"
        assert all(p.state == "done" for p in card.phases)
        assert card.entry.complete is True


@pytest.mark.asyncio
async def test_workflow_card_counts_sub_agents_by_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_workflow_event("workflow_start", name="wf", description="d", phases=[{"title": "Verify"}]))
        card = next(iter(app._workflow_activities.values()))

        # A workflow sub-agent carrying run_id + phase rolls into the card even though no
        # workflow_phase event was emitted (the agent(phase=...) kwarg path).
        evt = _sub_agent_event(agent_id="wf-a1", task="do it", text="working")
        assert evt.sub_agent_info is not None
        evt.sub_agent_info["workflow_run_id"] = "wf-1"
        evt.sub_agent_info["phase"] = "Verify"
        app._render_event(evt)

        assert card.agent_count == 1
        verify = card.phase_by_title("Verify")
        assert verify is not None
        assert verify.state == "active"
        assert verify.agents_total == 1
        assert verify.agents_done == 0

        # Completion bumps the done count and rolls up tokens.
        done = _sub_agent_event(agent_id="wf-a1", task="do it", status="STOPPED", message="Completed", total_tokens=500)
        assert done.sub_agent_info is not None
        done.sub_agent_info["workflow_run_id"] = "wf-1"
        done.sub_agent_info["phase"] = "Verify"
        app._render_event(done)

        assert verify.agents_done == 1
        assert card.tokens == 500


@pytest.mark.asyncio
async def test_thread_reset_closes_open_inspector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(uuid="u1", text="output"))
        app.action_open_sub_agent()
        await pilot.pause()
        assert app._sub_agent_inspector is not None

        await app._reset_current_thread()
        await pilot.pause()

        assert app._sub_agent_inspector is None


@pytest.mark.asyncio
async def test_sub_agent_inspector_tick_follow_and_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr(app, "copy_to_clipboard", lambda text: copied.append(text))
        monkeypatch.setattr(app, "_notify_user", lambda *a, **k: None)

        # A finished agent (exercises the completed status glyph) + a running one.
        app._render_event(_sub_agent_event(agent_id="done", status="GENERATING", message="Starting"))
        app._render_event(
            _sub_agent_event(
                agent_id="done",
                message_type="tool_call",
                text="Reading",
                tool_description="read_file",
                tool_call_id="t1",
            )
        )
        app._render_event(_sub_agent_event(agent_id="done", status="STOPPED", message="Completed", total_tokens=1500))

        app.action_open_sub_agent("done")
        await pilot.pause()
        screen = app._sub_agent_inspector
        assert screen is not None

        # Spinner/elapsed tick refresh must not raise on running or finished agents.
        screen._on_tick()
        await pilot.pause()

        # Follow toggles.
        assert screen._follow is True
        screen.action_toggle_follow()
        assert screen._follow is False

        # Copy gathers the selected agent's trajectory text.
        screen.action_copy_trajectory()
        assert copied and "read_file" in copied[0]
