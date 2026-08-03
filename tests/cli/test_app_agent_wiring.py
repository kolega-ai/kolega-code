# ruff: noqa: F401,F811,E402
from pathlib import Path
from typing import Any, cast
import asyncio
import json
import time
from unittest.mock import Mock

import pytest

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
from kolega_code.tools import tool_definition_from_callable

from ._app_test_utils import (
    FakeCoderAgent,
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    install_fake_agents,
    question_payload,
    renderable_text,
)


class FakePlanningAgent(FakeCoderAgent):
    pass


@pytest.mark.asyncio
async def test_textual_app_builds_the_configured_browser_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=FakePlanningAgent)
    browser_manager = Mock()
    build_browser_manager = Mock(return_value=browser_manager)
    monkeypatch.setattr(
        "kolega_code.cli.tui.agent_runtime.build_browser_manager",
        build_browser_manager,
    )

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(
        project_path=project,
        config=config,
        mode="code",
        store=store,
        session=session,
        browser_visible=True,
    )

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["browser_manager"] is browser_manager
        build_browser_manager.assert_called_once_with(
            store.root,
            session.session_id,
            browser_visible=True,
        )


@pytest.mark.asyncio
async def test_textual_app_passes_shared_task_list_tools_to_build_agent_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import PlanningMarkdown

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        assert isinstance(app.agent, FakeCoderAgent)
        task_list_extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list")
        build_tools = task_list_extension.tools
        assert {"get_task_list", "update_task_list"} == set(build_tools)
        # The task list is single-owner; it must not be inherited by sub-agents.
        assert task_list_extension.propagate_to_sub_agents is False
        # Build mode gets the question tool too, under its own framing.
        build_question_extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-planning-questions")
        assert set(build_question_extension.tools) == {"ask_user_choice"}
        assert build_question_extension.propagate_to_sub_agents is False
        assert "ask_user_choice" in build_question_extension.tool_groups["coder_agent_tools"]
        build_prompt_ids = {getattr(ext, "id", None) for ext in app.agent.kwargs["prompt_extensions"]}
        assert "cli-build-questions" in build_prompt_ids
        assert "cli-planning-questions" not in build_prompt_ids
        build_task_list_prompt = app.agent.kwargs["prompt_extensions"][0].markdown
        assert "After each meaningful task is completed" in build_task_list_prompt
        assert "Do not wait until every TODO is complete" in build_task_list_prompt
        update_task_list_doc = build_tools["update_task_list"].__doc__ or ""
        assert "progress is visible incrementally" in update_task_list_doc
        assert "do not wait" in update_task_list_doc.lower()

        assert await build_tools["get_task_list"]() == "No task list has been set."
        assert await build_tools["update_task_list"]("- [ ] inspect\n- [x] plan") == "Task list updated."
        assert app.session.task_list_markdown == "- [ ] inspect\n- [x] plan"
        assert app.query_one("#status_task_list_markdown", PlanningMarkdown).source == "- [ ] inspect\n- [x] plan"
        assert store.load(session.session_id).task_list_markdown == "- [ ] inspect\n- [x] plan"

        goal_extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-goal-control")
        goal_prompt = extension_by_name(app.agent.kwargs["prompt_extensions"], "cli-goal-control")
        assert set(goal_extension.tools) == {"set_goal"}
        assert goal_extension.propagate_to_sub_agents is False
        assert goal_prompt.propagate_to_sub_agents is False
        goal_doc = goal_extension.tools["set_goal"].__doc__ or ""
        goal_policy = goal_prompt.markdown
        for policy_text in (goal_doc, goal_policy):
            policy_lower = " ".join(policy_text.lower().split())
            assert "explicit governing instruction" in policy_lower
            assert "activated agent skill" in policy_lower
            assert "host-provided workflow" in policy_lower
            assert "do not infer goal mode" in policy_lower
            assert "repository contents" in policy_lower
            assert "not by itself authorization" in policy_lower
        goal_definition = tool_definition_from_callable("set_goal", goal_extension.tools["set_goal"])
        goal_schema = goal_definition.to_anthropic()["input_schema"]
        assert goal_schema["required"] == ["condition"]
        assert goal_schema["properties"]["condition"]["type"] == "string"

        # The worktree switch carries the same shape of authorization gate: the
        # model sees it in the tool description as well as the prompt section.
        switch_extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control")
        switch_prompt = extension_by_name(app.agent.kwargs["prompt_extensions"], "cli-worktree-control")
        switch_doc = switch_extension.tools["switch_worktree"].__doc__ or ""
        for policy_text in (switch_doc, switch_prompt.markdown):
            policy_lower = " ".join(policy_text.lower().split())
            assert "explicitly asked" in policy_lower or "explicitly asks" in policy_lower
            assert "does not create a worktree" in policy_lower or "never creates one" in policy_lower
            assert "may decline" in policy_lower

        await pilot.press("shift+tab")

        assert isinstance(app.agent, FakePlanningAgent)
        plan_extension_names = {getattr(ext, "name", None) for ext in app.agent.kwargs["tool_extensions"]}
        # Plan mode never gets the writable task list (build-mode only)...
        assert "cli-shared-task-list" not in plan_extension_names
        # ...but it does get a read-only view, so a re-plan started mid-build can
        # see what has already been completed.
        assert "cli-shared-task-list-readonly" in plan_extension_names
        plan_task_list_extension = extension_by_name(
            app.agent.kwargs["tool_extensions"], "cli-shared-task-list-readonly"
        )
        assert set(plan_task_list_extension.tools) == {"get_task_list"}
        assert plan_task_list_extension.propagate_to_sub_agents is False
        # PlanningAgent only exposes extension tools declared in its custom_tool_groups.
        assert plan_task_list_extension.tool_groups["planning_tools"] == ["get_task_list"]
        # The read-only getter sees the list build mode wrote a moment ago.
        assert await plan_task_list_extension.tools["get_task_list"]() == "- [ ] inspect\n- [x] plan"
        plan_prompt_ids = {getattr(ext, "id", None) for ext in app.agent.kwargs["prompt_extensions"]}
        assert "cli-shared-task-list-readonly" in plan_prompt_ids
        assert "cli-shared-task-list" not in plan_prompt_ids
        # ...and still gets the planning-question tool.
        assert "cli-planning-questions" in plan_extension_names
        # The goal and worktree control extensions are installed in both modes,
        # but neither declares planning_tools — the group that bypasses the
        # planning agent's read_only filter — so plan mode's registry drops both.
        assert "cli-goal-control" in plan_extension_names
        plan_goal_extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-goal-control")
        assert set(plan_goal_extension.tools) == {"set_goal"}
        assert plan_goal_extension.tool_groups == {"cli_goal_tools": ["set_goal"]}
        plan_switch_extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control")
        assert plan_switch_extension.tool_groups == {"cli_worktree_tools": ["switch_worktree"]}
        # Mode-aware prompt sections name the build-only handoffs while making it
        # explicit that the planning agent cannot call either tool.
        assert "cli-goal-control" in plan_prompt_ids
        plan_goal_prompt = extension_by_name(app.agent.kwargs["prompt_extensions"], "cli-goal-control")
        assert "Planning mode cannot call `set_goal`" in plan_goal_prompt.markdown
        assert "prerequisites may come first" in plan_goal_prompt.markdown
        plan_switch_prompt = extension_by_name(app.agent.kwargs["prompt_extensions"], "cli-worktree-control")
        assert "never run `git worktree add` here" in plan_switch_prompt.markdown
        assert "cannot call `switch_worktree` in planning mode" in plan_switch_prompt.markdown
        assert "make `switch_worktree` the next model response" in plan_switch_prompt.markdown
        assert "only after the continuation starts in the switched workspace" in plan_switch_prompt.markdown
        question_tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-planning-questions").tools
        assert {"ask_user_choice"} == set(question_tools)
        prompt_markdown = "\n".join(extension.markdown for extension in app.agent.kwargs["prompt_extensions"])
        assert "multiple-choice" in prompt_markdown
        # The task list captured in build mode persists and is untouched by plan mode.
        assert app.session.task_list_markdown == "- [ ] inspect\n- [x] plan"


@pytest.mark.asyncio
async def test_textual_app_passes_skill_extensions_to_build_and_plan_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=FakePlanningAgent)

    project = tmp_path / "project"
    skill_dir = project / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        assert isinstance(app.agent, FakeCoderAgent)
        skill_prompt = extension_by_name(app.agent.kwargs["prompt_extensions"], "cli-agent-skills")
        skill_tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-agent-skills").tools

        assert "demo-skill" in skill_prompt.markdown
        assert {"list_skills", "activate_skill", "read_skill_resource"} == set(skill_tools)
        assert "demo-skill" in await skill_tools["list_skills"]()

        await pilot.press("shift+tab")

        assert isinstance(app.agent, FakePlanningAgent)
        planning_skill_tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-agent-skills")
        assert "activate_skill" in planning_skill_tools.tools


@pytest.mark.asyncio
async def test_agent_rebuild_keeps_the_same_usage_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model/settings rebuilds construct a new agent but must keep accounting on
    the app's one process-wide ledger (same object, same run_id)."""
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.config import config_summary
    from kolega_code.cli.session_store import SessionStore

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await pilot.pause()
        first_agent = cast(Any, app.agent)
        assert first_agent is not None
        ledger = first_agent.kwargs["usage_ledger"]
        assert ledger is app._usage_ledger
        run_id = ledger.run_id

        await app._build_agent(config, rebuild=True)

        assert app.agent is not first_agent
        assert cast(Any, app.agent).kwargs["usage_ledger"] is ledger
        assert ledger.run_id == run_id


@pytest.mark.asyncio
async def test_usage_sink_attached_on_mount_and_drained_on_quit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_mount writes the accounting marker and wires the ledger observer;
    quit drains the sink; the snapshot helper carries the derived usage field."""
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.config import config_summary
    from kolega_code.cli.session_store import SessionStore
    from kolega_code.cli.session_usage import LLM_MESSAGE_EVENT, LLM_RUN_STARTED_EVENT
    from kolega_code.llm.ledger import LlmCallOrigin, llm_call_origin
    from kolega_code.llm.models import Message
    from kolega_code.llm.usage import normalize_usage

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._usage_ledger.observer is app._usage_sink

        ledger = app._usage_ledger
        usage = normalize_usage({"input_tokens": 10, "output_tokens": 5}, "anthropic", "m")
        with llm_call_origin(LlmCallOrigin(kind="sub_agent", agent_name="Investigator", agent_id="a1")):
            request_id = ledger.begin("anthropic", "m")
        ledger.record_response(request_id, usage, message=Message(role="assistant", content="s", usage=usage))

        # Snapshot helper carries the derived field.
        app.session.usage = {"total_tokens": 15}
        async with app._persistence_lock:
            snapshot = app._session_snapshot_locked()
        assert snapshot.usage == {"total_tokens": 15}

        await app.action_quit()

    events = store.journal(session.session_id).read_events()
    types = [e.event_type for e in events]
    assert types.count(LLM_RUN_STARTED_EVENT) == 1
    assert types.count(LLM_MESSAGE_EVENT) == 1
