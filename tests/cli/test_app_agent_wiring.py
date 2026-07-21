# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

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
        assert all("ask_user_choice" not in extension.tools for extension in app.agent.kwargs["tool_extensions"])
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

        await pilot.press("shift+tab")

        assert isinstance(app.agent, FakePlanningAgent)
        plan_extension_names = {getattr(ext, "name", None) for ext in app.agent.kwargs["tool_extensions"]}
        # Plan mode no longer gets the shared task list (build-mode only)...
        assert "cli-shared-task-list" not in plan_extension_names
        # ...but still gets the planning-question tool.
        assert "cli-planning-questions" in plan_extension_names
        assert "cli-goal-control" in plan_extension_names
        plan_goal_extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-goal-control")
        assert set(plan_goal_extension.tools) == {"set_goal"}
        assert "set_goal" in plan_goal_extension.tool_groups["planning_tools"]
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
