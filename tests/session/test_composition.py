from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest

from kolega_code.agent import AgentConfig, PromptExtension, ToolExtension
from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.custom_agents import CustomAgentCatalog, CustomAgentDefinition, CustomAgentDiagnostic
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli import skills as skills_module
from kolega_code.cli.skills import SkillCatalog, SkillRecord
from kolega_code.extensions import (
    ExtensionSelection,
    KolegaExtensionBundle,
    KolegaExtensionLoadError,
)
from kolega_code.llm.models import Message, TextBlock
from kolega_code.mcp.config import LoadedMCPConfig
from kolega_code.permissions import PermissionMode
from kolega_code.session import composition
from kolega_code.session.composition import (
    CompositionHost,
    CompositionRequest,
    HostExtensions,
    construct_agent,
    prepare_composition,
    select_agent_class,
)

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


class _RecordingAgent(BaseAgent):
    agent_name = "composition-test-agent"
    kwargs: dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_select_agent_class_preserves_mode_selection() -> None:
    class PlanAgent(_RecordingAgent):
        pass

    assert (
        select_agent_class("build", build_agent_class=_RecordingAgent, planning_agent_class=PlanAgent)
        is _RecordingAgent
    )
    assert select_agent_class("plan", build_agent_class=_RecordingAgent, planning_agent_class=PlanAgent) is PlanAgent


def _prompt(name: str) -> PromptExtension:
    return PromptExtension(id=name, title=name, markdown=name)


def _tool(name: str) -> ToolExtension:
    return ToolExtension(name=name, tools={})


def _make_request(root: Path) -> CompositionRequest:
    return CompositionRequest(
        config=AgentConfig(skills_enabled=False, anthropic_api_key="fake-composition-test-key"),
        project_path=root / "launch",
        active_project_path=root / "active",
        state_dir=root / "state",
        workspace_id="test-workspace",
        thread_id="test-thread",
        interaction_mode="build",
        agent_mode=AgentMode.ASK,
        agent_class=_RecordingAgent,
        connection_manager=Mock(),
        browser_manager=Mock(),
        memory_manager=Mock(),
        permission_mode=PermissionMode.ASK,
        permission_callback=Mock(),
        session_recorder=Mock(),
        usage_ledger=Mock(),
        extension_selection=None,
    )


def _make_host() -> CompositionHost:
    scratchpad = _prompt("scratchpad")
    return CompositionHost(
        extensions=Mock(
            return_value=HostExtensions(
                [_prompt("host-first"), scratchpad],
                [_tool("host-first")],
                scratchpad=scratchpad,
            )
        ),
        questions=Mock(return_value=(_prompt("question"), _tool("question"))),
        additional_prompts=Mock(return_value=[_prompt("additional-first"), _prompt("additional-last")]),
        history=Mock(return_value=[]),
        set_catalogs=Mock(),
        mcp_trusted=Mock(return_value=False),
        mcp_diagnostic=Mock(),
        mcp_names=Mock(),
        own_bundle=Mock(),
        hook_dispatcher=Mock(return_value=object()),
        mcp_builder=Mock(return_value=None),
    )


@pytest.fixture
def composition_request(tmp_path: Path) -> CompositionRequest:
    return _make_request(tmp_path)


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> CompositionHost:
    monkeypatch.setattr(composition, "discover_custom_agents", Mock(return_value=CustomAgentCatalog()))
    monkeypatch.setattr(composition, "discover_skills", Mock(return_value=SkillCatalog()))
    return _make_host()


@pytest.mark.parametrize("active_is_launch", [False, True], ids=["switched", "launch"])
@pytest.mark.parametrize("trusted", [False, True], ids=["untrusted", "trusted"])
@pytest.mark.parametrize("has_extension", [False, True], ids=["no-tools", "tools"])
def test_prepare_forwards_active_mcp_path_launch_cache_and_host_trust(
    composition_request: CompositionRequest,
    host: CompositionHost,
    active_is_launch: bool,
    trusted: bool,
    has_extension: bool,
) -> None:
    request = composition_request
    if active_is_launch:
        request.active_project_path = Path(str(request.project_path))
    cached = LoadedMCPConfig(diagnostics=["first diagnostic", "second diagnostic"], project_trusted=not trusted)
    request.config.mcp_config = cached
    mcp_extension = _tool("mcp") if has_extension else None
    host.mcp_builder = builder = Mock(return_value=mcp_extension)
    host.mcp_trusted = trust = Mock(return_value=trusted)

    prepared = prepare_composition(request, host)

    builder.assert_called_once_with(
        request.active_project_path,
        request.state_dir,
        project_trusted=trusted,
        loaded_config=cached if active_is_launch else None,
    )
    trust.assert_called_once_with()
    assert cast(Mock, host.mcp_diagnostic).call_args_list == [
        (("first diagnostic",), {}),
        (("second diagnostic",), {}),
    ]
    names = cast(Mock, host.mcp_names)
    if active_is_launch and has_extension:
        names.assert_called_once_with(cached)
    else:
        names.assert_not_called()
    assert [tool.name for tool in prepared.tools] == ["host-first", "question"] + (["mcp"] if has_extension else [])
    if has_extension:
        assert prepared.tools[-1] is mcp_extension


def test_prepare_without_cached_mcp_config_still_calls_host_builder(
    composition_request: CompositionRequest, host: CompositionHost
) -> None:
    prepared = prepare_composition(composition_request, host)

    cast(Mock, host.mcp_builder).assert_called_once_with(
        composition_request.active_project_path,
        composition_request.state_dir,
        project_trusted=False,
        loaded_config=None,
    )
    cast(Mock, host.mcp_diagnostic).assert_not_called()
    cast(Mock, host.mcp_names).assert_not_called()
    assert [tool.name for tool in prepared.tools] == ["host-first", "question"]


def test_disabled_skills_skip_discovery_budget_and_builders(
    composition_request: CompositionRequest, host: CompositionHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbidden = Mock(side_effect=AssertionError("disabled skills must not be consulted"))
    for name in (
        "discover_skills",
        "context_window_tokens_for_skill_budget",
        "build_skill_prompt_extension",
        "build_skill_tool_extension",
    ):
        monkeypatch.setattr(composition, name, forbidden)

    prepared = prepare_composition(composition_request, host)

    forbidden.assert_not_called()
    skills, agents = cast(Mock, host.set_catalogs).call_args.args
    assert isinstance(skills, SkillCatalog)
    assert skills.skills == {}
    assert agents is prepared.custom_agents
    assert [prompt.id for prompt in prepared.prompts] == [
        "host-first",
        "scratchpad",
        "question",
        "additional-first",
        "additional-last",
    ]
    extensions = cast(Mock, host.extensions).return_value
    assert prepared.prompts is extensions.prompts
    assert prepared.tools is extensions.tools
    assert prepared.scratchpad is extensions.scratchpad


@pytest.mark.asyncio
async def test_enabled_skills_use_active_project_agent_budget_and_live_history(
    composition_request: CompositionRequest,
    host: CompositionHost,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = composition_request
    request.config.skills_enabled = True
    skill_dir = request.active_project_path / ".agents" / "skills" / "demo"
    catalog = SkillCatalog(
        skills={"demo": SkillRecord("demo", "Use the demo workflow.", skill_dir, skill_dir / "SKILL.md", "project")}
    )
    discover = Mock(return_value=catalog)
    budget = Mock(return_value=12345)
    prompt_builder = Mock(wraps=composition.build_skill_prompt_extension)
    tool_builder = Mock(wraps=composition.build_skill_tool_extension)
    monkeypatch.setattr(composition, "discover_skills", discover)
    monkeypatch.setattr(composition, "context_window_tokens_for_skill_budget", budget)
    monkeypatch.setattr(composition, "build_skill_prompt_extension", prompt_builder)
    monkeypatch.setattr(composition, "build_skill_tool_extension", tool_builder)
    read_skill = Mock(return_value=({}, "Follow the demo steps."))
    monkeypatch.setattr(skills_module, "_parse_skill_file", read_skill)
    monkeypatch.setattr(SkillCatalog, "resource_paths", Mock(return_value=([], False)))
    host.history = history = Mock(return_value=[])
    host.mcp_builder = Mock(return_value=_tool("mcp"))

    prepared = prepare_composition(request, host)

    discover.assert_called_once_with(request.active_project_path)
    budget.assert_called_once_with(request.config, _RecordingAgent.agent_name)
    prompt_builder.assert_called_once_with(catalog, context_window_tokens=12345)
    tool_builder.assert_called_once_with(catalog, history)
    assert cast(Mock, host.set_catalogs).call_args.args[0] is catalog
    assert [prompt.id for prompt in prepared.prompts] == [
        "host-first",
        "scratchpad",
        "cli-agent-skills",
        "question",
        "additional-first",
        "additional-last",
    ]
    assert [tool.name for tool in prepared.tools] == ["host-first", "cli-agent-skills", "question", "mcp"]
    history.assert_not_called()
    activate = prepared.tools[1].tools["skill"]
    output = await activate(name="demo")
    assert "Follow the demo steps." in output
    history.return_value = [Message(role="user", content=[TextBlock(output)])]
    assert "already active" in await activate(name="demo")
    history.return_value = []
    assert "Follow the demo steps." in await activate(name="demo")
    assert history.call_count == 3
    assert read_skill.call_count == 2


def test_enabled_but_empty_skills_add_no_extensions(
    composition_request: CompositionRequest, host: CompositionHost
) -> None:
    composition_request.config.skills_enabled = True

    prepared = prepare_composition(composition_request, host)

    assert [tool.name for tool in prepared.tools] == ["host-first", "question"]
    assert "cli-agent-skills" not in [prompt.id for prompt in prepared.prompts]


@pytest.mark.parametrize("mode", ["build", "plan"])
def test_custom_catalog_is_validated_before_mode_filtering_and_shared_with_agent(
    composition_request: CompositionRequest,
    host: CompositionHost,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    request = composition_request
    request.interaction_mode = mode
    definitions = {
        name: CustomAgentDefinition(
            name=name,
            description=name,
            prompt=name,
            source_path=request.active_project_path / f"{name}.md",
            scope="project",
            mode=agent_mode,
        )
        for name, agent_mode in (("builder", "build"), ("planner", "plan"), ("shared", "all"), ("invalid", "all"))
    }
    diagnostic = CustomAgentDiagnostic("warning", "discovery warning", request.active_project_path)
    catalog = CustomAgentCatalog(agents=definitions, diagnostics=[diagnostic])
    discover = Mock(return_value=catalog)
    resolved: list[str] = []

    def resolve(definition: CustomAgentDefinition, config: AgentConfig) -> Any:
        assert config is request.config
        resolved.append(definition.name)
        if definition.name == "invalid":
            raise ValueError("invalid inherited model")
        return config.long_context_config

    monkeypatch.setattr(composition, "discover_custom_agents", discover)
    monkeypatch.setattr(CustomAgentDefinition, "resolve_model_config", resolve)

    prepared = prepare_composition(request, host)
    agent, _ = construct_agent(request, host, prepared)

    discover.assert_called_once_with(request.active_project_path, request.state_dir)
    assert resolved == list(definitions)
    assert prepared.custom_agents.names() == (["builder", "shared"] if mode == "build" else ["planner", "shared"])
    assert prepared.custom_agents.get("shared") is definitions["shared"]
    assert prepared.custom_agents.diagnostics[0] is diagnostic
    assert "invalid inherited model" in prepared.custom_agents.diagnostics[1].message
    assert catalog.names() == list(definitions)
    assert catalog.diagnostics == [diagnostic]
    assert cast(Mock, host.set_catalogs).call_args.args[1] is prepared.custom_agents
    assert isinstance(agent, _RecordingAgent)
    assert agent.kwargs["custom_agent_catalog"] is prepared.custom_agents


@pytest.mark.parametrize("with_bundle", [False, True], ids=["plain", "extension"])
def test_construct_preserves_dependencies_extension_order_and_host_ownership(
    composition_request: CompositionRequest, host: CompositionHost, with_bundle: bool
) -> None:
    request = composition_request
    contributed = KolegaExtensionBundle(
        prompt_extensions=[_prompt("selected-first"), _prompt("selected-last")],
        tool_extensions=[_tool("selected-first"), _tool("selected-last")],
        llm_trace_sink=Mock(),
        bind_agent=Mock(),
        cleanup=Mock(),
    )
    factory = Mock(return_value=contributed)
    if with_bundle:
        request.extension_selection = ExtensionSelection("test:factory", factory, request.state_dir / "opaque.json")
    prepared = prepare_composition(request, host)
    prompt_prefix, tool_prefix = list(prepared.prompts), list(prepared.tools)
    owned: list[KolegaExtensionBundle] = []
    host.own_bundle = Mock(side_effect=owned.append)

    class CheckingAgent(_RecordingAgent):
        def __init__(self, **kwargs: Any) -> None:
            assert bool(owned) is with_bundle
            super().__init__(**kwargs)

    request.agent_class = CheckingAgent

    agent, bundle = construct_agent(request, host, prepared)

    assert isinstance(agent, CheckingAgent)
    expected = {
        "project_path": request.active_project_path,
        "workspace_id": request.workspace_id,
        "thread_id": request.thread_id,
        "connection_manager": request.connection_manager,
        "config": request.config,
        "browser_manager": request.browser_manager,
        "agent_mode": request.agent_mode,
        "memory_manager": request.memory_manager,
        "permission_mode": request.permission_mode,
        "permission_callback": request.permission_callback,
        "session_recorder": request.session_recorder,
        "hook_dispatcher": cast(Mock, host.hook_dispatcher).return_value,
        "custom_agent_catalog": prepared.custom_agents,
        "usage_ledger": request.usage_ledger,
        "llm_trace_sink": contributed.llm_trace_sink if with_bundle else None,
    }
    assert agent.kwargs.keys() == expected.keys() | {"prompt_extensions", "tool_extensions"}
    for name, value in expected.items():
        assert agent.kwargs[name] is value, name
    cast(Mock, host.hook_dispatcher).assert_called_once_with()
    if with_bundle:
        assert bundle is not None
        assert owned == [bundle]
        cast(Mock, host.own_bundle).assert_called_once_with(bundle)
        factory.assert_called_once()
        extension_host, config_path = factory.call_args.args
        assert extension_host.project_path is request.active_project_path
        assert extension_host.workspace_id == request.workspace_id
        assert extension_host.thread_id == request.thread_id
        assert extension_host.config is request.config
        assert extension_host.agent_mode is request.agent_mode
        assert request.extension_selection is not None
        assert config_path is request.extension_selection.config_path
    else:
        assert bundle is None
        cast(Mock, host.own_bundle).assert_not_called()
        factory.assert_not_called()
    expected_prompts = prompt_prefix + (contributed.prompt_extensions if with_bundle else [])
    expected_tools = tool_prefix + (contributed.tool_extensions if with_bundle else [])
    for actual, expected_extension in zip(agent.kwargs["prompt_extensions"], expected_prompts, strict=True):
        assert actual is expected_extension
    for actual, expected_extension in zip(agent.kwargs["tool_extensions"], expected_tools, strict=True):
        assert actual is expected_extension
    assert agent.kwargs["prompt_extensions"] is not prepared.prompts
    assert agent.kwargs["tool_extensions"] is not prepared.tools
    assert prepared.prompts == prompt_prefix
    assert prepared.tools == tool_prefix
    cast(Mock, contributed.bind_agent).assert_not_called()
    cast(Mock, contributed.cleanup).assert_not_called()


@pytest.mark.parametrize("with_bundle", [False, True], ids=["plain", "extension"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_constructor_failure_retains_host_bundle_ownership_and_only_wraps_extension_value_errors(
    composition_request: CompositionRequest,
    host: CompositionHost,
    with_bundle: bool,
    error_type: type[Exception],
) -> None:
    request = composition_request
    contributed = KolegaExtensionBundle(
        prompt_extensions=[_prompt("selected")],
        tool_extensions=[_tool("selected")],
        cleanup=Mock(),
    )
    if with_bundle:
        request.extension_selection = ExtensionSelection("test:factory", Mock(return_value=contributed), None)
    original = error_type("constructor failed")
    owned: list[KolegaExtensionBundle] = []
    host.own_bundle = Mock(side_effect=owned.append)

    class FailingAgent(_RecordingAgent):
        def __init__(self, **kwargs: Any) -> None:
            assert bool(owned) is with_bundle
            if with_bundle:
                assert kwargs["prompt_extensions"][-1] is contributed.prompt_extensions[0]
                assert kwargs["tool_extensions"][-1] is contributed.tool_extensions[0]
            raise original

    request.agent_class = FailingAgent
    prepared = prepare_composition(request, host)
    wrapped = with_bundle and error_type is ValueError
    expected_error = KolegaExtensionLoadError if wrapped else error_type

    with pytest.raises(expected_error, match="constructor failed") as caught:
        construct_agent(request, host, prepared)

    if wrapped:
        assert caught.value.__cause__ is original
    else:
        assert caught.value is original
    assert len(owned) == int(with_bundle)
    cast(Mock, contributed.cleanup).assert_not_called()
    if with_bundle:
        assert owned[0].cleanup is contributed.cleanup
        cast(Mock, host.own_bundle).assert_called_once_with(owned[0])
    else:
        cast(Mock, host.own_bundle).assert_not_called()


def test_prepare_and_construct_in_fresh_process_with_textual_and_rich_blocked(tmp_path: Path) -> None:
    program = textwrap.dedent(
        """
        import runpy
        import sys
        from pathlib import Path

        class RejectUI:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"textual", "rich"}:
                    raise ImportError(f"{fullname} is blocked for this test")
                return None

        assert not any(name.split(".")[0] in {"textual", "rich"} for name in sys.modules)
        sys.meta_path.insert(0, RejectUI())
        helpers = runpy.run_path(sys.argv[1])
        composition = helpers["composition"]
        Mock = helpers["Mock"]
        with helpers["pytest"].MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(composition, "discover_custom_agents",
                                Mock(return_value=helpers["CustomAgentCatalog"]()))
            monkeypatch.setattr(composition, "discover_skills",
                                Mock(return_value=helpers["SkillCatalog"]()))
            request = helpers["_make_request"](Path(sys.argv[2]))
            request.config.skills_enabled = True
            request.extension_selection = helpers["ExtensionSelection"](
                "test:factory", Mock(return_value=helpers["KolegaExtensionBundle"]()), None
            )
            host = helpers["_make_host"]()
            prepared = composition.prepare_composition(request, host)
            agent, bundle = composition.construct_agent(request, host, prepared)
            assert isinstance(agent, helpers["_RecordingAgent"])
            assert agent.kwargs["config"] is request.config
            assert agent.kwargs["prompt_extensions"] == prepared.prompts
            assert agent.kwargs["prompt_extensions"] is not prepared.prompts
            assert bundle is not None
            host.own_bundle.assert_called_once_with(bundle)
        assert not any(name.split(".")[0] in {"textual", "rich"} for name in sys.modules)
        print("HEADLESS COMPOSITION OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HEADLESS COMPOSITION OK" in result.stdout
