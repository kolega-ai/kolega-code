"""Deterministic wire dump of every model-facing tool definition.

The dump captures exactly what a provider serialization reads from each
``ToolDefinition``: the name, the wire description, the resolved input-schema
object, the input kind, and any freeform format. Anything not captured here
(groups, parallel safety, permission gating) is runtime behavior, not wire
bytes, and is covered by other tests.

Scenarios pin every runtime-dependent input (custom-agent catalog, skill
catalog, browser targets, edit protocol) so the dump is byte-identical across
machines and runs. ``test_tool_definition_freeze.py`` asserts the rendered
dump matches the committed snapshot fixture; regenerate deliberately with::

    KOLEGA_UPDATE_TOOL_DEFINITION_SNAPSHOT=1 uv run pytest tests/agent/test_tool_definition_freeze.py

Dicts are rendered without key sorting: JSON object order is dict insertion
order, which is also the order provider payloads serialize, so an unchanged
snapshot certifies unchanged wire bytes, not merely equivalent schemas. The
``tool_order`` list per scenario freezes registry order for the same reason —
prompt caching matches on the byte prefix, so reordering definitions is a wire
change even when each definition is intact.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional
from unittest.mock import AsyncMock, Mock

from kolega_code.agent.browseragent import BrowserAgent
from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.custom_agents import CustomAgent, CustomAgentCatalog, CustomAgentDefinition
from kolega_code.agent.generalagent import GeneralAgent
from kolega_code.agent.investigationagent import InvestigationAgent
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode, PromptProvider
from kolega_code.agent.tools import ToolExtension
from kolega_code.cli.skills import build_skill_tool_extension, discover_skills
from kolega_code.cli.tui.agent_runtime import AgentRuntimeMixin
from kolega_code.cli.tui.prompt_flows import PromptFlowMixin
from kolega_code.config import AgentConfig, EditProtocol, ModelConfig, ModelProvider
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import ToolDefinition

SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "tool_definitions" / "wire_definitions.json"

_PINNED_MODEL = "claude-sonnet-4-5-20250929"


def wire_definition(definition: ToolDefinition) -> dict[str, Any]:
    """Everything provider serializations (to_anthropic/to_openai/to_google) read."""
    return {
        "description": definition.description,
        "input_schema": definition._object_schema(),
        "input_kind": definition.input_kind,
        "freeform_format": definition.freeform_format,
    }


def dump_definitions(definitions: Iterable[ToolDefinition], only: Optional[set[str]] = None) -> dict[str, Any]:
    selected = [d for d in definitions if only is None or d.name in only]
    return {
        "tool_order": [d.name for d in selected],
        "tools": {d.name: wire_definition(d) for d in sorted(selected, key=lambda d: d.name)},
    }


def _connection_manager() -> Mock:
    manager = Mock(spec=AgentConnectionManager)
    manager.workspace_id = "freeze-workspace"
    manager.send_message = AsyncMock()
    return manager


def _config(edit_protocol: EditProtocol = EditProtocol.SEARCH_REPLACE) -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="freeze-key",
        long_context_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model=_PINNED_MODEL),
        edit_protocol=edit_protocol,
    )


def _custom_agent_catalog() -> CustomAgentCatalog:
    """Pinned catalog: fixed names/descriptions, no filesystem or model lookups."""
    definitions = {
        "reviewer": CustomAgentDefinition(
            name="reviewer",
            description="Reviews code for correctness",
            prompt="Review.",
            source_path=Path("/kolega-freeze/agents/reviewer.md"),
            scope="project",
        ),
        "docs-writer": CustomAgentDefinition(
            name="docs-writer",
            description="Writes and updates documentation",
            prompt="Write docs.",
            source_path=Path("/kolega-freeze/agents/docs-writer.md"),
            scope="user",
        ),
    }
    return CustomAgentCatalog(agents=definitions)


def _agent_kwargs(project_path: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "project_path": project_path,
        "workspace_id": "freeze-workspace",
        "thread_id": str(uuid.uuid4()),
        "connection_manager": _connection_manager(),
        "config": _config(),
    }
    kwargs.update(overrides)
    return kwargs


def _hosted_prompt_provider(project_path: Path) -> PromptProvider:
    template_dir = project_path / "hosted_prompt_templates"
    agents_dir = template_dir / "system" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "coder_code.md.j2").write_text("Hosted freeze prompt.", encoding="utf-8")
    return PromptProvider(template_dirs=[template_dir])


class _StubTuiApp:
    """Build-time stand-in for the TUI app the extension factories close over.

    The factory methods only reference ``self`` inside the tool callbacks, which
    the dump never invokes, so an attribute-free stub is enough to build the
    extensions exactly as the TUI does.
    """


def _cli_build_tool_extensions() -> list[ToolExtension]:
    stub: Any = _StubTuiApp()
    return [
        PromptFlowMixin._shared_task_list_tool_extension(stub),
        PromptFlowMixin._planning_question_tool_extension(stub),
        AgentRuntimeMixin._worktree_control_tool_extension(stub),
        AgentRuntimeMixin._goal_control_tool_extension(stub),
    ]


def _cli_plan_tool_extensions() -> list[ToolExtension]:
    stub: Any = _StubTuiApp()
    return [
        PromptFlowMixin._shared_task_list_readonly_tool_extension(stub),
        PromptFlowMixin._planning_question_tool_extension(stub),
    ]


def _skill_tool_extension(project_path: Path) -> ToolExtension:
    skills_dir = project_path / ".agents" / "skills" / "demo"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill for the freeze dump\n---\n\nDemo body.\n",
        encoding="utf-8",
    )
    catalog = discover_skills(
        project_path,
        user_home=project_path / "no-user-skills",
        bundled_root=None,
    )
    extension = build_skill_tool_extension(catalog, lambda: [])
    assert extension is not None
    return extension


def _mcp_style_extension() -> ToolExtension:
    """A pinned extension shaped exactly like build_mcp_tool_extension's output.

    One tool's schema describes every property (the Args-strip rule applies) and
    one leaves a property undescribed (the keep rule applies), freezing both
    sides of the strip/keep behavior for description-plus-schema extensions.
    """

    async def _lookup(**inputs: Any) -> str:
        return ""

    _lookup.__name__ = "mcp_lookup"
    _lookup.__doc__ = "Look up a record in the pinned MCP fixture service."

    async def _report(**inputs: Any) -> str:
        return ""

    _report.__name__ = "mcp_report"
    _report.__doc__ = "Produce a report from the pinned MCP fixture service."

    return ToolExtension(
        name="mcp",
        tools={"mcp_lookup": _lookup, "mcp_report": _report},
        tool_groups={"mcp_tools": ["mcp_lookup", "mcp_report"]},
        tool_schemas={
            "mcp_lookup": {
                "type": "object",
                "properties": {"key": {"type": "string", "description": "Record key."}},
                "required": ["key"],
            },
            "mcp_report": {
                "type": "object",
                "properties": {"format": {"type": "string"}},
            },
        },
        propagate_to_sub_agents=False,
    )


def _coder(project_path: Path, **overrides: Any) -> CoderAgent:
    kwargs = _agent_kwargs(
        project_path,
        agent_mode=AgentMode.CLI,
        custom_agent_catalog=_custom_agent_catalog(),
    )
    kwargs.update(overrides)
    return CoderAgent(**kwargs)


def _definitions(agent: Any) -> list[ToolDefinition]:
    assert agent.tool_collection is not None
    return agent.tool_collection.get_tool_list()


def build_snapshot(project_path: Path) -> dict[str, Any]:
    """Build the complete wire dump under pinned fixtures.

    ``project_path`` must be a throwaway directory; scenario fixtures (skill
    files, hosted prompt templates) are written beneath it. No content derived
    from the path may reach the dump — ``test_tool_definition_freeze.py``
    asserts path independence by building the snapshot from two directories.
    """
    project_path.mkdir(parents=True, exist_ok=True)
    snapshot: dict[str, Any] = {}

    coder_cli = _coder(project_path)
    coder_cli.gigacode_enabled = True
    snapshot["coder_cli"] = dump_definitions(_definitions(coder_cli))

    # The registry is rebuilt per call, so post-construction pins reuse the same
    # collection: get_host needs a sandbox-capable terminal manager, and
    # dispatch_agent only offers browser_target with more than one target.
    collection = coder_cli.tool_collection
    assert collection is not None
    collection.terminal_manager.sandbox = Mock(get_host=Mock(return_value="host"))
    snapshot["coder_cli_sandbox_get_host"] = dump_definitions(collection.get_tool_list(), only={"get_host"})
    del collection.terminal_manager.sandbox

    collection._browser_targets = ("playwright", "chrome")
    snapshot["coder_cli_two_browser_targets"] = dump_definitions(collection.get_tool_list(), only={"dispatch_agent"})
    collection._browser_targets = ()

    snapshot["coder_ask"] = dump_definitions(_definitions(_coder(project_path, agent_mode=AgentMode.ASK)))
    snapshot["coder_code"] = dump_definitions(
        _definitions(
            _coder(
                project_path,
                agent_mode=AgentMode.CODE,
                prompt_provider=_hosted_prompt_provider(project_path),
            )
        )
    )

    for scenario, protocol in (
        ("coder_edit_claude_code", EditProtocol.CLAUDE_CODE),
        ("coder_edit_codex_apply_patch", EditProtocol.CODEX_APPLY_PATCH),
        ("coder_edit_hashline_v2", EditProtocol.HASHLINE_V2),
    ):
        agent = _coder(project_path, config=_config(edit_protocol=protocol))
        edit_names = {"edit", "multi_edit", "write", "apply_patch"}
        snapshot[scenario] = dump_definitions(_definitions(agent), only=edit_names)

    planning = PlanningAgent(**_agent_kwargs(project_path, agent_mode=AgentMode.CLI))
    snapshot["planning"] = dump_definitions(_definitions(planning))

    snapshot["general"] = dump_definitions(_definitions(GeneralAgent(**_agent_kwargs(project_path, sub_agent=True))))
    snapshot["investigation"] = dump_definitions(
        _definitions(InvestigationAgent(**_agent_kwargs(project_path, sub_agent=True)))
    )
    snapshot["browser"] = dump_definitions(_definitions(BrowserAgent(**_agent_kwargs(project_path, sub_agent=True))))

    custom = CustomAgent(
        **_agent_kwargs(project_path, sub_agent=True),
        definition=_custom_agent_catalog().agents["reviewer"],
        allowed_tools=["read", "exec_command", "web_search"],
    )
    snapshot["custom_agent"] = dump_definitions(_definitions(custom))

    build_extensions = [*_cli_build_tool_extensions(), _skill_tool_extension(project_path), _mcp_style_extension()]
    cli_build = _coder(project_path, tool_extensions=build_extensions)
    extension_names = {name for extension in build_extensions for name in extension.tools}
    snapshot["cli_build_extensions"] = dump_definitions(_definitions(cli_build), only=extension_names)

    plan_extensions = _cli_plan_tool_extensions()
    cli_plan = PlanningAgent(**_agent_kwargs(project_path, agent_mode=AgentMode.CLI, tool_extensions=plan_extensions))
    plan_extension_names = {name for extension in plan_extensions for name in extension.tools}
    snapshot["cli_plan_extensions"] = dump_definitions(_definitions(cli_plan), only=plan_extension_names)

    structured_host = _coder(project_path)
    host_collection = structured_host.tool_collection
    assert host_collection is not None
    for scenario, schema in (
        (
            "workflow_submit_result_object",
            {
                "type": "object",
                "properties": {"verdict": {"type": "string", "description": "The verdict."}},
                "required": ["verdict"],
            },
        ),
        ("workflow_submit_result_wrapped", {"type": "string"}),
    ):
        extension = host_collection.agent_tool._structured_output_extension(schema, {})
        carrier = _coder(project_path, tool_extensions=[extension])
        snapshot[scenario] = dump_definitions(_definitions(carrier), only={"submit_result"})

    return snapshot


def render_snapshot(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
