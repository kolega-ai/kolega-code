"""Tests for the ``kolega-code ask --gigacode`` headless orchestration flag.

The flag has to do everything the TUI's ``/gigacode`` does: expose the
``run_workflow`` / ``list_workflow_runs`` tools *and* inject the authoring
guide. The end-to-end tests below drive a real ``CoderAgent`` through
``main(["ask", ...])`` with only the LLM turn stubbed out, so a regression that
sets the flag without opening the tool gate fails here.
"""

from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.orchestration.guide import GIGACODE_AUTHORING_GUIDE

from ._app_test_utils import FakeCoderAgent

ORCHESTRATION_TOOLS = {"run_workflow", "list_workflow_runs"}


class RecordingCoderAgent(CoderAgent):
    """The real agent, minus the network turn, recording every instance built."""

    instances: list["RecordingCoderAgent"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        RecordingCoderAgent.instances.append(self)

    async def process_message_stream(self, message, attachments=None):
        yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    def tool_names(self) -> set[str]:
        assert self.tool_collection is not None
        return {tool.name for tool in self.tool_collection.get_tool_list()}

    def system_prompt_text(self) -> str:
        return "".join(getattr(block, "text", "") for block in self.system_prompt.content)


class GigacodeAskFakeAgent(FakeCoderAgent):
    agent_name = "coder"
    instances: list["GigacodeAskFakeAgent"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        GigacodeAskFakeAgent.instances.append(self)

    async def fire_hook(self, event, payload):
        class Result:
            additional_context = None
            blocked = False
            end_turn = False

        return Result()


def _setup(tmp_path, monkeypatch, agent_cls):
    from kolega_code.cli import main as main_module

    agent_cls.instances = []
    monkeypatch.setattr(main_module, "CoderAgent", agent_cls)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    return main_module, project


def test_ask_gigacode_exposes_orchestration_tools(tmp_path, monkeypatch, isolated_cli_env):
    """The flag opens the run_workflow tool gate, not just a boolean."""
    main_module, project = _setup(tmp_path, monkeypatch, RecordingCoderAgent)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project), "--gigacode"])

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]
    assert agent.gigacode_enabled is True
    assert ORCHESTRATION_TOOLS <= agent.tool_names()
    # The authoring guide has to reach the model, or the tools are unusable.
    assert GIGACODE_AUTHORING_GUIDE in agent.system_prompt_text()


def test_ask_without_gigacode_hides_orchestration_tools(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, RecordingCoderAgent)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project)])

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]
    assert agent.gigacode_enabled is False
    assert not (ORCHESTRATION_TOOLS & agent.tool_names())
    assert GIGACODE_AUTHORING_GUIDE not in agent.system_prompt_text()


def test_ask_gigacode_persists_and_resumes_with_the_session(tmp_path, monkeypatch, isolated_cli_env):
    """/gigacode sticks to a session; a resumed headless session inherits it."""
    main_module, project = _setup(tmp_path, monkeypatch, RecordingCoderAgent)

    assert main_module.main(["ask", "first", "--project", str(project), "--session", "s1", "--gigacode"]) == 0
    # No --gigacode on the resume: the stored session state carries it.
    assert main_module.main(["ask", "second", "--project", str(project), "--session", "s1"]) == 0

    resumed = RecordingCoderAgent.instances[1]
    assert resumed.gigacode_enabled is True
    assert ORCHESTRATION_TOOLS <= resumed.tool_names()


def test_ask_gigacode_applies_the_shared_prompt_extension(tmp_path, monkeypatch, isolated_cli_env):
    """ask and the TUI's /gigacode must hand the agent the same extension."""
    main_module, project = _setup(tmp_path, monkeypatch, GigacodeAskFakeAgent)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project), "--gigacode"])

    assert exit_code == 0
    agent = GigacodeAskFakeAgent.instances[0]
    assert agent.gigacode_enabled is True
    extension = agent.gigacode_prompt_extension
    assert extension is not None
    assert extension.id == "gigacode"
    assert extension.markdown == GIGACODE_AUTHORING_GUIDE
    # Sub-agents can't run workflows, so the guide must not propagate to them.
    assert extension.propagate_to_sub_agents is False
