"""Tests for the ``kolega-code ask --web-search`` mode flag and its plumbing.

The mode has one job spread over three layers: resolve (flag > env > settings >
"auto", cli/config.py), carry (AgentConfig.web_search_mode -> BaseAgent gates),
persist (SessionRecord / CliSettings round-trips). The e2e tests drive a real
``CoderAgent`` through ``main(["ask", ...])`` with only the LLM turn stubbed,
so a regression that sets the mode without flipping the tool gate fails here.
"""

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.cli.config import CliConfigError, CliConfigOverrides, _web_search_mode
from kolega_code.cli.session_store import SessionRecord
from kolega_code.cli.settings import CliSettings

WEB_TOOLS = {"web_search", "web_fetch"}


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


def _setup(tmp_path, monkeypatch):
    from kolega_code.cli import main as main_module

    RecordingCoderAgent.instances = []
    monkeypatch.setattr(main_module, "CoderAgent", RecordingCoderAgent)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    return main_module, project


# --- resolution -------------------------------------------------------------------


def test_mode_resolution_precedence():
    settings = CliSettings(web_search_mode="client")
    env = {"KOLEGA_CODE_WEB_SEARCH_MODE": "hosted"}
    # flag > env > settings > default
    assert _web_search_mode(env, settings, "off") == "off"
    assert _web_search_mode(env, settings, None) == "hosted"
    assert _web_search_mode({}, settings, None) == "client"
    assert _web_search_mode({}, None, None) == "auto"


def test_mode_resolution_rejects_unknown():
    with pytest.raises(CliConfigError):
        _web_search_mode({}, None, "sideways")


def test_overrides_dataclass_carries_the_mode():
    assert CliConfigOverrides(web_search_mode="off").web_search_mode == "off"


# --- persistence round-trips ------------------------------------------------------


def test_settings_round_trip_is_additive_optional():
    settings = CliSettings(web_search_mode="hosted")
    assert CliSettings.from_dict(settings.to_dict()).web_search_mode == "hosted"
    # Older files without the key load as None (default applied downstream).
    data = CliSettings().to_dict()
    del data["web_search_mode"]
    assert CliSettings.from_dict(data).web_search_mode is None


def test_session_record_round_trip(tmp_path):
    record = SessionRecord.create(tmp_path, "code", {})
    record.web_search_mode = "off"
    restored = SessionRecord.from_dict(record.to_dict())
    assert restored.web_search_mode == "off"
    # Absent in older files -> None.
    data = record.to_dict()
    del data["web_search_mode"]
    assert SessionRecord.from_dict(data).web_search_mode is None


# --- end to end -------------------------------------------------------------------


def test_ask_web_search_off_removes_both_client_tools(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project), "--web-search", "off"])

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]
    assert agent.web_search_mode == "off"
    assert not (WEB_TOOLS & agent.tool_names())
    assert agent.hosted_web_search_active is False


def test_ask_default_keeps_client_tools_on_non_hosted_model(tmp_path, monkeypatch, isolated_cli_env):
    """No flag: auto on a model without hosted support is exactly today's inventory."""
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project)])

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]
    assert agent.web_search_mode == "auto"
    assert WEB_TOOLS <= agent.tool_names()
    assert agent.hosted_web_search_active is False


def test_ask_mode_persists_and_resumes_with_the_session(tmp_path, monkeypatch, isolated_cli_env):
    """--web-search sticks to a session; a resumed headless session inherits it."""
    main_module, project = _setup(tmp_path, monkeypatch)

    assert main_module.main(["ask", "first", "--project", str(project), "--session", "s1", "--web-search", "off"]) == 0
    # No flag on the resume: the stored session state carries it.
    assert main_module.main(["ask", "second", "--project", str(project), "--session", "s1"]) == 0

    resumed = RecordingCoderAgent.instances[1]
    assert resumed.web_search_mode == "off"
    assert not (WEB_TOOLS & resumed.tool_names())
