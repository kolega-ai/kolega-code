"""Tests for the ``--lsp`` flag and the LSP master-switch plumbing.

The switch has one job spread over two layers: resolve (flag > env > settings >
enabled, cli/config.py) and carry (AgentConfig.lsp.enabled -> the registry gate
in ToolCollection._should_include_tool). Disabling must actually remove the
``lsp``/``lsp_edit`` tools from the model-facing toolset — not just null the
manager — so the e2e tests drive a real ``CoderAgent`` through
``main(["ask", ...])`` and assert on the tool inventory.
"""

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.cli.config import CliConfigError, CliConfigOverrides, _lsp_enabled
from kolega_code.cli.settings import CliSettings, SettingsStore

LSP_TOOLS = {"lsp", "lsp_edit"}


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


def _persist_lsp_disabled(tmp_path):
    """Write lsp_enabled=False to the isolated settings.json the CLI will read."""
    store = SettingsStore(tmp_path / "state")
    settings = store.load()
    settings.lsp_enabled = False
    store.save(settings)


# --- resolution -------------------------------------------------------------------


def test_lsp_resolution_precedence():
    settings = CliSettings(lsp_enabled=True)
    env = {"KOLEGA_CODE_LSP": "off"}
    # flag > env > settings > default (enabled)
    assert _lsp_enabled(env, settings, "on") is True
    assert _lsp_enabled(env, settings, None) is False
    assert _lsp_enabled({}, settings, None) is True
    assert _lsp_enabled({}, CliSettings(lsp_enabled=False), None) is False
    assert _lsp_enabled({}, None, None) is True


def test_lsp_flag_forces_on_over_disabled_settings():
    assert _lsp_enabled({}, CliSettings(lsp_enabled=False), "on") is True


def test_lsp_resolution_rejects_unknown_env_value():
    with pytest.raises(CliConfigError):
        _lsp_enabled({"KOLEGA_CODE_LSP": "sideways"}, None, None)


def test_overrides_dataclass_carries_the_mode():
    assert CliConfigOverrides(lsp_mode="off").lsp_mode == "off"


# --- end to end -------------------------------------------------------------------


def test_ask_lsp_off_removes_both_tools(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project), "--lsp", "off"])

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]
    assert not (LSP_TOOLS & agent.tool_names())
    assert agent.tool_collection.lsp_manager is None


def test_ask_default_keeps_lsp_tools(tmp_path, monkeypatch, isolated_cli_env):
    """No flag, no setting: today's inventory includes both LSP tools."""
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project)])

    assert exit_code == 0
    assert LSP_TOOLS <= RecordingCoderAgent.instances[0].tool_names()


def test_ask_disabled_settings_remove_tools_without_flag(tmp_path, monkeypatch, isolated_cli_env):
    """settings.json lsp_enabled=false alone must drop the tools (registry gate)."""
    main_module, project = _setup(tmp_path, monkeypatch)
    _persist_lsp_disabled(tmp_path)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project)])

    assert exit_code == 0
    assert not (LSP_TOOLS & RecordingCoderAgent.instances[0].tool_names())


def test_ask_flag_on_beats_disabled_settings(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)
    _persist_lsp_disabled(tmp_path)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project), "--lsp", "on"])

    assert exit_code == 0
    assert LSP_TOOLS <= RecordingCoderAgent.instances[0].tool_names()


def test_ask_env_off_removes_tools_and_flag_beats_env(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("KOLEGA_CODE_LSP", "off")

    assert main_module.main(["ask", "first", "--project", str(project)]) == 0
    assert main_module.main(["ask", "second", "--project", str(project), "--lsp", "on"]) == 0

    env_only, flag_wins = RecordingCoderAgent.instances
    assert not (LSP_TOOLS & env_only.tool_names())
    assert LSP_TOOLS <= flag_wins.tool_names()
