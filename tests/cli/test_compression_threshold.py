"""Tests for the configurable history-compression threshold.

Mirrors the LSP master-switch plumbing: resolution (flag > env > settings >
built-in default, cli/config.py) and carry (AgentConfig.history_compression_threshold
-> BaseAgent instance attribute -> HistoryCompressor). The e2e tests drive a real
``CoderAgent`` through ``main(["ask", ...])`` and assert on the compressor the
agent actually built.

User-facing surfaces carry a percent (10-100); the agent compares a fraction,
so every assertion below crosses that conversion exactly once.
"""

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.cli.config import CliConfigError, CliConfigOverrides, _compression_threshold
from kolega_code.cli.settings import CliSettings, SettingsStore


class RecordingCoderAgent(CoderAgent):
    """The real agent, minus the network turn, recording every instance built."""

    instances: list["RecordingCoderAgent"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        RecordingCoderAgent.instances.append(self)

    async def process_message_stream(self, message, attachments=None):
        yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}


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


def _persist_threshold(tmp_path, percent: float):
    """Write compression_threshold to the isolated settings.json the CLI will read."""
    store = SettingsStore(tmp_path / "state")
    settings = store.load()
    settings.compression_threshold = percent
    store.save(settings)


# --- resolution -------------------------------------------------------------------


def test_compression_resolution_precedence():
    settings = CliSettings(compression_threshold=65.0)
    env = {"KOLEGA_CODE_COMPRESSION_THRESHOLD": "70"}
    # flag > env > settings > default (None -> the agent's built-in 0.8)
    assert _compression_threshold(env, settings, "90") == pytest.approx(0.9)
    assert _compression_threshold(env, settings, None) == pytest.approx(0.7)
    assert _compression_threshold({}, settings, None) == pytest.approx(0.65)
    assert _compression_threshold({}, None, None) is None


def test_compression_resolution_accepts_range_boundaries():
    assert _compression_threshold({}, None, "10") == pytest.approx(0.1)
    assert _compression_threshold({}, None, "100") == pytest.approx(1.0)


@pytest.mark.parametrize("raw", ["abc", "5", "101", "0", "-20", ""])
def test_compression_resolution_rejects_invalid_flag_values(raw):
    with pytest.raises(CliConfigError, match="compression threshold"):
        _compression_threshold({}, None, raw)


@pytest.mark.parametrize("raw", ["lots", "200"])
def test_compression_resolution_rejects_invalid_env_values(raw):
    with pytest.raises(CliConfigError, match="KOLEGA_CODE_COMPRESSION_THRESHOLD"):
        _compression_threshold({"KOLEGA_CODE_COMPRESSION_THRESHOLD": raw}, None, None)


def test_overrides_dataclass_carries_the_threshold():
    assert CliConfigOverrides(compression_threshold="85").compression_threshold == "85"


# --- end to end -------------------------------------------------------------------


def test_ask_default_keeps_builtin_threshold(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project)])

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]
    assert agent.history_compression_threshold == pytest.approx(0.8)
    assert agent.compressor.threshold == pytest.approx(0.8)


def test_ask_flag_sets_threshold(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project), "--compression-threshold", "60"])

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]
    assert agent.history_compression_threshold == pytest.approx(0.6)
    assert agent.compressor.threshold == pytest.approx(0.6)


def test_ask_settings_threshold_applies_without_flag(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)
    _persist_threshold(tmp_path, 85.0)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project)])

    assert exit_code == 0
    assert RecordingCoderAgent.instances[0].compressor.threshold == pytest.approx(0.85)


def test_ask_flag_beats_env_and_settings(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)
    _persist_threshold(tmp_path, 85.0)
    monkeypatch.setenv("KOLEGA_CODE_COMPRESSION_THRESHOLD", "70")

    assert main_module.main(["ask", "first", "--project", str(project)]) == 0
    assert main_module.main(["ask", "second", "--project", str(project), "--compression-threshold", "50"]) == 0

    env_and_settings, flag_wins = RecordingCoderAgent.instances
    assert env_and_settings.compressor.threshold == pytest.approx(0.7)
    assert flag_wins.compressor.threshold == pytest.approx(0.5)


def test_ask_invalid_flag_fails_loudly(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project), "--compression-threshold", "500"])

    assert exit_code == 2
    assert RecordingCoderAgent.instances == []
