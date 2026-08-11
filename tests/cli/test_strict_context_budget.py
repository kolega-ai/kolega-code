"""Tests for the paired strict context-budget flags.

Resolution and validation live in cli/config.py (`_strict_context_budget`);
the pair carries through AgentConfig into BaseAgent, which re-validates
against its actual primary model and budgets from an effective spec with the
reservation forced. The run evidence is the `context_budget` marker
(`strict_context_budget_marker`) that `llm.run_started` and
`config_summary` expose; the experiment side audits billed usage against it.
"""

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.cli.config import (
    CliConfigError,
    CliConfigOverrides,
    _strict_context_budget,
    config_summary,
    strict_context_budget_marker,
)
from kolega_code.config import ModelProvider


class RecordingCoderAgent(CoderAgent):
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


def _overrides(window, output):
    return CliConfigOverrides(context_window_tokens=window, max_output_tokens=output)


# --- resolution and validation ---------------------------------------------------


def test_neither_flag_is_a_noop():
    assert _strict_context_budget(_overrides(None, None), ModelProvider.ANTHROPIC, "claude-opus-5") == (None, None)


@pytest.mark.parametrize("window,output", [("65536", None), (None, "8192")])
def test_single_flag_fails(window, output):
    with pytest.raises(CliConfigError, match="together"):
        _strict_context_budget(_overrides(window, output), ModelProvider.ANTHROPIC, "claude-opus-5")


@pytest.mark.parametrize("window,output", [("0", "8192"), ("-1", "8192"), ("abc", "8192"), ("65536.5", "8192")])
def test_non_positive_or_non_integer_window_fails(window, output):
    with pytest.raises(CliConfigError, match="positive integer"):
        _strict_context_budget(_overrides(window, output), ModelProvider.ANTHROPIC, "claude-opus-5")


@pytest.mark.parametrize("window,output", [("65536", "65536"), ("65536", "70000")])
def test_output_not_below_window_fails(window, output):
    with pytest.raises(CliConfigError, match="strictly smaller"):
        _strict_context_budget(_overrides(window, output), ModelProvider.ANTHROPIC, "claude-opus-5")


def test_window_above_catalog_fails():
    with pytest.raises(CliConfigError, match="catalogued context length"):
        _strict_context_budget(_overrides("2000000", "8192"), ModelProvider.ANTHROPIC, "claude-opus-5")


def test_output_above_catalog_fails():
    with pytest.raises(CliConfigError, match="catalogued completion maximum"):
        _strict_context_budget(_overrides("900000", "300000"), ModelProvider.ANTHROPIC, "claude-opus-5")


def test_valid_pair_resolves():
    assert _strict_context_budget(_overrides("65536", "8192"), ModelProvider.ANTHROPIC, "claude-opus-5") == (
        65536,
        8192,
    )


# --- end to end -------------------------------------------------------------------


def test_ask_without_flags_keeps_catalog_budget(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    assert main_module.main(["ask", "do the thing", "--project", str(project)]) == 0
    agent = RecordingCoderAgent.instances[0]
    assert agent.strict_context_budget is False
    assert agent.config.context_window_tokens is None
    assert agent.model_context_length == 1000000


def test_ask_with_flags_budgets_from_effective_spec(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(
        [
            "ask",
            "do the thing",
            "--project",
            str(project),
            "--context-window-tokens",
            "65536",
            "--max-output-tokens",
            "8192",
        ]
    )

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]
    assert agent.strict_context_budget is True
    assert agent.model_context_length == 65536
    assert agent.model_completion_tokens == 8192
    assert agent.model_max_input_tokens == 65536 - 8192


def test_ask_single_flag_fails_before_inference(tmp_path, monkeypatch, isolated_cli_env, capsys):
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project), "--context-window-tokens", "65536"])

    assert exit_code != 0
    assert not RecordingCoderAgent.instances


# --- evidence ---------------------------------------------------------------------


def test_marker_and_summary_expose_registered_values(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    assert (
        main_module.main(
            [
                "ask",
                "do it",
                "--project",
                str(project),
                "--context-window-tokens",
                "65536",
                "--max-output-tokens",
                "8192",
                "--compression-threshold",
                "90",
            ]
        )
        == 0
    )
    config = RecordingCoderAgent.instances[0].config
    marker = strict_context_budget_marker(config)
    assert marker is not None
    assert marker["provider"] == "anthropic"
    assert marker["model"] == "claude-opus-5"
    assert marker["context_window_tokens"] == 65536
    assert marker["max_output_tokens"] == 8192
    assert marker["max_input_tokens"] == 57344
    assert marker["catalog_context_window_tokens"] == 1000000
    assert marker["first_compaction_input_tokens"] == int(57344 * 0.9) + 1
    assert marker["source"] == "cli"
    assert config_summary(config)["context_budget"] == marker


def test_marker_absent_without_flags(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    assert main_module.main(["ask", "do it", "--project", str(project)]) == 0
    config = RecordingCoderAgent.instances[0].config
    assert strict_context_budget_marker(config) is None
    assert "context_budget" not in config_summary(config)
