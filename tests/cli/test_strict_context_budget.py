"""Tests for the paired strict context-budget flags.

Resolution and validation live in cli/config.py (`_strict_context_budget`);
the pair carries through AgentConfig into BaseAgent, which re-validates
against its actual primary model and budgets from an effective spec with the
reservation forced. The run evidence is the `context_budget` marker
(`strict_context_budget_marker`) that `llm.run_started` and
`config_summary` expose; the experiment side audits billed usage against it.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.cli.config import (
    CliConfigError,
    CliConfigOverrides,
    _strict_context_budget,
    config_summary,
    strict_context_budget_marker,
)
from kolega_code.cli.settings import SettingsStore
from kolega_code.config import ModelProvider


class RecordingCoderAgent(CoderAgent):
    instances: list["RecordingCoderAgent"] = []
    construction_attempts: int = 0
    process_calls: int = 0

    def __init__(self, **kwargs):
        RecordingCoderAgent.construction_attempts += 1
        super().__init__(**kwargs)
        RecordingCoderAgent.instances.append(self)

    async def process_message_stream(self, message, attachments=None):
        RecordingCoderAgent.process_calls += 1
        yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}


def _setup(tmp_path, monkeypatch):
    from kolega_code.cli import main as main_module

    RecordingCoderAgent.instances = []
    RecordingCoderAgent.construction_attempts = 0
    RecordingCoderAgent.process_calls = 0
    monkeypatch.setattr(main_module, "CoderAgent", RecordingCoderAgent)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    return main_module, project


def _overrides(window, output):
    return CliConfigOverrides(context_window_tokens=window, max_output_tokens=output)


def _persist_building_override(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "state")
    settings = store.load()
    settings.set_agent_model("building", "anthropic", "claude-haiku-4-5-20251001")
    store.save(settings)


def _saved_events(tmp_path: Path) -> list[dict[str, Any]]:
    sessions_dir = tmp_path / "state" / "sessions"
    if not sessions_dir.exists():
        return []
    event_paths = sorted(sessions_dir.glob("*/events.jsonl"))
    return [
        json.loads(line) for event_path in event_paths for line in event_path.read_text(encoding="utf-8").splitlines()
    ]


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


def test_ask_validates_compatible_saved_building_override_before_run_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    main_module, project = _setup(tmp_path, monkeypatch)
    _persist_building_override(tmp_path)

    exit_code = main_module.main(
        [
            "ask",
            "do the thing",
            "--project",
            str(project),
            "--save",
            "--context-window-tokens",
            "180000",
            "--max-output-tokens",
            "8192",
        ]
    )

    assert exit_code == 0
    assert RecordingCoderAgent.construction_attempts == 1
    assert RecordingCoderAgent.process_calls == 1
    agent = RecordingCoderAgent.instances[0]
    assert agent.primary_model_config.model == "claude-haiku-4-5-20251001"
    run_started = next(event for event in _saved_events(tmp_path) if event["type"] == "llm.run_started")
    assert run_started["payload"]["context_budget"]["model"] == "claude-haiku-4-5-20251001"


@pytest.mark.parametrize(
    "window,output,error_text",
    [
        ("300000", "8192", "catalogued context length"),
        ("180000", "20000", "catalogued completion maximum"),
    ],
)
def test_ask_rejects_incompatible_saved_building_override_before_run_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    capsys: pytest.CaptureFixture[str],
    window: str,
    output: str,
    error_text: str,
) -> None:
    main_module, project = _setup(tmp_path, monkeypatch)
    _persist_building_override(tmp_path)

    exit_code = main_module.main(
        [
            "ask",
            "do the thing",
            "--project",
            str(project),
            "--save",
            "--context-window-tokens",
            window,
            "--max-output-tokens",
            output,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert error_text in captured.err
    assert "anthropic/claude-haiku-4-5-20251001" in captured.err
    assert RecordingCoderAgent.construction_attempts == 0
    assert RecordingCoderAgent.process_calls == 0
    assert "llm.run_started" not in {event["type"] for event in _saved_events(tmp_path)}


@pytest.mark.parametrize(
    "window,output,error_text",
    [
        ("300000", "8192", "catalogued context length"),
        ("180000", "20000", "catalogued completion maximum"),
    ],
)
def test_tui_rejects_incompatible_saved_building_override_before_run_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    capsys: pytest.CaptureFixture[str],
    window: str,
    output: str,
    error_text: str,
) -> None:
    main_module, project = _setup(tmp_path, monkeypatch)
    _persist_building_override(tmp_path)
    from kolega_code.cli import app as app_module

    app_construction_attempts: list[None] = []

    class UnexpectedTuiApp:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            app_construction_attempts.append(None)
            raise AssertionError("TUI app constructed before strict context-budget validation")

    monkeypatch.setattr(app_module, "KolegaCodeApp", UnexpectedTuiApp)

    exit_code = main_module.main(
        [
            "tui",
            str(project),
            "--context-window-tokens",
            window,
            "--max-output-tokens",
            output,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert error_text in captured.err
    assert "anthropic/claude-haiku-4-5-20251001" in captured.err
    assert app_construction_attempts == []
    assert "llm.run_started" not in {event["type"] for event in _saved_events(tmp_path)}


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
