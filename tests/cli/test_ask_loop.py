"""Tests for the headless ``kolega-code ask --loop`` scheduled-prompt path."""

import json
from datetime import datetime, timedelta

import pytest

from kolega_code.cli import loop as loop_module
from kolega_code.cli import messages
from kolega_code.cli.loop import LOOP_MD_RELATIVE_PATH

from ._app_test_utils import FakeCoderAgent


class LoopAskFakeAgent(FakeCoderAgent):
    """Fake CoderAgent recording every turn prompt the loop driver sends."""

    agent_name = "coder"
    instances: list["LoopAskFakeAgent"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_extensions = list(kwargs.get("prompt_extensions", []))
        LoopAskFakeAgent.instances.append(self)

    def apply_loop(self, active, prompt_extension=None):
        self.loop_active = active
        self.loop_prompt_extension = prompt_extension if active else None
        exts = [e for e in (self.prompt_extensions or []) if getattr(e, "id", None) != "cli-active-loop"]
        if active and prompt_extension is not None:
            exts.append(prompt_extension)
        self.prompt_extensions = exts

    def restore_message_history(self, history):
        pass

    def dump_message_history(self):
        return []

    async def process_message_stream(self, message, attachments=None):
        self.messages.append(message)
        yield {"type": "response", "content": "checked", "complete": True, "uuid": "resp-1"}

    async def fire_hook(self, event, payload):
        class Result:
            additional_context = None
            blocked = False
            end_turn = False

        return Result()


@pytest.fixture
def ask_env(tmp_path, monkeypatch, isolated_cli_env):
    """Patch CoderAgent, neutralize sleeping, and return the project path.

    Depends on ``isolated_cli_env`` so the provider env vars set here survive:
    that fixture clears the developer's environment during its own setup.
    """
    from kolega_code.cli import main as main_module

    LoopAskFakeAgent.instances = []
    monkeypatch.setattr(main_module, "CoderAgent", LoopAskFakeAgent)

    # Drive the loop on a virtual clock: the driver sleeps in <=1s slices, and
    # each fake slice advances ``now_local`` by exactly that much. Schedules,
    # expiry, and next-fire math all stay real while the test runs instantly.
    clock = {"now": datetime(2026, 7, 27, 10, 0, 0)}
    monkeypatch.setattr(loop_module, "now_local", lambda: clock["now"])

    async def fake_sleep(seconds):
        clock["now"] += timedelta(seconds=seconds)

    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    return project


def write_loop_md(project, content: str):
    path = project / LOOP_MD_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def json_lines(captured) -> list[dict]:
    lines = []
    for line in captured.out.splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return lines


# ---------------------------------------------------------------------------
# Iteration behavior
# ---------------------------------------------------------------------------


def test_loop_runs_up_to_the_iteration_cap(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    exit_code = main_module.main(
        ["ask", "check the deploy", "--loop", "5m", "--loop-max-iterations", "3", "--project", str(ask_env)]
    )

    assert exit_code == 0
    agent = LoopAskFakeAgent.instances[0]
    assert len(agent.messages) == 3
    for index, message in enumerate(agent.messages, start=1):
        assert f"[Scheduled loop iteration {index} of 3]" in message
        assert "check the deploy" in message


def test_loop_applies_the_loop_prompt_extension(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    main_module.main(
        ["ask", "check the deploy", "--loop", "5m", "--loop-max-iterations", "1", "--project", str(ask_env)]
    )

    agent = LoopAskFakeAgent.instances[0]
    assert agent.loop_active is True
    assert "cli-active-loop" in {getattr(e, "id", None) for e in agent.prompt_extensions}


def test_interval_loop_does_not_sleep_before_the_first_iteration(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    main_module.main(["ask", "poll", "--loop", "5m", "--loop-max-iterations", "2", "--project", str(ask_env)])

    captured = capsys.readouterr()
    # One sleep between the two iterations, none before the first.
    assert captured.err.count("[loop] sleeping") == 1
    assert captured.err.index("[loop] iteration 1/2") < captured.err.index("[loop] sleeping")


def test_cron_loop_sleeps_before_the_first_iteration(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    main_module.main(
        ["ask", "briefing", "--loop-cron", "0 9 * * *", "--loop-max-iterations", "1", "--project", str(ask_env)]
    )

    captured = capsys.readouterr()
    assert captured.err.index("[loop] sleeping") < captured.err.index("[loop] iteration 1/1")


def test_loop_fresh_clears_history_from_the_second_iteration(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    main_module.main(
        [
            "ask",
            "poll",
            "--loop",
            "5m",
            "--loop-max-iterations",
            "3",
            "--loop-fresh",
            "--project",
            str(ask_env),
        ]
    )

    agent = LoopAskFakeAgent.instances[0]
    assert agent.clear_history_calls == 2  # before iterations 2 and 3, not 1
    assert "fresh conversation thread" in agent.messages[1]


def test_loop_without_fresh_keeps_history(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    main_module.main(["ask", "poll", "--loop", "5m", "--loop-max-iterations", "2", "--project", str(ask_env)])

    assert LoopAskFakeAgent.instances[0].clear_history_calls == 0


def test_expiry_ends_the_loop_before_the_cap(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    exit_code = main_module.main(
        [
            "ask",
            "poll",
            "--loop",
            "1h",
            "--loop-expires",
            "90m",
            "--loop-max-iterations",
            "50",
            "--project",
            str(ask_env),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "expired" in captured.err
    # Fires at t=0 and t=1h; the t=2h fire is past the 90m expiry.
    assert len(LoopAskFakeAgent.instances[0].messages) == 2


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def test_json_emits_loop_iteration_sleep_and_result_lines(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    main_module.main(["ask", "poll", "--loop", "5m", "--loop-max-iterations", "2", "--json", "--project", str(ask_env)])

    payloads = json_lines(capsys.readouterr())
    iterations = [item for item in payloads if item.get("kind") == "loop_iteration"]
    sleeps = [item for item in payloads if item.get("kind") == "loop_sleep"]
    results = [item for item in payloads if item.get("kind") == "loop_result"]

    assert [item["data"]["iteration"] for item in iterations] == [1, 2]
    assert all(item["data"]["prompt_source"] == "inline" for item in iterations)
    assert all(item["data"]["scheduled_at"] for item in iterations)
    assert len(sleeps) == 1
    assert sleeps[0]["data"]["seconds"] == pytest.approx(300, abs=2)
    assert sleeps[0]["data"]["next_fire_at"]
    assert results[0]["data"] == {"iterations": 2, "reason": "reached the iteration cap"}


# ---------------------------------------------------------------------------
# Prompt resolution and validation
# ---------------------------------------------------------------------------


def test_prompt_falls_back_to_loop_md(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    write_loop_md(ask_env, "schedule: 30m\n\ntend the branch\n")
    exit_code = main_module.main(["ask", "--loop", "5m", "--loop-max-iterations", "1", "--project", str(ask_env)])

    assert exit_code == 0
    assert "tend the branch" in LoopAskFakeAgent.instances[0].messages[0]


def test_json_reports_the_loop_md_prompt_source(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    write_loop_md(ask_env, "tend the branch\n")
    main_module.main(["ask", "--loop", "5m", "--loop-max-iterations", "1", "--json", "--project", str(ask_env)])

    payloads = json_lines(capsys.readouterr())
    iterations = [item for item in payloads if item.get("kind") == "loop_iteration"]
    assert iterations[0]["data"]["prompt_source"] == "loop_md"


def test_missing_prompt_and_missing_loop_md_exits_two(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    exit_code = main_module.main(["ask", "--loop", "5m", "--project", str(ask_env)])

    assert exit_code == 2
    assert messages.LOOP_MD_MISSING in capsys.readouterr().err


def test_missing_prompt_without_loop_still_exits_two(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    exit_code = main_module.main(["ask", "--project", str(ask_env)])

    assert exit_code == 2
    assert "prompt is required" in capsys.readouterr().err


def test_loop_and_goal_together_exit_two(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    exit_code = main_module.main(["ask", "poll", "--loop", "5m", "--goal", "ship it", "--project", str(ask_env)])

    assert exit_code == 2
    assert "--loop cannot be combined with --goal" in capsys.readouterr().err


def test_loop_and_loop_cron_are_mutually_exclusive(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    with pytest.raises(SystemExit) as excinfo:
        main_module.main(["ask", "poll", "--loop", "5m", "--loop-cron", "0 9 * * *", "--project", str(ask_env)])
    assert excinfo.value.code == 2


def test_invalid_interval_exits_two(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    exit_code = main_module.main(["ask", "poll", "--loop", "5s", "--project", str(ask_env)])

    assert exit_code == 2
    assert "at least" in capsys.readouterr().err


def test_invalid_cron_exits_two(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    exit_code = main_module.main(["ask", "poll", "--loop-cron", "0 9 * * MON", "--project", str(ask_env)])

    assert exit_code == 2
    assert "day-of-week" in capsys.readouterr().err


def test_symlinked_loop_md_exits_two(ask_env, capsys, isolated_cli_env, tmp_path):
    import sys

    if sys.platform == "win32":
        pytest.skip("symlink creation needs elevation on Windows")

    from kolega_code.cli import main as main_module

    real = tmp_path / "elsewhere.md"
    real.write_text("do something\n", encoding="utf-8")
    link = ask_env / LOOP_MD_RELATIVE_PATH
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)

    exit_code = main_module.main(["ask", "--loop", "5m", "--project", str(ask_env)])

    assert exit_code == 2
    assert "symlink" in capsys.readouterr().err


def test_a_plain_ask_is_unaffected(ask_env, capsys, isolated_cli_env):
    from kolega_code.cli import main as main_module

    exit_code = main_module.main(["ask", "just one question", "--project", str(ask_env)])

    assert exit_code == 0
    agent = LoopAskFakeAgent.instances[0]
    assert agent.messages == ["just one question"]
    assert agent.loop_active is False
    assert "[loop]" not in capsys.readouterr().err
