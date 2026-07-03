"""Tests for the ``kolega-code ask --goal`` run-to-completion path."""

import json


from kolega_code.agent.goal import GoalVerdict
from kolega_code.cli.goal import build_goal_task_prompt
from kolega_code.llm.models import Message


class GoalAskFakeAgent:
    """Fake CoderAgent supporting the goal loop contract."""

    instances: list["GoalAskFakeAgent"] = []
    # Class-level verdict queue so tests can pre-populate before ``main`` creates
    # the agent instance internally.
    evaluate_queue: list[GoalVerdict] = []
    default_met: bool = True
    default_reason: str = "done"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.history: list[Message] = []
        self.messages: list[str] = []
        self.active_goal_condition = None
        self.prompt_extensions = list(kwargs.get("prompt_extensions", []))
        self.evaluate_calls = 0
        GoalAskFakeAgent.instances.append(self)

    # -- goal contract --------------------------------------------------

    def apply_goal(self, condition, prompt_extension=None):
        self.active_goal_condition = condition
        exts = [e for e in (self.prompt_extensions or []) if getattr(e, "id", None) != "cli-active-goal"]
        if condition and prompt_extension is not None:
            exts.append(prompt_extension)
        self.prompt_extensions = exts

    async def evaluate_goal_condition(self, condition):
        self.evaluate_calls += 1
        if GoalAskFakeAgent.evaluate_queue:
            return GoalAskFakeAgent.evaluate_queue.pop(0)
        return GoalVerdict(
            met=GoalAskFakeAgent.default_met,
            reason=GoalAskFakeAgent.default_reason,
        )

    # -- messaging ------------------------------------------------------

    def append_user_message(self, content):
        self.history.append(Message(role="user", content=content))

    def restore_message_history(self, history):
        pass

    def dump_message_history(self):
        return []

    def dump_compaction_state(self):
        return {}

    def restore_compaction_state(self, data):
        pass

    async def process_message_stream(self, message, attachments=None):
        self.messages.append(message)
        yield {"type": "response", "content": "working", "complete": True, "uuid": "resp-1"}

    async def cleanup(self):
        pass

    async def fire_hook(self, event, payload):
        class Result:
            additional_context = None
            blocked = False
            end_turn = False

        return Result()


def _reset_fake(monkeypatch):
    """Reset the fake agent class state and monkeypatch CoderAgent."""
    from kolega_code.cli import main as main_module

    GoalAskFakeAgent.instances = []
    GoalAskFakeAgent.evaluate_queue = []
    GoalAskFakeAgent.default_met = True
    GoalAskFakeAgent.default_reason = "done"
    monkeypatch.setattr(main_module, "CoderAgent", GoalAskFakeAgent)


def _setup_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    return project


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ask_goal_no_prompt_synthesizes_task_prompt(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from kolega_code.cli import main as main_module

    _reset_fake(monkeypatch)
    project = _setup_project(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "--goal", "make tests pass", "--project", str(project)])

    assert exit_code == 0
    agent = GoalAskFakeAgent.instances[0]
    expected = build_goal_task_prompt("make tests pass")
    assert agent.messages[0] == expected
    assert agent.active_goal_condition == "make tests pass"


def test_ask_goal_with_prompt_uses_given_prompt(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from kolega_code.cli import main as main_module

    _reset_fake(monkeypatch)
    project = _setup_project(tmp_path, monkeypatch)

    exit_code = main_module.main(
        ["ask", "--goal", "make tests pass", "start by running pytest", "--project", str(project)]
    )

    assert exit_code == 0
    agent = GoalAskFakeAgent.instances[0]
    assert agent.messages[0] == "start by running pytest"
    # The synthesized task prompt must not appear as the first message.
    assert agent.messages[0] != build_goal_task_prompt("make tests pass")
    assert agent.active_goal_condition == "make tests pass"


def test_ask_goal_loops_to_completion(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from kolega_code.cli import main as main_module

    _reset_fake(monkeypatch)
    project = _setup_project(tmp_path, monkeypatch)
    GoalAskFakeAgent.evaluate_queue = [
        GoalVerdict(met=False, reason="not yet"),
        GoalVerdict(met=False, reason="still failing"),
        GoalVerdict(met=True, reason="all pass"),
    ]

    exit_code = main_module.main(["ask", "--goal", "make tests pass", "--project", str(project), "--json"])

    assert exit_code == 0
    agent = GoalAskFakeAgent.instances[0]
    # 1 task prompt + 2 nudges = 3 process_message_stream calls
    assert len(agent.messages) == 3
    assert agent.messages[0] == build_goal_task_prompt("make tests pass")
    assert agent.evaluate_calls == 3

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    eval_lines = [line for line in lines if line["kind"] == "goal_eval"]
    result_lines = [line for line in lines if line["kind"] == "goal_result"]
    assert len(eval_lines) == 3
    assert eval_lines[0]["data"]["met"] is False
    assert eval_lines[1]["data"]["met"] is False
    assert eval_lines[2]["data"]["met"] is True
    assert len(result_lines) == 1
    assert result_lines[0]["data"]["met"] is True
    assert result_lines[0]["data"]["turns"] == 3


def test_ask_goal_cap_returns_nonzero(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from kolega_code.cli import main as main_module

    _reset_fake(monkeypatch)
    project = _setup_project(tmp_path, monkeypatch)
    GoalAskFakeAgent.default_met = False
    GoalAskFakeAgent.default_reason = "nope"

    exit_code = main_module.main(
        ["ask", "--goal", "make tests pass", "--goal-max-turns", "2", "--project", str(project)]
    )

    assert exit_code == 1
    agent = GoalAskFakeAgent.instances[0]
    assert agent.evaluate_calls == 2
    # 1 task prompt + 2 nudges (a nudge follows every not-met eval, including
    # the one that reaches the cap — the while check is at the top).
    assert len(agent.messages) == 3


def test_ask_goal_json_emits_eval_and_result_lines(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from kolega_code.cli import main as main_module

    _reset_fake(monkeypatch)
    project = _setup_project(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "--goal", "x", "--project", str(project), "--json"])

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    eval_lines = [line for line in lines if line["kind"] == "goal_eval"]
    result_lines = [line for line in lines if line["kind"] == "goal_result"]
    assert len(eval_lines) == 1
    assert eval_lines[0]["data"]["met"] is True
    assert len(result_lines) == 1
    assert result_lines[0]["data"]["met"] is True


def test_ask_goal_no_prompt_and_no_goal_returns_error(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from kolega_code.cli import main as main_module

    project = tmp_path / "project"
    project.mkdir()

    exit_code = main_module.main(["ask", "--project", str(project)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "prompt is required" in captured.err


def test_ask_goal_plain_mode_emits_stderr_status(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from kolega_code.cli import main as main_module

    _reset_fake(monkeypatch)
    project = _setup_project(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "--goal", "x", "--project", str(project)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[goal]" in captured.err
    assert "MET" in captured.err
