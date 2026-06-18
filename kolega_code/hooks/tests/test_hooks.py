import json
import sys

import pytest

from kolega_code.hooks import (
    HookCapabilities,
    HookConfig,
    HookConfigError,
    HookDispatcher,
    HookEvent,
    HookMatcher,
    HookOutcome,
    HookSpec,
    LifecycleEvent,
    load_hook_config,
    merge,
)
from kolega_code.hooks.events import enter_hook, exit_hook


# --------------------------------------------------------------------------- #
# matcher
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pattern,target,expected",
    [
        ("", "Bash", True),
        ("*", "Bash", True),
        ("Edit|Write", "Edit", True),
        ("Edit|Write", "Read", False),
        (r"mcp__.*", "mcp__memory__read", True),
        (r"mcp__.*", "execute_terminal_command", False),
        ("execute_terminal_command", "execute_terminal_command", True),
    ],
)
def test_matcher(pattern, target, expected):
    assert HookMatcher(pattern).matches(target) is expected


# --------------------------------------------------------------------------- #
# outcome / merge
# --------------------------------------------------------------------------- #


def test_merge_first_block_wins_and_chains_input():
    merged = merge(
        [
            HookOutcome(updated_input={"command": "a"}),
            HookOutcome(updated_input={"command": "b"}),
            HookOutcome(blocked=True, reason="nope"),
        ]
    )
    assert merged.blocked is True
    assert merged.reason == "nope"
    assert merged.updated_input == {"command": "b"}
    assert merged.end_turn is True  # a block always ends the turn


def test_merge_concatenates_context():
    merged = merge([HookOutcome(additional_context="one"), HookOutcome(additional_context="two")])
    assert merged.additional_context == "one\ntwo"
    assert merged.blocked is False


# --------------------------------------------------------------------------- #
# config loading / trust / validation
# --------------------------------------------------------------------------- #


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_merges_global_then_project_when_trusted(tmp_path):
    state_dir = tmp_path / "state"
    project = tmp_path / "project"
    _write(
        state_dir / "hooks.json",
        {
            "schema_version": 1,
            "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "g"}]}]},
        },
    )
    _write(
        project / ".kolega" / "hooks.json",
        {
            "schema_version": 1,
            "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "p"}]}]},
        },
    )

    config = load_hook_config(project, state_dir, project_trusted=True)
    specs = config.specs_for(HookEvent.PRE_TOOL_USE, "execute_terminal_command")
    assert [s.command for s in specs] == ["g", "p"]  # global first, project last
    assert [s.scope for s in specs] == ["global", "project"]


def test_project_hooks_skipped_when_untrusted(tmp_path):
    state_dir = tmp_path / "state"
    project = tmp_path / "project"
    _write(
        project / ".kolega" / "hooks.json",
        {
            "schema_version": 1,
            "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "p"}]}]},
        },
    )
    config = load_hook_config(project, state_dir, project_trusted=False)
    assert config.is_empty
    assert any("trusted" in d for d in config.diagnostics)


def test_unsupported_schema_version_is_diagnostic(tmp_path):
    state_dir = tmp_path / "state"
    _write(state_dir / "hooks.json", {"schema_version": 99, "hooks": {}})
    config = load_hook_config(tmp_path / "project", state_dir, project_trusted=True)
    assert config.is_empty
    assert any("schema version" in d for d in config.diagnostics)


def test_malformed_json_is_diagnostic(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "hooks.json").write_text("{not json", encoding="utf-8")
    config = load_hook_config(tmp_path / "project", state_dir, project_trusted=True)
    assert config.is_empty
    assert any("valid JSON" in d for d in config.diagnostics)


def test_agent_hook_rejected_on_tool_event(tmp_path):
    state_dir = tmp_path / "state"
    _write(
        state_dir / "hooks.json",
        {"schema_version": 1, "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "agent", "prompt": "x"}]}]}},
    )
    config = load_hook_config(tmp_path / "project", state_dir, project_trusted=True)
    assert config.is_empty
    assert any("not allowed on tool events" in d for d in config.diagnostics)


def test_agent_hook_allowed_on_stop(tmp_path):
    state_dir = tmp_path / "state"
    _write(
        state_dir / "hooks.json",
        {"schema_version": 1, "hooks": {"Stop": [{"matcher": "*", "hooks": [{"type": "agent", "prompt": "x"}]}]}},
    )
    config = load_hook_config(tmp_path / "project", state_dir, project_trusted=True)
    assert config.specs_for(HookEvent.STOP)


def test_hookspec_requires_type_field():
    with pytest.raises(HookConfigError):
        HookSpec.from_dict({"type": "command"}, scope="global")  # missing command
    with pytest.raises(HookConfigError):
        HookSpec.from_dict({"type": "bogus", "command": "x"}, scope="global")


# --------------------------------------------------------------------------- #
# command backend
# --------------------------------------------------------------------------- #


def _command_dispatcher(command: str, event: HookEvent = HookEvent.PRE_TOOL_USE, timeout: int = 10) -> HookDispatcher:
    config = HookConfig(
        entries={
            event: [(HookMatcher("*"), [HookSpec(type="command", timeout=timeout, scope="global", command=command)])]
        }
    )
    return HookDispatcher(config)


def _script(tmp_path, body: str) -> str:
    path = tmp_path / "hook.py"
    path.write_text("import sys, json\nevent = json.load(sys.stdin)\n" + body, encoding="utf-8")
    return f"{sys.executable} {path}"


@pytest.mark.asyncio
async def test_command_block_via_exit_2(tmp_path):
    command = _script(tmp_path, "sys.stderr.write('denied by policy')\nsys.exit(2)\n")
    disp = _command_dispatcher(command)
    event = LifecycleEvent(name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "x", "tool_input": {}})
    outcome = await disp.dispatch(event, target="x", caps=HookCapabilities(project_path=tmp_path))
    assert outcome.blocked and "denied by policy" in outcome.reason


@pytest.mark.asyncio
async def test_command_modifies_input_via_stdout_json(tmp_path):
    command = _script(
        tmp_path,
        "print(json.dumps({'hookSpecificOutput': {'updatedInput': {'command': 'echo safe'}}}))\n",
    )
    disp = _command_dispatcher(command)
    event = LifecycleEvent(
        name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "x", "tool_input": {"command": "rm -rf /"}}
    )
    outcome = await disp.dispatch(event, target="x", caps=HookCapabilities(project_path=tmp_path))
    assert not outcome.blocked
    assert outcome.updated_input == {"command": "echo safe"}


@pytest.mark.asyncio
async def test_command_nonzero_is_nonblocking(tmp_path):
    logs = []
    command = _script(tmp_path, "sys.exit(1)\n")
    disp = _command_dispatcher(command)
    event = LifecycleEvent(name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "x", "tool_input": {}})

    async def log(msg):
        logs.append(msg)

    outcome = await disp.dispatch(event, target="x", caps=HookCapabilities(project_path=tmp_path, log=log))
    assert outcome.is_empty
    assert any("exited 1" in m for m in logs)


@pytest.mark.asyncio
async def test_command_timeout_is_nonblocking(tmp_path):
    logs = []
    command = _script(tmp_path, "import time\ntime.sleep(5)\n")
    disp = _command_dispatcher(command, timeout=1)
    event = LifecycleEvent(name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "x", "tool_input": {}})

    async def log(msg):
        logs.append(msg)

    outcome = await disp.dispatch(event, target="x", caps=HookCapabilities(project_path=tmp_path, log=log))
    assert outcome.is_empty
    assert any("timed out" in m for m in logs)


# --------------------------------------------------------------------------- #
# python backend
# --------------------------------------------------------------------------- #


def deny_python_hook(event: LifecycleEvent) -> HookOutcome:
    return HookOutcome.deny("python says no")


async def modify_python_hook(event: LifecycleEvent) -> HookOutcome:
    return HookOutcome(updated_input={"command": "ls"})


@pytest.mark.asyncio
async def test_python_backend_block():
    config = HookConfig(
        entries={
            HookEvent.PRE_TOOL_USE: [
                (
                    HookMatcher("*"),
                    [HookSpec(type="python", timeout=5, scope="global", callable=f"{__name__}:deny_python_hook")],
                )
            ]
        }
    )
    event = LifecycleEvent(name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "x", "tool_input": {}})
    outcome = await HookDispatcher(config).dispatch(event, target="x")
    assert outcome.blocked and outcome.reason == "python says no"


@pytest.mark.asyncio
async def test_python_backend_async_modify():
    config = HookConfig(
        entries={
            HookEvent.PRE_TOOL_USE: [
                (
                    HookMatcher("*"),
                    [HookSpec(type="python", timeout=5, scope="global", callable=f"{__name__}:modify_python_hook")],
                )
            ]
        }
    )
    event = LifecycleEvent(name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "x", "tool_input": {"command": "boom"}})
    outcome = await HookDispatcher(config).dispatch(event, target="x")
    assert outcome.updated_input == {"command": "ls"}


# --------------------------------------------------------------------------- #
# LLM backends (prompt / agent) via stub runners
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_prompt_backend_ok_reason():
    config = HookConfig(
        entries={
            HookEvent.STOP: [
                (HookMatcher("*"), [HookSpec(type="prompt", timeout=5, scope="global", prompt="done? $EVENT")])
            ]
        }
    )
    seen = {}

    async def prompt_runner(text, model):
        seen["text"] = text
        return 'The tasks are not finished. {"ok": false, "reason": "tests still failing"}'

    outcome = await HookDispatcher(config).dispatch(
        LifecycleEvent(name=HookEvent.STOP, payload={"stop_reason": "end_turn"}),
        caps=HookCapabilities(prompt_runner=prompt_runner),
    )
    assert "hook_event_name" in seen["text"]  # $EVENT was rendered
    assert outcome.blocked and outcome.reason == "tests still failing"


@pytest.mark.asyncio
async def test_prompt_backend_missing_runner_fails_open():
    logs = []
    config = HookConfig(
        entries={HookEvent.STOP: [(HookMatcher("*"), [HookSpec(type="prompt", timeout=5, scope="global", prompt="x")])]}
    )

    async def log(msg):
        logs.append(msg)

    outcome = await HookDispatcher(config).dispatch(
        LifecycleEvent(name=HookEvent.STOP, payload={}), caps=HookCapabilities(log=log)
    )
    assert outcome.is_empty
    assert any("no LLM runner" in m for m in logs)


@pytest.mark.asyncio
async def test_agent_backend_runs_under_reentrancy_guard():
    from kolega_code.hooks.events import in_hook

    config = HookConfig(
        entries={
            HookEvent.STOP: [(HookMatcher("*"), [HookSpec(type="agent", timeout=5, scope="global", prompt="verify")])]
        }
    )
    flags = {}

    async def agent_runner(task):
        flags["in_hook"] = in_hook()
        return '{"ok": true}'

    outcome = await HookDispatcher(config).dispatch(
        LifecycleEvent(name=HookEvent.STOP, payload={}), caps=HookCapabilities(agent_runner=agent_runner)
    )
    assert flags["in_hook"] is True  # guard active while the agent hook runs
    assert outcome.is_empty


# --------------------------------------------------------------------------- #
# dispatcher behavior
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dispatch_is_suppressed_when_already_in_hook():
    config = HookConfig(
        entries={
            HookEvent.PRE_TOOL_USE: [
                (
                    HookMatcher("*"),
                    [HookSpec(type="python", timeout=5, scope="global", callable=f"{__name__}:deny_python_hook")],
                )
            ]
        }
    )
    event = LifecycleEvent(name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "x", "tool_input": {}})
    token = enter_hook()
    try:
        outcome = await HookDispatcher(config).dispatch(event, target="x")
    finally:
        exit_hook(token)
    assert outcome.is_empty


@pytest.mark.asyncio
async def test_no_op_dispatcher_returns_empty():
    from kolega_code.hooks import NO_OP_DISPATCHER

    event = LifecycleEvent(name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "x", "tool_input": {}})
    assert (await NO_OP_DISPATCHER.dispatch(event, target="x")).is_empty


@pytest.mark.asyncio
async def test_matcher_scopes_specs_to_target():
    config = HookConfig(
        entries={
            HookEvent.PRE_TOOL_USE: [
                (
                    HookMatcher("execute_terminal_command"),
                    [HookSpec(type="python", timeout=5, scope="global", callable=f"{__name__}:deny_python_hook")],
                )
            ]
        }
    )
    disp = HookDispatcher(config)
    # Non-matching tool -> no hooks run.
    other = await disp.dispatch(
        LifecycleEvent(name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "read_file", "tool_input": {}}),
        target="read_file",
    )
    assert other.is_empty
    # Matching tool -> blocked.
    match = await disp.dispatch(
        LifecycleEvent(
            name=HookEvent.PRE_TOOL_USE, payload={"tool_name": "execute_terminal_command", "tool_input": {}}
        ),
        target="execute_terminal_command",
    )
    assert match.blocked
