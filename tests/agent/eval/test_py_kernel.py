"""Integration tests for the persistent Python kernel and the tool bridge path.

These run a real kernel subprocess on the host interpreter (eval_python_path
override skips managed-env provisioning). They stay fast and offline; the
managed-env provisioning path is covered by unit tests in test_env.py.
"""

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest
import pytest_asyncio

from kolega_code.agent.eval.kernel import EvalKernelManager
from kolega_code.llm.models import ToolResult

pytestmark = pytest.mark.asyncio


class FakeAgent:
    """Minimal agent stand-in routing bridge calls through execute_single_tool."""

    sub_agent = False
    supports_vision = False

    def __init__(self):
        self.calls = []
        self.tool_collection = None

    async def execute_single_tool(self, tool_call):
        self.calls.append((tool_call.name, tool_call.input))
        if tool_call.name == "echo":
            return ToolResult(
                tool_use_id=tool_call.id,
                content=json.dumps(tool_call.input, sort_keys=True),
                name=tool_call.name,
                is_error=False,
            )
        return ToolResult(
            tool_use_id=tool_call.id,
            content=f"unknown tool: {tool_call.name}",
            name=tool_call.name,
            is_error=True,
        )


@pytest.fixture
def config():
    mock = Mock()
    mock.eval_python_path = sys.executable
    mock.eval_python_version = None
    mock.eval_env_path = None
    mock.eval_kernel_packages = None
    mock.eval_js_runtime = None
    return mock


@pytest_asyncio.fixture
async def manager(tmp_path, config, isolated_cli_env):
    instance = EvalKernelManager.for_thread(
        workspace_id="test_ws",
        thread_id=f"test-{uuid.uuid4().hex}",
        project_path=tmp_path,
        config=config,
    )
    yield instance
    await instance.shutdown()


async def test_result_echo_and_stdout(manager):
    agent = FakeAgent()
    result = await manager.execute(language="py", code="print('hello kernel')\n40 + 2", agent=agent, timeout=30)
    assert result.status == "ok"
    assert result.stdout == "hello kernel\n"
    assert result.result_bundle["application/json"] == 42
    # First start on the override interpreter carries the custom-interpreter note.
    assert any("custom interpreter" in note for note in result.notes)


async def test_state_persists_across_cells(manager):
    agent = FakeAgent()
    await manager.execute(language="py", code="counter = 10", agent=agent, timeout=30)
    result = await manager.execute(language="py", code="counter += 5\ncounter", agent=agent, timeout=30)
    assert result.result_bundle["application/json"] == 15


async def test_reset_wipes_state(manager):
    agent = FakeAgent()
    await manager.execute(language="py", code="secret = 'x'", agent=agent, timeout=30)
    await manager.execute(language="py", code="pass", agent=agent, timeout=30, reset=True)
    result = await manager.execute(language="py", code="'secret' in globals()", agent=agent, timeout=30)
    assert result.result_bundle["application/json"] is False


async def test_top_level_await(manager):
    agent = FakeAgent()
    result = await manager.execute(
        language="py",
        code="import asyncio\nawait asyncio.sleep(0.05)\n'done'",
        agent=agent,
        timeout=30,
    )
    assert result.status == "ok"
    assert result.result_bundle["text/plain"] == "'done'"


async def test_error_traceback(manager):
    agent = FakeAgent()
    result = await manager.execute(language="py", code="def f():\n    return 1 / 0\nf()", agent=agent, timeout=30)
    assert result.status == "error"
    assert result.error is not None
    assert result.error.ename == "ZeroDivisionError"
    assert any("1 / 0" in line or "1/0" in line for line in result.error.traceback)


async def test_timeout_interrupts_and_kernel_survives(manager):
    agent = FakeAgent()
    result = await manager.execute(language="py", code="import time\ntime.sleep(60)", agent=agent, timeout=1)
    assert result.status == "error"
    assert result.interrupted is True
    assert any("timed out" in note for note in result.notes)
    assert result.error is not None and result.error.ename == "KeyboardInterrupt"

    alive = await manager.execute(language="py", code="'still alive'", agent=agent, timeout=30)
    assert alive.status == "ok"
    assert alive.result_bundle["text/plain"] == "'still alive'"


async def test_display_json_and_status_lines(manager):
    agent = FakeAgent()
    result = await manager.execute(
        language="py",
        code="log('loading')\nphase('aggregate')\ndisplay({'rows': 3})",
        agent=agent,
        timeout=30,
    )
    assert result.status == "ok"
    assert result.displays and result.displays[0]["application/json"] == {"rows": 3}
    ops = [event["op"] for event in result.statuses]
    assert ops[:2] == ["log", "phase"]


async def test_bridge_tool_call_round_trip(manager):
    agent = FakeAgent()
    result = await manager.execute(
        language="py",
        code="tool.echo({'path': 'a.py', 'n': 2})",
        agent=agent,
        timeout=30,
    )
    assert result.status == "ok"
    assert agent.calls == [("echo", {"path": "a.py", "n": 2})]
    echoed = json.loads(result.result_bundle["text/plain"].strip("'"))
    assert echoed == {"n": 2, "path": "a.py"}


async def test_bridge_tool_error_raises_in_kernel(manager):
    agent = FakeAgent()
    result = await manager.execute(language="py", code="tool.nope({'x': 1})", agent=agent, timeout=30)
    assert result.status == "error"
    assert result.error is not None
    assert result.error.ename == "RuntimeError"
    assert "unknown tool: nope" in result.error.evalue


async def test_parallel_bridge_calls(manager):
    agent = FakeAgent()
    result = await manager.execute(
        language="py",
        code="parallel([lambda i=i: tool.echo({'i': i}) for i in range(4)])",
        agent=agent,
        timeout=30,
    )
    assert result.status == "ok"
    assert len(agent.calls) == 4
    payloads = result.result_bundle["application/json"]
    assert len(payloads) == 4
    assert sorted(json.loads(item)["i"] for item in payloads) == [0, 1, 2, 3]


async def test_list_tools_empty_without_tool_collection(manager):
    agent = FakeAgent()
    result = await manager.execute(language="py", code="list_tools()", agent=agent, timeout=30)
    assert result.status == "ok"
    assert result.result_bundle["application/json"] == []


async def test_eval_via_bridge_into_busy_kernel_is_rejected(manager):
    agent = FakeAgent()
    result = await manager.execute(
        language="py",
        code="tool.eval({'language': 'py', 'code': '1'})",
        agent=agent,
        timeout=30,
    )
    assert result.status == "error"
    assert result.error is not None
    assert "busy" in result.error.evalue


async def test_project_path_importable_and_cwd(manager, tmp_path):
    (tmp_path / "myproj_mod.py").write_text("VALUE = 'from project'\n", encoding="utf-8")
    agent = FakeAgent()
    result = await manager.execute(
        language="py",
        code="import os, myproj_mod\n[os.getcwd(), myproj_mod.VALUE]",
        agent=agent,
        timeout=30,
    )
    assert result.status == "ok"
    cwd, value = result.result_bundle["application/json"]
    assert Path(cwd).resolve() == tmp_path.resolve()
    assert value == "from project"


async def test_read_write_env_helpers(manager, tmp_path):
    agent = FakeAgent()
    result = await manager.execute(
        language="py",
        code=("write('sub/out.txt', 'line1\\nline2\\nline3\\n')\nread('sub/out.txt', offset=2, limit=1)\n"),
        agent=agent,
        timeout=30,
    )
    assert result.status == "ok"
    assert result.result_bundle["text/plain"] == "'line2\\n'"
    assert (tmp_path / "sub" / "out.txt").read_text() == "line1\nline2\nline3\n"


async def test_sub_agent_shares_kernel_but_does_not_own_it(tmp_path, config, isolated_cli_env):
    thread_id = f"test-{uuid.uuid4().hex}"
    parent = EvalKernelManager.for_thread(
        workspace_id="test_ws", thread_id=thread_id, project_path=tmp_path, config=config
    )
    child = EvalKernelManager.for_thread(
        workspace_id="test_ws", thread_id=thread_id, project_path=tmp_path, config=config
    )
    assert child is parent
    agent = FakeAgent()
    await parent.execute(language="py", code="shared = 99", agent=agent, timeout=30)
    result = await child.execute(language="py", code="shared", agent=agent, timeout=30)
    assert result.result_bundle["application/json"] == 99
    await parent.shutdown()
    assert ("test_ws", thread_id) not in EvalKernelManager._registry
