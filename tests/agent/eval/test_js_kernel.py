"""Integration tests for the persistent JavaScript kernel (Bun/Node).

Skipped when neither runtime is on PATH. Marked integration since they spawn
external runtimes whose behavior can vary across versions.
"""

import sys
import uuid
from unittest.mock import Mock

import pytest
import pytest_asyncio

from kolega_code.agent.eval.js_kernel import probe_js_runtime
from kolega_code.agent.eval.kernel import EvalKernelManager
from kolega_code.llm.models import ToolResult

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _runtime_or_skip(config):
    runtime = probe_js_runtime(config)
    if runtime is None:
        pytest.skip("no JavaScript runtime (bun/node >= 18) on PATH")
    return runtime


class FakeAgent:
    sub_agent = False
    supports_vision = False

    def __init__(self):
        self.calls = []
        self.tool_collection = None

    async def execute_single_tool(self, tool_call):
        self.calls.append((tool_call.name, tool_call.input))
        if tool_call.name == "add":
            total = tool_call.input["a"] + tool_call.input["b"]
            return ToolResult(tool_use_id=tool_call.id, content=str(total), name=tool_call.name, is_error=False)
        return ToolResult(
            tool_use_id=tool_call.id, content=f"unknown tool: {tool_call.name}", name=tool_call.name, is_error=True
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
    _runtime_or_skip(config)
    instance = EvalKernelManager.for_thread(
        workspace_id="test_ws",
        thread_id=f"test-js-{uuid.uuid4().hex}",
        project_path=tmp_path,
        config=config,
    )
    yield instance
    await instance.shutdown()


async def test_result_echo_and_console(manager):
    agent = FakeAgent()
    result = await manager.execute(language="js", code="console.log('hello js');\n40 + 2", agent=agent, timeout=30)
    assert result.status == "ok"
    assert result.stdout == "hello js\n"
    assert result.result_bundle["application/json"] == 42


async def test_declarations_persist_across_cells(manager):
    agent = FakeAgent()
    await manager.execute(language="js", code="let base = 40; const two = 2;", agent=agent, timeout=30)
    result = await manager.execute(language="js", code="base + two", agent=agent, timeout=30)
    assert result.result_bundle["application/json"] == 42


async def test_redeclaration_error_is_actionable(manager):
    agent = FakeAgent()
    await manager.execute(language="js", code="let x = 1;", agent=agent, timeout=30)
    result = await manager.execute(language="js", code="let x = 2;", agent=agent, timeout=30)
    assert result.status == "error"
    assert result.error is not None
    assert result.error.ename == "SyntaxError"
    assert "without" in result.error.evalue and "reset=true" in result.error.evalue


async def test_top_level_await_and_set_global(manager):
    agent = FakeAgent()
    awaited = await manager.execute(
        language="js",
        code="const z = await Promise.resolve(7); setGlobal('z', z);",
        agent=agent,
        timeout=30,
    )
    assert awaited.status == "ok"
    result = await manager.execute(language="js", code="z + 1", agent=agent, timeout=30)
    assert result.result_bundle["application/json"] == 8


async def test_throw_produces_error_frame(manager):
    agent = FakeAgent()
    result = await manager.execute(language="js", code="throw new Error('boom')", agent=agent, timeout=30)
    assert result.status == "error"
    assert result.error is not None
    assert result.error.evalue == "boom"


async def test_timeout_interrupts_and_kernel_survives(manager):
    agent = FakeAgent()
    result = await manager.execute(
        language="js",
        code="await new Promise((resolve) => setTimeout(resolve, 60000))",
        agent=agent,
        timeout=1,
    )
    assert result.status == "error"
    assert result.interrupted is True

    alive = await manager.execute(language="js", code="'still alive'", agent=agent, timeout=30)
    assert alive.status == "ok"
    assert alive.result_bundle["text/plain"] == "still alive"


async def test_bridge_tool_call(manager):
    agent = FakeAgent()
    result = await manager.execute(
        language="js",
        code="return await tool.add({a: 5, b: 6});",
        agent=agent,
        timeout=30,
    )
    assert result.status == "ok"
    assert agent.calls == [("add", {"a": 5, "b": 6})]
    assert result.result_bundle["text/plain"] == "11"


async def test_read_write_helpers(manager, tmp_path):
    agent = FakeAgent()
    result = await manager.execute(
        language="js",
        code="write('note.txt', 'some data');\nread('note.txt');",
        agent=agent,
        timeout=30,
    )
    assert result.status == "ok"
    assert result.result_bundle["text/plain"] == "some data"
    assert (tmp_path / "note.txt").read_text() == "some data"


async def test_reset_wipes_state(manager):
    agent = FakeAgent()
    await manager.execute(language="js", code="var marker = 'x';", agent=agent, timeout=30)
    await manager.execute(language="js", code="'reset'", agent=agent, timeout=30, reset=True)
    result = await manager.execute(language="js", code="typeof marker", agent=agent, timeout=30)
    assert result.result_bundle["text/plain"] == "undefined"


async def test_json_values_in_results(manager):
    agent = FakeAgent()
    result = await manager.execute(language="js", code="({a: 1, b: [1, 2, 3]})", agent=agent, timeout=30)
    assert result.status == "ok"
    assert result.result_bundle["application/json"] == {"a": 1, "b": [1, 2, 3]}
