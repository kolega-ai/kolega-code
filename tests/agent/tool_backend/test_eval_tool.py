"""Tests for the model-facing eval tool backend (formatting, gating, wiring)."""

import sys
import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.eval.kernel import EvalCellResult, KernelErrorInfo
from kolega_code.agent.tool_backend.eval_tool import EvalTool, clamp_cell_timeout
from kolega_code.llm.models import ImageBlock, TextBlock, ToolResult
from kolega_code.permissions import PermissionKind, allow_rule_options, permission_request_for_tool
from kolega_code.tools import ToolError

PNG_1PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

pytestmark = pytest.mark.asyncio


def make_tool(tmp_path, *, supports_vision=False, sub_agent=False, config=None, scratchpad_dir=None):
    caller = Mock()
    caller.sub_agent = sub_agent
    caller.supports_vision = supports_vision
    caller.scratchpad_dir = scratchpad_dir

    async def execute_single_tool(tool_call):
        return ToolResult(
            tool_use_id=tool_call.id,
            content=f"called {tool_call.name}",
            name=tool_call.name,
            is_error=False,
        )

    caller.execute_single_tool = execute_single_tool
    if config is None:
        config = Mock()
        config.eval_python_path = sys.executable
        config.eval_python_version = None
        config.eval_env_path = None
        config.eval_kernel_packages = None
        config.eval_js_runtime = None
    tool = EvalTool(tmp_path, "test_ws", f"eval-tool-{uuid.uuid4().hex}", AsyncMock(), config, caller)
    return tool


# -- timeout clamping ---------------------------------------------------------


def test_clamp_timeout_defaults_and_bounds():
    assert clamp_cell_timeout(None) == 120.0
    assert clamp_cell_timeout(0) is None
    assert clamp_cell_timeout(30) == 30.0
    assert clamp_cell_timeout(99999) == 600.0
    assert clamp_cell_timeout(0.1) == 1.0
    assert clamp_cell_timeout("not-a-number") == 120.0


# -- formatting ---------------------------------------------------------------


def _format(tool, result, title=None):
    return tool._format(result, title=title)


async def test_format_stdout_result_and_title(tmp_path):
    tool = make_tool(tmp_path)
    result = EvalCellResult(stdout="hello\n", result_bundle={"text/plain": "42", "application/json": 42})
    text = _format(tool, result, title="load data")
    assert isinstance(text, str)
    assert text.startswith("## load data")
    assert "### stdout\nhello" in text
    assert "### result\n42" in text


async def test_format_error_with_traceback(tmp_path):
    tool = make_tool(tmp_path)
    result = EvalCellResult(
        status="error",
        error=KernelErrorInfo(ename="ValueError", evalue="bad", traceback=["Traceback...\n", "ValueError: bad\n"]),
    )
    text = _format(tool, result)
    assert "### error" in text
    assert "ValueError: bad" in text


async def test_format_notes_and_status_lines(tmp_path):
    tool = make_tool(tmp_path)
    result = EvalCellResult(
        notes=["Provisioned env (one-time setup)."],
        statuses=[{"op": "log", "message": "loading"}, {"op": "phase", "title": "aggregate"}],
        stdout="x\n",
    )
    text = _format(tool, result)
    assert "> Provisioned env (one-time setup)." in text
    assert "- log: loading" in text
    assert "- phase: aggregate" in text


async def test_format_truncates_long_stdout(tmp_path):
    tool = make_tool(tmp_path)
    result = EvalCellResult(stdout="y" * 30_000)
    text = _format(tool, result)
    assert "truncated" in text
    assert "y" * 30_000 not in text


async def test_format_display_json_chunk(tmp_path):
    tool = make_tool(tmp_path)
    result = EvalCellResult(displays=[{"application/json": {"a": 1}, "text/plain": "{'a': 1}"}])
    text = _format(tool, result)
    assert "display[1]:" in text
    assert "'a': 1" in text


async def test_format_images_hidden_without_vision(tmp_path):
    tool = make_tool(tmp_path, supports_vision=False)
    result = EvalCellResult(displays=[{"image/png": PNG_1PX, "text/plain": "<Figure>"}])
    text = _format(tool, result)
    assert isinstance(text, str)
    assert "not shown: model has no vision support" in text


async def test_format_images_returned_with_vision(tmp_path):
    tool = make_tool(tmp_path, supports_vision=True)
    result = EvalCellResult(stdout="plot done\n", displays=[{"image/png": PNG_1PX, "text/plain": "<Figure>"}])
    blocks = _format(tool, result)
    assert isinstance(blocks, list)
    assert isinstance(blocks[0], TextBlock)
    images = [block for block in blocks if isinstance(block, ImageBlock)]
    assert len(images) == 1
    assert images[0].media_type == "image/png"
    assert images[0].data == PNG_1PX


# -- validation + end-to-end ---------------------------------------------------


async def test_rejects_unknown_language(tmp_path):
    tool = make_tool(tmp_path)
    with pytest.raises(ToolError, match="unsupported eval language"):
        await tool.eval("ruby", "puts 1")


async def test_rejects_empty_code(tmp_path):
    tool = make_tool(tmp_path)
    with pytest.raises(ToolError, match="non-empty code"):
        await tool.eval("py", "   ")


async def test_end_to_end_cell(tmp_path, isolated_cli_env):
    tool = make_tool(tmp_path)
    try:
        text = await tool.eval("py", "total = 6 * 7\ntotal", title="compute")
        assert "### result\n42" in text
        assert "custom interpreter" in text  # eval_python_path override note
        text = await tool.eval("py", "total + 1")
        assert "### result\n43" in text
    finally:
        await tool.shutdown_if_owner()


async def test_end_to_end_kernel_env_has_scratchpad(tmp_path, isolated_cli_env):
    scratchpad = tmp_path / "scratchpad"
    tool = make_tool(tmp_path, scratchpad_dir=scratchpad)
    try:
        text = await tool.eval("py", "import os\nos.environ.get('KOLEGA_SCRATCHPAD', 'missing')")
        assert f"### result\n'{scratchpad}'" in text
    finally:
        await tool.shutdown_if_owner()


async def test_end_to_end_bridge_call_reaches_caller(tmp_path, isolated_cli_env):
    tool = make_tool(tmp_path)
    try:
        text = await tool.eval("py", "tool.read_file_section({'path': 'x.py', 'start_line': 1, 'end_line': 2})")
        assert "called read_file_section" in text
    finally:
        await tool.shutdown_if_owner()


async def test_shutdown_skipped_for_sub_agents(tmp_path, isolated_cli_env):
    tool = make_tool(tmp_path, sub_agent=True)
    await tool.eval("py", "1 + 1")
    manager = tool._manager
    assert manager is not None and not manager._shutdown
    await tool.shutdown_if_owner()
    assert not manager._shutdown  # sub-agents must not kill shared kernels
    await manager.shutdown()


# -- permissions ---------------------------------------------------------------


def test_permission_request_shapes_eval_command():
    request = permission_request_for_tool("eval", {"language": "py", "code": "print(1)"})
    assert request is not None
    assert request.kind == PermissionKind.COMMAND
    assert request.command == "[py] print(1)"


def test_permission_rule_options_include_language_wide_allow():
    request = permission_request_for_tool("eval", {"language": "js", "code": "console.log(1)"})
    assert request is not None
    options = allow_rule_options(request)
    executable = [option for option in options if option.rule.match_type == "executable"]
    assert executable and executable[0].rule.pattern == "[js]"
