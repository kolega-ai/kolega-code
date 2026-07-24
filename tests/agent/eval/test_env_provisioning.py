"""Slow end-to-end test: provision a real managed env and run a cell in it.

Uses an empty bundle so provisioning stays offline and fast (uv venv or
python -m venv only). The full bundle install path is covered by unit tests in
test_env.py with a mocked installer.
"""

import uuid
from unittest.mock import Mock

import pytest

from kolega_code.agent.eval import env as env_module
from kolega_code.agent.eval.kernel import EvalKernelManager

pytestmark = [pytest.mark.asyncio, pytest.mark.slow]


class FakeAgent:
    sub_agent = False
    supports_vision = False
    tool_collection = None

    async def execute_single_tool(self, tool_call):
        raise AssertionError("no bridge calls expected in this test")


@pytest.fixture
def config(tmp_path):
    mock = Mock()
    mock.eval_python_path = None
    mock.eval_python_version = "3.12"
    mock.eval_env_path = str(tmp_path / "eval-env")
    mock.eval_kernel_packages = None
    mock.eval_js_runtime = None
    return mock


async def test_provisions_real_env_and_runs_cell(tmp_path, config, isolated_cli_env, monkeypatch):
    # Empty bundle: provision the venv without downloading packages.
    monkeypatch.setattr(env_module.EvalEnvironmentManager, "_bundle_source", lambda self: b"")

    manager = EvalKernelManager.for_thread(
        workspace_id="test_ws",
        thread_id=f"test-provision-{uuid.uuid4().hex}",
        project_path=tmp_path,
        config=config,
    )
    try:
        result = await manager.execute(
            language="py",
            code="import sys\ninfo = python_info()\n[sys.version.split()[0], info['env_path']]",
            agent=FakeAgent(),
            timeout=120,
        )
        assert result.status == "ok", result.error
        assert any("one-time setup" in note for note in result.notes)
        version, env_path = result.result_bundle["application/json"]
        assert version
        assert env_path == str(tmp_path / "eval-env")

        # Second cell reuses the env (no provisioning note).
        again = await manager.execute(language="py", code="'warm'", agent=FakeAgent(), timeout=30)
        assert again.status == "ok"
        assert not any("one-time setup" in note for note in again.notes)

        # A kernel restart (reset) must not re-announce the one-time setup.
        restarted = await manager.execute(language="py", code="'fresh'", agent=FakeAgent(), timeout=30, reset=True)
        assert restarted.status == "ok"
        assert not any("one-time setup" in note for note in restarted.notes)
    finally:
        await manager.shutdown()
