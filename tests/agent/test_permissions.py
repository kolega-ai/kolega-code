import os
import stat
import uuid
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import ToolCall
from kolega_code.llm.models import ToolDefinition
from kolega_code.permissions import (
    PermissionDecision,
    PermissionKind,
    PermissionMode,
    PermissionRule,
    ProjectPermissionStore,
    allow_rule_options,
    permission_request_for_tool,
)
from kolega_code.tools import Tool, ToolRegistry


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )


def test_permission_store_writes_private_file_and_directory(tmp_path):
    store = ProjectPermissionStore(tmp_path)
    rule = PermissionRule.create(
        kind=PermissionKind.COMMAND,
        tool="*",
        match_type="exact",
        pattern="npm test",
    )

    old_umask = os.umask(0)
    try:
        store.save([rule])
    finally:
        os.umask(old_umask)

    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / ".kolega").stat().st_mode) == 0o700
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_permission_store_matches_command_rules(tmp_path):
    request = permission_request_for_tool(
        "exec_command",
        {"command": "npm run test -- --watch=false"},
    )
    assert request is not None
    store = ProjectPermissionStore(tmp_path)
    store.save(
        [
            PermissionRule.create(
                kind=PermissionKind.COMMAND,
                tool="*",
                match_type="prefix",
                pattern="npm run",
            )
        ]
    )

    assert store.first_match(request) is not None
    assert (tmp_path / ".kolega" / "permissions.json").exists()


def test_allow_rule_options_for_command_include_exact_prefix_and_executable():
    request = permission_request_for_tool("exec_command", {"command": "npm run test"})
    assert request is not None

    options = allow_rule_options(request)
    rules = {(option.rule.match_type, option.rule.pattern) for option in options}

    assert ("exact", "npm run test") in rules
    assert ("prefix", "npm run") in rules
    assert ("executable", "npm") in rules


def test_edit_permission_rule_can_scope_to_path():
    request = permission_request_for_tool("write", {"path": "src/new.py", "content": ""})
    assert request is not None
    rule = PermissionRule.create(
        kind=PermissionKind.EDIT,
        tool="write",
        match_type="path",
        pattern="src/new.py",
    )

    assert rule.matches(request)


def test_claude_edit_permission_uses_file_path():
    request = permission_request_for_tool(
        "edit",
        {"file_path": "src/app.py", "old_string": "old", "new_string": "new"},
    )

    assert request is not None
    assert request.path == "src/app.py"


def test_hashline_rename_permission_includes_source_and_destination():
    request = permission_request_for_tool(
        "edit",
        {"path": "src/old.py", "edits": [], "rename": "src/new.py"},
    )

    assert request is not None
    assert request.path == "src/old.py -> src/new.py"
    assert request.summary == "edit src/old.py -> src/new.py"


def test_single_file_apply_patch_permission_can_scope_to_path():
    request = permission_request_for_tool(
        "apply_patch",
        {"input": "*** Begin Patch\n*** Add File: src/new.py\n+x = 1\n*** End Patch\n"},
    )
    assert request is not None
    rule = PermissionRule.create(
        kind=PermissionKind.EDIT,
        tool="apply_patch",
        match_type="path",
        pattern="src/new.py",
    )

    assert request.path == "src/new.py"
    assert rule.matches(request)


def test_path_rule_never_authorizes_multi_file_apply_patch():
    request = permission_request_for_tool(
        "apply_patch",
        {
            "input": (
                "*** Begin Patch\n"
                "*** Add File: src/one.py\n+one = 1\n"
                "*** Add File: src/two.py\n+two = 2\n"
                "*** End Patch\n"
            )
        },
    )
    assert request is not None
    rule = PermissionRule.create(
        kind=PermissionKind.EDIT,
        tool="apply_patch",
        match_type="path",
        pattern="src/one.py",
    )

    assert request.path == ""
    assert not rule.matches(request)
    assert all(option.rule.match_type != "path" for option in allow_rule_options(request))


def test_lsp_edit_permission_is_gated_as_edit():
    request = permission_request_for_tool(
        "lsp_edit",
        {"operation": "rename", "path": "src/app.py", "line": 3, "symbol": "old", "new_name": "new"},
    )

    assert request is not None
    assert request.kind == PermissionKind.EDIT
    assert request.summary == "lsp_edit src/app.py"


def test_lsp_edit_rename_file_permission_summary_includes_destination():
    request = permission_request_for_tool(
        "lsp_edit",
        {"operation": "rename_file", "path": "src/old.py", "new_path": "src/new.py"},
    )

    assert request is not None
    assert request.kind == PermissionKind.EDIT
    assert request.summary == "lsp_edit src/old.py -> src/new.py"


@pytest.mark.asyncio
async def test_execute_single_tool_denies_gated_tool_before_dispatch(tmp_path, agent_config, monkeypatch):
    handler = AsyncMock(return_value="command ran")

    class TestTools:
        def registry(self):
            return ToolRegistry(
                [
                    Tool(
                        name="exec_command",
                        definition=ToolDefinition(name="exec_command", description="", parameters=[]),
                        handler=handler,
                    )
                ]
            )

    async def deny(_request):
        return PermissionDecision(allowed=False, reason="No.")

    agent = BaseAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=agent_config,
        permission_mode=PermissionMode.ASK,
        permission_callback=deny,
    )
    monkeypatch.setattr(agent, "tool_collection", TestTools())
    agent.send_chat_message = AsyncMock()
    agent.log_info = AsyncMock()
    agent.log_warning = AsyncMock()

    result = await agent.execute_single_tool(
        ToolCall(
            id="tool_1",
            name="exec_command",
            input={"command": "npm run test"},
            execution_id="exec_1",
        )
    )

    assert result.is_error is True
    assert "Permission denied" in result.content
    handler.assert_not_awaited()
