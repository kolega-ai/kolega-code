from __future__ import annotations

from kolega_code.permissions import PermissionKind, PermissionRequest, allow_rule_options, permission_request_for_tool


def test_mcp_permission_request_and_allow_rules_are_exact_tool_or_server() -> None:
    request = permission_request_for_tool("mcp__github__search", {"query": "kolega"})
    assert request is not None
    assert request.kind == PermissionKind.MCP
    assert request.mcp_server == "github"
    assert request.mcp_tool == "search"
    assert request.summary == "github/search"

    options = allow_rule_options(request)
    exact_tool_rule = options[0].rule
    server_rule = options[1].rule

    assert exact_tool_rule.matches(request)
    same_tool_other_server = PermissionRequest(
        kind=PermissionKind.MCP,
        tool_name="mcp__docs__search",
        inputs={},
        mcp_server="docs",
        mcp_tool="search",
    )
    assert not exact_tool_rule.matches(same_tool_other_server)

    other_tool_same_server = PermissionRequest(
        kind=PermissionKind.MCP,
        tool_name="mcp__github__issues",
        inputs={},
        mcp_server="github",
        mcp_tool="issues",
    )
    assert server_rule.matches(other_tool_same_server)
