"""Unit tests for SubAgentRouting.from_info — no Textual app required."""

from typing import Any, cast

from kolega_code.cli.tui.state import SubAgentRouting


def _info(effective=None, requested=None, **extra):
    info = {"agent_id": "a1", "agent_name": "general-agent", **extra}
    if effective is not None:
        info["effective_routing"] = effective
    if requested is not None:
        info["requested_routing"] = requested
    return info


def test_from_info_none_and_non_mapping():
    assert SubAgentRouting.from_info(None) is None
    assert SubAgentRouting.from_info(cast(Any, "not a mapping")) is None


def test_from_info_missing_or_malformed_effective_routing():
    assert SubAgentRouting.from_info(_info()) is None
    assert SubAgentRouting.from_info(_info(effective="anthropic/claude-opus-4-8")) is None
    assert SubAgentRouting.from_info(_info(effective={"provider": "anthropic"})) is None
    assert SubAgentRouting.from_info(_info(effective={"provider": "anthropic", "model": "  "})) is None
    assert SubAgentRouting.from_info(_info(effective={"provider": None, "model": "claude-opus-4-8"})) is None


def test_from_info_ordinary_dispatch_shape():
    routing = SubAgentRouting.from_info(
        _info(effective={"provider": "anthropic", "model": "claude-opus-4-8", "thinking_effort": "high"})
    )
    assert routing == SubAgentRouting(provider="anthropic", model="claude-opus-4-8", effort="high", overridden=False)


def test_from_info_workflow_effort_key_spelling():
    routing = SubAgentRouting.from_info(
        _info(effective={"provider": "anthropic", "model": "claude-opus-4-8", "effort": "low"})
    )
    assert routing is not None
    assert routing.effort == "low"


def test_from_info_effort_none_or_blank():
    for effort in (None, "", "   "):
        routing = SubAgentRouting.from_info(
            _info(effective={"provider": "anthropic", "model": "claude-opus-4-8", "thinking_effort": effort})
        )
        assert routing is not None
        assert routing.effort is None


def test_from_info_strips_whitespace():
    routing = SubAgentRouting.from_info(
        _info(effective={"provider": " anthropic ", "model": " claude-opus-4-8 ", "thinking_effort": "high"})
    )
    assert routing == SubAgentRouting(provider="anthropic", model="claude-opus-4-8", effort="high")


def test_from_info_requested_routing_marks_override():
    effective = {"provider": "deepseek", "model": "deepseek-r1", "thinking_effort": "high"}
    inherited = SubAgentRouting.from_info(_info(effective=effective))
    pinned = SubAgentRouting.from_info(_info(effective=effective, requested=effective))
    assert inherited is not None and inherited.overridden is False
    assert pinned is not None and pinned.overridden is True
