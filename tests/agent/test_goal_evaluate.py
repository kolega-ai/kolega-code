# ruff: noqa: F401
"""Tests for ``BaseAgent.evaluate_goal_condition``.

The method dispatches a read-only investigation sub-agent via
``self.tool_collection.agent_tool.dispatch_investigation_agent`` and parses the
returned text into a :class:`GoalVerdict`. Dispatch failures and malformed
verdicts are reported as not-met (never raise), while a missing tool collection
raises ``RuntimeError`` so the caller can surface a setup bug.
"""

import pytest
from unittest.mock import AsyncMock

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.goal import GoalVerdict


class FakeToolCollection:
    """Minimal stand-in for a tool collection exposing ``agent_tool`` dispatch."""

    def __init__(self, dispatch_return=None, dispatch_exc=None):
        # ``hasattr(self.tool_collection, "agent_tool")`` must be True, so expose
        # the agent_tool attribute pointing at a mock that records the call.
        self.agent_tool = AsyncMock()
        if dispatch_exc is not None:
            self.agent_tool.dispatch_investigation_agent.side_effect = dispatch_exc
        else:
            self.agent_tool.dispatch_investigation_agent.return_value = dispatch_return


@pytest.mark.asyncio
async def test_evaluate_goal_condition_met_true(base_agent):
    """A verifier reply of ``{"ok": true}`` yields ``met=True``."""
    fake = FakeToolCollection(dispatch_return='{"ok": true}')
    base_agent.tool_collection = fake

    verdict = await base_agent.evaluate_goal_condition("all tests pass")

    assert isinstance(verdict, GoalVerdict)
    assert verdict.met is True


@pytest.mark.asyncio
async def test_evaluate_goal_condition_met_false(base_agent):
    """A verifier reply of ``{"ok": false, ...}`` yields ``met=False`` with reason."""
    fake = FakeToolCollection(dispatch_return='{"ok": false, "reason": "still failing"}')
    base_agent.tool_collection = fake

    verdict = await base_agent.evaluate_goal_condition("all tests pass")

    assert isinstance(verdict, GoalVerdict)
    assert verdict.met is False
    assert verdict.reason == "still failing"


@pytest.mark.asyncio
async def test_evaluate_goal_condition_dispatch_exception(base_agent):
    """A dispatch failure is reported as not-met, never re-raised."""
    fake = FakeToolCollection(dispatch_exc=RuntimeError("network down"))
    base_agent.tool_collection = fake

    verdict = await base_agent.evaluate_goal_condition("all tests pass")

    assert isinstance(verdict, GoalVerdict)
    assert verdict.met is False
    assert "verifier error" in verdict.reason
    assert "network down" in verdict.reason


@pytest.mark.asyncio
async def test_evaluate_goal_condition_missing_tool_collection_raises(base_agent):
    """No tool collection (or one without ``agent_tool``) raises ``RuntimeError``."""
    base_agent.tool_collection = None

    with pytest.raises(RuntimeError):
        await base_agent.evaluate_goal_condition("all tests pass")


@pytest.mark.asyncio
async def test_evaluate_goal_condition_missing_agent_tool_raises(base_agent):
    """A tool collection lacking ``agent_tool`` also raises ``RuntimeError``."""

    class BareCollection:
        pass

    base_agent.tool_collection = BareCollection()

    with pytest.raises(RuntimeError):
        await base_agent.evaluate_goal_condition("all tests pass")


@pytest.mark.asyncio
async def test_evaluate_goal_condition_passes_condition_in_instruction(base_agent):
    """The condition text must appear inside the instruction sent to the verifier."""
    fake = FakeToolCollection(dispatch_return='{"ok": true}')
    base_agent.tool_collection = fake

    await base_agent.evaluate_goal_condition("all tests pass")

    assert fake.agent_tool.dispatch_investigation_agent.await_count == 1
    (instruction,), _ = fake.agent_tool.dispatch_investigation_agent.call_args
    assert "all tests pass" in instruction
