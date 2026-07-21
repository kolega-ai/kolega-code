"""Internal lifetime-agent and token accounting for one workflow run."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

from .budget import Budget
from .errors import WorkflowAgentCapExceeded


class AgentReservation:
    """Lifetime launch ticket that settles one agent's cumulative token usage."""

    def __init__(self, budget: Budget) -> None:
        self._budget = budget
        self._reported_tokens = 0

    @property
    def reported_tokens(self) -> int:
        """Largest valid cumulative token total reported for this agent."""
        return self._reported_tokens

    def report_total(self, tokens: Optional[int]) -> None:
        """Add only newly reported cumulative tokens to the shared budget."""
        if type(tokens) is not int:
            return
        total = tokens

        delta = total - self._reported_tokens
        if delta <= 0:
            return

        self._budget.add(delta)
        self._reported_tokens = total


class WorkflowRunAccounting:
    """Shared admission and usage state for a single workflow invocation."""

    def __init__(self, budget: Budget, agent_cap: int) -> None:
        self._budget = budget
        self._agent_cap = agent_cap
        self._agent_count = 0

    @property
    def budget(self) -> Budget:
        """The exact budget exposed to and shared by the workflow."""
        return self._budget

    @property
    def agent_cap(self) -> int:
        """Maximum number of lifetime agent attempts admitted for the run."""
        return self._agent_cap

    @property
    def agent_count(self) -> int:
        """Number of agent launches accepted for the lifetime of the run."""
        return self._agent_count

    def reserve_agent(self) -> AgentReservation:
        """Admit one lifetime agent attempt and return its usage ticket."""
        self._budget.check()
        if self._agent_count >= self._agent_cap:
            raise WorkflowAgentCapExceeded(
                f"workflow exceeded the lifetime agent cap ({self._agent_cap}); likely a runaway loop"
            )

        self._agent_count += 1
        return AgentReservation(self._budget)


_current_agent_reservation: ContextVar[Optional[AgentReservation]] = ContextVar(
    "current_workflow_agent_reservation",
    default=None,
)


def set_current_agent_reservation(reservation: AgentReservation) -> Token[Optional[AgentReservation]]:
    """Expose a live reservation to the internal dispatch adapter without serializing it."""
    return _current_agent_reservation.set(reservation)


def reset_current_agent_reservation(token: Token[Optional[AgentReservation]]) -> None:
    _current_agent_reservation.reset(token)


def get_current_agent_reservation() -> Optional[AgentReservation]:
    return _current_agent_reservation.get()
