"""Token budget for a workflow run.

Mirrors the ultracode ``budget`` global: a hard ceiling on output tokens spent
across every sub-agent the run dispatches. ``total`` of ``None`` means unbounded
(``remaining()`` is ``inf``), which is the common case when no ``+Nk`` directive
was given.
"""

from __future__ import annotations

from typing import Optional, Union

from .errors import WorkflowBudgetExceeded


class Budget:
    """Aggregate token accounting shared across all agents in a run."""

    def __init__(self, total: Optional[int] = None) -> None:
        self.total: Optional[int] = total
        self._spent: int = 0

    def add(self, tokens: Optional[int]) -> None:
        """Fold one sub-agent's token usage into the running total."""
        try:
            self._spent += max(0, int(tokens or 0))
        except (TypeError, ValueError):
            # Defensive: a provider that doesn't report usage shouldn't crash a run.
            pass

    def spent(self) -> int:
        """Output tokens spent so far across the whole run."""
        return self._spent

    def remaining(self) -> Union[int, float]:
        """Tokens left before the ceiling, or ``inf`` when unbounded."""
        if self.total is None:
            return float("inf")
        return max(0, self.total - self._spent)

    def check(self) -> None:
        """Raise if the ceiling has been reached. Called before each dispatch."""
        if self.total is not None and self._spent >= self.total:
            raise WorkflowBudgetExceeded(f"workflow token budget exhausted: spent {self._spent} of {self.total}")
