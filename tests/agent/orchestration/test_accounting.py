"""Focused tests for private workflow-run accounting."""

from typing import Any, cast

import pytest

from kolega_code.agent.orchestration.accounting import WorkflowRunAccounting
from kolega_code.agent.orchestration.budget import Budget
from kolega_code.agent.orchestration.errors import WorkflowAgentCapExceeded, WorkflowBudgetExceeded


def test_reservations_share_exact_budget_and_lifetime_count() -> None:
    budget = Budget()
    accounting = WorkflowRunAccounting(budget, agent_cap=3)

    first = accounting.reserve_agent()
    second = accounting.reserve_agent()
    first.report_total(4)
    second.report_total(6)

    assert accounting.budget is budget
    assert accounting.agent_cap == 3
    assert accounting.agent_count == 2
    assert budget.spent() == 10


def test_cap_rejection_is_prospective_and_does_not_increment_count() -> None:
    accounting = WorkflowRunAccounting(Budget(), agent_cap=1)

    accounting.reserve_agent()

    with pytest.raises(WorkflowAgentCapExceeded, match="lifetime agent cap \\(1\\)"):
        accounting.reserve_agent()
    assert accounting.agent_count == 1


def test_budget_rejection_does_not_increment_count() -> None:
    budget = Budget(total=5)
    budget.add(5)
    accounting = WorkflowRunAccounting(budget, agent_cap=2)

    with pytest.raises(WorkflowBudgetExceeded, match="budget exhausted"):
        accounting.reserve_agent()
    assert accounting.agent_count == 0


def test_cumulative_reports_add_only_positive_deltas_and_are_idempotent() -> None:
    budget = Budget()
    reservation = WorkflowRunAccounting(budget, agent_cap=1).reserve_agent()

    reservation.report_total(7)
    reservation.report_total(12)
    reservation.report_total(12)

    assert reservation.reported_tokens == 12
    assert budget.spent() == 12


def test_invalid_negative_and_regressing_reports_are_ignored() -> None:
    budget = Budget()
    reservation = WorkflowRunAccounting(budget, agent_cap=1).reserve_agent()
    reservation.report_total(9)

    invalid_totals: tuple[Any, ...] = (None, -4, 3, True, 2.9, "9", "invalid", object(), float("inf"))
    for total in invalid_totals:
        reservation.report_total(cast(Any, total))

    assert reservation.reported_tokens == 9
    assert budget.spent() == 9


def test_reservations_track_independent_cumulative_totals() -> None:
    budget = Budget()
    accounting = WorkflowRunAccounting(budget, agent_cap=2)
    first = accounting.reserve_agent()
    second = accounting.reserve_agent()

    first.report_total(10)
    second.report_total(7)
    first.report_total(15)
    second.report_total(9)

    assert first.reported_tokens == 15
    assert second.reported_tokens == 9
    assert budget.spent() == 24
