from __future__ import annotations

from typing import Any, Literal, cast

PlanApprovalAction = Literal["implement_plan", "implement_plan_clear", "discuss_plan"]

PLAN_APPROVAL_OPTIONS: tuple[tuple[PlanApprovalAction, str], ...] = (
    ("implement_plan", "Implement plan"),
    ("implement_plan_clear", "Clear context and implement plan"),
    ("discuss_plan", "Discuss further"),
)


def plan_approval_payload(plan: str) -> dict[str, Any]:
    if not isinstance(plan, str) or not plan.strip():
        raise ValueError("A plan approval requires nonempty plan text.")
    return {
        "plan": plan,
        "options": [{"id": action, "label": label} for action, label in PLAN_APPROVAL_OPTIONS],
    }


def plan_approval_action(value: object) -> PlanApprovalAction | None:
    if isinstance(value, str) and any(value == action for action, _ in PLAN_APPROVAL_OPTIONS):
        return cast(PlanApprovalAction, value)
    return None
