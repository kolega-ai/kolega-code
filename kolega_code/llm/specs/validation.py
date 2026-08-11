"""Validation shared by bundled and runtime model catalogs."""

from typing import Any, Mapping

INPUT_BUDGET_CONVENTIONS = ("window_minus_output", "separate_output_limit", "output_shares_window")


def validate_model_spec(specs: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a model spec cannot support context budgeting."""
    for key in ("context_length", "max_completion_tokens"):
        value = specs.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")

    convention = specs.get("input_budget")
    if convention not in INPUT_BUDGET_CONVENTIONS:
        raise ValueError(f"missing or invalid input_budget (got {convention!r})")
