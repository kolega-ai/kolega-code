"""Offline Argon specs and unified-window regressions, independent of UI registries."""

import pytest

from kolega_code.llm.specs import (
    build_thinking_request_params,
    default_thinking_effort,
    get_model_specs,
    normalize_thinking_effort,
    resolve_max_input_tokens,
    thinking_effort_options,
)
from kolega_code.llm.specs.catalog.google import GOOGLE_SPECS


def test_argon_catalog_specs() -> None:
    specs = get_model_specs("google", "gemini-4-argon")

    assert specs["context_length"] == 1048576
    assert specs["max_completion_tokens"] == 65536
    assert specs["input_budget"] == "window_minus_output"
    assert specs["supports_vision"] is True
    assert specs["supports_temperature"] is False
    assert specs["default_temperature"] == 1.0
    assert specs["thinking_effort"].mode == "google_thinking_level"


def test_argon_is_appended_without_reordering_existing_models() -> None:
    assert list(GOOGLE_SPECS) == [
        ("google", "gemini-3.8-flash"),
        ("google", "gemini-3.7-flash"),
        ("google", "gemini-3.6-flash"),
        ("google", "gemini-3.5-flash-lite"),
        ("google", "gemini-3.1-pro-preview"),
        ("google", "gemini-3.5-flash"),
        ("google", "gemini-4-argon"),
    ]


def test_argon_thinking_options_and_default() -> None:
    assert thinking_effort_options("google", "gemini-4-argon") == ("low", "medium", "high")
    assert default_thinking_effort("google", "gemini-4-argon") == "medium"
    assert normalize_thinking_effort("google", "gemini-4-argon", None) == "medium"


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_argon_thinking_request_params(effort: str) -> None:
    assert build_thinking_request_params("google", "gemini-4-argon", effort) == {
        "thinking_config": {"thinking_level": effort}
    }


@pytest.mark.parametrize("effort", ["none", "minimal", "xhigh", "max"])
def test_argon_rejects_unsupported_thinking(effort: str) -> None:
    with pytest.raises(ValueError, match="Unsupported thinking effort"):
        build_thinking_request_params("google", "gemini-4-argon", effort)


@pytest.mark.parametrize(
    "model,expected_input_budget",
    [("gemini-4-argon", 983040), ("gemini-3.8-flash", 1048576), ("gemini-3.1-pro-preview", 1048576)],
)
def test_argon_reserves_thinking_and_output_without_changing_gemini3(model: str, expected_input_budget: int) -> None:
    assert resolve_max_input_tokens(get_model_specs("google", model)) == expected_input_budget


@pytest.mark.parametrize(
    "output_limit,expected_input_budget",
    [(1024, 1047552), (32768, 1015808), (65536, 983040)],
)
def test_argon_shared_window_reserves_the_configured_output_limit(
    output_limit: int, expected_input_budget: int
) -> None:
    specs = {**get_model_specs("google", "gemini-4-argon"), "max_completion_tokens": output_limit}
    assert resolve_max_input_tokens(specs) == expected_input_budget
