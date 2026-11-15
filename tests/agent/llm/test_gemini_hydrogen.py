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


def test_hydrogen_catalog_specs() -> None:
    specs = get_model_specs("google", "gemini-4-hydrogen")

    assert specs["context_length"] == 1048576
    assert specs["max_completion_tokens"] == 65536
    assert specs["input_budget"] == "window_minus_output"
    assert specs["supports_vision"] is True
    assert specs["supports_temperature"] is False
    assert specs["thinking_effort"].mode == "google_thinking_level"
    assert resolve_max_input_tokens(specs) == 983040


def test_hydrogen_follows_existing_google_models() -> None:
    models = list(GOOGLE_SPECS)
    assert models.index(("google", "gemini-4-hydrogen")) > models.index(("google", "gemini-4-argon"))
    assert models[0] == ("google", "gemini-3.8-flash")


def test_hydrogen_thinking_options_and_default() -> None:
    assert thinking_effort_options("google", "gemini-4-hydrogen") == ("low", "medium", "high", "xhigh")
    assert default_thinking_effort("google", "gemini-4-hydrogen") == "high"
    assert normalize_thinking_effort("google", "gemini-4-hydrogen", None) == "high"
    assert normalize_thinking_effort("google", "gemini-4-hydrogen", "") == "high"
    assert normalize_thinking_effort("google", "gemini-4-hydrogen", " XHIGH ") == "xhigh"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
def test_hydrogen_thinking_request_params(effort: str) -> None:
    assert build_thinking_request_params("google", "gemini-4-hydrogen", effort) == {
        "thinking_config": {"thinking_level": effort}
    }


@pytest.mark.parametrize("effort", ["none", "minimal", "max"])
def test_hydrogen_rejects_undocumented_thinking(effort: str) -> None:
    with pytest.raises(ValueError, match="Unsupported thinking effort"):
        build_thinking_request_params("google", "gemini-4-hydrogen", effort)


@pytest.mark.parametrize(
    "output_limit,expected_input_budget",
    [(1024, 1047552), (32768, 1015808), (65536, 983040)],
)
def test_hydrogen_reserves_configured_thinking_and_output(output_limit: int, expected_input_budget: int) -> None:
    specs = {**get_model_specs("google", "gemini-4-hydrogen"), "max_completion_tokens": output_limit}
    assert resolve_max_input_tokens(specs) == expected_input_budget
