from __future__ import annotations

import json
from pathlib import Path

import pytest

from kolega_code.cli import model_catalog
from kolega_code.cli.provider_registry import (
    default_model_for_provider,
    default_ui_thinking_effort,
    get_ui_model,
    ui_model_options,
    ui_provider_options,
    ui_thinking_effort_options,
)
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import (
    MODEL_SPECS,
    get_model_specs,
    provider_has_featured_models,
    provider_has_wildcard_models,
    thinking_effort_options,
    validate_thinking_effort,
)
from kolega_code.llm.specs.validation import validate_model_spec

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

MODELS = (
    "glm-5.3-abliterated",
    "glm-5.3-flash-abliterated",
    "deepseek-v4.1-flash-abliterated",
    "deepseek-v4.1-flash-fast",
    "minimax-m3.1-flash",
)


def _efforts(model: str) -> tuple[str, ...]:
    return ("off", "low", "high", "max") if model.startswith("deepseek-") else ("low", "high", "max")


def test_refuseless_exact_catalog_order_default_and_unfiltered_picker() -> None:
    assert tuple(model for provider, model in MODEL_SPECS if provider == "refuseless") == MODELS
    assert tuple(model for _label, model in ui_model_options("refuseless")) == MODELS
    assert ("Refuseless", "refuseless") in ui_provider_options()
    assert default_model_for_provider(ModelProvider.REFUSELESS) == MODELS[0]
    assert not provider_has_featured_models("refuseless")
    assert not provider_has_wildcard_models("refuseless")
    assert tuple(model for _label, model in ui_model_options("refuseless", vision_only=True)) == MODELS[1:]
    for label, model in ui_model_options("refuseless"):
        for qualifier in ("abliterated", "fast"):
            if qualifier in model:
                assert qualifier in label.lower()


@pytest.mark.parametrize("model", MODELS)
def test_refuseless_catalog_capabilities_and_efforts_reach_ui(model: str) -> None:
    spec = get_model_specs("refuseless", model)
    validate_model_spec(spec)
    option = get_ui_model("refuseless", model)
    assert option is not None
    assert option.api_key_env == "REFUSELESS_API_KEY"
    assert option.provider_label == "Refuseless"
    assert option.supports_vision is (model != MODELS[0])
    assert option.supports_vision == spec["supports_vision"]
    assert spec["pricing"]["model"] == model
    assert spec["pricing"]["status"] == "priced"
    assert spec["pricing"]["currency"] == "USD"
    assert option.thinking_efforts == _efforts(model)
    assert thinking_effort_options("refuseless", model) == _efforts(model)
    assert default_ui_thinking_effort("refuseless", model) == "low"
    assert tuple(value for _label, value in ui_thinking_effort_options("refuseless", model)) == _efforts(model)
    if model.startswith("deepseek-"):
        assert ("Off", "off") in ui_thinking_effort_options("refuseless", model)
    for effort in _efforts(model):
        assert validate_thinking_effort("refuseless", model, effort) == effort


@pytest.mark.parametrize(
    ("model", "max_completion_tokens"),
    [
        ("glm-5.3-abliterated", 131072),
        ("glm-5.3-flash-abliterated", 131072),
        ("deepseek-v4.1-flash-abliterated", 65536),
        ("deepseek-v4.1-flash-fast", 262144),
        ("minimax-m3.1-flash", 512000),
    ],
)
def test_refuseless_exact_route_limits(model: str, max_completion_tokens: int) -> None:
    spec = get_model_specs("refuseless", model)
    assert spec["context_length"] == 1048576
    assert spec["max_completion_tokens"] == max_completion_tokens


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", ("medium", "minimal", "xhigh", "none"))
def test_refuseless_rejects_non_native_effort(model: str, effort: str) -> None:
    with pytest.raises(ValueError, match="Unsupported thinking effort"):
        validate_thinking_effort("refuseless", model, effort)


@pytest.mark.parametrize("model", (MODELS[0], MODELS[1], MODELS[-1]))
def test_refuseless_always_thinking_models_reject_off(model: str) -> None:
    with pytest.raises(ValueError, match="Unsupported thinking effort"):
        validate_thinking_effort("refuseless", model, "off")


@pytest.mark.parametrize("model", ("glm-5.3-unknown", "deepseek-v4.1-flash", "private/not-a-public-model"))
def test_refuseless_unknown_models_have_no_wildcard_fallback(model: str) -> None:
    assert get_ui_model("refuseless", model) is None
    with pytest.raises(ValueError, match="is not supported"):
        get_model_specs("refuseless", model)


@pytest.mark.parametrize("featured", (False, True), ids=("all", "picker-visible"))
def test_refuseless_models_list_json_uses_real_routes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], featured: bool
) -> None:
    from kolega_code.cli.main import main

    args = ["models", "list", "--provider", "refuseless", "--state-dir", str(tmp_path / "catalog"), "--json"]
    if featured:
        args.append("--featured")
    assert main(args) == 0
    output = capsys.readouterr()
    assert output.err == ""
    rows = json.loads(output.out)
    assert tuple(row["model"] for row in rows) == MODELS
    assert [row["rank"] for row in rows] == list(range(1, len(MODELS) + 1))
    for row in rows:
        model = row["model"]
        assert row["provider"] == "refuseless"
        assert row["thinking_efforts"] == list(_efforts(model))
        assert row["default_thinking_effort"] == "low"
        assert row["supports_vision"] is (model != MODELS[0])
        assert row["context_length"] == MODEL_SPECS[("refuseless", model)]["context_length"] == 1048576


def test_refuseless_models_list_table_uses_real_routes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from kolega_code.cli.main import main

    assert main(["models", "list", "--provider", "refuseless", "--state-dir", str(tmp_path / "catalog")]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    lines = output.out.splitlines()
    assert {"RANK", "PROVIDER", "MODEL", "CONTEXT", "VISION", "EFFORTS", "DEFAULT"} <= set(lines[0].split())
    rows = [line.split() for line in lines[1:] if len(line.split()) > 2 and line.split()[1] == "refuseless"]
    assert tuple(row[2] for row in rows) == MODELS
    for rank, row in enumerate(rows, 1):
        assert row[0] == str(rank)
        assert row[3] == "1,048,576"
        assert row[5] == ("no" if rank == 1 else "yes")
        assert row[6] == ",".join(_efforts(row[2]))
        assert row[7] == "low"
    assert tuple(row["model"] for row in model_catalog.catalog_rows(provider="refuseless")) == MODELS
