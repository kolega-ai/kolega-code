import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from kolega_code.llm.accounting import UsageObservation, nonnegative_decimal, reported_cost
from kolega_code.llm.models import Message
from kolega_code.llm.providers.responses_common import _usage_from_response
from kolega_code.llm.usage import attach_normalized_usage, observation_from_metadata


@pytest.mark.parametrize("raw", [True, False, -1, "-0.1", "NaN", "Infinity", float("inf"), float("nan"), {}, "oops"])
def test_invalid_decimal_charge_is_rejected(raw: Any) -> None:
    assert nonnegative_decimal(raw) is None
    assert reported_cost("openrouter", {"cost": raw}) is None


def test_charge_routes_and_zero_are_explicit() -> None:
    assert reported_cost("custom:gateway", {"cost": "1"}) is None
    assert reported_cost("openai", {"cost": "1"}) is None
    router = reported_cost("openrouter", {"cost": 0, "cost_details": {"upstream_inference_cost": "4"}})
    assert router is not None and router["amount"] == "0"
    assert router["scope"] == "gateway_request_charge"
    assert router["provenance"]["cost_details"]["upstream_inference_cost"] == "4"
    perplexity = reported_cost("perplexity_agent", {"cost": {"total_cost": "0.25", "input_tokens_cost": "0.1"}})
    assert perplexity is not None and perplexity["amount"] == "0.25"
    assert reported_cost("perplexity_agent", {"cost": "0.25"}) is None
    xai = reported_cost("xai", {"cost_in_usd_ticks": 12345678901})
    assert xai is not None and Decimal(xai["amount"]) == Decimal("1.2345678901")
    assert xai["provenance"]["cost_in_usd_ticks"] == 12345678901


@pytest.mark.parametrize("ticks", [True, -1, 1.5, "100", float("inf")])
def test_xai_ticks_require_nonnegative_integer(ticks: Any) -> None:
    assert reported_cost("xai", {"cost_in_usd_ticks": ticks}) is None


def test_observation_is_recursively_immutable_and_detached() -> None:
    quantities = {"cache_creation": {"ephemeral_5m_input_tokens": 12}, "items": ["a"]}
    value = UsageObservation("anthropic", input_tokens=0, pricing_quantities=quantities)
    quantities["cache_creation"]["ephemeral_5m_input_tokens"] = 99
    assert value.pricing_quantities["cache_creation"]["ephemeral_5m_input_tokens"] == 12
    with pytest.raises(TypeError):
        value.pricing_quantities["cache_creation"]["ephemeral_5m_input_tokens"] = 88
    with pytest.raises(TypeError):
        value.pricing_quantities.update({"a": 1})
    payload = json.loads(json.dumps(value.to_dict(), allow_nan=False))
    assert UsageObservation.from_dict(payload) == value
    payload["pricing_quantities"]["items"].append("b")
    assert value.pricing_quantities["items"] == ("a",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_tokens": True},
        {"output_tokens": -1},
        {"total_tokens": 1.0},
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 4},
        {"complete": True},
        {"cost": {"amount": True}},
        {"cost": {"amount": "Infinity"}},
        {"pricing_quantities": {"bad": float("nan")}},
    ],
)
def test_invalid_observations_are_not_deserialized(kwargs: dict[str, Any]) -> None:
    assert UsageObservation.from_dict({"provider": "openai", **kwargs}) is None


def test_partial_missing_and_zero_survive_independently() -> None:
    value = observation_from_metadata({"prompt_tokens": 0, "cost": "0.1"}, "openrouter", "m")
    assert value.input_tokens == 0 and value.output_tokens is None and value.total_tokens is None
    assert value.cost is not None and value.cost["amount"] == "0.1"
    assert not value.complete
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=0))
    raw = _usage_from_response(response)
    assert raw["completion_tokens"] is None
    value = observation_from_metadata(raw, "openai", "m", complete=True)
    assert value.input_tokens == 0 and value.output_tokens is None and not value.complete


def test_provider_inclusive_arithmetic_and_malformed_participant() -> None:
    anthropic = observation_from_metadata(
        {"input_tokens": 2, "cache_read_input_tokens": 10, "cache_write_input_tokens": 3}, "anthropic", "m"
    )
    assert anthropic.input_tokens == 15 and anthropic.output_tokens is None
    google = observation_from_metadata({"candidates_token_count": 4, "thoughts_token_count": 3}, "google", "m")
    assert google.output_tokens == 7 and google.input_tokens is None
    malformed = observation_from_metadata(
        {"prompt_tokens": 10, "completion_tokens": 2, "cache_write_input_tokens": -1}, "openai", "m"
    )
    assert malformed.input_tokens is None and malformed.output_tokens == 2


def test_late_partial_cannot_regress_complete_core_evidence() -> None:
    complete = UsageObservation("openai", input_tokens=10, output_tokens=5, complete=True)
    partial = UsageObservation("openai", input_tokens=0, cache_read_input_tokens=2)
    result = complete.refined_by(partial)
    assert result.complete and result.total_tokens == 15 and result.input_tokens == 10
    assert result.cache_read_input_tokens == 2


def test_message_observation_roundtrip_is_additive() -> None:
    message = attach_normalized_usage(
        Message("assistant", "ok", usage_metadata={"prompt_tokens": 0, "cost": "0"}), "openrouter", "m"
    )
    restored = Message.from_dict(message.to_dict())
    assert restored.usage == message.usage
    assert restored.usage_observation == message.usage_observation
    assert restored.usage_observation is not None and restored.usage_observation.input_tokens == 0


def test_total_only_usage_remains_known_but_incomplete() -> None:
    observation = observation_from_metadata({"total_tokens": 0}, "openai", "m", complete=True)
    assert observation.total_tokens == 0 and observation.input_tokens is None
    assert not observation.complete


def test_cache_ttl_and_tool_pricing_quantities_survive() -> None:
    observation = observation_from_metadata(
        {
            "cache_creation": {"ephemeral_5m_input_tokens": 12, "ephemeral_1h_input_tokens": 0},
            "server_tool_use": {"web_search_requests": 2},
        },
        "anthropic",
        "m",
    )
    assert observation.pricing_quantities["cache_write_5m"] == 12
    assert observation.pricing_quantities["cache_write_1h"] == 0
    assert observation.pricing_quantities["web_search_requests"] == 2


def test_invalid_optional_detail_does_not_destroy_valid_tokens() -> None:
    observation = observation_from_metadata(
        {"prompt_tokens": 1, "completion_tokens": 2, "prompt_tokens_details": {"bad": object()}},
        "openai",
        "m",
        complete=True,
    )
    assert observation.total_tokens == 3 and observation.complete
    assert "prompt_tokens_details" not in observation.pricing_quantities


def test_google_sdk_modality_splits_survive_wire_capture() -> None:
    from enum import Enum
    from types import SimpleNamespace

    from kolega_code.llm.usage import google_usage_metadata

    class Modality(Enum):
        TEXT = "TEXT"
        AUDIO = "AUDIO"

    raw = google_usage_metadata(
        SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=2,
            prompt_tokens_details=[
                SimpleNamespace(modality=Modality.TEXT, token_count=7),
                SimpleNamespace(modality=Modality.AUDIO, token_count=3),
            ],
            candidates_tokens_details=[SimpleNamespace(modality=Modality.TEXT, token_count=2)],
        )
    )
    observation = observation_from_metadata(raw, "google", "m")
    assert observation.pricing_quantities["text_input_tokens"] == 7
    assert observation.pricing_quantities["audio_input_tokens"] == 3
    assert observation.pricing_quantities["text_output_tokens"] == 2
    assert observation.pricing_quantities["prompt_tokens_details"][0]["modality"] == "TEXT"
    assert json.loads(json.dumps(observation.to_dict())) == observation.to_dict()


def test_openai_explicit_modality_detail_does_not_infer_missing_text() -> None:
    observation = observation_from_metadata(
        {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"audio_tokens": 3}}, "openai", "m"
    )
    assert observation.pricing_quantities["audio_input_tokens"] == 3
    assert "text_input_tokens" not in observation.pricing_quantities


def test_openai_wire_retains_hosted_tool_and_tier_pricing_quantities() -> None:
    from types import SimpleNamespace

    from kolega_code.llm.providers.openai import _usage_metadata

    raw = _usage_metadata(
        SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            pricing_tier_input_tokens=12,
            server_tool_use={"server_search_successful_requests": 1},
        )
    )
    observation = observation_from_metadata(raw, "deepseek", "m", complete=True)
    assert observation.pricing_quantities["server_search_successful_requests"] == 1
    assert observation.pricing_quantities["pricing_tier_input_tokens"] == 12
