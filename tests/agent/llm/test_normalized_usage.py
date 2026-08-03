"""Unit tests for kolega_code/llm/usage.py — the NormalizedUsage schema and
per-family normalization formulas (live-verified against provider APIs; see
findings/provider-usage-field-semantics.md)."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from kolega_code.config import ModelProvider
from kolega_code.llm.models import Message
from kolega_code.llm.usage import (
    REASON_MALFORMED,
    REASON_NOT_REPORTED,
    NormalizedUsage,
    normalize_usage,
    usage_token_fields,
)


@pytest.mark.parametrize("provider_name", [provider.value for provider in ModelProvider])
def test_every_model_provider_has_usage_family(provider_name: str) -> None:
    """Guard: adding a ModelProvider without a normalization family must fail."""
    assert usage_token_fields(provider_name) is not None
    usage = normalize_usage({}, provider_name, model=None)
    assert usage.reported is False
    assert usage.unavailable_reason == REASON_NOT_REPORTED


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="no-such-provider"):
        normalize_usage({}, "no-such-provider", model=None)


# --- anthropic family -----------------------------------------------------------


def test_anthropic_input_is_reconstructed_inclusive_of_cache() -> None:
    usage = normalize_usage(
        {"input_tokens": 13, "output_tokens": 4, "cache_read_input_tokens": 0, "cache_write_input_tokens": 19013},
        "anthropic",
        "claude-haiku-4-5-20251001",
    )
    assert usage.reported is True
    assert usage.input_tokens == 19026
    assert usage.output_tokens == 4
    assert usage.total_tokens == 19030
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_write_input_tokens == 19013
    assert usage.reasoning_output_tokens is None
    assert usage.unavailable_reason is None


def test_anthropic_absent_cache_fields_stay_none_but_core_reported() -> None:
    usage = normalize_usage({"input_tokens": 14, "output_tokens": 4}, "anthropic", None)
    assert usage.reported is True
    assert usage.input_tokens == 14
    assert usage.total_tokens == 18
    assert usage.cache_read_input_tokens is None
    assert usage.cache_write_input_tokens is None


def test_zai_null_cache_write_with_int_cache_read() -> None:
    # Live-verified zai shape: cache_creation is always null, reads are ints.
    usage = normalize_usage(
        {"input_tokens": 28, "output_tokens": 2, "cache_read_input_tokens": 18432, "cache_write_input_tokens": None},
        "zai",
        "glm-5.2",
    )
    assert usage.input_tokens == 18460
    assert usage.cache_write_input_tokens is None


def test_moonshot_thinking_subset_passthrough() -> None:
    usage = normalize_usage(
        {"input_tokens": 92, "output_tokens": 32, "reasoning_output_tokens": 29},
        "moonshot",
        "kimi-k3",
    )
    assert usage.output_tokens == 32  # thinking already billed inside output
    assert usage.reasoning_output_tokens == 29
    assert usage.total_tokens == 124


# --- openai family --------------------------------------------------------------


def test_openai_plain_provider() -> None:
    usage = normalize_usage(
        {"prompt_tokens": 15, "completion_tokens": 28, "total_tokens": 43, "reasoning_output_tokens": 25},
        "together",
        "moonshotai/Kimi-K2.7-Code",
    )
    assert usage.input_tokens == 15
    assert usage.output_tokens == 28  # reasoning billed inside completion
    assert usage.total_tokens == 43
    assert usage.reasoning_output_tokens == 25
    assert usage.cache_write_input_tokens is None


def test_openrouter_cache_write_add_back_restores_provider_total() -> None:
    # Capture stores prompt_tokens with cache writes subtracted (openai.py); the
    # normalizer's add-back must reproduce the provider's own inclusive counts.
    # Live probe: raw prompt=19026 incl. 19023 written -> stored prompt=3.
    usage = normalize_usage(
        {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 19030,
            "cache_write_input_tokens": 19023,
            "cache_read_input_tokens": 0,
        },
        "openrouter",
        "anthropic/claude-haiku-4.5",
    )
    assert usage.input_tokens == 19026
    assert usage.output_tokens == 4
    assert usage.total_tokens == 19030  # == the raw provider total
    assert usage.cache_write_input_tokens == 19023


def test_xai_reasoning_excluded_from_completion_is_added_to_output() -> None:
    # Live-verified: xAI's completion_tokens excludes reasoning and its raw
    # total is prompt + completion + reasoning.
    usage = normalize_usage(
        {
            "prompt_tokens": 213,
            "completion_tokens": 1,
            "total_tokens": 238,
            "cache_read_input_tokens": 128,
            "reasoning_output_tokens": 24,
        },
        "xai",
        "grok-4.5",
    )
    assert usage.input_tokens == 213
    assert usage.output_tokens == 25
    assert usage.total_tokens == 238  # matches xAI's own total
    assert usage.reasoning_output_tokens == 24


def test_xai_without_reasoning_field_keeps_completion_as_output() -> None:
    usage = normalize_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, "xai", "grok-4.5")
    assert usage.output_tokens == 5
    assert usage.total_tokens == 15


def test_responses_shape_normalizes_like_chat() -> None:
    # The Responses path stores chat-shaped keys; cache writes are never stored.
    usage = normalize_usage(
        {
            "prompt_tokens": 93,
            "completion_tokens": 57,
            "total_tokens": 150,
            "reasoning_output_tokens": 48,
        },
        "deepseek",
        "deepseek-v4-flash",
    )
    assert usage.input_tokens == 93
    assert usage.output_tokens == 57
    assert usage.total_tokens == 150
    assert usage.reasoning_output_tokens == 48
    assert usage.cache_write_input_tokens is None


# --- google ---------------------------------------------------------------------


def test_google_thoughts_and_tool_use_enter_arithmetic() -> None:
    usage = normalize_usage(
        {
            "prompt_token_count": 58,
            "candidates_token_count": 16,
            "total_token_count": 137,
            "thoughts_token_count": 63,
        },
        "google",
        "gemini-3.5-flash",
    )
    assert usage.input_tokens == 58
    assert usage.output_tokens == 79  # candidates excludes thoughts
    assert usage.total_tokens == 137  # matches Google's own total
    assert usage.reasoning_output_tokens == 63
    assert usage.cache_write_input_tokens is None

    with_tool_use = normalize_usage(
        {
            "prompt_token_count": 100,
            "candidates_token_count": 10,
            "tool_use_prompt_token_count": 40,
            "cached_content_token_count": 30,
        },
        "google",
        None,
    )
    assert with_tool_use.input_tokens == 140
    assert with_tool_use.output_tokens == 10
    assert with_tool_use.cache_read_input_tokens == 30


def test_google_none_valued_cores_are_not_reported() -> None:
    usage = normalize_usage(
        {"prompt_token_count": None, "candidates_token_count": None, "total_token_count": None, "provider": "google"},
        "google",
        None,
    )
    assert usage.reported is False
    assert usage.unavailable_reason == REASON_NOT_REPORTED
    assert usage.input_tokens is None


# --- missing / malformed --------------------------------------------------------


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"provider": "openai"},
        {"provider": "openai", "edit_protocol": "diff"},
        {"prompt_tokens": 5},  # one core missing
    ],
)
def test_missing_usage_is_unreported_never_zero(metadata) -> None:
    usage = normalize_usage(metadata, "openai", "gpt-5.4-mini")
    assert usage.reported is False
    assert usage.unavailable_reason == REASON_NOT_REPORTED
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None


@pytest.mark.parametrize("bad_value", [True, False, -1, "12", 1.0])
def test_malformed_core_rejected(bad_value) -> None:
    usage = normalize_usage({"prompt_tokens": bad_value, "completion_tokens": 5}, "openai", None)
    assert usage.reported is False
    assert usage.unavailable_reason == REASON_MALFORMED
    assert usage.input_tokens is None


def test_malformed_arithmetic_subset_poisons_whole_record() -> None:
    # Anthropic cache fields participate in the inclusive-input arithmetic;
    # nulling one would silently shift tokens, so the whole record is rejected.
    usage = normalize_usage(
        {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": "junk"},
        "anthropic",
        None,
    )
    assert usage.reported is False
    assert usage.unavailable_reason == REASON_MALFORMED


def test_malformed_informational_subset_is_nulled_core_kept() -> None:
    usage = normalize_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "cache_read_input_tokens": True},
        "openai",
        None,
    )
    assert usage.reported is True
    assert usage.input_tokens == 10
    assert usage.cache_read_input_tokens is None


def test_subset_exceeding_superset_is_discarded() -> None:
    usage = normalize_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "cache_read_input_tokens": 11},
        "openai",
        None,
    )
    assert usage.reported is True
    assert usage.cache_read_input_tokens is None


# --- serialization --------------------------------------------------------------


def test_normalized_usage_round_trip() -> None:
    usage = normalize_usage(
        {"prompt_tokens": 213, "completion_tokens": 1, "reasoning_output_tokens": 24},
        "xai",
        "grok-4.5",
    )
    data = usage.to_dict()
    assert set(data) == {
        "provider",
        "model",
        "reported",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "reasoning_output_tokens",
        "unavailable_reason",
    }
    assert NormalizedUsage.from_dict(data) == usage


@pytest.mark.parametrize(
    "junk",
    [
        None,
        "junk",
        42,
        [],
        {"reported": 1},
        {"provider": "", "reported": True},
        {"provider": "x", "reported": True, "input_tokens": -1},
    ],
)
def test_from_dict_rejects_malformed(junk) -> None:
    assert NormalizedUsage.from_dict(junk) is None


def test_message_serialization_round_trip_with_usage() -> None:
    message = Message(
        role="assistant",
        content="hi",
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "provider": "anthropic"},
        usage=normalize_usage({"input_tokens": 1, "output_tokens": 2}, "anthropic", "claude-opus-5"),
    )
    data = message.to_dict()
    restored = Message.from_dict(data)
    assert restored.usage == message.usage
    assert restored.to_dict() == data


def test_message_without_usage_serializes_without_usage_key() -> None:
    message = Message(role="user", content="hi")
    assert "usage" not in message.to_dict()


def test_legacy_message_dict_deserializes_with_usage_none() -> None:
    restored = Message.from_dict({"role": "assistant", "content": "x", "usage_metadata": {"prompt_tokens": 5}})
    assert restored.usage is None
    assert restored.usage_metadata == {"prompt_tokens": 5}


# --- raw-capture behaviors in Message converters --------------------------------


def test_from_google_without_usage_does_not_fabricate_zeros() -> None:
    response = SimpleNamespace(candidates=[], usage_metadata=None, finish_reason=None)
    message = Message.from_google(cast(Any, response))
    assert message.usage_metadata == {}


def test_from_google_captures_optional_token_fields() -> None:
    response = SimpleNamespace(
        candidates=[],
        finish_reason=None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=58,
            candidates_token_count=16,
            total_token_count=137,
            cached_content_token_count=None,
            thoughts_token_count=63,
            tool_use_prompt_token_count=None,
        ),
    )
    message = Message.from_google(cast(Any, response))
    assert message.usage_metadata["thoughts_token_count"] == 63
    assert "cached_content_token_count" not in message.usage_metadata
    assert "tool_use_prompt_token_count" not in message.usage_metadata


def test_from_anthropic_captures_moonshot_thinking_tokens() -> None:
    sdk_message = SimpleNamespace(
        role="assistant",
        content=[],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=92,
            output_tokens=32,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            output_tokens_details={"thinking_tokens": 29},
        ),
    )
    message = Message.from_anthropic(sdk_message)
    assert message.usage_metadata["reasoning_output_tokens"] == 29
