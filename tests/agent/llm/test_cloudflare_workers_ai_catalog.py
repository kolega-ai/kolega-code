"""Pinned Cloudflare Workers AI catalog entries (hermetic).

Windows, vision flags, and effort ladders mirror Cloudflare's published model
metadata (``tests/fixtures/cloudflare_workers_ai/models_search.json``, the
``/ai/models/search`` shape); the output caps are Kolega's reservation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from kolega_code.cli.provider_registry import (
    PROVIDER_LABELS,
    default_model_for_provider,
    get_ui_model,
    ui_model_options,
    ui_provider_options,
)
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import (
    MODEL_SPECS,
    default_thinking_effort,
    get_model_specs,
    preferred_edit_protocol,
    resolve_max_input_tokens,
    supports_vision,
    thinking_effort_options,
)
from kolega_code.llm.specs import cloudflare_workers_ai_catalog
from kolega_code.llm.specs.thinking import build_thinking_request_params, reasoning_replay_field

PROVIDER = ModelProvider.CLOUDFLARE_WORKERS_AI.value
FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "cloudflare_workers_ai" / "models_search.json"

# model -> (context, max output, vision, efforts, default effort, temperature)
EXPECTED: dict[str, tuple[int, int, bool, tuple[str, ...], Optional[str], float]] = {
    "@cf/zai-org/glm-5.3": (1048576, 131072, False, ("low", "high", "max"), "max", 0.6),
    "@cf/zai-org/glm-5.3-flash": (1048576, 131072, True, ("low", "high", "max"), "max", 0.6),
    "@cf/moonshotai/kimi-k2.7-code": (262144, 32768, True, (), None, 1.0),
    "@cf/deepseek-ai/deepseek-v4.1-flash-1017": (1048576, 65536, True, ("none", "low", "high", "max"), "high", 1.0),
    "@cf/deepseek-ai/deepseek-v4-pro-0813": (1048576, 65536, False, ("none", "low", "high", "max"), "high", 1.0),
    "@cf/zai-org/glm-5.2": (262144, 32768, False, ("none", "high", "max"), "max", 0.6),
    "@cf/moonshotai/kimi-k2.6": (262144, 32768, True, ("none", "high"), "high", 1.0),
    "@cf/qwen/qwen3.8-27b": (262144, 32768, True, ("low", "medium", "xhigh"), "xhigh", 0.6),
    "@cf/google/gemma-4-26b-a4b-it": (256000, 32768, True, (), None, 1.0),
    "@cf/nvidia/nemotron-3-120b-a12b": (256000, 32768, False, (), None, 1.0),
    "@cf/openai/gpt-oss-120b": (128000, 32768, False, ("low", "medium", "high"), "medium", 1.0),
    "@cf/openai/gpt-oss-20b": (128000, 32768, False, ("low", "medium", "high"), "medium", 1.0),
    "@cf/zai-org/glm-4.7-flash": (131072, 16384, False, (), None, 0.6),
}


def _catalogued() -> list[str]:
    return [model for provider, model in MODEL_SPECS if provider == PROVIDER]


def _published() -> dict[str, dict[str, Any]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return dict(cloudflare_workers_ai_catalog.catalog_entries(payload))


def test_curated_set_and_order_are_exact() -> None:
    assert _catalogued() == list(EXPECTED)


@pytest.mark.parametrize("model", list(EXPECTED))
def test_each_entry_is_pinned(model: str) -> None:
    context, max_out, vision, efforts, default, temperature = EXPECTED[model]
    specs = get_model_specs(PROVIDER, model)

    assert specs["context_length"] == context
    assert specs["max_completion_tokens"] == max_out
    assert specs["input_budget"] == "window_minus_output"
    assert resolve_max_input_tokens(specs) == context - max_out
    assert supports_vision(PROVIDER, model) is vision
    assert thinking_effort_options(PROVIDER, model) == efforts
    assert default_thinking_effort(PROVIDER, model) == default
    assert specs["default_temperature"] == temperature


@pytest.mark.parametrize("model", list(EXPECTED))
def test_curated_entries_agree_with_published_metadata(model: str) -> None:
    published = _published()[model]
    specs = get_model_specs(PROVIDER, model)

    assert specs["context_length"] == published["context_length"]
    assert specs["supports_vision"] == published["supports_vision"]
    published_effort = published.get("thinking_effort")
    curated_effort = specs.get("thinking_effort")
    curated = (curated_effort.options, curated_effort.default) if curated_effort else None
    expected = (published_effort.options, published_effort.default) if published_effort else None
    assert curated == expected


def test_efforts_ride_the_flat_reasoning_effort_field() -> None:
    assert build_thinking_request_params(PROVIDER, "@cf/zai-org/glm-5.3", "low") == {"reasoning_effort": "low"}
    assert build_thinking_request_params(PROVIDER, "@cf/moonshotai/kimi-k2.6", "none") == {"reasoning_effort": "none"}


def test_reasoning_replay_uses_the_server_emitted_field() -> None:
    assert reasoning_replay_field(PROVIDER, "@cf/zai-org/glm-5.3") == "auto"
    assert reasoning_replay_field(PROVIDER, "@cf/moonshotai/kimi-k2.7-code") == "auto"


def test_gemma_4_drops_prior_reasoning_from_requests() -> None:
    from kolega_code.agent.conversation import adapt_history_for_provider
    from kolega_code.llm.models import ContentBlock, Message, TextBlock, ThinkingBlock, ToolCall

    gemma = "@cf/google/gemma-4-26b-a4b-it"
    content: list[ContentBlock] = [
        ThinkingBlock(thinking="plan the call"),
        ToolCall(id="call_1", name="add_small_ints", input={"a": 19, "b": 23}),
    ]
    turn = Message(role="assistant", content=content)
    turn.usage_metadata["provider"] = PROVIDER

    assert reasoning_replay_field(PROVIDER, gemma) is None
    adapted = adapt_history_for_provider(
        [Message(role="user", content=[TextBlock(text="add")]), turn],
        target_provider=PROVIDER,
        target_model=gemma,
        supports_vision=True,
    )
    assert not any(isinstance(block, ThinkingBlock) for block in adapted[1].content)
    assert any(isinstance(block, ToolCall) for block in adapted[1].content)
    kept = adapt_history_for_provider(
        [turn], target_provider=PROVIDER, target_model="@cf/qwen/qwen3.8-27b", supports_vision=True
    )
    assert any(isinstance(block, ThinkingBlock) for block in kept[0].content)


def test_gpt_oss_prefers_codex_apply_patch_and_others_use_the_default() -> None:
    assert preferred_edit_protocol(PROVIDER, "@cf/openai/gpt-oss-120b") == "codex_apply_patch"
    assert preferred_edit_protocol(PROVIDER, "@cf/openai/gpt-oss-20b") == "codex_apply_patch"
    assert preferred_edit_protocol(PROVIDER, "@cf/zai-org/glm-5.3") is None


def test_registry_lists_the_provider_with_glm_53_first_and_default() -> None:
    assert PROVIDER_LABELS[ModelProvider.CLOUDFLARE_WORKERS_AI] == "Cloudflare Workers AI"
    assert ("Cloudflare Workers AI", PROVIDER) in ui_provider_options()
    assert ui_model_options(PROVIDER)[0] == ("GLM-5.3", "@cf/zai-org/glm-5.3")
    assert default_model_for_provider(ModelProvider.CLOUDFLARE_WORKERS_AI) == "@cf/zai-org/glm-5.3"
    option = get_ui_model(PROVIDER, "@cf/zai-org/glm-5.3")
    assert option is not None and option.api_key_env == "CLOUDFLARE_API_TOKEN"


def test_vision_only_options_follow_the_catalog() -> None:
    assert dict(ui_model_options(PROVIDER, vision_only=True)) == {
        "GLM-5.3 Flash": "@cf/zai-org/glm-5.3-flash",
        "Kimi K2.7 Code": "@cf/moonshotai/kimi-k2.7-code",
        "DeepSeek V4.1 Flash": "@cf/deepseek-ai/deepseek-v4.1-flash-1017",
        "Kimi K2.6": "@cf/moonshotai/kimi-k2.6",
        "Qwen 3.8 27B": "@cf/qwen/qwen3.8-27b",
        "Gemma 4 26B A4B": "@cf/google/gemma-4-26b-a4b-it",
    }


def test_every_curated_model_has_a_label() -> None:
    labels = dict((model, label) for label, model in ui_model_options(PROVIDER))
    assert all(labels[model] != model for model in EXPECTED)
