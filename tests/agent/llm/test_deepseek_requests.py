"""DeepSeek output-token cap: the model-keyed wire clamp on every provider.

DeepSeek's real per-response output ceiling (~64k) is far below its published
max_completion_tokens, and the server reports its own cutoff as a clean finish
(silent truncation) while enforcing AND honestly reporting an explicit client
cap — measured live by test_deepseek_output_cap_live.py (2026-08-03). These
tests pin the clamp (specs/accessors.py + the request rules) without touching
the network. Both ``stream`` and ``generate`` share ``_apply_model_request_rules``
("kept in one place so stream and generate cannot diverge"), so the stream-path
transport tests cover both entry points.
"""

import asyncio

import pytest

from kolega_code.llm.models import MessageHistory
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.specs import (
    DEEPSEEK_WIRE_OUTPUT_CAP,
    MODEL_SPECS,
    deepseek_output_token_cap,
    is_deepseek_model,
)

FLASH = "deepseek-v4-flash"
OLLAMA_DEEPSEEK_MODEL = "deepseek-test-small"


class _Recorder:
    """Minimal AsyncOpenAI stand-in that captures the request kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Stream:
            async def __anext__(self):
                raise StopAsyncIteration

            async def aclose(self):
                return None

        return _Stream()

    @property
    def last(self) -> dict:
        assert self.calls, "no request was issued"
        return self.calls[-1]


def _stream_request(monkeypatch, *, provider_name: str, model: str, max_completion_tokens=None) -> dict:
    provider = OpenAIProvider(api_key="sk-test", provider_name=provider_name)
    recorder = _Recorder()
    monkeypatch.setattr(provider, "async_client", recorder)

    async def run():
        await provider.stream(
            MessageHistory([]),
            params=GenerationParams(temperature=1.0, max_completion_tokens=max_completion_tokens),
            model=model,
        )

    asyncio.run(run())
    return recorder.last


# --- the accessor -----------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-v4-pro",  # first-party
        "accounts/fireworks/models/deepseek-v4-flash",  # fireworks path prefix
        "deepseek/deepseek-v4-pro",  # openrouter slug
        "deepseek-test:latest",  # ollama_cloud-style tag
        "DeepSeek-R1",  # case-insensitive
    ],
)
def test_is_deepseek_model_positive(model: str) -> None:
    assert is_deepseek_model(model)


@pytest.mark.parametrize("model", ["gpt-5.5", "qwen3p7-plus", "accounts/fireworks/models/kimi-k2p7-code"])
def test_is_deepseek_model_negative(model: str) -> None:
    assert not is_deepseek_model(model)


def test_deepseek_output_token_cap_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        MODEL_SPECS,
        ("ollama_cloud", OLLAMA_DEEPSEEK_MODEL),
        {"max_completion_tokens": 32768},
    )
    # Oversized request (the catalog's old fiction) clamps to the wire cap.
    assert deepseek_output_token_cap("deepseek", "deepseek-v4-pro", 384000) == DEEPSEEK_WIRE_OUTPUT_CAP
    # A below-cap request passes through untouched.
    assert deepseek_output_token_cap("deepseek", "deepseek-v4-pro", 4096) == 4096
    # No requested cap -> always emit (the wire cap; catalog is higher).
    assert deepseek_output_token_cap("deepseek", "deepseek-v4-pro", None) == DEEPSEEK_WIRE_OUTPUT_CAP
    # A catalog entry BELOW the wire cap wins the min (older/smaller models).
    assert deepseek_output_token_cap("ollama_cloud", OLLAMA_DEEPSEEK_MODEL, None) == 32768
    assert deepseek_output_token_cap("ollama_cloud", OLLAMA_DEEPSEEK_MODEL, 384000) == 32768
    # Unknown overlay model: no catalog entry, wire cap still applies.
    assert deepseek_output_token_cap("deepseek", "deepseek-experimental", None) == DEEPSEEK_WIRE_OUTPUT_CAP
    assert deepseek_output_token_cap("deepseek", "deepseek-experimental", 384000) == DEEPSEEK_WIRE_OUTPUT_CAP
    assert deepseek_output_token_cap("deepseek", "deepseek-experimental", 4096) == 4096


# --- chat path (fireworks, ollama_cloud) -----------------------------------------


@pytest.mark.parametrize(
    "provider_name,model,expected",
    [
        ("fireworks", "accounts/fireworks/models/deepseek-v4-pro", 64000),
        ("fireworks", "accounts/fireworks/models/deepseek-v4-flash", 64000),
    ],
)
def test_chat_stream_sends_clamped_cap_on_wire(monkeypatch, provider_name: str, model: str, expected: int) -> None:
    request = _stream_request(monkeypatch, provider_name=provider_name, model=model)
    assert request["max_tokens"] == expected


def test_chat_stream_uses_smaller_catalog_cap_for_ollama_deepseek_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        MODEL_SPECS,
        ("ollama_cloud", OLLAMA_DEEPSEEK_MODEL),
        {"max_completion_tokens": 32768},
    )

    request = _stream_request(monkeypatch, provider_name="ollama_cloud", model=OLLAMA_DEEPSEEK_MODEL)

    assert request["max_tokens"] == 32768


def test_chat_non_deepseek_model_on_same_provider_unaffected(monkeypatch) -> None:
    request = _stream_request(
        monkeypatch,
        provider_name="fireworks",
        model="accounts/fireworks/models/kimi-k2p7-code",
        max_completion_tokens=4096,
    )
    assert request["max_tokens"] == 4096

    request = _stream_request(monkeypatch, provider_name="fireworks", model="accounts/fireworks/models/kimi-k2p7-code")
    assert "max_tokens" not in request


# --- Responses path (first-party deepseek) ----------------------------------------


def _responses_request(params, model: str = FLASH) -> dict:
    provider = DeepSeekResponsesProvider(api_key="sk-test")
    return provider._build_request(MessageHistory([]), None, params, {"model": model})


def test_responses_build_request_emits_max_output_tokens() -> None:
    request = _responses_request(GenerationParams(max_completion_tokens=4096))
    assert request["parallel_tool_calls"] is True
    assert request["max_output_tokens"] == 4096
    # flash is unclamped: its catalog max_completion_tokens passes through.
    assert _responses_request(GenerationParams(max_completion_tokens=384000))["max_output_tokens"] == 384000
    assert _responses_request(None)["max_output_tokens"] == 384000


def test_responses_build_request_clamps_pro_cap() -> None:
    pro = "deepseek-v4-pro"
    assert _responses_request(None, model=pro)["max_output_tokens"] == 64000
    oversized = _responses_request(GenerationParams(max_completion_tokens=384000), model=pro)
    assert oversized["max_output_tokens"] == 64000
    assert _responses_request(GenerationParams(max_completion_tokens=4096), model=pro)["max_output_tokens"] == 4096


def test_responses_build_request_passes_vision_exp_cap_through() -> None:
    vision = "deepseek-v4-flash-vision-exp"
    # Flash-family: the catalog's real 384K ceiling passes through unclamped.
    assert _responses_request(None, model=vision)["max_output_tokens"] == 384000
    assert (
        _responses_request(GenerationParams(max_completion_tokens=384000), model=vision)["max_output_tokens"] == 384000
    )
    assert _responses_request(GenerationParams(max_completion_tokens=4096), model=vision)["max_output_tokens"] == 4096
