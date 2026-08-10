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


# --- chat path (first-party, fireworks, ollama_cloud) -----------------------------


@pytest.mark.parametrize(
    "provider_name,model,expected",
    [
        ("deepseek", "deepseek-v4-pro", 64000),
        ("fireworks", "accounts/fireworks/models/deepseek-v4-pro", 64000),
        ("fireworks", "accounts/fireworks/models/deepseek-v4-flash", 64000),
    ],
)
def test_chat_stream_sends_clamped_cap_on_wire(monkeypatch, provider_name: str, model: str, expected: int) -> None:
    # No caller-supplied cap: DeepSeek models still always get one (always-emit).
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


def test_chat_oversized_request_is_clamped(monkeypatch) -> None:
    request = _stream_request(
        monkeypatch, provider_name="deepseek", model="deepseek-v4-pro", max_completion_tokens=384000
    )
    assert request["max_tokens"] == 64000


def test_chat_below_cap_request_passes_through(monkeypatch) -> None:
    request = _stream_request(
        monkeypatch, provider_name="deepseek", model="deepseek-v4-pro", max_completion_tokens=4096
    )
    assert request["max_tokens"] == 4096


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


# --- Responses path (first-party flash) -------------------------------------------


def _responses_request(params) -> dict:
    provider = DeepSeekResponsesProvider(api_key="sk-test")
    return provider._build_request(MessageHistory([]), None, params, {"model": FLASH})


def test_responses_build_request_emits_max_output_tokens() -> None:
    # The shared Responses builder omits the cap; the DeepSeek override must add
    # it — without one the server truncates at its own 65536 default.
    assert _responses_request(GenerationParams(max_completion_tokens=4096))["max_output_tokens"] == 4096
    # flash is unclamped: its catalog max_completion_tokens passes through.
    assert _responses_request(GenerationParams(max_completion_tokens=384000))["max_output_tokens"] == 384000
    assert _responses_request(None)["max_output_tokens"] == 384000
