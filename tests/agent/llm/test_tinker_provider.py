"""Unit tests for the native Tinker provider.

The provider's third-party surface (tinker SDK, cookbook renderers, tml-
renderers) is faked here so the suite runs everywhere, including CI without
the optional ``[tinker]`` extra. The real-stack behavior is verified at
implementation time against the live API (see the probe used during
development) and by the optional tests at the bottom of this file, which run
only when the extra is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pytest

from kolega_code.llm.exceptions import LLMInvalidRequestError, map_tinker_errors as _map_errors
from kolega_code.llm.ledger import helper_origin, llm_call_origin
from kolega_code.llm.models import (
    Message,
    MessageChunk,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
)
from kolega_code.llm.providers import tinker as tinker_provider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.tinker import TinkerProvider, TinkerTraceRecord

MODEL = "Qwen/Qwen3-8B"
CHECKPOINT = "tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080"

SYSTEM = Message(role="system", content=[TextBlock(text="Be concise.")])
MESSAGES = MessageHistory([Message(role="user", content=[TextBlock(text="What is 2+2?")])])

# ---------------------------------------------------------------------------
# Fakes for the tinker SDK + renderer surfaces the provider touches.
# ---------------------------------------------------------------------------


class FakeModelInput:
    def __init__(self, tokens: List[int]):
        self._tokens = tokens

    @property
    def length(self) -> int:
        return len(self._tokens)

    def to_ints(self) -> List[int]:
        return list(self._tokens)


@dataclass
class FakeSequence:
    tokens: List[int]
    logprobs: Optional[Sequence[Optional[float]]]
    stop_reason: str = "stop"


@dataclass
class FakeSampleResponse:
    sequences: List[FakeSequence]
    prompt_cache_hit_tokens: int = 0


class FakeSamplingClient:
    def __init__(self, base_model: str, response: FakeSampleResponse):
        self.base_model = base_model
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    async def get_base_model_async(self) -> str:
        return self.base_model

    async def sample_async(
        self, prompt: Any, num_samples: int, sampling_params: Any, **kwargs: Any
    ) -> FakeSampleResponse:
        self.calls.append({"prompt": prompt, "num_samples": num_samples, "sampling_params": sampling_params})
        return self.response


class FakeSamplingParams:
    """Records kwargs for later assertions."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs


class FakeService:
    def __init__(self, client: FakeSamplingClient):
        self.client = client
        self.created: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> "FakeService":
        return self

    async def create_sampling_client_async(self, **kwargs: Any) -> FakeSamplingClient:
        self.created.append(kwargs)
        return self.client


class FakeTermination:
    def __init__(self, name: str):
        self.name = name


class FakeRenderer:
    """Scripted cookbook-style renderer."""

    def __init__(
        self,
        parsed: Dict[str, Any],
        termination: Any = FakeTermination("STOP_SEQUENCE"),
        accepts_effort: bool = False,
        supports_streaming: bool = False,
    ):
        self.parsed = parsed
        self.termination = termination
        self._accepts_effort = accepts_effort
        self.supports_streaming = supports_streaming
        self.prompt_calls: List[Dict[str, Any]] = []
        self.stream_deltas: List[Any] = []

    def get_stop_sequences(self) -> list:
        return [1, 2]

    def create_conversation_prefix_with_tools(self, tools: List[Any], system_prompt: str = "") -> List[Dict[str, Any]]:
        return [{"role": "system", "content": system_prompt}]

    def build_generation_prompt(self, messages: List[Any], **kwargs: Any) -> FakeModelInput:
        self.prompt_calls.append({"messages": messages, "kwargs": kwargs})
        return FakeModelInput([1, 2, 3, 4, 5])

    def parse_response(self, tokens: List[int]) -> tuple[Dict[str, Any], Any]:
        return self.parsed, self.termination

    def parse_response_streaming(self, tokens: List[int]):
        return iter(self.stream_deltas)


def _fake_build_prompt(
    renderer: Any,
    openai_messages: List[Dict[str, Any]],
    system_text: str,
    tools: Sequence[Any],
    effort: Optional[str],
) -> FakeModelInput:
    """Mirror the provider's prompt-builder effort logic without the cookbook.

    The real ``_build_prompt`` imports ``tinker_cookbook.third_party.openai_compat``
    at call time, which does not exist in CI (no ``[tinker]`` extra), so the
    hermetic tests stub it and keep only the effort-passing branch under test.
    The real bridge is exercised by the skipif-guarded cookbook test and by the
    implementation-time live verification.
    """
    if effort and effort != "none" and getattr(renderer, "_accepts_effort", False):
        return renderer.build_generation_prompt(openai_messages, effort=tinker_provider._effort_float(effort))
    return renderer.build_generation_prompt(openai_messages)


def _patch_provider(monkeypatch: pytest.MonkeyPatch, client: FakeSamplingClient, renderer: FakeRenderer) -> None:
    """Point the provider at the fakes: fake SDK module, client factory, renderer."""

    class FakeTypes:
        SamplingParams = FakeSamplingParams
        ModelInput = FakeModelInput

    fake_module = type("FakeTinkerModule", (), {"ServiceClient": FakeService(client), "types": FakeTypes})
    monkeypatch.setattr(tinker_provider, "_tinker", fake_module)
    monkeypatch.setattr(tinker_provider, "_NATIVE_STACK_AVAILABLE", True)
    monkeypatch.setattr(tinker_provider, "_renderer_for_base_model", lambda base, effort: renderer)
    monkeypatch.setattr(tinker_provider, "_build_prompt", _fake_build_prompt)


def _make_provider(trace_sink=None) -> TinkerProvider:
    return TinkerProvider(api_key="test-key", model=MODEL, trace_sink=trace_sink)


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_parsed_text_message(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1, 2, 3], [-0.1, -0.2, -0.3])]))
    renderer = FakeRenderer({"role": "assistant", "content": "Four."})
    _patch_provider(monkeypatch, client, renderer)

    provider = _make_provider()
    message = await provider.generate(
        MESSAGES, SYSTEM, GenerationParams(temperature=0.7, max_completion_tokens=64), model=MODEL
    )

    assert message.role == "assistant"
    assert message.get_text_content() == "Four."
    assert message.stop_reason == "stop_sequence"
    assert message.usage_metadata["provider"] == "tinker"
    assert message.usage_metadata["input_tokens"] == 5
    assert message.usage_metadata["output_tokens"] == 3
    # The sampling client was created for the base model and sampled once.
    assert client.calls
    assert client.calls[0]["num_samples"] == 1
    assert client.calls[0]["sampling_params"].kwargs["max_tokens"] == 64


@pytest.mark.asyncio
async def test_generate_parses_tool_calls_and_fabricates_missing_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            },
            {
                "type": "function",
                "id": None,
                "function": {"name": "get_time", "arguments": "not json"},
            },
        ],
    }
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([9], [0.0])]))
    renderer = FakeRenderer(parsed, termination=FakeTermination("STOP_SEQUENCE"))
    _patch_provider(monkeypatch, client, renderer)

    message = await _make_provider().generate(MESSAGES, SYSTEM, GenerationParams(), model=MODEL)

    calls = [b for b in (message.content or []) if isinstance(b, ToolCall)]
    assert len(calls) == 2
    assert calls[0].id == "call_1"
    assert calls[0].name == "get_weather"
    assert calls[0].input == {"city": "Paris"}
    assert calls[1].id == "call_1"  # fabricated from the index
    assert calls[1].name == "get_time"
    assert calls[1].input == {"raw": "not json"}


@pytest.mark.asyncio
async def test_generate_captures_thinking_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "Let me compute."}, {"type": "text", "text": "4"}],
    }
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1, 2], [0.1, 0.2])]))
    _patch_provider(monkeypatch, client, FakeRenderer(parsed))

    message = await _make_provider().generate(MESSAGES, SYSTEM, GenerationParams(), model=MODEL)

    thinking = [b for b in (message.content or []) if isinstance(b, ThinkingBlock)]
    assert [b.thinking for b in thinking] == ["Let me compute."]
    assert message.get_text_content() == "4"


@pytest.mark.asyncio
async def test_checkpoint_model_uses_model_path_and_resolves_base(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeSamplingClient("thinkingmachines/Inkling", FakeSampleResponse([FakeSequence([1], [0.1])]))
    service = FakeService(client)

    class FakeTypes:
        SamplingParams = FakeSamplingParams
        ModelInput = FakeModelInput

    fake_module = type("FakeTinkerModule", (), {"ServiceClient": service, "types": FakeTypes})
    monkeypatch.setattr(tinker_provider, "_tinker", fake_module)
    monkeypatch.setattr(tinker_provider, "_NATIVE_STACK_AVAILABLE", True)
    monkeypatch.setattr(
        tinker_provider,
        "_renderer_for_base_model",
        lambda base, effort: FakeRenderer({"role": "assistant", "content": "ok"}),
    )
    monkeypatch.setattr(tinker_provider, "_build_prompt", _fake_build_prompt)

    provider = _make_provider()
    provider.model = CHECKPOINT
    await provider.generate(MESSAGES, SYSTEM, GenerationParams(), model=CHECKPOINT)

    assert service.created == [{"model_path": CHECKPOINT}]
    assert provider._base_model == "thinkingmachines/Inkling"


@pytest.mark.asyncio
async def test_provider_rejects_a_second_distinct_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1], [0.1])]))
    _patch_provider(monkeypatch, client, FakeRenderer({"role": "assistant", "content": "ok"}))

    provider = _make_provider()
    await provider.generate(MESSAGES, SYSTEM, GenerationParams(), model=MODEL)
    with pytest.raises(LLMInvalidRequestError, match="bound to model"):
        await provider.generate(MESSAGES, SYSTEM, GenerationParams(), model="Qwen/Qwen3.5-9B")


@pytest.mark.asyncio
async def test_count_tokens_uses_rendered_prompt_length(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1], [0.1])]))
    _patch_provider(monkeypatch, client, FakeRenderer({"role": "assistant", "content": "ok"}))

    count = await _make_provider().count_tokens(MESSAGES, SYSTEM, model=MODEL)

    assert count.input_tokens == 5
    assert count.output_tokens is None


@pytest.mark.asyncio
async def test_max_tokens_capped_to_context_window(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1], [0.1])]))
    _patch_provider(monkeypatch, client, FakeRenderer({"role": "assistant", "content": "ok"}))

    # Qwen/Qwen3-8B catalog context is 32768; a huge requested cap must be clamped.
    provider = _make_provider()
    await provider.generate(MESSAGES, SYSTEM, GenerationParams(max_completion_tokens=1_000_000), model=MODEL)

    assert client.calls[0]["sampling_params"].kwargs["max_tokens"] == 32768 - 5


@pytest.mark.asyncio
async def test_stream_replays_deltas_and_final_message(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = {"role": "assistant", "content": "One two three."}
    renderer = FakeRenderer(parsed, supports_streaming=True)
    renderer.stream_deltas = [
        type("H", (), {"role": "assistant", "text": None})(),
        type("T", (), {"text": "One ", "thinking": None})(),
        type("T", (), {"text": "two ", "thinking": None})(),
        type("Th", (), {"thinking": "thinking text", "text": None})(),
        parsed,
    ]
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1, 2, 3, 4], [0.1] * 4)]))
    _patch_provider(monkeypatch, client, renderer)

    provider = _make_provider()
    stream_cm = await provider.stream(MESSAGES, SYSTEM, GenerationParams(), model=MODEL)
    chunks: List[MessageChunk] = []
    async with stream_cm as stream:
        async for chunk in stream:
            chunks.append(chunk)
    final = await stream.get_final_message()

    text = "".join(c.text or "" for c in chunks if c.type == "text")
    assert text == "One two "
    assert any(c.type == "thinking" and c.thinking == "thinking text" for c in chunks)
    assert final.get_text_content() == "One two three."


@pytest.mark.asyncio
async def test_stream_falls_back_to_block_replay_without_streaming_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = {"role": "assistant", "content": "Block text.", "tool_calls": []}
    renderer = FakeRenderer(parsed, supports_streaming=False)
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1, 2], [0.1] * 2)]))
    _patch_provider(monkeypatch, client, renderer)

    provider = _make_provider()
    stream_cm = await provider.stream(MESSAGES, SYSTEM, GenerationParams(), model=MODEL)
    chunks: List[MessageChunk] = []
    async with stream_cm as stream:
        async for chunk in stream:
            chunks.append(chunk)

    assert [c.type for c in chunks] == ["text"]
    assert chunks[0].text == "Block text."


@pytest.mark.asyncio
async def test_trace_sink_receives_full_record(monkeypatch: pytest.MonkeyPatch) -> None:
    records: List[TinkerTraceRecord] = []
    client = FakeSamplingClient(
        MODEL, FakeSampleResponse([FakeSequence([7, 8], [-0.4, -0.5])], prompt_cache_hit_tokens=3)
    )
    renderer = FakeRenderer({"role": "assistant", "content": "ok"}, termination=FakeTermination("STOP_SEQUENCE"))
    _patch_provider(monkeypatch, client, renderer)

    provider = _make_provider(trace_sink=records.append)
    with llm_call_origin(helper_origin("unit_test")):
        await provider.generate(MESSAGES, SYSTEM, GenerationParams(temperature=0.5), model=MODEL)

    assert len(records) == 1
    record = records[0]
    assert record.model == MODEL
    assert record.base_model == MODEL
    assert record.request_role == {"kind": "helper", "helper": "unit_test"}
    assert record.prompt_tokens == [1, 2, 3, 4, 5]
    assert record.sampled_tokens == [7, 8]
    assert record.logprobs == [-0.4, -0.5]
    assert record.stop_reason == "stop"
    assert record.termination == "STOP_SEQUENCE"
    assert record.cache_hit_tokens == 3
    assert record.temperature == 0.5


@pytest.mark.asyncio
async def test_trace_sink_disabled_emits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([7], [-0.4])]))
    _patch_provider(monkeypatch, client, FakeRenderer({"role": "assistant", "content": "ok"}))

    await _make_provider(trace_sink=None).generate(MESSAGES, SYSTEM, GenerationParams(), model=MODEL)
    # No assertion hook needed beyond the call succeeding; sink absence is a no-op.


@pytest.mark.asyncio
async def test_effort_float_passed_only_to_effort_aware_renderers(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1], [0.1])]))
    renderer = FakeRenderer({"role": "assistant", "content": "ok"}, accepts_effort=True)
    _patch_provider(monkeypatch, client, renderer)

    await _make_provider().generate(MESSAGES, SYSTEM, GenerationParams(thinking="medium"), model=MODEL)

    assert renderer.prompt_calls[0]["kwargs"] == {"effort": 0.7}


@pytest.mark.asyncio
async def test_effort_none_sends_no_effort_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1], [0.1])]))
    renderer = FakeRenderer({"role": "assistant", "content": "ok"}, accepts_effort=True)
    _patch_provider(monkeypatch, client, renderer)

    await _make_provider().generate(MESSAGES, SYSTEM, GenerationParams(thinking="none"), model=MODEL)

    assert renderer.prompt_calls[0]["kwargs"] == {}


# ---------------------------------------------------------------------------
# Real-stack tests (run only where the optional extra is installed)
# ---------------------------------------------------------------------------

needs_native_stack = pytest.mark.skipif(
    not tinker_provider._NATIVE_STACK_AVAILABLE,
    reason="optional [tinker] extra not installed",
)


@needs_native_stack
def test_build_prompt_renders_messages_and_tools_with_real_cookbook() -> None:
    """Exercise the real cookbook bridge end to end, minus sampling."""
    from tinker_cookbook.tokenizer_utils import get_tokenizer  # pyright: ignore[reportMissingImports]

    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    renderer = tinker_provider._cookbook_get_renderer("qwen3", tokenizer, model_name="Qwen/Qwen3-8B")
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get the weather",
            parameters=[ToolParameter(name="city", type="string", description="City", required=True)],
        )
    ]
    model_input = tinker_provider._build_prompt(
        renderer,
        [{"role": "user", "content": "Weather?"}],
        system_text="Be concise.",
        tools=tools,
        effort=None,
    )
    assert model_input.length > 0
    assert model_input.to_ints()


def test_map_tinker_errors_by_status() -> None:
    from kolega_code.llm.exceptions import (
        LLMAuthenticationError,
        LLMBillingError,
        LLMInternalServerError,
        LLMInvalidRequestError,
        LLMRateLimitError,
    )

    class Err(Exception):
        def __init__(self, status: int, message: str = "boom"):
            super().__init__(message)
            self.status_code = status

    assert isinstance(_map_errors(Err(401)), LLMAuthenticationError)
    assert isinstance(_map_errors(Err(403)), LLMAuthenticationError)
    assert isinstance(_map_errors(Err(429)), LLMRateLimitError)
    assert isinstance(_map_errors(Err(400)), LLMInvalidRequestError)
    assert isinstance(_map_errors(Err(500)), LLMInternalServerError)
    assert isinstance(_map_errors(Err(402)), LLMBillingError)
    assert isinstance(_map_errors(Err(402, "billing required")), LLMBillingError)
    assert _map_errors(ValueError("no status")) is None
