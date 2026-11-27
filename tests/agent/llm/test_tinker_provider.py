"""Unit tests for the native Tinker provider.

The provider's third-party surface (tinker SDK, cookbook renderers, tml-
renderers) is faked here so the suite runs everywhere, including CI without
the optional ``[tinker]`` extra. The real-stack behavior is verified at
implementation time against the live API (see the probe used during
development) and by the optional tests at the bottom of this file, which run
only when the extra is installed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from kolega_code.llm.exceptions import LLMInvalidRequestError, map_tinker_errors as _map_errors
from kolega_code.llm.client import LLMClient
from kolega_code.llm.ledger import HISTORY_ORIGIN, UsageLedger, helper_origin, llm_call_origin
from kolega_code.llm.models import (
    Message,
    MessageChunk,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers import tinker as tinker_provider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.tinker import TinkerProvider, TinkerTraceRecord
from kolega_code.llm.specs import MODEL_SPECS, ThinkingEffortSpec

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

    def get_base_model(self) -> str:
        return self.base_model

    async def sample(self, prompt: Any, num_samples: int, sampling_params: Any, **kwargs: Any) -> FakeSampleResponse:
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

    async def create_sampling_client(self, **kwargs: Any) -> FakeSamplingClient:
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
    if effort and getattr(renderer, "_accepts_effort", False):
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
    assert message.tool_calls == calls
    assert message.stop_reason == "tool_use"

    restored = Message.from_dict(message.to_dict())
    assert [(call.id, call.name, call.input) for call in restored.tool_calls] == [
        ("call_1", "get_weather", {"city": "Paris"}),
        ("call_1", "get_time", {"raw": "not json"}),
    ]


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
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("base_model", ["thinkingmachines/Inkling", "unknown/base", None])
async def test_checkpoint_requests_remain_unpriced_even_with_identified_native_base(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, base_model: str | None
) -> None:
    sampling = FakeSamplingClient(base_model or "", FakeSampleResponse([FakeSequence([1], [0.1])]))
    _patch_provider(monkeypatch, sampling, FakeRenderer({"role": "assistant", "content": "ok"}))
    if base_model is None:

        def unavailable() -> str:
            raise RuntimeError("checkpoint metadata unavailable")

        monkeypatch.setattr(sampling, "get_base_model", unavailable)

    ledger = UsageLedger()
    begun: list[Any] = []
    ledger.observer = SimpleNamespace(
        on_begin=begun.append,
        on_response=lambda *_args: None,
        on_failure=lambda *_args: None,
        on_observation=lambda *_args: None,
    )
    client = LLMClient(provider="tinker", api_key="test-key", model=CHECKPOINT, usage_ledger=ledger)
    assert isinstance(client.provider, TinkerProvider)
    assert client.provider._base_model is None
    for _ in range(2):
        if streaming:
            pending = client.stream(MESSAGES, SYSTEM, model=CHECKPOINT)
            wrapper = await pending if inspect.isawaitable(pending) else pending
            async with wrapper as stream:
                async for _chunk in stream:
                    pass
                await stream.get_final_message()
        else:
            await client.generate(MESSAGES, SYSTEM, model=CHECKPOINT)
    assert len(begun) == len(sampling.calls) == 2
    for index, request in enumerate(begun):
        assert request.model == CHECKPOINT
        assert request.pricing_context.get("base_model") == (base_model if index else None)
        assert not request.pinned_pricing["pricing"]
    snapshot = ledger.snapshot()
    assert snapshot.complete
    assert snapshot.requests == snapshot.responses == 2
    assert (snapshot.input_tokens, snapshot.output_tokens, snapshot.total_tokens) == (10, 2, 12)
    cost = snapshot.costs
    assert not cost["complete"]
    assert cost["currencies"] == {}


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
async def test_stream_final_message_exposes_tool_calls_and_fabricates_missing_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "id": None,
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }
    renderer = FakeRenderer(parsed, supports_streaming=False)
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1, 2], [0.1] * 2)]))
    _patch_provider(monkeypatch, client, renderer)

    stream_cm = await _make_provider().stream(MESSAGES, SYSTEM, GenerationParams(), model=MODEL)
    async with stream_cm as stream:
        chunks = [chunk async for chunk in stream]
    final = await stream.get_final_message()

    content_calls = [block for block in final.content if isinstance(block, ToolCall)]
    assert final.tool_calls == content_calls
    assert [(call.id, call.name, call.input) for call in final.tool_calls] == [
        ("call_0", "get_weather", {"city": "Paris"})
    ]
    assert final.stop_reason == "tool_use"
    assert [chunk.type for chunk in chunks] == ["tool_use_start", "tool_use_delta"]
    assert chunks[0].tool_call_delta == {
        "id": "call_0",
        "name": "get_weather",
        "input": "",
        "execution_id": final.tool_calls[0].execution_id,
    }


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
async def test_streaming_trace_record_carries_origin_active_at_context_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling and trace emission happen in __aenter__, so the origin that
    matters is the one active when the stream context is entered — not the one
    active at the stream() call."""
    records: List[TinkerTraceRecord] = []
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([7, 8], [-0.4, -0.5])]))
    _patch_provider(monkeypatch, client, FakeRenderer({"role": "assistant", "content": "ok"}))
    provider = _make_provider(trace_sink=records.append)

    stream_cm = await provider.stream(MESSAGES, SYSTEM, GenerationParams(), model=MODEL)
    with llm_call_origin(HISTORY_ORIGIN):
        async with stream_cm as stream:
            async for _chunk in stream:
                pass

    assert len(records) == 1
    assert records[0].request_role == {"kind": "history"}


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
async def test_effort_none_sends_zero_to_effort_aware_renderers(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1], [0.1])]))
    renderer = FakeRenderer({"role": "assistant", "content": "ok"}, accepts_effort=True)
    _patch_provider(monkeypatch, client, renderer)

    await _make_provider().generate(MESSAGES, SYSTEM, GenerationParams(thinking="none"), model=MODEL)

    assert renderer.prompt_calls[0]["kwargs"] == {"effort": 0.0}


def _multi_turn_thinking_history(assistant_provider: str = "tinker") -> MessageHistory:
    return MessageHistory(
        [
            Message(role="user", content=[TextBlock(text="Read a.py")]),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Need to inspect a.py."),
                    ToolCall(id="call_0", name="read", input={"file_path": "a.py"}),
                ],
                stop_reason="tool_use",
                usage_metadata={"provider": assistant_provider},
            ),
            Message(
                role="user",
                content=[ToolResult(tool_use_id="call_0", name="read", content="print(1)", is_error=False)],
            ),
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("request_model", "base_model"),
    [
        ("test-org/Tinker-Thinking-Model", "test-org/Tinker-Thinking-Model"),
        (CHECKPOINT, "test-org/Tinker-Thinking-Model"),
        (CHECKPOINT, "test-org/Uncatalogued-Checkpoint-Base"),
        ("test-org/Uncatalogued-Direct-Base", "test-org/Uncatalogued-Direct-Base"),
    ],
)
async def test_multi_turn_replays_same_provider_thinking_as_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    request_model: str,
    base_model: str,
) -> None:
    monkeypatch.setitem(
        MODEL_SPECS,
        ("tinker", "test-org/Tinker-Thinking-Model"),
        {
            "context_length": 65536,
            "max_completion_tokens": 32768,
            "input_budget": "window_minus_output",
            "default_temperature": 1.0,
            "thinking_effort": ThinkingEffortSpec(
                options=("none", "high"),
                default="high",
                mode="tinker_native_effort",
            ),
        },
    )
    client = FakeSamplingClient(base_model, FakeSampleResponse([FakeSequence([1], [0.1])]))
    renderer = FakeRenderer({"role": "assistant", "content": "Done."})
    _patch_provider(monkeypatch, client, renderer)

    provider = TinkerProvider(api_key="test-key", model=request_model)
    history = _multi_turn_thinking_history(assistant_provider="tinker")
    if streaming:
        stream_cm = await provider.stream(history, SYSTEM, GenerationParams(), model=request_model)
        async with stream_cm as stream:
            async for _chunk in stream:
                pass
    else:
        await provider.generate(history, SYSTEM, GenerationParams(), model=request_model)

    sent_messages = renderer.prompt_calls[0]["messages"]
    assistant_msg = sent_messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["reasoning_content"] == "Need to inspect a.py."
    assert assistant_msg["content"] == ""
    assert "*Thinking:*" not in str(assistant_msg["content"])
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "read"
    assert sent_messages[2]["role"] == "tool"


@pytest.mark.asyncio
async def test_count_tokens_replays_same_provider_thinking_as_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1], [0.1])]))
    renderer = FakeRenderer({"role": "assistant", "content": "Done."})
    _patch_provider(monkeypatch, client, renderer)

    provider = TinkerProvider(api_key="test-key", model=CHECKPOINT)
    await provider.count_tokens(_multi_turn_thinking_history(), SYSTEM, model=CHECKPOINT)

    assistant_msg = renderer.prompt_calls[0]["messages"][1]
    assert assistant_msg["reasoning_content"] == "Need to inspect a.py."
    assert assistant_msg["content"] == ""


@pytest.mark.asyncio
async def test_multi_turn_does_not_replay_foreign_provider_thinking_natively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeSamplingClient(MODEL, FakeSampleResponse([FakeSequence([1], [0.1])]))
    renderer = FakeRenderer({"role": "assistant", "content": "Done."})
    _patch_provider(monkeypatch, client, renderer)

    provider = _make_provider()
    await provider.generate(_multi_turn_thinking_history(assistant_provider="anthropic"), SYSTEM, GenerationParams())

    assistant_msg = renderer.prompt_calls[0]["messages"][1]
    assert "reasoning_content" not in assistant_msg


# ---------------------------------------------------------------------------
# Real-stack tests (run only where the optional extra is installed)
# ---------------------------------------------------------------------------

needs_native_stack = pytest.mark.skipif(
    not tinker_provider._NATIVE_STACK_AVAILABLE,
    reason="optional [tinker] extra not installed",
)


@needs_native_stack
@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", ":peft:262144"])
async def test_inkling_2_mini_renders_loop_prompt_on_exact_sampling_route(
    monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    model = f"thinkingmachines/Inkling-2-Mini{suffix}"
    service = FakeService(FakeSamplingClient(model, FakeSampleResponse([])))
    monkeypatch.setattr(tinker_provider, "_tinker", SimpleNamespace(ServiceClient=service))
    provider = TinkerProvider(api_key="test-key", model=model)

    await provider.count_tokens(MESSAGES, SYSTEM, thinking="none")
    renderer = await provider._get_renderer(model, "none")
    prompt = tinker_provider._build_prompt(
        renderer, [{"role": "user", "content": "Hello"}], system_text="", tools=[], effort="none"
    )

    assert service.created == [{"base_model": model}]
    assert "<|message_user|><|content_text|>Hello<|end_message|>" in renderer.tokenizer.decode(prompt.to_ints())


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


@needs_native_stack
def test_deepseek_v4_1_renderer_preserves_thinking_and_prefix_extension_with_tools() -> None:
    """Verify real DeepSeekV4_1Renderer preserves ThinkingBlock reasoning across
    tool-call turns and maintains the token prefix-extension property."""
    from tinker_cookbook.renderers.deepseek_v4_1 import DeepSeekV4_1Renderer  # pyright: ignore[reportMissingImports]

    special_tokens = [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<think>",
        "</think>",
    ]
    special_to_id = {tok: idx for idx, tok in enumerate(special_tokens)}
    id_to_special = {idx: tok for tok, idx in special_to_id.items()}
    byte_offset = len(special_tokens)

    class OfflineByteTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            _ = add_special_tokens
            tokens: List[int] = []
            pos = 0
            while pos < len(text):
                matched = next((tok for tok in special_tokens if text.startswith(tok, pos)), None)
                if matched is not None:
                    tokens.append(special_to_id[matched])
                    pos += len(matched)
                else:
                    tokens.extend(byte_offset + b for b in text[pos].encode("utf-8"))
                    pos += 1
            return tokens

        def decode(self, tokens: Sequence[int]) -> str:
            out: List[str] = []
            buf = bytearray()
            for tok in tokens:
                if tok in id_to_special:
                    if buf:
                        out.append(buf.decode("utf-8", errors="replace"))
                        buf.clear()
                    out.append(id_to_special[tok])
                else:
                    buf.append(tok - byte_offset)
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
            return "".join(out)

    tokenizer = OfflineByteTokenizer()
    renderer = DeepSeekV4_1Renderer(tokenizer)  # type: ignore[arg-type]
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get the weather",
            parameters=[ToolParameter(name="city", type="string", description="City", required=True)],
        )
    ]

    turn1_history = MessageHistory([Message(role="user", content=[TextBlock(text="Weather in Paris?")])])
    turn1_openai, system_text = tinker_provider._openai_messages(
        turn1_history, SYSTEM, provider="tinker", model=CHECKPOINT
    )
    turn1_prompt = tinker_provider._build_prompt(renderer, turn1_openai, system_text, tools, effort=None)
    turn1_tokens = list(turn1_prompt.to_ints())

    sampled_body = (
        "Let me check the weather first.</think>\n\n"
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="get_weather">\n'
        '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>"
    )
    sampled_tokens = tokenizer.encode(sampled_body, add_special_tokens=False) + [special_to_id["<｜end▁of▁sentence｜>"]]
    parsed, termination = renderer.parse_response(sampled_tokens)
    assistant_msg = tinker_provider._parsed_to_message(parsed, tinker_provider._kolega_stop_reason("stop", termination))
    assistant_msg.usage_metadata = {"provider": "tinker"}

    assert any(
        isinstance(b, ThinkingBlock) and b.thinking == "Let me check the weather first." for b in assistant_msg.content
    )
    assert assistant_msg.tool_calls and assistant_msg.tool_calls[0].name == "get_weather"

    turn2_history = MessageHistory(
        [
            Message(role="user", content=[TextBlock(text="Weather in Paris?")]),
            assistant_msg,
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id=assistant_msg.tool_calls[0].id,
                        name="get_weather",
                        content="Sunny, 20C",
                        is_error=False,
                    )
                ],
            ),
        ]
    )
    turn2_openai, turn2_system = tinker_provider._openai_messages(
        turn2_history, SYSTEM, provider="tinker", model=CHECKPOINT
    )
    turn2_prompt = tinker_provider._build_prompt(renderer, turn2_openai, turn2_system, tools, effort=None)
    turn2_tokens = list(turn2_prompt.to_ints())
    decoded_turn2 = tokenizer.decode(turn2_tokens)

    assert "<think>Let me check the weather first.</think>" in decoded_turn2
    assert "*Thinking:*" not in decoded_turn2
    assert "<think></think>" not in decoded_turn2
    assert turn2_tokens[: len(turn1_tokens) + len(sampled_tokens)] == turn1_tokens + sampled_tokens


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
