"""Tests for the api-key OpenAI Responses provider and its routing.

gpt-5.x reject ``function tools + reasoning_effort`` on Chat Completions, so the
api-key ``openai`` provider routes to the Responses API. The OpenAI-compatible
providers (fireworks, xai, …) keep using the Chat Completions ``OpenAIProvider``.
"""

import types
from typing import Any

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    Message,
    MessageHistory,
    ResponsesReasoningBlock,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolInputKind,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.providers.openai_responses import OpenAIResponsesProvider
from kolega_code.llm.providers.responses_common import to_responses_input


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _mk_call(cid, kind: ToolInputKind = "json"):
    inp = "{}" if kind == "freeform" else {"cmd": "x"}
    return ToolCall(id=cid, name="run", input=inp, execution_id=cid, input_kind=kind)


def _mk_result(cid, kind: ToolInputKind = "json"):
    return ToolResult(tool_use_id=cid, content="ok", name="run", is_error=False, input_kind=kind)


class TestOrphanedToolCallPadding:
    """Every function_call must be answered or the Responses API 400s with
    'No tool output found for tool call ...'. Parallel tool calls can leave one
    unanswered (e.g. a yielded command); to_responses_input pads the orphan."""

    def test_orphaned_parallel_call_gets_placeholder_output(self):
        ids = ["call_00_a", "call_01_b", "call_02_c", "call_03_d"]
        history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="inspect")]),
                Message(role="assistant", content=[_mk_call(i) for i in ids]),
                Message(role="user", content=[_mk_result(i) for i in ids[:3]]),  # call_03_d orphaned
            ]
        )
        items = to_responses_input(history)
        calls = {it["call_id"] for it in items if it.get("type") == "function_call"}
        outputs = {it["call_id"] for it in items if it.get("type") == "function_call_output"}
        assert calls == set(ids)
        assert calls == outputs  # every call answered, orphan padded
        placeholder = [it for it in items if it.get("type") == "function_call_output" and it["call_id"] == "call_03_d"]
        assert len(placeholder) == 1 and placeholder[0]["output"]

    def test_fully_answered_calls_are_not_double_padded(self):
        ids = ["call_00_a", "call_01_b"]
        history = MessageHistory(
            [
                Message(role="assistant", content=[_mk_call(i) for i in ids]),
                Message(role="user", content=[_mk_result(i) for i in ids]),
            ]
        )
        outputs = [it for it in to_responses_input(history) if it.get("type") == "function_call_output"]
        assert [o["call_id"] for o in outputs] == ids  # exactly one each, no extras

    def test_orphaned_freeform_call_gets_custom_placeholder(self):
        history = MessageHistory([Message(role="assistant", content=[_mk_call("call_00_ff", kind="freeform")])])
        items = to_responses_input(history)
        pad = [it for it in items if it.get("type") == "custom_tool_call_output"]
        assert len(pad) == 1 and pad[0]["call_id"] == "call_00_ff"


class TestAssistantTextToolCallOrdering:
    """When an assistant turn has both text and tool calls, the text must be
    emitted BEFORE the function_calls so the next message's outputs stay adjacent
    to the calls. An interposed assistant message makes DeepSeek's Responses API
    400 ("No tool output found for tool call ...")."""

    def test_assistant_text_precedes_tool_calls_and_outputs_are_adjacent(self):
        ids = ["call_00_a", "call_01_b"]
        history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="inspect the repo")]),
                # The model emitted a preamble sentence AND two tool calls.
                Message(
                    role="assistant",
                    content=[TextBlock(text="I'll check a couple things."), *[_mk_call(i) for i in ids]],
                ),
                Message(role="user", content=[_mk_result(i) for i in ids]),
            ]
        )
        types = [it.get("type") or it.get("role") for it in to_responses_input(history)]
        # assistant text must come before the first function_call
        assert types.index("assistant") < types.index("function_call")
        # and NO assistant/user message may sit between the calls and their outputs
        first_call = types.index("function_call")
        first_output = types.index("function_call_output")
        between = types[first_call:first_output]
        assert set(between) == {"function_call"}, f"non-call item interposed: {between}"


class _FakeStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


class _FakeResponses:
    def __init__(self, result):
        self._result = result
        self.last_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._result


# --- routing --------------------------------------------------------------------


def test_llmclient_routes_openai_to_responses_provider():
    client = LLMClient(provider="openai", api_key="sk-test")
    assert isinstance(client.provider, OpenAIResponsesProvider)
    assert client.provider.provider_name == "openai"


def test_llmclient_routes_compatible_provider_to_chat_completions():
    client = LLMClient(provider="fireworks", api_key="sk-test")
    assert isinstance(client.provider, OpenAIProvider)
    assert not isinstance(client.provider, OpenAIResponsesProvider)


# --- request building (no network) ----------------------------------------------


@pytest.mark.parametrize(
    "model,effort",
    [("gpt-5.5", "high"), *[("gpt-6-astra", effort) for effort in ("low", "medium", "high", "xhigh", "max")]],
)
def test_build_request_is_responses_shaped_with_reasoning(model: str, effort: str) -> None:
    provider = OpenAIResponsesProvider(api_key="sk-test")
    tool = ToolDefinition(
        name="read_file",
        description="Read a file",
        parameters=[ToolParameter(name="path", type="string", description="path", required=True)],
    )
    params = GenerationParams(tools=[tool], thinking=effort, max_completion_tokens=256, temperature=0.5)
    request = provider._build_request(
        MessageHistory([Message(role="user", content=[TextBlock(text="hello")])]),
        Message(role="system", content=[TextBlock(text="sys")]),
        params,
        {"model": model},
    )

    assert request["model"] == model
    assert request["stream"] is True
    assert request["store"] is False
    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is True
    assert request["instructions"] == "sys"
    assert request["reasoning"] == {"effort": effort, "summary": "auto"}
    assert request["include"] == ["reasoning.encrypted_content"]
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["name"] == "read_file"
    assert request["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
    # The Responses path must NOT send Chat-Completions-only fields. The absent
    # max_output_tokens also guards the non-DeepSeek builders: only
    # DeepSeekResponsesProvider overrides _build_request to add a clamped cap.
    assert "temperature" not in request
    assert "max_completion_tokens" not in request
    assert "max_output_tokens" not in request


def test_build_request_default_model_and_no_reasoning_without_thinking():
    provider = OpenAIResponsesProvider(api_key="sk-test")
    request = provider._build_request(
        MessageHistory([Message(role="user", content=[TextBlock(text="hi")])]),
        None,
        GenerationParams(),
        {},
    )
    assert request["model"] == "gpt-5.6-sol"
    assert "reasoning" not in request
    assert "include" not in request


def test_build_request_escapes_harmony_tokens_for_openai_provider():
    # gpt-5.x (Harmony-dialect) backends reject raw control-token spellings in
    # input with "Request blocked"; the api-key openai provider escapes too.
    provider = OpenAIResponsesProvider(api_key="sk-test")
    history = MessageHistory([Message(role="user", content=[TextBlock(text="see <|channel|>")])])
    request = provider._build_request(history, None, GenerationParams(), {"model": "gpt-5.5"})
    assert request["input"][0]["content"][0]["text"] == "see <\\|channel\\|>"


def test_build_request_does_not_escape_harmony_tokens_for_deepseek():
    # DeepSeek is not a Harmony backend: the escape must stay off so payloads
    # keep their byte-for-byte spelling.
    provider = DeepSeekResponsesProvider(api_key="sk-test")
    history = MessageHistory([Message(role="user", content=[TextBlock(text="see <|channel|>")])])
    request = provider._build_request(history, None, GenerationParams(), {"model": "deepseek-v4-flash"})
    assert request["input"][0]["content"][0]["text"] == "see <|channel|>"


# --- generate / stream (no network) ---------------------------------------------


@pytest.mark.asyncio
async def test_generate_tags_openai_provider_and_sends_include(monkeypatch):
    provider = OpenAIResponsesProvider(api_key="sk-test")
    completed = _ns(
        output=[_ns(type="message", content=[_ns(type="output_text", text="hi")])],
        usage=_ns(input_tokens=3, output_tokens=2, total_tokens=5, input_tokens_details=None),
        status="completed",
        incomplete_details=None,
    )
    fake = _FakeResponses(_FakeStream([_ns(type="response.completed", response=completed)]))
    monkeypatch.setattr(provider, "async_client", _ns(responses=fake))

    msg = await provider.generate(
        MessageHistory([Message(role="user", content=[TextBlock(text="hello")])]),
        params=GenerationParams(thinking="medium"),
        model="gpt-5.5",
    )

    assert msg.get_text_content() == "hi"
    assert msg.usage_metadata["provider"] == "openai"
    assert fake.last_kwargs["include"] == ["reasoning.encrypted_content"]
    assert fake.last_kwargs["reasoning"] == {"effort": "medium", "summary": "auto"}


@pytest.mark.asyncio
async def test_stream_captures_reasoning_for_continuity(monkeypatch):
    provider = OpenAIResponsesProvider(api_key="sk-test")
    completed = _ns(output=[], usage=None, status="completed", incomplete_details=None)
    events = [
        _ns(
            type="response.output_item.done", item=_ns(type="reasoning", id="rs_1", encrypted_content="ENC", summary=[])
        ),
        _ns(type="response.output_text.delta", delta="hi"),
        _ns(type="response.completed", response=completed),
    ]
    monkeypatch.setattr(provider, "async_client", _ns(responses=_FakeResponses(_FakeStream(events))))

    wrapper = await provider.stream(
        MessageHistory([Message(role="user", content=[TextBlock(text="x")])]),
        params=GenerationParams(thinking="high"),
        model="gpt-5.5",
    )
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()

    assert isinstance(message.content[0], ResponsesReasoningBlock)
    assert message.content[0].encrypted_content == "ENC"
