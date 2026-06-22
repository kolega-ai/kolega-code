"""Tests for the ChatGPT-subscription Responses provider and its wiring."""

import types

import httpx
import pytest

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    ResponsesReasoningBlock,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.chatgpt_oauth import (
    ChatGPTAuth,
    ChatGPTOAuthProvider,
    ResponsesStreamWrapper,
    instructions_from,
    responses_tools,
    to_responses_input,
)
from kolega_code.llm.providers.models import GenerationParams


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _tokens():
    return OAuthTokens(access_token="at", refresh_token="rt", expires_at=10**12, account_id="acct_1", plan_type="pro")


# --- conversion -----------------------------------------------------------------


def test_to_responses_input_user_and_assistant_text():
    history = MessageHistory(
        [
            Message(role="user", content=[TextBlock(text="hello")]),
            Message(role="assistant", content=[TextBlock(text="hi there")]),
        ]
    )
    items = to_responses_input(history)
    assert items == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "hi there"}]},
    ]


def test_to_responses_input_tool_call_and_result():
    history = MessageHistory(
        [
            Message(role="assistant", content=[ToolCall(id="call_1", name="read_file", input={"path": "a.py"})]),
            Message(
                role="user",
                content=[ToolResult(tool_use_id="call_1", content="file contents", name="read_file", is_error=False)],
            ),
        ]
    )
    items = to_responses_input(history)
    assert items[0] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read_file",
        "arguments": '{"path": "a.py"}',
    }
    assert items[1] == {"type": "function_call_output", "call_id": "call_1", "output": "file contents"}


def test_to_responses_input_image_and_system_skip():
    history = MessageHistory(
        [
            Message(role="system", content=[TextBlock(text="you are helpful")]),
            Message(role="user", content=[ImageBlock(image_type="base64", media_type="image/png", data="BASE64")]),
        ]
    )
    items = to_responses_input(history)
    # System message dropped; image becomes an input_image data URL.
    assert items == [
        {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,BASE64"}]}
    ]


def test_to_responses_input_image_tool_result_adds_followup_user_image():
    history = MessageHistory(
        [
            Message(role="assistant", content=[ToolCall(id="call_1", name="read_image", input={"path": "shot.png"})]),
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id="call_1",
                        content=[ImageBlock(image_type="base64", media_type="image/png", data="BASE64")],
                        name="read_image",
                        is_error=False,
                    )
                ],
            ),
        ]
    )

    items = to_responses_input(history)

    assert items[0]["type"] == "function_call"
    assert items[1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "[read_image returned 1 image; attached in the following user message.]",
    }
    assert items[2]["role"] == "user"
    assert items[2]["content"] == [
        {"type": "input_text", "text": "Image returned by tool read_image for tool call call_1."},
        {"type": "input_image", "image_url": "data:image/png;base64,BASE64"},
    ]


def test_to_responses_input_multiple_tool_outputs_before_image_followups():
    history = MessageHistory(
        [
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id="call_1",
                        content=[ImageBlock(image_type="base64", media_type="image/png", data="BASE64")],
                        name="read_image",
                        is_error=False,
                    ),
                    ToolResult(tool_use_id="call_2", content="file contents", name="read_file", is_error=False),
                ],
            )
        ]
    )

    items = to_responses_input(history)

    assert [item.get("type") for item in items[:2]] == ["function_call_output", "function_call_output"]
    assert items[0]["call_id"] == "call_1"
    assert items[1]["call_id"] == "call_2"
    assert items[2]["role"] == "user"
    assert any(part.get("type") == "input_image" for part in items[2]["content"])


def test_to_responses_input_resends_reasoning_before_tool_call():
    # A prior assistant turn carrying captured reasoning must resend it as a
    # reasoning item that *precedes* the function_call it belongs to.
    history = MessageHistory(
        [
            Message(
                role="assistant",
                content=[
                    ResponsesReasoningBlock(encrypted_content="ENC", summary=["thinking..."], item_id="rs_1"),
                    ToolCall(id="call_1", name="read_file", input={"path": "a.py"}),
                ],
            ),
        ]
    )
    items = to_responses_input(history)
    assert items[0] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "thinking..."}],
        "encrypted_content": "ENC",
    }
    # The opaque server id is intentionally not resent (matches Codex, store=false).
    assert "id" not in items[0]
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_1"


def test_responses_tools_flattens_function_shape():
    tool = ToolDefinition(
        name="read_file",
        description="Read a file",
        parameters=[ToolParameter(name="path", type="string", description="path", required=True)],
    )
    tools = responses_tools(GenerationParams(tools=[tool]))
    assert tools == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "path"}},
                "required": ["path"],
            },
        }
    ]


def test_instructions_from_system_message():
    system = Message(role="system", content=[TextBlock(text="be terse")])
    assert instructions_from(system, MessageHistory([])) == "be terse"


# --- streaming wrapper ----------------------------------------------------------


class _FakeStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


@pytest.mark.asyncio
async def test_responses_stream_wrapper_text_tools_and_usage():
    completed_response = _ns(
        output=[
            _ns(type="message", content=[_ns(type="output_text", text="Done.")]),
            _ns(type="function_call", call_id="call_1", id="fc_1", name="read_file", arguments='{"path": "a.py"}'),
        ],
        usage=_ns(input_tokens=12, output_tokens=4, total_tokens=16, input_tokens_details=None),
        status="completed",
        incomplete_details=None,
    )
    events = [
        _ns(type="response.output_text.delta", delta="Do"),
        _ns(type="response.output_text.delta", delta="ne."),
        _ns(type="response.output_item.added", item=_ns(type="function_call", call_id="call_1", id="fc_1", name="read_file")),
        _ns(type="response.function_call_arguments.delta", item_id="fc_1", delta='{"path"'),
        _ns(type="response.function_call_arguments.done", item_id="fc_1", arguments='{"path": "a.py"}'),
        _ns(type="response.completed", response=completed_response),
    ]

    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    text_chunks = []
    tool_starts = []
    async with wrapper as stream:
        async for chunk in stream:
            if chunk.type == "text" and chunk.text:
                text_chunks.append(chunk.text)
            elif chunk.type == "tool_use_start":
                tool_starts.append(chunk.tool_call_delta)

    assert "".join(text_chunks) == "Done."
    assert tool_starts == [{"id": "call_1", "name": "read_file", "input": ""}]

    message = await wrapper.get_final_message()
    assert message.get_text_content() == "Done."
    assert message.stop_reason == "tool_use"
    tool_calls = [b for b in message.content if isinstance(b, ToolCall)]
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].input == {"path": "a.py"}
    assert message.usage_metadata["prompt_tokens"] == 12
    assert message.usage_metadata["completion_tokens"] == 4


@pytest.mark.asyncio
async def test_responses_stream_wrapper_uses_deltas_when_completed_output_empty():
    # The ChatGPT/Codex backend streams the answer as deltas but sends
    # response.completed with an empty output[]. The final message must come from
    # the deltas, not the empty output (this was the "agent says one thing and
    # stops" bug).
    completed = _ns(
        output=[],
        usage=_ns(input_tokens=5, output_tokens=1, total_tokens=6, input_tokens_details=None),
        status="completed",
        incomplete_details=None,
    )
    events = [
        _ns(type="response.output_text.delta", delta="ready"),
        _ns(type="response.completed", response=completed),
    ]
    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    assert message.get_text_content() == "ready"
    assert message.usage_metadata["completion_tokens"] == 1
    assert message.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_responses_stream_wrapper_tool_call_from_output_item_done():
    # Tool calls must survive even when args arrive only via output_item.done
    # (and the completed output is empty, as on the ChatGPT backend).
    completed = _ns(output=[], usage=None, status="completed", incomplete_details=None)
    events = [
        _ns(
            type="response.output_item.done",
            item=_ns(type="function_call", id="fc_1", call_id="call_1", name="read_file", arguments='{"path": "a.py"}'),
        ),
        _ns(type="response.completed", response=completed),
    ]
    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    tool_calls = [b for b in message.content if isinstance(b, ToolCall)]
    assert tool_calls and tool_calls[0].id == "call_1"
    assert tool_calls[0].name == "read_file"
    assert tool_calls[0].input == {"path": "a.py"}
    assert message.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_provider_always_sends_non_empty_instructions():
    # The backend 400s with "Instructions are required" on an empty instructions.
    provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(_tokens()))
    fake = _FakeResponses(_FakeStream([]))
    provider.async_client = _ns(responses=fake)
    await provider.stream(
        MessageHistory([Message(role="user", content=[TextBlock(text="hi")])]),
        params=GenerationParams(),
        model="gpt-5.5",
    )
    assert fake.last_kwargs["instructions"]  # present and non-empty


@pytest.mark.asyncio
async def test_responses_stream_wrapper_max_tokens_stop_reason():
    completed = _ns(
        output=[_ns(type="message", content=[_ns(type="output_text", text="partial")])],
        usage=_ns(input_tokens=5, output_tokens=5, total_tokens=10, input_tokens_details=None),
        status="incomplete",
        incomplete_details=_ns(reason="max_output_tokens"),
    )
    wrapper = ResponsesStreamWrapper(_FakeStream([_ns(type="response.completed", response=completed)]))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    assert message.stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_responses_stream_wrapper_captures_reasoning_for_continuity():
    # With include=["reasoning.encrypted_content"], the backend emits a completed
    # reasoning item carrying the encrypted blob. We capture it (leading the
    # assistant message) so it round-trips into the next request for continuity.
    completed = _ns(output=[], usage=None, status="completed", incomplete_details=None)
    events = [
        _ns(
            type="response.output_item.done",
            item=_ns(
                type="reasoning",
                id="rs_1",
                encrypted_content="ENC",
                summary=[_ns(type="summary_text", text="planning")],
            ),
        ),
        _ns(
            type="response.output_item.done",
            item=_ns(type="function_call", id="fc_1", call_id="call_1", name="read_file", arguments='{"path": "a.py"}'),
        ),
        _ns(type="response.completed", response=completed),
    ]
    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()

    # Reasoning leads the assistant content so it precedes the tool call on resend.
    assert isinstance(message.content[0], ResponsesReasoningBlock)
    assert message.content[0].encrypted_content == "ENC"
    assert message.content[0].summary == ["planning"]

    items = to_responses_input(MessageHistory([message]))
    assert items[0]["type"] == "reasoning"
    assert items[0]["encrypted_content"] == "ENC"
    assert items[1]["type"] == "function_call"


@pytest.mark.asyncio
async def test_responses_stream_wrapper_reasoning_from_final_output_fallback():
    # Some backends populate reasoning only on the final response output, not via
    # output_item.done events. The fallback path must still capture it.
    completed = _ns(
        output=[
            _ns(type="reasoning", id="rs_1", encrypted_content="ENC2", summary=[]),
            _ns(type="message", content=[_ns(type="output_text", text="hi")]),
        ],
        usage=None,
        status="completed",
        incomplete_details=None,
    )
    wrapper = ResponsesStreamWrapper(_FakeStream([_ns(type="response.completed", response=completed)]))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    reasoning = [b for b in message.content if isinstance(b, ResponsesReasoningBlock)]
    assert reasoning and reasoning[0].encrypted_content == "ENC2"
    assert reasoning[0].summary == []


@pytest.mark.asyncio
async def test_responses_stream_wrapper_skips_reasoning_without_encrypted_content():
    # A reasoning item lacking encrypted_content is useless for continuity (and
    # would be rejected on resend), so it must be dropped.
    completed = _ns(output=[], usage=None, status="completed", incomplete_details=None)
    events = [
        _ns(type="response.output_item.done", item=_ns(type="reasoning", id="rs_1", encrypted_content=None, summary=[])),
        _ns(type="response.output_text.delta", delta="ok"),
        _ns(type="response.completed", response=completed),
    ]
    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    assert not [b for b in message.content if isinstance(b, ResponsesReasoningBlock)]


# --- provider request building (no network) -------------------------------------


class _FakeResponses:
    def __init__(self, result):
        self._result = result
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._result


@pytest.mark.asyncio
async def test_provider_generate_builds_codex_shaped_request():
    provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(_tokens()))
    completed = _ns(
        output=[_ns(type="message", content=[_ns(type="output_text", text="hi")])],
        usage=_ns(input_tokens=3, output_tokens=2, total_tokens=5, input_tokens_details=None),
        status="completed",
        incomplete_details=None,
    )
    fake = _FakeResponses(_FakeStream([_ns(type="response.completed", response=completed)]))
    provider.async_client = _ns(responses=fake)

    params = GenerationParams(max_completion_tokens=256, thinking="high")
    message = await provider.generate(
        MessageHistory([Message(role="user", content=[TextBlock(text="hello")])]),
        system=Message(role="system", content=[TextBlock(text="sys")]),
        params=params,
        model="gpt-5.5",
    )

    assert message.get_text_content() == "hi"
    kwargs = fake.last_kwargs
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["store"] is False
    assert kwargs["stream"] is True  # backend is SSE-only; generate streams too
    assert kwargs["instructions"] == "sys"
    # summary="auto" so the backend streams a reasoning summary for the thinking
    # display; this is independent of continuity, which rides on `include` below.
    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
    # Reasoning continuity: ask the backend for the encrypted reasoning blob so it
    # can be resent next turn (matches Codex; the cause of the long-thinking gap).
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is False
    assert "prompt_cache_key" in kwargs
    # Codex never sends max_output_tokens; sending it triggers a 400.
    assert "max_output_tokens" not in kwargs
    assert kwargs["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]


@pytest.mark.asyncio
async def test_provider_generate_omits_reasoning_and_include_without_thinking():
    provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(_tokens()))
    completed = _ns(
        output=[_ns(type="message", content=[_ns(type="output_text", text="hi")])],
        usage=None,
        status="completed",
        incomplete_details=None,
    )
    fake = _FakeResponses(_FakeStream([_ns(type="response.completed", response=completed)]))
    provider.async_client = _ns(responses=fake)
    await provider.generate(
        MessageHistory([Message(role="user", content=[TextBlock(text="hello")])]),
        params=GenerationParams(),  # no thinking effort
        model="gpt-5.5",
    )
    assert "reasoning" not in fake.last_kwargs
    assert "include" not in fake.last_kwargs


@pytest.mark.asyncio
async def test_provider_stream_returns_wrapper():
    provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(_tokens()))
    provider.async_client = _ns(responses=_FakeResponses(_FakeStream([])))
    stream = await provider.stream(
        MessageHistory([Message(role="user", content=[TextBlock(text="hi")])]),
        params=GenerationParams(),
        model="gpt-5.5",
    )
    assert isinstance(stream, ResponsesStreamWrapper)


# --- auth flow ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chatgpt_auth_sets_headers():
    auth = ChatGPTAuth(ChatGPTTokenManager(_tokens()))
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    flow = auth.async_auth_flow(request)
    sent = await flow.__anext__()
    assert sent.headers["Authorization"] == "Bearer at"
    assert sent.headers[chatgpt_constants.ACCOUNT_ID_HEADER] == "acct_1"
    await flow.aclose()


# --- LLMClient routing ----------------------------------------------------------


def test_llmclient_routes_to_chatgpt_provider():
    client = LLMClient(
        provider="openai_chatgpt",
        api_key="unused",
        token_manager=ChatGPTTokenManager(_tokens()),
    )
    assert isinstance(client.provider, ChatGPTOAuthProvider)


def test_llmclient_chatgpt_without_manager_raises():
    with pytest.raises(Exception):
        LLMClient(provider="openai_chatgpt", api_key="unused")


def test_agent_config_validates_with_chatgpt_tokens():
    config = AgentConfig(
        openai_chatgpt_tokens=_tokens(),
        long_context_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
        fast_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
        thinking_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
    )
    assert config.get_api_key(ModelProvider.OPENAI_CHATGPT) == "at"
    manager = config.get_chatgpt_token_manager()
    assert manager is not None


def test_agent_config_without_tokens_rejects_chatgpt_provider():
    with pytest.raises(ValueError, match="signed in"):
        AgentConfig(
            long_context_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
            fast_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
            thinking_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
        )
