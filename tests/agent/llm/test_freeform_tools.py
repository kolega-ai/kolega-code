import types
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.conversation import Conversation, adapt_history_for_provider
from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens
from kolega_code.config import AgentConfig, EditProtocol, ModelConfig, ModelProvider
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolDefinition, ToolResult
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.providers.openai_responses import OpenAIResponsesProvider
from kolega_code.llm.providers.chatgpt_oauth import ChatGPTOAuthProvider
from kolega_code.llm.providers.responses_common import ResponsesStreamWrapper, responses_tools, to_responses_input
from kolega_code.tools import Tool, ToolRegistry


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


class _FakeStream:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        async def generate():
            for event in self.events:
                yield event

        return generate()


def freeform_definition() -> ToolDefinition:
    return ToolDefinition(
        name="apply_patch",
        description="Apply a patch",
        parameters=[],
        input_kind="freeform",
        freeform_format={"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
    )


def test_responses_uses_native_custom_tool_definition() -> None:
    assert responses_tools(GenerationParams(tools=[freeform_definition()])) == [
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
        }
    ]


@pytest.mark.parametrize("provider_kind", ["openai", "openai_chatgpt"])
def test_both_responses_providers_build_native_custom_tool_requests(provider_kind: str) -> None:
    if provider_kind == "openai":
        provider = OpenAIResponsesProvider(api_key="test")
    else:
        tokens = OAuthTokens(
            access_token="at",
            refresh_token="rt",
            expires_at=10**12,
            account_id="account",
        )
        provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(tokens))

    request = provider._build_request(
        MessageHistory([Message(role="user", content=[TextBlock(text="edit")])]),
        None,
        GenerationParams(tools=[freeform_definition()]),
        {"model": "test-model"},
    )

    assert request["tools"][0]["type"] == "custom"
    assert request["tools"][0]["format"]["syntax"] == "lark"


def test_responses_round_trips_custom_call_and_output() -> None:
    raw = "*** Begin Patch\n*** Add File: a.txt\n+x\n*** End Patch\n"
    history = MessageHistory(
        [
            Message(
                role="assistant",
                content=[ToolCall(id="call-1", name="apply_patch", input=raw, input_kind="freeform")],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id="call-1",
                        name="apply_patch",
                        content="Success",
                        is_error=False,
                        input_kind="freeform",
                    )
                ],
            ),
        ]
    )

    assert to_responses_input(history) == [
        {"type": "custom_tool_call", "call_id": "call-1", "name": "apply_patch", "input": raw},
        {"type": "custom_tool_call_output", "call_id": "call-1", "output": "Success"},
    ]


@pytest.mark.asyncio
async def test_responses_stream_parses_custom_tool_events() -> None:
    raw = "*** Begin Patch\n*** Delete File: old.txt\n*** End Patch\n"
    completed = _ns(output=[], usage=None, status="completed", incomplete_details=None)
    events = [
        _ns(
            type="response.output_item.added",
            item=_ns(type="custom_tool_call", id="ct_1", call_id="call-1", name="apply_patch"),
        ),
        _ns(type="response.custom_tool_call_input.delta", item_id="ct_1", delta=raw[:20]),
        _ns(type="response.custom_tool_call_input.done", item_id="ct_1", input=raw),
        _ns(type="response.completed", response=completed),
    ]
    wrapper = ResponsesStreamWrapper(_FakeStream(events), provider_name="openai")
    async with wrapper as stream:
        async for _ in stream:
            pass

    message = await wrapper.get_final_message()

    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].input == raw
    assert message.tool_calls[0].input_kind == "freeform"


def test_non_responses_providers_receive_json_string_fallback() -> None:
    definition = freeform_definition()
    expected_schema = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Raw freeform tool input. Do not JSON-encode the value itself.",
            }
        },
        "required": ["input"],
    }

    assert definition.to_anthropic()["input_schema"] == expected_schema
    assert definition.to_openai()["function"]["parameters"] == expected_schema
    declarations = definition.to_google().function_declarations
    assert declarations is not None
    google = declarations[0]
    assert google.parameters is not None
    assert google.parameters.required == ["input"]
    assert google.parameters.properties is not None
    assert "input" in google.parameters.properties


@pytest.mark.parametrize(
    "provider_name",
    [
        provider.value
        for provider in ModelProvider
        if provider
        not in {
            ModelProvider.OPENAI,
            ModelProvider.OPENAI_CHATGPT,
            ModelProvider.ANTHROPIC,
            ModelProvider.GOOGLE,
        }
    ],
)
def test_every_chat_completions_provider_uses_freeform_fallback_contract(provider_name: str) -> None:
    provider = object.__new__(OpenAIProvider)
    provider.provider_name = provider_name
    params = provider._prepare_generation_params(GenerationParams(tools=[freeform_definition()]))

    assert params["tools"][0]["type"] == "function"
    assert params["tools"][0]["function"]["parameters"]["required"] == ["input"]


def test_freeform_call_fallback_serializers_wrap_raw_input() -> None:
    call = ToolCall(id="call-1", name="apply_patch", input="RAW", input_kind="freeform")

    assert call.to_anthropic()["input"] == {"input": "RAW"}
    assert call.to_openai()["function"]["arguments"] == '{"input": "RAW"}'
    function_call = call.to_google().function_call
    assert function_call is not None
    assert function_call.args == {"input": "RAW"}


@pytest.mark.parametrize("provider", ["anthropic", "google", "fireworks"])
def test_same_provider_fallback_history_remains_replayable(provider: str) -> None:
    call = ToolCall(id="call-1", name="apply_patch", input="RAW", input_kind="freeform")
    result = ToolResult(
        tool_use_id="call-1",
        name="apply_patch",
        content="Success",
        is_error=False,
        input_kind="freeform",
    )
    history = MessageHistory(
        [
            Message(role="assistant", content=[call], usage_metadata={"provider": provider}),
            Message(role="user", content=[result]),
        ]
    )
    adapted = adapt_history_for_provider(
        history,
        target_provider=provider,
        target_model="model",
        supports_vision=True,
        target_edit_protocol="codex_apply_patch",
    )

    assert adapted is history
    if provider == "anthropic":
        payload = history.to_anthropic()
        assert payload[0]["content"][0]["input"] == {"input": "RAW"}
    elif provider == "google":
        payload = history.to_google()
        assert payload[0].parts[0].function_call.args == {"input": "RAW"}
    else:
        payload = history.to_openai(provider=provider, model="model")
        assert payload[0]["tool_calls"][0]["function"]["arguments"] == '{"input": "RAW"}'


def test_freeform_transcript_round_trip_and_legacy_defaults() -> None:
    call = ToolCall(id="call-1", name="apply_patch", input="RAW", input_kind="freeform")
    result = ToolResult(
        tool_use_id="call-1",
        name="apply_patch",
        content="ok",
        is_error=False,
        input_kind="freeform",
    )

    assert ToolCall.from_dict(call.to_dict()).input_kind == "freeform"
    assert ToolResult.from_dict(result.to_dict()).input_kind == "freeform"
    assert ToolCall.from_dict({"type": "tool_call", "id": "old", "name": "read", "input": {}}).input_kind == "json"
    assert (
        ToolResult.from_dict(
            {"type": "tool_result", "tool_use_id": "old", "name": "read", "content": "ok", "is_error": False}
        ).input_kind
        == "json"
    )


def test_interrupted_freeform_call_repair_preserves_input_kind() -> None:
    call = ToolCall(id="call-1", name="apply_patch", input="RAW", input_kind="freeform")
    repaired = Conversation([Message(role="assistant", content=[call])]).repaired()

    result = repaired[1].content[0]
    assert isinstance(result, ToolResult)
    assert result.input_kind == "freeform"


@pytest.mark.parametrize("source,target", [("openai", "anthropic"), ("anthropic", "google"), ("google", "openai")])
def test_cross_provider_switch_neutralizes_freeform_exchange_without_mutating_storage(source: str, target: str) -> None:
    call = ToolCall(id="call-1", name="apply_patch", input="RAW", input_kind="freeform")
    result = ToolResult(
        tool_use_id="call-1",
        name="apply_patch",
        content="Success",
        is_error=False,
        input_kind="freeform",
    )
    history = [
        Message(role="assistant", content=[call], usage_metadata={"provider": source}),
        Message(role="user", content=[result]),
    ]

    adapted = adapt_history_for_provider(
        history,
        target_provider=target,
        target_model="target-model",
        supports_vision=True,
        target_edit_protocol="codex_apply_patch",
    )

    assert all(isinstance(block, TextBlock) for message in adapted for block in message.content)
    assert history[0].content[0] is call
    assert history[1].content[0] is result


def test_openai_backend_switch_preserves_native_custom_exchange() -> None:
    call = ToolCall(id="call-1", name="apply_patch", input="RAW", input_kind="freeform")
    result = ToolResult(
        tool_use_id="call-1",
        name="apply_patch",
        content="Success",
        is_error=False,
        input_kind="freeform",
    )
    history = [
        Message(role="assistant", content=[call], usage_metadata={"provider": "openai"}),
        Message(role="user", content=[result]),
    ]

    adapted = adapt_history_for_provider(
        history,
        target_provider="openai_chatgpt",
        target_model="gpt-test",
        supports_vision=True,
        target_edit_protocol="codex_apply_patch",
    )

    assert adapted is history


def test_edit_protocol_enum_preserves_same_provider_freeform_exchange() -> None:
    call = ToolCall(id="call-1", name="apply_patch", input="RAW", input_kind="freeform")
    history = [Message(role="assistant", content=[call], usage_metadata={"provider": "anthropic"})]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-test",
        supports_vision=True,
        target_edit_protocol=EditProtocol.CODEX_APPLY_PATCH,
    )

    assert adapted is history


def test_switching_edit_protocol_neutralizes_freeform_exchange() -> None:
    history = [
        Message(
            role="assistant",
            content=[ToolCall(id="call-1", name="apply_patch", input="RAW", input_kind="freeform")],
            usage_metadata={"provider": "anthropic"},
        )
    ]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-test",
        supports_vision=True,
        target_edit_protocol="search_replace",
    )

    assert isinstance(adapted[0].content[0], TextBlock)


def test_protocol_metadata_prevents_replaying_another_freeform_language() -> None:
    history = [
        Message(
            role="assistant",
            content=[ToolCall(id="call-1", name="apply_patch", input="RAW", input_kind="freeform")],
            usage_metadata={"provider": "anthropic", "edit_protocol": "future_hashline"},
        )
    ]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-test",
        supports_vision=True,
        target_edit_protocol="codex_apply_patch",
    )

    assert isinstance(adapted[0].content[0], TextBlock)


def test_json_edit_history_stays_portable_when_surface_schema_changes() -> None:
    call = ToolCall(
        id="call-1",
        name="edit",
        input={"file_path": "a.txt", "old_string": "old", "new_string": "new"},
    )
    history = [
        Message(
            role="assistant",
            content=[call],
            usage_metadata={"provider": "anthropic", "edit_protocol": "claude_code"},
        )
    ]

    adapted = adapt_history_for_provider(
        history,
        target_provider="openai",
        target_model="gpt-test",
        supports_vision=True,
        target_edit_protocol="search_replace",
    )

    assert adapted is history


@pytest.mark.asyncio
async def test_agent_normalizes_provider_fallback_before_storage_and_execution(tmp_path) -> None:
    captured = AsyncMock(return_value="Success")
    definition = freeform_definition()

    class FreeformTools:
        def registry(self):
            return ToolRegistry(
                [Tool(name="apply_patch", definition=definition, handler=captured, parallel_safe=False)]
            )

    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")
    config = AgentConfig(
        anthropic_api_key="test",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
    )
    agent = BaseAgent(tmp_path, "workspace", "thread", AsyncMock(), config)
    agent.tool_collection = cast(Any, FreeformTools())
    agent.send_chat_message = AsyncMock()
    agent.log_info = AsyncMock()
    call = ToolCall(id="call-1", name="apply_patch", input={"input": "RAW"})
    message = Message(role="assistant", content=[call], tool_calls=[call])

    agent._normalize_freeform_tool_calls(message)
    results = await agent.process_tool_calls(message.tool_calls)

    assert call.input == "RAW"
    assert call.input_kind == "freeform"
    captured.assert_awaited_once_with(input="RAW")
    assert results[0].input_kind == "freeform"
