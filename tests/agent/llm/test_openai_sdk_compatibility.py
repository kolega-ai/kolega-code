"""OpenAI Responses SDK-boundary compatibility regression tests."""

import json
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import ValidationError

from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolDefinition
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai_responses import OpenAIResponsesProvider


@pytest.mark.asyncio
async def test_responses_sdk_lenient_event_fallback_and_tool_serialization() -> None:
    incomplete_delta: dict[str, Any] = {
        "type": "response.output_text.delta",
        "delta": "useful delta",
    }
    # This confirms the fixture cannot take the SDK's strict model-validation path:
    # required event metadata is deliberately absent.
    with pytest.raises(ValidationError):
        ResponseTextDeltaEvent.model_validate(incomplete_delta)

    completed_event = {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {
            "id": "resp_mock",
            "created_at": 0,
            "model": "gpt-5.6-terra",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
        },
    }
    sse = (f"data: {json.dumps(incomplete_delta)}\n\ndata: {json.dumps(completed_event)}\n\ndata: [DONE]\n\n").encode()

    request_bodies: list[dict[str, Any]] = []
    request_urls: list[str] = []

    async def handle_request(request: httpx.Request) -> httpx.Response:
        request_urls.append(str(request.url))
        request_bodies.append(cast(dict[str, Any], json.loads(await request.aread())))
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=sse,
        )

    nested_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "model_override": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "minLength": 1},
                    "model": {"type": "string", "minLength": 1},
                    "thinking_effort": {
                        "anyOf": [
                            {"type": "string", "enum": ["low", "medium", "high"]},
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["provider", "model"],
                "additionalProperties": False,
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    }
    function_tool = ToolDefinition(
        name="dispatch_subagent",
        description="Dispatch a focused task to a sub-agent.",
        parameters=[],
        input_schema=nested_schema,
    )
    freeform_tool = ToolDefinition(
        name="apply_patch",
        description="Apply a patch using the native freeform edit protocol.",
        parameters=[],
        input_kind="freeform",
        freeform_format={
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: /.+/",
        },
    )

    provider = OpenAIResponsesProvider(api_key="sk-test", max_retries=0)
    original_client = provider.async_client
    await original_client.close()
    assert original_client.is_closed()

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    sdk_client = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=http_client,
    )
    provider.async_client = sdk_client

    try:
        wrapper = await provider.stream(
            MessageHistory([Message(role="user", content=[TextBlock(text="exercise the SDK boundary")])]),
            params=GenerationParams(tools=[function_tool, freeform_tool]),
            model="gpt-5.6-terra",
        )
        async with wrapper:
            chunks = [chunk async for chunk in wrapper]
        final_message = await wrapper.get_final_message()
    finally:
        await sdk_client.close()

    assert sdk_client.is_closed()
    assert http_client.is_closed
    # Reaching these assertions means the SDK's discriminated-union fallback did
    # not raise AttributeError, and the incomplete event retained its useful data.
    assert [(chunk.type, chunk.text) for chunk in chunks if chunk.type == "text"] == [("text", "useful delta")]
    assert final_message.get_text_content() == "useful delta"

    assert request_urls == ["https://openai.test/v1/responses"]
    assert len(request_bodies) == 1
    assert request_bodies[0]["tools"] == [
        {
            "type": "function",
            "name": "dispatch_subagent",
            "description": "Dispatch a focused task to a sub-agent.",
            "parameters": nested_schema,
        },
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch using the native freeform edit protocol.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": "start: /.+/",
            },
        },
    ]
