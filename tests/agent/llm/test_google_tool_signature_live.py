"""Live Gemini tool-call round-trip — reproduces the thought_signature 400.

Before the fix, a second request that resent a Gemini 3.x function call without its
thought_signature failed with `400 INVALID_ARGUMENT ... missing a thought_signature`.
These tests drive the full loop (call -> tool_use -> tool_result -> call again) against
the real API for both the non-streaming and streaming paths.
"""

import inspect
import os

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import ContentBlock, Message, MessageHistory, TextBlock, ToolCall, ToolResult
from kolega_code.llm.models import ToolDefinition, ToolParameter

pytestmark = pytest.mark.integration

MODEL = "gemini-3.5-flash"
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

SYSTEM = Message(role="system", content=[TextBlock(text="You are a helpful coding assistant.")])
LIST_DIR_TOOL = ToolDefinition(
    name="list_directory",
    description="List the files in a directory.",
    parameters=[ToolParameter(name="path", type="string", description="Directory path", required=True)],
)


def _client() -> LLMClient:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    return LLMClient(provider="google", api_key=api_key)


def _user(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


def _tool_results_for(assistant: Message) -> Message:
    """Fabricate tool results for every tool call the model just made."""
    results: list[ContentBlock] = [
        ToolResult(tool_use_id=tc.id, content="README.md\npyproject.toml", name=tc.name, is_error=False)
        for tc in assistant.content
        if isinstance(tc, ToolCall)
    ]
    return Message(role="user", content=results)


@pytest.mark.asyncio
async def test_google_tool_round_trip_non_streaming() -> None:
    client = _client()
    history = MessageHistory([_user("List the files in the current directory using the tool.")])

    first = await client.generate(
        messages=history,
        system=SYSTEM,
        model=MODEL,
        max_completion_tokens=8192,
        temperature=1.0,
        thinking="high",
        tools=[LIST_DIR_TOOL],
    )
    tool_calls = [b for b in first.content if isinstance(b, ToolCall)]
    if not tool_calls:
        pytest.skip("Model did not call the tool; cannot exercise the signature round-trip")
    assert tool_calls[0].thought_signature, "expected a thought_signature on the Gemini function call"

    # Resend history with the assistant tool call + the tool result. This is the request that
    # 400'd before the fix.
    history.append(first)
    history.append(_tool_results_for(first))
    second = await client.generate(
        messages=history,
        system=SYSTEM,
        model=MODEL,
        max_completion_tokens=8192,
        temperature=1.0,
        thinking="high",
        tools=[LIST_DIR_TOOL],
    )
    assert second is not None
    assert second.role == "assistant"


@pytest.mark.asyncio
async def test_google_tool_round_trip_streaming() -> None:
    client = _client()
    history = MessageHistory([_user("List the files in the current directory using the tool.")])

    stream_obj = client.stream(
        messages=history,
        system=SYSTEM,
        model=MODEL,
        max_completion_tokens=8192,
        temperature=1.0,
        thinking="high",
        tools=[LIST_DIR_TOOL],
    )
    # LLMClient.stream returns either an async context manager or a coroutine
    # resolving to one, depending on the provider.
    if inspect.isawaitable(stream_obj):
        stream_obj = await stream_obj
    async with stream_obj as ctx:
        async for _ in ctx:
            pass
        first = await ctx.get_final_message()

    tool_calls = [b for b in first.content if isinstance(b, ToolCall)]
    if not tool_calls:
        pytest.skip("Model did not call the tool; cannot exercise the signature round-trip")
    assert tool_calls[0].thought_signature, "expected a thought_signature on the streamed Gemini function call"

    history.append(first)
    history.append(_tool_results_for(first))
    # The follow-up request must not 400 on a missing thought_signature.
    second_stream_obj = client.stream(
        messages=history,
        system=SYSTEM,
        model=MODEL,
        max_completion_tokens=8192,
        temperature=1.0,
        thinking="high",
        tools=[LIST_DIR_TOOL],
    )
    if inspect.isawaitable(second_stream_obj):
        second_stream_obj = await second_stream_obj
    async with second_stream_obj as ctx:
        async for _ in ctx:
            pass
        second = await ctx.get_final_message()
    assert second is not None
    assert second.role == "assistant"
