# ruff: noqa: F401,F811,E402
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
