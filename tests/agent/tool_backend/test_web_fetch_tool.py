from unittest.mock import AsyncMock, Mock, patch

import pytest

from kolega_code.agent.tool_backend.web_fetch.answering import MAX_COMPLETION_TOKENS
from kolega_code.agent.tool_backend.web_fetch.pipeline import WebContent, WebContentError
from kolega_code.agent.tool_backend.web_fetch_tool import WebFetchTool
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.tools import ToolError


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key="test_key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="long-model", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="haiku-model", rate_limits=RateLimitConfig()),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="think-model",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )


@pytest.fixture
def caller():
    value = Mock()
    value.agent_name = "coder"
    value.current_tool_call_id = None
    value.workspace_id = "test_workspace"
    value.thread_id = "test_thread"
    value.llm = None
    value.user_id = "user-123"
    value.user_email = "user@example.com"
    return value


@pytest.fixture
def tool(tmp_path, agent_config, caller):
    return WebFetchTool(tmp_path, "test_workspace", "test_thread", AsyncMock(), agent_config, caller)


def _content(text: str = "Extracted content with the important fact.", **kwargs) -> WebContent:
    return WebContent(
        requested_url="https://example.com/",
        final_url="https://example.com/final",
        content=text,
        content_type="text/html",
        method="trafilatura",
        byte_count=512,
        **kwargs,
    )


def _model_response(text: str):
    response = Mock()
    response.get_text_content.return_value = text
    return response


def _configure_success(tool: WebFetchTool, content: WebContent | None = None, response_text: str | None = None):
    tool.content_pipeline.load = AsyncMock(return_value=content or _content())
    client = Mock()
    client.generate = AsyncMock(
        return_value=_model_response(
            response_text or '{"answer":"The important fact.","evidence":["the important fact"],"insufficient":false}'
        )
    )
    return client


@pytest.mark.asyncio
async def test_web_fetch_returns_source_grounded_answer_and_evidence(tool):
    client = _configure_success(tool)
    with (
        patch.object(tool, "_build_client", return_value=client),
        patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"context_length": 200_000, "max_completion_tokens": 1024},
        ),
    ):
        result = await tool.web_fetch("example.com", "Find the important fact")

    assert "Source: https://example.com/final" in result
    assert "Answer:\nThe important fact." in result
    assert '- "the important fact"' in result
    tool.content_pipeline.load.assert_awaited_once_with("example.com")
    assert client.generate.await_args.kwargs["max_completion_tokens"] == 1024


@pytest.mark.asyncio
async def test_web_fetch_caps_completion_tokens_without_clipping_answer_chars(tool):
    long_answer = "word " * 300
    response = (
        '{"answer":' + repr(long_answer).replace("'", '"') + ',"evidence":["important fact"],"insufficient":false}'
    )
    client = _configure_success(tool, response_text=response)
    with (
        patch.object(tool, "_build_client", return_value=client),
        patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"context_length": 200_000, "max_completion_tokens": 384_000},
        ),
    ):
        result = await tool.web_fetch("https://example.com", "Explain")

    assert "word word word" in result
    assert len(result) > 512
    assert client.generate.await_args.kwargs["max_completion_tokens"] == MAX_COMPLETION_TOKENS


@pytest.mark.asyncio
async def test_web_fetch_model_failure_returns_bounded_extracted_content(tool):
    text = "HEAD-" + ("x" * 90_000) + "-TAIL"
    client = _configure_success(tool, _content(text))
    client.generate.side_effect = RuntimeError("provider unavailable")
    with (
        patch.object(tool, "_build_client", return_value=client),
        patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"context_length": 200_000, "max_completion_tokens": 1024},
        ),
    ):
        result = await tool.web_fetch("https://example.com", "Explain")

    assert "Extracted content:" in result
    assert "HEAD-" in result and "-TAIL" in result
    assert "characters omitted from the middle" in result
    assert "Fast-model answering failed" in result


@pytest.mark.asyncio
async def test_web_fetch_insufficient_answer_degrades_to_content(tool):
    client = _configure_success(
        tool,
        response_text='{"answer":"Not present","evidence":[],"insufficient":true}',
    )
    with (
        patch.object(tool, "_build_client", return_value=client),
        patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"context_length": 200_000, "max_completion_tokens": 1024},
        ),
    ):
        result = await tool.web_fetch("https://example.com", "Find absent data")

    assert "Extracted content:" in result
    assert "could not find enough grounded evidence" in result


@pytest.mark.asyncio
async def test_web_fetch_terminal_content_failure_raises_tool_error(tool):
    tool.content_pipeline.load = AsyncMock(side_effect=WebContentError("HTTP 404 while fetching URL."))
    with pytest.raises(ToolError, match="HTTP 404"):
        await tool.web_fetch("https://example.com/missing", "Summarize")


@pytest.mark.asyncio
async def test_web_fetch_rejects_empty_instruction_before_network(tool):
    with pytest.raises(ToolError, match="non-empty instruction"):
        await tool.web_fetch("https://example.com", "  ")


def test_safe_display_url_removes_credentials_and_query() -> None:
    assert WebFetchTool._safe_display_url("https://user:secret@example.com/private?token=secret") == "the requested URL"
    assert WebFetchTool._safe_display_url("example.com/path?token=secret") == "https://example.com/path"


@pytest.mark.asyncio
async def test_progress_broadcast_failure_does_not_destroy_result(tool, caller):
    caller.current_tool_call_id = "call-1"
    tool.connection_manager.broadcast_event.side_effect = RuntimeError("UI disconnected")
    client = _configure_success(tool)
    with (
        patch.object(tool, "_build_client", return_value=client),
        patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"context_length": 200_000, "max_completion_tokens": 1024},
        ),
    ):
        result = await tool.web_fetch("https://example.com", "Summarize")

    assert "The important fact." in result
