from unittest.mock import AsyncMock, Mock, patch

import pytest

from kolega_code.agent.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.web_fetch_tool import WebFetchTool


@pytest.fixture
def mock_connection_manager():
    return AsyncMock()


@pytest.fixture
def project_path(tmp_path):
    return tmp_path


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key="test_key",
        openai_api_key="test_key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="long-model", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="haiku-model", rate_limits=RateLimitConfig()),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="think-model", rate_limits=RateLimitConfig(), thinking_tokens=512
        ),
    )


@pytest.fixture
def mock_caller():
    caller = Mock()
    caller.agent_name = "coder"
    caller.current_tool_call_id = None
    caller.workspace_id = "test_workspace"
    caller.thread_id = "test_thread"
    caller.llm = None
    caller.user_id = "user-123"
    caller.user_email = "user@example.com"
    return caller


@pytest.fixture
def web_fetch_tool(project_path, mock_connection_manager, agent_config, mock_caller):
    return WebFetchTool(
        project_path, "test_workspace", "test_thread", mock_connection_manager, agent_config, mock_caller
    )


class TestWebFetchTool:
    @pytest.mark.asyncio
    async def test_web_fetch_success(self, web_fetch_tool, agent_config):
        with patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.fetch_url",
            return_value="<html>content</html>",
        ) as mock_fetch, patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.extract", return_value="Extracted content"
        ) as mock_extract, patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"max_completion_tokens": 1024},
        ) as mock_specs, patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.LLMClient"
        ) as mock_llm_class:
            mock_response = Mock()
            mock_response.get_text_content.return_value = "Summarized answer"
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.generate = AsyncMock(return_value=mock_response)

            result = await web_fetch_tool.web_fetch("https://example.com", "Summarize the page")

            assert result == "Summarized answer"
            mock_fetch.assert_called_once_with("https://example.com")
            mock_extract.assert_called_once()
            mock_specs.assert_called_once()
            mock_llm_instance.generate.assert_awaited_once()

            await_args, await_kwargs = mock_llm_instance.generate.await_args
            assert await_kwargs["model"] == agent_config.fast_config.model
            assert await_kwargs["max_completion_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_web_fetch_applies_char_limit(self, web_fetch_tool):
        with patch.object(WebFetchTool, "DEFAULT_RESPONSE_CHAR_LIMIT", 10), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.fetch_url",
            return_value="<html>content</html>",
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.extract", return_value="Extracted content"
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"max_completion_tokens": 1024},
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.LLMClient"
        ) as mock_llm_class:
            long_text = "Alpha Beta Gamma Delta"
            mock_response = Mock()
            mock_response.get_text_content.return_value = long_text
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.generate = AsyncMock(return_value=mock_response)

            result = await web_fetch_tool.web_fetch("https://example.com", "Summarize")

            assert result == "Alpha…"

    @pytest.mark.asyncio
    async def test_web_fetch_caps_large_model_token_limit(self, web_fetch_tool):
        with patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.fetch_url",
            return_value="<html>content</html>",
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.extract", return_value="Extracted content"
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"max_completion_tokens": 384000},
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.LLMClient"
        ) as mock_llm_class:
            mock_response = Mock()
            mock_response.get_text_content.return_value = "Summarized answer"
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.generate = AsyncMock(return_value=mock_response)

            result = await web_fetch_tool.web_fetch("https://example.com", "Summarize")

            assert result == "Summarized answer"
            await_args, await_kwargs = mock_llm_instance.generate.await_args
            assert await_kwargs["max_completion_tokens"] == WebFetchTool.WEB_FETCH_MAX_COMPLETION_TOKENS

    @pytest.mark.asyncio
    async def test_web_fetch_preserves_smaller_model_token_limit(self, web_fetch_tool):
        with patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.fetch_url",
            return_value="<html>content</html>",
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.extract", return_value="Extracted content"
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"max_completion_tokens": 512},
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.LLMClient"
        ) as mock_llm_class:
            mock_response = Mock()
            mock_response.get_text_content.return_value = "Summarized answer"
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.generate = AsyncMock(return_value=mock_response)

            result = await web_fetch_tool.web_fetch("https://example.com", "Summarize")

            assert result == "Summarized answer"
            await_args, await_kwargs = mock_llm_instance.generate.await_args
            assert await_kwargs["max_completion_tokens"] == 512

    @pytest.mark.asyncio
    async def test_web_fetch_reports_empty_model_response(self, web_fetch_tool):
        with patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.fetch_url",
            return_value="<html>content</html>",
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.extract", return_value="Extracted content"
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.get_model_specs",
            return_value={"max_completion_tokens": 1024},
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.LLMClient"
        ) as mock_llm_class:
            mock_response = Mock()
            mock_response.get_text_content.return_value = ""
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.generate = AsyncMock(return_value=mock_response)

            result = await web_fetch_tool.web_fetch("https://example.com", "Summarize")

            assert result == "Error: Fast model returned an empty response for fetched content."
            mock_llm_instance.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_web_fetch_invalid_url(self, web_fetch_tool):
        result = await web_fetch_tool.web_fetch("ftp://example.com", "Summarize")
        assert result.startswith("Error: Provide a valid http(s) URL.")

    @pytest.mark.asyncio
    async def test_web_fetch_no_content_downloaded(self, web_fetch_tool):
        with patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.trafilatura.fetch_url",
            return_value=None,
        ), patch(
            "kolega_code.agent.tool_backend.web_fetch_tool.LLMClient"
        ) as mock_llm_class:
            result = await web_fetch_tool.web_fetch("https://example.com", "Summarize")
            assert result.startswith("Error: No content retrieved from https://example.com")
            mock_llm_class.assert_not_called()
