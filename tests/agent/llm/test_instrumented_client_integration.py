"""
Integration tests for the InstrumentedLLMClient class using real API keys.

These tests require valid API keys to be set in the environment and will be skipped
if the keys are not available. They test the actual integration with LLM providers
and Langfuse tracing.
"""

import asyncio
import inspect
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider as _OtelTracerProvider

from kolega_code.llm.instrumented_client import InstrumentedLLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall

# Load environment variables from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv_path = REPO_ROOT / ".env"
if dotenv_path.exists():
    print(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path)
    print(f"ANTHROPIC_API_KEY present: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
    print(f"OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"GOOGLE_API_KEY present: {bool(os.getenv('GOOGLE_API_KEY'))}")
    print(f"LANGFUSE_PUBLIC_KEY present: {bool(os.getenv('LANGFUSE_PUBLIC_KEY'))}")
    print(f"LANGFUSE_SECRET_KEY present: {bool(os.getenv('LANGFUSE_SECRET_KEY'))}")
else:
    print(f"Warning: .env file not found at {dotenv_path}")
    print("Integration tests requiring API keys may be skipped.")

# Test data
TEST_MESSAGES = MessageHistory(
    [Message(role="user", content=[TextBlock(text="What is 2+2? Reply with just the number.")])]
)
TEST_SYSTEM = Message(role="system", content=[TextBlock(text="You are a helpful math assistant. Be concise.")])
ANTHROPIC_CACHE_TEST_MODEL = "claude-sonnet-4-6"

# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


@pytest.fixture
def real_langfuse_client():
    """Create a real Langfuse client if credentials are available."""
    if not all([os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY"), os.getenv("LANGFUSE_HOST")]):
        return None

    try:
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            tracer_provider=_OtelTracerProvider(),  # isolates Langfuse from Sentry's global OTEL provider
        )
    except Exception as e:
        print(f"Failed to create Langfuse client: {e}")
        return None


@pytest.fixture
def mock_langfuse_client():
    """Create a mock Langfuse client for testing (v3 API)."""
    langfuse = Mock()

    # Create a mock generation that tracks calls
    generation = Mock()
    generation.update = Mock()
    generation.end = Mock()

    # Create a mock trace/span that returns the generation
    trace = Mock()
    trace.update_trace = Mock()
    trace.update = Mock()
    trace.end = Mock()
    trace.start_generation = Mock(return_value=generation)

    # Make langfuse.start_span() return the trace
    langfuse.start_span = Mock(return_value=trace)

    return langfuse


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(SKIP_IN_CI, reason="Skipping in CI environment")
class TestInstrumentedClientWithRealAPIs:
    """Test InstrumentedLLMClient with real API calls."""

    @pytest.mark.asyncio
    async def test_anthropic_generation_with_instrumentation(self, mock_langfuse_client):
        """Test Anthropic generation with instrumentation using real API."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key=api_key,
            langfuse_client=mock_langfuse_client,
            workspace_id="test-workspace",
            thread_id="test-thread",
            agent_type="test-agent",
            environment="test",
        )

        # Make real API call
        response = await client.generate(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="claude-haiku-4-5-20251001",
            max_completion_tokens=10,
            temperature=0,
        )

        # Verify response
        assert response is not None
        assert response.role == "assistant"
        assert response.get_text_content()
        assert "4" in response.get_text_content()

        # Verify Langfuse was called (v3 API)
        mock_langfuse_client.start_span.assert_called_once()
        trace = mock_langfuse_client.start_span.return_value
        trace.start_generation.assert_called_once()

        # Verify usage data was extracted
        generation = trace.start_generation.return_value
        generation.update.assert_called_once()
        generation.end.assert_called_once()
        update_call = generation.update.call_args
        assert update_call.kwargs["usage_details"] is not None
        assert update_call.kwargs["usage_details"]["input"] > 0
        assert update_call.kwargs["usage_details"]["output"] > 0
        assert update_call.kwargs["level"] == "DEFAULT"

    @pytest.mark.asyncio
    async def test_openai_generation_with_instrumentation(self, mock_langfuse_client):
        """Test OpenAI generation with instrumentation using real API."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        client = InstrumentedLLMClient(
            provider="openai",
            api_key=api_key,
            langfuse_client=mock_langfuse_client,
            workspace_id="test-workspace",
            thread_id="test-thread",
            agent_type="test-agent",
            environment="production",
        )

        # Make real API call
        response = await client.generate(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="gpt-5.4-mini",
            max_completion_tokens=10,
            temperature=0,
        )

        # Verify response
        assert response is not None
        assert response.role == "assistant"
        assert response.get_text_content()
        assert "4" in response.get_text_content()

        # Verify Langfuse was called with correct tags
        trace_call = mock_langfuse_client.start_span.return_value.update_trace.call_args
        assert trace_call.kwargs["tags"] == [
            "production",
            "workspace:test-workspace",
            "thread:test-thread",
            "agent:test-agent",
            "provider:openai",
        ]

        # Verify usage tracking
        generation = mock_langfuse_client.start_span.return_value.start_generation.return_value
        update_call = generation.update.call_args
        usage = update_call.kwargs["usage_details"]
        assert usage["input"] > 0
        assert usage["output"] > 0

    @pytest.mark.asyncio
    async def test_google_generation_with_instrumentation(self, mock_langfuse_client):
        """Test Google generation with instrumentation using real API."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")

        client = InstrumentedLLMClient(
            provider="google",
            api_key=api_key,
            langfuse_client=mock_langfuse_client,
            workspace_id="test-workspace",
            thread_id="test-thread",
            agent_type="test-agent",
            environment="development",
        )

        # Make real API call
        response = await client.generate(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="gemini-3.1-pro-preview",
            max_completion_tokens=128,
            temperature=0,
        )

        # Verify response
        assert response is not None
        assert response.role == "assistant"  # Normalized from Google's "model" role
        assert response.get_text_content()

        # Verify Langfuse integration
        assert mock_langfuse_client.start_span.called

    @pytest.mark.asyncio
    async def test_streaming_with_instrumentation(self, mock_langfuse_client):
        """Test streaming with instrumentation using real Anthropic API."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key=api_key,
            langfuse_client=mock_langfuse_client,
            workspace_id="test-workspace",
            thread_id="test-thread",
            agent_type="test-agent",
            environment="test",
        )

        # Stream response
        accumulated_text = ""
        stream_obj = client.stream(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="claude-haiku-4-5-20251001",
            max_completion_tokens=50,
            temperature=0,
        )
        # InstrumentedLLMClient.stream returns either an async context manager or a
        # coroutine resolving to one, depending on whether Langfuse is configured.
        if inspect.isawaitable(stream_obj):
            stream_obj = await stream_obj
        async with stream_obj as stream:
            async for chunk in stream:
                if chunk.type == "text":
                    accumulated_text += chunk.text

        # Verify response
        assert accumulated_text
        assert "4" in accumulated_text

        # Verify Langfuse was called
        mock_langfuse_client.start_span.assert_called_once()
        trace = mock_langfuse_client.start_span.return_value
        trace.start_generation.assert_called_once()

        # Verify streaming tag
        trace_update_call = trace.update_trace.call_args
        assert "streaming" in trace_update_call.kwargs["tags"]

        gen_call = trace.start_generation.call_args
        assert gen_call.kwargs["model_parameters"]["streaming"] is True

        # Verify generation.end was called
        generation = trace.start_generation.return_value
        generation.update.assert_called_once()
        generation.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_with_instrumentation(self, mock_langfuse_client):
        """Test error handling with instrumentation."""
        # Use invalid API key
        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key="invalid-key",
            langfuse_client=mock_langfuse_client,
            workspace_id="test-workspace",
            thread_id="test-thread",
            agent_type="test-agent",
            environment="test",
        )

        # Attempt API call with invalid key
        with pytest.raises(Exception):
            await client.generate(
                messages=TEST_MESSAGES,
                system=TEST_SYSTEM,
                model="claude-haiku-4-5-20251001",
            )

        # Verify error was logged to Langfuse
        trace = mock_langfuse_client.start_span.return_value
        generation = trace.start_generation.return_value
        generation.update.assert_called_once()
        generation.end.assert_called_once()
        update_call = generation.update.call_args
        assert update_call.kwargs["level"] == "ERROR"

    @pytest.mark.asyncio
    async def test_cache_tokens_tracking(self, mock_langfuse_client):
        """Test that Anthropic cache tokens are properly tracked."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key=api_key,
            langfuse_client=mock_langfuse_client,
            workspace_id="test-workspace",
            thread_id="test-thread",
            agent_type="test-agent",
            environment="test",
        )

        # Create a longer conversation to potentially trigger cache
        long_messages = MessageHistory(
            [Message(role="user", content=[TextBlock(text="Tell me about Python programming.")])]
        )

        # First call
        await client.generate(
            messages=long_messages,
            system=TEST_SYSTEM,
            model=ANTHROPIC_CACHE_TEST_MODEL,
            max_completion_tokens=100,
        )

        # Second call with same system prompt might use cache
        await client.generate(
            messages=long_messages,
            system=TEST_SYSTEM,
            model=ANTHROPIC_CACHE_TEST_MODEL,
            max_completion_tokens=100,
        )

        # Cache usage is not guaranteed, but usage details should be captured for each call.
        trace = mock_langfuse_client.start_span.return_value
        generation = trace.start_generation.return_value
        assert generation.update.call_count >= 2, "Should have made at least 2 calls"
        usage_updates = [
            call.kwargs["usage_details"]
            for call in generation.update.call_args_list
            if call.kwargs.get("usage_details")
        ]
        assert len(usage_updates) >= 2, "Should have captured usage details for both calls"


@pytest.mark.slow
@pytest.mark.integration
class TestRealLangfuseIntegration:
    """Test with real Langfuse service if credentials are available."""

    @pytest.mark.asyncio
    async def test_real_langfuse_integration(self, real_langfuse_client):
        """Test actual Langfuse integration if credentials are available."""
        if not real_langfuse_client:
            pytest.skip("Langfuse credentials not available")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key=api_key,
            langfuse_client=real_langfuse_client,
            workspace_id="integration-test",
            thread_id="test-thread-123",
            agent_type="integration-test-agent",
            environment=os.getenv("ENVIRONMENT", "test"),
        )

        # Make real API call
        response = await client.generate(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="claude-haiku-4-5-20251001",
            max_completion_tokens=10,
            temperature=0,
        )

        # Verify response
        assert response is not None
        assert "4" in response.get_text_content()

        # Flush to ensure trace is sent
        real_langfuse_client.flush()

        # Give Langfuse a moment to process
        await asyncio.sleep(1)

        print("✅ Real Langfuse trace sent successfully")
        print("Check Langfuse dashboard for trace with:")
        print("  - Workspace: integration-test")
        print("  - Thread: test-thread-123")
        print("  - Agent: integration-test-agent")


@pytest.mark.slow
@pytest.mark.integration
class TestInstrumentedClientCompatibility:
    """Test that instrumented client maintains compatibility with base client."""

    @pytest.mark.asyncio
    async def test_fallback_without_langfuse(self):
        """Test that client works normally without Langfuse."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        # Create client without Langfuse
        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key=api_key,
            langfuse_client=None,  # No Langfuse
        )

        # Should work normally
        response = await client.generate(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="claude-haiku-4-5-20251001",
            max_completion_tokens=10,
            temperature=0,
        )

        assert response is not None
        assert "4" in response.get_text_content()

    @pytest.mark.asyncio
    async def test_all_providers_with_instrumentation(self, mock_langfuse_client):
        """Test instrumentation works with all supported providers."""
        providers_to_test = [
            ("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
            ("openai", "OPENAI_API_KEY", "gpt-5.4-mini"),
            ("google", "GOOGLE_API_KEY", "gemini-3.1-pro-preview"),
        ]

        for provider, env_key, model in providers_to_test:
            api_key = os.getenv(env_key)
            if not api_key:
                print(f"Skipping {provider} - {env_key} not set")
                continue

            client = InstrumentedLLMClient(
                provider=provider,
                api_key=api_key,
                langfuse_client=mock_langfuse_client,
                workspace_id="test-workspace",
                thread_id="test-thread",
                agent_type=f"{provider}-test-agent",
                environment="test",
            )

            try:
                response = await client.generate(
                    messages=TEST_MESSAGES,
                    system=TEST_SYSTEM,
                    model=model,
                    max_completion_tokens=10,
                    temperature=0,
                )

                assert response is not None
                print(f"✅ {provider} instrumentation working")

                # Verify Langfuse was called
                assert mock_langfuse_client.start_span.called

            except Exception as e:
                pytest.fail(f"Failed to test {provider}: {str(e)}")
            finally:
                # Reset mock for next provider
                mock_langfuse_client.reset_mock()

    @pytest.mark.asyncio
    async def test_tool_calling_with_instrumentation(self, mock_langfuse_client):
        """Test that tool calling works with instrumentation."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key=api_key,
            langfuse_client=mock_langfuse_client,
            workspace_id="test-workspace",
            thread_id="test-thread",
            agent_type="test-agent",
            environment="test",
        )

        # Define a simple tool using proper ToolDefinition
        from kolega_code.llm.models import ToolDefinition, ToolParameter

        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get the weather for a location",
                parameters=[
                    ToolParameter(
                        name="location", type="string", description="The location to get weather for", required=True
                    )
                ],
            )
        ]

        messages = MessageHistory(
            [Message(role="user", content=[TextBlock(text="What's the weather in San Francisco?")])]
        )

        response = await client.generate(
            messages=messages,
            system=TEST_SYSTEM,  # Add system message to avoid None error
            model="claude-haiku-4-5-20251001",
            tools=tools,
            max_completion_tokens=200,
        )

        # Should either answer directly or call the tool
        assert response is not None
        content = response.content

        # Check if it made a tool call
        tool_calls = [c for c in content if isinstance(c, ToolCall)]
        if tool_calls:
            assert tool_calls[0].name == "get_weather"
            assert "location" in tool_calls[0].input

        # Verify Langfuse tracked it
        trace = mock_langfuse_client.start_span.return_value
        generation = trace.start_generation.return_value
        generation.update.assert_called_once()
        generation.end.assert_called_once()
