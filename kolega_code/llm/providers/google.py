from typing import AsyncContextManager, List, Optional

from google.genai import Client as genai_client
from google.genai import types as genai_types

from ..models import Message, MessageChunk, MessageHistory, ToolDefinition
from ..tool_execution_ids import ToolExecutionIdRegistry
from .base import BaseLLMProvider
from .models import GenerationParams, TokenCount


class GoogleStreamWrapper:
    def __init__(self, gemini_stream):
        self.gemini_stream = gemini_stream
        self.final_content = ""
        self.final_tool_calls = {}
        self.stop_reason = None
        self.tool_execution_ids = ToolExecutionIdRegistry()

        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.gemini_stream, "aclose"):
            await self.gemini_stream.aclose()

        self._closed = True
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration

        try:
            chunk = await self.gemini_stream.__anext__()

            content = chunk.text or ""
            self.final_content += content

            for idx, function_call in enumerate(chunk.function_calls or []):
                self.final_tool_calls[idx] = function_call

                # self.final_tool_calls[function_call_id].function.arguments += tool_call.function.arguments

            self.stop_reason = chunk.candidates[0].finish_reason.value if chunk.candidates[0].finish_reason else None

            return MessageChunk.from_google(chunk)

        except StopAsyncIteration:
            raise

    async def get_final_message(self):
        return Message.from_google_stream(
            role="assistant",
            content=self.final_content,
            tool_calls=self.final_tool_calls,
            stop_reason=self.stop_reason,
            tool_execution_ids=self.tool_execution_ids,
        )


class GoogleProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(api_key, max_retries, requests_per_minute, tokens_per_minute, base_url)
        self.async_client = genai_client(api_key=api_key)

    @property
    def retry_decorator(self):
        """Get retry decorator with configured max retries"""
        return self.get_retry_decorator()

    async def count_tokens(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        model: Optional[str] = None,
        tools: List[ToolDefinition] = None,
        **kwargs,
    ) -> TokenCount:
        """Count tokens for a list of messages using tiktoken

        Args:
            messages: List of messages to count tokens for
            system: Optional system message
            model: Optional model name to use for counting (defaults to gpt-4)

        Returns:
            TokenCount object with input token count
        """
        count = await self.async_client.aio.models.count_tokens(
            model=model,
            contents=messages.to_google(),
        )

        return TokenCount(input_tokens=count.total_tokens, output_tokens=None)

    async def stream(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> AsyncContextManager:
        """Generate a streaming response from Google

        Returns a coroutine that resolves to an async iterator.
        """
        config = genai_types.GenerateContentConfig(
            system_instruction=system.content[0].text,
            temperature=params.temperature,
            max_output_tokens=params.max_completion_tokens,
            tools=[t.to_google() for t in params.tools] if params.tools else None,
            thinking_config=params.thinking,
        )

        await self.rate_limiter.acquire()

        return GoogleStreamWrapper(
            await self.async_client.aio.models.generate_content_stream(
                model=kwargs["model"], contents=messages.to_google(), config=config
            )
        )

    async def generate(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> Message:
        config = genai_types.GenerateContentConfig(
            system_instruction=system.content[0].text,
            temperature=params.temperature,
            max_output_tokens=params.max_completion_tokens,
            tools=[t.to_google() for t in params.tools] if params.tools else None,
            thinking_config=params.thinking,
        )

        await self.rate_limiter.acquire()

        response = await self.async_client.aio.models.generate_content(
            model=kwargs["model"], contents=messages.to_google(), config=config
        )

        return Message.from_google(response)
