from typing import AsyncContextManager, List, Optional

from google.genai import Client as genai_client
from google.genai import types as genai_types

from ..models import Message, MessageChunk, MessageHistory, ToolDefinition
from ..specs import build_thinking_request_params
from ..tool_execution_ids import ToolExecutionIdRegistry
from .base import BaseLLMProvider
from .models import GenerationParams, TokenCount


class GoogleStreamWrapper:
    def __init__(self, gemini_stream):
        self.gemini_stream = gemini_stream
        self.final_content = ""
        # Maps a running index -> (FunctionCall, thought_signature). The signature is a
        # part-level field (Gemini 3.x), so we read it off the parts, not chunk.function_calls.
        self.final_tool_calls = {}
        self._tool_call_index = 0
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

            # Read function calls off the parts so we also capture each part's thought_signature
            # (Gemini 3.x requires it echoed back). Accumulate across chunks with a running index.
            candidate = chunk.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call:
                        self.final_tool_calls[self._tool_call_index] = (part.function_call, part.thought_signature)
                        self._tool_call_index += 1

            self.stop_reason = candidate.finish_reason.value if candidate.finish_reason else None

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
        # Wire the google-genai client's built-in retry (exponential backoff) so transient
        # 429/5xx are retried like the other providers. attempts is total tries = retries + 1.
        self.async_client = genai_client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(
                    attempts=max(1, max_retries + 1),
                    http_status_codes=[408, 429, 500, 502, 503, 504],
                )
            ),
        )

    @property
    def retry_decorator(self):
        """Get retry decorator with configured max retries"""
        return self.get_retry_decorator()

    def _prepare_thinking_config(self, model: str, params: Optional[GenerationParams]):
        if not params or not params.thinking:
            return None
        request_params = build_thinking_request_params("google", model, params.thinking)
        thinking_config = request_params.get("thinking_config")
        if not thinking_config:
            return None
        return genai_types.ThinkingConfig(**thinking_config)

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
        model = kwargs["model"]
        config = genai_types.GenerateContentConfig(
            system_instruction=system.content[0].text,
            temperature=params.temperature,
            max_output_tokens=params.max_completion_tokens,
            tools=[t.to_google() for t in params.tools] if params.tools else None,
            thinking_config=self._prepare_thinking_config(model, params),
        )

        await self.rate_limiter.acquire()

        # NOTE: no per-request streaming read timeout here. Unlike the httpx-based
        # providers (see kolega_code/llm/timeouts.py), google-genai's HttpOptions exposes
        # only a single *total-request* timeout, not a per-read (inter-chunk) one. A total
        # bound would wrongly cut legitimately long streams, so bounding a silent stall on
        # Google would require an inactivity watchdog around the loop instead. Left as-is
        # until a Google stall is actually observed.
        return GoogleStreamWrapper(
            await self.async_client.aio.models.generate_content_stream(
                model=model, contents=messages.to_google(), config=config
            )
        )

    async def generate(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> Message:
        model = kwargs["model"]
        config = genai_types.GenerateContentConfig(
            system_instruction=system.content[0].text,
            temperature=params.temperature,
            max_output_tokens=params.max_completion_tokens,
            tools=[t.to_google() for t in params.tools] if params.tools else None,
            thinking_config=self._prepare_thinking_config(model, params),
        )

        await self.rate_limiter.acquire()

        response = await self.async_client.aio.models.generate_content(
            model=model, contents=messages.to_google(), config=config
        )

        return Message.from_google(response)
