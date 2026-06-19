from typing import Any, AsyncContextManager, Dict, List, Optional
import json
import logging
import math

import tiktoken
from openai import AsyncOpenAI, OpenAI

from ..models import ImageBlock, Message, MessageChunk, MessageHistory, ToolCall, ToolDefinition, ToolResult
from ..specs import build_thinking_request_params
from ..tool_execution_ids import ToolExecutionIdRegistry
from .base import BaseLLMProvider
from .models import GenerationParams, TokenCount


class OpenAIStreamWrapper:
    def __init__(self, openai_stream, requested_include_usage: bool = False):
        self.openai_stream = openai_stream
        self.final_content = ""
        self.final_tool_calls = {}
        self.stop_reason = None
        self.usage_data = None
        self.tool_execution_ids = ToolExecutionIdRegistry()

        self._closed = False
        self._requested_include_usage = requested_include_usage

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.openai_stream, "aclose"):
            await self.openai_stream.aclose()

        self._closed = True
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration

        try:
            chunk = await self.openai_stream.__anext__()

            # Some providers emit usage-only events with no choices; guard accesses
            if hasattr(chunk, "choices") and chunk.choices:
                choice0 = chunk.choices[0]
                delta = getattr(choice0, "delta", None)
                if delta is not None:
                    content = getattr(delta, "content", None) or ""
                    if content:
                        self.final_content += content

                    for tool_call in getattr(delta, "tool_calls", []) or []:
                        index = tool_call.index

                        if index not in self.final_tool_calls:
                            self.final_tool_calls[index] = tool_call

                        if self.final_tool_calls[index].function.arguments != tool_call.function.arguments:
                            if self.final_tool_calls[index].function.arguments is None:
                                self.final_tool_calls[index].function.arguments = ""

                            self.final_tool_calls[index].function.arguments += tool_call.function.arguments

                # Capture stop reason if present
                self.stop_reason = getattr(choice0, "finish_reason", self.stop_reason)

            # Capture usage data from final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                self.usage_data = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
                # Capture cached prompt tokens if available (e.g., DashScope/Qwen)
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                cached = None
                if details is not None:
                    cached = getattr(details, "cached_tokens", None)
                    if cached is None and isinstance(details, dict):
                        cached = details.get("cached_tokens")
                if cached is not None:
                    self.usage_data["cache_read_input_tokens"] = cached

            # Return a safe chunk representation; ignore events with no choices
            if hasattr(chunk, "choices") and chunk.choices:
                return MessageChunk.from_openai(chunk)
            else:
                return MessageChunk(type="ignore", text="")

        except StopAsyncIteration:
            raise

    async def get_final_message(self):
        message = Message.from_openai_stream(
            role="assistant",
            content=self.final_content,
            tool_calls=self.final_tool_calls,
            stop_reason=self.stop_reason,
            tool_execution_ids=self.tool_execution_ids,
        )

        # Add usage data if available
        if self.usage_data:
            message.usage_metadata.update(self.usage_data)
        else:
            logger = logging.getLogger(__name__)
            if self._requested_include_usage:
                logger.warning(
                    "OpenAIStreamWrapper: include_usage requested but provider emitted no usage; billing may be skipped"
                )
            else:
                logger.warning(
                    "OpenAIStreamWrapper: no usage metadata captured from streaming response; billing may be skipped"
                )

        return message


class OpenAIProvider(BaseLLMProvider):

    models_max_completion_tokens = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.2"]

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
        provider_name: str = "openai",
    ):
        super().__init__(api_key, max_retries, requests_per_minute, tokens_per_minute, base_url)
        # OpenAI-compatible providers (xai, together, fireworks, dashscope, ...) reuse this
        # provider; provider_name is used to look up the model's thinking-effort spec.
        self.provider_name = provider_name
        # Forward max_retries so the SDK's built-in exponential backoff + jitter (which
        # honors retry-after and retries 429/5xx + connection errors) is actually used.
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)
        self.sync_client = OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)

    @property
    def retry_decorator(self):
        """Get retry decorator with configured max retries"""
        return self.get_retry_decorator()

    def _prepare_generation_params(self, params: Optional[GenerationParams] = None) -> Dict[str, Any]:
        """Convert common parameters to provider-specific format"""
        generation_params = {
            "model": "gpt-5.5",  # Default model
        }

        if params:
            if params.temperature is not None:
                generation_params["temperature"] = params.temperature
            if params.max_completion_tokens is not None:
                generation_params["max_tokens"] = params.max_completion_tokens
            if params.tools:
                generation_params["tools"] = [t.to_openai() for t in params.tools]

        return generation_params

    def _apply_thinking_params(self, generation_params: Dict[str, Any], params: Optional[GenerationParams]) -> None:
        if not params or not params.thinking:
            return
        generation_params.update(
            build_thinking_request_params(
                self.provider_name,
                str(generation_params["model"]),
                params.thinking,
            )
        )

    async def count_tokens(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        model: Optional[str] = None,
        tools: List[ToolDefinition] = None,
        **kwargs,
    ) -> TokenCount:
        """Count tokens for a list of messages using tiktoken.

        Provides comprehensive token counting including:
        - System prompts and messages with formatting overhead
        - Images with estimation based on base64 data size
        - Tool definitions with JSON serialization
        - Tool calls and tool results in message content

        Args:
            messages: List of messages to count tokens for
            system: Optional system message
            model: Optional model name to use for counting (defaults to gpt-4)
            tools: Optional tool definitions

        Returns:
            TokenCount object with input token count
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0

        # Combine system message with messages if provided
        all_messages = ([system] + list(messages)) if system else list(messages)

        # Count tokens for each message
        for message in all_messages:
            # Base tokens for message formatting
            num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n format

            # Add tokens for role
            if hasattr(message, "role") and message.role:
                num_tokens += len(encoding.encode(message.role))

            # Add tokens for content
            if hasattr(message, "content"):
                if isinstance(message.content, str):
                    num_tokens += len(encoding.encode(message.content))
                elif isinstance(message.content, list):
                    for item in message.content:
                        # Handle text blocks
                        if hasattr(item, "text") and item.text:
                            num_tokens += len(encoding.encode(item.text))
                        elif isinstance(item, dict) and "text" in item:
                            num_tokens += len(encoding.encode(item["text"]))
                        # Handle image blocks
                        elif isinstance(item, ImageBlock):
                            num_tokens += self._estimate_image_tokens(len(item.data))
                        elif hasattr(item, "data") and hasattr(item, "media_type"):
                            # ImageBlock - estimate tokens based on base64 data size
                            num_tokens += self._estimate_image_tokens(len(item.data))
                        # Handle tool calls
                        elif isinstance(item, ToolCall):
                            tool_call_json = json.dumps(item.to_openai())
                            num_tokens += len(encoding.encode(tool_call_json))
                            num_tokens += 2  # Minimal formatting overhead for tool calls
                        # Handle tool results
                        elif isinstance(item, ToolResult):
                            # Tool results contain content that needs to be counted
                            if isinstance(item.content, str):
                                num_tokens += len(encoding.encode(item.content))
                            elif isinstance(item.content, list):
                                for result_item in item.content:
                                    if hasattr(result_item, "text") and result_item.text:
                                        num_tokens += len(encoding.encode(result_item.text))
                            num_tokens += 2  # Minimal formatting overhead for tool results

        # Count tool definition tokens
        if tools:
            for tool in tools:
                tool_json = json.dumps(tool.to_openai())
                # Count JSON tokens
                json_tokens = len(encoding.encode(tool_json))
                # OpenAI uses highly optimized internal format (not JSON)
                # Empirically, their token count is ~79% of raw JSON token count
                # Apply scaling factor to match API behavior
                num_tokens += int(json_tokens * 0.79)

        return TokenCount(input_tokens=num_tokens)

    def _estimate_image_tokens(self, base64_data_length: int) -> int:
        """Estimate image token cost based on base64 data length.

        OpenAI charges for images based on their dimensions after resizing.
        Since we don't decode images (performance), we estimate based on data size.

        Uses same formula as Anthropic for consistency:
        tokens ≈ 20 + sqrt(base64_length * 6)

        This gives reasonable estimates:
        - Tiny images (96 chars base64): ~44 tokens
        - Small images (~50KB base64): ~659 tokens
        - Medium images (~200KB base64): ~1285 tokens
        - Large images (~800KB base64): ~2549 tokens

        Args:
            base64_data_length: Length of base64 encoded image data

        Returns:
            Estimated token count for the image
        """
        # Use square root scaling for better fit across image sizes
        # Base cost of 20 tokens + sqrt scaling
        estimated_tokens = 20 + int(math.sqrt(base64_data_length * 6))
        return estimated_tokens

    async def stream(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> AsyncContextManager:
        """Generate a streaming response from OpenAI

        Returns a coroutine that resolves to an async iterator.
        """
        generation_params = self._prepare_generation_params(params)
        generation_params.update(kwargs)
        self._apply_thinking_params(generation_params, params)
        generation_params["stream"] = True
        # Ask provider to include usage in the final stream chunk when supported
        try:
            existing_stream_options = generation_params.get("stream_options") or {}
            existing_stream_options["include_usage"] = True
            generation_params["stream_options"] = existing_stream_options
        except Exception:
            # Best-effort; some providers may not support stream_options
            pass

        # Reasoning models (gpt-5.x, etc.) use max_completion_tokens and only accept the
        # default temperature (1); sending any other value is a 400, so drop it.
        if generation_params["model"] in self.models_max_completion_tokens:
            if "max_tokens" in generation_params:
                generation_params["max_completion_tokens"] = generation_params["max_tokens"]
                del generation_params["max_tokens"]
            generation_params.pop("temperature", None)

        # Combine system message with messages if provided
        if system:
            messages = MessageHistory([system] + messages)

        await self.rate_limiter.acquire()

        return OpenAIStreamWrapper(
            await self.async_client.chat.completions.create(messages=messages.to_openai(), **generation_params),
            requested_include_usage=True,
        )

    async def generate(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> Message:
        generation_params = self._prepare_generation_params(params)
        generation_params.update(kwargs)
        self._apply_thinking_params(generation_params, params)

        # Reasoning models (gpt-5.x, etc.) use max_completion_tokens and only accept the
        # default temperature (1); sending any other value is a 400, so drop it.
        if generation_params["model"] in self.models_max_completion_tokens:
            if "max_tokens" in generation_params:
                generation_params["max_completion_tokens"] = generation_params["max_tokens"]
                del generation_params["max_tokens"]
            generation_params.pop("temperature", None)

        # Combine system message with messages if provided
        if system:
            messages = MessageHistory([system] + messages)

        await self.rate_limiter.acquire()
        response = await self.async_client.chat.completions.create(messages=messages.to_openai(), **generation_params)

        # Extract message and add usage data
        message = Message.from_openai(response.choices[0].message)

        # Add usage data from the response
        if hasattr(response, "usage") and response.usage:
            message.usage_metadata.update(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )
            # Capture cached prompt tokens if available (e.g., DashScope/Qwen)
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached = None
            if details is not None:
                cached = getattr(details, "cached_tokens", None)
                if cached is None and isinstance(details, dict):
                    cached = details.get("cached_tokens")
            if cached is not None:
                message.usage_metadata["cache_read_input_tokens"] = cached
        else:
            logging.getLogger(__name__).warning(
                "OpenAIProvider.generate: response contains no usage metadata; billing may be skipped"
            )

        return message
