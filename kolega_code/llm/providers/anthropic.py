import json
import os
from typing import Any, AsyncContextManager, Dict, List, Optional

import tiktoken
from anthropic import Anthropic, AsyncAnthropic

from ..models import Message, MessageChunk, MessageHistory, ToolDefinition
from ..specs import build_thinking_request_params, get_model_specs
from ..tool_execution_ids import ToolExecutionIdRegistry
from .base import BaseLLMProvider
from .models import GenerationParams, TokenCount


class AnthropicStreamWrapper:
    def __init__(self, anthropic_stream, provider_name: str = "anthropic"):
        self.anthropic_stream = anthropic_stream
        self.provider_name = provider_name
        self.generator = None
        self._closed = False

        # Track tool calls being streamed
        self.tool_execution_ids = ToolExecutionIdRegistry()
        self.current_tool_calls = {}  # Maps tool_call_id to accumulated data
        self.tool_call_order = []  # Track order of tool calls
        self.current_block_index = None  # Track which content block we're processing

    async def __aenter__(self):
        self.generator = await self.anthropic_stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.anthropic_stream.__aexit__(exc_type, exc_val, exc_tb)

    def __aiter__(self):
        if self.generator is None:
            raise RuntimeError("Must use 'async with' before iterating")
        return self

    async def __anext__(self):
        if self.generator is None:
            raise RuntimeError("Must use 'async with' before iterating")

        try:
            chunk = await self.generator.__anext__()

            # Handle content_block_start events for tool use
            if chunk.type == "content_block_start" and hasattr(chunk, "content_block"):
                if chunk.content_block.type == "tool_use":
                    # Track this new tool call
                    tool_id = chunk.content_block.id
                    self.current_tool_calls[tool_id] = {
                        "id": tool_id,
                        "name": chunk.content_block.name,
                        "input_json": "",
                        "block_index": chunk.index if hasattr(chunk, "index") else len(self.tool_call_order),
                        "execution_id": self.tool_execution_ids.get_or_create(tool_id),
                    }
                    self.tool_call_order.append(tool_id)
                    self.current_block_index = chunk.index if hasattr(chunk, "index") else None

            # Handle content_block_delta events for tool use input
            elif chunk.type == "content_block_delta" and hasattr(chunk, "delta"):
                if chunk.delta.type == "input_json_delta" and hasattr(chunk, "index"):
                    # Find the tool call by block index
                    for tool_id, tool_data in self.current_tool_calls.items():
                        if tool_data["block_index"] == chunk.index:
                            # Accumulate the JSON input
                            tool_data["input_json"] += chunk.delta.partial_json
                            break

            message_chunk = MessageChunk.from_anthropic(chunk)
            if message_chunk.type == "tool_use_start" and message_chunk.tool_call_delta:
                tool_id = message_chunk.tool_call_delta.get("id")
                tool_data = self.current_tool_calls.get(tool_id)
                if tool_data and tool_data.get("execution_id"):
                    message_chunk.tool_call_delta["execution_id"] = tool_data["execution_id"]

            return message_chunk

        except StopAsyncIteration:
            raise

    async def get_final_message(self):
        message = Message.from_anthropic(
            await self.generator.get_final_message(),
            tool_execution_ids=self.tool_execution_ids,
        )
        if message.usage_metadata:
            message.usage_metadata["provider"] = self.provider_name
        return message


class AnthropicProvider(BaseLLMProvider):
    SYSTEM_OVERHEAD = 4
    MESSAGE_OVERHEAD = 3
    TOOL_DEFINITION_OVERHEAD = 65

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
        provider_name: str = "anthropic",
    ):
        super().__init__(api_key, max_retries, requests_per_minute, tokens_per_minute, base_url)
        self.provider_name = provider_name
        self.async_client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self.sync_client = Anthropic(api_key=api_key, base_url=base_url)
        
        # OpenAI-compatible Anthropic-shaped APIs do not expose messages/count_tokens,
        # so local counting is only a preflight context-size estimate for those models.
        # Billing/accounting must use provider response usage metadata instead.
        self.use_local_token_counting = (
            provider_name in {"moonshot", "deepseek", "kimi_coding"}
            or os.getenv('ANTHROPIC_USE_LOCAL_TOKEN_COUNTING', 'false').lower() == 'true'
        )

    @property
    def retry_decorator(self):
        """Get retry decorator with configured max retries"""
        return self.get_retry_decorator()

    def _prepare_generation_params(self, params: Optional[GenerationParams] = None) -> Dict[str, Any]:
        """Convert common parameters to provider-specific format"""
        generation_params = {
            "model": "claude-opus-4-8",  # Default model
            "max_tokens": 1024,  # Default max tokens
        }

        if params:
            if params.temperature is not None:
                generation_params["temperature"] = params.temperature
            if params.max_completion_tokens is not None:
                generation_params["max_tokens"] = params.max_completion_tokens
            if params.tools:
                generation_params["tools"] = [t.to_anthropic() for t in params.tools]
                generation_params["tool_choice"] = {"type": "auto"}

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

    def _sanitize_generation_params(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove parameters unsupported by the selected Anthropic model."""
        model = generation_params.get("model")
        if self.provider_name != "anthropic" or not isinstance(model, str):
            return generation_params

        try:
            model_specs = get_model_specs(self.provider_name, model)
        except ValueError:
            return generation_params

        if model_specs.get("supports_temperature", True) is False:
            generation_params.pop("temperature", None)

        return generation_params

    async def count_tokens(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        model: Optional[str] = None,
        tools: List[ToolDefinition] = None,
        **kwargs,
    ) -> TokenCount:
        tools = tools or []
        if self.use_local_token_counting:
            # Use local tiktoken-based counting (no API call). This is an
            # estimate for context management, not authoritative billing usage.
            return self._count_tokens_local(messages, system, model, tools)
        else:
            # Use Anthropic API for token counting
            await self.rate_limiter.acquire()
            count = await self.async_client.messages.count_tokens(
                messages=messages.to_anthropic(),
                system=[c.to_anthropic() for c in system.content],
                model=model,
                tools=[t.to_anthropic() for t in tools],
                **kwargs,
            )

            # The API now only returns input_tokens
            return TokenCount(
                input_tokens=count.input_tokens,
                output_tokens=None,  # MessageTokensCount no longer includes output_tokens
            )

    def _count_tokens_local(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        model: Optional[str] = None,
        tools: List[ToolDefinition] = None,
    ) -> TokenCount:
        """Count tokens locally using tiktoken with p50k_base encoding.

        This provides a fast approximation without making an API call.
        Uses minimal overhead and direct text encoding for better accuracy.
        Handles images by estimating token cost based on data size.

        Args:
            messages: Message history to count tokens for
            system: Optional system message
            model: Optional model name (not used for local counting)
            tools: Optional tool definitions

        Returns:
            TokenCount object with estimated input token count
        """
        encoding = tiktoken.get_encoding("p50k_base")
        num_tokens = 0
        tools = tools or []

        if system:
            num_tokens += self.SYSTEM_OVERHEAD
            num_tokens += self._count_message_content_tokens(encoding, system.content)

        for message in messages:
            num_tokens += self.MESSAGE_OVERHEAD
            num_tokens += self._count_message_content_tokens(encoding, message.content)

        for tool in tools:
            num_tokens += self._count_value_tokens(encoding, tool.to_anthropic())
            num_tokens += self.TOOL_DEFINITION_OVERHEAD

        return TokenCount(input_tokens=num_tokens, output_tokens=None)

    def _count_message_content_tokens(self, encoding, content: Any) -> int:
        if isinstance(content, str):
            return len(encoding.encode(content))

        if isinstance(content, list):
            return sum(self._count_content_block_tokens(encoding, block) for block in content)

        return self._count_value_tokens(encoding, content)

    def _count_content_block_tokens(self, encoding, block: Any) -> int:
        if hasattr(block, "text"):
            return len(encoding.encode(block.text))

        if getattr(block, "type", None) == "image_url":
            data = getattr(block, "data", None)
            if isinstance(data, str):
                return self._estimate_image_tokens(len(data))

        if getattr(block, "type", None) == "tool_result":
            content = getattr(block, "content", "")
            return self._count_message_content_tokens(encoding, content)

        if hasattr(block, "thinking"):
            return len(encoding.encode(block.thinking))

        if hasattr(block, "data"):
            return len(encoding.encode(str(block.data)))

        if hasattr(block, "to_anthropic"):
            return self._count_value_tokens(encoding, block.to_anthropic())

        return self._count_value_tokens(encoding, block)

    def _count_value_tokens(self, encoding, value: Any) -> int:
        if value is None:
            return 0

        if isinstance(value, str):
            return len(encoding.encode(value))

        if isinstance(value, (int, float, bool)):
            return len(encoding.encode(str(value)))

        if isinstance(value, list):
            return 2 + sum(self._count_value_tokens(encoding, item) for item in value)

        if isinstance(value, dict):
            if value.get("type") == "image":
                source = value.get("source") or {}
                data = source.get("data")
                if isinstance(data, str):
                    return self._estimate_image_tokens(len(data))

            total = 2
            for key, item in value.items():
                total += len(encoding.encode(str(key)))
                total += self._count_value_tokens(encoding, item)
            return total

        return len(encoding.encode(json.dumps(value, ensure_ascii=False, default=str)))

    def _estimate_image_tokens(self, base64_data_length: int) -> int:
        """Estimate image token cost based on base64 data length.

        Anthropic charges for images based on their dimensions after resizing.
        Since we don't decode images (performance), we estimate based on data size.

        Empirically observed from tests:
        - Tiny images (96 chars base64, 1x1 px): ~25 tokens
        - Small images (~50-200KB base64): ~200-800 tokens
        - Medium images (~200-800KB base64): ~800-2000 tokens
        - Large images (~800KB+ base64): ~2000-4000 tokens

        Formula uses square root scaling for better approximation across sizes:
        tokens ≈ 20 + sqrt(base64_length * 6)

        This gives:
        - 96 chars → 20 + sqrt(576) = 44 tokens (~25 actual)
        - 50KB (68K chars) → 20 + sqrt(408K) = 659 tokens
        - 200KB (273K chars) → 20 + sqrt(1.6M) = 1285 tokens
        - 800KB (1.1M chars) → 20 + sqrt(6.4M) = 2549 tokens

        Args:
            base64_data_length: Length of base64 encoded image data

        Returns:
            Estimated token count for the image
        """
        import math

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
        """Generate a streaming response from Anthropic

        Returns a context manager that provides an async iterator when entered.
        The context manager also provides get_final_message() to retrieve the
        complete message after streaming.
        """
        generation_params = self._prepare_generation_params(params)
        generation_params.update(kwargs)
        self._apply_thinking_params(generation_params, params)
        generation_params = self._sanitize_generation_params(generation_params)

        await self.rate_limiter.acquire()

        # Return the stream context manager
        return AnthropicStreamWrapper(
            self.async_client.messages.stream(
                messages=messages.to_anthropic(),
                system=[c.to_anthropic() for c in system.content],
                **generation_params,
            ),
            provider_name=self.provider_name,
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
        generation_params = self._sanitize_generation_params(generation_params)

        await self.rate_limiter.acquire()
        response = await self.async_client.messages.create(
            messages=messages.to_anthropic(),
            system=[c.to_anthropic() for c in system.content],
            **generation_params,
        )
        message = Message.from_anthropic(response)
        if message.usage_metadata:
            message.usage_metadata["provider"] = self.provider_name
        return message
