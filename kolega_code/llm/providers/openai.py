from typing import Any, AsyncContextManager, Dict, List, Optional
from weakref import WeakKeyDictionary
import asyncio
import json
import logging
import math
import threading

import tiktoken
from openai import AsyncOpenAI

from ..models import (
    ImageBlock,
    Message,
    MessageChunk,
    MessageHistory,
    RedactedThinkingBlock,
    ResponsesReasoningBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from ..specs import build_thinking_request_params
from ..timeouts import streaming_timeout
from ..tool_execution_ids import ToolExecutionIdRegistry
from .base import BaseLLMProvider
from .models import GenerationParams, TokenCount

# tiktoken caches Encoding objects in a module-level registry after the first
# (BPE-table-loading) call, but we also hold our own reference so the hot
# count_tokens path never does even a registry dict lookup. cl100k_base is the
# de-facto encoding for every OpenAI-compatible provider (DeepSeek, xAI, Fireworks,
# ... have no native tiktoken table); the count is an estimate for context
# management, not billing.
_ENCODING_NAME = "cl100k_base"
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoding


class OpenAIStreamWrapper:
    def __init__(self, openai_stream, requested_include_usage: bool = False, provider_name: str = "openai"):
        self.openai_stream = openai_stream
        # Accumulate streamed deltas into lists and ''.join once at finalize. A
        # running ``str += delta`` is O(n^2) here because the buffer is an instance
        # attribute (CPython's in-place concat optimization only fires for locals
        # with refcount 1), and DeepSeek max-effort reasoning streams build enormous
        # buffers over thousands of tiny chunks. Lists make accumulation O(n).
        self._content_parts: List[str] = []
        self._reasoning_parts: List[str] = []
        # Per-index tool-call: the first delta's object (kept for id/name/index) plus
        # its argument fragments collected separately so the JSON arg string (which can
        # be large, e.g. a file-writing tool) is also joined once instead of grown.
        self.final_tool_calls = {}
        self._tool_call_arg_parts: Dict[int, List[str]] = {}
        self.stop_reason = None
        self.usage_data = None
        self.tool_execution_ids = ToolExecutionIdRegistry()
        self.provider_name = provider_name

        self._closed = False
        self._requested_include_usage = requested_include_usage

    @property
    def final_content(self) -> str:
        return "".join(self._content_parts)

    @property
    def final_reasoning_content(self) -> str:
        return "".join(self._reasoning_parts)

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
                        self._content_parts.append(content)

                    reasoning_content = (
                        getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
                    )
                    if reasoning_content:
                        self._reasoning_parts.append(reasoning_content)

                    for tool_call in getattr(delta, "tool_calls", []) or []:
                        index = tool_call.index

                        if index not in self.final_tool_calls:
                            # Keep the first delta's object for id/name/index; seed its
                            # argument fragments and clear the live attribute (the joined
                            # value is written back in get_final_message).
                            self.final_tool_calls[index] = tool_call
                            self._tool_call_arg_parts[index] = [tool_call.function.arguments or ""]
                        else:
                            fragment = tool_call.function.arguments
                            if fragment:
                                self._tool_call_arg_parts[index].append(fragment)

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
        # Materialize the accumulated tool-call argument fragments into each delta
        # object exactly once (the O(n) join the per-chunk path deliberately avoided).
        for index, parts in self._tool_call_arg_parts.items():
            tool_call = self.final_tool_calls.get(index)
            if tool_call is not None:
                tool_call.function.arguments = "".join(parts)

        message = Message.from_openai_stream(
            role="assistant",
            content=self.final_content,
            reasoning_content=self.final_reasoning_content,
            tool_calls=self.final_tool_calls,
            stop_reason=self.stop_reason,
            tool_execution_ids=self.tool_execution_ids,
        )

        # Add usage data if available
        if self.usage_data:
            message.usage_metadata.update(self.usage_data)
        message.usage_metadata["provider"] = self.provider_name
        if not self.usage_data:
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
        # Per-message / per-tool token-count memos. count_tokens runs every agent-loop
        # iteration over the whole history; without this it re-encodes the entire
        # conversation each time (O(history) on the event loop, the streaming-turn
        # freeze on slow machines). Keyed by object identity so unchanged messages are
        # counted once; a cheap length fingerprint catches in-place edits (e.g. an
        # oversized tool result being truncated). Compaction/adaptation/repair produce
        # new objects, which miss naturally.
        self._message_token_memo: "WeakKeyDictionary[Message, tuple]" = WeakKeyDictionary()
        self._tool_token_memo: "WeakKeyDictionary[ToolDefinition, int]" = WeakKeyDictionary()
        # Guards the memos when count_tokens is offloaded to a worker thread.
        self._token_memo_lock = threading.Lock()

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
        # Combine system message with messages if provided
        all_messages = ([system] + list(messages)) if system else list(messages)

        # tiktoken encoding is pure-CPU work. The agent loop runs on the Textual/asyncio
        # event loop, so encoding inline freezes the UI — worst on memo-cold passes
        # (compaction, /model switch) that re-encode the whole history at once. Offload
        # to a worker thread so a count never blocks rendering. The per-instance lock
        # keeps the (non-thread-safe) WeakKeyDictionary memos consistent if two agents
        # share this provider and count concurrently from different threads.
        num_tokens = await asyncio.to_thread(self._count_tokens_sync, all_messages, tools)
        return TokenCount(input_tokens=num_tokens)

    def _count_tokens_sync(self, all_messages: List[Message], tools: Optional[List[ToolDefinition]]) -> int:
        encoding = _get_encoding()
        self._ensure_token_memos()
        with self._token_memo_lock:
            # Sum memoized per-message counts: only new or changed messages are re-encoded.
            num_tokens = sum(self._memoized_message_tokens(encoding, message) for message in all_messages)
            # Tool definitions don't mutate within a session, so memoize per definition by identity.
            if tools:
                num_tokens += sum(self._memoized_tool_tokens(encoding, tool) for tool in tools)
        return num_tokens

    def _ensure_token_memos(self) -> None:
        # Instances built via __new__ (benchmarks/tests) skip __init__, so create lazily.
        if getattr(self, "_message_token_memo", None) is None:
            self._message_token_memo = WeakKeyDictionary()
        if getattr(self, "_tool_token_memo", None) is None:
            self._tool_token_memo = WeakKeyDictionary()
        if getattr(self, "_token_memo_lock", None) is None:
            self._token_memo_lock = threading.Lock()

    def _memoized_message_tokens(self, encoding, message) -> int:
        fingerprint = self._message_fingerprint(message)
        try:
            cached = self._message_token_memo.get(message)
        except TypeError:  # message not weak-referenceable / hashable
            cached = None
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        value = self._count_message_tokens(encoding, message)
        try:
            self._message_token_memo[message] = (fingerprint, value)
        except TypeError:
            pass
        return value

    def _memoized_tool_tokens(self, encoding, tool) -> int:
        try:
            cached = self._tool_token_memo.get(tool)
        except TypeError:
            cached = None
        if cached is not None:
            return cached
        # OpenAI uses a highly optimized internal format (not JSON); empirically their
        # token count is ~79% of the raw JSON token count.
        value = int(len(encoding.encode(json.dumps(tool.to_openai()))) * 0.79)
        try:
            self._tool_token_memo[tool] = value
        except TypeError:
            pass
        return value

    def _count_message_tokens(self, encoding, message) -> int:
        """Token count for one message (role + content blocks). Memoized by the caller."""
        num_tokens = 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n format
        if hasattr(message, "role") and message.role:
            num_tokens += len(encoding.encode(message.role))
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
                    # Handle reasoning blocks. DeepSeek (and other reasoning models)
                    # replay prior reasoning back to the same provider, so it counts
                    # toward the input budget. Without these branches ThinkingBlocks
                    # (.thinking, not .text) scored 0, the gauge undercounted, and
                    # auto-compaction fired late while payloads stayed large.
                    elif isinstance(item, ThinkingBlock):
                        num_tokens += len(encoding.encode(item.thinking or ""))
                        if item.signature:
                            num_tokens += len(encoding.encode(str(item.signature)))
                    elif isinstance(item, RedactedThinkingBlock):
                        num_tokens += len(encoding.encode(str(item.data)))
                    elif isinstance(item, ResponsesReasoningBlock):
                        num_tokens += len(encoding.encode(str(item.encrypted_content or "")))
                        for part in item.summary:
                            num_tokens += len(encoding.encode(str(part)))
                    # Handle tool calls. OpenAI's prompt accounting uses a compact
                    # internal representation for assistant tool calls rather than
                    # charging the full Chat Completions JSON wrapper. Counting the
                    # stable fields (id, function name, serialized arguments) tracks
                    # the API much more closely for resumed/provider-shaped history.
                    elif isinstance(item, ToolCall):
                        tool_call_payload = item.to_openai()
                        function_payload = tool_call_payload.get("function", {})
                        arguments = str(function_payload.get("arguments") or "")
                        num_tokens += len(encoding.encode(str(tool_call_payload.get("id") or "")))
                        num_tokens += len(encoding.encode(str(function_payload.get("name") or item.name)))
                        num_tokens += len(encoding.encode(arguments))
                        num_tokens += 1  # Compact formatting overhead for tool calls
                    # Handle tool results
                    elif isinstance(item, ToolResult):
                        # Tool results contain content that needs to be counted.
                        # OpenAI tool messages are text-only, but image-bearing
                        # tool results are serialized as a follow-up user image
                        # message, so nested images contribute image tokens.
                        if isinstance(item.content, str):
                            num_tokens += len(encoding.encode(item.content))
                        elif isinstance(item.content, list):
                            for result_item in item.content:
                                if isinstance(result_item, ImageBlock):
                                    num_tokens += self._estimate_image_tokens(len(result_item.data))
                                elif hasattr(result_item, "data") and hasattr(result_item, "media_type"):
                                    num_tokens += self._estimate_image_tokens(len(result_item.data))
                                elif hasattr(result_item, "text") and result_item.text:
                                    num_tokens += len(encoding.encode(result_item.text))
                        num_tokens += 2  # Minimal formatting overhead for tool results
        return num_tokens

    def _message_fingerprint(self, message):
        """Cheap (lengths-only) signature mirroring _count_message_tokens' inputs.

        The token count for a fixed message structure changes only when an encoded
        length changes, so a change in this fingerprint is necessary and sufficient to
        invalidate a memoized count for the same object (e.g. in-place truncation of an
        oversized tool result). Structural/type changes arrive as new objects (identity
        miss), so they need not be captured here."""
        role = getattr(message, "role", "") or ""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return (role, len(content))
        if isinstance(content, list):
            return (role, tuple(self._block_fingerprint(item) for item in content))
        return (role, None)

    def _block_fingerprint(self, item):
        # Branch order mirrors _count_message_tokens so classification stays aligned.
        if hasattr(item, "text") and getattr(item, "text", None):
            return ("txt", len(item.text))
        if isinstance(item, dict) and "text" in item:
            return ("dtxt", len(item.get("text") or ""))
        if isinstance(item, ImageBlock):
            return ("img", len(item.data))
        if hasattr(item, "data") and hasattr(item, "media_type"):
            return ("img", len(item.data))
        if isinstance(item, ThinkingBlock):
            return ("think", len(item.thinking or ""), len(str(item.signature or "")))
        if isinstance(item, RedactedThinkingBlock):
            return ("rthink", len(str(item.data)))
        if isinstance(item, ResponsesReasoningBlock):
            return ("rreason", len(str(item.encrypted_content or "")), tuple(len(str(p)) for p in item.summary))
        if isinstance(item, ToolCall):
            payload = item.to_openai()
            function_payload = payload.get("function", {})
            return (
                "tc",
                len(str(payload.get("id") or "")),
                len(str(function_payload.get("name") or item.name)),
                len(str(function_payload.get("arguments") or "")),
            )
        if isinstance(item, ToolResult):
            if isinstance(item.content, str):
                return ("tr", len(item.content))
            if isinstance(item.content, list):
                return ("tr", tuple(self._block_fingerprint(ri) for ri in item.content))
            return ("tr", None)
        return ("?",)

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

        # Per-request streaming timeout bounds the inter-chunk read wait (see
        # kolega_code/llm/timeouts.py): a stalled connection fails in minutes and is
        # retried, instead of hanging on the SDK's 600s default.
        return OpenAIStreamWrapper(
            await self.async_client.chat.completions.create(
                messages=messages.to_openai(provider=self.provider_name, model=generation_params["model"]),
                timeout=streaming_timeout(),
                **generation_params,
            ),
            requested_include_usage=True,
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
        response = await self.async_client.chat.completions.create(
            messages=messages.to_openai(provider=self.provider_name, model=generation_params["model"]),
            **generation_params,
        )

        # Extract message and add usage data
        message = Message.from_openai(response.choices[0].message)
        message.usage_metadata["provider"] = self.provider_name

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
