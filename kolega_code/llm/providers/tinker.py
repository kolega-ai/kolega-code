"""Native Tinker SamplingClient provider (on-policy RL trajectory source).

Unlike the ``thinking_machines`` provider (Tinker's Anthropic-compatible
serverless endpoint), this provider drives Tinker's **native sampling API**:
token-level ``ModelInput`` rendered client-side by the model's official
renderer, sampled token ids with per-token behavior logprobs, stop reasons, and
prefix-cache metadata. That complete training record is what agentic-RL needs
and what the Anthropic-compatible endpoint does not expose.

The provider integrates with Kolega's normal LLM abstraction — messages, tools,
reasoning, token counting, and the agent's streaming event protocol — so the
regular ``ask`` loop can generate on-policy trajectories. A structured trace
record is emitted per native call when a sink is attached.

Dependencies: the optional ``[tinker]`` extra (``tinker-cookbook`` +
``tml-renderers``; the native renderer stack needs torch). Importing this
module never requires them — the missing-extra case fails with a clear error at
client construction.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncContextManager, Callable, Dict, List, Optional, Sequence

from ..exceptions import LLMInvalidRequestError
from ..ledger import current_llm_call_origin
from ..models import (
    Message,
    MessageChunk,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
)
from ..specs import MODEL_SPECS, get_model_specs
from ..usage import attach_normalized_usage
from .base import BaseLLMProvider
from .models import GenerationParams, TokenCount
from .tinker_trace import TinkerTraceRecord

logger = logging.getLogger(__name__)

# The native stack is optional (the ``[tinker]`` extra). ``tinker`` is the SDK;
# the cookbook provides the official chat-template renderers. Guarded import so
# importing this module (and therefore any session that merely constructs a
# client) never requires the extra.
try:  # pragma: no cover - exercised by the missing-extra path
    import tinker as _tinker_module  # pyright: ignore[reportMissingImports]
    from tinker_cookbook.model_info import get_model_attributes as _get_model_attributes  # pyright: ignore[reportMissingImports]
    from tinker_cookbook.renderers import get_renderer as _cookbook_get_renderer  # pyright: ignore[reportMissingImports]
    from tinker_cookbook.tokenizer_utils import get_tokenizer as _cookbook_get_tokenizer  # pyright: ignore[reportMissingImports]

    _tinker: Any = _tinker_module
    _NATIVE_STACK_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on install extras
    _tinker = None
    _NATIVE_STACK_AVAILABLE = False

_MISSING_STACK_MESSAGE = (
    "The tinker provider needs the optional native Tinker stack. Install it with: pip install 'kolega-code[tinker]'"
)

# Named effort -> tml-renderers effort float, matching Tinker's documented
# mapping for its compatible endpoints (max aliases xhigh server-side).
_EFFORT_FLOATS = {
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}

# Conservative context fallback for checkpoints whose base is not catalogued.
_FALLBACK_CONTEXT_LENGTH = 65536


def _require_native_stack() -> None:
    if not _NATIVE_STACK_AVAILABLE:
        raise LLMInvalidRequestError(_MISSING_STACK_MESSAGE, provider="tinker")


def _effort_float(effort: str) -> Optional[float]:
    return _EFFORT_FLOATS.get(effort)


def _tool_openai_specs(tools: Sequence[ToolDefinition]) -> List[Dict[str, Any]]:
    return [t.to_openai() for t in tools]


def _renderer_for_base_model(base_model: str, effort: Optional[str]) -> Any:
    """Build the official chat renderer for a Tinker base model.

    Renderer and tokenizer both come from the cookbook's model registry: the
    ``tokenizer_utils`` package returns the right tokenizer per model (the
    tml-renderers adapter for Inkling), and ``model_info`` maps the base model
    to its recommended renderer, preferring the ``*_disable_thinking`` variant
    when ``effort == "none"`` and one is offered.
    """
    tokenizer = _cookbook_get_tokenizer(base_model)
    attributes = _get_model_attributes(base_model)
    renderers = list(attributes.recommended_renderers)
    if effort == "none":
        renderer_name = next((name for name in renderers if "disable_thinking" in name), renderers[0])
    else:
        renderer_name = renderers[0]
    renderer = _cookbook_get_renderer(renderer_name, tokenizer, model_name=base_model)
    # tml-renderers (Inkling) accepts an ``effort`` float per call; record that
    # so ``_build_prompt`` can pass it through.
    import inspect

    renderer._accepts_effort = "effort" in inspect.signature(renderer.build_generation_prompt).parameters  # type: ignore[attr-defined]
    return renderer


def _build_prompt(
    renderer: Any,
    openai_messages: List[Dict[str, Any]],
    system_text: str,
    tools: Sequence[ToolDefinition],
    effort: Optional[str],
) -> Any:
    """Render messages (plus tool declarations) into a native ``ModelInput``."""
    from tinker_cookbook.third_party.openai_compat import (  # pyright: ignore[reportMissingImports]
        openai_messages_to_tinker,
        openai_tools_to_tinker,
    )

    # Normalize OpenAI-shaped dicts into the cookbook's Message format (tool
    # calls become ToolCall objects; text content is stringified).
    messages = openai_messages_to_tinker(list(openai_messages))
    if tools:
        tool_specs = openai_tools_to_tinker(_tool_openai_specs(tools))
        prefix = renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt=system_text)
        prompt_messages = prefix + messages
    elif system_text:
        prompt_messages = [{"role": "system", "content": system_text}, *messages]
    else:
        prompt_messages = messages

    if effort and effort != "none" and getattr(renderer, "_accepts_effort", False):
        return renderer.build_generation_prompt(prompt_messages, effort=_effort_float(effort))
    return renderer.build_generation_prompt(prompt_messages)


def _openai_messages(messages: MessageHistory, system: Optional[Message]) -> tuple[List[Dict[str, Any]], str]:
    """Convert Kolega history to OpenAI-shaped dicts plus the system text.

    ``MessageHistory.to_openai`` already partitions ``ToolResult`` blocks into
    ``role="tool"`` messages, matching the shape the cookbook renderers expect.
    """
    system_text = ""
    if system is not None:
        system_text = system.get_text_content()
    return messages.to_openai(), system_text


def _parsed_to_message(parsed: Any, stop_reason: str) -> Message:
    """Convert a renderer-parsed response (OpenAI-shaped dict) to a Kolega Message.

    Handles ``content`` (str or part list), ``reasoning_content``-style thinking
    fields when the renderer separates them, and ``tool_calls``. Tool-call ids
    are fabricated when the renderer does not emit one.
    """
    blocks: List[Any] = []
    message_tool_calls: List[ToolCall] = []

    reasoning = (
        parsed.get("reasoning_content") if isinstance(parsed, dict) else getattr(parsed, "reasoning_content", None)
    )
    if reasoning:
        blocks.append(ThinkingBlock(thinking=str(reasoning)))

    content = parsed.get("content") if isinstance(parsed, dict) else getattr(parsed, "content", None)
    if isinstance(content, str):
        if content:
            blocks.append(TextBlock(text=content))
    elif isinstance(content, list):
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type == "text":
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", "")
                if text:
                    blocks.append(TextBlock(text=str(text)))
            elif part_type in ("thinking", "reasoning"):
                thinking = (
                    part.get("text") or part.get("thinking")
                    if isinstance(part, dict)
                    else getattr(part, "text", None) or getattr(part, "thinking", None)
                )
                if thinking:
                    blocks.append(ThinkingBlock(thinking=str(thinking)))
            elif isinstance(part, str) and part:
                blocks.append(TextBlock(text=part))

    tool_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else getattr(parsed, "tool_calls", None)
    if tool_calls:
        for index, call in enumerate(tool_calls):
            function = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
            if isinstance(function, dict):
                name = str(function.get("name") or "")
                arguments = function.get("arguments") or "{}"
            else:
                name = str(getattr(function, "name", "") or "")
                arguments = getattr(function, "arguments", None) or "{}"
            try:
                tool_input: Any = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                tool_input = {"raw": arguments}
            raw_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            call_id = str(raw_id) if raw_id else f"call_{index}"
            tool_call = ToolCall(id=call_id, name=name, input=tool_input)
            message_tool_calls.append(tool_call)
            blocks.append(tool_call)

    if not blocks:
        blocks.append(TextBlock(text=""))
    return Message(
        role="assistant",
        content=blocks,
        stop_reason="tool_use" if message_tool_calls else stop_reason,
        tool_calls=message_tool_calls,
    )


def _kolega_stop_reason(sdk_reason: str, termination: Any) -> str:
    """Map SDK/renderer stop signals onto Kolega's stop-reason vocabulary."""
    termination_name = str(getattr(termination, "name", termination) if termination is not None else "")
    if termination_name and "MALFORMED" in termination_name:
        return "stop_sequence"
    if sdk_reason == "length":
        return "max_tokens"
    return "stop_sequence"


class TinkerProvider(BaseLLMProvider):
    """Kolega LLM provider over Tinker's native ``SamplingClient``."""

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
        provider_name: str = "tinker",
        model: Optional[str] = None,
        trace_sink: Optional[Callable[[TinkerTraceRecord], None]] = None,
    ):
        super().__init__(
            api_key,
            max_retries=max_retries,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            base_url=base_url,
        )
        self.provider_name = provider_name
        self.model = model
        self.trace_sink = trace_sink
        # The SDK manages its own transport (overridable via TINKER_BASE_URL);
        # the attribute only feeds diagnostics display.
        self.base_url = base_url or "https://tinker.thinkingmachines.dev/services/tinker-prod"
        # Present for parity with the other provider classes (union attribute
        # access in tests); the Tinker transport is SDK-managed, so there is no
        # raw SDK client to expose.
        self.async_client = None
        self._sampling_client: Any = None
        self._service: Any = None
        self._client_model: Optional[str] = None
        self._base_model: Optional[str] = None
        self._renderer: Any = None
        _require_native_stack()

    @property
    def retry_decorator(self):
        return self.get_retry_decorator()

    async def _ensure_client(self, model: str) -> None:
        """Lazily create the SamplingClient for ``model`` (base id or tinker:// path).

        One provider instance serves exactly one model: the agent builds one
        client per model config. A second distinct model is a caller bug.
        """
        if self._sampling_client is not None:
            if model != self._client_model:
                raise LLMInvalidRequestError(
                    f"tinker provider is bound to model '{self._client_model}', got '{model}'. "
                    "Use a separate LLMClient per model.",
                    provider=self.provider_name,
                )
            return
        service = _tinker.ServiceClient(api_key=self.api_key)
        if model.startswith("tinker://"):
            self._sampling_client = await service.create_sampling_client_async(model_path=model)
        else:
            self._sampling_client = await service.create_sampling_client_async(base_model=model)
        self._client_model = model
        try:
            self._base_model = await self._sampling_client.get_base_model_async()
        except Exception:  # noqa: BLE001 - base models are known directly
            self._base_model = None
        if not self._base_model:
            self._base_model = model if not model.startswith("tinker://") else model

    async def _get_renderer(self, model: str, effort: Optional[str]) -> Any:
        if self._renderer is None:
            base_model = self._base_model or model
            self._renderer = await asyncio.to_thread(_renderer_for_base_model, base_model, effort)
        return self._renderer

    def _context_length(self, model: str) -> int:
        base_model = self._base_model or model
        specs = MODEL_SPECS.get(("tinker", base_model))
        if specs is None:
            try:
                specs = get_model_specs("tinker", model)
            except ValueError:
                specs = None
        return int((specs or {}).get("context_length") or _FALLBACK_CONTEXT_LENGTH)

    async def _sample(
        self,
        messages: MessageHistory,
        system: Optional[Message],
        params: Optional[GenerationParams],
        model: str,
    ) -> tuple[Any, Any, Any, Any, Dict[str, int]]:
        """Render and sample one response; returns
        ``(sequence, parsed, termination, model_input, usage)``."""
        await self._ensure_client(model)
        effort = params.thinking if params else None
        renderer = await self._get_renderer(model, effort)
        openai_messages, system_text = _openai_messages(messages, system)
        tools = list(params.tools) if params and params.tools else []
        prompt = await asyncio.to_thread(_build_prompt, renderer, openai_messages, system_text, tools, effort)

        stop = renderer.get_stop_sequences()
        requested_max = params.max_completion_tokens if params else None
        context = self._context_length(model)
        prompt_tokens = prompt.length
        if requested_max is None or requested_max > context - prompt_tokens:
            requested_max = max(1, context - prompt_tokens)
        temperature = params.temperature if params and params.temperature is not None else 1.0

        sampling_params = _tinker.types.SamplingParams(
            max_tokens=requested_max,
            temperature=temperature,
            stop=stop,
        )
        response = await self._sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        )
        sequence = response.sequences[0]
        parsed, termination = renderer.parse_response(sequence.tokens)
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": len(sequence.tokens),
            "prompt_cache_hit_tokens": int(getattr(response, "prompt_cache_hit_tokens", 0) or 0),
        }
        return sequence, parsed, termination, prompt, usage

    def _emit_trace(
        self,
        model: str,
        sequence: Any,
        termination: Any,
        prompt: Any,
        usage: Dict[str, int],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> None:
        if self.trace_sink is None:
            return
        origin = current_llm_call_origin()
        record = TinkerTraceRecord(
            model=model,
            base_model=self._base_model or model,
            request_role=origin.to_payload() if origin is not None else None,
            prompt_tokens=list(prompt.to_ints()),
            sampled_tokens=list(sequence.tokens),
            sampled_text=_renderer_decode(self._renderer, sequence.tokens),
            logprobs=list(sequence.logprobs) if sequence.logprobs is not None else [],
            stop_reason=str(sequence.stop_reason),
            termination=str(getattr(termination, "name", termination)),
            cache_hit_tokens=int(usage.get("prompt_cache_hit_tokens") or 0),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            self.trace_sink(record)
        except Exception:  # noqa: BLE001 - a sink failure must never break the loop
            logger.warning("tinker trace sink raised", exc_info=True)

    async def generate(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs: Any,
    ) -> Message:
        model = str(kwargs.get("model") or self.model or "")
        if not model:
            raise LLMInvalidRequestError("tinker provider requires a model", provider=self.provider_name)
        sequence, parsed, termination, prompt, usage = await self._sample(messages, system, params, model)

        message = _parsed_to_message(parsed, _kolega_stop_reason(str(sequence.stop_reason), termination))
        message.usage_metadata = {
            "provider": self.provider_name,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cache_read_input_tokens": usage["prompt_cache_hit_tokens"],
        }
        attach_normalized_usage(message, self.provider_name, model)

        self._emit_trace(
            model,
            sequence,
            termination,
            prompt,
            usage,
            params.temperature if params else None,
            params.max_completion_tokens if params else None,
        )
        return message

    async def count_tokens(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        model: Optional[str] = None,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> TokenCount:
        resolved_model = str(model or self.model or "")
        if not resolved_model:
            raise LLMInvalidRequestError("tinker provider requires a model", provider=self.provider_name)
        await self._ensure_client(resolved_model)
        effort = kwargs.get("thinking")
        renderer = await self._get_renderer(resolved_model, effort)
        openai_messages, system_text = _openai_messages(messages, system)
        prompt = await asyncio.to_thread(_build_prompt, renderer, openai_messages, system_text, tools or [], effort)
        return TokenCount(input_tokens=prompt.length, output_tokens=None)

    async def stream(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs: Any,
    ) -> AsyncContextManager[Any]:
        # The agent awaits the coroutine to obtain the context manager, then
        # enters it; the sample happens inside ``__aenter__``.
        return TinkerStreamWrapper(self, messages, system, params, kwargs)


def _renderer_decode(renderer: Any, tokens: Sequence[int]) -> str:
    """Best-effort decode of sampled token ids for the trace record."""
    tokenizer = getattr(renderer, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode(list(tokens)))
        except Exception:  # noqa: BLE001
            pass
    return ""


class TinkerStreamWrapper:
    """Stream adapter over a completed native sample.

    Tinker's native SamplingClient has no streaming API, so the sample is
    awaited up front and the parsed response is replayed as ``MessageChunk``
    events (text, thinking, tool-use start/delta) so the agent loop and TUI
    work unchanged. Nothing is shown until the sample completes — a documented
    limitation of the native path.
    """

    def __init__(
        self,
        provider: TinkerProvider,
        messages: MessageHistory,
        system: Optional[Message],
        params: Optional[GenerationParams],
        kwargs: Dict[str, Any],
    ):
        self.provider = provider
        self.messages = messages
        self.system = system
        self.params = params
        self.kwargs = kwargs
        self._chunks: List[MessageChunk] = []
        self._index = 0
        self._final_message: Optional[Message] = None
        self._entered = False

    async def __aenter__(self) -> "TinkerStreamWrapper":
        model = str(self.kwargs.get("model") or self.provider.model or "")
        sequence, parsed, termination, prompt, usage = await self.provider._sample(
            self.messages, self.system, self.params, model
        )
        self._final_message = _parsed_to_message(parsed, _kolega_stop_reason(str(sequence.stop_reason), termination))
        self._final_message.usage_metadata = {
            "provider": self.provider.provider_name,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cache_read_input_tokens": usage["prompt_cache_hit_tokens"],
        }
        attach_normalized_usage(self._final_message, self.provider.provider_name, model)
        renderer = self.provider._renderer
        if renderer is not None and getattr(renderer, "supports_streaming", False):
            self._chunks = _stream_chunks(renderer, sequence.tokens)
        else:
            # tml-renderers (Inkling) has no streaming parser; replay blocks.
            self._chunks = _message_to_chunks(self._final_message)
        self.provider._emit_trace(
            model,
            sequence,
            termination,
            prompt,
            usage,
            self.params.temperature if self.params else None,
            self.params.max_completion_tokens if self.params else None,
        )
        self._entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._entered = False

    def __aiter__(self) -> "TinkerStreamWrapper":
        if not self._entered:
            raise RuntimeError("Must use 'async with' before iterating")
        return self

    async def __anext__(self) -> MessageChunk:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def get_final_message(self) -> Message:
        if self._final_message is None:
            raise RuntimeError("Stream has not entered")
        return self._final_message


def _stream_chunks(renderer: Any, tokens: Sequence[int]) -> List[MessageChunk]:
    """Replay a renderer's post-hoc streaming parser as ``MessageChunk`` events.

    ``parse_response_streaming`` yields header/text/thinking deltas followed by
    the final parsed message; only text and thinking deltas become events.
    """
    chunks: List[MessageChunk] = []
    for delta in renderer.parse_response_streaming(list(tokens)):
        if hasattr(delta, "thinking") and delta.thinking:
            chunks.append(MessageChunk(type="thinking", thinking=delta.thinking))
        elif hasattr(delta, "text") and delta.text:
            chunks.append(MessageChunk(type="text", text=delta.text))
    return chunks


def _message_to_chunks(message: Message) -> List[MessageChunk]:
    """Split a parsed message into replayable ``MessageChunk`` events.

    Tool calls are emitted as a ``tool_use_start`` followed by one
    ``tool_use_delta`` carrying the complete JSON input, mirroring how the
    Anthropic wrapper presents streamed tool calls to the agent loop.
    """
    chunks: List[MessageChunk] = []
    for block in message.content or []:
        if isinstance(block, TextBlock):
            chunks.append(MessageChunk(type="text", text=block.text))
        elif isinstance(block, ThinkingBlock):
            chunks.append(MessageChunk(type="thinking", thinking=block.thinking))
        elif isinstance(block, ToolCall):
            chunks.append(
                MessageChunk(
                    type="tool_use_start",
                    tool_call_delta={"id": block.id, "name": block.name, "input": ""},
                )
            )
            chunks.append(
                MessageChunk(
                    type="tool_use_delta",
                    tool_call_delta={"input_delta": json.dumps(block.input)},
                )
            )
    return chunks
