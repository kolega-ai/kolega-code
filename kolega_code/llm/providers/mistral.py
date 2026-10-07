"""First-party Mistral Chat Completions, without a vendor SDK.

Native thinking chunks are not OpenAI reasoning fields. Keep their ordered,
nested replay payloads intact, and never turn an incomplete SSE response into a
successful message. Local token counts are conservative estimates, not billing
counts or an exact Mistral tokenizer.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import math
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any

import httpx

from ..exceptions import (
    LLMAuthenticationError,
    LLMBillingError,
    LLMConnectionError,
    LLMContentPolicyViolationError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
    LLMInvalidRequestError,
    LLMNotFoundError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMTimeout,
    LLMUnprocessableEntityError,
    LLMUnsupportedParamsError,
)
from ..models import ContentBlock, ImageBlock, Message, MessageChunk, MessageHistory, TextBlock, ThinkingBlock
from ..models import ToolCall, ToolDefinition
from ..specs import MODEL_SPECS, build_thinking_request_params
from ..timeouts import streaming_timeout
from ..tool_execution_ids import ToolExecutionIdRegistry
from ..usage import attach_normalized_usage
from ._token_encoding import get_counting_encoding
from .base import BaseLLMProvider
from .models import GenerationParams, TokenCount

_DEFAULT_MODEL = "mistral-medium-3-5"
_REQUEST_FIELDS = frozenset(
    {
        "model",
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "random_seed",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "presence_penalty",
        "frequency_penalty",
        "safe_prompt",
        "prompt_cache_key",
        "reasoning_effort",
        "prediction",
        "metadata",
        "prompt_mode",
        "service_tier",
        "guardrails",
    }
)


def _protocol_error(detail: str) -> LLMError:
    return LLMError(f"Mistral response error: {detail}", provider="mistral")


def _http_error(status: int, body: Any) -> LLMError:
    """Map native HTTP/error-event bodies, not SDK exception classes."""
    detail = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body)
    lowered = detail.lower()
    error_type: type[LLMError]
    if status == 402 or (
        status in (400, 403, 429)
        and any(phrase in lowered for phrase in ("insufficient credit", "insufficient balance", "payment required"))
    ):
        error_type = LLMBillingError
    elif status == 401:
        error_type = LLMAuthenticationError
    elif status == 403:
        error_type = LLMPermissionDeniedError
    elif status == 404:
        error_type = LLMNotFoundError
    elif status == 422:
        error_type = LLMUnprocessableEntityError
    elif status == 429:
        error_type = LLMRateLimitError
    elif status == 408:
        error_type = LLMTimeout
    elif status >= 500:
        error_type = LLMInternalServerError
    elif status == 413 or (
        status == 400
        and any(
            phrase in lowered
            for phrase in (
                "context_length_exceeded",
                "context length",
                "context window",
                "maximum context",
                "too many tokens",
                "prompt is too long",
                "model token limit",
            )
        )
    ):
        error_type = LLMContextWindowExceededError
    elif status == 400 and any(
        phrase in lowered for phrase in ("content_filter", "content policy", "content filtering")
    ):
        error_type = LLMContentPolicyViolationError
    elif status == 400:
        error_type = LLMInvalidRequestError
    else:
        error_type = LLMError
    return error_type(f"Mistral API error ({status}): {detail}", provider="mistral")


def _event_error(body: dict[str, Any]) -> LLMError:
    error = body.get("error", body)
    if isinstance(error, dict):
        status = error.get("status_code", error.get("status", error.get("code")))
        if isinstance(status, str) and status.isdigit():
            status = int(status)
        if type(status) is not int:
            kind = str(error.get("type", "")).lower()
            status = {
                "authentication_error": 401,
                "permission_error": 403,
                "invalid_request_error": 400,
                "rate_limit_error": 429,
            }.get(kind, 500)
    else:
        status = 500
    return _http_error(status, body)


def _transport_error(error: httpx.TransportError) -> LLMConnectionError:
    cls = LLMTimeout if isinstance(error, httpx.TimeoutException) else LLMConnectionError
    return cls(f"Mistral transport error: {error}", provider="mistral")


def _stop_reason(reason: Any) -> str:
    if reason == "error":
        raise LLMInternalServerError("Mistral returned finish_reason=error", provider="mistral")
    if reason == "content_filter":
        raise LLMContentPolicyViolationError("Mistral blocked the completion (content_filter)", provider="mistral")
    mapped = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens", "model_length": "max_tokens"}
    if not isinstance(reason, str) or reason not in mapped:
        raise _protocol_error(f"missing or unsupported finish_reason: {reason!r}")
    return mapped[reason]


def _usage_metadata(raw: Any) -> dict[str, Any]:
    """Retain reported usage; only expose subtotals actually supplied by Mistral."""
    result: dict[str, Any] = {"provider": "mistral"}
    if raw is None:
        return result
    if not isinstance(raw, dict):
        raise _protocol_error("usage must be an object or null")
    result.update(raw)
    result["provider"] = "mistral"
    for source, source_key, target in (
        ("prompt_tokens_details", "cached_tokens", "cache_read_input_tokens"),
        ("prompt_tokens_details", "cache_write_tokens", "cache_write_input_tokens"),
        ("completion_tokens_details", "reasoning_tokens", "reasoning_output_tokens"),
    ):
        details = raw.get(source)
        if isinstance(details, dict) and source_key in details:
            result[target] = details[source_key]
    # The common OpenAI-shaped normalizer adds writes to prompt_tokens. Native
    # prompt_tokens includes them, so match its established capture invariant.
    writes = result.get("cache_write_input_tokens")
    prompt = result.get("prompt_tokens")
    if type(writes) is int and writes >= 0 and type(prompt) is int and prompt >= writes:
        result["prompt_tokens"] = prompt - writes
    return result


def _content_blocks(content: Any) -> list[ContentBlock]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextBlock(content)] if content else []
    if not isinstance(content, list):
        raise _protocol_error("content must be a string, chunk list, or null")
    blocks: list[ContentBlock] = []
    for chunk in content:
        if not isinstance(chunk, dict):
            raise _protocol_error("content chunks must be objects")
        kind = chunk.get("type")
        if kind == "text":
            text = chunk.get("text")
            if not isinstance(text, str):
                raise _protocol_error("text chunk requires string text")
            blocks.append(TextBlock(text))
        elif kind == "thinking":
            nested = chunk.get("thinking")
            if not isinstance(nested, list):
                raise _protocol_error("thinking chunk requires a nested chunk list")
            parts: list[str] = []
            for item in nested:
                if not isinstance(item, dict) or item.get("type") != "text" or not isinstance(item.get("text"), str):
                    raise _protocol_error(f"unsupported nested thinking chunk: {item!r}")
                parts.append(item["text"])
            signature = chunk.get("signature")
            if signature is not None and not isinstance(signature, str):
                raise _protocol_error("thinking signature must be a string or null")
            if "closed" in chunk and chunk["closed"] is not None and type(chunk["closed"]) is not bool:
                raise _protocol_error("thinking closed must be a boolean or null")
            blocks.append(
                ThinkingBlock(
                    thinking="".join(parts),
                    signature=signature,
                    provider_metadata={key: value for key, value in chunk.items() if key != "type"},
                )
            )
        else:
            raise _protocol_error(f"unsupported content chunk type: {kind!r}")
    return blocks


class _ContentAccumulator:
    """Join adjacent text deltas once; keep each native thinking chunk verbatim."""

    def __init__(self) -> None:
        self.blocks: list[ContentBlock | list[str]] = []

    def add(self, blocks: list[ContentBlock]) -> None:
        for block in blocks:
            if isinstance(block, TextBlock):
                if self.blocks and isinstance(self.blocks[-1], list):
                    self.blocks[-1].append(block.text)
                else:
                    self.blocks.append([block.text])
            else:
                self.blocks.append(block)

    def finish(self) -> list[ContentBlock]:
        return [TextBlock("".join(block)) if isinstance(block, list) else block for block in self.blocks]


def _tool_call(raw: Any, registry: ToolExecutionIdRegistry, freeform_names: set[str]) -> ToolCall:
    if not isinstance(raw, dict) or raw.get("type", "function") != "function":
        raise _protocol_error("unsupported tool call")
    ident = raw.get("id")
    function = raw.get("function")
    if not isinstance(ident, str) or not ident or not isinstance(function, dict):
        raise _protocol_error("tool call requires an ID and function")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        raise _protocol_error("tool call requires a function name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError) as error:
            raise _protocol_error("malformed tool arguments JSON") from error
    if not isinstance(arguments, dict):
        raise _protocol_error("tool arguments must be a JSON object")
    if name in freeform_names:
        if not isinstance(arguments.get("input"), str):
            raise _protocol_error("freeform tool arguments require a string input")
        return ToolCall(
            id=ident,
            name=name,
            input=arguments["input"],
            input_kind="freeform",
            execution_id=registry.get_or_create(ident),
        )
    return ToolCall(id=ident, name=name, input=arguments, execution_id=registry.get_or_create(ident))


async def _sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Incremental strict UTF-8/SSE decoding with linear fragment accumulation."""
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    line_parts: list[str] = []
    data_parts: list[str] = []
    event = "message"
    pending_cr = False

    def accept_line(line: str) -> tuple[str, str] | None:
        nonlocal event
        if not line:
            if not data_parts:
                event = "message"
                return None
            result = (event, "\n".join(data_parts))
            data_parts.clear()
            event = "message"
            return result
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_parts.append(value)
        elif field == "event":
            event = value
        elif field not in ("id", "retry"):
            raise _protocol_error(f"invalid SSE field: {field!r}")
        return None

    async for raw in response.aiter_bytes():
        text = decoder.decode(raw)
        if not text:
            continue
        if pending_cr and text.startswith("\n"):
            text = text[1:]
        pending_cr = text.endswith("\r")
        lines = re.split(r"\r\n|\r|\n", text)
        line_parts.append(lines[0])
        if len(lines) == 1:
            continue
        first = "".join(line_parts)
        line_parts.clear()
        result = accept_line(first)
        if result is not None:
            yield result
        for line in lines[1:-1]:
            result = accept_line(line)
            if result is not None:
                yield result
        if lines[-1]:
            line_parts.append(lines[-1])
    trailing = decoder.decode(b"", final=True)
    if trailing:
        line_parts.append(trailing)
    if line_parts or data_parts:
        raise _protocol_error("truncated SSE event")
    raise _protocol_error("SSE ended without [DONE]")


class MistralStreamWrapper:
    """Single-consumer native stream; final usage is available only after DONE."""

    def __init__(self, response: httpx.Response, model: str, freeform_names: set[str]) -> None:
        self.response = response
        self.model = model
        self.freeform_names = freeform_names
        self.tool_execution_ids = ToolExecutionIdRegistry()
        self._content = _ContentAccumulator()
        self._tools: dict[int, dict[str, list[str]]] = {}
        self._tool_calls: list[ToolCall] = []
        self._usage: dict[str, Any] = {"provider": "mistral"}
        self._stop_reason: str | None = None
        self._final_message: Message | None = None
        self._failure: BaseException | None = None
        self._closed = False
        self._iterator = self._iterate()

    async def __aenter__(self) -> MistralStreamWrapper:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        await self.aclose()
        return False

    def __aiter__(self) -> MistralStreamWrapper:
        return self

    async def __anext__(self) -> MessageChunk:
        if self._closed:
            if self._failure is not None:
                raise self._failure
            raise StopAsyncIteration
        try:
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except httpx.TransportError as error:
            self._failure = _transport_error(error)
            await self.aclose()
            raise self._failure from error
        except (ValueError, UnicodeError) as error:
            self._failure = _protocol_error(f"invalid SSE/JSON: {error}")
            await self.aclose()
            raise self._failure from error
        except BaseException as error:
            self._failure = error
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                await self._iterator.aclose()
            finally:
                await self.response.aclose()

    async def get_final_message(self) -> Message:
        if self._failure is not None:
            raise self._failure
        if self._final_message is None:
            if self._closed:
                raise _protocol_error("stream closed before completion")
            async for _ in self:
                pass
        if self._final_message is None:
            raise _protocol_error("stream has no completed message")
        return self._final_message

    def _add_tools(self, calls: Any) -> None:
        if calls is None:
            return
        if not isinstance(calls, list):
            raise _protocol_error("delta.tool_calls must be a list")
        for call in calls:
            if not isinstance(call, dict) or call.get("type", "function") != "function":
                raise _protocol_error("unsupported streamed tool call")
            index = call.get("index")
            if type(index) is not int or index < 0:
                raise _protocol_error("streamed tool call requires a nonnegative index")
            state = self._tools.setdefault(index, {"id": [], "name": [], "arguments": []})
            function = call.get("function")
            if function is None:
                function = {}
            if not isinstance(function, dict):
                raise _protocol_error("streamed tool function must be an object")
            for field, value in (
                ("id", call.get("id")),
                ("name", function.get("name")),
                ("arguments", function.get("arguments")),
            ):
                if value is not None:
                    if not isinstance(value, str):
                        raise _protocol_error(f"streamed tool {field} must be a string")
                    state[field].append(value)

    async def _iterate(self) -> AsyncGenerator[MessageChunk, None]:
        async for event_name, data in _sse_events(self.response):
            if event_name not in ("message", "error"):
                raise _protocol_error(f"unsupported SSE event: {event_name!r}")
            if data.strip() == "[DONE]":
                if event_name == "error" or self._stop_reason is None:
                    raise _protocol_error("[DONE] arrived without a successful finish reason")
                message = Message(
                    role="assistant",
                    content=[*self._content.finish(), *self._tool_calls],
                    tool_calls=self._tool_calls,
                    stop_reason=self._stop_reason,
                    usage_metadata=self._usage,
                )
                self._final_message = attach_normalized_usage(message, "mistral", self.model)
                return
            body = json.loads(data)
            if not isinstance(body, dict):
                raise _protocol_error("SSE data must be a JSON object")
            if event_name == "error" or "error" in body or body.get("object") == "error":
                raise _event_error(body)
            choices = body.get("choices")
            if not isinstance(choices, list) or len(choices) > 1:
                raise _protocol_error("SSE requires zero or one completion choices")
            if not choices and body.get("usage") is None:
                raise _protocol_error("empty SSE event without usage")
            if body.get("usage") is not None:
                self._usage = _usage_metadata(body["usage"])
            if not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise _protocol_error("unsupported completion choice")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise _protocol_error("completion delta must be an object")
            unsupported = delta.keys() - {"role", "content", "tool_calls"}
            if unsupported:
                raise _protocol_error(f"unsupported delta fields: {sorted(unsupported)}")
            if delta.get("role") not in (None, "assistant"):
                raise _protocol_error("unexpected streamed message role")
            if self._stop_reason is not None:
                raise _protocol_error("completion delta after finish_reason")
            blocks = _content_blocks(delta.get("content"))
            self._content.add(blocks)
            self._add_tools(delta.get("tool_calls"))
            finish = choice.get("finish_reason")
            if finish is not None:
                self._stop_reason = _stop_reason(finish)
                ids: set[str] = set()
                for index in sorted(self._tools):
                    state = self._tools[index]
                    call = _tool_call(
                        {
                            "id": "".join(state["id"]),
                            "function": {"name": "".join(state["name"]), "arguments": "".join(state["arguments"])},
                        },
                        self.tool_execution_ids,
                        self.freeform_names,
                    )
                    if call.id in ids:
                        raise _protocol_error("duplicate tool call ID within one response")
                    ids.add(call.id)
                    self._tool_calls.append(call)
            for block in blocks:
                if isinstance(block, TextBlock) and block.text:
                    yield MessageChunk(type="text", text=block.text)
                elif isinstance(block, ThinkingBlock):
                    yield MessageChunk(type="thinking", thinking=block.thinking)
            if finish is not None:
                for call in self._tool_calls:
                    yield MessageChunk(type="tool_use", tool_call=call)


def _identity_retry(function: Callable[..., Any]) -> Callable[..., Any]:
    """Compatibility decorator: HTTP retries are owned by this adapter."""
    return function


class MistralProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        base_url: str | None = None,
        provider_name: str = "mistral",
    ) -> None:
        super().__init__(api_key, max_retries, requests_per_minute, tokens_per_minute, base_url)
        self.provider_name = provider_name
        self._session_id = str(uuid.uuid4())
        self.async_client = httpx.AsyncClient(
            base_url=(base_url or "https://api.mistral.ai/v1").rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=httpx.Timeout(600.0, connect=10.0),
        )

    @property
    def retry_decorator(self) -> Callable[..., Any]:
        return self.get_retry_decorator()

    def get_retry_decorator(self) -> Callable[..., Any]:
        return _identity_retry

    def _prepare_generation_params(self, params: GenerationParams | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"model": _DEFAULT_MODEL, "prompt_cache_key": self._session_id}
        if params is not None:
            if params.temperature is not None:
                result["temperature"] = params.temperature
            if params.max_completion_tokens is not None:
                result["max_tokens"] = params.max_completion_tokens
            if params.tools:
                result["tools"] = [tool.to_openai() for tool in params.tools]
        return result

    def _serialize_history(self, messages: MessageHistory, system: Message | None, model: str) -> list[dict[str, Any]]:
        history = MessageHistory([system, *messages]) if system is not None else messages
        spec = MODEL_SPECS.get(("mistral", model), {})
        if spec.get("supports_vision") is False:
            for message in history:
                if isinstance(message.content, list) and any(
                    isinstance(block, ImageBlock) for block in message.content
                ):
                    raise LLMInvalidRequestError(f"Mistral model {model} does not support vision", provider="mistral")
        # Reuse shared ordering, placeholders, image-result adaptation and native
        # reasoning replay. The final pass handles direct-provider orphan results.
        serialized = history.to_openai(provider="mistral", model=model)
        for item in serialized:
            if spec.get("supports_vision") is False and isinstance(item.get("content"), list):
                # Direct images were rejected above. Images remaining here are
                # shared-serializer followups from tool results: keep their text,
                # but do not send vision content to a known text-only model.
                item["content"] = [
                    {"type": "text", "text": f"[Tool result image omitted: {model} does not support vision]"}
                    if chunk.get("type") == "image_url"
                    else chunk
                    for chunk in item["content"]
                ]
        repaired: list[dict[str, Any]] = []
        pending: set[str] = set()
        for item in serialized:
            if item["role"] == "tool":
                ident = item.get("tool_call_id")
                if ident not in pending:
                    content = item.get("content", "")
                    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                    repaired.append(
                        {
                            "role": "user",
                            "content": f"[Unmatched tool result for {ident}; original call unavailable]\n{text}",
                        }
                    )
                    continue
                pending.remove(ident)
            else:
                if pending:
                    repaired.extend({"role": "tool", "tool_call_id": ident, "content": ""} for ident in sorted(pending))
                    pending.clear()
                if item["role"] == "assistant":
                    pending.update(call["id"] for call in item.get("tool_calls", []))
            repaired.append(item)
        repaired.extend({"role": "tool", "tool_call_id": ident, "content": ""} for ident in sorted(pending))
        return repaired

    def _build_request(
        self,
        messages: MessageHistory,
        system: Message | None,
        params: GenerationParams | None,
        streaming: bool,
        kwargs: dict[str, Any],
    ) -> tuple[bytes, str, set[str]]:
        unsupported = kwargs.keys() - _REQUEST_FIELDS
        if unsupported:
            raise LLMUnsupportedParamsError(
                f"Unsupported Mistral request fields: {sorted(unsupported)}", provider="mistral"
            )
        request = self._prepare_generation_params(params)
        request.update(kwargs)
        model = request["model"]
        if not isinstance(model, str) or not model:
            raise LLMInvalidRequestError("Mistral requires a model identifier", provider="mistral")
        spec = MODEL_SPECS.get(("mistral", model), {})
        request.setdefault("temperature", spec.get("default_temperature", 1.0))
        # This is our reserved-output default, NOT a hard native output ceiling.
        request.setdefault("max_tokens", spec.get("max_completion_tokens", 32768))
        if params is not None:
            request.update(build_thinking_request_params("mistral", model, params.thinking))
        if request.get("tools"):
            request.setdefault("parallel_tool_calls", True)
        request["messages"] = self._serialize_history(messages, system, model)
        request["stream"] = streaming
        freeform = {tool.name for tool in (params.tools or []) if tool.input_kind == "freeform"} if params else set()
        return json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), model, freeform

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        value = response.headers.get("Retry-After") if response is not None else None
        if value:
            try:
                seconds = float(value)
                if math.isfinite(seconds):
                    return max(0.0, seconds)
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    pass
        return float(min(2**attempt, 10))

    async def _request(self, payload: bytes, *, streaming: bool) -> httpx.Response:
        for attempt in range(max(0, self.max_retries) + 1):
            response: httpx.Response | None = None
            retryable = False
            try:
                await self.rate_limiter.acquire()
                request_kwargs: dict[str, Any] = {
                    "content": payload,
                    "headers": {
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream" if streaming else "application/json",
                    },
                }
                if streaming:
                    request_kwargs["timeout"] = streaming_timeout()
                request = self.async_client.build_request("POST", "chat/completions", **request_kwargs)
                response = await self.async_client.send(request, stream=True)
                if not response.is_success:
                    await response.aread()
                    try:
                        body = response.json()
                    except ValueError:
                        body = response.text
                    error = _http_error(response.status_code, body)
                    retryable = isinstance(error, (LLMRateLimitError, LLMInternalServerError, LLMTimeout))
                    if (
                        isinstance(error, LLMRateLimitError)
                        and response.headers.get("x-ratelimit-limit-req-minute") == "0"
                    ):
                        # A zero model/account allowance cannot recover by waiting.
                        error = LLMRateLimitError(
                            f"{error}. Mistral reports a zero request quota for this model/account; "
                            "enable model access in your Mistral account or select another model.",
                            provider="mistral",
                        )
                        retryable = False
                else:
                    if not streaming:
                        await response.aread()
                        await response.aclose()
                    return response
            except httpx.TransportError as caught:
                error = _transport_error(caught)
                retryable = True
            except BaseException:
                if response is not None:
                    await response.aclose()
                raise
            if response is not None:
                await response.aclose()
            if not retryable or attempt >= self.max_retries:
                raise error
            await asyncio.sleep(self._retry_delay(response, attempt))
        raise AssertionError("unreachable retry state")

    async def generate(
        self,
        messages: MessageHistory,
        system: Message | None = None,
        params: GenerationParams | None = None,
        **kwargs: Any,
    ) -> Message:
        payload, model, freeform = await asyncio.to_thread(self._build_request, messages, system, params, False, kwargs)
        response = await self._request(payload, streaming=False)
        try:
            body = response.json()
        except ValueError as error:
            raise _protocol_error("malformed completion JSON") from error
        if not isinstance(body, dict):
            raise _protocol_error("completion must be an object")
        if "error" in body or body.get("object") == "error":
            raise _event_error(body)
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise _protocol_error("completion requires exactly one choice")
        choice = choices[0]
        native = choice.get("message")
        if not isinstance(native, dict) or native.get("role") != "assistant":
            raise _protocol_error("completion requires an assistant message")
        accumulator = _ContentAccumulator()
        accumulator.add(_content_blocks(native.get("content")))
        registry = ToolExecutionIdRegistry()
        raw_calls = native.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise _protocol_error("tool_calls must be a list")
        calls = [_tool_call(call, registry, freeform) for call in raw_calls]
        if len({call.id for call in calls}) != len(calls):
            raise _protocol_error("duplicate tool call ID within one response")
        message = Message(
            role="assistant",
            content=[*accumulator.finish(), *calls],
            tool_calls=calls,
            stop_reason=_stop_reason(choice.get("finish_reason")),
            usage_metadata=_usage_metadata(body.get("usage")),
        )
        return attach_normalized_usage(message, "mistral", model)

    async def stream(
        self,
        messages: MessageHistory,
        system: Message | None = None,
        params: GenerationParams | None = None,
        **kwargs: Any,
    ) -> MistralStreamWrapper:
        payload, model, freeform = await asyncio.to_thread(self._build_request, messages, system, params, True, kwargs)
        response = await self._request(payload, streaming=True)
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type and content_type != "text/event-stream":
            await response.aclose()
            raise _protocol_error(f"expected text/event-stream, received {content_type!r}")
        return MistralStreamWrapper(response, model, freeform)

    def _count_tokens(
        self,
        messages: MessageHistory,
        system: Message | None,
        model: str,
        tools: list[ToolDefinition] | None,
    ) -> TokenCount:
        history = self._serialize_history(messages, system, model)
        images = 0

        def omit_images(value: Any) -> Any:
            nonlocal images
            if isinstance(value, dict):
                if value.get("type") == "image_url":
                    images += 1
                    return {"type": "image_url", "image_url": "[image]"}
                return {key: omit_images(item) for key, item in value.items()}
            if isinstance(value, list):
                return [omit_images(item) for item in value]
            return value

        text = json.dumps(
            {"messages": omit_images(history), "tools": [tool.to_openai() for tool in tools or []]}, ensure_ascii=False
        )
        encoding = get_counting_encoding("cl100k_base")
        # Unknown image dimensions/tokenizer: reserve 4096 per image without
        # tokenizing its base64. JSON/role framing plus 20% protect text budgets.
        estimate = math.ceil((len(encoding.encode(text)) + 8 * len(history) + 16) * 1.2) + images * 4096
        return TokenCount(input_tokens=estimate)

    async def count_tokens(
        self,
        messages: MessageHistory,
        system: Message | None = None,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> TokenCount:
        return await asyncio.to_thread(self._count_tokens, messages, system, model or _DEFAULT_MODEL, tools)
