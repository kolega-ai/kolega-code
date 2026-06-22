"""OpenAI provider variant that talks to the ChatGPT-subscription backend.

Unlike :class:`OpenAIProvider` (which speaks Chat Completions against
``api.openai.com``), this provider authenticates with a ChatGPT OAuth bearer
token and calls the **Responses API** at ``chatgpt.com/backend-api/codex`` —
the only surface that backend exposes. It reuses the base class only for token
counting; request building, streaming, and response parsing are Responses-shaped.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.tokens import ChatGPTTokenManager

from ..models import (
    ImageBlock,
    Message,
    MessageChunk,
    MessageHistory,
    ResponsesReasoningBlock,
    TextBlock,
    ToolCall,
    ToolResult,
    safe_parse_tool_arguments,
)
from ..specs import build_thinking_request_params
from ..tool_execution_ids import ToolExecutionIdRegistry
from .base import BaseLLMProvider
from .models import GenerationParams
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)


class ChatGPTAuth(httpx.Auth):
    """httpx auth flow that injects a fresh bearer token + account id per request.

    On a 401 it forces a token refresh and retries the request once, so an
    expired access token self-heals mid-session.
    """

    def __init__(self, token_manager: ChatGPTTokenManager) -> None:
        self._manager = token_manager

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        access_token, account_id = await self._manager.authorization()
        self._apply(request, access_token, account_id)
        response = yield request
        if response.status_code == 401:
            await self._manager.refresh()
            access_token, account_id = await self._manager.authorization()
            self._apply(request, access_token, account_id)
            yield request

    @staticmethod
    def _apply(request: httpx.Request, access_token: str, account_id: str) -> None:
        request.headers["Authorization"] = f"Bearer {access_token}"
        if account_id:
            request.headers[chatgpt_constants.ACCOUNT_ID_HEADER] = account_id


# --- message/tool conversion to the Responses `input` shape ---------------------


def _image_data_url(block: ImageBlock) -> str:
    return f"data:{block.media_type};base64,{block.data}"


def _tool_result_images(block: ToolResult) -> List[ImageBlock]:
    if not isinstance(block.content, list):
        return []
    return [item for item in block.content if isinstance(item, ImageBlock)]


def _tool_result_text(block: ToolResult) -> str:
    """Flatten a tool result to the plain text a function_call_output expects.

    Responses function_call_output.output is string-only. If a tool returned
    images, ``to_responses_input`` attaches them in a following user input item.
    """
    content = block.content
    if isinstance(content, str):
        return content
    parts: List[str] = []
    image_count = 0
    for item in content or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
        elif isinstance(item, ImageBlock) or getattr(item, "media_type", None):
            image_count += 1
    if image_count:
        parts.append(
            f"[{block.name} returned {image_count} image"
            f"{'s' if image_count != 1 else ''}; attached in the following user message.]"
        )
    if block.is_error and not parts:
        return "Tool execution error"
    return "\n".join(parts)


def _tool_result_image_message(block: ToolResult) -> Optional[Dict[str, Any]]:
    images = _tool_result_images(block)
    if not images:
        return None
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": f"Image returned by tool {block.name} for tool call {block.tool_use_id}.",
        }
    ]
    content.extend({"type": "input_image", "image_url": _image_data_url(image)} for image in images)
    return {"role": "user", "content": content}


def _role_message_item(role: str, parts: List[tuple]) -> Optional[Dict[str, Any]]:
    """Build a Responses role message item from (kind, value) content parts."""
    is_user = role != "assistant"
    text_type = "input_text" if is_user else "output_text"
    content: List[Dict[str, Any]] = []
    for kind, value in parts:
        if kind == "text" and value:
            content.append({"type": text_type, "text": value})
        elif kind == "image" and is_user:
            content.append({"type": "input_image", "image_url": value})
    if not content:
        return None
    return {"role": "user" if is_user else "assistant", "content": content}


def to_responses_input(messages: MessageHistory) -> List[Dict[str, Any]]:
    """Convert unified message history into Responses API ``input`` items.

    System/developer messages are dropped here — they are folded into the
    top-level ``instructions`` field instead (see :func:`instructions_from`).
    """
    items: List[Dict[str, Any]] = []
    for message in messages:
        if getattr(message, "role", None) in ("system", "developer"):
            continue
        if isinstance(message.content, str):
            item = _role_message_item(message.role, [("text", message.content)])
            if item:
                items.append(item)
            continue

        text_parts: List[tuple] = []
        tool_image_messages: List[Dict[str, Any]] = []
        for block in message.content or []:
            if isinstance(block, ResponsesReasoningBlock):
                # Resend the model's prior reasoning (opaque encrypted blob) so it
                # continues its chain-of-thought across tool calls instead of
                # re-deriving it every round — the key to matching Codex's
                # thinking time. Reasoning items must precede the function_call
                # they belong to, which holds because the stream wrapper puts the
                # reasoning block first in the assistant message.
                items.append(block.to_responses_item())
            elif isinstance(block, ToolCall):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": block.input if isinstance(block.input, str) else json.dumps(block.input),
                    }
                )
            elif isinstance(block, ToolResult):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": block.tool_use_id,
                        "output": _tool_result_text(block),
                    }
                )
                image_message = _tool_result_image_message(block)
                if image_message is not None:
                    tool_image_messages.append(image_message)
            elif isinstance(block, TextBlock):
                text_parts.append(("text", block.text))
            elif isinstance(block, ImageBlock):
                text_parts.append(("image", _image_data_url(block)))
            # Foreign Anthropic thinking/redacted-thinking blocks (from a prior
            # provider) are skipped here; adapt_history_for_provider has already
            # replaced them with text placeholders before this conversion runs.
        if text_parts:
            item = _role_message_item(message.role, text_parts)
            if item:
                items.append(item)
        items.extend(tool_image_messages)
    return items


def instructions_from(system: Optional[Message], messages: MessageHistory) -> str:
    """Collect the system prompt (and any system/developer messages) as instructions."""
    parts: List[str] = []
    if system is not None:
        parts.append(system.get_text_content() if hasattr(system, "get_text_content") else str(system))
    for message in messages:
        if getattr(message, "role", None) in ("system", "developer"):
            parts.append(message.get_text_content())
    return "\n\n".join(part for part in parts if part)


def responses_tools(params: Optional[GenerationParams]) -> Optional[List[Dict[str, Any]]]:
    """Flatten Chat-Completions tool defs into the Responses (un-nested) shape."""
    if not params or not params.tools:
        return None
    tools: List[Dict[str, Any]] = []
    for definition in params.tools:
        chat_shape = definition.to_openai()
        fn = chat_shape.get("function", chat_shape)
        tools.append(
            {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            }
        )
    return tools


def _usage_from_response(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)
    metadata: Dict[str, Any] = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "provider": "openai",
    }
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    if cached is None and isinstance(details, dict):
        cached = details.get("cached_tokens")
    if cached:
        metadata["cache_read_input_tokens"] = cached
    return metadata


def _blocks_from_response(response: Any, tool_execution_ids: ToolExecutionIdRegistry):
    """Parse a completed Responses object into (content_blocks, tool_use_blocks)."""
    content_blocks: list = []
    tool_use_blocks: list = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for part in getattr(item, "content", None) or []:
                text = getattr(part, "text", None)
                if getattr(part, "type", None) in ("output_text", "text") and text:
                    content_blocks.append(TextBlock(text=text))
        elif item_type == "function_call":
            call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or ""
            tool_call = ToolCall(
                id=call_id,
                name=getattr(item, "name", "") or "",
                input=safe_parse_tool_arguments(getattr(item, "arguments", "") or ""),
                execution_id=tool_execution_ids.get_or_create(call_id),
            )
            content_blocks.append(tool_call)
            tool_use_blocks.append(tool_call)
    return content_blocks, tool_use_blocks


def _stop_reason_from_response(response: Any, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_use"
    if getattr(response, "status", None) == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) if details is not None else None
        if reason == "max_output_tokens":
            return "max_tokens"
    return "end_turn"


# --- streaming wrapper ----------------------------------------------------------


class ResponsesStreamWrapper:
    """Adapts the Responses streaming events to the MessageChunk contract.

    Mirrors :class:`OpenAIStreamWrapper`: yields ``text``/``thinking`` chunks for
    live display and a ``tool_use_start`` chunk when a function call begins, then
    builds the authoritative final Message from the ``response.completed`` event.
    """

    def __init__(self, responses_stream: Any) -> None:
        self._stream = responses_stream
        self._iterator: Optional[AsyncIterator[Any]] = None
        self._closed = False
        self._text = ""
        self._reasoning = ""
        # Completed reasoning items (with encrypted_content), in stream order, so
        # they can be resent next turn for chain-of-thought continuity.
        self._reasoning_items: List[Dict[str, Any]] = []
        self._final_response: Any = None
        self._function_calls: Dict[str, Dict[str, str]] = {}
        self._tool_execution_ids = ToolExecutionIdRegistry()
        self._started_calls: set[str] = set()

    async def __aenter__(self) -> "ResponsesStreamWrapper":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        aclose = getattr(self._stream, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # pragma: no cover - best effort
                pass
        self._closed = True
        return False

    def __aiter__(self) -> "ResponsesStreamWrapper":
        return self

    async def __anext__(self) -> MessageChunk:
        if self._closed:
            raise StopAsyncIteration
        iterator = self._iterator
        if iterator is None:
            iterator = self._stream.__aiter__()
            self._iterator = iterator
        event = await iterator.__anext__()  # propagates StopAsyncIteration
        return self._handle_event(event)

    def _handle_event(self, event: Any) -> MessageChunk:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            self._text += delta
            return MessageChunk(type="text", text=delta)
        if event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            delta = getattr(event, "delta", "") or ""
            self._reasoning += delta
            return MessageChunk(type="thinking", thinking=delta)
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if getattr(item, "type", None) == "function_call":
                # Key accumulators by the stream item id (what argument-delta events
                # reference); remember the call_id used to link the tool result.
                item_id = getattr(item, "id", None) or getattr(item, "call_id", "")
                call_id = getattr(item, "call_id", None) or item_id
                name = getattr(item, "name", "") or ""
                self._function_calls.setdefault(item_id, {"call_id": call_id, "name": name, "arguments": ""})
                if item_id not in self._started_calls:
                    self._started_calls.add(item_id)
                    return MessageChunk(
                        type="tool_use_start",
                        tool_call_delta={"id": call_id, "name": name, "input": ""},
                    )
        elif event_type == "response.function_call_arguments.delta":
            call_id = getattr(event, "item_id", None) or getattr(event, "call_id", "")
            record = self._function_calls.setdefault(call_id, {"call_id": call_id, "name": "", "arguments": ""})
            record["arguments"] += getattr(event, "delta", "") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = getattr(event, "item_id", None) or getattr(event, "call_id", "")
            record = self._function_calls.setdefault(call_id, {"call_id": call_id, "name": "", "arguments": ""})
            arguments = getattr(event, "arguments", None)
            if arguments is not None:
                record["arguments"] = arguments
        elif event_type == "response.output_item.done":
            item = getattr(event, "item", None)
            item_kind = getattr(item, "type", None)
            if item_kind == "reasoning":
                # Capture the completed reasoning item (carries encrypted_content)
                # so it can be resent next turn for chain-of-thought continuity.
                captured = self._reasoning_dict(item)
                if captured:
                    self._reasoning_items.append(captured)
            elif item_kind == "function_call":
                # The completed function_call item carries the final name +
                # arguments; capture it so tool calls survive even if argument
                # deltas were sparse.
                item_id = getattr(item, "id", None) or getattr(item, "call_id", "")
                call_id = getattr(item, "call_id", None) or item_id
                record = self._function_calls.setdefault(item_id, {"call_id": call_id, "name": "", "arguments": ""})
                record["call_id"] = call_id or record["call_id"]
                name = getattr(item, "name", None)
                arguments = getattr(item, "arguments", None)
                if name:
                    record["name"] = name
                if arguments:
                    record["arguments"] = arguments
        elif event_type == "response.completed":
            self._final_response = getattr(event, "response", None)
        elif event_type in ("response.failed", "error"):
            message = getattr(event, "message", None) or "ChatGPT Responses stream failed."
            raise RuntimeError(message)
        return MessageChunk(type="ignore", text="")

    async def get_final_message(self) -> Message:
        # Build content from the streamed deltas, NOT response.completed.output:
        # the ChatGPT/Codex backend sends response.completed with an empty output[]
        # (the text/tool calls arrive only as deltas), so parsing the final
        # response there would drop the whole answer.
        content_blocks, tool_use_blocks = self._blocks_from_accumulators()
        if self._final_response is not None:
            usage_metadata = _usage_from_response(self._final_response)
            if not content_blocks and not tool_use_blocks:
                # Fallback for backends that only populate the final output (and
                # not deltas) — e.g. a non-streaming-shaped response.
                content_blocks, tool_use_blocks = _blocks_from_response(
                    self._final_response, self._tool_execution_ids
                )
            stop_reason = _stop_reason_from_response(self._final_response, bool(tool_use_blocks))
        else:
            usage_metadata = {}
            stop_reason = "tool_use" if tool_use_blocks else "end_turn"
            logger.warning("ResponsesStreamWrapper: no response.completed event; billing may be skipped")

        # Reasoning items lead the assistant turn (they precede the model's
        # message/function_call output) so the backend continues the prior
        # chain-of-thought when this message is resent next turn.
        reasoning_blocks = self._collect_reasoning_blocks()
        return Message(
            role="assistant",
            content=reasoning_blocks + content_blocks,
            tool_calls=tool_use_blocks or None,
            stop_reason=stop_reason,
            usage_metadata=usage_metadata,
        )

    def _collect_reasoning_blocks(self) -> list:
        """Build the reasoning blocks to carry forward for continuity.

        Prefers items captured from ``response.output_item.done`` events; falls
        back to the final response output for backends that only populate
        reasoning there.
        """
        items = list(self._reasoning_items)
        if not items and self._final_response is not None:
            for out in getattr(self._final_response, "output", None) or []:
                if getattr(out, "type", None) == "reasoning":
                    captured = self._reasoning_dict(out)
                    if captured:
                        items.append(captured)
        return [
            ResponsesReasoningBlock(
                encrypted_content=item["encrypted_content"],
                summary=item.get("summary") or [],
                item_id=item.get("item_id"),
            )
            for item in items
        ]

    @staticmethod
    def _reasoning_dict(item: Any) -> Optional[Dict[str, Any]]:
        """Extract resend-relevant fields from a Responses reasoning item.

        Only items carrying ``encrypted_content`` are useful for continuity, so
        items without it are dropped. Summary parts (if any) are preserved so the
        resent item matches what the backend returned.
        """
        encrypted = getattr(item, "encrypted_content", None)
        if not encrypted:
            return None
        summary: List[str] = []
        for part in getattr(item, "summary", None) or []:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                summary.append(text)
        return {
            "encrypted_content": encrypted,
            "summary": summary,
            "item_id": getattr(item, "id", None),
        }

    def _blocks_from_accumulators(self):
        content_blocks: list = []
        tool_use_blocks: list = []
        if self._text:
            content_blocks.append(TextBlock(text=self._text))
        for record in self._function_calls.values():
            tool_call = ToolCall(
                id=record["call_id"],
                name=record["name"],
                input=safe_parse_tool_arguments(record["arguments"]),
                execution_id=self._tool_execution_ids.get_or_create(record["call_id"]),
            )
            content_blocks.append(tool_call)
            tool_use_blocks.append(tool_call)
        return content_blocks, tool_use_blocks


# --- provider -------------------------------------------------------------------


class ChatGPTOAuthProvider(OpenAIProvider):
    """OpenAI Responses-API provider backed by a ChatGPT subscription token."""

    def __init__(
        self,
        token_manager: ChatGPTTokenManager,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
        provider_name: str = chatgpt_constants.PROVIDER_KEY,
    ) -> None:
        # Skip OpenAIProvider.__init__ (it builds Chat Completions clients with an
        # api_key); wire our own Responses client with the refreshing auth flow.
        BaseLLMProvider.__init__(
            self,
            api_key="chatgpt-oauth",
            max_retries=max_retries,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            base_url=base_url or chatgpt_constants.INFERENCE_BASE_URL,
        )
        self.provider_name = provider_name
        self._token_manager = token_manager
        self._session_id = str(uuid.uuid4())
        # Match Codex's client identity. The HTTP /responses backend does NOT take
        # an OpenAI-Beta header (that's the WebSocket path only); it expects the
        # codex_cli_rs originator + User-Agent.
        default_headers = {
            "originator": chatgpt_constants.ORIGINATOR,
            "User-Agent": chatgpt_constants.USER_AGENT,
            "session-id": self._session_id,
        }
        self.async_client = AsyncOpenAI(
            api_key="chatgpt-oauth",
            base_url=self.base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=httpx.AsyncClient(auth=ChatGPTAuth(token_manager), timeout=600.0),
        )
        # The Responses path is async-only; the sync client is unused.
        self.sync_client = None

    def _prepare_generation_params(self, params: Optional[GenerationParams] = None) -> Dict[str, Any]:
        return {"model": chatgpt_constants.DEFAULT_MODEL}

    def _build_request(
        self,
        messages: MessageHistory,
        system: Optional[Message],
        params: Optional[GenerationParams],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a Responses request matching what Codex sends to this backend.

        Notably this does NOT send ``max_output_tokens`` (Codex never does; the
        backend rejects it) and always streams (the backend is SSE-only).
        """
        model = str(kwargs.get("model") or chatgpt_constants.DEFAULT_MODEL)
        request: Dict[str, Any] = {
            "model": model,
            "input": to_responses_input(messages),
            "tools": responses_tools(params) or [],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "store": False,
            "stream": True,
            "prompt_cache_key": self._session_id,
        }
        # The ChatGPT backend hard-requires a non-empty instructions field
        # (400 "Instructions are required" otherwise), so always send one.
        request["instructions"] = instructions_from(system, messages) or "You are a helpful coding assistant."
        if params and params.thinking:
            thinking_params = build_thinking_request_params(self.provider_name, model, params.thinking)
            reasoning = thinking_params.get("reasoning")
            if reasoning:
                request["reasoning"] = reasoning
                # Ask the backend to return the model's reasoning as an opaque
                # encrypted item so we can resend it next turn for continuity
                # (mirrors Codex). Without it the model re-reasons from scratch on
                # every tool-call round and "thinks" far longer at high effort.
                request["include"] = ["reasoning.encrypted_content"]
        return request

    async def stream(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> ResponsesStreamWrapper:
        request = self._build_request(messages, system, params, kwargs)
        await self.rate_limiter.acquire()
        responses_stream = await self.async_client.responses.create(**request)
        return ResponsesStreamWrapper(responses_stream)

    async def generate(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> Message:
        # The backend only supports streaming, so drain a stream into a full message.
        request = self._build_request(messages, system, params, kwargs)
        await self.rate_limiter.acquire()
        responses_stream = await self.async_client.responses.create(**request)
        wrapper = ResponsesStreamWrapper(responses_stream)
        async with wrapper:
            async for _chunk in wrapper:
                pass
        message = await wrapper.get_final_message()
        if not message.usage_metadata:
            logger.warning("ChatGPTOAuthProvider.generate: response had no usage metadata; billing may be skipped")
        return message
