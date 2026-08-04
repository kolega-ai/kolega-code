"""Hosted web_search on the Responses providers: request, capture, replay, count.

The provider's server-side web_search tool is requested with a bare
``{"type": "web_search"}`` entry; the server executes ``search`` / ``open_page``
actions and returns ``web_search_call`` output items whose content never
reaches the client (it is injected server-side and billed as input). The
wrapper captures those items in stream order as ``WebSearchCallBlock``s so
``to_responses_input`` can replay them — the item id keys the server-side
restore of the searched content (verified live 2026-08-04,
findings/probes/hosted_web_search_probe.py).

Hermetic: fake streams, no network. Live counterpart planned in
test_hosted_web_search_live.py.
"""

from types import SimpleNamespace as _ns

import pytest

from kolega_code.llm.models import (
    ContentBlock,
    Message,
    MessageHistory,
    ResponsesReasoningBlock,
    TextBlock,
    ToolCall,
    WebSearchCallBlock,
)
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.providers.openai_responses import OpenAIResponsesProvider
from kolega_code.llm.providers.responses_common import (
    ResponsesStreamWrapper,
    responses_tools,
    to_responses_input,
)
from kolega_code.llm.providers._token_encoding import get_counting_encoding


def _function_tool():
    from kolega_code.llm.models import ToolDefinition, ToolParameter

    return ToolDefinition(
        name="read_file",
        description="Read a file.",
        parameters=[ToolParameter(name="path", type="string", description="Path", required=True)],
    )


class TestHostedToolInRequest:
    def test_appended_after_function_tools_when_enabled(self):
        params = GenerationParams(tools=[_function_tool()], hosted_web_search=True)
        tools = responses_tools(params)
        assert tools is not None
        assert tools[-1] == {"type": "web_search"}
        assert tools[0]["type"] == "function"

    def test_absent_by_default(self):
        tools = responses_tools(GenerationParams(tools=[_function_tool()]))
        assert tools is not None
        assert {"type": "web_search"} not in tools

    def test_emitted_even_without_client_tools(self):
        assert responses_tools(GenerationParams(hosted_web_search=True)) == [{"type": "web_search"}]

    def test_no_tools_at_all_returns_none(self):
        assert responses_tools(GenerationParams()) is None
        assert responses_tools(None) is None

    @pytest.mark.parametrize(
        "provider_cls,model",
        [
            (DeepSeekResponsesProvider, "deepseek-v4-flash"),
            (OpenAIResponsesProvider, "gpt-5.6-sol"),
        ],
    )
    def test_build_request_carries_hosted_tool(self, provider_cls, model):
        provider = provider_cls(api_key="sk-test")
        request = provider._build_request(
            MessageHistory([]),
            None,
            GenerationParams(hosted_web_search=True),
            {"model": model},
        )
        assert {"type": "web_search"} in request["tools"]

        request_off = provider._build_request(
            MessageHistory([]),
            None,
            GenerationParams(),
            {"model": model},
        )
        assert {"type": "web_search"} not in (request_off["tools"] or [])


# --- stream capture ---------------------------------------------------------------


class _FakeStream:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def aclose(self):
        pass


def _reasoning_event(item_id="rs_1", text="Let me check the docs."):
    return _ns(
        type="response.output_item.done",
        item=_ns(
            type="reasoning",
            id=item_id,
            encrypted_content=None,
            summary=[],
            content=[_ns(type="reasoning_text", text=text)],
        ),
    )


def _web_search_event(item_id="ws_1", action=None):
    return _ns(
        type="response.output_item.done",
        item=_ns(
            type="web_search_call",
            id=item_id,
            status="completed",
            action=action if action is not None else {"type": "search", "queries": ["rust releases"]},
        ),
    )


def _completed_event():
    return _ns(
        type="response.completed",
        response=_ns(output=[], usage=None, status="completed", incomplete_details=None),
    )


async def _drain_with_chunks(wrapper):
    chunks = []
    async with wrapper as stream:
        async for chunk in stream:
            chunks.append(chunk)
    return chunks, await wrapper.get_final_message()


class TestStreamCapture:
    @pytest.mark.asyncio
    async def test_interleaved_items_captured_in_stream_order(self):
        events = [
            _reasoning_event("rs_1", "Search first."),
            _web_search_event("ws_1", {"type": "search", "queries": ["a", "b"]}),
            _reasoning_event("rs_2", "Now open the page."),
            _web_search_event("ws_2", {"type": "open_page", "url": "https://example.com/x"}),
            _completed_event(),
        ]
        chunks, message = await _drain_with_chunks(
            ResponsesStreamWrapper(_FakeStream(events), provider_name="deepseek")
        )

        hosted = [c for c in chunks if c.type == "hosted_tool_call"]
        assert [c.tool_call_delta["id"] for c in hosted] == ["ws_1", "ws_2"]
        assert hosted[0].tool_call_delta["action"] == {"type": "search", "queries": ["a", "b"]}
        assert hosted[0].tool_call_delta["name"] == "web_search"

        kinds = [type(block).__name__ for block in message.content]
        assert kinds == [
            "ResponsesReasoningBlock",
            "WebSearchCallBlock",
            "ResponsesReasoningBlock",
            "WebSearchCallBlock",
        ]
        ws = message.content[1]
        assert ws.item_id == "ws_1"
        assert ws.status == "completed"
        assert ws.queries == ["a", "b"]

    @pytest.mark.asyncio
    async def test_namespace_action_converts_to_dict(self):
        # SDK objects arrive as attribute bags, not dicts.
        events = [
            _web_search_event("ws_1", _ns(type="open_page", url="https://example.com/y", queries=None)),
            _completed_event(),
        ]
        _, message = await _drain_with_chunks(ResponsesStreamWrapper(_FakeStream(events)))
        ws = message.content[0]
        assert isinstance(ws, WebSearchCallBlock)
        assert ws.action == {"type": "open_page", "url": "https://example.com/y"}

    @pytest.mark.asyncio
    async def test_fallback_scan_of_final_response_output(self):
        # Backends that only populate the final response (no item.done events).
        final = _ns(
            type="response.completed",
            response=_ns(
                output=[
                    _ns(
                        type="reasoning",
                        id="rs_1",
                        encrypted_content=None,
                        summary=[],
                        content=[_ns(type="reasoning_text", text="hm")],
                    ),
                    _ns(
                        type="web_search_call",
                        id="ws_9",
                        status="completed",
                        action={"type": "search", "queries": ["q"]},
                    ),
                ],
                usage=None,
                status="completed",
                incomplete_details=None,
            ),
        )
        _, message = await _drain_with_chunks(ResponsesStreamWrapper(_FakeStream([final])))
        kinds = [type(block).__name__ for block in message.content]
        assert kinds == ["ResponsesReasoningBlock", "WebSearchCallBlock"]
        assert message.content[1].item_id == "ws_9"


# --- replay -----------------------------------------------------------------------


class TestReplay:
    def test_to_responses_input_preserves_prefix_order_and_calls_last(self):
        message = Message(
            role="assistant",
            content=[
                ResponsesReasoningBlock(content=["think 1"], item_id="rs_1"),
                WebSearchCallBlock(item_id="ws_1", status="completed", action={"type": "search", "queries": ["q"]}),
                ResponsesReasoningBlock(content=["think 2"], item_id="rs_2"),
                TextBlock(text="Found it."),
                ToolCall(id="call_1", name="read_file", input={"path": "a.py"}, execution_id="te_1"),
            ],
        )
        items = to_responses_input(MessageHistory([message]))
        types = [item.get("type") or item.get("role") for item in items]
        # Prefix in block order, assistant text next, function_call after it
        # (its orphan-padding output trails, keeping the request valid).
        assert types[:5] == ["reasoning", "web_search_call", "reasoning", "assistant", "function_call"]
        ws_item = items[1]
        assert ws_item == {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "queries": ["q"]},
        }
        # function_call items stay last (before their padded output).
        assert types.index("function_call") > types.index("assistant")

    def test_item_id_omitted_when_unknown(self):
        block = WebSearchCallBlock(status="completed", action={"type": "search", "queries": []})
        assert "id" not in block.to_responses_item()


# --- persistence round-trip -------------------------------------------------------


class TestRoundTrip:
    def test_dict_round_trip_via_registry(self):
        block = WebSearchCallBlock(
            item_id="ws_1", status="completed", action={"type": "open_page", "url": "https://e.com"}
        )
        restored = ContentBlock.from_dict(block.to_dict())
        assert isinstance(restored, WebSearchCallBlock)
        assert restored.item_id == "ws_1"
        assert restored.status == "completed"
        assert restored.action == {"type": "open_page", "url": "https://e.com"}
        assert restored.url == "https://e.com"

    def test_markdown_label(self):
        search = WebSearchCallBlock(action={"type": "search", "queries": ["a", "b"]})
        assert "'a', 'b'" in search.to_markdown()
        page = WebSearchCallBlock(action={"type": "open_page", "url": "https://e.com"})
        # Not URL sanitization — a test assertion on a display label.
        assert "https://e.com" in page.to_markdown()  # codeql[py/incomplete-url-substring-sanitization]


# --- token counting ---------------------------------------------------------------


class TestTokenCounting:
    def _provider(self):
        return OpenAIProvider(api_key="sk-test")

    def test_block_counts_nonzero_metadata_only(self):
        provider = self._provider()
        encoding = get_counting_encoding("o200k_base")
        message = Message(
            role="assistant",
            content=[
                WebSearchCallBlock(
                    item_id="ws_1",
                    status="completed",
                    action={"type": "search", "queries": ["rust releases august"]},
                )
            ],
        )
        count = provider._count_message_tokens(encoding, message)
        # More than the bare message overhead, far less than any page of content.
        assert 5 < count < 200

    def test_fingerprint_tracks_action_changes(self):
        provider = self._provider()
        block = WebSearchCallBlock(item_id="ws_1", status="completed", action={"type": "search", "queries": ["q"]})
        before = provider._block_fingerprint(block)
        assert before[0] == "wsc"
        block.action["queries"] = ["a much longer query string"]
        assert provider._block_fingerprint(block) != before


# --- cross-provider handling ------------------------------------------------------


class TestCrossProvider:
    def _blocks(self):
        return [
            TextBlock(text="answer"),
            WebSearchCallBlock(item_id="ws_1", status="completed", action={"type": "search", "queries": ["q"]}),
        ]

    def test_kept_for_same_provider(self):
        from kolega_code.agent.conversation import _adapt_content_blocks_for_provider

        adapted, _changed = _adapt_content_blocks_for_provider(
            self._blocks(),
            source_provider="deepseek",
            target_provider="deepseek",
            target_model="deepseek-v4-flash",
            supports_vision=False,
        )
        assert any(isinstance(b, WebSearchCallBlock) for b in adapted)

    def test_dropped_for_foreign_provider_without_placeholder(self):
        from kolega_code.agent.conversation import _adapt_content_blocks_for_provider

        adapted, changed = _adapt_content_blocks_for_provider(
            self._blocks(),
            source_provider="deepseek",
            target_provider="anthropic",
            target_model="claude-sonnet-4-5-20250929",
            supports_vision=True,
        )
        assert changed
        assert not any(isinstance(b, WebSearchCallBlock) for b in adapted)
        assert not any("[Web search]" in getattr(b, "text", "") for b in adapted)

    def test_message_to_openai_omits_block(self):
        message = Message(role="assistant", content=self._blocks())
        payload = message.to_openai()
        assert "[Web search]" not in str(payload)
