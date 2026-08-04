"""Retain+resend of DeepSeek flash's plain-text reasoning items.

Flash's /responses exposes the raw chain-of-thought as ``reasoning_text`` content
parts and returns no ``encrypted_content``. The stream wrapper retains those
items and ``to_responses_input`` resends them — the exact item shape Codex
records and replays against this endpoint (see rollout files in
runs/tb21-matrix/codex). This keeps the context gauge honest (reasoning counts
from the history like any other content) and keeps CoT continuity independent
of the server-side call_id-keyed restore, which dedupes explicit copies.

Hermetic: fake streams, no network. Live counterpart:
test_deepseek_flash_context_gauge_live.py.
"""

from types import SimpleNamespace as _ns

import pytest

from kolega_code.llm.models import (
    ContentBlock,
    Message,
    MessageHistory,
    ResponsesReasoningBlock,
    TextBlock,
)
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.providers.responses_common import ResponsesStreamWrapper, to_responses_input


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


def _reasoning_item_event(item_id="rs_1", text="I should read the file first."):
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


def _function_call_event(call_id="call_1"):
    return _ns(
        type="response.output_item.done",
        item=_ns(type="function_call", id="fc_1", call_id=call_id, name="read_file", arguments='{"path": "a.py"}'),
    )


def _completed_event():
    return _ns(
        type="response.completed",
        response=_ns(output=[], usage=None, status="completed", incomplete_details=None),
    )


async def _drain(wrapper):
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    return await wrapper.get_final_message()


class TestWrapperRetention:
    @pytest.mark.asyncio
    async def test_plain_text_reasoning_is_retained_and_resent(self):
        events = [_reasoning_item_event(), _function_call_event(), _completed_event()]
        message = await _drain(ResponsesStreamWrapper(_FakeStream(events), provider_name="deepseek"))

        block = message.content[0]
        assert isinstance(block, ResponsesReasoningBlock)
        assert block.content == ["I should read the file first."]
        assert block.encrypted_content is None
        assert block.item_id == "rs_1"

        items = to_responses_input(MessageHistory([message]))
        assert items[0]["type"] == "reasoning"
        assert items[0]["content"] == [{"type": "reasoning_text", "text": "I should read the file first."}]
        # Codex replays the item id to DeepSeek; the verified server-side dedupe
        # was measured on that exact shape, so keep it.
        assert items[0]["id"] == "rs_1"
        assert "encrypted_content" not in items[0]
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_item_with_neither_content_nor_blob_is_dropped(self):
        bare = _ns(
            type="response.output_item.done",
            item=_ns(type="reasoning", id="rs_1", encrypted_content=None, summary=[], content=[]),
        )
        message = await _drain(ResponsesStreamWrapper(_FakeStream([bare, _completed_event()])))
        assert not [b for b in message.content or [] if isinstance(b, ResponsesReasoningBlock)]

    @pytest.mark.asyncio
    async def test_encrypted_retention_unchanged(self):
        # The OpenAI/Codex path (encrypted blob, no plain content) must be untouched.
        event = _ns(
            type="response.output_item.done",
            item=_ns(type="reasoning", id="rs_9", encrypted_content="ENC", summary=[]),
        )
        message = await _drain(ResponsesStreamWrapper(_FakeStream([event, _completed_event()])))
        block = message.content[0]
        assert isinstance(block, ResponsesReasoningBlock)
        assert block.encrypted_content == "ENC"
        assert block.content == []
        item = to_responses_input(MessageHistory([message]))[0]
        assert item["encrypted_content"] == "ENC"
        assert "content" not in item
        assert "id" not in item


class TestSerialization:
    def test_round_trip_preserves_reasoning_text(self):
        block = ResponsesReasoningBlock(content=["step one", "step two"], item_id="rs_1")
        restored = ContentBlock.from_dict(block.to_dict())
        assert isinstance(restored, ResponsesReasoningBlock)
        assert restored.content == ["step one", "step two"]
        assert restored.item_id == "rs_1"
        assert restored.encrypted_content is None

    def test_legacy_encrypted_only_dict_still_loads(self):
        restored = ContentBlock.from_dict({"type": "responses_reasoning", "encrypted_content": "ENC", "summary": ["s"]})
        assert isinstance(restored, ResponsesReasoningBlock)
        assert restored.encrypted_content == "ENC"
        assert restored.content == []


class TestChatPathAndMarkdown:
    def test_chat_serialization_omits_reasoning_block(self):
        # Reachable when a session moves flash -> pro (same provider, chat API).
        # A placeholder would be echoed by the model (see
        # _ECHOED_REASONING_PLACEHOLDER in agent/conversation.py), so the block
        # must vanish, not render.
        message = Message(
            role="assistant",
            content=[
                ResponsesReasoningBlock(content=["private chain of thought"], item_id="rs_1"),
                TextBlock(text="the answer"),
            ],
        )
        payload = message.to_openai()
        assert payload["content"] == [{"type": "text", "text": "the answer"}]

    def test_markdown_hides_reasoning_text(self):
        # Compaction prompts are built from the markdown conversation; leaking a
        # 100k-token chain of thought into them would blow the summary budget.
        block = ResponsesReasoningBlock(content=["private chain of thought"])
        assert "private chain of thought" not in block.to_markdown()


class TestTokenCounting:
    def test_reasoning_text_counts_toward_input(self):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        reasoning_text = "carefully consider the tiling recurrence " * 50
        with_reasoning = [
            Message(
                role="assistant",
                content=[
                    ResponsesReasoningBlock(content=[reasoning_text], item_id="rs_1"),
                    TextBlock(text="ok"),
                ],
            )
        ]
        without = [Message(role="assistant", content=[TextBlock(text="ok")])]

        counted_with = provider._count_tokens_sync(with_reasoning, None)
        counted_without = provider._count_tokens_sync(without, None)

        assert counted_with - counted_without >= 100  # ~350 tokens of reasoning
