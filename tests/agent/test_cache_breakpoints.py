"""Where cache breakpoints land, and that there are never more than the API allows.

Anthropic permits four ``cache_control`` breakpoints per request. The budget is spent as: the
tool list, the system prompt, and two rolling markers in the history. Exceeding it is a 400;
misplacing one silently costs a cache hit, which is worse because nothing fails loudly.
"""

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.conversation import Conversation
from kolega_code.llm.models import (
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from kolega_code.tools import Tool, ToolRegistry

from .compaction_helpers import text_msg

MAX_BREAKPOINTS = 4


def make_tool(name: str) -> Tool:
    async def handler(**inputs):
        return "ok"

    return Tool(
        name=name,
        definition=ToolDefinition(name=name, description=f"{name} tool", parameters=[]),
        handler=handler,
    )


def first_block(message: Message):
    assert isinstance(message.content, list)
    return message.content[0]


def marked_blocks(conversation: Conversation) -> list:
    return [
        block
        for message in conversation.history
        if isinstance(message.content, list)
        for block in message.content
        if getattr(block, "cache_checkpoint", False)
    ]


class TestRollingMarkers:
    def test_first_call_marks_only_the_latest_block(self):
        conversation = Conversation([text_msg("user", "one")])

        conversation.mark_cache_checkpoint()

        assert marked_blocks(conversation) == [conversation.history[-1].content[-1]]

    def test_second_call_keeps_the_previous_marker(self):
        """Two markers a turn apart: if the newest misses its lookback, the older still hits."""
        conversation = Conversation([text_msg("user", "one")])
        conversation.mark_cache_checkpoint()
        first = conversation.history[-1].content[-1]

        conversation.history.append(text_msg("assistant", "reply"))
        conversation.history.append(text_msg("user", "two"))
        conversation.mark_cache_checkpoint()

        marked = marked_blocks(conversation)
        assert len(marked) == 2
        assert first in marked
        assert conversation.history[-1].content[-1] in marked

    def test_markers_never_accumulate_beyond_two(self):
        conversation = Conversation([text_msg("user", "one")])
        for turn in range(6):
            conversation.history.append(text_msg("assistant", f"reply {turn}"))
            conversation.history.append(text_msg("user", f"ask {turn}"))
            conversation.mark_cache_checkpoint()
            assert len(marked_blocks(conversation)) <= 2

    def test_a_wide_parallel_tool_turn_still_leaves_a_marker_behind(self):
        """The case a single marker cannot survive: one turn adding more blocks than the window."""
        conversation = Conversation([text_msg("user", "start")])
        conversation.mark_cache_checkpoint()
        early = conversation.history[-1].content[-1]

        calls = [ToolCall(id=f"t{i}", name="read_file", input={}) for i in range(12)]
        results = [ToolResult(tool_use_id=f"t{i}", name="read_file", content="ok", is_error=False) for i in range(12)]
        conversation.history.append(Message(role="assistant", content=list(calls)))
        conversation.history.append(Message(role="user", content=list(results)))
        conversation.mark_cache_checkpoint()

        marked = marked_blocks(conversation)
        assert early in marked, "the older marker is the whole point: it survives a long jump"
        assert len(marked) == 2

    def test_replaced_history_drops_the_stale_marker(self):
        """Compaction and restore swap blocks wholesale; a dangling reference must not be marked."""
        conversation = Conversation([text_msg("user", "one")])
        conversation.mark_cache_checkpoint()

        conversation.history = [text_msg("user", "fresh")]
        conversation.mark_cache_checkpoint()

        marked = marked_blocks(conversation)
        assert marked == [conversation.history[-1].content[-1]]


class TestUnsupportedBlockTypes:
    def test_thinking_block_is_skipped_in_favour_of_a_cacheable_block(self):
        conversation = Conversation(
            [
                text_msg("user", "ask"),
                Message(role="assistant", content=[TextBlock(text="answer"), ThinkingBlock(thinking="pondering")]),
            ]
        )

        conversation.mark_cache_checkpoint()

        marked = marked_blocks(conversation)
        assert len(marked) == 1
        assert isinstance(marked[0], TextBlock)

    def test_thinking_block_never_serializes_a_breakpoint(self):
        """Belt and braces: Anthropic rejects cache_control on a thinking block."""
        block = ThinkingBlock(thinking="pondering", cache_checkpoint=True)

        assert "cache_control" not in block.to_anthropic()

    def test_redacted_thinking_is_also_skipped(self):
        """Same trap as ThinkingBlock: it carries the flag but never serializes a breakpoint."""
        conversation = Conversation(
            [
                Message(
                    role="assistant",
                    content=[TextBlock(text="answer"), RedactedThinkingBlock(data="opaque")],
                )
            ]
        )

        conversation.mark_cache_checkpoint()

        marked = marked_blocks(conversation)
        assert len(marked) == 1
        assert isinstance(marked[0], TextBlock)

    def test_every_marked_block_type_actually_serializes_its_marker(self):
        """The invariant behind SUPPORTS_CACHE_CONTROL: selection and emission must agree."""
        blocks = [
            TextBlock(text="t", cache_checkpoint=True),
            ToolCall(id="t1", name="read", input={}, cache_checkpoint=True),
            ToolResult(tool_use_id="t1", name="read", content="ok", is_error=False, cache_checkpoint=True),
            ThinkingBlock(thinking="p", cache_checkpoint=True),
            RedactedThinkingBlock(data="opaque", cache_checkpoint=True),
        ]

        for block in blocks:
            emits = "cache_control" in block.to_anthropic()
            assert emits is block.SUPPORTS_CACHE_CONTROL, f"{type(block).__name__} disagrees with its flag"


class TestTtl:
    def test_tool_list_breakpoint_uses_the_long_ttl(self):
        registry = ToolRegistry().add(make_tool("read"), make_tool("write"))

        definitions = registry.definitions()

        assert definitions[-1].to_anthropic()["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "cache_control" not in definitions[0].to_anthropic()

    def test_system_prompt_breakpoint_uses_the_long_ttl(self):
        message = BaseAgent.system_prompt_message("you are a helpful agent")

        block = first_block(message).to_anthropic()
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_rolling_markers_keep_the_default_ttl(self):
        """Rolling markers are rewritten every turn, so the cheaper 5m write is the right trade."""
        conversation = Conversation([text_msg("user", "one")])
        conversation.mark_cache_checkpoint()

        control = marked_blocks(conversation)[0].to_anthropic()["cache_control"]
        assert control == {"type": "ephemeral"}


class TestSerialization:
    """Breakpoints are recomputed per turn, so persisting them would only carry stale state."""

    def test_markers_are_not_persisted(self):
        block = TextBlock(text="hello", cache_checkpoint=True, cache_ttl="1h")

        stored = block.to_dict()

        assert "cache_checkpoint" not in stored
        assert "cache_ttl" not in stored

    def test_a_marker_left_in_an_older_session_file_is_ignored(self):
        """Sessions written before this change still carry the key; loading must not honour it."""
        legacy = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_checkpoint": True},
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "name": "read",
                    "content": "ok",
                    "is_error": False,
                    "cache_checkpoint": True,
                },
            ],
        }

        restored = Message.from_dict(legacy)

        assert isinstance(restored.content, list)
        assert [block.cache_checkpoint for block in restored.content] == [False, False]


class TestTotalBudget:
    def test_a_full_request_stays_within_the_api_limit(self):
        registry = ToolRegistry().add(make_tool("read"), make_tool("write"))
        system = BaseAgent.system_prompt_message("system")
        conversation = Conversation([text_msg("user", "one")])
        conversation.mark_cache_checkpoint()
        conversation.history.append(text_msg("assistant", "reply"))
        conversation.history.append(text_msg("user", "two"))
        conversation.mark_cache_checkpoint()

        assert isinstance(system.content, list)
        breakpoints = (
            sum("cache_control" in definition.to_anthropic() for definition in registry.definitions())
            + sum("cache_control" in block.to_anthropic() for block in system.content)
            + len(marked_blocks(conversation))
        )

        assert breakpoints == MAX_BREAKPOINTS
