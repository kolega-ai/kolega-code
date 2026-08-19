"""The prefix sent to the provider must stay byte-stable across turns.

This is the property the whole volatile-context mechanism exists to protect. Providers cache on
a prefix match, and the system prompt is hashed ahead of every message, so a single changed byte
in the system prompt re-bills the entire conversation. Editing memory used to do exactly that.

These tests drive real turns through ``BaseAgent`` against a recording fake LLM and compare the
actual request payloads.
"""

import copy
from typing import Any

import pytest

from kolega_code.agent.prompt_provider import AgentMode, AgentType
from kolega_code.memory import ProjectMemoryManager

from .compaction_helpers import FakeLLM, build_agent


class RecordingLLM(FakeLLM):
    """FakeLLM that snapshots each request payload.

    The history is passed by reference and mutated in place between turns (blocks are appended,
    cache markers move), so the payload has to be serialized at call time to be comparable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.payloads: list[dict[str, Any]] = []

    async def _stream(self, *args: Any, **kwargs: Any):
        messages = kwargs.get("messages")
        system = kwargs.get("system")
        tools = kwargs.get("tools")
        self.payloads.append(
            {
                "messages": copy.deepcopy(messages.to_anthropic()) if messages is not None else None,
                "system": copy.deepcopy([block.to_anthropic() for block in system.content]) if system else None,
                "tools": copy.deepcopy([tool.to_anthropic() for tool in tools]) if tools else [],
            }
        )
        return await super()._stream(*args, **kwargs)


def strip_cache_control(value: Any) -> Any:
    """Drop ``cache_control`` markers so payloads can be compared on content alone.

    The rolling breakpoint deliberately moves to the newest block each turn, so its position
    differs between requests. That is placement, not content: the cached prefix is matched on
    the content bytes, which is what these tests are asserting about.
    """
    if isinstance(value, dict):
        return {k: strip_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [strip_cache_control(item) for item in value]
    return value


def reminder_sources(payload: dict[str, Any]) -> list[str]:
    """Every ``source="..."`` attribute appearing in injected reminder blocks in this payload."""
    import re

    sources: list[str] = []
    for message in payload["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                sources.extend(re.findall(r'<system-reminder source="([^"]+)"', block.get("text", "")))
    return sources


@pytest.fixture
def recording_agent(tmp_path):
    llm = RecordingLLM()
    agent, _cm = build_agent(tmp_path, llm=llm, model_context_length=1_000_000)
    manager = ProjectMemoryManager(tmp_path, tmp_path.parent / f"{tmp_path.name}-memory-state")
    agent.memory_manager = manager
    agent.context.services.memory_manager = manager

    # The shared harness stubs system_prompt to a constant, which would make every stability
    # assertion below pass vacuously. Build the real coder prompt instead, so the payloads
    # reflect the actual prompt pipeline — memory policy, guidance handling and all.
    def initialize() -> None:
        # Via system_prompt_message, exactly as every real agent does, so the payload carries the
        # breakpoint and its TTL rather than a bare block.
        agent.system_prompt = agent.system_prompt_message(
            agent.build_agent_system_prompt(AgentType.CODER, AgentMode.CLI)
        )

    agent._initialize_system_prompt = initialize
    initialize()
    return agent, llm, manager


class TestHarnessIsMeaningful:
    """Guard the fixture itself: these tests are worthless against a stubbed prompt."""

    @pytest.mark.asyncio
    async def test_agent_renders_the_real_system_prompt(self, recording_agent):
        agent, llm, _manager = recording_agent

        await run_turn(agent, "first")

        system_text = llm.payloads[0]["system"][0]["text"]
        assert len(system_text) > 2000
        assert "## Private project memory" in system_text
        assert "## Context Updates" in system_text


async def run_turn(agent, message: str) -> None:
    async for _chunk in agent.process_message_stream(message):
        pass


class TestPrefixStability:
    @pytest.mark.asyncio
    async def test_memory_write_does_not_change_the_system_prompt(self, recording_agent):
        """The regression this change exists to prevent."""
        agent, llm, manager = recording_agent

        await run_turn(agent, "first")
        assert manager.write_entry("MEMORY.md", "- A durable project fact.").ok
        await run_turn(agent, "second")
        assert manager.write_entry("MEMORY.md", "- A durable project fact.\n- And another.").ok
        # Rebuild the prompt from scratch, which is what a memory refresh used to do. Even then
        # the bytes must not move: the memory body is no longer part of the prompt at all.
        agent._initialize_system_prompt()
        await run_turn(agent, "third")

        assert len(llm.payloads) == 3
        systems = [payload["system"] for payload in llm.payloads]
        assert systems[0] == systems[1] == systems[2]
        # And the memory itself never reached the prompt.
        assert "A durable project fact" not in systems[2][0]["text"]

    @pytest.mark.asyncio
    async def test_earlier_messages_are_never_rewritten(self, recording_agent):
        """Every block already sent must reappear unchanged, or the prefix match breaks."""
        agent, llm, manager = recording_agent

        await run_turn(agent, "first")
        assert manager.write_entry("MEMORY.md", "- A durable project fact.").ok
        await run_turn(agent, "second")
        await run_turn(agent, "third")

        for earlier, later in zip(llm.payloads, llm.payloads[1:]):
            earlier_messages = strip_cache_control(earlier["messages"])
            later_messages = strip_cache_control(later["messages"])
            # History is append-only: the later request must start with the earlier one verbatim.
            assert later_messages[: len(earlier_messages)] == earlier_messages

    @pytest.mark.asyncio
    async def test_memory_change_is_injected_once(self, recording_agent):
        agent, llm, manager = recording_agent

        await run_turn(agent, "first")
        assert manager.write_entry("MEMORY.md", "- A durable project fact.").ok
        await run_turn(agent, "second")
        await run_turn(agent, "third")

        # Turn 1 seeds whatever exists at session start; turn 2 carries the change; turn 3, having
        # nothing new to say, must add no new volatile reminder at all.
        assert "memory" in reminder_sources(llm.payloads[1])
        assert reminder_sources(llm.payloads[2]) == reminder_sources(llm.payloads[1])

    @pytest.mark.asyncio
    async def test_guidance_edit_is_injected_without_touching_the_prompt(self, recording_agent, tmp_path):
        agent, llm, _manager = recording_agent

        await run_turn(agent, "first")
        (tmp_path / "AGENTS.md").write_text("Prefer tabs.", encoding="utf-8")
        await run_turn(agent, "second")

        assert llm.payloads[0]["system"] == llm.payloads[1]["system"]
        new_sources = set(reminder_sources(llm.payloads[1])) - set(reminder_sources(llm.payloads[0]))
        assert "guidance" in new_sources

    @pytest.mark.asyncio
    async def test_reminder_is_its_own_message_not_appended_to_user_text(self, recording_agent, tmp_path):
        """Keeping it separate is what makes it unambiguous to the model and to the transcript."""
        agent, llm, _manager = recording_agent

        (tmp_path / "AGENTS.md").write_text("Prefer tabs.", encoding="utf-8")
        await run_turn(agent, "do the thing")

        messages = llm.payloads[0]["messages"]
        reminder_messages = [
            message
            for message in messages
            if any(
                isinstance(block, dict) and "<system-reminder" in str(block.get("text", ""))
                for block in message.get("content", [])
                if isinstance(block, dict)
            )
        ]
        session_messages = [
            message
            for message in reminder_messages
            if any(
                isinstance(block, dict) and 'source="session"' in str(block.get("text", ""))
                for block in message.get("content", [])
                if isinstance(block, dict)
            )
        ]
        volatile_messages = [message for message in reminder_messages if message not in session_messages]

        assert len(session_messages) == 1
        assert len(volatile_messages) == 1
        session_reminder = session_messages[0]
        volatile_reminder = volatile_messages[0]

        assert session_reminder["role"] == "user"
        assert volatile_reminder["role"] == "user"
        # Exactly one block in each reminder message, and the user's own text is not in either.
        assert len(session_reminder["content"]) == 1
        assert len(volatile_reminder["content"]) == 1
        assert "do the thing" not in session_reminder["content"][0]["text"]
        assert "do the thing" not in volatile_reminder["content"][0]["text"]
        # The volatile reminder trails the user's message rather than preceding it, so that the
        # in-memory history matches the journal — the journal records the turn's user message first.
        assert "do the thing" in str(messages[1]["content"])
        assert session_reminder is messages[0]
        assert volatile_reminder is messages[2]


class TestInjectionDoesNotBreakToolPairing:
    @pytest.mark.asyncio
    async def test_unanswered_tool_call_is_still_repaired_around_the_injected_turn(self, recording_agent):
        """A cancelled turn can leave an assistant tool_use with no result.

        The injected reminder lands right after it, so the repair pass has to place the
        placeholder results adjacent to the call regardless. Anthropic requires the message
        immediately following a tool_use to carry the matching tool_result.
        """
        from kolega_code.llm.models import Message, TextBlock, ToolCall

        agent, llm, _manager = recording_agent
        agent.conversation.history.append(Message(role="user", content=[TextBlock(text="earlier ask")]))
        agent.conversation.history.append(
            Message(
                role="assistant",
                content=[ToolCall(id="orphan-1", name="read_file", input={})],
                tool_calls=[ToolCall(id="orphan-1", name="read_file", input={})],
                stop_reason="tool_use",
            )
        )

        await run_turn(agent, "next ask")

        messages = llm.payloads[0]["messages"]
        for index, message in enumerate(messages):
            call_ids = {
                block["id"] for block in message["content"] if isinstance(block, dict) and block["type"] == "tool_use"
            }
            if not call_ids:
                continue
            assert index + 1 < len(messages), "a tool_use must not be the final message"
            following = messages[index + 1]
            result_ids = {
                block["tool_use_id"]
                for block in following["content"]
                if isinstance(block, dict) and block["type"] == "tool_result"
            }
            assert call_ids <= result_ids, f"tool_use at {index} is not answered by the next message"


class TestBreakpointBudgetInRealRequests:
    @pytest.mark.asyncio
    async def test_cancelled_tool_followup_stays_within_four_breakpoints(self, recording_agent):
        """Repair must not serialize the marked follow-up twice and create a fifth breakpoint."""
        from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolDefinition
        from kolega_code.tools import Tool, ToolRegistry

        agent, llm, _manager = recording_agent

        async def handler(**inputs):
            return "ok"

        registry = ToolRegistry().add(
            Tool(
                name="write_stdin",
                definition=ToolDefinition(name="write_stdin", description="write", parameters=[]),
                handler=handler,
            )
        )
        agent.tool_collection.get_tool_list = lambda: registry.definitions()

        # Seed both the prior rolling checkpoint and the volatile-context tracker. The next turn
        # then recreates the production failure exactly: an older marker survives while the
        # ordinary user message immediately following an orphaned tool call is the newest marker.
        await run_turn(agent, "prime cache")
        agent.conversation.history.append(Message(role="user", content=[TextBlock(text="earlier ask")]))
        agent.conversation.history.append(
            Message(
                role="assistant",
                content=[ToolCall(id="orphan-1", name="write_stdin", input={})],
                stop_reason="tool_use",
            )
        )

        await run_turn(agent, "continue after cancellation")

        payload = llm.payloads[-1]
        matching_text = [
            block
            for message in payload["messages"]
            for block in message["content"]
            if isinstance(block, dict)
            and block.get("type") == "text"
            and block.get("text") == "continue after cancellation"
        ]
        assert len(matching_text) == 1

        orphan_index = next(
            index
            for index, message in enumerate(payload["messages"])
            if any(isinstance(block, dict) and block.get("id") == "orphan-1" for block in message["content"])
        )
        repaired_following = payload["messages"][orphan_index + 1]
        assert any(
            isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") == "orphan-1"
            for block in repaired_following["content"]
        )

        tools = sum("cache_control" in tool for tool in payload["tools"])
        system = sum("cache_control" in block for block in payload["system"])
        history = sum(
            "cache_control" in block
            for message in payload["messages"]
            for block in message["content"]
            if isinstance(block, dict)
        )
        assert (tools, system, history) == (1, 1, 2)
        assert tools + system + history == 4

    @pytest.mark.asyncio
    async def test_a_multi_turn_request_never_exceeds_four_breakpoints(self, recording_agent):
        """The API rejects a fifth. Assert on real payloads, not on the pieces in isolation."""
        from kolega_code.llm.models import ToolDefinition
        from kolega_code.tools import Tool, ToolRegistry

        agent, llm, manager = recording_agent

        async def handler(**inputs):
            return "ok"

        registry = ToolRegistry().add(
            *(
                Tool(name=name, definition=ToolDefinition(name=name, description=name, parameters=[]), handler=handler)
                for name in ("read", "write", "grep")
            )
        )
        agent.tool_collection.get_tool_list = lambda: registry.definitions()

        await run_turn(agent, "first")
        assert manager.write_entry("MEMORY.md", "- A fact.").ok
        await run_turn(agent, "second")
        await run_turn(agent, "third")

        for index, payload in enumerate(llm.payloads):
            tools = sum("cache_control" in definition.to_anthropic() for definition in registry.definitions())
            system = sum("cache_control" in block for block in payload["system"])
            history = sum(
                "cache_control" in block
                for message in payload["messages"]
                for block in message["content"]
                if isinstance(block, dict)
            )
            total = tools + system + history
            assert total <= 4, (
                f"turn {index} used {total} breakpoints (tools={tools} system={system} history={history})"
            )
            # Tools and system are always marked; the history carries at most the two rolling ones.
            assert tools == 1
            assert system == 1
