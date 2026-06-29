"""Reusable, hermetic fakes for conversation-compaction tests.

No network and no API keys: every test installs a ``FakeLLM`` on the agent so
``count_tokens``/``generate``/``stream`` are fully scripted.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolResult
from kolega_code.llm.providers.models import TokenCount


class FakeStream:
    """Minimal end_turn stream: yields no events, returns one assistant message."""

    def __init__(self, final_message=None):
        self._final = final_message or Message(role="assistant", content=[TextBlock(text="ok")], stop_reason="end_turn")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return self._final


class FakeLLM:
    """Drop-in replacement for ``agent.llm`` (the LLMClient surface BaseAgent uses).

    - ``count_tokens`` returns a scripted sequence of ``TokenCount``s (pops per call
      and repeats the final value), so a single over-budget turn — which counts
      tokens more than once — is fully deterministic. With ``proxy=True`` it instead
      counts a chars/4 proxy of the messages, which lets a test prove that the
      effective token count drops after compaction.
    - ``generate`` returns a scripted summary message.
    - ``stream`` returns a ``FakeStream`` that ends the agent loop after one pass.
    """

    def __init__(
        self, token_script=None, *, summary_text="SUMMARY: condensed older turns.", proxy=False, final_message=None
    ):
        self._token_script = list(token_script) if token_script else None
        self._summary_text = summary_text
        self._proxy = proxy
        self._final_message = final_message
        # Mirror the LLMClient surface: a provider object exposing base_url, which
        # the diagnostics layer reads (agent.llm.provider.base_url).
        self.provider = MagicMock(base_url="https://api.test.example/v1")
        self.count_tokens = AsyncMock(side_effect=self._count_tokens)
        self.generate = AsyncMock(side_effect=self._generate)
        self.stream = AsyncMock(side_effect=self._stream)

    async def _count_tokens(self, *args, **kwargs):
        if self._proxy:
            messages = kwargs.get("messages")
            if messages is None and args:
                messages = args[0]
            total = sum(len(m.get_text_content()) for m in (messages or []))
            return TokenCount(input_tokens=total // 4)
        if self._token_script:
            value = self._token_script.pop(0) if len(self._token_script) > 1 else self._token_script[0]
            return TokenCount(input_tokens=value)
        return TokenCount(input_tokens=0)

    async def _generate(self, *args, **kwargs):
        return Message(role="assistant", content=[TextBlock(text=self._summary_text)], stop_reason="end_turn")

    async def _stream(self, *args, **kwargs):
        # The summary is produced by streaming (compaction) and the turn loop also
        # streams; a single end_turn message carrying the summary text serves both.
        final = self._final_message or Message(
            role="assistant", content=[TextBlock(text=self._summary_text)], stop_reason="end_turn"
        )
        return FakeStream(final)


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def text_msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def skill_msg(name: str = "kolega-skill", body: str = "protected body") -> Message:
    """A user message carrying skill content (protected from compaction)."""
    return Message(role="user", content=[TextBlock(text=f'<skill_content name="{name}">{body}</skill_content>')])


def tool_pair(call_id: str = "t1", name: str = "read_file"):
    """Return (assistant-with-toolcall, user-with-toolresult) — a valid atomic pair."""
    return (
        Message(role="assistant", content=[ToolCall(id=call_id, name=name, input={})]),
        Message(role="user", content=[ToolResult(tool_use_id=call_id, name=name, content="ok", is_error=False)]),
    )


def long_history(n_pairs: int = 6) -> MessageHistory:
    """``n_pairs`` user/assistant exchanges (>= MIN_MESSAGES_TO_COMPRESS for n_pairs>=3)."""
    messages = []
    for i in range(n_pairs):
        messages.append(text_msg("user", f"user turn {i} " + "x" * 40))
        messages.append(text_msg("assistant", f"assistant turn {i} " + "y" * 40))
    return MessageHistory(messages)


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------


def make_agent_config():
    from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig

    def cfg(**extra):
        return ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
            **extra,
        )

    return AgentConfig(
        anthropic_api_key="test_key",
        openai_api_key="test_key",
        long_context_config=cfg(),
        fast_config=cfg(),
        thinking_config=cfg(thinking_effort="medium"),
    )


def build_agent(
    tmp_path, *, connection_manager=None, agent_cls=None, sub_agent=False, llm=None, model_context_length=1000
):
    """Construct an agent wired for hermetic compaction tests.

    Returns ``(agent, connection_manager)``. The agent has a FakeLLM (if provided),
    a stub tool collection + system prompt, and AsyncMock log/chat sinks.
    """
    from kolega_code.agent.baseagent import BaseAgent

    cm = connection_manager or AsyncMock(spec=AgentConnectionManager)
    cls = agent_cls or BaseAgent
    agent = cls(
        project_path=tmp_path,
        workspace_id="test_ws",
        thread_id=str(uuid.uuid4()),
        connection_manager=cm,
        config=make_agent_config(),
        sub_agent=sub_agent,
    )
    agent.send_chat_message = AsyncMock()
    agent.log_info = AsyncMock()
    agent.log_error = AsyncMock()
    agent.system_prompt = Message(role="system", content=[TextBlock(text="test agent")])
    agent.tool_collection = MagicMock()
    agent.tool_collection.get_tool_list = MagicMock(return_value=[])
    if llm is not None:
        agent.llm = llm
    agent.model_context_length = model_context_length
    return agent, cm


def context_update_events(connection_manager):
    """All llm_context_update events broadcast through ``connection_manager``."""
    events = []
    for call in connection_manager.broadcast_event.call_args_list:
        event = call.args[0] if call.args else call.kwargs.get("event")
        if event is not None and getattr(event, "event_type", None) == "llm_context_update":
            events.append(event)
    return events


def compaction_status_events(connection_manager):
    """All compaction_status events broadcast through ``connection_manager``."""
    events = []
    for call in connection_manager.broadcast_event.call_args_list:
        event = call.args[0] if call.args else call.kwargs.get("event")
        if event is not None and getattr(event, "event_type", None) == "compaction_status":
            events.append(event)
    return events
