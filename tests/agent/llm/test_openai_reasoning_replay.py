"""Reasoning replay for the OpenAI-compatible Chat Completions path.

Prior reasoning captured from reasoning models (DeepSeek, Fireworks, Ollama
Cloud) must be replayed via each provider's native reasoning field
(``reasoning_content`` / ``reasoning``) on the assistant message, not re-injected
as visible ``*Thinking:*`` prompt text. Providers/models without native support
keep the visible-text fallback. Foreign-provider reasoning never serializes as
native replay metadata.
"""

from kolega_code.llm.models import (
    Message,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)
from kolega_code.llm.specs.thinking import reasoning_replay_field

DEEPSEEK_MODEL = "deepseek-v4-pro"
FIREWORKS_MODEL = "accounts/fireworks/models/glm-5p2"
OLLAMA_MODEL = "deepseek-v3.2"


def _assistant(*blocks, provider):
    return Message(role="assistant", content=list(blocks), usage_metadata={"provider": provider})


# --- resolver -------------------------------------------------------------


def test_reasoning_replay_field_deepseek_fireworks_xai_use_reasoning_content():
    assert reasoning_replay_field("deepseek", DEEPSEEK_MODEL) == "reasoning_content"
    assert reasoning_replay_field("fireworks", FIREWORKS_MODEL) == "reasoning_content"
    # xAI's Chat Completions endpoint returns/accepts reasoning_content (verified live).
    assert reasoning_replay_field("xai", "grok-4.3") == "reasoning_content"


def test_reasoning_replay_field_ollama_uses_reasoning():
    assert reasoning_replay_field("ollama_cloud", OLLAMA_MODEL) == "reasoning"


def test_reasoning_replay_field_none_for_unmapped_provider_unknown_and_non_reasoning():
    # Provider not in the map at all.
    assert reasoning_replay_field("together", "anything") is None
    assert reasoning_replay_field("groq", "anything") is None
    # Mapped provider but a non-reasoning model on it (no thinking_effort spec).
    assert reasoning_replay_field("xai", "grok-build-0.1") is None
    # Unknown provider/model pair (get_model_specs raises -> None).
    assert reasoning_replay_field("deepseek", "no-such-model") is None


# --- native serialization -------------------------------------------------


def test_deepseek_thinking_block_serializes_as_reasoning_content():
    asst = _assistant(
        ThinkingBlock(thinking="let me think"),
        TextBlock("the answer"),
        ToolCall(id="t1", name="get_weather", input={"city": "Paris"}),
        provider="deepseek",
    )

    out = MessageHistory([asst]).to_openai(provider="deepseek", model=DEEPSEEK_MODEL)[0]

    assert out["reasoning_content"] == "let me think"
    # Answer text stays in content; thinking is pulled out.
    assert out["content"] == [{"type": "text", "text": "the answer"}]
    assert "*Thinking:*" not in str(out["content"])
    # Tool calls preserved.
    assert out["tool_calls"][0]["id"] == "t1"
    assert out["tool_calls"][0]["function"]["name"] == "get_weather"


def test_ollama_cloud_uses_reasoning_not_reasoning_content():
    asst = _assistant(ThinkingBlock(thinking="r"), TextBlock("a"), provider="ollama_cloud")

    out = MessageHistory([asst]).to_openai(provider="ollama_cloud", model=OLLAMA_MODEL)[0]

    assert out["reasoning"] == "r"
    assert "reasoning_content" not in out
    assert out["content"] == [{"type": "text", "text": "a"}]


def test_fireworks_uses_reasoning_content():
    asst = _assistant(ThinkingBlock(thinking="r"), TextBlock("a"), provider="fireworks")

    out = MessageHistory([asst]).to_openai(provider="fireworks", model=FIREWORKS_MODEL)[0]

    assert out["reasoning_content"] == "r"
    assert out["content"] == [{"type": "text", "text": "a"}]


def test_multiple_thinking_blocks_join_with_blank_line():
    asst = _assistant(
        ThinkingBlock(thinking="first"),
        ThinkingBlock(thinking="second"),
        TextBlock("a"),
        provider="deepseek",
    )

    out = MessageHistory([asst]).to_openai(provider="deepseek", model=DEEPSEEK_MODEL)[0]

    assert out["reasoning_content"] == "first\n\nsecond"


def test_empty_content_normalized_to_empty_string_with_tool_calls():
    # [ThinkingBlock, ToolCall]: after pulling reasoning out, content is empty.
    asst = _assistant(
        ThinkingBlock(thinking="r"),
        ToolCall(id="t1", name="read_file", input={"path": "a.py"}),
        provider="deepseek",
    )

    out = MessageHistory([asst]).to_openai(provider="deepseek", model=DEEPSEEK_MODEL)[0]

    assert out["content"] == ""
    assert out["reasoning_content"] == "r"
    assert out["tool_calls"][0]["id"] == "t1"


def test_native_serialization_through_tool_result_partition_path():
    # An assistant message that also carries a ToolResult goes through the
    # temp_message branch of MessageHistory.to_openai; it must still serialize
    # reasoning natively.
    asst = _assistant(
        ThinkingBlock(thinking="r"),
        TextBlock("a"),
        ToolCall(id="t1", name="f", input={}),
        ToolResult(tool_use_id="t1", name="f", content="ok", is_error=False),
        provider="deepseek",
    )

    out = MessageHistory([asst]).to_openai(provider="deepseek", model=DEEPSEEK_MODEL)

    assistant_msg = out[0]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["reasoning_content"] == "r"
    assert assistant_msg["content"] == [{"type": "text", "text": "a"}]
    # The tool result becomes its own role=tool message.
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "t1" for m in out)


# --- fallback / safety ----------------------------------------------------


def test_non_reasoning_target_keeps_visible_text_fallback():
    asst = _assistant(ThinkingBlock(thinking="r"), TextBlock("a"), provider="together")

    out = MessageHistory([asst]).to_openai(provider="together", model="some-chat-model")[0]

    assert "reasoning_content" not in out
    assert out["content"] == [{"type": "text", "text": "*Thinking:*\nr"}, {"type": "text", "text": "a"}]


def test_default_call_without_provider_is_unchanged():
    asst = _assistant(ThinkingBlock(thinking="r"), TextBlock("a"), provider="deepseek")

    out = MessageHistory([asst]).to_openai()[0]

    assert "reasoning_content" not in out
    assert out["content"][0] == {"type": "text", "text": "*Thinking:*\nr"}


def test_foreign_provider_reasoning_not_replayed_natively():
    # Reasoning produced by anthropic, target deepseek: must NOT become
    # reasoning_content (same-provider gate); falls back to visible text.
    asst = _assistant(ThinkingBlock(thinking="secret"), TextBlock("a"), provider="anthropic")

    out = MessageHistory([asst]).to_openai(provider="deepseek", model=DEEPSEEK_MODEL)[0]

    assert "reasoning_content" not in out
    assert {"type": "text", "text": "*Thinking:*\nsecret"} in out["content"]


def test_cross_provider_chat_reasoning_not_replayed_natively():
    # deepseek-origin reasoning, target fireworks: different provider -> fallback.
    asst = _assistant(ThinkingBlock(thinking="r"), TextBlock("a"), provider="deepseek")

    out = MessageHistory([asst]).to_openai(provider="fireworks", model=FIREWORKS_MODEL)[0]

    assert "reasoning_content" not in out
    assert {"type": "text", "text": "*Thinking:*\nr"} in out["content"]


def test_reasoning_only_message_not_dropped():
    # No answer text, no tool calls: pulling reasoning out would empty the
    # message; keep the visible-text fallback so it isn't dropped.
    asst = _assistant(ThinkingBlock(thinking="only"), provider="deepseek")

    out = MessageHistory([asst]).to_openai(provider="deepseek", model=DEEPSEEK_MODEL)

    assert len(out) == 1
    assert out[0]["content"] == [{"type": "text", "text": "*Thinking:*\nonly"}]
    assert "reasoning_content" not in out[0]


def test_streamed_deepseek_reasoning_round_trips_to_reasoning_content():
    # Capture: streamed reasoning_content becomes a ThinkingBlock.
    msg = Message.from_openai_stream(
        role="assistant",
        reasoning_content="streamed cot",
        content="final",
        stop_reason="stop",
    )
    assert isinstance(msg.content[0], ThinkingBlock)
    # Stamp provider as the live provider does, then replay.
    msg.usage_metadata["provider"] = "deepseek"

    out = MessageHistory([msg]).to_openai(provider="deepseek", model=DEEPSEEK_MODEL)[0]

    assert out["reasoning_content"] == "streamed cot"
    assert out["content"] == [{"type": "text", "text": "final"}]
