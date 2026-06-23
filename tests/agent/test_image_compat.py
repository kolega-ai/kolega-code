"""Tests for image-block detection and placeholder replacement in conversation history.

Covers the graceful-degradation path: when a non-vision model is active in a thread
that already contains images, the images are replaced with text placeholders on the
request copy (stored history is never mutated), and the compaction-aware detector
only reports images that would actually be sent.
"""

from kolega_code.agent.conversation import (
    Conversation,
    adapt_history_for_provider,
    count_image_blocks,
    replace_image_blocks_with_placeholders,
)
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    RedactedThinkingBlock,
    ResponsesReasoningBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)


def _image(media_type: str = "image/png") -> ImageBlock:
    return ImageBlock(image_type="base64", media_type=media_type, data="ZmFrZQ==")


def _user(*blocks) -> Message:
    return Message(role="user", content=list(blocks))


def _assistant(*blocks, provider: str | None = None) -> Message:
    return Message(
        role="assistant",
        content=list(blocks),
        usage_metadata={"provider": provider} if provider else {},
    )


# ---------------------------------------------------------------------------
# adapt_history_for_provider
# ---------------------------------------------------------------------------


def test_adapt_converts_kimi_thinking_when_targeting_anthropic():
    tool_call = ToolCall(id="tool1", name="read_image", input={"path": "a.png"})
    history = [
        _assistant(ThinkingBlock(thinking="foreign reasoning", signature="kimi-sig"), tool_call, provider="kimi_coding")
    ]

    out = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-4-8",
        supports_vision=True,
    )

    assert out[0] is not history[0]
    assert not any(isinstance(block, ThinkingBlock) for block in out[0].content)
    assert isinstance(out[0].content[0], TextBlock)
    assert "Prior reasoning from kimi_coding omitted" in out[0].content[0].text
    assert any(isinstance(block, ToolCall) and block.id == "tool1" for block in out[0].content)


def test_adapt_preserves_anthropic_thinking_when_targeting_anthropic():
    history = [_assistant(ThinkingBlock(thinking="native reasoning", signature="anthropic-sig"), provider="anthropic")]

    out = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-4-8",
        supports_vision=True,
    )

    assert out is history
    assert isinstance(out[0].content[0], ThinkingBlock)
    assert out[0].content[0].signature == "anthropic-sig"


def test_adapt_preserves_zai_thinking_when_targeting_zai():
    history = [_assistant(ThinkingBlock(thinking="native reasoning", signature="zai-sig"), provider="zai")]

    out = adapt_history_for_provider(
        history,
        target_provider="zai",
        target_model="glm-5.2",
        supports_vision=False,
    )

    assert out is history
    assert isinstance(out[0].content[0], ThinkingBlock)
    assert out[0].content[0].signature == "zai-sig"


def test_adapt_preserves_deepseek_thinking_when_targeting_deepseek():
    history = [_assistant(ThinkingBlock(thinking="native reasoning", signature="deepseek-sig"), provider="deepseek")]

    out = adapt_history_for_provider(
        history,
        target_provider="deepseek",
        target_model="deepseek-v4-pro",
        supports_vision=False,
    )

    assert out is history
    assert isinstance(out[0].content[0], ThinkingBlock)
    assert out[0].content[0].signature == "deepseek-sig"


def test_adapt_preserves_fireworks_thinking_when_targeting_fireworks():
    history = [_assistant(ThinkingBlock(thinking="native reasoning"), provider="fireworks")]

    out = adapt_history_for_provider(
        history,
        target_provider="fireworks",
        target_model="accounts/fireworks/models/glm-5p2",
        supports_vision=False,
    )

    assert out is history
    assert isinstance(out[0].content[0], ThinkingBlock)
    assert out[0].content[0].thinking == "native reasoning"


def test_adapt_converts_thinking_without_source_provider():
    history = [_assistant(ThinkingBlock(thinking="unknown-source reasoning"))]

    out = adapt_history_for_provider(
        history,
        target_provider="fireworks",
        target_model="accounts/fireworks/models/glm-5p2",
        supports_vision=False,
    )

    assert out[0] is not history[0]
    assert isinstance(out[0].content[0], TextBlock)
    assert "Prior reasoning from unknown provider omitted" in out[0].content[0].text


def test_adapt_preserves_images_for_vision_target_while_converting_foreign_thinking():
    tr = ToolResult(tool_use_id="t1", name="read_image", content=[_image("image/jpeg")], is_error=False)
    history = [
        _user(TextBlock(text="look"), _image("image/png"), tr),
        _assistant(ThinkingBlock(thinking="foreign", signature="kimi-sig"), provider="kimi_coding"),
    ]

    out = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-4-8",
        supports_vision=True,
    )

    assert count_image_blocks(out) == 2
    assert isinstance(out[0].content[1], ImageBlock)
    assert isinstance(out[0].content[2].content[0], ImageBlock)
    assert isinstance(out[1].content[0], TextBlock)


def test_adapt_replaces_images_for_non_vision_target_and_converts_foreign_thinking():
    tr = ToolResult(tool_use_id="t1", name="read_image", content=[_image("image/jpeg")], is_error=False)
    history = [
        _user(TextBlock(text="look"), _image("image/png"), tr),
        _assistant(RedactedThinkingBlock(data="encrypted"), provider="kimi_coding"),
    ]

    out = adapt_history_for_provider(
        history,
        target_provider="deepseek",
        target_model="deepseek-v4-pro",
        supports_vision=False,
    )

    assert count_image_blocks(out) == 0
    assert isinstance(out[0].content[1], TextBlock)
    assert "not visible" in out[0].content[1].text
    assert isinstance(out[0].content[2].content[0], TextBlock)
    assert isinstance(out[1].content[0], TextBlock)
    assert "redacted reasoning from kimi_coding omitted" in out[1].content[0].text


def test_adapt_does_not_mutate_stored_history():
    image = _image("image/png")
    thinking = ThinkingBlock(thinking="foreign", signature="kimi-sig")
    history = [_user(image), _assistant(thinking, provider="kimi_coding")]

    out = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-4-8",
        supports_vision=True,
    )

    assert out is not history
    assert isinstance(history[0].content[0], ImageBlock)
    assert history[0].content[0] is image
    assert isinstance(history[1].content[0], ThinkingBlock)
    assert history[1].content[0] is thinking
    assert isinstance(out[1].content[0], TextBlock)


def test_adapted_kimi_thinking_is_not_serialized_as_anthropic_thinking():
    tool_call = ToolCall(id="tool1", name="read_file", input={"path": "README.md"})
    result = ToolResult(tool_use_id="tool1", name="read_file", content="ok", is_error=False)
    history = [
        _assistant(ThinkingBlock(thinking="foreign", signature="kimi-sig"), tool_call, provider="kimi_coding"),
        _user(result),
    ]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-4-8",
        supports_vision=True,
    )
    payload = [message.to_anthropic() for message in adapted]

    assert payload[0]["role"] == "assistant"
    assert all(block.get("type") != "thinking" for block in payload[0]["content"])
    assert payload[0]["content"][1]["type"] == "tool_use"
    assert payload[1]["role"] == "user"
    assert payload[1]["content"][0]["type"] == "tool_result"
    assert payload[1]["content"][0]["tool_use_id"] == "tool1"


def test_adapt_preserves_responses_reasoning_when_targeting_chatgpt():
    block = ResponsesReasoningBlock(encrypted_content="ENC", summary=["plan"], item_id="rs_1")
    tool_call = ToolCall(id="t1", name="read_file", input={"path": "a.py"})
    history = [_assistant(block, tool_call, provider="openai_chatgpt")]

    out = adapt_history_for_provider(
        history,
        target_provider="openai_chatgpt",
        target_model="gpt-5.5",
        supports_vision=True,
    )

    # Same provider -> reasoning is preserved so continuity keeps working.
    assert out is history
    assert isinstance(out[0].content[0], ResponsesReasoningBlock)
    assert out[0].content[0].encrypted_content == "ENC"


def test_adapt_preserves_responses_reasoning_across_openai_backends():
    # api-key `openai` and `openai_chatgpt` are the same OpenAI Responses API, so
    # reasoning produced by one replays cleanly to the other — preserved, not a
    # placeholder. (Both directions.)
    for source, target in (("openai_chatgpt", "openai"), ("openai", "openai_chatgpt")):
        block = ResponsesReasoningBlock(encrypted_content="ENC", summary=["plan"], item_id="rs_1")
        tool_call = ToolCall(id="t1", name="read_file", input={"path": "a.py"})
        history = [_assistant(block, tool_call, provider=source)]

        out = adapt_history_for_provider(
            history,
            target_provider=target,
            target_model="gpt-5.5",
            supports_vision=True,
        )

        assert out is history  # nothing changed -> reasoning preserved
        assert isinstance(out[0].content[0], ResponsesReasoningBlock)
        assert out[0].content[0].encrypted_content == "ENC"


def test_adapt_converts_responses_reasoning_when_targeting_anthropic():
    block = ResponsesReasoningBlock(encrypted_content="ENC", summary=[], item_id="rs_1")
    history = [_assistant(block, provider="openai_chatgpt")]

    out = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-4-8",
        supports_vision=True,
    )

    assert out[0] is not history[0]
    assert isinstance(out[0].content[0], TextBlock)
    assert "Prior reasoning from openai_chatgpt omitted" in out[0].content[0].text


# ---------------------------------------------------------------------------
# count_image_blocks
# ---------------------------------------------------------------------------


def test_count_image_blocks_empty_history():
    assert count_image_blocks([]) == 0


def test_count_image_blocks_no_images():
    history = [_user(TextBlock(text="hi")), _assistant(TextBlock(text="hello"))]
    assert count_image_blocks(history) == 0


def test_count_image_blocks_user_message_image():
    history = [_user(TextBlock(text="look"), _image())]
    assert count_image_blocks(history) == 1


def test_count_image_blocks_multiple_top_level():
    history = [_user(_image("image/png"), _image("image/jpeg"))]
    assert count_image_blocks(history) == 2


def test_count_image_blocks_nested_in_tool_result():
    tr = ToolResult(tool_use_id="t1", name="read_image", content=[_image()], is_error=False)
    history = [_user(tr)]
    assert count_image_blocks(history) == 1


def test_count_image_blocks_nested_and_top_level():
    tr = ToolResult(tool_use_id="t1", name="read_image", content=[_image(), TextBlock(text="ok")], is_error=False)
    history = [_user(TextBlock(text="see"), _image()), _user(tr)]
    assert count_image_blocks(history) == 2


def test_count_image_blocks_ignores_string_content():
    history = [_user("just a string")]
    assert count_image_blocks(history) == 0


# ---------------------------------------------------------------------------
# Conversation.has_image_blocks (compaction-aware)
# ---------------------------------------------------------------------------


def test_has_image_blocks_true_for_user_image():
    conv = Conversation([_user(TextBlock(text="hi"), _image())])
    assert conv.has_image_blocks() is True


def test_has_image_blocks_true_for_tool_result_image():
    tr = ToolResult(tool_use_id="t1", name="read_image", content=[_image()], is_error=False)
    conv = Conversation([_user(tr)])
    assert conv.has_image_blocks() is True


def test_has_image_blocks_false_for_text_only():
    conv = Conversation([_user(TextBlock(text="hi")), _assistant(TextBlock(text="hello"))])
    assert conv.has_image_blocks() is False


def test_has_image_blocks_false_when_folded_into_summary():
    """Images in the compacted prefix are gone from the effective history."""
    conv = Conversation([_user(TextBlock(text="u0"), _image()), _assistant(TextBlock(text="a0"))])
    conv.apply_compaction("SUMMARY", split_point=2)  # fold both messages into the summary
    assert conv.has_image_blocks() is False


def test_has_image_blocks_true_when_image_in_verbatim_tail():
    conv = Conversation(
        [_user(TextBlock(text="u0")), _assistant(TextBlock(text="a0")), _user(TextBlock(text="u1"), _image())]
    )
    conv.apply_compaction("SUMMARY", split_point=2)  # keep the image-bearing user message verbatim
    assert conv.has_image_blocks() is True


# ---------------------------------------------------------------------------
# replace_image_blocks_with_placeholders
# ---------------------------------------------------------------------------


def test_replace_returns_same_object_for_messages_without_images():
    m = _user(TextBlock(text="hi"))
    out = replace_image_blocks_with_placeholders([m], "deepseek-v4-pro")
    assert out[0] is m


def test_replace_does_not_mutate_input():
    original = _user(TextBlock(text="look"), _image("image/png"))
    snapshot_content = list(original.content)
    _ = replace_image_blocks_with_placeholders([original], "deepseek-v4-pro")
    assert original.content == snapshot_content
    assert isinstance(original.content[1], ImageBlock)


def test_replace_top_level_image_with_placeholder():
    history = [_user(TextBlock(text="look"), _image("image/png"))]
    out = replace_image_blocks_with_placeholders(history, "deepseek-v4-pro")
    assert out is not history
    assert out[0] is not history[0]
    blocks = out[0].content
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "look"
    assert isinstance(blocks[1], TextBlock)
    assert "image/png" in blocks[1].text
    assert "deepseek-v4-pro" in blocks[1].text
    assert "not visible" in blocks[1].text


def test_replace_preserves_message_metadata():
    tool_call = ToolCall(id="call1", name="read_file", input={"path": "a.py"})
    msg = Message(
        role="assistant",
        content=[TextBlock(text="thinking"), tool_call, _image("image/jpeg")],
        stop_reason="end_turn",
        tool_calls=[tool_call],
        usage_metadata={"input_tokens": 10},
    )
    out = replace_image_blocks_with_placeholders([msg], "glm-5.2")
    new_msg = out[0]
    assert new_msg.role == "assistant"
    assert new_msg.stop_reason == "end_turn"
    assert new_msg.tool_calls == [tool_call]
    assert new_msg.usage_metadata == {"input_tokens": 10}
    # Tool call block survives if present alongside an image in the same message.
    assert any(isinstance(b, ToolCall) for b in new_msg.content)


def test_replace_nested_tool_result_image():
    tr = ToolResult(
        tool_use_id="t1",
        name="read_image",
        content=[_image("image/png"), TextBlock(text="caption")],
        is_error=False,
        execution_id="exec-1",
    )
    history = [_user(tr)]
    out = replace_image_blocks_with_placeholders(history, "deepseek-v4-pro")
    new_tr = out[0].content[0]
    assert isinstance(new_tr, ToolResult)
    assert new_tr is not tr  # new object, input not mutated
    assert new_tr.tool_use_id == "t1"
    assert new_tr.name == "read_image"
    assert new_tr.is_error is False
    assert new_tr.execution_id == "exec-1"
    assert isinstance(new_tr.content[0], TextBlock)
    assert "image/png" in new_tr.content[0].text
    assert isinstance(new_tr.content[1], TextBlock)
    assert new_tr.content[1].text == "caption"
    # original untouched
    assert isinstance(tr.content[0], ImageBlock)


def test_replace_preserves_string_tool_result():
    tr = ToolResult(tool_use_id="t1", name="read_file", content="plain text result", is_error=False)
    history = [_user(tr)]
    out = replace_image_blocks_with_placeholders(history, "deepseek-v4-pro")
    assert out[0] is history[0]  # no images -> unchanged


def test_replace_mixed_history_only_changes_image_messages():
    u_img = _user(TextBlock(text="look"), _image())
    a_text = _assistant(TextBlock(text="ok"))
    u_text = _user(TextBlock(text="next"))
    history = [u_img, a_text, u_text]
    out = replace_image_blocks_with_placeholders(history, "deepseek-v4-pro")
    assert out[0] is not u_img
    assert out[1] is a_text  # unchanged
    assert out[2] is u_text  # unchanged
    assert not any(isinstance(b, ImageBlock) for b in out[0].content)


def test_replace_eliminates_all_image_blocks():
    tr = ToolResult(tool_use_id="t1", name="read_image", content=[_image()], is_error=False)
    history = [_user(TextBlock(text="a"), _image()), _user(tr), _assistant(TextBlock(text="b"))]
    out = replace_image_blocks_with_placeholders(history, "deepseek-v4-pro")
    assert count_image_blocks(out) == 0


def test_replace_preserves_cache_checkpoint_on_tool_result():
    tr = ToolResult(
        tool_use_id="t1",
        name="read_image",
        content=[_image()],
        is_error=False,
        cache_checkpoint=True,
    )
    history = [_user(tr)]
    out = replace_image_blocks_with_placeholders(history, "deepseek-v4-pro")
    new_tr = out[0].content[0]
    assert new_tr.cache_checkpoint is True
