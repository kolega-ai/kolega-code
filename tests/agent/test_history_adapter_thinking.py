# ruff: noqa: F401,F811,E402
"""Tests for image-block detection and placeholder replacement in conversation history.

Covers the graceful-degradation path: when a non-vision model is active in a thread
that already contains images, the images are replaced with text placeholders on the
request copy (stored history is never mutated), and the compaction-aware detector
only reports images that would actually be sent.
"""

import base64
import io

from PIL import Image

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
    image_format = "JPEG" if media_type == "image/jpeg" else "PNG"
    image = Image.new("RGB", (1, 1), "navy")
    output = io.BytesIO()
    image.save(output, format=image_format)
    return ImageBlock(
        image_type="base64",
        media_type=media_type,
        data=base64.b64encode(output.getvalue()).decode("ascii"),
    )


def _user(*blocks) -> Message:
    return Message(role="user", content=list(blocks))


def _assistant(*blocks, provider: str | None = None) -> Message:
    return Message(
        role="assistant",
        content=list(blocks),
        usage_metadata={"provider": provider} if provider else {},
    )


def test_adapt_drops_kimi_thinking_when_targeting_anthropic():
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
    # Dropped outright: a text stand-in would be echoed back by the model.
    assert not any(isinstance(block, TextBlock) for block in out[0].content)
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


def test_adapt_preserves_ollama_cloud_thinking_when_targeting_ollama_cloud():
    history = [_assistant(ThinkingBlock(thinking="ollama reasoning"), provider="ollama_cloud")]

    out = adapt_history_for_provider(
        history,
        target_provider="ollama_cloud",
        target_model="gpt-oss:20b",
        supports_vision=False,
    )

    assert out is history
    assert isinstance(out[0].content[0], ThinkingBlock)
    assert out[0].content[0].thinking == "ollama reasoning"


def test_adapt_drops_ollama_cloud_thinking_when_targeting_foreign_provider():
    history = [_assistant(ThinkingBlock(thinking="ollama reasoning"), provider="ollama_cloud")]

    out = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-4-8",
        supports_vision=True,
    )

    # Reasoning was the whole message, so the message itself goes away.
    assert out == []


def test_adapt_drops_ollama_cloud_thinking_without_source_provider():
    history = [_assistant(ThinkingBlock(thinking="ollama reasoning"))]

    out = adapt_history_for_provider(
        history,
        target_provider="ollama_cloud",
        target_model="gpt-oss:20b",
        supports_vision=False,
    )

    assert out == []


def test_adapt_drops_thinking_without_source_provider():
    history = [_assistant(ThinkingBlock(thinking="unknown-source reasoning"))]

    out = adapt_history_for_provider(
        history,
        target_provider="fireworks",
        target_model="accounts/fireworks/models/glm-5p2",
        supports_vision=False,
    )

    assert out == []


def test_adapt_preserves_images_for_vision_target_while_dropping_foreign_thinking():
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
    nested_tr = out[0].content[2]
    assert isinstance(nested_tr, ToolResult)
    assert isinstance(nested_tr.content[0], ImageBlock)
    # The reasoning-only assistant message is dropped; the user message remains.
    assert len(out) == 1


def test_adapt_replaces_images_for_non_vision_target_and_drops_foreign_thinking():
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
    nested_tr = out[0].content[2]
    assert isinstance(nested_tr, ToolResult)
    assert isinstance(nested_tr.content[0], TextBlock)
    # Redacted reasoning is dropped too, taking its otherwise-empty message with it.
    assert len(out) == 1


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
    # Storage keeps both messages; only the request copy loses the reasoning one.
    assert len(history) == 2
    assert len(out) == 1


def test_repaired_preserves_usage_metadata_when_rebuilding_tool_result_message():
    conversation = Conversation()
    tool_call = ToolCall(id="tool1", name="read_file", input={"path": "README.md"})
    tool_result = ToolResult(tool_use_id="tool1", name="read_file", content="ok", is_error=False)
    messages = [
        _assistant(tool_call, provider="ollama_cloud"),
        Message(role="user", content=[tool_result], usage_metadata={"provider": "tool_runner"}),
    ]

    repaired = conversation.repaired(messages)

    assert repaired[0].usage_metadata["provider"] == "ollama_cloud"
    assert repaired[1].usage_metadata["provider"] == "tool_runner"


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
    # No text stand-in is emitted, so the tool call is the only remaining block.
    assert len(payload[0]["content"]) == 1
    assert payload[0]["content"][0]["type"] == "tool_use"
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


def test_adapt_drops_responses_reasoning_when_targeting_anthropic():
    block = ResponsesReasoningBlock(encrypted_content="ENC", summary=[], item_id="rs_1")
    history = [_assistant(block, provider="openai_chatgpt")]

    out = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-4-8",
        supports_vision=True,
    )

    assert out == []
    # Storage is untouched, so switching back to OpenAI restores continuity.
    assert isinstance(history[0].content[0], ResponsesReasoningBlock)


# ---------------------------------------------------------------------------
# Scrubbing placeholders that earlier builds already echoed into stored history
# ---------------------------------------------------------------------------


def test_adapt_scrubs_echoed_placeholder_from_assistant_prose():
    """Reproduces the shape seen in a real contaminated session.

    The model reproduced the placeholder as the opening tokens of its own reply, so
    it landed in stored history fused with genuine prose in a single text block.
    """
    text = "[Prior reasoning from openai_chatgpt omitted for compatibility.]I see what is happening."
    stored = _assistant(TextBlock(text=text), provider="google")
    history = [stored]

    out = adapt_history_for_provider(
        history,
        target_provider="google",
        target_model="gemini-3.1-pro-preview",
        supports_vision=True,
    )

    adapted_block = out[0].content[0]
    assert isinstance(adapted_block, TextBlock)
    assert adapted_block.text == "I see what is happening."
    # Request copy only: storage keeps what the model actually produced.
    stored_block = stored.content[0]
    assert isinstance(stored_block, TextBlock)
    assert stored_block.text == text


def test_adapt_scrubs_echoed_placeholder_before_a_tool_call():
    tool_call = ToolCall(id="t1", name="exec_command", input={"command": "ls"})
    history = [
        _assistant(
            TextBlock(text="[Prior reasoning from openai_chatgpt omitted for compatibility.]"),
            tool_call,
            provider="google",
        )
    ]

    out = adapt_history_for_provider(
        history,
        target_provider="google",
        target_model="gemini-3.1-pro-preview",
        supports_vision=True,
    )

    # The text block held nothing else, so it goes; the tool call survives.
    assert len(out[0].content) == 1
    assert isinstance(out[0].content[0], ToolCall)
    assert out[0].content[0].id == "t1"


def test_adapt_drops_message_that_was_only_an_echoed_placeholder():
    history = [_assistant(TextBlock(text="[Prior reasoning from kimi_coding omitted for compatibility.]"))]

    out = adapt_history_for_provider(
        history,
        target_provider="google",
        target_model="gemini-3.1-pro-preview",
        supports_vision=True,
    )

    assert out == []


def test_adapt_scrubs_redacted_placeholder_spelling():
    history = [
        _assistant(
            TextBlock(text="[Prior redacted reasoning from anthropic omitted for compatibility.]done"),
            provider="google",
        )
    ]

    out = adapt_history_for_provider(
        history,
        target_provider="google",
        target_model="gemini-3.1-pro-preview",
        supports_vision=True,
    )

    adapted_block = out[0].content[0]
    assert isinstance(adapted_block, TextBlock)
    assert adapted_block.text == "done"


def test_adapt_leaves_placeholder_text_in_user_messages_alone():
    """A user quoting the phrase (e.g. pasting a transcript) must not be rewritten."""
    text = "why does it say [Prior reasoning from openai_chatgpt omitted for compatibility.] ?"
    history = [_user(TextBlock(text=text))]

    out = adapt_history_for_provider(
        history,
        target_provider="google",
        target_model="gemini-3.1-pro-preview",
        supports_vision=True,
    )

    assert out is history
    user_block = out[0].content[0]
    assert isinstance(user_block, TextBlock)
    assert user_block.text == text


def test_adapt_leaves_ordinary_assistant_prose_untouched():
    history = [_assistant(TextBlock(text="Here is the plan."), provider="google")]

    out = adapt_history_for_provider(
        history,
        target_provider="google",
        target_model="gemini-3.1-pro-preview",
        supports_vision=True,
    )

    assert out is history
