# ruff: noqa: F401,F811,E402
"""Tests for image-block detection and placeholder replacement in conversation history.

Covers the graceful-degradation path: when a non-vision model is active in a thread
that already contains images, the images are replaced with text placeholders on the
request copy (stored history is never mutated), and the compaction-aware detector
only reports images that would actually be sent.
"""

import base64
import io

import pytest
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


def _solid_png(width: int, height: int) -> ImageBlock:
    image = Image.new("RGB", (width, height), "navy")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return ImageBlock(
        image_type="base64",
        media_type="image/png",
        data=base64.b64encode(output.getvalue()).decode("ascii"),
    )


def _image_dimensions(block: ImageBlock) -> tuple[int, int]:
    with Image.open(io.BytesIO(base64.b64decode(block.data))) as image:
        return image.size


def _noise_jpeg() -> ImageBlock:
    image = Image.effect_noise((160, 120), 80).convert("RGB")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return ImageBlock(
        image_type="base64",
        media_type="image/jpeg",
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
    assert isinstance(new_tr, ToolResult)
    assert new_tr.cache_checkpoint is True


def test_adapt_keeps_normal_anthropic_dimensions_at_exactly_twenty_images() -> None:
    large = _solid_png(2_001, 100)
    history = [_user(large, *[_image() for _ in range(19)])]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-5",
        supports_vision=True,
    )

    assert count_image_blocks(adapted) == 20
    assert adapted is history
    assert adapted[0].content[0] is large
    assert _image_dimensions(large) == (2_001, 100)


def test_adapt_constrains_direct_and_nested_images_when_request_has_more_than_twenty() -> None:
    top_level = _solid_png(2_001, 100)
    nested = _solid_png(100, 2_001)
    valid = [_image() for _ in range(19)]
    result = ToolResult(
        tool_use_id="many-image-call",
        name="browser_take_screenshot",
        content=[nested, TextBlock(text="caption")],
        is_error=False,
        cache_checkpoint=True,
        execution_id="many-image-exec",
    )
    history = [_user(top_level, *valid), _user(result)]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-5",
        supports_vision=True,
    )

    assert count_image_blocks(adapted) == 21
    adapted_top = adapted[0].content[0]
    assert isinstance(adapted_top, ImageBlock)
    assert adapted_top is not top_level
    assert max(_image_dimensions(adapted_top)) <= 2_000
    assert all(adapted[0].content[index + 1] is block for index, block in enumerate(valid))

    adapted_result = adapted[1].content[0]
    assert isinstance(adapted_result, ToolResult)
    assert adapted_result is not result
    assert adapted_result.tool_use_id == "many-image-call"
    assert adapted_result.name == "browser_take_screenshot"
    assert adapted_result.cache_checkpoint is True
    assert adapted_result.execution_id == "many-image-exec"
    adapted_nested = adapted_result.content[0]
    assert isinstance(adapted_nested, ImageBlock)
    assert adapted_nested is not nested
    assert max(_image_dimensions(adapted_nested)) <= 2_000
    assert isinstance(adapted_result.content[1], TextBlock)
    assert adapted_result.content[1].text == "caption"

    # Only the outbound request receives derivatives.
    assert history[0].content[0] is top_level
    assert history[1].content[0] is result
    assert result.content[0] is nested
    assert _image_dimensions(top_level) == (2_001, 100)
    assert _image_dimensions(nested) == (100, 2_001)


def test_adapt_constrains_single_anthropic_image_to_normal_dimension_limit() -> None:
    oversized = _solid_png(8_001, 2)
    history = [_user(oversized)]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-5",
        supports_vision=True,
    )

    adapted_image = adapted[0].content[0]
    assert isinstance(adapted_image, ImageBlock)
    assert adapted_image is not oversized
    assert max(_image_dimensions(adapted_image)) <= 8_000
    assert history[0].content[0] is oversized
    assert _image_dimensions(oversized) == (8_001, 2)


def test_adapt_resizes_top_level_and_nested_anthropic_images_without_mutating_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kolega_code.utils.images as image_utils

    limit = 1_600
    monkeypatch.setattr(image_utils, "ANTHROPIC_MAX_IMAGE_BASE64_BYTES", limit)
    top_level = _noise_jpeg()
    nested = _noise_jpeg()
    result = ToolResult(
        tool_use_id="image-call",
        name="read_image",
        content=[nested, TextBlock(text="caption")],
        is_error=False,
        cache_checkpoint=True,
        execution_id="image-exec",
    )
    history = [_user(TextBlock(text="look"), top_level), _user(result)]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-5",
        supports_vision=True,
    )

    adapted_top = adapted[0].content[1]
    assert isinstance(adapted_top, ImageBlock)
    assert adapted_top is not top_level
    assert adapted_top.media_type == "image/jpeg"
    assert len(adapted_top.data) < limit
    assert adapted_top.data != top_level.data

    adapted_result = adapted[1].content[0]
    assert isinstance(adapted_result, ToolResult)
    assert adapted_result is not result
    assert adapted_result.tool_use_id == "image-call"
    assert adapted_result.name == "read_image"
    assert adapted_result.is_error is False
    assert adapted_result.cache_checkpoint is True
    assert adapted_result.execution_id == "image-exec"
    adapted_nested = adapted_result.content[0]
    assert isinstance(adapted_nested, ImageBlock)
    assert adapted_nested is not nested
    assert len(adapted_nested.data) < limit
    assert isinstance(adapted_result.content[1], TextBlock)
    assert adapted_result.content[1].text == "caption"

    # The canonical stored history remains full-resolution and untouched.
    assert history[0].content[1] is top_level
    assert history[1].content[0] is result
    assert result.content[0] is nested


def test_adapt_resizes_image_nested_in_compatible_freeform_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    import kolega_code.utils.images as image_utils

    limit = 1_600
    monkeypatch.setattr(image_utils, "ANTHROPIC_MAX_IMAGE_BASE64_BYTES", limit)
    nested = _noise_jpeg()
    result = ToolResult(
        tool_use_id="freeform-image-call",
        name="read_image",
        content=[nested],
        is_error=False,
        input_kind="freeform",
    )
    history = [
        Message(
            role="user",
            content=[result],
            usage_metadata={"provider": "anthropic"},
        )
    ]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-5",
        supports_vision=True,
    )

    adapted_result = adapted[0].content[0]
    assert isinstance(adapted_result, ToolResult)
    assert isinstance(adapted_result.content, list)
    adapted_image = adapted_result.content[0]
    assert isinstance(adapted_image, ImageBlock)
    assert len(adapted_image.data) < limit
    assert adapted_image.data != nested.data
    assert result.content[0] is nested


def test_adapt_applies_many_image_dimension_to_compatible_freeform_tool_result() -> None:
    nested = _solid_png(2_001, 100)
    result = ToolResult(
        tool_use_id="freeform-many-image-call",
        name="browser_take_screenshot",
        content=[nested],
        is_error=False,
        input_kind="freeform",
    )
    history = [
        Message(
            role="user",
            content=[*[_image() for _ in range(20)], result],
            usage_metadata={"provider": "anthropic"},
        )
    ]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-5",
        supports_vision=True,
    )

    adapted_result = adapted[0].content[-1]
    assert isinstance(adapted_result, ToolResult)
    assert isinstance(adapted_result.content, list)
    adapted_image = adapted_result.content[0]
    assert isinstance(adapted_image, ImageBlock)
    assert max(_image_dimensions(adapted_image)) <= 2_000
    assert result.content[0] is nested
    assert _image_dimensions(nested) == (2_001, 100)


def test_adapt_leaves_valid_url_and_non_anthropic_images_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    import kolega_code.utils.images as image_utils

    monkeypatch.setattr(image_utils, "ANTHROPIC_MAX_IMAGE_BASE64_BYTES", 128)
    valid = _image()
    url = ImageBlock(image_type="url", media_type="image/png", data="https://example.com/image.png")
    oversized = _noise_jpeg()
    history = [_user(valid, url, oversized)]

    anthropic = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-5",
        supports_vision=True,
    )
    assert anthropic[0].content[0] is valid
    assert anthropic[0].content[1] is url

    openai = adapt_history_for_provider(
        history,
        target_provider="openai",
        target_model="gpt-5.4",
        supports_vision=True,
    )
    assert openai is history
    assert openai[0].content[2] is oversized


def test_adapt_omits_oversized_anthropic_image_when_resize_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import kolega_code.utils.images as image_utils

    monkeypatch.setattr(image_utils, "ANTHROPIC_MAX_IMAGE_BASE64_BYTES", 64)
    invalid = ImageBlock(
        image_type="base64",
        media_type="image/png",
        data=base64.b64encode(b"not-an-image" * 100).decode("ascii"),
        cache_checkpoint=True,
    )
    history = [_user(invalid)]

    adapted = adapt_history_for_provider(
        history,
        target_provider="anthropic",
        target_model="claude-opus-5",
        supports_vision=True,
    )

    placeholder = adapted[0].content[0]
    assert isinstance(placeholder, TextBlock)
    assert "could not be resized safely" in placeholder.text
    assert placeholder.cache_checkpoint is True
    assert history[0].content[0] is invalid
