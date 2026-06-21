"""Tests for coder agent image attachment handling."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.coder import CoderAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.llm.models import ImageBlock, Message, TextBlock
from kolega_code.llm.providers.models import TokenCount
from kolega_code.agent.prompt_provider import AgentMode


def _deepseek_config() -> AgentConfig:
    model_config = ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model="deepseek-v4-pro",
        rate_limits=RateLimitConfig(),
    )
    return AgentConfig(
        deepseek_api_key="test-key",
        long_context_config=model_config,
        fast_config=model_config,
        thinking_config=model_config,
    )


def _image_attachment() -> dict:
    return {
        "type": "image",
        "media_type": "image/png",
        "data": base64.b64encode(b"fake-image-data").decode("utf-8"),
        "filename": "test-image.png",
    }


class _EmptyStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return Message("assistant", [TextBlock("done")], stop_reason="end_turn")


def test_non_vision_image_attachment_is_rejected_by_provider_check():
    agent = object.__new__(BaseAgent)
    agent.primary_model_config = SimpleNamespace(
        provider=ModelProvider.DEEPSEEK.value, model="deepseek-v4-pro"
    )

    message = agent._unsupported_attachment_message([_image_attachment()])
    assert message is not None
    assert "does not support image input" in message
    assert "deepseek-v4-pro" in message
    assert "not sent to the model" in message


def test_vision_image_attachment_is_allowed():
    agent = object.__new__(BaseAgent)
    agent.primary_model_config = SimpleNamespace(
        provider=ModelProvider.ANTHROPIC, model="claude-opus-4-8"
    )
    assert agent._unsupported_attachment_message([_image_attachment()]) is None


def test_attachment_check_allows_non_images_for_non_vision_model():
    agent = object.__new__(BaseAgent)
    agent.primary_model_config = SimpleNamespace(
        provider=ModelProvider.DEEPSEEK, model="deepseek-v4-pro"
    )
    assert agent._unsupported_attachment_message(None) is None
    assert agent._unsupported_attachment_message([{"type": "document", "data": "abc"}]) is None
    assert agent._unsupported_attachment_message([{"type": "file", "path": "a.py", "content": "x"}]) is None


@pytest.mark.asyncio
async def test_coder_agent_rejects_deepseek_image_without_llm_call(tmp_path):
    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()
    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=connection_manager,
        config=_deepseek_config(),
        agent_mode=AgentMode.CLI,
    )
    agent.llm = Mock()

    chunks = [
        chunk
        async for chunk in agent.process_message_stream("What is in this image?", [_image_attachment()])
    ]

    assert len(chunks) == 1
    assert chunks[0]["type"] == "response"
    assert "does not support image input" in chunks[0]["content"]
    assert "deepseek-v4-pro" in chunks[0]["content"]
    assert "not sent to the model" in chunks[0]["content"]
    assert chunks[0]["complete"] is True
    assert agent.history == []
    agent.llm.stream.assert_not_called()


@pytest.mark.asyncio
async def test_coder_agent_does_not_print_context_token_counts(tmp_path, capsys):
    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()
    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=connection_manager,
        config=_deepseek_config(),
        agent_mode=AgentMode.CLI,
    )
    agent.count_current_context = AsyncMock(return_value=TokenCount(input_tokens=42))
    agent.llm = Mock()
    agent.llm.stream = AsyncMock(return_value=_EmptyStream())

    chunks = [chunk async for chunk in agent.process_message_stream("hello")]

    assert chunks[-1]["complete"] is True
    assert capsys.readouterr().out == ""


class TestCoderAgentAttachments:
    """Test suite for verifying image attachment handling logic."""

    def test_single_image_attachment_processing(self):
        """Test that a single image attachment is correctly processed into an ImageBlock."""
        test_image_data = base64.b64encode(b"fake-image-data").decode("utf-8")
        attachment = {
            "type": "image",
            "media_type": "image/png",
            "data": test_image_data,
            "filename": "test-image.png",
        }

        # Process the attachment as the coder agent would
        image_block = ImageBlock(
            image_type="base64", media_type=attachment.get("media_type", "image/png"), data=attachment["data"]
        )

        assert image_block.image_type == "base64"
        assert image_block.media_type == "image/png"
        assert image_block.data == test_image_data

    def test_multiple_image_attachments_processing(self):
        """Test that multiple image attachments are correctly processed."""
        attachments = [
            {
                "type": "image",
                "media_type": "image/png",
                "data": base64.b64encode(b"image1").decode("utf-8"),
                "filename": "image1.png",
            },
            {
                "type": "image",
                "media_type": "image/jpeg",
                "data": base64.b64encode(b"image2").decode("utf-8"),
                "filename": "image2.jpg",
            },
        ]

        # Process attachments as the coder agent would
        image_blocks = []
        for attachment in attachments:
            if attachment.get("type") == "image":
                image_block = ImageBlock(
                    image_type="base64", media_type=attachment.get("media_type", "image/png"), data=attachment["data"]
                )
                image_blocks.append(image_block)

        assert len(image_blocks) == 2
        assert image_blocks[0].media_type == "image/png"
        assert image_blocks[1].media_type == "image/jpeg"

    def test_non_image_attachments_filtered(self):
        """Test that non-image attachments are filtered out."""
        attachments = [
            {
                "type": "document",
                "media_type": "application/pdf",
                "data": base64.b64encode(b"pdf-data").decode("utf-8"),
                "filename": "document.pdf",
            },
            {
                "type": "image",
                "media_type": "image/png",
                "data": base64.b64encode(b"image-data").decode("utf-8"),
                "filename": "image.png",
            },
        ]

        # Process attachments, filtering non-images
        image_blocks = []
        for attachment in attachments:
            if attachment.get("type") == "image":
                image_block = ImageBlock(
                    image_type="base64", media_type=attachment.get("media_type", "image/png"), data=attachment["data"]
                )
                image_blocks.append(image_block)

        assert len(image_blocks) == 1
        assert image_blocks[0].media_type == "image/png"

    def test_empty_attachments_handling(self):
        """Test that empty or None attachments are handled gracefully."""
        # Test with None
        image_blocks = []
        attachments = None
        if attachments:
            for attachment in attachments:
                if attachment.get("type") == "image":
                    image_block = ImageBlock(
                        image_type="base64",
                        media_type=attachment.get("media_type", "image/png"),
                        data=attachment["data"],
                    )
                    image_blocks.append(image_block)

        assert len(image_blocks) == 0

        # Test with empty list
        image_blocks = []
        attachments = []
        for attachment in attachments:
            if attachment.get("type") == "image":
                image_block = ImageBlock(
                    image_type="base64", media_type=attachment.get("media_type", "image/png"), data=attachment["data"]
                )
                image_blocks.append(image_block)

        assert len(image_blocks) == 0

    def test_message_content_with_attachments(self):
        """Test that message content is correctly structured with text and image blocks."""
        test_message = "What is in this image?"
        test_image_data = base64.b64encode(b"fake-image-data").decode("utf-8")
        attachments = [
            {
                "type": "image",
                "media_type": "image/png",
                "data": test_image_data,
                "filename": "test-image.png",
            }
        ]

        # Build content blocks as the coder agent would
        content_blocks = [TextBlock(text=test_message)]

        if attachments:
            for attachment in attachments:
                if attachment.get("type") == "image":
                    image_block = ImageBlock(
                        image_type="base64",
                        media_type=attachment.get("media_type", "image/png"),
                        data=attachment["data"],
                    )
                    content_blocks.append(image_block)

        assert len(content_blocks) == 2
        assert isinstance(content_blocks[0], TextBlock)
        assert content_blocks[0].text == test_message
        assert isinstance(content_blocks[1], ImageBlock)
        assert content_blocks[1].data == test_image_data


@pytest.mark.asyncio
async def test_coder_agent_process_message_imports():
    """Test that the coder agent has the necessary imports for image handling."""
    # This test verifies the imports are correct
    from kolega_code.agent.coder import CoderAgent  # noqa: F401
    from kolega_code.llm.models import ImageBlock, TextBlock

    # Just verify the imports work
    assert ImageBlock is not None
    assert TextBlock is not None
    assert CoderAgent is not None


def test_attachment_blocks_mixes_images_and_files():
    agent = object.__new__(BaseAgent)
    blocks = agent._attachment_blocks(
        [
            _image_attachment(),
            {"type": "file", "path": "src/app.py", "content": "print('hi')"},
            {"type": "unknown", "data": "ignored"},
        ]
    )

    assert len(blocks) == 2
    assert isinstance(blocks[0], ImageBlock)
    assert isinstance(blocks[1], TextBlock)
    assert blocks[1].text == '<attached-file path="src/app.py">\nprint(\'hi\')\n</attached-file>'


def test_attachment_blocks_handles_none():
    agent = object.__new__(BaseAgent)
    assert agent._attachment_blocks(None) == []


@pytest.mark.asyncio
async def test_coder_agent_file_attachment_added_to_history(tmp_path):
    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()
    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=connection_manager,
        config=_deepseek_config(),
        agent_mode=AgentMode.CLI,
    )
    agent.count_current_context = AsyncMock(return_value=TokenCount(input_tokens=42))
    agent.llm = Mock()
    agent.llm.stream = AsyncMock(return_value=_EmptyStream())

    attachments = [{"type": "file", "path": "notes.md", "content": "remember the milk"}]
    chunks = [chunk async for chunk in agent.process_message_stream("see the notes", attachments)]

    assert chunks[-1]["complete"] is True
    user_message = agent.history[0]
    texts = [block.text for block in user_message.content if isinstance(block, TextBlock)]
    assert texts[0] == "see the notes"
    assert texts[1] == '<attached-file path="notes.md">\nremember the milk\n</attached-file>'


# ---------------------------------------------------------------------------
# Historical image handling: non-vision model in a thread that already has images
# ---------------------------------------------------------------------------


def _anthropic_config() -> AgentConfig:
    model_config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model="claude-haiku-4-5-20251001",
        rate_limits=RateLimitConfig(),
    )
    return AgentConfig(
        anthropic_api_key="test-key",
        long_context_config=model_config,
        fast_config=model_config,
        thinking_config=model_config,
    )


def _image_history_message() -> Message:
    """A user message carrying a text block and an image block from an earlier turn."""
    return Message(
        role="user",
        content=[
            TextBlock(text="what is in this image?"),
            ImageBlock(image_type="base64", media_type="image/png", data="ZmFrZQ=="),
        ],
    )


def _sent_messages(agent) -> list:
    """The messages passed to llm.stream on the most recent call."""
    return list(agent.llm.stream.call_args.kwargs["messages"])


@pytest.mark.asyncio
async def test_non_vision_model_strips_historical_images_before_llm_call(tmp_path):
    from kolega_code.agent.conversation import count_image_blocks

    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()
    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=connection_manager,
        config=_deepseek_config(),
        agent_mode=AgentMode.CLI,
    )
    assert agent.supports_vision is False
    # Pre-populate history with an image-bearing message from an earlier turn.
    agent.append_user_message([_image_history_message().content[0], _image_history_message().content[1]])
    assert count_image_blocks(list(agent.history)) == 1

    agent.count_current_context = AsyncMock(return_value=TokenCount(input_tokens=42))
    agent.llm = Mock()
    agent.llm.stream = AsyncMock(return_value=_EmptyStream())

    chunks = [chunk async for chunk in agent.process_message_stream("follow up question")]

    # Turn completed normally (no provider error surfaced).
    assert chunks[-1]["complete"] is True
    # The request sent to the LLM contains NO image blocks.
    sent = _sent_messages(agent)
    assert count_image_blocks(sent) == 0
    # A text placeholder describing the image is present instead.
    placeholders = [
        block.text
        for m in sent
        for block in (m.content if isinstance(m.content, list) else [])
        if isinstance(block, TextBlock) and "not visible" in block.text
    ]
    assert placeholders, "expected an image placeholder text block in the request"
    assert "image/png" in placeholders[0]
    assert "deepseek-v4-pro" in placeholders[0]
    # Stored history is NOT mutated — the original image survives for a switch back.
    assert count_image_blocks(list(agent.history)) == 1


@pytest.mark.asyncio
async def test_vision_model_passes_historical_images_through(tmp_path):
    from kolega_code.agent.conversation import count_image_blocks

    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()
    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=connection_manager,
        config=_anthropic_config(),
        agent_mode=AgentMode.CLI,
    )
    assert agent.supports_vision is True
    agent.append_user_message([_image_history_message().content[0], _image_history_message().content[1]])

    agent.count_current_context = AsyncMock(return_value=TokenCount(input_tokens=42))
    agent.llm = Mock()
    agent.llm.stream = AsyncMock(return_value=_EmptyStream())

    chunks = [chunk async for chunk in agent.process_message_stream("follow up question")]
    assert chunks[-1]["complete"] is True

    # Vision-capable model receives the image block unchanged.
    sent = _sent_messages(agent)
    assert count_image_blocks(sent) == 1


@pytest.mark.asyncio
async def test_non_vision_model_count_context_also_strips_images(tmp_path):
    """count_current_context must not send image blocks to a non-vision model's tokenizer."""
    from kolega_code.agent.conversation import count_image_blocks

    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()
    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=connection_manager,
        config=_deepseek_config(),
        agent_mode=AgentMode.CLI,
    )
    agent.append_user_message([_image_history_message().content[0], _image_history_message().content[1]])

    captured = {}

    async def _capture_count(*args, **kwargs):
        captured["messages"] = list(kwargs.get("messages", []))
        return TokenCount(input_tokens=42)

    agent.llm = Mock()
    agent.llm.count_tokens = AsyncMock(side_effect=_capture_count)
    # Tool list needed by count_current_context.
    agent.tool_collection = Mock()
    agent.tool_collection.get_tool_list = Mock(return_value=[])

    await agent.count_current_context()

    assert count_image_blocks(captured["messages"]) == 0
    placeholders = [
        block.text
        for m in captured["messages"]
        for block in (m.content if isinstance(m.content, list) else [])
        if isinstance(block, TextBlock) and "not visible" in block.text
    ]
    assert placeholders
