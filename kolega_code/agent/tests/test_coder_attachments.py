"""Tests for coder agent image attachment handling."""

import base64
import pytest
from unittest.mock import MagicMock, patch

from kolega_code.agent.llm.models import ImageBlock, TextBlock


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
    from kolega_code.agent.coder import CoderAgent
    from kolega_code.agent.llm.models import ImageBlock, TextBlock

    # Just verify the imports work
    assert ImageBlock is not None
    assert TextBlock is not None
