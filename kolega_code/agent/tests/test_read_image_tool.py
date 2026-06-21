import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.tool_backend.read_image_tool import ReadImageTool
from kolega_code.agent.tools import ToolCollection
from kolega_code.llm.models import ImageBlock
from kolega_code.services.file_system import LocalFileSystem


def _make_tool(tmp_path, caller=None, config=None):
    return ReadImageTool(
        tmp_path,
        "ws",
        "thread",
        Mock(),  # connection_manager
        config or SimpleNamespace(),
        caller or object(),
        LocalFileSystem(root_path=tmp_path),
    )


class TestReadImageToolBackend:
    @pytest.mark.asyncio
    async def test_read_png_returns_image_block(self, tmp_path):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        path = tmp_path / "shot.png"
        path.write_bytes(png_bytes)

        tool = _make_tool(tmp_path)
        result = await tool.read_image(str(path))

        assert isinstance(result, list)
        assert len(result) == 1
        block = result[0]
        assert isinstance(block, ImageBlock)
        assert block.image_type == "base64"
        assert block.media_type == "image/png"
        assert base64.b64decode(block.data) == png_bytes

    @pytest.mark.asyncio
    async def test_read_jpeg_returns_jpeg_media_type(self, tmp_path):
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16
        path = tmp_path / "photo.jpg"
        path.write_bytes(jpeg_bytes)

        tool = _make_tool(tmp_path)
        result = await tool.read_image(str(path))

        assert result[0].media_type == "image/jpeg"
        assert base64.b64decode(result[0].data) == jpeg_bytes

    @pytest.mark.asyncio
    async def test_missing_path_raises_file_not_found(self, tmp_path):
        tool = _make_tool(tmp_path)
        with pytest.raises(FileNotFoundError):
            await tool.read_image(str(tmp_path / "nope.png"))

    @pytest.mark.asyncio
    async def test_directory_path_raises_value_error(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        tool = _make_tool(tmp_path)
        with pytest.raises(ValueError):
            await tool.read_image(str(tmp_path / "subdir"))

    @pytest.mark.asyncio
    async def test_unsupported_extension_raises_value_error(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello")

        tool = _make_tool(tmp_path)
        with pytest.raises(ValueError):
            await tool.read_image(str(path))


class TestReadImageWrapper:
    @pytest.mark.asyncio
    async def test_wrapper_delegates_to_backend(self, tmp_path):
        collection = ToolCollection.__new__(ToolCollection)
        expected = [ImageBlock(image_type="base64", media_type="image/png", data="")]
        collection.read_image_tool = Mock()
        collection.read_image_tool.read_image = AsyncMock(return_value=expected)
        collection.caller = object()

        result = await collection.read_image("x.png")
        assert result is expected
        collection.read_image_tool.read_image.assert_awaited_once_with("x.png")

    @pytest.mark.asyncio
    async def test_wrapper_chatgpt_provider_returns_string(self, tmp_path):
        collection = ToolCollection.__new__(ToolCollection)
        collection.read_image_tool = Mock()
        collection.read_image_tool.read_image = AsyncMock()
        collection.caller = SimpleNamespace(
            primary_model_config=SimpleNamespace(provider="openai_chatgpt")
        )

        result = await collection.read_image("img.png")
        assert isinstance(result, str)
        assert "ChatGPT Responses backend" in result
        collection.read_image_tool.read_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wrapper_chatgpt_provider_enum_value(self, tmp_path):
        from kolega_code.config import ModelProvider

        collection = ToolCollection.__new__(ToolCollection)
        collection.read_image_tool = Mock()
        collection.read_image_tool.read_image = AsyncMock()
        collection.caller = SimpleNamespace(
            primary_model_config=SimpleNamespace(provider=ModelProvider.OPENAI_CHATGPT)
        )

        result = await collection.read_image("img.png")
        assert isinstance(result, str)
        collection.read_image_tool.read_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wrapper_non_chatgpt_provider_delegates(self, tmp_path):
        collection = ToolCollection.__new__(ToolCollection)
        expected = [ImageBlock(image_type="base64", media_type="image/png", data="")]
        collection.read_image_tool = Mock()
        collection.read_image_tool.read_image = AsyncMock(return_value=expected)
        collection.caller = SimpleNamespace(
            primary_model_config=SimpleNamespace(provider="anthropic")
        )

        result = await collection.read_image("img.png")
        assert result is expected


class TestShouldIncludeReadImage:
    def _make_collection(self):
        collection = ToolCollection.__new__(ToolCollection)
        collection.caller = object()
        collection.orchestration_tools = []
        collection.tool_config = SimpleNamespace(custom_tool_groups=None)
        return collection

    def test_vision_enabled_includes_tool(self):
        collection = self._make_collection()
        collection.caller = SimpleNamespace(supports_vision=True)
        assert collection._should_include_tool("read_image") is True

    def test_vision_disabled_excludes_tool(self):
        collection = self._make_collection()
        collection.caller = SimpleNamespace(supports_vision=False)
        assert collection._should_include_tool("read_image") is False

    def test_no_supports_vision_attr_excludes_tool(self):
        collection = self._make_collection()
        collection.caller = object()
        assert collection._should_include_tool("read_image") is False


def test_read_image_in_read_only_tools():
    assert "read_image" in ToolCollection.read_only_tools
    assert hasattr(ToolCollection, "read_image")
