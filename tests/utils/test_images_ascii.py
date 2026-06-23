"""Tests for the ASCII-art thumbnail renderer and ``ImageBlock.to_markdown``."""

import base64
import io

from PIL import Image

from kolega_code.llm.models import ImageBlock
from kolega_code.utils.images import _ASCII_RAMP, ascii_thumbnail_from_base64


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_ascii_thumbnail_dimensions() -> None:
    img = Image.new("RGB", (200, 100), (123, 45, 6))
    md = ascii_thumbnail_from_base64(_png_b64(img), "image/png", width=10, height=4)

    lines = md.split("\n")
    assert len(lines) == 4
    for line in lines:
        assert len(line) == 10


def test_ascii_thumbnail_brightness_mapping() -> None:
    # 2x2 image: black pixel (0,0), white pixel (1,0), and two mid-gray pixels.
    img = Image.new("L", (2, 2))
    img.putpixel((0, 0), 0)  # black  -> darkest char (space)
    img.putpixel((1, 0), 255)  # white  -> brightest char (@)
    img.putpixel((0, 1), 128)
    img.putpixel((1, 1), 64)

    md = ascii_thumbnail_from_base64(_png_b64(img), "image/png", width=2, height=2)
    lines = md.split("\n")

    assert lines[0][0] == _ASCII_RAMP[0]  # black -> space
    assert lines[0][1] == _ASCII_RAMP[-1]  # white -> @
    # Brightness monotonically increases along the ramp.
    assert _ASCII_RAMP.index(lines[1][0]) > _ASCII_RAMP.index(lines[1][1])


def test_ascii_thumbnail_invalid_data_returns_placeholder() -> None:
    result = ascii_thumbnail_from_base64("!!!not-base64!!!", "image/png")
    assert "image/png" in result
    assert "could not render" in result


def test_ascii_thumbnail_corrupt_png_returns_placeholder() -> None:
    # Valid base64 but not a real image — Pillow should fail to identify it.
    bad = base64.b64encode(b"\x00\x01\x02not-an-image").decode("ascii")
    result = ascii_thumbnail_from_base64(bad, "image/png")
    assert "image/png" in result
    assert "could not render" in result


def test_ascii_thumbnail_default_size() -> None:
    img = Image.new("RGB", (800, 600), (200, 200, 200))
    md = ascii_thumbnail_from_base64(_png_b64(img), "image/png")

    lines = md.split("\n")
    assert len(lines) == 20
    for line in lines:
        assert len(line) == 40


def test_imageblock_to_markdown_returns_ascii_for_base64() -> None:
    img = Image.new("L", (40, 20), 255)  # pure white
    block = ImageBlock(image_type="base64", media_type="image/png", data=_png_b64(img))

    md = block.to_markdown()

    assert "data:" not in md
    assert ";base64," not in md
    assert md.count("\n") == 19  # 20 rows
    # A pure-white solid image should render as the brightest ramp char.
    assert set(md.replace("\n", "")) == {_ASCII_RAMP[-1]}


def test_imageblock_to_markdown_url_image() -> None:
    block = ImageBlock(
        image_type="url", media_type="image/jpeg", data="https://example.com/shot.jpg"
    )

    assert block.to_markdown() == "[Image URL — image/jpeg]"


def test_imageblock_to_markdown_preserves_provider_data() -> None:
    """``to_markdown`` is display-only; the raw base64 is still on ``.data``."""
    img = Image.new("L", (4, 4), 50)
    data = _png_b64(img)
    block = ImageBlock(image_type="base64", media_type="image/png", data=data)

    assert block.data == data  # untouched
    assert base64.b64decode(block.data) == base64.b64decode(data)
