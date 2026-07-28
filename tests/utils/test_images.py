"""Tests for the shared image-encoding helpers."""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from kolega_code.utils.images import (
    ANTHROPIC_MAX_IMAGE_BASE64_BYTES,
    MAX_IMAGE_BYTES,
    base64_encoded_size,
    encode_image_attachment,
    encode_image_file,
    image_media_type,
    is_supported_image,
    max_raw_size_for_base64_limit,
    resize_base64_image_to_limit,
)


def test_image_media_type_maps_known_extensions() -> None:
    assert image_media_type(".png") == "image/png"
    assert image_media_type(".PNG") == "image/png"
    assert image_media_type(".jpg") == "image/jpeg"
    assert image_media_type(".jpeg") == "image/jpeg"
    assert image_media_type(".gif") == "image/gif"
    assert image_media_type(".webp") == "image/webp"
    assert image_media_type(".bmp") == "image/bmp"


def test_image_media_type_from_path() -> None:
    assert image_media_type("assets/logo.png") == "image/png"
    assert image_media_type(Path("/tmp/shot.JPEG")) == "image/jpeg"
    assert image_media_type("png") == "image/png"


def test_image_media_type_unsupported_returns_none() -> None:
    assert image_media_type(".txt") is None
    assert image_media_type("readme.md") is None
    assert image_media_type("") is None


def test_is_supported_image() -> None:
    assert is_supported_image("logo.png") is True
    assert is_supported_image("notes.txt") is False


def test_encode_image_attachment_builds_dict() -> None:
    data = b"\x89PNG\r\n\x1a\nfake"
    att = encode_image_attachment(data, "image/png", path="x.png")
    assert att["type"] == "image"
    assert att["media_type"] == "image/png"
    assert att["path"] == "x.png"
    assert base64.b64decode(att["data"]) == data


def test_encode_image_attachment_omits_path_when_none() -> None:
    att = encode_image_attachment(b"x", "image/png")
    assert "path" not in att
    assert att["type"] == "image"


def test_encode_image_file_reads_and_encodes(tmp_path: Path) -> None:
    png = tmp_path / "shot.png"
    payload = b"\x89PNG\r\n\x1a\nfake-image-bytes"
    png.write_bytes(payload)

    att = encode_image_file(png)
    assert att is not None
    assert att["type"] == "image"
    assert att["media_type"] == "image/png"
    assert base64.b64decode(att["data"]) == payload
    assert att["path"].endswith("shot.png")


def test_encode_image_file_non_image_returns_none(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    assert encode_image_file(f) is None


def test_encode_image_file_missing_returns_none(tmp_path: Path) -> None:
    assert encode_image_file(tmp_path / "nope.png") is None


def test_encode_image_file_oversized_returns_none(tmp_path: Path, monkeypatch) -> None:
    import kolega_code.utils.images as images_mod

    monkeypatch.setattr(images_mod, "MAX_IMAGE_BYTES", 4)
    big = tmp_path / "big.png"
    big.write_bytes(b"\x89PNG" + b"\x00" * 100)
    assert encode_image_file(big) is None


def test_encode_image_file_respects_real_max(tmp_path: Path) -> None:
    assert MAX_IMAGE_BYTES == 20 * 1024 * 1024


def _encoded_image(
    image: Image.Image,
    image_format: str,
    *,
    exif: Image.Exif | None = None,
) -> str:
    output = io.BytesIO()
    save_kwargs = {"exif": exif} if exif is not None else {}
    image.save(output, format=image_format, **save_kwargs)
    return base64.b64encode(output.getvalue()).decode("ascii")


def test_base64_size_math_matches_anthropic_boundary_and_reported_session_image() -> None:
    assert max_raw_size_for_base64_limit(ANTHROPIC_MAX_IMAGE_BASE64_BYTES) == 7_864_320
    assert base64_encoded_size(7_864_320) == ANTHROPIC_MAX_IMAGE_BASE64_BYTES
    assert base64_encoded_size(7_864_321) == ANTHROPIC_MAX_IMAGE_BASE64_BYTES + 4

    # Session ef87fee1e3884ce3907157a352278a85: a 10,106,238-byte
    # JPEG became the exact 13,474,984-byte field rejected by Anthropic.
    assert base64_encoded_size(10_106_238) == 13_474_984


def test_resize_base64_image_returns_fitting_image_unchanged() -> None:
    data = _encoded_image(Image.new("RGB", (4, 4), "navy"), "PNG")

    result = resize_base64_image_to_limit(data, "image/png", max_base64_bytes=len(data))

    assert result.succeeded is True
    assert result.resized is False
    assert result.data == data
    assert result.media_type == "image/png"


def test_resize_base64_image_constrains_byte_small_image_dimensions() -> None:
    data = _encoded_image(Image.new("RGB", (2_001, 100), "navy"), "PNG")
    assert len(data) < ANTHROPIC_MAX_IMAGE_BASE64_BYTES

    result = resize_base64_image_to_limit(
        data,
        "image/png",
        max_base64_bytes=ANTHROPIC_MAX_IMAGE_BASE64_BYTES,
        max_dimension=2_000,
    )

    assert result.succeeded is True
    assert result.resized is True
    assert result.data is not None
    with Image.open(io.BytesIO(base64.b64decode(result.data))) as derivative:
        assert derivative.width <= 2_000
        assert derivative.height <= 2_000
        assert derivative.width / derivative.height == pytest.approx(2_001 / 100, rel=0.02)


def test_resize_base64_image_returns_dimension_boundary_unchanged() -> None:
    data = _encoded_image(Image.new("RGB", (2_000, 100), "navy"), "PNG")

    result = resize_base64_image_to_limit(
        data,
        "image/png",
        max_base64_bytes=ANTHROPIC_MAX_IMAGE_BASE64_BYTES,
        max_dimension=2_000,
    )

    assert result.succeeded is True
    assert result.resized is False
    assert result.data == data
    assert result.media_type == "image/png"


def test_resize_base64_image_satisfies_dimension_and_payload_limits() -> None:
    image = Image.effect_noise((400, 300), 80).convert("RGB")
    data = _encoded_image(image, "JPEG")
    limit = 4_000
    assert len(data) > limit

    result = resize_base64_image_to_limit(
        data,
        "image/jpeg",
        max_base64_bytes=limit,
        max_dimension=200,
    )

    assert result.succeeded is True
    assert result.resized is True
    assert result.data is not None
    assert len(result.data) < limit
    with Image.open(io.BytesIO(base64.b64decode(result.data))) as derivative:
        assert derivative.width <= 200
        assert derivative.height <= 200


def test_resize_base64_jpeg_produces_decodable_derivative_below_limit() -> None:
    image = Image.effect_noise((160, 120), 80).convert("RGB")
    data = _encoded_image(image, "JPEG")
    limit = 1_600
    assert len(data) > limit

    result = resize_base64_image_to_limit(data, "image/jpeg", max_base64_bytes=limit)

    assert result.succeeded is True
    assert result.resized is True
    assert result.data is not None
    assert result.media_type == "image/jpeg"
    assert len(result.data) < limit
    with Image.open(io.BytesIO(base64.b64decode(result.data))) as derivative:
        derivative.verify()


def test_resize_base64_transparent_png_preserves_alpha() -> None:
    image = Image.new("RGBA", (96, 96))
    image.putdata(
        [
            ((x * 17) % 256, (y * 29) % 256, ((x + y) * 11) % 256, (x * y) % 256)
            for y in range(image.height)
            for x in range(image.width)
        ]
    )
    data = _encoded_image(image, "PNG")
    limit = 1_500
    assert len(data) > limit

    result = resize_base64_image_to_limit(data, "image/png", max_base64_bytes=limit)

    assert result.succeeded is True
    assert result.resized is True
    assert result.data is not None
    assert result.media_type == "image/png"
    assert len(result.data) < limit
    with Image.open(io.BytesIO(base64.b64decode(result.data))) as derivative:
        assert "A" in derivative.getbands()


def test_resize_base64_image_applies_exif_orientation() -> None:
    image = Image.effect_noise((80, 40), 80).convert("RGB")
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    data = _encoded_image(image, "JPEG", exif=exif)
    limit = 1_200
    assert len(data) > limit

    result = resize_base64_image_to_limit(data, "image/jpeg", max_base64_bytes=limit)

    assert result.succeeded is True
    assert result.data is not None
    with Image.open(io.BytesIO(base64.b64decode(result.data))) as derivative:
        assert derivative.height > derivative.width


def test_resize_base64_animated_image_uses_static_supported_frame() -> None:
    first = Image.effect_noise((96, 96), 80).convert("RGB")
    second = Image.new("RGB", (96, 96), "red")
    output = io.BytesIO()
    first.save(output, format="GIF", save_all=True, append_images=[second])
    data = base64.b64encode(output.getvalue()).decode("ascii")
    limit = 1_500
    assert len(data) > limit

    result = resize_base64_image_to_limit(data, "image/gif", max_base64_bytes=limit)

    assert result.succeeded is True
    assert result.resized is True
    assert result.data is not None
    assert result.media_type == "image/jpeg"
    assert len(result.data) < limit
    with Image.open(io.BytesIO(base64.b64decode(result.data))) as derivative:
        assert getattr(derivative, "n_frames", 1) == 1


def test_resize_base64_image_returns_safe_failure_for_invalid_image() -> None:
    data = base64.b64encode(b"not-an-image" * 100).decode("ascii")

    result = resize_base64_image_to_limit(data, "image/png", max_base64_bytes=64)

    assert result.succeeded is False
    assert result.data is None
    assert result.media_type is None
    assert result.error is not None
    assert "decode image" in result.error


def test_resize_base64_image_rejects_unsafe_decoded_pixel_count(monkeypatch) -> None:
    import kolega_code.utils.images as images_mod

    monkeypatch.setattr(images_mod, "_IMAGE_RESIZE_MAX_PIXELS", 32)
    data = _encoded_image(Image.new("RGB", (8, 8), "red"), "PNG")

    result = resize_base64_image_to_limit(data, "image/png", max_base64_bytes=64)

    assert result.succeeded is False
    assert result.data is None
    assert result.error is not None
    assert "pixel resize safety limit" in result.error


def test_resize_base64_image_returns_safe_failure_when_limit_cannot_hold_base64() -> None:
    data = _encoded_image(Image.new("RGB", (1, 1), "red"), "PNG")

    result = resize_base64_image_to_limit(data, "image/png", max_base64_bytes=3)

    assert result.succeeded is False
    assert result.data is None
    assert result.error is not None
