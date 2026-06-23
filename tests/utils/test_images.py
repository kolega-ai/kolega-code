"""Tests for the shared image-encoding helpers."""

import base64
from pathlib import Path

from kolega_code.utils.images import (
    MAX_IMAGE_BYTES,
    encode_image_attachment,
    encode_image_file,
    image_media_type,
    is_supported_image,
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
