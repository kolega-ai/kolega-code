"""Shared image-encoding and MIME helpers for vision support.

Used by the CLI attachment paths (@-mention detection, ``/attach`` slash
command, clipboard paste, the ``ask --image`` flag) and by the agent
``read_image`` tool backend. Keeping the MIME map and base64 encoding in one
place avoids three copies of the logic and CLI<->agent import cycles.

``ascii_thumbnail_from_base64`` is the display-only renderer used by
``ImageBlock.to_markdown`` so that expanding an image tool call in the TUI
shows a small ASCII-art preview instead of a wall of base64.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image, ImageOps

# Supported image extensions -> MIME types. These are the common web image
# formats accepted by all four provider backends (Anthropic, OpenAI, Google,
# ChatGPT/Responses).
IMAGE_MIME_TYPES: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Skip images larger than this to avoid blowing the context budget.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Anthropic validates the encoded ``source.base64`` field itself, not the
# decoded image size. A raw image just under 10 MiB may therefore exceed this
# limit after base64's 4/3 expansion.
ANTHROPIC_MAX_IMAGE_BASE64_BYTES = 10 * 1024 * 1024

# Leave headroom below the provider's hard boundary when producing a
# derivative. This avoids repeatedly re-encoding an image for tiny accounting
# differences and gives later request serialization a conservative margin.
_IMAGE_RESIZE_SAFETY_RATIO = 0.95
_IMAGE_RESIZE_MAX_ATTEMPTS = 12
# Bound decoded working memory before Pillow copies/converts a provider
# derivative. Forty megapixels can already require 160 MiB as RGBA.
_IMAGE_RESIZE_MAX_PIXELS = 40_000_000

# Dark -> light ramp for grayscale ASCII rendering. Index 0 (space) is the
# darkest pixel; the last char (``@``) is the brightest.
_ASCII_RAMP = " .:-=+*#%@"


@dataclass(frozen=True)
class ImageResizeResult:
    """Result of fitting a base64 image to a provider payload limit."""

    data: Optional[str]
    media_type: Optional[str]
    resized: bool
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.data is not None and self.media_type is not None


def base64_encoded_size(raw_size: int) -> int:
    """Return the exact base64 length for ``raw_size`` bytes."""
    if raw_size < 0:
        raise ValueError("raw_size must be non-negative")
    return 4 * ((raw_size + 2) // 3)


def max_raw_size_for_base64_limit(encoded_limit: int) -> int:
    """Return the largest raw byte count whose base64 fits ``encoded_limit``."""
    if encoded_limit < 0:
        raise ValueError("encoded_limit must be non-negative")
    return 3 * (encoded_limit // 4)


def resize_base64_image_to_limit(
    data: str,
    media_type: str,
    *,
    max_base64_bytes: int,
) -> ImageResizeResult:
    """Return a non-mutating provider-safe derivative of a base64 image.

    Images already within the hard limit are returned byte-for-byte unchanged.
    Oversized images are decoded and re-encoded as PNG when transparency must
    be retained, or optimized JPEG otherwise. Dimensions are reduced
    iteratively until the payload has a safety margin below the provider limit.
    Animated images use their first frame for the ephemeral provider copy.
    """
    if max_base64_bytes <= 0:
        raise ValueError("max_base64_bytes must be positive")

    if not data.isascii():
        return ImageResizeResult(None, None, resized=False, error="Image payload is not ASCII base64")
    encoded_size = len(data)

    if encoded_size <= max_base64_bytes:
        return ImageResizeResult(data, media_type, resized=False)

    try:
        raw = base64.b64decode(data, validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            opened.seek(0)
            width, height = opened.size
            if width * height > _IMAGE_RESIZE_MAX_PIXELS:
                return ImageResizeResult(
                    None,
                    None,
                    resized=False,
                    error=f"Image dimensions exceed the {_IMAGE_RESIZE_MAX_PIXELS:,}-pixel resize safety limit",
                )
            image = ImageOps.exif_transpose(opened).copy()
    except Exception as exc:
        return ImageResizeResult(None, None, resized=False, error=f"Could not decode image: {exc}")

    has_transparency = "A" in image.getbands() or "transparency" in image.info
    if has_transparency:
        image = image.convert("RGBA")
        output_format = "PNG"
        output_media_type = "image/png"
    else:
        image = image.convert("RGB")
        output_format = "JPEG"
        output_media_type = "image/jpeg"

    target_size = int(max_base64_bytes * _IMAGE_RESIZE_SAFETY_RATIO)
    target_size -= target_size % 4
    if target_size < 4:
        return ImageResizeResult(
            None,
            None,
            resized=False,
            error=f"Could not fit image below the {max_base64_bytes}-byte base64 limit",
        )

    for _ in range(_IMAGE_RESIZE_MAX_ATTEMPTS):
        output = io.BytesIO()
        try:
            if output_format == "PNG":
                image.save(output, format=output_format, optimize=True, compress_level=9)
            else:
                image.save(output, format=output_format, quality=90, optimize=True, progressive=True)
        except Exception as exc:
            return ImageResizeResult(None, None, resized=False, error=f"Could not encode resized image: {exc}")

        derivative = base64.b64encode(output.getvalue()).decode("ascii")
        if len(derivative) <= target_size:
            return ImageResizeResult(derivative, output_media_type, resized=True)

        width, height = image.size
        if width <= 1 and height <= 1:
            break

        # Encoded size trends with pixel area, so the square root estimates a
        # useful one-step dimension ratio. The extra margin guarantees progress
        # when compression ratios are nonlinear.
        ratio = math.sqrt(target_size / len(derivative)) * 0.95
        ratio = min(0.9, max(0.1, ratio))
        next_width = max(1, int(width * ratio))
        next_height = max(1, int(height * ratio))
        if next_width == width and width > 1:
            next_width -= 1
        if next_height == height and height > 1:
            next_height -= 1
        image = image.resize((next_width, next_height), Image.Resampling.LANCZOS)

    return ImageResizeResult(
        None,
        None,
        resized=False,
        error=f"Could not fit image below the {max_base64_bytes}-byte base64 limit",
    )


def image_media_type(path_or_suffix: Union[str, Path]) -> Optional[str]:
    """Return the MIME type for an image path/suffix, or ``None`` if unsupported.

    Args:
        path_or_suffix: A file path (e.g. ``"assets/logo.png"``) or a bare
            extension (e.g. ``".png"`` or ``"png"``). Case-insensitive.
    """
    text = str(path_or_suffix).lower()
    if text.startswith("."):
        suffix = text
    else:
        suffix = Path(text).suffix.lower()
        if not suffix:  # bare extension like "png" with no dot or path sep
            suffix = f".{text}" if "/" not in text and "\\" not in text else ""
    return IMAGE_MIME_TYPES.get(suffix)


def is_supported_image(path: Union[str, Path]) -> bool:
    """True if ``path`` has a supported image extension."""
    return image_media_type(path) is not None


def encode_image_attachment(data: bytes, media_type: str, *, path: Optional[str] = None) -> Dict[str, Any]:
    """Build an image attachment dict from raw bytes.

    Args:
        data: Raw image bytes.
        media_type: MIME type (e.g. ``"image/png"``).
        path: Optional source path included for display/debugging.

    Returns:
        ``{"type": "image", "media_type": ..., "data": "<base64>", ...}`` —
        the shape consumed by ``BaseAgent._attachment_blocks``.
    """
    attachment: Dict[str, Any] = {
        "type": "image",
        "media_type": media_type,
        "data": base64.b64encode(data).decode("ascii"),
    }
    if path:
        attachment["path"] = path
    return attachment


def encode_image_file(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read, validate, and base64-encode an image file.

    Returns ``None`` if the path is not a supported image format, the file is
    missing/unreadable, or it exceeds ``MAX_IMAGE_BYTES`` (so callers can fall
    back to a friendly message instead of crashing).
    """
    path = Path(path)
    media_type = image_media_type(path)
    if media_type is None:
        return None
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
    except OSError:
        return None
    if size > MAX_IMAGE_BYTES:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return encode_image_attachment(data, media_type, path=str(path))


def ascii_thumbnail_from_base64(data: str, media_type: str, *, width: int = 40, height: int = 20) -> str:
    """Render a grayscale ASCII-art thumbnail from a base64-encoded image.

    Decodes the image with Pillow, converts to grayscale, downsamples to
    ``width`` x ``height``, and maps each pixel's brightness to a character
    in ``_ASCII_RAMP`` (dark = space, light = ``@``).

    Returns the multi-line ASCII string. If the image cannot be decoded
    (corrupt data, unsupported format, etc.), returns a short placeholder
    string instead of raising — a malformed image must never crash the TUI.
    """
    try:
        raw = base64.b64decode(data)
        img = Image.open(io.BytesIO(raw))
        img = img.convert("L").resize((width, height), Image.Resampling.LANCZOS)
    except Exception:
        return f"[Image — {media_type} — could not render preview]"

    pixels = img.load()
    if pixels is None:
        return f"[Image — {media_type} — could not render preview]"
    ramp_last = len(_ASCII_RAMP) - 1
    lines: list[str] = []
    for y in range(height):
        chars: list[str] = []
        for x in range(width):
            v = pixels[x, y]
            if isinstance(v, tuple):
                v = v[0]
            chars.append(_ASCII_RAMP[int(v) * ramp_last // 255])
        lines.append("".join(chars))
    return "\n".join(lines)
