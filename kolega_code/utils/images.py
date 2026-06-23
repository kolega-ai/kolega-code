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
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

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

# Dark -> light ramp for grayscale ASCII rendering. Index 0 (space) is the
# darkest pixel; the last char (``@``) is the brightest.
_ASCII_RAMP = " .:-=+*#%@"


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
    ramp_last = len(_ASCII_RAMP) - 1
    lines: list[str] = []
    for y in range(height):
        row = "".join(_ASCII_RAMP[pixels[x, y] * ramp_last // 255] for x in range(width))
        lines.append(row)
    return "\n".join(lines)
