"""Chrome Native Messaging framing for bounded JSON objects."""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, BinaryIO, Mapping, cast

MAX_NATIVE_MESSAGE_BYTES = 1_048_576
_HEADER = struct.Struct("<I")


class FramingError(ValueError):
    """A safe-to-report native-message framing failure."""


class MessageTooLargeError(FramingError):
    """A frame exceeded the configured size bound."""


class TruncatedMessageError(FramingError):
    """EOF occurred in the middle of a frame."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FramingError("Native message contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise FramingError("Native message contains a non-finite number")


def _decode_object(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise FramingError("Native message is not valid UTF-8") from None
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except FramingError:
        raise
    except (json.JSONDecodeError, RecursionError):
        raise FramingError("Native message is not valid JSON") from None
    if not isinstance(value, dict):
        raise FramingError("Native message must be a JSON object")
    return cast(dict[str, Any], value)


def encode_message(message: Mapping[str, object], *, max_bytes: int = MAX_NATIVE_MESSAGE_BYTES) -> bytes:
    """Encode one mapping using Chrome's little-endian length framing."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    try:
        payload = json.dumps(
            dict(message),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise FramingError("Native message is not JSON serializable") from None
    if len(payload) > max_bytes:
        raise MessageTooLargeError("Native message exceeds the size limit")
    return _HEADER.pack(len(payload)) + payload


def _read_exact(stream: BinaryIO, size: int, *, allow_clean_eof: bool = False) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            if not chunks and allow_clean_eof:
                return None
            raise TruncatedMessageError("Native message ended before the frame was complete")
        chunks.extend(chunk)
    return bytes(chunks)


def read_message(stream: BinaryIO, *, max_bytes: int = MAX_NATIVE_MESSAGE_BYTES) -> dict[str, Any] | None:
    """Read one message, returning ``None`` only for EOF between frames."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    header = _read_exact(stream, _HEADER.size, allow_clean_eof=True)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > max_bytes:
        raise MessageTooLargeError("Native message exceeds the size limit")
    payload = _read_exact(stream, length)
    assert payload is not None
    return _decode_object(payload)


def _write_all(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = stream.write(view[written:])
        if count is None:
            written = len(view)
        elif count <= 0:
            raise FramingError("Native message could not be written")
        else:
            written += count
    flush = getattr(stream, "flush", None)
    if flush is not None:
        flush()


def write_message(
    stream: BinaryIO,
    message: Mapping[str, object],
    *,
    max_bytes: int = MAX_NATIVE_MESSAGE_BYTES,
) -> None:
    """Write one complete native-message frame."""
    _write_all(stream, encode_message(message, max_bytes=max_bytes))


async def read_message_async(
    reader: asyncio.StreamReader,
    *,
    max_bytes: int = MAX_NATIVE_MESSAGE_BYTES,
) -> dict[str, Any] | None:
    """Read one bounded native message from an asyncio byte stream."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    try:
        header = await reader.readexactly(_HEADER.size)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise TruncatedMessageError("Native message ended before the frame was complete") from None
    (length,) = _HEADER.unpack(header)
    if length > max_bytes:
        raise MessageTooLargeError("Native message exceeds the size limit")
    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        raise TruncatedMessageError("Native message ended before the frame was complete") from None
    return _decode_object(payload)


async def write_message_async(
    writer: asyncio.StreamWriter,
    message: Mapping[str, object],
    *,
    max_bytes: int = MAX_NATIVE_MESSAGE_BYTES,
) -> None:
    """Write and drain one complete native-message frame."""
    writer.write(encode_message(message, max_bytes=max_bytes))
    await writer.drain()
