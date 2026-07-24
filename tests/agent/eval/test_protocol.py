"""Frame codec tests for the eval kernel NDJSON protocol."""

import pytest

from kolega_code.agent.eval.protocol import MAX_FRAME_BYTES, ProtocolError, encode_frame, parse_frame


def test_round_trip_preserves_frame():
    frame = {"type": "exec", "id": 7, "run": "abc", "code": "print('héllo')\n", "silent": False}
    line = encode_frame(frame)
    assert line.endswith(b"\n")
    assert parse_frame(line) == frame


def test_round_trip_large_image_bundle():
    bundle = {"image/png": "a" * 200_000, "text/plain": "fig"}
    frame = {"type": "display", "id": 3, "bundle": bundle}
    assert parse_frame(encode_frame(frame)) == frame


def test_parse_rejects_non_json():
    with pytest.raises(ProtocolError, match="malformed"):
        parse_frame(b"this is not json\n")


def test_parse_rejects_non_object():
    with pytest.raises(ProtocolError, match="not an object"):
        parse_frame(b"[1, 2, 3]\n")


def test_parse_rejects_missing_type():
    with pytest.raises(ProtocolError, match="not an object with a type"):
        parse_frame(b'{"id": 1}\n')


def test_parse_rejects_oversized_frame():
    with pytest.raises(ProtocolError, match="exceeds"):
        parse_frame(b"x" * (MAX_FRAME_BYTES + 1))
