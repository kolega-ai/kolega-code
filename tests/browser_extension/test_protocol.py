from __future__ import annotations

import io
import json
import re
import struct
import time
from typing import Any, cast

import pytest

from kolega_code.browser_extension.framing import (
    FramingError,
    MessageTooLargeError,
    TruncatedMessageError,
    encode_message,
    read_message,
    write_message,
)
from kolega_code.browser_extension.protocol import (
    ALLOWED_OPERATIONS,
    MAX_ERROR_MESSAGE_LENGTH,
    PROTOCOL_VERSION,
    Envelope,
    MessageDirection,
    MessageType,
    ProtocolValidationError,
    validate_discovery_request,
    validate_discovery_response,
    validate_operation_request,
)

NOW = int(time.time() * 1000)


def request(operation: str = "browser.navigate", params: dict[str, Any] | None = None) -> Envelope:
    return Envelope.request(
        direction=MessageDirection.RUNTIME_TO_EXTENSION,
        request_id="request_1",
        runtime_id="runtime_1",
        session_id="session_1",
        deadline_ms=NOW + 10_000,
        operation=operation,
        params=params if params is not None else {"url": "https://example.com"},
    )


def test_v1_envelope_is_strict_and_round_trips() -> None:
    envelope = request()
    assert PROTOCOL_VERSION == 1
    assert set(envelope.to_mapping()) == {
        "version",
        "direction",
        "type",
        "request_id",
        "runtime_id",
        "session_id",
        "deadline_ms",
        "payload",
    }
    assert Envelope.from_json(envelope.to_json()) == envelope
    mapping = envelope.to_mapping()
    mapping["extra"] = None
    with pytest.raises(ProtocolValidationError, match="invalid schema"):
        Envelope.from_mapping(mapping)
    raw = envelope.to_json().replace('"version":1', '"version":1,"version":1')
    with pytest.raises(ProtocolValidationError) as duplicate:
        Envelope.from_json(raw)
    assert duplicate.value.code == "duplicate_key"


def test_payload_types_and_bounded_errors() -> None:
    original = request()
    result = {"page": {"html": "<main>normal browser data</main>"}, "items": [1, True, None]}
    assert Envelope.response_for(original, result).payload["result"] == result
    bounded = Envelope.error_for(original, code="failed", message="x" * 500)
    assert len(cast(str, bounded.payload["message"])) == MAX_ERROR_MESSAGE_LENGTH
    invalid = bounded.to_mapping()
    invalid["payload"] = {"code": "failed", "message": "x" * 241, "retryable": False}
    with pytest.raises(ProtocolValidationError, match="error message"):
        Envelope.from_mapping(invalid)
    assert Envelope.cancel_for(original, reason="done").direction is original.direction
    event = Envelope.event(
        direction=MessageDirection.EXTENSION_TO_RUNTIME,
        request_id="event_1",
        runtime_id="runtime_1",
        session_id="session_1",
        deadline_ms=NOW + 10_000,
        event="browser.session_ready",
        data={},
    )
    assert event.type is MessageType.EVENT


def test_protocol_rejects_lone_surrogates_before_utf8_serialization() -> None:
    original = request()
    with pytest.raises(ProtocolValidationError, match="Unicode scalar"):
        Envelope.response_for(original, {"text": "\ud800"})

    escaped = json.dumps(
        {
            **Envelope.response_for(original, {"text": "safe"}).to_mapping(),
            "payload": {"result": {"text": "\udfff"}},
        }
    ).encode()
    with pytest.raises(ProtocolValidationError, match="Unicode scalar"):
        Envelope.from_json(escaped)
    with pytest.raises(ProtocolValidationError, match="valid Unicode"):
        Envelope.from_json("\ud800")


VALID_PARAMS: dict[str, dict[str, object]] = {
    "browser.navigate": {"url": "https://example.com/"},
    "browser.navigate_back": {},
    "browser.snapshot": {"target": None, "depth": 3},
    "browser.find": {"text": "Save", "regex": None},
    "browser.wait_for": {"time": 0.1, "text": None, "text_gone": None},
    "browser.click": {"target": "e1", "double_click": False, "button": "left", "modifiers": ["Shift"]},
    "browser.type": {"target": "e1", "text": "hello", "submit": True, "slowly": False},
    "browser.fill_form": {"fields": [{"name": "Email", "target": "e1", "type": "textbox", "value": "a@example.com"}]},
    "browser.select_option": {"target": "e1", "values": ["blue"]},
    "browser.hover": {"target": "e1"},
    "browser.drag": {"start_target": "e1", "end_target": "e2"},
    "browser.press_key": {"key": "ControlOrMeta+L"},
    "browser.scroll": {"by_pages": 2, "target": None, "x": None, "y": None},
    "browser.tabs": {"action": "list", "index": None, "url": None},
    "browser.network_requests": {"include_static": False, "filter_pattern": None},
    "browser.screenshot": {"target": None, "image_type": "png", "full_page": True, "scale": "css"},
    "browser.detach": {},
}


def test_fixed_operation_surface_has_exactly_seventeen_schemas() -> None:
    assert ALLOWED_OPERATIONS == frozenset(VALID_PARAMS)
    assert len(ALLOWED_OPERATIONS) == 17


@pytest.mark.parametrize(("operation", "params"), VALID_PARAMS.items())
def test_fixed_operation_schemas(operation: str, params: dict[str, object]) -> None:
    assert operation in ALLOWED_OPERATIONS
    assert validate_operation_request(operation, params) == params
    assert request(operation, params).payload["operation"] == operation


@pytest.mark.parametrize(
    "operation",
    [
        "browser.evaluate",
        "browser.cdp",
        "browser.cookies",
        "browser.storage",
        "browser.network_request",
        "browser.file_upload",
        "browser.drop",
        "browser.console_messages",
    ],
)
def test_dangerous_or_unknown_operations_are_rejected(operation: str) -> None:
    with pytest.raises(ProtocolValidationError) as error:
        validate_operation_request(operation, {})
    assert error.value.code == "unsupported_operation"


@pytest.mark.parametrize(
    ("operation", "params"),
    [
        ("browser.navigate", {"url": "file:///tmp/a"}),
        ("browser.find", {"text": "x", "regex": "x"}),
        ("browser.wait_for", {"time": None, "text": None, "text_gone": None}),
        ("browser.click", {"target": "e1", "double_click": False, "button": "left", "modifiers": ["Shift", "Shift"]}),
        ("browser.fill_form", {"fields": []}),
        ("browser.tabs", {"action": "select", "index": None, "url": None}),
        ("browser.screenshot", {"target": "e1", "image_type": "png", "full_page": True, "scale": "css"}),
    ],
)
def test_invalid_operation_parameters_are_rejected(operation: str, params: dict[str, object]) -> None:
    with pytest.raises(ProtocolValidationError) as error:
        validate_operation_request(operation, params)
    assert error.value.code == "invalid_params"


def test_scroll_accepts_exactly_one_movement() -> None:
    base: dict[str, object] = {"by_pages": None, "target": None, "x": None, "y": None}
    for movement in ({"by_pages": 3}, {"by_pages": -1.5}, {"y": 15_000}, {"x": 0, "y": 0}, {"target": "#main"}):
        params = {**base, **movement}
        assert validate_operation_request("browser.scroll", params) == params

    ambiguous = "Provide exactly one of target, by_pages, or x/y"
    for params, expected in (
        (base, ambiguous),
        ({**base, "by_pages": 1, "y": 10}, ambiguous),
        ({**base, "target": "#main", "by_pages": 1}, ambiguous),
        ({**base, "by_pages": 11}, "by_pages is invalid"),
        # Message text is mirrored character for character by nullableInteger in
        # the extension's src/operations.js.
        ({**base, "y": -1}, "y must be an integer between 0 and 10000000, or JSON null"),
        ({**base, "y": 1.5}, "y must be an integer between 0 and 10000000, or JSON null"),
        ({**base, "y": "null"}, "y must be an integer or JSON null, not the string 'null'"),
        # Inapplicable fields must be null, never omitted.
        ({"by_pages": 1}, "invalid schema"),
        ({**base, "by_pages": 1, "extra": True}, "invalid schema"),
    ):
        with pytest.raises(ProtocolValidationError, match=re.escape(expected)) as error:
            validate_operation_request("browser.scroll", params)
        assert error.value.code == "invalid_params"


def test_url_normalization_and_regex_escapes_match_the_extension_contract() -> None:
    assert validate_operation_request("browser.navigate", {"url": "localhost:3000/path"}) == {
        "url": "http://localhost:3000/path"
    }
    assert validate_operation_request("browser.navigate", {"url": "example.com"}) == {"url": "https://example.com/"}
    assert validate_operation_request("browser.navigate", {"url": "https://faß.de/"}) == {
        "url": "https://xn--fa-hia.de/"
    }
    assert validate_operation_request("browser.navigate", {"url": "https://[0:0:0:0:0:0:0:1]/"}) == {
        "url": "https://[::1]/"
    }
    with pytest.raises(ProtocolValidationError):
        validate_operation_request("browser.navigate", {"url": "https://exa mple.com"})
    for hostname in ("foo..bar", ".example.com", "-foo.com", "foo_.com"):
        with pytest.raises(ProtocolValidationError, match="valid HTTP"):
            validate_operation_request("browser.navigate", {"url": f"https://{hostname}/"})
    for hostname in (
        "chrome.google.com",
        "chrome.google.com.",
        "CHROMEWEBSTORE.GOOGLE.COM",
        "chromewebstore.google.com.",
    ):
        with pytest.raises(ProtocolValidationError, match="restricted"):
            validate_operation_request("browser.navigate", {"url": f"https://{hostname}/detail/example"})
    for expression in (r"\xGG", r"\x0", r"\u123", r"\uZZZZ"):
        with pytest.raises(ProtocolValidationError, match="unsupported syntax"):
            validate_operation_request("browser.find", {"text": None, "regex": expression})
    for expression in (r"/\q/u", r"/\c/u", r"/\_/u"):
        with pytest.raises(ProtocolValidationError, match="unsupported syntax"):
            validate_operation_request("browser.find", {"text": None, "regex": expression})


# Mirrors tests/operations.test.js in kolega-chrome-extension. The grammar is
# duplicated across the two repos, so these cases must stay in lockstep or the
# backends will accept different regex languages.
ACCEPTED_PATTERNS = (
    "QX-[0-9]{4}-ZT",
    r"QX-\d{4}-ZT",
    "QX-[0-9]+-ZT",
    "QX-48.1-ZT",
    "QX-[0-9][0-9][0-9][0-9]-ZT",
    "a?b",
    "a*b",
    "[a-z]{2,5}",
    "x{3,}",
    "^abc$",
    r"\x41+",
    "/abc[0-9]+/i",
    "a{1000}",
    "a.{1,3}b",
    "[0-9]{4}[a-z]{2}x*y+",
    r"[a\-z]+",
    "/x{2}/u",
    r"\u0041{2}",
)

REJECTED_PATTERNS = (
    ("(abc)+", "does not support groups"),
    ("(?=a)", "does not support groups"),
    ("(a+)+$", "does not support groups"),
    ("[a](?:b)", "does not support groups"),
    (r"(a)\1", "does not support groups"),
    ("a|b", "does not support alternation"),
    (r"\p{L}", "backreferences or unicode property escapes"),
    (r"\k<x>", "backreferences or unicode property escapes"),
    ("a*b*c*d*e*", "at most 4 quantifiers"),
    ("a{1001}", "repetition counts up to 1000"),
    ("*abc", "nothing to repeat"),
    ("^*", "nothing to repeat"),
    ("a**", "nothing to repeat"),
    ("a{2,1}", "invalid repetition bound"),
    ("a{abc}", "invalid repetition bound"),
    ("a{", "unterminated repetition bound"),
    ("[[a]]", "unsupported syntax"),
    ("]", "unsupported syntax"),
    ("}", "unsupported syntax"),
    ("[abc", "is invalid"),
    ("abc\\", "is invalid"),
    ("/abc/z", "unsupported flags"),
    ("/abc/ii", "unsupported flags"),
)


@pytest.mark.parametrize("expression", ACCEPTED_PATTERNS)
def test_find_accepts_single_atom_quantifiers(expression: str) -> None:
    """Quantifiers bound to one atom are linear-time, so they are supported."""
    validated = validate_operation_request("browser.find", {"text": None, "regex": expression})
    assert validated["regex"] == expression


@pytest.mark.parametrize(("expression", "message"), REJECTED_PATTERNS)
def test_find_rejects_unsupported_regex_constructs(expression: str, message: str) -> None:
    """Grouping and alternation stay banned: they are what make backtracking blow up."""
    with pytest.raises(ProtocolValidationError, match=message):
        validate_operation_request("browser.find", {"text": None, "regex": expression})


def test_network_filter_pattern_shares_the_find_grammar() -> None:
    validated = validate_operation_request(
        "browser.network_requests",
        {"include_static": False, "filter_pattern": "/api/v[0-9]{1,2}/"},
    )
    assert validated["filter_pattern"] == "/api/v[0-9]{1,2}/"
    with pytest.raises(ProtocolValidationError, match="does not support alternation"):
        validate_operation_request(
            "browser.network_requests",
            {"include_static": False, "filter_pattern": "api|cdn"},
        )


def test_wait_for_accepts_text_and_text_gone_conditions() -> None:
    """The text branches had no coverage, which is how a null deref shipped."""
    for params in (
        {"time": None, "text": "Ready", "text_gone": None},
        {"time": None, "text": None, "text_gone": "Loading"},
        {"time": None, "text": "Ready", "text_gone": "Loading"},
        {"time": 1.5, "text": "Ready", "text_gone": "Loading"},
    ):
        assert validate_operation_request("browser.wait_for", dict(params)) == params
    with pytest.raises(ProtocolValidationError, match="Provide time, text, or text_gone"):
        validate_operation_request("browser.wait_for", {"time": None, "text": None, "text_gone": None})


def test_tabs_parameter_errors_name_the_correct_shape() -> None:
    """A bare 0 or "" for an inapplicable field burned a whole acceptance run."""
    with pytest.raises(ProtocolValidationError, match=r"index is not accepted for the list tab action"):
        validate_operation_request("browser.tabs", {"action": "list", "index": 0, "url": None})
    with pytest.raises(ProtocolValidationError, match="pass index: null"):
        validate_operation_request("browser.tabs", {"action": "new", "index": 0, "url": "https://example.com/"})
    with pytest.raises(ProtocolValidationError, match="url is invalid"):
        validate_operation_request("browser.tabs", {"action": "list", "index": None, "url": ""})
    with pytest.raises(ProtocolValidationError, match="index is required when selecting a tab"):
        validate_operation_request("browser.tabs", {"action": "select", "index": None, "url": None})
    with pytest.raises(ProtocolValidationError, match="url is only accepted when creating a tab"):
        validate_operation_request("browser.tabs", {"action": "select", "index": 2, "url": "https://example.com/"})
    # A missing key must name itself rather than say "invalid schema".
    with pytest.raises(ProtocolValidationError, match=r"missing index, url \(pass null when not applicable\)"):
        validate_operation_request("browser.tabs", {"action": "list"})
    with pytest.raises(ProtocolValidationError, match="unexpected extra"):
        validate_operation_request(
            "browser.tabs",
            {"action": "list", "index": None, "url": None, "extra": 1},
        )


def test_discovery_messages_are_exact_v1_shapes() -> None:
    discovery = {"kind": "list_runtimes", "protocol_version": 1, "request_id": "discover_1"}
    assert validate_discovery_request(discovery) == discovery
    response = {
        "kind": "runtimes",
        "protocol_version": 1,
        "request_id": "discover_1",
        "runtimes": [
            {
                "runtime_id": "runtime_1",
                "session_id": "session_1",
                "created_at_ms": NOW,
                "expires_at_ms": NOW + 1_000,
            }
        ],
    }
    assert validate_discovery_response(response, expected_request_id="discover_1") == response
    for malformed in ({**discovery, "extra": True}, {**discovery, "protocol_version": 999}):
        with pytest.raises(ProtocolValidationError):
            validate_discovery_request(malformed)


class _ChunkedBytesIO(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        return super().read(min(size, 1))


class _PartialWriter(io.BytesIO):
    def write(self, value: bytes | bytearray | memoryview) -> int:
        return super().write(bytes(value[:2]))


def test_native_framing_handles_partial_io_and_strict_json() -> None:
    frame = encode_message({"hello": "world"})
    assert read_message(_ChunkedBytesIO(frame)) == {"hello": "world"}
    output = _PartialWriter()
    write_message(output, {"hello": "world"})
    assert read_message(io.BytesIO(output.getvalue())) == {"hello": "world"}
    with pytest.raises(TruncatedMessageError):
        read_message(io.BytesIO(struct.pack("<I", 4) + b"{}"))
    duplicate = b'{"a":1,"a":2}'
    with pytest.raises(FramingError, match="duplicate"):
        read_message(io.BytesIO(struct.pack("<I", len(duplicate)) + duplicate))
    with pytest.raises(MessageTooLargeError):
        encode_message({"x": "large"}, max_bytes=2)
    assert read_message(io.BytesIO()) is None
    with pytest.raises(FramingError):
        encode_message({"bad": float("nan")})


def test_native_framing_counts_utf8_bytes_without_ascii_escape_inflation() -> None:
    value = {"text": "é" * 10}
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    frame = encode_message(value, max_bytes=len(encoded))
    assert struct.unpack("<I", frame[:4]) == (len(encoded),)
    assert frame[4:] == encoded
    with pytest.raises(MessageTooLargeError):
        encode_message(value, max_bytes=len(encoded) - 1)


def test_generated_envelopes_cannot_exceed_the_protocol_frame_bound() -> None:
    with pytest.raises(ProtocolValidationError) as error:
        Envelope.response_for(request(), {"text": "é" * 600_000})
    assert error.value.code == "message_too_large"


def test_protocol_json_rejects_non_finite_and_non_object_payloads() -> None:
    raw = request().to_mapping()
    raw["payload"] = {"result": float("inf")}
    raw["type"] = "response"
    with pytest.raises(ProtocolValidationError):
        Envelope.from_mapping(raw)
    encoded = json.dumps(request().to_mapping()).encode()
    assert Envelope.from_json(encoded) == request()
    with pytest.raises(ProtocolValidationError) as error:
        Envelope.from_json(b'{"version":NaN}')
    assert error.value.code == "invalid_json"
