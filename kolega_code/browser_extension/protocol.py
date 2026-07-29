"""Version 1 of the Kolega Chrome-extension protocol."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, TypeAlias, cast
from urllib.parse import quote, urlsplit, urlunsplit

import idna

PROTOCOL_VERSION = 1
MAX_SAFE_INTEGER = (1 << 53) - 1
MAX_IDENTIFIER_LENGTH = 128
MAX_ERROR_MESSAGE_LENGTH = 240
MAX_PROTOCOL_JSON_BYTES = 1_048_576
MAX_DISCOVERY_RUNTIMES = 64
MAX_DEADLINE_AHEAD_MS = 300_000

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ENVELOPE_KEYS = frozenset(
    {
        "version",
        "direction",
        "type",
        "request_id",
        "runtime_id",
        "session_id",
        "deadline_ms",
        "payload",
    }
)
_DISCOVERY_REQUEST_KEYS = frozenset({"kind", "protocol_version", "request_id"})
_DISCOVERY_RESPONSE_KEYS = frozenset({"kind", "protocol_version", "request_id", "runtimes"})
_DISCOVERY_RUNTIME_KEYS = frozenset({"runtime_id", "session_id", "created_at_ms", "expires_at_ms"})


class MessageDirection(str, Enum):
    """The endpoint that emitted an envelope."""

    RUNTIME_TO_EXTENSION = "runtime_to_extension"
    EXTENSION_TO_RUNTIME = "extension_to_runtime"

    @property
    def opposite(self) -> MessageDirection:
        if self is MessageDirection.RUNTIME_TO_EXTENSION:
            return MessageDirection.EXTENSION_TO_RUNTIME
        return MessageDirection.RUNTIME_TO_EXTENSION


class MessageType(str, Enum):
    """Protocol message kinds."""

    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    CANCEL = "cancel"
    EVENT = "event"


class ProtocolValidationError(ValueError):
    """A bounded protocol validation failure."""

    def __init__(self, code: str, message: str) -> None:
        bounded = message[:MAX_ERROR_MESSAGE_LENGTH]
        super().__init__(bounded)
        self.code = code
        self.message = bounded


def _invalid(code: str, message: str) -> ProtocolValidationError:
    return ProtocolValidationError(code, message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise _invalid("invalid_identifier", f"{field} is invalid")
    return value


def _exact_mapping(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys or not all(isinstance(key, str) for key in value):
        raise _invalid("invalid_schema", f"{label} has an invalid schema")
    return cast(dict[str, Any], value)


def _validate_json_value(value: object, *, depth: int = 0) -> JSONValue:
    if depth > 64:
        raise _invalid("invalid_json_value", "JSON nesting is too deep")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise _invalid("invalid_json_value", "JSON strings must contain valid Unicode scalar values")
        return value
    if _is_int(value):
        if abs(cast(int, value)) > MAX_SAFE_INTEGER:
            raise _invalid("invalid_json_value", "JSON integer exceeds the safe range")
        return cast(int, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _invalid("invalid_json_value", "JSON numbers must be finite")
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise _invalid("invalid_json_value", "JSON object keys must be strings")
        if any(any(0xD800 <= ord(character) <= 0xDFFF for character in key) for key in value):
            raise _invalid("invalid_json_value", "JSON strings must contain valid Unicode scalar values")
        return {key: _validate_json_value(item, depth=depth + 1) for key, item in value.items()}
    raise _invalid("invalid_json_value", "payload contains a non-JSON value")


ALLOWED_OPERATIONS = frozenset(
    {
        "browser.navigate",
        "browser.navigate_back",
        "browser.snapshot",
        "browser.find",
        "browser.wait_for",
        "browser.click",
        "browser.type",
        "browser.fill_form",
        "browser.select_option",
        "browser.hover",
        "browser.drag",
        "browser.press_key",
        "browser.tabs",
        "browser.network_requests",
        "browser.screenshot",
        "browser.detach",
    }
)
_MODIFIERS = frozenset({"Alt", "Control", "ControlOrMeta", "Meta", "Shift"})
_FORM_TYPES = frozenset({"textbox", "checkbox", "radio", "combobox", "slider"})
_RESTRICTED_HOSTS = frozenset({"chrome.google.com", "chromewebstore.google.com"})


def _params_error(message: str) -> ProtocolValidationError:
    return _invalid("invalid_params", message)


def _exact_params(params: dict[str, Any], keys: frozenset[str]) -> None:
    if set(params) != keys:
        raise _params_error("Operation parameters have an invalid schema")


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _string_param(
    value: object,
    name: str,
    *,
    nullable: bool = False,
    maximum: int = 20_000,
    nonempty: bool = True,
) -> str | None:
    if nullable and value is None:
        return None
    if (
        not isinstance(value, str)
        or _utf16_length(value) > maximum
        or (nonempty and not value.strip())
        or "\0" in value
    ):
        raise _params_error(f"{name} is invalid")
    return value


def _target(value: object, name: str = "target", *, nullable: bool = False) -> str | None:
    return _string_param(value, name, nullable=nullable, maximum=2_000)


def _nullable_integer(value: object, name: str, minimum: int, maximum: int) -> None:
    if value is not None and (not _is_int(value) or not minimum <= cast(int, value) <= maximum):
        raise _params_error(f"{name} is invalid")


def _nullable_number(value: object, name: str, minimum: float, maximum: float) -> None:
    if value is not None and (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise _params_error(f"{name} is invalid")


def _normalize_http_url(value: object, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    raw = _string_param(value, "url", maximum=8_192)
    assert raw is not None
    candidate = raw.strip()
    if "://" not in candidate:
        local_address = re.match(r"^(?:localhost|127\.0\.0\.1|\[::1\])(?::|/|$)", candidate, re.IGNORECASE)
        candidate = f"{'http' if local_address else 'https'}://{candidate}"
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        raise _params_error("url must be a valid HTTP or HTTPS URL") from None
    hostname = parsed.hostname
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise _params_error("url must be a valid HTTP or HTTPS URL")
    if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in hostname):
        raise _params_error("url must be a valid HTTP or HTTPS URL")
    try:
        if ":" in hostname:
            ascii_hostname = ipaddress.IPv6Address(hostname).compressed
        else:
            ascii_hostname = (
                idna.encode(
                    hostname,
                    uts46=True,
                    transitional=False,
                    std3_rules=True,
                )
                .decode("ascii")
                .lower()
            )
    except (UnicodeError, ValueError, idna.IDNAError):
        raise _params_error("url must be a valid HTTP or HTTPS URL") from None
    if ascii_hostname.rstrip(".") in _RESTRICTED_HOSTS:
        raise _params_error("url is restricted")
    if ":" in ascii_hostname:
        ascii_hostname = f"[{ascii_hostname}]"
    scheme = parsed.scheme.lower()
    if port is not None and not (scheme == "http" and port == 80) and not (scheme == "https" and port == 443):
        ascii_hostname = f"{ascii_hostname}:{port}"
    path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parsed.query, safe="/?:@!$&'()*+,;=%-._~")
    fragment = quote(parsed.fragment, safe="/?:@!$&'()*+,;=%-._~")
    return urlunsplit((scheme, ascii_hostname, path, query, fragment))


def _validate_safe_pattern(value: object, name: str = "regex") -> None:
    expression = _string_param(value, name, maximum=200)
    assert expression is not None
    source = expression
    flags = ""
    literal = re.fullmatch(r"/(.*)/([a-z]*)", expression)
    if literal is not None:
        source, flags = literal.groups()
    if flags and (not re.fullmatch(r"[imu]*", flags) or len(set(flags)) != len(flags)):
        raise _params_error(f"{name} has unsupported flags")
    in_class = False
    escaped = False
    unicode_mode = "u" in flags
    simple_escapes = frozenset("bBdfnrsStvwW0")
    syntax_escapes = frozenset("^$\\.*+?()[]{}|/")
    for index, character in enumerate(source):
        if escaped:
            if re.fullmatch(r"[1-9kpP]", character) or (
                character == "0" and index + 1 < len(source) and source[index + 1].isdigit()
            ):
                raise _params_error(f"{name} uses an unsafe expression")
            if character == "x" and (
                index + 2 >= len(source)
                or not all(item in "0123456789abcdefABCDEF" for item in source[index + 1 : index + 3])
            ):
                raise _params_error(f"{name} uses an unsafe expression")
            if character == "u" and (
                index + 4 >= len(source)
                or not all(item in "0123456789abcdefABCDEF" for item in source[index + 1 : index + 5])
            ):
                raise _params_error(f"{name} uses an unsafe expression")
            if unicode_mode and (
                character not in simple_escapes
                and character not in syntax_escapes
                and not (character == "-" and in_class)
                and character not in {"x", "u"}
            ):
                raise _params_error(f"{name} uses an unsafe expression")
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "[":
            if in_class:
                raise _params_error(f"{name} uses an unsafe expression")
            in_class = True
        elif character == "]":
            if not in_class:
                raise _params_error(f"{name} uses an unsafe expression")
            in_class = False
        elif not in_class and character in "*+?{}()|":
            raise _params_error(f"{name} uses an unsafe expression")
    if escaped or in_class:
        raise _params_error(f"{name} is invalid")


def validate_operation_request(operation: object, params: object) -> dict[str, JSONValue]:
    """Validate one fixed browser operation and its exact parameter schema."""
    if not isinstance(operation, str) or not _NAME_RE.fullmatch(operation):
        raise _invalid("invalid_operation", "request operation is invalid")
    if operation not in ALLOWED_OPERATIONS:
        raise _invalid("unsupported_operation", "Operation is not supported")
    if not isinstance(params, dict):
        raise _params_error("Operation params must be an object")
    values = dict(cast(dict[str, Any], params))

    if operation == "browser.navigate":
        _exact_params(values, frozenset({"url"}))
        values["url"] = _normalize_http_url(values["url"])
    elif operation in {"browser.navigate_back", "browser.detach"}:
        _exact_params(values, frozenset())
    elif operation == "browser.snapshot":
        _exact_params(values, frozenset({"depth", "target"}))
        _target(values["target"], nullable=True)
        _nullable_integer(values["depth"], "depth", 1, 20)
    elif operation == "browser.find":
        _exact_params(values, frozenset({"regex", "text"}))
        _string_param(values["text"], "text", nullable=True, maximum=500)
        _string_param(values["regex"], "regex", nullable=True, maximum=200)
        if (values["text"] is None) == (values["regex"] is None):
            raise _params_error("Provide exactly one of text or regex")
        if values["regex"] is not None:
            _validate_safe_pattern(values["regex"])
    elif operation == "browser.wait_for":
        _exact_params(values, frozenset({"text", "text_gone", "time"}))
        _nullable_number(values["time"], "time", 0, 30)
        _string_param(values["text"], "text", nullable=True, maximum=500)
        _string_param(values["text_gone"], "text_gone", nullable=True, maximum=500)
        if values["time"] is None and values["text"] is None and values["text_gone"] is None:
            raise _params_error("Provide time, text, or text_gone")
    elif operation == "browser.click":
        _exact_params(values, frozenset({"button", "double_click", "modifiers", "target"}))
        _target(values["target"])
        modifiers = values["modifiers"]
        if (
            not isinstance(values["double_click"], bool)
            or values["button"] not in {"left", "right", "middle"}
            or not isinstance(modifiers, list)
            or len(modifiers) > len(_MODIFIERS)
            or any(not isinstance(item, str) or item not in _MODIFIERS for item in modifiers)
            or len(set(cast(list[str], modifiers))) != len(modifiers)
        ):
            raise _params_error("click parameters are invalid")
    elif operation == "browser.type":
        _exact_params(values, frozenset({"slowly", "submit", "target", "text"}))
        _target(values["target"])
        _string_param(values["text"], "text", maximum=50_000, nonempty=False)
        if not isinstance(values["submit"], bool) or not isinstance(values["slowly"], bool):
            raise _params_error("type flags must be booleans")
    elif operation == "browser.fill_form":
        _exact_params(values, frozenset({"fields"}))
        fields = values["fields"]
        if not isinstance(fields, list) or not 1 <= len(fields) <= 50:
            raise _params_error("fields must contain between 1 and 50 entries")
        for field in fields:
            if not isinstance(field, dict) or set(field) != {"name", "target", "type", "value"}:
                raise _params_error("A form field has an invalid schema")
            _string_param(field["name"], "field name", maximum=200)
            _target(field["target"])
            if field["type"] not in _FORM_TYPES:
                raise _params_error("field type is invalid")
            field_value = _string_param(field["value"], "field value", maximum=20_000, nonempty=False)
            assert field_value is not None
            if field["type"] in {"checkbox", "radio"} and field_value.lower() not in {"true", "false"}:
                raise _params_error(f"{field['type']} value must be true or false")
    elif operation == "browser.select_option":
        _exact_params(values, frozenset({"target", "values"}))
        _target(values["target"])
        options = values["values"]
        if not isinstance(options, list) or not 1 <= len(options) <= 100:
            raise _params_error("values are invalid")
        for option in options:
            _string_param(option, "option value", maximum=2_000, nonempty=False)
    elif operation == "browser.hover":
        _exact_params(values, frozenset({"target"}))
        _target(values["target"])
    elif operation == "browser.drag":
        _exact_params(values, frozenset({"end_target", "start_target"}))
        _target(values["start_target"], "start_target")
        _target(values["end_target"], "end_target")
    elif operation == "browser.press_key":
        _exact_params(values, frozenset({"key"}))
        key = _string_param(values["key"], "key", maximum=64)
        assert key is not None
        if re.search(r"[\x00-\x1f\x7f]", key):
            raise _params_error("key is invalid")
    elif operation == "browser.tabs":
        _exact_params(values, frozenset({"action", "index", "url"}))
        action = values["action"]
        if action not in {"list", "new", "close", "select"}:
            raise _params_error("tab action is invalid")
        _nullable_integer(values["index"], "index", 0, 1_000)
        values["url"] = _normalize_http_url(values["url"], nullable=True)
        if action == "select" and values["index"] is None:
            raise _params_error("index is required when selecting a tab")
        if action in {"list", "new"} and values["index"] is not None:
            raise _params_error("index is not accepted for this tab action")
        if action != "new" and values["url"] is not None:
            raise _params_error("url is only accepted when creating a tab")
    elif operation == "browser.network_requests":
        _exact_params(values, frozenset({"filter_pattern", "include_static"}))
        if not isinstance(values["include_static"], bool):
            raise _params_error("include_static must be a boolean")
        _string_param(values["filter_pattern"], "filter_pattern", nullable=True, maximum=200)
        if values["filter_pattern"] is not None:
            _validate_safe_pattern(values["filter_pattern"], "filter_pattern")
    elif operation == "browser.screenshot":
        _exact_params(values, frozenset({"full_page", "image_type", "scale", "target"}))
        _target(values["target"], nullable=True)
        if not isinstance(values["full_page"], bool):
            raise _params_error("full_page must be a boolean")
        if values["image_type"] not in {"png", "jpeg"} or values["scale"] not in {"css", "device"}:
            raise _params_error("screenshot format or scale is invalid")
        if values["target"] is not None and values["full_page"]:
            raise _params_error("full_page cannot be combined with an element target")

    validated = _validate_json_value(values)
    assert isinstance(validated, dict)
    return validated


def _validate_payload(message_type: MessageType, value: object) -> dict[str, JSONValue]:
    schemas = {
        MessageType.REQUEST: frozenset({"operation", "params"}),
        MessageType.RESPONSE: frozenset({"result"}),
        MessageType.ERROR: frozenset({"code", "message", "retryable"}),
        MessageType.CANCEL: frozenset({"reason"}),
        MessageType.EVENT: frozenset({"event", "data"}),
    }
    payload = _exact_mapping(value, schemas[message_type], f"{message_type.value} payload")
    if message_type is MessageType.REQUEST:
        payload["params"] = validate_operation_request(payload["operation"], payload["params"])
    elif message_type is MessageType.ERROR:
        if not isinstance(payload["code"], str) or not _ERROR_CODE_RE.fullmatch(payload["code"]):
            raise _invalid("invalid_error", "error code is invalid")
        if (
            not isinstance(payload["message"], str)
            or not payload["message"]
            or len(payload["message"]) > MAX_ERROR_MESSAGE_LENGTH
        ):
            raise _invalid("invalid_error", "error message is invalid")
        if not isinstance(payload["retryable"], bool):
            raise _invalid("invalid_error", "error retryable flag is invalid")
    elif message_type is MessageType.CANCEL:
        reason = payload["reason"]
        if not isinstance(reason, str) or not reason or len(reason) > MAX_ERROR_MESSAGE_LENGTH:
            raise _invalid("invalid_cancel", "cancel reason is invalid")
    elif message_type is MessageType.EVENT:
        if not isinstance(payload["event"], str) or not _NAME_RE.fullmatch(payload["event"]):
            raise _invalid("invalid_event", "event name is invalid")
        if not isinstance(payload["data"], dict):
            raise _invalid("invalid_schema", "event data must be an object")
    validated = _validate_json_value(payload)
    assert isinstance(validated, dict)
    return validated


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _invalid("duplicate_key", "JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise _invalid("invalid_json", "JSON contains a non-finite number")


def validate_discovery_request(value: object) -> dict[str, JSONValue]:
    """Validate the exact extension-to-host runtime discovery request."""
    request = _exact_mapping(value, _DISCOVERY_REQUEST_KEYS, "runtime discovery request")
    if request["kind"] != "list_runtimes" or request["protocol_version"] != PROTOCOL_VERSION:
        raise _invalid("invalid_discovery", "runtime discovery request is invalid")
    _identifier(request["request_id"], "request_id")
    return cast(dict[str, JSONValue], request)


def validate_discovery_response(value: object, *, expected_request_id: str) -> dict[str, JSONValue]:
    """Validate the exact host-to-extension runtime discovery response."""
    response = _exact_mapping(value, _DISCOVERY_RESPONSE_KEYS, "runtime discovery response")
    runtimes = response["runtimes"]
    if (
        response["kind"] != "runtimes"
        or response["protocol_version"] != PROTOCOL_VERSION
        or response["request_id"] != expected_request_id
        or not isinstance(runtimes, list)
        or len(runtimes) > MAX_DISCOVERY_RUNTIMES
    ):
        raise _invalid("invalid_discovery", "runtime discovery response is invalid")
    for item in runtimes:
        runtime = _exact_mapping(item, _DISCOVERY_RUNTIME_KEYS, "runtime descriptor")
        _identifier(runtime["runtime_id"], "runtime_id")
        _identifier(runtime["session_id"], "session_id")
        created = runtime["created_at_ms"]
        expires = runtime["expires_at_ms"]
        if (
            not _is_int(created)
            or not _is_int(expires)
            or created <= 0
            or expires <= created
            or expires > MAX_SAFE_INTEGER
        ):
            raise _invalid("invalid_discovery", "runtime descriptor is invalid")
    validated = _validate_json_value(response)
    assert isinstance(validated, dict)
    return validated


@dataclass(frozen=True, slots=True)
class Envelope:
    """A validated protocol v1 envelope."""

    direction: MessageDirection
    type: MessageType
    request_id: str
    runtime_id: str
    session_id: str
    deadline_ms: int
    payload: dict[str, JSONValue]
    version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        validated = Envelope.from_mapping(self.to_mapping())
        object.__setattr__(self, "payload", validated.payload)
        self.to_json()

    @classmethod
    def from_mapping(cls, value: object) -> Envelope:
        envelope = _exact_mapping(value, _ENVELOPE_KEYS, "envelope")
        if not _is_int(envelope["version"]) or envelope["version"] != PROTOCOL_VERSION:
            raise _invalid("unsupported_version", "protocol version is unsupported")
        try:
            direction = MessageDirection(envelope["direction"])
        except (TypeError, ValueError):
            raise _invalid("invalid_direction", "message direction is invalid") from None
        try:
            message_type = MessageType(envelope["type"])
        except (TypeError, ValueError):
            raise _invalid("invalid_type", "message type is invalid") from None
        request_id = _identifier(envelope["request_id"], "request_id")
        runtime_id = _identifier(envelope["runtime_id"], "runtime_id")
        session_id = _identifier(envelope["session_id"], "session_id")
        deadline_ms = envelope["deadline_ms"]
        if not _is_int(deadline_ms) or deadline_ms <= 0 or deadline_ms > MAX_SAFE_INTEGER:
            raise _invalid("invalid_deadline", "deadline_ms is invalid")
        payload = _validate_payload(message_type, envelope["payload"])

        instance = object.__new__(cls)
        object.__setattr__(instance, "direction", direction)
        object.__setattr__(instance, "type", message_type)
        object.__setattr__(instance, "request_id", request_id)
        object.__setattr__(instance, "runtime_id", runtime_id)
        object.__setattr__(instance, "session_id", session_id)
        object.__setattr__(instance, "deadline_ms", deadline_ms)
        object.__setattr__(instance, "payload", payload)
        object.__setattr__(instance, "version", PROTOCOL_VERSION)
        return instance

    @classmethod
    def from_json(cls, raw: str | bytes) -> Envelope:
        if isinstance(raw, bytes):
            if len(raw) > MAX_PROTOCOL_JSON_BYTES:
                raise _invalid("message_too_large", "protocol message exceeds the size limit")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise _invalid("invalid_json", "protocol message is not valid UTF-8") from None
        else:
            text = raw
            try:
                encoded_size = len(text.encode("utf-8"))
            except UnicodeEncodeError:
                raise _invalid("invalid_json", "protocol message is not valid Unicode") from None
            if encoded_size > MAX_PROTOCOL_JSON_BYTES:
                raise _invalid("message_too_large", "protocol message exceeds the size limit")
        try:
            value = json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
        except ProtocolValidationError:
            raise
        except (json.JSONDecodeError, RecursionError):
            raise _invalid("invalid_json", "protocol message is not valid JSON") from None
        return cls.from_mapping(value)

    def to_mapping(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "direction": self.direction.value,
            "type": self.type.value,
            "request_id": self.request_id,
            "runtime_id": self.runtime_id,
            "session_id": self.session_id,
            "deadline_ms": self.deadline_ms,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        encoded = json.dumps(
            self.to_mapping(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(encoded.encode("utf-8")) > MAX_PROTOCOL_JSON_BYTES:
            raise _invalid("message_too_large", "protocol message exceeds the size limit")
        return encoded

    def is_expired(self, *, now_ms: int | None = None) -> bool:
        current = int(time.time() * 1000) if now_ms is None else now_ms
        return self.deadline_ms <= current

    @classmethod
    def request(
        cls,
        *,
        direction: MessageDirection,
        request_id: str,
        runtime_id: str,
        session_id: str,
        deadline_ms: int,
        operation: str,
        params: Mapping[str, JSONValue],
    ) -> Envelope:
        return cls(
            direction=direction,
            type=MessageType.REQUEST,
            request_id=request_id,
            runtime_id=runtime_id,
            session_id=session_id,
            deadline_ms=deadline_ms,
            payload={"operation": operation, "params": dict(params)},
        )

    @classmethod
    def response_for(cls, request: Envelope, result: JSONValue) -> Envelope:
        return cls(
            direction=request.direction.opposite,
            type=MessageType.RESPONSE,
            request_id=request.request_id,
            runtime_id=request.runtime_id,
            session_id=request.session_id,
            deadline_ms=request.deadline_ms,
            payload={"result": result},
        )

    @classmethod
    def error_for(
        cls,
        request: Envelope,
        *,
        code: str,
        message: str,
        retryable: bool = False,
    ) -> Envelope:
        return cls(
            direction=request.direction.opposite,
            type=MessageType.ERROR,
            request_id=request.request_id,
            runtime_id=request.runtime_id,
            session_id=request.session_id,
            deadline_ms=request.deadline_ms,
            payload={"code": code, "message": message[:MAX_ERROR_MESSAGE_LENGTH], "retryable": retryable},
        )

    @classmethod
    def cancel_for(cls, request: Envelope, *, reason: str) -> Envelope:
        return cls(
            direction=request.direction,
            type=MessageType.CANCEL,
            request_id=request.request_id,
            runtime_id=request.runtime_id,
            session_id=request.session_id,
            deadline_ms=request.deadline_ms,
            payload={"reason": reason[:MAX_ERROR_MESSAGE_LENGTH]},
        )

    @classmethod
    def event(
        cls,
        *,
        direction: MessageDirection,
        request_id: str,
        runtime_id: str,
        session_id: str,
        deadline_ms: int,
        event: str,
        data: Mapping[str, JSONValue],
    ) -> Envelope:
        return cls(
            direction=direction,
            type=MessageType.EVENT,
            request_id=request_id,
            runtime_id=runtime_id,
            session_id=session_id,
            deadline_ms=deadline_ms,
            payload={"event": event, "data": dict(data)},
        )
