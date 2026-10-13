"""Parsing a tool call's raw arguments."""

from kolega_code.llm.models import safe_parse_tool_arguments


def test_valid_json_is_returned_as_is() -> None:
    assert safe_parse_tool_arguments('{"path": "a.py", "line": 3}') == {"path": "a.py", "line": 3}


def test_empty_arguments_are_an_empty_dict() -> None:
    assert safe_parse_tool_arguments("") == {}


def test_raw_tab_inside_a_string_is_kept() -> None:
    raw = '{"items": [{"text": "\tindented line"}]}'
    assert safe_parse_tool_arguments(raw) == {"items": [{"text": "\tindented line"}]}


def test_raw_newline_inside_a_string_is_kept() -> None:
    raw = '{"text": "first line\nsecond line", "count": 1}'
    assert safe_parse_tool_arguments(raw) == {"text": "first line\nsecond line", "count": 1}


def test_trailing_comma_is_repaired() -> None:
    assert safe_parse_tool_arguments('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}


def test_one_missing_closing_brace_is_repaired() -> None:
    assert safe_parse_tool_arguments('{"a": {"b": 1}') == {"a": {"b": 1}}


def test_repair_and_raw_control_character_together() -> None:
    assert safe_parse_tool_arguments('{"text": "a\tb",}') == {"text": "a\tb"}


def test_unparseable_arguments_fall_back_without_raising() -> None:
    raw = '{"items": [{"name": "x" "count": }'
    assert safe_parse_tool_arguments(raw) == {"_raw_arguments": raw, "_parse_error": "json_decode_error"}
