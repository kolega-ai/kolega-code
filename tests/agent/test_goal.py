# ruff: noqa
"""Unit tests for the goal verifier module :mod:`kolega_code.agent.goal`.

Covers :class:`GoalVerdict`, :func:`build_goal_verifier_instruction`,
:func:`parse_goal_verdict`, and the internal :func:`_iter_json_objects` helper.
"""

from kolega_code.agent.goal import (
    GoalVerdict,
    _iter_json_objects,
    build_goal_verifier_instruction,
    parse_goal_verdict,
)


# ---------------------------------------------------------------------------
# parse_goal_verdict
# ---------------------------------------------------------------------------


def test_parse_verdict_ok_true_no_reason():
    assert parse_goal_verdict('{"ok": true}') == GoalVerdict(met=True, reason="")


def test_parse_verdict_ok_false_with_reason():
    assert parse_goal_verdict('{"ok": false, "reason": "tests fail"}') == GoalVerdict(met=False, reason="tests fail")


def test_parse_verdict_prose_before_json():
    verdict = parse_goal_verdict('I checked the files.\n{"ok": true}')
    assert verdict.met is True
    assert verdict.reason == ""


def test_parse_verdict_picks_last_with_ok_key():
    text = '{"ok": false, "reason": "first"} ... {"ok": true}'
    verdict = parse_goal_verdict(text)
    assert verdict.met is True
    assert verdict.reason == ""


def test_parse_verdict_missing_ok_key():
    verdict = parse_goal_verdict('{"status": "done"}')
    assert verdict.met is False
    assert "not a valid verdict" in verdict.reason


def test_parse_verdict_malformed_json():
    verdict = parse_goal_verdict("not json at all")
    assert verdict.met is False
    assert "not a valid verdict" in verdict.reason


def test_parse_verdict_empty_string():
    verdict = parse_goal_verdict("")
    assert verdict.met is False
    assert verdict.reason == "verifier returned no output"


def test_parse_verdict_ok_not_boolean_is_truthy():
    # bool("yes") is True — documented behavior, not a strict type check.
    verdict = parse_goal_verdict('{"ok": "yes"}')
    assert verdict.met is True
    assert verdict.reason == ""


def test_parse_verdict_reason_not_string_is_ignored():
    verdict = parse_goal_verdict('{"ok": false, "reason": 42}')
    assert verdict.met is False
    # A non-string reason is dropped, leaving an empty reason.
    assert verdict.reason == ""


def test_parse_verdict_truncates_long_non_json_text():
    long_text = "x" * 1000
    verdict = parse_goal_verdict(long_text)
    assert verdict.met is False
    assert "not a valid verdict" in verdict.reason
    # The reason should be truncated to ~280 chars plus the ellipsis.
    assert len(verdict.reason) <= 280 + len("verifier reply was not a valid verdict: ") + 1
    assert verdict.reason.endswith("…")


def test_parse_verdict_braces_inside_string_value():
    verdict = parse_goal_verdict('{"ok": true, "reason": "has } brace"}')
    assert verdict.met is True
    assert verdict.reason == "has } brace"


# ---------------------------------------------------------------------------
# build_goal_verifier_instruction
# ---------------------------------------------------------------------------


def test_instruction_includes_condition_text():
    condition = "All pytest tests in tests/ must pass"
    instruction = build_goal_verifier_instruction(condition)
    assert condition in instruction


def test_instruction_includes_json_contract():
    instruction = build_goal_verifier_instruction("do the thing")
    assert '{"ok": true}' in instruction
    assert '{"ok": false' in instruction
    assert "reason" in instruction


def test_instruction_instructs_not_to_modify():
    instruction = build_goal_verifier_instruction("do the thing")
    assert "not modify" in instruction.lower() or "do not" in instruction.lower()


# ---------------------------------------------------------------------------
# _iter_json_objects
# ---------------------------------------------------------------------------


def test_iter_json_objects_simple():
    spans = list(_iter_json_objects('prefix {"a": 1} suffix'))
    assert spans == [(7, 15)]


def test_iter_json_objects_multiple():
    spans = list(_iter_json_objects('{"a": 1} {"b": 2}'))
    assert spans == [(0, 8), (9, 17)]


def test_iter_json_objects_nested_yields_outermost_only():
    text = '{"outer": {"inner": 1}}'
    spans = list(_iter_json_objects(text))
    assert spans == [(0, len(text))]


def test_iter_json_objects_braces_inside_strings():
    text = '{"reason": "has } and { inside"}'
    spans = list(_iter_json_objects(text))
    assert spans == [(0, len(text))]


def test_iter_json_objects_empty_or_no_objects():
    assert list(_iter_json_objects("no braces here")) == []
    assert list(_iter_json_objects("")) == []


def test_iter_json_objects_unbalanced_braces_not_yielded():
    # An opening brace that never closes should not be yielded.
    assert list(_iter_json_objects('{"broken": {')) == []
