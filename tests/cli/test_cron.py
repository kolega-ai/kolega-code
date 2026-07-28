"""Unit tests for the dependency-free cron parser in :mod:`kolega_code.cli.cron`."""

from datetime import datetime

import pytest

from kolega_code.cli.cron import SEARCH_HORIZON_DAYS, CronError, CronSchedule, parse_cron, try_parse_cron

BASE = datetime(2026, 7, 27, 10, 17, 30)  # a Monday


# ----------------------------------------------------------------------
# Field parsing
# ----------------------------------------------------------------------


def test_star_expands_to_the_full_range():
    schedule = parse_cron("* * * * *")
    assert schedule.minutes == frozenset(range(60))
    assert schedule.hours == frozenset(range(24))
    assert schedule.days_of_month == frozenset(range(1, 32))
    assert schedule.months == frozenset(range(1, 13))
    assert schedule.days_of_week == frozenset(range(7))
    assert schedule.dom_restricted is False
    assert schedule.dow_restricted is False


def test_single_value_range_step_and_list():
    assert parse_cron("7 * * * *").minutes == frozenset({7})
    assert parse_cron("10-13 * * * *").minutes == frozenset({10, 11, 12, 13})
    assert parse_cron("*/15 * * * *").minutes == frozenset({0, 15, 30, 45})
    assert parse_cron("0-30/10 * * * *").minutes == frozenset({0, 10, 20, 30})
    assert parse_cron("5,35,55 * * * *").minutes == frozenset({5, 35, 55})
    assert parse_cron("0 9-17/4 * * *").hours == frozenset({9, 13, 17})


def test_bare_value_with_step_runs_to_the_field_maximum():
    assert parse_cron("40/10 * * * *").minutes == frozenset({40, 50})


def test_day_of_week_seven_normalizes_to_sunday():
    assert parse_cron("0 0 * * 7").days_of_week == frozenset({0})
    assert parse_cron("0 0 * * 0").days_of_week == parse_cron("0 0 * * 7").days_of_week
    assert parse_cron("0 0 * * 5-7").days_of_week == frozenset({5, 6, 0})


def test_restriction_flags_track_non_star_fields():
    schedule = parse_cron("0 0 13 * 5")
    assert schedule.dom_restricted is True
    assert schedule.dow_restricted is True
    assert parse_cron("0 0 */2 * *").dom_restricted is True


def test_expression_is_normalized_for_display():
    schedule = parse_cron("  0   9  *  *  1-5 ")
    assert schedule.expression == "0 9 * * 1-5"
    assert schedule.describe() == "cron 0 9 * * 1-5"


# ----------------------------------------------------------------------
# Rejections
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "0 9 * * MON",  # day names
        "0 JAN * * *",  # month names
        "0 0 L * *",  # last-day-of-month
        "0 0 15W * *",  # nearest weekday
        "0 0 ? * *",  # no-specific-value
        "0 0 * * 5#2",  # nth weekday
        "@daily",  # macros
        "@reboot",
        "* * * *",  # too few fields
        "* * * * * *",  # too many fields
        "",  # empty
        "60 * * * *",  # minute out of range
        "* 24 * * *",  # hour out of range
        "* * 0 * *",  # day-of-month out of range
        "* * * 13 *",  # month out of range
        "* * * * 8",  # day-of-week out of range
        "*/0 * * * *",  # zero step
        "10-5 * * * *",  # inverted range
        "1,,2 * * * *",  # empty list item
    ],
)
def test_unsupported_syntax_is_rejected(expression):
    with pytest.raises(CronError):
        parse_cron(expression)


def test_rejection_message_names_the_supported_syntax():
    with pytest.raises(CronError) as excinfo:
        parse_cron("0 9 * * MON")
    message = str(excinfo.value)
    assert "day-of-week" in message
    assert "Supported cron syntax" in message


def test_try_parse_cron_returns_none_instead_of_raising():
    assert try_parse_cron("@daily") is None
    assert try_parse_cron("0 9 * * *") is not None


# ----------------------------------------------------------------------
# next_fire_after
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("*/15 * * * *", datetime(2026, 7, 27, 10, 30)),
        ("0 9 * * 1-5", datetime(2026, 7, 28, 9, 0)),
        ("0 0 1 * *", datetime(2026, 8, 1, 0, 0)),
        ("0 0 * * 0", datetime(2026, 8, 2, 0, 0)),
        ("0 0 * * 7", datetime(2026, 8, 2, 0, 0)),
        ("5,35 * * * *", datetime(2026, 7, 27, 10, 35)),
        ("* * * * *", datetime(2026, 7, 27, 10, 18)),
    ],
)
def test_next_fire_after(expression, expected):
    assert parse_cron(expression).next_fire_after(BASE) == expected


def test_next_fire_is_strictly_after_the_given_moment():
    exact = datetime(2026, 7, 27, 10, 30, 0)
    assert parse_cron("*/15 * * * *").next_fire_after(exact) == datetime(2026, 7, 27, 10, 45)


def test_next_fire_rolls_over_midnight():
    late = datetime(2026, 7, 27, 23, 59, 30)
    assert parse_cron("0 0 * * *").next_fire_after(late) == datetime(2026, 7, 28, 0, 0)


def test_dom_and_dow_both_restricted_matches_either():
    # "the 13th, and every Friday" — vixie-cron OR semantics.
    schedule = parse_cron("0 0 13 * 5")
    assert schedule.matches(datetime(2026, 8, 13, 0, 0))  # a Thursday, matches by day-of-month
    assert schedule.matches(datetime(2026, 7, 31, 0, 0))  # a Friday, matches by day-of-week
    assert not schedule.matches(datetime(2026, 7, 28, 0, 0))  # Tuesday the 28th, matches neither
    assert schedule.next_fire_after(BASE) == datetime(2026, 7, 31, 0, 0)


def test_only_one_day_field_restricted_uses_and_semantics():
    # Day-of-week alone: every Friday, regardless of the date.
    friday_only = parse_cron("0 0 * * 5")
    assert friday_only.matches(datetime(2026, 7, 31, 0, 0))
    assert not friday_only.matches(datetime(2026, 8, 13, 0, 0))
    # Day-of-month alone: the 13th, regardless of the weekday.
    thirteenth_only = parse_cron("0 0 13 * *")
    assert thirteenth_only.matches(datetime(2026, 8, 13, 0, 0))
    assert not thirteenth_only.matches(datetime(2026, 7, 31, 0, 0))


def test_leap_day_lands_on_a_leap_year():
    assert parse_cron("0 0 29 2 *").next_fire_after(BASE) == datetime(2028, 2, 29, 0, 0)


def test_impossible_date_raises_within_the_search_horizon():
    # February 30th never happens.
    with pytest.raises(CronError) as excinfo:
        parse_cron("0 0 30 2 *").next_fire_after(BASE)
    assert "no matching time" in str(excinfo.value)


def test_search_horizon_covers_a_leap_cycle():
    assert SEARCH_HORIZON_DAYS >= 366 * 4


def test_matches_ignores_seconds():
    schedule = parse_cron("30 10 * * *")
    assert schedule.matches(datetime(2026, 7, 27, 10, 30, 59))
    assert not schedule.matches(datetime(2026, 7, 27, 10, 31, 0))


def test_schedule_is_hashable_and_frozen():
    schedule = parse_cron("0 9 * * *")
    assert {schedule, parse_cron("0 9 * * *")} == {schedule}
    with pytest.raises(Exception):
        schedule.expression = "changed"  # type: ignore[misc]


def test_parse_is_available_as_a_classmethod():
    assert CronSchedule.parse("0 9 * * *").expression == "0 9 * * *"
