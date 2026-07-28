"""Dependency-free 5-field cron parsing for the CLI ``/loop`` scheduler.

Supports the classic vixie-cron subset that covers essentially every schedule a
developer writes by hand: ``*``, single values, ranges (``a-b``), steps
(``*/n`` and ``a-b/n``), and comma-separated lists of those.  Extended syntax
(``L``, ``W``, ``?``, ``#``, month/day names, ``@daily`` macros) is rejected
with an explanatory message rather than silently mis-scheduled.

Everything here operates on naive **local wall-clock** ``datetime`` values.  The
loop scheduler recomputes the next fire after every iteration, so a DST
transition can at worst skip or repeat a single fire.

This module is intentionally standalone (stdlib only, no Textual, no sibling CLI
imports) so it stays cheap to import and trivial to unit test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

#: How far ahead ``next_fire_after`` will search before giving up. Four years
#: covers the worst realistic case, ``0 0 29 2 *`` (February 29th), from any
#: starting point.
SEARCH_HORIZON_DAYS = 366 * 4

#: The only characters a supported cron field may contain.
_ALLOWED_FIELD_CHARS = re.compile(r"^[0-9*/,\-]+$")

SUPPORTED_SYNTAX_HELP = (
    "Supported cron syntax: 5 fields (minute hour day-of-month month day-of-week) "
    "using *, a value, a-b ranges, */n or a-b/n steps, and comma-separated lists. "
    "Names (MON, JAN), macros (@daily) and extended syntax (L, W, ?, #) are not supported."
)


class CronError(ValueError):
    """Raised when a cron expression cannot be parsed or never matches."""


@dataclass(frozen=True)
class _FieldSpec:
    name: str
    low: int
    high: int


_MINUTE = _FieldSpec("minute", 0, 59)
_HOUR = _FieldSpec("hour", 0, 23)
_DOM = _FieldSpec("day-of-month", 1, 31)
_MONTH = _FieldSpec("month", 1, 12)
#: Day-of-week accepts 0-7 on input; 7 is normalized to 0 (both are Sunday).
_DOW = _FieldSpec("day-of-week", 0, 7)


def _parse_field(text: str, spec: _FieldSpec) -> frozenset[int]:
    """Expand one cron field into the set of values it matches."""
    raw = text.strip()
    if not raw:
        raise CronError(f"The {spec.name} field is empty. {SUPPORTED_SYNTAX_HELP}")
    if not _ALLOWED_FIELD_CHARS.match(raw):
        raise CronError(f"Unsupported {spec.name} field {text!r}. {SUPPORTED_SYNTAX_HELP}")

    values: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            raise CronError(f"Empty list item in the {spec.name} field {text!r}. {SUPPORTED_SYNTAX_HELP}")
        values.update(_parse_field_item(item, spec, text))
    return frozenset(values)


def _parse_field_item(item: str, spec: _FieldSpec, field_text: str) -> set[int]:
    step = 1
    body = item
    if "/" in item:
        body, _, step_text = item.partition("/")
        if "/" in step_text:
            raise CronError(f"Nested step in the {spec.name} field {field_text!r}. {SUPPORTED_SYNTAX_HELP}")
        if not step_text.isdigit() or int(step_text) < 1:
            raise CronError(
                f"Step must be a positive integer in the {spec.name} field {field_text!r}. {SUPPORTED_SYNTAX_HELP}"
            )
        step = int(step_text)
        body = body.strip()

    if body == "*":
        low, high = spec.low, spec.high
    elif "-" in body:
        start_text, _, end_text = body.partition("-")
        low = _parse_value(start_text, spec, field_text)
        high = _parse_value(end_text, spec, field_text)
        if low > high:
            raise CronError(
                f"Range {body!r} in the {spec.name} field is inverted; use a-b with a <= b. {SUPPORTED_SYNTAX_HELP}"
            )
    else:
        low = _parse_value(body, spec, field_text)
        # A bare value with a step (``5/10``) means "from 5 to the field maximum".
        high = spec.high if step > 1 else low

    return set(range(low, high + 1, step))


def _parse_value(text: str, spec: _FieldSpec, field_text: str) -> int:
    body = text.strip()
    if not body.isdigit():
        raise CronError(f"Unsupported value {body!r} in the {spec.name} field. {SUPPORTED_SYNTAX_HELP}")
    value = int(body)
    if not (spec.low <= value <= spec.high):
        raise CronError(f"Value {value} is out of range for the {spec.name} field ({spec.low}-{spec.high}).")
    return value


@dataclass(frozen=True)
class CronSchedule:
    """A parsed 5-field cron expression.

    ``dom_restricted`` / ``dow_restricted`` record whether the day-of-month and
    day-of-week fields were written as something other than a bare ``*``.  When
    both are restricted, vixie-cron matches a day if **either** field matches
    (so ``0 0 13 * 5`` means "the 13th, and every Friday"), which is what
    :meth:`matches` implements.
    """

    expression: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    dom_restricted: bool
    dow_restricted: bool

    @classmethod
    def parse(cls, expression: str) -> "CronSchedule":
        raw = " ".join(str(expression or "").split())
        if not raw:
            raise CronError(f"Empty cron expression. {SUPPORTED_SYNTAX_HELP}")
        if raw.startswith("@"):
            raise CronError(f"Cron macros like {raw.split()[0]!r} are not supported. {SUPPORTED_SYNTAX_HELP}")

        fields = raw.split(" ")
        if len(fields) != 5:
            raise CronError(f"A cron expression needs exactly 5 fields, got {len(fields)}. {SUPPORTED_SYNTAX_HELP}")

        minute_text, hour_text, dom_text, month_text, dow_text = fields
        dow_values = {value % 7 for value in _parse_field(dow_text, _DOW)}

        return cls(
            expression=raw,
            minutes=_parse_field(minute_text, _MINUTE),
            hours=_parse_field(hour_text, _HOUR),
            days_of_month=_parse_field(dom_text, _DOM),
            months=_parse_field(month_text, _MONTH),
            days_of_week=frozenset(dow_values),
            dom_restricted=dom_text.strip() != "*",
            dow_restricted=dow_text.strip() != "*",
        )

    def describe(self) -> str:
        """Short human label for status output."""
        return f"cron {self.expression}"

    def matches(self, moment: datetime) -> bool:
        """Whether ``moment`` (to the minute) is a firing time."""
        if moment.minute not in self.minutes or moment.hour not in self.hours:
            return False
        return self._matches_date(moment.date())

    def _matches_date(self, day: date) -> bool:
        if day.month not in self.months:
            return False
        dom_ok = day.day in self.days_of_month
        # Python's weekday() is Monday=0..Sunday=6; cron is Sunday=0..Saturday=6.
        dow_ok = ((day.weekday() + 1) % 7) in self.days_of_week
        if self.dom_restricted and self.dow_restricted:
            return dom_ok or dow_ok
        return dom_ok and dow_ok

    def next_fire_after(self, after: datetime) -> datetime:
        """The first matching wall-clock minute strictly after ``after``."""
        start = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        sorted_hours = sorted(self.hours)
        sorted_minutes = sorted(self.minutes)
        day = start.date()
        last_day = (after + timedelta(days=SEARCH_HORIZON_DAYS)).date()
        first_day = True

        while day <= last_day:
            if self._matches_date(day):
                for hour in sorted_hours:
                    if first_day and hour < start.hour:
                        continue
                    for minute in sorted_minutes:
                        if first_day and hour == start.hour and minute < start.minute:
                            continue
                        return datetime(day.year, day.month, day.day, hour, minute)
            day += timedelta(days=1)
            first_day = False

        raise CronError(
            f"Cron expression {self.expression!r} has no matching time within "
            f"{SEARCH_HORIZON_DAYS // 366} years — check the day-of-month and month fields."
        )


def parse_cron(expression: str) -> CronSchedule:
    """Convenience wrapper around :meth:`CronSchedule.parse`."""
    return CronSchedule.parse(expression)


def try_parse_cron(expression: str) -> Optional[CronSchedule]:
    """Parse ``expression``, returning ``None`` instead of raising on failure."""
    try:
        return CronSchedule.parse(expression)
    except CronError:
        return None
