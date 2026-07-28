"""Unit tests for :mod:`kolega_code.cli.loop`: durations, command parsing, state, loop.md."""

import sys
from datetime import datetime, timedelta

import pytest

from kolega_code.cli import loop as loop_module
from kolega_code.cli.loop import (
    DEFAULT_LOOP_EXPIRY_DAYS,
    DEFAULT_LOOP_MAX_ITERATIONS,
    LOOP_MD_MAX_BYTES,
    LOOP_MD_RELATIVE_PATH,
    LOOP_STATUS_ALIASES,
    LOOP_STOP_ALIASES,
    MIN_INTERVAL_SECONDS,
    PROMPT_SOURCE_INLINE,
    PROMPT_SOURCE_LOOP_MD,
    LoopError,
    LoopSchedule,
    LoopState,
    build_loop_iteration_prompt,
    build_loop_prompt_extension_markdown,
    format_countdown,
    format_duration_short,
    format_loop_status,
    interval_schedule,
    loop_state_label,
    loop_status_summary,
    parse_duration,
    parse_interval,
    parse_loop_command,
    parse_schedule_text,
    read_loop_md,
)

NOW = datetime(2026, 7, 27, 10, 0, 0)


@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze :func:`loop.now_local`, returning a setter to advance it."""
    current = {"now": NOW}
    monkeypatch.setattr(loop_module, "now_local", lambda: current["now"])
    return current


def write_loop_md(project, content: str):
    path = project / LOOP_MD_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Durations
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("30s", 30),
        ("5m", 300),
        ("2h", 7200),
        ("1d", 86400),
        ("90 minutes", 5400),
        ("2 hours", 7200),
        ("every 2 hours", 7200),
        ("EVERY 15 Mins", 900),
        ("45sec", 45),
        ("3 days", 259200),
    ],
)
def test_parse_duration_accepts_compact_and_natural_forms(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "soon", "5x", "m5", "-5m", "0m", "5 5", "check the deploy"])
def test_parse_duration_rejects_non_durations(text):
    with pytest.raises(LoopError):
        parse_duration(text)


def test_parse_interval_enforces_the_floor():
    assert parse_interval("15s") == MIN_INTERVAL_SECONDS
    with pytest.raises(LoopError) as excinfo:
        parse_interval("5s")
    assert str(MIN_INTERVAL_SECONDS) in str(excinfo.value)


def test_interval_schedule_enforces_the_floor():
    with pytest.raises(LoopError):
        interval_schedule(1)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0s"), (45, "45s"), (60, "1m"), (90, "1m 30s"), (300, "5m"), (5400, "1h 30m"), (86400, "1d")],
)
def test_format_duration_short(seconds, expected):
    assert format_duration_short(seconds) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(-5, "now"), (0, "now"), (5, "5s"), (59, "59s"), (60, "1m"), (119, "1m"), (3630, "1h")],
)
def test_format_countdown_is_coarse_above_a_minute(seconds, expected):
    assert format_countdown(seconds) == expected


# ----------------------------------------------------------------------
# Schedules
# ----------------------------------------------------------------------


def test_interval_schedule_properties():
    schedule = parse_schedule_text("5m")
    assert schedule.kind == "interval"
    assert schedule.value == "300"
    assert schedule.label() == "every 5m"
    assert schedule.first_fire_is_immediate is True
    assert schedule.next_after(NOW) == NOW + timedelta(minutes=5)


def test_cron_schedule_properties():
    schedule = parse_schedule_text("0 9 * * *")
    assert schedule.kind == "cron"
    assert schedule.value == "0 9 * * *"
    assert schedule.label() == "cron 0 9 * * *"
    assert schedule.first_fire_is_immediate is False
    assert schedule.next_after(NOW) == datetime(2026, 7, 28, 9, 0)


def test_parse_schedule_text_prefers_intervals_then_cron():
    assert parse_schedule_text("every 2 hours").interval_seconds == 7200
    assert parse_schedule_text('"0 9 * * *"').cron is not None
    with pytest.raises(LoopError):
        parse_schedule_text("")
    with pytest.raises(LoopError):
        parse_schedule_text("not a schedule at all")


def test_schedule_must_be_exactly_one_kind():
    with pytest.raises(LoopError):
        LoopSchedule()
    with pytest.raises(LoopError):
        LoopSchedule(interval_seconds=60, cron=parse_schedule_text("0 9 * * *").cron)


# ----------------------------------------------------------------------
# Command parsing
# ----------------------------------------------------------------------


def test_interval_and_prompt():
    command = parse_loop_command("5m check if CI went green")
    assert command.action == "start"
    assert command.spec is not None
    assert command.spec.schedule is not None
    assert command.spec.schedule.interval_seconds == 300
    assert command.spec.prompt == "check if CI went green"
    assert command.spec.fresh is False
    assert command.spec.max_iterations == DEFAULT_LOOP_MAX_ITERATIONS
    assert command.spec.expires_seconds == DEFAULT_LOOP_EXPIRY_DAYS * 86400


def test_natural_interval_and_prompt():
    command = parse_loop_command("every 2 hours summarize open PRs")
    assert command.spec is not None
    assert command.spec.schedule is not None
    assert command.spec.schedule.interval_seconds == 7200
    assert command.spec.prompt == "summarize open PRs"


def test_cron_via_flag_and_via_leading_quotes():
    by_flag = parse_loop_command('--cron "0 9 * * 1-5" summarize open PRs')
    by_quotes = parse_loop_command('"0 9 * * 1-5" summarize open PRs')
    for command in (by_flag, by_quotes):
        assert command.action == "start"
        assert command.spec is not None
        assert command.spec.schedule is not None
        assert command.spec.schedule.cron is not None
        assert command.spec.schedule.cron.expression == "0 9 * * 1-5"
        assert command.spec.prompt == "summarize open PRs"


def test_flags_are_accepted_in_any_order():
    first = parse_loop_command("--fresh --max-iterations 5 --expires 2h 30s poll the build")
    second = parse_loop_command("--expires 2h --max-iterations 5 --fresh 30s poll the build")
    for command in (first, second):
        assert command.spec is not None
        assert command.spec.fresh is True
        assert command.spec.max_iterations == 5
        assert command.spec.expires_seconds == 7200
        assert command.spec.schedule is not None
        assert command.spec.schedule.interval_seconds == 30
        assert command.spec.prompt == "poll the build"


def test_max_iters_alias():
    command = parse_loop_command("--max-iters 3 5m x")
    assert command.spec is not None
    assert command.spec.max_iterations == 3


@pytest.mark.parametrize("alias", sorted(LOOP_STOP_ALIASES))
def test_stop_aliases(alias):
    assert parse_loop_command(alias).action == "stop"
    assert parse_loop_command(alias.upper()).action == "stop"


@pytest.mark.parametrize("alias", sorted(LOOP_STATUS_ALIASES))
def test_status_aliases(alias):
    assert parse_loop_command(alias).action == "status"


def test_bare_loop_defers_to_loop_md():
    command = parse_loop_command("")
    assert command.action == "start"
    assert command.spec is not None
    assert command.spec.schedule is None
    assert command.spec.prompt == ""


def test_prompt_without_schedule_defers_the_schedule_to_loop_md():
    command = parse_loop_command("check the deploy")
    assert command.action == "start"
    assert command.spec is not None
    assert command.spec.schedule is None
    assert command.spec.prompt == "check the deploy"


def test_prompt_internal_whitespace_is_preserved():
    command = parse_loop_command("5m check   the    deploy")
    assert command.spec is not None
    assert command.spec.prompt == "check   the    deploy"


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ("5s too fast", "at least"),
        ("--max-iterations 0 5m x", "positive whole number"),
        ("--max-iterations abc 5m x", "positive whole number"),
        ("--bogus 5m x", "Unknown option"),
        ('--cron "0 9 * * MON" x', "day-of-week"),
        ("--expires nope 5m x", "duration"),
        ('--cron "" x', "must not be empty"),
    ],
)
def test_invalid_invocations_return_usage_with_a_reason(args, fragment):
    command = parse_loop_command(args)
    assert command.action == "usage"
    assert fragment in command.reason


def test_unterminated_quote_is_treated_as_prompt_text():
    command = parse_loop_command('5m say "hello')
    assert command.action == "start"
    assert command.spec is not None
    assert command.spec.prompt == 'say "hello'


# ----------------------------------------------------------------------
# LoopState
# ----------------------------------------------------------------------


def test_create_interval_loop_is_due_immediately():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    assert state.next_fire_at == NOW.isoformat()
    assert state.is_due(NOW) is True
    assert state.iterations == 0
    assert state.prompt_source == PROMPT_SOURCE_INLINE


def test_create_cron_loop_waits_for_the_first_match():
    state = LoopState.create(parse_schedule_text("0 9 * * *"), "briefing", now=NOW)
    assert state.next_fire_at == datetime(2026, 7, 28, 9, 0).isoformat()
    assert state.is_due(NOW) is False


def test_round_trip_through_dict():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW, fresh=True, max_iterations=7)
    state.mark_fired(NOW)
    state.advance_after_completion(NOW + timedelta(seconds=30))
    assert LoopState.from_dict(state.to_dict()) == state


def test_from_dict_tolerates_missing_keys():
    state = LoopState.from_dict({})
    assert state.iterations == 0
    assert state.max_iterations == DEFAULT_LOOP_MAX_ITERATIONS
    assert state.schedule_kind == "interval"
    assert state.schedule_label() == f"every {format_duration_short(MIN_INTERVAL_SECONDS)}"


def test_interval_delay_is_measured_from_completion():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    state.mark_fired(NOW)
    # A slow iteration: the next fire is five minutes after it *finished*.
    finished = NOW + timedelta(minutes=8)
    state.advance_after_completion(finished)
    assert state.next_fire_at == (finished + timedelta(minutes=5)).isoformat()


def test_cron_advance_uses_the_wall_clock():
    state = LoopState.create(parse_schedule_text("0 9 * * *"), "briefing", now=NOW)
    state.mark_fired(datetime(2026, 7, 28, 9, 0))
    state.advance_after_completion(datetime(2026, 7, 28, 9, 2))
    assert state.next_fire_at == datetime(2026, 7, 29, 9, 0).isoformat()


def test_mark_fired_increments_and_clears_deferred():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    state.deferred = True
    state.mark_fired(NOW)
    assert state.iterations == 1
    assert state.last_fired_at == NOW.isoformat()
    assert state.deferred is False


def test_cap_and_expiry_end_the_loop(frozen_clock):
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW, max_iterations=2, expires_seconds=3600)
    assert state.is_active is True
    state.mark_fired(NOW)
    assert state.is_active is True
    state.mark_fired(NOW)
    assert state.reached_cap is True
    assert state.is_active is False

    fresh = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW, expires_seconds=3600)
    assert fresh.is_expired(NOW) is False
    assert fresh.is_expired(NOW + timedelta(hours=2)) is True
    frozen_clock["now"] = NOW + timedelta(hours=2)
    assert fresh.is_active is False


def test_stopped_loop_is_inactive():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    state.stopped = True
    assert state.is_active is False


def test_advance_with_an_unreadable_schedule_stops_instead_of_spinning():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    state.schedule_value = "not-a-number"
    state.advance_after_completion(NOW)
    assert state.stopped is True
    assert state.status_note


def test_seconds_until_and_is_due():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    state.mark_fired(NOW)
    state.advance_after_completion(NOW)
    assert state.seconds_until(NOW) == 300
    assert state.is_due(NOW) is False
    assert state.is_due(NOW + timedelta(minutes=5)) is True


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def test_status_summary_shows_schedule_countdown_and_counter():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW, max_iterations=4)
    state.mark_fired(NOW)
    state.advance_after_completion(NOW)
    summary = loop_status_summary(state, now=NOW + timedelta(minutes=2))
    assert summary == "check CI (every 5m · next 3m · 1/4)"


def test_status_summary_truncates_a_long_prompt():
    state = LoopState.create(parse_schedule_text("5m"), "x" * 200, now=NOW)
    summary = loop_status_summary(state, now=NOW)
    assert summary.startswith("x" * 40)
    assert "…" in summary
    assert len(summary.split(" (")[0]) <= 48


def test_status_summary_reports_waiting_for_idle():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    state.deferred = True
    assert "waiting for idle" in loop_status_summary(state, now=NOW)


def test_status_summary_reports_a_terminal_state():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    state.stopped = True
    assert loop_status_summary(state, now=NOW) == "check CI (stopped)"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda s: None, "armed"),
        (lambda s: setattr(s, "deferred", True), "waiting"),
        (lambda s: setattr(s, "stopped", True), "stopped"),
        (lambda s: setattr(s, "iterations", 999), "finished"),
    ],
)
def test_loop_state_label(mutate, expected):
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    mutate(state)
    assert loop_state_label(state, now=NOW) == expected


def test_format_loop_status_covers_the_useful_fields():
    state = LoopState.create(
        parse_schedule_text("5m"), "check CI", now=NOW, max_iterations=4, fresh=True, expires_seconds=3600
    )
    state.mark_fired(NOW)
    state.advance_after_completion(NOW)
    state.tokens_spent = 1234
    text = format_loop_status(state, now=NOW + timedelta(minutes=1))
    assert "Loop (armed): check CI" in text
    assert "Schedule: every 5m" in text
    assert "Iterations: 1/4" in text
    assert "Next iteration: in 4m" in text
    assert "Expires in 59m" in text
    assert "Prompt source: the command." in text
    assert "fresh conversation thread" in text
    assert "Tokens spent: 1,234" in text


def test_format_loop_status_notes_a_loop_md_prompt():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW, prompt_source=PROMPT_SOURCE_LOOP_MD)
    assert "Prompt source: the .kolega/loop.md file." in format_loop_status(state, now=NOW)


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------


def test_iteration_prompt_carries_the_counter_and_the_body():
    prompt = build_loop_iteration_prompt("check CI", iteration=2, max_iterations=5)
    assert prompt.startswith("[Scheduled loop iteration 2 of 5]")
    assert prompt.endswith("check CI")


def test_iteration_prompt_mentions_a_fresh_thread_only_after_the_first():
    first = build_loop_iteration_prompt("check CI", iteration=1, max_iterations=5, fresh=True)
    later = build_loop_iteration_prompt("check CI", iteration=2, max_iterations=5, fresh=True)
    assert "fresh conversation thread" not in first
    assert "fresh conversation thread" in later


def test_prompt_extension_sets_unattended_expectations():
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=NOW)
    markdown = build_loop_prompt_extension_markdown(state)
    assert "## Scheduled loop" in markdown
    assert "every 5m" in markdown
    assert "Do not ask questions" in markdown
    assert "read-only" in markdown


# ----------------------------------------------------------------------
# .kolega/loop.md
# ----------------------------------------------------------------------


def test_read_loop_md_returns_none_when_absent(tmp_path):
    assert read_loop_md(tmp_path) is None


def test_read_loop_md_plain_body(tmp_path):
    write_loop_md(tmp_path, "check the deploy and report\n")
    loop_md = read_loop_md(tmp_path)
    assert loop_md is not None
    assert loop_md.schedule_text is None
    assert loop_md.prompt == "check the deploy and report"
    assert loop_md.truncated is False


def test_read_loop_md_schedule_header(tmp_path):
    write_loop_md(tmp_path, "schedule: 15m\n\ncheck the deploy\n")
    loop_md = read_loop_md(tmp_path)
    assert loop_md is not None
    assert loop_md.schedule_text == "15m"
    assert loop_md.prompt == "check the deploy"


def test_read_loop_md_allows_a_heading_before_the_header(tmp_path):
    write_loop_md(tmp_path, '# Project loop\n\nschedule: "0 9 * * 1-5"\n\nsummarize open PRs\n')
    loop_md = read_loop_md(tmp_path)
    assert loop_md is not None
    assert loop_md.schedule_text == "0 9 * * 1-5"
    assert loop_md.prompt == "summarize open PRs"


def test_read_loop_md_header_is_case_insensitive(tmp_path):
    write_loop_md(tmp_path, "Schedule:   2h  \n\nrun the audit\n")
    loop_md = read_loop_md(tmp_path)
    assert loop_md is not None
    assert loop_md.schedule_text == "2h"


def test_read_loop_md_keeps_a_multiline_body(tmp_path):
    write_loop_md(tmp_path, "schedule: 5m\n\nstep one\n\nstep two\n")
    loop_md = read_loop_md(tmp_path)
    assert loop_md is not None
    assert loop_md.prompt == "step one\n\nstep two"


def test_read_loop_md_truncates_oversize_content(tmp_path):
    write_loop_md(tmp_path, "x" * (LOOP_MD_MAX_BYTES + 500))
    loop_md = read_loop_md(tmp_path)
    assert loop_md is not None
    assert loop_md.truncated is True
    assert len(loop_md.prompt) == LOOP_MD_MAX_BYTES


def test_read_loop_md_rejects_an_empty_body(tmp_path):
    write_loop_md(tmp_path, "schedule: 5m\n\n   \n")
    with pytest.raises(LoopError):
        read_loop_md(tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs elevation on Windows")
def test_read_loop_md_refuses_a_symlinked_file(tmp_path):
    real = tmp_path / "elsewhere.md"
    real.write_text("do something\n", encoding="utf-8")
    link = tmp_path / LOOP_MD_RELATIVE_PATH
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    with pytest.raises(LoopError) as excinfo:
        read_loop_md(tmp_path)
    assert "symlink" in str(excinfo.value)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs elevation on Windows")
def test_read_loop_md_refuses_a_symlinked_directory(tmp_path):
    real_dir = tmp_path / "elsewhere"
    real_dir.mkdir()
    (real_dir / "loop.md").write_text("do something\n", encoding="utf-8")
    (tmp_path / ".kolega").symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(LoopError):
        read_loop_md(tmp_path)


def test_now_local_is_naive_and_second_resolution():
    moment = loop_module.now_local()
    assert moment.tzinfo is None
    assert moment.microsecond == 0
