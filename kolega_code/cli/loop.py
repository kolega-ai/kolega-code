"""Schedule parsing, state, and prompt builders for the ``/loop`` command.

``/loop`` re-runs a prompt on a schedule — a fixed interval or a 5-field cron
expression — inside the current session.  This is the shared CLI layer,
importable by both the Textual TUI mixin (:mod:`kolega_code.cli.tui.loop_runtime`)
and the non-interactive ``ask --loop`` path in :mod:`kolega_code.cli.main`.  It
mirrors :mod:`kolega_code.cli.goal`: no Textual imports, standard library plus
:mod:`kolega_code.cli.cron` and the microcopy in :mod:`kolega_code.cli.messages`
only, so it stays cheap to import and easy to test.

All times are naive **local wall-clock** ``datetime`` values produced by
:func:`now_local`, which is the single clock seam the tests monkeypatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from . import messages
from .cron import CronError, CronSchedule

#: Safety backstop on how many iterations one loop may run.
DEFAULT_LOOP_MAX_ITERATIONS = 100

#: A loop stops this long after it was created, even if it is under the cap.
DEFAULT_LOOP_EXPIRY_DAYS = 7

#: Floor on interval schedules. The scheduler ticks once a second, so sub-minute
#: intervals work, but anything faster than this is a runaway-spend hazard.
MIN_INTERVAL_SECONDS = 15

#: Intervals below this get a one-line advisory (not a block) when the loop starts.
SUB_MINUTE_ADVISORY_SECONDS = 60

#: Project-scoped default prompt file, consulted only on an explicit ``/loop``.
LOOP_MD_RELATIVE_PATH = Path(".kolega") / "loop.md"
LOOP_MD_MAX_BYTES = 25_000

LOOP_STOP_ALIASES: frozenset[str] = frozenset({"stop", "clear", "off", "cancel", "none", "reset"})
LOOP_STATUS_ALIASES: frozenset[str] = frozenset({"status", "show", "info"})

#: How often the TUI scheduler checks whether a loop is due.
LOOP_TICK_SECONDS = 1.0

#: Prompt preview length in the one-line status-dashboard summary.
LOOP_SUMMARY_MAX_CHARS = 48

PROMPT_SOURCE_INLINE = "inline"
PROMPT_SOURCE_LOOP_MD = "loop_md"

SCHEDULE_KIND_INTERVAL = "interval"
SCHEDULE_KIND_CRON = "cron"

_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}

_DURATION_RE = re.compile(r"^(\d+)\s*([a-z]+)$")
_SCHEDULE_HEADER_RE = re.compile(r"^\s*schedule\s*:\s*(.+?)\s*$", re.IGNORECASE)


class LoopError(ValueError):
    """Raised when a loop schedule, option, or ``loop.md`` file is invalid."""


def now_local() -> datetime:
    """Current naive local wall-clock time, truncated to the second.

    The single clock seam for the whole loop feature: monkeypatch this in tests
    to drive the scheduler deterministically.
    """
    return datetime.now().replace(microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    # Loop timestamps are naive local; drop any tzinfo a hand-edited session carried.
    return parsed.replace(tzinfo=None)


# ----------------------------------------------------------------------
# Durations
# ----------------------------------------------------------------------


def format_duration_short(seconds: float) -> str:
    """Compact duration label: ``45s``, ``5m``, ``1h 30m``, ``2d``."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs and not days and not hours:
        parts.append(f"{secs}s")
    return " ".join(parts) or "0s"


def format_countdown(seconds: float) -> str:
    """Coarse "time until" label.

    Seconds below a minute, whole minutes above.  Deliberately low-resolution:
    the status dashboard only re-renders when this string changes, and Textual
    reflow costs scale with the number of mounted widgets.
    """
    remaining = int(seconds)
    if remaining <= 0:
        return "now"
    if remaining < 60:
        return f"{remaining}s"
    return format_duration_short((remaining // 60) * 60)


def parse_duration(text: str) -> int:
    """Parse ``30s`` / ``5m`` / ``2 hours`` / ``every 2 hours`` into seconds."""
    candidate = " ".join(str(text or "").strip().lower().split())
    if candidate.startswith("every "):
        candidate = candidate[len("every ") :].strip()
    match = _DURATION_RE.match(candidate)
    if match is None:
        raise LoopError(messages.LOOP_BAD_DURATION.format(value=text))
    amount, unit = match.groups()
    if unit not in _UNIT_SECONDS:
        raise LoopError(messages.LOOP_BAD_DURATION.format(value=text))
    seconds = int(amount) * _UNIT_SECONDS[unit]
    if seconds <= 0:
        raise LoopError(messages.LOOP_BAD_DURATION.format(value=text))
    return seconds


def _try_duration(text: str) -> Optional[int]:
    """Parse a duration, returning ``None`` when ``text`` is not one at all.

    A value that parses but is too short still returns its seconds so the caller
    can raise a precise "interval too short" error instead of silently treating
    the token as the start of the prompt.
    """
    try:
        return parse_duration(text)
    except LoopError:
        return None


def parse_interval(text: str) -> int:
    """Parse an interval and enforce :data:`MIN_INTERVAL_SECONDS`."""
    seconds = parse_duration(text)
    if seconds < MIN_INTERVAL_SECONDS:
        raise LoopError(messages.LOOP_INTERVAL_TOO_SHORT.format(minimum=MIN_INTERVAL_SECONDS))
    return seconds


# ----------------------------------------------------------------------
# Schedules
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LoopSchedule:
    """Either a fixed interval or a cron expression. Exactly one is set."""

    interval_seconds: Optional[int] = None
    cron: Optional[CronSchedule] = None

    def __post_init__(self) -> None:
        if (self.interval_seconds is None) == (self.cron is None):
            raise LoopError("A loop schedule must be exactly one of an interval or a cron expression.")

    @property
    def kind(self) -> str:
        return SCHEDULE_KIND_INTERVAL if self.interval_seconds is not None else SCHEDULE_KIND_CRON

    @property
    def value(self) -> str:
        if self.interval_seconds is not None:
            return str(self.interval_seconds)
        assert self.cron is not None
        return self.cron.expression

    @property
    def first_fire_is_immediate(self) -> bool:
        """Interval loops run right away; cron loops wait for the first match."""
        return self.interval_seconds is not None

    def label(self) -> str:
        if self.interval_seconds is not None:
            return f"every {format_duration_short(self.interval_seconds)}"
        assert self.cron is not None
        return self.cron.describe()

    def next_after(self, completed_at: datetime) -> datetime:
        """Next fire time.

        Interval schedules measure the delay from when the previous iteration
        *finished*, so a slow iteration never queues an immediately-due fire.
        Cron schedules use the wall clock.
        """
        if self.interval_seconds is not None:
            return completed_at + timedelta(seconds=self.interval_seconds)
        assert self.cron is not None
        return self.cron.next_fire_after(completed_at)


def interval_schedule(seconds: int) -> LoopSchedule:
    """Build an interval schedule, enforcing the minimum interval."""
    if seconds < MIN_INTERVAL_SECONDS:
        raise LoopError(messages.LOOP_INTERVAL_TOO_SHORT.format(minimum=MIN_INTERVAL_SECONDS))
    return LoopSchedule(interval_seconds=int(seconds))


def parse_schedule_text(text: str) -> LoopSchedule:
    """Parse a free-form schedule: an interval first, otherwise a cron expression."""
    candidate = str(text or "").strip().strip('"').strip()
    if not candidate:
        raise LoopError(messages.LOOP_SCHEDULE_EMPTY)
    seconds = _try_duration(candidate)
    if seconds is not None:
        return interval_schedule(seconds)
    try:
        return LoopSchedule(cron=CronSchedule.parse(candidate))
    except CronError as exc:
        raise LoopError(str(exc)) from exc


@lru_cache(maxsize=64)
def _schedule_for(kind: str, value: str) -> LoopSchedule:
    if kind == SCHEDULE_KIND_INTERVAL:
        return LoopSchedule(interval_seconds=int(value))
    return LoopSchedule(cron=CronSchedule.parse(value))


# ----------------------------------------------------------------------
# Command parsing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LoopSpec:
    """A validated ``/loop`` start request.

    ``schedule`` is ``None`` when the user gave none (fall back to the
    ``schedule:`` header in ``loop.md``); ``prompt`` is empty when the user gave
    none (fall back to the ``loop.md`` body).
    """

    schedule: Optional[LoopSchedule] = None
    prompt: str = ""
    fresh: bool = False
    max_iterations: int = DEFAULT_LOOP_MAX_ITERATIONS
    expires_seconds: int = DEFAULT_LOOP_EXPIRY_DAYS * 86400


@dataclass(frozen=True)
class LoopCommand:
    """Parsed ``/loop`` invocation: ``start``, ``status``, ``stop``, or ``usage``."""

    action: str
    spec: Optional[LoopSpec] = None
    reason: str = ""


def _take_token(text: str) -> tuple[str, str]:
    stripped = text.lstrip()
    if not stripped:
        return "", ""
    match = re.match(r"\S+", stripped)
    assert match is not None
    token = match.group(0)
    return token, stripped[len(token) :].lstrip()


def _take_quoted(text: str) -> tuple[Optional[str], str]:
    stripped = text.lstrip()
    if not stripped.startswith('"'):
        return None, text
    closing = stripped.find('"', 1)
    if closing == -1:
        return None, text
    return stripped[1:closing], stripped[closing + 1 :].lstrip()


def _take_argument(text: str) -> tuple[str, str]:
    quoted, rest = _take_quoted(text)
    if quoted is not None:
        return quoted, rest
    return _take_token(text)


def _take_schedule(text: str) -> tuple[Optional[LoopSchedule], str]:
    """Consume a leading schedule if there is one, else leave the text untouched."""
    quoted, rest = _take_quoted(text)
    if quoted is not None:
        return parse_schedule_text(quoted), rest

    token, after_token = _take_token(text)
    if not token:
        return None, text

    if token.lower() == "every":
        second, after_second = _take_token(after_token)
        third, after_third = _take_token(after_second)
        for candidate, remainder in ((f"{second} {third}", after_third), (second, after_second)):
            seconds = _try_duration(candidate)
            if seconds is not None:
                return interval_schedule(seconds), remainder
        return None, text

    seconds = _try_duration(token)
    if seconds is not None:
        return interval_schedule(seconds), after_token
    return None, text


def parse_loop_command(args: str) -> LoopCommand:
    """Parse the argument string of a ``/loop`` invocation.

    Grammar: leading flags (``--fresh``, ``--max-iterations N``, ``--expires D``,
    ``--cron "expr"``) in any order, then an optional schedule (a compact
    interval, ``every <n> <unit>``, or a double-quoted cron expression), then the
    rest of the line as the prompt.
    """
    text = str(args or "").strip()
    lowered = text.lower()
    if lowered in LOOP_STOP_ALIASES:
        return LoopCommand(action="stop")
    if lowered in LOOP_STATUS_ALIASES:
        return LoopCommand(action="status")

    fresh = False
    max_iterations = DEFAULT_LOOP_MAX_ITERATIONS
    expires_seconds = DEFAULT_LOOP_EXPIRY_DAYS * 86400
    cron_expression: Optional[str] = None
    rest = text

    try:
        while True:
            token, remainder = _take_token(rest)
            flag = token.lower()
            if flag == "--fresh":
                fresh = True
                rest = remainder
                continue
            if flag in ("--max-iterations", "--max-iters"):
                value, after_value = _take_argument(remainder)
                if not value.isdigit() or int(value) < 1:
                    return LoopCommand(action="usage", reason=messages.LOOP_BAD_MAX_ITERATIONS)
                max_iterations = int(value)
                rest = after_value
                continue
            if flag == "--expires":
                value, after_value = _take_argument(remainder)
                expires_seconds = parse_duration(value)
                rest = after_value
                continue
            if flag == "--cron":
                value, after_value = _take_argument(remainder)
                if not value:
                    return LoopCommand(action="usage", reason=messages.LOOP_SCHEDULE_EMPTY)
                cron_expression = value
                rest = after_value
                continue
            if flag.startswith("--"):
                return LoopCommand(action="usage", reason=messages.LOOP_UNKNOWN_OPTION.format(option=token))
            break

        if cron_expression is not None:
            schedule: Optional[LoopSchedule] = LoopSchedule(cron=CronSchedule.parse(cron_expression))
        else:
            schedule, rest = _take_schedule(rest)
    except (LoopError, CronError) as exc:
        return LoopCommand(action="usage", reason=str(exc))

    return LoopCommand(
        action="start",
        spec=LoopSpec(
            schedule=schedule,
            prompt=rest.strip(),
            fresh=fresh,
            max_iterations=max_iterations,
            expires_seconds=expires_seconds,
        ),
    )


# ----------------------------------------------------------------------
# .kolega/loop.md
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LoopMd:
    """Contents of ``.kolega/loop.md``."""

    schedule_text: Optional[str]
    prompt: str
    truncated: bool = False


def read_loop_md(project_path: Path) -> Optional[LoopMd]:
    """Read ``<project>/.kolega/loop.md``.

    Returns ``None`` when the file does not exist.  Raises :class:`LoopError`
    when the file or its parent directory is a symlink (a loop prompt drives
    autonomous turns, so it must not be redirectable) or when it holds no prompt.
    """
    path = Path(project_path) / LOOP_MD_RELATIVE_PATH
    if path.is_symlink() or path.parent.is_symlink():
        raise LoopError(messages.LOOP_MD_SYMLINK.format(path=LOOP_MD_RELATIVE_PATH.as_posix()))
    if not path.is_file():
        return None

    data = path.read_bytes()
    truncated = len(data) > LOOP_MD_MAX_BYTES
    if truncated:
        data = data[:LOOP_MD_MAX_BYTES]
    lines = data.decode("utf-8", errors="replace").splitlines()

    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    # Allow a single Markdown heading before the schedule header.
    if index < len(lines) and lines[index].lstrip().startswith("#"):
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1

    schedule_text: Optional[str] = None
    if index < len(lines):
        match = _SCHEDULE_HEADER_RE.match(lines[index])
        if match is not None:
            schedule_text = match.group(1).strip().strip('"').strip()
            index += 1

    prompt = "\n".join(lines[index:]).strip()
    if not prompt:
        raise LoopError(messages.LOOP_MD_EMPTY.format(path=LOOP_MD_RELATIVE_PATH.as_posix()))
    return LoopMd(schedule_text=schedule_text, prompt=prompt, truncated=truncated)


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------


@dataclass
class LoopState:
    """Persistent state of one scheduled loop.

    Serialized to/from a dict for session persistence; unknown or missing keys
    are tolerated so older sessions load.
    """

    schedule_kind: str
    schedule_value: str
    prompt: str
    prompt_source: str = PROMPT_SOURCE_INLINE
    fresh: bool = False
    created_at: str = ""
    next_fire_at: str = ""
    last_fired_at: Optional[str] = None
    last_completed_at: Optional[str] = None
    iterations: int = 0
    max_iterations: int = DEFAULT_LOOP_MAX_ITERATIONS
    expires_at: str = ""
    tokens_spent: int = 0
    stopped: bool = False
    deferred: bool = False
    status_note: str = ""

    # -- serialization -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_kind": self.schedule_kind,
            "schedule_value": self.schedule_value,
            "prompt": self.prompt,
            "prompt_source": self.prompt_source,
            "fresh": self.fresh,
            "created_at": self.created_at,
            "next_fire_at": self.next_fire_at,
            "last_fired_at": self.last_fired_at,
            "last_completed_at": self.last_completed_at,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "expires_at": self.expires_at,
            "tokens_spent": self.tokens_spent,
            "stopped": self.stopped,
            "deferred": self.deferred,
            "status_note": self.status_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopState":
        return cls(
            schedule_kind=str(data.get("schedule_kind") or SCHEDULE_KIND_INTERVAL),
            schedule_value=str(data.get("schedule_value") or MIN_INTERVAL_SECONDS),
            prompt=str(data.get("prompt") or ""),
            prompt_source=str(data.get("prompt_source") or PROMPT_SOURCE_INLINE),
            fresh=bool(data.get("fresh", False)),
            created_at=str(data.get("created_at") or ""),
            next_fire_at=str(data.get("next_fire_at") or ""),
            last_fired_at=data.get("last_fired_at"),
            last_completed_at=data.get("last_completed_at"),
            iterations=int(data.get("iterations") or 0),
            max_iterations=int(data.get("max_iterations") or DEFAULT_LOOP_MAX_ITERATIONS),
            expires_at=str(data.get("expires_at") or ""),
            tokens_spent=int(data.get("tokens_spent") or 0),
            stopped=bool(data.get("stopped", False)),
            deferred=bool(data.get("deferred", False)),
            status_note=str(data.get("status_note") or ""),
        )

    @classmethod
    def create(
        cls,
        schedule: LoopSchedule,
        prompt: str,
        *,
        prompt_source: str = PROMPT_SOURCE_INLINE,
        fresh: bool = False,
        max_iterations: int = DEFAULT_LOOP_MAX_ITERATIONS,
        expires_seconds: int = DEFAULT_LOOP_EXPIRY_DAYS * 86400,
        now: Optional[datetime] = None,
    ) -> "LoopState":
        moment = now or now_local()
        first_fire = moment if schedule.first_fire_is_immediate else schedule.next_after(moment)
        return cls(
            schedule_kind=schedule.kind,
            schedule_value=schedule.value,
            prompt=prompt.strip(),
            prompt_source=prompt_source,
            fresh=fresh,
            created_at=_iso(moment),
            next_fire_at=_iso(first_fire),
            max_iterations=max_iterations,
            expires_at=_iso(moment + timedelta(seconds=expires_seconds)),
        )

    # -- derived state -------------------------------------------------

    @property
    def schedule(self) -> LoopSchedule:
        return _schedule_for(self.schedule_kind, self.schedule_value)

    def schedule_label(self) -> str:
        try:
            return self.schedule.label()
        except (LoopError, CronError, ValueError):
            return f"{self.schedule_kind} {self.schedule_value}"

    @property
    def reached_cap(self) -> bool:
        return self.iterations >= self.max_iterations

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        expires = _parse_iso(self.expires_at)
        if expires is None:
            return False
        return (now or now_local()) >= expires

    def is_active_at(self, now: Optional[datetime] = None) -> bool:
        """Whether the loop should still fire as of ``now``.

        Rendering helpers must use this rather than :attr:`is_active` so a
        caller-supplied clock is honored instead of silently falling back to the
        wall clock.
        """
        return not self.stopped and not self.reached_cap and not self.is_expired(now)

    @property
    def is_active(self) -> bool:
        """A loop that should still fire: not stopped, under the cap, unexpired."""
        return self.is_active_at()

    def seconds_until(self, now: Optional[datetime] = None) -> float:
        target = _parse_iso(self.next_fire_at)
        if target is None:
            return 0.0
        return (target - (now or now_local())).total_seconds()

    def is_due(self, now: Optional[datetime] = None) -> bool:
        return self.seconds_until(now) <= 0

    def advance_after_completion(self, completed_at: Optional[datetime] = None) -> None:
        """Record a finished iteration and arm the next fire."""
        moment = completed_at or now_local()
        self.last_completed_at = _iso(moment)
        try:
            self.next_fire_at = _iso(self.schedule.next_after(moment))
        except (LoopError, CronError, ValueError):
            # An unparseable persisted schedule cannot be advanced; stop rather
            # than spin. The caller surfaces ``status_note``.
            self.stopped = True
            self.status_note = messages.LOOP_SCHEDULE_UNREADABLE

    def mark_fired(self, fired_at: Optional[datetime] = None) -> None:
        self.iterations += 1
        self.last_fired_at = _iso(fired_at or now_local())
        self.deferred = False


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _prompt_preview(prompt: str, limit: int = LOOP_SUMMARY_MAX_CHARS) -> str:
    collapsed = " ".join(prompt.split()) or "(empty prompt)"
    if len(collapsed) > limit:
        return collapsed[: limit - 1].rstrip() + "…"
    return collapsed


def loop_state_label(state: LoopState, *, now: Optional[datetime] = None) -> str:
    """Short lifecycle word for status output."""
    if state.stopped:
        return "stopped"
    if state.reached_cap:
        return "finished"
    if state.is_expired(now):
        return "expired"
    if state.deferred:
        return "waiting"
    return "armed"


def loop_status_summary(state: LoopState, *, now: Optional[datetime] = None) -> str:
    """One-line summary for the status dashboard.

    Deliberately coarse: the dashboard is only re-rendered when this string
    changes, so the countdown must not tick every second above a minute.
    """
    preview = _prompt_preview(state.prompt)
    if not state.is_active_at(now):
        return f"{preview} ({loop_state_label(state, now=now)})"
    counter = f"{state.iterations}/{state.max_iterations}"
    if state.deferred:
        return f"{preview} ({state.schedule_label()} · waiting for idle · {counter})"
    countdown = format_countdown(state.seconds_until(now))
    return f"{preview} ({state.schedule_label()} · next {countdown} · {counter})"


def format_loop_status(state: LoopState, *, now: Optional[datetime] = None) -> str:
    """Render the ``/loop status`` block as plain text."""
    moment = now or now_local()
    lines = [
        f"Loop ({loop_state_label(state, now=moment)}): {' '.join(state.prompt.split()) or '(empty prompt)'}",
        f"Schedule: {state.schedule_label()}  |  Iterations: {state.iterations}/{state.max_iterations}",
    ]
    if state.is_active_at(moment):
        if state.deferred:
            lines.append("Next iteration: waiting for the current work to finish.")
        else:
            lines.append(f"Next iteration: in {format_countdown(state.seconds_until(moment))} ({state.next_fire_at}).")
    expires = _parse_iso(state.expires_at)
    if expires is not None and not state.stopped:
        if expires <= moment:
            lines.append(f"Expired at {state.expires_at}.")
        else:
            lines.append(f"Expires in {format_countdown((expires - moment).total_seconds())} ({state.expires_at}).")
    source = "the .kolega/loop.md file" if state.prompt_source == PROMPT_SOURCE_LOOP_MD else "the command"
    lines.append(f"Prompt source: {source}.")
    if state.fresh:
        lines.append("Each iteration after the first starts from a fresh conversation thread.")
    if state.last_completed_at:
        lines.append(f"Last iteration finished at {state.last_completed_at}.")
    if state.tokens_spent:
        lines.append(f"Tokens spent: {state.tokens_spent:,}")
    if state.status_note:
        lines.append(f"Status: {state.status_note}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------


def build_loop_iteration_prompt(prompt: str, *, iteration: int, max_iterations: int, fresh: bool = False) -> str:
    """The message sent to the agent for one scheduled iteration."""
    header = f"[Scheduled loop iteration {iteration} of {max_iterations}]"
    if fresh and iteration > 1:
        header += (
            " This iteration starts from a fresh conversation thread, so read the "
            "repository, Git history, and any notes on disk for prior context."
        )
    return f"{header}\n\n{prompt.strip()}"


def build_loop_prompt_extension_markdown(state: LoopState) -> str:
    """Body for the ``cli-active-loop`` system-prompt extension."""
    return (
        "## Scheduled loop\n\n"
        "This turn was started automatically by a scheduled loop "
        f"({state.schedule_label()}), not by someone typing a message. Assume the "
        "user is away and will read a transcript later.\n\n"
        "- Be concise and lead with the answer: what changed, what is broken, what you did.\n"
        "- Do not ask questions. Nobody may be present to answer, and an unanswered "
        "question wastes the whole iteration.\n"
        "- Prefer read-only checks. If an action would need approval you cannot get, "
        "report what you would do instead of blocking on a prompt.\n"
        "- Do not start large refactors or sweeping changes unless the loop prompt "
        "explicitly asks for them. Keep each iteration small and self-contained.\n"
        "- If there is nothing to do this iteration, say so briefly rather than "
        "inventing work."
    )
