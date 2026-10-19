"""Pure presentation tests for the persistent background-agent status strip."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from rich.text import Text

from kolega_code.cli import theme
from kolega_code.cli.theme import Color
from kolega_code.cli.tui.background_status import BackgroundStatus

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.fixture(autouse=True)
def unicode_theme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme, "supports_unicode", lambda: True)


def row(**overrides: Any) -> dict[str, Any]:
    return {
        "job_id": "job",
        "completion_id": "completion",
        "agent_type": "investigation",
        "status": "running",
        "delivered": False,
        "label": "research",
        "archived": False,
        "ready": False,
        "settling": False,
        "persistence_error": False,
        **overrides,
    }


def test_from_rows_counts_each_visible_job_once_and_ignores_archived_rows() -> None:
    status = BackgroundStatus.from_rows(
        [
            row(label="running"),
            row(status="queued", label="queued"),
            row(status="completed", ready=True),
            row(status="cancelled", ready=True),
            row(status="failed", ready=True),
            row(status="interrupted", ready=True),
            row(status="completed", settling=True),
            row(status="failed", settling=True, persistence_error=True),
            row(status="completed", delivered=True),
            row(status="failed", delivered=True),
            row(status="failed", archived=True, persistence_error=True),
        ],
        set(),
        delivery_failed=True,
    )
    assert status == BackgroundStatus(
        running=1,
        queued=1,
        ready=2,
        failed=2,
        interrupted=1,
        saving=1,
        storage_failed=1,
        labels=("running", "queued"),
        delivery_failed=True,
    )
    assert status.total == 9
    assert status.visible is True
    assert status.animating is True


@pytest.mark.parametrize(
    ("data", "acknowledged", "expected"),
    [
        (row(status="completed", delivered=True), set(), BackgroundStatus()),
        (row(status="failed", delivered=True), set(), BackgroundStatus(failed=1, labels=("research",))),
        (
            row(status="interrupted", delivered=True),
            set(),
            BackgroundStatus(interrupted=1, labels=("research",)),
        ),
        (row(status="failed", delivered=True), {"completion"}, BackgroundStatus()),
        (row(status="interrupted", delivered=True), {"completion"}, BackgroundStatus()),
        (
            row(status="failed", ready=True),
            {"completion"},
            BackgroundStatus(ready=1, labels=("research",)),
        ),
        (
            row(status="interrupted", ready=True),
            {"completion"},
            BackgroundStatus(ready=1, labels=("research",)),
        ),
        (row(status="cancelled", ready=True), set(), BackgroundStatus(ready=1, labels=("research",))),
        (
            row(status="failed", settling=True),
            set(),
            BackgroundStatus(saving=1, labels=("research",)),
        ),
        (
            row(status="failed", settling=True, persistence_error=True),
            {"completion"},
            BackgroundStatus(storage_failed=1, labels=("research",)),
        ),
    ],
    ids=[
        "delivered-success-hidden",
        "delivered-failure-persists",
        "delivered-interruption-persists",
        "acknowledged-delivered-failure-hidden",
        "acknowledged-delivered-interruption-hidden",
        "acknowledged-pending-failure-ready",
        "acknowledged-pending-interruption-ready",
        "cancelled-result-ready",
        "terminal-write-saving",
        "storage-error-not-acknowledged-away",
    ],
)
def test_terminal_visibility_and_acknowledgment(
    data: dict[str, Any], acknowledged: set[str], expected: BackgroundStatus
) -> None:
    assert BackgroundStatus.from_rows([data], acknowledged) == expected


def test_failure_acknowledgment_uses_completion_id_not_reused_job_id() -> None:
    rows = [
        row(status="failed", delivered=True, completion_id="old-completion", label="old failure"),
        row(status="failed", delivered=True, completion_id="new-completion", label="new failure"),
    ]
    assert BackgroundStatus.from_rows(rows, {"job", "old-completion"}) == BackgroundStatus(
        failed=1, labels=("new failure",)
    )


@pytest.mark.parametrize(
    "data",
    [
        row(),
        row(status="queued"),
        row(status="failed", ready=True),
        row(status="interrupted", delivered=True),
        row(status="completed", settling=True, persistence_error=True),
    ],
)
def test_archived_jobs_never_contribute_activity_or_warnings(data: dict[str, Any]) -> None:
    data["archived"] = True
    status = BackgroundStatus.from_rows([data], set())
    assert status == BackgroundStatus()
    assert status.visible is False
    assert status.animating is False


def test_status_is_immutable_and_does_not_retain_mutable_inputs() -> None:
    rows = [row(status="failed", ready=True, label="original label")]
    acknowledged: set[str] = set()
    status = BackgroundStatus.from_rows(rows, acknowledged)
    rows[0].update(status="completed", label="changed label")
    rows.clear()
    acknowledged.add("completion")
    assert status == BackgroundStatus(failed=1, labels=("original label",))
    assert isinstance(status.labels, tuple)
    with pytest.raises(FrozenInstanceError):
        setattr(status, "failed", 0)


def test_labels_fall_back_to_agent_type_and_only_keep_two_visible_jobs() -> None:
    status = BackgroundStatus.from_rows(
        [
            row(status="completed", delivered=True, label="hidden"),
            row(label=""),
            row(status="queued", label="second task"),
            row(label="third task"),
        ],
        set(),
    )
    assert status == BackgroundStatus(running=2, queued=1, labels=("investigation", "second task"))


@pytest.mark.parametrize(
    ("status", "expected", "color", "animating"),
    [
        (BackgroundStatus(running=1), "⠋ 1 background agent · Ctrl+G", Color.ACCENT, True),
        (BackgroundStatus(queued=2), "⠋ 2 queued agents · Ctrl+G", Color.ACCENT, True),
        (BackgroundStatus(ready=1), "✓ 1 result ready · Ctrl+G", Color.SUCCESS, False),
        (BackgroundStatus(ready=2), "✓ 2 results ready · Ctrl+G", Color.SUCCESS, False),
        (BackgroundStatus(failed=1), "✗ 1 failed agent · Ctrl+G", Color.WARNING, False),
        (BackgroundStatus(interrupted=2), "✗ 2 interrupted agents · Ctrl+G", Color.WARNING, False),
        (
            BackgroundStatus(saving=1),
            "⠋ 1 agent · 1 saving result · Ctrl+G",
            Color.ACCENT,
            True,
        ),
        (
            BackgroundStatus(saving=2),
            "⠋ 2 agents · 2 saving results · Ctrl+G",
            Color.ACCENT,
            True,
        ),
        (
            BackgroundStatus(storage_failed=1),
            "✗ 1 agent · 1 save failed · Ctrl+G",
            Color.WARNING,
            False,
        ),
        (
            BackgroundStatus(delivery_failed=True),
            "✗ 0 agents · delivery failed · Ctrl+G",
            Color.WARNING,
            False,
        ),
    ],
)
def test_render_state_counts_and_semantic_style(
    status: BackgroundStatus, expected: str, color: str, animating: bool
) -> None:
    rendered = status.render("⠋")
    assert isinstance(rendered, Text)
    assert rendered.plain == expected
    assert rendered.style == color
    assert status.visible is True
    assert status.animating is animating


def test_mixed_render_preserves_every_count_and_warning_before_task_labels() -> None:
    status = BackgroundStatus(
        running=2,
        queued=1,
        ready=2,
        failed=1,
        interrupted=1,
        saving=1,
        storage_failed=1,
        delivery_failed=True,
        labels=("research",),
    )
    assert status.render("⠋").plain == (
        "✗ 9 agents · 2 running · 1 queued · 2 results ready · 1 failed · 1 interrupted"
        " · 1 saving result · 1 save failed · delivery failed · research · Ctrl+G"
    )
    assert status.render("⠋").style == Color.WARNING


def test_compact_render_omits_task_labels_but_not_count_or_shortcut() -> None:
    status = BackgroundStatus(running=2, labels=("task one", "task two"))
    assert status.render("⠋", compact=True).plain == "⠋ 2 background agents · Ctrl+G"
    assert status.render("⠋").plain == "⠋ 2 background agents · task one, task two · Ctrl+G"


def test_empty_status_has_no_strip() -> None:
    assert BackgroundStatus().render("⠋").plain == ""


@pytest.mark.parametrize("width", [-1, 0, 1, 8, 10, 13, 17, 24, 32, 48, 72, 120])
@pytest.mark.parametrize(
    "status",
    [
        BackgroundStatus(running=12, queued=3, labels=("界" * 12, "e\u0301" * 12)),
        BackgroundStatus(failed=2, storage_failed=1, delivery_failed=True, labels=("[red]task[/red]",)),
    ],
)
def test_render_stays_single_line_and_within_available_terminal_cells(width: int, status: BackgroundStatus) -> None:
    rendered = status.render("⠋", width=width)
    assert rendered.cell_len <= max(0, width)
    assert "\n" not in rendered.plain
    assert "\r" not in rendered.plain
    if width <= 0:
        assert rendered.plain == ""
    if width >= 13:
        assert "Ctrl+G" in rendered.plain


def test_width_budget_uses_unicode_cells_not_python_string_length() -> None:
    status = BackgroundStatus.from_rows([row(label="界" * 10)], set())
    assert status.render("⠋", width=45).plain == "⠋ 1 background agent · Ctrl+G"
    wide = status.render("⠋", width=64)
    assert wide.plain == "⠋ 1 background agent · " + "界" * 10 + " · Ctrl+G"
    assert wide.cell_len > len(wide.plain)
    assert wide.cell_len <= 64


def test_labels_are_single_line_literal_text_not_rich_markup() -> None:
    status = BackgroundStatus.from_rows([row(label="  [red]界[/red]\n\t e\u0301\r  ")], set())
    assert status.labels == ("[red]界[/red] e\u0301",)
    rendered = status.render("⠋")
    assert rendered.plain == "⠋ 1 background agent · [red]界[/red] e\u0301 · Ctrl+G"
    assert rendered.style == Color.ACCENT
    assert all(span.style == "dim" for span in rendered.spans)


def test_long_unicode_labels_are_bounded_before_rendering() -> None:
    status = BackgroundStatus.from_rows([row(label="界" * 100)], set())
    assert len(status.labels) == 1
    assert Text(status.labels[0]).cell_len <= 28
    assert status.labels[0].endswith("…")
    assert status.render("⠋", width=60).cell_len <= 60


@pytest.mark.parametrize(
    ("status", "width", "expected"),
    [
        (
            BackgroundStatus(running=12, labels=("a long task label",)),
            33,
            "⠋ 12 background agents · Ctrl+G",
        ),
        (BackgroundStatus(running=12, labels=("a long task label",)), 23, "⠋ 12 agents · Ctrl+G"),
        (BackgroundStatus(running=12), 13, "12 · Ctrl+G"),
        (
            BackgroundStatus(running=8, failed=2, labels=("task label",)),
            27,
            "✗ 2 failed · Ctrl+G",
        ),
        (
            BackgroundStatus(running=8, delivery_failed=True, labels=("task label",)),
            29,
            "✗ delivery failed · Ctrl+G",
        ),
        (
            BackgroundStatus(running=8, storage_failed=1, labels=("task label",)),
            27,
            "✗ 1 save failed · Ctrl+G",
        ),
    ],
)
def test_narrow_render_prioritizes_shortcut_counts_and_warnings(
    status: BackgroundStatus, width: int, expected: str
) -> None:
    assert status.render("⠋", width=width).plain == expected


def test_warning_glyph_respects_ascii_terminal_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme, "supports_unicode", lambda: False)
    assert BackgroundStatus(failed=1).render("|").plain == "x 1 failed agent · Ctrl+G"
    assert BackgroundStatus(ready=1).render("|").plain == "ok 1 result ready · Ctrl+G"
