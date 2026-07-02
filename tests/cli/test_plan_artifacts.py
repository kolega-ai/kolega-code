import os
import stat
from pathlib import Path

from kolega_code.cli.plan_artifacts import (
    CURRENT_PLAN_FILENAME,
    current_plan_artifact_path,
    normalize_plan_artifact_content,
    write_current_plan_artifact,
)


def test_current_plan_artifact_path_is_under_state_dir(tmp_path: Path) -> None:
    path = current_plan_artifact_path(tmp_path / "state", "session-123")

    assert path == tmp_path / "state" / "plans" / "session-123" / CURRENT_PLAN_FILENAME


def test_normalize_plan_artifact_content_strips_and_adds_trailing_newline() -> None:
    assert normalize_plan_artifact_content("\n# Plan\n\nBuild it.  \n") == "# Plan\n\nBuild it.\n"


def test_write_current_plan_artifact_writes_private_markdown(tmp_path: Path) -> None:
    state_root = tmp_path / "state"

    old_umask = os.umask(0)
    try:
        path = write_current_plan_artifact(state_root, "session-123", "\n# Plan\n\nBuild it.\n")
    finally:
        os.umask(old_umask)

    assert path == current_plan_artifact_path(state_root, "session-123")
    assert path.read_text(encoding="utf-8") == "# Plan\n\nBuild it.\n"

    if os.name != "nt":
        assert stat.S_IMODE(state_root.stat().st_mode) == 0o700
        assert stat.S_IMODE((state_root / "plans").stat().st_mode) == 0o700
        assert stat.S_IMODE((state_root / "plans" / "session-123").stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_current_plan_artifact_rewrites_when_plan_changes(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    path = write_current_plan_artifact(state_root, "session-123", "# Plan\n\nFirst.")

    write_current_plan_artifact(state_root, "session-123", "# Plan\n\nSecond.")

    assert path.read_text(encoding="utf-8") == "# Plan\n\nSecond.\n"
