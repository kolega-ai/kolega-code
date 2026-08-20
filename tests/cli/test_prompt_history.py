import json
from pathlib import Path

from kolega_code.cli.prompt_history import (
    PROMPT_HISTORY_MAX,
    load_prompt_history,
    prompt_history_path,
    save_prompt_history,
)


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_prompt_history(tmp_path) == []


def test_save_load_roundtrip_newest_last(tmp_path: Path) -> None:
    save_prompt_history(tmp_path, ["one", "two", "three"])
    assert load_prompt_history(tmp_path) == ["one", "two", "three"]


def test_save_caps_at_limit(tmp_path: Path) -> None:
    entries = [f"prompt {index}" for index in range(PROMPT_HISTORY_MAX + 20)]
    save_prompt_history(tmp_path, entries)
    loaded = load_prompt_history(tmp_path)
    assert len(loaded) == PROMPT_HISTORY_MAX
    assert loaded[0] == "prompt 20"
    assert loaded[-1] == f"prompt {PROMPT_HISTORY_MAX + 19}"


def test_save_empty_list_writes_no_file(tmp_path: Path) -> None:
    save_prompt_history(tmp_path, [])
    assert not prompt_history_path(tmp_path).exists()


def test_load_corrupt_json_returns_empty(tmp_path: Path) -> None:
    path = prompt_history_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_prompt_history(tmp_path) == []


def test_load_filters_non_list_payload_and_non_string_entries(tmp_path: Path) -> None:
    path = prompt_history_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nope": True}), encoding="utf-8")
    assert load_prompt_history(tmp_path) == []
    path.write_text(json.dumps(["ok", 42, None, "fine"]), encoding="utf-8")
    assert load_prompt_history(tmp_path) == ["ok", "fine"]
