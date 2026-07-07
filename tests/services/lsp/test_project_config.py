"""T3 + F7: ``.kolega/lsp.json`` parsing, merge semantics, and type validation."""

from __future__ import annotations

import json
from pathlib import Path


from kolega_code.services.lsp.config import LspConfig
from kolega_code.services.lsp.manager import _merge_lsp_config
from kolega_code.services.lsp.registry import load_project_lsp_config


def _write_config(project: Path, payload: dict) -> None:
    (project / ".kolega").mkdir(parents=True, exist_ok=True)
    (project / ".kolega" / "lsp.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# load_project_lsp_config
# ---------------------------------------------------------------------------


def test_load_returns_none_when_absent(tmp_path: Path):
    assert load_project_lsp_config(tmp_path) is None


def test_load_returns_only_present_keys(tmp_path: Path):
    """The loader returns only keys present in the file (F5 merge support)."""
    _write_config(tmp_path, {"max_diagnostics": 50, "disabled_languages": ["go"]})
    result = load_project_lsp_config(tmp_path)

    assert result is not None
    assert result == {"max_diagnostics": 50, "disabled_languages": ["go"]}


def test_load_returns_none_for_unparseable(tmp_path: Path):
    """F7: a corrupt file returns None instead of crashing init."""
    (tmp_path / ".kolega").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".kolega" / "lsp.json").write_text("{ not valid json ", encoding="utf-8")
    assert load_project_lsp_config(tmp_path) is None


def test_load_returns_none_for_non_dict(tmp_path: Path):
    _write_config(tmp_path, [])  # type: ignore[arg-type]  # intentionally malformed
    # json.dumps([]) -> "[]"; write directly to keep it a JSON array
    (tmp_path / ".kolega" / "lsp.json").write_text("[]", encoding="utf-8")
    assert load_project_lsp_config(tmp_path) is None


# ---------------------------------------------------------------------------
# F7: type validation (no silent character-split / no crash)
# ---------------------------------------------------------------------------


def test_load_string_for_list_field_is_not_character_split(tmp_path: Path):
    """F7: a string where a list is expected is dropped, not split into chars."""
    _write_config(tmp_path, {"disabled_languages": "python"})
    result = load_project_lsp_config(tmp_path)

    assert result is not None
    # The mistyped field is ignored entirely (not ['p','y','t','h','o','n']).
    assert "disabled_languages" not in result


def test_load_string_for_dict_field_does_not_crash(tmp_path: Path):
    """F7: a string where a dict is expected is dropped, not raised as ValueError."""
    _write_config(tmp_path, {"preferences": "x"})
    result = load_project_lsp_config(tmp_path)

    assert result is not None
    assert "preferences" not in result


def test_load_bool_for_int_field_is_rejected(tmp_path: Path):
    """F7: ``True`` is not accepted as an integer (bool is a subclass of int)."""
    _write_config(tmp_path, {"max_diagnostics": True})
    result = load_project_lsp_config(tmp_path)

    assert result is not None
    assert "max_diagnostics" not in result


def test_load_rejects_path_bearing_server_bin(tmp_path: Path):
    """F1 defense-in-depth: project servers with a path-bearing bin are dropped."""
    _write_config(
        tmp_path,
        {"servers": {"ok": {"bin": "real-server"}, "bad": {"bin": "/bin/sh"}}},
    )
    result = load_project_lsp_config(tmp_path)

    assert result is not None
    assert "ok" in result["servers"]
    assert "bad" not in result["servers"]


# ---------------------------------------------------------------------------
# _merge_lsp_config (F5)
# ---------------------------------------------------------------------------


def test_merge_preserves_base_fields_not_in_overrides():
    """F5: fields absent from the project file keep the base (user) value."""
    base = LspConfig(
        enabled=True,
        max_diagnostics=10,
        preferences={"python": "basedpyright"},
        prompt_on_missing=False,
    )
    overrides = {"max_diagnostics": 50}

    merged = _merge_lsp_config(base, overrides)

    assert merged.max_diagnostics == 50  # overridden
    assert merged.preferences == {"python": "basedpyright"}  # preserved
    assert merged.prompt_on_missing is False  # preserved


def test_merge_always_preserves_user_enabled():
    """F5: the user's kill-switch wins; a project file cannot flip ``enabled``."""
    base = LspConfig(enabled=True)
    merged_off = _merge_lsp_config(base, {"enabled": False})
    assert merged_off.enabled is True

    base_off = LspConfig(enabled=False)
    merged_on = _merge_lsp_config(base_off, {"enabled": True})
    assert merged_on.enabled is False


def test_merge_maps_servers_to_custom_servers():
    """F5: the JSON ``servers`` key maps to the ``custom_servers`` field."""
    base = LspConfig(enabled=True)
    overrides = {"servers": {"my-server": {"bin": "my-server"}}}

    merged = _merge_lsp_config(base, overrides)

    assert merged.custom_servers == {"my-server": {"bin": "my-server"}}


def test_merge_maps_diagnostic_servers():
    """F5: the JSON ``diagnostic_servers`` key maps to the field."""
    base = LspConfig(enabled=True)
    merged = _merge_lsp_config(base, {"diagnostic_servers": ["ruff-lsp"]})

    assert merged.diagnostic_servers == ["ruff-lsp"]


def test_merge_ignores_unknown_keys():
    base = LspConfig(enabled=True, max_diagnostics=10)
    merged = _merge_lsp_config(base, {"unknown_key": "value", "enabled": False})

    assert merged.max_diagnostics == 10
    assert merged.enabled is True  # preserved + unknown key ignored
