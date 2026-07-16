import pytest

from kolega_code.agent.prompt_dump import (
    dump_prompt_overrides,
    format_prompt_validation_result,
    list_prompt_overrides,
    prompt_dump_contents,
    select_prompt_dump_specs,
    validate_prompt_overrides,
)
from kolega_code.agent.prompt_overrides import PROMPT_OVERRIDE_DIR


def test_dump_prompt_overrides_creates_all_uppercase_files(tmp_path):
    result = dump_prompt_overrides(tmp_path)

    assert result.errors == []
    assert len(result.written) == 6
    expected = {"CODER.md", "PLANNING.md", "GENERAL.md", "INVESTIGATION.md", "BROWSER.md", "COMPACTION.md"}
    assert {path.name for path in result.written} == expected
    for filename in expected:
        assert (tmp_path / PROMPT_OVERRIDE_DIR / filename).is_file()

    coder_text = (tmp_path / PROMPT_OVERRIDE_DIR / "CODER.md").read_text(encoding="utf-8")
    assert "powerful AI coding assistant" in coder_text
    assert "- Working directory: {{ context.project_path }}" in coder_text
    assert "- Is directory a git repo: {{ context.is_git_repo }}" in coder_text
    assert "- Platform: {{ context.platform }}" in coder_text
    assert "- Today's date: {{ context.date_today }}" in coder_text
    assert "- Model: {{ context.model_name }}" in coder_text
    assert str(tmp_path) not in coder_text
    assert "continuity briefing" in (tmp_path / PROMPT_OVERRIDE_DIR / "COMPACTION.md").read_text(encoding="utf-8")


def test_dump_prompt_overrides_skips_existing_by_default_and_force_overwrites(tmp_path):
    prompt_dir = tmp_path / PROMPT_OVERRIDE_DIR
    prompt_dir.mkdir(parents=True)
    coder = prompt_dir / "CODER.md"
    coder.write_text("custom", encoding="utf-8")

    result = dump_prompt_overrides(tmp_path)

    assert coder in result.skipped
    assert coder.read_text(encoding="utf-8") == "custom"

    forced = dump_prompt_overrides(tmp_path, force=True)

    assert coder in forced.written
    assert "powerful AI coding assistant" in coder.read_text(encoding="utf-8")


def test_prompt_dump_contents_use_placeholders_for_agent_environment(tmp_path):
    contents = prompt_dump_contents(tmp_path)

    for filename in ["CODER.md", "PLANNING.md", "GENERAL.md", "INVESTIGATION.md", "BROWSER.md"]:
        text = contents[filename]
        assert "{{ context.project_path }}" in text
        assert "{{ context.is_git_repo }}" in text
        assert "{{ context.platform }}" in text
        assert "{{ context.date_today }}" in text
        assert "{{ context.model_name }}" in text
        assert str(tmp_path) not in text


def test_prompt_dump_contents_do_not_include_dynamic_project_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Do not bake into dump", encoding="utf-8")

    contents = prompt_dump_contents(tmp_path)

    assert all("Do not bake into dump" not in text for text in contents.values())


def test_list_prompt_overrides_reports_existing_and_missing(tmp_path):
    prompt_dir = tmp_path / PROMPT_OVERRIDE_DIR
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text("custom", encoding="utf-8")

    result = list_prompt_overrides(tmp_path)

    existing = {item.path.name for item in result.existing}
    missing = {item.path.name for item in result.missing}
    assert existing == {"CODER.md"}
    assert "COMPACTION.md" in missing
    assert len(result.files) == 6


def test_select_prompt_dump_specs_accepts_keys_and_filename_aliases():
    specs = select_prompt_dump_specs(["CODER.md", "coder", "PLANNING", "planning.md"])

    assert [spec.key for spec in specs] == ["coder", "planning"]


def test_select_prompt_dump_specs_rejects_unknown_selector():
    with pytest.raises(ValueError, match="Unknown prompt selector: nope"):
        select_prompt_dump_specs(["nope"])


def test_dump_prompt_overrides_can_write_only_selected_files(tmp_path):
    result = dump_prompt_overrides(tmp_path, selectors=["coder", "compaction"])

    assert result.errors == []
    assert {path.name for path in result.written} == {"CODER.md", "COMPACTION.md"}
    assert (tmp_path / PROMPT_OVERRIDE_DIR / "CODER.md").is_file()
    assert (tmp_path / PROMPT_OVERRIDE_DIR / "COMPACTION.md").is_file()
    assert not (tmp_path / PROMPT_OVERRIDE_DIR / "PLANNING.md").exists()


def test_dump_prompt_overrides_force_only_overwrites_selected_files(tmp_path):
    prompt_dir = tmp_path / PROMPT_OVERRIDE_DIR
    prompt_dir.mkdir(parents=True)
    coder = prompt_dir / "CODER.md"
    compaction = prompt_dir / "COMPACTION.md"
    coder.write_text("custom coder", encoding="utf-8")
    compaction.write_text("custom compaction", encoding="utf-8")

    result = dump_prompt_overrides(tmp_path, force=True, selectors=["compaction"])

    assert result.errors == []
    assert result.written == [compaction]
    assert coder.read_text(encoding="utf-8") == "custom coder"
    assert "continuity briefing" in compaction.read_text(encoding="utf-8")


def test_validate_prompt_overrides_reports_nothing_to_validate(tmp_path):
    result = validate_prompt_overrides(tmp_path)

    assert result.ok is True
    assert result.checked == ()
    assert "nothing to validate" in format_prompt_validation_result(result)


def test_validate_prompt_overrides_reports_valid_existing_overrides(tmp_path):
    prompt_dir = tmp_path / PROMPT_OVERRIDE_DIR
    prompt_dir.mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("Project guidance", encoding="utf-8")
    (prompt_dir / "CODER.md").write_text(
        "{{ context.project_guidance }}\n{{ context.project_path }}",
        encoding="utf-8",
    )

    result = validate_prompt_overrides(tmp_path)

    assert result.ok is True
    assert [item.path.name for item in result.checked] == ["CODER.md"]
    assert "All existing prompt overrides are valid" in format_prompt_validation_result(result)


def test_validate_prompt_overrides_flags_removed_agent_memory_field(tmp_path):
    """Overrides referencing the removed legacy field are reported by name."""
    prompt_dir = tmp_path / PROMPT_OVERRIDE_DIR
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text("{{ context.agent_memory }}", encoding="utf-8")

    result = validate_prompt_overrides(tmp_path)

    assert result.ok is False
    assert any("agent_memory" in diagnostic.message for diagnostic in result.diagnostics)


def test_validate_prompt_overrides_reports_startup_style_errors(tmp_path):
    prompt_dir = tmp_path / PROMPT_OVERRIDE_DIR
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "GENERAL.md").write_text("{{ missing_variable }}", encoding="utf-8")

    result = validate_prompt_overrides(tmp_path)

    assert result.ok is False
    assert len(result.diagnostics) == 1
    formatted = format_prompt_validation_result(result)
    assert "Could not render prompt override .kolega/prompts/GENERAL.md" in formatted
    assert "missing_variable" in formatted
    assert "Falling back to the default prompt" in formatted
