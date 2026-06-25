from kolega_code.agent.prompt_dump import dump_prompt_overrides, list_prompt_overrides, prompt_dump_contents
from kolega_code.agent.prompt_overrides import PROMPT_OVERRIDE_DIR


def test_dump_prompt_overrides_creates_all_uppercase_files(tmp_path):
    result = dump_prompt_overrides(tmp_path)

    assert result.errors == []
    assert len(result.written) == 6
    expected = {"CODER.md", "PLANNING.md", "GENERAL.md", "INVESTIGATION.md", "BROWSER.md", "COMPACTION.md"}
    assert {path.name for path in result.written} == expected
    for filename in expected:
        assert (tmp_path / PROMPT_OVERRIDE_DIR / filename).is_file()

    assert "powerful AI coding assistant" in (tmp_path / PROMPT_OVERRIDE_DIR / "CODER.md").read_text(encoding="utf-8")
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


def test_prompt_dump_contents_do_not_include_dynamic_project_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Do not bake into dump", encoding="utf-8")
    (tmp_path / "AGENT_MEMORY.md").write_text("Do not bake memory", encoding="utf-8")

    contents = prompt_dump_contents(tmp_path)

    assert all("Do not bake into dump" not in text for text in contents.values())
    assert all("Do not bake memory" not in text for text in contents.values())


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
