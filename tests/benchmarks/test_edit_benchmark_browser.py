import json
from pathlib import Path

from benchmarks.edit_tools.browser import build_corpus_browser
from benchmarks.edit_tools.models import FileContent, SuiteSpec, TaskSpec


def test_static_browser_contains_task_data_diff_and_assets(tmp_path: Path) -> None:
    task = TaskSpec(
        id="browser-example",
        prompt="Replace `old` with `new`; preserve the literal </script> text.",
        before_files={"src/example.py": FileContent(text="value = 'old'\n")},
        expected_files={"src/example.py": FileContent(text="value = 'new'\n")},
        language="python",
        family="localized-replacement",
        difficulty="easy",
        target_length="short",
        primary_target="src/example.py",
        tags=["browser-test"],
    )
    suite = SuiteSpec(id="browser-suite", description="Browser fixture", curated_tasks=[task])

    index = build_corpus_browser(tmp_path / "site", suite, [task], package_root=tmp_path)

    assert index == tmp_path / "site" / "index.html"
    assert index.is_file()
    assert (index.parent / "styles.css").is_file()
    assert (index.parent / "app.js").is_file()
    data_source = (index.parent / "data.js").read_text(encoding="utf-8")
    assert "</script>" not in data_source
    prefix = "window.BENCHMARK_DATA="
    payload = json.loads(data_source.removeprefix(prefix).removesuffix(";\n"))
    assert payload["suite"] == {"id": "browser-suite", "description": "Browser fixture"}
    assert payload["summary"]["tasks"] == 1
    assert payload["tasks"][0]["id"] == "browser-example"
    assert payload["tasks"][0]["files"][0]["path"] == "src/example.py"
    assert "-value = 'old'" in payload["tasks"][0]["files"][0]["diff"]
    assert "+value = 'new'" in payload["tasks"][0]["files"][0]["diff"]


def test_browser_assets_expose_search_filters_and_source_views() -> None:
    root = Path(__file__).resolve().parents[2] / "benchmarks" / "edit_tools" / "browser_assets"
    html = (root / "index.html").read_text(encoding="utf-8")
    javascript = (root / "app.js").read_text(encoding="utf-8")

    assert 'id="task-search"' in html
    assert 'id="language-filter"' in html
    assert 'id="task-detail"' in html
    assert "Kolega evaluation lab" not in html
    assert "brand-mark" not in html
    assert '["diff", "before", "expected"]' in javascript
    assert "task.snapshot.repository" in javascript
