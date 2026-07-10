"""Unit tests for LSP language auto-detection from project files."""

from pathlib import Path

import pytest

from kolega_code.services.lsp import LspRegistry
from kolega_code.services.lsp.detector import DetectionReport, detect_languages
from kolega_code.services.workspace_scan import ScanOutcome, ScannedPath


@pytest.mark.asyncio
async def test_detect_python_project(tmp_path: Path):
    """A project with pyproject.toml and .py files is detected as Python."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (tmp_path / "main.py").write_text("print('hello')\n")
    (tmp_path / "lib.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("# Test\n")

    registry = LspRegistry()
    report = await detect_languages(tmp_path, registry)

    assert isinstance(report, DetectionReport)
    python_detected = next((d for d in report.detected if d.language_id == "python"), None)
    assert python_detected is not None
    assert "pyproject.toml" in python_detected.detection_reason.lower()


@pytest.mark.asyncio
async def test_detect_rust_project(tmp_path: Path):
    """A project with Cargo.toml and .rs files is detected as Rust."""
    (tmp_path / "Cargo.toml").write_text("[package]\nname='test'\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text("fn main() {}\n")

    registry = LspRegistry()
    report = await detect_languages(tmp_path, registry)

    rust_detected = next((d for d in report.detected if d.language_id == "rust"), None)
    assert rust_detected is not None
    assert "Cargo.toml" in rust_detected.detection_reason


@pytest.mark.asyncio
async def test_detect_markdown_only(tmp_path: Path):
    """A project with only .md files detects markdown."""
    (tmp_path / "README.md").write_text("# Test\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("content\n")

    registry = LspRegistry()
    report = await detect_languages(tmp_path, registry)

    md_detected = next((d for d in report.detected if d.language_id == "markdown"), None)
    assert md_detected is not None


@pytest.mark.asyncio
async def test_detect_dockerfile(tmp_path: Path):
    """A Dockerfile is detected as docker language."""
    (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")

    registry = LspRegistry()
    report = await detect_languages(tmp_path, registry)

    docker_detected = next((d for d in report.detected if d.language_id == "docker"), None)
    assert docker_detected is not None


@pytest.mark.asyncio
async def test_detect_empty_project(tmp_path: Path):
    """An empty project produces no detected languages."""
    registry = LspRegistry()
    report = await detect_languages(tmp_path, registry)
    assert report.detected == []
    assert report.resolved == []
    assert report.missing == []


@pytest.mark.asyncio
async def test_detect_languages_retains_partial_scan_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def partial_scan(*args, **kwargs):
        return ScanOutcome(
            paths=[ScannedPath("main.py", is_dir=False)],
            visited_entries=50_000,
            elapsed_seconds=5.0,
            complete=False,
            stop_reason="entry_limit",
        )

    monkeypatch.setattr("kolega_code.services.lsp.detector.scan_workspace", partial_scan)
    report = await detect_languages(tmp_path, LspRegistry())

    assert report.scan_complete is False
    assert report.scan_stop_reason == "entry_limit"
    assert report.scanned_entries == 50_000
    assert any(language.language_id == "python" for language in report.detected)
