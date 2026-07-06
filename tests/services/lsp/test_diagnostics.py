"""Unit tests for LSP diagnostic formatting."""

from kolega_code.services.lsp.client import LspDiagnostic
from kolega_code.services.lsp.diagnostics import (
    MissingServer,
    format_detected_summary,
    format_diagnostics,
    format_missing_prompt,
    format_no_diagnostics,
)


def make_diag(line: int, message: str, severity: int = 1, source: str = "", code: str = ""):
    """Helper to create an LspDiagnostic with a line number."""
    return LspDiagnostic(
        range={"start": {"line": line - 1, "character": 0}, "end": {"line": line - 1, "character": 99}},
        severity=severity,
        code=code,
        message=message,
        source=source,
    )


def test_format_diagnostics_errors():
    """Error diagnostics are formatted with red circle."""
    diags = [
        make_diag(42, "'foo' is not defined", severity=1, source="pyright"),
        make_diag(15, "Unused import 'os'", severity=2),
    ]
    result = format_diagnostics(diags, "test.py", source="pyright")
    assert "Line 42" in result
    assert "'foo' is not defined" in result
    assert "Line 15" in result
    assert "Unused import 'os'" in result
    assert "### LSP Diagnostics" in result
    assert "pyright" in result


def test_format_diagnostics_caps_at_max():
    """Diagnostics are capped at max_diags."""
    diags = [make_diag(i, f"Issue {i}") for i in range(100)]
    result = format_diagnostics(diags, "test.py", max_diags=20)
    assert "and 80 more diagnostics" in result
    assert result.count("Line ") <= 20


def test_format_diagnostics_empty():
    """Empty diagnostics produce empty string."""
    result = format_diagnostics([], "test.py")
    assert result == ""


def test_format_no_diagnostics():
    """No-diagnostics message is positive."""
    result = format_no_diagnostics()
    assert "✅" in result
    assert "No LSP diagnostics" in result


def test_format_missing_prompt():
    """Missing server prompt includes install commands and alternatives."""
    missing = [
        MissingServer(
            language_id="python",
            display_name="Python",
            detection_reason="pyproject.toml + 42 .py files",
            server_name="pyright",
            server_bin="pyright-langserver",
            install_commands=["pip install pyright"],
            alternatives=["basedpyright", "ruff-lsp"],
        ),
        MissingServer(
            language_id="rust",
            display_name="Rust",
            detection_reason="Cargo.toml + 18 .rs files",
            server_name="rust-analyzer",
            server_bin="rust-analyzer",
            install_commands=["rustup component add rust-analyzer"],
            alternatives=[],
        ),
    ]
    result = format_missing_prompt(missing)
    assert "⚠" in result
    assert "pyright" in result
    assert "pip install pyright" in result
    assert "rust-analyzer" in result
    assert "rustup component add rust-analyzer" in result
    assert "basedpyright" in result
    assert "ruff-lsp" in result
    assert ".kolega/lsp.json" in result


def test_format_missing_prompt_empty():
    """Empty missing list produces empty string."""
    assert format_missing_prompt([]) == ""


def test_format_detected_summary():
    """Detected summary lists all languages."""
    detected = [
        ("python", "Python", "pyproject.toml + 42 .py files"),
        ("rust", "Rust", "Cargo.toml + 18 .rs files"),
    ]
    result = format_detected_summary(detected)
    assert "🔍" in result
    assert "Python" in result
    assert "pyproject.toml" in result
    assert "Rust" in result
    assert "Cargo.toml" in result


def test_format_detected_summary_empty():
    """Empty list produces empty string."""
    assert format_detected_summary([]) == ""
