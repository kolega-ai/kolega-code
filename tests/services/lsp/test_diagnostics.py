"""Unit tests for LSP diagnostic formatting."""

from kolega_code.services.lsp.client import LspDiagnostic
from kolega_code.services.lsp.diagnostics import (
    MissingServer,
    extract_lsp_label,
    format_detected_summary,
    format_diagnostics,
    format_missing_prompt,
    format_no_diagnostics,
    severity_counts,
    summary_label,
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
    """Diagnostics are formatted with a clean header and severity summary."""
    diags = [
        make_diag(42, "'foo' is not defined", severity=1, source="pyright"),
        make_diag(15, "Unused import 'os'", severity=2),
    ]
    result = format_diagnostics(diags, "test.py", source="pyright")
    assert "Line 42" in result
    assert "'foo' is not defined" in result
    assert "Line 15" in result
    assert "Unused import 'os'" in result
    # Clean plain-text header (no markdown) carrying the severity summary.
    assert "LSP diagnostics" in result
    assert "###" not in result
    assert "1 error, 1 warning" in result
    assert "pyright" in result
    # No leading newline: the caller adds the separator it wants.
    assert not result.startswith("\n")


def test_format_diagnostics_caps_at_max():
    """Diagnostics are capped at max_diags."""
    diags = [make_diag(i, f"Issue {i}") for i in range(100)]
    result = format_diagnostics(diags, "test.py", max_diags=20)
    assert "and 80 more" in result
    assert result.count("Line ") <= 20


def test_format_diagnostics_empty():
    """Empty diagnostics produce empty string."""
    result = format_diagnostics([], "test.py")
    assert result == ""


def test_severity_counts():
    """severity_counts tallies diagnostics by LSP severity."""
    diags = [
        make_diag(1, "e1", severity=1),
        make_diag(2, "e2", severity=1),
        make_diag(3, "w1", severity=2),
        make_diag(4, "i1", severity=3),
        make_diag(5, "h1", severity=4),
        make_diag(6, "unknown", severity=99),
    ]
    counts = severity_counts(diags)
    assert counts == {"total": 6, "errors": 2, "warnings": 1, "infos": 1, "hints": 1}


def test_summary_label_pluralization_and_mix():
    """summary_label pluralizes and joins mixed severities."""
    assert summary_label({"total": 1, "errors": 1, "warnings": 0, "infos": 0, "hints": 0}) == "1 error"
    assert summary_label({"total": 2, "errors": 0, "warnings": 2, "infos": 0, "hints": 0}) == "2 warnings"
    assert summary_label({"total": 3, "errors": 1, "warnings": 2, "infos": 0, "hints": 0}) == "1 error, 2 warnings"
    # Infos/hints collapse into "notes".
    assert summary_label({"total": 1, "errors": 0, "warnings": 0, "infos": 1, "hints": 0}) == "1 note"
    assert summary_label({"total": 2, "errors": 0, "warnings": 0, "infos": 1, "hints": 1}) == "2 notes"
    # No recognized severities -> generic total.
    assert summary_label({"total": 3, "errors": 0, "warnings": 0, "infos": 0, "hints": 0}) == "3 diagnostics"


def test_extract_lsp_label_round_trips_format_diagnostics():
    """extract_lsp_label recovers the summary from format_diagnostics output."""
    diags = [
        make_diag(5, "boom", severity=1),
        make_diag(6, "careful", severity=2),
        make_diag(7, "careful too", severity=2),
    ]
    text = "Edited foo.py\n\n" + format_diagnostics(diags, "foo.py", source="pyright")
    assert extract_lsp_label(text) == "1 error, 2 warnings"


def test_extract_lsp_label_absent():
    """extract_lsp_label returns None when there is no diagnostics block."""
    assert extract_lsp_label("Edited foo.py") is None
    assert extract_lsp_label(format_no_diagnostics()) is None
    assert extract_lsp_label("") is None


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
