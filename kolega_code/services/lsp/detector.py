"""Auto-detection of project languages from config files and file extensions.

Scans a project root to determine which languages are in use, then resolves
language servers via the ``LspRegistry``.
"""

from __future__ import annotations

import logging
import asyncio
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from kolega_code.services.workspace_scan import ScanLimits, scan_workspace

from .registry import LspRegistry

logger = logging.getLogger(__name__)

# Directories to skip during the extension survey.
_SKIP_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "bower_components",
    "vendor",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    "target",
    "out",
    ".next",
    ".nuxt",
    ".cache",
    "coverage",
    ".coverage",
    ".idea",
    ".vscode",
    ".fleet",
}

# Threshold: minimum files of a language's extensions needed to consider it
# when no config file was found.
_MIN_FILES_THRESHOLD = 5
# When the project has zero config file hits, lower the threshold to 1
# (a single-file project should still get LSP).
_MIN_FILES_NO_CONFIG = 1


@dataclass
class DetectionResult:
    """The outcome of auto-detection for one language."""

    language_id: str
    display_name: str
    config_files_found: list[str]  # config files that were detected
    file_count: int  # number of files with matching extensions
    detection_reason: str  # human-readable summary, e.g. "pyproject.toml + 42 .py files"


@dataclass
class ResolvedLanguage:
    """A detected language with its resolved server (or missing info)."""

    language_id: str
    display_name: str
    detection_reason: str
    server_name: str
    server_bin: str | None  # absolute path if resolved, None if missing
    server_args: list[str]
    install_commands: list[str]
    alternatives: list[str]  # other server names available
    family: Optional[str] = None  # if set, reuse this language's server
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables from the matched ``LanguageServerSpec``."""
    initialization_options: dict = field(default_factory=dict)
    """Server-specific initialization options (from config or spec)."""


@dataclass
class DetectionReport:
    """Full auto-detection report for a project."""

    detected: list[DetectionResult] = field(default_factory=list)
    resolved: list[ResolvedLanguage] = field(default_factory=list)
    missing: list[ResolvedLanguage] = field(default_factory=list)
    scan_complete: bool = True
    scan_stop_reason: Optional[str] = None
    scanned_entries: int = 0
    scan_elapsed_seconds: float = 0.0


async def detect_languages(project_path: str | Path, registry: LspRegistry) -> DetectionReport:
    """Auto-detect languages used in *project_path* and resolve their servers.

    The detection happens in two phases:

    1. Scan project root for well-known config files (strong signal).
    2. Recursively count file extensions (recall signal).

    Languages with config-file hits are always included. Languages with many
    matching extension files are included as candidates. All detected languages
    are then resolved against the registry: servers found on PATH are marked as
    ``resolved``, others as ``missing``.
    """
    root = Path(project_path).resolve()
    all_languages = registry.languages

    # A single bounded traversal supplies both shallow config signals and the
    # recursive extension survey. This avoids a separate unbounded root probe.
    scan = await scan_workspace(
        root,
        pattern="**/*",
        include_files=True,
        include_directories=False,
        exclude_directories=frozenset(_SKIP_DIRS),
        skip_hidden_directories=True,
        collect_metadata=False,
        limits=ScanLimits(timeout_seconds=5.0, max_entries=50_000),
    )
    scanned_paths = [scanned.path for scanned in scan.paths]
    config_hits, filename_hits = await asyncio.to_thread(
        _detect_config_signals,
        scanned_paths,
        all_languages,
    )

    # Phase B: extension survey
    ext_counts: dict[str, int] = {}
    for scanned in scan.paths:
        suffix = Path(scanned.path).suffix.lower()
        if suffix:
            ext_counts[suffix] = ext_counts.get(suffix, 0) + 1

    # Phase C: merge signals
    detected: list[DetectionResult] = []
    has_any_config = bool(config_hits) or bool(filename_hits)

    for lang_id, spec in all_languages.items():
        cf = config_hits.get(lang_id, [])
        ff = filename_hits.get(lang_id, [])
        fc = 0
        for ext in spec.extensions:
            fc += ext_counts.get(ext.lower(), 0)

        if cf or ff:
            # Strong signal: always include
            reason_parts = []
            if cf:
                reason_parts.append(", ".join(cf))
            if ff:
                reason_parts.append(", ".join(ff))
            if fc > 0:
                reason_parts.append(f"{fc} {spec.extensions[0] if spec.extensions else ''} files")
            detected.append(
                DetectionResult(
                    language_id=lang_id,
                    display_name=spec.display_name,
                    config_files_found=cf + ff,
                    file_count=fc,
                    detection_reason=" + ".join(reason_parts),
                )
            )
        elif fc >= (_MIN_FILES_NO_CONFIG if not has_any_config else _MIN_FILES_THRESHOLD):
            # Extension-only signal
            detected.append(
                DetectionResult(
                    language_id=lang_id,
                    display_name=spec.display_name,
                    config_files_found=[],
                    file_count=fc,
                    detection_reason=f"{fc} {spec.extensions[0] if spec.extensions else ''} files",
                )
            )

    # Resolve language servers
    resolved: list[ResolvedLanguage] = []
    missing: list[ResolvedLanguage] = []

    for dr in detected:
        lang_id = dr.language_id
        spec = registry.get(lang_id)
        if not spec or not spec.language_servers:
            continue

        resolved_bin, matched, all_candidates = registry.resolve_server(
            lang_id,
            auto_fallback=getattr(registry.config, "auto_fallback", True),
        )

        alternatives = [s.name for s in all_candidates if matched is None or s.name != matched.name]
        first = all_candidates[0] if all_candidates else None

        chosen = matched or first
        rl = ResolvedLanguage(
            language_id=lang_id,
            display_name=dr.display_name,
            detection_reason=dr.detection_reason,
            server_name=matched.name if matched else (first.name if first else "unknown"),
            server_bin=resolved_bin,
            server_args=list(matched.args) if matched else (list(first.args) if first else []),
            install_commands=registry.install_commands_for_platform(chosen) if chosen is not None else [],
            alternatives=alternatives,
            family=spec.family,
            env=dict(chosen.env) if chosen is not None else {},
            initialization_options=registry.initialization_options_for(chosen.name) if chosen is not None else {},
        )

        if resolved_bin:
            resolved.append(rl)
        else:
            missing.append(rl)

    return DetectionReport(
        detected=detected,
        resolved=resolved,
        missing=missing,
        scan_complete=scan.complete,
        scan_stop_reason=scan.stop_reason,
        scanned_entries=scan.visited_entries,
        scan_elapsed_seconds=scan.elapsed_seconds,
    )


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _detect_config_signals(
    scanned_paths: list[str], all_languages: dict[str, Any]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Derive root/one-level signals from the already bounded path snapshot."""
    shallow = [path for path in (Path(raw_path) for raw_path in scanned_paths) if len(path.parts) <= 2]
    root_names = {path.name for path in shallow if len(path.parts) == 1}

    config_hits: dict[str, list[str]] = {}
    filename_hits: dict[str, list[str]] = {}
    for lang_id, spec in all_languages.items():
        configs: list[str] = []
        for pattern in spec.config_files:
            for path in shallow:
                if fnmatch.fnmatchcase(path.name, pattern):
                    configs.append(path.as_posix())
        if configs:
            config_hits[lang_id] = sorted(dict.fromkeys(configs))

        filenames = [fname for fname in spec.filename_map if fname in root_names]
        if filenames:
            filename_hits[lang_id] = filenames

    return config_hits, filename_hits


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
