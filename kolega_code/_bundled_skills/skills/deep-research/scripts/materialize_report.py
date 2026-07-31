#!/usr/bin/env python3
"""Materialize a deep-research workflow result as a validated Markdown report.

This module owns the authoritative deterministic report logic: Markdown link
extraction, URL identity, ``## Sources`` construction, and structural
validation. The bundled workflow keeps a mirrored in-sandbox copy for audit
hints only, because a Gigacode script has no filesystem access. The regression
suite asserts both implementations agree on a shared fixture corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Query parameters that never identify a distinct document.
TRACKING_PARAMS = frozenset({"gclid", "fbclid", "ref", "ref_src", "s", "share"})

# Bounds on delivered length relative to the requested word target.
SHORTFALL_RATIO = 0.8
OVERLENGTH_RATIO = 1.35

# Internal identifiers that must never reach a reader-facing gaps section.
_INTERNAL_ID = re.compile(r"\b(?:lane-\d+|followup-\d+|escalation-\d+)(?:/[A-Za-z]+\d+)?\b")
_BARE_RECORD_ID = re.compile(r"\b[CES]\d{1,3}\b")
_INTERNAL_MARKERS = ("lane", "scout", "verifier", "claim_id", "evidence_id", "downstream")

# A reader-facing note is a short disclosure, not a research ledger.
MAX_READER_FACING_GAPS = 10

# Content-word overlap above which two gaps are treated as the same disclosure.
# Scouts and verifiers routinely report one gap in different words.
NEAR_DUPLICATE_OVERLAP = 0.6
# ...but boilerplate alone must never merge two distinct gaps, so also require a
# floor of shared distinctive words.
NEAR_DUPLICATE_MIN_SHARED = 4
# Duplicated gaps almost always open on the same subject ("Agrippa's table of
# Saturn...", "Ficino's chapters on engraved images..."), which distinguishes them
# from two unrelated gaps that merely share stock phrasing.
GAP_SUBJECT_WORDS = 6
GAP_SUBJECT_MIN_SHARED = 4

# Vocabulary common to almost every sourcing gap; useless for telling two apart.
_GAP_BOILERPLATE = frozenset(
    {
        "about",
        "accessible",
        "also",
        "available",
        "because",
        "been",
        "checked",
        "claim",
        "claims",
        "consulted",
        "could",
        "directly",
        "established",
        "evidence",
        "failed",
        "from",
        "have",
        "here",
        "inaccessible",
        "into",
        "known",
        "located",
        "never",
        "obtained",
        "only",
        "read",
        "reached",
        "recovered",
        "remain",
        "remains",
        "report",
        "rests",
        "scholarship",
        "secured",
        "settled",
        "source",
        "sources",
        "still",
        "than",
        "that",
        "their",
        "them",
        "there",
        "then",
        "this",
        "unverified",
        "verified",
        "were",
        "which",
        "with",
        "would",
    }
)


class MaterializationError(ValueError):
    """Raised when a workflow result cannot be materialized safely."""


# ---------------------------------------------------------------------------
# Deterministic text helpers (mirrored by scripts/deep-research.workflow)
# ---------------------------------------------------------------------------
def canonical_url(value: Any) -> str:
    """Return a comparison key that preserves meaningful query parameters.

    Drops the fragment and known tracking parameters, lowercases the scheme and
    host, sorts the surviving query parameters, and strips one trailing slash.
    Two URLs differing only in a meaningful query parameter stay distinct.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.split("#", 1)[0]
    scheme = ""
    rest = text
    marker = text.find("://")
    if marker >= 0:
        scheme = text[:marker].lower()
        rest = text[marker + 3 :]
    query = ""
    question = rest.find("?")
    if question >= 0:
        query = rest[question + 1 :]
        rest = rest[:question]
    slash = rest.find("/")
    if slash >= 0:
        host = rest[:slash].lower()
        path = rest[slash:]
    else:
        host = rest.lower()
        path = ""
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    kept = []
    for part in query.split("&"):
        if not part:
            continue
        key = part.split("=", 1)[0].lower()
        if key in TRACKING_PARAMS or key.startswith("utm_"):
            continue
        kept.append(part)
    base = f"{scheme}://" if scheme else ""
    base += host + path
    if kept:
        return base + "?" + "&".join(sorted(kept))
    return base


def strip_fenced_code(text: Any) -> str:
    """Blank out fenced code blocks so their contents cannot look like citations."""
    kept: list[str] = []
    in_fence = False
    fence = ""
    for line in str(text or "").split("\n"):
        stripped = line.strip()
        marker = ""
        if stripped.startswith("```"):
            marker = "```"
        elif stripped.startswith("~~~"):
            marker = "~~~"
        if marker:
            if in_fence:
                if marker == fence:
                    in_fence = False
                    fence = ""
            else:
                in_fence = True
                fence = marker
            kept.append("")
            continue
        kept.append("" if in_fence else line)
    return "\n".join(kept)


def strip_inline_code(text: Any) -> str:
    """Drop inline code spans so `[a](b)` is not read as a citation."""
    source = str(text or "")
    length = len(source)
    kept: list[str] = []
    index = 0
    while index < length:
        if source[index] == "`":
            run = 0
            while index + run < length and source[index + run] == "`":
                run += 1
            closing = source.find("`" * run, index + run)
            if closing < 0:
                index += run
                continue
            index = closing + run
            continue
        kept.append(source[index])
        index += 1
    return "".join(kept)


def matching_bracket(text: str, start: int) -> int:
    """Index of the ``]`` closing the ``[`` at ``start``, or -1."""
    depth = 0
    index = start
    length = len(text)
    while index < length:
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def link_destination(text: str, start: int) -> tuple[str, int]:
    """Parse a Markdown link destination, returning (destination, closing index)."""
    length = len(text)
    index = start
    while index < length and text[index] in " \t\n":
        index += 1
    destination = ""
    if index < length and text[index] == "<":
        end = text.find(">", index + 1)
        if end < 0:
            return "", -1
        destination = text[index + 1 : end]
        index = end + 1
    else:
        depth = 0
        chars: list[str] = []
        while index < length:
            char = text[index]
            if char in " \t\n":
                break
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            chars.append(char)
            index += 1
        destination = "".join(chars)
    while index < length and text[index] in " \t\n":
        index += 1
    if index < length and text[index] in ('"', "'", "("):
        closer = ")" if text[index] == "(" else text[index]
        end = text.find(closer, index + 1)
        if end < 0:
            return "", -1
        index = end + 1
        while index < length and text[index] in " \t\n":
            index += 1
    if index < length and text[index] == ")":
        return destination.strip(), index
    return "", -1


def markdown_urls(markdown: Any) -> list[str]:
    """Extract http(s) destinations from Markdown inline links, in order.

    Skips fenced blocks, inline code, and images; honors balanced parentheses in
    the destination and an optional link title; ignores non-http destinations so
    anchors and mail links are not reported as unknown citations.
    """
    text = strip_inline_code(strip_fenced_code(markdown))
    length = len(text)
    urls: list[str] = []
    index = 0
    while index < length:
        if text[index] != "[":
            index += 1
            continue
        is_image = index > 0 and text[index - 1] == "!"
        label_end = matching_bracket(text, index)
        if label_end < 0 or label_end + 1 >= length or text[label_end + 1] != "(":
            index += 1
            continue
        destination, close = link_destination(text, label_end + 2)
        if close < 0:
            index += 1
            continue
        if not is_image and destination:
            lowered = destination.lower()
            if lowered.startswith(("http://", "https://")) and destination not in urls:
                urls.append(destination)
        index = close + 1
    return urls


def strip_sources_section(markdown: Any) -> str:
    """Return the body with any trailing ``## Sources`` section and title removed."""
    lines = str(markdown or "").strip().splitlines()
    kept: list[str] = []
    for line in lines:
        if line.strip().lower() == "## sources":
            break
        kept.append(line)
    if kept and kept[0].startswith("# "):
        kept = kept[1:]
    return "\n".join(kept).strip()


def structural_issues(report: str) -> list[str]:
    """Structural defects that make a report unfit to deliver."""
    issues: list[str] = []
    text = str(report or "").strip()
    if not text.startswith("# "):
        issues.append("missing level-one title")
    if text.count("\n## Sources\n") != 1:
        issues.append("report must contain exactly one `## Sources` section")
    lines = text.splitlines()
    seen: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        key = line.strip().lower()
        if key in seen:
            issues.append(f"duplicate heading: {line.strip()}")
        seen.append(key)
        cursor = index + 1
        has_content = False
        while cursor < len(lines) and not lines[cursor].startswith("## "):
            if lines[cursor].strip():
                has_content = True
                break
            cursor += 1
        if not has_content:
            issues.append(f"empty section: {line.strip()}")
    return issues


def source_entry(source: dict[str, Any], fallback_url: str) -> str:
    """One deduplicated bibliography line."""
    title = str(source.get("title") or "").strip() or fallback_url
    url = str(source.get("url") or "").strip() or fallback_url
    detail = str(source.get("publisher") or source.get("source_type") or "").strip()
    date = str(source.get("date") or "").strip()
    if date:
        detail = f"{detail}, {date}".strip(", ")
    suffix = f" — {detail}" if detail else ""
    return f"- [{title}]({url}){suffix}"


def assemble_report(
    title: str, body: str, registry: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Build the final report and its ``## Sources`` section from the body's links."""
    clean_title = str(title or "").strip().lstrip("#").strip()
    clean_body = strip_sources_section(body)
    by_url = {canonical_url(source.get("url")): source for source in registry}
    by_url.pop("", None)

    cited_keys: list[str] = []
    unknown: list[str] = []
    for url in markdown_urls(clean_body):
        key = canonical_url(url)
        if key in by_url:
            if key not in cited_keys:
                cited_keys.append(key)
        elif url not in unknown:
            unknown.append(url)

    cited_sources = [by_url[key] for key in cited_keys]
    lines = [source_entry(by_url[key], key) for key in cited_keys]
    report = f"# {clean_title}\n\n{clean_body}\n\n## Sources\n\n" + "\n".join(lines)
    report = report.strip() + "\n"

    issues: list[str] = []
    if not clean_title:
        issues.append("report has no title")
    if not clean_body:
        issues.append("report body is empty")
    if not cited_sources:
        issues.append("report body cites no source from the workflow registry")
    if unknown:
        issues.append(f"citations absent from the source registry: {', '.join(unknown)}")
    return report, cited_sources, issues


def _distinctive_words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z']+", text.lower())
        if len(word) > 3 and word not in _GAP_BOILERPLATE
    ]


def _gap_signature(text: str) -> set[str]:
    """Distinctive words used to recognize the same gap phrased two ways."""
    return set(_distinctive_words(text))


def _gap_subject(text: str) -> set[str]:
    """The distinctive words that open a gap, i.e. what it is about."""
    return set(_distinctive_words(text)[:GAP_SUBJECT_WORDS])


def _near_duplicate_index(
    signature: set[str],
    subject: set[str],
    existing: list[tuple[set[str], set[str]]],
) -> int | None:
    """Index of an already-kept gap that says substantially the same thing.

    Two gaps merge when they open on the same subject, or when their distinctive
    vocabulary overlaps heavily. The bar is deliberately conservative: showing one
    disclosure twice is a cosmetic wart, whereas merging two distinct gaps hides a
    limitation from the reader.
    """
    if len(signature) < NEAR_DUPLICATE_MIN_SHARED:
        return None
    for index, (other_signature, other_subject) in enumerate(existing):
        if len(subject & other_subject) >= GAP_SUBJECT_MIN_SHARED:
            return index
        if len(other_signature) < NEAR_DUPLICATE_MIN_SHARED:
            continue
        shared = signature & other_signature
        if len(shared) < NEAR_DUPLICATE_MIN_SHARED:
            continue
        if len(shared) / min(len(signature), len(other_signature)) >= NEAR_DUPLICATE_OVERLAP:
            return index
    return None


def reader_facing_gaps(gaps: list[str], limit: int = MAX_READER_FACING_GAPS) -> list[str]:
    """Filter workflow gaps down to disclosures a reader can actually use.

    Workflow gaps are written for the operator and routinely carry internal claim
    and lane identifiers, and the same gap often arrives several times in slightly
    different words. Strip the identifiers, drop entries that are about research
    machinery rather than evidence, collapse near-duplicates to their fullest
    wording, and cap the list so a report ends with a short disclosure instead of a
    research ledger.
    """
    cleaned: list[str] = []
    fingerprints: list[tuple[set[str], set[str]]] = []
    for gap in gaps:
        text = str(gap).strip()
        if not text:
            continue
        text = _INTERNAL_ID.sub("", text)
        text = _BARE_RECORD_ID.sub("", text)
        text = re.sub(r"\(\s*[,;'\"]*\s*\)", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ,;:.—-")
        if len(text) < 20:
            continue
        lowered = text.lower()
        if any(marker in lowered for marker in _INTERNAL_MARKERS):
            continue
        if text[0].islower():
            text = text[0].upper() + text[1:]
        if not text.endswith("."):
            text += "."
        signature = _gap_signature(text)
        subject = _gap_subject(text)
        duplicate = _near_duplicate_index(signature, subject, fingerprints)
        if duplicate is not None:
            # Keep whichever phrasing tells the reader more.
            if len(text) > len(cleaned[duplicate]):
                cleaned[duplicate] = text
                fingerprints[duplicate] = (signature, subject)
            continue
        fingerprints.append((signature, subject))
        cleaned.append(text)

    kept = cleaned[:limit]
    overflow = len(cleaned) - len(kept)
    if overflow > 0:
        kept.append(
            f"{overflow} further sourcing gaps are recorded in the research notes for this report."
        )
    return kept


def insert_gaps_section(report: str, gaps: list[str]) -> str:
    """Insert a ``## Scope and gaps`` section ahead of ``## Sources``."""
    entries = reader_facing_gaps(gaps)
    if not entries:
        return report
    block = "## Scope and gaps\n\nThis report is a supported partial result. The "
    block += "following points remain unresolved:\n\n"
    block += "\n".join(f"- {entry}" for entry in entries)
    marker = "\n## Sources\n"
    if marker in report:
        head, tail = report.split(marker, 1)
        return f"{head}\n\n{block}\n{marker}{tail}"
    return f"{report.rstrip()}\n\n{block}\n"


# ---------------------------------------------------------------------------
# Workflow result loading
# ---------------------------------------------------------------------------
def _decode_json(text: str, source: Path) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise MaterializationError(f"{source}: invalid JSON: {exc}") from exc


def _extract_json_from_markdown(text: str, source: Path) -> Any:
    heading_index = text.find("## Full return value")
    search_from = heading_index if heading_index >= 0 else 0
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text[search_from:], re.DOTALL)
    if not match:
        raise MaterializationError(f"{source}: could not find a fenced JSON workflow return value")
    return _decode_json(match.group(1), source)


def load_workflow_result(path: Path) -> dict[str, Any]:
    """Load a persisted workflow result from JSON or the readable Markdown artifact."""
    path = Path(path)
    if not path.is_file():
        raise MaterializationError(f"{path}: workflow result does not exist")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = _decode_json(text, path)
    else:
        payload = _extract_json_from_markdown(text, path)

    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        payload = payload["result"]
    if not isinstance(payload, dict):
        raise MaterializationError(f"{path}: workflow return value must be an object")
    return payload


def _registry(payload: dict[str, Any]) -> list[dict[str, Any]]:
    registry = payload.get("source_registry")
    if isinstance(registry, list):
        return [source for source in registry if isinstance(source, dict)]
    cited = payload.get("cited_sources")
    if isinstance(cited, list):
        return [source for source in cited if isinstance(source, dict)]
    return []


def _word_budget_warning(payload: dict[str, Any], report: str) -> str | None:
    """Warn when the delivered length departs materially from the confirmed target."""
    summary = payload.get("run_summary")
    if not isinstance(summary, dict):
        return None
    target = summary.get("target_words")
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        return None
    words = len(strip_sources_section(report).split())
    if words < int(target * SHORTFALL_RATIO):
        return f"report is short: {words} words against a requested target of about {target}"
    if words > int(target * OVERLENGTH_RATIO):
        return f"report is long: {words} words against a requested target of about {target}"
    return None


def _degraded_stage_warning(payload: dict[str, Any]) -> str | None:
    """Surface stages whose structured output was unusable."""
    summary = payload.get("run_summary")
    if not isinstance(summary, dict):
        return None
    degraded = summary.get("degraded_stages")
    if not isinstance(degraded, list) or not degraded:
        return None
    names = ", ".join(str(stage) for stage in degraded)
    return f"one or more stages returned an unusable record and were skipped: {names}"


def resolve_report(payload: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    """Return (report Markdown, status, cited sources) after structural validation.

    A ``report_plan`` is preferred: its body file is the authoritative text for a
    long report, and the bibliography is built here from the workflow registry.
    """
    status = payload.get("status")
    if status not in {"complete", "partial"}:
        raise MaterializationError(
            f"workflow status is {status!r}; only complete or supported partial "
            "reports can be written"
        )

    plan = payload.get("report_plan")
    if isinstance(plan, dict) and str(plan.get("body_path") or "").strip():
        body_path = Path(str(plan["body_path"]).strip())
        if not body_path.is_file():
            raise MaterializationError(f"{body_path}: report body file does not exist")
        body_text = body_path.read_text(encoding="utf-8")
        if not body_text.strip():
            raise MaterializationError(f"{body_path}: report body file is empty")
        title = str(plan.get("title") or "").strip()
        first_line = body_text.strip().splitlines()[0]
        if not title and first_line.startswith("# "):
            title = first_line[2:].strip()
        report, cited, issues = assemble_report(title, body_text, _registry(payload))
    else:
        raw = payload.get("report_markdown")
        if not isinstance(raw, str) or not raw.strip():
            raise MaterializationError(
                "workflow result has neither a report_plan body file nor a nonempty report_markdown"
            )
        report = raw.strip() + "\n"
        cited = [source for source in payload.get("cited_sources", []) if isinstance(source, dict)]
        issues = []

    if not strip_sources_section(report).strip():
        issues.append("report body is empty")
    issues.extend(structural_issues(report))
    if issues:
        raise MaterializationError("report is not deliverable: " + "; ".join(issues))
    return report, status, cited


def collision_safe_path(path: Path) -> Path:
    """Return path or the first available numbered sibling."""
    path = Path(path)
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def materialize_report(
    result_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
    collision_safe: bool = False,
) -> tuple[Path, str, list[str]]:
    """Validate a workflow result and write its report.

    Returns the destination path, the workflow status, and any advisory warnings.
    """
    if overwrite and collision_safe:
        raise MaterializationError("choose either overwrite or collision-safe output, not both")

    payload = load_workflow_result(result_path)
    report, status, _cited = resolve_report(payload)

    warnings: list[str] = []
    for warning in (_word_budget_warning(payload, report), _degraded_stage_warning(payload)):
        if warning:
            warnings.append(warning)

    if status == "partial":
        gaps = [str(gap) for gap in payload.get("gaps", []) if str(gap).strip()]
        report = insert_gaps_section(report, gaps)

    destination = Path(output_path)
    if destination.exists():
        if overwrite:
            pass
        elif collision_safe:
            destination = collision_safe_path(destination)
        else:
            raise MaterializationError(
                f"{destination}: output exists; use --overwrite or --collision-safe"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    if destination.stat().st_size == 0:
        raise MaterializationError(f"{destination}: report write produced an empty file")
    return destination, status, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_path", type=Path, help="workflow result.md or result.json")
    parser.add_argument("output_path", type=Path, help="destination Markdown report")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file (only with explicit user approval)",
    )
    destination.add_argument(
        "--collision-safe",
        action="store_true",
        help="choose a numbered filename when the destination exists",
    )
    args = parser.parse_args(argv)

    try:
        path, status, warnings = materialize_report(
            args.result_path,
            args.output_path,
            overwrite=args.overwrite,
            collision_safe=args.collision_safe,
        )
    except (OSError, MaterializationError) as exc:
        print(f"Materialization failed: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Materialized {status} report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
