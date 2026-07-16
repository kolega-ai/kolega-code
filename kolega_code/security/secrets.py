"""Deterministic probable-secret detection without entropy heuristics."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

SECRET_PLACEHOLDER = "‹secret›"


@dataclass(frozen=True, slots=True)
class SecretFinding:
    category: str
    start: int
    end: int


_PATTERNS: tuple[tuple[str, re.Pattern[str], str | None], ...] = (
    (
        "private-key",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?"
            r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        None,
    ),
    ("anthropic-token", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"), None),
    ("api-token", re.compile(r"\b(?:sk|xai)-[A-Za-z0-9_-]{8,}"), None),
    ("google-api-key", re.compile(r"\bAIza[A-Za-z0-9_-]{10,}"), None),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), None),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), None),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), None),
    ("npm-token", re.compile(r"\bnpm_[A-Za-z0-9]{20,}"), None),
    (
        "authorization",
        re.compile(r"(?i)authorization\s*:\s*(?:bearer|basic)\s+(?P<value>\S+)"),
        "value",
    ),
    ("api-key-header", re.compile(r"(?i)x-api-key\s*[:=]\s*(?P<value>\S+)"), "value"),
    (
        "credential-assignment",
        re.compile(
            r"(?im)\b[A-Za-z0-9_]*(?:API_?KEY|ACCESS_?KEY|TOKEN|SECRET|"
            r"PASSWORD|PASSWD|PRIVATE_?KEY|CLIENT_?SECRET)[A-Za-z0-9_]*\s*[=:]\s*"
            r"(?P<value>[^\s\"']+|\"[^\"]+\"|'[^']+')"
        ),
        "value",
    ),
    (
        "credential-url",
        re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:(?P<value>[^@\s/]+)@"),
        "value",
    ),
)


def _environment_values() -> list[str]:
    values: list[str] = []
    for name, value in os.environ.items():
        if (
            value
            and len(value) >= 8
            and any(marker in name.upper() for marker in ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD"))
        ):
            values.append(value)
    return values


def detect_secrets(
    text: str, extra_values: Iterable[str] | None = None, *, include_environment: bool = False
) -> tuple[SecretFinding, ...]:
    """Return non-overlapping probable-secret spans and categories."""
    findings: list[SecretFinding] = []
    for category, pattern, group in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span(group) if group else match.span()
            if end > start:
                findings.append(SecretFinding(category, start, end))
    values = list(extra_values or [])
    if include_environment:
        values.extend(_environment_values())
    for value in values:
        if not value or len(value) < 8:
            continue
        start = 0
        while (index := text.find(value, start)) >= 0:
            findings.append(SecretFinding("configured-secret", index, index + len(value)))
            start = index + len(value)
    findings.sort(key=lambda finding: (finding.start, -(finding.end - finding.start), finding.category))
    result: list[SecretFinding] = []
    for finding in findings:
        if result and finding.start < result[-1].end:
            if finding.end <= result[-1].end:
                continue
            finding = SecretFinding(finding.category, result[-1].end, finding.end)
        result.append(finding)
    return tuple(result)


def redact_secrets(
    text: str,
    extra_values: Iterable[str] | None = None,
    *,
    include_environment: bool = False,
) -> str:
    findings = detect_secrets(text, extra_values, include_environment=include_environment)
    for finding in reversed(findings):
        text = text[: finding.start] + SECRET_PLACEHOLDER + text[finding.end :]
    return text
