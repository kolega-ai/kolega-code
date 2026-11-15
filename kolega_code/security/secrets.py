"""Deterministic probable-secret detection without entropy heuristics."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

SECRET_PLACEHOLDER = "‹secret›"


@dataclass(frozen=True, slots=True)
class SecretFinding:
    category: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _PatternSpec:
    """One detector plus the literal every match of it must contain.

    The literal is checked before the regex runs: most text contains none of
    them, and a failed ``in`` test costs a fraction of a full regex pass over a
    tool result or exported session payload.
    """

    category: str
    pattern: re.Pattern[str]
    group: str | None = None
    #: Every match contains one of these literals. ``folded`` literals are
    #: lowercase and compared against ``text.casefold()``, which is what an
    #: ``(?i)`` pattern can match; the others are compared case-sensitively.
    needles: tuple[str, ...] = ()
    folded: bool = False
    #: Starts can be located by finding ``needles`` in an ASCII-lowercased copy
    #: instead of letting the engine retry a keyword alternation at every
    #: character of every word.
    keyword_prefixed: bool = False


_CREDENTIAL_KEYWORDS = (
    "apikey",
    "api_key",
    "accesskey",
    "access_key",
    "token",
    "secret",
    "password",
    "passwd",
    "privatekey",
    "private_key",
    "clientsecret",
    "client_secret",
)

_PATTERNS: tuple[_PatternSpec, ...] = (
    _PatternSpec(
        "private-key",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?"
            r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        needles=("-----BEGIN ",),
    ),
    _PatternSpec("anthropic-token", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"), needles=("sk-ant-",)),
    _PatternSpec("api-token", re.compile(r"\b(?:sk|xai)-[A-Za-z0-9_-]{8,}"), needles=("sk-", "xai-")),
    _PatternSpec("google-api-key", re.compile(r"\bAIza[A-Za-z0-9_-]{10,}"), needles=("AIza",)),
    _PatternSpec(
        "github-token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
        needles=("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
    ),
    _PatternSpec("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), needles=("AKIA", "ASIA")),
    _PatternSpec(
        "slack-token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
        needles=("xoxb-", "xoxa-", "xoxp-", "xoxr-", "xoxs-"),
    ),
    _PatternSpec("npm-token", re.compile(r"\bnpm_[A-Za-z0-9]{20,}"), needles=("npm_",)),
    _PatternSpec(
        "cloudflare-token",
        re.compile(r"\bcf(?:ut|at|k)_[A-Za-z0-9]{40,}"),
        needles=("cfut_", "cfat_", "cfk_"),
    ),
    _PatternSpec(
        "authorization",
        re.compile(r"(?i)authorization\s*:\s*(?:bearer|basic)\s+(?P<value>\S+)"),
        group="value",
        needles=("authorization",),
        folded=True,
    ),
    _PatternSpec(
        "api-key-header",
        re.compile(r"(?i)x-api-key\s*[:=]\s*(?P<value>\S+)"),
        group="value",
        needles=("x-api-key",),
        folded=True,
    ),
    _PatternSpec(
        # The unquoted alternative excludes backslash as well as quotes. Without
        # that, a secret sitting inside already-serialized JSON absorbs the
        # leading backslash of the following \" escape; replacing the span then
        # leaves a bare quote that terminates the JSON string and corrupts the
        # document. Real credentials do not contain backslashes, so excluding it
        # only ever shortens a match.
        "credential-assignment",
        re.compile(
            r"(?im)\b[A-Za-z0-9_]*(?:API_?KEY|ACCESS_?KEY|TOKEN|SECRET|"
            r"PASSWORD|PASSWD|PRIVATE_?KEY|CLIENT_?SECRET)[A-Za-z0-9_]*\s*[=:]\s*"
            r"(?P<value>[^\s\"'\\]+|\"[^\"]+\"|'[^']+')"
        ),
        group="value",
        needles=_CREDENTIAL_KEYWORDS,
        folded=True,
        keyword_prefixed=True,
    ),
    _PatternSpec(
        "credential-url",
        re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:(?P<value>[^@\s/]+)@"),
        group="value",
        needles=("://",),
    ),
)


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _keyword_prefixed_matches(spec: _PatternSpec, text: str) -> list[re.Match[str]]:
    """Return the same matches as ``spec.pattern.finditer(text)``, found from the keyword.

    ``\\b[A-Za-z0-9_]*(?:API_?KEY|...)`` makes the engine retry the keyword
    alternation from every word character of every word. For ASCII text the
    alternative can only match where its lowercase spelling occurs, so locating
    those occurrences and matching once per containing word start yields the
    same leftmost, non-overlapping spans for a fraction of the cost. Non-ASCII
    text can fold into a keyword by a rule ``str.lower`` does not model, so it
    keeps the plain scan.
    """
    if not text.isascii():
        return list(spec.pattern.finditer(text))
    lowered = text.lower()
    starts: set[int] = set()
    for needle in spec.needles:
        index = lowered.find(needle)
        while index >= 0:
            start = index
            while start and _is_word_char(lowered[start - 1]):
                start -= 1
            starts.add(start)
            index = lowered.find(needle, index + 1)
    matches: list[re.Match[str]] = []
    end = 0
    for start in sorted(starts):
        if start < end:
            continue
        match = spec.pattern.match(text, start)
        if match is not None:
            matches.append(match)
            end = match.end()
    return matches


def _iter_matches(spec: _PatternSpec, text: str) -> Iterable[re.Match[str]]:
    if spec.keyword_prefixed:
        return _keyword_prefixed_matches(spec, text)
    return spec.pattern.finditer(text)


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
    folded: str | None = None
    # `(?i)` matches an ASCII letter for U+0130 and U+0131, which `str.casefold`
    # does not turn into that letter; such text skips the screen rather than
    # risk screening out a real match.
    screenable = "\u0130" not in text and "\u0131" not in text
    for spec in _PATTERNS:
        if spec.needles and screenable:
            if spec.folded:
                folded = folded if folded is not None else text.casefold()
                haystack = folded
            else:
                haystack = text
            if not any(needle in haystack for needle in spec.needles):
                continue
        for match in _iter_matches(spec, text):
            start, end = match.span(spec.group) if spec.group else match.span()
            if end > start:
                findings.append(SecretFinding(spec.category, start, end))
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


def redact_secrets_in_obj(
    obj: Any,
    extra_values: Iterable[str] | None = None,
    *,
    include_environment: bool = False,
) -> Any:
    """Redact string leaves of a JSON-compatible structure, preserving its shape.

    Prefer this over running ``redact_secrets`` on serialized JSON. Redacting the
    encoded form can splice a replacement across an escape sequence and produce a
    document that no longer parses; redacting the decoded leaves and re-serializing
    is valid by construction, and the patterns match real text rather than escaped
    text.

    Leaves that are not JSON-native are coerced with ``str()`` and *then* redacted.
    That closes a hole in the ``json.dumps(..., default=str)`` pattern, where an
    object stringified during serialization would otherwise never be scrubbed.
    Dict keys are left alone: they are structural, and rewriting them risks
    collisions.
    """
    values = list(extra_values or [])
    if include_environment:
        # Snapshot once: rebuilding the environment value list for every string
        # leaf dominated exporting payloads with thousands of blocks.
        values.extend(_environment_values())

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return redact_secrets(node, values)
        if isinstance(node, bool) or node is None or isinstance(node, (int, float)):
            return node
        if isinstance(node, dict):
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(item) for item in node]
        return redact_secrets(str(node), values)

    return walk(obj)
