"""Small, plain-text tool subjects shared by live and persisted displays.

This is a display boundary, not a shell parser or arbitrary-secret detector.
Unrecognized payloads are never stringified. Ambiguous credentials, inline code,
and oversized subjects are suppressed rather than displayed optimistically.
Callers must render the returned string as literal text, not markup.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import re
import shlex
import unicodedata
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from kolega_code.security.secrets import SECRET_PLACEHOLDER, redact_secrets

if TYPE_CHECKING:
    from kolega_code.config import AgentConfig

MAX_SUBJECT_LENGTH = 240
_MAX_INPUT_LENGTH = 16_384
_MAX_SECRET_VALUES = 256
_MAX_PATCH_PATHS = 8
_ANSI = re.compile(
    r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c|$)"
    r"|(?:\x1b[PX^_]|\x90|\x98|\x9e|\x9f)[^\x1b\x9c]*(?:\x1b\\|\x9c|$)"
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*(?:[@-~]|$)|\x1b[@-_]"
)
_URL = re.compile(r"(?i)(?:[a-z][a-z0-9+.-]*:)?//[^\s<>\"']+")
_PATCH_HEADER = re.compile(r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)$", re.MULTILINE)
_SENSITIVE_NAME = re.compile(
    r"key|token|secret|password|passwd|credential|auth|cookie|session|signature|"
    r"header|username|connection.?string",
    re.IGNORECASE,
)
_SENSITIVE_WORDS = frozenset({"auth", "key", "pass", "pwd", "sig", "jwt", "bearer", "basic"})
_PAYLOAD_FLAGS = frozenset(
    {
        "header",
        "proxy-header",
        "user",
        "proxy-user",
        "http-user",
        "ftp-user",
        "cert",
        "certificate",
        "passphrase",
        "dsn",
        "data",
        "data-raw",
        "data-binary",
        "data-urlencode",
        "data-ascii",
        "json",
        "form",
        "form-string",
        "config",
    }
)
_FILE_TOOLS = frozenset(
    {
        "read",
        "read_image",
        "write",
        "edit",
        "claude_edit",
        "claude_write",
        "hashline_edit",
        "hashline_write",
        "multi_edit",
        "read_memory",
        "write_memory",
        "edit_memory",
    }
)
_ALIASES = {
    "read_file": "read",
    "read_entire_file": "read",
    "read_file_section": "read",
    "write_file": "write",
    "edit_file": "edit",
}
_NO_SUBJECT_TOOLS = frozenset({"eval", "browser_evaluate", "write_stdin", "dispatch_agent"})
_INTERPRETERS = frozenset(
    {"sh", "bash", "zsh", "fish", "python", "python3", "node", "ruby", "perl", "powershell", "pwsh"}
)


def configured_tool_subject_secrets(config: AgentConfig | None) -> tuple[str, ...]:
    """Credentials already in memory, shared by execution and history rendering.

    Never loads settings, reads credential files, or refreshes OAuth tokens.
    """
    if config is None:
        return ()
    from kolega_code.config import ModelProvider

    values = [config.get_api_key(provider) for provider in ModelProvider]
    values.extend((config.web_search_api_key, config.langfuse_public_key, config.langfuse_secret_key))
    values.extend(endpoint.api_key for endpoint in config.custom_endpoints.values())
    manager = config._chatgpt_token_manager
    for tokens in (config.openai_chatgpt_tokens, getattr(manager, "tokens", None)):
        if tokens is not None:
            values.extend((tokens.access_token, tokens.refresh_token, tokens.id_token))
    return tuple(value for value in values if isinstance(value, str) and value)


def _string(value: object) -> str:
    # Do not cut an arbitrary string before redaction: that could expose a
    # configured secret whose suffix lies beyond the work limit.
    return value if isinstance(value, str) and len(value) <= _MAX_INPUT_LENGTH else ""


def _first_string(data: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _string(data.get(key))
        if value.strip():
            return value
    return ""


def _joined_strings(value: object, separator: str) -> str:
    if not isinstance(value, list) or not value or len(value) > 32:
        return ""
    if any(not isinstance(item, str) or len(item) > _MAX_INPUT_LENGTH for item in value):
        return ""
    return separator.join(value)


def _clean(text: str) -> str:
    text = _ANSI.sub("", text)
    return "".join(
        " " if char.isspace() else char
        for char in text
        if char.isspace() or unicodedata.category(char) not in {"Cc", "Cf", "Cs"}
    )


def _safe_url(value: str) -> str:
    """Keep only the URL location; query/fragment payloads are not subjects."""
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in {"", "http", "https", "ftp", "ftps", "ssh", "git", "ws", "wss"}:
            return ""
        host = parts.netloc.rsplit("@", 1)[-1]
        if not host or not parts.hostname or any(char.isspace() for char in host):
            return ""
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return ""


def _sensitive_name(name: str) -> bool:
    name = name.lower().strip("-").replace("_", "-")
    return bool(_SENSITIVE_NAME.search(name)) or name in _SENSITIVE_WORDS


def _safe_command_parts(text: str, values: tuple[str, ...]) -> str:
    """Suppress from the first suspect argument, including malformed quotes.

    Tokenization is only for inspection. Ordinary commands keep their original
    quoting; a redacted prefix may be re-quoted, and is never meant for execution.
    """
    # Shell expansions and heredocs can contain whole programs or file bodies.
    embedded = re.search(r"<<|`|\$\(|<\(|>\(", text)
    if embedded:
        prefix = _safe_command_parts(text[: embedded.start()], values)
        return f"{prefix} {SECRET_PLACEHOLDER}".strip()
    try:
        # Decode quote-concatenated URL arguments before removing their query
        # or userinfo. Stripping a raw URL prefix first could orphan its secret
        # suffix (for example ?token='a value with spaces').
        arguments = shlex.split(text)
        safe_arguments: list[str] = []
        for argument in arguments:
            match = _URL.search(argument)
            if match:
                argument = argument[: match.start()] + (_safe_url(argument[match.start() :]) or SECRET_PLACEHOLDER)
            safe_arguments.append(argument)
        if safe_arguments != arguments:
            text = shlex.join(safe_arguments)
        lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        # A subject can be prose or a path, not shell input. Preserve ordinary
        # word apostrophes only after inspecting the dequoted form for secrets.
        # Real command inputs are validated separately before reaching here.
        words = re.sub(r"(?<=\w)'(?=\w)", "", text)
        if words != text:
            inspected = _safe_command_parts(words, values)
            return text if inspected == words else inspected
        # A quote/backslash may hide the boundary of a sensitive value. Do not
        # display even a partial token from an invalid command/metadata string.
        return ""
    programs = {posixpath.basename(token).lower() for token in tokens}
    interpreter = bool(programs & _INTERPRETERS) or any(
        re.fullmatch(r"python\d+(?:\.\d+)?", program) for program in programs
    )
    for index, token in enumerate(tokens):
        name = re.split(r"[=:]", token, maxsplit=1)[0]
        assignment = "=" in token and _sensitive_name(name)
        header = ":" in token and _sensitive_name(name)
        separated_value = (
            _sensitive_name(token) and index + 1 < len(tokens) and tokens[index + 1].startswith(("=", ":"))
        )
        long_flag = token.startswith("--") and (_sensitive_name(name) or name[2:] in _PAYLOAD_FLAGS)
        # These short options commonly carry credentials (including attached
        # forms such as -uuser:pass, -HAuthorization:..., and -pfake-password).
        short_flag = bool(re.match(r"^-[uUpH]", token))
        curl_payload = "curl" in programs and bool(re.match(r"^-[a-zA-Z]*[uUpHbFdEK]", token))
        inline_code = interpreter and token.lower() in {"-c", "-e", "-command", "-encodedcommand", "-enc"}
        if assignment or header or separated_value or long_flag or short_flag or curl_payload or inline_code:
            prefix = shlex.join(tokens[:index])
            return f"{prefix} {SECRET_PLACEHOLDER}".strip()
    # Shell quotes/backslashes can also split a known secret. Preserve ordinary
    # display quoting unless this second inspection actually changes something.
    decoded = shlex.join(tokens)
    redacted = _redact(decoded, values)
    if redacted != decoded:
        return redacted
    return text


def _redact(text: str, values: tuple[str, ...]) -> str:
    text = redact_secrets(text, extra_values=values, include_environment=True)
    # The shared detector deliberately ignores values shorter than eight
    # characters. Explicitly supplied short secrets still belong off displays.
    short_values = {value for value in values if 0 < len(value) < 8}
    if short_values:
        # One pass, protecting placeholders: repeated short values must not
        # recursively expand the output (e.g. many copies of the secret "e").
        pattern = "|".join(re.escape(value) for value in sorted(short_values, key=lambda value: (-len(value), value)))
        text = re.sub(re.escape(SECRET_PLACEHOLDER) + "|" + pattern, lambda _: SECRET_PLACEHOLDER, text)
    return text


def sanitize_tool_subject(subject: object, *, secret_values: Iterable[str] = ()) -> str:
    """Sanitize already-emitted metadata into at most 240 single-line characters.

    Only string inputs are accepted. All URL query/fragment values are omitted;
    probable shell credential/payload arguments suppress the remainder. Known
    environment secrets and explicit values are redacted before truncation.
    Strings above the work limit and ambiguous quoting fail closed. Ordinary
    word apostrophes are preserved in path and query subjects.
    """
    text = _string(subject)
    if not text:
        return ""
    supplied = (secret_values,) if isinstance(secret_values, str) else secret_values
    values = tuple(islice(supplied, _MAX_SECRET_VALUES + 1))
    if len(values) > _MAX_SECRET_VALUES or any(not isinstance(value, str) for value in values):
        return ""
    # Redact both before and after control removal, to handle configured values
    # containing controls as well as controls inserted into recognizable tokens.
    text = _clean(_redact(text, values))
    text = _safe_command_parts(text, values)
    text = " ".join(_clean(_redact(text, values)).split())
    if len(text) > MAX_SUBJECT_LENGTH:
        return text[: MAX_SUBJECT_LENGTH - 1] + "…"
    return text


def _shorten_path(value: str, project_path: str | Path | None) -> str:
    """Lexical only: no resolve, stat, cwd lookup, or home-directory lookup."""
    # Normalizing a URL as a path destroys its :// (or // authority), hiding
    # credentials from later sanitization. Preserve URL/control syntax until
    # the sanitizer has redacted the original text and removed unsafe parts.
    if _URL.search(value) or _clean(value) != value:
        return value
    path_module = ntpath if ntpath.splitdrive(value)[0] else posixpath
    path = path_module.normpath(value)
    project = str(project_path) if isinstance(project_path, (str, Path)) else ""
    home = os.environ.get("USERPROFILE" if path_module is ntpath else "HOME", "")
    for root, replacement in ((project, ""), (home, "~")):
        if not root or len(root) > _MAX_INPUT_LENGTH:
            continue
        root = path_module.normpath(root)
        prefix = root.rstrip(path_module.sep) + path_module.sep
        if path_module.normcase(path) == path_module.normcase(root):
            return replacement or "."
        if path_module.normcase(path).startswith(path_module.normcase(prefix)):
            tail = path[len(prefix) :]
            return replacement + path_module.sep + tail if replacement else tail
    return path


def _patch_subject(patch: object, project_path: str | Path | None) -> str:
    if not isinstance(patch, str):
        return ""
    # A bounded scan is safe here because only complete, anchored file headers
    # are selected, never a partial final line or a hunk's +/-/space-prefixed body.
    prefix = patch[:_MAX_INPUT_LENGTH]
    if len(patch) > _MAX_INPUT_LENGTH:
        prefix = prefix[: prefix.rfind("\n")] if "\n" in prefix else ""
    paths: list[str] = []
    for match in _PATCH_HEADER.finditer(prefix):
        path = _shorten_path(match.group(1).rstrip("\r"), project_path)
        if path not in paths:
            paths.append(path)
        if len(paths) == _MAX_PATCH_PATHS:
            break
    return ", ".join(paths)


def build_tool_subject(
    tool_name: str,
    tool_input: object,
    *,
    project_path: str | Path | None = None,
    secret_values: Iterable[str] = (),
) -> str:
    """Extract a safe subject without inspecting files or arbitrary payloads.

    Supports built-in edit protocols, legacy read names, and namespaced names.
    Unknown/MCP tools use only top-level path, URL, query/pattern, or command
    fields. Code, prompts, stdin, file contents, and nested dictionaries are
    never used as fallback labels.
    """
    if not isinstance(tool_name, str) or len(tool_name) > 256:
        return ""
    name = re.split(r"__|[.:/]", tool_name.strip().lower())[-1]
    name = _ALIASES.get(name, name)
    if not name or name in _NO_SUBJECT_TOOLS:
        return ""
    data = tool_input if isinstance(tool_input, dict) else {}
    if name.endswith(" (hosted)"):
        # Hosted history carries action metadata rather than a local ToolCall.
        # Use the same extraction here and in live emission, with bounded joins.
        subject = _first_string(data, ("query",)) or _joined_strings(data.get("queries"), ", ")
        if not subject:
            subject = _first_string(data, ("url",)) or _joined_strings(data.get("urls"), " ")
    elif name == "apply_patch":
        patch = tool_input if isinstance(tool_input, str) else data.get("input", data.get("patch"))
        subject = _patch_subject(patch, project_path)
    elif name in _FILE_TOOLS:
        # read/Claude use file_path; search/replace and hashline use path.
        keys = ("file_path", "path") if name in {"read", "claude_edit", "claude_write"} else ("path", "file_path")
        path = _first_string(data, keys)
        subject = _shorten_path(path, project_path) if path else ""
    elif name == "exec_command":
        subject = _first_string(data, ("command",))
        try:
            shlex.split(subject)
        except ValueError:
            return ""
    elif name == "browser_find":
        subject = _first_string(data, ("text", "regex"))
    elif name in {"web_search", "search", "search_codebase", "grep", "glob", "find_files_by_pattern"}:
        subject = _first_string(data, ("query", "pattern"))
    elif name in {"web_fetch", "fetch", "browser_navigate", "navigate"}:
        subject = _safe_url(_clean(_first_string(data, ("url",))))
    else:
        path = _first_string(data, ("file_path", "path"))
        url = _first_string(data, ("url",))
        if path:
            subject = _shorten_path(path, project_path)
        elif url:
            subject = _safe_url(_clean(url))
        else:
            subject = _first_string(data, ("query", "pattern", "command"))
    return sanitize_tool_subject(subject, secret_values=secret_values)
