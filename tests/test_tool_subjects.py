"""Focused extraction and security-boundary tests for shared tool subjects."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

from kolega_code.security.secrets import SECRET_PLACEHOLDER
from kolega_code.tool_subjects import (
    MAX_SUBJECT_LENGTH,
    build_tool_display,
    build_tool_subject,
    sanitize_tool_display,
    sanitize_tool_subject,
)

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.fixture(autouse=True)
def subject_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace, rather than inspect, the environment used by the helper.
    monkeypatch.setattr(os, "environ", {"HOME": "/home/tester"})


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("read", {"file_path": "/project/src/main.py", "offset": 8}, "src/main.py"),
        ("read", {"path": "/project/src/alias.py"}, "src/alias.py"),
        ("read", {"file_path": "canonical", "path": "alias"}, "canonical"),
        ("read_file", {"path": "legacy.py"}, "legacy.py"),
        ("read_file_section", {"file_path": "legacy.py"}, "legacy.py"),
        ("read_entire_file", {"file_path": "legacy.py"}, "legacy.py"),
        ("read_image", {"path": "assets/image.png"}, "assets/image.png"),
        ("write", {"path": "new.py", "content": "BODY_DO_NOT_SHOW"}, "new.py"),
        ("write", {"file_path": "claude.py", "content": "BODY_DO_NOT_SHOW"}, "claude.py"),
        ("edit", {"path": "edit.py", "block": "BODY_DO_NOT_SHOW"}, "edit.py"),
        ("edit", {"file_path": "edit.py", "new_string": "BODY_DO_NOT_SHOW"}, "edit.py"),
        ("claude_edit", {"file_path": "edit.py", "old_string": "BODY_DO_NOT_SHOW"}, "edit.py"),
        ("claude_write", {"file_path": "new.py", "content": "BODY_DO_NOT_SHOW"}, "new.py"),
        ("hashline_edit", {"path": "old.py", "rename": "new.py", "edits": [{"content": "BODY"}]}, "old.py"),
        ("hashline_write", {"path": "hash.py", "content": "BODY_DO_NOT_SHOW"}, "hash.py"),
        ("multi_edit", {"path": "multi.py", "blocks": "BODY_DO_NOT_SHOW"}, "multi.py"),
        (
            "exec_command",
            {"command": "pytest -q tests/test_one.py", "workdir": "/project"},
            "pytest -q tests/test_one.py",
        ),
        ("exec_command", {"command": "rg -n 'hello world' src"}, "rg -n 'hello world' src"),
        ("exec_command", {"command": "cd /project && pytest -q"}, "cd /project && pytest -q"),
        ("web_search", {"query": "Python examples", "max_results": 3}, "Python examples"),
        ("search_codebase", {"pattern": "def main", "path": "/project"}, "def main"),
        ("browser_find", {"text": "Save button"}, "Save button"),
        ("browser_find", {"regex": "Save.*button"}, "Save.*button"),
        ("glob", {"pattern": "**/*.py"}, "**/*.py"),
        ("web_fetch", {"url": "https://example.test/docs", "instruction": "PROMPT"}, "https://example.test/docs"),
        ("browser_navigate", {"url": "https://example.test/"}, "https://example.test/"),
        ("lsp", {"path": "main.py", "query": "ignored"}, "main.py"),
        ("lsp", {"operation": "workspace_symbols", "query": "Widget"}, "Widget"),
        ("mcp__server__unknown", {"file_path": "/project/file.txt", "content": "BODY"}, "file.txt"),
        ("mcp__server__unknown", {"pattern": "needle", "prompt": "PROMPT"}, "needle"),
        ("mcp__server__unknown", {"url": "https://example.test/path?token=FAKE"}, "https://example.test/path"),
        ("unknown", {"command": "pytest -q", "code": "CODE"}, "pytest -q"),
    ],
)
def test_tool_families(name: str, args: object, expected: str) -> None:
    assert build_tool_subject(name, args, project_path=Path("/project")) == expected


@pytest.mark.parametrize(
    "name", ["functions.read", "tools.read", "functions::read", "functions/read", "mcp__files__read", " READ "]
)
def test_namespaced_aliases(name: str) -> None:
    assert build_tool_subject(name, {"file_path": "main.py"}) == "main.py"


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("", {"path": "hidden"}),
        ("x" * 257, {"path": "hidden"}),
        ("read", None),
        ("read", "not a dictionary"),
        ("read", {"file_path": 12, "content": "BODY"}),
        ("read", {"file_path": {"path": "nested"}}),
        ("edit", {"block": "BODY"}),
        ("multi_edit", {"edits": [{"file_path": "nested", "content": "BODY"}]}),
        ("hashline_edit", {"rename": "new.py"}),
        ("apply_patch", {"input": None}),
        ("apply_patch", 123),
        ("exec_command", {"title": "not the command"}),
        ("eval", {"command": "CODE", "code": "CODE", "title": "title"}),
        ("browser_evaluate", {"code": "CODE"}),
        ("write_stdin", {"chars": "STDIN", "command": "STDIN"}),
        ("dispatch_agent", {"prompt": "PROMPT"}),
        ("unknown", {"code": "CODE", "prompt": "PROMPT", "content": "BODY", "text": "BODY", "title": "BODY"}),
        ("unknown", [{"path": "nested"}]),
        ("web_fetch", {"url": "javascript:alert('CODE')"}),
        ("web_fetch", {"url": "https://[invalid"}),
        ("web_fetch", {"url": "not a URL"}),
    ],
)
def test_missing_and_unsafe_inputs(name: str, args: object) -> None:
    assert build_tool_subject(name, args) == ""


@pytest.mark.parametrize("value", [None, 12, True, ["hello"], {"subject": "hello"}, object()])
def test_sanitize_never_stringifies(value: object) -> None:
    assert sanitize_tool_subject(value) == ""


@pytest.mark.parametrize(
    ("path", "project", "expected"),
    [
        ("/project/a/../b.py", "/project", "b.py"),
        ("/project-other/a.py", "/project", "/project-other/a.py"),
        ("/project", "/project", "."),
        ("/home/tester/.config/test", None, "~/.config/test"),
        ("/home/tester-other/test", None, "/home/tester-other/test"),
        ("../sibling/file.py", "/project", "../sibling/file.py"),
        ("/project/../../outside.py", "/project", "/outside.py"),
        (r"C:\project\src\main.py", r"C:\project", r"src\main.py"),
    ],
)
def test_lexical_path_shortening(path: str, project: str | None, expected: str) -> None:
    assert build_tool_subject("read", {"path": path}, project_path=project) == expected


def test_no_filesystem_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("subject construction must not access the filesystem")

    for method in ("resolve", "absolute", "stat", "exists", "read_text", "home", "cwd"):
        monkeypatch.setattr(Path, method, forbidden)
    assert build_tool_subject("write", {"path": "/project/file"}, project_path=Path("/project")) == "file"


_PATCH = """*** Begin Patch
*** Update File: /project/old.py
*** Move to: /project/new.py
@@
-PRIVATE_BODY_REMOVED
+PRIVATE_BODY_ADDED
 *** Update File: BODY_FAKE_HEADER
+*** Add File: BODY_FAKE_HEADER
*** Add File: /project/added.py
+PRIVATE_BODY_ADDED
*** Delete File: /project/deleted.py
*** End Patch
"""


@pytest.mark.parametrize("wrapper", ["freeform", "input", "patch"])
def test_patch_headers_only(wrapper: str) -> None:
    payload = _PATCH if wrapper == "freeform" else {wrapper: _PATCH}
    result = build_tool_subject("functions.apply_patch", payload, project_path="/project")
    assert result == "old.py, new.py, added.py, deleted.py"
    assert "BODY" not in result


def test_huge_patch_and_partial_header_never_expose_body() -> None:
    payload = "*** Begin Patch\n*** Update File: main.py\n@@\n+" + "BODY" * 100_000
    assert build_tool_subject("apply_patch", payload) == "main.py"
    payload = "*** Begin Patch\n*** Update File: " + "partial" * 10_000
    assert build_tool_subject("apply_patch", payload) == ""
    assert build_tool_subject("apply_patch", "*** Begin Patch\n@@\n+BODY\n*** End Patch") == ""


def test_patch_path_count_is_bounded_and_duplicates_are_omitted() -> None:
    patch = "*** Begin Patch\n" + "".join(f"*** Update File: file{i}.py\n@@\n+BODY\n" for i in range(10))
    assert build_tool_subject("apply_patch", patch) == ", ".join(f"file{i}.py" for i in range(8))
    assert build_tool_subject("apply_patch", "*** Update File: file.py\n*** Move to: file.py\n") == "file.py"


@pytest.mark.parametrize(
    "command",
    [
        "curl --token FAKE_CREDENTIAL_TAIL",
        "curl --api-key=FAKE_CREDENTIAL_TAIL",
        "curl --password 'FAKE CREDENTIAL TAIL'",
        "curl --password 'FAKE_CREDENTIAL_TAIL",
        'curl --"pass"word FAKE_CREDENTIAL_TAIL',
        r"curl --pa\ssword FAKE_CREDENTIAL_TAIL",
        "curl -uuser:FAKE_CREDENTIAL_TAIL",
        "curl -suuser:FAKE_CREDENTIAL_TAIL",
        "mysql -pFAKE_CREDENTIAL_TAIL",
        "curl -H 'Authorization: Bearer FAKE_CREDENTIAL_TAIL'",
        "curl -HAuthorization:Bearer\\ FAKE_CREDENTIAL_TAIL",
        "curl --header Authorization: Bearer FAKE_CREDENTIAL_TAIL",
        "curl --proxy-user user:FAKE_CREDENTIAL_TAIL",
        "curl --oauth2-bearer FAKE_CREDENTIAL_TAIL",
        "curl -Ecert:FAKE_CREDENTIAL_TAIL",
        "git -c 'http.extraHeader=Authorization: Custom FAKE_CREDENTIAL_TAIL' fetch",
        "curl -bFAKE_CREDENTIAL_TAIL",
        "curl --cookie session=FAKE_CREDENTIAL_TAIL",
        "curl --data 'password=FAKE_CREDENTIAL_TAIL'",
        "curl -dFAKE_CREDENTIAL_TAIL",
        "curl -Ffield=FAKE_CREDENTIAL_TAIL",
        "env API_KEY='FAKE CREDENTIAL TAIL' pytest",
        "export PASSWORD=FAKE_CREDENTIAL_TAIL; pytest",
        "echo safe;pwd=FAKE_CREDENTIAL_TAIL",
        "echo safe;PASS=FAKE_CREDENTIAL_TAIL",
        "PASSWORD = FAKE_CREDENTIAL_TAIL",
        "pwd: FAKE_CREDENTIAL_TAIL",
        "pwd = FAKE_CREDENTIAL_TAIL",
        "Authorization: Custom FAKE_CREDENTIAL_TAIL",
        "Authorization : Custom FAKE_CREDENTIAL_TAIL",
        "X-Api-Key: FAKE_CREDENTIAL_TAIL",
        "python -c 'print(\"FAKE_CREDENTIAL_TAIL\")'",
        "bash -c 'echo FAKE_CREDENTIAL_TAIL'",
        "cat <<EOF\nFAKE_CREDENTIAL_TAIL\nEOF",
        "echo $(printf FAKE_CREDENTIAL_TAIL)",
        "echo `printf FAKE_CREDENTIAL_TAIL`",
    ],
)
def test_sensitive_shell_portions_are_suppressed(command: str) -> None:
    for result in (build_tool_subject("exec_command", {"command": command}), sanitize_tool_subject(command)):
        assert "FAKE" not in result
        assert "CREDENTIAL" not in result
        assert "TAIL" not in result


@pytest.mark.parametrize(
    "url",
    [
        "https://fake-user:fake-password@example.test/path?api_key=FAKE_QUERY#FAKE_FRAGMENT",
        "https://fake-user@example.test/path?access_token=FAKE_QUERY",
        "https://example.test/path?X-Amz-Signature=FAKE_QUERY",
        "https://example.test/path?%74oken=FAKE_QUERY",
        "https://example.test/path?q=harmless&redirect=https%3A%2F%2Fu%3Ap%40host",
        "https://example.test/path#access_token=FAKE_FRAGMENT",
    ],
)
def test_url_credentials_queries_and_fragments(url: str) -> None:
    assert build_tool_subject("web_fetch", {"url": url}) == "https://example.test/path"
    assert sanitize_tool_subject(url) == "https://example.test/path"
    assert build_tool_subject("exec_command", {"command": f"curl '{url}'"}) == "curl https://example.test/path"


@pytest.mark.parametrize("tool_name", ["read", "mcp__files__unknown", "apply_patch"])
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://fake-user:fake-password@example.test/path?token=FAKE_QUERY#FAKE_FRAGMENT",
            "https://example.test/path",
        ),
        (
            "//fake-user:fake-password@example.test/path?token=FAKE_QUERY#FAKE_FRAGMENT",
            "//example.test/path",
        ),
        (
            "https:\x00//fake-user:fake-password@example.test/path?token=FAKE_QUERY#FAKE_FRAGMENT",
            "https://example.test/path",
        ),
        (
            "\x1b[33mhttps://fake-user:fake-password@example.test/path?token=FAKE_QUERY#FAKE_FRAGMENT\x1b[0m",
            "https://example.test/path",
        ),
    ],
)
def test_url_in_path_preserves_credential_boundaries(tool_name: str, url: str, expected: str) -> None:
    inputs: object = f"*** Update File: {url}\n" if tool_name == "apply_patch" else {"file_path": url}
    assert build_tool_subject(tool_name, inputs) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  first\nsecond\tthird\r\n", "first second third"),
        ("\x1b[31mred\x1b[0m", "red"),
        ("\x9b31mred\x9b0m", "red"),
        ("\x9d8;;https://hidden.test\x9clabel\x9d8;;\x9c", "label"),
        ("\x1b]8;;https://hidden.test\x07label\x1b]8;;\x07", "label"),
        ("safe\x1b]unterminated hidden", "safe"),
        ("a\x00b\x08c\u202edef\u2066ghi\u2069\u200b", "abcdefghi"),
        ("[bold]literal[/bold] <tag>文字</tag>", "[bold]literal[/bold] <tag>文字</tag>"),
        ("文件名\u2028搜索\u2029测试", "文件名 搜索 测试"),
    ],
)
def test_single_line_plain_unicode(value: str, expected: str) -> None:
    assert sanitize_tool_subject(value) == expected


@pytest.mark.parametrize(
    ("tool_name", "inputs", "expected"),
    [
        ("web_search", {"query": "what's new in Textual?"}, "what's new in Textual?"),
        ("read", {"file_path": "docs/author's notes.md"}, "docs/author's notes.md"),
        ("exec_command", {"command": "printf %s foo'bar"}, ""),
        ("web_search (hosted)", {"queries": ["first query", "second query"]}, "first query, second query"),
        ("web_search (hosted)", {"queries": [None, "second query"]}, ""),
        ("web_search (hosted)", {"queries": ["too many"] * 33}, ""),
        ("web_search (hosted)", {"queries": ["too long" * 3000]}, ""),
        ("fetch_url (hosted)", {"urls": ["https://u:p@example.test/a?token=FAKE"]}, "https://example.test/a"),
    ],
)
def test_plain_apostrophes_and_hosted_metadata(tool_name: str, inputs: object, expected: str) -> None:
    subject = build_tool_subject(tool_name, inputs)
    assert subject == expected
    assert sanitize_tool_subject(subject) == expected


def test_size_bounds_and_redaction_before_final_truncation() -> None:
    assert sanitize_tool_subject("界" * 1000) == "界" * (MAX_SUBJECT_LENGTH - 1) + "…"
    assert sanitize_tool_subject("a" * 100_000) == ""
    assert build_tool_subject("exec_command", {"command": "pytest " + "a" * 100_000}) == ""
    secret = "FAKE_CONFIGURED_CREDENTIAL"
    result = sanitize_tool_subject("x" * 230 + secret, secret_values=[secret])
    assert result == "x" * 230 + SECRET_PLACEHOLDER
    # A secret crossing the work boundary must not leave its prefix behind.
    assert sanitize_tool_subject("x" * 16_380 + secret, secret_values=[secret]) == ""


def test_environment_and_configured_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "FAKE_ENVIRONMENT_CREDENTIAL")
    assert sanitize_tool_subject("look FAKE_ENVIRONMENT_CREDENTIAL here") == f"look {SECRET_PLACEHOLDER} here"
    assert (
        build_tool_subject(
            "web_search",
            {"query": "look FAKE_CONFIGURED_CREDENTIAL"},
            secret_values=iter(["FAKE_CONFIGURED_CREDENTIAL"]),
        )
        == f"look {SECRET_PLACEHOLDER}"
    )
    assert sanitize_tool_subject("value abc here", secret_values=["abc"]) == f"value {SECRET_PLACEHOLDER} here"
    assert sanitize_tool_subject("FAKE_CONFIGURED_CREDENTIAL", secret_values="FAKE_CONFIGURED_CREDENTIAL") == (
        SECRET_PLACEHOLDER
    )


def test_controls_cannot_bypass_known_secret_redaction() -> None:
    assert sanitize_tool_subject("sk-\u202eFAKE123456789") == SECRET_PLACEHOLDER
    assert sanitize_tool_subject("fake\x00credential", secret_values=["fake\x00credential"]) == SECRET_PLACEHOLDER


def test_shell_quoting_cannot_split_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "FAKE_ENVIRONMENT_CREDENTIAL")
    result = build_tool_subject("exec_command", {"command": "echo FAKE_ENVIRONMENT_'CREDENTIAL'"})
    assert result == f"echo {SECRET_PLACEHOLDER}"
    result = build_tool_subject("exec_command", {"command": "curl https://fake-user:'FAKE_PASSWORD'@example.test/path"})
    assert result == "curl https://example.test/path"


@pytest.mark.parametrize(
    "command",
    [
        'curl "https://example.test/path?token=FAKE CREDENTIAL TAIL"',
        "curl https://example.test/path?token='FAKE CREDENTIAL TAIL'",
        "curl --url='https://example.test/path?token=FAKE CREDENTIAL TAIL'",
        "curl https://example.test/path#'FAKE CREDENTIAL TAIL'",
    ],
)
def test_quoted_url_query_values_do_not_leave_orphaned_suffixes(command: str) -> None:
    result = build_tool_subject("exec_command", {"command": command})
    assert "FAKE" not in result
    assert "CREDENTIAL" not in result
    assert "TAIL" not in result
    assert "https://example.test/path" in result


def test_bounded_secret_iterable() -> None:
    def endless_values() -> Iterable[str]:
        while True:
            yield "FAKE_CONFIGURED_CREDENTIAL"

    assert sanitize_tool_subject("safe", secret_values=endless_values()) == ""


def test_repeated_short_secrets_do_not_expand_placeholders_recursively() -> None:
    assert sanitize_tool_subject("e", secret_values=["e"] * 256) == SECRET_PLACEHOLDER
    assert len(sanitize_tool_subject("e" * 1000, secret_values=["e"] * 256)) <= MAX_SUBJECT_LENGTH


def test_tool_display_paths_keep_full_safe_paths_and_multi_file_patches() -> None:
    long_name = "src/" + ("文件" * 130) + ".py"
    assert build_tool_display("read", {"file_path": f"/project/{long_name}"}, project_path="/project") == {
        "paths": [long_name]
    }
    assert len(
        build_tool_display("read", {"file_path": f"/project/{long_name}"}, project_path="/project")["paths"][0]
    ) > (MAX_SUBJECT_LENGTH)
    assert build_tool_display("read", {"file_path": "/project-other/outside.py"}, project_path="/project") == {
        "paths": ["/project-other/outside.py"]
    }
    assert build_tool_display("read", {"file_path": "/home/tester/.config/test"}, project_path="/project") == {
        "paths": ["~/.config/test"]
    }
    assert build_tool_display("apply_patch", _PATCH, project_path="/project") == {
        "paths": ["old.py", "new.py", "added.py", "deleted.py"]
    }


def test_tool_display_paths_redact_secrets_before_control_cleaning() -> None:
    secret = "fake\x00credential"
    assert sanitize_tool_display(
        {"paths": [f"/tmp/{secret}/report.txt"]},
        secret_values=[secret],
    ) == {"paths": [f"/tmp/{SECRET_PLACEHOLDER}/report.txt"]}


def test_tool_display_paths_allow_long_scratchpad_absolute_and_windows_forms() -> None:
    scratchpad_path = (
        "/var/folders/ny/myl4qfms7svd_wp4yywp3nzr0000gn/T/kolega-code-501/"
        "kolega-code-9dacd4aeafb1bdcd72578a1a/c0d65426a0c84a3cba954bc7987a2ef4/"
        + ("nested/" * 20)
        + "final-output.json"
    )
    result = build_tool_display("read", {"file_path": scratchpad_path}, project_path="/project")
    assert result == {"paths": [scratchpad_path]}
    assert len(result["paths"][0]) > MAX_SUBJECT_LENGTH
    assert result["paths"][0].endswith("/final-output.json")
    assert build_tool_display(
        "read", {"file_path": r"C:\scratchpad\logs\final-output.json"}, project_path="/project"
    ) == {"paths": [r"C:\scratchpad\logs\final-output.json"]}
    assert build_tool_display(
        "read",
        {"file_path": r"\\server\share\scratchpad\final-output.json"},
        project_path="/project",
    ) == {"paths": [r"\\server\share\scratchpad\final-output.json"]}


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("unknown", {"path": "src/app.py"}),
        ("web_fetch", {"url": "https://example.test/path"}),
        ("exec_command", {"title": "ls"}),
        ("read", {"file_path": "/project/unsafe;name.py"}),
        (
            "apply_patch",
            "*** Begin Patch\n*** Update File: /project/ok.py\n*** Update File: --password=nope\n*** End Patch\n",
        ),
    ],
)
def test_tool_display_unsupported_and_unsafe_inputs_fail_closed(name: str, args: object) -> None:
    assert build_tool_display(name, args, project_path="/project") == {}


def test_tool_display_preserves_benign_multiline_commands() -> None:
    command = "cd src\npytest -q tests/test_tool_subjects.py\n"
    assert build_tool_display("exec_command", {"command": command}) == {"command": command}
    assert sanitize_tool_display({"command": command}) == {"command": command}


def test_tool_display_suppresses_inline_code_and_heredoc_bodies() -> None:
    inline = build_tool_display("exec_command", {"command": "python -c 'print(1)'"}).get("command", "")
    assert isinstance(inline, str)
    assert inline == f"python {SECRET_PLACEHOLDER}"
    heredoc = build_tool_display("exec_command", {"command": "cat <<'EOF'\nhello\nworld\nEOF\nprintf done\n"}).get(
        "command", ""
    )
    assert isinstance(heredoc, str)
    assert "hello" not in heredoc and "world" not in heredoc
    assert heredoc.startswith("cat <<'EOF'\n")
    assert f"\n{SECRET_PLACEHOLDER}\nEOF\nprintf done\n" in heredoc


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'END-DATA'\nPRIVATE_FILE_BODY\nEND-DATA\n",
        "cat <<\\EOF\nPRIVATE_FILE_BODY\nEOF\n",
        "cat <<END-DATA\nEND\nPRIVATE_FILE_BODY\nEND-DATA\n",
        "cat <<'EOF'X\nEOF\nPRIVATE_FILE_BODY\nEOFX\n",
        "cat <<EOF <<'END-DATA'\nfirst\nEOF\nPRIVATE_FILE_BODY\nEND-DATA\n",
        "cat <<<PRIVATE_FILE_BODY",
    ],
)
def test_tool_display_unsupported_heredocs_fail_closed(command: str) -> None:
    assert build_tool_display("exec_command", {"command": command}) == {}
    assert sanitize_tool_display({"command": command}) == {}


@pytest.mark.parametrize(
    "suffix",
    [
        "--password PRIVATE_ARGUMENT",
        "$(printf PRIVATE_ARGUMENT)",
        "`printf PRIVATE_ARGUMENT`",
        "<(printf PRIVATE_ARGUMENT)",
        "-c 'print(\"PRIVATE_ARGUMENT\")'",
    ],
)
@pytest.mark.parametrize(
    "prefix",
    [
        "echo FAKE_CONFIGURED_'CREDENTIAL'",
        "curl 'https://fake-user:FAKE_PASSWORD@example.test/path?token=FAKE_QUERY'",
    ],
)
def test_tool_display_sanitizes_retained_prefix_before_suppressing_payload(prefix: str, suffix: str) -> None:
    command = f"{prefix} python {suffix}"
    secrets = ["FAKE_CONFIGURED_CREDENTIAL"]
    built = build_tool_display("exec_command", {"command": command}, secret_values=secrets)
    sanitized = sanitize_tool_display({"command": command}, secret_values=secrets)
    assert built == sanitized
    assert built
    displayed = str(built["command"])
    assert "FAKE" not in displayed
    assert "CREDENTIAL" not in displayed
    assert "PRIVATE_ARGUMENT" not in displayed
    assert "fake-user" not in displayed
    assert sanitize_tool_display(built) == built


@pytest.mark.parametrize(
    "command",
    [
        "curl 'https://fake-user:fake-password@example.test/path?token=FAKE_QUERY#FAKE_FRAGMENT'\nprintf ok\n",
        "curl --data 'password=FAKE_CREDENTIAL_TAIL' https://example.test/path\nprintf ok\n",
        "env API_KEY='FAKE CREDENTIAL TAIL' pytest -q\nprintf ok\n",
    ],
)
def test_tool_display_redacts_urls_credentials_and_payloads(command: str) -> None:
    result = build_tool_display("exec_command", {"command": command}).get("command", "")
    assert "FAKE" not in result
    assert "CREDENTIAL" not in result
    assert "TAIL" not in result
    assert "fake-password" not in result
    assert "token=" not in result


def test_tool_display_sanitizer_is_strict_and_literal() -> None:
    assert sanitize_tool_display({"paths": ["src/main.py", "src/main.py"]}) == {"paths": ["src/main.py"]}
    assert sanitize_tool_display({"paths": ["/abs/path"]}) == {"paths": ["/abs/path"]}
    assert sanitize_tool_display({"paths": ["curl --password nope"]}) == {}
    assert sanitize_tool_display({"paths": ["--password=nope"]}) == {}
    assert sanitize_tool_display({"command": ["ls"]}) == {}
    assert sanitize_tool_display({"command": "unterminated '"}) == {}
    assert sanitize_tool_display({"command": "printf '[bold]x[/bold]\\n'"}) == {"command": "printf '[bold]x[/bold]\\n'"}
    assert sanitize_tool_display({"paths": ["src/main.py"], "command": "ls"}) == {}


def test_tool_display_size_bounds_and_secret_boundary() -> None:
    assert sanitize_tool_display({"command": "a" * 100_000}) == {}
    secret = "FAKE_CONFIGURED_CREDENTIAL"
    assert sanitize_tool_display({"command": "x" * 230 + secret}, secret_values=[secret]) == {
        "command": "x" * 230 + SECRET_PLACEHOLDER
    }
    assert build_tool_display("exec_command", {"command": "x" * 16_380 + secret}, secret_values=[secret]) == {}


def test_tool_display_command_redacts_quote_split_known_secret_and_roundtrips() -> None:
    secret = "FAKE_CONFIGURED_CREDENTIAL"
    built = build_tool_display(
        "exec_command",
        {"command": "echo FAKE_CONFIGURED_'CREDENTIAL'\nprintf ok\n"},
        secret_values=[secret],
    )
    assert built == {"command": f"echo {SECRET_PLACEHOLDER}\nprintf ok\n"}
    assert sanitize_tool_display(built, secret_values=[secret]) == built
    assert sanitize_tool_display(sanitize_tool_display(built, secret_values=[secret]), secret_values=[secret]) == built


@pytest.mark.parametrize("tool", ["read", "apply_patch"])
def test_display_redacts_before_lexical_path_normalization(tool: str) -> None:
    secret = "FAKE//PRIVATE/VALUE"
    path = f"/var/tmp/{secret}/file.py"
    arguments = {"file_path": path} if tool == "read" else {"patch": f"*** Update File: {path}\n"}
    built = build_tool_display(tool, arguments, secret_values=[secret])
    assert built == {"paths": [f"/var/tmp/{SECRET_PLACEHOLDER}/file.py"]}


def test_tool_display_path_roundtrips_with_absolute_redacted_path() -> None:
    secret = "FAKE_CONFIGURED_CREDENTIAL"
    built = build_tool_display(
        "read",
        {"file_path": f"/var/tmp/{secret}/artifact.log"},
        project_path="/project",
        secret_values=[secret],
    )
    assert built == {"paths": [f"/var/tmp/{SECRET_PLACEHOLDER}/artifact.log"]}
    assert sanitize_tool_display(built, secret_values=[secret]) == built
    assert sanitize_tool_display(sanitize_tool_display(built, secret_values=[secret]), secret_values=[secret]) == built
