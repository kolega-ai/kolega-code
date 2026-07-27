"""``kolega-code share export``: what actually leaves the machine.

Export is the one command that turns a private session into a file meant to be
handed to somebody else, so these are security tests. They assert on the bytes
of the written bundle rather than on the redactor in isolation, because the leak
that mattered was not a broken redactor — it was a caller that never told the
redactor what its secrets were.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import re
from pathlib import Path

import pytest

from kolega_code.cli.main import main
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.events import AgentEvent, KnownEventType

#: A shape no shipped pattern recognises, which is the whole point: the real
#: Fireworks, Tavily, and bare-hex provider keys all look like this.
UNPATTERNED_KEY = "fw_3ZUftNOTAREALKEYvalue"


def _session_that_printed(store: SessionStore, project: Path, text: str) -> str:
    """Record a session whose tool output contains ``text``.

    Synchronous because ``main`` owns an event loop of its own, so these cannot
    be async tests.
    """
    record = store.create(project, "code", {"model": "test"}, title="leaky")
    events = FileSessionEventStore(store.journal(record.session_id))

    async def seed() -> None:
        await events.append(
            AgentEvent(
                session_id=record.session_id,
                sender="agent",
                event_type=KnownEventType.TURN_STARTED,
                content={"turn_id": "t1", "user_text": "print the config"},
            )
        )
        await events.append(
            AgentEvent(
                session_id=record.session_id,
                sender="agent",
                event_type=KnownEventType.CHAT_MESSAGE,
                content={"message_type": "tool_result", "text": text, "tool_description": "exec_command"},
                elapsed_ms=10,
            )
        )

    asyncio.run(seed())
    return record.session_id


def _single_file_payload(path: Path) -> str:
    """Everything a recipient can read out of a one-file replay."""
    document = path.read_text(encoding="utf-8")
    match = re.search(r"globalThis\.__KC_REPLAY__ = (\{.*?\});</script>", document, re.S)
    assert match is not None, "the single file no longer embeds its payload where this test looks"
    payload = json.loads(match.group(1).replace("\\u003c", "<"))
    events = gzip.decompress(base64.b64decode(payload["events"])).decode("utf-8")
    return "\n".join([document, events, json.dumps(payload["manifest"])])


def test_export_redacts_configured_api_keys_that_match_no_pattern(
    tmp_path: Path, capsys, isolated_cli_env: None
) -> None:
    """The CLI knows every key it holds, so it must not make the redactor guess.

    ``export_bundle`` has always accepted ``extra_secrets`` and the only caller
    never passed it, leaving detection to pattern matching alone. Keys whose
    shape nobody anticipated were exported in clear.
    """
    state = tmp_path / "state"
    store = SessionStore(root=state)
    settings_store = SettingsStore(root=state)
    settings_store.save(CliSettings(api_keys={"fireworks": UNPATTERNED_KEY}))
    session_id = _session_that_printed(store, tmp_path / "project", f"FIREWORKS_TOKEN {UNPATTERNED_KEY} ok")

    out = tmp_path / "replay.html"
    assert main(["share", "export", session_id, "--out", str(out), "--state-dir", str(state)]) == 0

    assert UNPATTERNED_KEY not in _single_file_payload(out), "a configured API key was exported in clear"
    assert "‹secret›" in _single_file_payload(out)
    assert "best-effort" in capsys.readouterr().out


def test_export_rewrites_the_home_directory_out_of_every_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    """Single file, directory, and zip must all be free of the host's layout."""
    from kolega_code.web import redaction as redaction_module

    home = tmp_path / "home" / "someone"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    redaction_module._prefixes_for_home.cache_clear()

    state = tmp_path / "state"
    store = SessionStore(root=state)
    session_id = _session_that_printed(store, tmp_path / "project", f"ls {home}/git/private-thing")

    single = tmp_path / "one.html"
    directory = tmp_path / "dir"
    archive = tmp_path / "bundle.zip"
    for extra in (["--out", str(single)], ["--dir", "--out", str(directory)], ["--zip", "--out", str(archive)]):
        assert main(["share", "export", session_id, "--state-dir", str(state), *extra]) == 0

    assert str(home) not in _single_file_payload(single)
    assert str(home) not in (directory / "events.jsonl").read_text(encoding="utf-8")

    import zipfile

    with zipfile.ZipFile(archive) as bundle:
        assert str(home).encode() not in bundle.read("events.jsonl")


def test_zip_export_does_not_delete_a_neighbouring_directory(tmp_path: Path, isolated_cli_env: None) -> None:
    """The end-to-end form of the repro: ``--out backup.zip`` beside ``backup/``."""
    state = tmp_path / "state"
    store = SessionStore(root=state)
    session_id = _session_that_printed(store, tmp_path / "project", "nothing interesting")

    victim = tmp_path / "backup"
    (victim / "nested").mkdir(parents=True)
    (victim / "nested" / "keep.txt").write_text("precious")

    assert (
        main(["share", "export", session_id, "--zip", "--out", str(tmp_path / "backup.zip"), "--state-dir", str(state)])
        == 0
    )

    assert (victim / "nested" / "keep.txt").read_text() == "precious"
    assert (tmp_path / "backup.zip").is_file()
