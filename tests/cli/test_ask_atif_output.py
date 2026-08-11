"""`ask --atif-output` and `sessions export --format atif` end to end."""

import base64
import json

import pytest

from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.exceptions import LLMBillingError
from kolega_code.llm.models import ImageBlock, Message, TextBlock

from .test_ask_semantic_json import SemanticAskFakeAgent, _setup


def _validate(path):
    import atif

    document = json.loads(path.read_text(encoding="utf-8"))
    return atif.Trajectory.model_validate(document)


def test_saved_run_writes_a_validated_trajectory(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)
    out = tmp_path / "exports" / "trajectory.json"

    exit_code = main_module.main(
        ["ask", "run the tests", "--project", str(project), "--save", "--atif-output", str(out)]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    # Plain stdout is unchanged; the export notice goes to stderr.
    assert captured.out == "Checking the tests.All green.\n"
    assert "Wrote ATIF trajectory" in captured.err

    trajectory = _validate(out)
    assert trajectory.schema_version == "ATIF-v1.7"
    sources = [step.source for step in trajectory.steps]
    assert sources.count("agent") == 2
    assert trajectory.extra is not None
    assert trajectory.extra["kolega"]["status"]["outcome"] == "completed"


def test_unsaved_run_writes_a_trajectory_and_no_session_state(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)
    out = tmp_path / "trajectory.json"

    exit_code = main_module.main(["ask", "run the tests", "--project", str(project), "--atif-output", str(out)])
    assert exit_code == 0
    trajectory = _validate(out)
    assert [step.source for step in trajectory.steps].count("agent") == 2
    store = SessionStore()
    assert store.list() == []


def test_failed_run_still_produces_a_trajectory_and_keeps_exit_one(tmp_path, capsys, monkeypatch, isolated_cli_env):
    class BillingFailureAgent(SemanticAskFakeAgent):
        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            raise LLMBillingError("Insufficient Balance", provider="anthropic")
            yield {}

    main_module, project = _setup(tmp_path, monkeypatch, BillingFailureAgent)
    out = tmp_path / "trajectory.json"

    exit_code = main_module.main(["ask", "go", "--project", str(project), "--atif-output", str(out)])
    assert exit_code == 1
    trajectory = _validate(out)
    assert trajectory.extra is not None
    status = trajectory.extra["kolega"]["status"]
    assert status["outcome"] == "failed"
    assert status["error"]["code"] == "billing_error"


def test_conversion_failure_on_a_clean_run_exits_one(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)
    out = tmp_path / "trajectory.json"

    from kolega_code.cli import atif_export as atif_module

    def boom(document):
        raise atif_module.AtifExportError("forced failure")

    monkeypatch.setattr(atif_module, "validate_atif_document", boom)
    exit_code = main_module.main(["ask", "go", "--project", str(project), "--atif-output", str(out)])
    assert exit_code == 1
    assert not out.exists()
    assert "ATIF export failed" in capsys.readouterr().err


def test_sessions_export_atif_stdout_and_json_parity(tmp_path, capsys, monkeypatch, isolated_cli_env):
    import atif

    main_module, project = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)
    exit_code = main_module.main(["ask", "run the tests", "--project", str(project), "--save", "--json"])
    assert exit_code == 0
    session_id = json.loads(capsys.readouterr().out.splitlines()[0])["session_id"]

    exit_code = main_module.main(["sessions", "export", session_id, "--format", "atif"])
    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)
    atif.Trajectory.model_validate(document)
    assert document["session_id"] == session_id

    # The default replay snapshot is untouched by the new formats.
    exit_code = main_module.main(["sessions", "export", session_id])
    assert exit_code == 0
    assert capsys.readouterr().out == SessionStore().export(session_id)


def test_sessions_export_atif_with_images_needs_output(tmp_path, capsys, monkeypatch, isolated_cli_env):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
    )

    class ImageTurnAgent(SemanticAskFakeAgent):
        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            recorder = self.session_recorder
            assert recorder is not None
            recorder.start_turn(
                Message(
                    role="user",
                    content=[TextBlock("look"), ImageBlock("base64", "image/png", base64.b64encode(png).decode())],
                )
            )
            recorder.finish_turn("completed")
            yield {"type": "response", "content": "seen", "complete": True, "uuid": "r1"}

    main_module, project = _setup(tmp_path, monkeypatch, ImageTurnAgent)
    exit_code = main_module.main(["ask", "look", "--project", str(project), "--save", "--json"])
    assert exit_code == 0
    session_id = json.loads(capsys.readouterr().out.splitlines()[0])["session_id"]

    # Stdout export refuses before writing anything.
    exit_code = main_module.main(["sessions", "export", session_id, "--format", "atif"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "--output" in captured.err

    out = tmp_path / "img" / "trajectory.json"
    exit_code = main_module.main(["sessions", "export", session_id, "--format", "atif", "--output", str(out)])
    assert exit_code == 0
    trajectory = _validate(out)
    assert trajectory.has_multimodal_content()
    assets = sorted((out.parent / "trajectory.assets").iterdir())
    assert len(assets) == 1 and assets[0].read_bytes() == png


def test_sessions_export_events_jsonl_round_trip(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)
    exit_code = main_module.main(["ask", "run the tests", "--project", str(project), "--save", "--json"])
    assert exit_code == 0
    live = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]

    exit_code = main_module.main(["sessions", "export", live[0]["session_id"], "--format", "events-jsonl"])
    assert exit_code == 0
    exported = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert exported == live


def test_unknown_session_exits_two(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, _ = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)
    exit_code = main_module.main(["sessions", "export", "does-not-exist", "--format", "atif"])
    assert exit_code == 2


@pytest.mark.parametrize("bad_format", ["yaml", "atiff"])
def test_invalid_format_is_rejected_by_argparse(tmp_path, capsys, monkeypatch, isolated_cli_env, bad_format):
    main_module, _ = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)
    with pytest.raises(SystemExit) as excinfo:
        main_module.main(["sessions", "export", "x", "--format", bad_format])
    assert excinfo.value.code == 2
