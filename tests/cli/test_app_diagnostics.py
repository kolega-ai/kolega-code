# ruff: noqa: F401,E402
"""App-level diagnostics: the event tee persists structured LLM errors, and /bug bundles them."""

import json
from pathlib import Path

import pytest

from kolega_code.events import AgentEvent

from ._app_test_utils import _build_mention_test_app


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_event_tee_persists_structured_llm_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        assert app._diag is not None and app._diag.enabled
        # Simulate the agent emitting a structured error (e.g. the DeepSeek /v1 image 400).
        app._render_event(
            AgentEvent(
                event_type="llm_error",
                sender="agent",
                content={
                    "provider": "deepseek",
                    "model": "deepseek-v4-pro",
                    "endpoint": "https://api.deepseek.com/v1",
                    "http_status": 400,
                    "error_type": "LLMInvalidRequestError",
                    "message": "unknown variant `image_url`, expected `text`",
                },
            )
        )
        rows = _read(app._diag.path)

    errors = [r for r in rows if r["kind"] == "llm_error"]
    assert errors, "structured llm_error was not persisted to the diagnostics timeline"
    err = errors[0]
    assert err["http_status"] == 400
    assert err["provider"] == "deepseek"
    assert "api.deepseek.com/v1" in err["endpoint"]
    assert "image_url" in err["message"]


@pytest.mark.asyncio
async def test_diagnostics_and_bug_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._command_diagnostics("")
        await app._command_bug("")
        diag_dir = app._diag.dir

    # /bug wrote a shareable bundle with the summary + the full session JSON.
    bundles = list(diag_dir.glob("bug-*"))
    assert bundles, "/bug did not write a bundle"
    bundle = bundles[0]
    assert (bundle / "summary.md").exists()
    assert (bundle / "session.json").exists()

    system_entries = [e.content for e in app.conversation_entries if e.kind == "system"]
    assert any("Diagnostics log" in c for c in system_entries)  # /diagnostics summary
    assert any("bundle written to" in c for c in system_entries)  # /bug output
