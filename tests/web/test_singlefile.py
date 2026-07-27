"""A single-file replay must open by double-click, which is the whole point.

The directory bundle is unopenable from ``file://`` — browsers block module
imports and ``fetch`` on that origin — and it fails silently, because the script
that would report the error is the one that was blocked. That is what the single
file exists to fix, so the load-bearing test here drives a real browser at a real
``file://`` URL rather than asserting on the document's text.

The rest guard the inlining itself: no external reference may survive, redaction
must hold in the new output shape, and the template surgery must fail loudly if
someone edits ``player.html`` without updating the inliner.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

from kolega_code.events import AgentEvent, ArtifactPurpose, KnownEventType
from kolega_code.session.inmemory import InMemoryArtifactStore
from kolega_code.web import singlefile
from kolega_code.web.bundle import export_bundle
from kolega_code.web.singlefile import SingleFileError, build_single_file

SECRET = "sk-live-51H9xTOTALLYSECRETvalue0000"

#: A 1x1 transparent PNG, so the image path is exercised with real image bytes.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _event(event_type: str, seq: int, *, elapsed_ms: int = 0, **content) -> AgentEvent:
    return AgentEvent(
        session_id="s1",
        sender="agent",
        event_type=event_type,
        content=dict(content),
        seq=seq,
        elapsed_ms=elapsed_ms,
    )


def _session() -> list[AgentEvent]:
    return [
        _event(KnownEventType.TURN_STARTED, 1, turn_id="t1", user_text="ship the thing"),
        _event(KnownEventType.ASSISTANT_DELTA, 2, elapsed_ms=1000, text="Working on it. ", complete=False),
        _event(KnownEventType.ASSISTANT_DELTA, 3, elapsed_ms=2000, text="Done.", complete=True),
        _event(KnownEventType.TURN_ENDED, 4, elapsed_ms=2500, turn_id="t1", status="completed"),
    ]


@pytest.mark.asyncio
async def test_export_defaults_to_one_html_file(tmp_path: Path) -> None:
    result = await export_bundle(_session(), tmp_path / "replay.html", session_id="s1", single_file=True)

    assert result.single_file is True
    assert result.path == tmp_path / "replay.html"
    # One file, and nothing beside it: a directory would not survive being emailed.
    assert [item.name for item in tmp_path.iterdir()] == ["replay.html"]


@pytest.mark.asyncio
async def test_single_file_has_no_external_references(tmp_path: Path) -> None:
    """Anything fetched at runtime is a blank page on file://, so nothing may be."""
    await export_bundle(_session(), tmp_path / "replay.html", session_id="s1", single_file=True)
    document = (tmp_path / "replay.html").read_text(encoding="utf-8")

    # The fetch and import paths survive in the inlined source as dead branches,
    # so this asserts on what the browser would actually load, not on the text.
    assert "<script" in document
    assert not re.search(r"""<link\b[^>]*\brel=["']stylesheet""", document)
    assert not re.search(r"""(?:src|href)=["'](?!data:|#)""", document), "found a reference to an external resource"
    assert "globalThis.__KC_REPLAY__" in document


@pytest.mark.asyncio
async def test_single_file_scrubs_secrets(tmp_path: Path) -> None:
    """Redaction runs before inlining, including inside the compressed payload."""
    events = _session()
    events.append(_event(KnownEventType.ASSISTANT_DELTA, 5, elapsed_ms=3000, text=f"key {SECRET}", complete=True))

    result = await export_bundle(events, tmp_path / "replay.html", session_id="s1", single_file=True)

    document = (tmp_path / "replay.html").read_text(encoding="utf-8")
    assert SECRET not in document
    assert result.report.strings_redacted >= 1


@pytest.mark.asyncio
async def test_only_images_are_embedded(tmp_path: Path) -> None:
    """Text artifacts are already described by their preview; images are not."""
    store = InMemoryArtifactStore()
    image = await store.put(PNG_1X1, media_type="image/png", purpose=ArtifactPurpose.IMAGE, encoding="binary")
    text = await store.put(
        b"a long tool result",
        media_type="text/plain",
        purpose=ArtifactPurpose.TOOL_RESULT,
        encoding="utf-8",
        chars=18,
    )
    events = _session()
    events.append(
        AgentEvent(
            session_id="s1",
            sender="agent",
            event_type=KnownEventType.CHAT_MESSAGE,
            content={"text": "captured a screenshot"},
            seq=5,
            elapsed_ms=3000,
            artifacts=[image, text],
        )
    )

    result = await export_bundle(
        events, tmp_path / "replay.html", session_id="s1", artifact_store=store, single_file=True
    )

    document = (tmp_path / "replay.html").read_text(encoding="utf-8")
    assert result.artifact_count == 1
    assert "data:image/png;base64," in document
    assert image.sha256 in document
    assert text.sha256 not in document


def test_inliner_refuses_a_template_it_no_longer_understands() -> None:
    """A silent mismatch would ship a blank page, so the surgery asserts its markers."""
    with pytest.raises(SingleFileError, match="player.css link"):
        original = singlefile._read_asset

        def missing(name: str) -> str:
            source = original(name)
            return source.replace(singlefile._PLAYER_LINK, "") if name == "player.html" else source

        singlefile._read_asset = missing  # pyright: ignore[reportAttributeAccessIssue]
        try:
            build_single_file(manifest={}, events_jsonl=b"", theme_css="")
        finally:
            singlefile._read_asset = original  # pyright: ignore[reportAttributeAccessIssue]


def test_payload_cannot_break_out_of_its_script_tag() -> None:
    """A session that discusses HTML must not be able to close the tag holding it."""
    document = build_single_file(
        manifest={"title": "</script><img src=x onerror=alert(1)>"},
        events_jsonl=b"",
        theme_css="",
    )
    assert "</script><img" not in document
    assert "\\u003c/script>" in document or "\\u003c/script\\u003e" in document


# --------------------------------------------------------------------- browser ---

playwright_api = pytest.importorskip("playwright.async_api", reason="playwright is required for player tests")


@pytest.mark.asyncio
async def test_double_clicking_the_file_plays_the_replay(tmp_path: Path) -> None:
    """The end-to-end promise: a recipient opens the file and sees the session.

    Loaded over ``file://`` with no server anywhere, which is the exact condition
    the directory bundle fails under.
    """
    await export_bundle(_session(), tmp_path / "replay.html", session_id="s1", title="shipped", single_file=True)

    async with playwright_api.async_playwright() as driver:
        try:
            browser = await driver.chromium.launch()
        except Exception as exc:  # pragma: no cover - environment without chromium
            pytest.skip(f"chromium is unavailable: {exc}")
        page = await browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("requestfailed", lambda request: errors.append(f"failed request: {request.url}"))

        await page.goto((tmp_path / "replay.html").as_uri(), wait_until="load")
        await page.wait_for_selector("#scrub")
        # Seek to the end: proves the gzipped payload decoded and the fold ran,
        # not merely that the document's static shell rendered.
        await page.eval_on_selector(
            "#scrub", "node => { node.value = node.max; node.dispatchEvent(new Event('input')); }"
        )
        await page.wait_for_function("document.querySelectorAll('.kc-entry').length > 0")

        text = await page.eval_on_selector("#transcript", "node => node.textContent")
        banner = await page.eval_on_selector("#error", "node => node.textContent.trim()")
        await browser.close()

    assert "ship the thing" in text
    assert "Done." in text
    assert banner == ""
    assert errors == []
