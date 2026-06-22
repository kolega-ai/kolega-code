"""Tests for vision mismatch warnings when attaching/sending images on non-vision models.

Covers all four image attachment entry points:
  1. Clipboard paste (Ctrl+Shift+V)  — _paste_clipboard_image_worker
  2. on_paste event (data-URI / file path) — ChatComposer.on_paste -> add_pending_image_attachment
  3. /attach <path> slash command      — _command_attach -> add_pending_image_attachment
  4. @file.png mention + submit        — _build_mention_attachments at submit time

Plus the centralised vision check in ``add_pending_image_attachment`` and the
submit-time pre-send gate in ``on_chat_composer_submitted``.
"""

import base64

import pytest

from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.session_store import SessionStore


def build_test_config(project):
    return build_agent_config(
        project,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
        },
    )


def _image_attachment(path: str = "clipboard") -> dict:
    return {
        "type": "image",
        "media_type": "image/png",
        "data": base64.b64encode(b"fake-image").decode("ascii"),
        "path": path,
    }


class _FakeConversation:
    def __init__(self, has_images: bool = False):
        self._has_images = has_images

    def has_image_blocks(self) -> bool:
        return self._has_images


class _FakeAgent:
    """Minimal agent stand-in exposing vision capability + conversation probe."""

    def __init__(self, *, supports_vision: bool, has_images: bool = False, model: str = "test-model"):
        self.supports_vision = supports_vision
        self.conversation = _FakeConversation(has_images)
        self.primary_model_config = type("Cfg", (), {"model": model})()


def _make_app(tmp_path, monkeypatch):
    """Build a KolegaCodeApp with a no-op FakeCoderAgent; agent set by the caller."""
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    return app


# ---------------------------------------------------------------------------
# add_pending_image_attachment — the centralised vision check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_pending_image_non_vision_shows_warning_hint(tmp_path, monkeypatch):
    """Non-vision model: warning-tone hint + transcript system message."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=False)
        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))
        entries: list = []
        app._add_conversation_entry = lambda entry: entries.append(entry)

        app.add_pending_image_attachment(_image_attachment("clipboard"))

        assert hints, "expected a composer hint"
        text, tone = hints[0]
        assert tone == "warning"
        assert "clipboard" in text
        assert "can't see images" in text
        # Transcript system message added.
        system = [e for e in entries if e.kind == "system"]
        assert system, "expected a system message in the transcript"
        assert system[0].tone == "warning"
        assert "does not support vision" in system[0].content


@pytest.mark.asyncio
async def test_add_pending_image_vision_shows_info_hint(tmp_path, monkeypatch):
    """Vision model: info-tone hint, no transcript system message."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=True)
        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))
        entries: list = []
        app._add_conversation_entry = lambda entry: entries.append(entry)

        app.add_pending_image_attachment(_image_attachment("photo.png"))

        assert hints, "expected a composer hint"
        text, tone = hints[0]
        assert tone == "info"
        assert "photo.png" in text
        assert entries == [], "vision model should not add a system message"


@pytest.mark.asyncio
async def test_multiple_attachments_non_vision_one_transcript_message(tmp_path, monkeypatch):
    """Multiple attachments on a non-vision model: one transcript message, hint updates."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=False)
        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))
        entries: list = []
        app._add_conversation_entry = lambda entry: entries.append(entry)

        app.add_pending_image_attachment(_image_attachment("img1.png"))
        app.add_pending_image_attachment(_image_attachment("img2.png"))

        # Two hint updates (each shows all attached names).
        assert len(hints) == 2
        assert "img1.png" in hints[0][0]
        assert "img2.png" in hints[1][0]
        assert "img1.png" in hints[1][0]  # both names in the latest hint
        # Only ONE transcript system message (deduped).
        system = [e for e in entries if e.kind == "system"]
        assert len(system) == 1


# ---------------------------------------------------------------------------
# _paste_clipboard_image_worker — warning not overwritten
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paste_clipboard_image_non_vision_warning_not_overwritten(tmp_path, monkeypatch):
    """The old bug: _paste_clipboard_image_worker showed a warning then add_pending_image_attachment
    overwrote it with an info hint. Now the warning survives because the check is centralised."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=False)

        # Mock the clipboard reader.
        async def _fake_read():
            return (b"fake-image", "image/png")

        monkeypatch.setattr(
            "kolega_code.cli.clipboard_image.read_clipboard_image",
            _fake_read,
        )

        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))
        entries: list = []
        app._add_conversation_entry = lambda entry: entries.append(entry)

        await app._paste_clipboard_image_worker()

        # The final hint must be a warning (not info) and mention vision.
        assert hints, "expected at least one hint"
        final_text, final_tone = hints[-1]
        assert final_tone == "warning", "warning should not be overwritten by info hint"
        assert "can't see images" in final_text
        # And a transcript system message.
        assert any(e.kind == "system" for e in entries)


# ---------------------------------------------------------------------------
# /attach slash command — warning on non-vision model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attach_command_non_vision_shows_warning(tmp_path, monkeypatch):
    """/attach on a non-vision model shows the vision warning (centralised check)."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=False)

        # Create a real image file for /attach to read.
        import struct
        import zlib

        # Minimal 1x1 PNG.
        def _minimal_png() -> bytes:
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(
                ">IHHBBBB", 13, 1, 1, 8, 2, 0, 0
            )  # length=13, w=1, h=1, depth=8, color=2 (RGB)
            ihdr_chunk = b"IHDR" + ihdr
            ihdr_crc = struct.pack(">I", zlib.crc32(ihdr_chunk) & 0xFFFFFFFF)
            ihdr_full = struct.pack(">I", 13) + ihdr_chunk + ihdr_crc
            raw = b"\x00\xff\x00\x00"  # filter=none, R=0, G=0, B=0
            idat_data = zlib.compress(raw)
            idat_chunk = b"IDAT" + idat_data
            idat_crc = struct.pack(">I", zlib.crc32(idat_chunk) & 0xFFFFFFFF)
            idat_full = struct.pack(">I", len(idat_data)) + idat_chunk + idat_crc
            iend_chunk = b"IEND"
            iend_crc = struct.pack(">I", zlib.crc32(iend_chunk) & 0xFFFFFFFF)
            iend_full = struct.pack(">I", 0) + iend_chunk + iend_crc
            return sig + ihdr_full + idat_full + iend_full

        img_path = tmp_path / "project" / "test.png"
        img_path.write_bytes(_minimal_png())

        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))
        entries: list = []
        app._add_conversation_entry = lambda entry: entries.append(entry)

        await app._command_attach("test.png")

        # Image was attached successfully and a warning was shown.
        assert app._pending_image_attachments, "image should be stashed"
        assert hints, "expected a hint"
        final_text, final_tone = hints[-1]
        assert final_tone == "warning"
        assert "can't see images" in final_text
        assert any(e.kind == "system" for e in entries)


# ---------------------------------------------------------------------------
# Submit-time pre-send gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_with_pending_image_non_vision_blocked(tmp_path, monkeypatch):
    """Submitting with a pending image on a non-vision model blocks the send at the CLI gate.
    No agent worker is spawned — the user sees a transcript message, not an API failure."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=False)
        app._pending_image_attachments.append(_image_attachment("photo.png"))

        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))
        entries: list = []
        app._add_conversation_entry = lambda entry: entries.append(entry)
        spawned: list = []

        def _track_worker(coro, *a, **k):
            spawned.append((coro, a, k))
            coro.close()  # avoid "coroutine never awaited" warning
            return None

        app.run_worker = _track_worker

        from kolega_code.cli.tui.widgets import ChatComposer

        composer = app.query_one("#composer", ChatComposer)
        composer.text = "describe this"

        # Simulate submit.
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, "describe this"))

        # No agent worker spawned (send blocked).
        assert spawned == [], "no agent worker should be spawned when the send is blocked"
        # The user's message must NOT appear in the transcript (it was never sent).
        kinds = [e.kind for e in entries]
        assert "user" not in kinds, "blocked message should not appear in the transcript"
        # A system warning IS added to the transcript.
        assert "system" in kinds
        system = [e for e in entries if e.kind == "system"]
        assert "does not support vision" in system[0].content
        # Composer hint shows the blocked message.
        assert hints
        assert "not sent" in hints[-1][0].lower() or "Not sent" in hints[-1][0]
        # Composer text is preserved so the user can edit and resend.
        assert composer.text == "describe this", "composer text should be preserved when blocked"
        # Pending attachment is preserved so the user can /detach it.
        assert app._pending_image_attachments, "pending attachment should be preserved when blocked"


@pytest.mark.asyncio
async def test_submit_with_mention_image_non_vision_blocked(tmp_path, monkeypatch):
    """@file.png mention resolved at submit time on a non-vision model: blocked at CLI gate."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=False)

        # Create an image file so the @mention resolves to an image attachment.
        import struct
        import zlib

        def _minimal_png() -> bytes:
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(">IHHBBBB", 13, 1, 1, 8, 2, 0, 0)
            ihdr_chunk = b"IHDR" + ihdr
            ihdr_crc = struct.pack(">I", zlib.crc32(ihdr_chunk) & 0xFFFFFFFF)
            ihdr_full = struct.pack(">I", 13) + ihdr_chunk + ihdr_crc
            raw = b"\x00\xff\x00\x00"
            idat_data = zlib.compress(raw)
            idat_chunk = b"IDAT" + idat_data
            idat_crc = struct.pack(">I", zlib.crc32(idat_chunk) & 0xFFFFFFFF)
            idat_full = struct.pack(">I", len(idat_data)) + idat_chunk + idat_crc
            iend_chunk = b"IEND"
            iend_crc = struct.pack(">I", zlib.crc32(iend_chunk) & 0xFFFFFFFF)
            iend_full = struct.pack(">I", 0) + iend_chunk + iend_crc
            return sig + ihdr_full + idat_full + iend_full

        img_path = tmp_path / "project" / "screenshot.png"
        img_path.write_bytes(_minimal_png())
        app.file_index.refresh()

        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))
        entries: list = []
        app._add_conversation_entry = lambda entry: entries.append(entry)
        spawned: list = []

        def _track_worker(coro, *a, **k):
            spawned.append((coro, a, k))
            coro.close()
            return None

        app.run_worker = _track_worker

        from kolega_code.cli.tui.widgets import ChatComposer

        composer = app.query_one("#composer", ChatComposer)
        text = "@screenshot.png describe this"
        composer.text = text

        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, text))

        # Send blocked — no agent worker.
        assert spawned == [], "no agent worker should be spawned"
        # The user's message must NOT appear in the transcript (it was never sent).
        kinds = [e.kind for e in entries]
        assert "user" not in kinds, "blocked message should not appear in the transcript"
        system = [e for e in entries if e.kind == "system"]
        assert system, "expected a system message"
        # Composer text preserved so the user can edit the @mention and resend.
        assert composer.text == text, "composer text should be preserved when blocked"


@pytest.mark.asyncio
async def test_submit_with_image_vision_model_proceeds(tmp_path, monkeypatch):
    """Submitting with a pending image on a vision model: send proceeds normally."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=True)
        app._pending_image_attachments.append(_image_attachment("photo.png"))

        entries: list = []
        app._add_conversation_entry = lambda entry: entries.append(entry)
        spawned: list = []

        def _track_worker(coro, *a, **k):
            spawned.append((coro, a, k))
            coro.close()
            return None

        app.run_worker = _track_worker

        from kolega_code.cli.tui.widgets import ChatComposer

        composer = app.query_one("#composer", ChatComposer)
        composer.text = "describe this"

        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, "describe this"))

        # Agent worker spawned (send proceeds).
        assert spawned, "agent worker should be spawned for vision model"

        # The composer hint must be cleared after a successful submit — it
        # should not linger during generation.
        from textual.containers import Horizontal

        row = app.query_one("#composer_hint_row", Horizontal)
        assert not row.display, "composer hint row should be hidden after submit"
        # And the detach button should be hidden (no pending attachments).
        from textual.widgets import Button

        btn = app.query_one("#detach_btn", Button)
        assert not btn.display, "detach button should be hidden after submit"


@pytest.mark.asyncio
async def test_submit_text_only_non_vision_with_history_images_proceeds(tmp_path, monkeypatch):
    """Text-only message on a non-vision model with image history: send proceeds.
    History images are stripped by _history_for_llm, not blocked by the CLI gate."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        # Non-vision model WITH image history, but NO new image attachments.
        app.agent = _FakeAgent(supports_vision=False, has_images=True)

        entries: list = []
        app._add_conversation_entry = lambda entry: entries.append(entry)
        spawned: list = []

        def _track_worker(coro, *a, **k):
            spawned.append((coro, a, k))
            coro.close()
            return None

        app.run_worker = _track_worker

        from kolega_code.cli.tui.widgets import ChatComposer

        composer = app.query_one("#composer", ChatComposer)
        composer.text = "follow up question"

        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, "follow up question"))

        # Agent worker spawned — the CLI gate only blocks NEW image attachments,
        # not history images (those are handled by _history_for_llm).
        assert spawned, "agent worker should be spawned for text-only message"


# ---------------------------------------------------------------------------
# /detach — remove pending image attachments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detach_clears_pending_attachments(tmp_path, monkeypatch):
    """/detach removes all pending image attachments and confirms via hint."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=False)
        app._pending_image_attachments.append(_image_attachment("photo.png"))
        app._pending_image_attachments.append(_image_attachment("chart.png"))

        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))

        await app._command_detach("")

        assert app._pending_image_attachments == [], "pending attachments should be cleared"
        assert hints, "expected a confirmation hint"
        text, tone = hints[-1]
        assert tone == "info"
        assert "photo.png" in text
        assert "chart.png" in text
        assert "Removed" in text


@pytest.mark.asyncio
async def test_detach_with_no_attachments(tmp_path, monkeypatch):
    """/detach with nothing pending shows an info hint, not an error."""
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=False)

        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))

        await app._command_detach("")

        assert app._pending_image_attachments == []
        assert hints, "expected an info hint"
        text, tone = hints[-1]
        assert tone == "info"
        assert "No pending" in text


@pytest.mark.asyncio
async def test_detach_command_registered(tmp_path, monkeypatch):
    """/detach is in the TUI command handler dict and the slash command list."""
    from kolega_code.cli.slash_commands import TUI_COMMAND_NAMES

    assert "/detach" in TUI_COMMAND_NAMES

    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        handlers = app._tui_command_handlers()
        assert "/detach" in handlers
        assert handlers["/detach"] == app._command_detach


# ---------------------------------------------------------------------------
# Detach × button — clickable UI element in the composer hint row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detach_button_visible_when_image_attached(tmp_path, monkeypatch):
    """The × button in the hint row is visible when there are pending image attachments."""
    from textual.widgets import Button

    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=True)
        app.add_pending_image_attachment(_image_attachment("photo.png"))

        btn = app.query_one("#detach_btn", Button)
        assert btn.display, "detach button should be visible when an image is attached"


@pytest.mark.asyncio
async def test_detach_button_hidden_when_no_attachments(tmp_path, monkeypatch):
    """The × button is hidden when there are no pending image attachments."""
    from textual.widgets import Button

    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=True)

        btn = app.query_one("#detach_btn", Button)
        assert not btn.display, "detach button should be hidden when no image is attached"


@pytest.mark.asyncio
async def test_detach_button_click_clears_attachments(tmp_path, monkeypatch):
    """Clicking the × button calls _command_detach and clears pending attachments."""
    from textual.widgets import Button

    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.agent = _FakeAgent(supports_vision=True)
        app.add_pending_image_attachment(_image_attachment("photo.png"))
        app.add_pending_image_attachment(_image_attachment("chart.png"))

        assert app._pending_image_attachments, "should have pending attachments"

        # Simulate clicking the detach button.
        btn = app.query_one("#detach_btn", Button)
        await app.on_button_pressed(Button.Pressed(btn))

        assert app._pending_image_attachments == [], "clicking × should clear attachments"

        # Button should now be hidden.
        assert not btn.display, "detach button should be hidden after clearing"
