"""Tests for the model-switch warning when a non-vision model inherits image history."""

import pytest

from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import DEEPSEEK_DEFAULT_MODEL
from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.models import ImageBlock, Message, TextBlock


def build_test_config(project):
    return build_agent_config(
        project,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
        },
    )


class _FakeConversation:
    def __init__(self, has_images: bool):
        self._has_images = has_images

    def has_image_blocks(self) -> bool:
        return self._has_images


class _FakeAgent:
    """Minimal agent stand-in exposing the vision capability + conversation probe."""

    def __init__(self, *, supports_vision: bool, has_images: bool):
        self.supports_vision = supports_vision
        self.conversation = _FakeConversation(has_images)


def _image_history() -> list:
    return [
        Message(
            role="user",
            content=[TextBlock(text="look"), ImageBlock(image_type="base64", media_type="image/png", data="ZmFrZQ==")],
        ).to_dict()
    ]


@pytest.mark.asyncio
async def test_switch_to_non_vision_model_with_image_history_warns(tmp_path, monkeypatch):
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

    async with app.run_test():
        # Install a non-vision agent with image history, as if the rebuild produced it.
        async def _fake_ensure(rebuild=False):
            app.agent = _FakeAgent(supports_vision=False, has_images=True)

        app._ensure_agent_from_settings = _fake_ensure
        app._populate_settings_controls = lambda: None
        app._restore_composer_placeholder = lambda: None
        app._notify_user = lambda *a, **k: None
        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))

        await app._switch_model("deepseek", DEEPSEEK_DEFAULT_MODEL)

        assert hints, "expected a composer hint warning for non-vision model with image history"
        text, tone = hints[0]
        assert "images from earlier turns" in text
        assert tone == "warning"


@pytest.mark.asyncio
async def test_switch_to_vision_model_with_image_history_no_warn(tmp_path, monkeypatch):
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

    async with app.run_test():
        async def _fake_ensure(rebuild=False):
            app.agent = _FakeAgent(supports_vision=True, has_images=True)

        app._ensure_agent_from_settings = _fake_ensure
        app._populate_settings_controls = lambda: None
        app._restore_composer_placeholder = lambda: None
        app._notify_user = lambda *a, **k: None
        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))

        await app._switch_model("anthropic", "claude-opus-4-8")

        assert hints == [], "vision-capable model should not trigger the image-history warning"


@pytest.mark.asyncio
async def test_switch_to_non_vision_model_without_image_history_no_warn(tmp_path, monkeypatch):
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

    async with app.run_test():
        async def _fake_ensure(rebuild=False):
            app.agent = _FakeAgent(supports_vision=False, has_images=False)

        app._ensure_agent_from_settings = _fake_ensure
        app._populate_settings_controls = lambda: None
        app._restore_composer_placeholder = lambda: None
        app._notify_user = lambda *a, **k: None
        hints: list[tuple] = []
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))

        await app._switch_model("deepseek", DEEPSEEK_DEFAULT_MODEL)

        assert hints == [], "no image history should not trigger the warning"
