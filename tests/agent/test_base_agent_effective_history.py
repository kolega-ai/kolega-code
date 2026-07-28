# ruff: noqa: F401,F811,E402
import base64
import io
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from PIL import Image

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.errors import MaxAgentIterationsExceeded
from kolega_code.cli.session_store import SessionStore
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMContextWindowExceededError,
    LLMInternalServerError,
    LLMRateLimitError,
)
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)

from .compaction_helpers import FakeLLM

# Load environment variables
load_dotenv()


class TestBaseAgent:
    def test_get_effective_history_preserves_thinking_blocks(self, base_agent):
        base_agent.history = MessageHistory(
            [
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="unsigned thinking"),
                        ThinkingBlock(thinking="signed thinking", signature="sig"),
                        RedactedThinkingBlock(data="encrypted-redacted-thinking"),
                        TextBlock(text="final answer"),
                    ],
                )
            ]
        )

        effective = base_agent.get_effective_history_for_llm()

        assert len(effective) == 1
        assert [block.type for block in effective[0].content] == [
            "thinking",
            "thinking",
            "redacted_thinking",
            "text",
        ]
        assert effective[0].content[0].thinking == "unsigned thinking"
        assert effective[0].content[1].thinking == "signed thinking"
        assert effective[0].content[1].signature == "sig"
        assert effective[0].content[2].data == "encrypted-redacted-thinking"

    def test_history_for_llm_drops_foreign_thinking_when_switching_to_anthropic(self, base_agent):
        tool_call = ToolCall(id="tool1", name="read_file", input={"path": "README.md"})
        base_agent.primary_model_config.provider = ModelProvider.ANTHROPIC
        base_agent.primary_model_config.model = "claude-opus-4-8"
        base_agent.supports_vision = True
        base_agent.history = MessageHistory(
            [
                Message(
                    role="assistant",
                    content=[ThinkingBlock(thinking="kimi reasoning", signature="kimi-sig"), tool_call],
                    usage_metadata={"provider": "kimi_coding"},
                ),
                Message(
                    role="user",
                    content=[ToolResult(tool_use_id="tool1", name="read_file", content="ok", is_error=False)],
                ),
            ]
        )

        history = base_agent._history_for_llm()

        assert not any(isinstance(block, ThinkingBlock) for block in history[0].content)
        # Dropped, not replaced: a text stand-in gets echoed back by the model.
        assert not any(isinstance(block, TextBlock) for block in history[0].content)
        assert len(history[0].content) == 1
        assert isinstance(history[0].content[0], ToolCall)
        assert history[1].role == "user"
        assert isinstance(history[1].content[0], ToolResult)
        assert history[1].content[0].tool_use_id == "tool1"

    def test_history_for_llm_preserves_fireworks_thinking_when_targeting_fireworks(self, base_agent):
        base_agent.primary_model_config.provider = ModelProvider.FIREWORKS
        base_agent.primary_model_config.model = "accounts/fireworks/models/glm-5p2"
        base_agent.supports_vision = False
        base_agent.history = MessageHistory(
            [
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="fireworks reasoning"),
                        TextBlock(text="final answer"),
                    ],
                    usage_metadata={"provider": "fireworks"},
                )
            ]
        )

        history = base_agent._history_for_llm()

        assert isinstance(history[0].content[0], ThinkingBlock)
        assert history[0].content[0].thinking == "fireworks reasoning"
        assert isinstance(history[0].content[1], TextBlock)
        assert history[0].content[1].text == "final answer"

    def test_history_for_llm_preserves_images_when_target_anthropic_supports_vision(self, base_agent):
        image = ImageBlock(image_type="base64", media_type="image/png", data="BASE64")
        nested_image = ImageBlock(image_type="base64", media_type="image/jpeg", data="BASE642")
        base_agent.primary_model_config.provider = ModelProvider.ANTHROPIC
        base_agent.primary_model_config.model = "claude-opus-4-8"
        base_agent.supports_vision = True
        base_agent.history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="look"), image]),
                Message(
                    role="user",
                    content=[
                        ToolResult(
                            tool_use_id="tool1",
                            name="read_image",
                            content=[nested_image],
                            is_error=False,
                        )
                    ],
                ),
            ]
        )

        history = base_agent._history_for_llm()

        assert history[0].content[1] is image
        assert history[1].content[0].content[0] is nested_image

    def test_anthropic_history_resize_preserves_hydrated_history_and_artifact(
        self,
        base_agent,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import kolega_code.utils.images as image_utils

        image = Image.effect_noise((160, 120), 80).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG")
        original_bytes = output.getvalue()
        original_base64 = base64.b64encode(original_bytes).decode("ascii")
        limit = 1_600
        assert len(original_base64) > limit

        project = tmp_path / "project"
        project.mkdir()
        store = SessionStore(tmp_path / "state")
        record = store.create(project, "code", {})
        store.recorder(record.session_id).record_context_message(
            Message(
                role="user",
                content=[ImageBlock(image_type="base64", media_type="image/jpeg", data=original_base64)],
            )
        )
        journal = store.journal(record.session_id)
        event = next(event for event in journal.read_events() if event.artifacts)
        artifact_ref = event.artifacts[0]
        artifact_before = journal.read_artifact(artifact_ref)
        assert artifact_before == original_bytes

        loaded = store.load(record.session_id)
        base_agent.restore_message_history(loaded.history)
        restored_image = base_agent.history[0].content[0]
        assert isinstance(restored_image, ImageBlock)
        assert restored_image.data == original_base64

        monkeypatch.setattr(image_utils, "ANTHROPIC_MAX_IMAGE_BASE64_BYTES", limit)
        base_agent.primary_model_config.provider = ModelProvider.ANTHROPIC
        base_agent.primary_model_config.model = "claude-opus-4-8"
        base_agent.supports_vision = True

        outbound = base_agent._history_for_llm()

        outbound_image = outbound[0].content[0]
        assert isinstance(outbound_image, ImageBlock)
        assert outbound_image.data != original_base64
        assert len(outbound_image.data) < limit
        assert restored_image.data == original_base64
        assert journal.read_artifact(artifact_ref) == artifact_before

    def test_get_effective_history_falls_back_when_no_compression(self, base_agent):
        # With no compression, effective == full history
        base_agent.history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="hi")]),
                Message(role="assistant", content=[TextBlock(text="yo")]),
            ]
        )
        eff = base_agent.get_effective_history_for_llm()
        assert len(eff) == 2

    def test_get_effective_history_after_markers(self, base_agent):
        base_agent.history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="a")]),
                Message(role="assistant", content=[TextBlock(text="b")]),
                Message(role="user", content=[TextBlock(text="c")]),
            ]
        )
        # Compact all three messages into a summary (empty verbatim tail).
        base_agent.conversation.apply_compaction("CONVERSATION HISTORY SUMMARY (compressed at ...)", split_point=3)
        eff = base_agent.get_effective_history_for_llm()
        # everything folded, tail empty -> just the summary
        assert len(eff) == 1
        assert "CONVERSATION HISTORY SUMMARY" in eff[0].content[0].text
