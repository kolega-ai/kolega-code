# ruff: noqa: F401,F811,E402
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.errors import MaxAgentIterationsExceeded
from kolega_code.agent.prompt_provider import AgentMode, AgentType
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
    def test_build_prompt_context_loads_agents_md(self, base_agent, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Use AGENTS guidance", encoding="utf-8")

        context = base_agent.build_prompt_context()

        assert context.project_guidance_file == "AGENTS.md"
        assert context.project_guidance == "Use AGENTS guidance"
        assert context.kolega_md == "Use AGENTS guidance"

    def test_build_prompt_context_falls_back_to_kolega_md(self, base_agent, tmp_path):
        (tmp_path / "KOLEGA.md").write_text("Use legacy guidance", encoding="utf-8")

        context = base_agent.build_prompt_context()

        assert context.project_guidance_file == "KOLEGA.md"
        assert context.project_guidance == "Use legacy guidance"

    def test_build_prompt_context_prefers_agents_md(self, base_agent, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Use canonical guidance", encoding="utf-8")
        (tmp_path / "KOLEGA.md").write_text("Ignore legacy guidance", encoding="utf-8")

        context = base_agent.build_prompt_context()

        assert context.project_guidance_file == "AGENTS.md"
        assert context.project_guidance == "Use canonical guidance"
        assert "Ignore legacy guidance" not in context.project_guidance

    def test_build_prompt_context_without_guidance(self, base_agent):
        context = base_agent.build_prompt_context()

        assert context.project_guidance_file == ""
        assert context.project_guidance == ""

    def test_build_prompt_context_ignores_removed_agent_memory_file(self, base_agent, tmp_path):
        legacy_content = "Legacy repository memory must not reach the model."
        (tmp_path / "AGENT_MEMORY.md").write_text(legacy_content, encoding="utf-8")

        context = base_agent.build_prompt_context()
        prompt = base_agent.build_agent_system_prompt(AgentType.CODER, AgentMode.CLI)

        assert not hasattr(context, "agent_memory")
        assert legacy_content not in prompt
