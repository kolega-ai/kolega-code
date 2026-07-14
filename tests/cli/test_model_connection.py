from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kolega_code.cli.config import build_agent_config, key_status
from kolega_code.cli.model_connection import test_model_connection as run_model_connection_probe
from kolega_code.config import ModelProvider
from kolega_code.llm.exceptions import LLMAuthenticationError
from kolega_code.llm.models import Message, TextBlock


def _config(project: Path):
    return build_agent_config(
        project,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
        },
    )


@pytest.mark.asyncio
async def test_model_connection_sends_a_tiny_tool_free_request(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeClient:
        async def generate(self, **kwargs):
            calls["generate"] = kwargs
            return Message(role="assistant", content=[TextBlock(text="OK")])

    def factory(**kwargs):
        calls["factory"] = kwargs
        return FakeClient()

    config = _config(tmp_path)
    result = await run_model_connection_probe(config, client_factory=factory)

    assert result.ok is True
    assert config.long_context_config.model in result.message
    factory_args = calls["factory"]
    assert isinstance(factory_args, dict)
    assert factory_args["provider"] == ModelProvider.ANTHROPIC
    assert factory_args["api_key"] == "test-key"
    assert factory_args["max_retries"] == 0
    request = calls["generate"]
    assert isinstance(request, dict)
    assert request["max_completion_tokens"] == 32
    assert request["tools"] == []
    history = request["messages"]
    assert history[0].content[0].text == "Reply with OK."


@pytest.mark.asyncio
async def test_model_connection_normalizes_authentication_errors(tmp_path: Path) -> None:
    def factory(**_kwargs):
        raise LLMAuthenticationError("bad key", provider="anthropic")

    result = await run_model_connection_probe(_config(tmp_path), client_factory=factory)

    assert result.ok is False
    assert "could not authenticate" in result.message
    assert "test-key" not in result.message


@pytest.mark.asyncio
async def test_model_connection_times_out(tmp_path: Path) -> None:
    class SlowClient:
        async def generate(self, **_kwargs):
            await asyncio.sleep(1)
            return Message(role="assistant", content=[TextBlock(text="OK")])

    result = await run_model_connection_probe(
        _config(tmp_path), client_factory=lambda **_kwargs: SlowClient(), timeout=0.001
    )

    assert result.ok is False
    assert result.message == "Connection test timed out after 0.001 seconds."


def test_local_provider_credential_status_is_not_required(tmp_path: Path) -> None:
    assert key_status(ModelProvider.LLAMA.value, tmp_path) == "not required for the local provider"
