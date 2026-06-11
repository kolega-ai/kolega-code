import os
import types
import pytest

from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.models import Message, MessageHistory

# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


class _UsageDetails:
    def __init__(self):
        self.cached_tokens = 2048


class _Usage:
    def __init__(self):
        self.prompt_tokens = 3019
        self.completion_tokens = 104
        self.total_tokens = 3123
        self.prompt_tokens_details = _UsageDetails()


class _ChoiceMsg:
    def __init__(self):
        self.content = 'ok'
        self.tool_calls = None
        self.finish_reason = 'stop'


class _Response:
    def __init__(self):
        self.usage = _Usage()
        self.choices = [types.SimpleNamespace(message=_ChoiceMsg())]


# TODO: Fix after qwen-3-coder-plus PR is merged - needs OpenAI cache token extraction from prompt_tokens_details
@pytest.mark.asyncio
@pytest.mark.skipif(SKIP_IN_CI, reason="Skipping slow test in CI environment")
async def test_openai_generate_includes_cached_tokens(monkeypatch):
    provider = OpenAIProvider(api_key='sk-test', base_url='https://api.openai.com/v1')

    async def fake_create(*args, **kwargs):
        return _Response()

    monkeypatch.setattr(provider.async_client.chat.completions, 'create', fake_create)

    messages = MessageHistory([Message(role='user', content='hi')])

    msg = await provider.generate(messages=messages)
    assert msg.usage_metadata['prompt_tokens'] == 3019
    assert msg.usage_metadata['completion_tokens'] == 104
    assert msg.usage_metadata['total_tokens'] == 3123
    assert msg.usage_metadata['cache_read_input_tokens'] == 2048



