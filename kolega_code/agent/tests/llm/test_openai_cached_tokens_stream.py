import pytest

from kolega_code.agent.llm.providers.openai import OpenAIProvider
from kolega_code.agent.llm.models import Message, MessageHistory


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, text=None, is_final=False):
        self.choices = [_Choice(delta=_Delta(content=text), finish_reason='stop' if is_final else None)]
        self.usage = None


class _Usage:
    def __init__(self):
        self.prompt_tokens = 3019
        self.completion_tokens = 104
        self.total_tokens = 3123
        self.prompt_tokens_details = {'cached_tokens': 2048}


class _AsyncStream:
    def __init__(self):
        self._chunks = [_Chunk('hello '), _Chunk('world', is_final=True)]
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        if self._i == len(self._chunks):
            chunk.usage = _Usage()  # final chunk carries usage
        return chunk

    async def aclose(self):
        pass


# TODO: Fix after qwen-3-coder-plus PR is merged - needs OpenAI cache token extraction in streaming
@pytest.mark.asyncio
async def test_openai_stream_includes_cached_tokens(monkeypatch):
    provider = OpenAIProvider(api_key='sk-test', base_url='https://api.openai.com/v1')

    async def fake_create(*args, **kwargs):
        return _AsyncStream()

    monkeypatch.setattr(provider.async_client.chat.completions, 'create', fake_create)

    stream = await provider.stream(messages=MessageHistory([Message(role='user', content='hi')]))
    async with stream as s:
        async for _ in s:
            pass
        final = await s.get_final_message()
        assert final.usage_metadata['prompt_tokens'] == 3019
        assert final.usage_metadata['total_tokens'] == 3123
        assert final.usage_metadata['cache_read_input_tokens'] == 2048



