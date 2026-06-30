"""OpenAIStreamWrapper accumulates streamed deltas into the final message.

The wrapper collects content / reasoning_content / tool-call argument fragments into
lists and ``''.join``s them once at finalize (instead of a per-chunk ``str +=``, which
is O(n^2) for large DeepSeek reasoning streams because the buffers are instance
attributes). These tests pin that the joined result is byte-identical to a naive
concatenation of every fragment, in order.
"""

import pytest

from kolega_code.llm.models import ThinkingBlock
from kolega_code.llm.providers.openai import OpenAIStreamWrapper


class _Fn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolCallDelta:
    def __init__(self, index, name=None, arguments=None, id=None):
        self.index = index
        self.id = id
        self.function = _Fn(name=name, arguments=arguments)


class _Delta:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls or []


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, delta, finish_reason=None):
        self.choices = [_Choice(delta, finish_reason)]
        self.usage = None


class _AsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_stream_accumulation_is_byte_identical():
    # A DeepSeek-shaped stream: a flood of tiny reasoning + content deltas, plus a tool
    # call whose JSON arguments arrive in fragments across several chunks.
    reasoning_parts = [f"r{i} " for i in range(500)]
    content_parts = [f"c{i} " for i in range(300)]
    arg_parts = ['{"path": "', "a/b/c.txt", '", "data": "', "x" * 100, '"}']

    chunks = []
    for r in reasoning_parts:
        chunks.append(_Chunk(_Delta(reasoning_content=r)))
    for c in content_parts:
        chunks.append(_Chunk(_Delta(content=c)))
    chunks.append(_Chunk(_Delta(tool_calls=[_ToolCallDelta(0, name="write", arguments=arg_parts[0], id="call_0")])))
    for frag in arg_parts[1:]:
        chunks.append(_Chunk(_Delta(tool_calls=[_ToolCallDelta(0, arguments=frag)])))
    chunks.append(_Chunk(_Delta(), finish_reason="tool_calls"))

    wrapper = OpenAIStreamWrapper(_AsyncStream(chunks), provider_name="deepseek")
    async with wrapper as stream:
        async for _ in stream:
            pass

        assert stream.final_reasoning_content == "".join(reasoning_parts)
        assert stream.final_content == "".join(content_parts)

        final = await stream.get_final_message()

    # Tool-call arguments are joined once at finalize, in arrival order.
    assert stream.final_tool_calls[0].function.arguments == "".join(arg_parts)

    # The final message carries the full reasoning as a ThinkingBlock and the answer text.
    thinking = [b for b in final.content if isinstance(b, ThinkingBlock)]
    assert thinking and thinking[0].thinking == "".join(reasoning_parts)
    assert final.tool_calls and final.tool_calls[0].name == "write"


@pytest.mark.asyncio
async def test_empty_stream_yields_empty_buffers():
    wrapper = OpenAIStreamWrapper(_AsyncStream([]), provider_name="deepseek")
    async with wrapper as stream:
        async for _ in stream:
            pass
        assert stream.final_content == ""
        assert stream.final_reasoning_content == ""
