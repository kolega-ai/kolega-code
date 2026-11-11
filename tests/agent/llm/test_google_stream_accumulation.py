"""GoogleStreamWrapper accumulates streamed text in parts.

The wrapper joins the text fragments once at finalize instead of running
``self.final_content += chunk.text`` per chunk. That attribute concat is O(n^2)
because CPython's in-place optimization needs a refcount-1 local, so a long
Gemini stream copies the whole buffer on every chunk. These tests pin that the
joined result stays byte-identical to naive in-order concatenation.
"""

from types import SimpleNamespace

import pytest

from kolega_code.llm.models import TextBlock
from kolega_code.llm.providers.google import GoogleStreamWrapper


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


def _chunk(text):
    candidate = SimpleNamespace(content=None, finish_reason=None)
    return SimpleNamespace(text=text, usage_metadata=None, model_version=None, candidates=[candidate])


@pytest.mark.asyncio
async def test_stream_accumulation_is_byte_identical():
    parts = [f"part {i}; " for i in range(2_000)]
    wrapper = GoogleStreamWrapper(_AsyncStream([_chunk(part) for part in parts]), model="gemini-3.5-flash")

    async with wrapper as stream:
        chunks = [chunk async for chunk in stream]
        assert all(chunk.type == "text" for chunk in chunks)
        assert stream.final_content == "".join(parts)

        final = await stream.get_final_message()

    assert isinstance(final.content, list)
    text_blocks = [block for block in final.content if isinstance(block, TextBlock)]
    assert text_blocks[0].text == "".join(parts)


@pytest.mark.asyncio
async def test_empty_and_blank_chunks_do_not_change_output():
    wrapper = GoogleStreamWrapper(_AsyncStream([_chunk(""), _chunk(None), _chunk("tail")]), model="gemini-3.5-flash")

    async with wrapper as stream:
        async for _ in stream:
            pass
        assert stream.final_content == "tail"


@pytest.mark.asyncio
async def test_empty_stream_yields_empty_content():
    wrapper = GoogleStreamWrapper(_AsyncStream([]), model="gemini-3.5-flash")

    async with wrapper as stream:
        async for _ in stream:
            pass
        assert stream.final_content == ""
