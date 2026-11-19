#!/usr/bin/env python3
"""Measure a foreground sub-agent dispatch while the child streams its reply.

``AgentTool._execute_agent`` consumes ``process_message_stream`` chunks. Chunk
types the agent already mirrors onto the event stream (``response`` and
``thinking``) are not rebroadcast by the tool, so each chunk's cost is whatever
the dispatch loop itself does per delta. Run this before and after touching
that loop:

    uv run python scripts/bench_subagent_stream.py

The fake child yields ``chunks`` chunks of 50 characters, then one completing
chunk. Dispatch time must grow linearly with the streamed total.
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import sys
import tempfile
import time
import tracemalloc
import uuid
from typing import Any, ClassVar
from unittest.mock import AsyncMock, Mock

from kolega_code.agent.tool_backend.agent_tool import AgentTool
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.llm.models import Message, TextBlock

CHUNK_CHARS = 50
CASES = (2_000, 10_000, 40_000)
REPEATS = 3


def _config() -> AgentConfig:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig())
    return AgentConfig(
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        long_context_config=model,
        fast_config=model,
    )


def _caller() -> Mock:
    caller = Mock()
    caller.agent_name = "bench-caller"
    caller.log_info = AsyncMock()
    caller.current_tool_call_id = None
    caller.current_tool_execution_id = None
    caller.sub_agent = False
    caller.protected_files = []
    caller.workspace_env_var_descriptions = {}
    caller.workspace_memories = []
    caller.prompt_extensions = []
    caller.tool_extensions = []
    caller.usage_recorder = None
    caller.sub_agent_recorder = None
    caller.session_recorder = None
    caller.sub_agent_context = None
    caller.max_iterations = None
    caller.get_background_jobs = lambda: None
    return caller


class _StreamingChild:
    agent_name = "general-agent"
    chunks: ClassVar[int] = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.session_context_message = Message(
            role="user",
            content=[TextBlock('<system-reminder source="session">\nBench session context\n</system-reminder>')],
        )
        self._history = [self.session_context_message.to_dict()]
        self.total_tokens_used = 0
        self.cleanup = AsyncMock()

    async def process_message_stream(self, task: str):
        self._history.append({"role": "user", "content": [{"type": "text", "text": task}]})
        body = "x" * CHUNK_CHARS
        for _ in range(self.chunks):
            yield {"type": "response", "content": body, "complete": False, "uuid": "stream"}
        self._history.append({"role": "assistant", "content": [{"type": "text", "text": body}]})
        yield {"type": "response", "content": "done", "complete": True, "uuid": "stream"}

    def dump_message_history(self):
        return self._history

    async def recap_agent_outcome(self):
        return "ok"


async def _dispatch(chunks: int) -> float:
    from kolega_code.agent import generalagent

    _StreamingChild.chunks = chunks
    original = generalagent.GeneralAgent
    generalagent.GeneralAgent = _StreamingChild  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as project:
            tool = AgentTool(
                project_path=project,
                workspace_id="bench-workspace",
                thread_id=str(uuid.uuid4()),
                connection_manager=AsyncMock(),
                config=_config(),
                caller=_caller(),
            )
            started = time.perf_counter()
            result = await tool.dispatch_general_agent("bench task")
            elapsed = time.perf_counter() - started
        assert result == "ok", result
        return elapsed
    finally:
        generalagent.GeneralAgent = original


async def main() -> int:
    print(f"{'chunks':>7} {'streamed chars':>15} {'median ms':>10} {'peak MiB':>9}")
    for chunks in CASES:
        gc.collect()
        await _dispatch(chunks)  # warm imports and the recorder seam
        timings = [await _dispatch(chunks) for _ in range(REPEATS)]
        tracemalloc.start()
        await _dispatch(chunks)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(
            f"{chunks:>7} {chunks * CHUNK_CHARS:>15,} "
            f"{statistics.median(timings) * 1000:>10.1f} {peak / (1024 * 1024):>9.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
