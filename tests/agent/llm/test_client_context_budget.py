"""Strict per-run context budget enforcement in LLMClient/InstrumentedLLMClient.

When the paired caps (context_window_tokens + max_output_tokens) are propagated
from AgentConfig through AgentContext.create_llm_client, generate()/stream()
cap the effective output maximum, count the fully rendered request, and raise
LLMContextWindowExceededError before any provider generation when the input
exceeds context_window_tokens - effective_output (equality allowed). Without
the pair, behavior is byte-for-byte the legacy path (no counting, params
untouched).
"""

from typing import Any, AsyncContextManager, Coroutine, Dict, cast
from unittest.mock import MagicMock

import pytest

from kolega_code.llm.client import GenerationParams, LLMClient, TokenCount
from kolega_code.llm.exceptions import LLMContextWindowExceededError
from kolega_code.llm.instrumented_client import InstrumentedLLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolDefinition

WINDOW = 1000
RUN_OUTPUT = 200

MESSAGES = MessageHistory([Message("user", [TextBlock("Hello")])])
SYSTEM = Message("system", [TextBlock("You are helpful.")])
TOOLS = [ToolDefinition(name="echo", description="Echo input", parameters=[])]


def _client(**overrides) -> LLMClient:
    kwargs: Dict[str, Any] = dict(
        provider="anthropic",
        api_key="test-key",
        context_window_tokens=WINDOW,
        max_output_tokens=RUN_OUTPUT,
    )
    kwargs.update(overrides)
    return LLMClient(**kwargs)


def _mock_provider(client: LLMClient, input_tokens: int) -> Dict[str, Any]:
    """Patch the client's provider with recording mocks; return captured state."""
    captured: Dict[str, Any] = {"count_calls": 0, "generate_calls": 0, "stream_calls": 0}

    async def _count_tokens(messages=None, system=None, model=None, tools=None, thinking=None, **kwargs):
        captured["count_calls"] += 1
        captured["count_args"] = {
            "messages": messages,
            "system": system,
            "model": model,
            "tools": tools,
            "thinking": thinking,
        }
        return TokenCount(input_tokens=input_tokens)

    async def _generate(messages, system, params, **kwargs):
        captured["generate_calls"] += 1
        captured["generate_params"] = params
        captured["generate_kwargs"] = kwargs
        return Message("assistant", [TextBlock("ok")])

    async def _stream(messages, system, params, **kwargs):
        captured["stream_calls"] += 1
        captured["stream_params"] = params
        return _FakeStreamCM()

    client.provider.count_tokens = _count_tokens  # type: ignore[assignment]
    client.provider.generate = _generate  # type: ignore[assignment]
    client.provider.stream = _stream  # type: ignore[assignment]
    return captured


class _FakeStreamCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return Message("assistant", [TextBlock("summary text")])


async def _open_stream(client: LLMClient, *args, **kwargs) -> AsyncContextManager[Any]:
    """Await client.stream()'s union return (every concrete provider is async)."""
    return await cast(Coroutine[Any, Any, AsyncContextManager[Any]], client.stream(*args, **kwargs))


def _mock_langfuse():
    langfuse = MagicMock()
    generation = MagicMock()
    trace = MagicMock()
    trace.start_generation = MagicMock(return_value=generation)
    langfuse.start_span = MagicMock(return_value=trace)
    return langfuse, trace, generation


# ---------------------------------------------------------------------------
# Constructor pairing
# ---------------------------------------------------------------------------


def test_constructor_requires_paired_caps():
    with pytest.raises(ValueError, match="supplied together"):
        LLMClient(provider="anthropic", api_key="test-key", context_window_tokens=WINDOW)
    with pytest.raises(ValueError, match="supplied together"):
        LLMClient(provider="anthropic", api_key="test-key", max_output_tokens=RUN_OUTPUT)


# ---------------------------------------------------------------------------
# generate(): effective output maximum
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_caps_caller_output_to_run_max():
    client = _client()
    captured = _mock_provider(client, input_tokens=100)
    await client.generate(MESSAGES, SYSTEM, max_completion_tokens=500, model="m")
    assert captured["generate_params"].max_completion_tokens == RUN_OUTPUT


@pytest.mark.asyncio
async def test_generate_uses_run_max_when_caller_supplies_none():
    client = _client()
    captured = _mock_provider(client, input_tokens=100)
    await client.generate(MESSAGES, SYSTEM, model="m")
    assert captured["generate_params"].max_completion_tokens == RUN_OUTPUT


@pytest.mark.asyncio
async def test_generate_keeps_smaller_caller_output_limit():
    client = _client()
    captured = _mock_provider(client, input_tokens=100)
    await client.generate(MESSAGES, SYSTEM, max_completion_tokens=50, model="m")
    assert captured["generate_params"].max_completion_tokens == 50


@pytest.mark.asyncio
async def test_generate_input_ceiling_derives_from_effective_output():
    # 900 input tokens fit when the caller's smaller output limit (50) raises
    # the input ceiling to 950...
    client = _client()
    captured = _mock_provider(client, input_tokens=900)
    await client.generate(MESSAGES, SYSTEM, max_completion_tokens=50, model="m")
    assert captured["generate_calls"] == 1

    # ...but not when the run-wide output reservation (200) lowers it to 800.
    client = _client()
    captured = _mock_provider(client, input_tokens=900)
    with pytest.raises(LLMContextWindowExceededError):
        await client.generate(MESSAGES, SYSTEM, model="m")
    assert captured["generate_calls"] == 0


# ---------------------------------------------------------------------------
# generate(): boundary and rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_exact_boundary_is_allowed():
    client = _client()
    captured = _mock_provider(client, input_tokens=WINDOW - RUN_OUTPUT)
    await client.generate(MESSAGES, SYSTEM, model="m")
    assert captured["generate_calls"] == 1


@pytest.mark.asyncio
async def test_generate_over_limit_rejected_before_provider_invocation():
    client = _client()
    captured = _mock_provider(client, input_tokens=WINDOW - RUN_OUTPUT + 1)
    with pytest.raises(LLMContextWindowExceededError):
        await client.generate(MESSAGES, SYSTEM, model="m")
    assert captured["generate_calls"] == 0
    assert captured["count_calls"] == 1


@pytest.mark.asyncio
async def test_generate_counts_fully_rendered_request():
    client = _client()
    captured = _mock_provider(client, input_tokens=100)
    params = GenerationParams(max_completion_tokens=100, tools=TOOLS, thinking="high")
    await client.generate(MESSAGES, SYSTEM, params=params, model="claude-test")
    count_args = captured["count_args"]
    assert count_args["messages"] is MESSAGES
    assert count_args["system"] is SYSTEM
    assert count_args["tools"] is TOOLS
    assert count_args["model"] == "claude-test"
    assert count_args["thinking"] == "high"


@pytest.mark.asyncio
async def test_generate_does_not_mutate_supplied_params():
    client = _client()
    captured = _mock_provider(client, input_tokens=100)
    params = GenerationParams(temperature=0.3, max_completion_tokens=500, tools=TOOLS)
    await client.generate(MESSAGES, SYSTEM, params=params, model="m")
    assert params.max_completion_tokens == 500
    sent = captured["generate_params"]
    assert sent is not params
    assert sent.max_completion_tokens == RUN_OUTPUT
    assert sent.temperature == 0.3
    assert sent.tools is TOOLS


# ---------------------------------------------------------------------------
# generate(): precomputed count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_precomputed_count_skips_recount():
    client = _client()
    captured = _mock_provider(client, input_tokens=999_999)
    await client.generate(MESSAGES, SYSTEM, model="m", _precomputed_input_tokens=WINDOW - RUN_OUTPUT)
    assert captured["count_calls"] == 0
    assert captured["generate_calls"] == 1


@pytest.mark.asyncio
async def test_generate_over_limit_precomputed_count_rejected_without_recount():
    client = _client()
    captured = _mock_provider(client, input_tokens=0)
    with pytest.raises(LLMContextWindowExceededError):
        await client.generate(MESSAGES, SYSTEM, model="m", _precomputed_input_tokens=WINDOW - RUN_OUTPUT + 1)
    assert captured["count_calls"] == 0
    assert captured["generate_calls"] == 0


# ---------------------------------------------------------------------------
# stream(): same contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_caps_caller_output_to_run_max():
    client = _client()
    captured = _mock_provider(client, input_tokens=100)
    cm = await _open_stream(client, MESSAGES, SYSTEM, max_completion_tokens=500, model="m")
    assert captured["stream_params"].max_completion_tokens == RUN_OUTPUT
    async with cm:
        pass


@pytest.mark.asyncio
async def test_stream_exact_boundary_is_allowed():
    client = _client()
    captured = _mock_provider(client, input_tokens=WINDOW - RUN_OUTPUT)
    cm = await _open_stream(client, MESSAGES, SYSTEM, model="m")
    assert captured["stream_calls"] == 1
    async with cm:
        pass


@pytest.mark.asyncio
async def test_stream_over_limit_rejected_before_provider_invocation():
    client = _client()
    captured = _mock_provider(client, input_tokens=WINDOW - RUN_OUTPUT + 1)
    with pytest.raises(LLMContextWindowExceededError):
        await _open_stream(client, MESSAGES, SYSTEM, model="m")
    assert captured["stream_calls"] == 0
    assert captured["count_calls"] == 1


@pytest.mark.asyncio
async def test_stream_precomputed_count_skips_recount():
    client = _client()
    captured = _mock_provider(client, input_tokens=999_999)
    cm = await _open_stream(client, MESSAGES, SYSTEM, model="m", _precomputed_input_tokens=WINDOW - RUN_OUTPUT)
    assert captured["count_calls"] == 0
    assert captured["stream_calls"] == 1
    async with cm:
        pass


@pytest.mark.asyncio
async def test_stream_does_not_mutate_supplied_params():
    client = _client()
    captured = _mock_provider(client, input_tokens=100)
    params = GenerationParams(max_completion_tokens=None)
    cm = await _open_stream(client, MESSAGES, SYSTEM, params=params, model="m")
    assert params.max_completion_tokens is None
    assert captured["stream_params"].max_completion_tokens == RUN_OUTPUT
    async with cm:
        pass


# ---------------------------------------------------------------------------
# No caps: unchanged legacy behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_without_caps_does_not_count_and_passes_params_through():
    client = LLMClient(provider="anthropic", api_key="test-key")
    captured = _mock_provider(client, input_tokens=999_999)
    await client.generate(MESSAGES, SYSTEM, model="m")
    assert captured["count_calls"] == 0
    assert captured["generate_calls"] == 1
    assert captured["generate_params"].max_completion_tokens is None


@pytest.mark.asyncio
async def test_generate_without_caps_keeps_caller_output_limit():
    client = LLMClient(provider="anthropic", api_key="test-key")
    captured = _mock_provider(client, input_tokens=999_999)
    await client.generate(MESSAGES, SYSTEM, max_completion_tokens=500, model="m")
    assert captured["count_calls"] == 0
    assert captured["generate_params"].max_completion_tokens == 500


@pytest.mark.asyncio
async def test_stream_without_caps_returns_provider_stream_directly():
    client = LLMClient(provider="anthropic", api_key="test-key")
    captured = _mock_provider(client, input_tokens=999_999)
    cm = await _open_stream(client, MESSAGES, SYSTEM, model="m")
    assert captured["count_calls"] == 0
    assert captured["stream_calls"] == 1
    assert captured["stream_params"].max_completion_tokens is None
    async with cm:
        pass


# ---------------------------------------------------------------------------
# Instrumented client
# ---------------------------------------------------------------------------


def _instrumented_client(**overrides) -> InstrumentedLLMClient:
    kwargs: Dict[str, Any] = dict(
        provider="anthropic",
        api_key="test-key",
        context_window_tokens=WINDOW,
        max_output_tokens=RUN_OUTPUT,
        workspace_id="ws",
        thread_id="th",
        agent_type="test-agent",
    )
    kwargs.update(overrides)
    return InstrumentedLLMClient(**kwargs)


@pytest.mark.asyncio
async def test_instrumented_generate_enforces_budget_and_caps_output():
    langfuse, _trace, _generation = _mock_langfuse()
    client = _instrumented_client(langfuse_client=langfuse)
    captured = _mock_provider(client, input_tokens=100)
    await client.generate(MESSAGES, SYSTEM, max_completion_tokens=500, model="m")
    assert captured["generate_params"].max_completion_tokens == RUN_OUTPUT

    captured = _mock_provider(client, input_tokens=WINDOW - RUN_OUTPUT + 1)
    with pytest.raises(LLMContextWindowExceededError):
        await client.generate(MESSAGES, SYSTEM, model="m")
    assert captured["generate_calls"] == 0


@pytest.mark.asyncio
async def test_instrumented_stream_enforces_budget_and_caps_output():
    langfuse, _trace, _generation = _mock_langfuse()
    client = _instrumented_client(langfuse_client=langfuse)
    captured = _mock_provider(client, input_tokens=100)
    cm = await _open_stream(client, MESSAGES, SYSTEM, max_completion_tokens=500, model="m")
    assert captured["stream_params"].max_completion_tokens == RUN_OUTPUT
    async with cm:
        pass

    captured = _mock_provider(client, input_tokens=WINDOW - RUN_OUTPUT + 1)
    with pytest.raises(LLMContextWindowExceededError):
        await _open_stream(client, MESSAGES, SYSTEM, model="m")
    assert captured["stream_calls"] == 0


@pytest.mark.asyncio
async def test_instrumented_precomputed_count_used_and_kept_out_of_trace_metadata():
    langfuse, trace, _generation = _mock_langfuse()
    client = _instrumented_client(langfuse_client=langfuse)
    captured = _mock_provider(client, input_tokens=999_999)
    await client.generate(MESSAGES, SYSTEM, model="m", _precomputed_input_tokens=WINDOW - RUN_OUTPUT)
    assert captured["count_calls"] == 0
    assert captured["generate_calls"] == 1
    assert "_precomputed_input_tokens" not in captured["generate_kwargs"]

    span_metadata = trace.start_generation.call_args.kwargs["metadata"]
    assert "_precomputed_input_tokens" not in span_metadata


@pytest.mark.asyncio
async def test_instrumented_without_langfuse_still_enforces_budget():
    client = _instrumented_client(langfuse_client=None)
    captured = _mock_provider(client, input_tokens=WINDOW - RUN_OUTPUT + 1)
    with pytest.raises(LLMContextWindowExceededError):
        await client.generate(MESSAGES, SYSTEM, model="m")
    assert captured["generate_calls"] == 0


# ---------------------------------------------------------------------------
# AgentContext propagation
# ---------------------------------------------------------------------------


def _make_agent(tmp_path, *, strict_budget, langfuse_client=None):
    import uuid
    from unittest.mock import AsyncMock

    from kolega_code.agent.coder import CoderAgent
    from kolega_code.agent.prompt_provider import AgentMode
    from kolega_code.events import AgentConnectionManager

    from ..compaction_helpers import make_agent_config

    return CoderAgent(
        project_path=tmp_path,
        workspace_id="test_ws",
        thread_id=str(uuid.uuid4()),
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=make_agent_config(strict_budget=strict_budget),
        agent_mode=AgentMode.CLI,
        langfuse_client=langfuse_client,
    )


def test_create_llm_client_propagates_caps_to_plain_client(tmp_path):
    agent = _make_agent(tmp_path, strict_budget=(WINDOW, RUN_OUTPUT))
    client = agent.context.create_llm_client(agent.agent_name)
    assert not isinstance(client, InstrumentedLLMClient)
    assert client._context_window_tokens == WINDOW
    assert client._max_output_tokens == RUN_OUTPUT


def test_create_llm_client_propagates_caps_to_instrumented_client(tmp_path):
    agent = _make_agent(tmp_path, strict_budget=(WINDOW, RUN_OUTPUT), langfuse_client=MagicMock())
    client = agent.context.create_llm_client(agent.agent_name)
    assert isinstance(client, InstrumentedLLMClient)
    assert client._context_window_tokens == WINDOW
    assert client._max_output_tokens == RUN_OUTPUT


def test_create_llm_client_without_caps_leaves_budget_unset(tmp_path):
    agent = _make_agent(tmp_path, strict_budget=None)
    client = agent.context.create_llm_client(agent.agent_name)
    assert client._context_window_tokens is None
    assert client._max_output_tokens is None
    assert not client._has_run_context_budget


# ---------------------------------------------------------------------------
# Ordinary history compactor: covered automatically via the agent's client
# ---------------------------------------------------------------------------


def _compressible_conversation():
    from kolega_code.agent.conversation import Conversation

    messages = []
    for i in range(12):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(Message(role, [TextBlock(f"turn {i}")]))
    return Conversation(messages)


@pytest.mark.asyncio
async def test_history_compactor_output_capped_to_run_max():
    from kolega_code.agent.compression import HistoryCompressor

    client = _client()
    captured = _mock_provider(client, input_tokens=100)
    result = await HistoryCompressor().summarize(
        _compressible_conversation(), llm=client, model="m", temperature=1.0, thinking=None
    )
    assert result.ok, result.message
    # The compactor asks for SUMMARY_MAX_TOKENS (8192); the run-wide cap wins.
    assert captured["stream_params"].max_completion_tokens == RUN_OUTPUT


@pytest.mark.asyncio
async def test_history_compactor_over_limit_fails_before_provider_invocation():
    from kolega_code.agent.compression import HistoryCompressor

    client = _client()
    captured = _mock_provider(client, input_tokens=WINDOW - RUN_OUTPUT + 1)
    result = await HistoryCompressor().summarize(
        _compressible_conversation(), llm=client, model="m", temperature=1.0, thinking=None
    )
    assert not result.ok
    assert result.reason == "llm_error"
    assert captured["count_calls"] == 1
    assert captured["stream_calls"] == 0
