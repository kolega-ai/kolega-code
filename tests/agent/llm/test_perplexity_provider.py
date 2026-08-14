"""Perplexity providers: Gateway (Chat Completions) and Agent API (Responses).

Covers routing, base URLs, auth, request shaping (including the anthropic
``max_output_tokens`` rule and server-side tools), streaming, harness function
calls, Perplexity server-side tool result items (``search_results`` /
``fetch_url_results``), usage accounting, error mapping, config, and the
generated catalogs + runtime overlay.

Hermetic: fake streams and recorder clients, no network. Wire facts sourced
from docs.perplexity.ai (verified 2026-08-14); live probing was blocked by an
invalid ``PERPLEXITY_API_KEY`` in the repo ``.env`` (401 on every endpoint).
"""

import asyncio
import json
from types import SimpleNamespace as _ns
from typing import Any, Coroutine, cast

import pytest

from kolega_code.config import (
    AgentConfig,
    ModelConfig,
    ModelProvider,
)
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    map_to_llm_error,
)
from kolega_code.llm.models import (
    Message,
    MessageHistory,
    TextBlock,
    ToolDefinition,
    ToolParameter,
    WebSearchCallBlock,
)
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.providers.perplexity_responses import (
    DEFAULT_BASE_URL as AGENT_BASE_URL,
    PerplexityResponsesProvider,
)
from kolega_code.llm.providers.responses_common import (
    ResponsesStreamWrapper,
    responses_tools,
    to_responses_input,
)
from kolega_code.llm.usage import OPENAI_USAGE_PROVIDERS
from kolega_code.llm.specs.perplexity_catalog import AGENT_SERVER_TOOLS


def _function_tool() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read a file.",
        parameters=[ToolParameter(name="path", type="string", description="Path", required=True)],
    )


def _history(*messages: Message) -> MessageHistory:
    return MessageHistory(list(messages))


# --- routing, base URLs, auth -----------------------------------------------------


def test_provider_class_routes_agent_to_perplexity_responses_provider():
    assert LLMClient._provider_class("perplexity_agent") is PerplexityResponsesProvider


def test_agent_client_uses_v1_base_url_and_key():
    client = LLMClient(provider="perplexity_agent", api_key="pplx-test", model="openai/gpt-5.6-sol")
    provider = client.provider
    assert isinstance(provider, PerplexityResponsesProvider)
    assert str(provider.async_client.base_url).rstrip("/") == AGENT_BASE_URL
    assert provider.async_client.api_key == "pplx-test"
    assert provider.provider_name == "perplexity_agent"


def test_agent_sdk_client_posts_to_responses_alias():
    client = LLMClient(provider="perplexity_agent", api_key="pplx-test", model="openai/gpt-5.6-sol")
    provider = client.provider
    assert isinstance(provider, PerplexityResponsesProvider)
    assert str(provider.async_client.base_url) == "https://api.perplexity.ai/v1/"


def test_api_key_env_maps_agent_provider_to_perplexity_key():
    from kolega_code.cli.config import API_KEY_ENV

    assert API_KEY_ENV[ModelProvider.PERPLEXITY_AGENT] == "PERPLEXITY_API_KEY"


def test_usage_provider_set_includes_perplexity_agent():
    assert "perplexity_agent" in OPENAI_USAGE_PROVIDERS


def test_agent_error_mapping():
    from openai import OpenAIError

    class _StatusError(OpenAIError):
        def __init__(self, message, status_code):
            super().__init__(message)
            self.status_code = status_code

    for status, expected in ((401, LLMAuthenticationError), (429, LLMRateLimitError)):
        mapped = map_to_llm_error(_StatusError("nope", status), "perplexity_agent")
        assert isinstance(mapped, expected)
        assert mapped.provider == "perplexity_agent"


# --- agent (Responses) request shaping ---------------------------------------------


def _agent_provider() -> PerplexityResponsesProvider:
    return PerplexityResponsesProvider(api_key="pplx-test")


def _agent_request(model: str, params=None, messages=None, system=None) -> dict:
    provider = _agent_provider()
    return provider._build_request(
        messages or _history(Message(role="user", content=[TextBlock(text="hi")])),
        system or Message(role="system", content=[TextBlock(text="be brief")]),
        params,
        {"model": model},
    )


def test_agent_request_is_responses_shaped():
    request = _agent_request("openai/gpt-5.6-sol", GenerationParams(tools=[_function_tool()]))
    assert request["model"] == "openai/gpt-5.6-sol"
    assert request["store"] is False
    assert request["stream"] is True
    assert request["input"][0]["role"] == "user"
    assert "be brief" in request["instructions"]
    assert request["tools"][0] == {
        "type": "function",
        "name": "read_file",
        "description": "Read a file.",
        "parameters": request["tools"][0]["parameters"],
    }


def test_agent_request_sends_max_output_tokens_for_anthropic_models():
    request = _agent_request("anthropic/claude-opus-5", GenerationParams(max_completion_tokens=4096))
    assert request["max_output_tokens"] == 4096


def test_agent_request_derives_anthropic_cap_from_catalog_when_unset():
    request = _agent_request("anthropic/claude-opus-5")
    assert request["max_output_tokens"] > 0


def test_agent_request_omits_max_output_tokens_for_other_models():
    request = _agent_request("openai/gpt-5.6-sol")
    assert "max_output_tokens" not in request


def test_agent_request_omits_reasoning_include_param():
    request = _agent_request("openai/gpt-5.6-sol", GenerationParams(thinking="medium"))
    assert "include" not in request


def test_server_tools_absent_by_default():
    tools = responses_tools(GenerationParams(tools=[_function_tool()]))
    assert tools is not None
    assert all(tool["type"] != "web_search" for tool in tools)


def test_server_tools_appended_when_configured():
    tools = responses_tools(GenerationParams(tools=[_function_tool()], server_tools=["web_search", "fetch_url"]))
    assert tools is not None
    assert tools[-2:] == [{"type": "web_search"}, {"type": "fetch_url"}]
    assert tools[0]["type"] == "function"


def test_server_tools_dedupe_against_hosted_web_search():
    tools = responses_tools(GenerationParams(hosted_web_search=True, server_tools=["web_search", "people_search"]))
    assert tools is not None
    web_search_entries = [tool for tool in tools if tool == {"type": "web_search"}]
    assert len(web_search_entries) == 1
    assert {"type": "people_search"} in tools


def test_agent_request_carries_server_tools():
    request = _agent_request(
        "openai/gpt-5.6-sol",
        GenerationParams(tools=[_function_tool()], server_tools=["web_search", "fetch_url"]),
    )
    assert request["tools"][-2:] == [{"type": "web_search"}, {"type": "fetch_url"}]


# --- agent streaming ---------------------------------------------------------------


class _ResponsesStream:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def aclose(self):
        return None


def _text_delta(text):
    return _ns(type="response.output_text.delta", delta=text)


def _function_call_item(item_id="fc_1", call_id="call_1", name="read_file", arguments='{"path": "a.py"}'):
    return _ns(type="function_call", id=item_id, call_id=call_id, name=name, arguments=arguments)


def _completed_event(output=None, usage=None):
    usage = usage or _ns(
        input_tokens=5,
        output_tokens=2,
        total_tokens=7,
        input_tokens_details=_ns(cached_tokens=3),
        output_tokens_details=None,
        model_extra={"cost": {"total_cost": 0.01}, "tool_calls_details": {"search_web": {"invocation": 1}}},
    )
    return _ns(type="response.completed", response=_ns(status="completed", output=output or [], usage=usage))


async def _drain(wrapper: ResponsesStreamWrapper):
    chunks = []
    async with wrapper:
        async for chunk in wrapper:
            chunks.append(chunk)
    return chunks, await wrapper.get_final_message()


def test_agent_stream_text_and_usage():
    wrapper = ResponsesStreamWrapper(
        _ResponsesStream([_text_delta("Hi"), _text_delta(" there"), _completed_event()]),
        provider_name="perplexity_agent",
        model="openai/gpt-5.6-sol",
    )
    chunks, message = asyncio.run(_drain(wrapper))
    assert [chunk.text for chunk in chunks if chunk.type == "text"] == ["Hi", " there"]
    assert message.get_text_content() == "Hi there"
    assert message.usage_metadata["prompt_tokens"] == 5
    assert message.usage_metadata["cache_read_input_tokens"] == 3
    assert message.usage_metadata["cost"] == {"total_cost": 0.01}
    assert message.usage_metadata["tool_calls_details"] == {"search_web": {"invocation": 1}}
    assert message.usage_metadata["provider"] == "perplexity_agent"
    assert message.stop_reason == "end_turn"


def test_agent_stream_harness_function_call():
    events = [
        _ns(type="response.output_item.added", item=_function_call_item()),
        _ns(type="response.function_call_arguments.delta", item_id="fc_1", delta='{"path": '),
        _ns(type="response.function_call_arguments.delta", item_id="fc_1", delta='"a.py"}'),
        _completed_event(),
    ]
    wrapper = ResponsesStreamWrapper(
        _ResponsesStream(events), provider_name="perplexity_agent", model="openai/gpt-5.6-sol"
    )
    chunks, message = asyncio.run(_drain(wrapper))
    starts = [chunk for chunk in chunks if chunk.type == "tool_use_start"]
    assert starts and starts[0].tool_call_delta["name"] == "read_file"
    assert message.stop_reason == "tool_use"
    assert message.tool_calls[0].input == {"path": "a.py"}


def _search_results_item(item_id="sr_1", queries=("llm benchmarks",)):
    results = [{"id": 1, "title": "Leaderboard", "url": "https://example.com/llm", "snippet": "Benchmarks..."}]
    return _ns(type="search_results", id=item_id, queries=list(queries), results=results, status="completed")


def _fetch_url_results_item(item_id="fr_1"):
    contents = [{"title": "Page", "url": "https://example.com/page", "snippet": "Content..."}]
    return _ns(type="fetch_url_results", id=item_id, contents=contents, status="completed")


def test_agent_stream_server_tool_results_emit_hosted_tool_call_chunks():
    events = [
        _ns(type="response.output_item.done", item=_search_results_item()),
        _ns(type="response.output_item.done", item=_fetch_url_results_item()),
        _text_delta("Answer"),
        _completed_event(),
    ]
    wrapper = ResponsesStreamWrapper(
        _ResponsesStream(events), provider_name="perplexity_agent", model="openai/gpt-5.6-sol"
    )
    chunks, message = asyncio.run(_drain(wrapper))
    hosted = [chunk for chunk in chunks if chunk.type == "hosted_tool_call"]
    assert len(hosted) == 2
    first, second = hosted
    assert first.tool_call_delta["id"] == "sr_1"
    assert first.tool_call_delta["item_type"] == "search_results"
    assert first.tool_call_delta["action"]["queries"] == ["llm benchmarks"]
    assert first.tool_call_delta["payload"]["type"] == "search_results"
    assert second.tool_call_delta["item_type"] == "fetch_url_results"
    assert second.tool_call_delta["action"]["urls"] == ["https://example.com/page"]

    # Output-only items are dropped from the transport copy (replay 400s).
    prefix = [block for block in message.content if isinstance(block, WebSearchCallBlock)]
    assert [block.item_type for block in prefix] == ["search_results", "fetch_url_results"]
    assert prefix[0].tool_label == "search (hosted)"
    assert prefix[1].tool_label == "fetch_url (hosted)"
    assert "1 result returned" in prefix[0].result_summary()
    assert prefix[0].payload is not None
    assert prefix[0].payload["results"][0]["url"] == "https://example.com/llm"
    replay = to_responses_input(_history(message))
    assert all(item.get("type") not in ("search_results", "fetch_url_results") for item in replay)


def test_agent_stream_server_tool_results_from_final_response_fallback():
    # No output_item.done: no live chunk, but the block is still captured.
    events = [
        _text_delta("Answer"),
        _completed_event(output=[_search_results_item()]),
    ]
    wrapper = ResponsesStreamWrapper(
        _ResponsesStream(events), provider_name="perplexity_agent", model="openai/gpt-5.6-sol"
    )
    chunks, message = asyncio.run(_drain(wrapper))
    assert not any(chunk.type == "hosted_tool_call" for chunk in chunks)
    prefix = [block for block in message.content if isinstance(block, WebSearchCallBlock)]
    assert prefix and prefix[0].item_type == "search_results"
    assert all(item.get("type") != "search_results" for item in to_responses_input(_history(message)))


def test_server_tool_blocks_survive_serialization_round_trip():
    block = WebSearchCallBlock(
        item_id="sr_1",
        status="completed",
        action={"type": "search", "queries": ["q"]},
        item_type="search_results",
        payload={"type": "search_results", "id": "sr_1", "queries": ["q"], "results": []},
    )
    restored = WebSearchCallBlock.from_dict(block.to_dict())
    assert restored.item_type == "search_results"
    assert restored.payload == block.payload
    assert restored.to_responses_item()["type"] == "search_results"
    classic = WebSearchCallBlock(item_id="ws_1", status="completed", action={"type": "search", "queries": ["q"]})
    assert "item_type" not in classic.to_dict()
    assert classic.to_responses_item() == {
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["q"]},
        "id": "ws_1",
    }


def test_server_tool_blocks_contribute_no_input_tokens():
    """Output-only items are never resent, so they add nothing to the input count."""
    provider = OpenAIProvider(api_key="pplx-test", provider_name="perplexity_agent")
    payload_block = WebSearchCallBlock(
        item_type="search_results",
        action={"type": "search", "queries": ["q"]},
        payload={"type": "search_results", "queries": ["q"], "results": [{"title": "t", "snippet": "s" * 500}]},
    )
    empty = asyncio.run(provider.count_tokens(_history(Message(role="assistant", content=[]))))
    with_block = asyncio.run(provider.count_tokens(_history(Message(role="assistant", content=[payload_block]))))
    assert with_block.input_tokens == empty.input_tokens
    metadata_block = WebSearchCallBlock(action={"type": "search", "queries": ["q"]})
    with_replayable = asyncio.run(provider.count_tokens(_history(Message(role="assistant", content=[metadata_block]))))
    assert with_replayable.input_tokens > empty.input_tokens


# --- LLMClient passthrough ----------------------------------------------------------


class _ParamsRecorder:
    def __init__(self):
        self.params: list[GenerationParams] = []

    async def stream(self, messages, system, params, **kwargs):
        self.params.append(params)

        class _Empty:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def get_final_message(self):
                return Message(role="assistant", content=[TextBlock(text="")])

        return _Empty()


def _run_stream_call(call) -> None:
    """Await LLMClient.stream's union return (the provider stream coroutine)."""
    coroutine = cast("Coroutine[Any, Any, Any]", call)
    asyncio.run(coroutine)


def test_llmclient_folds_server_tools_into_params():
    client = LLMClient(provider="perplexity_agent", api_key="pplx-test", model="openai/gpt-5.6-sol")
    recorder = _ParamsRecorder()
    client.provider = recorder  # type: ignore[assignment]
    _run_stream_call(
        client.stream(
            _history(Message(role="user", content=[TextBlock(text="hi")])),
            server_tools=["web_search", "fetch_url"],
            model="openai/gpt-5.6-sol",
        )
    )
    assert recorder.params[-1].server_tools == ["web_search", "fetch_url"]


def test_llmclient_default_params_have_no_server_tools():
    client = LLMClient(provider="perplexity_agent", api_key="pplx-test", model="openai/gpt-5.6-sol")
    recorder = _ParamsRecorder()
    client.provider = recorder  # type: ignore[assignment]
    _run_stream_call(
        client.stream(_history(Message(role="user", content=[TextBlock(text="hi")])), model="openai/gpt-5.6-sol")
    )
    assert recorder.params[-1].server_tools is None


# --- config ------------------------------------------------------------------------


def _config() -> AgentConfig:
    model = ModelConfig(provider=ModelProvider.PERPLEXITY_AGENT, model="openai/gpt-5.6-sol")
    return AgentConfig(
        perplexity_api_key="pplx-test",
        long_context_config=model,
        fast_config=model,
    )


def test_get_api_key_serves_agent_provider():
    config = _config()
    assert config.get_api_key(ModelProvider.PERPLEXITY_AGENT) == "pplx-test"


# --- agent server-tools resolution (BaseAgent) --------------------------------------


def _agent_state(spec_tools, *, mode="auto"):
    """A minimally-constructed BaseAgent carrying only the web-tool state inputs."""
    from kolega_code.agent.baseagent import BaseAgent

    agent = BaseAgent.__new__(BaseAgent)
    agent.supports_hosted_web_search = False
    agent._model_server_tools = list(spec_tools)
    agent.web_search_mode = mode
    agent._apply_web_search_state()
    return agent


def test_server_tools_without_spec_tools_stay_empty():
    state = _agent_state([])
    assert state.server_tools == []
    assert state.client_web_tools_enabled is True


def test_server_tools_from_spec_are_on_by_default():
    state = _agent_state(AGENT_SERVER_TOOLS)
    assert state.server_tools == list(AGENT_SERVER_TOOLS)
    assert state.client_web_tools_enabled is False


def test_server_tools_mode_hosted_keeps_them_on():
    state = _agent_state(AGENT_SERVER_TOOLS, mode="hosted")
    assert state.server_tools == list(AGENT_SERVER_TOOLS)
    assert state.client_web_tools_enabled is False


def test_server_tools_mode_off_disables_all_web_tooling():
    state = _agent_state(AGENT_SERVER_TOOLS, mode="off")
    assert state.server_tools == []
    assert state.client_web_tools_enabled is False


def test_server_tools_mode_client_defers_to_client_web_tools():
    state = _agent_state(AGENT_SERVER_TOOLS, mode="client")
    assert state.server_tools == []
    assert state.client_web_tools_enabled is True


# --- catalogs ------------------------------------------------------------------------


def test_seeded_catalogs_resolve():
    from kolega_code.llm.specs import MODEL_SPECS, get_model_specs, model_is_known

    assert ("perplexity_agent", "anthropic/claude-opus-5") in MODEL_SPECS
    assert model_is_known("perplexity_agent", "xai/grok-4.6")
    # No wildcards: unknown ids are rejected like every other catalog provider.
    assert not model_is_known("perplexity_agent", "openai/gpt-99")
    assert get_model_specs("perplexity_agent", "anthropic/claude-opus-5")["context_length"] == 131072


def test_server_tools_are_a_spec_property_of_the_agent_catalog():
    """Every agent entry declares the full server-tool set."""
    from kolega_code.llm.specs import MODEL_SPECS

    agent_entries = [spec for (provider, _), spec in MODEL_SPECS.items() if provider == "perplexity_agent"]
    assert agent_entries and all(spec["server_tools"] == list(AGENT_SERVER_TOOLS) for spec in agent_entries)
    assert AGENT_SERVER_TOOLS == ("web_search", "fetch_url", "finance_search", "people_search")


def test_catalog_entries_are_uniform_regardless_of_id():
    """Every entry gets the same conservative spec; ids are not matched against
    native-provider catalogs (the same model behind a different API has no
    guaranteed shared behavior)."""
    from kolega_code.llm.specs import perplexity_catalog as catalog
    from kolega_code.llm.specs.validation import validate_model_spec

    payload = {
        "data": [
            {"id": "openai/gpt-5.6-sol", "object": "model", "created": 0, "owned_by": "openai"},
            {"id": "perplexity/brand-new-model", "object": "model", "created": 0, "owned_by": "perplexity"},
        ]
    }
    entries = catalog.catalog_entries(payload)
    assert [identifier for identifier, _ in entries] == ["openai/gpt-5.6-sol", "perplexity/brand-new-model"]
    sol_spec, fallback_spec = entries[0][1], entries[1][1]
    assert sol_spec == fallback_spec
    assert sol_spec["context_length"] == catalog.FALLBACK_CONTEXT_LENGTH
    assert sol_spec["max_completion_tokens"] == catalog.FALLBACK_MAX_COMPLETION_TOKENS
    assert sol_spec["thinking_effort"].mode == "openai_responses_reasoning"
    for _, spec in entries:
        validate_model_spec(spec)


def test_catalog_efforts_come_from_the_probed_vocabulary():
    """Agent entries carry the live-probed effort vocabulary; the Gateway carries none.

    Probed 2026-08-14: an invalid effort makes the Agent API enumerate the
    accepted set uniformly for every model — minimal/low/medium/high/xhigh/max,
    notably no "none".
    """
    from kolega_code.llm.specs import build_thinking_request_params, get_model_specs
    from kolega_code.llm.specs import perplexity_catalog as catalog

    payload = {"data": [{"id": "perplexity/deepseek-v4-flash-0731"}, {"id": "xai/grok-4.5"}]}

    for identifier, spec in catalog.AGENT_CATALOG.catalog_entries(payload):
        assert spec["thinking_effort"].mode == "openai_responses_reasoning"
        assert spec["thinking_effort"].options == catalog.AGENT_EFFORT_OPTIONS
        assert "none" not in spec["thinking_effort"].options

    agent_spec = get_model_specs("perplexity_agent", "perplexity/glm-5.2")
    assert agent_spec["thinking_effort"].options == catalog.AGENT_EFFORT_OPTIONS
    params = build_thinking_request_params("perplexity_agent", "perplexity/glm-5.2", "high")
    assert params == {"reasoning": {"effort": "high", "summary": "auto"}}


def test_catalog_entries_rejects_malformed_payloads():
    from kolega_code.llm.specs import perplexity_catalog as catalog

    with pytest.raises(catalog.PerplexityCatalogError):
        catalog.catalog_entries({"data": []})
    with pytest.raises(catalog.PerplexityCatalogError):
        catalog.catalog_entries({"data": [{"id": ""}]})


def test_catalog_cache_round_trip(tmp_path):
    from kolega_code.llm.specs import perplexity_catalog as catalog

    payload = {"data": [{"id": "openai/gpt-5.6-sol"}, {"id": "perplexity/brand-new"}]}
    entries = catalog.AGENT_CATALOG.catalog_entries(payload)
    path = tmp_path / catalog.AGENT_CATALOG.CACHE_FILENAME
    catalog.AGENT_CATALOG.save_cache(path, entries, fetched_at="2026-08-14T00:00:00+00:00")
    restored = catalog.AGENT_CATALOG.load_cache(path)
    assert [identifier for identifier, _ in restored] == ["openai/gpt-5.6-sol", "perplexity/brand-new"]
    assert restored[0][1]["thinking_effort"].options == entries[0][1]["thinking_effort"].options
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps(
            {
                "schema_version": catalog.CACHE_SCHEMA_VERSION,
                "provider": "openrouter",
                "models": [{"id": "x", "spec": {}}],
            }
        )
    )
    assert catalog.AGENT_CATALOG.load_cache(other) == []


def test_catalog_overlay_sources_registered():
    from kolega_code.cli.model_catalog import OVERLAY_SOURCES

    assert OVERLAY_SOURCES["perplexity_agent"].MODELS_URL == "https://api.perplexity.ai/v1/models"


def test_fetch_models_requires_api_key(monkeypatch):
    from kolega_code.llm.specs import perplexity_catalog as catalog

    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(catalog.PerplexityCatalogError, match="PERPLEXITY_API_KEY"):
        catalog.fetch_models(catalog.AGENT_MODELS_URL)


def test_provider_registry_defaults_and_labels():
    from kolega_code.cli.provider_registry import (
        PROVIDER_DEFAULT_MODEL,
        PROVIDER_LABELS,
        _api_key_env,
    )

    assert PROVIDER_LABELS[ModelProvider.PERPLEXITY_AGENT] == "Perplexity Agent API"
    assert PROVIDER_DEFAULT_MODEL[ModelProvider.PERPLEXITY_AGENT] == "openai/gpt-5.6-sol"
    assert _api_key_env(ModelProvider.PERPLEXITY_AGENT) == "PERPLEXITY_API_KEY"
