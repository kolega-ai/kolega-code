import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from kolega_code.agent.tool_backend.network_status import ConnectFailureTracker
from kolega_code.agent.tool_backend.search_backends import (
    DEFAULT_BACKEND,
    SearchBackendError,
    SearchBackendNotConfigured,
    SearchBackendRateLimited,
    SearchBackendUnavailable,
    SearchBackendUnreachable,
    SearchResponse,
    SearchResult,
    available_backends,
    backend_names,
    build_search_backend,
    get_backend_class,
)
from kolega_code.agent.tool_backend.search_backends.base import clamp_results
from kolega_code.agent.tool_backend.search_backends.firecrawl import _new_client
from kolega_code.agent.tool_backend.search_backends.tavily import _new_client as _new_tavily_client
from kolega_code.agent.tool_backend.web_search_tool import WebSearchTool
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider

_DUCK = "kolega_code.agent.tool_backend.search_backends.duckduckgo"
_TAVILY = "kolega_code.agent.tool_backend.search_backends.tavily"
_FIRE = "kolega_code.agent.tool_backend.search_backends.firecrawl"
_SEARX = "kolega_code.agent.tool_backend.search_backends.searxng"


# --------------------------------------------------------------------------- registry


def test_registry_lists_all_backends() -> None:
    assert set(backend_names()) == {"duckduckgo", "firecrawl", "tavily", "searxng"}
    # Default backend is first in the TUI option list and is keyless.
    labels = available_backends()
    assert labels[0][1] == DEFAULT_BACKEND == "duckduckgo"
    assert get_backend_class("duckduckgo").requires_api_key is False
    # Optional-key model: Firecrawl is keyless-capable but accepts a key for higher limits;
    # Tavily requires one; DuckDuckGo/SearXNG take no key.
    assert get_backend_class("firecrawl").requires_api_key is False
    assert get_backend_class("firecrawl").accepts_api_key is True
    assert get_backend_class("tavily").requires_api_key is True
    assert get_backend_class("tavily").accepts_api_key is True
    assert get_backend_class("duckduckgo").accepts_api_key is False
    assert get_backend_class("searxng").accepts_api_key is False


def test_build_unknown_backend_raises_typed_error() -> None:
    with pytest.raises(SearchBackendError, match="Unknown web_search backend"):
        build_search_backend("nope")


@pytest.mark.parametrize(("requested", "expected"), [(-3, 1), (0, 1), (5, 5), (50, 10), ("bad", 5)])
def test_clamp_results(requested, expected) -> None:
    assert clamp_results(requested) == expected


# ------------------------------------------------------------------------- duckduckgo


class _DummyDDGS:
    def __init__(self, rows=None, exc=None):
        self._rows = rows or []
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, query, max_results=5):
        if self._exc:
            raise self._exc
        return self._rows


@pytest.mark.asyncio
async def test_duckduckgo_maps_rows() -> None:
    rows = [{"title": "T", "href": "https://a.test", "body": "snippet"}, {"title": "", "href": "", "body": "x"}]
    with patch(f"{_DUCK}.DDGS", lambda: _DummyDDGS(rows=rows)):
        response = await build_search_backend("duckduckgo").search("q", max_results=3)
    assert response.backend == "duckduckgo"
    # The second row has no href and is dropped.
    assert [r.url for r in response.results] == ["https://a.test"]
    assert response.results[0].snippet == "snippet"


@pytest.mark.asyncio
async def test_duckduckgo_rate_limit_mapped() -> None:
    from ddgs.exceptions import RatelimitException

    with patch(f"{_DUCK}.DDGS", lambda: _DummyDDGS(exc=RatelimitException("429"))):
        with pytest.raises(SearchBackendRateLimited):
            await build_search_backend("duckduckgo").search("q")


@pytest.mark.asyncio
async def test_duckduckgo_generic_error_unavailable() -> None:
    with patch(f"{_DUCK}.DDGS", lambda: _DummyDDGS(exc=RuntimeError("boom"))):
        with pytest.raises(SearchBackendUnavailable):
            await build_search_backend("duckduckgo").search("q")


# ----------------------------------------------------------------------------- tavily


@pytest.mark.asyncio
async def test_tavily_requires_key() -> None:
    with pytest.raises(SearchBackendNotConfigured):
        await build_search_backend("tavily", api_key=None).search("q")


@pytest.mark.asyncio
async def test_tavily_success_with_answer() -> None:
    payload = {
        "answer": "the answer",
        "results": [{"title": "T", "url": "https://a.test", "content": "body"}, {"url": ""}],
    }
    client = Mock()
    client.search.return_value = payload
    with patch(f"{_TAVILY}._new_client", return_value=client) as new_client:
        response = await build_search_backend("tavily", api_key="k").search("q", max_results=2)
    assert response.answer == "the answer"
    assert [r.url for r in response.results] == ["https://a.test"]
    # Key + count are passed through.
    assert new_client.call_args.kwargs == {"api_key": "k"}
    _, kwargs = client.search.call_args
    assert kwargs["max_results"] == 2 and kwargs["include_answer"] is True


@pytest.mark.asyncio
async def test_tavily_invalid_key_mapped_to_not_configured() -> None:
    from tavily.errors import InvalidAPIKeyError

    client = Mock()
    client.search.side_effect = InvalidAPIKeyError("bad key")
    with patch(f"{_TAVILY}._new_client", return_value=client):
        with pytest.raises(SearchBackendNotConfigured):
            await build_search_backend("tavily", api_key="bad").search("q")


@pytest.mark.asyncio
async def test_tavily_usage_limit_mapped_to_rate_limited() -> None:
    from tavily.errors import UsageLimitExceededError

    client = Mock()
    client.search.side_effect = UsageLimitExceededError("quota gone")
    with patch(f"{_TAVILY}._new_client", return_value=client):
        with pytest.raises(SearchBackendRateLimited, match="quota gone"):
            await build_search_backend("tavily", api_key="k").search("q")


def test_tavily_client_is_built_from_the_imported_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keyed path reaches the real SDK import, not just the patched seam.

    That import is deferred to first keyed search, so this also proves it still
    resolves tavily-python and passes the credentials through.
    """
    import tavily

    created: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            created.update(kwargs)

    monkeypatch.setattr(tavily, "TavilyClient", _FakeClient)
    assert isinstance(_new_tavily_client(api_key="tvly-obviously-fake"), _FakeClient)
    assert created == {"api_key": "tvly-obviously-fake"}


def test_tavily_sdk_stays_off_the_startup_import_path() -> None:
    """Importing the backend registry must not import the Tavily SDK.

    That import also pulls ``requests`` (~48 ms), which every CLI start would otherwise
    pay for a backend most sessions never select. A fresh interpreter is required:
    other tests import ``tavily`` in this process.
    """
    probe = "import sys; import kolega_code.agent.tool_backend.search_backends; print('tavily' in sys.modules)"
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"


# --------------------------------------------------------------------------- firecrawl


@pytest.mark.asyncio
async def test_firecrawl_keyless_uses_rest() -> None:
    # No key -> direct REST (Firecrawl's free tier), no SDK involved.
    payload = {
        "success": True,
        "data": {
            "web": [
                {"url": "https://a.test", "title": "T", "description": "desc"},
                {"url": "", "title": "skip"},
            ]
        },
    }
    with patch(f"{_FIRE}.httpx.AsyncClient", _post_client(_DummyResponse(200, payload))):
        response = await build_search_backend("firecrawl", api_key=None).search("q", max_results=3)
    assert response.backend == "firecrawl"
    assert [r.url for r in response.results] == ["https://a.test"]
    assert response.results[0].snippet == "desc"


@pytest.mark.asyncio
async def test_firecrawl_keyless_429_rate_limited() -> None:
    with patch(f"{_FIRE}.httpx.AsyncClient", _post_client(_DummyResponse(429))):
        with pytest.raises(SearchBackendRateLimited):
            await build_search_backend("firecrawl", api_key=None).search("q")


@pytest.mark.asyncio
async def test_firecrawl_keyed_uses_sdk() -> None:
    data = SimpleNamespace(
        web=[
            SimpleNamespace(url="https://a.test", title="T", description="desc", markdown="full"),
            SimpleNamespace(url=None, title="skip", description="", markdown=""),
        ]
    )
    client = Mock()
    client.search.return_value = data
    with patch(f"{_FIRE}._new_client", return_value=client) as new_client:
        response = await build_search_backend("firecrawl", api_key="fc").search("q", max_results=4)
    assert [r.url for r in response.results] == ["https://a.test"]
    assert response.results[0].snippet == "desc"
    assert response.results[0].content == "full"
    # The client is built with the configured key and timeout.
    assert new_client.call_args.kwargs == {"api_key": "fc", "timeout": WebSearchTool.SEARCH_TIMEOUT_SECONDS}
    # limit is forwarded; firecrawl timeout is milliseconds.
    _, kwargs = client.search.call_args
    assert kwargs["limit"] == 4 and kwargs["timeout"] == int(WebSearchTool.SEARCH_TIMEOUT_SECONDS * 1000)


@pytest.mark.asyncio
async def test_firecrawl_401_mapped_to_not_configured() -> None:
    client = Mock()
    client.search.side_effect = RuntimeError("HTTP 401 Unauthorized")
    with patch(f"{_FIRE}._new_client", return_value=client):
        with pytest.raises(SearchBackendNotConfigured):
            await build_search_backend("firecrawl", api_key="fc").search("q")


def test_firecrawl_client_is_built_from_the_imported_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keyed path reaches the real SDK import, not just the patched seam.

    That import is deferred to first keyed search, so this also proves it still
    resolves firecrawl-py and passes the credentials through.
    """
    import firecrawl

    created: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            created.update(kwargs)

    monkeypatch.setattr(firecrawl, "Firecrawl", _FakeClient)
    assert isinstance(_new_client(api_key="fc-obviously-fake", timeout=3.0), _FakeClient)
    assert created == {"api_key": "fc-obviously-fake", "timeout": 3.0}


# ----------------------------------------------------------------------------- searxng


class _DummyResponse:
    def __init__(self, status_code=200, payload=None, json_error=False):
        self.status_code = status_code
        self._payload = payload or {}
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload


def _searx_client(response=None, get_exc=None):
    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None):
            if get_exc:
                raise get_exc
            return response

    return _DummyAsyncClient


def _post_client(response=None, post_exc=None):
    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None):
            if post_exc:
                raise post_exc
            return response

    return _DummyAsyncClient


@pytest.mark.asyncio
async def test_searxng_requires_base_url() -> None:
    with pytest.raises(SearchBackendNotConfigured):
        await build_search_backend("searxng", base_url=None).search("q")


@pytest.mark.asyncio
async def test_searxng_success() -> None:
    payload = {"results": [{"title": "T", "url": "https://a.test", "content": "c"}, {"url": ""}]}
    with patch(f"{_SEARX}.httpx.AsyncClient", _searx_client(_DummyResponse(200, payload))):
        response = await build_search_backend("searxng", base_url="https://s.test/").search("q", max_results=5)
    assert [r.url for r in response.results] == ["https://a.test"]


@pytest.mark.asyncio
async def test_searxng_429_rate_limited() -> None:
    with patch(f"{_SEARX}.httpx.AsyncClient", _searx_client(_DummyResponse(429))):
        with pytest.raises(SearchBackendRateLimited):
            await build_search_backend("searxng", base_url="https://s.test").search("q")


@pytest.mark.asyncio
async def test_searxng_http_error_unavailable() -> None:
    with patch(f"{_SEARX}.httpx.AsyncClient", _searx_client(_DummyResponse(503))):
        with pytest.raises(SearchBackendUnavailable):
            await build_search_backend("searxng", base_url="https://s.test").search("q")


@pytest.mark.asyncio
async def test_searxng_timeout_unavailable() -> None:
    with patch(f"{_SEARX}.httpx.AsyncClient", _searx_client(get_exc=httpx.TimeoutException("slow"))):
        with pytest.raises(SearchBackendUnavailable):
            await build_search_backend("searxng", base_url="https://s.test").search("q")


# ---------------------------------------------------------------- connect-level failures


@pytest.mark.asyncio
async def test_searxng_connect_error_raises_unreachable_with_host() -> None:
    with patch(
        f"{_SEARX}.httpx.AsyncClient", _searx_client(get_exc=httpx.ConnectError("All connection attempts failed"))
    ):
        with pytest.raises(SearchBackendUnreachable) as excinfo:
            await build_search_backend("searxng", base_url="https://s.test").search("q")
    assert excinfo.value.host == "s.test"


@pytest.mark.asyncio
async def test_firecrawl_keyless_connect_error_raises_unreachable() -> None:
    with patch(f"{_FIRE}.httpx.AsyncClient", _post_client(post_exc=httpx.ConnectError("connection refused"))):
        with pytest.raises(SearchBackendUnreachable) as excinfo:
            await build_search_backend("firecrawl").search("q")
    assert excinfo.value.host == "api.firecrawl.dev"


@pytest.mark.asyncio
async def test_duckduckgo_connect_failure_sniffed_to_unreachable() -> None:
    from ddgs.exceptions import DDGSException

    with patch(f"{_DUCK}.DDGS", lambda: _DummyDDGS(exc=DDGSException("Temporary failure in name resolution"))):
        with pytest.raises(SearchBackendUnreachable):
            await build_search_backend("duckduckgo").search("q")


# ------------------------------------------------------------------------- WebSearchTool


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key="k",
        long_context_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="m"),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="f"),
    )


@pytest.fixture
def caller():
    c = Mock()
    c.agent_name = "coder"
    c.current_tool_call_id = "call-1"
    c.sub_agent = False
    return c


@pytest.fixture
def tool(tmp_path, agent_config, caller):
    return WebSearchTool(tmp_path, "ws", "th", AsyncMock(), agent_config, caller)


@pytest.mark.asyncio
async def test_tool_empty_query_guard(tool) -> None:
    assert (await tool.web_search("   ")).startswith("Error: Provide a non-empty search query.")


@pytest.mark.asyncio
async def test_tool_formats_results_and_streams(tool) -> None:
    response = SearchResponse(
        query="cats",
        backend="duckduckgo",
        results=[SearchResult(title="Cats", url="https://cats.test", snippet="meow")],
    )
    backend = Mock()
    backend.search = AsyncMock(return_value=response)
    with patch("kolega_code.agent.tool_backend.web_search_tool.build_search_backend", return_value=backend):
        result = await tool.web_search("cats", max_results=3)
    # The URL appears as its own line (exact match, not a substring check).
    assert any(line.strip() == "https://cats.test" for line in result.splitlines())
    assert "Cats" in result
    assert "meow" in result
    # A final (is_complete) streaming update is emitted.
    final = [c for c in tool.connection_manager.broadcast_event.call_args_list]
    assert final, "expected streaming updates"


@pytest.mark.asyncio
async def test_tool_empty_results_message(tool) -> None:
    response = SearchResponse(query="zzz", backend="duckduckgo", results=[])
    backend = Mock()
    backend.search = AsyncMock(return_value=response)
    with patch("kolega_code.agent.tool_backend.web_search_tool.build_search_backend", return_value=backend):
        result = await tool.web_search("zzz")
    assert result == "No results found for: zzz"


@pytest.mark.asyncio
async def test_tool_not_configured_is_friendly(tool) -> None:
    backend = Mock()
    backend.search = AsyncMock(side_effect=SearchBackendNotConfigured("needs a key"))
    with patch("kolega_code.agent.tool_backend.web_search_tool.build_search_backend", return_value=backend):
        result = await tool.web_search("q")
    assert result.startswith("Error:") and "Settings > Web Search" in result


@pytest.mark.asyncio
async def test_tool_never_raises_on_unexpected(tool) -> None:
    backend = Mock()
    backend.search = AsyncMock(side_effect=RuntimeError("kaboom"))
    with patch("kolega_code.agent.tool_backend.web_search_tool.build_search_backend", return_value=backend):
        result = await tool.web_search("q")
    assert result.startswith("Error: web_search failed:")


@pytest.mark.asyncio
async def test_tool_unreachable_message_is_causal(tool) -> None:
    backend = Mock()
    backend.search = AsyncMock(side_effect=SearchBackendUnreachable("All connection attempts failed", host="s.test"))
    with patch("kolega_code.agent.tool_backend.web_search_tool.build_search_backend", return_value=backend):
        result = await tool.web_search("q")
    assert result.startswith("Error: Could not connect to s.test")
    assert "may restrict outbound network access" in result
    assert "ConnectError" not in result


@pytest.mark.asyncio
async def test_tool_escalates_on_second_distinct_unreachable_host(tool) -> None:
    def unreachable(host):
        backend = Mock()
        backend.search = AsyncMock(side_effect=SearchBackendUnreachable("connection refused", host=host))
        return backend

    with patch(
        "kolega_code.agent.tool_backend.web_search_tool.build_search_backend", return_value=unreachable("a.test")
    ):
        first = await tool.web_search("q")
    assert "may restrict outbound network access" in first
    with patch(
        "kolega_code.agent.tool_backend.web_search_tool.build_search_backend", return_value=unreachable("b.test")
    ):
        second = await tool.web_search("q")
    assert "2 distinct external hosts have failed to connect" in second
    assert "treat outbound network access as unavailable" in second


@pytest.mark.asyncio
async def test_tool_rate_limited_message_has_no_settings_hint(tool) -> None:
    backend = Mock()
    backend.search = AsyncMock(side_effect=SearchBackendRateLimited("too many requests"))
    with patch("kolega_code.agent.tool_backend.web_search_tool.build_search_backend", return_value=backend):
        result = await tool.web_search("q")
    assert result.startswith("Error:")
    assert "retry shortly" in result
    assert "Settings > Web Search" not in result


def test_tool_threads_shared_tracker(tmp_path, agent_config, caller) -> None:
    tracker = ConnectFailureTracker()
    tool = WebSearchTool(tmp_path, "ws", "th", AsyncMock(), agent_config, caller, connect_failures=tracker)
    assert tool.connect_failures is tracker
